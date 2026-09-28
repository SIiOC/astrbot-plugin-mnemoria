"""记忆星图的边计算与缓存。

「点点相连」的两类边：
1. 语义骨干边：向量余弦相似度，每节点取最强 top_k 条、全局阈值过滤——
   与设计稿 v4 的 BACKBONE 同构（常态只画最强几条，弱边隐入焦点交互）。
2. 演化链：memories.superseded_by 软链（DB 直出，无需计算）。

纯 Python 余弦（项目不依赖 numpy/faiss），896 节点全两两 ≈ 40 万对
约需几十秒——所以放后台线程算一次，结果连同指纹缓存到 state 目录，
指纹（条数 + 最大 updated_at）变化才重算。/graph 端点读缓存，
缓存未就绪时返回 computing=true 由前端轮询。
"""

from __future__ import annotations

import json
from astrbot.api import logger
import math
import sqlite3
import threading
from pathlib import Path
from typing import Any


TOP_K = 3          # 每节点保留的最强关联数（设计稿方向 B 的骨干）
MIN_SIM = 0.55     # 低于此相似度的边不进骨干（真实嵌入校准：同义改写 0.68+）
# 参与连边的节点上限（v0.1.9）：边计算是 O(n²) 纯 Python——实测 887 节点
# 约 21 秒，5000 节点量级会拖到分钟级（缓存与后台线程只能缓解不能解救）。
# 超过上限时只拿"最近创建"的 MAX_NODES 条算边（其余节点仍会画出来，
# 只是没有骨干连线），保证星图永不断死。
MAX_NODES = 1500

_lock = threading.Lock()
_computing = False
_thread: threading.Thread | None = None
_stopping = False


def _cache_path(plugin) -> Path:
    return plugin.paths.state / "graph_edges.json"


def _fingerprint_conn(conn: sqlite3.Connection) -> tuple[int, int, int]:
    mem = conn.execute(
        "SELECT COUNT(*), SUM(CASE WHEN superseded_by IS NOT NULL THEN 1 ELSE 0 END) "
        "FROM memories WHERE deleted_at IS NULL"
    ).fetchone()
    vec = conn.execute(
        "SELECT COUNT(*) FROM vectors v JOIN memories m ON m.id=v.memory_id "
        "WHERE m.deleted_at IS NULL"
    ).fetchone()
    # 指纹保持用"未截断"的向量总数（与旧版一致）：任何向量增删都让缓存失效，
    # 不因 MAX_NODES 截断而丢掉失效信号（节点选择的变化由记忆总数携入）。
    return int(mem[0] or 0), int(vec[0] or 0), int(mem[1] or 0)


def _fingerprint(plugin) -> tuple[int, int, int]:
    """连线结构指纹：只统计「会影响边」的量，避开召回/衰减的记账字段。

    ⚠️ 曾用 MAX(updated_at) 作指纹——但 mark_recalled（每轮召回）与衰减扫描
    都会刷新 updated_at，于是**每次对话都让缓存失效**，用户一开星图就重算
    ~21 秒（887 节点 / 2048 维实测）。改用：
      - 记忆条数：节点增删
      - 向量条数：内容编辑会删向量（web 编辑路径），也代表「边是否可算」
      - 取代链条数：演化链变化
    三者都不随召回 / 衰减变动，缓存命中率大幅提升。
    """
    return _fingerprint_conn(plugin.store.conn)


def _load_cache(plugin) -> dict[str, Any] | None:
    path = _cache_path(plugin)
    if not path.is_file():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    if tuple(data.get("fingerprint", ())) != _fingerprint(plugin):
        return None
    return data


def _compute_edges(conn: sqlite3.Connection) -> list[dict]:
    # 独立连接直读（后台线程内），不再经 plugin.store 共享连接
    from .vector import unpack
    rows = conn.execute(
        "SELECT v.memory_id, v.dim, v.vec, m.created_at FROM vectors v "
        "JOIN memories m ON m.id=v.memory_id WHERE m.deleted_at IS NULL "
        "ORDER BY m.created_at DESC LIMIT ?",
        (int(MAX_NODES),),
    ).fetchall()
    if rows:
        total = conn.execute(
            "SELECT COUNT(*) FROM vectors v JOIN memories m ON m.id=v.memory_id "
            "WHERE m.deleted_at IS NULL"
        ).fetchone()[0]
        if total and int(total) > int(MAX_NODES):
            logger.info(
                "星图节点数 %d 超过上限 %d：仅对最近 %d 条计算骨干边"
                "（O(n²) 保护，其余节点仍会显示）",
                int(total), int(MAX_NODES), int(MAX_NODES),
            )
    vecs = [(r["memory_id"], unpack(r["vec"], r["dim"])) for r in rows]
    # 预归一化：后面点积即余弦
    normed: list[tuple[str, list[float]]] = []
    for mid, v in vecs:
        n = math.sqrt(sum(x * x for x in v))
        if n <= 0:
            continue
        normed.append((mid, [x / n for x in v]))
    from operator import mul
    edges: dict[str, float] = {}
    for i, (id_a, va) in enumerate(normed):
        best: list[tuple[float, str]] = []
        for j, (id_b, vb) in enumerate(normed):
            if i == j:
                continue
            sim = sum(map(mul, va, vb))
            if sim >= MIN_SIM:
                best.append((sim, id_b))
        best.sort(reverse=True)
        for sim, id_b in best[:TOP_K]:
            key = "|".join(sorted((id_a, id_b)))
            if sim > edges.get(key, 0.0):
                edges[key] = sim
    return [
        {"a": k.split("|")[0], "b": k.split("|")[1], "w": round(w, 3)}
        for k, w in edges.items()
    ]


def _compute_and_cache(plugin) -> None:
    global _computing
    try:
        # 独立连接：后台线程不再借用 plugin.store.conn——主线程在 terminate
        # 里 close 共享连接时，跨线程访问会直接崩溃。只做读，普通打开即可
        # 全程参与 WAL（mode=ro 在 WAL 库上依赖 -shm 只读映射，兼容性不稳）
        conn = sqlite3.connect(str(plugin.paths.db), timeout=30.0)
        try:
            conn.row_factory = sqlite3.Row
            edges = _compute_edges(conn)
            sup = conn.execute(
                "SELECT id, superseded_by FROM memories "
                "WHERE superseded_by IS NOT NULL AND deleted_at IS NULL"
            ).fetchall()
            fingerprint = list(_fingerprint_conn(conn))
        finally:
            conn.close()
        payload = {
            "fingerprint": fingerprint,
            "edges": edges,
            "superseded": [
                {"from": r[0], "to": r[1]} for r in sup
            ],
        }
        path = _cache_path(plugin)
        tmp = path.with_suffix(".tmp")
        tmp.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
        tmp.replace(path)
        logger.info("星图连线已生成：%d 条骨干边 / %d 条演化链", len(edges), len(sup))
    except Exception:
        logger.warning("星图连线计算失败", exc_info=True)
    finally:
        _computing = False


def stop(timeout: float = 1.0) -> None:
    """插件 terminate 时调用：停止再派生新计算并等待在跑的线程退出。"""
    global _stopping
    _stopping = True
    with _lock:
        thread = _thread
    if thread is not None and thread.is_alive():
        thread.join(timeout)


def reset() -> None:
    """新插件实例接管时重置停机标志（热重载场景：同进程新旧实例
    共用本模块的全局状态，不重置的话 stop 一次后星图永久停摆）。"""
    global _stopping
    _stopping = False


def ensure_edges(plugin) -> dict[str, Any] | None:
    """读缓存；指纹过期则后台重算（当次返回 None，前端下次再来取）。"""
    global _computing, _thread
    cached = _load_cache(plugin)
    if cached is not None:
        return cached
    with _lock:
        if not _computing and not _stopping:
            _computing = True
            _thread = threading.Thread(
                target=_compute_and_cache, args=(plugin,), daemon=True
            )
            _thread.start()
    return None
