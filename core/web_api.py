"""WebAPI 路由：供插件页控制台与调试使用。

约定（与 AstrBot 4.24.2+ 插件页机制一致，参照 meme_manager 生产实现）：
- 路由路径必须带插件名前缀：`/{PLUGIN_NAME}/{subpath}`
- handler 无位置参数，读取 quart 的 `request`（GET 用 request.args，POST 用 await request.get_json()）
- 返回 `(jsonify(payload), status_code)`
- 前端页面用相对 endpoint（如 `apiGet("overview")`），宿主会补上插件名

所有端点做统一异常包装，失败返回 {ok:false, error}，不让异常穿透到 dashboard。
"""

from __future__ import annotations

import logging
from typing import Any, Callable

from quart import jsonify, request

from . import graph
from .paths import to_local_str

logger = logging.getLogger(__name__)

PLUGIN_NAME = "astrbot_plugin_mnemoria"


def register_routes(context, plugin) -> None:
    reg = getattr(context, "register_web_api", None)
    if reg is None:
        logger.warning("当前 AstrBot 版本不支持 register_web_api，插件页将不可用。")
        return

    def route(subpath: str, handler: Callable, methods: list[str], desc: str) -> None:
        path = f"/{PLUGIN_NAME}/{subpath.strip('/')}"

        async def wrapped(*args, **kwargs):
            try:
                data = handler(*args, **kwargs)
                if hasattr(data, "__await__"):
                    data = await data
                return jsonify({"ok": True, "data": data}), 200
            except Exception as exc:  # noqa: BLE001
                logger.warning("mnemoria API %s 错误: %s", subpath, exc, exc_info=True)
                return jsonify({"ok": False, "error": str(exc)}), 500

        wrapped.__name__ = f"mnemoria_{subpath.replace('/', '_')}"
        try:
            reg(path, wrapped, methods, desc)
        except Exception as exc:  # noqa: BLE001
            logger.warning("注册路由 %s 失败: %s", path, exc)

    route("overview", lambda: _overview(plugin), ["GET"], "记忆系统总览")
    route("memories", lambda: _list_memories(plugin), ["GET"], "记忆列表/搜索")
    route("memory/create", lambda: _create_memory(plugin), ["POST"], "新增记忆")
    route("memory/get", lambda: _get_memory(plugin), ["GET"], "记忆详情")
    route("memory/update", lambda: _update_memory(plugin), ["POST"], "编辑记忆")
    route("memory/delete", lambda: _delete_memory(plugin), ["POST"], "删除记忆（回收站）")
    route("memory/restore", lambda: _restore_memory(plugin), ["POST"], "从回收站恢复")
    route("memory/purge", lambda: _purge_memory(plugin), ["POST"], "彻底删除记忆（不可恢复）")
    route("trash", lambda: _list_trash(plugin), ["GET"], "回收站列表")
    route("profiles", lambda: _list_profiles(plugin), ["GET"], "画像列表")
    route("profile/save", lambda: _save_profile(plugin), ["POST"], "新增/更新画像条目")
    route("profile/delete", lambda: _del_profile(plugin), ["POST"], "删除画像条目")
    route("ledger", lambda: _list_ledger(plugin), ["GET"], "账本回看")
    route("ledger/search", lambda: _search_ledger(plugin), ["GET"], "账本搜索")
    route("recall", lambda: _recall(plugin), ["GET"], "检索探针")
    route("decay/run", lambda: _run_decay(plugin), ["POST"], "手动执行衰减")
    route("backup/run", lambda: _run_backup(plugin), ["POST"], "手动导出备份")
    route("backups", lambda: _list_backups(plugin), ["GET"], "备份列表")
    # 笔记知识库
    route("notes", lambda: _list_notes(plugin), ["GET"], "笔记列表/搜索")
    route("note/create", lambda: _create_note(plugin), ["POST"], "新增笔记")
    route("note/update", lambda: _update_note(plugin), ["POST"], "编辑笔记")
    route("note/delete", lambda: _delete_note(plugin), ["POST"], "删除笔记（回收站）")
    route("note/restore", lambda: _restore_note(plugin), ["POST"], "从回收站恢复笔记（v0.1.9）")
    route("note/purge", lambda: _purge_note(plugin), ["POST"], "彻底删除笔记（不可恢复）")
    route("note/import", lambda: _import_notes(plugin), ["POST"], "导入 .md 笔记")
    # 星图与配置（UI 融合新增）
    route("graph", lambda: _graph(plugin), ["GET"], "星图：节点+骨干边+演化链")
    route("config", lambda: _get_config(plugin), ["GET"], "读取插件配置")
    route("config/save", lambda: _save_config(plugin), ["POST"], "保存插件配置（就地改+落盘+热生效）")
    route("providers", lambda: _list_providers(plugin), ["GET"], "可用模型列表（设置页下拉）")


# --------------------------------------------------------------------- helpers
def _rows(rows) -> list[dict]:
    out = []
    for r in rows:
        d = dict(r)
        for k in ("created_at", "updated_at", "last_recalled_at", "last_decay_at", "valid_to", "deleted_at", "ts"):
            if k in d:
                d[k + "_local"] = to_local_str(d.get(k))
        out.append(d)
    return out


def _q(name: str, default: str = "") -> str:
    try:
        return (request.args.get(name) or default).strip()
    except Exception:  # noqa: BLE001
        return default


def _qi(name: str, default: int) -> int:
    try:
        return int(request.args.get(name))
    except (TypeError, ValueError):
        return default


async def _body() -> dict:
    try:
        data = await request.get_json()
        return data if isinstance(data, dict) else {}
    except Exception:  # noqa: BLE001
        return {}


# --------------------------------------------------------------------- handlers
def _overview(plugin) -> dict:
    return plugin.engine.stats()


async def _list_memories(plugin) -> dict:
    scope = _q("scope", "default") or "default"
    query = _q("q")
    limit = _qi("limit", 100)
    if query:
        # 搜索与生产检索同源（含向量通道）；不计热度、预算放宽便于浏览
        cands = await plugin.engine.recall(
            query, scope=scope, top_k=limit, token_budget=100000, mark_recalled=False,
        )
        ids = [c.id for c in cands]
        if not ids:
            return {"items": [], "scope": scope}
        rows = plugin.store.conn.execute(
            "SELECT id, content, memory_type, is_active, strength, useful_score, hit_count, "
            "proof_count, quarantined, superseded_by, observed_at, scope, session_id, "
            "tags_json, created_at, updated_at FROM memories WHERE id IN (%s)" % ",".join("?" * len(ids)),
            ids,
        ).fetchall()
        items = []
        for r in _rows(rows):
            r["quarantined"] = bool(r.get("quarantined"))
            ts = r.get("observed_at") or r.get("created_at") or r.get("updated_at")
            r["t"] = float(ts) if ts else 0.0
            r["tags"] = plugin.store.parse_tags(r)
            items.append(r)
        return {"items": items, "scope": scope}
    rows = plugin.store.conn.execute(
        "SELECT id, content, memory_type, is_active, strength, useful_score, hit_count, "
        "proof_count, quarantined, superseded_by, observed_at, scope, session_id, "
        "tags_json, created_at, updated_at FROM memories "
        "WHERE scope=? AND deleted_at IS NULL AND superseded_by IS NULL "
        "ORDER BY updated_at DESC LIMIT ?",
        (scope, limit),
    ).fetchall()
    items = []
    for r in _rows(rows):
        r["quarantined"] = bool(r.get("quarantined"))
        ts = r.get("observed_at") or r.get("created_at") or r.get("updated_at")
        r["t"] = float(ts) if ts else 0.0
        r["tags"] = plugin.store.parse_tags(r)
        items.append(r)
    return {"items": items, "scope": scope}


def _get_memory(plugin) -> dict:
    mid = _q("id")
    row = plugin.store.get_memory(mid) if mid else None
    return {"memory": _rows([row])[0] if row else None}


async def _create_memory(plugin) -> dict:
    """面板手动新增记忆：复用引擎写入门（α 门+去重），不绕过检验。

    重复内容会走「强化」而非重复插入（与抽取路径同语义）；
    主动记忆由调用方指定，默认被动。
    """
    body = await _body()
    content = str(body.get("content") or "").strip()
    if not content:
        raise ValueError("缺少 content")
    scope = str(body.get("scope") or "default") or "default"
    mtype = str(body.get("memory_type") or "fact") or "fact"
    is_active = body.get("is_active") in (True, "true", "1", 1)
    speaker = str(body.get("speaker") or "")
    before = plugin.store.count().get("total", 0)
    ok = await plugin.engine.remember(
        content,
        memory_type=mtype,
        alpha=1.0,          # 人工写入视为高价值
        source="manual",
        scope=scope,
        speaker=speaker,
        is_active=is_active,
    )
    after = plugin.store.count().get("total", 0)
    if not ok:
        return {"created": False, "reason": "被写入门拒绝（重复已强化 / 元指令 / 隐私串）"}
    return {"created": after > before, "reinforced": after == before, "scope": scope}


async def _update_memory(plugin) -> dict:
    body = await _body()
    mid = str(body.get("id") or "")
    if not mid:
        raise ValueError("缺少 id")
    fields: dict[str, Any] = {}
    if "content" in body:
        fields["content"] = str(body["content"])
    if "memory_type" in body:
        fields["memory_type"] = str(body["memory_type"])
    if body.get("strength") is not None:
        fields["strength"] = float(body["strength"])
    if "is_active" in body:
        fields["is_active"] = 1 if body["is_active"] in (True, "true", "1", 1) else 0
    if body.get("useful_score") is not None:
        fields["useful_score"] = float(body["useful_score"])
    if "quarantined" in body:
        fields["quarantined"] = 1 if body["quarantined"] in (True, "true", "1", 1) else 0
        if fields["quarantined"] == 0:
            # v0.2.11：通过审核 = 真正回到活性检索面。隔离条目出生即
            # deleted_at=now（落在回收站），只清 quarantined 会得到
            # 「既不隔离也不可见」的隐身行——一并清掉 deleted_at。
            fields["deleted_at"] = None
    plugin.store.update_memory(mid, **fields)
    # 内容被人工修改后旧向量不再代表新语义，删掉让检索退回关键词通道；
    # 同时置脏缓存，否则去重/巩固在下次 remember() 前仍拿旧向量做判定（同 _merge_cluster 缺陷）
    if "content" in fields:
        plugin.store.delete_vector(mid)
        plugin.engine._vectors_dirty = True
    # 隔离条目通过审核 = 进入活跃检索面（此前无向量），向量缓存置脏待补嵌
    if fields.get("quarantined") == 0:
        plugin.engine._vectors_dirty = True
    return {"updated": mid}


async def _delete_memory(plugin) -> dict:
    body = await _body()
    mid = str(body.get("id") or "")
    if not mid:
        raise ValueError("缺少 id")
    plugin.store.trash(mid)
    return {"trashed": mid}


async def _restore_memory(plugin) -> dict:
    body = await _body()
    mid = str(body.get("id") or "")
    if not mid:
        raise ValueError("缺少 id")
    # 普通恢复保留 superseded_by，避免被新说法取代的旧事实复活到检索面；
    # clear_superseded=true 是人工明确选择的彻底恢复。
    clear_superseded = body.get("clear_superseded") in (True, "true", "1", 1)
    plugin.store.restore(mid, clear_superseded=clear_superseded)
    # v0.1.4：恢复 = 该条向量重新进入活跃检索面，向量缓存必须置脏——
    # 此前漏置脏，与「编辑删除向量」同类的失效纪律缺口（编辑路径已置脏，恢复路径漏了）
    plugin.engine._vectors_dirty = True
    return {"restored": mid, "clear_superseded": clear_superseded}


async def _purge_memory(plugin) -> dict:
    """彻底删除（回收站里的硬删）：store.purge 连带删向量，不可恢复。"""
    body = await _body()
    mid = str(body.get("id") or "")
    if not mid:
        raise ValueError("缺少 id")
    plugin.store.purge(mid)
    plugin.engine._vectors_dirty = True
    return {"purged": mid}


def _list_trash(plugin) -> dict:
    """回收站列表：记忆 + 笔记（v0.1.9）。

    笔记此前只有软删没有恢复也没有清理，面板看不到、也恢复不了；
    现在与记忆一起回到同一张回收站列表，并支持 note/restore。
    """
    return {
        "items": _rows(plugin.store.list_trash()),
        "notes": _rows(plugin.store.list_trash_notes()),
    }


def _list_profiles(plugin) -> dict:
    scope = _q("scope", "default") or "default"
    user_key = _q("user_key")
    if user_key:
        rows = plugin.store.get_profile(scope, user_key)
    else:
        rows = plugin.store.conn.execute(
            "SELECT scope, user_key, key, value, confidence, updated_at FROM profiles "
            "WHERE scope=? ORDER BY user_key, updated_at DESC",
            (scope,),
        ).fetchall()
    return {"items": _rows(rows)}


async def _save_profile(plugin) -> dict:
    """面板新增/更新画像条目（同键 upsert，与自动提炼/工具写入同表同语义）。"""
    body = await _body()
    scope = str(body.get("scope") or "default") or "default"
    user_key = str(body.get("user_key") or "").strip()
    key = str(body.get("key") or "").strip()
    value = str(body.get("value") or "").strip()
    if not user_key or not key or not value:
        raise ValueError("用户 ID、维度、取值均不能为空")
    try:
        conf = float(body.get("confidence") if body.get("confidence") is not None else 1.0)
    except (TypeError, ValueError):
        conf = 1.0
    conf = max(0.0, min(1.0, conf))
    plugin.store.upsert_profile(scope, user_key, key, value, confidence=conf)
    return {"saved": True}


async def _del_profile(plugin) -> dict:
    body = await _body()
    plugin.store.delete_profile(
        str(body.get("scope") or "default"),
        str(body.get("user_key") or ""),
        str(body.get("key") or ""),
    )
    return {"deleted": True}


def _list_ledger(plugin) -> dict:
    session_id = _q("session_id")
    limit = _qi("limit", 50)
    if session_id:
        rows = plugin.store.recent_ledger(session_id, limit=limit)
    else:
        rows = plugin.store.conn.execute(
            "SELECT session_id, role, substr(content,1,200) AS content, ts FROM ledger "
            "ORDER BY ts DESC LIMIT ?",
            (limit,),
        ).fetchall()
    # 面板隐私：用户自己的消息不展示（也不出网），只回放助手侧时间线。
    # 抽取/反思走 store.recent_ledger，不受此影响。
    rows = [r for r in rows if r["role"] == "assistant"]
    return {"items": _rows(rows)}


def _search_ledger(plugin) -> dict:
    q = _q("q")
    session_id = _q("session_id") or None
    rows = plugin.store.search_ledger(q, session_id=session_id, limit=_qi("limit", 30))
    rows = [r for r in rows if r["role"] == "assistant"]
    return {"items": _rows(rows)}


async def _recall(plugin) -> dict:
    """检索探针：走生产引擎的完整四路检索（向量+关键词+标签+时间 RRF），
    与注入/工具路径同源——探针看到的就是模型看到的。不计热度。"""
    q = _q("q")
    scope = _q("scope", "default") or "default"
    if not q:
        return {"items": []}
    cands = await plugin.engine.recall(
        q, scope=scope, top_k=_qi("limit", 8), token_budget=4000, mark_recalled=False,
    )
    return {"items": [{"id": c.id, "content": c.content, "receipt": c.receipt,
                       "channels": c.channels, "hotness": c.hotness} for c in cands]}


def _run_decay(plugin) -> dict:
    return plugin.engine.decay_sweep()


def _run_backup(plugin) -> dict:
    from .backup import write_backup
    path = write_backup(plugin.paths, plugin.conn, keep=int(plugin.config.get("backup.keep_copies", 3) or 3))
    return {"file": path.name if path else None}


def _list_backups(plugin) -> dict:
    from .backup import list_backups
    return {"items": list_backups(plugin.paths)}


# --------------------------------------------------------------------- 笔记
def _list_notes(plugin) -> dict:
    scope = _q("scope", "default") or "default"
    query = _q("q")
    limit = _qi("limit", 200)
    if query:
        rows = plugin.store.search_notes(query, scope=scope, limit=limit)
    else:
        rows = plugin.store.list_notes(scope=scope, limit=limit)
    return {"items": _rows(rows), "scope": scope}


async def _create_note(plugin) -> dict:
    body = await _body()
    content = str(body.get("content") or "").strip()
    if not content:
        raise ValueError("缺少 content")
    scope = str(body.get("scope") or "default") or "default"
    nid = await plugin.engine.add_note(
        content,
        title=str(body.get("title") or ""),
        tags=str(body.get("tags") or ""),
        source="manual",
        scope=scope,
    )
    return {"created": bool(nid), "id": nid}


async def _update_note(plugin) -> dict:
    body = await _body()
    nid = str(body.get("id") or "")
    if not nid:
        raise ValueError("缺少 id")
    fields: dict[str, Any] = {}
    for k in ("title", "content", "tags"):
        if k in body:
            fields[k] = str(body[k])
    # v0.2.0：走引擎统一入口——内容变更时重嵌向量并重建切片派生层
    await plugin.engine.update_note(nid, **fields)
    return {"updated": nid}


async def _delete_note(plugin) -> dict:
    body = await _body()
    nid = str(body.get("id") or "")
    if not nid:
        raise ValueError("缺少 id")
    plugin.store.trash_note(nid)
    return {"trashed": nid}


async def _restore_note(plugin) -> dict:
    """从回收站恢复笔记（v0.1.9：与记忆同一套生命周期）。"""
    body = await _body()
    nid = str(body.get("id") or "")
    if not nid:
        raise ValueError("缺少 id")
    plugin.store.restore_note(nid)
    return {"restored": nid}


async def _purge_note(plugin) -> dict:
    """彻底删除笔记（连带切片），不可恢复。"""
    body = await _body()
    nid = str(body.get("id") or "")
    if not nid:
        raise ValueError("缺少 id")
    plugin.store.purge_note(nid)
    return {"purged": nid}


async def _import_notes(plugin) -> dict:
    """导入 .md：浏览器端读文件文本后 POST 过来，这里分块入库。"""
    body = await _body()
    text = str(body.get("text") or "")
    if not text.strip():
        raise ValueError("文件内容为空")
    scope = str(body.get("scope") or "default") or "default"
    file_name = str(body.get("file_name") or "")
    n = await plugin.engine.import_markdown(text, file_name=file_name, scope=scope)
    return {"imported": n, "file_name": file_name}


# ---------------------------------------------------------------- 星图
def _graph(plugin) -> dict:
    """星图数据：节点（全部未删除记忆）+ 边（骨干相似度 + 演化链）。

    边计算为后台线程+磁盘缓存（指纹=条数+最大updated_at），
    未就绪时返回 computing=true，前端稍后再拉。
    """
    scope = _q("scope", "default") or "default"
    rows = plugin.store.conn.execute(
        "SELECT id, content, memory_type, is_active, strength, hit_count, "
        "quarantined, superseded_by, observed_at, updated_at "
        "FROM memories WHERE scope=? AND deleted_at IS NULL",
        (scope,),
    ).fetchall()
    nodes = [
        {
            "id": r[0], "content": r[1], "type": r[2],
            "active": bool(r[3]), "strength": r[4] or 10.0,
            "heat": r[5] or 0, "quarantined": bool(r[6]),
            "superseded_by": r[7], "t": r[8] or r[9] or 0,
        }
        for r in rows
    ]
    cached = graph.ensure_edges(plugin)
    if cached is None:
        return {"nodes": nodes, "edges": None, "superseded": None, "computing": True}
    return {
        "nodes": nodes,
        "edges": cached["edges"],
        "superseded": cached["superseded"],
        "computing": False,
    }


# ---------------------------------------------------------------- 配置
def _list_providers(plugin) -> dict:
    """设置页 provider 下拉的选项源：枚举 AstrBot 已配置的模型 id。

    provider_manager 的内部结构随 AstrBot 版本变动，全部防御式读取，
    拿不到就返回空列表（前端退化为「当前值 + 手输」）。
    """
    chat: list[str] = []
    embedding: list[str] = []
    pm = getattr(getattr(plugin, "ctx", None), "provider_manager", None)
    insts = getattr(pm, "provider_insts", None) or []
    for p in insts:
        try:
            meta = p.meta()
            pid = str(getattr(meta, "id", "") or "").strip()
            ptype = str(getattr(meta, "type", "") or "").lower()
        except Exception:  # noqa: BLE001
            continue
        if not pid:
            continue
        (embedding if "embedding" in ptype else chat).append(pid)
    return {"chat": chat, "embedding": embedding}


def _flatten(raw: dict, prefix: str = "") -> dict:
    out: dict[str, Any] = {}
    for k, v in raw.items():
        key = f"{prefix}.{k}" if prefix else str(k)
        if isinstance(v, dict):
            out.update(_flatten(v, key))
        else:
            out[key] = v
    return out


def _set_dotted(raw: dict, dotted: str, value) -> None:
    parts = dotted.split(".")
    cur: dict = raw
    for p in parts[:-1]:
        nxt = cur.get(p)
        if not isinstance(nxt, dict):
            nxt = {}
            cur[p] = nxt
        cur = nxt
    cur[parts[-1]] = value


def _get_config(plugin) -> dict:
    raw = plugin.config._raw if isinstance(plugin.config._raw, dict) else {}
    return {"flat": _flatten(raw), "raw": raw}


def _allowed_config_keys(plugin) -> set[str]:
    """把 _conf_schema.json 展平成允许写入的 dotted 键集合。

    组（有 items 的对象）递归下钻；叶子即允许键。
    schema 取不到时返回空集合——调用方据此**拒绝保存**（fail-closed：
    白名单校验不可用时放行任意键等于没有校验）。
    """
    schema = getattr(plugin.astrbot_config, "schema", None)
    if not isinstance(schema, dict) or not schema:
        return set()
    allowed: set[str] = set()

    def walk(node: dict, prefix: str) -> None:
        for k, v in node.items():
            key = f"{prefix}.{k}" if prefix else str(k)
            items = v.get("items") if isinstance(v, dict) else None
            if isinstance(items, dict) and items:
                walk(items, key)
            else:
                allowed.add(key)

    walk(schema, "")
    return allowed


def _schema_nodes(plugin) -> dict[str, dict]:
    """展平 schema，返回 dotted 键 -> 叶子节点（供类型/范围校验）。"""
    schema = getattr(plugin.astrbot_config, "schema", None)
    if not isinstance(schema, dict) or not schema:
        return {}
    nodes: dict[str, dict] = {}

    def walk(node: dict, prefix: str) -> None:
        for k, v in node.items():
            key = f"{prefix}.{k}" if prefix else str(k)
            if isinstance(v, dict):
                items = v.get("items")
                if isinstance(items, dict) and items:
                    walk(items, key)
                    continue
                nodes[key] = v

    walk(schema, "")
    return nodes


def _validate_value(node: dict, key: str, value):
    """按 schema 声明校验并钳制单个配置值；非法类型抛 ValueError。

    拒绝在白名单之外的类型（如把 int 键写成字符串），否则毒化配置会让
    后续所有 int()/float() 读取抛错——那等于一个键瘫痪整个配置读取。
    """
    t = str(node.get("type") or "").lower()
    if t in ("int",):
        if isinstance(value, bool) or not isinstance(value, int):
            raise ValueError(f"配置 {key} 需要 int，收到 {type(value).__name__}")
    elif t in ("float",):
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError(f"配置 {key} 需要 float，收到 {type(value).__name__}")
    elif t in ("bool",):
        if not isinstance(value, bool):
            raise ValueError(f"配置 {key} 需要 bool，收到 {type(value).__name__}")
    elif t in ("string", "text", "select"):
        if not isinstance(value, str):
            raise ValueError(f"配置 {key} 需要 string，收到 {type(value).__name__}")
        options = node.get("options")
        if isinstance(options, list) and options and value not in options:
            raise ValueError(f"配置 {key} 不在允许的取值内：{value}")
    elif t in ("list",):
        if not isinstance(value, list):
            raise ValueError(f"配置 {key} 需要 list，收到 {type(value).__name__}")
    elif t in ("object",):
        if not isinstance(value, dict):
            raise ValueError(f"配置 {key} 需要 object，收到 {type(value).__name__}")
    # 未知类型不强校验（schema 扩展类型放行，保守不误杀）
    if t in ("int", "float") and not isinstance(value, bool):
        for bound, op in (("min", max), ("max", min)):
            limit = node.get(bound)
            if limit is None or isinstance(limit, bool):
                continue
            try:
                limit = float(limit)
            except (TypeError, ValueError):
                continue
            value = op(value, limit)
        if t == "int":
            value = int(value)
    return value


async def _save_config(plugin) -> dict:
    """保存配置：updates={dotted键: 值}。

    Config._raw 与框架 AstrBotConfig 是同一个 dict——就地写入即热生效
    （桥接对象惰性读）；再用框架对象 save_config_async 落盘。

    白名单校验：只接受 _conf_schema.json 里声明过的键，防拼错/恶意键被
    写入配置并落盘（未知键会被 check_config_integrity 永久留存）。
    schema 不可用时**拒绝保存**（fail-closed）；值按 schema 类型/范围校验。
    """
    body = await _body()
    updates = body.get("updates") or {}
    if not isinstance(updates, dict) or not updates:
        raise ValueError("缺少 updates")
    raw = plugin.config._raw
    if not isinstance(raw, dict):
        raise ValueError("配置对象不可写")

    allowed = _allowed_config_keys(plugin)
    if not allowed:
        raise ValueError("无法读取配置 schema，拒绝保存（fail-closed）")
    nodes = _schema_nodes(plugin)
    clean: dict = {}
    rejected: list[str] = []
    for dotted, value in updates.items():
        if not isinstance(dotted, str) or not dotted.strip():
            continue
        key = dotted.strip()
        if key not in allowed:
            rejected.append(key)
            continue
        node = nodes.get(key)
        if node:
            value = _validate_value(node, key, value)
        clean[key] = value
    if rejected:
        raise ValueError("未知配置键（未在 schema 中声明）：" + "、".join(sorted(rejected)))
    if not clean:
        raise ValueError("没有可应用的配置项")

    for dotted, value in clean.items():
        _set_dotted(raw, dotted, value)
    saved = False
    if plugin.astrbot_config is not None and hasattr(plugin.astrbot_config, "save_config_async"):
        saved = await plugin.astrbot_config.save_config_async()
    return {"applied": list(clean.keys()), "persisted": bool(saved)}
