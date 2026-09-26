"""记忆导入/导出（好想记住你 JSON 格式）。

用于：
1. 接收 migrate_from_angel.py 的产物
2. 常规备份/恢复

命令行：
    python scripts/import_export.py export --db <好想记住你db> --out dump.json
    python scripts/import_export.py import --db <好想记住你db> --in dump.json \
        [--vectorize --provider-id <嵌入提供商id>] [--no-preserve-ts]

导入会做去重（content_hash），已在库中的内容会走「强化」而非重复插入；
可重复执行（幂等），中断后重跑只补缺失部分。

--vectorize：导入后对「还没有向量的未删除记忆」批量补嵌（每批 20 条，
直连 provider 的 /embeddings 端点，读 AstrBot 主配置里的 key）。
只补缺失的行，因此天然可断点重跑。迁移大库时必须加此开关——引擎没有
"检索时按需补建"机制，缺向量的记忆在语义通道不可见。

时间保真（默认开）：导入条目若带 observed_at（迁移导出均有），回填到
created_at/observed_at/valid_from——否则迁移记忆全部伪装成"刚记住"，
时间近因通道会把它排到最前，衰减锚点也被重置。
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import sqlite3  # noqa: E402

from core import db as dbm  # noqa: E402
from core.paths import utc_now_ts  # noqa: E402
from core.store import MemoryStore  # noqa: E402

# AstrBot 主配置（provider key 所在）：环境变量 ASTRBOT_CMD_CONFIG 或
# --config 必填。Path("") 会归一化成当前目录、.exists() 恒真——必须用
# None 显式表达"未配置"（v0.2.16 审查 D1：空串默认让友好报错成死代码，
# 用户照文档示例跑会吃到 IsADirectoryError 原始栈）。
CFG_PATH = None
_env_cfg = os.environ.get("ASTRBOT_CMD_CONFIG", "").strip()
if _env_cfg:
    CFG_PATH = Path(_env_cfg)


def _open_readonly(db_path: Path) -> sqlite3.Connection:
    """只读打开导出源库（绝不 init_schema——那会给只读操作附带建表写入）。

    WAL 库上 mode=ro 依赖 -shm 只读映射，失败时退回普通连接（仅读，
    不写任何东西，效果等价）。
    """
    try:
        uri = db_path.resolve().as_uri() + "?mode=ro"
        conn = sqlite3.connect(uri, uri=True, timeout=10.0)
    except (sqlite3.Error, ValueError, OSError):
        conn = sqlite3.connect(str(db_path), timeout=10.0)
    conn.row_factory = sqlite3.Row
    return conn


def do_export(db_path: Path, out_path: Path) -> int:
    conn = _open_readonly(db_path)
    payload = {
        "source": "mnemoria",
        "exported_at": utc_now_ts(),
        "memories": [
            dict(r) for r in conn.execute(
                "SELECT id, content, reasoning, memory_type, source, speaker, speaker_key, "
                "is_active, strength, useful_score, useful_count, hit_count, proof_count, "
                "observed_at, valid_from, valid_to, superseded_by, deleted_at, quarantined, "
                "scope, session_id, tags_json, created_at, updated_at FROM memories"
            ).fetchall()
        ],
        "profiles": [
            dict(r) for r in conn.execute(
                "SELECT scope, user_key, key, value, confidence, updated_at FROM profiles"
            ).fetchall()
        ],
        "notes": [
            dict(r) for r in conn.execute(
                "SELECT id, title, content, tags, source, file_name, heading, scope, "
                "content_hash, deleted_at, created_at, updated_at FROM notes"
            ).fetchall()
        ],
    }
    out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    conn.close()
    print(f"[完成] 导出 {len(payload['memories'])} 条记忆、{len(payload['profiles'])} 条画像、"
          f"{len(payload['notes'])} 条笔记 → {out_path}")
    return 0


def do_import(db_path: Path, in_path: Path, vectorize: bool = False,
              preserve_ts: bool = True, provider_id: str = "") -> int:
    data = json.loads(in_path.read_text(encoding="utf-8"))
    memories = data.get("memories") or []
    conn = dbm.connect(db_path)
    dbm.init_schema(conn)
    store = MemoryStore(conn)

    added = reinforced = skipped = 0
    for m in memories:
        content = str(m.get("content") or "").strip()
        if not content:
            skipped += 1
            continue
        scope = str(m.get("scope") or data.get("scope") or "default")
        from core.text import content_hash
        existing = store.get_by_hash(content_hash(content), scope)
        if existing:
            store.reinforce(existing["id"], useful_delta=0.0, strength_delta=0.5)
            # angel 内部同一句话可能同时存在主动/被动两个版本：
            # 任一版本 is_active=1，合并后的条目也应升为主动（永不衰减）
            if m.get("is_active") and not existing["is_active"]:
                store.update_memory(existing["id"], is_active=1)
            reinforced += 1
            continue
        store.add_memory(
            content,
            reasoning=str(m.get("reasoning") or ""),
            memory_type=str(m.get("memory_type") or "fact"),
            source=str(m.get("source") or "user"),
            speaker=str(m.get("speaker") or ""),
            # v0.2.0：稳定身份键随备份往返（旧备份无此字段则空串，不猜测）
            speaker_key=str(m.get("speaker_key") or ""),
            scope=scope,
            session_id=str(m.get("session_id") or ""),
            is_active=bool(m.get("is_active")),
            strength=float(m.get("strength") or 10.0),
            proof_count=int(m.get("proof_count") or 1),
            valid_from=_ts_or_none(m.get("observed_at")) if preserve_ts else None,
            # v0.1.8：备份往返保留 tags（导出 JSON 里是 tags_json 列名）
            tags=store.parse_tags({"tags_json": m.get("tags_json")}) or None,
        )
        # 回填 useful_score 等分数（add_memory 默认 0）
        row = store.get_by_hash(content_hash(content), scope)
        if row:
            store.update_memory(
                row["id"],
                useful_score=float(m.get("useful_score") or 0.0),
                useful_count=int(m.get("useful_count") or 0),
            )
            # 时间保真：created_at/observed_at 用原始观察时间（update_memory
            # 的白名单不含这两列，脚本自持连接直改）
            ts = _ts_or_none(m.get("observed_at")) if preserve_ts else None
            if ts:
                conn.execute(
                    "UPDATE memories SET created_at=?, observed_at=? WHERE id=?",
                    (ts, ts, row["id"]),
                )
                conn.commit()
        added += 1

    for p in data.get("profiles") or []:
        store.upsert_profile(
            str(p.get("scope") or "default"),
            str(p.get("user_key") or ""),
            str(p.get("key") or ""),
            str(p.get("value") or ""),
            confidence=float(p.get("confidence") or 1.0),
        )

    notes_added = notes_skipped = 0
    for n in data.get("notes") or []:
        content = str(n.get("content") or "").strip()
        nid = str(n.get("id") or "").strip()
        if not content or not nid:
            notes_skipped += 1
            continue
        chash = str(n.get("content_hash") or "")
        # v0.2.16（审查 D5）：缺省 scope 对齐检索/注入的默认域 "default"——
        # 此前回落 "public"，迁移笔记在默认域静默不可检索。
        nscope = str(n.get("scope") or "default")
        # 幂等：同 id 或同 content_hash+scope 已存在则跳过（重复执行不重复插入）
        exists = conn.execute("SELECT 1 FROM notes WHERE id=?", (nid,)).fetchone()
        if not exists and chash:
            exists = conn.execute(
                "SELECT 1 FROM notes WHERE content_hash=? AND scope=?", (chash, nscope)
            ).fetchone()
        if exists:
            notes_skipped += 1
            continue
        conn.execute(
            "INSERT INTO notes(id, title, content, tags, source, file_name, heading, "
            "scope, content_hash, deleted_at, created_at, updated_at) "
            "VALUES(?,?,?,?,?,?,?,?,?,?,?,?)",
            (nid, str(n.get("title") or ""), content, str(n.get("tags") or ""),
             str(n.get("source") or "manual"), str(n.get("file_name") or ""),
             str(n.get("heading") or ""), nscope, chash,
             _ts_or_none(n.get("deleted_at")),
             _ts_or_none(n.get("created_at")) or utc_now_ts(),
             _ts_or_none(n.get("updated_at")) or utc_now_ts()),
        )
        notes_added += 1
    if data.get("notes") is not None:
        conn.commit()
        print(f"[笔记] 新增 {notes_added} 条，跳过 {notes_skipped} 条（id/指纹重复）")

    print(f"[完成] 新增 {added} 条，强化已存在 {reinforced} 条，跳过 {skipped} 条")

    if vectorize:
        # v0.1.9：此前 --provider-id 被解析了但从未传到这里（恒用默认
        # nvidia_embedding），换嵌入供应商迁移时用户以为参数生效了。
        grp = _load_embedding_provider(provider_id)
        backfill_vectors(conn, grp)

    conn.close()
    return 0


def _ts_or_none(value) -> float | None:
    try:
        v = float(value)
    except (TypeError, ValueError):
        return None
    return v if v > 0 else None


# ---------------------------------------------------------------- 向量回填
def _load_embedding_provider(provider_id: str = "") -> dict:
    """从 AstrBot 主配置读嵌入 provider 组（与 calibrate_embeddings.py 同款）。

    v0.2.16（审查 D1/D2）：CFG_PATH 未配置/不存在给清晰报错；provider_id
    为空时列出配置内全部可用嵌入提供商（不再按作者环境默认值静默外发）。
    """
    if CFG_PATH is None or not CFG_PATH.exists():
        where = "ASTRBOT_CMD_CONFIG" if CFG_PATH is None else str(CFG_PATH)
        raise SystemExit(
            f"未找到 AstrBot 主配置：{where}\n"
            "请设置 ASTRBOT_CMD_CONFIG 环境变量，或用 --config <path> 指定 cmd_config.json。"
        )
    cfg = json.loads(CFG_PATH.read_text(encoding="utf-8-sig"))
    if not provider_id:
        candidates = [str(g.get("id") or "") for g in cfg.get("provider", [])
                      if g.get("embedding_api_key")]
        raise SystemExit(
            "未指定嵌入提供商。请用 --provider-id 从以下候选中选择：\n  "
            + ("\n  ".join(candidates) if candidates else "（配置里没有带 embedding_api_key 的提供商）")
        )
    for grp in cfg.get("provider", []):
        if grp.get("id") == provider_id:
            if not grp.get("embedding_api_key"):
                raise SystemExit(f"提供商 {provider_id} 没有 API key")
            return grp
    raise SystemExit(f"找不到提供商 {provider_id}")


def missing_vector_rows(conn) -> list[dict]:
    """缺向量且未删除的记忆（回填与断点重跑的选择器）。"""
    return [
        {"id": r["id"], "content": r["content"]}
        for r in conn.execute(
            "SELECT m.id, m.content FROM memories m "
            "LEFT JOIN vectors v ON v.memory_id = m.id "
            "WHERE v.memory_id IS NULL AND m.deleted_at IS NULL"
        ).fetchall()
    ]


def backfill_vectors(conn, grp: dict, batch: int = 20, embed_fn=None) -> int:
    """给缺向量的记忆补嵌。embed_fn 可注入（测试用）；默认直调 /embeddings。

    单批失败只告警并继续（该批行保持无向量，重跑即补）。返回成功补嵌条数。
    """
    if embed_fn is None:
        def embed_fn(texts):
            return _embed_openai(grp["embedding_api_base"], grp["embedding_api_key"],
                                 grp["embedding_model"], texts)

    rows = missing_vector_rows(conn)
    if not rows:
        print("[向量] 全部记忆已有向量，无需补建")
        return 0
    print(f"[向量] 待补嵌 {len(rows)} 条（模型 {grp.get('embedding_model')}，每批 {batch}）")
    from core.vector import pack

    done = 0
    for i in range(0, len(rows), batch):
        chunk = rows[i:i + batch]
        try:
            vecs = embed_fn([r["content"] for r in chunk])
        except Exception as exc:  # noqa: BLE001
            print(f"[向量] 第 {i // batch + 1} 批失败（{exc}），重跑本命令可续补")
            break
        if not vecs or len(vecs) != len(chunk):
            print(f"[向量] 第 {i // batch + 1} 批返回数量不符，跳过（重跑可续补）")
            continue
        for r, vec in zip(chunk, vecs):
            conn.execute(
                "INSERT INTO vectors(memory_id, dim, vec) VALUES(?,?,?) "
                "ON CONFLICT(memory_id) DO UPDATE SET dim=excluded.dim, vec=excluded.vec",
                (r["id"], len(vec), pack(vec)),
            )
        conn.commit()
        done += len(chunk)
        print(f"[向量] 进度 {done}/{len(rows)}")
    still = len(missing_vector_rows(conn))
    print(f"[向量] 本轮补嵌 {done} 条，剩余缺向量 {still} 条" + ("（全部完成）" if still == 0 else "（重跑续补）"))
    return done


def _embed_openai(base: str, key: str, model: str, texts: list[str]) -> list[list[float]]:
    body = json.dumps({"model": model, "input": texts, "encoding_format": "float"}).encode()
    req = urllib.request.Request(
        f"{base.rstrip('/')}/embeddings", data=body,
        headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=60) as r:
        data = json.loads(r.read())
    return [d["embedding"] for d in sorted(data["data"], key=lambda x: x["index"])]


def main() -> int:
    ap = argparse.ArgumentParser(description="好想记住你 记忆导入/导出")
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("export", "import"):
        sub_parser = sub.add_parser(name)
        sub_parser.add_argument("--db", required=True)
        sub_parser.add_argument("--config", default=None,
                                help="AstrBot 主配置路径（或设 ASTRBOT_CMD_CONFIG 环境变量）")
    ex = sub.choices["export"]
    ex.add_argument("--out", required=True)
    im = sub.choices["import"]
    im.add_argument("--in", dest="inp", required=True)
    im.add_argument("--vectorize", action="store_true",
                    help="导入后给缺向量的记忆补嵌（读 AstrBot 主配置的 key）")
    im.add_argument("--provider-id", default="",
                    help="补嵌用的嵌入提供商 id（必填；不传时列出配置内可用候选）")
    im.add_argument("--no-preserve-ts", action="store_true",
                    help="不回填原始观察时间（全部按导入时间记）")
    args = ap.parse_args()

    global CFG_PATH
    if getattr(args, "config", None):
        CFG_PATH = Path(args.config)

    if args.cmd == "export":
        return do_export(Path(args.db), Path(args.out))
    return do_import(Path(args.db), Path(args.inp),
                     vectorize=args.vectorize, preserve_ts=not args.no_preserve_ts,
                     provider_id=getattr(args, "provider_id", "") or "")


if __name__ == "__main__":
    raise SystemExit(main())
