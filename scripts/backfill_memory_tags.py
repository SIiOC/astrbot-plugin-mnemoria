"""存量记忆 tags 回填（v0.2.2）。

背景：v0.2.1 起抽取提示词用案例教学让模型输出 tags，新记忆覆盖率约 98%，
但此前写入的历史记忆大量没有 tags（线上实测约 90%）——这些记忆的标签
检索通道是瞎的。本脚本用 core.tags.derive_tags 纯规则派生 tags 回填。

安全约定（与画像归并脚本一致）：
- 默认 dry-run，只打印计划与抽样，不动库；
- `--apply` 才写入，且**先落全量 JSON 快照**到 `<数据目录>/backups/`；
- 只填 tags 为空的记忆；已有 tags 的一律不碰；
- 单事务提交，中途失败整体回滚。

用法：
    python scripts/backfill_memory_tags.py --db <mnemoria.db>            # 预览
    python scripts/backfill_memory_tags.py --db <mnemoria.db> --apply    # 执行
    python scripts/backfill_memory_tags.py --db <...> --limit 50         # 只处理前 50 条
"""

from __future__ import annotations

import argparse
import json
import sqlite3
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from core import db as dbm  # noqa: E402
from core.tags import derive_tags  # noqa: E402


def _identities(conn: sqlite3.Connection) -> list[str]:
    """从 user_ledger 收集已知身份名（昵称历史），作为身份锚点候选。"""
    out: list[str] = []
    try:
        rows = conn.execute("SELECT user_names FROM user_ledger").fetchall()
    except sqlite3.Error:
        return out
    for r in rows:
        try:
            names = json.loads(r["user_names"] or "[]")
        except (TypeError, ValueError):
            continue
        if isinstance(names, list):
            out.extend(str(n) for n in names if n)
    return list(dict.fromkeys(out))


def plan(conn: sqlite3.Connection, limit: int = 0) -> list[dict]:
    """挑出 tags 为空的活性记忆，算出派生 tags。返回待写清单。"""
    identities = _identities(conn)
    sql = ("SELECT id, content, memory_type FROM memories "
           "WHERE deleted_at IS NULL AND (tags_json IS NULL OR tags_json IN ('', '[]'))")
    if limit:
        sql += f" LIMIT {int(limit)}"
    out: list[dict] = []
    for r in conn.execute(sql).fetchall():
        tags = derive_tags(str(r["content"] or ""),
                           memory_type=str(r["memory_type"] or ""),
                           identities=identities)
        if tags:
            out.append({"id": r["id"], "content": str(r["content"] or ""),
                        "tags": tags})
    return out


def apply(conn: sqlite3.Connection, items: list[dict], backup_dir: Path) -> Path:
    """先快照后写入；单事务，返回快照路径。"""
    backup_dir.mkdir(parents=True, exist_ok=True)
    snap = backup_dir / f"tags-backfill-{time.strftime('%Y%m%d-%H%M%S')}.json"
    snap.write_text(json.dumps({
        "created_at": time.time(),
        "reason": "pre-tags-backfill",
        "items": items,
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    now = time.time()
    conn.execute("BEGIN")
    try:
        for it in items:
            conn.execute(
                "UPDATE memories SET tags_json=?, updated_at=? WHERE id=?",
                (json.dumps(it["tags"], ensure_ascii=False), now, it["id"]),
            )
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    return snap


def main() -> int:
    ap = argparse.ArgumentParser(description="存量记忆 tags 回填（规则派生）")
    ap.add_argument("--db", required=True, help="mnemoria.db 路径")
    ap.add_argument("--apply", action="store_true", help="真正写入（默认只预览）")
    ap.add_argument("--limit", type=int, default=0, help="最多处理几条（0=全部）")
    args = ap.parse_args()

    db = Path(args.db)
    if not db.exists():
        raise SystemExit(f"数据库不存在: {db}")
    conn = dbm.connect(db)
    total = conn.execute(
        "SELECT COUNT(*) FROM memories WHERE deleted_at IS NULL").fetchone()[0]
    empty = conn.execute(
        "SELECT COUNT(*) FROM memories WHERE deleted_at IS NULL "
        "AND (tags_json IS NULL OR tags_json IN ('', '[]'))").fetchone()[0]
    items = plan(conn, limit=args.limit)
    derived = sum(1 for _ in items)
    print(f"活性记忆 {total} 条，其中 tags 为空 {empty} 条"
          + (f"（本次limit={args.limit}）" if args.limit else ""))
    print(f"可派生 tags 的 {derived} 条；其余为空内容/纯噪声，跳过")
    for it in items[:8]:
        print(f"  - {it['content'][:40]}… -> {it['tags']}")
    if not args.apply:
        print("dry-run：未写入。确认无误后加 --apply 执行（会先落 JSON 快照）。")
        conn.close()
        return 0
    snap = apply(conn, items, db.parent / "backups")
    left = conn.execute(
        "SELECT COUNT(*) FROM memories WHERE deleted_at IS NULL "
        "AND (tags_json IS NULL OR tags_json IN ('', '[]'))").fetchone()[0]
    conn.close()
    print(f"已回填 {len(items)} 条；剩余 tags 为空 {left} 条；快照: {snap.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
