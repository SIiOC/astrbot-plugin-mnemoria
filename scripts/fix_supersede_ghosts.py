"""存量「三不见」死数据修复（v0.2.7 再审配套，幂等、可恢复）。

背景（2026-09-22 全量审查发现）：夜间巩固路径 `_merge_cluster` 此前只
`supersede` 不 `trash`——被合并取代的旧条被挤出活性视图与检索面，却没进
回收站，于是**三不见**：活性列表没有、回收站没有、检索不到。既不生效也不
可恢复，随每次巩固无限堆积（线上实测积 489 条）。v0.2.7 已修代码路径
（supersede 后立即 trash），本脚本把存量此类行补送进回收站：

- 判定：`superseded_by IS NOT NULL AND deleted_at IS NULL`（水久腐死的行）；
- 动作：只补 `deleted_at`（+ updated_at），使其与 v0.2.7 之后的行为一致——
  进回收站、面板可见（v0.2.14 起点「彻底恢复」才清血缘链回检索面，
  普通「恢复」对已被取代条目是无效操作）、逾期由回收站保留策略物理清理；
- 绝不动 `superseded_by`/`valid_to`（血缘是真财产，恢复逻辑依赖它）；
- 默认 dry-run；--apply 先落 JSON 快照；单事务；幂等（跑第二遍为 0 条）。

用法：
    python scripts/fix_supersede_ghosts.py --db <mnemoria.db>          # 预览
    python scripts/fix_supersede_ghosts.py --db <...> --apply         # 执行
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
from core.paths import utc_now_ts  # noqa: E402


def plan(conn: sqlite3.Connection) -> list[dict]:
    """挑出「superseded 但未入回收站」的死数据行。"""
    rows = conn.execute(
        "SELECT id, scope, superseded_by, valid_to, substr(content,1,60) AS head "
        "FROM memories WHERE superseded_by IS NOT NULL AND deleted_at IS NULL "
        "ORDER BY valid_to ASC").fetchall()
    return [dict(r) for r in rows]


def apply(conn: sqlite3.Connection, items: list[dict], backup_dir: Path) -> Path:
    """先快照后写入；单事务；只补 deleted_at，不动血缘。"""
    backup_dir.mkdir(parents=True, exist_ok=True)
    snap = backup_dir / f"supersede-ghosts-fix-{time.strftime('%Y%m%d-%H%M%S')}.json"
    snap.write_text(json.dumps({
        "created_at": time.time(),
        "reason": "pre-supersede-ghosts-fix",
        "count": len(items),
        "items": items,
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    now = utc_now_ts()
    conn.execute("BEGIN")
    try:
        for it in items:
            # deleted_at 取 now（而非 valid_to）：给用户一个完整的回收站
            # 保留期来人工复核/恢复，而不是一入回收站就逾期物理删除
            conn.execute(
                "UPDATE memories SET deleted_at=?, updated_at=? WHERE id=?",
                (now, now, it["id"]),
            )
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    return snap


def main() -> int:
    ap = argparse.ArgumentParser(
        description="把「superseded 但未入回收站」的死数据补送进回收站")
    ap.add_argument("--db", required=True, help="mnemoria.db 路径")
    ap.add_argument("--apply", action="store_true", help="真正写入（默认只预览）")
    ap.add_argument("--limit", type=int, default=0, help="最多处理几条（0=全部）")
    args = ap.parse_args()

    db = Path(args.db)
    if not db.exists():
        raise SystemExit(f"数据库不存在: {db}")
    conn = dbm.connect(db)
    items = plan(conn)
    if args.limit:
        items = items[: int(args.limit)]
    print(f"「superseded 但未入回收站」的死数据 {len(items)} 条"
          f"（将只补 deleted_at 送入回收站，血缘链保持不动）")
    for it in items[:5]:
        print(f"  - [{it['id'][:8]}] {it['head']}")
    if not args.apply:
        print("\ndry-run：未写入。确认无误后加 --apply 执行（会先落 JSON 快照）。")
        conn.close()
        return 0
    snap = apply(conn, items, db.parent / "backups")
    left = conn.execute(
        "SELECT COUNT(*) FROM memories WHERE superseded_by IS NOT NULL "
        "AND deleted_at IS NULL").fetchone()[0]
    conn.close()
    print(f"\n已修复 {len(items)} 条；剩余死数据 {left} 条；快照: {snap.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
