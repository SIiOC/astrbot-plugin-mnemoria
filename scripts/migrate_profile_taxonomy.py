"""画像维度归并（v0.2.1，借鉴 angel_memory 的固定画像体系）。

把历史漂移的画像键（喜好/兴趣/使用习惯/关系状态…）按固定五维归并：
- 默认 dry-run，只打印计划，不动库；
- `--apply` 才写入，且**先落全量 JSON 快照**到 `<数据目录>/backups/`；
- 只合并映射表内的键；未识别的键原样保留；不碰记忆/笔记/账本。

用法：
    python scripts/migrate_profile_taxonomy.py --db <mnemoria.db>            # 预览
    python scripts/migrate_profile_taxonomy.py --db <mnemoria.db> --apply    # 执行
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
from core import profile_taxonomy as ptax  # noqa: E402


def plan(conn: sqlite3.Connection):
    """返回 (合并计划, 未识别保留行)。

    合并计划：{(scope, user_key, 固定维度): [rows...]}，同一维度的固定键
    与全部同义键都纳入同一组统一归并。
    """
    rows = conn.execute(
        "SELECT scope, user_key, key, value, confidence, updated_at FROM profiles"
    ).fetchall()
    groups: dict[tuple[str, str, str], list] = {}
    untouched: list = []
    for r in rows:
        canonical = ptax.normalize_profile_key(str(r["key"]))
        if not ptax.is_canonical(canonical):
            untouched.append(r)
            continue
        groups.setdefault((r["scope"], r["user_key"], canonical), []).append(r)
    return groups, untouched


def apply(conn: sqlite3.Connection, groups: dict, backup_dir: Path):
    """先快照后归并；返回 (快照路径, 归并组数)。"""
    backup_dir.mkdir(parents=True, exist_ok=True)
    snap = backup_dir / f"profiles-pre-taxonomy-{time.strftime('%Y%m%d-%H%M%S')}.json"
    all_rows = [dict(r) for r in conn.execute("SELECT * FROM profiles").fetchall()]
    snap.write_text(json.dumps({
        "created_at": time.time(),
        "reason": "pre-profile-taxonomy-migration",
        "profiles": all_rows,
    }, ensure_ascii=False, indent=2), encoding="utf-8")

    merged = 0
    for (scope, user_key, canonical), rows in groups.items():
        # 已是唯一固定键 → 无需动
        if len(rows) == 1 and str(rows[0]["key"]) == canonical:
            continue
        ordered = sorted(rows, key=lambda r: float(r["updated_at"] or 0), reverse=True)
        values: list[str] = []
        conf = 0.0
        for r in ordered:
            for piece in str(r["value"] or "").split("；"):
                piece = piece.strip()
                if piece and piece not in values:
                    values.append(piece)
            conf = max(conf, float(r["confidence"] or 0.0))
        text = "；".join(values)
        if len(text) > 4000:
            text = text[:4000]
            print(f"  [提示] {canonical} 合并后超 4000 字已截断，请到面板确认")
        # 先删同组全部旧行（含固定键行），再写归并后的新值，避免自删
        for r in rows:
            conn.execute(
                "DELETE FROM profiles WHERE scope=? AND user_key=? AND key=?",
                (r["scope"], r["user_key"], r["key"]),
            )
        conn.execute(
            "INSERT INTO profiles(scope, user_key, key, value, confidence, updated_at) "
            "VALUES(?,?,?,?,?,?)",
            (scope, user_key, canonical, text, conf, time.time()),
        )
        merged += 1
    conn.commit()
    return snap, merged


def main() -> int:
    ap = argparse.ArgumentParser(description="画像维度归并（固定五维）")
    ap.add_argument("--db", required=True, help="mnemoria.db 路径")
    ap.add_argument("--apply", action="store_true", help="真正写入（默认只预览）")
    args = ap.parse_args()

    db = Path(args.db)
    if not db.exists():
        raise SystemExit(f"数据库不存在: {db}")
    conn = dbm.connect(db)
    groups, untouched = plan(conn)
    touched_groups = 0
    for (scope, user_key, canonical), rows in sorted(groups.items()):
        if len(rows) == 1 and str(rows[0]["key"]) == canonical:
            continue
        touched_groups += 1
        print(f"[{scope}/{user_key}] {canonical} ← " + "、".join(str(r["key"]) for r in rows))
    print(f"计划归并 {touched_groups} 组；未识别保留 {len(untouched)} 行（原样不动）")
    if not args.apply:
        print("dry-run：未写入。确认无误后加 --apply 执行（会先落 JSON 快照）。")
        conn.close()
        return 0
    snap, merged = apply(conn, groups, db.parent / "backups")
    print(f"已归并 {merged} 组；执行前快照: {snap.name}")
    conn.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
