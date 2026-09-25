"""存量记忆 reasoning（依据）补写（v0.2.3 配套工具，只用真实语料）。

reasoning 是「这条记忆的对话依据」，是证据字段——**没有原始对话就绝不编造**。
本脚本只做一件事：给 reasoning 为空的活性记忆，从**账本里仍留存的真实原话**
中取最相关一条，写成 `日期 对话原话：「…」` 形式的依据。

两条匹配通道（2026-09-22 按线上实况扩充）：
1. **同会话**：记忆带 session_id 时，在该会话的流水里找最相关原话（相似度
   ≥ min_match 0.15）；
2. **时间窗推定**：angel 迁移来的遗产记忆 session_id 为空（线上实测 87%），
   同会话通道对它们恒不命中。退而求其次：在记忆 created_at ±30 分钟的
   **用户**流水里按内容相似度找原话，阈值提高到 window_min_match 0.25
   （邻近是弱证据，必须更像才采信），且依据里显式标注「（时间窗推定）」，
   不让推定来源伪装成同会话来源。

- 账本只保留近期流水（线上实测跨度约 6 天）：更早的记忆无源可考，一律跳过；
- 与记忆内容文本相似度低于阈值的流水不采信（防张冠李戴）；
- 已有 reasoning 的一律不碰；单条截断到 80 字，保持字段精简。

安全约定：默认 dry-run；--apply 先落全量 JSON 快照；单事务；幂等。

用法：
    python scripts/backfill_reasoning.py --db <mnemoria.db>            # 预览
    python scripts/backfill_reasoning.py --db <...> --apply           # 执行
    python scripts/backfill_reasoning.py --db <...> --limit 20        # 只处理前 20 条
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
from core import admission  # noqa: E402
from core.paths import utc_now_ts  # noqa: E402

#: 依据字段的最大长度（原话截断）
_MAX_REASONING = 80


def _session_turns(conn: sqlite3.Connection, session_id: str,
                   center_ts: float = 0.0, window_seconds: float = 1800.0,
                   limit: int = 400) -> list[dict]:
    """某会话的流水。传入 center_ts 时只取 created_at 邻近窗口内的行。

    ⚠️ 2026-09-22 修复：此前是 `ORDER BY id DESC LIMIT 60` 的盲截断——
    账本一长（线上 475 行），早期记忆的原话就滑出窗口，同会话通道对
    「几天前写入的记忆」恒不命中（实测 2 条 sim 0.25/0.30 的合格匹配被漏掉）。
    改为按记忆创建时间取邻近窗口；center_ts 缺失时才退化为最近的 limit 行。
    """
    if not session_id:
        return []
    try:
        if center_ts:
            rows = conn.execute(
                "SELECT id, role, content, ts FROM ledger WHERE session_id=? "
                "AND ts BETWEEN ? AND ? ORDER BY ABS(ts - ?) LIMIT ?",
                (session_id, float(center_ts) - window_seconds,
                 float(center_ts) + window_seconds, float(center_ts),
                 int(limit))).fetchall()
        else:
            rows = conn.execute(
                "SELECT id, role, content, ts FROM ledger WHERE session_id=? "
                "ORDER BY id DESC LIMIT ?", (session_id, int(limit))).fetchall()
    except sqlite3.Error:
        return []
    return [dict(r) for r in rows]


def _window_turns(conn: sqlite3.Connection, ts: float,
                  window_seconds: float = 1800.0, limit: int = 200) -> list[dict]:
    """created_at 邻近窗口内的用户流水（遗产记忆无 session 时的推定通道）。"""
    if not ts:
        return []
    try:
        rows = conn.execute(
            "SELECT id, role, content, ts FROM ledger "
            "WHERE role='user' AND ts BETWEEN ? AND ? "
            "ORDER BY ABS(ts - ?) LIMIT ?",
            (float(ts) - window_seconds, float(ts) + window_seconds,
             float(ts), int(limit))).fetchall()
    except sqlite3.Error:
        return []
    return [dict(r) for r in rows]


def plan(conn: sqlite3.Connection, *, limit: int = 0,
         min_match: float = 0.15, window_min_match: float = 0.25,
         window_seconds: float = 1800.0) -> list[dict]:
    """挑出 reasoning 为空的活性记忆，尽量从账本找到原话依据。"""
    sql = ("SELECT id, session_id, content, scope, created_at FROM memories "
           "WHERE deleted_at IS NULL AND (reasoning IS NULL OR reasoning='')")
    if limit:
        sql += f" LIMIT {int(limit)}"
    out: list[dict] = []
    for r in conn.execute(sql).fetchall():
        content = str(r["content"] or "")
        if not content:
            continue
        best = None
        best_sim = 0.0
        via_window = False
        for turn in _session_turns(conn, str(r["session_id"] or ""),
                                   center_ts=float(r["created_at"] or 0.0),
                                   window_seconds=window_seconds):
            text = str(turn["content"] or "")
            if not text:
                continue
            sim = admission.text_similarity(content, text)
            if sim > best_sim:
                best_sim, best = sim, turn
        # 通道2：无 session（或同会话没匹配上）时走时间窗推定，阈值更严
        if best is None and not str(r["session_id"] or "").strip():
            for turn in _window_turns(conn, float(r["created_at"] or 0.0),
                                      window_seconds=window_seconds):
                text = str(turn["content"] or "")
                if not text:
                    continue
                sim = admission.text_similarity(content, text)
                if sim > best_sim:
                    best_sim, best, via_window = sim, turn, True
        floor = window_min_match if via_window else min_match
        if best is None or best_sim < floor:
            continue  # 没有来源不编造
        quote = str(best["content"] or "").strip()[:_MAX_REASONING]
        day = time.strftime("%Y-%m-%d", time.localtime(best["ts"] or time.time()))
        tag = "（时间窗推定）" if via_window else ""
        out.append({"id": r["id"], "scope": str(r["scope"] or ""),
                    "content": content, "match": round(best_sim, 3),
                    "via_window": via_window,
                    "reasoning": f"{day} 对话原话{tag}：「{quote}」"})
    return out


def apply(conn: sqlite3.Connection, items: list[dict], backup_dir: Path) -> Path:
    """先快照后写入；单事务；返回快照路径。"""
    backup_dir.mkdir(parents=True, exist_ok=True)
    snap = backup_dir / f"reasoning-backfill-{time.strftime('%Y%m%d-%H%M%S')}.json"
    snap.write_text(json.dumps({
        "created_at": time.time(),
        "reason": "pre-reasoning-backfill",
        "items": items,
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    now = utc_now_ts()
    conn.execute("BEGIN")
    try:
        for it in items:
            conn.execute(
                "UPDATE memories SET reasoning=?, updated_at=? WHERE id=?",
                (it["reasoning"], now, it["id"]),
            )
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    return snap


def main() -> int:
    ap = argparse.ArgumentParser(description="存量记忆 reasoning 补写（只用账本原话）")
    ap.add_argument("--db", required=True, help="mnemoria.db 路径")
    ap.add_argument("--apply", action="store_true", help="真正写入（默认只预览）")
    ap.add_argument("--limit", type=int, default=0, help="最多处理几条（0=全部）")
    ap.add_argument("--min-match", type=float, default=0.15,
                    help="同会话通道：原话与记忆的最低文本相似度")
    ap.add_argument("--window-min-match", type=float, default=0.25,
                    help="时间窗推定通道（无 session 的遗产记忆）的最低相似度，比同会话更严")
    ap.add_argument("--window-seconds", type=float, default=1800.0,
                    help="时间窗推定通道的半窗宽度（秒，默认 ±30 分钟）")
    args = ap.parse_args()

    db = Path(args.db)
    if not db.exists():
        raise SystemExit(f"数据库不存在: {db}")
    conn = dbm.connect(db)
    empty = conn.execute(
        "SELECT COUNT(*) FROM memories WHERE deleted_at IS NULL "
        "AND (reasoning IS NULL OR reasoning='')").fetchone()[0]
    items = plan(conn, limit=args.limit, min_match=args.min_match,
                 window_min_match=args.window_min_match,
                 window_seconds=args.window_seconds)
    print(f"reasoning 为空的活性记忆 {empty} 条；其中能在账本找到原话依据的 "
          f"{len(items)} 条（其余无源可考，按设计跳过——不编造依据）")
    for it in items[:8]:
        print(f"  - {it['content'][:36]}… <- {it['reasoning'][:46]}（匹配 {it['match']}）")
    if not args.apply:
        print("\ndry-run：未写入。确认无误后加 --apply 执行（会先落 JSON 快照）。")
        conn.close()
        return 0
    snap = apply(conn, items, db.parent / "backups")
    left = conn.execute(
        "SELECT COUNT(*) FROM memories WHERE deleted_at IS NULL "
        "AND (reasoning IS NULL OR reasoning='')").fetchone()[0]
    conn.close()
    print(f"\n已补写 {len(items)} 条；剩余空 reasoning {left} 条；快照: {snap.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
