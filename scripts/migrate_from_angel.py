"""从 angel_memory 导出记忆并转换为好想记住你可导入的 JSON。

用法（只读 angel 的库、只写文件，不动 angel 任何数据）：

    python scripts/migrate_from_angel.py \
        --angel-db "<AstrBot>/data/plugin_data/astrbot_plugin_angel_memory/memory_center/index/simple_memory.db" \
        --out angel_export.json \
        [--active-only]

【只读保证】本脚本以 SQLite 只读模式（mode=ro）打开 angel 数据库，
任何情况下不写、不改、不删 angel 的数据；停用 angel 插件也只是禁用加载，
其数据文件原样保留，可随时重新启用回滚。迁移是"导出副本"，不是"搬走"。

转换要点（angel -> mnemoria）：
- 正文列自动探测：content / judgment（angel 1.6.8 实际用 judgment）/ text …
- reasoning     -> reasoning
- memory_type   -> memory_type
- is_active     -> is_active（1 的记忆在好想记住你里同样永不衰减）
- strength / useful_score / useful_count -> 同名列
- 时间字段（若为 ISO 串）-> 统一转 UTC 秒
- proof_count 默认 1；scope 由 --scope 指定（默认 default）

注意：angel 的表结构随版本变动，本脚本会先探测列名再取值并打印；
正文列全部为空的行会跳过并计入 skipped，导出后请核对条数。
"""

from __future__ import annotations

import argparse
import json
import sqlite3
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def parse_ts(value) -> float:
    """把 angel 的时间值（可能是 ISO 串或数字）统一转为 UTC 秒。"""
    if value is None:
        return 0.0
    if isinstance(value, (int, float)):
        v = float(value)
        return v / 1000.0 if v > 1e11 else v  # 毫秒时间戳归一到秒
    text = str(value).strip()
    if not text:
        return 0.0
    try:
        dt = datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError:
        return 0.0
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.timestamp()


def table_columns(conn: sqlite3.Connection, table: str) -> list[str]:
    try:
        rows = conn.execute(f"PRAGMA table_info({table})").fetchall()
    except sqlite3.Error:
        return []
    return [r[1] for r in rows]


# 正文可能所在的列（按优先级）。angel 各版本不同：
# 部分版本用 content，但实测 1.6.8 的 memory_records 表正文在 judgment 列。
_CONTENT_CANDIDATES = ("content", "judgment", "text", "memory", "summary")


def find_memory_table(conn: sqlite3.Connection) -> str | None:
    rows = conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table'"
    ).fetchall()
    candidates = [r[0] for r in rows if "memor" in r[0].lower()]
    # 优先选带任一正文候选列的表
    for name in candidates:
        cols = table_columns(conn, name)
        if any(c in cols for c in _CONTENT_CANDIDATES):
            return name
    return candidates[0] if candidates else None


# angel 的 memory_type 是中文（实测 v1.6.8：知识/事件/情感/技能记忆），
# 好想记住你的分型 TTL 权重表按英文键工作——导出时归一化，否则迁移记忆全部落默认权重。
TYPE_MAP = {
    "知识记忆": "knowledge",
    "事件记忆": "event",
    "情感记忆": "emotional",
    "技能记忆": "skill",
    "任务记忆": "task",
    "事实记忆": "fact",
}


def map_type(raw: str) -> str:
    raw = (raw or "").strip()
    return TYPE_MAP.get(raw, raw if raw in ("fact", "knowledge", "event", "skill", "emotional", "task") else "fact")


def pick_content(d: dict, cols: list[str]) -> str:
    """按候选顺序取第一个非空正文列。"""
    for c in _CONTENT_CANDIDATES:
        if c in cols:
            v = str(d.get(c) or "").strip()
            if v:
                return v
    return ""


def to_epoch(value, cols):
    """按可用列挑一个时间字段转 UTC 秒。"""
    for key in ("created_at", "updated_at", "created", "timestamp"):
        if key in cols and value.get(key) is not None:
            return parse_ts(value[key])
    return 0.0


def main() -> int:
    ap = argparse.ArgumentParser(description="从 angel_memory 导出记忆为好想记住你导入格式")
    ap.add_argument("--angel-db", required=True, help="angel 的 simple_memory.db 路径")
    ap.add_argument("--out", default="angel_export.json", help="输出 JSON 路径")
    ap.add_argument("--scope", default="default", help="导入到好想记住你的隔离域")
    ap.add_argument("--active-only", action="store_true", help="只导出主动记忆（永不衰减的那批）")
    args = ap.parse_args()

    db_path = Path(args.angel_db)
    if not db_path.exists():
        print(f"[错误] 找不到 angel 数据库：{db_path}", file=sys.stderr)
        return 2

    conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    conn.row_factory = sqlite3.Row
    try:
        table = find_memory_table(conn)
        if not table:
            print("[错误] 未在 angel 库中找到记忆表", file=sys.stderr)
            return 3
        cols = table_columns(conn, table)
        print(f"[信息] 使用表 {table}，可用列：{', '.join(cols)}")

        sql = f"SELECT * FROM {table}"
        if args.active_only and "is_active" in cols:
            sql += " WHERE is_active=1"
        rows = conn.execute(sql).fetchall()

        out = []
        skipped = 0
        for r in rows:
            d = dict(r)
            # 正文列按候选顺序探测（angel 1.6.8 实际用 judgment 列）
            content = pick_content(d, cols)
            if not content:
                skipped += 1
                continue
            out.append({
                "content": content,
                "reasoning": str(d.get("reasoning") or ""),
                "memory_type": map_type(str(d.get("memory_type") or "")),
                "is_active": int(d.get("is_active") or 0),
                "strength": float(d.get("strength") or 10.0),
                "useful_score": float(d.get("useful_score") or 0.0),
                "useful_count": int(d.get("useful_count") or 0),
                "proof_count": int(d.get("proof_count") or 1),
                "observed_at": to_epoch(d, cols),
                "scope": args.scope,
                "source": "user",
            })

        payload = {
            "source": "angel_memory",
            "source_db": str(db_path),
            "table": table,
            "scope": args.scope,
            "count": len(out),
            "skipped": skipped,
            "active_count": sum(1 for x in out if x["is_active"]),
            "memories": out,
        }
        Path(args.out).write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"[完成] 导出 {len(out)} 条（其中主动 {payload['active_count']} 条，"
              f"正文为空跳过 {skipped} 条）→ {args.out}")
        print("[下一步] 在好想记住你插件页或命令行执行导入（导入函数见 scripts/import_export.py）")
        print("[提示] angel 原库全程只读未动；确认好想记住你运行正常前请勿删除 angel 数据目录")
        return 0
    finally:
        conn.close()


if __name__ == "__main__":
    raise SystemExit(main())
