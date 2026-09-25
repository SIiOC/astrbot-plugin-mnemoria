"""数据目录布局与时间工具。

全链路统一以 UTC 存时间戳（避免 naive/aware 混用引发的时区炸弹），
仅在展示层转本地时区。
"""

from __future__ import annotations

import time
from datetime import datetime, timezone
from pathlib import Path


def utc_now_iso() -> str:
    """当前 UTC 时间的 ISO8601 字符串（含时区后缀）。"""
    return datetime.now(timezone.utc).isoformat()


def utc_now_ts() -> float:
    """当前 UTC 时间戳（秒，浮点）。"""
    return time.time()


def to_local_str(iso_or_ts: str | float | None) -> str:
    """把 UTC 时间戳/ISO 串转为本地时区的可读字符串，供展示层使用。"""
    if iso_or_ts is None:
        return ""
    try:
        if isinstance(iso_or_ts, (int, float)):
            dt = datetime.fromtimestamp(float(iso_or_ts), tz=timezone.utc)
        else:
            dt = datetime.fromisoformat(str(iso_or_ts))
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
    except (ValueError, TypeError, OSError):
        return str(iso_or_ts)
    return dt.astimezone().strftime("%Y-%m-%d %H:%M:%S")


class DataPaths:
    """插件数据目录布局，集中在 <plugin_data>/<plugin_name>/ 之下。"""

    def __init__(self, root: Path | str) -> None:
        self.root = Path(root)
        self.db = self.root / "mnemoria.db"
        self.backups = self.root / "backups"
        self.state = self.root / "state"
        self.logs = self.root / "logs"
        # 配置版本等元信息（不改动 AstrBot 的 config schema 文件）
        self.meta = self.state / "meta.json"
        self.ledger_dir = self.root / "ledger"

    def ensure(self) -> "DataPaths":
        """创建所有必需目录（幂等）。"""
        for d in (self.root, self.backups, self.state, self.logs, self.ledger_dir):
            d.mkdir(parents=True, exist_ok=True)
        return self

    def backup_path(self, stamp: str | None = None) -> Path:
        stamp = stamp or datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
        return self.backups / f"memories-{stamp}.json"

    def schema_backup_path(self, version: int, stamp: str | None = None) -> Path:
        """返回 schema 升级前的一致性 SQLite 备份路径。"""
        stamp = stamp or datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
        return self.backups / f"pre-schema-v{int(version)}-{stamp}.db"
