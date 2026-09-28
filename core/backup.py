"""JSON 备份与恢复（每日导出，保留 N 份）。"""

from __future__ import annotations

import json
from astrbot.api import logger
import sqlite3
from pathlib import Path

from .paths import DataPaths, to_local_str, utc_now_ts


_MEM_COLS = (
    "id, content, reasoning, memory_type, source, speaker, speaker_key, is_active, strength, "
    "useful_score, useful_count, hit_count, last_recalled_at, last_decay_at, "
    "proof_count, observed_at, valid_from, valid_to, superseded_by, deleted_at, "
    "quarantined, scope, session_id, tags_json, created_at, updated_at"
)

# vec/vec_dim 不导出（BLOB 可在导入侧按需回填）
_NOTE_COLS = (
    "id, title, content, tags, source, file_name, heading, scope, "
    "content_hash, deleted_at, created_at, updated_at"
)


def export_all(conn: sqlite3.Connection) -> dict:
    memories = [dict(r) for r in conn.execute(f"SELECT {_MEM_COLS} FROM memories").fetchall()]
    profiles = [dict(r) for r in conn.execute(
        "SELECT scope, user_key, key, value, confidence, updated_at FROM profiles"
    ).fetchall()]
    # v0.1.1：笔记并入备份（此前只导 memories+profiles，笔记导入/手写的内容
    # 不在备份里，误删即失守）
    notes = [dict(r) for r in conn.execute(f"SELECT {_NOTE_COLS} FROM notes").fetchall()]
    return {
        "exported_at": utc_now_ts(),
        "memories": memories,
        "profiles": profiles,
        "notes": notes,
    }


def write_backup(paths: DataPaths, conn: sqlite3.Connection, keep: int = 3) -> Path | None:
    """导出备份。任何失败都只记日志并返回 None——备份绝不能拖垮插件。"""
    try:
        paths.ensure()
        data = export_all(conn)
        target = paths.backup_path()
        tmp = target.with_suffix(".tmp")
        tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
        tmp.replace(target)
        _prune(paths.backups, keep)
        return target
    except (OSError, sqlite3.Error) as exc:
        logger.warning("备份写入失败: %s", exc)
        return None
    except Exception as exc:  # noqa: BLE001
        logger.warning("备份出现意外错误: %s", exc)
        return None


def _prune(dir_: Path, keep: int) -> None:
    files = sorted(dir_.glob("memories-*.json"), key=lambda p: p.stat().st_mtime, reverse=True)
    for old in files[max(1, keep):]:
        try:
            old.unlink()
        except OSError:
            pass


def list_backups(paths: DataPaths) -> list[dict]:
    out = []
    for p in sorted(paths.backups.glob("memories-*.json"), key=lambda x: x.stat().st_mtime, reverse=True):
        st = p.stat()
        out.append({"name": p.name, "size": st.st_size, "mtime": to_local_str(st.st_mtime)})
    return out
