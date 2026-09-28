"""配置读取与版本迁移。

AstrBot 的配置由 _conf_schema.json 驱动，升级插件时旧配置可能缺键或多键。
这里统一做三件事：
1) 用默认值补齐缺失键（dotted get，避免 KeyError 全家桶——mnemosyne #148/#151 的教训）；
2) 记录 schema 版本号，为将来的键改名/删键提供一次性迁移钩子；
3) 迁移状态落在数据目录 state/meta.json，不污染 AstrBot 的配置 schema。
"""

from __future__ import annotations

import json
from astrbot.api import logger
from pathlib import Path
from typing import Any


# 当前配置 schema 版本。每次破坏性改键（改名/删键/改语义）时 +1，并在 _MIGRATIONS 里补一步。
CONFIG_SCHEMA_VERSION = 1

# schema_version -> 迁移函数(config: dict) -> dict
_MIGRATIONS: dict[int, Any] = {}


def _register(from_version: int):
    def deco(fn):
        _MIGRATIONS[from_version] = fn
        return fn
    return deco


# 示例（占位）：0 -> 1 无历史键，仅建立版本基线。
@_register(0)
def _migrate_0_to_1(config: dict) -> dict:
    return config


class Config:
    """围绕 AstrBot 传入的配置 dict 提供带默认值的只读访问。"""

    def __init__(self, raw: dict | None, meta_file: Path | None = None) -> None:
        # 持引用而非拷贝：AstrBot 保存配置时会就地修改同一个 dict，
        # 拷贝会导致热更新永远看不到新值（曾因此使注入/阈值改动不生效）。
        if isinstance(raw, dict):
            self._raw: dict = raw
        else:
            self._raw = {}
        self._meta_file = meta_file
        self._version = CONFIG_SCHEMA_VERSION
        self._apply_migrations()

    # ---- 迁移 ----------------------------------------------------------
    def _apply_migrations(self) -> None:
        stored = self._read_stored_version()
        if stored >= CONFIG_SCHEMA_VERSION:
            return
        version = stored
        chain_complete = True
        while version < CONFIG_SCHEMA_VERSION:
            fn = _MIGRATIONS.get(version)
            if fn is not None:
                try:
                    self._raw = fn(self._raw) or self._raw
                except Exception as exc:  # 迁移失败不得致命
                    logger.warning("配置迁移 %s->%s 失败，保留原配置: %s", version, version + 1, exc)
                    chain_complete = False
                    break
            version += 1
        # 只有迁移链走完才写新版本号：中途失败也标记到最新会让失败的
        # 步骤永远不再重试（每次启动都直接跳过迁移）
        if chain_complete:
            self._write_stored_version(CONFIG_SCHEMA_VERSION)

    def _read_stored_version(self) -> int:
        if not self._meta_file or not self._meta_file.exists():
            return 0
        try:
            data = json.loads(self._meta_file.read_text(encoding="utf-8"))
            return int(data.get("config_schema_version", 0))
        except (OSError, ValueError, TypeError):
            return 0

    def _write_stored_version(self, version: int) -> None:
        if not self._meta_file:
            return
        try:
            self._meta_file.parent.mkdir(parents=True, exist_ok=True)
            data: dict = {}
            if self._meta_file.exists():
                try:
                    data = json.loads(self._meta_file.read_text(encoding="utf-8"))
                except (OSError, ValueError):
                    data = {}
            data["config_schema_version"] = version
            tmp = self._meta_file.with_suffix(".tmp")
            tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
            tmp.replace(self._meta_file)
        except OSError as exc:
            logger.debug("写入配置版本失败（不影响运行）: %s", exc)

    # ---- 访问 ----------------------------------------------------------
    def get(self, path: str, default: Any = None) -> Any:
        """以 'a.b.c' 形式取值，缺失返回默认值。"""
        node: Any = self._raw
        for key in path.split("."):
            if not isinstance(node, dict) or key not in node:
                return default
            node = node[key]
        return node if node is not None else default

    def get_num(self, path: str, default: Any = None) -> Any:
        """数值配置取值：显式设置的 0 是合法值（如 penalty_ratio=0 不扣分、
        consolidation_text_floor=0 关闭文本守卫、inject_max_items=0 不注入），
        不能像 get(...) or default 那样把 0 当 falsy 吞掉；仅缺失/None/空串
        才回落默认。"""
        v = self.get(path, default)
        return default if v is None or v == "" else v

    # 常用便捷属性
    @property
    def provider_id(self) -> str:
        return str(self.get("provider_id", "") or "").strip()

    @property
    def embedding_provider_id(self) -> str:
        return str(self.get("retrieval.embedding_provider_id", "") or "").strip()

    @property
    def rerank_provider_id(self) -> str:
        return str(self.get("retrieval.rerank_provider_id", "") or "").strip()

    @property
    def schema_version(self) -> int:
        return self._version
