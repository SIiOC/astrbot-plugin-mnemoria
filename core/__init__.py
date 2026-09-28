"""好想记住你（mnemoria）插件核心包。

仅依赖 AstrBot 框架抽象与 Python 标准库：
- 存储：SQLite(WAL) + FTS5
- 向量：纯 Python 暴力余弦（无 numpy/faiss）
- 无跨插件引用，与 Humanizer 等插件零耦合
"""

from __future__ import annotations

import os
import sys

# CLI 脚本（scripts/）脱离框架直跑时没有 AstrBot 进程：astrbot 包在
# AstrBot 源码根而不在 venv 里。core 各模块统一 `from astrbot.api
# import logger`（插件审查规范：logger 必须且只能来自 astrbot.api），
# 因此这里在导入任何 core.* 之前把 AstrBot 根装配进 sys.path；
# 未提供环境变量时给出可操作报错而不是裸 ModuleNotFoundError。
_AB_ROOT = (os.environ.get("ASTRBOT_IMPORT_ROOT")
            or os.environ.get("ASTRBOT_ROOT") or "").strip()
if _AB_ROOT and _AB_ROOT not in sys.path:
    sys.path.insert(0, _AB_ROOT)
try:
    import astrbot  # noqa: F401
except ImportError:
    raise SystemExit(
        "core 模块的 logger 依赖 astrbot.api：请设置 ASTRBOT_ROOT 指向 "
        "AstrBot 源码根（含 astrbot 包）后重跑，"
        "例如 set ASTRBOT_ROOT=<你的 AstrBot 目录>"
    )

__version__ = "0.1.0"

__all__ = ["__version__"]
