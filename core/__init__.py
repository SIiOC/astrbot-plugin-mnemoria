"""好想记住你（mnemoria）插件核心包。

仅依赖 AstrBot 框架抽象与 Python 标准库：
- 存储：SQLite(WAL) + FTS5
- 向量：纯 Python 暴力余弦（无 numpy/faiss）
- 无跨插件引用，与 Humanizer 等插件零耦合
"""

__version__ = "0.1.0"

__all__ = ["__version__"]
