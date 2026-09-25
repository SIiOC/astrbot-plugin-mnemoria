"""独立测试脚本的共享引导。

必须在 `import astrbot` 之前导入本模块：

    from _bootstrap import bootstrap  # noqa: F401
    bootstrap()

作用有两件，缺一不可：
1. 把 AstrBot 源码根与插件包父目录加入 sys.path；
2. **把进程工作目录切到临时目录** —— AstrBot 初始化时会往 cwd 写
   `data/cmd_config.json`（内含 dashboard jwt_secret / 口令哈希）。
   若以插件目录为 cwd 运行，敏感文件就会落进插件源码树（历史上已误落多次）。
"""

from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path

_BOOTSTRAPPED = False
_TMP_DIR: str | None = None


def bootstrap(plugin_dir: Path | None = None) -> str:
    """返回实际使用的临时工作目录。幂等，可重复调用。"""
    global _BOOTSTRAPPED, _TMP_DIR

    if plugin_dir is None:
        plugin_dir = Path(__file__).resolve().parents[1]

    ab_root = os.environ.get(
        "ASTRBOT_IMPORT_ROOT", os.environ.get("ASTRBOT_ROOT", "")
    )
    for p in (ab_root, str(plugin_dir), str(plugin_dir.parent)):
        if p and p not in sys.path:
            sys.path.insert(0, p)

    if not _BOOTSTRAPPED:
        _TMP_DIR = tempfile.mkdtemp(prefix="mnemoria-test-")
        # 双保险：
        #  1) cwd 切到临时目录（AstrBot 会往 cwd 写 data/）；
        #  2) ASTRBOT_ROOT 指向临时目录 —— 框架的 data 路径由它推导
        #     （astrbot_path.get_astrbot_root() 读这个变量）。
        #     用 ASTRBOT_ROOT 而非自造变量名，否则是假保护。
        os.chdir(_TMP_DIR)
        os.environ["ASTRBOT_ROOT"] = _TMP_DIR
        _BOOTSTRAPPED = True
    return _TMP_DIR or os.getcwd()


# 导入即保护（幂等）：任何一次性调试脚本只要 `import _bootstrap`
# 就完成 chdir + sys.path 装配，不再依赖记得手动调用 bootstrap()。
bootstrap()
