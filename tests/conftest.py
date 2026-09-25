"""pytest 全局夹具与防护。

关键防护：把工作目录切到临时目录 + 重定向插件数据目录，
避免 AstrBot 被以插件目录为 cwd 导入时误落 data/cmd_config.json
（该文件含 dashboard jwt_secret，历史上已误落过多次）。
"""

from __future__ import annotations

import asyncio
import inspect
import os
import sys
from pathlib import Path

import pytest

# 让测试能 import astrbot 与本插件
#   _AB_ROOT       : AstrBot 源码根（提供 astrbot 包）
#   _PLUGIN_DIR    : 插件自身目录（提供 core / tools / scripts 顶层导入）
#   _PLUGIN_PARENT : 插件目录的父级（提供 astrbot_plugin_mnemoria 包导入）
_AB_ROOT = os.environ.get("ASTRBOT_ROOT", "")
if not _AB_ROOT:
    raise SystemExit(
        "请先设置环境变量 ASTRBOT_ROOT 指向 AstrBot 源码根目录（含 astrbot 包），\n"
        "例如：set ASTRBOT_ROOT=C:\\path\\to\\AstrBot"
    )
_PLUGIN_DIR = str(Path(__file__).resolve().parents[1])
_PLUGIN_PARENT = str(Path(__file__).resolve().parents[2])
for p in (_AB_ROOT, _PLUGIN_DIR, _PLUGIN_PARENT):
    if p and p not in sys.path:
        sys.path.insert(0, p)


def pytest_configure(config):
    """注册 asyncio marker，避免未注册告警。"""
    config.addinivalue_line("markers", "asyncio: mark test as async (run by local hook)")


def pytest_pyfunc_call(pyfuncitem):
    """原生支持 async 测试函数（无需 pytest-asyncio，不污染宿主 venv）。

    仅接管协程函数；同步函数交回默认实现（返回 None）。
    """
    func = pyfuncitem.obj
    if not inspect.iscoroutinefunction(func):
        return None
    kwargs = {
        name: pyfuncitem.funcargs[name]
        for name in pyfuncitem._fixtureinfo.argnames
    }
    asyncio.run(func(**kwargs))
    return True


@pytest.fixture(autouse=True)
def _safe_cwd(tmp_path, monkeypatch):
    """每个用例都在临时目录里跑，且插件数据目录指向临时区。"""
    monkeypatch.chdir(tmp_path)
    try:
        from astrbot.core.star.star_tools import StarTools
        data_root = tmp_path / "plugin_data"
        data_root.mkdir(parents=True, exist_ok=True)
        monkeypatch.setattr(
            StarTools, "get_data_dir",
            classmethod(lambda cls, name=None: (data_root / (name or "p")).resolve()),
            raising=False,
        )
    except Exception:  # noqa: BLE001
        pass
    yield


@pytest.fixture
def fake_embedder():
    """确定性假嵌入：不同文本得到近正交向量（近似真实 embedding 的判别性）。

    刻意不做 bag-of-chars —— 那种向量会让模板化短句（"第1条"/"第10条"）
    余弦高达 0.9+，引发非真实的去重合并，掩盖问题。
    相同文本必然得到相同向量；据此可测「指纹去重」路径。
    """
    class _E:
        enabled = True

        @staticmethod
        def _vec(text: str):
            import hashlib
            dim = 32
            h = hashlib.sha256(text.strip().encode("utf-8")).digest()
            v = [0.0] * dim
            for i, byte in enumerate(h[:dim]):
                v[i] = (byte / 255.0) - 0.5
            return v

        async def embed_one(self, text, timeout=None):
            return self._vec(text)

        async def embed(self, texts):
            return [self._vec(t) for t in texts]
    return _E()


@pytest.fixture
def make_engine(tmp_path, fake_embedder):
    """构造 (engine, store, conn)，可选注入假 LLM。"""
    from core import config as cfgmod
    from core import db as dbm
    from core.engine import MemoryEngine
    from core.paths import DataPaths
    from core.store import MemoryStore

    def _make(conf=None, payload=None, embedder="default"):
        paths = DataPaths(tmp_path / "pd").ensure()
        conn = dbm.connect(paths.db)
        dbm.init_schema(conn)
        store = MemoryStore(conn)
        cfg = cfgmod.Config(conf or {}, paths.meta)

        class _LLM:
            enabled = payload is not None

            async def generate_json(self, prompt, system_prompt=None):
                return payload

            async def generate(self, prompt, system_prompt=None):
                return ""

        eng = MemoryEngine(
            store, cfg,
            embedder=fake_embedder if embedder == "default" else embedder,
            reranker=None,
            llm=_LLM() if payload is not None else None,
        )
        return eng, store, conn

    return _make


@pytest.fixture
def plugin(tmp_path, monkeypatch):
    """构造真实插件实例（数据目录已由 _safe_cwd 重定向）。"""
    class Ctx:
        class provider_manager:
            @staticmethod
            async def get_provider_by_id(p):
                return None

        def __init__(self):
            self.tools = []
            self.routes = []

        def add_llm_tools(self, *tools):
            self.tools.extend(tools)

        def register_web_api(self, route, handler, methods, desc):
            self.routes.append((route, handler, methods, desc))

    from astrbot_plugin_mnemoria.main import MnemoriaPlugin
    ctx = Ctx()
    inst = MnemoriaPlugin(ctx, config={})
    inst._test_ctx = ctx
    return inst
