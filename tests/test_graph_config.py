"""星图缓存指纹 与 配置白名单校验 的回归测试（2026-09-16 审查发现）。"""

from __future__ import annotations

import pytest


class _FakePlugin:
    """graph.py / web_api 配置端点所需的最小插件替身。"""

    def __init__(self, store, paths, schema=None, astrbot_cfg=None):
        self.store = store
        self.paths = paths
        self.astrbot_config = astrbot_cfg

    class config:
        pass


@pytest.fixture()
def graph_plugin(tmp_path, fake_embedder):
    from core import db as dbm
    from core.paths import DataPaths
    from core.store import MemoryStore

    paths = DataPaths(tmp_path / "pdgraph").ensure()
    conn = dbm.connect(paths.db)
    dbm.init_schema(conn)
    store = MemoryStore(conn)
    yield _FakePlugin(store, paths), store, conn
    conn.close()


class TestGraphFingerprint:
    def test_recall_does_not_invalidate_cache(self, graph_plugin):
        """召回（每轮对话都发生）不得让星图缓存失效。

        缺陷背景：指纹原用 MAX(updated_at)，而 mark_recalled 会刷新 updated_at
        → 每次对话后打开星图都重算约 21 秒。
        """
        from core import graph
        plugin, store, conn = graph_plugin
        mid = store.add_memory("星图指纹测试记忆", scope="default")
        store.set_vector(mid, [0.1, 0.2, 0.3])
        fp1 = graph._fingerprint(plugin)

        store.mark_recalled([mid])           # 模拟一次召回
        fp2 = graph._fingerprint(plugin)
        assert fp1 == fp2, "召回不得改变结构指纹"

        store.update_memory(mid, strength=3.0)  # 衰减也会动 updated_at
        assert graph._fingerprint(plugin) == fp1, "衰减记账不得改变指纹"

    def test_structure_change_invalidates(self, graph_plugin):
        """节点/向量/取代链变化时指纹必须变化（否则会读到过期连线）。"""
        from core import graph
        plugin, store, conn = graph_plugin
        m1 = store.add_memory("结构变化A", scope="default")
        store.set_vector(m1, [1.0, 0.0, 0.0])
        fp0 = graph._fingerprint(plugin)

        m2 = store.add_memory("结构变化B", scope="default")   # 新增节点
        assert graph._fingerprint(plugin) != fp0

        fp1 = graph._fingerprint(plugin)
        store.set_vector(m2, [0.0, 1.0, 0.0])                # 新增向量
        assert graph._fingerprint(plugin) != fp1

        fp2 = graph._fingerprint(plugin)
        store.supersede(m2, m1)                              # 演化链
        assert graph._fingerprint(plugin) != fp2

    def test_cache_roundtrip(self, graph_plugin):
        """指纹一致时缓存命中，不一致时视为过期。"""
        from core import graph
        plugin, store, conn = graph_plugin
        mid = store.add_memory("缓存往返测试", scope="default")
        store.set_vector(mid, [0.5, 0.5, 0.0])

        graph._compute_and_cache(plugin)     # 直接算一次并落盘
        assert graph._load_cache(plugin) is not None, "同指纹应命中缓存"

        store.add_memory("让缓存过期的新记忆", scope="default")
        assert graph._load_cache(plugin) is None, "结构变化后缓存应判定过期"

    def test_single_node_no_crash(self, graph_plugin):
        """只有 0~1 个向量时不得崩（点积循环边界）。"""
        from core import graph
        plugin, store, conn = graph_plugin
        assert graph._compute_edges(conn) == []
        mid = store.add_memory("单节点", scope="default")
        store.set_vector(mid, [1.0, 0.0, 0.0])
        assert graph._compute_edges(conn) == []


class TestConfigWhitelist:
    """config/save 只接受 schema 中声明过的键。"""

    def _plugin_with_schema(self, make_engine):
        eng, store, conn = make_engine()
        schema = {
            "provider_id": {"type": "string", "default": ""},
            "injection": {"type": "object", "items": {
                "enabled": {"type": "bool", "default": True},
                "token_budget": {"type": "int", "default": 800},
            }},
        }

        class Cfg:
            pass
        cfg = Cfg(); cfg.schema = schema

        class P:
            pass
        p = P(); p.store = store; p.astrbot_config = cfg
        return p, store, conn

    def test_allowed_keys_collected(self, make_engine):
        from core import web_api
        p, store, conn = self._plugin_with_schema(make_engine)
        try:
            allowed = web_api._allowed_config_keys(p)
            assert allowed == {"provider_id", "injection.enabled", "injection.token_budget"}
        finally:
            conn.close()

    async def test_unknown_key_rejected(self, make_engine, monkeypatch):
        from core import web_api
        p, store, conn = self._plugin_with_schema(make_engine)

        class Eng:
            pass
        p.config = Eng(); p.config._raw = {}

        async def body():
            return {"updates": {"injection.enabled": True, "hacked.key": 1}}
        monkeypatch.setattr(web_api, "_body", body)
        try:
            with pytest.raises(ValueError) as ei:
                await web_api._save_config(p)
            assert "hacked.key" in str(ei.value)
            assert p.config._raw == {}, "校验失败时不得写入任何键"
        finally:
            conn.close()

    async def test_known_keys_applied(self, make_engine, monkeypatch):
        from core import web_api
        p, store, conn = self._plugin_with_schema(make_engine)

        class Eng:
            pass
        p.config = Eng(); p.config._raw = {}

        async def body():
            return {"updates": {"injection.enabled": False, "injection.token_budget": 400}}
        monkeypatch.setattr(web_api, "_body", body)
        try:
            d = await web_api._save_config(p)
            assert d["applied"] == ["injection.enabled", "injection.token_budget"]
            assert p.config._raw["injection"]["enabled"] is False
            assert p.config._raw["injection"]["token_budget"] == 400
        finally:
            conn.close()

    async def test_schema_missing_is_fail_closed(self, make_engine, monkeypatch):
        """v0.1.1：取不到 schema 时**拒绝保存**——fail-open 等于白名单形同虚设，
        任意键都能写入并落盘（check_config_integrity 会把未知键永久留存）。"""
        from core import web_api
        p, store, conn = self._plugin_with_schema(make_engine)
        p.astrbot_config = None

        class Eng:
            pass
        p.config = Eng(); p.config._raw = {}

        async def body():
            return {"updates": {"anything.goes": 1}}
        monkeypatch.setattr(web_api, "_body", body)
        try:
            with pytest.raises(ValueError) as ei:
                await web_api._save_config(p)
            assert "fail-closed" in str(ei.value)
            assert p.config._raw == {}, "拒绝保存时不得写入任何键"
        finally:
            conn.close()
