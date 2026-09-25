"""tags 规则派生与存量回填测试（v0.2.2）。

覆盖：
- core.tags.derive_tags：身份锚点 / 「」引用词 / 画像维度 / 拉丁 token /
  场合词同义词 / 限量去重 / 空内容；
- store.set_tags：只改 tags_json，不动其它列；
- 回填脚本：dry-run 不写、--apply 先快照后写、幂等、不碰已有 tags；
- 抽取回退：模型漏输出 tags 时落库的不是空数组。
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

import pytest

from core import db as dbm
from core.store import MemoryStore
from core.tags import (derive_tags, identities_from_label,
                       normalize_tags)

PLUGIN_ROOT = Path(__file__).resolve().parents[1]


def _mk_store(tmp_path):
    conn = dbm.connect(tmp_path / "t.db")
    dbm.init_schema(conn)
    return MemoryStore(conn), conn


# ---------------------------------------------------------------- 1. 规则派生
class TestDeriveTags:

    def test_identity_anchor_from_param_and_text(self):
        tags = derive_tags("Star Lantern（3141592653）说他周末在跑步",
                           identities=["Star Lantern"])
        assert "Star Lantern" in tags, "调用方给的已知身份名要作为锚点"
        assert "3141592653" in tags, "正文里的数字 ID 要作为锚点"

    def test_wechat_openid_anchor(self):
        tags = derive_tags("用户模z0xw804vjXb1VAlfuUxb-TY2hSYM@im.wechat在问插件")
        assert "z0xw804vjXb1VAlfuUxb-TY2hSYM@im.wechat" in tags

    def test_date_like_number_is_not_identity(self):
        tags = derive_tags("20260920 那天聊了跑步")
        assert "20260920" not in tags, "日期形态数字不是身份锚点"

    def test_quoted_terms(self):
        tags = derive_tags("他要求我「设成主动记忆」并且叫他「主理人」")
        assert "设成主动记忆" in tags and "主理人" in tags

    def test_profile_dim_words(self):
        assert "技能树" in derive_tags("他会用 mimo desktop 软件，还在学新工具")
        assert "关系图谱" in derive_tags("两人的互动风格是调侃中带亲密")
        assert "活跃项目" in derive_tags("他正在推进真人感回复训练项目")

    def test_latin_tokens(self):
        tags = derive_tags("测试模型 ox 是智谱的 glm5.3turbo，也试过 qwen-image-3.0-pro")
        assert "glm5.3turbo" in tags and "qwen-image-3.0-pro" in tags

    def test_occasion_synonyms(self, monkeypatch):
        """发布版 _OCCASION_LEXICON 为空——注入临时词典测机制本身。"""
        import core.tags as tags_mod
        monkeypatch.setattr(tags_mod, "_OCCASION_LEXICON",
                            ((("跑步",), ("运动", "跑步")),))
        tags = derive_tags("他周末晚上在家跑步")
        assert "运动" in tags, "正文写跑步，查询可能说运动"

    def test_limit_and_dedupe(self, monkeypatch):
        """发布版 _OCCASION_LEXICON 为空——注入临时词典测 limit/去重机制。"""
        import core.tags as tags_mod
        monkeypatch.setattr(tags_mod, "_OCCASION_LEXICON", (
            (("跑步",), ("运动", "跑步")),
            (("古筝",), ("乐器", "古筝")),
            (("插件",), ("插件", "配置")),
            (("语音",), ("语音", "tts")),
            (("语料",), ("训练", "语料")),
        ))
        content = ("Star Lantern（3141592653）用「A」「B」「C」「D」「E」"
                   "跑步、古筝、英语、插件、图片、语音、记忆、语料……")
        tags = derive_tags(content, identities=["Star Lantern"], limit=5)
        assert len(tags) == 5
        assert len(set(tags)) == len(tags)

    def test_empty_content(self):
        assert derive_tags("") == []
        assert derive_tags("   ") == []

    def test_identities_from_label(self):
        assert identities_from_label("Star Lantern（3141592653）") == \
            ["Star Lantern", "3141592653"]
        assert identities_from_label("用户（u1）") == [], "占位指称不当锚点"

    def test_normalize_tags(self):
        assert normalize_tags(["  a ", "a", "", "b", "c"], limit=2) == ["a", "b"]


# ---------------------------------------------------------------- 2. store 写入
class TestSetTags:

    def test_set_tags_roundtrip_and_isolated(self, tmp_path):
        store, conn = _mk_store(tmp_path)
        try:
            mid = store.add_memory("旧记忆没有 tags", scope="d")
            before = store.get_memory(mid)
            store.set_tags(mid, ["运动", "跑步", "运动"])  # 去重
            row = store.get_memory(mid)
            assert json.loads(row["tags_json"]) == ["运动", "跑步"]
            # 其它列不被碰
            assert row["content"] == before["content"]
            assert row["scope"] == before["scope"]
            assert row["strength"] == before["strength"]
        finally:
            conn.close()


# ---------------------------------------------------------------- 3. 回填脚本
def _load_script():
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "backfill_memory_tags", PLUGIN_ROOT / "scripts" / "backfill_memory_tags.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


class TestBackfillScript:

    def _seed(self, tmp_path):
        store, conn = _mk_store(tmp_path)
        a = store.add_memory("Star Lantern（3141592653）周末在家跑步", scope="d")
        b = store.add_memory("他会用 mimo desktop 软件配置插件", scope="d")
        c = store.add_memory("已有 tags 的旧记忆", scope="d")
        store.set_tags(c, ["保留"])
        return store, conn, (a, b, c)

    def test_dry_run_writes_nothing(self, tmp_path):
        store, conn, ids = self._seed(tmp_path)
        try:
            mod = _load_script()
            items = mod.plan(conn)
            assert {i["id"] for i in items} == {ids[0], ids[1]}
            assert not list((tmp_path / "backups").glob("tags-backfill-*.json"))
            assert json.loads(store.get_memory(ids[0])["tags_json"]) == []
        finally:
            conn.close()

    def test_apply_snapshots_and_fills(self, tmp_path):
        store, conn, ids = self._seed(tmp_path)
        try:
            mod = _load_script()
            items = mod.plan(conn)
            snap = mod.apply(conn, items, tmp_path / "backups")
            assert snap.exists(), "apply 前必须落快照"
            payload = json.loads(snap.read_text(encoding="utf-8"))
            assert len(payload["items"]) == 2
            assert json.loads(store.get_memory(ids[0])["tags_json"]) != []
            # 已有 tags 的不被覆盖
            assert json.loads(store.get_memory(ids[2])["tags_json"]) == ["保留"]
            # 幂等：再跑一次没有可填的
            assert mod.plan(conn) == []
        finally:
            conn.close()

    def test_main_dry_run_exit_zero(self, tmp_path, capsys):
        store, conn, _ = self._seed(tmp_path)
        conn.close()
        mod = _load_script()
        import sys
        old = sys.argv
        try:
            sys.argv = ["x", "--db", str(tmp_path / "t.db")]
            assert mod.main() == 0
            out = capsys.readouterr().out
            assert "dry-run" in out
        finally:
            sys.argv = old


# ---------------------------------------------------------------- 4. 抽取回退
class _FakeLLM:
    enabled = True
    provider_id = "fake"

    def __init__(self, payload):
        self.payload = payload

    async def generate_json(self, prompt, system_prompt=None):
        return self.payload

    async def generate(self, prompt, system_prompt=None):
        return ""


class TestExtractionFallback:

    def test_missing_tags_get_derived(self, tmp_path):
        from core import config as cfgmod
        from core.engine import MemoryEngine
        from core.paths import DataPaths

        paths = DataPaths(tmp_path / "pd").ensure()
        conn = dbm.connect(paths.db)
        dbm.init_schema(conn)
        store = MemoryStore(conn)
        eng = MemoryEngine(store, cfgmod.Config({}, paths.meta), llm=_FakeLLM({
            "memories": [{"content": "Star Lantern（3141592653）周末在家跑步",
                          "type": "fact", "alpha": 0.8, "speaker": "user"}],
            "profile": [],
        }))
        try:
            eng.record_turn("s1", "user", "我周末在家跑步", scope="default")
            n = asyncio.run(eng.extract_session("s1", scope="default",
                                                user_key="3141592653"))
            assert n == 1
            row = store.active_memories("default")[0]
            tags = json.loads(row["tags_json"])
            assert tags, "模型漏输出 tags 时必须规则派生兜底，不能落空数组"
            assert "3141592653" in tags
        finally:
            conn.close()
