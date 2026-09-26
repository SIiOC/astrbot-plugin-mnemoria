"""v0.1.6 指称修复 + 抽取纪律批回归测试。

背景（历史事故，v0.1.6 修复的 P0）：画像 称呼/名字 键可能被 bot 侧信息
占用（名字=bot 人设名），v0.1.5 的 _user_label 读它们，曾把用户标成
bot 人设（记忆正文出现「人设名(用户ID)在晚上活跃…」式污染）。

覆盖：
- _user_label 新解析链（config map → 全局默认 → 专用画像键 → 回退），
  且绝不读取 称呼/名字/昵称/姓名/name；
- transcript 身份锚点（[昵称（ID）]: / [助理]:）；
- type 六类枚举归一化（自造类型 → fact）；
- allow_relationship marker 扩充（恋人/交往类）。
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


# ---------------------------------------------------------------- 指称修复
class TestUserLabelP0:
    def test_bot_occupied_keys_never_used(self, make_engine):
        """P0 回归钉子：称呼=宝宝、名字=星棠(青鸾) 存在时，
        指称绝不能取到它们（历史事故形态复刻）。"""
        eng, store, conn = make_engine()
        try:
            store.upsert_profile("default", "3141592653", "称呼", "宝宝", 0.9)
            store.upsert_profile("default", "3141592653", "名字", "星棠(青鸾)", 0.9)
            store.upsert_profile("default", "3141592653", "昵称", "念念", 0.8)
            store.upsert_profile("default", "3141592653", "姓名", "青鸾", 0.8)
            store.upsert_profile("default", "3141592653", "name", "Xingtaoyue", 0.8)
            label = eng._user_label("default", "3141592653")
            assert label == "用户（3141592653）", \
                f"bot 占用键必须全部失效，实际得到 {label!r}"
            for bad in ("宝宝", "星棠", "青鸾", "念念", "Xingtaoyue"):
                assert bad not in label
        finally:
            conn.close()

    def test_config_map_wins(self, make_engine):
        eng, store, conn = make_engine({"runtime": {
            "user_display_name_map": {"u1": "Star Lantern"},
        }})
        try:
            # 即使画像里塞了 bot 键，map 优先
            store.upsert_profile("default", "u1", "名字", "星棠(青鸾)", 0.9)
            assert eng._user_label("default", "u1") == "Star Lantern（u1）"
            # 其他用户不受影响
            assert eng._user_label("default", "u2") == "用户（u2）"
        finally:
            conn.close()

    def test_global_display_name_used_as_is(self, make_engine):
        eng, store, conn = make_engine({"runtime": {
            "user_display_name": "小明",
        }})
        try:
            assert eng._user_label("default", "u1") == "小明"
            assert eng._user_label("default", "") == "小明"
        finally:
            conn.close()

    def test_dedicated_profile_key(self, make_engine):
        eng, store, conn = make_engine()
        try:
            store.upsert_profile("default", "u1", "用户昵称", "小明")
            assert eng._user_label("default", "u1") == "小明（u1）"
        finally:
            conn.close()

    def test_map_beats_global_and_profile(self, make_engine):
        eng, store, conn = make_engine({"runtime": {
            "user_display_name": "全局名",
            "user_display_name_map": {"u1": "地图名"},
        }})
        try:
            store.upsert_profile("default", "u1", "用户昵称", "画像名")
            assert eng._user_label("default", "u1") == "地图名（u1）"
            assert eng._user_label("default", "u2") == "全局名"
        finally:
            conn.close()

    def test_malformed_map_tolerated(self, make_engine):
        """map 配成非 dict（脏配置）不得崩，回落正常链路。"""
        eng, store, conn = make_engine({"runtime": {
            "user_display_name_map": "not-a-dict",
        }})
        try:
            assert eng._user_label("default", "u1") == "用户（u1）"
        finally:
            conn.close()


class TestIdentityTranscript:
    def test_build_transcript_labels(self):
        from core.templates import build_transcript
        text = build_transcript(
            [("user", "我感冒了"), ("assistant", "多喝热水早点睡")],
            user_label="Star Lantern（3141592653）",
        )
        assert "[Star Lantern（3141592653）]: 我感冒了" in text
        assert "[助理]: 多喝热水早点睡" in text
        assert "用户" not in text, "身份化后 transcript 不得再出现裸「用户」"

    def test_default_transcript_backwards_compatible(self):
        from core.templates import build_transcript
        text = build_transcript([("user", "hi")])
        assert "[用户]: hi" in text

    @pytest.mark.asyncio
    async def test_extract_prompt_carries_label_and_anchor(self, make_engine):
        """端到端：config map 的名字进 transcript 锚点与规则行。"""
        eng, store, conn = make_engine({"runtime": {
            "user_display_name_map": {"u1": "Star Lantern"},
        }}, payload={"memories": [], "profile": []})
        try:
            captured = {}
            orig = eng.llm.generate_json

            async def spy(prompt, system_prompt=None):
                captured["prompt"] = prompt
                return await orig(prompt, system_prompt=system_prompt)

            eng.llm.generate_json = spy
            eng.record_turn("s", "user", "我周四想去跑步", scope="default")
            await eng.extract_session("s", scope="default", user_key="u1")
            p = captured.get("prompt", "")
            assert "[Star Lantern（u1）]: 我周四想去跑步" in p
            assert "[助理]" in p or "助理" in p
        finally:
            conn.close()


# ---------------------------------------------------------------- type 归一化
class TestTypeNormalization:
    @pytest.mark.asyncio
    async def test_bogus_type_becomes_fact(self, make_engine):
        eng, store, conn = make_engine(payload={
            "memories": [
                {"content": "小明（u1）喜欢跑步", "type": "relationship", "alpha": 0.9},
                {"content": "小明（u1）作息偏晚", "type": "preference", "alpha": 0.8},
                {"content": "小明（u1）会弹古筝", "type": "skill", "alpha": 0.8},
            ],
            "profile": [],
        })
        try:
            eng.record_turn("s", "user", "我喜欢跑步作息晚会弹古筝", scope="default")
            n = await eng.extract_session("s", scope="default", user_key="u1")
            assert n == 3
            types = {r["memory_type"] for r in store.active_memories("default")}
            assert types == {"fact", "skill"}, \
                f"自造类型必须归一化（fact），实际 {types}"
        finally:
            conn.close()

    def test_normalize_function_edges(self):
        from core.engine import _normalize_memory_type
        for ok in ("fact", "knowledge", "skill", "event", "emotional", "task"):
            assert _normalize_memory_type(ok) == ok
            assert _normalize_memory_type(ok.upper()) == ok
        for bad in ("relationship", "preference", "", None, "事实", 123):
            assert _normalize_memory_type(bad) == "fact"


# ---------------------------------------------------------------- 框架解析契约
class TestSchemaParsesInFramework:
    def test_full_schema_through_astrbotconfig(self, tmp_path):
        """端到端钉子：_conf_schema.json 必须能被框架 AstrBotConfig 真实加载。

        v0.1.5 的 select 事故与 v0.1.6 的 object 缺 items 事故（KeyError
        'items'，插件装载失败）都是 schema 写法不符框架契约——单元级的
        type ∈ DEFAULT_VALUE_MAP 检查拦不住递归解析层的坑，必须真跑一遍。
        同时验证 type=dict 的自由映射在完整性合并后用户键存活。
        """
        import json as _json
        from astrbot.core.config.astrbot_config import AstrBotConfig
        root = Path(__file__).resolve().parents[1]
        schema = _json.loads((root / "_conf_schema.json").read_text(encoding="utf-8-sig"))
        cfg_path = tmp_path / "astrbot_plugin_mnemoria_config.json"
        cfg_path.write_text(_json.dumps({
            "runtime": {
                "default_scope": "default",
                "user_display_name": "",
                "user_display_name_map": {"u1": "Star Lantern"},
            },
        }, ensure_ascii=False), encoding="utf-8")
        conf = AstrBotConfig(config_path=str(cfg_path), schema=schema)
        # 自由映射（type=dict）的用户键必须在合并后存活（否则指称功能失效）
        assert conf["runtime"]["user_display_name_map"].get("u1") == "Star Lantern"


# ---------------------------------------------------------------- marker 扩充
class TestRelationshipMarkers:
    def _assess(self, text, mtype="fact"):
        from core.admission import assess
        return assess(text, alpha=0.9, alpha_threshold=0.4, source="assistant",
                      deny_assistant_claims=True,
                      assistant_claim_policy="allow_relationship", memory_type=mtype)

    def test_couple_marker_fact_type_accepted(self):
        from core.admission import Verdict
        assert self._assess("两人是恋人关系，相互以宝宝相称").verdict == Verdict.ACCEPT
        assert self._assess("我们正在交往，对象是他").verdict == Verdict.ACCEPT

    def test_plain_assistant_fact_still_rejected(self):
        from core.admission import Verdict
        assert self._assess("用户好像感冒了").verdict == Verdict.REJECT
