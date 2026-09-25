"""v0.1.5 记忆质量批回归测试。

覆盖：
- 抽取提示词三类记忆 + 用户标识（昵称（ID））注入；
- assistant 分级闸门 assistant_claim_policy（reject_all / allow_relationship /
  allow_all / 非法值回落旧开关）；
- 画像入口不受新策略影响（仍只看 deny_assistant_claims）。
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


# ---------------------------------------------------------------- 提示词契约
class TestExtractPromptContract:
    def test_prompt_declares_user_label(self):
        from core import templates
        assert "{user_label}" in templates.EXTRACT_PROMPT, "提示词必须带用户标识占位符"
        assert "[{user_label}]" in templates.EXTRACT_PROMPT, "v0.1.6 起 transcript 规则须声明身份锚点"
        # 旧契约不回退：speaker 字段仍须声明（v0.1.2 钉子）
        assert templates.EXTRACT_PROMPT.count('"speaker"') >= 2
        assert "assistant" in templates.EXTRACT_PROMPT

    def test_prompt_mentions_three_kinds(self):
        from core import templates
        for kw in ("关系", "约定", "我会"):
            assert kw in templates.EXTRACT_PROMPT, f"提示词应覆盖三类记忆（缺 {kw}）"


# ---------------------------------------------------------------- 用户标识
class TestUserLabel:
    @pytest.mark.asyncio
    async def test_prompt_contains_nick_and_id(self, make_engine):
        """画像里有专用键「用户昵称」→ 提示词注入「昵称（用户ID）」+ transcript 身份锚点。"""
        eng, store, conn = make_engine(payload={"memories": [], "profile": []})
        try:
            store.upsert_profile("default", "u1", "用户昵称", "小明")
            captured = {}
            orig = eng.llm.generate_json

            async def spy(prompt, system_prompt=None):
                captured["prompt"] = prompt
                return await orig(prompt, system_prompt=system_prompt)

            eng.llm.generate_json = spy
            eng.record_turn("s", "user", "我今天聊了很多心里话", scope="default")
            await eng.extract_session("s", scope="default", user_key="u1")
            assert "小明（u1）" in captured.get("prompt", ""), "昵称（ID）必须进入提示词"
            assert "[小明（u1）]" in captured.get("prompt", ""), "v0.1.6 transcript 必须带身份锚点"
        finally:
            conn.close()

    @pytest.mark.asyncio
    async def test_label_fallback_without_profile(self, make_engine):
        """无画像 → 回退「用户（ID）」；无 user_key → 回退「用户」。"""
        eng, store, conn = make_engine(payload={"memories": [], "profile": []})
        try:
            assert eng._user_label("default", "u1") == "用户（u1）"
            assert eng._user_label("default", "") == "用户"
            store.upsert_profile("default", "u2", "用户昵称", "阿崇")
            assert eng._user_label("default", "u2") == "阿崇（u2）"
            # 专用键以外的画像键不参与指称
            store.upsert_profile("default", "u3", "喜好", "猫")
            assert eng._user_label("default", "u3") == "用户（u3）"
        finally:
            conn.close()


# ---------------------------------------------------------------- 分级闸门（admission 层）
class TestAssistantPolicy:
    def _assess(self, text, *, source="assistant", mtype="fact", policy="", deny=True):
        from core.admission import assess
        return assess(text, alpha=0.9, alpha_threshold=0.4, source=source,
                      deny_assistant_claims=deny,
                      assistant_claim_policy=policy, memory_type=mtype)

    def test_reject_all_rejects(self):
        from core.admission import Verdict
        r = self._assess("当用户emo时我会陪他聊聊", policy="reject_all", mtype="emotional")
        assert r.verdict == Verdict.REJECT

    def test_allow_all_accepts_plain_fact(self):
        from core.admission import Verdict
        r = self._assess("用户喜欢猫", policy="allow_all")
        assert r.verdict == Verdict.ACCEPT

    def test_relationship_type_whitelist(self):
        from core.admission import Verdict
        for mt in ("event", "emotional"):
            r = self._assess("用户生日那天我陪聊到很晚", policy="allow_relationship", mtype=mt)
            assert r.verdict == Verdict.ACCEPT, f"type={mt} 应放行"

    def test_relationship_marker_text(self):
        from core.admission import Verdict
        r = self._assess("两人约定周末一起复盘近况", policy="allow_relationship", mtype="fact")
        assert r.verdict == Verdict.ACCEPT, "含互动/约定特征的事实应放行"

    def test_relationship_rejects_plain_assistant_fact(self):
        from core.admission import Verdict
        r = self._assess("用户好像喜欢猫", policy="allow_relationship", mtype="fact")
        assert r.verdict == Verdict.REJECT, "纯代述用户事实仍须拒收"

    def test_cliche_still_rejected_under_relationship(self):
        from core.admission import Verdict
        r = self._assess("作为一个人工智能助手，我会一直陪着你", policy="allow_relationship",
                         mtype="emotional")
        assert r.verdict == Verdict.REJECT, "套话不得因策略放宽而入库"

    def test_invalid_policy_falls_back_to_legacy(self):
        from core.admission import Verdict
        # 非法值 + deny=True → 拒
        assert self._assess("用户喜欢猫", policy="bogus").verdict == Verdict.REJECT
        # 缺失 + deny=False → 放
        assert self._assess("用户喜欢猫", policy="", deny=False).verdict == Verdict.ACCEPT

    def test_user_source_unaffected(self):
        from core.admission import Verdict
        r = self._assess("用户喜欢猫", source="user", policy="reject_all")
        assert r.verdict == Verdict.ACCEPT


# ---------------------------------------------------------------- schema 契约
class TestSchemaTypesSupportedByFramework:
    def test_all_schema_types_in_framework_map(self):
        """_conf_schema.json 的每个 type 必须在框架 DEFAULT_VALUE_MAP 内。

        v0.1.5 事故：新增 assistant_claim_policy 时误用 "select"（别的框架的
        记忆），框架 _config_schema_to_default_config 直接 TypeError 报
        「不受支持的配置类型 select」→ 插件装载失败。此钉直接引用框架
        真源（default.py DEFAULT_VALUE_MAP），防再犯。
        """
        import json
        from astrbot.core.config.default import DEFAULT_VALUE_MAP
        schema = json.load(open(
            Path(__file__).resolve().parents[1] / "_conf_schema.json",
            encoding="utf-8-sig"))

        def _walk(node, path=""):
            for k, v in node.items():
                if not isinstance(v, dict) or "type" not in v:
                    continue
                t = v["type"]
                assert t in DEFAULT_VALUE_MAP, \
                    f"schema {path}{k} 用了框架不支持的类型 {t!r}"
                if t in ("object", "template_list") and isinstance(v.get("items"), dict):
                    _walk(v["items"], f"{path}{k}.")

        _walk(schema)


# ---------------------------------------------------------------- 引擎接线
class TestEnginePolicyWiring:
    @pytest.mark.asyncio
    async def test_relationship_memory_written(self, make_engine):
        """allow_relationship：assistant 互动记忆（当…我会…）入库。"""
        eng, store, conn = make_engine({"admission": {
            "assistant_claim_policy": "allow_relationship",
        }}, payload={
            "memories": [{"content": "当小明（u1）深夜emo时，我会先听他说完再安慰",
                          "type": "emotional", "alpha": 0.8, "speaker": "assistant"}],
            "profile": [],
        })
        try:
            eng.record_turn("s", "user", "我今晚有点难受", scope="default")
            n = await eng.extract_session("s", scope="default", user_key="u1")
            assert n == 1, "互动类 assistant 记忆应入库"
            assert store.count()["total"] == 1
        finally:
            conn.close()

    @pytest.mark.asyncio
    async def test_plain_assistant_fact_still_rejected(self, make_engine):
        eng, store, conn = make_engine({"admission": {
            "assistant_claim_policy": "allow_relationship",
        }}, payload={
            "memories": [{"content": "用户感冒了", "type": "fact",
                          "alpha": 0.9, "speaker": "assistant"}],
            "profile": [],
        })
        try:
            eng.record_turn("s", "assistant", "你好像感冒了", scope="default")
            n = await eng.extract_session("s", scope="default", user_key="u1")
            assert n == 0 and store.count()["total"] == 0
        finally:
            conn.close()

    @pytest.mark.asyncio
    async def test_policy_supersedes_legacy_switch(self, make_engine):
        """显式 reject_all 优先于 deny_assistant_claims=false。"""
        eng, store, conn = make_engine({"admission": {
            "deny_assistant_claims": False,
            "assistant_claim_policy": "reject_all",
        }}, payload={
            "memories": [{"content": "用户喜欢猫", "type": "fact",
                          "alpha": 0.9, "speaker": "assistant"}],
            "profile": [],
        })
        try:
            eng.record_turn("s", "assistant", "你喜欢猫吧", scope="default")
            n = await eng.extract_session("s", scope="default", user_key="u1")
            assert n == 0, "policy 显式设置时必须压过旧开关"
        finally:
            conn.close()

    @pytest.mark.asyncio
    async def test_profile_gate_not_opened_by_policy(self, make_engine):
        """allow_relationship 只开记忆闸门：画像 assistant 条目仍拒收。"""
        eng, store, conn = make_engine({"admission": {
            "assistant_claim_policy": "allow_relationship",
        }}, payload={
            "memories": [],
            "profile": [{"key": "称呼", "value": "宝宝", "speaker": "assistant",
                         "confidence": 0.9}],
        })
        try:
            eng.record_turn("s", "assistant", "那我以后叫你宝宝好不好", scope="default")
            await eng.extract_session("s", scope="default", user_key="u1")
            assert store.get_profile("default", "u1") == [], \
                "画像入口不受 assistant_claim_policy 影响（保守设计）"
        finally:
            conn.close()
