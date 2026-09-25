"""v0.1.7 主动记忆判定收紧 + 证据落库回归测试。

背景（用户报障）：工具写入此前一律 is_active=True 永不衰减 +
alpha=1.0 绕过价值门 + source="user" 伪装来源。对照 angel
（angel_remember 结构强制 judgment/tags 必填、reflection 产物默认
被动、is_active 语义留给显式调用方）收紧：
- 工具默认 retention=normal → 被动记忆（可衰减，反思强化续命）；
- retention=permanent 才写主动记忆；
- evidence 落 reasoning 列（工具与抽取两条路径）；
- source 如实标 tool。
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def _mk_event(engine, session_id="s1", sender="u1"):
    class _Ev:
        get_session_id = staticmethod(lambda: session_id)
        get_sender_id = staticmethod(lambda: sender)

    ev = _Ev()
    ev.mnemoria_engine = engine  # 工具经 getattr(event, "mnemoria_engine") 取引擎
    return ev


def _tool(engine):
    from tools.remember import MemoryRememberTool
    return MemoryRememberTool()


# ---------------------------------------------------------------- 工具判定
class TestToolRetention:
    @pytest.mark.asyncio
    async def test_default_is_passive(self, make_engine):
        """🔴 回归钉子：不传 retention 默认被动（此前一律永生）。"""
        eng, store, conn = make_engine()
        try:
            eng._session_user["s1"] = "u1"
            msg = await _tool(eng).run(_mk_event(eng), "小明（u1）每周四打篮球")
            assert "普通记忆" in msg
            row = store.active_memories("default")[0]
            assert row["is_active"] == 0, "默认必须是被动手感（v0.1.7 核心收紧）"
        finally:
            conn.close()

    @pytest.mark.asyncio
    async def test_permanent_is_active(self, make_engine):
        eng, store, conn = make_engine()
        try:
            eng._session_user["s1"] = "u1"
            msg = await _tool(eng).run(
                _mk_event(eng), "小明（u1）的生日是10月9日", retention="permanent")
            assert "永不遗忘" in msg
            row = store.active_memories("default")[0]
            assert row["is_active"] == 1
            assert row["strength"] == 50.0
        finally:
            conn.close()

    @pytest.mark.asyncio
    async def test_evidence_lands_in_reasoning(self, make_engine):
        eng, store, conn = make_engine()
        try:
            eng._session_user["s1"] = "u1"
            await _tool(eng).run(
                _mk_event(eng), "小明（u1）雷点是政治课",
                evidence="「我最烦政治课了」")
            row = store.active_memories("default")[0]
            assert row["reasoning"] == "「我最烦政治课了」", "evidence 必须落 reasoning 列"
        finally:
            conn.close()

    @pytest.mark.asyncio
    async def test_source_is_tool_not_user(self, make_engine):
        """来源如实标注：工具写入不再是 user（审计可查）。"""
        eng, store, conn = make_engine()
        try:
            eng._session_user["s1"] = "u1"
            await _tool(eng).run(_mk_event(eng), "小明（u1）喜欢跑步")
            row = store.active_memories("default")[0]
            assert row["source"] == "tool"
        finally:
            conn.close()

    @pytest.mark.asyncio
    async def test_tool_write_survives_assistant_gate(self, make_engine):
        """source=tool 不得被 deny_assistant_claims 误杀（只拦 assistant）。"""
        eng, store, conn = make_engine()
        try:
            eng._session_user["s1"] = "u1"
            await _tool(eng).run(_mk_event(eng), "小明（u1）在准备考试")
            assert store.count()["total"] == 1
        finally:
            conn.close()

    @pytest.mark.asyncio
    async def test_invalid_retention_falls_back_normal(self, make_engine):
        eng, store, conn = make_engine()
        try:
            eng._session_user["s1"] = "u1"
            await _tool(eng).run(_mk_event(eng), "小明（u1）会弹古筝",
                                 retention="bogus")
            row = store.active_memories("default")[0]
            assert row["is_active"] == 0, "非法 retention 必须回落 normal"
        finally:
            conn.close()


# ---------------------------------------------------------------- 抽取证据落库
class TestExtractionEvidence:
    @pytest.mark.asyncio
    async def test_extract_evidence_persisted(self, make_engine):
        """抽取路径：LLM 返回的 evidence 不再丢弃。"""
        eng, store, conn = make_engine(payload={
            "memories": [{"content": "小明（u1）喜欢跑步", "type": "fact",
                          "alpha": 0.9, "speaker": "user",
                          "evidence": "「明天去跑步」"}],
            "profile": [],
        })
        try:
            eng.record_turn("s", "user", "我明天想去跑步", scope="default")
            n = await eng.extract_session("s", scope="default", user_key="u1")
            assert n == 1
            row = store.active_memories("default")[0]
            assert row["reasoning"] == "「明天去跑步」"
        finally:
            conn.close()

    @pytest.mark.asyncio
    async def test_extract_missing_evidence_ok(self, make_engine):
        eng, store, conn = make_engine(payload={
            "memories": [{"content": "小明（u1）作息偏晚", "type": "fact",
                          "alpha": 0.9}],
            "profile": [],
        })
        try:
            eng.record_turn("s", "user", "我睡得很晚", scope="default")
            assert await eng.extract_session("s", scope="default", user_key="u1") == 1
            row = store.active_memories("default")[0]
            assert row["reasoning"] == ""
        finally:
            conn.close()
