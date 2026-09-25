"""v0.1.3 自审修复批回归测试（对 v0.1.2 的对抗性复查产物）。

覆盖：notes_fts 迁移失败路径保触发器（v0.1.2 的实缺陷）、触发器缺失/旧列集
的启动守卫、代述闸门覆盖画像数组、LLM 停用期间的空闲计时刷新。
"""

from __future__ import annotations

import asyncio
import sqlite3
import sys
import time
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


# ---------------------------------------------------------------- 迁移失败与恢复
class TestMigrationTriggerSafety:
    def _to_legacy(self, conn, store):
        """回退到 v0.1.0 旧结构（旧表+旧触发器）并写入一条笔记。"""
        conn.execute("DROP TABLE notes_fts")
        for trig in ("note_fts_ai", "note_fts_au", "note_fts_ad"):
            conn.execute(f"DROP TRIGGER IF EXISTS {trig}")
        conn.execute(
            "CREATE VIRTUAL TABLE notes_fts USING fts5("
            "content, row_id UNINDEXED, tokenize='trigram')"
        )
        conn.execute(
            "CREATE TRIGGER note_fts_ai AFTER INSERT ON notes BEGIN "
            "INSERT INTO notes_fts(content, row_id) VALUES (new.content, new.id); END;"
        )
        conn.commit()
        store.add_note("正文完全无关词汇", title="猫粮囤货攻略", scope="default")

    def test_missing_triggers_restored_by_startup_guard(self, make_engine):
        """触发器丢失（中断残留）：init_schema 自动修复并保持功能。"""
        from core import db as dbm
        eng, store, conn = make_engine()
        try:
            store.add_note("守卫笔记甲", title="标题甲", scope="default")
            for trig in ("note_fts_ai", "note_fts_au", "note_fts_ad"):
                conn.execute(f"DROP TRIGGER IF EXISTS {trig}")
            conn.commit()
            assert conn.execute(
                "SELECT COUNT(*) FROM sqlite_master WHERE type='trigger' "
                "AND name LIKE 'note_fts_%'").fetchone()[0] == 0
            dbm.init_schema(conn)
            n = conn.execute(
                "SELECT COUNT(*) FROM sqlite_master WHERE type='trigger' "
                "AND name IN ('note_fts_ai','note_fts_au','note_fts_ad')").fetchone()[0]
            assert n == 3, "启动守卫必须补回三个触发器"
            store.add_note("守卫笔记乙", title="标题乙", scope="default")
            hits = store.search_notes("守卫笔记乙")
            assert hits, "补回的触发器必须真正工作（新笔记进索引）"
        finally:
            conn.close()

    def test_legacy_triggers_upgraded(self, make_engine):
        """新结构表配旧列集触发器：守卫识别并升级（否则 title 更新不同步）。"""
        from core import db as dbm
        eng, store, conn = make_engine()
        try:
            store.add_note("升级测试笔记", title="原标题", scope="default")
            for trig in ("note_fts_ai", "note_fts_au", "note_fts_ad"):
                conn.execute(f"DROP TRIGGER IF EXISTS {trig}")
            # 手写旧列集触发器（模拟与新表错配的残留）
            conn.executescript(
                "CREATE TRIGGER note_fts_ai AFTER INSERT ON notes BEGIN "
                "INSERT INTO notes_fts(content, row_id) VALUES (new.content, new.id); END;"
            )
            conn.commit()
            assert not dbm._note_triggers_ok(conn), "旧列集触发器必须被判定为不合格"
            dbm.init_schema(conn)
            assert dbm._note_triggers_ok(conn), "启动后触发器应升级为新列集"
            ai = conn.execute(
                "SELECT sql FROM sqlite_master WHERE name='note_fts_ai'").fetchone()[0]
            assert "title" in ai
        finally:
            conn.close()

    def test_healthy_state_not_rebuilt(self, make_engine):
        """健康现场：守卫不得误触发重建（副本表不得出现）。"""
        from core import db as dbm
        eng, store, conn = make_engine()
        try:
            store.add_note("健康状态笔记", title="标题", scope="default")
            assert dbm._note_triggers_ok(conn)
            dbm.init_schema(conn)
            leftover = conn.execute(
                "SELECT COUNT(*) FROM sqlite_master WHERE name='notes_fts_new'").fetchone()[0]
            assert leftover == 0, "健康现场不得派生迁移副本"
            assert conn.execute("SELECT COUNT(*) FROM notes_fts").fetchone()[0] == 1
        finally:
            conn.close()


# ---------------------------------------------------------------- 画像代述闸门
class TestProfileSpeakerGate:
    @pytest.mark.asyncio
    async def test_profile_assistant_rejected(self, make_engine):
        """画像注入 system_prompt——代述闸门必须同样覆盖画像数组。"""
        eng, store, conn = make_engine(payload={
            "memories": [],
            "profile": [
                {"key": "爱好", "value": "助手认为用户喜欢登山",
                 "speaker": "assistant", "confidence": 0.9},
                {"key": "城市", "value": "杭州", "speaker": "user", "confidence": 0.9},
            ],
        })
        try:
            eng.record_turn("s", "user", "今天随便聊了聊最近的生活近况", scope="default")
            await eng.extract_session("s", scope="default", user_key="u1")
            rows = store.get_profile("default", "u1")
            keys = [r["key"] for r in rows]
            assert "城市" in keys, "用户亲口说的画像必须写入"
            assert "爱好" not in keys, "assistant 代述画像必须被闸门拦下"
        finally:
            conn.close()

    @pytest.mark.asyncio
    async def test_profile_assistant_allowed_when_disabled(self, make_engine):
        eng, store, conn = make_engine(
            {"admission": {"deny_assistant_claims": False}},
            payload={
                "memories": [],
                "profile": [{"key": "爱好", "value": "登山",
                             "speaker": "assistant", "confidence": 0.9}],
            })
        try:
            eng.record_turn("s", "user", "今天随便聊了聊最近的生活近况", scope="default")
            await eng.extract_session("s", scope="default", user_key="u1")
            assert any(r["key"] == "事实属性" for r in store.get_profile("default", "u1")), \
                "闸门关闭时 assistant 画像应放行（键归一到固定维度）"
        finally:
            conn.close()

    @pytest.mark.asyncio
    async def test_profile_without_speaker_still_accepted(self, make_engine):
        """缺 speaker 的画像（旧模型输出习惯）按 user 放行——宁放过不错杀。"""
        eng, store, conn = make_engine(payload={
            "memories": [],
            "profile": [{"key": "称呼", "value": "小明", "confidence": 0.9}],
        })
        try:
            eng.record_turn("s", "user", "今天随便聊了聊最近的生活近况", scope="default")
            await eng.extract_session("s", scope="default", user_key="u1")
            assert any(r["key"] == "用户别名" for r in store.get_profile("default", "u1"))
        finally:
            conn.close()


# ---------------------------------------------------------------- LLM 停用空闲
class TestIdleRefreshWhenLLMDisabled:
    @pytest.mark.asyncio
    async def test_disabled_llm_refreshes_last_seen(self, make_engine):
        """LLM 未启用：抽取调用也要刷新空闲计时，防每 tick 恒真空转。"""
        eng, store, conn = make_engine(  # payload=None → llm=None
            {"memory_behavior": {"trigger_turns": 99, "idle_seconds": 0.001}})
        try:
            eng.record_turn("s", "user", "消息", scope="default")
            await asyncio.sleep(0.01)
            assert eng.should_extract("s") is True
            n = await eng.extract_session("s", scope="default", user_key="u1")
            assert n == 0
            assert eng.should_extract("s") is False, \
                "LLM 停用期间也必须刷新 _last_seen（否则空转任务每 tick 被 spawn）"
        finally:
            conn.close()


# ---------------------------------------------------------------- prompt 契约
class TestSpeakerPromptContract:
    def test_profile_speaker_declared(self):
        from core import templates
        assert templates.EXTRACT_PROMPT.count('"speaker"') >= 2, \
            "记忆与画像两个数组都必须声明 speaker"
