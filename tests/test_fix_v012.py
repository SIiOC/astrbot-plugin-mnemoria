"""v0.1.2 二次审查修复批回归测试。

覆盖：deny_assistant_claims 真正接线（speaker 标注透传）、notes_fts 迁移
copy-then-swap（失败保旧索引 / 空索引中断残留守卫）、笔记 LIKE 兜底三列
口径（注入路径）、有内容抽取后空闲计时统一、脚本配置缺失清晰报错。
"""

from __future__ import annotations

import asyncio
import sqlite3
import sys
import time
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


# ---------------------------------------------------------------- 1A speaker 接线
class TestSpeakerWiring:
    @pytest.mark.asyncio
    async def test_assistant_speaker_rejected_by_default(self, make_engine):
        """LLM 标注 speaker=assistant 的条目：默认开关下拒收（此前恒写入）。"""
        eng, store, conn = make_engine(payload={
            "memories": [{"content": "助手觉得用户应该早睡", "type": "fact",
                          "alpha": 0.9, "speaker": "assistant"}],
            "profile": [],
        })
        try:
            eng.record_turn("s", "assistant", "你应该早点睡", scope="default")
            n = await eng.extract_session("s", scope="default", user_key="u1")
            assert n == 0, "assistant 代述必须被 deny_assistant_claims 闸门拒收"
            assert store.count()["total"] == 0
        finally:
            conn.close()

    @pytest.mark.asyncio
    async def test_assistant_speaker_allowed_when_disabled(self, make_engine):
        eng, store, conn = make_engine({
            "admission": {"deny_assistant_claims": False},
        }, payload={
            "memories": [{"content": "助手觉得用户应该早睡", "type": "fact",
                          "alpha": 0.9, "speaker": "assistant"}],
            "profile": [],
        })
        try:
            eng.record_turn("s", "assistant", "你应该早点睡", scope="default")
            n = await eng.extract_session("s", scope="default", user_key="u1")
            assert n == 1, "开关关闭时 assistant 代述放行"
        finally:
            conn.close()

    @pytest.mark.asyncio
    async def test_user_speaker_and_garbage_fall_back_to_user(self, make_engine):
        """speaker=user / 缺失 / 非法值都按 user 处理（宁放过不错杀）。"""
        eng, store, conn = make_engine(payload={
            "memories": [
                {"content": "用户的名字是张三", "type": "fact", "alpha": 0.9, "speaker": "user"},
                {"content": "用户住在杭州", "type": "fact", "alpha": 0.9},
                {"content": "用户喜欢猫", "type": "fact", "alpha": 0.9, "speaker": "bogus"},
            ],
            "profile": [],
        })
        try:
            eng.record_turn("s", "user", "我叫张三住杭州喜欢猫", scope="default")
            n = await eng.extract_session("s", scope="default", user_key="u1")
            assert n == 3, f"三条都应按 user 写入，实际 {n}"
        finally:
            conn.close()

    def test_extract_prompt_declares_speaker(self):
        """契约：抽取提示词必须声明 speaker 字段（接线的前提）。"""
        from core import templates
        assert '"speaker"' in templates.EXTRACT_PROMPT
        assert "assistant" in templates.EXTRACT_PROMPT


# ---------------------------------------------------------------- 4 notes_fts 迁移
class TestNotesFtsCopyThenSwap:
    def _legacy_state(self, conn, store):
        """把库回退到 v0.1.0 旧结构（只索引 content）并写入一条笔记。"""
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

    def test_failure_preserves_old_index(self, make_engine):
        """灌数据失败：旧索引原样保留、_new 副本清理、不再谎称保持旧结构。"""
        from core import db as dbm
        eng, store, conn = make_engine()
        try:
            self._legacy_state(conn, store)

            class Proxy:
                def __init__(self, real):
                    self._real = real

                def execute(self, sql, params=()):
                    if "INSERT INTO notes_fts_new" in sql:
                        raise sqlite3.OperationalError("模拟灌库失败")
                    return self._real.execute(sql, params)

                def executescript(self, script):
                    return self._real.executescript(script)

                def commit(self):
                    self._real.commit()

                def rollback(self):
                    self._real.rollback()

            dbm._upgrade_notes_fts(Proxy(conn))
            cols = [r[1] for r in conn.execute("PRAGMA table_info(notes_fts)").fetchall()]
            assert "title" not in cols, "失败后旧结构必须原样保留（copy-then-swap 契约）"
            n = conn.execute("SELECT COUNT(*) FROM notes_fts").fetchone()[0]
            assert n == 1, "旧索引数据不得丢失"
            leftover = [r[1] for r in conn.execute("PRAGMA table_info(notes_fts_new)").fetchall()]
            assert not leftover, "失败的 _new 副本必须清理"
            # v0.1.3：失败路径不得丢触发器（v0.1.2 曾在灌数据前就摘除，
            # 失败后旧表在、触发器没了，新笔记静默不进索引）
            trig = conn.execute(
                "SELECT COUNT(*) FROM sqlite_master WHERE type='trigger' AND name='note_fts_ai'"
            ).fetchone()[0]
            assert trig == 1, "失败路径必须保留原触发器"
            conn.execute("INSERT INTO notes(id, title, content, scope) "
                         "VALUES('extra','后补标题','后补笔记内容','default')")
            conn.commit()
            assert conn.execute("SELECT COUNT(*) FROM notes_fts").fetchone()[0] == 2, \
                "陈旧但有效的触发器仍在工作：新笔记必须继续进旧索引"
            # 旧索引仍可检索（FTS 本身没坏；查询须 ≥3 字过 trigram 下限）
            hits = conn.execute("SELECT row_id FROM notes_fts WHERE notes_fts MATCH '完全无关'").fetchall()
            assert hits
        finally:
            conn.close()

    def test_empty_index_with_notes_gets_backfilled(self, make_engine):
        """崩溃残留（结构正确但 0 行 + 笔记非空）：守卫触发补灌，不再永久空索引。"""
        from core import db as dbm
        eng, store, conn = make_engine()
        try:
            store.add_note("守卫测试笔记", title="狗粮攻略", scope="default")
            conn.execute("DELETE FROM notes_fts")  # 制造"结构对但空"的残留
            conn.commit()
            dbm.init_schema(conn)
            n = conn.execute("SELECT COUNT(*) FROM notes_fts").fetchone()[0]
            assert n == 1, "空索引守卫必须补灌"
            hits = store.search_notes("狗粮")
            assert hits and any(h["title"] == "狗粮攻略" for h in hits)
        finally:
            conn.close()

    def test_healthy_index_untouched(self, make_engine):
        """结构正确且非空：迁移不得重复重建（幂等守卫）。"""
        from core import db as dbm
        eng, store, conn = make_engine()
        try:
            store.add_note("健康索引", title="标题", scope="default")
            before = conn.execute(
                "SELECT sql FROM sqlite_master WHERE name='note_fts_ai'").fetchone()
            dbm.init_schema(conn)
            after = conn.execute(
                "SELECT sql FROM sqlite_master WHERE name='note_fts_ai'").fetchone()
            assert before is not None and after is not None
            n = conn.execute("SELECT COUNT(*) FROM notes_fts").fetchone()[0]
            assert n == 1
        finally:
            conn.close()


# ---------------------------------------------------------------- 5 注入路径兜底口径
class TestLexicalRanksFallback:
    def test_two_char_query_hits_title(self, make_engine):
        """两字查询达不到 trigram 下限必然走 LIKE 兜底——标题必须能命中。"""
        from core.notes import lexical_ranks
        eng, store, conn = make_engine()
        try:
            store.add_note("正文完全无关的内容", title="猫粮囤货攻略", scope="default")
            ranks = lexical_ranks(conn, "猫粮", "default", 5, store.fts)
            assert ranks, "注入路径的 LIKE 兜底必须能搜到标题（与面板口径一致）"
        finally:
            conn.close()

    def test_content_still_matched(self, make_engine):
        from core.notes import lexical_ranks
        eng, store, conn = make_engine()
        try:
            store.add_note("蓝山咖啡豆风味记录", title="", scope="default")
            assert lexical_ranks(conn, "咖啡豆", "default", 5, store.fts)
        finally:
            conn.close()


# ---------------------------------------------------------------- 6 空闲计时统一
class TestIdleRefreshOnContentExtract:
    @pytest.mark.asyncio
    async def test_content_extract_refreshes_last_seen(self, make_engine):
        """有内容抽取后立即刷新空闲计时：不再多空转一轮。"""
        eng, store, conn = make_engine({"memory_behavior": {
            "trigger_turns": 99, "idle_seconds": 0.001,
        }}, payload={
            "memories": [{"content": "用户在测试空闲计时统一", "type": "fact", "alpha": 0.9}],
            "profile": [],
        })
        try:
                eng.record_turn("s", "user", "我最近在测试一个东西啊", scope="default")
                await asyncio.sleep(0.01)
                assert eng.should_extract("s") is True
                t_before = time.time()
                n = await eng.extract_session("s", scope="default", user_key="u1")
                assert n == 1
                age = time.time() - eng._last_seen["s"]
                assert age < 1.0, "有内容抽取后 _last_seen 必须刷新"
                # v0.1.5：末次断言改为对比刷新时刻与计数器。idle=0.001s 下
                # 「抽取完立刻 should_extract 为 False」是天然竞态——refresh 之后
                # 任何 >1ms 的耗时（含 v0.1.5 新增的 _user_label 画像查询）都会
                # 再次越限，历史上在开发机实地复现翻车过
                assert eng._last_seen["s"] >= t_before, "刷新必须发生在本次抽取期间"
                assert eng._turn_counter["s"] == 0, "计数器必须清零（轮次触发不多发）"
        finally:
            conn.close()


# ---------------------------------------------------------------- 3 脚本路径报错
class TestScriptConfigGuidance:
    def test_import_export_missing_cfg_message(self, tmp_path, monkeypatch):
        sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
        import import_export
        monkeypatch.setattr(import_export, "CFG_PATH", tmp_path / "nope.json")
        with pytest.raises(SystemExit) as ei:
            import_export._load_embedding_provider()
        assert "ASTRBOT_CMD_CONFIG" in str(ei.value), "报错必须给出可行指引"
        assert "--config" in str(ei.value)

    def test_calibrate_missing_cfg_message(self, tmp_path, monkeypatch):
        sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
        import calibrate_embeddings
        monkeypatch.setattr(calibrate_embeddings, "CFG_PATH", tmp_path / "nope.json")
        with pytest.raises(SystemExit) as ei:
            calibrate_embeddings.load_provider("nvidia_embedding")
        assert "ASTRBOT_CMD_CONFIG" in str(ei.value)
