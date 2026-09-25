"""v0.1.8 记忆演进批回归测试（angel 蓝本：tags 检索锚点 + update/merge 动作）。

覆盖：
- schema v4 迁移（旧库补 tags_json 列）；
- 抽取 tags 落库 + 清洗；
- tag 检索通道（tag 有而正文无的关键词可召回）；
- 相关旧记忆注入 + update（改口）/ merge（合并）软删演进；
- 无效编号降级 create、主动记忆不可被 LLM 演进、evolve_enabled 开关。
"""

from __future__ import annotations

import asyncio
import sqlite3
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


# ---------------------------------------------------------------- schema v4
class TestSchemaV4:
    def test_fresh_schema_has_tags_json(self, make_engine):
        eng, store, conn = make_engine()
        try:
            cols = [r[1] for r in conn.execute("PRAGMA table_info(memories)").fetchall()]
            assert "tags_json" in cols
        finally:
            conn.close()

    def test_old_db_gets_tags_json(self, tmp_path):
        """旧 v3 库（无 tags_json）启动后自动补列。"""
        from core import db as dbm
        from core.paths import DataPaths
        paths = DataPaths(tmp_path / "pd").ensure()
        conn = dbm.connect(paths.db)
        # 模拟 v3：把新列删掉（SQLite 3.35+ 支持 DROP COLUMN）
        try:
            conn.execute("ALTER TABLE memories DROP COLUMN tags_json")
            conn.commit()
        except sqlite3.OperationalError:
            pytest.skip("SQLite 不支持 DROP COLUMN")
        ver = conn.execute("SELECT value FROM meta WHERE key='schema_version'").fetchone()[0]
        assert ver == "3"
        dbm.init_schema(conn)
        cols = [r[1] for r in conn.execute("PRAGMA table_info(memories)").fetchall()]
        assert "tags_json" in cols, "v3→v4 迁移必须补 tags_json 列"
        conn.close()


# ---------------------------------------------------------------- tags 落库
class TestTagsStorage:
    @pytest.mark.asyncio
    async def test_extraction_stores_tags(self, make_engine):
        eng, store, conn = make_engine(payload={
            "memories": [{"content": "小明（u1）喜欢跑步", "type": "fact",
                          "alpha": 0.9, "speaker": "user",
                          "tags": ["小明", "u1", "跑步", "空闲"]}],
            "profile": [],
        })
        try:
            eng.record_turn("s", "user", "我最近迷上跑步了", scope="default")
            assert await eng.extract_session("s", scope="default", user_key="u1") == 1
            row = store.active_memories("default")[0]
            tags = store.parse_tags(row)
            assert "跑步" in tags and "u1" in tags
        finally:
            conn.close()

    @pytest.mark.asyncio
    async def test_tags_capped_and_deduped(self, make_engine):
        from core.engine import _clean_tags
        raw = ["a", "a", " b ", "", None, 1, "c", "d", "e", "f", "g"]
        out = _clean_tags(raw)
        # _clean_tags 只清洗不截断——截 6 个在 store._tags_to_json（落库唯一入口）
        assert out == ["a", "b", "1", "c", "d", "e", "f", "g"]
        assert _clean_tags("not-a-list") == []
        assert _clean_tags(None) == []

    @pytest.mark.asyncio
    async def test_store_caps_tags_at_six(self, make_engine):
        eng, store, conn = make_engine()
        try:
            await eng.remember("小明（u1）喜欢很多东西", alpha=0.9,
                               tags=["一", "二", "三", "四", "五", "六", "七", "八"])
            row = store.active_memories("default")[0]
            tags = store.parse_tags(row)
            assert len(tags) == 6, "落库层必须截到 6 个"
        finally:
            conn.close()

    @pytest.mark.asyncio
    async def test_parse_tags_robust(self, make_engine):
        eng, store, conn = make_engine()
        try:
            assert store.parse_tags({"tags_json": None}) == []
            assert store.parse_tags({"tags_json": "not-json"}) == []
            assert store.parse_tags({"tags_json": '["x","y"]'}) == ["x", "y"]
        finally:
            conn.close()


# ---------------------------------------------------------------- tag 检索通道
class TestTagChannel:
    @pytest.mark.asyncio
    async def test_tag_only_keyword_recallable(self, make_engine):
        """tag「跑步」不在正文里：正文 FTS/LIKE 全不中，仅 tag 通道命中。"""
        eng, store, conn = make_engine()
        try:
            await eng.remember("小明每周四下午有空去做运动", alpha=0.9,
                               tags=["跑步", "空闲"])
            cands = await eng.recall("跑步", scope="default", require_match=True,
                                     fast=True)
            assert cands, "tag 通道必须能召回（正文无该词）"
            assert any("运动" in c.content for c in cands)
            assert any("tag" in c.channels for c in cands), "receipt 应含 tag 通道"
        finally:
            conn.close()

    @pytest.mark.asyncio
    async def test_no_tag_no_ghost_hit(self, make_engine):
        eng, store, conn = make_engine()
        try:
            await eng.remember("小明喜欢猫", alpha=0.9)
            cands = await eng.recall("跑步", scope="default", require_match=True,
                                     fast=True)
            # 语义通道按设计无相似度地板（RRF+预算兜底），这里只钉死：
            # tag 通道不得幽灵命中（该记忆没有 tags）
            assert all("tag" not in c.channels for c in cands)
        finally:
            conn.close()


# ---------------------------------------------------------------- 演进动作
class TestEvolution:
    async def _seed_and_extract(self, make_engine, *, seed_contents, user_turn,
                                item, seed_active=False):
        """公共流程：播种旧记忆 → 记录本轮 → 抽取。返回 (eng, store, conn, payload)。"""
        eng, store, conn = make_engine(payload={
            "memories": [item], "profile": []})
        try:
            for c in seed_contents:
                await eng.remember(c, alpha=0.9,
                                   is_active=seed_active)
            eng.record_turn("s", "user", user_turn, scope="default")
            n = await eng.extract_session("s", scope="default", user_key="u1")
            return eng, store, conn, n
        except Exception:
            conn.close()
            raise

    @pytest.mark.asyncio
    async def test_update_supersedes_old(self, make_engine):
        eng, store, conn, n = await self._seed_and_extract(
            make_engine,
            seed_contents=["小明喜欢猫"],
            user_turn="小明喜欢猫 我现在对猫过敏了",
            item={"content": "小明（u1）对猫过敏，现在不喜欢猫了", "type": "fact",
                  "alpha": 0.9, "speaker": "user", "action": "update",
                  "update_ids": [1], "evidence": "「我对猫过敏了」",
                  "tags": ["猫", "过敏"]},
        )
        try:
            assert n == 1
            rows = store.active_memories("default")
            assert len(rows) == 1, "旧记忆必须退场，只留新记忆"
            assert "过敏" in rows[0]["content"]
            assert rows[0]["reasoning"] == "「我对猫过敏了」"
            assert store.parse_tags(rows[0]) == ["猫", "过敏"]
            old = [r for r in conn.execute(
                "SELECT * FROM memories WHERE content='小明喜欢猫'")][0]
            assert old["superseded_by"] == rows[0]["id"], "旧记忆必须软链到新记忆"
            assert old["deleted_at"] is not None, "演进旧行必须入回收站（第三轮审查修复）"
            assert any(r["id"] == old["id"] for r in store.list_trash()),                 "面板回收站必须可见（否则无法人工复核）"
        finally:
            conn.close()

    @pytest.mark.asyncio
    async def test_merge_inherits_proof_and_supersedes(self, make_engine):
        eng, store, conn, n = await self._seed_and_extract(
            make_engine,
            seed_contents=["小明喜欢跑步", "小明每周四去跑步"],
            user_turn="小明喜欢跑步 小明每周四去跑步 其实是一回事",
            item={"content": "小明（u1）喜欢跑步，常在周四去", "type": "fact",
                  "alpha": 0.9, "action": "merge", "merge_ids": [1, 2]},
        )
        try:
            assert n == 1
            rows = store.active_memories("default")
            assert len(rows) == 1 and "周四" in rows[0]["content"]
            assert rows[0]["proof_count"] == 2, "merge 继承 proof_count 总和"
            old_rows = list(conn.execute(
                "SELECT superseded_by FROM memories WHERE content LIKE '小明%' "
                "AND content != ?", (rows[0]["content"],)))
            assert len(old_rows) == 2
            assert all(r["superseded_by"] == rows[0]["id"] for r in old_rows)
        finally:
            conn.close()

    @pytest.mark.asyncio
    async def test_invalid_ids_degrade_to_create(self, make_engine):
        eng, store, conn, n = await self._seed_and_extract(
            make_engine,
            seed_contents=["小明喜欢猫"],
            user_turn="小明喜欢猫 随便聊聊",
            item={"content": "小明（u1）昨天去了猫咖", "type": "event",
                  "alpha": 0.9, "action": "update", "update_ids": [99]},
        )
        try:
            assert n == 1
            rows = store.active_memories("default")
            assert len(rows) == 2, "编号无效必须降级 create，旧记忆不动"
            assert any(r["content"] == "小明喜欢猫" for r in rows)
        finally:
            conn.close()

    @pytest.mark.asyncio
    async def test_active_memory_never_superseded(self, make_engine):
        """🔴 主动记忆不可被 LLM 演进取代（v0.1.7 收紧精神的延续）。"""
        eng, store, conn, n = await self._seed_and_extract(
            make_engine,
            seed_contents=["小明的生日是10月9日"],
            user_turn="小明的生日是10月9日 改成10月10日了",
            item={"content": "小明（u1）的生日其实是10月10日", "type": "fact",
                  "alpha": 0.9, "action": "update", "update_ids": [1]},
            seed_active=True,
        )
        try:
            assert n == 1
            old = [r for r in conn.execute(
                "SELECT * FROM memories WHERE content='小明的生日是10月9日'")][0]
            assert old["is_active"] == 1 and old["superseded_by"] is None, \
                "主动记忆必须原样保留"
        finally:
            conn.close()

    @pytest.mark.asyncio
    async def test_evolve_disabled_no_injection(self, make_engine):
        """开关关闭：无相关旧记忆段，update 必然降级 create。"""
        eng, store, conn = make_engine({"memory_behavior": {"evolve_enabled": False}},
                                       payload={
            "memories": [{"content": "小明（u1）现在不喜欢猫了", "type": "fact",
                          "alpha": 0.9, "action": "update", "update_ids": [1]}],
            "profile": []})
        try:
            await eng.remember("小明喜欢猫", alpha=0.9)
            captured = {}
            orig = eng.llm.generate_json

            async def spy(prompt, system_prompt=None):
                captured["prompt"] = prompt
                return await orig(prompt, system_prompt=system_prompt)

            eng.llm.generate_json = spy
            eng.record_turn("s", "user", "我现在不喜欢猫了", scope="default")
            await eng.extract_session("s", scope="default", user_key="u1")
            assert "（无）" in captured["prompt"]
            rows = store.active_memories("default")
            assert len(rows) == 2, "关闭演进时旧记忆不得被动"
        finally:
            conn.close()

    @pytest.mark.asyncio
    async def test_related_section_in_prompt(self, make_engine):
        """相关旧记忆以编号清单进入提示词。"""
        eng, store, conn = make_engine(payload={"memories": [], "profile": []})
        try:
            await eng.remember("小明喜欢跑步", alpha=0.9)
            captured = {}
            orig = eng.llm.generate_json

            async def spy(prompt, system_prompt=None):
                captured["prompt"] = prompt
                return await orig(prompt, system_prompt=system_prompt)

            eng.llm.generate_json = spy
            eng.record_turn("s", "user", "小明喜欢跑步 这周末还去吗", scope="default")
            await eng.extract_session("s", scope="default", user_key="u1")
            assert "[1] (fact) 小明喜欢跑步" in captured["prompt"]
        finally:
            conn.close()


# ---------------------------------------------------------------- 契约
class TestEvolutionContract:
    def test_prompt_declares_tags_and_actions(self):
        from core import templates
        for kw in ('"tags"', '"action"', '"update_ids"', '"merge_ids"', "相关旧记忆"):
            assert kw in templates.EXTRACT_PROMPT, f"提示词缺 {kw}"

    def test_tool_accepts_tags(self, make_engine):
        """工具 tags 参数进库。"""
        from tools.remember import MemoryRememberTool

        class _Ev:
            get_session_id = staticmethod(lambda: "s1")
            get_sender_id = staticmethod(lambda: "u1")

        async def _run():
            eng, store, conn = make_engine()
            try:
                eng._session_user["s1"] = "u1"
                ev = _Ev()
                ev.mnemoria_engine = eng
                await MemoryRememberTool().run(
                    ev, "小明（u1）每周四打篮球", tags=["篮球", "周四"])
                row = store.active_memories("default")[0]
                assert "篮球" in store.parse_tags(row)
            finally:
                conn.close()

        asyncio.run(_run())


# ---------------------------------------------------------------- 审查修复回归
class TestEvolutionGuards:
    @pytest.mark.asyncio
    async def test_self_reference_never_superseded(self, make_engine):
        """🔴 守卫A：LLM 用重复内容「update」同一条 → 只强化，绝不自指向
        supersede（自指向会把记忆挤出检索面，等于变相删除）。"""
        eng, store, conn = make_engine()
        try:
            await eng.remember("小明喜欢猫", alpha=0.9)
            mid = store.active_memories("default")[0]["id"]
            eng._apply_evolution("小明喜欢猫", [mid], "update", "default")
            row = store.get_memory(mid)
            assert row["superseded_by"] is None, "自指向 supersede = 记忆丢失"
            assert row["deleted_at"] is None
        finally:
            conn.close()

    @pytest.mark.asyncio
    async def test_superseded_hash_follows_chain_to_live(self, make_engine):
        """🔴 守卫B：演进目标哈希命中退场行 → 沿 superseded_by 链找活口，
        引用只能挂到活记忆上。"""
        eng, store, conn = make_engine()
        try:
            await eng.remember("旧说法A", alpha=0.9)
            await eng.remember("新说法B", alpha=0.9)
            await eng.remember("待合并C", alpha=0.9)
            rows = {r["content"]: r["id"] for r in store.active_memories("default")}
            a_id, b_id, c_id = rows["旧说法A"], rows["新说法B"], rows["待合并C"]
            store.supersede(a_id, b_id)  # A 已退场，但其哈希仍是 "旧说法A"
            eng._apply_evolution("旧说法A", [c_id], "update", "default")
            c_row = store.get_memory(c_id)
            assert c_row["superseded_by"] == b_id, \
                "引用必须沿链挂到活记忆 B，而不是退场的 A"
            assert store.get_memory(b_id)["superseded_by"] is None
        finally:
            conn.close()

    @pytest.mark.asyncio
    async def test_evolution_restore_roundtrip(self, make_engine):
        """v0.2.14：普通 restore 保留取代血缘，旧说法留在回收站不回检索面；
        clear_superseded=True 才是明确的彻底恢复。"""
        eng, store, conn = make_engine(payload={
            "memories": [{"content": "小明（u1）现在不喜欢猫了", "type": "fact",
                          "alpha": 0.9, "action": "update", "update_ids": [1]}],
            "profile": []})
        try:
            await eng.remember("小明喜欢猫", alpha=0.9)
            eng.record_turn("s", "user", "小明喜欢猫 我现在不喜欢猫了", scope="default")
            await eng.extract_session("s", scope="default", user_key="u1")
            trashed = [r for r in store.list_trash()
                       if r["content"] == "小明喜欢猫"]
            assert len(trashed) == 1
            old_id = trashed[0]["id"]
            store.restore(old_id)
            row = store.get_memory(old_id)
            assert row["deleted_at"] is not None
            assert row["superseded_by"] is not None, "普通恢复必须保留取代血缘"
            assert row["valid_to"] is not None, "普通恢复必须保留双时态终点"
            assert any(r["id"] == old_id for r in store.list_trash()), \
                "保留血缘的旧说法必须继续在回收站可见"
            cands = await eng.recall("小明喜欢猫", scope="default",
                                     require_match=True, fast=True)
            assert all(c.id != old_id for c in cands), "旧说法不得重新进入检索"
            store.restore(old_id, clear_superseded=True)
            row = store.get_memory(old_id)
            assert row["superseded_by"] is None and row["valid_to"] is None
            cands = await eng.recall("小明喜欢猫", scope="default",
                                     require_match=True, fast=True)
            assert any(c.id == old_id for c in cands), "彻底恢复后必须重新可检索"
        finally:
            conn.close()

    @pytest.mark.asyncio
    async def test_mixed_actions_one_extraction(self, make_engine):
        """端到端：一次抽取同时产出 create + update + merge。"""
        eng, store, conn = make_engine(payload={
            "memories": [
                {"content": "小明（u1）上周去了猫咖", "type": "event",
                 "alpha": 0.9, "action": "create", "tags": ["猫咖"]},
                {"content": "小明（u1）现在喜欢狗了", "type": "fact",
                 "alpha": 0.9, "action": "update", "update_ids": [1],
                 "tags": ["宠物"]},
                {"content": "小明（u1）周四跑步周五跑步合一条", "type": "fact",
                 "alpha": 0.9, "action": "merge", "merge_ids": [2, 3]},
            ],
            "profile": []})
        try:
            await eng.remember("小明喜欢猫", alpha=0.9)
            await eng.remember("小明周四跑步", alpha=0.9)
            await eng.remember("小明周五跑步", alpha=0.9)
            eng.record_turn("s", "user",
                            "小明喜欢猫 小明周四跑步 小明周五跑步 最近变了",
                            scope="default")
            n = await eng.extract_session("s", scope="default", user_key="u1")
            assert n == 3, f"三条动作都应写入，实际 {n}"
            active = {r["content"]: r for r in store.active_memories("default")}
            assert "小明(u1)上周去了猫咖" in active, "create 应入库"
            assert "小明(u1)现在喜欢狗了" in active, "update 新条应入库"
            assert "小明(u1)周四跑步周五跑步合一条" in active, "merge 新条应入库"
            # 旧的：猫（update）+ 两条跑步（merge）全部退场入回收站
            assert len(store.active_memories("default")) == 3
            trashed_contents = {r["content"] for r in store.list_trash()}
            assert {"小明喜欢猫", "小明周四跑步", "小明周五跑步"} <= trashed_contents
        finally:
            conn.close()

    def test_import_roundtrips_tags(self, tmp_path):
        """导出 JSON 的 tags_json 导入后还原（v0.1.8 前会静默丢 tags）。"""
        import json as _json
        sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
        from import_export import do_import
        from core import db as dbm
        from core.paths import DataPaths
        from core.store import MemoryStore

        dump = tmp_path / "dump.json"
        dump.write_text(_json.dumps({
            "memories": [{"content": "带标签的记忆", "memory_type": "fact",
                          "scope": "default", "tags_json": '["篮球", "周四"]'}],
        }, ensure_ascii=False), encoding="utf-8")
        paths = DataPaths(tmp_path / "pd").ensure()
        conn = dbm.connect(paths.db)
        dbm.init_schema(conn)
        do_import(paths.db, dump)
        store = MemoryStore(conn)
        row = store.active_memories("default")[0]
        assert store.parse_tags(row) == ["篮球", "周四"]
        conn.close()
