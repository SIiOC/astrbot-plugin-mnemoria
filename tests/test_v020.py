"""v0.2.0 记忆质量升级回归测试。

覆盖五条链路：
1. schema v5 无损迁移 + 升级前一致性备份；
2. 写入裁决（add/reinforce/merge/update/noop、低置信/超时/编号守卫回退）；
3. 夜间退休评审加固（快照、审计、低置信、超时整批跳过）；
4. 笔记切片派生层（切分、检索、重建、回退、配额、清理）；
5. 稳定身份账本 platform:user_id 与记忆/抽取接线。
"""

from __future__ import annotations

import asyncio
import json
import sqlite3

import pytest


# ------------------------------------------------------------------ 工具
class VecEmbedder:
    """按文本查表的确定性嵌入（可控相似度）。"""

    enabled = True

    def __init__(self, mapping: dict | None = None):
        self.mapping = mapping or {}

    def _v(self, text: str):
        return list(self.mapping.get(text, [0.0, 1.0, 0.0]))

    async def embed_one(self, text, timeout=None):
        return self._v(text)

    async def embed(self, texts):
        return [self._v(t) for t in texts]


class FakeLLM:
    enabled = True
    provider_id = "fake-provider"

    def __init__(self, payload, delay: float = 0.0):
        self.payload = payload
        self.delay = delay
        self.calls = 0
        self.prompts: list[str] = []

    async def generate_json(self, prompt, system_prompt=None):
        self.calls += 1
        self.prompts.append(prompt)
        if self.delay:
            await asyncio.sleep(self.delay)
        return self.payload

    async def generate(self, prompt, system_prompt=None):
        return ""


def _mk_engine(tmp_path, conf=None, llm=None, embedder=None, name="pdv020"):
    from core import config as cfgmod, db as dbm
    from core.engine import MemoryEngine
    from core.paths import DataPaths
    from core.store import MemoryStore

    paths = DataPaths(tmp_path / name).ensure()
    conn = dbm.connect(paths.db)
    dbm.init_schema(conn)
    store = MemoryStore(conn)
    eng = MemoryEngine(store, cfgmod.Config(conf or {}, paths.meta),
                       embedder=embedder, llm=llm)
    return eng, store, conn, paths


# ------------------------------------------------------------------ 1. schema v5
class TestSchemaV5:
    def test_fresh_schema_has_v5_objects(self, make_engine):
        eng, store, conn = make_engine()
        try:
            cols = [r[1] for r in conn.execute("PRAGMA table_info(memories)").fetchall()]
            assert "speaker_key" in cols
            tables = {r[0] for r in conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            ).fetchall()}
            assert {"note_chunks", "memory_events", "user_ledger", "group_ledger"} <= tables
            chunks = {r[0] for r in conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name LIKE 'note_chunks_fts%'"
            ).fetchall()}
            assert chunks, "切片 FTS 表必须建立"
        finally:
            conn.close()

    def test_v4_db_migrates_with_backup(self, tmp_path):
        """旧 v4 库：先落一致性备份，再无损补列建表，旧数据不动。

        运动建 v4 结构（不含 speaker_key 与 v5 新表），避免依赖
        DROP COLUMN（该列上有索引，SQLite 不允许直接删）。
        """
        from core import db as dbm
        from core.paths import DataPaths

        paths = DataPaths(tmp_path / "pd").ensure()
        raw = sqlite3.connect(str(paths.db))
        raw.executescript("""
CREATE TABLE meta (key TEXT PRIMARY KEY, value TEXT);
CREATE TABLE memories (
    id TEXT PRIMARY KEY, content TEXT NOT NULL, reasoning TEXT DEFAULT '',
    memory_type TEXT DEFAULT 'fact', source TEXT DEFAULT 'user', speaker TEXT DEFAULT '',
    is_active INTEGER DEFAULT 0, strength REAL DEFAULT 10.0, useful_score REAL DEFAULT 0.0,
    useful_count INTEGER DEFAULT 0, hit_count INTEGER DEFAULT 0, last_recalled_at REAL DEFAULT 0,
    last_decay_at REAL DEFAULT 0, proof_count INTEGER DEFAULT 1, observed_at REAL DEFAULT 0,
    valid_from REAL DEFAULT 0, valid_to REAL, superseded_by TEXT, deleted_at REAL,
    quarantined INTEGER DEFAULT 0, tags_json TEXT DEFAULT '[]', scope TEXT DEFAULT 'public',
    session_id TEXT DEFAULT '', content_hash TEXT DEFAULT '', created_at REAL DEFAULT 0,
    updated_at REAL DEFAULT 0
);
CREATE TABLE profiles (
    scope TEXT NOT NULL, user_key TEXT NOT NULL, key TEXT NOT NULL, value TEXT NOT NULL,
    confidence REAL DEFAULT 1.0, updated_at REAL DEFAULT 0, PRIMARY KEY (scope, user_key, key)
);
INSERT INTO meta(key, value) VALUES ('schema_version', '4'), ('fts5', '0');
INSERT INTO memories (id, content, scope, speaker, created_at, updated_at)
    VALUES ('m1', '迁移前记忆', 'd', '旧昵称', 1, 1);
INSERT INTO profiles (scope, user_key, key, value, confidence, updated_at)
    VALUES ('d', 'u1', '称呼', '小一', 1.0, 1);
""")
        raw.commit()
        raw.close()

        conn = dbm.connect(paths.db)
        try:
            assert dbm.has_existing_schema(conn), "旧库必须被识别为需要迁移保护"
            assert dbm.current_schema_version(conn) == 4
            backup = dbm.backup_before_migration(
                conn, paths.schema_backup_path(dbm.SCHEMA_VERSION))
            assert backup is not None and backup.exists(), "升级前必须生成一致性备份"
            bconn = sqlite3.connect(str(backup))
            backed = bconn.execute("SELECT content, speaker FROM memories").fetchall()
            bconn.close()
            assert backed and backed[0][0] == "迁移前记忆", "备份必须是升级前快照"

            dbm.init_schema(conn)
            cols = [r[1] for r in conn.execute("PRAGMA table_info(memories)").fetchall()]
            assert "speaker_key" in cols, "v4→v5 必须补 speaker_key 列"
            assert dbm.current_schema_version(conn) == dbm.SCHEMA_VERSION
            row = conn.execute(
                "SELECT content, speaker, speaker_key FROM memories WHERE id='m1'"
            ).fetchone()
            assert row["content"] == "迁移前记忆"
            assert row["speaker"] == "旧昵称", "旧 speaker 不得被改写"
            assert row["speaker_key"] == "", "迁移只补空列，不猜测旧身份"
            prof = conn.execute("SELECT value FROM profiles WHERE user_key='u1'").fetchone()
            assert prof["value"] == "小一", "画像数据必须无损"
            tables = {r[0] for r in conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table'").fetchall()}
            assert "memory_events" in tables and "note_chunks" in tables
            idx = conn.execute(
                "SELECT 1 FROM sqlite_master WHERE type='index' AND name='idx_mem_speaker'"
            ).fetchone()
            assert idx, "v5 身份索引必须建立"
        finally:
            conn.close()

    def test_new_db_no_backup(self, tmp_path):
        """全新空库不产生迁移备份（没有需要保护的数据）。"""
        from core import db as dbm
        paths = tmp_path / "fresh"
        paths.mkdir()
        target = paths / "pre-schema-v5-x.db"
        conn = dbm.connect(paths / "x.db")
        try:
            assert dbm.backup_before_migration(conn, target) is None
            assert not target.exists()
        finally:
            conn.close()


# ------------------------------------------------------------------ 2. 写入裁决
class TestAdjudicatedStore:
    def test_add_records_event_and_identity(self, make_engine):
        eng, store, conn = make_engine()
        try:
            res = store.adjudicated_write(
                action="add", scope="default", content="一条新记忆",
                speaker="u1", speaker_key="qq:u1",
                vec=[1.0, 0.0, 0.0], confidence=0.9, reason="test",
            )
            assert res and res["action"] == "add"
            row = store.get_memory(res["target_id"])
            assert row["speaker_key"] == "qq:u1"
            evs = store.list_memory_events(target_id=res["target_id"])
            assert evs and evs[0]["action"] == "add"
        finally:
            conn.close()

    def test_merge_supersedes_and_inherits(self, make_engine):
        eng, store, conn = make_engine()
        try:
            old = store.add_memory("旧说法", scope="default", proof_count=3)
            store.update_memory(old, useful_score=5.0)
            res = store.adjudicated_write(
                action="merge", scope="default", content="合并后说法",
                target_ids=[old], vec=[1.0, 0.0, 0.0], confidence=0.8,
            )
            assert res and res["action"] == "merge"
            new = store.get_memory(res["target_id"])
            assert new["proof_count"] == 3, "merge 必须继承证据计数"
            assert new["useful_score"] == 5.0, "merge 必须继承有用分"
            row = store.get_memory(old)
            assert row["superseded_by"] == res["target_id"]
            assert row["valid_to"] is not None and row["deleted_at"] is not None
            evs = store.list_memory_events(target_id=res["target_id"])
            assert json.loads(evs[0]["source_ids_json"]) == [old]
        finally:
            conn.close()

    def test_invalid_target_rolls_back(self, make_engine):
        eng, store, conn = make_engine()
        try:
            before = store.count()["total"]
            res = store.adjudicated_write(
                action="update", scope="default", content="新说法",
                target_ids=["does-not-exist"], vec=[1.0, 0.0, 0.0],
            )
            assert res is None, "无效目标必须整组回滚"
            assert store.count()["total"] == before, "不得留下半截新增"
        finally:
            conn.close()

    def test_active_memory_not_replaceable(self, make_engine):
        eng, store, conn = make_engine()
        try:
            old = store.add_memory("永生条目", scope="default", is_active=True)
            res = store.adjudicated_write(
                action="update", scope="default", content="想取代它",
                target_ids=[old], vec=[1.0, 0.0, 0.0],
            )
            assert res is None
            assert store.get_memory(old)["superseded_by"] is None
        finally:
            conn.close()

    def test_cross_scope_target_rejected(self, make_engine):
        eng, store, conn = make_engine()
        try:
            old = store.add_memory("别的域", scope="other")
            res = store.adjudicated_write(
                action="update", scope="default", content="跨域覆盖",
                target_ids=[old], vec=[1.0, 0.0, 0.0],
            )
            assert res is None
            assert store.get_memory(old)["superseded_by"] is None
        finally:
            conn.close()

    def test_reinforce_bumps_proof(self, make_engine):
        eng, store, conn = make_engine()
        try:
            mid = store.add_memory("被强化的", scope="default")
            res = store.adjudicated_write(
                action="reinforce", scope="default", content="x",
                target_ids=[mid], useful_delta=1.5, confidence=0.9,
            )
            assert res and res["target_id"] == mid
            row = store.get_memory(mid)
            assert row["proof_count"] == 2 and row["useful_score"] == 1.5
        finally:
            conn.close()

    def test_noop_writes_no_memory(self, make_engine):
        eng, store, conn = make_engine()
        try:
            before = store.count()["total"]
            res = store.adjudicated_write(
                action="noop", scope="default", content="重复噪声",
                target_ids=[], confidence=0.9,
            )
            assert res and res["action"] == "noop"
            assert store.count()["total"] == before
            evs = store.list_memory_events()
            assert evs and evs[0]["action"] == "noop"
        finally:
            conn.close()


class TestWriteAdjudicationFlow:
    V1 = [1.0, 0.0, 0.0]
    V2 = [0.85, 0.5268, 0.0]  # 与 V1 余弦 ≈ 0.85（裁决带内）

    def _setup(self, tmp_path, payload, conf=None, delay=0.0):
        emb = VecEmbedder({"小明喜欢猫": self.V1,
                           "小明现在喜欢猫科动物了": self.V2})
        llm = FakeLLM(payload, delay=delay)
        base = {"admission": {"write_adjudication_enabled": True}}
        if conf:
            base["admission"].update(conf.get("admission", {}))
            for k, v in conf.items():
                if k != "admission":
                    base[k] = v
        eng, store, conn, paths = _mk_engine(
            tmp_path, base, llm=llm, embedder=emb)
        return eng, store, conn, llm

    async def test_merge_via_llm(self, tmp_path):
        eng, store, conn, llm = self._setup(tmp_path, {
            "action": "merge", "target_ids": ["1"],
            "content": "小明喜欢猫科动物", "confidence": 0.9, "reason": "同偏好",
        })
        try:
            assert await eng.remember("小明喜欢猫", alpha=0.9) is True
            old_id = store.active_memories("default")[0]["id"]
            assert await eng.remember("小明现在喜欢猫科动物了", alpha=0.9) is True
            active = store.active_memories("default")
            assert len(active) == 1, "merge 后只留合并条"
            assert active[0]["content"] == "小明喜欢猫科动物"
            old = store.get_memory(old_id)
            assert old["superseded_by"] == active[0]["id"]
            assert old["deleted_at"] is not None, "旧条入回收站可恢复"
            assert llm.calls == 1
            evs = store.list_memory_events(target_id=active[0]["id"])
            assert evs[0]["action"] == "merge"
        finally:
            conn.close()

    async def test_low_confidence_falls_back_to_add(self, tmp_path):
        eng, store, conn, llm = self._setup(tmp_path, {
            "action": "merge", "target_ids": ["1"],
            "content": "不该被采纳", "confidence": 0.2,
        })
        try:
            await eng.remember("小明喜欢猫", alpha=0.9)
            await eng.remember("小明现在喜欢猫科动物了", alpha=0.9)
            active = store.active_memories("default")
            assert len(active) == 2, "低置信必须回退普通新增"
            assert all(r["superseded_by"] is None for r in active)
            assert llm.calls == 1
        finally:
            conn.close()

    async def test_timeout_falls_back_to_add(self, tmp_path):
        eng, store, conn, llm = self._setup(tmp_path, {
            "action": "merge", "target_ids": ["1"], "content": "超时内容",
            "confidence": 0.9,
        }, conf={"admission": {"adjudication_timeout_seconds": 0.01}}, delay=0.3)
        try:
            await eng.remember("小明喜欢猫", alpha=0.9)
            await eng.remember("小明现在喜欢猫科动物了", alpha=0.9)
            assert len(store.active_memories("default")) == 2
            assert llm.calls == 1
        finally:
            conn.close()

    async def test_number_only_guard_skips_llm(self, tmp_path):
        emb = VecEmbedder({"用户的第1条记忆": self.V1,
                           "用户的第10条记忆": self.V2})
        llm = FakeLLM({"action": "merge", "target_ids": ["1"],
                       "content": "不该合并", "confidence": 0.99})
        eng, store, conn, _ = _mk_engine(
            tmp_path, {"admission": {"write_adjudication_enabled": True}},
            llm=llm, embedder=emb)
        try:
            await eng.remember("用户的第1条记忆", alpha=0.9)
            await eng.remember("用户的第10条记忆", alpha=0.9)
            assert len(store.active_memories("default")) == 2, "仅编号不同不得合并"
            assert llm.calls == 0, "编号守卫应在调用 LLM 前拦截"
        finally:
            conn.close()

    async def test_noop_swallows_duplicate(self, tmp_path):
        eng, store, conn, llm = self._setup(tmp_path, {
            "action": "noop", "target_ids": [], "confidence": 0.9,
        })
        try:
            await eng.remember("小明喜欢猫", alpha=0.9)
            assert await eng.remember("小明现在喜欢猫科动物了", alpha=0.9) is True
            assert len(store.active_memories("default")) == 1, "noop 不应新增"
            assert llm.calls == 1
        finally:
            conn.close()

    async def test_disabled_adjudication_never_calls_llm(self, tmp_path):
        eng, store, conn, llm = self._setup(
            tmp_path,
            {"action": "merge", "target_ids": ["1"], "content": "x", "confidence": 0.99},
            conf={"admission": {"write_adjudication_enabled": False}})
        try:
            await eng.remember("小明喜欢猫", alpha=0.9)
            await eng.remember("小明现在喜欢猫科动物了", alpha=0.9)
            assert llm.calls == 0
            assert len(store.active_memories("default")) == 2
        finally:
            conn.close()

    async def test_dedup_reinforce_records_event(self, make_engine):
        eng, store, conn = make_engine()
        try:
            assert await eng.remember("完全相同的记忆", alpha=0.9) is True
            assert await eng.remember("完全相同的记忆", alpha=0.9) is True
            assert store.count()["total"] == 1
            evs = store.list_memory_events()
            assert evs and evs[0]["action"] == "reinforce"
            assert str(evs[0]["reason"]).startswith("dedup:")
        finally:
            conn.close()
    async def test_number_only_candidate_filtered_from_prompt(self, tmp_path):
        """一个合法候选混着一个编号模板候选：模板必须被逐条剔除，
        而不是只在“全都是模板”时才跳过。"""
        emb = VecEmbedder({"小明的第10条记录是跑步": [1.0, 0.0, 0.0]})
        llm = FakeLLM({"action": "update", "target_ids": ["1"],
                       "content": "合并结果", "confidence": 0.9})
        eng, store, conn, _ = _mk_engine(
            tmp_path, {"admission": {"write_adjudication_enabled": True}},
            llm=llm, embedder=emb)
        try:
            bad = store.add_memory("小明的第1条记录是跑步", scope="default")
            good = store.add_memory("小明的周四安排是跑步", scope="default")
            decision = await eng._adjudicate_write(
                "小明的第10条记录是跑步", "default", [1.0, 0.0, 0.0],
                [(bad, "小明的第1条记录是跑步", [1.0, 0.0, 0.0]),
                 (good, "小明的周四安排是跑步", [0.9, 0.4358, 0.0])],
                "fact")
            assert decision is not None
            assert decision["target_ids"] == [good], "编号模板候选不得进候选清单"
            assert "小明的第1条记录是跑步" not in llm.prompts[0], "模板候选不得出现在提示词里"
        finally:
            conn.close()


# ------------------------------------------------------------------ 3. 退休评审
class TestRetirementV2:
    def _mk(self, tmp_path, payload, conf=None, delay=0.0, name="pdret2"):
        llm = FakeLLM(payload, delay=delay)
        eng, store, conn, paths = _mk_engine(tmp_path, conf or {}, llm=llm, name=name)
        return eng, store, conn, paths, llm

    async def test_snapshot_written_and_event_recorded(self, tmp_path):
        eng, store, conn, paths, llm = self._mk(tmp_path, {
            "delete": ["1"], "keep": [], "promote": [], "confidence": 0.9,
            "reason": "琐事",
        })
        try:
            mid = store.add_memory("该忘的琐事", scope="d", strength=1.0)
            st = await eng.review_retirement()
            assert st["deleted"] == 1
            snaps = list(paths.backups.glob("retirement-*.json"))
            assert snaps, "删除前必须落 JSON 快照"
            payload = json.loads(snaps[0].read_text(encoding="utf-8"))
            assert any(m["id"] == mid for m in payload["memories"])
            evs = store.list_memory_events(target_id=mid)
            assert evs and evs[0]["action"] == "retire_delete"
        finally:
            conn.close()

    async def test_low_confidence_keeps_batch(self, tmp_path):
        eng, store, conn, paths, llm = self._mk(tmp_path, {
            "delete": ["1"], "keep": [], "promote": [], "confidence": 0.1,
        })
        try:
            mid = store.add_memory("也许不该删", scope="d", strength=1.0)
            st = await eng.review_retirement()
            assert st["deleted"] == 0 and st["skipped"] == 1 and st["kept"] == 1
            assert store.get_memory(mid)["deleted_at"] is None
            assert not list(paths.backups.glob("retirement-*.json")), "低置信不应动手"
        finally:
            conn.close()

    async def test_timeout_skips_batch(self, tmp_path):
        eng, store, conn, paths, llm = self._mk(
            tmp_path, {"delete": ["1"], "keep": [], "promote": [], "confidence": 0.9},
            conf={"retirement": {"enabled": True, "timeout_seconds": 0.01}},
            delay=0.3)
        try:
            mid = store.add_memory("超时不该删", scope="d", strength=1.0)
            st = await eng.review_retirement()
            assert st["skipped"] == 1 and st["deleted"] == 0
            assert store.get_memory(mid)["deleted_at"] is None
        finally:
            conn.close()

    async def test_delete_and_promote_same_id(self, tmp_path):
        """同一编号同时进 delete 与 promote：delete 优先，不留「已删除但
        高分」的不一致状态（第三轮再审：模型偶发重复列同一编号）。"""
        eng, store, conn, paths, llm = self._mk(tmp_path, {
            "delete": ["1"], "keep": [], "promote": ["1"], "confidence": 0.9,
            "reason": "矛盾输出",
        })
        try:
            mid = store.add_memory("矛盾的裁决", scope="d", strength=1.0)
            st = await eng.review_retirement()
            assert st["deleted"] == 1 and st["promoted"] == 0
            row = store.get_memory(mid)
            assert row["deleted_at"] is not None
            evs = [e["action"] for e in store.list_memory_events(target_id=mid)]
            assert "retire_delete" in evs and "retire_promote" not in evs
        finally:
            conn.close()

    async def test_default_enabled_with_llm(self, tmp_path):
        """v0.2.0 起未配置 retirement.enabled 时默认开启（有 LLM 即跑）。"""
        eng, store, conn, paths, llm = self._mk(tmp_path, {
            "delete": [], "keep": ["1"], "promote": [], "confidence": 0.9,
        })
        try:
            store.add_memory("普通冷记忆", scope="d", strength=1.0)
            st = await eng.review_retirement()
            assert st["candidates"] == 1 and st["kept"] == 1
        finally:
            conn.close()


# ------------------------------------------------------------------ 4. 笔记切片
class TestNoteChunks:
    def test_split_content_short_and_long(self):
        from core.notes import split_content
        assert split_content("") == []
        assert split_content("短正文") == ["短正文"]
        body = "a" * 2000
        pieces = split_content(body, max_chars=700, overlap=100, max_pieces=8)
        assert len(pieces) == 4, f"2000 字符应切 4 片（步长 600），实得 {len(pieces)}"
        assert all(len(p) <= 700 for p in pieces)
        assert pieces[1][:50] in pieces[0], "切片间必须保留重叠上下文"

    def test_split_content_caps_pieces(self):
        from core.notes import split_content
        body = "句子。" * 3000
        pieces = split_content(body, max_chars=200, overlap=40, max_pieces=3)
        assert len(pieces) == 3, "超上限必须合并尾部而不是无限切片"
        joined = "".join(pieces)
        assert len(joined) >= len(body) - 10, "合并尾部不得丢失正文"

    async def test_add_note_builds_chunks(self, make_engine):
        eng, store, conn = make_engine()
        try:
            content = "背景。" + "无关内容。" * 150 + "\n第二部分 小明喜欢跑步并在周四去运动店。"
            nid = await eng.add_note(content, title="长笔记", scope="default")
            chunks = store.list_note_chunks(nid)
            assert len(chunks) >= 2, f"长笔记应切多片，实得 {len(chunks)}"
            assert all(int(c["vec_dim"]) > 0 for c in chunks), "切片应带向量"
        finally:
            conn.close()

    async def test_chunk_retrieval_hits_deep_paragraph(self, make_engine):
        eng, store, conn = make_engine()
        try:
            content = "背景。" + "无关内容。" * 150 + "\n第二部分 小明喜欢跑步并在周四去运动店。"
            nid = await eng.add_note(content, title="长笔记", scope="default")
            hits = await eng.notes_recall("跑步", scope="default", top_k=3)
            assert hits and hits[0]["id"] == nid
            assert hits[0]["via"] == "chunk", "有切片时应走段落检索"
            assert "跑步" in hits[0]["content"]
        finally:
            conn.close()

    async def test_legacy_note_without_chunks_still_retrievable(self, make_engine):
        """存量笔记未回填时行为与 v0.1.9 一致（整篇检索）。"""
        eng, store, conn = make_engine()
        try:
            nid = store.add_note("旧版整篇笔记 含特有关键词香草", scope="default",
                                 vec=[0.0, 1.0, 0.0])
            assert store.list_note_chunks(nid) == []
            hits = await eng.notes_recall("香草", scope="default", top_k=3)
            assert hits and hits[0]["id"] == nid
            assert hits[0]["via"] == "note"
        finally:
            conn.close()

    async def test_update_note_rebuilds_chunks(self, make_engine):
        eng, store, conn = make_engine()
        try:
            nid = await eng.add_note("旧内容 包含旧词条", title="t", scope="default")
            await eng.update_note(nid, content="全新内容 包含特有关键词羚羊角")
            row = store.get_note(nid)
            assert row["vec"] is not None, "内容变更后必须重嵌向量"
            chunks = store.list_note_chunks(nid)
            assert chunks, "内容变更后必须重建切片"
            hits = await eng.notes_recall("羚羊角", scope="default", top_k=3)
            assert hits and "羚羊角" in hits[0]["content"]
            assert all("旧内容" not in c["content"] for c in chunks), "旧切片必须清除"
        finally:
            conn.close()

    async def test_chunk_cap_limits_embeddings(self, tmp_path):
        from core import config as cfgmod, db as dbm
        from core.engine import MemoryEngine
        from core.paths import DataPaths
        from core.store import MemoryStore

        paths = DataPaths(tmp_path / "pdcap").ensure()
        conn = dbm.connect(paths.db)
        dbm.init_schema(conn)
        store = MemoryStore(conn)
        eng = MemoryEngine(store, cfgmod.Config({
            "notes": {"chunk_size_chars": 200, "chunk_overlap_chars": 40,
                      "max_chunks_per_note": 3},
        }, paths.meta), embedder=VecEmbedder(), llm=None)
        try:
            nid = await eng.add_note("长文。" + "填充文本。" * 500, scope="default")
            assert len(store.list_note_chunks(nid)) == 3
        finally:
            conn.close()

    async def test_scope_isolation(self, make_engine):
        eng, store, conn = make_engine()
        try:
            nid = await eng.add_note("A 域独有词条雪狐", scope="a")
            hits = await eng.notes_recall("雪狐", scope="b", top_k=5)
            assert all(h["id"] != nid for h in hits), "切片检索不得跨域泄漏"
        finally:
            conn.close()

    async def test_purge_note_cleans_chunks_and_fts(self, make_engine):
        eng, store, conn = make_engine()
        try:
            nid = await eng.add_note("将被彻底删除的词条鲸落", scope="default")
            assert store.list_note_chunks(nid)
            store.purge_note(nid)
            assert store.list_note_chunks(nid) == []
            from core.notes import chunk_lexical_ranks
            hits = chunk_lexical_ranks(store.conn, "鲸落", "default", 5, store.fts)
            assert all(r[0] != nid for r in hits), "FTS 不得残留已删笔记的切片"
        finally:
            conn.close()

    async def test_backfill_builds_missing_chunks(self, make_engine):
        eng, store, conn = make_engine()
        try:
            nid = store.add_note("存量笔记 关键词蜂鸟", scope="default",
                                 vec=[0.0, 1.0, 0.0])
            assert store.list_note_chunks(nid) == []
            n = await eng.backfill_note_chunks(limit=10)
            assert n == 1
            assert store.list_note_chunks(nid), "回填后应有切片"
        finally:
            conn.close()

    async def test_chunk_rrf_no_note_accumulation_bias(self, tmp_path):
        """同一笔记的多个切片不得各自累加 RRF 分，否则“切片多的噪声笔记”
        会压过真正精确命中的笔记（离线评测实测的排序缺陷）。"""

        class PrefixEmbedder:
            enabled = True

            def _v(self, text):
                if "火烈鸟" in text or "DECOY" in text:
                    return [1.0, 0.0, 0.0]
                return [0.0, 1.0, 0.0]

            async def embed_one(self, text, timeout=None):
                return self._v(text)

            async def embed(self, texts):
                return [self._v(t) for t in texts]

        eng, store, conn, paths = _mk_engine(
            tmp_path,
            {"notes": {"chunk_size_chars": 200, "chunk_overlap_chars": 40}},
            embedder=PrefixEmbedder(), name="pdrrf")
        try:
            good = await eng.add_note("短文：小明看到了火烈鸟。", scope="default")
            decoy = await eng.add_note("DECOY " * 350, scope="default")
            assert len(store.list_note_chunks(decoy)) >= 4, "噪声笔记需要多切片"
            hits = await eng.notes_recall("火烈鸟", scope="default", top_k=3)
            assert hits and hits[0]["id"] == good, \
                "精确命中的短笔记必须压过多切片噪声笔记"
        finally:
            conn.close()
    async def test_backfill_skipped_when_notes_disabled(self, tmp_path):
        """笔记库关闭时不得跑回填（省一次无意义的嵌入调用）。"""
        eng, store, conn, paths = _mk_engine(
            tmp_path, {"notes": {"enabled": False}}, embedder=VecEmbedder(),
            name="pdnoff")
        try:
            nid = store.add_note("存量笔记 关键词蜂鸟", scope="default",
                                 vec=[0.0, 1.0, 0.0])
            assert await eng.backfill_note_chunks(limit=10) == 0
            assert store.list_note_chunks(nid) == []
        finally:
            conn.close()

    def test_chunks_scan_respects_note_window(self, make_engine):
        """切片语义扫描必须只在最近 note_limit 篇窗口内（防大库全表解包）。"""
        eng, store, conn = make_engine()
        try:
            ids = []
            for i in range(3):
                nid = store.add_note(f"笔记{i} 内容", scope="default")
                store.replace_note_chunks(
                    nid, [{"content": f"切片{i}", "vec": [1.0, 0.0, 0.0]}])
                ids.append(nid)
            for i, nid in enumerate(ids):
                conn.execute("UPDATE notes SET updated_at=? WHERE id=?",
                             (100.0 + i, nid))
            conn.commit()
            rows = store.chunks_with_vectors(scope="default", note_limit=2)
            got = {r["note_id"] for r in rows}
            assert got == {ids[1], ids[2]}, "切片扫描必须限定在最近笔记窗口内"
        finally:
            conn.close()


# ------------------------------------------------------------------ 5. 身份账本
class TestIdentityLedger:
    def test_upsert_accumulates_names(self, make_engine):
        eng, store, conn = make_engine()
        try:
            key = store.upsert_user_identity("qq", "123", "小明")
            assert key == "qq:123"
            store.upsert_user_identity("qq", "123", "阿明")
            store.upsert_user_identity("qq", "123", "阿明")  # 重复不追加
            assert store.identity_aliases("qq:123") == ["小明", "阿明"]
            rows = store.list_identities()
            assert rows and rows[0]["platform"] == "qq"
        finally:
            conn.close()

    def test_identity_for_registers_and_maps_session(self, make_engine):
        eng, store, conn = make_engine()
        try:
            class E:
                def get_sender_id(self):
                    return "123"

                def get_session_id(self):
                    return "s1"

                def get_platform_name(self):
                    return "qq"

                def get_sender_name(self):
                    return "小明"

            scope, user_key, speaker_key = eng.identity_for(E())
            assert (scope, user_key, speaker_key) == ("default", "123", "qq:123")
            assert store.identity_aliases("qq:123") == ["小明"]
            assert eng._session_identity["s1"] == "qq:123"
            # active_sessions 三元素快照（需先有会话活动记录）
            eng.record_turn("s1", "user", "你好", scope="default")
            snap = dict((sid, (uk, sk)) for sid, uk, sk in eng.active_sessions())
            assert snap["s1"][1] == "qq:123"
        finally:
            conn.close()

    async def test_extraction_propagates_speaker_key(self, make_engine):
        eng, store, conn = make_engine(payload={
            "memories": [{"content": "小明（123）喜欢跑步", "type": "fact",
                          "alpha": 0.9, "speaker": "user"}],
            "profile": [],
        })
        try:
            class E:
                def get_sender_id(self):
                    return "123"

                def get_session_id(self):
                    return "s9"

                def get_platform_name(self):
                    return "qq"

                def get_sender_name(self):
                    return "小明"

            eng.identity_for(E())
            eng.record_turn("s9", "user", "我最近迷上跑步了", scope="default")
            n = await eng.extract_session("s9", scope="default", user_key="123")
            assert n == 1
            row = store.active_memories("default")[0]
            assert row["speaker_key"] == "qq:123", "抽取记忆必须带稳定身份键"
            assert row["speaker"] == "123", "旧 user_key 语义保持不变"
        finally:
            conn.close()

    def test_identity_for_without_platform_is_empty(self, make_engine):
        eng, store, conn = make_engine()
        try:
            class E:
                def get_sender_id(self):
                    return "u1"

                def get_session_id(self):
                    return "s"

            scope, user_key, speaker_key = eng.identity_for(E())
            assert speaker_key == "", "平台不可知时不得编造身份键"
            assert store.list_identities() == []
        finally:
            conn.close()

    def test_identity_registers_without_nickname(self, make_engine):
        """平台不提供昵称时也要落账本：身份锚点不依赖昵称存在。"""
        eng, store, conn = make_engine()
        try:
            class E:
                def get_sender_id(self):
                    return "u9"

                def get_session_id(self):
                    return "s9"

                def get_platform_name(self):
                    return "qq"

            eng.identity_for(E())
            rows = store.list_identities()
            assert len(rows) == 1 and rows[0]["user_id"] == "u9"
            assert store.identity_aliases("qq:u9") == []
            eng.identity_for(E())  # 二次调用幂等，不重复写
            assert len(store.list_identities()) == 1
        finally:
            conn.close()

    def test_identity_name_history_capped(self, make_engine):
        eng, store, conn = make_engine()
        try:
            for i in range(14):
                store.upsert_user_identity("qq", "u1", f"昵称{i}")
            names = store.identity_aliases("qq:u1")
            assert len(names) == 10, f"昵称历史上限 10，实得 {len(names)}"
            assert names[-1] == "昵称13"
        finally:
            conn.close()
