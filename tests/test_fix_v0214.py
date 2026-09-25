"""v0.2.14 记忆质量修复回归测试。

覆盖：异维向量入队/惰性回填、回收站安全恢复、演进召回配置、
裁决失败的全候选文本守卫，以及一次性回填脚本的 dry-run 规划。
"""

from __future__ import annotations

import asyncio
import json
import sqlite3
from pathlib import Path

import pytest


class BatchEmbedder:
    enabled = True

    def __init__(self, dim=4, fail=False):
        self.dim = dim
        self.fail = fail
        self.calls = []

    async def embed_one(self, text, timeout=None):
        return [float(len(text) % 7 + 1)] * self.dim

    async def embed(self, texts, timeout=None):
        self.calls.append(list(texts))
        if self.fail:
            return None
        return [[float(i + 1)] * self.dim for i, _ in enumerate(texts)]


class TestVectorBacklog:
    def test_mismatched_retrieval_enqueues(self, make_engine):
        from core.retrieve import hybrid_retrieve

        eng, store, conn = make_engine()
        try:
            mid = store.add_memory("旧向量记忆", scope="default")
            store.set_vector(mid, [1.0, 2.0])
            out = hybrid_retrieve(
                conn, query="旧向量记忆", scope="default", query_vec=[1.0] * 4,
                fts_ok=store.fts, include_recency=False,
                vector_backlog=store.enqueue_vector_backlog,
            )
            assert out
            row = conn.execute(
                "SELECT dim_seen FROM vector_backlog WHERE memory_id=?", (mid,)
            ).fetchone()
            assert row and row[0] == 2
            assert all(c.id != mid or "semantic" not in c.channels for c in out)
        finally:
            conn.close()

    @pytest.mark.asyncio
    async def test_backfill_consumes_and_removes(self, make_engine):
        embedder = BatchEmbedder(dim=4)
        eng, store, conn = make_engine(embedder=embedder)
        try:
            mid = store.add_memory("待回填记忆", scope="default")
            store.set_vector(mid, [1.0, 2.0])
            assert store.enqueue_vector_backlog(mid, 2)
            result = await eng.backfill_vectors(batch=16)
            assert result["processed"] == 1
            assert store.vector_backlog_count() == 0
            row = conn.execute("SELECT dim FROM vectors WHERE memory_id=?", (mid,)).fetchone()
            assert row[0] == 4
            assert embedder.calls == [["待回填记忆"]]
        finally:
            conn.close()

    @pytest.mark.asyncio
    async def test_backfill_respects_batch_and_skips_trash(self, make_engine):
        embedder = BatchEmbedder(dim=4)
        eng, store, conn = make_engine(embedder=embedder)
        try:
            live = []
            for i in range(3):
                mid = store.add_memory(f"队列 {i}", scope="default")
                store.set_vector(mid, [1.0, 2.0])
                store.enqueue_vector_backlog(mid, 2)
                live.append(mid)
            trash = store.add_memory("回收站不回填", scope="default")
            store.set_vector(trash, [1.0, 2.0])
            store.trash(trash)
            store.enqueue_vector_backlog(trash, 2)
            result = await eng.backfill_vectors(batch=2)
            assert result["processed"] == 2
            assert store.vector_backlog_count() == 1
            assert conn.execute(
                "SELECT 1 FROM vector_backlog WHERE memory_id=?", (trash,)
            ).fetchone() is None
        finally:
            conn.close()

    @pytest.mark.asyncio
    async def test_backfill_failure_increments_attempts(self, make_engine):
        embedder = BatchEmbedder(dim=4, fail=True)
        eng, store, conn = make_engine(embedder=embedder)
        try:
            mid = store.add_memory("失败回填", scope="default")
            store.enqueue_vector_backlog(mid, 2)
            result = await eng.backfill_vectors()
            assert result["failed"] == 1
            attempts = conn.execute(
                "SELECT attempts FROM vector_backlog WHERE memory_id=?", (mid,)
            ).fetchone()[0]
            assert attempts == 1
        finally:
            conn.close()


class TestNightlyBackfillQuota:
    async def test_daily_limit_truncates_and_zero_disables(self, plugin):
        """main 层夜间循环：日限按剩余量截批；显式 0 禁用（不被 or 吞）。"""
        calls = []

        async def stub(batch=16):
            calls.append(batch)
            return {"processed": batch, "failed": 0, "skipped": 0}

        plugin.engine.backfill_vectors = stub
        raw = plugin.config._raw.setdefault("retrieval", {})
        raw["vector_backfill_daily_limit"] = 40
        await plugin._nightly("20260924")
        assert calls == [16, 16, 8], "日限 40 应截成 16+16+8 三批"
        calls.clear()
        raw["vector_backfill_daily_limit"] = 0
        await plugin._nightly("20260924b")
        assert calls == [], "daily_limit=0 必须完全禁用夜间消费"


class TestRestoreSemantics:
    def test_restore_preserves_supersede_until_explicit_clear(self, make_engine):
        _eng, store, conn = make_engine()
        try:
            old = store.add_memory("旧说法", scope="default")
            new = store.add_memory("新说法", scope="default")
            store.supersede(old, new)
            store.trash(old)
            store.restore(old)
            row = store.get_memory(old)
            assert row["deleted_at"] is not None
            assert row["superseded_by"] == new
            assert store.list_trash() and store.list_trash()[0]["id"] == old
            store.restore(old, clear_superseded=True)
            row = store.get_memory(old)
            assert row["superseded_by"] is None and row["valid_to"] is None
            assert not store.list_trash()
        finally:
            conn.close()


class TestFallbackAndConfig:
    def test_fallback_scans_all_candidates(self, make_engine):
        eng, store, conn = make_engine()
        try:
            decision = eng._fallback_reinforce(
                "用户喜欢跑步运动",
                [(0.81, "first", "用户住在北京"),
                 (0.82, "second", "用户喜欢跑步运动")],
            )
            assert decision and decision["target_ids"] == ["second"]
        finally:
            conn.close()

    @pytest.mark.asyncio
    async def test_evolve_recall_top_k_configured(self, make_engine):
        payload = {"memories": [], "profile": []}
        eng, store, conn = make_engine(conf={
            "memory_behavior": {"evolve_recall_top_k": 16}
        }, payload=payload)
        try:
            calls = []
            original = eng.recall

            async def wrapped(*args, **kwargs):
                calls.append(kwargs.get("top_k"))
                return await original(*args, **kwargs)

            eng.recall = wrapped
            eng.record_turn("s", "user", "用户喜欢跑步运动", scope="default")
            await eng.extract_session("s", scope="default", user_key="u1")
            assert calls and calls[0] == 16
        finally:
            conn.close()


def _load_script():
    import importlib.util
    path = Path(__file__).resolve().parents[1] / "scripts" / "reembed_vectors.py"
    spec = importlib.util.spec_from_file_location("reembed_vectors", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


class TestReembedScript:
    def test_plan_prefers_never_recalled_and_includes_missing(self, make_engine):
        mod = _load_script()
        _eng, store, conn = make_engine()
        try:
            old = store.add_memory("旧维度", scope="default")
            store.set_vector(old, [1.0, 2.0])
            hit = store.add_memory("新维度但有召回", scope="default")
            store.set_vector(hit, [1.0, 2.0])
            store.mark_recalled([hit])
            missing = store.add_memory("没有向量", scope="default")
            rows = mod.plan(conn, current_dim=4)
            ids = [r["id"] for r in rows]
            assert ids[0] in (old, missing)
            assert old in ids and missing in ids and hit in ids
        finally:
            conn.close()
