"""v0.1.4 检索规模优化回归测试。

覆盖：归一化向量缓存路径与旧 DB 余弦路径的**排序等价性**（最强钉子）、
候选正文按需取回不丢失、恢复记忆置脏向量缓存、speaker 归一化公共函数、
笔记基础集 id-only 优化后输出仍完整。
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


# ---------------------------------------------------------------- 1 检索等价性
class TestRetrievalEquivalence:
    def _build(self, make_engine, n=12):
        eng, store, conn = make_engine()
        for i in range(n):
            store.add_memory(f"用户的生活事实条目第 {i} 号，记录偏好与日常事务", scope="default")
        rows = store.active_memories("default")
        # 确定性伪向量（含一个零向量边界）
        for i, r in enumerate(rows):
            if i == 3:
                store.set_vector(r["id"], [0.0] * 8)
            else:
                v = [0.0] * 8
                v[i % 8] = 1.0
                v[(i + 1) % 8] = 0.5 + i * 0.01
                store.set_vector(r["id"], v)
        return eng, store, conn, rows

    def test_normalized_cache_matches_db_cosine_ranking(self, make_engine):
        """核心钉子：vectors=归一化缓存 与 vectors=None(DB 余弦) 的排序逐位一致。"""
        from core.retrieve import hybrid_retrieve
        from core.vector import normalize_vec
        eng, store, conn, rows = self._build(make_engine)
        try:
            q = [0.3, 0.9, 0.1, 0.0, 0.2, 0.4, 0.0, 0.1]
            base = hybrid_retrieve(
                conn, query="生活事实", scope="default", query_vec=q,
                fts_ok=store.fts, top_k=8,
            )
            normed = [(mid, normalize_vec(v)) for mid, v in store.get_all_vectors()]
            fast = hybrid_retrieve(
                conn, query="生活事实", scope="default", query_vec=q,
                fts_ok=store.fts, top_k=8, vectors=normed,
            )
            assert [c.id for c in base] == [c.id for c in fast], \
                "点积快路径必须与 DB 余弦路径同排序"
            for a, b in zip(base, fast):
                assert a.rrf == pytest.approx(b.rrf, abs=1e-9)
                assert a.content == b.content, "按需取回的正文必须一致"
        finally:
            conn.close()

    def test_engine_recall_uses_normalized_cache(self, make_engine):
        """引擎 recall 走缓存路径：命中语义通道且内容完整。"""
        eng, store, conn, _rows = self._build(make_engine)
        try:
            import asyncio

            async def scenario():
                qv = await eng.embedder.embed_one("生活事实")
                cands = await eng.recall(
                    "生活事实", scope="default", mark_recalled=False,
                    query_vec=qv, top_k=5, token_budget=4000,
                )
                assert cands, "缓存路径必须能召回"
                assert all(c.content for c in cands), "候选正文不得为空"
                assert eng._vectors_dirty is False, "recall 后缓存应为已刷新态"
            asyncio.run(scenario())
        finally:
            conn.close()

    def test_cache_normalized_and_dirty_rebuild(self, make_engine):
        """缓存内容确为归一化向量；置脏后重建反映新向量。"""
        from core.vector import normalize_vec
        eng, store, conn, _rows = self._build(make_engine)
        try:
            eng._refresh_vector_cache()
            for mid, v in eng._cached_vectors:
                norm = sum(x * x for x in v) ** 0.5
                if norm > 0:
                    assert abs(norm - 1.0) < 1e-6, "缓存必须存归一化向量"
            m = store.add_memory("触发缓存置脏的新记忆", scope="default")
            store.set_vector(m, [0.5, 0.5, 0.5, 0.5, 0.0, 0.0, 0.0, 0.0])
            eng._vectors_dirty = True
            eng._refresh_vector_cache()
            ids = {mid for mid, _ in eng._cached_vectors}
            assert m in ids
            expect = normalize_vec([0.5, 0.5, 0.5, 0.5, 0.0, 0.0, 0.0, 0.0])
            got = next(v for mid, v in eng._cached_vectors if mid == m)
            assert got == pytest.approx(expect, abs=1e-9)
        finally:
            conn.close()

    def test_zero_vector_not_ranked(self, make_engine):
        """零向量（归一化后全零）与旧实现同判：得分 0，不产生虚假高排。"""
        from core.retrieve import _semantic_ranks
        q = [1.0, 0.0]
        ranked = _semantic_ranks(q, [("zero", [0.0, 0.0]), ("real", [0.9, 0.1])],
                                 top_n=5, normalized=True)
        assert ranked[0][0] == "real"
        assert ranked[0][2] > ranked[1][2] >= 0.0


class TestRestoreDirtyFlag:
    @pytest.mark.asyncio
    async def test_restore_sets_dirty(self, plugin, make_engine):
        """恢复记忆必须置脏向量缓存（v0.1.4 补，与编辑路径同纪律）。"""
        from core import web_api
        plugin.engine._vectors_dirty = False
        mid = plugin.store.add_memory("待恢复的记忆", scope="default")
        plugin.store.trash(mid)

        class Body(dict):
            pass

        async def fake_body():
            return {"id": mid}
        orig = web_api._body
        web_api._body = fake_body
        try:
            out = await web_api._restore_memory(plugin)
            assert out["restored"] == mid
            assert plugin.engine._vectors_dirty is True, "恢复后必须置脏"
        finally:
            web_api._body = orig


# ---------------------------------------------------------------- 4 归一化统一
class TestNormalizeClaimSource:
    def test_matrix(self):
        from core.engine import _normalize_claim_source
        assert _normalize_claim_source("assistant") == "assistant"
        assert _normalize_claim_source(" ASSISTANT ") == "assistant"
        assert _normalize_claim_source("user") == "user"
        assert _normalize_claim_source(None) == "user"
        assert _normalize_claim_source("") == "user"
        assert _normalize_claim_source("bogus") == "user"
        assert _normalize_claim_source(123) == "user"

    def test_memory_and_profile_paths_share_it(self):
        """两条路径必须共用同一函数（AST 契约：调用点存在且不再内联）。"""
        import ast
        from pathlib import Path as P
        src = P(__file__).resolve().parents[1] / "core" / "engine.py"
        tree = ast.parse(src.read_text(encoding="utf-8"))
        calls = [
            n for n in ast.walk(tree)
            if isinstance(n, ast.Call)
            and isinstance(n.func, ast.Name)
            and n.func.id == "_normalize_claim_source"
        ]
        assert len(calls) >= 2, "记忆与画像路径都须调用公共归一化函数"
        text = src.read_text(encoding="utf-8")
        assert 'str(item.get("speaker")' not in text, "内联归一化写法应已移除"


# ---------------------------------------------------------------- notes id-only
class TestNotesRetrieveIdOnly:
    def test_retrieve_output_complete(self, make_engine):
        """笔记基础集改 id-only 后：输出字段（含正文/标题）仍完整。"""
        eng, store, conn = make_engine()
        try:
            store.add_note("蓝山咖啡豆风味记录正文", title="咖啡笔记", scope="default")
            store.add_note("与查询无关的另一条", title="无关", scope="default")
            from core.notes import retrieve
            out = retrieve(store, query="咖啡", scope="default", top_k=3)
            assert out, "id-only 优化后仍须命中"
            hit = out[0]
            assert hit["title"] == "咖啡笔记"
            assert "咖啡豆" in hit["content"], "正文必须按需取回，不得为空"
        finally:
            conn.close()

    def test_empty_scope_returns_early(self, make_engine):
        from core.notes import retrieve
        eng, store, conn = make_engine()
        try:
            assert retrieve(store, query="任意", scope="empty_scope", top_k=3) == []
        finally:
            conn.close()
