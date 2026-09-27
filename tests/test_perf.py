"""性能与规模测试：确认在数千条记忆下检索/写入不出现退化或超时。

不追求精确基准，只做「不爆炸」的守门测试（避免 O(n²) 或递归爆栈）。
"""

from __future__ import annotations

import time

import pytest


class TestScale:
    @pytest.mark.asyncio
    async def test_bulk_write_and_count(self, make_engine):
        eng, store, conn = make_engine()
        try:
            t0 = time.time()
            for i in range(500):
                await eng.remember(f"用户的第{i}条记忆内容，关于编号{i}的事实", alpha=0.9)
            elapsed = time.time() - t0
            assert store.count()["total"] == 500
            # 500 条写入应在合理时间完成（宽松上限，避免 CI 波动）
            assert elapsed < 60, f"500 条写入耗时 {elapsed:.1f}s，疑似性能退化"
        finally:
            conn.close()

    @pytest.mark.asyncio
    async def test_recall_speed_on_large_db(self, make_engine):
        eng, store, conn = make_engine()
        try:
            for i in range(800):
                await eng.remember(f"编号{i}的记忆：用户提到过主题{i}的一些事情", alpha=0.9)
            t0 = time.time()
            out = await eng.recall("主题123", top_k=8, token_budget=1000)
            elapsed = time.time() - t0
            assert out  # 应有结果
            assert elapsed < 10, f"800 条库检索耗时 {elapsed:.1f}s"
        finally:
            conn.close()

    @pytest.mark.asyncio
    async def test_decay_sweep_scales(self, make_engine):
        eng, store, conn = make_engine()
        try:
            for i in range(400):
                await eng.remember(f"待衰减记忆编号{i}", alpha=0.9)
            t0 = time.time()
            stats = eng.decay_sweep("default")
            elapsed = time.time() - t0
            assert stats["scanned"] == 400
            assert elapsed < 20, f"400 条衰减耗时 {elapsed:.1f}s"
        finally:
            conn.close()

    @pytest.mark.asyncio
    async def test_ledger_many_rows(self, make_engine):
        eng, store, conn = make_engine()
        try:
            for i in range(2000):
                eng.record_turn("s", "user", f"第{i}条历史消息内容", scope="default")
            rows = store.recent_ledger("s", limit=50)
            assert len(rows) == 50
            hit = store.search_ledger("第1999条", session_id="s", scope="default")
            assert len(hit) >= 1
        finally:
            conn.close()

    @pytest.mark.asyncio
    async def test_many_scopes_isolated(self, make_engine):
        eng, store, conn = make_engine()
        try:
            for s in range(30):
                await eng.remember(f"域{s}的专属记忆内容", alpha=0.9, scope=f"sc{s}")
            for s in range(30):
                out = await eng.recall("专属", scope=f"sc{s}", top_k=5, token_budget=500)
                assert len(out) == 1
                assert f"域{s}" in out[0].content
        finally:
            conn.close()
