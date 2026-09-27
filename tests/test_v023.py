"""v0.2.3：重复存储 / 噪音入库 / 裁决超时三项修复的回归测试。

覆盖：
- admission.dedup 守卫三（文本级近重复）：换措辞的同事实判重复；
  不同事实不误判；仅编号不同的模板内容仍放行；
- 裁决超时/坏输出回退：首位候选高度相似 → 保守强化（防重复入库）；
  相似度不够 → 仍普通新增（宁可重复不丢事实）；
- 每轮抽取条数兜底：超过 max_extract_per_turn 按 alpha 取前 N 条；
- 提示词含五问筛选与「一条记忆一件事」。
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

import pytest

from core import admission
from core import db as dbm
from core.store import MemoryStore
from core.engine import MemoryEngine
from core import config as cfgmod
from core.paths import DataPaths

PLUGIN_ROOT = Path(__file__).resolve().parents[1]


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
    provider_id = "fake"

    def __init__(self, payload, delay: float = 0.0):
        self.payload = payload
        self.delay = delay
        self.calls = 0

    async def generate_json(self, prompt, system_prompt=None):
        self.calls += 1
        if self.delay:
            await asyncio.sleep(self.delay)
        return self.payload

    async def generate(self, prompt, system_prompt=None):
        return ""


def _mk(tmp_path, conf=None, llm=None, embedder=None, name="pdv023"):
    paths = DataPaths(tmp_path / name).ensure()
    conn = dbm.connect(paths.db)
    dbm.init_schema(conn)
    store = MemoryStore(conn)
    eng = MemoryEngine(store, cfgmod.Config(conf or {}, paths.meta),
                       embedder=embedder, llm=llm)
    return eng, store, conn, paths


# ---------------------------------------------------------------- 1. 文本级去重
class TestTextDedupGuard:

    def test_reworded_same_fact_reinforced_without_vector(self):
        """无向量时，换措辞的同事实（Jaccard 0.81）必须判重复。"""
        existing = [("m1", "Star Lantern(3141592653)喜欢跑步", None)]
        v = admission.dedup("Star Lantern(3141592653)爱跑步",
                            scope="d", existing=existing, new_vec=None,
                            threshold=0.92)
        assert v.verdict is admission.Verdict.REINFORCE, v.reason
        assert v.target_id == "m1"

    def test_different_facts_not_merged(self):
        """不同事实（实测 Jaccard 0.73）不得误判重复。"""
        existing = [("m1", "Star Lantern(3141592653)喜欢跑步", None)]
        v = admission.dedup("Star Lantern(3141592653)喜欢在深夜听广播",
                            scope="d", existing=existing, new_vec=None,
                            threshold=0.92)
        assert v.verdict is admission.Verdict.ACCEPT, v.reason

    def test_number_template_still_guarded(self):
        """仅编号不同的模板内容（文本相似度很高）绝不合并。"""
        existing = [("m1", "用户的第1条记忆内容", None)]
        v = admission.dedup("用户的第10条记忆内容",
                            scope="d", existing=existing, new_vec=None,
                            threshold=0.92)
        assert v.verdict is admission.Verdict.ACCEPT, v.reason

    def test_threshold_respected(self):
        """阈值可调：0.95 时 0.81 的对子不再判重复。"""
        existing = [("m1", "Star Lantern(3141592653)喜欢跑步", None)]
        v = admission.dedup("Star Lantern(3141592653)爱跑步",
                            scope="d", existing=existing, new_vec=None,
                            threshold=0.92, text_dedup=0.95)
        assert v.verdict is admission.Verdict.ACCEPT, v.reason

    def test_scan_cap_limits_work(self):
        """超过扫描上限的旧记忆不参与文本比对（写入路径开销有界）。"""
        existing = [(f"m{i}", f"第{i}号无关内容", None) for i in range(50)]
        existing.append(("mTarget", "Star Lantern(3141592653)喜欢跑步", None))
        v = admission.dedup("Star Lantern(3141592653)爱跑步",
                            scope="d", existing=existing, new_vec=None,
                            threshold=0.92, text_scan_cap=10)
        assert v.verdict is admission.Verdict.ACCEPT, "上限外的目标不应被扫到"


# ---------------------------------------------------------------- 2. 裁决回退
class TestConservativeFallback:

    def _setup(self, tmp_path, payload, delay, sim_high=True):
        # 基准向量 30°；high=55°夹角(cos≈0.906，低于去重线0.92、高于保守线0.90
        # ——v0.2.11 dashscope 重标定：保守线 0.85→0.90，旧 0.87 档已改为 0.906 档)；
        # low=67°夹角(cos≈0.80，仍≥裁决候选线0.78、但低于保守线0.90)
        v_base = [0.866, 0.5, 0.0]
        v_new = [0.574, 0.819, 0.0] if sim_high else [0.393, 0.919, 0.0]
        emb = VecEmbedder({"小明喜欢跑步": v_base,
                           "小明热衷运动艺术创作": v_new})
        llm = FakeLLM(payload, delay=delay)
        conf = {"admission": {"write_adjudication_enabled": True,
                              "adjudication_timeout_seconds": 0.05}}
        return _mk(tmp_path, conf, llm=llm, embedder=emb)

    @pytest.mark.asyncio
    async def test_timeout_high_similarity_reinforces(self, tmp_path):
        """超时 + 首位候选 0.906 → 保守强化，不得裸新增（用户实测痛症）。"""
        eng, store, conn, _ = self._setup(tmp_path, {}, 0.3, sim_high=True)
        try:
            assert await eng.remember("小明喜欢跑步", alpha=0.9) is True
            first = store.active_memories("default")[0]
            assert await eng.remember("小明热衷运动艺术创作", alpha=0.9) is True
            active = store.active_memories("default")
            assert len(active) == 2, "高向量但文本差异大时不得误合并"
            assert any(r["id"] == first["id"] for r in active)
            evs = store.list_memory_events(target_id=first["id"])
            assert not any(e["action"] == "reinforce" for e in evs), "文本确认失败不得伪造强化审计"
        finally:
            conn.close()

    @pytest.mark.asyncio
    async def test_timeout_low_similarity_still_adds(self, tmp_path):
        """超时 + 相似度只有 0.80 → 仍普通新增（宁可重复不丢事实）。"""
        eng, store, conn, _ = self._setup(tmp_path, {}, 0.3, sim_high=False)
        try:
            assert await eng.remember("小明喜欢跑步", alpha=0.9) is True
            assert await eng.remember("小明热衷运动艺术创作", alpha=0.9) is True
            assert len(store.active_memories("default")) == 2
        finally:
            conn.close()

    @pytest.mark.asyncio
    async def test_bad_output_high_similarity_reinforces(self, tmp_path):
        """坏输出（非 dict）+ 高相似 → 同样保守强化。"""
        eng, store, conn, _ = self._setup(tmp_path, ["not", "a", "dict"], 0.0,
                                          sim_high=True)
        try:
            assert await eng.remember("小明喜欢跑步", alpha=0.9) is True
            assert await eng.remember("小明热衷运动艺术创作", alpha=0.9) is True
            assert len(store.active_memories("default")) == 2
        finally:
            conn.close()

    @pytest.mark.asyncio
    async def test_normal_adjudication_unaffected(self, tmp_path):
        """裁决正常时行为不变（merge 仍按模型决策走）。"""
        eng, store, conn, _ = self._setup(tmp_path, {
            "action": "merge", "target_ids": ["1"],
            "content": "小明喜欢猫科动物", "confidence": 0.9, "reason": "同偏好",
        }, 0.0, sim_high=True)
        try:
            assert await eng.remember("小明喜欢跑步", alpha=0.9) is True
            old_id = store.active_memories("default")[0]["id"]
            assert await eng.remember("小明热衷运动艺术创作", alpha=0.9) is True
            active = store.active_memories("default")
            assert len(active) == 1
            assert active[0]["content"] == "小明喜欢猫科动物"
            assert store.get_memory(old_id)["deleted_at"] is not None
        finally:
            conn.close()


# ---------------------------------------------------------------- 3. 每轮条数兜底
class TestExtractCap:

    @pytest.mark.asyncio
    async def test_over_cap_keeps_top_alpha(self, tmp_path):
        items = [{"content": f"第{i}号事实内容", "type": "fact",
                  "alpha": 0.3 + i * 0.05, "speaker": "user"}
                 for i in range(8)]
        eng, store, conn, _ = _mk(tmp_path, llm=FakeLLM(
            {"memories": items, "profile": []}))
        try:
            eng.record_turn("s1", "user", "聊了很多事实", scope="default")
            n = await eng.extract_session("s1", scope="default", user_key="u1")
            active = store.active_memories("default")
            assert n == 6 and len(active) == 6, "超过 6 条必须按 alpha 截断"
            kept = {m["content"] for m in active}
            assert "第7号事实内容" in kept and "第6号事实内容" in kept
            assert "第0号事实内容" not in kept and "第1号事实内容" not in kept
        finally:
            conn.close()

    @pytest.mark.asyncio
    async def test_under_cap_unchanged(self, tmp_path):
        items = [{"content": f"第{i}号事实内容", "type": "fact",
                  "alpha": 0.9, "speaker": "user"} for i in range(3)]
        eng, store, conn, _ = _mk(tmp_path, llm=FakeLLM(
            {"memories": items, "profile": []}))
        try:
            eng.record_turn("s1", "user", "聊了三件事", scope="default")
            n = await eng.extract_session("s1", scope="default", user_key="u1")
            assert n == 3
        finally:
            conn.close()


# ---------------------------------------------------------------- 4. 提示词
class TestPromptFilters:

    def test_five_questions_present(self):
        from core import templates
        p = templates.EXTRACT_PROMPT
        assert "五问筛选" in p
        for kw in ("明确对话依据", "临时状态", "低价值噪音", "高敏信息", "贬损性判断"):
            assert kw in p, f"五问缺少：{kw}"
        assert "一条记忆一件事" in p
