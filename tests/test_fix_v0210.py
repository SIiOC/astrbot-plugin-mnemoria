"""v0.2.10：数值配置「显式设 0 被 or 默认值吞掉」修复 回归测试。

`config.get(key, default) or default` 会把用户显式设置的 0 当 falsy 吞掉，
静默回落默认值——而这几个键的 0 都有明确语义：
- `reflection.penalty_ratio`（控制台滑块 min=0）：0 = 「召回但没用」不扣分 → 被吞成 0.5；
- `notes.inject_max_items`（min=0）：0 = 不注入笔记 → 被吞成 3；
- `memory_behavior.consolidation_text_floor`：0 = 关闭巩固的文本确认守卫 → 被吞成 0.55；
- `memory_behavior.consolidation_max_size`：0 被吞成 8（外层 max(2,…) 兜底，语义应为最小簇 2）。

修复：`Config.get_num()`——仅缺失/None/空串回落默认，0 原样透传；
上述 4 处调用点全部改用 get_num（core/engine.py ×3、main.py ×1）。

行为测试复用 v0.2.8 的 SameVec 夹具（任意文本余弦恒 1.0）：
floor=0 时文本守卫被关闭，余弦满分即并簇——旧 `or 0.55` 行为下本测试必红。
"""

from __future__ import annotations

import pytest

pytestmark = pytest.mark.asyncio


class TestGetNum:
    def _cfg(self, raw):
        from core.config import Config
        return Config(raw)

    def test_zero_int_preserved(self):
        assert self._cfg({"reflection": {"penalty_ratio": 0}}).get_num(
            "reflection.penalty_ratio", 0.5) == 0

    def test_zero_float_preserved(self):
        assert self._cfg({"memory_behavior": {"consolidation_text_floor": 0.0}}).get_num(
            "memory_behavior.consolidation_text_floor", 0.55) == 0.0

    def test_missing_falls_back(self):
        assert self._cfg({}).get_num("reflection.penalty_ratio", 0.5) == 0.5

    def test_none_falls_back(self):
        assert self._cfg({"reflection": {"penalty_ratio": None}}).get_num(
            "reflection.penalty_ratio", 0.5) == 0.5

    def test_empty_string_falls_back(self):
        # 文本框清空后保存会得到 ""，此时回落默认比 int(""/float("") 报错友好
        assert self._cfg({"reflection": {"penalty_ratio": ""}}).get_num(
            "reflection.penalty_ratio", 0.5) == 0.5

    def test_nonzero_passthrough(self):
        assert self._cfg({"reflection": {"penalty_ratio": 0.3}}).get_num(
            "reflection.penalty_ratio", 0.5) == 0.3


class SameVec:
    """任意文本余弦都是 1.0（都过 0.86 聚类地板）。"""

    enabled = True

    async def embed_one(self, text):
        return [1.0, 0.0, 0.0]

    async def embed(self, texts):
        return [[1.0, 0.0, 0.0] for _ in texts]


class MergingLLM:
    enabled = True
    provider_id = "fake"

    async def generate_json(self, prompt, system_prompt=None):
        return {"content": "合并后的记忆", "merged_ids": []}

    async def generate(self, p, s=None):
        return ""


def _mk(tmp_path, texts, raw_config):
    from core import config as cfgmod
    from core import db as dbm
    from core.engine import MemoryEngine
    from core.paths import DataPaths
    from core.store import MemoryStore

    paths = DataPaths(tmp_path / "pd").ensure()
    conn = dbm.connect(paths.db)
    dbm.init_schema(conn)
    store = MemoryStore(conn)
    for t in texts:
        mid = store.add_memory(t, scope="default", strength=30.0)
        store.set_vector(mid, [1.0, 0.0, 0.0])
    eng = MemoryEngine(store, cfgmod.Config(raw_config, paths.meta),
                       embedder=SameVec(), llm=MergingLLM())
    return conn, store, eng


# 同主语不同事实（两两文本 n-gram 相似度 < 0.55，v0.2.8 已实证）。
# 必须 ≥3 条：consolidate() 对活跃记忆 <3 条的 scope 直接跳过。
_THREE_TEXTS = [
    "用户Star Lantern在备考学术英语，明天要口译",
    "用户Star Lantern喜欢全糖果茶，一杯约450千卡",
    "用户Star Lantern的凯夫拉手机壳脏了用小苏打去味",
]


class TestConsolidationFloorZero:
    async def test_floor_zero_disables_text_guard(self, tmp_path):
        """显式 floor=0：文本确认守卫关闭，余弦满分即并簇。

        旧代码 `or 0.55` 会把 0 吞成 0.55，不相似文本过不了守卫，
        merged==0——本测试在旧行为下必红。
        """
        conn, _, eng = _mk(tmp_path, _THREE_TEXTS,
                           {"memory_behavior": {"consolidation_text_floor": 0}})
        try:
            stats = await eng.consolidate()
            assert stats["merged"] >= 1
            superseded = conn.execute(
                "SELECT count(*) FROM memories WHERE superseded_by IS NOT NULL"
            ).fetchone()[0]
            assert superseded >= 1
        finally:
            conn.close()

    async def test_floor_default_still_guards(self, tmp_path):
        """默认 0.55 不变：同主语不同事实仍不合并——
        防修复把文本守卫整个拆掉（守卫本体是 v0.2.8 的止血线）。"""
        conn, _, eng = _mk(tmp_path, _THREE_TEXTS, {})
        try:
            stats = await eng.consolidate()
            assert stats["merged"] == 0
        finally:
            conn.close()
