"""v0.2.5：写入裁决自适应超时 + 熔断 回归测试。

背景：线上提供商是推理型模型（mimo-v2.5-pro），9-20 日志实测裁决
66 成功 / 63 超时——固定 20s 阈值撞掉近半数裁决。超时的请求在服务端
继续计费，本地却拿不到 merge/update 判定，同 scope 写入还白卡 20s。
v0.2.5 把固定阈值改为「超时 ×2、成功 ×0.98 缓慢回收、封顶自适应」，
持续失败则熔断一段时间（期间写入直接走保守回退，不再调 LLM）。

覆盖：
- 超时后生效超时 ×2、封顶 cap；
- 成功后失败计数清零、学到的超时 ×0.98 回收且不低于基线；
- 连续超时到上限且满阈值 → 熔断：熔断期间不再调 LLM，直接保守回退；
- 冷却结束后自动恢复试探；
- 异常计入熔断（不要求到达超时上限）；
- 输出不可解析不计入熔断（提供商是通的，只是 JSON 抠不出来）；
- 熔断期间高相似候选仍走保守强化（回退路径不受熔断破坏）；
- schema 三个新键存在且带默认值。
"""

from __future__ import annotations

import asyncio
import json
import time
from pathlib import Path

import pytest

PLUGIN_ROOT = Path(__file__).resolve().parents[1]

_CONF = {"admission": {
    "adjudication_timeout_seconds": 0.05,
    "adjudication_timeout_max": 0.2,
    "adjudication_breaker_threshold": 3,
    "adjudication_breaker_cooldown_seconds": 600,
}}

_NEW_CONTENT = "用户最近迷上了羊毛毡"
_CAND = ("cand0001id", "用户的旧兴趣是跑步")


class SlowLLM:
    """延迟超过基线的假 LLM（触发真实 wait_for 超时）。"""

    enabled = True
    provider_id = "slow"

    def __init__(self, payload=None, delay: float = 0.5):
        self.payload = payload
        self.delay = delay
        self.calls = 0

    async def generate_json(self, prompt, system_prompt=None):
        self.calls += 1
        await asyncio.sleep(self.delay)
        return self.payload


class FastLLM:
    """立即出解的假 LLM。"""

    enabled = True
    provider_id = "fast"

    def __init__(self, payload):
        self.payload = payload
        self.calls = 0

    async def generate_json(self, prompt, system_prompt=None):
        self.calls += 1
        return self.payload


class BoomLLM:
    """每次调用都抛异常的假 LLM（网络/鉴权故障面）。"""

    enabled = True
    provider_id = "boom"

    def __init__(self):
        self.calls = 0

    async def generate_json(self, prompt, system_prompt=None):
        self.calls += 1
        raise RuntimeError("provider boom")


def _mk_engine(tmp_path, llm, conf=None, name="pdv025"):
    from core import config as cfgmod, db as dbm
    from core.engine import MemoryEngine
    from core.paths import DataPaths
    from core.store import MemoryStore

    paths = DataPaths(tmp_path / name).ensure()
    conn = dbm.connect(paths.db)
    dbm.init_schema(conn)
    store = MemoryStore(conn)
    eng = MemoryEngine(store, cfgmod.Config(conf or _CONF, paths.meta), llm=llm)
    return eng, store, conn


async def _adjudicate(eng):
    """向量候选 sim=1.0（同向量），超时/失败时保守回退必然命中强化。"""
    return await eng._adjudicate_write(
        _NEW_CONTENT, "default", [1.0, 0.0, 0.0],
        [(_CAND[0], _CAND[1], [1.0, 0.0, 0.0])], "fact")


class TestAdaptiveTimeout:
    async def test_timeout_doubles_learned_timeout(self, tmp_path):
        eng, store, conn = _mk_engine(tmp_path, SlowLLM())
        try:
            d = await _adjudicate(eng)
            assert eng._adj_fail_streak == 1
            assert eng._adj_learned_timeout == pytest.approx(0.1)
            # sim=1.0 候选 → 超时回退为保守强化而非普通新增
            assert d and d["action"] == "reinforce" and d.get("fallback")
        finally:
            conn.close()

    async def test_learned_timeout_capped(self, tmp_path):
        eng, store, conn = _mk_engine(tmp_path, SlowLLM())
        try:
            eng._adj_learned_timeout = 0.15  # ×2=0.3 超 cap(0.2)
            await _adjudicate(eng)
            assert eng._adj_learned_timeout == pytest.approx(0.2)
        finally:
            conn.close()

    async def test_success_resets_streak_and_decays(self, tmp_path):
        eng, store, conn = _mk_engine(
            tmp_path, FastLLM({"action": "add", "confidence": 0.9}))
        try:
            eng._adj_fail_streak = 2
            eng._adj_learned_timeout = 0.2
            d = await _adjudicate(eng)
            assert d and d["action"] == "add"
            assert eng._adj_fail_streak == 0
            # 缓慢回收（×0.98），不直接回落基线——推理模型每次都要想这么久
            assert eng._adj_learned_timeout == pytest.approx(0.2 * 0.98)
        finally:
            conn.close()

    async def test_decay_never_below_base(self, tmp_path):
        eng, store, conn = _mk_engine(
            tmp_path, FastLLM({"action": "add", "confidence": 0.9}))
        try:
            eng._adj_learned_timeout = 0.051  # ×0.98 后低于基线 0.05
            await _adjudicate(eng)
            assert eng._adj_learned_timeout == pytest.approx(0.05)
        finally:
            conn.close()


class TestBreaker:
    async def test_opens_after_threshold_at_cap(self, tmp_path):
        llm = SlowLLM()
        eng, store, conn = _mk_engine(tmp_path, llm)
        try:
            for _ in range(3):  # learned: 0.1 → 0.2(cap) → 0.2；第 3 次满阈值
                await _adjudicate(eng)
            assert eng._adj_fail_streak == 3
            assert eng._adj_breaker_until > time.time()
            # 熔断期间：不再调 LLM，直接保守回退（sim=1.0 → reinforce）
            calls_before = llm.calls
            d = await _adjudicate(eng)
            assert llm.calls == calls_before
            assert d and d["action"] == "reinforce" and d.get("fallback")
        finally:
            conn.close()

    async def test_recovers_after_cooldown(self, tmp_path):
        llm = FastLLM({"action": "add", "confidence": 0.9})
        eng, store, conn = _mk_engine(tmp_path, llm)
        try:
            eng._adj_breaker_until = time.time() - 1  # 冷却已过期
            d = await _adjudicate(eng)
            assert llm.calls == 1
            assert d and d["action"] == "add"
        finally:
            conn.close()

    async def test_exceptions_trip_breaker_without_cap(self, tmp_path):
        llm = BoomLLM()
        eng, store, conn = _mk_engine(
            tmp_path, llm,
            conf={"admission": {
                "adjudication_timeout_seconds": 0.05,
                "adjudication_breaker_threshold": 2,
                "adjudication_breaker_cooldown_seconds": 600,
            }}, name="pdv025boom")
        try:
            await _adjudicate(eng)
            assert eng._adj_breaker_until == 0.0  # 第 1 次未满阈值
            await _adjudicate(eng)
            # 异常不是「慢」：未到超时上限也熔断
            assert eng._adj_breaker_until > time.time()
            assert eng._adj_learned_timeout == 0.0
        finally:
            conn.close()

    async def test_unparseable_output_not_counted(self, tmp_path):
        """拿到响应但抠不出 JSON：提供商是通的，不得计入超时/熔断。"""
        llm = FastLLM(None)  # generate_json → None（桥接层 120s 内空响应同构）
        eng, store, conn = _mk_engine(tmp_path, llm)
        try:
            for _ in range(5):  # 远超熔断阈值
                d = await _adjudicate(eng)
                assert d and d.get("fallback")  # 仍走保守回退
            assert eng._adj_fail_streak == 0
            assert eng._adj_breaker_until == 0.0
        finally:
            conn.close()


class TestSchema:
    def test_adaptive_breaker_keys_present(self):
        schema = json.loads(
            (PLUGIN_ROOT / "_conf_schema.json").read_text(encoding="utf-8"))
        admission = schema["admission"]["items"]
        assert admission["adjudication_timeout_max"]["default"] == 90
        assert admission["adjudication_breaker_threshold"]["default"] == 5
        assert admission["adjudication_breaker_cooldown_seconds"]["default"] == 600
