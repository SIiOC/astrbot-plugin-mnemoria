"""离线自测：不依赖 AstrBot 框架，仅验证 core 纯逻辑与存储/检索/衰减链路。

运行： python tests/selftest_offline.py
"""

from __future__ import annotations

import asyncio
import os
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from _bootstrap import bootstrap  # noqa: E402

bootstrap()

from core import config as cfgmod  # noqa: E402
from core import db as dbm  # noqa: E402
from core.engine import MemoryEngine  # noqa: E402
from core.paths import DataPaths, utc_now_ts  # noqa: E402
from core.store import MemoryStore  # noqa: E402
from core.admission import Verdict, assess, dedup  # noqa: E402
from core import scoring  # noqa: E402
from core.text import content_hash, estimate_tokens, normalize  # noqa: E402
from core.vector import cosine, pack, unpack  # noqa: E402

PASS, FAIL = [], []


def check(name, cond):
    (PASS if cond else FAIL).append(name)
    print(("  ok  " if cond else "FAIL  ") + name)


class FakeEmbedder:
    enabled = True

    async def embed_one(self, text, timeout=None):
        # 极简确定性向量：按字符散列到 8 维
        v = [0.0] * 8
        for ch in text:
            v[ord(ch) % 8] += 1.0
        return v

    async def embed(self, texts):
        return [await self.embed_one(t) for t in texts]


class FakeLLM:
    enabled = True

    def __init__(self, payload):
        self.payload = payload

    async def generate_json(self, prompt, system_prompt=None):
        return self.payload

    async def generate(self, prompt, system_prompt=None):
        return ""


def make_engine(tmp, payload=None):
    paths = DataPaths(tmp).ensure()
    conn = dbm.connect(paths.db)
    dbm.init_schema(conn)
    store = MemoryStore(conn)
    conf = cfgmod.Config({}, paths.meta)
    llm = FakeLLM(payload) if payload is not None else None
    eng = MemoryEngine(store, conf, embedder=FakeEmbedder(), reranker=None, llm=llm)
    return eng, store, conn, paths


def test_text():
    print("[text]")
    check("normalize 折叠空白", normalize("  a   b  ") == "a b")
    check("hash 一致", content_hash("你好，世界") == content_hash("你好 世界"))
    check("token 估算>0", estimate_tokens("你好world") >= 2)


def test_vector():
    print("[vector]")
    check("pack/unpack 往返", unpack(pack([1.0, 2.0, 3.0]), 3) == [1.0, 2.0, 3.0])
    check("cosine 自相似=1", abs(cosine([1, 2, 3], [1, 2, 3]) - 1.0) < 1e-6)
    check("cosine 零向量=0", cosine([0, 0], [1, 1]) == 0.0)


def test_scoring():
    print("[scoring]")
    now = utc_now_ts()
    h0 = scoring.hotness(0, now, now, 7.0)
    h_many = scoring.hotness(20, now, now, 7.0)
    check("命中越多热度越高", h_many > h0)
    old = scoring.hotness(0, now - 30 * 86400, now, 7.0)
    check("越旧热度越低", old < h0)
    check("tier 分档", scoring.tier(0, 3, 10) == 0 and scoring.tier(5, 3, 10) == 1 and scoring.tier(12, 3, 10) == 2)
    check("访问延长半衰期", scoring.effective_half_life(7, 10) > 7)


def test_admission():
    print("[admission]")
    r = assess("嗯嗯", alpha=1.0, alpha_threshold=0.4, source="user", deny_assistant_claims=True)
    check("寒暄拒收", r.verdict == Verdict.REJECT)
    r = assess("sk-abcdefghijklmnopqrstuvwxyz012345", alpha=1.0, alpha_threshold=0.4, source="user", deny_assistant_claims=True)
    check("密钥隔离", r.verdict == Verdict.QUARANTINE)
    r = assess("用户的名字是张三，是一名高三学生", alpha=0.8, alpha_threshold=0.4, source="user", deny_assistant_claims=True)
    check("正常事实接收", r.verdict == Verdict.ACCEPT)
    r = assess("用户喜欢猫", alpha=0.1, alpha_threshold=0.4, source="user", deny_assistant_claims=True)
    check("低 α 拒收", r.verdict == Verdict.REJECT)
    r = assess("作为一个人工智能助手，我建议你多喝水", alpha=0.9, alpha_threshold=0.4, source="user", deny_assistant_claims=True)
    check("套话拒收", r.verdict == Verdict.REJECT)
    r = assess("用户喜欢猫", alpha=0.9, alpha_threshold=0.4, source="assistant", deny_assistant_claims=True)
    check("助手代述拒收", r.verdict == Verdict.REJECT)
    # 指纹去重
    d = dedup("用户喜欢猫", scope="s", existing=[("m1", "用户喜欢 猫", None)], new_vec=None, threshold=0.9)
    check("指纹去重→强化", d.verdict == Verdict.REINFORCE and d.target_id == "m1")


def test_store_and_engine():
    print("[store+engine]")
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
        eng, store, conn, paths = make_engine(tmp)

        async def run():
            # 写入
            ok1 = await eng.remember("用户的名字是张三", memory_type="fact", alpha=0.9, scope="default")
            ok2 = await eng.remember("用户的名字是张三", memory_type="fact", alpha=0.9, scope="default")  # 重复→强化
            ok3 = await eng.remember("嗯嗯", alpha=0.1, scope="default")  # 拒收
            check("新增成功", ok1 is True)
            check("重复走强化", ok2 is True)
            check("垃圾拒收", ok3 is False)
            cnt = store.count()
            check("库内仅 1 条", cnt["total"] == 1)
            row = store.active_memories("default")[0]
            check("重复写入使 proof_count>1", row["proof_count"] >= 2)

            # 主动记忆
            await eng.remember("用户养了一只叫豆豆的猫", alpha=1.0, scope="default", is_active=True)
            act = [r for r in store.active_memories("default") if r["is_active"]]
            check("主动记忆标记", len(act) == 1)

            # 检索
            cands = await eng.recall("名字", scope="default", top_k=5, token_budget=2000)
            check("检索命中", any("张三" in c.content for c in cands))
            check("召回后 hit_count 增加", store.get_memory(cands[0].id)["hit_count"] >= 1)

            # 账本
            eng.record_turn("s1", "user", "我最近在准备高考", scope="default")
            eng.record_turn("s1", "assistant", "加油", scope="default")
            led = store.search_ledger("高考", session_id="s1")
            check("账本检索命中", len(led) >= 1)

            # 画像
            store.upsert_profile("default", "u1", "称呼", "小张")
            pb = eng.profile_block("default", "u1")
            check("画像块含内容", "小张" in pb)

            # 抽取管线（使用独立库，模拟模型返回结构化 JSON）
            with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp2:
                eng3, store3, conn3, _ = make_engine(tmp2, payload={
                    "memories": [{"content": "用户正在准备高考", "type": "event", "alpha": 0.8}],
                    "profile": [{"key": "当前状态", "value": "备考高考", "confidence": 0.9}],
                })
                try:
                    eng3.record_turn("s2", "user", "我明年要参加高考了", scope="default")
                    n = await eng3.extract_session("s2", scope="default", user_key="u1")
                    check("抽取写入记忆", n >= 1 and store3.count()["total"] >= 1)
                    prof = store3.get_profile("default", "u1")
                    check("抽取更新画像", any(p["key"] == "当前状态" for p in prof))
                finally:
                    conn3.close()

            # 衰减
            store.update_memory(store.active_memories("default")[0]["id"], strength=0.5)
            stats = eng.decay_sweep("default")
            check("衰减扫描有动作", stats["scanned"] >= 1)
            # 主动记忆不应被淘汰
            act_ids = {r["id"] for r in store.active_memories("default") if r["is_active"]}
            trash_ids = {r["id"] for r in store.list_trash()}
            check("主动记忆免于淘汰", not (act_ids & trash_ids))

        try:
            asyncio.run(run())
        finally:
            conn.close()


def test_config_migration():
    print("[config]")
    with tempfile.TemporaryDirectory() as tmp:
        meta = Path(tmp) / "meta.json"
        c = cfgmod.Config({"provider_id": "x", "retrieval": {"rrf_k": 42}}, meta)
        check("点号取值", c.get("retrieval.rrf_k") == 42)
        check("缺省回落", c.get("nope.deep", "D") == "D")
        check("provider 属性", c.provider_id == "x")
        check("版本已写入", meta.exists())


if __name__ == "__main__":
    test_text()
    test_vector()
    test_scoring()
    test_admission()
    test_config_migration()
    test_store_and_engine()
    print(f"\n通过 {len(PASS)} / 失败 {len(FAIL)}")
    if FAIL:
        print("失败项：", ", ".join(FAIL))
        sys.exit(1)
