"""全功能场景验证矩阵：逐组逐项在各条件下真实驱动引擎，验证行为契约。

运行： python tests/feature_matrix.py

组别：
  A 抽取触发时机      B 用户指称链        C speaker×闸门矩阵
  D 记忆演进          E 检索四路          F 衰减/回收站
  G 反思闭环          H 工具四件套        I 注入组装
  J 备份/导入往返
"""

from __future__ import annotations

import asyncio
import os
import sys
import tempfile
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from _bootstrap import bootstrap  # noqa: E402

bootstrap()

from core import config as cfgmod  # noqa: E402
from core import db as dbm  # noqa: E402
from core.engine import MemoryEngine  # noqa: E402
from core.paths import DataPaths, utc_now_ts  # noqa: E402
from core.store import MemoryStore  # noqa: E402

PASS, FAIL = [], []


def check(name, cond):
    (PASS if cond else FAIL).append(name)
    print(("  ok  " if cond else "FAIL  ") + name)


class FakeEmbedder:
    """确定性近正交向量（同 conftest 思路）：不同文本低相似，同文本全同。"""
    enabled = True

    @staticmethod
    def _vec(text: str):
        import hashlib
        dim = 32
        h = hashlib.sha256(text.strip().encode("utf-8")).digest()
        return [(b / 255.0) - 0.5 for b in h[:dim]]

    async def embed_one(self, text, timeout=None):
        return self._vec(text)

    async def embed(self, texts):
        return [self._vec(t) for t in texts]


class FakeLLM:
    """按序返回 payload；payload=None 表示本轮返回不可解析输出。"""
    enabled = True

    def __init__(self, payloads):
        self.seq = list(payloads) if isinstance(payloads, list) else [payloads]
        self.prompts: list[str] = []

    async def generate_json(self, prompt, system_prompt=None):
        self.prompts.append(prompt)
        if not self.seq:
            return {"memories": [], "profile": []}
        p = self.seq.pop(0)
        return p


def new_engine(payloads=None, conf: dict | None = None):
    tmp = tempfile.mkdtemp(prefix="fm_")
    paths = DataPaths(Path(tmp) / "pd").ensure()
    conn = dbm.connect(paths.db)
    dbm.init_schema(conn)
    store = MemoryStore(conn)
    c = cfgmod.Config(conf or {}, paths.meta)
    llm = FakeLLM(payloads) if payloads is not None else None
    eng = MemoryEngine(store, c, embedder=FakeEmbedder(), reranker=None, llm=llm)
    return eng, store, conn, paths


def mem_item(content, mtype="fact", alpha=0.9, speaker="user", **extra):
    d = {"content": content, "type": mtype, "alpha": alpha, "speaker": speaker,
         "evidence": extra.pop("evidence", ""), "tags": extra.pop("tags", []),
         "action": extra.pop("action", "create"), "update_ids": extra.pop("update_ids", []),
         "merge_ids": extra.pop("merge_ids", [])}
    d.update(extra)
    return d


def seed_turn(eng, sid, text, role="user", scope="default"):
    eng.record_turn(sid, role, text, scope=scope)


# ================================================================ A 抽取触发时机
def group_a():
    print("[A 抽取触发时机]")
    # A1 LLM 停用
    eng, store, conn, _ = new_engine(payloads=None)
    async def a1():
        seed_turn(eng, "s", "这是一条足够长的测试消息内容")
        return await eng.extract_session("s", scope="default", user_key="u1")
    check("A1 LLM停用→返回0不炸", asyncio.run(a1()) == 0)
    conn.close()

    # A2 轮次触发 + 游标防重复
    eng, store, conn, _ = new_engine([{"memories": [mem_item("小明（u1）喜欢跑步")],
                                       "profile": []}])
    async def a2():
        eng.config._raw["memory_behavior"] = {"trigger_turns": 2, "idle_seconds": 0}
        seed_turn(eng, "s", "第一条消息足够长不惧最小长度")
        seed_turn(eng, "s", "第二条消息同样有足够长度")
        n1 = await eng.extract_session("s", scope="default", user_key="u1")
        n2 = await eng.extract_session("s", scope="default", user_key="u1")
        return n1, n2
    n1, n2 = asyncio.run(a2())
    check("A2a 轮次触发抽取=1", n1 == 1)
    check("A2b 游标推进后重复抽取=0", n2 == 0)
    conn.close()

    # A3 空窗口
    eng, store, conn, _ = new_engine([{"memories": [], "profile": []}])
    async def a3():
        return await eng.extract_session("s", scope="default", user_key="u1")
    check("A3 空账本窗口→0", asyncio.run(a3()) == 0)
    conn.close()

    # A4 全短消息
    eng, store, conn, _ = new_engine([{"memories": [], "profile": []}])
    async def a4():
        seed_turn(eng, "s", "嗯")
        seed_turn(eng, "s", "哦")
        return await eng.extract_session("s", scope="default", user_key="u1")
    check("A4 全短消息(<min_length)→0", asyncio.run(a4()) == 0)
    conn.close()

    # A5 should_extract 触发矩阵
    eng, store, conn, _ = new_engine(None)
    eng.config._raw["memory_behavior"] = {"trigger_turns": 3, "idle_seconds": 300}
    for _ in range(2):
        eng.record_turn("s", "user", "计数一", scope="default")
    check("A5a 轮次未到→False", eng.should_extract("s") is False)
    eng.record_turn("s", "user", "计数三", scope="default")
    check("A5b 轮次到达→True", eng.should_extract("s") is True)
    eng2, _, conn2, _ = new_engine(None)
    eng2.config._raw["memory_behavior"] = {"trigger_turns": 99, "idle_seconds": 300}
    eng2.record_turn("s", "user", "空闲测试", scope="default")
    eng2._last_seen["s"] = time.time() - 400
    check("A5c 空闲超时→True", eng2.should_extract("s") is True)
    eng2._last_seen["s"] = time.time()
    check("A5d 空闲未到且轮次未到→False", eng2.should_extract("s") is False)
    eng3, _, conn3, _ = new_engine(None)
    eng3.config._raw["memory_behavior"] = {"trigger_turns": 99, "idle_seconds": 0}
    eng3.record_turn("s", "user", "禁用空闲", scope="default")
    eng3._last_seen["s"] = time.time() - 99999
    check("A5e idle_seconds=0 显式禁用空闲触发", eng3.should_extract("s") is False)
    conn2.close(); conn3.close(); conn.close()

    # A6 LLM 输出不可解析
    eng, store, conn, _ = new_engine([None])
    async def a6():
        seed_turn(eng, "s", "这段对话内容足够长可以触发抽取")
        n = await eng.extract_session("s", scope="default", user_key="u1")
        return n, "s" in eng._ledger_cursor
    n, cur = asyncio.run(a6())
    check("A6 输出不可解析→0且游标已推进", n == 0 and cur)
    conn.close()


# ================================================================ B 用户指称链
def group_b():
    print("[B 用户指称链]")
    # B1 map 命中
    eng, store, conn, _ = new_engine([{"memories": [], "profile": []}],
                                     {"runtime": {"user_display_name_map": {"u1": "Star Lantern"}}})
    check("B1 map命中→昵称（ID）", eng._user_label("default", "u1") == "Star Lantern（u1）")
    # B2 map miss + 全局名
    check("B2 map未命中回落全局名", eng._user_label("default", "u2") == "Star Lantern（u2）"
          if eng.config.get("runtime.user_display_name") else
          eng._user_label("default", "u2") == "用户（u2）")
    conn.close()
    eng, store, conn, _ = new_engine(None, {"runtime": {"user_display_name": "小明"}})
    check("B2' 仅全局名→按原样", eng._user_label("default", "u1") == "小明")
    conn.close()
    # B3 画像专用键
    eng, store, conn, _ = new_engine(None)
    store.upsert_profile("default", "u1", "用户昵称", "阿崇")
    check("B3 画像专用键→昵称（ID）", eng._user_label("default", "u1") == "阿崇（u1）")
    # B4 bot 占用键不参与
    store.upsert_profile("default", "u1", "称呼", "宝宝")
    store.upsert_profile("default", "u1", "名字", "星棠(青鸾)")
    label = eng._user_label("default", "u1")
    check("B4 bot占用键（称呼/名字）被无视", "宝宝" not in label and "星棠" not in label)
    # B5 无 user_key
    check("B5 无user_key→「用户」", eng._user_label("default", "") == "用户")
    conn.close()
    # B6 脏 map
    eng, store, conn, _ = new_engine(None, {"runtime": {"user_display_name_map": "dirty"}})
    check("B6 脏map不炸回落用户（ID）", eng._user_label("default", "u1") == "用户（u1）")
    conn.close()
    # B7 transcript 身份锚点进 prompt
    eng, store, conn, _ = new_engine([{"memories": [], "profile": []}],
                                     {"runtime": {"user_display_name_map": {"u1": "Star Lantern"}}})
    async def b7():
        seed_turn(eng, "s", "我今晚想吃烤鱼", scope="default")
        seed_turn(eng, "s", "烤鱼好呀，想吃什么锅底", role="assistant", scope="default")
        await eng.extract_session("s", scope="default", user_key="u1")
        p = eng.llm.prompts[-1]
        return p
    p = asyncio.run(b7())
    check("B7a transcript带[昵称（ID）]锚点", "[Star Lantern（u1）]: 我今晚想吃烤鱼" in p)
    check("B7b 助理侧标[助理]", "[助理]" in p)
    check("B7c 规则行禁模糊代词", "禁止用「用户/提问者」" in p)
    conn.close()


# ================================================================ C speaker×闸门
def group_c():
    print("[C speaker×闸门矩阵]")

    def run_extract(conf_admission, item):
        eng, store, conn, _ = new_engine([{"memories": [item], "profile": []}],
                                         {"admission": conf_admission})
        async def _x():
            seed_turn(eng, "s", "这段对话内容足够长可以触发抽取流程")
            return await eng.extract_session("s", scope="default", user_key="u1"), store
        n, st = asyncio.run(_x())
        rows = st.active_memories("default")
        conn.close()
        return n, rows

    n, _ = run_extract({}, mem_item("小明（u1）喜欢跑步"))
    check("C1 user亲口→入库", n == 1)

    n, _ = run_extract({"assistant_claim_policy": "reject_all"},
                       mem_item("用户感冒了", speaker="assistant"))
    check("C2 assistant reject_all→拒", n == 0)

    n, _ = run_extract({"assistant_claim_policy": "allow_relationship"},
                       mem_item("当小明emo时，我会先听他说完", mtype="emotional", speaker="assistant"))
    check("C3 assistant relationship+emotional→入", n == 1)

    n, _ = run_extract({"assistant_claim_policy": "allow_relationship"},
                       mem_item("两人约定周末一起复盘", speaker="assistant"))
    check("C4 assistant fact含互动marker→入", n == 1)

    n, _ = run_extract({"assistant_claim_policy": "allow_relationship"},
                       mem_item("用户好像感冒了", speaker="assistant"))
    check("C5 assistant纯代述事实→拒", n == 0)

    n, _ = run_extract({"assistant_claim_policy": "allow_all"},
                       mem_item("用户好像感冒了", speaker="assistant"))
    check("C6 assistant allow_all→入", n == 1)

    n, _ = run_extract({"deny_assistant_claims": True, "assistant_claim_policy": "bogus"},
                       mem_item("用户喜欢猫", speaker="assistant"))
    check("C7 非法policy回落旧开关→拒", n == 0)

    n, _ = run_extract({"assistant_claim_policy": "allow_relationship"},
                       mem_item("作为一个人工智能，我会一直陪着你", mtype="emotional", speaker="assistant"))
    check("C8 套话闸门优先于策略放行→拒", n == 0)

    n, _ = run_extract({"assistant_claim_policy": "allow_all"},
                       mem_item("用户喜欢猫", alpha=0.1, speaker="assistant"))
    check("C9 α低于门槛策略也救不了→拒", n == 0)

    eng, store, conn, _ = new_engine(
        [{"memories": [], "profile": [{"key": "称呼", "value": "宝宝", "speaker": "assistant",
                                       "confidence": 0.9}]}],
        {"admission": {"assistant_claim_policy": "allow_all"}})
    async def c10():
        seed_turn(eng, "s", "那我以后叫你宝宝好不好呀", scope="default")
        await eng.extract_session("s", scope="default", user_key="u1")
        return store.get_profile("default", "u1")
    check("C10 画像入口不受policy影响→assistant画像拒", asyncio.run(c10()) == [])
    conn.close()

    # type 自造归一化
    n, rows = run_extract({}, mem_item("小明（u1）的关系状态是恋人", mtype="relationship"))
    check("C11 自造type归一化为fact", n == 1 and rows and rows[0]["memory_type"] == "fact")


# ================================================================ D 记忆演进
def group_d():
    print("[D 记忆演进]")

    def run(payload, seed_texts, user_turn, conf=None):
        eng, store, conn, _ = new_engine([{"memories": payload, "profile": []}], conf)
        async def _x():
            for t in seed_texts:
                await eng.remember(t, alpha=0.9)
            seed_turn(eng, "s", user_turn, scope="default")
            return await eng.extract_session("s", scope="default", user_key="u1")
        n = asyncio.run(_x())
        return eng, store, conn, n

    # D1 create 默认
    eng, store, conn, n = run([mem_item("小明（u1）喜欢跑步")], [], "我最近迷上跑步")
    check("D1 create默认入库", n == 1 and len(store.active_memories("default")) == 1)
    conn.close()

    # D2 update 有效
    eng, store, conn, n = run([mem_item("小明（u1）现在对猫过敏", action="update",
                                        update_ids=[1], tags=["猫", "过敏"])],
                              ["小明喜欢猫"], "小明喜欢猫 我现在对猫过敏了")
    active = store.active_memories("default")
    old = [r for r in conn.execute("SELECT * FROM memories WHERE content='小明喜欢猫'")][0]
    check("D2a update旧条退场新条在库", n == 1 and len(active) == 1 and "过敏" in active[0]["content"])
    check("D2b 旧条血缘链指向新条", old["superseded_by"] == active[0]["id"])
    check("D2c 旧条入回收站（面板可见）", any(r["id"] == old["id"] for r in store.list_trash()))
    # D2d 复活往返（v0.2.14 语义：普通恢复保留血缘；彻底恢复才清链回检索面）
    store.restore(old["id"])
    row = store.get_memory(old["id"])
    check("D2d 普通restore保留血缘（旧说法不复活）",
          row["superseded_by"] is not None and row["valid_to"] is not None
          and row["deleted_at"] is not None
          and any(r["id"] == old["id"] for r in store.list_trash()))
    store.restore(old["id"], clear_superseded=True)
    row = store.get_memory(old["id"])
    check("D2d2 彻底恢复清血缘回检索面",
          row["superseded_by"] is None and row["valid_to"] is None
          and row["deleted_at"] is None)
    conn.close()

    # D3/D4 update 降级
    eng, store, conn, n = run([mem_item("小明（u1）昨天去了猫咖", action="update", update_ids=[99])],
                              ["小明喜欢猫"], "小明喜欢猫 随便聊聊")
    check("D3 编号无效降级create（旧条不动）",
          n == 1 and len(store.active_memories("default")) == 2)
    conn.close()
    eng, store, conn, n = run([mem_item("小明（u1）昨天去了猫咖撸猫", action="update", update_ids=[1, 2])],
                              ["小明喜欢猫"], "小明喜欢猫 聊聊")
    check("D4 update编号数≠1降级", n == 1 and len(store.active_memories("default")) == 2)
    conn.close()

    # D5 merge 继承
    eng, store, conn, n = run([mem_item("小明（u1）喜欢跑步常在周四", action="merge",
                                        merge_ids=[1, 2])],
                              ["小明喜欢跑步", "小明每周四去跑步"],
                              "小明喜欢跑步 小明每周四去跑步 一回事")
    active = store.active_memories("default")
    check("D5a merge合二为一", n == 1 and len(active) == 1)
    check("D5b proof继承=max(sum(2),own(1))=2", active[0]["proof_count"] == 2)
    conn.close()

    # D6 merge 数量越界
    eng, store, conn, n = run([mem_item("小明（u1）昨天去了猫咖撸猫", action="merge", merge_ids=[1])],
                              ["小明喜欢猫"], "小明喜欢猫 聊")
    check("D6a merge仅1个编号降级", n == 1 and len(store.active_memories("default")) == 2)
    conn.close()
    eng, store, conn, n = run([mem_item("小明（u1）昨天去了猫咖撸猫", action="merge", merge_ids=[1, 2, 3, 4, 5, 6])],
                              ["小明喜欢猫"], "小明喜欢猫 聊")
    check("D6b merge超5个编号降级", n == 1 and len(store.active_memories("default")) == 2)
    conn.close()

    # D7 重复编号去重后不足
    eng, store, conn, n = run([mem_item("小明（u1）昨天去了猫咖撸猫", action="merge", merge_ids=[2, 2])],
                              ["小明喜欢猫"], "小明喜欢猫 聊")
    check("D7 merge重复编号去重后<2降级", n == 1 and len(store.active_memories("default")) == 2)
    conn.close()

    # D8 evolve off
    eng, store, conn, _ = new_engine([{"memories": [mem_item("小明（u1）现在不喜欢猫了",
                                                             action="update", update_ids=[1])],
                                       "profile": []}],
                                     {"memory_behavior": {"evolve_enabled": False}})
    async def d8():
        await eng.remember("小明喜欢猫", alpha=0.9)
        seed_turn(eng, "s", "我现在不喜欢猫了", scope="default")
        await eng.extract_session("s", scope="default", user_key="u1")
        return eng.llm.prompts[-1]
    p = asyncio.run(d8())
    check("D8 关闭演进：无相关段+旧条不动",
          "（无）" in p and len(store.active_memories("default")) == 2)
    conn.close()

    # D9 自指向守卫
    eng, store, conn, _ = new_engine(None)
    async def d9():
        await eng.remember("小明喜欢猫", alpha=0.9)
        mid = store.active_memories("default")[0]["id"]
        eng._apply_evolution("小明喜欢猫", [mid], "update", "default")
        return store.get_memory(mid)
    row = asyncio.run(d9())
    check("D9 自指向supersede被守卫拦截", row["superseded_by"] is None)
    conn.close()

    # D10 死行哈希链追踪
    eng, store, conn, _ = new_engine(None)
    async def d10():
        await eng.remember("旧说法A", alpha=0.9)
        await eng.remember("新说法B", alpha=0.9)
        await eng.remember("待合并C", alpha=0.9)
        ids = {r["content"]: r["id"] for r in store.active_memories("default")}
        store.supersede(ids["旧说法A"], ids["新说法B"])
        eng._apply_evolution("旧说法A", [ids["待合并C"]], "update", "default")
        return ids
    ids = asyncio.run(d10())
    c_row = store.get_memory(ids["待合并C"])
    check("D10 哈希命中退场行→沿链挂到活口B",
          c_row["superseded_by"] == ids["新说法B"])
    conn.close()

    # D11 混合动作
    eng, store, conn, n = run(
        [mem_item("小明（u1）上周去了猫咖", mtype="event", tags=["猫咖"]),
         mem_item("小明（u1）现在喜欢狗了", action="update", update_ids=[1], tags=["宠物"]),
         mem_item("小明（u1）周四五都跑步", action="merge", merge_ids=[2, 3])],
        ["小明喜欢猫", "小明周四跑步", "小明周五跑步"],
        "小明喜欢猫 小明周四跑步 小明周五跑步 最近变了")
    check("D11 一次抽取混合三动作", n == 3 and len(store.active_memories("default")) == 3
          and len(store.list_trash()) == 3)
    conn.close()

    # D12 相关旧记忆编号清单进 prompt
    eng, store, conn, _ = new_engine([{"memories": [], "profile": []}])
    async def d12():
        await eng.remember("小明喜欢跑步", alpha=0.9)
        seed_turn(eng, "s", "小明喜欢跑步 这周末还去吗", scope="default")
        await eng.extract_session("s", scope="default", user_key="u1")
        return eng.llm.prompts[-1]
    p = asyncio.run(d12())
    check("D12 相关旧记忆以编号清单注入", "[1] (fact) 小明喜欢跑步" in p)
    conn.close()

    # D13 回收站 30 天逾期 purge
    eng, store, conn, _ = new_engine(None)
    async def d13():
        await eng.remember("过期记忆内容", alpha=0.9)
        await eng.remember("新鲜记忆内容", alpha=0.9)
        rows = {r["content"]: r["id"] for r in store.active_memories("default")}
        store.trash(rows["过期记忆内容"])
        conn.execute("UPDATE memories SET deleted_at=? WHERE id=?",
                     (utc_now_ts() - 31 * 86400, rows["过期记忆内容"]))
        conn.commit()
        purged = eng.purge_trash()
        return purged, store.get_memory(rows["过期记忆内容"]), store.get_memory(rows["新鲜记忆内容"])
    purged, gone, kept = asyncio.run(d13())
    check("D13 purge只清逾期回收站行", purged == 1 and gone is None and kept is not None)
    conn.close()


# ================================================================ E 检索四路
def group_e():
    print("[E 检索四路]")

    def setup():
        eng, store, conn, _ = new_engine(None)
        return eng, store, conn

    # E1 tag-only
    eng, store, conn = setup()
    async def e1():
        await eng.remember("小明每周四下午有空去做运动", alpha=0.9, tags=["跑步", "空闲"])
        return await eng.recall("跑步", scope="default", require_match=True, fast=True)
    cands = asyncio.run(e1())
    check("E1 tag通道命中（正文无该词）",
          bool(cands) and any("tag" in c.channels for c in cands))
    conn.close()

    # E2 lexical
    eng, store, conn = setup()
    async def e2():
        await eng.remember("小明喜欢跑步", alpha=0.9)
        return await eng.recall("跑步", scope="default", require_match=True, fast=True)
    cands = asyncio.run(e2())
    check("E2 正文词法通道命中", bool(cands) and any("跑步" in c.content for c in cands))
    conn.close()

    # E3 require_match 无匹配 → 空（recency 关闭）
    eng, store, conn = setup()
    async def e3():
        await eng.remember("小明喜欢跑步", alpha=0.9)
        return await eng.recall("完全无关的词xyzzy", scope="default",
                                require_match=True, fast=True)
    check("E3 显式搜索无匹配→空（不兜底）", asyncio.run(e3()) == [])
    conn.close()

    # E4 注入路径 recency 兜底
    eng, store, conn = setup()
    async def e4():
        await eng.remember("小明喜欢跑步", alpha=0.9)
        return await eng.recall("完全无关的词xyzzy", scope="default",
                                require_match=False, fast=True)
    check("E4 注入路径时间近因兜底非空", bool(asyncio.run(e4())))
    conn.close()

    # E5 scope 隔离
    eng, store, conn = setup()
    async def e5():
        await eng.remember("甲域记忆跑步专属", alpha=0.9, scope="A")
        await eng.remember("乙域记忆跑步隔离", alpha=0.9, scope="B")
        ra = await eng.recall("跑步", scope="A", require_match=True, fast=True)
        rb = await eng.recall("跑步", scope="B", require_match=True, fast=True)
        return [c.content for c in ra], [c.content for c in rb]
    ca, cb = asyncio.run(e5())
    check("E5 scope隔离互不可见", all("甲域" in x for x in ca) and all("乙域" in x for x in cb))
    conn.close()

    # E6 superseded+trashed 排除
    eng, store, conn = setup()
    async def e6():
        await eng.remember("小明喜欢猫", alpha=0.9)
        await eng.remember("小明喜欢狗", alpha=0.9)
        ids = {r["content"]: r["id"] for r in store.active_memories("default")}
        store.supersede(ids["小明喜欢猫"], ids["小明喜欢狗"])
        store.trash(ids["小明喜欢狗"])
        return await eng.recall("喜欢猫", scope="default", require_match=True, fast=True)
    check("E6 退场/回收站行不参与检索", asyncio.run(e6()) == [])
    conn.close()

    # E7 per_type_limit 多样性
    eng, store, conn = setup()
    eng.config._raw["retrieval"] = {"per_type_limit": 1}
    async def e7():
        for i in range(4):
            await eng.remember(f"小明喜欢跑步第{i}种说法", alpha=0.9, memory_type="fact")
        await eng.remember("小明会弹古筝", alpha=0.9, memory_type="skill")
        # 分组限额主要服务注入路径（require_match=False，无语义地板）；
        # 显式搜索路径有地板，低相关的 skill 条目按新契约本就应被滤除
        cands = await eng.recall("跑步", scope="default", require_match=False, fast=True)
        cands2 = await eng.recall("跑步", scope="default", require_match=True, fast=True)
        return cands, cands2
    cands, cands2 = asyncio.run(e7())
    types = {c.memory_type for c in cands}
    check("E7a 注入路径分组限额下skill不缺席", "skill" in types)
    check("E7b 显式搜索路径语义地板滤除低相关", all("跑步" in c.content for c in cands2))
    conn.close()

    # E8 token 预算极小保底 1 条
    eng, store, conn = setup()
    async def e8():
        await eng.remember("小明喜欢跑步内容比较长一点", alpha=0.9)
        return await eng.recall("跑步", scope="default", require_match=True,
                                fast=True, token_budget=5, top_k=8)
    cands = asyncio.run(e8())
    check("E8 极小预算保底1条", len(cands) == 1)
    conn.close()

    # E9 active_only
    eng, store, conn = setup()
    async def e9():
        await eng.remember("小明喜欢跑步", alpha=0.9)
        await eng.remember("小明的生日", alpha=1.0, is_active=True)
        return await eng.recall("跑步", scope="default", require_match=True,
                                fast=True, active_only=True)
    cands = asyncio.run(e9())
    check("E9 active_only仅永生条", all(c.is_active for c in cands))
    conn.close()


# ================================================================ F 衰减/回收站
def group_f():
    print("[F 衰减/回收站]")
    # F1 时间衰减
    eng, store, conn, _ = new_engine(None)
    async def f1():
        await eng.remember("会随时间淡忘的普通记忆", alpha=0.9)
        mid = store.active_memories("default")[0]["id"]
        conn.execute("UPDATE memories SET created_at=?, strength=2.0 WHERE id=?",
                     (utc_now_ts() - 60 * 86400, mid))
        conn.commit()
        before = store.get_memory(mid)["strength"]
        stats = eng.decay_sweep()
        return before, store.get_memory(mid)["strength"], stats
    before, after, stats = asyncio.run(f1())
    check("F1 旧被动记忆被衰减扣分", after < before)
    conn.close()

    # F2 T2 永生
    eng, store, conn, _ = new_engine(None)
    async def f2():
        await eng.remember("长期有用被抬到T2的记忆", alpha=0.9)
        mid = store.active_memories("default")[0]["id"]
        conn.execute("UPDATE memories SET created_at=?, strength=5.0, useful_score=12.0 WHERE id=?",
                     (utc_now_ts() - 60 * 86400, mid))
        conn.commit()
        eng.decay_sweep()
        return store.get_memory(mid)
    row = asyncio.run(f2())
    check("F2 T2高信念永生不衰减", row["strength"] == 5.0)
    conn.close()

    # F3 主动记忆免疫
    eng, store, conn, _ = new_engine(None)
    async def f3():
        await eng.remember("永生条目", alpha=1.0, is_active=True)
        mid = store.active_memories("default")[0]["id"]
        conn.execute("UPDATE memories SET created_at=? WHERE id=?",
                     (utc_now_ts() - 60 * 86400, mid))
        conn.commit()
        eng.decay_sweep()
        return store.get_memory(mid)
    row = asyncio.run(f3())
    check("F3 主动记忆衰减免疫", row["strength"] == 50.0 and row["deleted_at"] is None)
    conn.close()

    # F4 衰减到 0 → 回收站 → 复活
    eng, store, conn, _ = new_engine(None)
    async def f4():
        await eng.remember("即将被遗忘的记忆", alpha=0.9)
        mid = store.active_memories("default")[0]["id"]
        conn.execute("UPDATE memories SET created_at=?, strength=0.3 WHERE id=?",
                     (utc_now_ts() - 60 * 86400, mid))
        conn.commit()
        stats = eng.decay_sweep()
        trashed = stats["trashed"] == 1 and store.get_memory(mid)["deleted_at"] is not None
        store.restore(mid)
        back = store.get_memory(mid)
        return trashed, back["deleted_at"] is None and back["superseded_by"] is None
    t, restored = asyncio.run(f4())
    check("F4a 衰减到0入回收站", t)
    check("F4b 复活后重新在库", restored)
    conn.close()

    # F5 停用衰减
    eng, store, conn, _ = new_engine(None, {"decay_policy": {"enabled": False}})
    async def f5():
        await eng.remember("不停衰", alpha=0.9)
        mid = store.active_memories("default")[0]["id"]
        conn.execute("UPDATE memories SET created_at=? WHERE id=?",
                     (utc_now_ts() - 60 * 86400, mid))
        conn.commit()
        return eng.decay_sweep()
    check("F5 衰减停用→零动作", asyncio.run(f5())["decayed"] == 0)
    conn.close()


# ================================================================ G 反思闭环
def group_g():
    print("[G 反思闭环]")
    conf = {"reflection": {"enabled": True, "turn_threshold": 2, "idle_seconds": 0}}
    # G1/G2 useful/useless
    eng, store, conn, _ = new_engine([{"useful": [1], "useless": [2], "reason": "测试"}], conf)
    async def g12():
        await eng.remember("被判定有用的记忆", alpha=0.9)
        await eng.remember("被判定无用的记忆", alpha=0.9)
        ids = {r["content"]: r["id"] for r in store.active_memories("default")}
        eng.record_turn("s", "user", "最近聊过一些话题", scope="default")
        eng._recall_buffer["s"] = [(ids["被判定有用的记忆"], "x"), (ids["被判定无用的记忆"], "y")]
        eng._reflect_counter["s"] = 5
        stats = await eng.reflect_session("s")
        return (ids, stats, store.get_memory(ids["被判定有用的记忆"])["useful_score"],
                store.get_memory(ids["被判定无用的记忆"])["useful_score"])
    ids, stats, u1, u2 = asyncio.run(g12())
    check("G1 useful→加分", stats["useful"] == 1 and u1 > 0)
    check("G2 useless→扣分", stats["useless"] == 1 and u2 == 0)
    check("G2' 反馈后缓冲清空防重复计分", not eng._recall_buffer.get("s"))
    conn.close()

    # G3 LLM 失败：缓冲取走不重试
    eng, store, conn, _ = new_engine([None], conf)
    async def g3():
        await eng.remember("反思失败场景记忆", alpha=0.9)
        mid = store.active_memories("default")[0]["id"]
        eng._recall_buffer["s"] = [(mid, "x")]
        eng._reflect_counter["s"] = 5
        stats = await eng.reflect_session("s")
        return stats, eng._recall_buffer.get("s"), eng._reflect_counter["s"]
    stats, buf, cnt = asyncio.run(g3())
    check("G3 LLM失败缓冲取走计数清零（不烧额度）",
          stats == {"useful": 0, "useless": 0} and not buf and cnt == 0)
    conn.close()

    # G4 disabled
    eng, store, conn, _ = new_engine([{"useful": [], "useless": [], "reason": ""}],
                                     {"reflection": {"enabled": False}})
    async def g4():
        eng._recall_buffer["s"] = [("m", "x")]
        eng._reflect_counter["s"] = 5
        return eng.should_reflect("s"), await eng.reflect_session("s")
    check("G4 关闭反思不触发", asyncio.run(g4()) == (False, {"useful": 0, "useless": 0}))
    conn.close()

    # G5 无召回缓冲
    eng, store, conn, _ = new_engine([{"useful": [], "useless": [], "reason": ""}], conf)
    check("G5 期间无召回→不触发", eng.should_reflect("s") is False)
    conn.close()


# ================================================================ H 工具
def group_h():
    print("[H 工具四件套]")
    from astrbot_plugin_mnemoria.tools.remember import MemoryRememberTool
    from astrbot_plugin_mnemoria.tools.recall import MemoryRecallTool
    from astrbot_plugin_mnemoria.tools.profile import ProfileUpdateTool

    def ev(eng):
        class _E:
            get_session_id = staticmethod(lambda: "s1")
            get_sender_id = staticmethod(lambda: "u1")
        e = _E()
        e.mnemoria_engine = eng
        return e

    # H1/H2/H3 retention/evidence/tags/source
    eng, store, conn, _ = new_engine(None)
    async def h1():
        eng._session_user["s1"] = "u1"
        await MemoryRememberTool().run(ev(eng), "小明（u1）每周四打篮球",
                                       evidence="「我周四有球局」", tags=["篮球", "周四"])
        r1 = store.active_memories("default")[0]
        await MemoryRememberTool().run(ev(eng), "小明（u1）的生日是10月9日",
                                       retention="permanent")
        await MemoryRememberTool().run(ev(eng), "小明（u1）随口提了个电影",
                                       retention="bogus")
        rows = store.active_memories("default")
        return r1, rows
    r1, rows = asyncio.run(h1())
    check("H1 默认normal被动", r1["is_active"] == 0)
    check("H2 evidence→reasoning/tags→tags_json",
          r1["reasoning"] == "「我周四有球局」" and "篮球" in store.parse_tags(r1))
    check("H3 source=tool如实标注", r1["source"] == "tool")
    check("H1' permanent→永生", any(r["is_active"] == 1 for r in rows))
    check("H1'' 非法retention回落normal", sum(1 for r in rows if r["is_active"] == 1) == 1)
    conn.close()

    # H4 recall 工具
    eng, store, conn, _ = new_engine(None)
    async def h4():
        await eng.remember("小明喜欢跑步", alpha=0.9)
        out = await MemoryRecallTool().run(ev(eng), "跑步")
        return out
    out = asyncio.run(h4())
    check("H4 recall工具返回相关记忆", "跑步" in out)
    conn.close()

    # H5 profile 工具
    eng, store, conn, _ = new_engine(None)
    async def h5():
        await ProfileUpdateTool().run(ev(eng), key="喜好", value="跑步与运动")
        return store.get_profile("default", "u1")
    prof = asyncio.run(h5())
    check("H5 profile工具写入画像", bool(prof) and prof[0]["value"] == "跑步与运动")
    conn.close()

    # H6 note 工具 + 注入块
    from astrbot_plugin_mnemoria.tools.note import NoteCreateTool
    eng, store, conn, _ = new_engine(None)
    async def h6():
        out = await NoteCreateTool().run(ev(eng), "卡扣生图的四个注意点：并发、超时、参考图、尺寸",
                                         title="生图备忘")
        notes = await eng.notes_recall("生图 注意点", scope="default")
        block = eng.notes_block(notes)
        return out, block
    out, block = asyncio.run(h6())
    check("H6 note工具创建可检索且UNTRUSTED包裹",
          "已" in out and "UNTRUSTED" in block and "生图" in block)
    conn.close()


# ================================================================ I 注入组装
def group_i():
    print("[I 注入组装]")
    eng, store, conn, _ = new_engine(None)
    async def i1():
        await eng.remember("小明喜欢跑步", alpha=0.9)
        cands = await eng.recall("跑步", scope="default", require_match=True, fast=True)
        return eng.memories_block(cands)
    block = asyncio.run(i1())
    check("I1 记忆块UNTRUSTED包裹+消毒正文",
          block.startswith("[UNTRUSTED DATA]") and "<relevant_memories>" in block)
    conn.close()

    eng, store, conn, _ = new_engine(None)
    async def i2():
        store.upsert_profile("default", "u1", "称呼", "宝宝")
        return eng.profile_block("default", "u1")
    block = asyncio.run(i2())
    check("I2 画像块含键值且包裹", "用户别名" in block and "UNTRUSTED" in block)
    conn.close()

    eng, store, conn, _ = new_engine(None)
    eng.config._raw["injection"] = {"untrusted_wrap": False}
    async def i3():
        await eng.remember("小明喜欢跑步", alpha=0.9)
        cands = await eng.recall("跑步", scope="default", require_match=True, fast=True)
        return eng.memories_block(cands)
    check("I3 关闭包裹→裸块", not asyncio.run(i3()).startswith("[UNTRUSTED"))
    conn.close()

    # I4 节流
    eng, store, conn, _ = new_engine(None)
    eng.config._raw["injection"] = {"throttle_turns": 3}
    seq = [eng.should_inject_now("s") for _ in range(6)]
    check("I4 throttle=3每三轮注入一次", seq == [False, False, True, False, False, True])
    conn.close()

    # I5 空态
    eng, store, conn, _ = new_engine(None)
    check("I5 空候选→空块", eng.memories_block([]) == "" and eng.profile_block("default", "u1") == "")
    conn.close()


# ================================================================ J 备份/导入
def group_j():
    print("[J 备份/导入往返]")
    import importlib
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
    import import_export

    # J1 导出含 tags_json 与回收站行
    eng, store, conn, paths = new_engine(None)
    async def j1_seed():
        await eng.remember("带标签的正式记忆", alpha=0.9, tags=["标签甲"])
        await eng.remember("被淘汰的记忆", alpha=0.9)
        rows = {r["content"]: r["id"] for r in store.active_memories("default")}
        store.trash(rows["被淘汰的记忆"])
    asyncio.run(j1_seed())
    import json as _json
    out_path = Path(tempfile.mkdtemp()) / "dump.json"
    ie = importlib.import_module("import_export")
    out = ie.do_export(paths.db, out_path)
    exported = _json.loads(out_path.read_text(encoding="utf-8"))
    mems = {m["content"]: m for m in exported.get("memories", [])}
    check("J1a 导出含全部行（含回收站行）", len(mems) >= 2 and "被淘汰的记忆" in mems)
    check("J1b 导出带 tags_json（第四轮修复：do_export 列清单补齐）",
          mems.get("带标签的正式记忆", {}).get("tags_json") == '["标签甲"]')
    conn.close()

    # J2/J3 导入往返（含 tags）+ 重导入去重
    tmp = Path(tempfile.mkdtemp())
    dump = tmp / "dump.json"
    dump.write_text(_json.dumps({"memories": [
        {"content": "往返记忆带标签", "memory_type": "fact", "scope": "default",
         "tags_json": '["篮球", "周四"]', "is_active": 0, "strength": 10.0,
         "useful_score": 1.5, "useful_count": 1},
    ]}, ensure_ascii=False), encoding="utf-8")
    eng, store, conn, paths = new_engine(None)
    n1 = ie.do_import(paths.db, dump)
    row = store.active_memories("default")[0]
    check("J2 导入还原tags", store.parse_tags(row) == ["篮球", "周四"])
    n2 = ie.do_import(paths.db, dump)
    check("J3 重导入去重强化不重复", n2 == 0 and len(store.active_memories("default")) == 1)
    conn.close()


def main():
    for g in (group_a, group_b, group_c, group_d, group_e,
              group_f, group_g, group_h, group_i, group_j):
        g()
    print(f"\n矩阵通过 {len(PASS)} / 失败 {len(FAIL)}")
    if FAIL:
        print("失败项：")
        for f in FAIL:
            print("  -", f)
    return 1 if FAIL else 0


if __name__ == "__main__":
    sys.exit(main())
