"""v0.2.4：过时记忆治理（对齐 angel 实现）回归测试。

背景（过时记忆审查 R1/R3/R4/R5/R7/R8）：改口型事实（住杭州→搬到上海）
向量上不相似，永远进不了相似度门控的裁决视野，新旧矛盾并存；画像四维
append-only 且提示词要求「更正与旧值并存」；注入/检索没有时间信息，
旧记忆反而吃热度加成；淘汰审查看不到时效信息。

覆盖：
- 注入时间戳：memories_block/profile_block 带相对时间标注（angel
  memory_formatter 同款），开关可关，未知时间不标；
- 检索年龄衰减：age_decay_rate>0 时老记忆融合分按 1/(1+rate*age_days)
  打折（angel _apply_time_decay 同公式），0 时行为不变；
- 槽位裁决：tag 重叠但向量不相似的「改口」进入裁决视野并被 update
  取代；开关关闭时恢复普通新增；无向量时槽位通道仍可裁决；
- 画像更正替换：replaces 移除被推翻旧片段（精确/模糊/不命中三路），
  对齐 angel「画像纠正必须 updata，不能仅 create」；
- 演进越权边界：allow_active_evolution=True（工具显式更正）可取代
  is_active 条目（入回收站可复活），默认仍禁止（v0.1.7 收紧不动）；
- 短编号解析：resolve_memory_prefix 唯一命中/过短/非法/歧义；
- 工具动作协议：memory_remember 的 update/merge/bad-id/permanent 白名单；
- 淘汰审查上下文：候选行附创建/最后召回/次数/强度（angel 审查同款）；
- schema 默认值：reflection.enabled=true 及四个新键存在。
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import pytest

from core.retrieve import Candidate, hybrid_retrieve

PLUGIN_ROOT = Path(__file__).resolve().parents[1]
DAY = 86400.0
NOW = 1_800_000_000.0


class CapLLM:
    """记录提示词的假 LLM（反思/淘汰审查断言用）。"""

    enabled = True
    provider_id = "cap"

    def __init__(self, payload):
        self.payload = payload
        self.prompts: list[str] = []

    async def generate_json(self, prompt, system_prompt=None):
        self.prompts.append(str(prompt))
        return self.payload

    async def generate(self, prompt, system_prompt=None):
        return ""


def _mk_event(engine, session_id="s1", sender="u1"):
    class _Ev:
        get_session_id = staticmethod(lambda: session_id)
        get_sender_id = staticmethod(lambda: sender)

    ev = _Ev()
    ev.mnemoria_engine = engine
    return ev


# ---------------------------------------------------------------- 1. 注入时间戳
class TestInjectionAgeLabel:

    def test_memory_block_shows_relative_age(self, make_engine):
        eng, store, conn = make_engine()
        try:
            import time
            now = time.time()
            old = Candidate(id="a", content="小明住在杭州", created_at=now - 3 * DAY)
            new = Candidate(id="b", content="小明搬到上海了", created_at=now)
            block = eng.memories_block([old, new])
            assert "（3天前）" in block and "（刚刚）" in block
        finally:
            conn.close()

    def test_age_label_can_be_disabled(self, make_engine):
        eng, store, conn = make_engine(conf={"injection": {"show_memory_age": False}})
        try:
            import time
            block = eng.memories_block(
                [Candidate(id="a", content="小明住在杭州", created_at=time.time() - 3 * DAY)])
            assert "天前" not in block
        finally:
            conn.close()

    def test_unknown_time_not_labeled(self, make_engine):
        eng, store, conn = make_engine()
        try:
            block = eng.memories_block([Candidate(id="a", content="简单事实")])
            assert "（" not in block, "created_at 未知（0）不得瞎编时间"
        finally:
            conn.close()

    def test_profile_block_shows_update_time(self, make_engine):
        eng, store, conn = make_engine()
        try:
            eng.write_profile("default", "u1", "事实属性", "喜欢跑步")
            block = eng.profile_block("default", "u1")
            assert "（刚刚）" in block
            # 抽取提示词喂的画像不带时间标注（with_time=False 路径不变）
            plain = eng._profile_lines("default", "u1")
            assert plain and "（" not in plain[0]
        finally:
            conn.close()


# ---------------------------------------------------------------- 2. 检索年龄衰减
class TestAgeDecay:

    def _two_facts(self, store, conn):
        old = store.add_memory("小明每周四打篮球旧事", scope="d")
        new = store.add_memory("小明每周四打篮球新事", scope="d")
        conn.execute("UPDATE memories SET created_at=? WHERE id=?",
                     (NOW - 100 * DAY, old))
        conn.execute("UPDATE memories SET created_at=? WHERE id=?", (NOW, new))
        conn.commit()
        return old, new

    def _run(self, conn, store, rate):
        return hybrid_retrieve(
            conn, query="打篮球", scope="d", query_vec=None, fts_ok=store.fts,
            include_recency=False, now_ts=NOW, top_k=8, age_decay_rate=rate,
        )

    def test_old_memory_demoted_by_age(self, make_engine):
        eng, store, conn = make_engine(embedder=None)
        try:
            old, new = self._two_facts(store, conn)
            base = {c.id: c.rrf for c in self._run(conn, store, 0.0)}
            aged = {c.id: c.rrf for c in self._run(conn, store, 0.01)}
            assert old in base and new in base, "rate=0 时两条都应在结果里"
            # angel 公式：1/(1+0.01*100)=0.5，老记忆分必须精确减半
            assert math.isclose(aged[old], base[old] * 0.5, rel_tol=1e-6)
            assert math.isclose(aged[new], base[new], rel_tol=1e-6), "新记忆不得受影响"
        finally:
            conn.close()

    def test_strong_rate_evicts_stale_candidate(self, make_engine):
        eng, store, conn = make_engine(embedder=None)
        try:
            old, new = self._two_facts(store, conn)
            base_ids = {c.id for c in self._run(conn, store, 0.0)}
            aged_ids = {c.id for c in self._run(conn, store, 1.0)}
            assert old in base_ids
            assert old not in aged_ids and new in aged_ids, \
                "强衰减下百天旧条应被相对截断淘汰、新条保留"
        finally:
            conn.close()


# ---------------------------------------------------------------- 3. 槽位裁决
class TestSlotAdjudication:

    @pytest.mark.asyncio
    async def test_correction_reaches_adjudication_via_tags(self, make_engine):
        """「改口」向量不相似（<0.78 地板），靠共享 tag 进裁决并被 update。"""
        payload = {"action": "update", "target_ids": ["1"],
                   "content": "小明（u1）搬到上海了", "confidence": 0.9,
                   "reason": "所在地改口"}
        eng, store, conn = make_engine(payload=payload)
        try:
            assert await eng.remember("小明（u1）住在杭州", alpha=0.9,
                                      tags=["所在地"]) is True
            old_id = store.active_memories("default")[0]["id"]
            assert await eng.remember("小明（u1）搬到上海了", alpha=0.9,
                                      tags=["所在地"]) is True
            active = store.active_memories("default")
            assert len(active) == 1, "矛盾对必须收敛为一条"
            assert active[0]["content"] == "小明(u1)搬到上海了", \
                "normalize 会把全角括号折半角，断言以落库形态为准"
            old = store.get_memory(old_id)
            assert old["superseded_by"] == active[0]["id"]
            assert old["deleted_at"] is not None, "被取代旧条入回收站（可复活）"
        finally:
            conn.close()

    @pytest.mark.asyncio
    async def test_slot_channel_disabled_keeps_both(self, make_engine):
        """关闭槽位通道：向量不相似的矛盾对走普通新增（旧行为）。"""
        payload = {"action": "update", "target_ids": ["1"],
                   "content": "小明（u1）搬到上海了", "confidence": 0.9}
        eng, store, conn = make_engine(
            conf={"admission": {"slot_candidate_enabled": False}}, payload=payload)
        try:
            await eng.remember("小明（u1）住在杭州", alpha=0.9, tags=["所在地"])
            await eng.remember("小明（u1）搬到上海了", alpha=0.9, tags=["所在地"])
            assert len(store.active_memories("default")) == 2
        finally:
            conn.close()

    @pytest.mark.asyncio
    async def test_slot_channel_works_without_vectors(self, make_engine):
        """无嵌入时槽位通道仍可裁决（此前无向量=完全放弃裁决）。"""
        payload = {"action": "update", "target_ids": ["1"],
                   "content": "小明（u1）搬到上海了", "confidence": 0.9}
        eng, store, conn = make_engine(payload=payload, embedder=None)
        try:
            await eng.remember("小明（u1）住在杭州", alpha=0.9, tags=["所在地"])
            old_id = store.active_memories("default")[0]["id"]
            await eng.remember("小明（u1）搬到上海了", alpha=0.9, tags=["所在地"])
            active = store.active_memories("default")
            assert len(active) == 1
            assert store.get_memory(old_id)["superseded_by"] == active[0]["id"]
        finally:
            conn.close()


# ---------------------------------------------------------------- 4. 画像更正替换
class TestProfileReplace:

    def test_exact_segment_replaced(self, make_engine):
        eng, store, conn = make_engine()
        try:
            store.merge_profile("default", "u1", "事实属性", "职业是教师；喜欢跑步")
            store.merge_profile("default", "u1", "事实属性", "职业是程序员",
                                replace_segment="职业是教师")
            row = store.get_profile("default", "u1")[0]
            assert "程序员" in row["value"] and "教师" not in row["value"]
            assert "喜欢跑步" in row["value"], "无关片段不得误删"
        finally:
            conn.close()

    def test_fuzzy_segment_replaced(self, make_engine):
        """模型照抄时略有出入（标点/截断）也能命中。"""
        eng, store, conn = make_engine()
        try:
            store.merge_profile("default", "u1", "事实属性", "职业是教师")
            store.merge_profile("default", "u1", "事实属性", "职业是程序员",
                                replace_segment="职业是教师。")
            row = store.get_profile("default", "u1")[0]
            assert "教师" not in row["value"]
        finally:
            conn.close()

    def test_no_match_only_appends(self, make_engine):
        """replaces 不命中时宁追加不误删（保守原则）。"""
        eng, store, conn = make_engine()
        try:
            store.merge_profile("default", "u1", "事实属性", "职业是教师")
            store.merge_profile("default", "u1", "事实属性", "养了一只猫",
                                replace_segment="完全不相关的旧片段")
            row = store.get_profile("default", "u1")[0]
            assert "教师" in row["value"] and "猫" in row["value"]
        finally:
            conn.close()

    def test_extract_path_honors_replaces(self, make_engine):
        """抽取管线：profile 条目的 replaces 字段直达写入。"""
        eng, store, conn = make_engine()
        try:
            eng.write_profile("default", "u1", "事实属性", "现居杭州")
            eng._update_profile([{
                "key": "事实属性", "value": "现居上海", "speaker": "user",
                "confidence": 0.9, "replaces": "现居杭州",
            }], "default", "u1")
            row = store.get_profile("default", "u1")[0]
            assert "上海" in row["value"] and "杭州" not in row["value"]
        finally:
            conn.close()


# ---------------------------------------------------------------- 5. 演进越权边界
class TestEvolutionActiveGuard:

    @pytest.mark.asyncio
    async def test_active_memory_protected_by_default(self, make_engine):
        """默认（自动路径）仍不得取代 is_active（v0.1.7 收紧不动）。"""
        eng, store, conn = make_engine()
        try:
            old = store.add_memory("小明的生日是10月9日", scope="default",
                                   is_active=True)
            await eng.remember("小明的生日是10月10日", alpha=0.9,
                               evolution_action="update", evolution_ids=[old])
            row = store.get_memory(old)
            assert row["superseded_by"] is None and row["deleted_at"] is None
            assert len(store.active_memories("default")) == 2
        finally:
            conn.close()

    @pytest.mark.asyncio
    async def test_explicit_tool_update_can_supersede_active(self, make_engine):
        """工具显式更正（allow_active_evolution=True）可取代永生条目，
        旧条入回收站可复活（比 angel 的物理删除多一层兜底）。"""
        eng, store, conn = make_engine()
        try:
            old = store.add_memory("小明的生日是10月9日", scope="default",
                                   is_active=True)
            await eng.remember("小明的生日是10月10日", alpha=0.9,
                               evolution_action="update", evolution_ids=[old],
                               allow_active_evolution=True)
            row = store.get_memory(old)
            assert row["superseded_by"] is not None
            assert row["deleted_at"] is not None, "取代即入回收站，可复活"
            active = store.active_memories("default")
            assert len(active) == 1 and "10月10日" in active[0]["content"]
        finally:
            conn.close()


# ---------------------------------------------------------------- 6. 短编号解析
class TestResolvePrefix:

    def test_unique_hit(self, make_engine):
        eng, store, conn = make_engine()
        try:
            mid = store.add_memory("小明喜欢跑步", scope="default")
            assert eng.resolve_memory_prefix("default", mid[:6]) == mid
            assert eng.resolve_memory_prefix("default", f"[{mid[:6]}]") == mid
        finally:
            conn.close()

    def test_too_short_and_illegal_rejected(self, make_engine):
        eng, store, conn = make_engine()
        try:
            mid = store.add_memory("小明喜欢跑步", scope="default")
            assert eng.resolve_memory_prefix("default", mid[:4]) is None
            assert eng.resolve_memory_prefix("default", "zzzzzz") is None
            assert eng.resolve_memory_prefix("default", "") is None
        finally:
            conn.close()

    def test_ambiguous_rejected(self, make_engine):
        eng, store, conn = make_engine()
        try:
            store.add_memory("甲事实", scope="default", mem_id="abcdef" + "1" * 26)
            store.add_memory("乙事实", scope="default", mem_id="abcdef" + "2" * 26)
            assert eng.resolve_memory_prefix("default", "abcdef") is None
        finally:
            conn.close()


# ---------------------------------------------------------------- 7. 工具动作协议
class TestRememberToolActions:

    @pytest.mark.asyncio
    async def test_update_action_end_to_end(self, make_engine):
        eng, store, conn = make_engine()
        try:
            from tools.remember import MemoryRememberTool
            tool = MemoryRememberTool()
            eng._session_user["s1"] = "u1"
            await tool.run(_mk_event(eng), "小明（u1）现居杭州")
            old = store.active_memories("default")[0]
            msg = await tool.run(_mk_event(eng), "小明（u1）现居上海",
                                 action="update", target_ids=[old["id"][:6]])
            assert "更正" in msg
            active = store.active_memories("default")
            assert len(active) == 1 and "上海" in active[0]["content"]
            assert store.get_memory(old["id"])["deleted_at"] is not None
        finally:
            conn.close()

    @pytest.mark.asyncio
    async def test_update_with_bad_id_guided_to_recall(self, make_engine):
        eng, store, conn = make_engine()
        try:
            from tools.remember import MemoryRememberTool
            tool = MemoryRememberTool()
            msg = await tool.run(_mk_event(eng), "小明（u1）现居上海",
                                 action="update", target_ids=["eeeeee"])
            assert "memory_recall" in msg, "编号无效必须引导先检索"
            assert store.active_memories("default") == []
        finally:
            conn.close()

    @pytest.mark.asyncio
    async def test_merge_action(self, make_engine):
        eng, store, conn = make_engine()
        try:
            from tools.remember import MemoryRememberTool
            tool = MemoryRememberTool()
            eng._session_user["s1"] = "u1"
            await tool.run(_mk_event(eng), "小明（u1）喜欢跑步")
            await tool.run(_mk_event(eng), "小明（u1）喜欢拼图")
            ids = [m["id"] for m in store.active_memories("default")]
            msg = await tool.run(_mk_event(eng), "小明（u1）喜欢跑步和拼图",
                                 action="merge", target_ids=[i[:6] for i in ids])
            assert "合并" in msg
            active = store.active_memories("default")
            assert len(active) == 1 and "跑步和拼图" in active[0]["content"]
        finally:
            conn.close()

    @pytest.mark.asyncio
    async def test_permanent_blocked_when_disallowed(self, make_engine):
        eng, store, conn = make_engine(
            conf={"memory_behavior": {"permanent_allow_tool": False}})
        try:
            from tools.remember import MemoryRememberTool
            tool = MemoryRememberTool()
            msg = await tool.run(_mk_event(eng), "小明（u1）的生日是10月9日",
                                 retention="permanent")
            row = store.active_memories("default")[0]
            assert row["is_active"] == 0, "白名单关闭时 permanent 必须降级"
            assert "普通记忆" in msg
        finally:
            conn.close()

    @pytest.mark.asyncio
    async def test_recall_exposes_short_ids(self, make_engine):
        eng, store, conn = make_engine()
        try:
            from tools.recall import MemoryRecallTool
            mid = store.add_memory("小明（u1）喜欢跑步", scope="default")
            out = await MemoryRecallTool().run(_mk_event(eng), "跑步")
            assert f"[{mid[:6]}]" in out, "recall 必须带短编号供 update 引用"
        finally:
            conn.close()


# ---------------------------------------------------------------- 8. 淘汰审查上下文
class TestRetirementContext:

    @pytest.mark.asyncio
    async def test_candidates_carry_time_info(self, make_engine):
        import time
        eng, store, conn = make_engine()
        try:
            mid = store.add_memory("冷门旧记忆", scope="default", strength=1.0)
            conn.execute("UPDATE memories SET created_at=? WHERE id=?",
                         (time.time() - 20 * DAY, mid))
            conn.commit()
            cands = eng.retirement_candidates(limit=5)
            assert cands and cands[0]["id"] == mid
            for key in ("created_at", "last_recalled_at", "hit_count", "proof_count"):
                assert key in cands[0], f"候选缺少 {key}"

            cap = CapLLM({"keep": ["1"], "confidence": 0.9, "reason": "留着"})
            eng.llm = cap
            stats = await eng.review_retirement(limit=5)
            assert stats["kept"] >= 1
            assert "创建于" in cap.prompts[0] and "（20天前）" in cap.prompts[0]
            assert "（从未被召回）" in cap.prompts[0]
        finally:
            conn.close()


# ---------------------------------------------------------------- 9. 提示词与 schema
class TestPromptsAndSchema:

    def test_adjudicate_prompt_has_slot_rule(self):
        from core import templates
        assert "同一属性槽位" in templates.ADJUDICATE_PROMPT
        assert "update" in templates.ADJUDICATE_PROMPT

    def test_extract_prompt_has_replaces(self):
        from core import templates
        assert "replaces" in templates.EXTRACT_PROMPT
        assert "优先修正画像" in templates.EXTRACT_PROMPT
        assert "并存可溯源" not in templates.EXTRACT_PROMPT, "旧的并存语义必须移除"

    def test_schema_defaults(self):
        schema = json.loads(
            (PLUGIN_ROOT / "_conf_schema.json").read_text(encoding="utf-8"))
        assert schema["reflection"]["items"]["enabled"]["default"] is True
        assert schema["retrieval"]["items"]["age_decay_rate"]["default"] == 0.01
        assert schema["injection"]["items"]["show_memory_age"]["default"] is True
        assert schema["admission"]["items"]["slot_candidate_enabled"]["default"] is True
        assert schema["memory_behavior"]["items"]["permanent_allow_tool"]["default"] is True
