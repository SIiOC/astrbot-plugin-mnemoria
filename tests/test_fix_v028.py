"""v0.2.8：夜间巩固误合并止血 回归测试（第三轮再审 P0）。

线上实证（2026-09-22 血缘链审计）：534 次「取代」里 **483 次**新旧文本
相似度 <0.35——夜间巩固只凭向量余弦聚类（0.86/超容量放宽 0.80），
而本嵌入器对「同主语不同事实」的余弦普遍 >0.86，于是把毫不相干的记忆
滚进同一簇：实测单个巨型 keeper 一次吞掉 214 条不同事实
（「全糖果茶热量」「手机壳去味」「古筝 riff」全被并进「亲密关系」那条），
活性库从 977 掉到 552，且每晚继续。修复三处：
- 聚类过余弦后必须再过**文本确认 + 编号模板守卫**（与写入守卫同源）；
- 簇大小上限（默认 8）；
- 提示词去诱导（原文断言「以下记忆讲述的是同一主题」，LLM 只会顺从）。

注意 SameVec 夹具：所有向量余弦=1.0，正好复现「向量过关但文本不重叠」
的事故条件——合并与否完全由新增的两道守卫决定。
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

pytestmark = pytest.mark.asyncio


class SameVec:
    """任意文本余弦都是 1.0（都过 0.86 地板）。"""

    enabled = True

    async def embed_one(self, text):
        return [1.0, 0.0, 0.0]

    async def embed(self, texts):
        return [[1.0, 0.0, 0.0] for _ in texts]


class MergingLLM:
    enabled = True
    provider_id = "fake"

    def __init__(self):
        self.seen_items: list[str] = []

    async def generate_json(self, prompt, system_prompt=None):
        return {"content": "合并后的记忆", "merged_ids": []}

    async def generate(self, p, s=None):
        return ""


class RefusingLLM(MergingLLM):
    async def generate_json(self, prompt, system_prompt=None):
        return {"content": None}


def _mk(tmp_path, texts, llm=None):
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
    eng = MemoryEngine(store, cfgmod.Config({}, paths.meta),
                       embedder=SameVec(), llm=llm or MergingLLM())
    return conn, store, eng


async def _consolidated_count(conn):
    return conn.execute(
        "SELECT count(*) FROM memories WHERE superseded_by IS NOT NULL"
    ).fetchone()[0]


class TestConsolidationGuards:
    async def test_different_facts_not_merged_despite_cosine_1(self, tmp_path):
        """核心回归：余弦满分但文本不重叠的「同主语不同事实」绝不合并。

        复刻 09-21 事故形态（一条 keeper 吞 214 条不同事实）。
        """
        conn, _, eng = _mk(tmp_path, [
            "用户Star Lantern在备考学术英语，明天要口译",
            "用户Star Lantern喜欢全糖果茶，一杯约450千卡",
            "用户Star Lantern的凯夫拉手机壳脏了用小苏打去味",
        ])
        try:
            stats = await eng.consolidate()
            assert stats["merged"] == 0
            assert await _consolidated_count(conn) == 0
            # 三条都必须还活着
            live = conn.execute(
                "SELECT count(*) FROM memories WHERE deleted_at IS NULL"
            ).fetchone()[0]
            assert live == 3
        finally:
            conn.close()

    async def test_true_paraphrase_still_merges(self, tmp_path):
        """同事实换措辞（文本高重叠）仍然合并——守卫不能一刀切。"""
        conn, _, eng = _mk(tmp_path, [
            "用户酷爱芹菜，顿顿都要放芹菜",
            "用户酷爱芹菜，每顿都要放芹菜",
            "用户酷爱芹菜，顿顿都爱放芹菜",
        ])
        try:
            stats = await eng.consolidate()
            assert stats["merged"] >= 1
            assert await _consolidated_count(conn) >= 1
        finally:
            conn.close()

    async def test_number_template_never_merged(self, tmp_path):
        """仅编号不同的模板内容：向量余弦 1.0、字符重叠 0.73 也绝不合并。"""
        conn, _, eng = _mk(tmp_path, [
            "用户的第1条记忆内容关于饮食",
            "用户的第10条记忆内容关于饮食",
            # 第三条不相关但同 scope，保证聚类循环真的走到编号守卫（防测试空转）
            "用户Star Lantern今天下午去晨跑房撸铁练了深蹲",
        ])
        try:
            await eng.consolidate()
            assert await _consolidated_count(conn) == 0
        finally:
            conn.close()

    async def test_cluster_size_cap(self, tmp_path):
        """簇大小上限：9 条同措辞种子能收多少，封顶 max_size(8)。"""
        texts = [f"用户酷爱芹菜，顿顿都要放芹菜{'。' * i}" for i in range(9)]
        conn, store, eng = _mk(tmp_path, texts)
        seen_sizes: list[int] = []
        orig = eng._merge_cluster
        async def spy(scope, cluster):
            seen_sizes.append(len(cluster))
            return await orig(scope, cluster)
        eng._merge_cluster = spy
        try:
            stats = await eng.consolidate()
            assert stats["merged"] >= 1
            assert seen_sizes, "_merge_cluster 未被调用"
            assert max(seen_sizes) <= 8, f"簇大小突破上限: {seen_sizes}"
        finally:
            conn.close()

    async def test_consolidate_records_audit_event(self, tmp_path):
        """巩固不再是无审计通道：成功合并必须落 memory_events。"""
        conn, store, eng = _mk(tmp_path, [
            "用户酷爱芹菜，顿顿都要放芹菜",
            "用户酷爱芹菜，每顿都要放芹菜",
            "用户酷爱芹菜，顿顿都爱放芹菜",
        ])
        try:
            stats = await eng.consolidate()
            assert stats["merged"] >= 1
            evs = conn.execute(
                "SELECT * FROM memory_events WHERE action='consolidate'").fetchall()
            assert len(evs) >= 1
            assert evs[0]["source_ids_json"]  # 来源清单非空
            assert evs[0]["target_id"]
        finally:
            conn.close()

    async def test_llm_refusal_changes_nothing(self, tmp_path):
        """LLM 返回 null（判定无法合并）：一条都不动。"""
        conn, store, eng = _mk(tmp_path, [
            "用户酷爱芹菜，顿顿都要放芹菜",
            "用户酷爱芹菜，每顿都要放芹菜",
            "用户酷爱芹菜，顿顿都爱放芹菜",
        ], llm=RefusingLLM())
        try:
            stats = await eng.consolidate()
            assert stats["merged"] == 0
            assert await _consolidated_count(conn) == 0
        finally:
            conn.close()


class TestPromptDebias:
    def test_prompt_no_longer_asserts_same_topic(self):
        from core import templates
        p = templates.CONSOLIDATE_PROMPT
        assert "讲述的是同一主题" not in p
        assert "可能" in p and "null" in p

    def test_floor_configurable_via_schema(self):
        import json
        schema = json.loads((Path(__file__).resolve().parents[1]
                             / "_conf_schema.json").read_text(encoding="utf-8"))
        mb = schema["memory_behavior"]["items"]
        assert mb["consolidation_text_floor"]["default"] == 0.55
        assert mb["consolidation_max_size"]["default"] == 8


class TestUndoBadConsolidations:
    """回滚脚本自身：护栏必须可测，否则一次批量回滚就是不可审计的赌博。"""

    def _load(self):
        import importlib.util
        path = Path(__file__).resolve().parents[1] / "scripts" / "undo_bad_consolidations.py"
        spec = importlib.util.spec_from_file_location("undo_bad_consolidations", path)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        return mod

    def _conn(self, tmp_path):
        from core import db as dbm
        conn = dbm.connect(tmp_path / "t.db")
        dbm.init_schema(conn)
        return conn

    def _chain(self, store, keeper_content, victim_contents, with_event=False):
        import time as _t
        kid = store.add_memory(keeper_content, scope="default", strength=30.0)
        vids = []
        for vc in victim_contents:
            vid = store.add_memory(vc, scope="default", strength=20.0)
            store.conn.execute(
                "UPDATE memories SET deleted_at=?, superseded_by=?, valid_to=? WHERE id=?",
                (_t.time(), kid, _t.time(), vid))
            vids.append(vid)
        if with_event:
            store.record_memory_event("merge", scope="default",
                                      source_ids=vids, target_id=kid,
                                      reason="test")
        store.conn.commit()
        return kid, vids

    def test_event_covered_chain_never_resurrected(self, tmp_path):
        """有事件佐证的链（改口更正/裁决 merge）绝不复活——最重要的护栏。"""
        mod = self._load()
        conn = self._conn(tmp_path)
        from core.store import MemoryStore
        store = MemoryStore(conn)
        kid, vids = self._chain(store, "小明搬到上海，开始新生活。",
                                 ["小明住在杭州，喜欢西湖。"], with_event=True)
        try:
            rb = mod.plan(conn, floor=0.55)
            assert [r["victim"] for r in rb] == []
            assert mod._keepers_to_retire(conn, rb) == []
        finally:
            conn.close()

    def test_low_sim_no_event_chain_rolled_back(self, tmp_path):
        mod = self._load()
        conn = self._conn(tmp_path)
        from core.store import MemoryStore
        store = MemoryStore(conn)
        kid, vids = self._chain(store, "小明建立了亲密关系，自称老夫老妻。",
                                 ["一杯全糖果茶约400-500千卡，冰淇淋容易让人连续吃。",
                                  "凯夫拉手机壳脏了可以用湿巾擦拭后通风晾干。",
                                  "用户追过凡人修仙传小说，仍在关注这部作品。"])
        try:
            rb = mod.plan(conn, floor=0.55)
            assert {r["victim"] for r in rb} == set(vids)
            ret = mod._keepers_to_retire(conn, rb)
            assert ret == [kid]  # 全部受害者回滚且 ≥2 个 → keeper 退休
        finally:
            conn.close()

    def test_single_victim_short_keeper_not_retired(self, tmp_path):
        """单受害者短 keeper：另一同簇成员可能早已逾期物理删除，
        退休 keeper 会连带丢那部分内容——宁留轻度重复，不赌完整性。"""
        mod = self._load()
        conn = self._conn(tmp_path)
        from core.store import MemoryStore
        store = MemoryStore(conn)
        kid, vids = self._chain(store, "Star Lantern明天计划去进行跑步活动。",
                                 ["Star Lantern在晨跑房完成了今日全部撸铁计划。"])
        try:
            rb = mod.plan(conn, floor=0.55)
            assert len(rb) == 1  # 受害者照滚
            assert mod._keepers_to_retire(conn, rb) == []
        finally:
            conn.close()

    def test_mixed_keeper_keeps_legit_and_survives(self, tmp_path):
        """混子 keeper（长拼接文本里含一条真换措辞重复 + 两条误并异事）：
        只滚误并的两条；keeper 因持有未被回滚的合法吸收内容而保留。
        （keeper >200 字且回滚数≥2 → 退休阈值已满足，唯一拦得住它的就是
        「受害者未全滚不得退休」这条护栏——变异它测试必须变红。）"""
        mod = self._load()
        conn = self._conn(tmp_path)
        from core.store import MemoryStore
        store = MemoryStore(conn)
        import time as _t
        base = ("小明对AI模型的使用习惯非常固定，白天用MiMo套餐跑日常对话，"
                "闲时用DeepSeek写代码，遇到复杂任务才切到最强模型；他对额度"
                "和计费很敏感，每次涨价都会重新比较一轮性价比，并且喜欢把结论"
                "写进备忘里反复确认。" * 2)  # >200 字的复合 keeper
        kid = store.add_memory(base, scope="default", strength=30.0)
        legit = store.add_memory(base.replace("写进备忘里反复确认", "写进备忘里多次确认"),
                                 scope="default", strength=20.0)
        bad1 = store.add_memory("凯夫拉手机壳脏了可以用湿巾擦拭后通风晾干。",
                                scope="default", strength=20.0)
        bad2 = store.add_memory("一杯全糖果茶约四百到五百千卡，冰淇淋容易连吃。",
                                scope="default", strength=20.0)
        for vid in (legit, bad1, bad2):
            store.conn.execute(
                "UPDATE memories SET deleted_at=?, superseded_by=?, valid_to=? WHERE id=?",
                (_t.time(), kid, _t.time(), vid))
        store.conn.commit()
        try:
            rb = mod.plan(conn, floor=0.55)
            assert {r["victim"] for r in rb} == {bad1, bad2}, "真换措辞不得被滚"
            ret = mod._keepers_to_retire(conn, rb)
            assert ret == [], "keeper 还持有合法吸收（legit 未被滚），不得退休"
            # 执行后：误并条复活、合法吸收条仍在回收站且血缘不变
            mod.apply(conn, rb, ret, tmp_path / "bk")
            row = conn.execute("SELECT deleted_at, superseded_by FROM memories WHERE id=?",
                               (bad1,)).fetchone()
            assert row["deleted_at"] is None and row["superseded_by"] is None
            row2 = conn.execute("SELECT deleted_at, superseded_by FROM memories WHERE id=?",
                                (legit,)).fetchone()
            assert row2["deleted_at"] is not None and row2["superseded_by"] == kid
        finally:
            conn.close()

    def test_high_sim_stays_absorbed_and_idempotent(self, tmp_path):
        """换措辞真重复：保持被吸收；重复跑第二遍应无可滚（幂等）。"""
        mod = self._load()
        conn = self._conn(tmp_path)
        from core.store import MemoryStore
        store = MemoryStore(conn)
        kid, vids = self._chain(store, "用户酷爱芹菜，顿顿都要放芹菜",
                                 ["用户酷爱芹菜，每顿都要放芹菜"])
        try:
            assert mod.plan(conn, floor=0.55) == []   # 高相似：不动
            # 再造一条低相似的滚一次，二次运行应为空（幂等）
            _, v2 = self._chain(store, "小明喜欢蓝色，新买的所有东西都是蓝色的。",
                                 ["小明养了三只仓鼠，名字叫糯米和团子。"])
            assert len(mod.plan(conn, floor=0.55)) == 1
            snap_dir = tmp_path / "bk"
            mod.apply(conn, mod.plan(conn, floor=0.55),
                      mod._keepers_to_retire(conn, mod.plan(conn, floor=0.55)),
                      snap_dir)
            assert mod.plan(conn, floor=0.55) == []  # 幂等
        finally:
            conn.close()
