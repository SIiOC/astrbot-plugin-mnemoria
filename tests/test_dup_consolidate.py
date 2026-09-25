"""v0.2.3 配套工具测试：存量近重复合并 + reasoning 有据补写。

覆盖安全规格：dry-run 不写、先快照后写、幂等、保护主动记忆/编号模板/
已有 reasoning、无源不编造。
"""

from __future__ import annotations

import importlib.util
import json
import time
from pathlib import Path

from core import db as dbm
from core.store import MemoryStore

PLUGIN_ROOT = Path(__file__).resolve().parents[1]


def _load(name: str):
    spec = importlib.util.spec_from_file_location(
        name, PLUGIN_ROOT / "scripts" / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _mk(tmp_path):
    conn = dbm.connect(tmp_path / "t.db")
    dbm.init_schema(conn)
    return MemoryStore(conn), conn


# ---------------------------------------------------------------- 近重复合并
class TestConsolidateDuplicates:

    def _seed(self, tmp_path):
        store, conn = _mk(tmp_path)
        # 用线上校准过的真实对：同事实换措辞 0.81 / 不同事实 0.73
        a = store.add_memory("Star Lantern(3141592653)喜欢跑步", scope="d",
                             strength=20.0, proof_count=2)
        store.update_memory(a, useful_score=3.0)
        b = store.add_memory("Star Lantern(3141592653)爱跑步", scope="d",
                             strength=10.0, proof_count=1)
        store.update_memory(b, useful_score=1.0)
        c = store.add_memory("Star Lantern(3141592653)沉迷深夜广播节目", scope="d",
                             strength=15.0)
        act = store.add_memory("Star Lantern(3141592653)热爱跑步艺术（主动记忆）",
                               scope="d", strength=50.0, is_active=True)
        tpl1 = store.add_memory("用户的第1条记忆内容", scope="d", strength=5.0)
        tpl2 = store.add_memory("用户的第10条记忆内容", scope="d", strength=5.0)
        return store, conn, (a, b, c, act, tpl1, tpl2)

    def test_plan_groups_near_dupes_only(self, tmp_path):
        store, conn, ids = self._seed(tmp_path)
        try:
            mod = _load("consolidate_duplicates")
            clusters = mod.find_clusters(conn)
            grouped = {m["id"] for cl in clusters for m in cl["members"]}
            assert {ids[0], ids[1]} <= grouped, "同事实换措辞必须聚成簇"
            assert ids[2] not in grouped, "不同事实不得入簇"
            assert ids[3] not in grouped, "主动记忆不得入簇"
            assert ids[4] not in grouped and ids[5] not in grouped, \
                "仅编号不同的模板内容不得合并"
        finally:
            conn.close()

    def test_dry_run_writes_nothing(self, tmp_path):
        store, conn, ids = self._seed(tmp_path)
        try:
            mod = _load("consolidate_duplicates")
            clusters = mod.find_clusters(conn)
            assert clusters
            assert not list((tmp_path / "backups").glob("dupes-*.json"))
            assert store.get_memory(ids[1])["deleted_at"] is None
        finally:
            conn.close()

    def test_apply_upgrades_kept_and_trashes_rest(self, tmp_path):
        store, conn, ids = self._seed(tmp_path)
        try:
            mod = _load("consolidate_duplicates")
            clusters = mod.find_clusters(conn)
            snap = mod.apply(conn, clusters, tmp_path / "backups")
            assert snap.exists()
            payload = json.loads(snap.read_text(encoding="utf-8"))
            assert payload["clusters"]
            kept = store.get_memory(ids[0])
            gone = store.get_memory(ids[1])
            assert gone["deleted_at"] is not None, "其余成员软删可恢复"
            assert gone["superseded_by"] == ids[0], "血缘指向保留条"
            assert kept["strength"] == 20.0 and kept["proof_count"] == 3, \
                "强度取最大、证据数求和"
            # v0.2.11 起事件字段语义正位：source=被取代旧条、target=保留条
            evs = store.list_memory_events(target_id=ids[0])
            assert evs and evs[0]["action"] == "manual_dedup"
            assert json.loads(evs[0]["source_ids_json"]) == [ids[1]], \
                "source_ids 应为被取代的旧条"
            # 幂等
            assert mod.find_clusters(conn) == []
        finally:
            conn.close()

    def test_main_dry_run_exit_zero(self, tmp_path, capsys):
        store, conn, _ = self._seed(tmp_path)
        conn.close()
        mod = _load("consolidate_duplicates")
        import sys
        old = sys.argv
        try:
            sys.argv = ["x", "--db", str(tmp_path / "t.db")]
            assert mod.main() == 0
            assert "dry-run" in capsys.readouterr().out
        finally:
            sys.argv = old


# ---------------------------------------------------------------- reasoning 补写
class TestBackfillReasoning:

    def _seed(self, tmp_path):
        store, conn = _mk(tmp_path)
        # 有源：账本里留着同会话原话
        store.append_ledger("s1", "user", "我最近迷上跑步了，周末打算外出跑步",
                            ts=time.time(), scope="d")
        mid = store.add_memory("小明喜欢跑步，周末计划外出跑步", scope="d",
                               session_id="s1")
        # 无源：会话流水已被清掉
        mid2 = store.add_memory("小明对调酒有兴趣", scope="d", session_id="sGone")
        # 已有 reasoning：不许碰
        mid3 = store.add_memory("小明怕鬼", scope="d", session_id="s1",
                                reasoning="2026-08-01 他说怕鬼")
        return store, conn, (mid, mid2, mid3)

    def test_plan_only_with_source(self, tmp_path):
        store, conn, ids = self._seed(tmp_path)
        try:
            mod = _load("backfill_reasoning")
            items = mod.plan(conn)
            by_id = {i["id"]: i for i in items}
            assert ids[0] in by_id, "有账本原话的要补"
            assert "对话原话" in by_id[ids[0]]["reasoning"]
            assert "跑步" in by_id[ids[0]]["reasoning"]
            assert ids[1] not in by_id, "无源可考的不补（不编造）"
            assert ids[2] not in by_id, "已有 reasoning 的不碰"
        finally:
            conn.close()

    def test_apply_writes_and_snapshots(self, tmp_path):
        store, conn, ids = self._seed(tmp_path)
        try:
            mod = _load("backfill_reasoning")
            items = mod.plan(conn)
            snap = mod.apply(conn, items, tmp_path / "backups")
            assert snap.exists()
            row = store.get_memory(ids[0])
            assert row["reasoning"].startswith("20"), "依据带日期"
            assert store.get_memory(ids[2])["reasoning"] == "2026-08-01 他说怕鬼"
            assert mod.plan(conn) == [], "幂等"
        finally:
            conn.close()

    def test_min_match_filters_unrelated_turn(self, tmp_path):
        """零重叠=无证据（不张冠李戴）；有重叠时才按阈值过滤。"""
        store, conn = _mk(tmp_path)
        try:
            store.append_ledger("s1", "user", "今天股市大涨",
                                ts=time.time(), scope="d")
            store.add_memory("小明喜欢跑步", scope="d", session_id="s1")
            store.append_ledger("s2", "user", "我迷上了跑步运动",
                                ts=time.time(), scope="d")
            mid2 = store.add_memory("小明喜欢跑步", scope="d", session_id="s2")
            mod = _load("backfill_reasoning")
            assert mod.plan(conn, min_match=0.0) == [x for x in mod.plan(conn) if x["id"] == mid2] or                 [i["id"] for i in mod.plan(conn, min_match=0.0)] == [mid2]
            assert mod.plan(conn, min_match=0.5) == [], "弱匹配不采信"
        finally:
            conn.close()
