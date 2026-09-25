# -*- coding: utf-8 -*-
"""v0.1.9 回归钉子：2026-09-19 审查修复批。

覆盖四类修复（每条对应审查报告里的编号）：
- M-1 笔记回收站：软删 → 恢复 → 逾期清理（此前笔记既不能恢复也不被清理）；
- M-2 抽取游标持久化：跨引擎实例（重启）不再重复抽取同一段对话；
- M-3 星图节点上限：MAX_NODES 生效且指纹仍随向量增删变化；
- M-7 元指令误伤：祈使前缀剥离后入库、提示注入仍隔离。
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from core import db as dbm  # noqa: E402
from core import graph  # noqa: E402
from core.config import Config  # noqa: E402
from core.engine import MemoryEngine  # noqa: E402
from core.paths import DataPaths  # noqa: E402
from core.store import MemoryStore  # noqa: E402


def _make(tmp_path, payload=None):
    """构造 (engine, store, conn)（与 conftest.make_engine 同形，独立一份避免耦合）。"""
    paths = DataPaths(tmp_path / "pd").ensure()
    conn = dbm.connect(paths.db)
    dbm.init_schema(conn)
    store = MemoryStore(conn)
    cfg = Config({}, paths.meta)

    class _LLM:
        enabled = payload is not None

        async def generate_json(self, prompt, system_prompt=None):
            return payload

        async def generate(self, prompt, system_prompt=None):
            return ""

    eng = MemoryEngine(store, cfg, embedder=None, reranker=None, llm=_LLM())
    return eng, store, conn


class TestNoteTrashLifecycle:
    """M-1：笔记回收站可恢复 + 逾期可清理。"""

    def test_trash_note_visible_and_restorable(self, tmp_path):
        eng, store, conn = _make(tmp_path)
        try:
            nid = store.add_note("把语料导入流程记下来", title="流程", scope="default")
            assert store.list_notes(scope="default"), "新建笔记应在列表里"
            store.trash_note(nid)
            assert store.list_notes(scope="default") == [], "软删后不进正常列表"
            trash = store.list_trash_notes()
            assert len(trash) == 1 and trash[0]["id"] == nid
            store.restore_note(nid)
            assert store.list_trash_notes() == [], "恢复后不在回收站"
            assert len(store.list_notes(scope="default")) == 1, "恢复后回到正常列表"
        finally:
            conn.close()

    def test_purge_trash_cleans_notes_too(self, tmp_path):
        """purge_trash 必须同时清记忆与笔记（此前只清记忆）。"""
        eng, store, conn = _make(tmp_path)
        try:
            mem_id = store.add_memory("旧的被动记忆", scope="default")
            store.trash(mem_id)
            note_id = store.add_note("旧的笔记", scope="default")
            store.trash_note(note_id)
            # 把两条的回收站时间都推到很久以前（逾期）
            long_ago = 0.0
            conn.execute("UPDATE memories SET deleted_at=? WHERE id=?", (long_ago, mem_id))
            conn.execute("UPDATE notes SET deleted_at=? WHERE id=?", (long_ago, note_id))
            conn.commit()
            purged = eng.purge_trash()
            assert purged == 2, f"记忆+笔记都应被清理，实际 {purged}"
            assert store.list_trash() == [] and store.list_trash_notes() == []
            assert conn.execute("SELECT COUNT(*) FROM notes").fetchone()[0] == 0
        finally:
            conn.close()

    def test_purge_trash_keeps_fresh_notes(self, tmp_path):
        """未逾期的笔记不得被清理（回收站保留期对笔记同样生效）。"""
        eng, store, conn = _make(tmp_path)
        try:
            nid = store.add_note("刚删的笔记", scope="default")
            store.trash_note(nid)
            assert eng.purge_trash() == 0
            assert len(store.list_trash_notes()) == 1
        finally:
            conn.close()


class TestExtractCursorPersistence:
    """M-2：抽取游标跨引擎实例（重启）保持。"""

    class _LLM:
        enabled = True

        async def generate_json(self, prompt, system_prompt=None):
            return {"memories": [{"content": "用户喜欢深夜听广播", "type": "fact",
                                  "alpha": 0.9, "speaker": "user"}], "profile": []}

        async def generate(self, prompt, system_prompt=None):
            return ""

    @pytest.mark.asyncio
    async def test_cursor_survives_new_engine_instance(self, tmp_path):
        paths = DataPaths(tmp_path / "pd").ensure()
        conn = dbm.connect(paths.db)
        dbm.init_schema(conn)
        store = MemoryStore(conn)
        try:
            cfg = Config({}, paths.meta)
            eng1 = MemoryEngine(store, cfg, embedder=None, reranker=None, llm=self._LLM())
            eng1.record_turn("s", "user", "我最近在准备考试", scope="default")
            n1 = await eng1.extract_session("s", scope="default", user_key="u1")
            assert n1 == 1
            cursor1 = eng1._ledger_cursor["s"]
            assert cursor1 == store.max_ledger_id("s")
            assert dbm.get_meta(conn, "cursor:s", "") == str(cursor1), "游标必须落 meta 表"

            # 模拟插件重启：全新引擎实例（内存游标为空）
            eng2 = MemoryEngine(store, cfg, embedder=None, reranker=None, llm=self._LLM())
            assert eng2._ledger_cursor == {}
            assert eng2._cursor_for("s") == cursor1, "新实例应从 meta 表恢复游标"
            n2 = await eng2.extract_session("s", scope="default", user_key="u1")
            assert n2 == 0, "重启后不得重复抽取已处理过的对话"
        finally:
            conn.close()

    @pytest.mark.asyncio
    async def test_cursor_read_failure_falls_back_to_zero(self, tmp_path):
        """meta 读失败（表缺失等）按 0 处理，不阻断抽取。"""
        eng, store, conn = _make(tmp_path)
        try:
            conn.execute("DROP TABLE meta")  # 极端损坏现场
            conn.commit()
            store.conn = conn
            assert eng._cursor_for("s") == 0
        finally:
            conn.close()


class TestGraphNodeCap:
    """M-3：星图 O(n²) 保护。"""

    def test_max_nodes_constant_and_sql_limit(self):
        src = (_ROOT / "core" / "graph.py").read_text(encoding="utf-8")
        assert "MAX_NODES = 1500" in src
        assert "ORDER BY m.created_at DESC LIMIT ?" in src, "边计算必须按最近节点截断"

    def test_fingerprint_still_counts_all_vectors(self, tmp_path):
        """指纹仍用未截断的向量总数（向量增删必须让缓存失效）。"""
        eng, store, conn = _make(tmp_path)
        try:
            fp0 = graph._fingerprint_conn(conn)
            mid = store.add_memory("用户喜欢猫", scope="default")
            store.set_vector(mid, [1.0, 0.0, 0.0])
            fp1 = graph._fingerprint_conn(conn)
            assert fp1[1] == fp0[1] + 1, "向量数变化必须反映到指纹"
        finally:
            conn.close()

    def test_edges_computed_within_cap(self, tmp_path):
        """小库（未触顶）时边计算照常产出。"""
        eng, store, conn = _make(tmp_path)
        try:
            a = store.add_memory("用户喜欢在深夜听广播", scope="default")
            b = store.add_memory("用户在深夜听广播节目", scope="default")
            store.set_vector(a, [1.0, 0.0, 0.0])
            store.set_vector(b, [0.99, 0.01, 0.0])
            edges = graph._compute_edges(conn)
            assert edges, "相似度高于 MIN_SIM 的两条应产生骨干边"
        finally:
            conn.close()


class TestStrayConfigGuard:
    """v0.1.9：插件目录残留含凭据的主配置副本时，启动必须告警（不自动删文件）。

    机制（实测）：`from astrbot.api import ...` 会往当前工作目录写
    data/cmd_config.json，故“以插件目录为 cwd”跑任何东西都会落一份到插件目录。
    """

    def test_helper_defined_and_called_in_init(self):
        import ast

        tree = ast.parse((_ROOT / "main.py").read_text(encoding="utf-8"))
        init = None
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == "__init__":
                init = node
                break
        assert init is not None, "未找到 __init__"
        called = {
            n.func.attr for n in ast.walk(init)
            if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)
        }
        assert "_warn_stray_runtime_config" in called, "启动未调用凭据残留护栏"
        dumped = ast.dump(init)
        assert "_warn_stray_runtime_config" in dumped

    def test_warns_and_never_deletes(self, plugin, tmp_path, monkeypatch):
        """行为断言：命中时告警、路径正确，且不得调用任何删除 API。"""
        # 注意：ASTRBOT_ROOT 也在 sys.path 上，裸 `import main` 会命中 AstrBot 根目录的
        # main.py，必须按包路径导入插件自身的模块。
        import astrbot_plugin_mnemoria.main as mnemoria_main

        plug_dir = tmp_path / "plug"
        (plug_dir / "data").mkdir(parents=True)
        (plug_dir / "data" / "cmd_config.json").write_text("{}", encoding="utf-8")
        monkeypatch.setattr(mnemoria_main, "_PLUGIN_DIR", str(plug_dir))

        captured: list[str] = []

        class _Rec:
            def warning(self, msg, *args):
                captured.append(msg % args if args else msg)

            def info(self, *a):
                pass

            def debug(self, *a):
                pass

            def error(self, *a):
                pass

        plugin.logger = _Rec()
        plugin._warn_stray_runtime_config()

        assert captured, "存在残留副本时应告警"
        joined = captured[0]
        assert "cmd_config.json" in joined and "data" in joined
        assert "打包" in joined or "分享" in joined, "告警缺少可操作建议"
        # 只告警不删除：残留文件必须原样保留
        assert (plug_dir / "data" / "cmd_config.json").is_file(), "护栏不得删除文件"

    def test_silent_when_absent(self, plugin, tmp_path, monkeypatch):
        # 注意：ASTRBOT_ROOT 也在 sys.path 上，裸 `import main` 会命中 AstrBot 根目录的
        # main.py，必须按包路径导入插件自身的模块。
        import astrbot_plugin_mnemoria.main as mnemoria_main

        monkeypatch.setattr(mnemoria_main, "_PLUGIN_DIR", str(tmp_path / "empty"))
        captured: list[str] = []

        class _Rec:
            def warning(self, msg, *args):
                captured.append(msg)

        plugin.logger = _Rec()
        plugin._warn_stray_runtime_config()
        assert captured == [], "无残留时不应告警"


class TestMetaPrefixStripping:
    """M-7：祈使前缀剥离（与 test_core/test_engine 的行为断言互补，这里钉契约细节）。"""

    def test_strip_is_iterative_and_keeps_meaning(self):
        from core.admission import strip_meta_prefix

        text, changed = strip_meta_prefix("别忘了提醒我明天开会")
        assert changed and text == "明天开会"

    def test_strip_empty_remainder_keeps_original(self):
        from core.admission import strip_meta_prefix

        text, changed = strip_meta_prefix("记住")
        assert not changed and text == "记住", "只剩前缀时不剥离，交噪声门处理"

    def test_prefix_and_content_used_for_dedup(self, tmp_path):
        """剥离后与既有无前缀记忆同文 → 走强化而非新增（不产生重复）。"""
        import asyncio

        eng, store, conn = _make(tmp_path)
        try:
            async def run():
                ok1 = await eng.remember("小明喜欢在深夜听广播", alpha=0.9, scope="default")
                ok2 = await eng.remember("记住小明喜欢在深夜听广播", alpha=0.9, scope="default")
                return ok1, ok2

            ok1, ok2 = asyncio.run(run())
            assert ok1 and ok2
            assert store.count()["total"] == 1, "剥离前缀后应与既有条目判重，不新增"
        finally:
            conn.close()
