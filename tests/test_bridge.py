"""桥接层与工具函数测试：LLM JSON 解析、embedding/rerank 降级、备份、锁。"""

from __future__ import annotations

import asyncio
import json

import pytest

from core.llm import _extract_text, _slice_braces, parse_json_loose
from core.templates import build_transcript


# ---------------------------------------------------------------- JSON 解析
class TestJsonParse:
    @pytest.mark.parametrize("raw,expect", [
        ('{"a": 1}', {"a": 1}),
        ('```json\n{"a": 1}\n```', {"a": 1}),
        ('```\n{"a": 1}\n```', {"a": 1}),
        ('前置噪声 {"a": 1} 后置噪声', {"a": 1}),
        ('[{"a": 1}]', [{"a": 1}]),
        ('前面文字 [1, 2, 3] 后面', [1, 2, 3]),
    ])
    def test_parse_variants(self, raw, expect):
        assert parse_json_loose(raw) == expect

    def test_parse_invalid_returns_none(self):
        assert parse_json_loose("完全不是JSON") is None
        assert parse_json_loose("") is None
        assert parse_json_loose(None) is None

    def test_parse_truncated_json_returns_none(self):
        assert parse_json_loose('{"a": ') is None

    def test_slice_braces(self):
        assert _slice_braces('xx{"a":1}yy') == '{"a":1}'

    def test_slice_braces_no_brace(self):
        assert _slice_braces("no braces") == ""

    def test_extract_text_from_object(self):
        class R:
            completion_text = "hello"
        assert _extract_text(R()) == "hello"

    def test_extract_text_from_dict(self):
        assert _extract_text({"text": "hi"}) == "hi"

    def test_extract_text_from_str(self):
        assert _extract_text("direct") == "direct"

    def test_extract_text_none(self):
        assert _extract_text(None) is None


# ---------------------------------------------------------------- 桥接降级
class TestBridgeDegradation:
    async def test_embedder_disabled_without_provider(self):
        from core.bridge import Embedder
        e = Embedder(None, "")
        assert e.enabled is False
        assert await e.embed(["x"]) is None
        assert await e.embed_one("x") is None

    async def test_embedder_handles_provider_missing(self):
        from core.bridge import Embedder

        class PM:
            @staticmethod
            async def get_provider_by_id(pid):
                return None

        class Ctx:
            provider_manager = PM()

        e = Embedder(Ctx(), "nonexistent")
        assert await e.embed(["x"]) is None
        assert e.enabled is False  # 失败后标记不可用

    async def test_embedder_timeout_returns_none(self):
        from core.bridge import Embedder

        class Slow:
            async def get_embeddings(self, texts):
                await asyncio.sleep(5)
                return [[1.0]]

        class PM:
            @staticmethod
            async def get_provider_by_id(pid):
                return Slow()

        class Ctx:
            provider_manager = PM()

        e = Embedder(Ctx(), "p", timeout=0.05)
        assert await e.embed(["x"]) is None

    async def test_embedder_batches_by_20(self):
        from core.bridge import Embedder

        calls = []

        class P:
            async def get_embeddings(self, texts):
                calls.append(len(texts))
                return [[0.1] for _ in texts]

        class PM:
            @staticmethod
            async def get_provider_by_id(pid):
                return P()

        class Ctx:
            provider_manager = PM()

        e = Embedder(Ctx(), "p")
        out = await e.embed([f"t{i}" for i in range(45)])
        assert out is not None and len(out) == 45
        assert calls == [20, 20, 5]

    async def test_reranker_disabled_without_provider(self):
        from core.bridge import Reranker
        r = Reranker(None, "")
        assert r.enabled is False
        assert await r.rerank("q", ["a", "b"]) is None

    async def test_reranker_parses_dict_results(self):
        from core.bridge import Reranker

        class P:
            async def rerank(self, query, documents):
                return [{"index": 1, "relevance_score": 0.9}, {"index": 0, "relevance_score": 0.1}]

        class PM:
            @staticmethod
            async def get_provider_by_id(pid):
                return P()

        class Ctx:
            provider_manager = PM()

        r = Reranker(Ctx(), "p")
        assert await r.rerank("q", ["a", "b"]) == [1, 0]

    async def test_reranker_failure_returns_none(self):
        from core.bridge import Reranker

        class P:
            async def rerank(self, query, documents):
                raise RuntimeError("boom")

        class PM:
            @staticmethod
            async def get_provider_by_id(pid):
                return P()

        class Ctx:
            provider_manager = PM()

        r = Reranker(Ctx(), "p")
        assert await r.rerank("q", ["a"]) is None


# ---------------------------------------------------------------- LLM 桥接
class TestLLMBridge:
    async def test_disabled_without_provider(self):
        from core.llm import LLMBridge
        b = LLMBridge(None, "")
        assert b.enabled is False
        assert await b.generate("p") is None
        assert await b.generate_json("p") is None

    async def test_generate_success(self):
        from core.llm import LLMBridge

        class Ctx:
            async def llm_generate(self, **kw):
                class R:
                    completion_text = "结果"
                return R()

        b = LLMBridge(Ctx(), "p")
        assert await b.generate("hi") == "结果"

    async def test_generate_timeout(self):
        from core.llm import LLMBridge

        class Ctx:
            async def llm_generate(self, **kw):
                await asyncio.sleep(5)

        b = LLMBridge(Ctx(), "p", timeout=0.05)
        assert await b.generate("hi") is None

    async def test_generate_exception(self):
        from core.llm import LLMBridge

        class Ctx:
            async def llm_generate(self, **kw):
                raise RuntimeError("provider down")

        b = LLMBridge(Ctx(), "p")
        assert await b.generate("hi") is None

    async def test_generate_json_parses(self):
        from core.llm import LLMBridge

        class Ctx:
            async def llm_generate(self, **kw):
                class R:
                    completion_text = '```json\n{"x": 1}\n```'
                return R()

        b = LLMBridge(Ctx(), "p")
        assert await b.generate_json("hi") == {"x": 1}


# ---------------------------------------------------------------- 模板
class TestTemplates:
    def test_build_transcript_roles(self):
        # v0.1.6：身份化格式 [用户]/[助理]（user_label 可覆盖用户侧）
        t = build_transcript([("user", "你好"), ("assistant", "在的")])
        assert "[用户]: 你好" in t and "[助理]: 在的" in t
        t2 = build_transcript([("user", "你好")], user_label="小明（u1）")
        assert "[小明（u1）]: 你好" in t2

    def test_build_transcript_truncates(self):
        t = build_transcript([("user", "x" * 10000)], max_chars=100)
        assert len(t) <= 130 and "前略" in t


# ---------------------------------------------------------------- 备份
class TestBackup:
    def test_backup_and_prune(self, tmp_path):
        from core import db as dbm
        from core.backup import list_backups, write_backup
        from core.paths import DataPaths
        from core.store import MemoryStore

        paths = DataPaths(tmp_path / "pd").ensure()
        conn = dbm.connect(paths.db); dbm.init_schema(conn)
        store = MemoryStore(conn)
        store.add_memory("用户喜欢猫", scope="s")

        for i in range(5):
            write_backup(paths, conn, keep=3)
        files = list_backups(paths)
        assert len(files) <= 3
        # 内容可解析
        p = paths.backups / files[0]["name"]
        data = json.loads(p.read_text(encoding="utf-8"))
        assert any(m["content"] == "用户喜欢猫" for m in data["memories"])
        conn.close()

    def test_backup_recovers_from_bad_dir(self, tmp_path):
        from core.backup import write_backup
        from core.paths import DataPaths
        import sqlite3
        paths = DataPaths(tmp_path / "pd2").ensure()
        conn = sqlite3.connect(":memory:")
        # 未建表 → 查询失败应被吞掉，返回 None 而非抛错
        assert write_backup(paths, conn) is None


# ---------------------------------------------------------------- 锁
class TestLocks:
    def test_different_keys_parallel(self):
        from core.locks import PartitionLock

        async def drive():
            pl = PartitionLock()
            async with pl.hold("a"), pl.hold("b"):
                pass  # 不同 key 不应死锁

        asyncio.run(drive())
