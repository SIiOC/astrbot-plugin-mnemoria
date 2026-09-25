"""import_export.py 脚本级回归（部署迁移前置把关，2026-09-15）。

覆盖部署前发现的两个缺口：
1. --vectorize 做实：缺向量筛选 / 批量补嵌 / 批次失败可续补 / 幂等重跑
2. 时间保真：observed_at 回填 created_at/observed_at（否则迁移记忆伪装成新记忆）

全部离线：embed_fn 注入假函数，不碰网络、不读 AstrBot 主配置。
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))


@pytest.fixture()
def fresh_db(tmp_path):
    from core import db as dbm
    p = tmp_path / "mig.db"
    conn = dbm.connect(p)
    dbm.init_schema(conn)
    yield conn
    conn.close()


def _write_dump(tmp_path, memories):
    p = tmp_path / "dump.json"
    p.write_text(json.dumps({"source": "angel_memory", "memories": memories},
                            ensure_ascii=False), encoding="utf-8")
    return p


def _fake_embed_factory(calls):
    """记录调用批次的假嵌入函数：每条文本一个可识别向量。"""
    def fake(texts):
        calls.append(list(texts))
        return [[1.0, float(len(t) % 7), 0.5] for t in texts]
    return fake


class TestPreserveTs:
    def test_observed_at_backfilled(self, tmp_path, fresh_db):
        from import_export import do_import
        dump = _write_dump(tmp_path, [{
            "content": "用户养了一只叫豆豆的猫", "memory_type": "fact",
            "is_active": 0, "strength": 10.0, "useful_score": 2.5, "useful_count": 1,
            "observed_at": 1700000000.0, "scope": "default",
        }])
        do_import(fresh_db_path(tmp_path), dump, preserve_ts=True)
        row = fresh_db.execute(
            "SELECT created_at, observed_at, valid_from FROM memories").fetchone()
        assert row["created_at"] == 1700000000.0, "created_at 必须回填原始观察时间"
        assert row["observed_at"] == 1700000000.0
        assert row["valid_from"] == 1700000000.0

    def test_no_preserve_ts_keeps_now(self, tmp_path, fresh_db):
        from import_export import do_import
        from core.paths import utc_now_ts
        dump = _write_dump(tmp_path, [{
            "content": "时间不保真的记忆", "observed_at": 1700000000.0, "scope": "default",
        }])
        do_import(fresh_db_path(tmp_path), dump, preserve_ts=False)
        row = fresh_db.execute("SELECT created_at FROM memories").fetchone()
        assert abs(row["created_at"] - utc_now_ts()) < 60, "关闭保真时应按导入时间"

    def test_zero_or_missing_ts_skipped(self, tmp_path, fresh_db):
        from import_export import do_import
        dump = _write_dump(tmp_path, [
            {"content": "无时间戳的记忆A", "scope": "default"},
            {"content": "时间戳为0的记忆B", "observed_at": 0, "scope": "default"},
        ])
        do_import(fresh_db_path(tmp_path), dump, preserve_ts=True)
        rows = fresh_db.execute(
            "SELECT content, created_at FROM memories ORDER BY content").fetchall()
        assert len(rows) == 2
        now = fresh_db.execute("SELECT MAX(created_at) m FROM memories").fetchone()["m"]
        for r in rows:
            assert abs(r["created_at"] - now) < 60, "无效时间戳不得回填"


def fresh_db_path(tmp_path):
    return tmp_path / "mig.db"


class TestVectorize:
    def test_backfill_embeds_missing_only(self, tmp_path, fresh_db):
        from import_export import do_import, backfill_vectors, missing_vector_rows
        dump = _write_dump(tmp_path, [
            {"content": "待补嵌记忆甲", "scope": "default"},
            {"content": "待补嵌记忆乙", "scope": "default"},
        ])
        do_import(fresh_db_path(tmp_path), dump)
        assert len(missing_vector_rows(fresh_db)) == 2

        grp = {"embedding_api_base": "http://x", "embedding_api_key": "k", "embedding_model": "m"}
        calls = []
        n = backfill_vectors(fresh_db, grp, batch=20, embed_fn=_fake_embed_factory(calls))
        assert n == 2 and len(calls) == 1 and len(calls[0]) == 2
        assert missing_vector_rows(fresh_db) == []

        vecs = fresh_db.execute("SELECT memory_id, dim FROM vectors").fetchall()
        assert len(vecs) == 2 and all(v["dim"] == 3 for v in vecs)

    def test_backfill_resumable_after_batch_fail(self, tmp_path, fresh_db):
        from import_export import do_import, backfill_vectors, missing_vector_rows
        dump = _write_dump(tmp_path, [
            {"content": f"批量失败续补记忆{i}", "scope": "default"} for i in range(5)
        ])
        do_import(fresh_db_path(tmp_path), dump)

        def boom(texts):
            raise RuntimeError("网络炸了")

        grp = {"embedding_model": "m"}
        n = backfill_vectors(fresh_db, grp, batch=2, embed_fn=boom)
        assert n == 0, "批次全失败时不得虚报完成"
        assert len(missing_vector_rows(fresh_db)) == 5, "失败行保持无向量等待重跑"

        calls = []
        n2 = backfill_vectors(fresh_db, grp, batch=2, embed_fn=_fake_embed_factory(calls))
        assert n2 == 5 and missing_vector_rows(fresh_db) == [], "重跑后全部补齐"
        assert [len(c) for c in calls] == [2, 2, 1], "按批推进"

    def test_reimport_reinforces_not_duplicates(self, tmp_path, fresh_db):
        """幂等重跑：同一 dump 导两次，第二次只强化不新增。"""
        from import_export import do_import
        dump = _write_dump(tmp_path, [{"content": "幂等重跑的记忆", "scope": "default"}])
        do_import(fresh_db_path(tmp_path), dump)
        do_import(fresh_db_path(tmp_path), dump)
        assert fresh_db.execute("SELECT COUNT(*) c FROM memories").fetchone()["c"] == 1
        row = fresh_db.execute("SELECT proof_count, strength FROM memories").fetchone()
        assert row["proof_count"] == 2 and row["strength"] == 10.5, "强化走 +0.5 strength"

    def test_active_promotion_on_merge(self, tmp_path, fresh_db):
        """angel 同文双版本（主动+被动）：合并后必须升为主动。"""
        from import_export import do_import
        dump = _write_dump(tmp_path, [
            {"content": "同文双版本记忆", "is_active": 0, "scope": "default"},
            {"content": "同文双版本记忆", "is_active": 1, "scope": "default"},
        ])
        do_import(fresh_db_path(tmp_path), dump)
        row = fresh_db.execute(
            "SELECT is_active, proof_count FROM memories").fetchone()
        assert row["is_active"] == 1 and row["proof_count"] == 2
