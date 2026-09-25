"""记忆与账本的持久化操作（SQLite）。

本模块只做数据进出，不含业务判定；业务逻辑在 admission/retrieve/decay。
所有写操作在一个短事务内完成，配合连接级 busy_timeout 与分区锁防竞态。
"""

from __future__ import annotations

import logging
import json
import sqlite3
import uuid
from typing import Any, Iterable

from . import db as dbm
from .paths import utc_now_ts
from .tags import normalize_tags
from .text import content_hash

logger = logging.getLogger(__name__)

def _tags_to_json(tags: list[str] | None) -> str:
    """tags 规范化落库：去空白/去重/截 6 个/每个截 24 字。"""
    if not tags:
        return "[]"
    out: list[str] = []
    for t in tags:
        t = str(t or "").strip()[:24]
        if t and t not in out:
            out.append(t)
        if len(out) >= 6:
            break
    return json.dumps(out, ensure_ascii=False)


def _tags_from_json(raw) -> list[str]:
    try:
        v = json.loads(raw) if raw else []
        return [str(x) for x in v] if isinstance(v, list) else []
    except Exception:  # noqa: BLE001
        return []


_MEM_COLS = (
    "id, content, reasoning, memory_type, source, speaker, speaker_key, is_active, strength, "
    "useful_score, useful_count, hit_count, last_recalled_at, last_decay_at, "
    "proof_count, observed_at, valid_from, valid_to, superseded_by, deleted_at, "
    "quarantined, scope, session_id, content_hash, tags_json, created_at, updated_at"
)

# 笔记对外列：刻意排除 vec/vec_dim（BLOB，不可 JSON 序列化）
_NOTE_PUBLIC_FIELDS = (
    "id", "title", "content", "tags", "source", "file_name", "heading", "scope",
    "content_hash", "deleted_at", "created_at", "updated_at",
)
_NOTE_PUBLIC_COLS = ", ".join(_NOTE_PUBLIC_FIELDS)
_NOTE_PUBLIC_COLS_PREFIXED = ", ".join("n." + f for f in _NOTE_PUBLIC_FIELDS)


class MemoryStore:
    def __init__(self, conn: sqlite3.Connection) -> None:
        self.conn = conn
        self.fts = dbm.fts_available(conn)

    # ------------------------------------------------------------------ 记忆
    def add_memory(
        self,
        content: str,
        *,
        reasoning: str = "",
        memory_type: str = "fact",
        source: str = "user",
        speaker: str = "",
        speaker_key: str = "",
        scope: str = "public",
        session_id: str = "",
        is_active: bool = False,
        strength: float = 10.0,
        proof_count: int = 1,
        valid_from: float | None = None,
        mem_id: str | None = None,
        quarantined: bool = False,
        tags: list[str] | None = None,
    ) -> str:
        mid = self._insert_memory(
            content,
            reasoning=reasoning, memory_type=memory_type, source=source,
            speaker=speaker, speaker_key=speaker_key, scope=scope,
            session_id=session_id, is_active=is_active, strength=strength,
            proof_count=proof_count, valid_from=valid_from, mem_id=mem_id,
            quarantined=quarantined, tags=tags,
        )
        self.conn.commit()
        return mid

    def _insert_memory(
        self,
        content: str,
        *,
        reasoning: str = "",
        memory_type: str = "fact",
        source: str = "user",
        speaker: str = "",
        speaker_key: str = "",
        scope: str = "public",
        session_id: str = "",
        is_active: bool = False,
        strength: float = 10.0,
        proof_count: int = 1,
        valid_from: float | None = None,
        mem_id: str | None = None,
        quarantined: bool = False,
        tags: list[str] | None = None,
    ) -> str:
        """执行插入但不提交（供单事务裁决使用）。"""
        now = utc_now_ts()
        mid = mem_id or uuid.uuid4().hex
        self.conn.execute(
            "INSERT INTO memories (id, content, reasoning, memory_type, source, speaker, speaker_key, "
            "is_active, strength, useful_score, useful_count, hit_count, last_recalled_at, "
            "last_decay_at, proof_count, observed_at, valid_from, valid_to, superseded_by, "
            "deleted_at, quarantined, scope, session_id, content_hash, tags_json, created_at, updated_at) "
            "VALUES (?,?,?,?,?,?,?,?,?,0,0,0,0,0,?,?,?,NULL,NULL,?,?,?,?,?,?,?,?)",
            (
                mid, content, reasoning, memory_type, source, speaker, speaker_key,
                1 if is_active else 0, float(strength), int(proof_count),
                now, (valid_from if valid_from is not None else now),
                (now if quarantined else None), 1 if quarantined else 0,
                scope, session_id, content_hash(content),
                _tags_to_json(tags), now, now,
            ),
        )
        return mid

    @staticmethod
    def parse_tags(row) -> list[str]:
        """从行对象安全解出 tags（旧库/异常值回 []）。"""
        return _tags_from_json(row["tags_json"] if "tags_json" in row.keys() else None)

    def get_memory(self, mem_id: str) -> sqlite3.Row | None:
        return self.conn.execute(
            f"SELECT {_MEM_COLS} FROM memories WHERE id=?", (mem_id,)
        ).fetchone()

    def get_by_hash(self, chash: str, scope: str) -> sqlite3.Row | None:
        return self.conn.execute(
            f"SELECT {_MEM_COLS} FROM memories WHERE content_hash=? AND scope=? AND deleted_at IS NULL",
            (chash, scope),
        ).fetchone()

    def update_memory(self, mem_id: str, **fields: Any) -> None:
        if not fields:
            return
        allowed = {
            "content", "reasoning", "memory_type", "strength", "useful_score",
            "useful_count", "hit_count", "last_recalled_at", "last_decay_at",
            "proof_count", "valid_to", "superseded_by", "deleted_at", "is_active",
            "scope", "session_id", "quarantined", "speaker_key",
        }
        sets = [f"{k}=?" for k in fields if k in allowed]
        vals = [fields[k] for k in fields if k in allowed]
        if not sets:
            return
        sets.append("updated_at=?")
        vals.append(utc_now_ts())
        vals.append(mem_id)
        self.conn.execute(f"UPDATE memories SET {', '.join(sets)} WHERE id=?", vals)
        self.conn.commit()

    def reinforce(self, mem_id: str, useful_delta: float, strength_delta: float = 1.0) -> None:
        """被判定有用：useful_score/strength 增量、proof_count+1（增量信念）。"""
        now = utc_now_ts()
        self.conn.execute(
            "UPDATE memories SET useful_score=useful_score+?, useful_count=useful_count+1, "
            "strength=strength+?, proof_count=proof_count+1, updated_at=? WHERE id=?",
            (float(useful_delta), float(strength_delta), now, mem_id),
        )
        self.conn.commit()

    def mark_recalled(self, mem_ids: Iterable[str]) -> None:
        """召回命中：hit_count+1 并刷新 last_recalled_at（访问强化，也用于衰减锚点）。"""
        now = utc_now_ts()
        ids = list(mem_ids)
        if not ids:
            return
        self.conn.executemany(
            "UPDATE memories SET hit_count=hit_count+1, last_recalled_at=?, updated_at=? WHERE id=?",
            [(now, now, mid) for mid in ids],
        )
        self.conn.commit()

    def penalize(self, mem_id: str, useful_delta: float) -> None:
        """被判定「召回但没用」：扣减 useful_score（下限 0），不动 strength。

        这是本插件唯一让 useful_score 递减的入口（反射闭环），
        用于把「老被召回却从没帮上忙」的记忆慢慢降出长期保留档。
        """
        now = utc_now_ts()
        self.conn.execute(
            "UPDATE memories SET useful_score=MAX(0.0, useful_score-?), updated_at=? WHERE id=?",
            (float(useful_delta), now, mem_id),
        )
        self.conn.commit()

    def supersede(self, old_id: str, new_id: str) -> None:
        """矛盾时旧记忆标记被取代（软链，不物理覆盖，可复活）。"""
        now = utc_now_ts()
        self.conn.execute(
            "UPDATE memories SET superseded_by=?, valid_to=?, updated_at=? WHERE id=?",
            (new_id, now, now, old_id),
        )
        self.conn.commit()

    def adjudicated_write(
        self,
        *,
        action: str,
        scope: str,
        content: str,
        memory_type: str = "fact",
        source: str = "user",
        speaker: str = "",
        speaker_key: str = "",
        session_id: str = "",
        is_active: bool = False,
        strength: float = 10.0,
        reasoning: str = "",
        tags: list[str] | None = None,
        target_ids: list[str] | None = None,
        vec: list[float] | None = None,
        useful_delta: float = 1.0,
        reason: str = "",
        confidence: float = 0.0,
        provider: str = "",
    ) -> dict | None:
        """单事务执行一次写入裁决，失败整组回滚并返回 None。

        - reinforce：只强化目标（不新增）
        - merge/update：新增一条，旧条 superseded_by/valid_to 失效并入回收站
          （面板可见可恢复），merge 额外继承有用分与证据计数
        - add：普通新增
        - noop：不落库，仅记审计事件
        主动记忆（is_active=1）永不赦免于被取代——返回 None 让调用方回退普通新增。
        """
        action = (action or "").strip().lower()
        if action not in ("add", "reinforce", "merge", "update", "noop"):
            return None
        targets = [str(x) for x in (target_ids or []) if str(x).strip()]
        if action in ("reinforce", "merge", "update") and not targets:
            return None
        now = utc_now_ts()
        try:
            if action == "noop":
                self._event_conn(
                    "noop", scope=scope, source_ids=targets, target_id="",
                    reason=reason, confidence=confidence, provider=provider,
                )
                self.conn.commit()
                return {"ok": True, "action": "noop", "target_id": ""}

            if action == "reinforce":
                tid = targets[0]
                row = self.conn.execute(
                    "SELECT id, scope, deleted_at, superseded_by FROM memories WHERE id=?",
                    (tid,),
                ).fetchone()
                if row is None or str(row["scope"] or "") != scope \
                        or row["deleted_at"] or row["superseded_by"]:
                    self.conn.rollback()
                    return None
                self.conn.execute(
                    "UPDATE memories SET useful_score=useful_score+?, useful_count=useful_count+1, "
                    "strength=strength+1.0, proof_count=proof_count+1, updated_at=? WHERE id=?",
                    (float(useful_delta), now, tid),
                )
                self._event_conn(
                    "reinforce", scope=scope, source_ids=[], target_id=tid,
                    reason=reason, confidence=confidence, provider=provider,
                )
                self.conn.commit()
                return {"ok": True, "action": "reinforce", "target_id": tid}

            # merge/update：先校验目标（跨域/退场/永生条目一律不算数）
            valid: list[sqlite3.Row] = []
            for oid in targets:
                row = self.conn.execute(
                    "SELECT id, scope, deleted_at, superseded_by, is_active, "
                    "useful_score, proof_count FROM memories WHERE id=?",
                    (oid,),
                ).fetchone()
                if row is None or str(row["scope"] or "") != scope:
                    continue
                if row["deleted_at"] or row["superseded_by"]:
                    continue
                if row["is_active"]:
                    logger.info("写入裁决跳过主动记忆 %s（永生条目不可被取代）", oid)
                    continue
                valid.append(row)
            if action in ("merge", "update") and not valid:
                self.conn.rollback()
                return None

            mid = self._insert_memory(
                content, memory_type=memory_type, source=source, speaker=speaker,
                speaker_key=speaker_key, scope=scope, session_id=session_id,
                is_active=is_active, strength=strength, reasoning=reasoning,
                valid_from=None, tags=tags,
            )
            if vec:
                self._set_vector_conn(mid, vec)
            for row in valid:
                self.conn.execute(
                    "UPDATE memories SET superseded_by=?, valid_to=?, deleted_at=?, "
                    "updated_at=? WHERE id=?",
                    (mid, now, now, now, row["id"]),
                )
            if action == "merge" and valid:
                useful = max([float(r["useful_score"] or 0.0) for r in valid])
                proof = sum(int(r["proof_count"] or 1) for r in valid)
                self.conn.execute(
                    "UPDATE memories SET useful_score=?, proof_count=?, updated_at=? WHERE id=?",
                    (useful, proof, now, mid),
                )
            self._event_conn(
                action, scope=scope, source_ids=[r["id"] for r in valid],
                target_id=mid, reason=reason, confidence=confidence, provider=provider,
            )
            self.conn.commit()
            return {"ok": True, "action": action, "target_id": mid}
        except sqlite3.Error as exc:
            try:
                self.conn.rollback()
            except sqlite3.Error:
                pass
            logger.warning("写入裁决落库失败，已回滚（调用方回退普通写入）: %s", exc)
            return None

    def _set_vector_conn(self, mem_id: str, vec: list[float]) -> None:
        """写向量但不提交（事务内部使用）。"""
        from .vector import pack
        self.conn.execute(
            "INSERT INTO vectors(memory_id, dim, vec) VALUES(?,?,?) "
            "ON CONFLICT(memory_id) DO UPDATE SET dim=excluded.dim, vec=excluded.vec",
            (mem_id, len(vec), pack(vec)),
        )

    def _event_conn(
        self, action: str, *, scope: str, source_ids: list[str] | None = None,
        target_id: str = "", reason: str = "", confidence: float = 0.0,
        provider: str = "",
    ) -> None:
        """写审计事件但不提交（事务内部使用）。"""
        self.conn.execute(
            "INSERT INTO memory_events(action,scope,source_ids_json,target_id,reason,"
            "confidence,provider,created_at) VALUES(?,?,?,?,?,?,?,?)",
            (
                str(action), str(scope),
                json.dumps(list(source_ids or []), ensure_ascii=False),
                str(target_id or ""), str(reason or ""), float(confidence or 0.0),
                str(provider or ""), utc_now_ts(),
            ),
        )

    def record_memory_event(
        self,
        action: str,
        *,
        scope: str,
        source_ids: list[str] | None = None,
        target_id: str = "",
        reason: str = "",
        confidence: float = 0.0,
        provider: str = "",
    ) -> int:
        """记录不可变的记忆动作审计。"""
        self._event_conn(
            action, scope=scope, source_ids=source_ids, target_id=target_id,
            reason=reason, confidence=confidence, provider=provider,
        )
        self.conn.commit()
        row = self.conn.execute("SELECT MAX(id) AS m FROM memory_events").fetchone()
        return int(row["m"] or 0) if row else 0

    def list_memory_events(self, *, target_id: str = "", limit: int = 100) -> list[sqlite3.Row]:
        if target_id:
            return self.conn.execute(
                "SELECT * FROM memory_events WHERE target_id=? ORDER BY id DESC LIMIT ?",
                (target_id, int(limit)),
            ).fetchall()
        return self.conn.execute(
            "SELECT * FROM memory_events ORDER BY id DESC LIMIT ?", (int(limit),)
        ).fetchall()

    def active_memories(self, scope: str, include_superseded: bool = False) -> list[sqlite3.Row]:
        sql = f"SELECT {_MEM_COLS} FROM memories WHERE scope=? AND deleted_at IS NULL"
        if not include_superseded:
            sql += " AND superseded_by IS NULL"
        return self.conn.execute(sql, (scope,)).fetchall()

    def set_tags(self, mem_id: str, tags) -> None:
        """只改 tags_json（存量回填/面板编辑用），不动其它列。

        归一化走 core.tags.normalize_tags（与抽取写入、回填脚本同一份规则）。
        """
        clean = normalize_tags(tags)
        self.conn.execute(
            "UPDATE memories SET tags_json=?, updated_at=? WHERE id=?",
            (json.dumps(clean, ensure_ascii=False), utc_now_ts(), mem_id),
        )
        self.conn.commit()

    def trash(self, mem_id: str) -> None:
        """放入回收站（软删）。"""
        self.conn.execute(
            "UPDATE memories SET deleted_at=?, updated_at=? WHERE id=?",
            (utc_now_ts(), utc_now_ts(), mem_id),
        )
        self.conn.commit()

    def restore(self, mem_id: str, *, clear_superseded: bool = False) -> None:
        """从回收站恢复。

        默认恢复普通软删/隔离条；若条目已被新说法取代，则保留 deleted_at、
        superseded_by 与 valid_to，继续留在回收站供复核且不进入检索面。面板的
        「彻底恢复」或人工明确指定 clear_superseded=True，才清除血缘并重新可检索。
        """
        if clear_superseded:
            self.conn.execute(
                "UPDATE memories SET deleted_at=NULL, quarantined=0, "
                "superseded_by=NULL, valid_to=NULL, updated_at=? WHERE id=?",
                (utc_now_ts(), mem_id),
            )
        else:
            # 已被取代的旧说法继续保留 deleted_at，因而留在回收站可见层，
            # 也继续受 trash_retention_days 清理；普通软删/隔离条目则恢复到活跃面。
            self.conn.execute(
                "UPDATE memories SET deleted_at=CASE WHEN superseded_by IS NULL "
                "THEN NULL ELSE deleted_at END, quarantined=0, updated_at=? WHERE id=?",
                (utc_now_ts(), mem_id),
            )
        self.conn.commit()

    def purge(self, mem_id: str) -> None:
        """物理删除（回收站逾期清理时用）。"""
        self.conn.execute("DELETE FROM memories WHERE id=?", (mem_id,))
        self.conn.execute("DELETE FROM vectors WHERE memory_id=?", (mem_id,))
        self.conn.commit()

    def list_trash(self, older_than_ts: float | None = None) -> list[sqlite3.Row]:
        # superseded 旧说法即使已点过「恢复」也留在回收站可见层，
        # 直到 clear_superseded=True 或 purge；检索仍由 superseded_by 条件排除。
        where = "(deleted_at IS NOT NULL OR superseded_by IS NOT NULL)"
        params: tuple = ()
        if older_than_ts is not None:
            # 没有 deleted_at 的历史异常行只能人工彻底恢复，不能因夜间清理
            # 的时间条件被误删；正常演进旧条始终带 deleted_at。
            where += " AND deleted_at IS NOT NULL AND deleted_at < ?"
            params = (older_than_ts,)
        return self.conn.execute(
            f"SELECT {_MEM_COLS} FROM memories WHERE {where}", params
        ).fetchall()

    # ------------------------------------------------------------------ 笔记回收站
    def list_trash_notes(self, older_than_ts: float | None = None,
                         limit: int = 500) -> list[sqlite3.Row]:
        """回收站里的笔记（v0.1.9）。

        此前笔记只有软删没有恢复也没有清理：purge_trash 只扫 memories，
        restore_note/purge_note 成了死代码，trash_retention_days 对笔记不生效
        （删掉即永久消失且占用库）。本方法与 web_api 的 note/restore 一起
        把笔记纳入与记忆同一套回收站生命周期。
        只取公开列（vec 是 BLOB，不可 JSON 序列化）。
        """
        sql = (
            f"SELECT {_NOTE_PUBLIC_COLS} FROM notes WHERE deleted_at IS NOT NULL"
        )
        params: list = []
        if older_than_ts is not None:
            sql += " AND deleted_at < ?"
            params.append(float(older_than_ts))
        sql += " ORDER BY deleted_at DESC LIMIT ?"
        params.append(int(limit))
        return self.conn.execute(sql, params).fetchall()

    def count(self, scope: str | None = None) -> dict[str, int]:
        where = "WHERE deleted_at IS NULL"
        params: tuple = ()
        if scope:
            where += " AND scope=?"
            params = (scope,)
        row = self.conn.execute(
            f"SELECT COUNT(*) n, SUM(is_active) a FROM memories {where}", params
        ).fetchone()
        trash = self.conn.execute(
            "SELECT COUNT(*) n FROM memories WHERE deleted_at IS NOT NULL"
        ).fetchone()
        return {"total": int(row["n"] or 0), "active": int(row["a"] or 0), "trash": int(trash["n"] or 0)}

    # ------------------------------------------------------------------ 向量
    def set_vector(self, mem_id: str, vec: list[float]) -> None:
        from .vector import pack
        self.conn.execute(
            "INSERT INTO vectors(memory_id, dim, vec) VALUES(?,?,?) "
            "ON CONFLICT(memory_id) DO UPDATE SET dim=excluded.dim, vec=excluded.vec",
            (mem_id, len(vec), pack(vec)),
        )
        self.conn.commit()

    def get_all_vectors(self) -> list[tuple[str, list[float]]]:
        from .vector import unpack
        rows = self.conn.execute(
            "SELECT v.memory_id, v.dim, v.vec FROM vectors v "
            "JOIN memories m ON m.id=v.memory_id WHERE m.deleted_at IS NULL"
        ).fetchall()
        return [(r["memory_id"], unpack(r["vec"], r["dim"])) for r in rows]

    def delete_vector(self, mem_id: str) -> None:
        self.conn.execute("DELETE FROM vectors WHERE memory_id=?", (mem_id,))
        self.conn.commit()

    # ---------------------------------------------------------- 向量回填队列
    def enqueue_vector_backlog(self, mem_id: str, dim_seen: int = 0) -> bool:
        """把活跃记忆放入惰性回填队列；重复入队保持首次时间与失败次数。"""
        row = self.conn.execute(
            "SELECT id FROM memories WHERE id=? AND deleted_at IS NULL", (str(mem_id),)
        ).fetchone()
        if row is None:
            return False
        cur = self.conn.execute(
            "INSERT OR IGNORE INTO vector_backlog(memory_id, dim_seen, enqueued_at, attempts) "
            "VALUES(?,?,?,0)",
            (str(mem_id), int(dim_seen or 0), utc_now_ts()),
        )
        if cur.rowcount == 0 and dim_seen:
            self.conn.execute(
                "UPDATE vector_backlog SET dim_seen=? WHERE memory_id=? AND dim_seen=0",
                (int(dim_seen), str(mem_id)),
            )
        self.conn.commit()
        return True

    def vector_backlog(self, limit: int = 16, max_attempts: int = 3) -> list[sqlite3.Row]:
        """按入队时间取活跃待回填记忆；达到失败上限的条目暂时跳过。"""
        return self.conn.execute(
            "SELECT b.memory_id, b.dim_seen, b.enqueued_at, b.attempts, m.content, m.scope "
            "FROM vector_backlog b JOIN memories m ON m.id=b.memory_id "
            "WHERE m.deleted_at IS NULL AND b.attempts<? "
            "ORDER BY b.enqueued_at ASC, b.memory_id ASC LIMIT ?",
            (max(0, int(max_attempts)), max(1, int(limit))),
        ).fetchall()

    def vector_backlog_count(self, *, include_failed: bool = False) -> int:
        where = ""
        params: tuple = ()
        if not include_failed:
            where = " AND b.attempts < 3"
        row = self.conn.execute(
            "SELECT COUNT(*) FROM vector_backlog b JOIN memories m ON m.id=b.memory_id "
            "WHERE m.deleted_at IS NULL" + where, params,
        ).fetchone()
        return int(row[0] or 0)

    def mark_vector_backlog_failed(self, mem_id: str) -> int:
        """记录一次回填失败并返回新的失败次数；达到 3 次由消费端记录跳过日志。"""
        self.conn.execute(
            "UPDATE vector_backlog SET attempts=attempts+1 WHERE memory_id=?",
            (str(mem_id),),
        )
        self.conn.commit()
        row = self.conn.execute(
            "SELECT attempts FROM vector_backlog WHERE memory_id=?", (str(mem_id),)
        ).fetchone()
        return int(row[0] or 0) if row else 0

    def remove_vector_backlog(self, mem_id: str) -> None:
        self.conn.execute("DELETE FROM vector_backlog WHERE memory_id=?", (str(mem_id),))
        self.conn.commit()

    # ------------------------------------------------------------------ 账本
    def append_ledger(
        self, session_id: str, role: str, content: str, ts: float, scope: str = "public",
        message_id: str | None = None,
    ) -> bool:
        """写入一条流水；message_id 重复则忽略（幂等）。返回是否写入。"""
        try:
            cur = self.conn.execute(
                "INSERT OR IGNORE INTO ledger(session_id, scope, role, content, message_id, ts) "
                "VALUES(?,?,?,?,?,?)",
                (session_id, scope, role, content, message_id, ts),
            )
            if cur.rowcount and self.fts:
                rid = cur.lastrowid
                self.conn.execute(
                    "INSERT INTO ledger_fts(content, session_id, row_id) VALUES(?,?,?)",
                    (content, session_id, rid),
                )
            self.conn.commit()
            return bool(cur.rowcount)
        except sqlite3.Error as exc:
            # 必须显式回滚：FTS 插入失败时主表插入已挂起在未提交事务里，
            # 会被后续任意一次 commit 顺带提交，造成主表与 FTS 失同步
            try:
                self.conn.rollback()
            except sqlite3.Error:
                pass
            logger.warning("账本写入失败: %s", exc)
            return False

    def recent_ledger(self, session_id: str, limit: int = 40,
                      after_id: int = 0, oldest_first: bool = False) -> list[sqlite3.Row]:
        """取某会话流水（按时间正序返回）。

        after_id: 只取 rowid 大于该值的行——抽取游标，防止同一段对话被
        反复送入抽取 LLM（会造成记忆膨胀，这是审查发现的缺陷）。
        排序用 id 而非 ts：同秒多条消息时 ts DESC 顺序不稳定，LIMIT 截断
        会任意挑选子集；id 单调递增且与写入序一致。
        oldest_first=True 取「游标之后最早的 N 条」（抽取用：配合游标
        逐窗口推进，被截断的剩余消息留给下一轮）；默认取最新的 N 条
        （近因查询用）。两种模式都返回时间正序。
        """
        rows = self.conn.execute(
            "SELECT id, role, content, ts, message_id FROM ledger "
            f"WHERE session_id=? AND id>? ORDER BY id {'ASC' if oldest_first else 'DESC'} LIMIT ?",
            (session_id, int(after_id), int(limit)),
        ).fetchall()
        return rows if oldest_first else rows[::-1]

    def max_ledger_id(self, session_id: str | None = None) -> int:
        if session_id:
            row = self.conn.execute(
                "SELECT MAX(id) AS m FROM ledger WHERE session_id=?", (session_id,)
            ).fetchone()
        else:
            row = self.conn.execute("SELECT MAX(id) AS m FROM ledger").fetchone()
        return int(row["m"] or 0)

    def search_ledger(self, query: str, session_id: str | None = None, limit: int = 20) -> list[sqlite3.Row]:
        if not query.strip():
            return []
        rows: list[sqlite3.Row] = []
        if self.fts:
            try:
                sql = (
                    "SELECT l.role, l.content, l.ts, l.session_id FROM ledger_fts f "
                    "JOIN ledger l ON l.id=f.row_id WHERE ledger_fts MATCH ?"
                )
                params: list = [_fts_query(query)]
                if session_id:
                    sql += " AND l.session_id=?"
                    params.append(session_id)
                sql += " ORDER BY rank LIMIT ?"
                params.append(int(limit))
                rows = self.conn.execute(sql, params).fetchall()
            except sqlite3.OperationalError as exc:
                logger.debug("FTS 查询失败，降级 LIKE: %s", exc)
                rows = []
        if rows:
            return rows
        # 降级/兜底：LIKE（覆盖 trigram 对 <3 字符查询的局限）
        sql = "SELECT role, content, ts, session_id FROM ledger WHERE content LIKE ?"
        params = [f"%{query}%"]
        if session_id:
            sql += " AND session_id=?"
            params.append(session_id)
        sql += " ORDER BY ts DESC LIMIT ?"
        params.append(int(limit))
        return self.conn.execute(sql, params).fetchall()

    def prune_ledger(self, before_ts: float) -> int:
        cur = self.conn.execute("DELETE FROM ledger WHERE ts < ?", (before_ts,))
        self.conn.execute("DELETE FROM ledger_fts WHERE row_id NOT IN (SELECT id FROM ledger)")
        self.conn.commit()
        return cur.rowcount or 0

    # ------------------------------------------------------------------ 画像
    def upsert_profile(self, scope: str, user_key: str, key: str, value: str, confidence: float = 1.0) -> None:
        self.conn.execute(
            "INSERT INTO profiles(scope, user_key, key, value, confidence, updated_at) "
            "VALUES(?,?,?,?,?,?) ON CONFLICT(scope, user_key, key) DO UPDATE SET "
            "value=excluded.value, confidence=excluded.confidence, updated_at=excluded.updated_at",
            (scope, user_key, key, value, float(confidence), utc_now_ts()),
        )
        self.conn.commit()

    def merge_profile(self, scope: str, user_key: str, key: str, value: str,
                      confidence: float = 1.0, max_total: int = 4000,
                      replace_segment: str = "") -> None:
        """合并写入画像（追加去重，不覆盖旧值；v0.2.1）。

        固定维度里的「事实属性/技能树/关系图谱/活跃项目」是累积型维度：
        新事实追加到同一行而不是替换（否则模型每次只写片段就会把旧事实冲掉）。
        用户在面板上的手动编辑仍走 upsert_profile（显式覆盖）。

        replace_segment（v0.2.4，对齐 angel「画像纠正必须 updata 而非并存」）：
        非空时表示本片段是对某条旧片段的更正——先把被推翻的旧片段从该维度
        移除再追加新值。匹配从严：先精确匹配（归一化后），再退到字符级
        相似度 ≥0.70 的最近片段；都不命中则只追加（宁可并存也不误删）。
        """
        now = utc_now_ts()
        new = str(value or "").strip()
        if not new:
            return
        row = self.conn.execute(
            "SELECT value, confidence FROM profiles WHERE scope=? AND user_key=? AND key=?",
            (scope, user_key, key),
        ).fetchone()
        if row is None:
            self.upsert_profile(scope, user_key, key, new, confidence=confidence)
            return
        parts = [p.strip() for p in str(row["value"] or "").split("；") if p.strip()]
        target = str(replace_segment or "").strip()
        removed = ""
        if target:
            removed = self._drop_profile_segment(parts, target)
            if removed:
                logger.info("画像更正：[%s] 移除被推翻的旧片段「%s」", key,
                            removed[:40])
        if new in parts and not removed:
            return  # 已有同款事实：不重复追加（有移除时仍需落库持久化）
        for piece in new.split("；"):
            piece = piece.strip()
            if piece and piece not in parts:
                parts.append(piece)
        # 限长必须按片段边界截断：硬切会把最后一个片段切成半截，
        # 既展示脏数据，又让后续去重判断失准（再审发现）
        cap = max(200, int(max_total))
        kept: list[str] = []
        total = 0
        for piece in parts:
            add = len(piece) + (1 if kept else 0)
            if total + add > cap:
                break
            kept.append(piece)
            total += add
        merged = "；".join(kept)
        conf = max(float(row["confidence"] or 0.0), float(confidence))
        self.conn.execute(
            "UPDATE profiles SET value=?, confidence=?, updated_at=? "
            "WHERE scope=? AND user_key=? AND key=?",
            (merged, conf, now, scope, user_key, key),
        )
        self.conn.commit()

    @staticmethod
    def _drop_profile_segment(parts: list[str], target: str) -> str:
        """从画像片段列表中移除被更正推翻的旧片段，返回被移除的片段（无命中返空串）。

        两级匹配（从严）：1) 归一化精确匹配或互相包含；2) 字符级 Jaccard
        相似度 ≥0.70 的最佳片段（模型照抄时可能截断/微调措辞）。
        只移除一个最可能的片段，绝不批量清除。
        """
        t = target.strip()
        if not t or not parts:
            return ""
        for p in list(parts):
            if p == t or (len(t) >= 6 and (t in p or p in t)):
                parts.remove(p)
                return p
        from .admission import text_similarity
        best_p, best_s = "", 0.0
        for p in parts:
            s = text_similarity(t, p)
            if s > best_s:
                best_p, best_s = p, s
        if best_p and best_s >= 0.70:
            parts.remove(best_p)
            return best_p
        return ""

    def get_profile(self, scope: str, user_key: str) -> list[sqlite3.Row]:
        return self.conn.execute(
            "SELECT key, value, confidence, updated_at FROM profiles "
            "WHERE scope=? AND user_key=? ORDER BY confidence DESC, updated_at DESC",
            (scope, user_key),
        ).fetchall()

    def delete_profile(self, scope: str, user_key: str, key: str) -> None:
        self.conn.execute(
            "DELETE FROM profiles WHERE scope=? AND user_key=? AND key=?",
            (scope, user_key, key),
        )
        self.conn.commit()

    # ------------------------------------------------------------------ 笔记
    def add_note(self, content: str, *, title: str = "", tags: str = "",
                 source: str = "manual", file_name: str = "", heading: str = "",
                 scope: str = "public", vec: list[float] | None = None,
                 note_id: str | None = None) -> str:
        now = utc_now_ts()
        nid = note_id or uuid.uuid4().hex
        blob = None
        dim = 0
        if vec:
            from .vector import pack
            blob = pack(vec)
            dim = len(vec)
        self.conn.execute(
            "INSERT INTO notes(id, title, content, tags, source, file_name, heading, "
            "scope, content_hash, vec, vec_dim, created_at, updated_at) "
            "VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?)",
            (nid, title, content, tags, source, file_name, heading, scope,
             content_hash(content), blob, dim, now, now),
        )
        self.conn.commit()
        return nid

    def get_note(self, note_id: str) -> sqlite3.Row | None:
        return self.conn.execute(
            "SELECT * FROM notes WHERE id=?", (note_id,)
        ).fetchone()

    def update_note(self, note_id: str, **fields: Any) -> None:
        allowed = {"title", "content", "tags", "scope"}
        sets = [f"{k}=?" for k in fields if k in allowed]
        vals = [fields[k] for k in fields if k in allowed]
        if not sets:
            return
        if "content" in fields:
            sets.append("content_hash=?")
            vals.append(content_hash(str(fields["content"])))
            # 内容变了，旧向量作废（与记忆编辑同语义）
            sets.append("vec=NULL")
            sets.append("vec_dim=0")
        sets.append("updated_at=?")
        vals.append(utc_now_ts())
        vals.append(note_id)
        self.conn.execute(f"UPDATE notes SET {', '.join(sets)} WHERE id=?", vals)
        # v0.2.0：内容变更后旧切片派生层立即失效，避免检索到旧正文；
        # 重建由 engine.update_note / _sync_note_chunks 负责，失败自然回退整篇检索
        if "content" in fields:
            self.conn.execute("DELETE FROM note_chunks WHERE note_id=?", (note_id,))
        self.conn.commit()

    def trash_note(self, note_id: str) -> None:
        self.conn.execute("UPDATE notes SET deleted_at=?, updated_at=? WHERE id=?",
                          (utc_now_ts(), utc_now_ts(), note_id))
        self.conn.commit()

    def restore_note(self, note_id: str) -> None:
        self.conn.execute("UPDATE notes SET deleted_at=NULL, updated_at=? WHERE id=?",
                          (utc_now_ts(), note_id))
        self.conn.commit()

    def purge_note(self, note_id: str) -> None:
        # 先删切片（显式 DELETE 会驱动 chunk_fts 清理触发器；不能只靠
        # 外键级联——级联不保证触发触发器，FTS 会残留脏行）
        self.conn.execute("DELETE FROM note_chunks WHERE note_id=?", (note_id,))
        self.conn.execute("DELETE FROM notes WHERE id=?", (note_id,))
        self.conn.commit()

    def list_notes(self, scope: str | None = None, include_deleted: bool = False,
                   limit: int = 200) -> list[sqlite3.Row]:
        # 显式列：绝不能 SELECT * —— vec 是 BLOB(bytes)，进 JSON 序列化会抛
        # TypeError 导致接口 500，且每条数千字节白白撑大响应（审查缺陷）。
        cols = _NOTE_PUBLIC_COLS
        where = [] if include_deleted else ["deleted_at IS NULL"]
        params: list = []
        if scope:
            where.append("scope=?")
            params.append(scope)
        sql = f"SELECT {cols} FROM notes"
        if where:
            sql += " WHERE " + " AND ".join(where)
        sql += " ORDER BY updated_at DESC LIMIT ?"
        params.append(int(limit))
        return self.conn.execute(sql, params).fetchall()

    def search_notes(self, query: str, scope: str | None = None, limit: int = 20) -> list[sqlite3.Row]:
        if not query.strip():
            return []
        cols = _NOTE_PUBLIC_COLS
        rows: list[sqlite3.Row] = []
        if self.fts:
            try:
                terms = [t for t in query.replace('"', " ").split() if t]
                if terms:
                    match = " OR ".join(f'"{t}"' for t in terms)
                    sql = (f"SELECT {_NOTE_PUBLIC_COLS_PREFIXED} FROM notes_fts f "
                           "JOIN notes n ON n.id=f.row_id "
                           "WHERE notes_fts MATCH ? AND n.deleted_at IS NULL")
                    params: list = [match]
                    if scope:
                        sql += " AND n.scope=?"
                        params.append(scope)
                    sql += " ORDER BY rank LIMIT ?"
                    params.append(int(limit))
                    rows = self.conn.execute(sql, params).fetchall()
            except sqlite3.OperationalError as exc:
                logger.debug("笔记 FTS 查询失败，降级 LIKE: %s", exc)
        if rows:
            return rows
        # LIKE 兜底与 FTS 口径对齐：content / title / tags 三列都搜，
        # 否则 FTS 不可用的环境里标题/标签永远搜不到
        pattern = f"%{query}%"
        sql = (
            f"SELECT {cols} FROM notes WHERE deleted_at IS NULL "
            "AND (content LIKE ? OR title LIKE ? OR COALESCE(tags,'') LIKE ?)"
        )
        params = [pattern, pattern, pattern]
        if scope:
            sql += " AND scope=?"
            params.append(scope)
        sql += " ORDER BY updated_at DESC LIMIT ?"
        params.append(int(limit))
        return self.conn.execute(sql, params).fetchall()

    def notes_with_vectors(self, scope: str | None = None) -> list[sqlite3.Row]:
        sql = "SELECT id, content, vec, vec_dim FROM notes WHERE deleted_at IS NULL AND vec IS NOT NULL"
        params: list = []
        if scope:
            sql += " AND scope=?"
            params.append(scope)
        return self.conn.execute(sql, params).fetchall()

    def set_note_vector(self, note_id: str, vec: list[float]) -> None:
        from .vector import pack
        self.conn.execute("UPDATE notes SET vec=?, vec_dim=?, updated_at=? WHERE id=?",
                          (pack(vec), len(vec), utc_now_ts(), note_id))
        self.conn.commit()

    # ------------------------------------------------------------------ 笔记切片（v5）
    def replace_note_chunks(self, note_id: str, chunks: list[dict]) -> None:
        """在一个事务中替换某篇笔记的全部切片；失败自动回滚。"""
        from .vector import pack
        now = utc_now_ts()
        try:
            self.conn.execute("DELETE FROM note_chunks WHERE note_id=?", (note_id,))
            for i, chunk in enumerate(chunks):
                text = str(chunk.get("content") or "").strip()
                if not text:
                    continue
                vec = chunk.get("vec")
                self.conn.execute(
                    "INSERT INTO note_chunks(id,note_id,chunk_index,heading_path,content,"
                    "content_hash,vec,vec_dim,created_at,updated_at) VALUES(?,?,?,?,?,?,?,?,?,?)",
                    (
                        str(chunk.get("id") or f"{note_id}:{i}"), note_id, int(i),
                        str(chunk.get("heading_path") or ""), text, content_hash(text),
                        pack(vec) if vec else None, len(vec) if vec else 0, now, now,
                    ),
                )
            self.conn.commit()
        except sqlite3.Error:
            self.conn.rollback()
            raise

    def list_note_chunks(self, note_id: str) -> list[sqlite3.Row]:
        return self.conn.execute(
            "SELECT id,note_id,chunk_index,heading_path,content,content_hash,vec_dim,"
            "created_at,updated_at FROM note_chunks WHERE note_id=? ORDER BY chunk_index",
            (note_id,),
        ).fetchall()

    def chunks_with_vectors(self, scope: str | None = None,
                            note_limit: int = 1000) -> list[sqlite3.Row]:
        """取有向量的切片（按笔记窗口限量，防大库全表解包）。

        窗口与 notes.retrieve 的基础集合（最近 1000 篇）保持一致：
        窗口外的笔记本就不参与检索，其切片也无需进语义通道。
        """
        sql = (
            "SELECT c.id,c.note_id,c.heading_path,c.content,c.vec,c.vec_dim "
            "FROM note_chunks c JOIN notes n ON n.id=c.note_id "
            "WHERE n.deleted_at IS NULL AND c.vec IS NOT NULL "
            "AND n.id IN (SELECT id FROM notes WHERE deleted_at IS NULL"
        )
        params: list = []
        if scope:
            sql += " AND scope=?"
            params.append(scope)
        sql += " ORDER BY updated_at DESC LIMIT ?)"
        params.append(max(1, int(note_limit)))
        return self.conn.execute(sql, params).fetchall()

    def notes_missing_chunks(self, limit: int = 20) -> list[sqlite3.Row]:
        return self.conn.execute(
            f"SELECT {_NOTE_PUBLIC_COLS_PREFIXED} FROM notes n "
            "WHERE n.deleted_at IS NULL AND NOT EXISTS "
            "(SELECT 1 FROM note_chunks c WHERE c.note_id=n.id) "
            "ORDER BY n.updated_at ASC LIMIT ?",
            (int(limit),),
        ).fetchall()

    # ------------------------------------------------------------------ 身份账本（v5）
    def upsert_user_identity(self, platform: str, user_id: str, name: str = "") -> str:
        """登记稳定身份并累计昵称；返回 canonical key。"""
        p = (platform or "unknown").strip().lower() or "unknown"
        uid = str(user_id or "").strip()
        if not uid:
            return ""
        now = utc_now_ts()
        row = self.conn.execute(
            "SELECT names_json,first_seen_at FROM user_ledger WHERE platform=? AND user_id=?",
            (p, uid),
        ).fetchone()
        names = []
        if row:
            try:
                raw = json.loads(row["names_json"] or "[]")
                if isinstance(raw, list):
                    names = [str(x) for x in raw if str(x).strip()]
            except Exception:  # noqa: BLE001
                names = []
        clean = str(name or "").strip()
        if clean and clean.lower() not in {"unknown", "anonymous", "匿名", "用户"} and clean not in names:
            names.append(clean[:80])
            # 昵称历史只做展示锚点，保留最近 10 个防无界增长
            names = names[-10:]
        self.conn.execute(
            "INSERT INTO user_ledger(platform,user_id,names_json,first_seen_at,last_seen_at) "
            "VALUES(?,?,?,?,?) ON CONFLICT(platform,user_id) DO UPDATE SET "
            "names_json=excluded.names_json,last_seen_at=excluded.last_seen_at",
            (p, uid, json.dumps(names, ensure_ascii=False),
             float(row["first_seen_at"] if row else now), now),
        )
        self.conn.commit()
        return f"{p}:{uid}"

    def upsert_group_identity(self, platform: str, group_id: str, name: str = "") -> str:
        p = (platform or "unknown").strip().lower() or "unknown"
        gid = str(group_id or "").strip()
        if not gid:
            return ""
        now = utc_now_ts()
        row = self.conn.execute(
            "SELECT first_seen_at FROM group_ledger WHERE platform=? AND group_id=?", (p, gid)
        ).fetchone()
        self.conn.execute(
            "INSERT INTO group_ledger(platform,group_id,group_name,first_seen_at,last_seen_at) "
            "VALUES(?,?,?,?,?) ON CONFLICT(platform,group_id) DO UPDATE SET "
            "group_name=CASE WHEN excluded.group_name<>'' THEN excluded.group_name "
            "ELSE group_ledger.group_name END,last_seen_at=excluded.last_seen_at",
            (p, gid, str(name or "").strip()[:80],
             float(row["first_seen_at"] if row else now), now),
        )
        self.conn.commit()
        return f"{p}:{gid}"

    def identity_aliases(self, speaker_key: str) -> list[str]:
        if ":" not in str(speaker_key or ""):
            return []
        platform, uid = speaker_key.split(":", 1)
        row = self.conn.execute(
            "SELECT names_json FROM user_ledger WHERE platform=? AND user_id=?",
            (platform, uid),
        ).fetchone()
        if not row:
            return []
        try:
            names = json.loads(row["names_json"] or "[]")
            return [str(x) for x in names] if isinstance(names, list) else []
        except Exception:  # noqa: BLE001
            return []

    def list_identities(self, limit: int = 200) -> list[sqlite3.Row]:
        return self.conn.execute(
            "SELECT platform,user_id,names_json,first_seen_at,last_seen_at "
            "FROM user_ledger ORDER BY last_seen_at DESC LIMIT ?", (int(limit),)
        ).fetchall()


def _fts_query(raw: str) -> str:
    """把用户查询转成 FTS5 安全查询串：每个词加引号用 OR 连接。"""
    terms = [t for t in raw.replace('"', " ").split() if t]
    if not terms:
        return '""'
    return " OR ".join(f'"{t}"' for t in terms)
