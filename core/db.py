"""SQLite 存储层：连接、schema 与版本迁移。

设计要点：
- 单文件 SQLite + WAL，落插件数据目录。
- 记忆表含双时态（observed_at/valid_from/valid_to）、增量信念（proof_count）、
  软删除（deleted_at + superseded_by，可复活）、召回热度（hit_count/last_recalled_at）。
- 隔离域 scope 落在 schema 层（不靠查询侧过滤——graphiti 的教训）。
- 账本用 FTS5 做关键词检索；FTS5 不可用时自动降级为 LIKE。
"""

from __future__ import annotations

from astrbot.api import logger
import sqlite3
from pathlib import Path


SCHEMA_VERSION = 6


def connect(db_path: Path | str) -> sqlite3.Connection:
    """打开连接并设置运行参数。"""
    conn = sqlite3.connect(str(db_path), timeout=30.0, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
    conn.execute("PRAGMA busy_timeout=30000")
    conn.execute("PRAGMA synchronous=NORMAL")
    return conn


_CORE_TABLES = """
CREATE TABLE IF NOT EXISTS meta (
    key   TEXT PRIMARY KEY,
    value TEXT
);

CREATE TABLE IF NOT EXISTS memories (
    id               TEXT PRIMARY KEY,
    content          TEXT NOT NULL,
    reasoning        TEXT DEFAULT '',
    memory_type      TEXT DEFAULT 'fact',      -- fact/knowledge/event/skill/emotional/task
    source           TEXT DEFAULT 'user',      -- user/assistant/tool/system（防回声过滤依据）
    speaker          TEXT DEFAULT '',
    speaker_key      TEXT DEFAULT '',          -- 稳定身份键 platform:user_id（v5）
    is_active        INTEGER DEFAULT 0,        -- 1=主动记忆，永不衰减
    strength         REAL    DEFAULT 10.0,
    useful_score     REAL    DEFAULT 0.0,
    useful_count     INTEGER DEFAULT 0,
    hit_count        INTEGER DEFAULT 0,        -- 被召回次数（热度）
    last_recalled_at REAL    DEFAULT 0,
    last_decay_at    REAL    DEFAULT 0,
    proof_count      INTEGER DEFAULT 1,        -- 增量信念：支持该记忆的证据条数
    observed_at      REAL    DEFAULT 0,        -- 首次观察到的时间
    valid_from       REAL    DEFAULT 0,        -- 双时态：事实生效时间
    valid_to         REAL,                     -- 双时态：事实失效时间（NULL=仍有效）
    superseded_by    TEXT,                     -- 被哪条新记忆取代（软链，不物理覆盖）
    deleted_at       REAL,                     -- 回收站时间（NULL=未删除）
    quarantined      INTEGER DEFAULT 0,        -- 隔离区标记（1=疑似敏感被隔离待审，memoripy 同款）
    tags_json        TEXT DEFAULT '[]',        -- 检索锚点标签（v0.1.8，angel tags 同思想；只进检索通道不进嵌入）
    scope            TEXT    DEFAULT 'public', -- 隔离域
    session_id       TEXT    DEFAULT '',
    content_hash     TEXT    DEFAULT '',
    created_at       REAL    DEFAULT 0,
    updated_at       REAL    DEFAULT 0
);
CREATE INDEX IF NOT EXISTS idx_mem_scope     ON memories(scope, is_active);
CREATE INDEX IF NOT EXISTS idx_mem_hash      ON memories(content_hash);
CREATE INDEX IF NOT EXISTS idx_mem_deleted   ON memories(deleted_at);
CREATE INDEX IF NOT EXISTS idx_mem_validto   ON memories(valid_to);

CREATE TABLE IF NOT EXISTS vectors (
    memory_id TEXT PRIMARY KEY,
    dim       INTEGER NOT NULL,
    vec       BLOB    NOT NULL,               -- float32 小端打包
    FOREIGN KEY (memory_id) REFERENCES memories(id) ON DELETE CASCADE
);

-- v6：嵌入器换型/混部时的惰性回填队列。只记录待回填 id，
-- 具体嵌入在后台任务执行，绝不阻塞消息写入或检索。
CREATE TABLE IF NOT EXISTS vector_backlog (
    memory_id  TEXT PRIMARY KEY,
    dim_seen   INTEGER NOT NULL DEFAULT 0,
    enqueued_at REAL NOT NULL,
    attempts   INTEGER NOT NULL DEFAULT 0,
    FOREIGN KEY (memory_id) REFERENCES memories(id) ON DELETE CASCADE
);
CREATE INDEX IF NOT EXISTS idx_vector_backlog_order
    ON vector_backlog(enqueued_at, attempts);

CREATE TABLE IF NOT EXISTS profiles (
    scope      TEXT NOT NULL,
    user_key   TEXT NOT NULL,
    key        TEXT NOT NULL,
    value      TEXT NOT NULL,
    confidence REAL DEFAULT 1.0,
    updated_at REAL DEFAULT 0,
    PRIMARY KEY (scope, user_key, key)
);

CREATE TABLE IF NOT EXISTS ledger (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id TEXT NOT NULL,
    scope      TEXT NOT NULL DEFAULT 'public',
    role       TEXT NOT NULL,
    content    TEXT NOT NULL,
    message_id TEXT,
    ts         REAL NOT NULL,
    UNIQUE(message_id)
);
CREATE INDEX IF NOT EXISTS idx_ledger_session ON ledger(session_id, ts);
CREATE INDEX IF NOT EXISTS idx_ledger_ts      ON ledger(ts);

-- 笔记知识库（v3）：与「记忆」分开管理。记忆是抽取出的事实，笔记是用户/AI
-- 主动整理的知识条目（可含 .md 文档分块），参与检索与注入但走独立表。
CREATE TABLE IF NOT EXISTS notes (
    id           TEXT PRIMARY KEY,
    title        TEXT DEFAULT '',
    content      TEXT NOT NULL,
    tags         TEXT DEFAULT '',          -- 逗号分隔
    source       TEXT DEFAULT 'manual',    -- manual / ai / file
    file_name    TEXT DEFAULT '',          -- .md 导入时的来源文件名
    heading      TEXT DEFAULT '',          -- .md 导入时的标题层级路径
    scope        TEXT DEFAULT 'public',
    content_hash TEXT DEFAULT '',
    vec          BLOB,                     -- float32 打包（可空：未回填向量的笔记）
    vec_dim      INTEGER DEFAULT 0,
    deleted_at   REAL,                     -- 软删（回收站）
    created_at   REAL DEFAULT 0,
    updated_at   REAL DEFAULT 0
);
CREATE INDEX IF NOT EXISTS idx_note_scope   ON notes(scope);
CREATE INDEX IF NOT EXISTS idx_note_deleted ON notes(deleted_at);
CREATE INDEX IF NOT EXISTS idx_note_file    ON notes(file_name);

-- v5：笔记正文切片。notes 仍是真相源，本表是可重建派生层。
CREATE TABLE IF NOT EXISTS note_chunks (
    id           TEXT PRIMARY KEY,
    note_id      TEXT NOT NULL,
    chunk_index  INTEGER NOT NULL,
    heading_path TEXT DEFAULT '',
    content      TEXT NOT NULL,
    content_hash TEXT DEFAULT '',
    vec          BLOB,
    vec_dim      INTEGER DEFAULT 0,
    created_at   REAL DEFAULT 0,
    updated_at   REAL DEFAULT 0,
    UNIQUE(note_id, chunk_index),
    FOREIGN KEY (note_id) REFERENCES notes(id) ON DELETE CASCADE
);
CREATE INDEX IF NOT EXISTS idx_chunk_note ON note_chunks(note_id, chunk_index);

-- v5：记忆动作审计与血缘。即使源记忆最终物理清理，事件仍保留。
CREATE TABLE IF NOT EXISTS memory_events (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    action          TEXT NOT NULL,
    scope           TEXT NOT NULL DEFAULT 'public',
    source_ids_json TEXT NOT NULL DEFAULT '[]',
    target_id       TEXT DEFAULT '',
    reason          TEXT DEFAULT '',
    confidence      REAL DEFAULT 0,
    provider        TEXT DEFAULT '',
    created_at      REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_event_scope_time ON memory_events(scope, created_at);
CREATE INDEX IF NOT EXISTS idx_event_target ON memory_events(target_id);

-- v5：稳定身份账本。昵称只是展示信息，(platform,id) 才是身份锚点。
CREATE TABLE IF NOT EXISTS user_ledger (
    platform      TEXT NOT NULL,
    user_id       TEXT NOT NULL,
    names_json    TEXT NOT NULL DEFAULT '[]',
    first_seen_at REAL NOT NULL,
    last_seen_at  REAL NOT NULL,
    PRIMARY KEY(platform, user_id)
);
CREATE TABLE IF NOT EXISTS group_ledger (
    platform      TEXT NOT NULL,
    group_id      TEXT NOT NULL,
    group_name    TEXT DEFAULT '',
    first_seen_at REAL NOT NULL,
    last_seen_at  REAL NOT NULL,
    PRIMARY KEY(platform, group_id)
);
"""

_FTS_TABLES = """
CREATE VIRTUAL TABLE IF NOT EXISTS ledger_fts USING fts5(
    content, session_id UNINDEXED, row_id UNINDEXED, tokenize='trigram'
);

CREATE VIRTUAL TABLE IF NOT EXISTS memories_fts USING fts5(
    content, row_id UNINDEXED, tokenize='trigram'
);
CREATE TRIGGER IF NOT EXISTS mem_fts_ai AFTER INSERT ON memories BEGIN
    INSERT INTO memories_fts(content, row_id) VALUES (new.content, new.id);
END;
CREATE TRIGGER IF NOT EXISTS mem_fts_au AFTER UPDATE OF content ON memories BEGIN
    UPDATE memories_fts SET content=new.content WHERE row_id=new.id;
END;
CREATE TRIGGER IF NOT EXISTS mem_fts_ad AFTER DELETE ON memories BEGIN
    DELETE FROM memories_fts WHERE row_id=old.id;
END;

CREATE VIRTUAL TABLE IF NOT EXISTS notes_fts USING fts5(
    content, title, tags, row_id UNINDEXED, tokenize='trigram'
);
CREATE TRIGGER IF NOT EXISTS note_fts_ai AFTER INSERT ON notes BEGIN
    INSERT INTO notes_fts(content, title, tags, row_id)
    VALUES (new.content, COALESCE(new.title,''), COALESCE(new.tags,''), new.id);
END;
CREATE TRIGGER IF NOT EXISTS note_fts_au AFTER UPDATE OF content, title, tags ON notes BEGIN
    UPDATE notes_fts SET content=new.content, title=COALESCE(new.title,''),
        tags=COALESCE(new.tags,'') WHERE row_id=new.id;
END;
CREATE TRIGGER IF NOT EXISTS note_fts_ad AFTER DELETE ON notes BEGIN
    DELETE FROM notes_fts WHERE row_id=old.id;
END;

CREATE VIRTUAL TABLE IF NOT EXISTS note_chunks_fts USING fts5(
    content, heading_path, row_id UNINDEXED, tokenize='trigram'
);
CREATE TRIGGER IF NOT EXISTS chunk_fts_ai AFTER INSERT ON note_chunks BEGIN
    INSERT INTO note_chunks_fts(content, heading_path, row_id)
    VALUES (new.content, COALESCE(new.heading_path,''), new.id);
END;
CREATE TRIGGER IF NOT EXISTS chunk_fts_au AFTER UPDATE OF content, heading_path ON note_chunks BEGIN
    UPDATE note_chunks_fts SET content=new.content,
        heading_path=COALESCE(new.heading_path,'') WHERE row_id=new.id;
END;
CREATE TRIGGER IF NOT EXISTS chunk_fts_ad AFTER DELETE ON note_chunks BEGIN
    DELETE FROM note_chunks_fts WHERE row_id=old.id;
END;
"""

# 不支持 trigram（老 SQLite）时的降级建表
_FTS_TABLES_PLAIN = """
CREATE VIRTUAL TABLE IF NOT EXISTS ledger_fts USING fts5(
    content, session_id UNINDEXED, row_id UNINDEXED
);
CREATE VIRTUAL TABLE IF NOT EXISTS memories_fts USING fts5(
    content, row_id UNINDEXED
);
CREATE TRIGGER IF NOT EXISTS mem_fts_ai AFTER INSERT ON memories BEGIN
    INSERT INTO memories_fts(content, row_id) VALUES (new.content, new.id);
END;
CREATE TRIGGER IF NOT EXISTS mem_fts_au AFTER UPDATE OF content ON memories BEGIN
    UPDATE memories_fts SET content=new.content WHERE row_id=new.id;
END;
CREATE TRIGGER IF NOT EXISTS mem_fts_ad AFTER DELETE ON memories BEGIN
    DELETE FROM memories_fts WHERE row_id=old.id;
END;

CREATE VIRTUAL TABLE IF NOT EXISTS notes_fts USING fts5(
    content, title, tags, row_id UNINDEXED
);
CREATE TRIGGER IF NOT EXISTS note_fts_ai AFTER INSERT ON notes BEGIN
    INSERT INTO notes_fts(content, title, tags, row_id)
    VALUES (new.content, COALESCE(new.title,''), COALESCE(new.tags,''), new.id);
END;
CREATE TRIGGER IF NOT EXISTS note_fts_au AFTER UPDATE OF content, title, tags ON notes BEGIN
    UPDATE notes_fts SET content=new.content, title=COALESCE(new.title,''),
        tags=COALESCE(new.tags,'') WHERE row_id=new.id;
END;
CREATE TRIGGER IF NOT EXISTS note_fts_ad AFTER DELETE ON notes BEGIN
    DELETE FROM notes_fts WHERE row_id=old.id;
END;

CREATE VIRTUAL TABLE IF NOT EXISTS note_chunks_fts USING fts5(
    content, heading_path, row_id UNINDEXED
);
CREATE TRIGGER IF NOT EXISTS chunk_fts_ai AFTER INSERT ON note_chunks BEGIN
    INSERT INTO note_chunks_fts(content, heading_path, row_id)
    VALUES (new.content, COALESCE(new.heading_path,''), new.id);
END;
CREATE TRIGGER IF NOT EXISTS chunk_fts_au AFTER UPDATE OF content, heading_path ON note_chunks BEGIN
    UPDATE note_chunks_fts SET content=new.content,
        heading_path=COALESCE(new.heading_path,'') WHERE row_id=new.id;
END;
CREATE TRIGGER IF NOT EXISTS chunk_fts_ad AFTER DELETE ON note_chunks BEGIN
    DELETE FROM note_chunks_fts WHERE row_id=old.id;
END;
"""


def current_schema_version(conn: sqlite3.Connection) -> int:
    """安全读取当前 schema 版本；空库/旧库返回 0。"""
    try:
        row = conn.execute("SELECT value FROM meta WHERE key='schema_version'").fetchone()
        return int(row[0]) if row else 0
    except (sqlite3.Error, TypeError, ValueError):
        return 0


def has_existing_schema(conn: sqlite3.Connection) -> bool:
    """判断数据库中是否已经存在需要升级保护的业务表。"""
    try:
        return conn.execute(
            "SELECT 1 FROM sqlite_master WHERE type='table' AND name='memories' LIMIT 1"
        ).fetchone() is not None
    except sqlite3.Error:
        return False


def backup_before_migration(
    conn: sqlite3.Connection,
    target: Path | str,
    target_version: int = SCHEMA_VERSION,
) -> Path | None:
    """升级前用 SQLite backup API 生成一致性快照。

    新空库和已是目标版本的库不备份。调用方在旧库备份失败时应停止升级，
    不能冒险直接修改用户数据。
    """
    old = current_schema_version(conn)
    try:
        has_tables = conn.execute(
            "SELECT 1 FROM sqlite_master WHERE type='table' "
            "AND name NOT LIKE 'sqlite_%' LIMIT 1"
        ).fetchone() is not None
    except sqlite3.Error:
        has_tables = False
    if not has_tables or old >= int(target_version):
        return None
    path = Path(target)
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        dst = sqlite3.connect(str(path))
        try:
            conn.backup(dst)
            dst.commit()
        finally:
            dst.close()
        return path
    except (OSError, sqlite3.Error) as exc:
        logger.warning("schema v%d 升级前数据库备份失败: %s", target_version, exc)
        return None


def init_schema(conn: sqlite3.Connection) -> bool:
    """建表（幂等）。返回 FTS5 是否可用。

    优先用 trigram 分词器（支持中文子串匹配），不支持时退回默认分词器。
    v2 迁移：旧库补 quarantined 列（隔离区，memoripy 同款机制）。
    v6 迁移：创建 vector_backlog 惰性回填队列（由 _CORE_TABLES 幂等建表）。
    """
    conn.executescript(_CORE_TABLES)
    # v1→v2 迁移：老库没有 quarantined 列，ALTER 补上（新库 CREATE 已含，会跳过）
    cols = [r[1] for r in conn.execute("PRAGMA table_info(memories)").fetchall()]
    if "quarantined" not in cols:
        conn.execute("ALTER TABLE memories ADD COLUMN quarantined INTEGER DEFAULT 0")
        logger.info("schema v2 迁移：memories 表已补 quarantined 列")
    # v3→v4 迁移：tags_json 列（检索锚点标签）
    if "tags_json" not in cols:
        conn.execute("ALTER TABLE memories ADD COLUMN tags_json TEXT DEFAULT '[]'")
        logger.info("schema v4 迁移：memories 表已补 tags_json 列")
    # v4→v5：稳定身份键。只加空列，不猜测/重写旧 speaker。
    if "speaker_key" not in cols:
        conn.execute("ALTER TABLE memories ADD COLUMN speaker_key TEXT DEFAULT ''")
        logger.info("schema v5 迁移：memories 表已补 speaker_key 列")
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_mem_speaker ON memories(scope, speaker_key)"
    )
    fts_ok = False
    for script in (_FTS_TABLES, _FTS_TABLES_PLAIN):
        try:
            conn.executescript(script)
            fts_ok = True
            break
        except sqlite3.OperationalError as exc:
            logger.warning("FTS5 建表失败，尝试降级: %s", exc)
    if not fts_ok:
        logger.warning("FTS5 不可用，检索降级为 LIKE 匹配")
    if fts_ok:
        _upgrade_notes_fts(conn)
    _set_meta(conn, "schema_version", str(SCHEMA_VERSION))
    _set_meta(conn, "fts5", "1" if fts_ok else "0")
    conn.commit()
    return fts_ok


_NEW_TRIGGERS = """
CREATE TRIGGER IF NOT EXISTS note_fts_ai AFTER INSERT ON notes BEGIN
    INSERT INTO notes_fts(content, title, tags, row_id)
    VALUES (new.content, COALESCE(new.title,''), COALESCE(new.tags,''), new.id);
END;
CREATE TRIGGER IF NOT EXISTS note_fts_au AFTER UPDATE OF content, title, tags ON notes BEGIN
    UPDATE notes_fts SET content=new.content, title=COALESCE(new.title,''),
        tags=COALESCE(new.tags,'') WHERE row_id=new.id;
END;
CREATE TRIGGER IF NOT EXISTS note_fts_ad AFTER DELETE ON notes BEGIN
    DELETE FROM notes_fts WHERE row_id=old.id;
END;
"""

# 恢复用：旧列集的触发器（内容级同步）。仅当表仍是旧结构时使用。
_LEGACY_TRIGGERS = """
CREATE TRIGGER IF NOT EXISTS note_fts_ai AFTER INSERT ON notes BEGIN
    INSERT INTO notes_fts(content, row_id) VALUES (new.content, new.id);
END;
CREATE TRIGGER IF NOT EXISTS note_fts_au AFTER UPDATE OF content ON notes BEGIN
    UPDATE notes_fts SET content=new.content WHERE row_id=new.id;
END;
CREATE TRIGGER IF NOT EXISTS note_fts_ad AFTER DELETE ON notes BEGIN
    DELETE FROM notes_fts WHERE row_id=old.id;
END;
"""

_TRIGGER_NAMES = ("note_fts_ai", "note_fts_au", "note_fts_ad")


def _ensure_note_triggers(conn: sqlite3.Connection) -> bool:
    """按 notes_fts 当前列集补建触发器（新列集→新触发器，旧列集→旧触发器）。

    返回是否成功。用于两处：迁移成功后的正常收尾，以及切换中途出错时
    的补救——触发器挂在 notes 表上，DROP notes_fts 不会连带删它们，
    一旦丢失或悬挂就会静默停摆（新笔记不进索引/开火报 no such table）。
    """
    try:
        cols = [r[1] for r in conn.execute("PRAGMA table_info(notes_fts)").fetchall()]
        if not cols:
            return False
        script = _NEW_TRIGGERS if "title" in cols else _LEGACY_TRIGGERS
        conn.executescript(script)
        conn.commit()
        return True
    except sqlite3.Error as exc:
        logger.warning("笔记 FTS 触发器补建失败: %s", exc)
        return False


def _note_triggers_ok(conn: sqlite3.Connection) -> bool:
    """触发器三项齐全且为主索引触发器的新列集版本。"""
    try:
        rows = conn.execute(
            "SELECT name, sql FROM sqlite_master WHERE type='trigger' "
            "AND name IN ('note_fts_ai','note_fts_au','note_fts_ad')"
        ).fetchall()
    except sqlite3.Error:
        return True  # 探测失败不触发重建（保守）
    got = {r[0] for r in rows}
    if got != set(_TRIGGER_NAMES):
        return False
    ai = next((r[1] or "" for r in rows if r[0] == "note_fts_ai"), "")
    return "title" in ai


def _upgrade_notes_fts(conn: sqlite3.Connection) -> None:
    """notes_fts 扩列迁移（v0.1.1 引入；v0.1.3 修正失败路径与中断恢复）。

    FTS5 虚拟表不支持 ALTER ADD COLUMN，需要整表重建。流程（notes 是
    真相源，灌数据不依赖旧 FTS）：
      1. 建 notes_fts_new 副本并灌好、commit——此步失败旧表旧触发器全在；
      2. 才摘旧触发器、DROP 旧表、RENAME 切换（FTS5 影子表随更名）；
      3. 按新列集重建触发器。
    ⚠️ v0.1.2 的教训：触发器必须放到第 2 步再摘——它挂在 notes 表上，
    DROP notes_fts 不会连带删除；先前在开头就摘除，灌数据一失败就落得
    「旧表在、触发器没了」，新笔记静默不进索引（已实验复现）。
    守卫四类现场，均自动修复：
      - 旧结构（缺 title/tags 列）；
      - 结构正确但索引为空而笔记非空（灌数据后中断的残留）；
      - 触发器缺失（摘除后中断/失败路径的另一版本）；
      - 触发器仍是旧列集（曾与旧表配套）。
    """
    try:
        cols = [r[1] for r in conn.execute("PRAGMA table_info(notes_fts)").fetchall()]
        fts_rows = conn.execute("SELECT COUNT(*) FROM notes_fts").fetchone()[0] if cols else 0
        note_rows = conn.execute("SELECT COUNT(*) FROM notes").fetchone()[0]
    except sqlite3.Error:
        return
    trig_ok = _note_triggers_ok(conn)
    needs = (
        not cols
        or "title" not in cols
        or not trig_ok
        or (fts_rows == 0 and note_rows > 0)
    )
    if not needs:
        return
    reasons = []
    if not cols:
        reasons.append("索引表不存在")
    elif "title" not in cols:
        reasons.append("旧结构缺 title/tags 列")
    if fts_rows == 0 and note_rows > 0:
        reasons.append("索引为空而笔记表非空（中断残留）")
    if not trig_ok:
        reasons.append("触发器缺失或为旧列集")
    reason = "；".join(reasons) or "需要重建"

    # ---- 第 1 步：备好新表并灌满（此步失败不动旧表旧触发器） ----
    conn.execute("DROP TABLE IF EXISTS notes_fts_new")
    built = False
    for ddl in (
        "CREATE VIRTUAL TABLE notes_fts_new USING fts5("
        "content, title, tags, row_id UNINDEXED, tokenize='trigram')",
        "CREATE VIRTUAL TABLE notes_fts_new USING fts5("
        "content, title, tags, row_id UNINDEXED)",
    ):
        try:
            conn.execute(ddl)
            built = True
            break
        except sqlite3.OperationalError:
            continue
    if not built:
        logger.warning("notes_fts 迁移失败（%s）：新表创建不可用，"
                       "旧索引与触发器保持原状，笔记检索走 LIKE 兜底", reason)
        return
    try:
        conn.execute(
            "INSERT INTO notes_fts_new(row_id, content, title, tags) "
            "SELECT id, content, COALESCE(title,''), COALESCE(tags,'') FROM notes"
        )
        conn.commit()
    except sqlite3.Error as exc:
        conn.rollback()
        conn.execute("DROP TABLE IF EXISTS notes_fts_new")
        logger.warning("notes_fts 迁移失败（%s）：新索引灌数据失败，"
                       "旧索引与触发器保持原状，下次启动自动重试: %s", reason, exc)
        return

    # ---- 第 2 步：摘触发器并切换（快速局部操作；出错尽力恢复） ----
    try:
        for trig in _TRIGGER_NAMES:
            conn.execute(f"DROP TRIGGER IF EXISTS {trig}")
        conn.execute("DROP TABLE IF EXISTS notes_fts")
        conn.execute("ALTER TABLE notes_fts_new RENAME TO notes_fts")
        conn.commit()
    except sqlite3.Error as exc:
        conn.rollback()
        try:
            has_old = conn.execute(
                "SELECT 1 FROM sqlite_master WHERE name='notes_fts'"
            ).fetchone() is not None
            has_new = conn.execute(
                "SELECT 1 FROM sqlite_master WHERE name='notes_fts_new'"
            ).fetchone() is not None
            if not has_old and has_new:
                # 旧表已摘、更名未成：把灌好的副本直接顶上去（等价重试切换）
                conn.execute("ALTER TABLE notes_fts_new RENAME TO notes_fts")
                conn.commit()
                has_old, has_new = True, False
            if has_new:
                conn.execute("DROP TABLE IF EXISTS notes_fts_new")
            if has_old:
                _ensure_note_triggers(conn)
        except sqlite3.Error:
            pass
        logger.warning("notes_fts 迁移失败（%s）：切换中途出错，已尽力恢复"
                       "（触发器按现存表列集补回）；笔记检索走 LIKE 兜底，"
                       "下次启动自动重试: %s", reason, exc)
        return

    # ---- 第 3 步：按新列集重建触发器 ----
    if _ensure_note_triggers(conn):
        logger.info("notes_fts 迁移完成（%s）", reason)
    else:
        logger.warning("notes_fts 已切换但触发器重建失败（%s）："
                       "新增/修改的笔记暂不进索引，下次启动自动修复", reason)


def get_meta(conn: sqlite3.Connection, key: str, default: str | None = None) -> str | None:
    row = conn.execute("SELECT value FROM meta WHERE key=?", (key,)).fetchone()
    return row["value"] if row else default


def set_meta(conn: sqlite3.Connection, key: str, value: str) -> None:
    """写入 meta 键值（公开版本，供「上次衰减/巩固日期」等跨重启状态使用）。"""
    _set_meta(conn, key, value)
    conn.commit()


def _set_meta(conn: sqlite3.Connection, key: str, value: str) -> None:
    conn.execute(
        "INSERT INTO meta(key, value) VALUES(?, ?) "
        "ON CONFLICT(key) DO UPDATE SET value=excluded.value",
        (key, value),
    )


def fts_available(conn: sqlite3.Connection) -> bool:
    return get_meta(conn, "fts5", "0") == "1"
