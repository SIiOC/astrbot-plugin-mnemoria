"""笔记知识库：Markdown 分块、检索与注入组装。

与「记忆」分开：记忆是从对话抽取出的事实，笔记是用户/AI 主动整理的知识条目
（可来自 .md 文档导入）。两者共用向量与 FTS 工具链，但独立成表、独立检索。

分块策略（.md 导入）：按 Markdown 标题（# / ## / ###）切分，每个标题下的段落
作为一条笔记；标题作为 title，标题层级路径作为 heading（如「设定 > 主角」），
便于检索时定位来源。超长段落按 chunk_max_chars 二次切分。
"""

from __future__ import annotations

import re
import sqlite3

from .text import normalize
from .vector import cosine

_HEADING_RE = re.compile(r"^(#{1,6})\s+(.*)$")


def parse_markdown(text: str, *, file_name: str = "", chunk_max_chars: int = 1200,
                   min_chars: int = 6) -> list[dict]:
    """把 Markdown 文本切成笔记块。

    返回 [{"title","content","heading","file_name"}, ...]。
    - 以标题为界分节；节内文本作为一条笔记。
    - 无标题的文档：整篇（或按段落聚合到 chunk_max_chars）作为一条。
    - 过短（< min_chars）的节丢弃，避免导入一堆空标题。
    """
    lines = (text or "").replace("\r\n", "\n").replace("\r", "\n").split("\n")
    chunks: list[dict] = []
    stack: list[tuple[int, str]] = []       # (level, title) 层级栈
    cur_title = ""
    cur_heading = file_name
    buf: list[str] = []

    def flush():
        body = normalize("\n".join(buf))
        if len(body) >= min_chars:
            for piece in _split_long(body, chunk_max_chars):
                chunks.append({
                    "title": cur_title or (file_name or "未命名"),
                    "content": piece,
                    "heading": cur_heading,
                    "file_name": file_name,
                })

    for line in lines:
        m = _HEADING_RE.match(line.strip())
        if m:
            flush()
            buf = []
            level = len(m.group(1))
            title = normalize(m.group(2))
            while stack and stack[-1][0] >= level:
                stack.pop()
            stack.append((level, title))
            cur_title = title
            cur_heading = " > ".join(t for _, t in stack) if stack else title
        else:
            buf.append(line)
    flush()
    return chunks


def _split_long(body: str, max_chars: int) -> list[str]:
    """超长正文按段落聚合切分，尽量不切断句子。"""
    if len(body) <= max_chars:
        return [body]
    out: list[str] = []
    cur = ""
    for para in body.split("\n"):
        if len(cur) + len(para) + 1 > max_chars and cur:
            out.append(cur.strip())
            cur = para
        else:
            cur = (cur + "\n" + para) if cur else para
    if cur.strip():
        out.append(cur.strip())
    # 单段仍超长时硬切
    final: list[str] = []
    for piece in out:
        if len(piece) <= max_chars:
            final.append(piece)
        else:
            for i in range(0, len(piece), max_chars):
                final.append(piece[i:i + max_chars])
    return final


def split_content(text: str, *, max_chars: int = 700, overlap: int = 100,
                  max_pieces: int = 8) -> list[str]:
    """把笔记正文切成带重叠的片段（v0.2.0 切片派生层）。

    - 不超上限的正文整篇一片（调用方复用整篇向量，免重复嵌入）；
    - 超长正文滑动窗口切分，窗口间保留 overlap 字符重叠；
    - 优先在换行/句号收尾，避免把句子拦腰截断；
    - 片段数超 max_pieces 时合并尾部（文本不丢，控制嵌入成本）。
    """
    body = normalize(text or "")
    if not body:
        return []
    limit = max(120, int(max_chars))
    ov = max(0, min(int(overlap), limit // 2))
    cap = max(1, int(max_pieces))
    if len(body) <= limit:
        return [body]
    out: list[str] = []
    start = 0
    total = len(body)
    while start < total:
        end = min(total, start + limit)
        if end < total:
            window_start = start + int(limit * 0.6)
            cut = -1
            for sep in ("\n", "。", "！", "？", "；", ". "):
                pos = body.rfind(sep, window_start, end)
                if pos > cut:
                    cut = pos + len(sep)
            if cut > start:
                end = cut
        piece = body[start:end].strip()
        if piece:
            out.append(piece)
        if end >= total:
            break
        start = max(start + 1, end - ov)
    if len(out) > cap:
        out = out[:cap - 1] + ["\n".join(out[cap - 1:])]
    return out


def _notes_with_chunks(conn, scope: str | None) -> set[str]:
    """已建立切片的笔记 id 集合（含暂无向量的切片）。"""
    sql = ("SELECT DISTINCT c.note_id AS id FROM note_chunks c "
           "JOIN notes n ON n.id=c.note_id WHERE n.deleted_at IS NULL")
    params: list = []
    if scope:
        sql += " AND n.scope=?"
        params.append(scope)
    return {r["id"] for r in conn.execute(sql, params).fetchall()}


def chunk_lexical_ranks(conn, query: str, scope: str | None, top_n: int, fts_ok: bool):
    """切片关键词通道：FTS5 切片索引优先，失败/空则 LIKE 兜底。

    返回 [(note_id, chunk_id, rank)]。
    """
    rows = []
    if fts_ok:
        try:
            terms = [t for t in query.replace('"', " ").split() if t]
            if terms:
                match = " OR ".join(f'"{t}"' for t in terms)
                sql = ("SELECT c.id AS cid, c.note_id AS nid FROM note_chunks_fts f "
                       "JOIN note_chunks c ON c.id=f.row_id "
                       "JOIN notes n ON n.id=c.note_id "
                       "WHERE note_chunks_fts MATCH ? AND n.deleted_at IS NULL")
                params: list = [match]
                if scope:
                    sql += " AND n.scope=?"
                    params.append(scope)
                sql += " ORDER BY rank LIMIT ?"
                params.append(int(top_n))
                rows = conn.execute(sql, params).fetchall()
        except sqlite3.OperationalError:
            rows = []
    if rows:
        return [(r["nid"], r["cid"], i + 1) for i, r in enumerate(rows)]
    pattern = f"%{query}%"
    sql = ("SELECT c.id AS cid, c.note_id AS nid FROM note_chunks c "
           "JOIN notes n ON n.id=c.note_id "
           "WHERE n.deleted_at IS NULL AND c.content LIKE ?")
    params = [pattern]
    if scope:
        sql += " AND n.scope=?"
        params.append(scope)
    sql += " ORDER BY c.updated_at DESC LIMIT ?"
    params.append(int(top_n))
    rows = conn.execute(sql, params).fetchall()
    return [(r["nid"], r["cid"], i + 1) for i, r in enumerate(rows)]


def chunk_semantic_ranks(query_vec, rows, top_n: int):
    """切片语义通道：对已加载的切片向量排序，返回 [(note_id, chunk_id, rank)]。"""
    from .vector import unpack
    scored = []
    for r in rows:
        v = unpack(r["vec"], r["vec_dim"])
        if not v or len(v) != len(query_vec):
            continue
        scored.append((r["id"], r["note_id"], cosine(query_vec, v)))
    scored.sort(key=lambda x: x[2], reverse=True)
    return [(nid, cid, i + 1) for i, (cid, nid, _) in enumerate(scored[:top_n])]


def lexical_ranks(conn, query: str, scope: str | None, top_n: int, fts_ok: bool):
    """笔记关键词通道：FTS5(BM25) 优先，失败或空则 LIKE 兜底。"""
    import sqlite3
    rows = []
    if fts_ok:
        try:
            terms = [t for t in query.replace('"', " ").split() if t]
            if terms:
                match = " OR ".join(f'"{t}"' for t in terms)
                sql = ("SELECT n.id AS id FROM notes_fts f JOIN notes n ON n.id=f.row_id "
                       "WHERE notes_fts MATCH ? AND n.deleted_at IS NULL")
                params: list = [match]
                if scope:
                    sql += " AND n.scope=?"
                    params.append(scope)
                sql += " ORDER BY rank LIMIT ?"
                params.append(int(top_n))
                rows = conn.execute(sql, params).fetchall()
        except sqlite3.OperationalError:
            rows = []
    if rows:
        return [(r["id"], i + 1) for i, r in enumerate(rows)]
    # v0.1.2：LIKE 兜底与 store.search_notes 对齐三列口径。此前只搜
    # content——中文两字短查询（如"猫粮"）达不到 trigram 3 字下限，必然
    # 走兜底，标题命中全部漏掉（面板路径能搜到、注入路径搜不到）
    pattern = f"%{query}%"
    sql = ("SELECT id FROM notes WHERE deleted_at IS NULL "
           "AND (content LIKE ? OR title LIKE ? OR COALESCE(tags,'') LIKE ?)")
    params = [pattern, pattern, pattern]
    if scope:
        sql += " AND scope=?"
        params.append(scope)
    sql += " ORDER BY updated_at DESC LIMIT ?"
    params.append(int(top_n))
    rows = conn.execute(sql, params).fetchall()
    return [(r["id"], i + 1) for i, r in enumerate(rows)]


def semantic_ranks(query_vec, rows, top_n: int):
    """笔记语义通道：对已加载的 (id, vec) 做余弦排序。"""
    from .vector import unpack
    scored = []
    for r in rows:
        v = unpack(r["vec"], r["vec_dim"])
        if not v or len(v) != len(query_vec):
            continue
        scored.append((r["id"], cosine(query_vec, v)))
    scored.sort(key=lambda x: x[1], reverse=True)
    return [(mid, i + 1) for i, (mid, _) in enumerate(scored[:top_n])]


def retrieve(store, *, query: str, scope: str, query_vec=None, top_k: int = 3,
             candidate_pool: int = 20, rrf_k: int = 60,
             max_chunks_per_note: int = 2) -> list[dict]:
    """笔记混合检索（切片 + 整篇 RRF），返回按笔记聚合的命中。

    v0.2.0：有切片的笔记走切片通道（命中段落而非整篇），没有切片的存量
    笔记走整篇通道——两条通道在同一 RRF 里融合，旧库未回填时行为与
    v0.1.9 完全一致（全部走整篇）。每篇笔记最多取 max_chunks_per_note 个
    最优切片拼接注入，防长文挤爆预算。
    """
    # 与 store.list_notes 同口径（ORDER/LIMIT）的轻量 id 集合
    sql = "SELECT id FROM notes WHERE deleted_at IS NULL"
    params: list = []
    if scope:
        sql += " AND scope=?"
        params.append(scope)
    sql += " ORDER BY updated_at DESC LIMIT 1000"
    by_id = {r["id"] for r in store.conn.execute(sql, params).fetchall()}
    if not by_id:
        return []
    chunk_note_ids = _notes_with_chunks(store.conn, scope) & by_id
    legacy_ids = by_id - chunk_note_ids

    # 通道输出统一为 (note_id, chunk_id_or_None, rank)
    channels: dict[str, list[tuple[str, str | None, int]]] = {}
    if query_vec:
        if chunk_note_ids:
            rows = [r for r in store.chunks_with_vectors(scope=scope)
                    if r["note_id"] in chunk_note_ids]
            ch = chunk_semantic_ranks(query_vec, rows, candidate_pool)
            if ch:
                channels["semantic_chunk"] = ch
        if legacy_ids:
            rows = [r for r in store.notes_with_vectors(scope=scope)
                    if r["id"] in legacy_ids]
            ch = [(mid, None, rank)
                  for mid, rank in semantic_ranks(query_vec, rows, candidate_pool)]
            if ch:
                channels["semantic_note"] = ch
    if query.strip():
        if chunk_note_ids:
            ch = [c for c in chunk_lexical_ranks(store.conn, query, scope,
                                                 candidate_pool, store.fts)
                  if c[0] in chunk_note_ids]
            if ch:
                channels["lexical_chunk"] = ch
        if legacy_ids:
            ch = [(mid, None, rank)
                  for mid, rank in lexical_ranks(store.conn, query, scope,
                                                 candidate_pool, store.fts)
                  if mid in legacy_ids]
            if ch:
                channels["lexical_note"] = ch
    if not channels:
        return []

    k = max(1, int(rrf_k))
    agg: dict[str, float] = {}
    chunk_hits: dict[str, list[tuple[int, str]]] = {}
    for ranked in channels.values():
        # 通道内每篇笔记只取最佳名次：同一笔记的多个切片各自加分会
        # 叠加成“切片越多分越高”的偏差，压过真正的精确命中（离线评测实测）。
        best: dict[str, tuple[int, str | None]] = {}
        for mid, cid, rank in ranked:
            if mid not in by_id:
                continue
            cur = best.get(mid)
            if cur is None or rank < cur[0]:
                best[mid] = (rank, cid)
        for mid, (rank, cid) in best.items():
            agg[mid] = agg.get(mid, 0.0) + 1.0 / (k + rank)
            if cid:
                chunk_hits.setdefault(mid, []).append((rank, cid))
    ordered = sorted(agg.items(), key=lambda kv: kv[1], reverse=True)[:max(1, int(top_k))]
    if not ordered:
        return []

    # 每篇取最多 max_chunks_per_note 个最优切片正文
    take_per_note: dict[str, list[str]] = {}
    for mid, _ in ordered:
        picks = sorted(chunk_hits.get(mid, []))[:max(1, int(max_chunks_per_note))]
        take_per_note[mid] = [cid for _, cid in picks]
    chunk_ids = [cid for ids in take_per_note.values() for cid in ids]
    chunk_text: dict[str, str] = {}
    if chunk_ids:
        qmarks = ",".join("?" * len(chunk_ids))
        for r in store.conn.execute(
            f"SELECT id, content FROM note_chunks WHERE id IN ({qmarks})", chunk_ids
        ).fetchall():
            chunk_text[r["id"]] = r["content"]

    wanted = [mid for mid, _ in ordered]
    qmarks = ",".join("?" * len(wanted))
    full: dict[str, sqlite3.Row] = {}
    for r in store.conn.execute(
        f"SELECT id, title, content, heading, file_name FROM notes WHERE id IN ({qmarks})",
        wanted,
    ).fetchall():
        full[r["id"]] = r
    out = []
    for mid, score in ordered:
        r = full.get(mid)
        if r is None:
            continue
        picks = [chunk_text[cid] for cid in take_per_note.get(mid, []) if cid in chunk_text]
        content = "\n".join(picks) if picks else r["content"]
        out.append({
            "id": mid, "title": r["title"] or "", "content": content,
            "heading": r["heading"] or "", "file_name": r["file_name"] or "",
            "rrf": score, "chunks": take_per_note.get(mid, []),
            "via": "chunk" if picks else "note",
        })
    return out


def format_block(notes: list[dict], max_chars: int = 200) -> str:
    """把笔记列表拼成注入块（条目级消毒，防伪标签注入）。"""
    from .text import sanitize_for_context
    if not notes:
        return ""
    lines = []
    for n in notes:
        # src（heading/file_name）来自导入的 .md，同样消毒——否则伪标签可以
        # 借道标题字段闭合注入块
        src_raw = str(n.get("heading") or n.get("file_name") or "").strip()
        src = f"（{sanitize_for_context(src_raw, 40)}）" if src_raw else ""
        lines.append(f"- {sanitize_for_context(n['content'], max_chars)}{src}")
    return "<notes>\n" + "\n".join(lines) + "\n</notes>"
