"""四路检索 + RRF 融合。

四路：
  semantic  向量余弦（需 embedding）
  lexical   FTS5 BM25（对记忆内容）
  tag       标签锚点（v0.1.8，angel tags 同思想：场合词常不在正文里）
  recency   时间近因（越新越靠前）

融合用 Reciprocal Rank Fusion：score = Σ 1/(k + rank)，k 默认 60。
每路结果带排名与纳入理由（receipt），便于排查"为什么召回了这条"。
"""

from __future__ import annotations

import json
from astrbot.api import logger
import sqlite3
from dataclasses import dataclass, field
from operator import mul

from .scoring import anchor_ts, hotness
from .text import estimate_tokens
from .vector import cosine



@dataclass
class Candidate:
    id: str
    content: str
    memory_type: str = "fact"
    is_active: bool = False
    hotness: float = 0.0
    # v0.2.4：透出创建时间——注入块据此标注相对时间（angel memory_formatter
    # 同思想），也是年龄衰减加权的数据源；缺省 0 表示未知（不标注不衰减）。
    created_at: float = 0.0
    channels: dict[str, int] = field(default_factory=dict)  # 通道 -> 排名(1-based)
    rrf: float = 0.0
    receipt: str = ""


def _semantic_ranks(query_vec, vectors, top_n, *, normalized: bool = False,
                    floor: float = 0.0):
    """语义通道排序。

    normalized=True：vectors 为预归一化向量（引擎 _cached_vectors 的契约，
    见 engine._refresh_vector_cache）——cosine 尺度不变，点积与余弦数学
    等价（实测最大偏差 ~1e-16），而 sum(map(mul)) 的内联 C 循环比三累乘
    Python 循环快约 4 倍，且免去每查询一次的向量解包（v0.1.4）。
    """
    scored = []
    if normalized:
        from .vector import normalize_vec
        q = normalize_vec(query_vec)
        if not q:
            return []
        for mid, vec in vectors:
            if not vec or len(vec) != len(q):
                continue
            scored.append((mid, sum(map(mul, q, vec))))
    else:
        for mid, vec in vectors:
            if not vec or len(vec) != len(query_vec):
                continue
            scored.append((mid, cosine(query_vec, vec)))
    scored.sort(key=lambda x: x[1], reverse=True)
    # 相似度地板（第四轮矩阵验证发现）：require_match 场景「无匹配就该空」
    # 的设计意图需要地板才成立，否则语义通道永远兜底返回最相近垃圾
    if floor > 0.0:
        scored = [(mid, sim) for mid, sim in scored if sim >= floor]
    return [(mid, i + 1, sim) for i, (mid, sim) in enumerate(scored[:top_n])]


def _lexical_ranks(conn, query, scope, top_n, fts_ok):
    rows: list[sqlite3.Row]
    if fts_ok:
        try:
            terms = [t for t in query.replace('"', " ").split() if t]
            if terms:
                match = " OR ".join(f'"{t}"' for t in terms)
                rows = conn.execute(
                    "SELECT m.id AS id FROM memories_fts "
                    "JOIN memories m ON m.id=memories_fts.row_id "
                    "WHERE memories_fts MATCH ? AND m.scope=? AND m.deleted_at IS NULL "
                    "AND m.superseded_by IS NULL "
                    "ORDER BY rank LIMIT ?",
                    (match, scope, top_n),
                ).fetchall()
                if rows:
                    return [(r["id"], i + 1, 0.0) for i, r in enumerate(rows)]
        except sqlite3.OperationalError:
            pass
    like = f"%{query}%"
    rows = conn.execute(
        "SELECT id FROM memories WHERE content LIKE ? AND scope=? AND deleted_at IS NULL "
        "AND superseded_by IS NULL "
        "ORDER BY created_at DESC LIMIT ?",
        (like, scope, top_n),
    ).fetchall()
    return [(r["id"], i + 1, 0.0) for i, r in enumerate(rows)]


def _recency_ranks(rows, top_n):
    ordered = sorted(rows, key=lambda r: r["created_at"] or 0, reverse=True)
    return [(r["id"], i + 1, 0.0) for i, r in enumerate(ordered[:top_n])]


def _tag_ranks(rows, lex, top_n):
    """标签通道：query 与记忆 tags 互为子串即命中（中文无分词也能匹配短标签）。

    lex 是完整查询串：tag「跑步」⊂ query「想找跑步的店」直接命中；
    英文多词查询再按空白拆词双向比对。命中序跟随主查询行序（rowid 稳定）。
    """
    lex_l = lex.strip().lower()
    if not lex_l:
        return []
    terms = [t for t in lex_l.split() if t] if " " in lex_l else [lex_l]
    hits: list[str] = []
    for r in rows:
        raw = r["tags_json"] if "tags_json" in r.keys() else None
        if not raw or raw == "[]":
            continue
        try:
            tags = json.loads(raw)
        except Exception:  # noqa: BLE001
            continue
        if not isinstance(tags, list):
            continue
        for tag in tags:
            t = str(tag).strip().lower()
            if t and (t in lex_l or any(t in term or term in t for term in terms)):
                hits.append(r["id"])
                break
    return [(mid, i + 1, 0.0) for i, mid in enumerate(hits[:top_n])]


def hybrid_retrieve(
    conn,
    *,
    query: str,
    scope: str,
    query_vec=None,
    fts_ok: bool = True,
    candidate_pool: int = 40,
    rrf_k: int = 60,
    half_life_days: float = 7.0,
    now_ts: float = 0.0,
    top_k: int = 8,
    token_budget: int = 800,
    active_only: bool = False,
    include_recency: bool = True,
    lexical_query: str | None = None,
    per_type_limit: int = 0,
    vectors: list[tuple[str, list[float]]] | None = None,
    semantic_floor: float = 0.0,
    age_decay_rate: float = 0.0,
    vector_backlog=None,
) -> list[Candidate]:
    """返回融合后的候选（已按 RRF 排序并截断到预算）。

    include_recency=False 用于显式搜索场景（/记忆搜索、/忘记）：
    时间近因通道只在「注入兜底」时有意义，用户明确搜索时无匹配就该说无匹配。

    lexical_query：关键词通道可单独用扩展后的查询（查询扩展时语义通道
    仍用原查询——扩展词会稀释嵌入向量）。

    per_type_limit>0 时对每种 memory_type 各限该条数（angel 链式回忆的分组
    限额同思想）：防单一类型霸榜，保证召回多样性；超出该类型的条目被降级到
    候补，在全部类型选完后按 RRF 顺序补满 top_k，不会凭空丢记忆。

    vectors（v0.1.4）：调用方预载的**归一化**向量对（引擎 _cached_vectors）；
    传入时语义通道走点积快路径并免去每查询解包。None 时此函数自 DB 现取
    原始向量走余弦路径（向后兼容的旧行为）。

    age_decay_rate（v0.2.4，对齐 angel `_apply_time_decay`）：>0 时对被动
    记忆的融合分乘年龄因子 1/(1+rate*age_days)——hotness 是「上次被想起」
    的召回热度，本项才是「这条事实本身有多旧」的新鲜度惩罚；两个方向一正
    一负，矛盾记忆并列时新条天然靠前。is_active 条目不衰减。0 关闭。
    """
    where = "scope=? AND deleted_at IS NULL AND superseded_by IS NULL"
    params: list = [scope]
    if active_only:
        where += " AND is_active=1"
    # v0.1.4：主查询只取轻量列（不含 content）——正文在通道产出候选后按 id
    # 取回（候选合计 ≤ 3×candidate_pool 条），避免每轮注入把 scope 内全部
    # 正文拉进内存；行为与旧版全量取行完全一致，只是取行时机后移
    rows = conn.execute(
        "SELECT id, memory_type, is_active, hit_count, "
        "last_recalled_at, last_decay_at, created_at, tags_json FROM memories WHERE " + where,
        params,
    ).fetchall()
    if not rows:
        return []
    by_id = {r["id"]: r for r in rows}

    channel_results: dict[str, list] = {}
    if query_vec:
        if vectors is not None:
            vecs = [(mid, v) for mid, v in vectors if mid in by_id]
            channel_results["semantic"] = _semantic_ranks(
                query_vec, vecs, candidate_pool, normalized=True,
                floor=semantic_floor,
            )
        else:
            vecs = [(mid, v) for mid, v in _all_scope_vectors(
                conn, scope, expected_dim=len(query_vec), backlog=vector_backlog,
            ) if mid in by_id]
            channel_results["semantic"] = _semantic_ranks(
                query_vec, vecs, candidate_pool, floor=semantic_floor)
    lex = (lexical_query or query).strip()
    if lex:
        channel_results["lexical"] = _lexical_ranks(conn, lex, scope, candidate_pool, fts_ok)
        # v0.1.8 标签通道：tags 是「场合词」，常不出现在正文里——
        # 只靠正文 FTS 会漏掉「tag 有而 content 无」的检索（angel recall_by_tags 同思想）。
        # 规模内（千条级）全表 LIKE 扫描代价可忽略，无需倒排索引。
        channel_results["tag"] = _tag_ranks(rows, lex, candidate_pool)
    if include_recency:
        channel_results["recency"] = _recency_ranks(rows, candidate_pool)

    wanted = sorted({mid for ranked in channel_results.values() for mid, _rank, _sim in ranked})
    contents: dict[str, str] = {}
    for i in range(0, len(wanted), 500):
        chunk = wanted[i:i + 500]
        qmarks = ",".join("?" * len(chunk))
        for r in conn.execute(
            f"SELECT id, content FROM memories WHERE id IN ({qmarks})", chunk,
        ).fetchall():
            contents[r["id"]] = r["content"]

    k = max(1, int(rrf_k))
    agg: dict[str, Candidate] = {}
    for ch, ranked in channel_results.items():
        for mid, rank, sim in ranked:
            row = by_id.get(mid)
            if row is None:
                continue
            cand = agg.get(mid)
            if cand is None:
                anchor = anchor_ts(row["last_recalled_at"], row["last_decay_at"], row["created_at"])
                cand = Candidate(
                    id=mid,
                    content=contents.get(mid, ""),
                    memory_type=row["memory_type"],
                    is_active=bool(row["is_active"]),
                    hotness=hotness(row["hit_count"] or 0, anchor, now_ts, half_life_days),
                    created_at=float(row["created_at"] or 0.0),
                )
                agg[mid] = cand
            cand.channels[ch] = rank
            cand.rrf += 1.0 / (k + rank)

    candidates = list(agg.values())
    # 主动记忆与高热度做轻微加权，但不颠覆 RRF 主体
    for c in candidates:
        c.rrf *= (1.0 + 0.15 * c.hotness) * (1.15 if c.is_active else 1.0)
    # v0.2.4 年龄衰减（angel _apply_time_decay 同公式）：对被动记忆按
    # 「事实本身的年龄」乘 1/(1+rate*age_days)。hotness 奖励「常被想起」，
    # 本项惩罚「本身太旧」——此前检索只有前者没有后者，旧矛盾记忆反而
    # 吃热度加成排在前面（过时记忆审查 R4/R5）。
    rate = float(age_decay_rate or 0.0)
    if rate > 0.0 and now_ts:
        for c in candidates:
            if c.is_active or not c.created_at:
                continue
            age_days = max(0.0, (now_ts - c.created_at) / 86400.0)
            c.rrf *= 1.0 / (1.0 + rate * age_days)
    candidates.sort(key=lambda c: c.rrf, reverse=True)

    # 相对截断（Paramecium「垃圾线」思想）：与头名差距过大的候选是噪声，
    # 不得占满 top_k。阈值取头名的 25%——RRF 每通道贡献 1/(k+rank)，
    # 只有单一通道靠后排名的条目通常落在 head*0.25 以下。至少保留 1 条。
    if candidates:
        head = candidates[0].rrf
        if head > 0:
            candidates = [c for c in candidates if c.rrf >= head * 0.25] or candidates[:1]

    # 分组限额（angel 链式回忆同思想）：每种类型先各取 per_type_limit 条，
    # 其余进候补；全部类型选完后若未满 top_k，再按 RRF 顺序补上候补。
    # 这样既保证类型多样性，又不会因限额而漏掉高相关条目。
    if per_type_limit > 0 and candidates:
        primary: list[Candidate] = []
        deferred: list[Candidate] = []
        type_counts: dict[str, int] = {}
        for c in candidates:
            mt = c.memory_type or "fact"
            if type_counts.get(mt, 0) < per_type_limit:
                type_counts[mt] = type_counts.get(mt, 0) + 1
                primary.append(c)
            else:
                deferred.append(c)
        candidates = primary + deferred

    # token 预算截断
    out: list[Candidate] = []
    used = 0
    for c in candidates:
        cost = estimate_tokens(c.content) + 4
        if out and used + cost > token_budget:
            break
        used += cost
        c.receipt = _receipt(c)
        out.append(c)
        if len(out) >= top_k:
            break
    return out


def _all_scope_vectors(conn, scope: str, expected_dim: int = 0,
                        backlog=None) -> list[tuple[str, list[float]]]:
    """按 scope 取向量；异维条目入队后跳过，不让混部污染余弦排序。"""
    from .vector import unpack
    rows = conn.execute(
        "SELECT v.memory_id mid, v.dim dim, v.vec vec FROM vectors v "
        "JOIN memories m ON m.id=v.memory_id "
        "WHERE m.deleted_at IS NULL AND m.scope=?",
        (scope,),
    ).fetchall()
    out = []
    for r in rows:
        if expected_dim and int(r["dim"] or 0) != int(expected_dim):
            if backlog is not None:
                try:
                    backlog(r["mid"], int(r["dim"] or 0))
                except Exception:  # noqa: BLE001
                    pass
            continue
        out.append((r["mid"], unpack(r["vec"], r["dim"])))
    return out


def _receipt(c: Candidate) -> str:
    parts = [f"{ch}#{rank}" for ch, rank in sorted(c.channels.items())]
    return f"rrf={c.rrf:.4f} hot={c.hotness:.2f} via[{', '.join(parts)}]"
