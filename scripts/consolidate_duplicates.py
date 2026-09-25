"""存量近重复记忆合并（v0.2.3 配套工具，确定性策略）。

背景：v0.2.3 之前「同事实换措辞」会被反复入库（线上跑步兴趣簇 5 条并存）。
新守卫只防未来写入，存量簇需要一次性收敛。夜间巩固的聚类阈值（0.86）抓不到
这类簇，故提供本脚本按**与写入守卫同一套判定**扫描：

- 判定（v0.2.6 起**与嵌入器解耦**的纯文本规则）：双方正文≥10字、非编号
  模板、**剥身份后**字符 Jaccard≥text_floor（0.60）、**剥掉身份token后共享稀有中文
  二元组**（DF≤max(3,2%·语料)）。线上校准（2026-09-21 实测）：真重复
  剥身份后 Jaccard 0.60~0.98 且共享「跑步」类稀有词；不同事实 ≤0.50，
  灰色带靠「稀有中文词」分辨（假对共享的是「助理」类高频词，DF 上百）。
  ⚠️ 向量分降级为可选 --vector-floor（默认 0=不启用）：同主语不同事实的
  余弦实测也能到 0.85~0.96，向量无法单独分辨真假，且嵌入器已从 nemotron
  换成 dashscope，旧校准失效——文本信号才与用哪个嵌入器无关；
- 保护：主动记忆（is_active）不参与；已软删/隔离的不参与。
- 合并策略（确定性，不调 LLM、不生成新文本，杜绝改写丢细节）：
  保留簇内最强成员（强度→有用度→内容长度），**原地升级**它
  （强度/有用度/证据数取簇内最大或求和，tags 取并集）；
  其余成员软删进回收站并记 superseded_by 指向保留条 + 审计事件——
  全部可从面板恢复。
- 安全约定：默认 dry-run；--apply 先落全量 JSON 快照；单事务；幂等。

用法：
    python scripts/consolidate_duplicates.py --db <mnemoria.db>            # 预览
    python scripts/consolidate_duplicates.py --db <...> --apply           # 执行
    python scripts/consolidate_duplicates.py --db <...> --limit 10        # 只处理前 10 簇
"""

from __future__ import annotations

import argparse
import json
import re
import sqlite3
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from core import db as dbm  # noqa: E402
from core import admission  # noqa: E402
from core.tags import normalize_tags  # noqa: E402
from core.vector import cosine  # noqa: E402


def _tags_of(row) -> list[str]:
    try:
        raw = row["tags_json"] if "tags_json" in row.keys() else None
        tags = json.loads(raw) if raw else []
        return [str(t) for t in tags] if isinstance(tags, list) else []
    except (TypeError, ValueError):
        return []


#: 身份token（openid/数字ID/带ID的括号块）——相似度计算前先剥掉，
#: 否则「小明（z0xw…@im.wechat）」这类长前缀会把同一用户的不同事实
#: 也推高相似度（线上实测：不同事实对的 cos 高达 0.96、Jaccard 0.5+）。
_RE_ID_TOKEN = re.compile(
    r"[A-Za-z0-9_-]{6,}@im\.[A-Za-z0-9.]+"      # 微信 openid
    r"|(?<!\d)\d{7,12}(?!\d)"                    # QQ 风格数字 ID
    r"|[（(][^）)]*(?:@im\.|\d{7,})[^）)]*[）)]"   # 带 ID 的括号块
)


def _bigrams(text: str) -> set[str]:
    t = str(text or "")
    return {t[i:i + 2] for i in range(len(t) - 1)} if len(t) >= 2 else set()


def _has_cjk(s: str) -> bool:
    return any("\u4e00" <= ch <= "\u9fff" for ch in s)


def _shares_distinctive(a: str, b: str, df: dict, total: int,
                        df_ratio: float = 0.02) -> bool:
    """剥掉身份token后，两段正文是否共享**稀有的中文内容词**。

    线上校准（2026-09-21，nvidia/nemotron 嵌入）：
    - 真重复对（跑步簇）cos 0.92-0.97、Jaccard 0.5-0.82，共享「跑步」等稀有词；
    - 同主语不同事实对 cos 也能到 0.85-0.96，剥掉数字 ID 后仍共享英文名
      （Star Lantern）的字符片段——**必须要求共享词含中文**，否则姓名碎片
      会在小语料里被误判为稀有信号；
    - 向量阈值在此嵌入器上无法区分两类，故只作 ≥0.80 的兜底 sanity check。
    """
    ca = _RE_ID_TOKEN.sub(" ", str(a or ""))
    cb = _RE_ID_TOKEN.sub(" ", str(b or ""))
    shared = _bigrams(ca) & _bigrams(cb)
    if not shared:
        return False
    cap = max(3, int(df_ratio * max(1, total)))
    return any(_has_cjk(g) and df.get(g, 0) <= cap for g in shared)


def find_clusters(conn: sqlite3.Connection, *, scope: str = "",
                  vector_floor: float = 0.0, text_floor: float = 0.60,
                  max_clusters: int = 0) -> list[dict]:
    """扫描近重复簇。返回 [{scope, members:[row...]}]，成员按强度降序。

    判定（v0.2.3 校准版，全部满足）：
    1) 双方正文 ≥10 字（短文本 Jaccard 噪声大）；
    2) 非「仅编号不同」的模板内容；
    3) 剥身份后字符 Jaccard ≥ text_floor（0.60，线上校准值：真重复 0.60~0.98、不同事实 ≤0.50）；
    4) 剥掉身份token后共享**稀有**二元组（DF≤2% 语料）——分辨
       「同事实换措辞」与「同主语不同事实」的关键信号；
    5) 可选：传了 --vector-floor（>0）时，有向量还要求余弦 ≥ 该值
       （默认不启用，原因见 _similar）。
    """
    sql = ("SELECT id, scope, content, memory_type, speaker, speaker_key, "
           "strength, useful_score, proof_count, is_active, tags_json, created_at "
           "FROM memories WHERE deleted_at IS NULL AND quarantined=0")
    params: tuple = ()
    if scope:
        sql += " AND scope=?"
        params = (scope,)
    rows = [dict(r) for r in conn.execute(sql, params).fetchall()]
    vec_map = {mid: v for mid, v in conn.execute(
        "SELECT memory_id, vec FROM vectors").fetchall()} if _has_vectors(conn) else {}

    cand = [r for r in rows if not int(r["is_active"] or 0)]
    cand.sort(key=lambda r: (-float(r["strength"] or 0),
                             -float(r["useful_score"] or 0),
                             -len(str(r["content"] or ""))))
    # 语料二元组文档频率（供「共享稀有词」判定；只统计活性记忆）
    df: dict[str, int] = {}
    for r in rows:
        for g in _bigrams(_RE_ID_TOKEN.sub(" ", str(r["content"] or ""))):
            df[g] = df.get(g, 0) + 1
    total = max(1, len(rows))
    assigned: set[str] = set()
    clusters: list[dict] = []
    for i, seed in enumerate(cand):
        if seed["id"] in assigned:
            continue
        group = [seed]
        for other in cand[i + 1:]:
            if other["id"] in assigned:
                continue
            if str(other["scope"]) != str(seed["scope"]):
                continue
            if _similar(seed, other, vec_map, df, total,
                        vector_floor, text_floor):
                group.append(other)
        if len(group) < 2:
            continue
        for r in group:
            assigned.add(r["id"])
        clusters.append({"scope": str(seed["scope"]),
                         "members": group})
        if max_clusters and len(clusters) >= max_clusters:
            break
    return clusters


def _has_vectors(conn: sqlite3.Connection) -> bool:
    try:
        return conn.execute(
            "SELECT 1 FROM sqlite_master WHERE type='table' AND name='vectors'"
        ).fetchone() is not None
    except sqlite3.Error:
        return False


def _unpack_vec(blob) -> list[float] | None:
    if blob is None:
        return None
    try:
        import struct
        return list(struct.unpack(f"<{len(blob)//4}f", blob))
    except Exception:  # noqa: BLE001
        return None


def _similar(a: dict, b: dict, vec_map: dict, df: dict, total: int,
             vector_floor: float, text_floor: float) -> bool:
    """纯文本判定（v0.2.6 起与嵌入器解耦）。

    ⚠️ 2026-09-21 线上实测（nemotron 嵌入器）：同主语不同事实的余弦
    也能到 0.85~0.96，完全无关的短记忆也有 0.80~0.87——向量阈值在此
    语料上**无法区分真假重复**。且 9-22 线上嵌入器已换成 dashscope，
    旧校准整体失效。故向量分降级为**可选**的 --vector-floor（默认 0
    = 不启用），主判定完全落在「剥身份 token 后的文本信号」上——它
    与用哪个嵌入器无关，也不会因换提供商而静默失灵。
    """
    ca, cb = str(a["content"] or ""), str(b["content"] or "")
    if not ca or not cb:
        return False
    if len(ca) < 10 or len(cb) < 10:
        return False
    if admission.differs_only_by_numbers(ca, cb):
        return False
    # ⚠️ Jaccard 必须算在**剥掉身份 token 后**的文本上：带 30 字长身份前缀
    # 算原文会把任意两条同用户记忆灌到 0.45+（2026-09-22 dry-run 实测：
    # 127 簇/577 条，把「英语晨读课」和「AI 回复准确性」并成一簇）。
    # 线上校准（2026-09-21）：真重复剥身份后 0.60~0.98，不同事实 ≤0.50。
    sa = _RE_ID_TOKEN.sub(" ", ca)
    sb = _RE_ID_TOKEN.sub(" ", cb)
    if admission.text_similarity(sa, sb) < text_floor:
        return False
    if not _shares_distinctive(ca, cb, df, total):
        return False
    if vector_floor > 0.0:
        va, vb = _unpack_vec(vec_map.get(a["id"])), _unpack_vec(vec_map.get(b["id"]))
        if va and vb and len(va) == len(vb) and cosine(va, vb) < vector_floor:
            return False
    return True


def apply(conn: sqlite3.Connection, clusters: list[dict], backup_dir: Path) -> Path:
    """先快照后合并；单事务；返回快照路径。"""
    backup_dir.mkdir(parents=True, exist_ok=True)
    snap = backup_dir / f"dupes-consolidate-{time.strftime('%Y%m%d-%H%M%S')}.json"
    snap.write_text(json.dumps({
        "created_at": time.time(),
        "reason": "pre-duplicates-consolidation",
        "clusters": clusters,
    }, ensure_ascii=False, indent=2), encoding="utf-8")

    now = time.time()
    conn.execute("BEGIN")
    try:
        for cl in clusters:
            members = cl["members"]
            kept = members[0]  # find_clusters 已按强度降序
            strength = max(float(m["strength"] or 0) for m in members)
            useful = max(float(m["useful_score"] or 0) for m in members)
            proof = sum(int(m["proof_count"] or 1) for m in members)
            tags: list[str] = []
            for m in members:
                tags.extend(_tags_of(m))
            conn.execute(
                "UPDATE memories SET strength=?, useful_score=?, proof_count=?, "
                "updated_at=? WHERE id=?",
                (strength, useful, proof, now, kept["id"]),
            )
            conn.execute(
                "UPDATE memories SET tags_json=?, updated_at=? WHERE id=?",
                (json.dumps(normalize_tags(tags, limit=8), ensure_ascii=False),
                 now, kept["id"]),
            )
            for m in members[1:]:
                # 软删 + 血缘指向保留条 + 双时态终点，全部可恢复
                conn.execute(
                    "UPDATE memories SET deleted_at=?, superseded_by=?, valid_to=?, "
                    "updated_at=? WHERE id=?",
                    (now, kept["id"], now, now, m["id"]),
                )
                conn.execute(
                    "INSERT INTO memory_events(action, scope, source_ids_json, "
                    "target_id, reason, confidence, provider, created_at) "
                    "VALUES(?,?,?,?,?,?,?,?)",
                    # v0.2.11 字段语义正位：source=被取代的旧条、target=保留条，
                    # 与 merge/update/undo_consolidation 一致。初版（09-22 线上
                    # 执行的 6 条）source 为空、受害者只落在 target_id——
                    # undo 脚本 v0.2.15 起按血缘方向判定兼容，不再误标 keeper
                    ("manual_dedup", cl["scope"], json.dumps([m["id"]]),
                     kept["id"],
                     f"近重复合并至 {kept['id'][:8]}（脚本确定性合并）", 1.0,
                     "consolidate_duplicates.py", now),
                )
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    return snap


def main() -> int:
    ap = argparse.ArgumentParser(description="存量近重复记忆合并（确定性）")
    ap.add_argument("--db", required=True, help="mnemoria.db 路径")
    ap.add_argument("--apply", action="store_true", help="真正写入（默认只预览）")
    ap.add_argument("--limit", type=int, default=0, help="最多处理几簇（0=全部）")
    ap.add_argument("--scope", default="", help="只处理某个隔离域")
    ap.add_argument("--vector-floor", type=float, default=0.0,
                    help="可选的向量余弦兜底门槛；默认 0=不启用（2026-09-21 实测本语料"
                         "向量无法区分真假重复，且嵌入器已更换，故主判定完全走文本）")
    ap.add_argument("--text-floor", type=float, default=0.60,
                    help="剥身份后的字符 Jaccard 下限（线上校准值 0.60，配合稀有共享词判定）")
    args = ap.parse_args()

    db = Path(args.db)
    if not db.exists():
        raise SystemExit(f"数据库不存在: {db}")
    conn = dbm.connect(db)
    clusters = find_clusters(conn, scope=args.scope,
                             vector_floor=args.vector_floor,
                             text_floor=args.text_floor,
                             max_clusters=args.limit)
    n_members = sum(len(c["members"]) for c in clusters)
    print(f"近重复簇 {len(clusters)} 个，涉及记忆 {n_members} 条"
          f"（合并后减少 {n_members - len(clusters)} 条）")
    for cl in clusters[:10]:
        kept = cl["members"][0]
        print(f"\n[{cl['scope']}] 保留: {str(kept['content'])[:50]}")
        for m in cl["members"][1:]:
            print(f"    合并: {str(m['content'])[:50]}")
    if not args.apply:
        print("\ndry-run：未写入。确认无误后加 --apply 执行（会先落 JSON 快照）。")
        conn.close()
        return 0
    snap = apply(conn, clusters, db.parent / "backups")
    left = find_clusters(conn, scope=args.scope,
                         vector_floor=args.vector_floor,
                         text_floor=args.text_floor)
    conn.close()
    print(f"\n已合并 {len(clusters)} 簇；剩余近重复簇 {len(left)} 个；快照: {snap.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
