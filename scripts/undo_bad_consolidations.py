"""回滚夜间巩固的误合并（v0.2.8 配套止血脚本，事故 2026-09-22 登记）。

事故：`_consolidate_scope` 只凭向量余弦聚类（无文本确认），而线上嵌入器
对「同主语不同事实」的余弦普遍 >0.86——单个 keeper 一晚吞掉 214 条不相干
记忆，活性库 977→552。血缘审计：534 条被取代记录中 518 条**无任何审计事件**
（巩固是当时唯一无审计的通道），其中 483 条与 keeper 的文本相似度 <0.35，
即不同事实被强行拼进同一条复合记忆。

本脚本按「宁恢复、不猜」原则逐条回滚：

回滚对象（两个条件必须同时满足）：
1. **无审计事件佐证**：被取代条不在任何 memory_events.source_ids_json 里。
   有事件（merge/update/manual_dedup）的链是**故意**的更正或去重——
   例如「住杭州→搬到上海」改口链新旧文本必然不相似，若不按事件排除、
   会被误判成误合并并把旧事实复活，这是本脚本最重要的护栏；
2. **文本相似度 < floor（默认 0.55）**：旧条与其 keeper 内容明显不是
   同一事实 → 当时就不该进同一簇。

对 keeper 的处理：某个 keeper 的**全部**无事件受害者都被回滚、且它没有
任何被保留的受害者（说明它只是这些不同事实的拼接产物）→ keeper 一并送
回收站（可恢复）。keeper 里若有哪怕一条真该吸收的内容（换措辞重复被
floor 保留），keeper 保留。

- 回滚 = 撤销软删与血缘（deleted_at/superseded_by/valid_to 置空）。注意：本脚本
  走直接 SQL，不受 v0.2.14 面板 restore 新语义（默认保留血缘）影响；面板里等价
  操作是「彻底恢复」；不复活任何隔离条目；
- 默认 dry-run；--apply 先落全量 JSON 快照；单事务；幂等（第二遍 0 条）；
- 执行后**必须重启插件**：引擎的向量缓存要到下次写入/重启才刷新，
  不重启的话回滚条检索不到（数据已在库里，只是暂时看不见）。

用法：
    python scripts/undo_bad_consolidations.py --db <mnemoria.db>          # 预览
    python scripts/undo_bad_consolidations.py --db <...> --apply          # 执行
"""

from __future__ import annotations

import argparse
import json
import sqlite3
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from core import admission  # noqa: E402
from core import db as dbm  # noqa: E402
from core.paths import utc_now_ts  # noqa: E402


def _event_covered_ids(conn: sqlite3.Connection) -> set[str]:
    """所有出现在审计事件 source_ids 里的记忆 id（=有意的演进/去重/裁决）。

    v0.2.15 兼容扫描（按线上真实形态修正——2026-09-23 只读核查）：
    2026-09-22 执行的 6 条 manual_dedup 旧事件 source_ids_json 为**空**、
    受害者落在 target_id（keeper 只存在于 reason 文本）；v0.2.11 字段正位后
    新事件 source=[victim]、target=keeper。两代事件不能只看字段名——用血缘
    方向判定：**target 行自身处于被吸收态（superseded_by 非空）才说明它是
    受害者**，计入 covered；keeper（superseded_by 为空的存活行）不会被误标，
    将来若被夜间巩固误并可正常回滚。
    注意：SQL 里不得加 `source_ids_json <> '[]'` 过滤——旧 6 条 source 恰为
    空，滤掉即丢失其受害者的护栏。
    """
    covered: set[str] = set()
    for row in conn.execute(
            "SELECT action, target_id, source_ids_json FROM memory_events "
            "WHERE source_ids_json IS NOT NULL"):
        try:
            ids = json.loads(row["source_ids_json"] or "[]")
        except (ValueError, TypeError):
            continue
        if isinstance(ids, list):
            covered.update(str(i) for i in ids if str(i))
        if str(row["action"] or "") == "manual_dedup" and row["target_id"]:
            t = conn.execute(
                "SELECT superseded_by FROM memories WHERE id=?",
                (str(row["target_id"]),)).fetchone()
            if t is not None and t["superseded_by"]:
                covered.add(str(row["target_id"]))
    return covered


def plan(conn: sqlite3.Connection, *, floor: float = 0.55) -> list[dict]:
    """列出「无事件佐证 + 与 keeper 文本不相似」的回滚候选链。"""
    covered = _event_covered_ids(conn)
    victims = conn.execute(
        "SELECT m.id AS id, m.content AS content, m.scope AS scope, "
        "       m.superseded_by AS kid, k.content AS kcontent "
        "FROM memories m JOIN memories k ON k.id = m.superseded_by "
        "WHERE m.superseded_by IS NOT NULL").fetchall()
    out: list[dict] = []
    for v in victims:
        if v["id"] in covered:
            continue  # 有意的演进/去重链：永不复活旧事实
        sim = admission.text_similarity(str(v["content"] or ""),
                                        str(v["kcontent"] or ""))
        if sim >= floor:
            continue  # 换措辞重复：吸收是对的，保持原状
        out.append({"victim": v["id"], "keeper": v["kid"], "scope": v["scope"],
                    "sim": round(sim, 3),
                    "victim_head": str(v["content"] or "")[:40],
                    "keeper_head": str(v["kcontent"] or "")[:40]})
    return out


def _keepers_to_retire(conn: sqlite3.Connection, rollbacks: list[dict]) -> list[str]:
    """keeper 的全部无事件受害者都被回滚，且 keeper 确属拼接产物 → 送回收站。

    拼接产物的判定（v0.2.8 补救脚本自检后收紧）：回滚受害者 ≥2 个，
    或 keeper 长度 >200 字。单受害者短 keeper 不回滚 keeper：它的其它
    同簇成员可能早已逾期物理删除，退休 keeper 会连带丢失那部分内容；
    宁留一条轻度重复，不赌信息完整性（受害者本身照常恢复）。
    """
    by_keeper: dict[str, list[dict]] = {}
    for r in rollbacks:
        by_keeper.setdefault(r["keeper"], []).append(r)
    retiring: list[str] = []
    for kid, rb_list in by_keeper.items():
        k = conn.execute("SELECT id, scope, deleted_at, length(content) AS clen, "
                         "content FROM memories WHERE id=?", (kid,)).fetchone()
        if k is None or k["deleted_at"] is not None:
            continue  # keeper 本身已不在活性面（可能被别的链取代），不动
        # 「有事件佐证的受害者不必单独判断」：plan() 已把 covered 条目挡在
        # 回滚集外，它们必然出现在 all_v 但不在 rolled → 由下面的全滚检查
        # 统一拦住退休。不写重复分支（第三轮变异审查：重复分支=死代码+测试盲区）。
        all_v = conn.execute(
            "SELECT id FROM memories WHERE superseded_by=?", (kid,)).fetchall()
        rolled = {r["victim"] for r in rb_list}
        if not all(str(r["id"]) in rolled for r in all_v):
            continue
        if len(rb_list) >= 2 or int(k["clen"] or 0) > 200:
            retiring.append(kid)
    return retiring


def apply(conn: sqlite3.Connection, rollbacks: list[dict], retire: list[str],
          backup_dir: Path) -> Path:
    backup_dir.mkdir(parents=True, exist_ok=True)
    snap = backup_dir / f"undo-consolidations-{time.strftime('%Y%m%d-%H%M%S')}.json"
    snap.write_text(json.dumps({
        "created_at": time.time(),
        "reason": "pre-undo-bad-consolidations",
        "rollbacks": rollbacks,
        "keepers_retired": retire,
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    now = utc_now_ts()
    conn.execute("BEGIN")
    try:
        # 回滚受害者：撤销软删与血缘（与 store.restore 同款字段处理，
        # 但不动 quarantined——回滚对象不该有隔离条）
        for r in rollbacks:
            conn.execute(
                "UPDATE memories SET deleted_at=NULL, superseded_by=NULL, "
                "valid_to=NULL, updated_at=? WHERE id=?",
                (now, r["victim"]))
        # 退休误拼 keeper：只进回收站（可恢复），不清血缘（它没有上家）
        for kid in retire:
            conn.execute(
                "UPDATE memories SET deleted_at=?, updated_at=? "
                "WHERE id=? AND deleted_at IS NULL",
                (now, now, kid))
        # 审计：逐条事件（与 manual_dedup 同款风格，便于面板反查）
        for r in rollbacks:
            conn.execute(
                "INSERT INTO memory_events(action, scope, source_ids_json, "
                "target_id, reason, confidence, provider, created_at) "
                "VALUES(?,?,?,?,?,?,?,?)",
                ("undo_consolidation", r["scope"], json.dumps([r["victim"]]),
                 r["keeper"], f"误合并回滚（sim={r['sim']}，无事件佐证）", 1.0,
                 "undo_bad_consolidations.py", now))
        for kid in retire:
            conn.execute(
                "INSERT INTO memory_events(action, scope, source_ids_json, "
                "target_id, reason, confidence, provider, created_at) "
                "VALUES(?,?,?,?,?,?,?,?)",
                ("undo_consolidation", "default", "[]", kid,
                 "误拼 keeper 退休入回收站（可恢复）", 1.0,
                 "undo_bad_consolidations.py", now))
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    return snap


def main() -> int:
    ap = argparse.ArgumentParser(description="回滚夜间巩固的误合并（宁恢复不猜）")
    ap.add_argument("--db", required=True, help="mnemoria.db 路径")
    ap.add_argument("--apply", action="store_true", help="真正写入（默认只预览）")
    ap.add_argument("--floor", type=float, default=0.55,
                    help="与 keeper 文本相似度低于该值的无事件受害者才回滚")
    args = ap.parse_args()

    db = Path(args.db)
    if not db.exists():
        raise SystemExit(f"数据库不存在: {db}")
    conn = dbm.connect(db)
    rb = plan(conn, floor=args.floor)
    retire = _keepers_to_retire(conn, rb)
    total_v = conn.execute(
        "SELECT count(*) FROM memories WHERE superseded_by IS NOT NULL").fetchone()[0]
    print(f"被取代记忆 {total_v} 条；其中误合并回滚候选 {len(rb)} 条，"
          f"连带退休的误拼 keeper {len(retire)} 个；"
          f"有事件佐证的链一律不动（护栏：改口更正不复活）")
    for r in rb[:6]:
        print(f"  - [sim={r['sim']}] {r['victim_head']}  ←keeper: {r['keeper_head']}")
    if not args.apply:
        print("\ndry-run：未写入。确认无误后加 --apply（会先落 JSON 快照）。"
              "\n⚠️ 执行后必须重启插件（向量缓存/检索可见性）。")
        conn.close()
        return 0
    snap = apply(conn, rb, retire, db.parent / "backups")
    left = conn.execute(
        "SELECT count(*) FROM memories WHERE superseded_by IS NOT NULL "
        "AND superseded_by IN (SELECT id FROM memories WHERE deleted_at IS NULL)"
    ).fetchone()[0]
    conn.close()
    print(f"\n已回滚 {len(rb)} 条、退休 keeper {len(retire)} 个；快照: {snap.name}")
    print(f"提示：剩余被取代链 {left} 条（有事件佐证或文本相似，属正常）。重启插件生效。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
