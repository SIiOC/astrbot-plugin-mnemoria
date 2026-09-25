"""离线记忆质量评测（v0.2.0 验收用）。

同一套指标可以分别跑在 v0.1.9 与 v0.2.0 的插件副本上，对比升级前后：
- 记忆检索 Recall@5 / MRR@8（关键词查询 → 期望记忆命中）
- 笔记段落命中率与平均注入长度（切片派生层是否更精准）
- scope 泄漏计数（跨域检索必须为 0）
- 写入裁决守卫错误数（仅 v0.2.0：编号守卫/跨域/永生条目/低置信）
- 10,000 条记忆上的检索 p50/p95 延迟

本脚本完全离线：使用本地字符 bigram 哈希嵌入（不联网、不调 LLM），
度量的是检索与裁决管线本身；生产环境请再用真实嵌入 provider 复测语义指标。

用法：
    python scripts/eval_memory_quality.py --plugin <插件根> --label v0.2.0 \
        [--out eval_v020.json]
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import math
import sys
import tempfile
import time
from pathlib import Path


# ---------------------------------------------------------------- 本地嵌入
class LocalHashEmbedder:
    """确定性字符 bigram 哈希嵌入（离线可复现，用于管线评测）。"""

    enabled = True
    dim = 96

    def __init__(self) -> None:
        import unicodedata

        self._normalize = lambda s: unicodedata.normalize("NFKC", str(s or "")).lower()

    def _vec(self, text: str) -> list[float]:
        t = self._normalize(text)
        v = [0.0] * self.dim
        for i in range(len(t) - 1):
            gram = t[i:i + 2]
            h = int(hashlib.md5(gram.encode("utf-8")).hexdigest()[:8], 16)
            v[h % self.dim] += 1.0
        for ch in t:
            h = int(hashlib.md5(ch.encode("utf-8")).hexdigest()[:8], 16)
            v[h % self.dim] += 0.3
        norm = math.sqrt(sum(x * x for x in v)) or 1.0
        return [x / norm for x in v]

    async def embed_one(self, text, timeout=None):
        return self._vec(text)

    async def embed(self, texts):
        return [self._vec(t) for t in texts]


class ScriptedLLM:
    """按调用次序返回预设裁决输出（离线裁决守卫测试用）。"""

    enabled = True
    provider_id = "eval-scripted"

    def __init__(self, payloads: list):
        self.payloads = list(payloads)
        self.calls = 0

    async def generate_json(self, prompt, system_prompt=None):
        self.calls += 1
        if self.payloads:
            return self.payloads.pop(0)
        return None

    async def generate(self, prompt, system_prompt=None):
        return ""


# ---------------------------------------------------------------- 语料
MEM_CASES = [
    ("跑步", "小明每周四晚上去运动店跑步"),
    ("运动店", "小明喜欢在运动店做跑步"),
    ("猫过敏", "小明对猫过敏，看到猫会打喷嚏"),
    ("生日", "小明的生日是三月五日"),
    ("篮球", "小明每周六下午打篮球"),
    ("美式咖啡", "小明早上习惯喝美式咖啡"),
    ("高考", "小明在准备明年的高考"),
    ("高达模型", "小明喜欢拼装高达模型"),
    ("索尼耳机", "小明的耳机是索尼 WH-1000XM5"),
    ("晨跑房", "小明每周三次去晨跑房练背"),
    ("大熊猫", "小明计划国庆去成都看大熊猫"),
    ("星际迷航", "小明最爱看的科幻剧是星际迷航"),
]

MEM_DISTRACTORS = [
    "楼下便利店新上了关东煮",
    "今天下雨出门要带伞",
    "同事推荐了一家川菜馆",
    "新买的键盘手感不错",
    "周末打算把房间收拾一遍",
    "小区门口在修路",
    "最近在读一本讲海洋的书",
    "手机系统更新后省电了",
    "常去的理发店换了位置",
    "楼上的邻居在装修",
]

NOTE_KEY = "冰岛极光"
NOTE_CASES = [
    (NOTE_KEY, "小明最想看的是冰岛极光"),
    ("敦煌壁画", "小明最想看的是敦煌壁画"),
    ("深海鮟鱇", "小明最好奇的是深海鮟鱇"),
    ("黑胶唱片", "小明收集了很多黑胶唱片"),
    ("唐代茶具", "小明在研究唐代茶具的形制"),
    ("南极科考", "小明关注南极科考队的进展"),
    ("苔藓造景", "小明在阳台养了一片苔藓造景"),
    ("手冲壶", "小明想入手一把不锈钢手冲壶"),
]

_FILLER = "以下是背景资料，用于拉长笔记并验证切片检索的定位能力。" * 60


def _long_note(target: str) -> str:
    """把目标句埋进长文中部，整篇约 3300+ 字符（验证只注入命中切片）。"""
    return "背景介绍。" + _FILLER + "重点记录：" + target + "。" + _FILLER + "以上为补充说明。"


# ---------------------------------------------------------------- 评测主体
async def run_eval(plugin_root: Path, label: str) -> dict:
    sys.path.insert(0, str(plugin_root))
    from core import config as cfgmod, db as dbm
    from core.engine import MemoryEngine
    from core.paths import DataPaths
    from core.store import MemoryStore
    from core.text import content_hash

    tmp = Path(tempfile.mkdtemp(prefix="mnemoria_eval_"))
    paths = DataPaths(tmp / "pd").ensure()
    conn = dbm.connect(paths.db)
    dbm.init_schema(conn)
    store = MemoryStore(conn)
    embedder = LocalHashEmbedder()
    engine = MemoryEngine(store, cfgmod.Config({}, paths.meta), embedder=embedder, llm=None)

    schema_row = conn.execute(
        "SELECT value FROM meta WHERE key='schema_version'").fetchone()
    schema_version = int(schema_row[0]) if schema_row else 0

    result: dict = {"label": label, "plugin": str(plugin_root),
                    "schema_version": schema_version}

    # ---------------- 1) 记忆检索 Recall@5 / MRR@8
    for content in [c for _, c in MEM_CASES] + MEM_DISTRACTORS:
        await engine.remember(content, alpha=0.9, scope="default")
    id_by_content = {r["content"]: r["id"] for r in store.active_memories("default")}
    hits5 = 0
    rr_sum = 0.0
    for query, content in MEM_CASES:
        mid = id_by_content[content]
        cands = await engine.recall(query, scope="default", top_k=8,
                                    mark_recalled=False, require_match=True)
        ids = [c.id for c in cands]
        if mid in ids[:5]:
            hits5 += 1
        if mid in ids:
            rr_sum += 1.0 / (ids.index(mid) + 1)
    result["memory"] = {
        "cases": len(MEM_CASES),
        "recall_at_5": round(hits5 / len(MEM_CASES), 4),
        "mrr_at_8": round(rr_sum / len(MEM_CASES), 4),
    }

    # ---------------- 2) 笔记段落命中率与注入长度
    note_hits = 0
    hit_chars: list[int] = []
    for query, target in NOTE_CASES:
        nid = await engine.add_note(_long_note(target), title="评测笔记", scope="default")
        cands = await engine.notes_recall(query, scope="default", top_k=3)
        if cands and any(target in c["content"] for c in cands):
            note_hits += 1
            hit_chars.append(len(cands[0]["content"]))
        del nid
    chunk_mode = any(
        r[0] for r in conn.execute(
            "SELECT 1 FROM sqlite_master WHERE type='table' AND name='note_chunks' LIMIT 1"
        ).fetchall()
    )
    result["notes"] = {
        "cases": len(NOTE_CASES),
        "paragraph_hit_rate": round(note_hits / len(NOTE_CASES), 4),
        "avg_hit_chars": round(sum(hit_chars) / len(hit_chars), 1) if hit_chars else 0,
        "chunk_mode": bool(chunk_mode),
    }

    # ---------------- 3) scope 泄漏
    secret_ids = set()
    for content in ["绝密雪狐暗号：开会地点在钟楼", "绝密雪狐暗号：接头时间改到周五"]:
        await engine.remember(content, alpha=0.9, scope="secret")
        row = store.get_by_hash(content_hash(content), "secret")
        if row is not None:
            secret_ids.add(row["id"])
    secret_note = await engine.add_note("绝密雪狐暗号：备用路线沿河道", scope="secret")
    leaks = 0
    checked = 0
    for query in ["雪狐暗号", "钟楼", "河道"]:
        checked += 1
        for c in await engine.recall(query, scope="default", top_k=8,
                                     mark_recalled=False, require_match=True):
            if c.id in secret_ids:
                leaks += 1
        for h in await engine.notes_recall(query, scope="default", top_k=8):
            if h["id"] == secret_note:
                leaks += 1
    result["scope_leak"] = {"checked": checked, "leaks": leaks}

    # ---------------- 4) 写入裁决守卫（仅 v0.2.0）
    if hasattr(store, "adjudicated_write") and hasattr(engine, "_adjudicate_write"):
        result["adjudication"] = await _adjudication_checks(engine, store)
    else:
        result["adjudication"] = None

    # ---------------- 5) 10,000 条检索延迟
    result["latency"] = await _latency_probe(engine, store, embedder, tmp, content_hash)

    conn.close()
    return result


async def _adjudication_checks(engine, store) -> dict:
    """守卫错误计数：每一项一次检查（0 错误 = 通过）。"""
    errors = 0
    checks = 0

    # 守卫一：仅编号不同绝不合并（LLM 想合并也不行）
    v1 = [1.0, 0.0, 0.0, 0.0]
    v2 = [0.85, 0.5268, 0.0, 0.0]
    old_id = store.add_memory("用户的第1条记忆", scope="eval", strength=1.0)
    engine.llm = ScriptedLLM([{"action": "merge", "target_ids": ["1"],
                               "content": "用户的所有记忆", "confidence": 0.99}])
    decision = await engine._adjudicate_write(
        "用户的第10条记忆", "eval", v2, [(old_id, "用户的第1条记忆", v1)], "fact")
    checks += 1
    if decision is not None:
        errors += 1

    # 守卫二：跨域目标不接受
    other = store.add_memory("别的域记忆", scope="other-eval")
    checks += 1
    if store.adjudicated_write(action="update", scope="eval", content="x",
                               target_ids=[other], vec=v1) is not None:
        errors += 1

    # 守卫三：永生（主动）条目不可被取代
    active = store.add_memory("永生条目", scope="eval", is_active=True)
    checks += 1
    if store.adjudicated_write(action="update", scope="eval", content="x",
                               target_ids=[active], vec=v1) is not None:
        errors += 1

    # 守卫四：低置信裁决回退普通新增
    engine.llm = ScriptedLLM([{"action": "update", "target_ids": ["1"],
                               "content": "低置信覆盖", "confidence": 0.1}])
    checks += 1
    if await engine._adjudicate_write("新说法", "eval", v2,
                                      [(old_id, "用户的第1条记忆", v1)], "fact") is not None:
        errors += 1

    # 正向：合法合并必须生效
    engine.llm = ScriptedLLM([{"action": "merge", "target_ids": ["1"],
                               "content": "合并后的用户记忆", "confidence": 0.9}])
    decision = await engine._adjudicate_write("用户的两条记忆", "eval", v2,
                                              [(old_id, "用户的第1条记忆", v1)], "fact")
    checks += 1
    if decision is None or decision["action"] != "merge":
        errors += 1

    return {"checks": checks, "guard_errors": errors}


async def _latency_probe(engine, store, embedder, tmp: Path, content_hash) -> dict:
    """在 10,000 条合成记忆上测检索 p50/p95（不含建库时间）。"""
    from core import db as dbm
    from core.paths import DataPaths
    from core.store import MemoryStore
    from core.vector import pack

    paths = DataPaths(tmp / "pdlat").ensure()
    conn = dbm.connect(paths.db)
    dbm.init_schema(conn)
    store2 = MemoryStore(conn)
    engine.store = store2
    engine._vectors_dirty = True

    now = time.time()
    keywords = [k for k, _ in MEM_CASES]
    rows = []
    vecs = []
    conn.execute("BEGIN")
    for i in range(10000):
        kw = keywords[i % len(keywords)]
        content = f"合成记忆{i}：{kw}相关的一条背景信息"
        mid = f"gen{i}"
        rows.append((mid, content, content_hash(content), now, now,
                     10.0, 1, now, now, "fact", "user", "default", "[]"))
        vecs.append((mid, embedder._vec(content)))
    conn.executemany(
        "INSERT INTO memories (id, content, content_hash, created_at, updated_at, "
        "strength, proof_count, observed_at, valid_from, memory_type, source, scope, tags_json) "
        "VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)", rows)
    conn.executemany(
        "INSERT OR REPLACE INTO vectors(memory_id, dim, vec) VALUES (?,?,?)",
        [(mid, len(v), pack(v)) for mid, v in vecs])
    conn.commit()

    queries = [keywords[i % len(keywords)] for i in range(50)]
    times: list[float] = []
    for q in queries:
        t0 = time.perf_counter()
        await engine.recall(q, scope="default", top_k=8, fast=True,
                            mark_recalled=False, require_match=True)
        times.append((time.perf_counter() - t0) * 1000.0)
    times.sort()
    n = len(times)
    p50 = times[n // 2]
    p95 = times[max(0, math.ceil(n * 0.95) - 1)]
    conn.close()
    return {"corpus": 10000, "queries": n,
            "p50_ms": round(p50, 2), "p95_ms": round(p95, 2)}


def main() -> int:
    ap = argparse.ArgumentParser(description="离线记忆质量评测（v0.2.0 验收）")
    ap.add_argument("--plugin", default=str(Path(__file__).resolve().parents[1]),
                    help="插件根目录（默认脚本所在插件）")
    ap.add_argument("--label", default="", help="本次评测标签（如 v0.1.9 / v0.2.0）")
    ap.add_argument("--out", default="", help="结果 JSON 输出路径（可选）")
    args = ap.parse_args()

    root = Path(args.plugin).resolve()
    if not (root / "core" / "engine.py").exists():
        raise SystemExit(f"不是有效的插件目录：{root}")
    label = args.label or root.name
    report = asyncio.run(run_eval(root, label))
    text = json.dumps(report, ensure_ascii=False, indent=2)
    print(text)
    if args.out:
        Path(args.out).write_text(text + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
