"""存量记忆向量回填（v0.2.14）。

默认 dry-run，仅展示将要处理的条目；加 ``--apply`` 才写库。

用法：
    python scripts/reembed_vectors.py --db <mnemoria.db> --config <cmd_config.json>
    python scripts/reembed_vectors.py --db <...> --config <...> --limit 20 --apply

脚本直接复用 AstrBot 的 embedding provider 配置与 HTTP 兼容调用，
不启动插件、不触碰回收站；apply 前将 vectors 表复制到同库
``vectors_backup``（若已存在则保留，不覆盖旧快照）。
"""

from __future__ import annotations

import argparse
import json
import logging
import sqlite3
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from core import db as dbm  # noqa: E402
from core.bridge import _call_embed  # noqa: E402

logger = logging.getLogger(__name__)


def _load_provider(config_path: Path, provider_id: str) -> dict:
    data = json.loads(config_path.read_text(encoding="utf-8-sig"))
    for group in data.get("provider", []) or []:
        if str(group.get("id") or "") == provider_id:
            return group
    raise SystemExit(f"找不到嵌入提供商：{provider_id}")


async def _embed(provider, texts: list[str]) -> list[list[float]] | None:
    return await _call_embed(provider, texts)


def _make_provider(config_path: Path, provider_id: str):
    """脚本使用框架 provider 实例时由调用方注入；HTTP fallback 留给 CLI runner。"""
    return _load_provider(config_path, provider_id)


def _http_embed(group: dict, texts: list[str]) -> list[list[float]]:
    """同步 HTTP fallback，支持 DashScope 原生与 OpenAI 兼容端点。"""
    import urllib.request

    base = str(group.get("embedding_api_base") or "").rstrip("/")
    key = str(group.get("embedding_api_key") or "")
    model = str(group.get("embedding_model") or "")
    kind = str(group.get("type") or "").lower()
    out: list[list[float]] = []
    if kind.startswith("dashscope"):
        url = base + "/services/embeddings/text-embedding/text-embedding"
        for i in range(0, len(texts), 20):
            batch = texts[i:i + 20]
            body = json.dumps({"model": model, "input": {"texts": batch}}).encode()
            req = urllib.request.Request(
                url, data=body,
                headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
            )
            with urllib.request.urlopen(req, timeout=90) as resp:
                data = json.loads(resp.read())
            out.extend(e["embedding"] for e in data["output"]["embeddings"])
        return out
    url = base + "/embeddings"
    for i in range(0, len(texts), 8):
        batch = texts[i:i + 8]
        body = json.dumps({"model": model, "input": batch, "encoding_format": "float"}).encode()
        req = urllib.request.Request(
            url, data=body,
            headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=90) as resp:
            data = json.loads(resp.read())
        out.extend(x["embedding"] for x in sorted(data["data"], key=lambda x: x["index"]))
    return out


def plan(conn: sqlite3.Connection, current_dim: int | None = None, limit: int = 0) -> list[sqlite3.Row]:
    """活跃且向量缺失/异维的条目；优先从未召回、再按最老创建时间。"""
    clauses = ["m.deleted_at IS NULL", "(v.memory_id IS NULL"]
    params: list = []
    if current_dim:
        clauses[-1] += " OR v.dim<>?"
        params.append(int(current_dim))
    clauses[-1] += ")"
    sql = (
        "SELECT m.id, m.content, m.created_at, m.hit_count, v.dim AS old_dim "
        "FROM memories m LEFT JOIN vectors v ON v.memory_id=m.id "
        "WHERE " + " AND ".join(clauses) + " "
        "ORDER BY CASE WHEN COALESCE(m.hit_count,0)=0 THEN 0 ELSE 1 END, "
        "m.created_at ASC, m.id ASC"
    )
    if limit:
        sql += " LIMIT ?"
        params.append(max(1, int(limit)))
    return conn.execute(sql, params).fetchall()


def ensure_backup(conn: sqlite3.Connection) -> None:
    conn.execute(
        "CREATE TABLE IF NOT EXISTS vectors_backup AS SELECT * FROM vectors WHERE 0"
    )
    count = conn.execute("SELECT COUNT(*) FROM vectors_backup").fetchone()[0]
    if count == 0:
        conn.execute("INSERT INTO vectors_backup SELECT * FROM vectors")
        conn.commit()


def write_report(log_dir: Path, report: dict) -> Path:
    log_dir.mkdir(parents=True, exist_ok=True)
    path = log_dir / f"reembed-{time.strftime('%Y%m%d-%H%M%S')}.json"
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def main() -> int:
    ap = argparse.ArgumentParser(description="存量记忆向量回填（默认 dry-run）")
    ap.add_argument("--db", required=True, help="mnemoria.db 路径")
    ap.add_argument("--config", required=True, help="AstrBot cmd_config.json 路径")
    ap.add_argument("--provider-id", default="", help="嵌入 provider id；默认从插件配置读取")
    ap.add_argument("--plugin-config", default="", help="插件配置 JSON；默认与 db 同目录向上找不到时留空")
    ap.add_argument("--dim", type=int, default=0, help="当前模型维度；0=首批探测")
    ap.add_argument("--apply", action="store_true", help="真正写入")
    ap.add_argument("--limit", type=int, default=0, help="最多处理 K 条；0=全部")
    ap.add_argument("--batch", type=int, default=32, help="批量大小（DashScope 会再按 20 分片）")
    ap.add_argument("--retry", type=int, default=3, help="单批失败重试次数")
    args = ap.parse_args()

    db = Path(args.db)
    cfg_path = Path(args.config)
    if not db.exists():
        raise SystemExit(f"数据库不存在：{db}")
    if not cfg_path.exists():
        raise SystemExit(f"主配置不存在：{cfg_path}")
    conn = dbm.connect(db)
    try:
        plugin_cfg_path = Path(args.plugin_config) if args.plugin_config else (
            cfg_path.parent / "config" / "astrbot_plugin_mnemoria_config.json"
        )
        provider_id = args.provider_id
        if not provider_id and plugin_cfg_path.exists():
            pconf = json.loads(plugin_cfg_path.read_text(encoding="utf-8-sig"))
            provider_id = str(pconf.get("retrieval", {}).get("embedding_provider_id") or "")
        if not provider_id:
            raise SystemExit("未指定嵌入 provider：请传 --provider-id 或 --plugin-config")
        group = _make_provider(cfg_path, provider_id)
        current_dim = args.dim or int(group.get("embedding_dimensions") or 0)
        rows = plan(conn, current_dim=current_dim or None, limit=args.limit)
        print(f"将重算 {len(rows)} 条（旧维度混合/缺失 → 当前 provider={provider_id}）")
        for row in rows[:5]:
            print(f"  - {row['id']} old_dim={row['old_dim'] or 0}: {str(row['content'] or '')[:80]}")
        if not args.apply:
            print("dry-run：未写库。确认抽样后加 --apply 执行。")
            return 0

        ensure_backup(conn)
        processed = failed = 0
        failed_ids: list[str] = []
        for start in range(0, len(rows), max(1, int(args.batch))):
            chunk = rows[start:start + max(1, int(args.batch))]
            texts = [str(r["content"] or "") for r in chunk]
            vectors = None
            error = ""
            for attempt in range(max(1, int(args.retry))):
                try:
                    vectors = _http_embed(group, texts)
                    if vectors and len(vectors) == len(chunk):
                        break
                    error = f"返回数量异常：{len(vectors or [])}/{len(chunk)}"
                except Exception as exc:  # noqa: BLE001
                    error = str(exc)
                if attempt + 1 < max(1, int(args.retry)):
                    time.sleep(2 ** attempt)
            if not vectors or len(vectors) != len(chunk):
                failed += len(chunk)
                failed_ids.extend(str(r["id"]) for r in chunk)
                print(f"批次失败（{len(chunk)} 条）：{error}", file=sys.stderr)
                continue
            for row, vec in zip(chunk, vectors):
                if not vec:
                    failed += 1
                    failed_ids.append(str(row["id"]))
                    continue
                conn.execute(
                    "INSERT INTO vectors(memory_id, dim, vec) VALUES(?,?,?) "
                    "ON CONFLICT(memory_id) DO UPDATE SET dim=excluded.dim, vec=excluded.vec",
                    (str(row["id"]), len(vec), __import__("core.vector", fromlist=["pack"]).pack(vec)),
                )
                processed += 1
            conn.commit()
            print(f"进度：{min(start + len(chunk), len(rows))}/{len(rows)}")
        report = {
            "created_at": time.time(), "provider_id": provider_id,
            "processed": processed, "failed": failed, "failed_ids": failed_ids,
            "planned": len(rows), "backup_table": "vectors_backup",
        }
        report_path = write_report(db.parent / "logs", report)
        print(f"完成：处理 {processed}，失败 {failed}；报告：{report_path}")
        return 0 if not failed else 2
    finally:
        conn.close()


if __name__ == "__main__":
    raise SystemExit(main())
