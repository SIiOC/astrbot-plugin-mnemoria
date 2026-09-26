"""嵌入相似度校准工具：用真实嵌入端点测量语义相似度分布，为去重/聚类阈值提供依据。

用法（需 AstrBot 配置里已有可用的 embedding 提供商）：
    python scripts/calibrate_embeddings.py --provider-id <嵌入提供商id>

最近一次实测记录（2026-09-15，nvidia/nemotron-3-embed-1b，2048 维，直连）：
    真重复      均值 1.000  [1.000, 1.000]   → 指纹层拦截
    同义改写    均值 0.747  [0.679, 0.841]   → 低于 dedup 0.92：不误合（保守），
                                                也低于夜间聚类 0.86：同义改写实际
                                                不会被归并，库会冗余但绝不丢事实
    不同事实    均值 0.779  [0.647, 0.875]   → 峰值 0.875 < 0.92 ✅
    模板编号    均值 0.903  [0.784, 0.971]   → ⚠️ 峰值 0.971 > 0.92！
        「完成作业1 vs 完成作业2」类模板内容真实余弦可达 0.97，
        必须依赖 differs_only_by_numbers 结构守卫 + 文本二次确认双防线，
        单靠向量阈值不可能安全。
    无关对照    均值 0.609  [0.575, 0.642]

结论：dedup_threshold=0.92 + 文本守卫是真实嵌入下的正确组合；
调低阈值会导致误合并，调高（>0.97）则模板防线失效——守卫逻辑不可移除。

最近一次实测记录（2026-09-23，dashscope qwen3.7-text-embedding-flash，1024 维）：
    真重复      均值 1.000
    同义改写    均值 0.937  [0.884, 0.955]   → 跨过 dedup 0.92！由文本二次确认
                                                （Jaccard≥0.70）放行小幅换词、
                                                拦截大改写；不同事实不触发，无误合风险
    不同事实    均值 0.714  [0.590, 0.860]   → 峰值 0.860 < 0.92 ✅
    模板编号    均值 0.914  [0.875, 0.956]   → ⚠️ >0.92 的部分照旧靠编号守卫
    无关对照    均值 0.403
与 nemotron 的关键差异：语义近重复的余弦整体上移 ~0.2（北京/上海这对不同事实
0.86）。故 v0.2.11 把 conservative_fallback_similarity 默认 0.85→0.90——
0.85 会在新嵌入器下把 0.86 级的不同事实对误强并（丢新事实，不可逆）。
dedup 0.92 经复核维持：文本守卫与编号守卫兜底的组合在两个嵌入器下都成立。
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import urllib.request
from collections import defaultdict
from pathlib import Path

# AstrBot 主配置：ASTRBOT_CMD_CONFIG 环境变量或 --config 必填。
# Path("") 会归一化成当前目录、.exists() 恒真——用 None 表达"未配置"
# （v0.2.16 审查 D1：空串默认让友好报错成死代码）。
CFG_PATH = None
_env_cfg = os.environ.get("ASTRBOT_CMD_CONFIG", "").strip()
if _env_cfg:
    CFG_PATH = Path(_env_cfg)

PAIRS = [
    ("同义改写", "用户的名字是张三", "用户叫张三"),
    ("同义改写", "用户住在杭州", "用户目前居住在杭州"),
    ("同义改写", "用户养了一只猫叫豆豆", "用户有一只名叫豆豆的猫"),
    ("同义改写", "用户不喜欢吃芹菜", "用户在饮食上排斥芹菜"),
    ("同义改写", "用户是一名后端工程师", "用户从事后端开发工作"),
    ("同义改写", "用户的生日是1998年3月5日", "用户出生于1998年3月5日"),
    ("同义改写", "用户最近在准备高考", "用户眼下正在备考高考"),
    ("同义改写", "用户每天早晨六点跑步", "用户习惯清晨六点跑步"),
    ("不同事实", "用户喜欢蓝色", "用户喜欢跑步"),
    ("不同事实", "用户的名字是张三", "用户的职业是教师"),
    ("不同事实", "用户养了猫", "用户养了狗"),
    ("不同事实", "用户住在北京", "用户住在上海"),
    ("不同事实", "用户不吃辣", "用户爱喝咖啡"),
    ("模板编号", "用户的第1条记忆内容关于编号1", "用户的第10条记忆内容关于编号10"),
    ("模板编号", "用户完成作业1", "用户完成作业2"),
    ("模板编号", "用户的考试排名第3", "用户的考试排名第30"),
    ("无关对照", "用户喜欢猫", "今天股市大幅上涨"),
    ("无关对照", "用户住在杭州", "明朝万历年间修了长城"),
    ("真重复", "用户养了一只猫", "用户养了一只猫"),
]


def load_provider(provider_id: str) -> dict:
    if CFG_PATH is None or not CFG_PATH.exists():
        where = "ASTRBOT_CMD_CONFIG" if CFG_PATH is None else str(CFG_PATH)
        raise SystemExit(
            f"未找到 AstrBot 主配置：{where}\n"
            "请设置 ASTRBOT_CMD_CONFIG 环境变量，或用 --config <path> 指定 cmd_config.json。"
        )
    cfg = json.loads(CFG_PATH.read_text(encoding="utf-8-sig"))
    if not provider_id:
        candidates = [str(g.get("id") or "") for g in cfg.get("provider", [])
                      if g.get("embedding_api_key")]
        raise SystemExit(
            "未指定嵌入提供商。请用 --provider-id 从以下候选中选择：\n  "
            + ("\n  ".join(candidates) if candidates else "（配置里没有带 embedding_api_key 的提供商）")
        )
    for grp in cfg.get("provider", []):
        if grp.get("id") == provider_id:
            if not grp.get("embedding_api_key"):
                raise SystemExit(f"提供商 {provider_id} 没有 API key")
            return grp
    raise SystemExit(f"找不到提供商 {provider_id}")


def embed_openai(base: str, key: str, model: str, texts: list[str]) -> list[list[float]]:
    out: list[list[float]] = []
    for i in range(0, len(texts), 8):
        batch = texts[i:i + 8]
        body = json.dumps({"model": model, "input": batch, "encoding_format": "float"}).encode()
        req = urllib.request.Request(
            f"{base.rstrip('/')}/embeddings", data=body,
            headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=60) as r:
            data = json.loads(r.read())
        out.extend(d["embedding"] for d in sorted(data["data"], key=lambda x: x["index"]))
    return out


def embed_dashscope(base: str, key: str, model: str, texts: list[str]) -> list[list[float]]:
    """阿里云百炼原生 embedding API（base=dashscope 时 OpenAI 兼容路径 404）。

    单请求批量 ≤20 条（超限返回 400 InvalidParameter，见 2026-08-27 实测）。
    """
    out: list[list[float]] = []
    url = base.rstrip("/") + "/services/embeddings/text-embedding/text-embedding"
    for i in range(0, len(texts), 20):
        batch = texts[i:i + 20]
        body = json.dumps({"model": model, "input": {"texts": batch}}).encode()
        req = urllib.request.Request(
            url, data=body,
            headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=60) as r:
            data = json.loads(r.read())
        out.extend(e["embedding"] for e in data["output"]["embeddings"])
    return out


def cos(a: list[float], b: list[float]) -> float:
    d = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(x * x for x in b))
    return d / (na * nb) if na and nb else 0.0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--provider-id", default="",
                    help="嵌入提供商 id（必填；不传时列出配置内可用候选）")
    ap.add_argument("--config", default=None,
                    help="AstrBot 主配置路径（或设 ASTRBOT_CMD_CONFIG 环境变量）")
    args = ap.parse_args()

    global CFG_PATH
    if args.config:
        CFG_PATH = Path(args.config)

    grp = load_provider(args.provider_id)
    texts = sorted({t for _, a, b in PAIRS for t in (a, b)})
    embed_fn = (embed_dashscope if str(grp.get("type", "")).startswith("dashscope")
                else embed_openai)
    vectors = embed_fn(grp["embedding_api_base"], grp["embedding_api_key"],
                       grp["embedding_model"], texts)
    vecs = dict(zip(texts, vectors))
    print(f"模型: {grp['embedding_model']} | 维度: {len(vectors[0])} | 句数: {len(texts)}\n")

    groups: dict[str, list[float]] = defaultdict(list)
    for cat, a, b in PAIRS:
        groups[cat].append(cos(vecs[a], vecs[b]))
    for cat in ("真重复", "同义改写", "不同事实", "模板编号", "无关对照"):
        vals = groups[cat]
        print(f"{cat}: 均值 {sum(vals)/len(vals):.3f} | 范围 [{min(vals):.3f}, {max(vals):.3f}]")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
