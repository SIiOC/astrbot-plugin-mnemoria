"""纯函数：热度衰减与三档评分。

热度公式（OpenViking 系，唯一被代码证实的实装写法）：
    hotness = sigmoid(log1p(hit_count)) * 2 ** (-age_days / half_life)
其中 age 以「最后一次被召回 / 最后一次衰减 / 创建」三者中最晚者为锚点，
使「重新被想起」重置时钟（access 延长半衰期）。

三档（angel 骨架的简化实装）：
    T0 useful_score < tier0     自然遗忘档（按热度扣 strength）
    T1 tier0 <= score < tier1   过渡档（当前与 T0 同样按热度衰减）
    T2 score >= tier1           长期保留（不再自然遗忘）

与 angel 的差异：useful_score 的降分（召回判"无用"）由 store.penalize 实现，
由夜间反思闭环（engine.reflect_session）调用——本模块只做纯计算，不感知调用方。
"""

from __future__ import annotations

import math

DAY = 86400.0


def sigmoid(x: float) -> float:
    if x >= 0:
        z = math.exp(-x)
        return 1.0 / (1.0 + z)
    z = math.exp(x)
    return z / (1.0 + z)


def hotness(hit_count: int, anchor_ts: float, now_ts: float, half_life_days: float) -> float:
    """返回 [0,1] 热度分。"""
    hl = max(1e-6, float(half_life_days))
    age_days = max(0.0, (now_ts - float(anchor_ts)) / DAY)
    decay = 2.0 ** (-age_days / hl)
    return sigmoid(math.log1p(max(0, int(hit_count)))) * decay


def anchor_ts(last_recalled_at: float, last_decay_at: float, created_at: float) -> float:
    """衰减锚点 = 三者中最晚者（重新被提起则时钟重置）。"""
    return max(float(last_recalled_at or 0), float(last_decay_at or 0), float(created_at or 0))


def tier(useful_score: float, tier0: float, tier1: float) -> int:
    if useful_score >= tier1:
        return 2
    if useful_score >= tier0:
        return 1
    return 0


def effective_half_life(base_days: float, hit_count: int, per_hit_bonus: float = 0.15) -> float:
    """访问延长半衰期：命中越多衰减越慢，上限 4 倍。"""
    return float(base_days) * min(4.0, 1.0 + max(0, int(hit_count)) * per_hit_bonus)


# 分型 TTL（livingmemory compute_ttl 同思想）：不同类型的记忆天然寿命不同。
# 权重乘在衰减损失上：<1 更耐久（知识/事实），>1 消退更快（任务是一次性的）。
DEFAULT_TYPE_WEIGHTS = {
    "fact": 0.7,
    "knowledge": 0.6,
    "skill": 0.6,
    "event": 1.0,
    "emotional": 0.8,
    "task": 2.0,
}


def type_weight(memory_type: str, weights: dict | None) -> float:
    """查某记忆类型的衰减权重，未知类型按 1.0。"""
    if not weights:
        return DEFAULT_TYPE_WEIGHTS.get(memory_type, 1.0)
    try:
        return max(0.1, min(5.0, float(weights.get(memory_type, 1.0))))
    except (TypeError, ValueError, AttributeError):
        return 1.0
