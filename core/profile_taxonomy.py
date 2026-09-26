"""画像维度分类（v0.2.1，借鉴 angel_memory 的固定画像体系）。

angel 的做法：画像只允许 5 个固定维度标签，配「画像判定规则 / 空画像硬底线 /
冲突修正」三条规则，从机制上杜绝「同一维度换着写」造成的键漂移。

本模块把该体系移植到 mnemoria：
- ``CANONICAL_ATTRS``：固定五维（写入与展示都收敛到这里）；
- ``SYNONYM_MAP``：历史键 → 固定维度的同义映射。只映射语义明确的键，
  刻意不含「名字/姓名」——它们常被助手侧人设占用（v0.1.5 事故教训）；
- ``normalize_profile_key`` 供写入路径（抽取/工具）归一；
- ``aggregate_profile_rows`` 供展示路径（注入块/抽取提示词）按维度聚合，
  把历史漂移的多行同义键合并成一行，省注入预算、防重复维度。
"""

from __future__ import annotations

#: 固定画像维度（与模板提示词、迁移脚本共用同一份定义）
CANONICAL_ATTRS: tuple[str, ...] = ("用户别名", "事实属性", "技能树", "关系图谱", "活跃项目")

#: 历史键 → 固定维度。
#: 映射原则：语义明确才映射，拿不准或可能被助手侧占用的键保持原样。
_SYNONYM_MAP: dict[str, str] = {
    # —— 用户别名：称呼类（不含「名字/姓名」，那是助手人设的历史占用键）——
    "称呼": "用户别名",
    "昵称": "用户别名",
    "别名": "用户别名",
    "常用称呼": "用户别名",
    "用户称呼": "用户别名",
    "称谓": "用户别名",
    "用户昵称": "用户别名",
    # —— 技能树 ——
    "技能": "技能树",
    "能力": "技能树",
    "专长": "技能树",
    "技能特长": "技能树",
    "擅长": "技能树",
    # —— 关系图谱 ——
    "关系": "关系图谱",
    "关系类型": "关系图谱",
    "关系状态": "关系图谱",
    "关系偏好": "关系图谱",
    "关系行为": "关系图谱",
    "关系/情感偏好": "关系图谱",
    "人际": "关系图谱",
    "社交": "关系图谱",
    "互动偏好": "关系图谱",
    # —— 活跃项目 ——
    "项目": "活跃项目",
    "任务": "活跃项目",
    "计划": "活跃项目",
    "长期项目": "活跃项目",
    "业务动态": "活跃项目",
    "研究方向": "活跃项目",
    "长期任务": "活跃项目",
    # —— 事实属性（最大类：兴趣/习惯/状态/背景等稳定事实）——
    "喜好": "事实属性",
    "兴趣": "事实属性",
    "偏好": "事实属性",
    "雷点": "事实属性",
    "忌讳": "事实属性",
    "作息": "事实属性",
    "作息习惯": "事实属性",
    "健康情况": "事实属性",
    "健康状况": "事实属性",
    "健康状态": "事实属性",
    "生活状态": "事实属性",
    "生活习惯": "事实属性",
    "使用习惯": "事实属性",
    "使用偏好": "事实属性",
    "内容偏好": "事实属性",
    "信息": "事实属性",
    "情绪": "事实属性",
    "情绪状态": "事实属性",
    "性格": "事实属性",
    "性格特点": "事实属性",
    "家庭背景": "事实属性",
    "学习状态": "事实属性",
    "学业情况": "事实属性",
    "学业": "事实属性",
    "学习情况": "事实属性",
    "职业": "事实属性",
    "所在地": "事实属性",
    "年龄": "事实属性",
    "生日": "事实属性",
    "关键关注": "事实属性",
    "关注点": "事实属性",
    "兴趣与投入": "事实属性",
    "喜好/兴趣": "事实属性",
    "空闲时间": "事实属性",
    "身份": "事实属性",
    # —— 线上实测的其余漂移键（2026-09-20 归并前扫描发现）——
    "业余活动": "事实属性",
    "互动习惯": "事实属性",
    "互动关注": "事实属性",
    "互动风格": "事实属性",
    "健康": "事实属性",
    "关系/互动偏好": "事实属性",
    "关系互动": "事实属性",
    "创作偏好": "事实属性",
    "喜好/性趣": "事实属性",
    "工具偏好": "事实属性",
    "性趣": "事实属性",
    "活动": "事实属性",
    "爱好": "事实属性",
    "行为": "事实属性",
    "角色投入": "事实属性",
    "身份偏好": "事实属性",
    "饮食偏好": "事实属性",
    "技术活动": "活跃项目",
    "技术需求": "活跃项目",
    "技能/兴趣": "技能树",
}


def normalize_profile_key(key: str) -> str:
    """把画像键归一到固定维度；未知键原样返回（不猜测、不误归类）。"""
    k = str(key or "").strip()
    if not k or k in CANONICAL_ATTRS:
        return k
    return _SYNONYM_MAP.get(k, k)


def is_canonical(key: str) -> bool:
    return str(key or "").strip() in CANONICAL_ATTRS


def aggregate_profile_rows(rows, *, limit: int = 12, max_value: int = 240,
                           include_updated: bool = False) -> list:
    """把画像行按固定维度聚合为 [(维度, "值1；值2"), ...]。

    - 保序：按传入行序（调用方按 confidence/updated 排序）首次出现的维度在前；
    - 去重：相同值只保留一次；
    - 限长：单维度值拼接超过 max_value 截断加省略号（省注入预算）；
    - 未知键不合并，单独成行。
    - include_updated=True（v0.2.4）：返回 (维度, 值, 该维度最近 updated_at)
      三元组，供注入块标注「更新于 N 天前」；默认 False 保持二元组契约。
    """
    agg: dict[str, list[str]] = {}
    latest: dict[str, float] = {}
    for row in rows:
        try:
            attr = normalize_profile_key(row["key"])
            value = str(row["value"] or "").strip()
        except (TypeError, IndexError, KeyError):
            continue
        if not attr or not value:
            continue
        bucket = agg.setdefault(attr, [])
        if value not in bucket:
            bucket.append(value)
        try:
            ts = float(row["updated_at"] or 0.0)
        except (TypeError, ValueError, IndexError, KeyError):
            ts = 0.0
        if ts > latest.get(attr, 0.0):
            latest[attr] = ts
    out: list = []
    for attr, values in list(agg.items())[: max(1, int(limit))]:
        text = "；".join(values)
        if len(text) > max_value:
            text = text[: max(0, max_value - 1)] + "…"
        if include_updated:
            out.append((attr, text, latest.get(attr, 0.0)))
        else:
            out.append((attr, text))
    return out
