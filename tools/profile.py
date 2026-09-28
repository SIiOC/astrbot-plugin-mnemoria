"""profile_update：写入用户画像（永不衰减）。"""

from __future__ import annotations

from astrbot.api import logger
from dataclasses import dataclass, field

from astrbot.api import FunctionTool
from astrbot.api.event import AstrMessageEvent



@dataclass
class ProfileUpdateTool(FunctionTool):
    name: str = "profile_update"
    description: str = (
        "更新对用户的稳定认知。key 只能从五个固定维度里选："
        "用户别名（稳定称呼/代号）、事实属性（兴趣/偏好/雷点/作息/健康/职业等稳定事实）、"
        "技能树（技能与工具）、关系图谱（稳定关系）、活跃项目（正在推进的事）。"
        "同一维度用同一个 key；用户别名写最新称呼，其余维度只写本次新增的"
        "具体事实（系统会自动与旧值合并去重）。这些信息会长期保留，不会随时间遗忘。"
    )
    parameters: dict = field(default_factory=lambda: {
        "type": "object",
        "properties": {
            "key": {
                "type": "string",
                "enum": ["用户别名", "事实属性", "技能树", "关系图谱", "活跃项目"],
                "description": "画像维度（固定五选一）。",
            },
            "value": {"type": "string", "description": "该维度的简洁取值。", "minLength": 1},
            "confidence": {"type": "number", "description": "置信度 0~1，默认 0.8。"},
        },
        "required": ["key", "value"],
    })

    async def run(self, event: AstrMessageEvent, key: str, value: str, confidence: float = 0.8) -> str:
        engine = getattr(event, "mnemoria_engine", None)
        if engine is None:
            return "记忆系统未就绪。"
        k = (key or "").strip()
        v = (value or "").strip()
        if not k or not v:
            return "参数不完整。"
        try:
            scope, user_key = engine.scope_for(event)
        except Exception:  # noqa: BLE001
            scope, user_key = "default", ""
        if not user_key:
            return "无法确定用户身份，未写入画像。"
        try:
            conf = max(0.0, min(1.0, float(confidence)))
        except (TypeError, ValueError):
            conf = 0.8
        # v0.2.1（借鉴 angel 固定画像体系）：统一入口做维度归一 +
        # 别名覆盖/其余维度合并 + 旧同义键清理。
        canonical = engine.write_profile(scope, user_key, k, v, confidence=conf)
        return f"已更新画像：{canonical} = {v}"
