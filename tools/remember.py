"""memory_remember：把一条信息写入长期记忆。

v0.1.7 收紧主动记忆判定（此前工具写入一律 is_active=True 永不衰减 +
alpha=1.0 绕过价值门 + source 伪装成 user）：
- retention="normal"（默认）→ 被动记忆，可被衰减/淘汰，反思闭环会给
  真正有用的记忆加分续命；
- retention="permanent" → 主动记忆，永不衰减。仅限「失去它会破坏人格
  或关系连续性」的核心信息：身份与生日、双方的重要约定、用户明确要求
  永远记住的事、关系底线与雷点。
- evidence（可填）→ 落 reasoning 列（angel 的 reasoning 同位），
  供将来回顾「当时为什么记下这条」。
- source 如实标 "tool"，不再伪装成用户亲口（审计可查）。

v0.2.4（对齐 angel_remember 的动作协议）：
- action=create（默认）新增；update 更正 1 条旧记忆；merge 合并多条——
  目标用 memory_recall 返回的短编号（target_ids）指定。update/merge
  是「显式更正」路径：可取代 is_active 条目（旧条入回收站可复活）。
- retention=permanent 默认保持 v0.1.7 语义（模型显式要求即永生）；
  配置 memory_behavior.permanent_allow_tool=false 时自动降为 normal
  并在回复中说明（防模型滥标 permanent 造出永久矛盾——过时记忆
  审查 R7）。v0.2.4 的 update/merge 显式更正本就可取代 is_active
  条目（旧条入回收站可复活），永生不再是不可消除的永久矛盾。
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

from astrbot.api import FunctionTool
from astrbot.api.event import AstrMessageEvent

try:
    from astrbot.api import logger
except ImportError:
    logger = logging.getLogger(__name__)


def _engine_from(event: AstrMessageEvent):
    return getattr(event, "mnemoria_engine", None)


@dataclass
class MemoryRememberTool(FunctionTool):
    name: str = "memory_remember"
    description: str = (
        "把值得长期记住的信息写入记忆库（例如称呼、喜好、约定、关系、重要事件）。"
        "action=create（默认）新增；update=更正一条过时/错误的旧记忆（content 写新说法，"
        "target_ids 填 memory_recall 返回的该条编号，恰 1 个）；merge=把多条讲同一件事的"
        "旧记忆合并为一条（target_ids 填 2~5 个编号）。发现旧记忆已过时或被用户改口时"
        "应优先 update，不要让新旧说法并存。"
        "retention 选 normal（默认，普通记忆，会随时间自然淡忘，有用会被反思机制强化）；"
        "选 permanent 仅限核心信息：用户身份与生日、双方重要约定、用户明确要求永远记住的事、"
        "关系底线与雷点——这类信息失去会破坏关系连续性。"
        "不要记录寒暄、情绪附和、一次性琐事或运维流水。"
    )
    parameters: dict = field(default_factory=lambda: {
        "type": "object",
        "properties": {
            "content": {
                "type": "string",
                "description": "自包含的一句话论断，含具体称呼，如「小明（3141592653）每周四晚上打篮球」。",
                "minLength": 2,
            },
            "action": {
                "type": "string",
                "enum": ["create", "update", "merge"],
                "description": "create=新增（默认）；update=更正 1 条旧记忆；merge=合并 2~5 条旧记忆。",
            },
            "target_ids": {
                "type": "array",
                "items": {"type": "string"},
                "description": "update/merge 的目标记忆编号（memory_recall 返回的方括号短编号）。update 恰 1 个，merge 2~5 个，create 留空。",
            },
            "category": {
                "type": "string",
                "enum": ["fact", "event", "knowledge", "skill", "emotional", "task"],
                "description": "记忆类型，默认 fact。",
            },
            "retention": {
                "type": "string",
                "enum": ["normal", "permanent"],
                "description": (
                    "保存强度。normal=普通记忆（默认）；permanent=永不遗忘，"
                    "只用于身份/生日/重要约定/明确要求永久记住的事/关系底线。"
                ),
            },
            "evidence": {
                "type": "string",
                "description": "支撑这条记忆的原话片段或依据（可空）。写下来便于以后理解当时为什么记。",
            },
            "tags": {
                "type": "array",
                "items": {"type": "string"},
                "description": "检索锚点短词 1~4 个（可空）：人名/ID 等实体锚点 + 场合词（将来提起这条记忆时查询里会出现的词，可不在正文里）。",
            },
        },
        "required": ["content"],
    })

    async def run(
        self,
        event: AstrMessageEvent,
        content: str,
        action: str = "create",
        target_ids: list[str] | None = None,
        category: str = "fact",
        retention: str = "normal",
        evidence: str = "",
        tags: list[str] | None = None,
    ) -> str:
        engine = _engine_from(event)
        if engine is None:
            return "记忆系统未就绪，本次未写入。"
        text = (content or "").strip()
        if not text:
            return "内容为空，未写入。"
        permanent = str(retention or "normal").strip().lower() == "permanent"
        # v0.2.4：permanent 白名单——默认 true 保持 v0.1.7 语义；
        # 配置 false 时工具的 permanent 降为 normal（防模型滥标永生）
        perm_blocked = False
        if permanent:
            try:
                allow = bool(engine.config.get("memory_behavior.permanent_allow_tool", True))
            except Exception:  # noqa: BLE001
                allow = True
            if not allow:
                permanent = False
                perm_blocked = True
        scope, user_key, speaker_key = _identity_of(engine, event)

        # v0.2.4：update/merge 显式更正路径（angel_remember 同款动作协议）
        act = str(action or "create").strip().lower()
        if act in ("update", "merge"):
            need = "恰 1 个" if act == "update" else "2~5 个"
            raw = [str(t).strip() for t in (target_ids or []) if str(t).strip()]
            resolved: list[str] = []
            bad: list[str] = []
            for t in dict.fromkeys(raw):
                mid = engine.resolve_memory_prefix(scope, t) \
                    if hasattr(engine, "resolve_memory_prefix") else None
                (resolved if mid else bad).append(mid or t)
            if (act == "update" and len(resolved) != 1) or \
                    (act == "merge" and not (2 <= len(resolved) <= 5)):
                return (
                    f"{act} 需要{need}有效目标编号，本次解析到 {len(resolved)} 个"
                    + (f"（无法识别：{'、'.join(bad)}）" if bad else "")
                    + "。请先调用 memory_recall 检索，用返回的 [编号] 作为 target_ids。"
                )
            ok = await engine.remember(
                text,
                memory_type=category or "fact",
                alpha=1.0 if permanent else 0.8,
                source="tool",
                scope=scope,
                session_id=event.get_session_id(),
                speaker=user_key,
                speaker_key=speaker_key,
                is_active=permanent,
                reasoning=str(evidence or ""),
                tags=tags if isinstance(tags, list) else None,
                evolution_action=act,
                evolution_ids=resolved,
                # 显式更正可取代 is_active（旧条入回收站可复活）
                allow_active_evolution=True,
            )
            if not ok:
                return "该内容被写入门拒绝（可能是元指令或隐私串）。"
            suffix = "（permanent 未获配置允许，已按普通记忆保存）" if perm_blocked else ""
            return f"已{('更正' if act == 'update' else '合并')}旧记忆。{suffix}"

        ok = await engine.remember(
            text,
            memory_type=category or "fact",
            # 不再无条件 1.0：permanent 顶格，normal 走高价值但留门槛
            alpha=1.0 if permanent else 0.8,
            source="tool",
            scope=scope,
            session_id=event.get_session_id(),
            speaker=user_key,
            speaker_key=speaker_key,
            is_active=permanent,  # v0.1.7：仅显式 permanent 才永不衰减
            reasoning=str(evidence or ""),
            tags=tags if isinstance(tags, list) else None,
        )
        if not ok:
            return "该内容被写入门拒绝（可能是重复、元指令或隐私串）。"
        if perm_blocked:
            return "已记住（普通记忆；permanent 未获配置允许，如需永生请在记忆面板手动设置）。"
        return "已记住（永不遗忘）。" if permanent else "已记住（普通记忆）。"


def _identity_of(engine, event) -> tuple[str, str, str]:
    """返回 (scope, user_key, speaker_key)；新引擎走 identity_for。"""
    try:
        if hasattr(engine, "identity_for"):
            return engine.identity_for(event)
        scope, user_key = engine.scope_for(event)
        return scope, user_key, ""
    except Exception:  # noqa: BLE001
        return "default", "", ""
