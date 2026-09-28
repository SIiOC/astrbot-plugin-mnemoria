"""memory_recall：主动检索长期记忆。"""

from __future__ import annotations

from astrbot.api import logger
from dataclasses import dataclass, field

from astrbot.api import FunctionTool
from astrbot.api.event import AstrMessageEvent



@dataclass
class MemoryRecallTool(FunctionTool):
    name: str = "memory_recall"
    description: str = (
        "检索长期记忆与历史对话。当需要回忆用户此前说过的事、身份信息或历史约定时调用。"
        "每条记忆前的方括号编号（如 [a1b2c3]）可在 memory_remember 的 update/merge 动作中引用。"
    )
    parameters: dict = field(default_factory=lambda: {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "检索关键字，可用空格分隔多个词。",
                "minLength": 1,
            },
            "limit": {
                "type": "integer",
                "description": "返回条数上限，默认 6。",
            },
        },
        "required": ["query"],
    })

    async def run(self, event: AstrMessageEvent, query: str, limit: int = 6) -> str:
        engine = getattr(event, "mnemoria_engine", None)
        if engine is None:
            return "记忆系统未就绪。"
        q = (query or "").strip()
        if not q:
            return "查询为空。"
        try:
            scope, user_key = engine.scope_for(event)
        except Exception:  # noqa: BLE001
            scope, user_key = "default", ""
        try:
            n = max(1, min(20, int(limit)))
        except (TypeError, ValueError):
            n = 6
        cands = await engine.recall(q, scope=scope, top_k=n, token_budget=1200)
        try:
            from ..core.text import sanitize_for_context
            from ..core.engine import rel_time_label
        except ImportError:  # 测试环境以顶层包导入（插件目录直接入 sys.path）
            from core.text import sanitize_for_context
            from core.engine import rel_time_label
        lines: list[str] = []
        if cands:
            lines.append("【长期记忆】")
            # v0.2.4：带短编号（angel 短 ID 同款），供 memory_remember 的
            # update/merge 动作引用；同时带相对时间，便于判断新旧
            lines.extend(
                f"- [{str(c.id)[:6]}] {sanitize_for_context(c.content)}"
                f"{rel_time_label(getattr(c, 'created_at', 0.0))}"
                for c in cands
            )
        session_id = event.get_session_id()
        ledger = engine.store.search_ledger(
            q, session_id=session_id, limit=n, scope=scope, role="assistant"
        )
        if not ledger:
            # 允许同一隔离域跨会话回退，但不得省略 scope，避免账本跨域泄漏。
            ledger = engine.store.search_ledger(
                q, limit=n, scope=scope, role="assistant"
            )
        if ledger:
            lines.append("【历史对话片段】")
            for r in ledger:
                who = "用户" if r["role"] == "user" else "助手"
                lines.append(f"- {who}：{sanitize_for_context(r['content'])}")
        return "\n".join(lines) if lines else "没有检索到相关记忆。"
