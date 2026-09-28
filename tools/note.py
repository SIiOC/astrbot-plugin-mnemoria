"""note_create：把值得长期留存的知识/设定写成一条笔记。"""

from __future__ import annotations

from astrbot.api import logger
from dataclasses import dataclass, field

from astrbot.api import FunctionTool
from astrbot.api.event import AstrMessageEvent



@dataclass
class NoteCreateTool(FunctionTool):
    name: str = "note_create"
    description: str = (
        "整理并保存一条知识笔记（与记忆不同：记忆是事实，笔记是可查阅的知识条目，"
        "如设定、攻略、要点总结）。当用户在交流中形成可复用的知识时调用。"
    )
    parameters: dict = field(default_factory=lambda: {
        "type": "object",
        "properties": {
            "content": {
                "type": "string",
                "description": "笔记正文，自包含、可独立阅读。",
                "minLength": 2,
            },
            "title": {
                "type": "string",
                "description": "简短标题，便于检索定位。",
            },
            "tags": {
                "type": "string",
                "description": "逗号分隔的标签（可选）。",
            },
        },
        "required": ["content"],
    })

    async def run(self, event: AstrMessageEvent, content: str, title: str = "",
                  tags: str = "") -> str:
        engine = getattr(event, "mnemoria_engine", None)
        if engine is None:
            return "记忆系统未就绪，本次未写入笔记。"
        if not engine.config.get("notes.enabled", True):
            return "笔记知识库当前未启用。"
        text = (content or "").strip()
        if not text:
            return "内容为空，未写入。"
        try:
            scope, _ = engine.scope_for(event)
        except Exception:  # noqa: BLE001
            scope = "default"
        nid = await engine.add_note(text, title=title, tags=tags, source="ai", scope=scope)
        return "笔记已保存。" if nid else "笔记未写入（内容为空）。"
