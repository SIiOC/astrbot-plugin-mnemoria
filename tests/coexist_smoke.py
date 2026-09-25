"""共存联调测试：验证好想记住你与 Humanizer / 框架其它插件同开时不打架。

运行（必须用 AstrBot 的 venv）：
  python tests/coexist_smoke.py

关注点（M5）：
  1. 钩子优先级排序：数字大者先执行 → 本插件 on_llm_response(-100) 必须最后跑，
     这样捕获到的是 Humanizer 深度改写后的最终文本。
  2. 注入互不覆盖：Humanizer 与好想记住你都往 system_prompt 追加、都往
     extra_user_content_parts 里放内容块；验证两者内容同时存在。
  3. 账本记录的是最终回复（而非改写前的原始回复）。
"""

from __future__ import annotations

import asyncio
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from _bootstrap import bootstrap  # noqa: E402

bootstrap()

PASS, FAIL = [], []


def check(name, cond):
    (PASS if cond else FAIL).append(name)
    print(("  ok  " if cond else "FAIL  ") + name)


class FakeEvent:
    def __init__(self, text="我最近在准备高考", session="s1", sender="u1"):
        self._text, self._session, self._sender = text, session, sender

    def get_message_str(self):
        return self._text

    def get_sender_id(self):
        return self._sender

    def get_group_id(self):
        return ""

    def get_session_id(self):
        return self._session


class FakeRequest:
    def __init__(self):
        self.system_prompt = "你是助手"
        self.extra_user_content_parts = []


def test_priority_ordering():
    """验证 handler 排序语义：priority 数字大者先执行。"""
    print("[priority ordering]")
    from astrbot.core.star.star_handler import StarHandlerRegistry

    reg = StarHandlerRegistry()

    class FakeHandler:
        def __init__(self, name, pri):
            self.handler_full_name = name
            self.extras_configs = {"priority": pri}

    # 模拟：Humanizer 默认 0，好想记住你 -100
    reg.append(FakeHandler("humanizer.on_llm_response", 0))
    reg.append(FakeHandler("mnemoria.on_llm_response", -100))
    order = [h.handler_full_name for h in reg._handlers]
    check("优先级高者排前（0 在 -100 前）", order.index("humanizer.on_llm_response") < order.index("mnemoria.on_llm_response"))
    check("故好想记住你最后执行（捕获改写后文本）", order[-1] == "mnemoria.on_llm_response")


def test_injection_coexist():
    """两个插件的注入互不覆盖。"""
    print("[injection coexist]")
    from astrbot.core.star.star_tools import StarTools

    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
        orig = StarTools.get_data_dir
        StarTools.get_data_dir = classmethod(lambda cls, name=None: Path(tmp) / (name or "p"))

        class Ctx:
            class provider_manager:
                @staticmethod
                async def get_provider_by_id(p):
                    return None

            def add_llm_tools(self, *t):
                pass

            def register_web_api(self, *a, **k):
                pass

        from astrbot_plugin_mnemoria.main import MnemoriaPlugin
        plugin = MnemoriaPlugin(Ctx(), config={})
        StarTools.get_data_dir = orig

        async def flow():
            plugin.store.upsert_profile("default", "u1", "称呼", "小张")
            await plugin.engine.remember("用户的生日是三月五日", alpha=0.9, scope="default")

            req = FakeRequest()
            # 模拟 Humanizer 先注入（它是默认优先级 0，先跑）
            from astrbot.core.agent.message import TextPart
            req.system_prompt += "\n<说话风格>口语化，别用书面语</说话风格>"
            req.extra_user_content_parts.append(TextPart(text="<style_hint>语气随意</style_hint>").mark_as_temp())

            # 好想记住你注入（priority 45 → 更早，但此处直接调用模拟合并结果）
            await plugin.inject_memories(FakeEvent(), req)

            check("Humanizer 系统提示保留", "说话风格" in req.system_prompt)
            check("好想记住你画像也进了系统提示", "小张" in req.system_prompt)
            texts = []
            for p in req.extra_user_content_parts:
                t = getattr(p, "text", None) or (p.get("text") if isinstance(p, dict) else "")
                if t:
                    texts.append(t)
            joined = "\n".join(texts)
            check("Humanizer 风格块保留", "style_hint" in joined)
            check("好想记住你记忆块也在", "三月五日" in joined)
            check("两条块都未被覆盖", len(req.extra_user_content_parts) >= 2)

            # 账本记的是最终（改写后）文本
            class Resp:
                completion_text = "嗯，好好准备，别太累啦"
            await plugin.capture_reply(FakeEvent(), Resp())
            await plugin.after_sent(FakeEvent())
            rows = plugin.store.recent_ledger("s1", limit=10)
            finals = [r["content"] for r in rows if r["role"] == "assistant"]
            check("账本记录最终回复", any("别太累" in c for c in finals))

            await plugin.terminate()

        asyncio.run(flow())


def test_no_tool_name_conflict():
    """工具名与其他插件不冲突（angel_memory/其它记忆插件的工具名）。"""
    print("[tool name conflicts]")
    from astrbot_plugin_mnemoria.tools import MemoryRecallTool, MemoryRememberTool, ProfileUpdateTool
    names = {MemoryRememberTool().name, MemoryRecallTool().name, ProfileUpdateTool().name}
    # angel_memory 用 angel_remember / angel_recall / angel_note_*；Humanizer 用 wiki_* 等
    foreign = {"angel_remember", "angel_recall", "angel_note_read", "angel_note_create"}
    check("与 angel 工具名无冲突", not (names & foreign))
    check("工具名统一 memory_/profile_ 前缀", all(n.startswith(("memory_", "profile_")) for n in names))


if __name__ == "__main__":
    test_priority_ordering()
    test_no_tool_name_conflict()
    test_injection_coexist()
    print(f"\n通过 {len(PASS)} / 失败 {len(FAIL)}")
    if FAIL:
        print("失败项：", ", ".join(FAIL))
        sys.exit(1)
