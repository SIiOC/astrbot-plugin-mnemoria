"""框架集成冒烟测试：在 AstrBot 真框架下实例化插件并走一遍钩子链路。

运行（必须用 AstrBot 的 venv）：
  python tests/integration_smoke.py

不启动 AstrBot 主程序，只构造最小 Context 替身，验证：
  1. 插件类能被 Star 体系实例化（装饰器注册无冲突）
  2. on_llm_request 注入链路（画像 + 记忆临时块）产物正确
  3. after_message_sent 账本记录
  4. 三个 LLM 工具的 run() 可调用
  5. terminate 能干净收尾
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


# ---------------------------------------------------------------- 假框架替身
class FakeEvent:
    def __init__(self, text="我最近在准备高考，你记一下", sender="u1", group="", session="sess-1"):
        self._text, self._sender, self._group, self._session = text, sender, group, session

    def get_message_str(self):
        return self._text

    def get_sender_id(self):
        return self._sender

    def get_group_id(self):
        return self._group

    def get_session_id(self):
        return self._session


class FakeRequest:
    def __init__(self):
        self.system_prompt = "你是一个助手"
        self.extra_user_content_parts = []
        self.contexts = []


class FakeProviderRequest:
    pass


class FakeResponse:
    completion_text = "好的，我记住了。"


class FakeProviderManager:
    async def get_provider_by_id(self, pid):
        return None

    class llm_tools:
        func_list = []

        @staticmethod
        def remove_func(name):
            pass


class FakeContext:
    def __init__(self):
        self.provider_manager = FakeProviderManager()
        self._tools = []

    def add_llm_tools(self, *tools):
        self._tools.extend(tools)

    def register_web_api(self, *a, **kw):
        pass


def run():
    from astrbot.api.star import Star  # noqa: F401
    from astrbot_plugin_mnemoria.main import MnemoriaPlugin

    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
        # 用 StarTools 替身把数据目录指到临时目录，避免污染真实 plugin_data
        from astrbot.core.star.star_tools import StarTools
        orig = StarTools.get_data_dir
        StarTools.get_data_dir = classmethod(lambda cls, name=None: Path(tmp) / (name or "p"))
        try:
            ctx = FakeContext()
            plugin = MnemoriaPlugin(ctx, config={})

            check("插件实例化", plugin is not None)
            check("注册了 4 个工具", len(ctx._tools) == 4)
            check("工具名正确", {t.name for t in ctx._tools} == {"memory_remember", "memory_recall", "profile_update", "note_create"})
            check("fts5 可用", plugin.fts_ok is True)
            # 直接给引擎塞一个假 LLM，以便测试抽取（真实 config 无 provider）
            check("默认 scope 可解析", plugin.scope_for(FakeEvent())[0] == "default")

            async def flow():
                # 1) on_llm_request：应先记录用户账本
                req = FakeRequest()
                ev = FakeEvent()
                await plugin.inject_memories(ev, req)
                led = plugin.store.recent_ledger("sess-1", limit=5)
                check("on_llm_request 记录用户账本", any(r["role"] == "user" for r in led))

                # 2) 预置一条记忆与画像，确认注入块出现
                await plugin.engine.remember("用户的名字是张三", alpha=0.9, scope="default")
                plugin.store.upsert_profile("default", "u1", "称呼", "小张")
                req2 = FakeRequest()
                await plugin.inject_memories(FakeEvent(text="我叫什么名字"), req2)
                check("画像注入到 system_prompt", "小张" in req2.system_prompt)
                has_mem = any("张三" in getattr(p, "text", "") or (isinstance(p, dict) and "张三" in p.get("text", ""))
                              for p in req2.extra_user_content_parts)
                check("记忆注入为临时块", has_mem)
                check("临时块标 _no_save", all(getattr(p, "_no_save", False) or isinstance(p, dict) for p in req2.extra_user_content_parts))

                # 3) on_llm_response + after_message_sent 记账本
                await plugin.capture_reply(FakeEvent(), FakeResponse())
                await plugin.after_sent(FakeEvent())
                led2 = plugin.store.recent_ledger("sess-1", limit=10)
                check("助手回复入账本", any(r["role"] == "assistant" for r in led2))

                # 4) 工具调用
                tool = {t.name: t for t in ctx._tools}
                ev_tool = FakeEvent()
                ev_tool.mnemoria_engine = plugin.engine
                r = await tool["memory_remember"].run(ev_tool, "用户养了一只猫", "fact")
                check("remember 工具可用", "记住" in r)
                r = await tool["profile_update"].run(ev_tool, "喜好", "猫")
                check("profile 工具可用", "已更新画像" in r)
                r = await tool["memory_recall"].run(ev_tool, "猫", 5)
                check("recall 工具可用", isinstance(r, str) and len(r) > 0)

                # 5) 收尾
                await plugin.terminate()
                check("terminate 无异常", True)

            asyncio.run(flow())
        finally:
            StarTools.get_data_dir = orig


if __name__ == "__main__":
    run()
    print(f"\n通过 {len(PASS)} / 失败 {len(FAIL)}")
    if FAIL:
        print("失败项：", ", ".join(FAIL))
        sys.exit(1)
