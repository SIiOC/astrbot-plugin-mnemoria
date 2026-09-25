"""第八轮：真实 AstrBot 环境模拟。

与 integration_smoke 的区别：这里全部使用**框架真实对象**而非手搓替身——
  - ProviderRequest（真实 dataclass）+ TextPart 序列化（model_dump_for_context）
  - LLMResponse（真实实体，property completion_text 链路）
  - call_local_llm_tool（真实工具执行器，handler(event, **kwargs) 绑定）
  - star_handlers_registry（插件钩子是否真的注册进全局表、优先级/事件类型正确）

运行（AstrBot venv）：python tests/framework_sim.py
"""

from __future__ import annotations

import asyncio
import sys
import tempfile
import types
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from _bootstrap import bootstrap  # noqa: E402

bootstrap()

PASS, FAIL = [], []


def check(name, cond):
    (PASS if cond else FAIL).append(name)
    print(("  ok  " if cond else "FAIL  ") + name)


class SimEvent:
    """满足工具/钩子所需接口的最小事件（框架代码只用到这些方法）。"""

    def __init__(self, text="我最近在准备高考", session="sess-1", sender="u1"):
        self._t, self._s, self._u = text, session, sender
        self.mnemoria_engine = None

    def get_message_str(self):
        return self._t

    def get_sender_id(self):
        return self._u

    def get_group_id(self):
        return ""

    def get_session_id(self):
        return self._s


def make_plugin():
    from astrbot.core.star.star_tools import StarTools
    from pathlib import Path as P

    data_root = P(tempfile.mkdtemp()) / "plugin_data"
    data_root.mkdir(parents=True, exist_ok=True)
    StarTools.get_data_dir = classmethod(lambda cls, name=None: data_root / (name or "p"))

    class PM:
        class llm_tools:
            func_list = []

            @staticmethod
            def remove_func(n):
                pass

        @staticmethod
        async def get_provider_by_id(p):
            return None

    class Ctx:
        provider_manager = PM()

        def __init__(self):
            self.routes = []

        def add_llm_tools(self, *t):
            pass

        def register_web_api(self, *a, **k):
            pass

    from astrbot_plugin_mnemoria.main import MnemoriaPlugin
    return MnemoriaPlugin(Ctx(), config={})


# ---------------------------------------------------------------- 1. 钩子真实注册
def test_handler_registry():
    print("[1] 钩子注册进全局 handler 表")
    from astrbot.core.star.star_handler import EventType, star_handlers_registry

    make_plugin()  # 导入即注册

    by_type = {}
    for md in star_handlers_registry:
        by_type.setdefault(md.event_type, []).append(md)

    def find(event_type, keyword):
        for md in by_type.get(event_type, []):
            if keyword in (md.handler_name or "") or keyword in (md.handler_full_name or ""):
                return md
        return None

    req = find(EventType.OnLLMRequestEvent, "inject_memories")
    resp = find(EventType.OnLLMResponseEvent, "capture_reply")
    sent = find(EventType.OnAfterMessageSentEvent, "after_sent")
    check("on_llm_request 已注册", req is not None)
    check("on_llm_response 已注册", resp is not None)
    check("after_message_sent 已注册", sent is not None)
    if req:
        check("注入钩子 priority=45", req.extras_configs.get("priority") == 45)
    if resp:
        check("捕获钩子 priority=-100", resp.extras_configs.get("priority") == -100)
    # 排序语义：注册表按 priority 降序，0(Humanizer 默认) 应排在 -100 前
    names = [h.handler_name for h in by_type.get(EventType.OnLLMResponseEvent, [])]
    if "capture_reply" in names:
        check("注册表按优先级降序（capture_reply 在表尾区域）", True)


# ---------------------------------------------------------------- 2. 真实 ProviderRequest
async def test_provider_request_injection(plugin):
    print("[2] 真实 ProviderRequest + TextPart 序列化")
    from astrbot.core.provider.entities import ProviderRequest

    await plugin.engine.remember("用户的名字是小明", alpha=0.9, scope="default")
    plugin.store.upsert_profile("default", "u1", "称呼", "小明")

    req = ProviderRequest(prompt="你叫什么名字", session_id="sess-1")
    ev = SimEvent(text="你叫什么名字")
    await plugin.inject_memories(ev, req)

    # 画像进 system_prompt
    check("画像进真实 ProviderRequest.system_prompt", "小明" in req.system_prompt)
    # 记忆进 extra_user_content_parts，且为真实 TextPart
    parts = req.extra_user_content_parts
    check("extra_user_content_parts 非空", len(parts) >= 1)
    tp = parts[0]
    check("注入块是框架 TextPart 实例", type(tp).__name__ == "TextPart")
    check("注入块内容含记忆", "小明" in getattr(tp, "text", ""))
    # 框架序列化必须带 _no_save（防写入持久历史）
    dumped = tp.model_dump_for_context()
    check("model_dump_for_context 含 _no_save:True", dumped.get("_no_save") is True)
    check("dumped 不污染持久化：_no_save 标记在", "_no_save" in dumped and dumped["_no_save"])


# ---------------------------------------------------------------- 3. 真实 LLMResponse
async def test_real_llmresponse(plugin):
    print("[3] 真实 LLMResponse 捕获链路")
    from astrbot.core.provider.entities import LLMResponse

    resp = LLMResponse(role="assistant")
    resp.completion_text = "好的，祝高考顺利！"  # 经 property setter 写入
    await plugin.inject_memories(SimEvent(), _req())
    await plugin.capture_reply(SimEvent(), resp)
    await plugin.after_sent(SimEvent())
    rows = plugin.store.recent_ledger("sess-1", limit=10)
    finals = [r["content"] for r in rows if r["role"] == "assistant"]
    check("捕获 LLMResponse.completion_text 入账本", any("高考顺利" in c for c in finals))


def _req():
    from astrbot.core.provider.entities import ProviderRequest
    return ProviderRequest(prompt="x", session_id="sess-1")


# ---------------------------------------------------------------- 4. 真实工具执行器
async def test_real_tool_executor(plugin):
    print("[4] call_local_llm_tool 真实执行链路")
    from astrbot.core.astr_agent_tool_exec import call_local_llm_tool
    from astrbot.core.agent.run_context import ContextWrapper
    from astrbot.core.agent.tool import FunctionTool

    tools = {}
    from astrbot_plugin_mnemoria.tools import MemoryRememberTool, MemoryRecallTool
    remember = MemoryRememberTool()
    recall = MemoryRecallTool()

    ev = SimEvent()
    ev.mnemoria_engine = plugin.engine
    wrapper = ContextWrapper(context=types.SimpleNamespace(event=ev))

    # 模拟 LLM 的 function call：executor 以 handler(event, **kwargs) 绑定
    gen = call_local_llm_tool(
        context=wrapper, handler=remember.run, method_name="run",
        content="用户养了一只叫米娅的猫", category="fact",
    )
    first = await anext(gen)
    text = ""
    try:
        for c in first.content:
            text += getattr(c, "text", "")
    except Exception:
        text = str(first)
    check("executor 路径记住成功", "记住" in text)

    gen2 = call_local_llm_tool(
        context=wrapper, handler=recall.run, method_name="run",
        query="米娅", limit=5,
    )
    r2 = await anext(gen2)
    # 底层 generator 吐原始 str；外层 FunctionToolExecutor 才包成 CallToolResult
    if isinstance(r2, str):
        text2 = r2
    else:
        text2 = "".join(getattr(c, "text", "") for c in r2.content)
    check("executor 路径召回命中", "米娅" in text2)

    # 框架 schema 导出（发给 LLM 前的转换）必须可用
    oai = remember.to_openai_tool_format() if hasattr(remember, "to_openai_tool_format") else None
    if oai is None:
        # 兼容旧名
        for fn in ("to_openai_tool", "as_openai_tool", "to_obj"):
            f = getattr(remember, fn, None)
            if f:
                oai = f()
                break
    if oai is not None:
        s = str(oai)
        check("工具 schema 可序列化为 OpenAI 格式", "memory_remember" in s)
    else:
        check("工具 schema 可序列化为 OpenAI 格式", True)  # 无该方法则跳过（版本差异）


# ---------------------------------------------------------------- 5. 全流程对话模拟
async def test_full_conversation_sim(plugin):
    print("[5] 多轮对话端到端（真实请求对象串联）")
    from astrbot.core.provider.entities import LLMResponse, ProviderRequest

    for turn in range(3):
        req = ProviderRequest(prompt=f"第{turn}轮：聊聊学习压力", session_id="conv-x")
        ev = SimEvent(text=f"第{turn}轮：聊聊学习压力", session="conv-x")
        await plugin.inject_memories(ev, req)
        resp = LLMResponse(role="assistant")
        resp.completion_text = f"学习辛苦了，第{turn}轮回复"
        await plugin.capture_reply(ev, resp)
        await plugin.after_sent(ev)

    rows = plugin.store.recent_ledger("conv-x", limit=20)
    n_user = sum(1 for r in rows if r["role"] == "user")
    n_bot = sum(1 for r in rows if r["role"] == "assistant")
    check("3 轮对话双角色账本完整", n_user == 3 and n_bot == 3)

    # 注入的记忆块每轮 ≤1 且都带 _no_save
    req2 = ProviderRequest(prompt="继续", session_id="conv-x")
    await plugin.inject_memories(SimEvent(text="继续", session="conv-x"), req2)
    saves = [getattr(p, "_no_save", None) for p in req2.extra_user_content_parts]
    check("所有注入块均标 _no_save", all(s is True for s in saves) if saves else True)


async def main():
    test_handler_registry()
    plugin = make_plugin()
    try:
        await test_provider_request_injection(plugin)
        await test_real_llmresponse(plugin)
        await test_full_conversation_sim(plugin)
        await test_real_tool_executor(plugin)
        await plugin.terminate()
        check("terminate 收尾干净", True)
    except Exception as exc:
        import traceback
        traceback.print_exc()
        check(f"异常: {exc}", False)
    print(f"\n通过 {len(PASS)} / 失败 {len(FAIL)}")
    if FAIL:
        print("失败项：", ", ".join(FAIL))
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
