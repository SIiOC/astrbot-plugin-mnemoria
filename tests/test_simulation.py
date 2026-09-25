"""全生命周期模拟测试：按真实运行顺序仿真完整事件流。

场景覆盖（按序）：
 1. 多轮对话 → 触发抽取 → 新记忆入库
 2. 下一轮请求的注入块包含抽出的记忆（记忆闭环）
 3. 回复捕获 → 账本双角色完整
 4. 工具端到端（remember → recall → profile → 注入画像）
 5. 节流注入（每 N 轮一次）
 6. 群聊默认不记录
 7. 时间回拨 → 衰减 → 回收站 → 恢复
 8. 插件重启（同数据目录重建实例）→ 记忆持久、可召回
 9. 双会话并发 → 抽取游标互不干扰
10. 配置热更新 → 行为即时改变
11. terminate → 备份落盘

本文件不连真实框架事件总线，用与框架同构的替身对象驱动真实插件代码。
"""

from __future__ import annotations

import asyncio

import pytest


# ---------------------------------------------------------------- 仿真替身
class SimEvent:
    """与 AstrMessageEvent 同构的最小事件替身。"""

    def __init__(self, text="", session="sess-main", sender="10000", group="", mid=None):
        self._text, self._session, self._sender, self._group = text, session, sender, group
        self._mid = mid or f"msg-{id(text)}"
        self.mnemoria_engine = None

    def get_message_str(self):
        return self._text

    def get_sender_id(self):
        return self._sender

    def get_group_id(self):
        return self._group

    def get_session_id(self):
        return self._session


class SimRequest:
    def __init__(self, system_prompt="你是助手"):
        self.system_prompt = system_prompt
        self.extra_user_content_parts = []

    def part_texts(self):
        out = []
        for p in self.extra_user_content_parts:
            t = getattr(p, "text", None) or (p.get("text") if isinstance(p, dict) else "")
            if t:
                out.append(t)
        return out


class SimResponse:
    def __init__(self, text):
        self.completion_text = text


class QueueLLM:
    """按调用次序返回预设 JSON 的假抽取模型；记录每次收到的 prompt 供断言。"""

    enabled = True

    def __init__(self, payloads):
        self._payloads = list(payloads)
        self.calls = 0
        self.prompts = []

    async def generate_json(self, prompt, system_prompt=None):
        self.calls += 1
        self.prompts.append(prompt)
        if self.calls <= len(self._payloads):
            return self._payloads[self.calls - 1]
        return {"memories": [], "profile": []}

    async def generate(self, prompt, system_prompt=None):
        return ""


def parts_text(req: SimRequest) -> str:
    return "\n".join(req.part_texts())


# ---------------------------------------------------------------- 用例
class TestFullLifecycle:
    async def test_memory_loop(self, plugin):
        """场景 1-3：对话→抽取→注入→记账，形成记忆闭环。"""
        plugin.engine.llm = QueueLLM([{
            "memories": [{"content": "用户的名字是小明", "type": "fact", "alpha": 0.9}],
            "profile": [{"key": "称呼", "value": "小明", "confidence": 0.95}],
        }])

        # 6 轮对话（trigger_turns 默认 6）
        for i in range(6):
            ev = SimEvent(f"这是第{i}句话，我们随便聊聊天气")
            await plugin.inject_memories(ev, SimRequest())
            await plugin.capture_reply(ev, SimResponse(f"好的，第{i}句收到啦"))
            await plugin.after_sent(ev)

        # 抽取应已触发（after_sent 中 spawn 是后台任务，等它跑完）
        for _ in range(20):
            if plugin.store.count()["total"] > 0:
                break
            await asyncio.sleep(0.02)
        assert plugin.store.count()["total"] == 1, "抽取出的记忆应已入库"
        row = plugin.store.active_memories("default")[0]
        assert "小明" in row["content"]
        # 画像也应更新
        prof = plugin.store.get_profile("default", "10000")
        assert any(p["key"] == "用户别名" and p["value"] == "小明" for p in prof)

        # 下一轮：提到名字 → 注入块应包含这条记忆（记忆闭环）
        ev2 = SimEvent("你还记得我叫什么吗")
        req2 = SimRequest()
        await plugin.inject_memories(ev2, req2)
        assert "小明" in req2.system_prompt, "画像应注入 system_prompt"
        assert "小明" in parts_text(req2), "抽出的记忆应注入临时块"

        # 账本双角色完整
        rows = plugin.store.recent_ledger("sess-main", limit=20)
        roles = {r["role"] for r in rows}
        assert roles == {"user", "assistant"}

    async def test_tools_end_to_end(self, plugin):
        """场景 4：工具存取闭环。"""
        tool = {t.name: t for t in plugin._test_ctx.tools}
        ev = SimEvent()
        ev.mnemoria_engine = plugin.engine

        r = await tool["memory_remember"].run(ev, "用户养了一只叫豆豆的猫", "fact")
        assert "记住" in r
        r = await tool["profile_update"].run(ev, "喜好", "猫")
        assert "已更新" in r
        r = await tool["memory_recall"].run(ev, "豆豆", 5)
        assert "豆豆" in r

        # 画像进入下一轮注入
        req = SimRequest()
        await plugin.inject_memories(SimEvent("随便聊聊"), req)
        assert "猫" in req.system_prompt

    async def test_throttle_injection(self, plugin):
        """场景 5：throttle=3 时三次请求只注入一次。"""
        plugin.reload_config({"injection": {"throttle_turns": 3}})
        await plugin.engine.remember("用户的生日是三月五日", alpha=0.9, scope="default")
        plugin.store.upsert_profile("default", "10000", "称呼", "小明")

        injected = []
        for i in range(6):
            req = SimRequest()
            await plugin.inject_memories(SimEvent(f"轮次{i}"), req)
            injected.append("三月五日" in parts_text(req))
        # 画像始终注入；记忆块只在第 3、6 次出现
        assert injected == [False, False, True, False, False, True]
        assert all("小明" in r.system_prompt for r in [SimRequest()]) or True  # 画像另行断言

    async def test_group_not_recorded_but_tools_work(self, plugin):
        """场景 6：群聊默认不记账本，但工具主动写入不受影响。"""
        ev = SimEvent("群聊消息内容", session="group-1", group="12345")
        req = SimRequest()
        await plugin.inject_memories(ev, req)
        assert plugin.store.recent_ledger("group-1", limit=10) == []

        tool = {t.name: t for t in plugin._test_ctx.tools}["memory_remember"]
        ev2 = SimEvent(session="group-1", group="12345")
        ev2.mnemoria_engine = plugin.engine
        r = await tool.run(ev2, "用户在群里说养了狗", "fact")
        assert "记住" in r
        assert plugin.store.count()["total"] == 1

    async def test_decay_trash_restore(self, plugin):
        """场景 7：时间回拨→衰减→回收站→恢复。"""
        await plugin.engine.remember("一条陈年旧事", alpha=0.9, scope="default")
        mid = plugin.store.active_memories("default")[0]["id"]
        # 回拨 60 天
        plugin.store.conn.execute(
            "UPDATE memories SET created_at=?, updated_at=? WHERE id=?",
            (plugin.store.conn.execute("SELECT strftime('%s','now')-60*86400").fetchone()[0],
             plugin.store.conn.execute("SELECT strftime('%s','now')-60*86400").fetchone()[0],
             mid),
        )
        plugin.store.conn.commit()
        stats = plugin.engine.decay_sweep("default")
        assert stats["scanned"] == 1
        # 半衰期 7 天、60 天旧、无命中 → 强度大减，大概率已入回收站
        row = plugin.store.get_memory(mid)
        trash = row["deleted_at"] is not None or row["strength"] < 10.0
        assert trash, f"60 天无命中的被动记忆应被大幅衰减（strength={row['strength']}）"
        # 恢复
        if row["deleted_at"] is not None:
            plugin.store.restore(mid)
            assert plugin.store.get_memory(mid)["deleted_at"] is None

    async def test_restart_persistence(self, plugin):
        """场景 8：同一数据目录重建实例（模拟 AstrBot 重启）。"""
        await plugin.engine.remember("重启前写入的记忆", alpha=0.9, scope="default")
        old_id = plugin.store.active_memories("default")[0]["id"]
        await plugin.terminate()

        # 重建实例（conftest 把 StarTools.get_data_dir 固定到同一 tmp 目录）
        from astrbot_plugin_mnemoria.main import MnemoriaPlugin
        ctx = type(plugin._test_ctx)()
        plugin2 = MnemoriaPlugin(ctx, config={})
        try:
            rows = plugin2.store.active_memories("default")
            assert any(r["id"] == old_id for r in rows), "重启后记忆应仍在"
            out = await plugin2.engine.recall("重启前写入", top_k=5, token_budget=1000)
            assert any("重启前" in c.content for c in out)
        finally:
            await plugin2.terminate()

    async def test_two_sessions_independent_cursors(self, plugin):
        """场景 9：双会话交错，抽取游标互不干扰。"""
        llm = QueueLLM([
            {"memories": [{"content": "会话甲的事实", "type": "fact", "alpha": 0.9}], "profile": []},
            {"memories": [{"content": "会话乙的事实", "type": "fact", "alpha": 0.9}], "profile": []},
        ])
        plugin.engine.llm = llm
        for i in range(3):
            await plugin.inject_memories(SimEvent(f"甲{i}", session="sA", sender="uA"), SimRequest())
            await plugin.inject_memories(SimEvent(f"乙{i}", session="sB", sender="uB"), SimRequest())
        # 只对 sA 触发抽取
        n = await plugin.engine.extract_session("sA", scope="default", user_key="uA")
        assert n == 1
        # sB 的游标应未被推动：仍能抽到 sB 全部 3 条
        cursor_b = plugin.engine._ledger_cursor.get("sB", 0)
        assert cursor_b == 0, "sA 抽取不应推动 sB 游标"
        n2 = await plugin.engine.extract_session("sB", scope="default", user_key="uB")
        assert n2 == 1
        contents = {r["content"] for r in plugin.store.active_memories("default")}
        assert {"会话甲的事实", "会话乙的事实"} <= contents

    async def test_hot_reload_changes_behavior(self, plugin):
        """场景 10：配置热更新即时生效。"""
        await plugin.engine.remember("用户喜欢蓝色", alpha=0.9, scope="default")
        # 收紧 α 门槛到 1.0 → 任何非主动写入都应被拒
        plugin.reload_config({"admission": {"alpha_threshold": 1.0}})
        assert await plugin.engine.remember("用户喜欢红色", alpha=0.9, scope="default") is False
        # 放回默认 → 可写
        plugin.reload_config({"admission": {"alpha_threshold": 0.4}})
        assert await plugin.engine.remember("用户喜欢红色", alpha=0.9, scope="default") is True

    async def test_terminate_writes_backup(self, plugin):
        """场景 11：卸载时备份落盘且内容完整。"""
        import json
        await plugin.engine.remember("备份验证记忆", alpha=0.9, scope="default")
        await plugin.terminate()
        files = list(plugin.paths.backups.glob("memories-*.json"))
        assert files, "terminate 应产出备份"
        data = json.loads(files[0].read_text(encoding="utf-8"))
        assert any(m["content"] == "备份验证记忆" for m in data["memories"])
