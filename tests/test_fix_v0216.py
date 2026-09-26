"""v0.2.16 发布审查修复批回归测试。

覆盖（2026-09-26 三源审查：自查两轮 + 深扫代理）：
1. **群聊隐私门控（P1）**：`ledger.group_chats=false` 此前只挡记账不挡注入，
   群聊消息仍把发送者画像与共享域记忆注入模型上下文。新增
   `injection.group_inject`（默认 false）：群聊事件默认不注入。
2. **CLI 脚本加固（P2/P3）**：
   - ASTRBOT_CMD_CONFIG 未设时 CFG_PATH 为 None（不再是 Path("") 恒真
     死代码）；
   - `--provider-id` 默认空：为空时列出配置里的可用嵌入提供商而不是
     按作者环境默认值 nvidia_embedding 静默外发；
   - 导入笔记缺省 scope 对齐 "default"（此前回落 "public"，与检索/
     注入的默认域不一致，迁移笔记静默不可检索）。
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


# ---------------------------------------------------------------- 群聊门控
class _Ev:
    """最小事件替身：group 非空串即群聊。"""

    def __init__(self, group=""):
        self._group = group

    def get_message_str(self):
        return "群里问候"

    def get_sender_id(self):
        return "u1"

    def get_sender_name(self):
        return "测试用户"

    def get_group_id(self):
        return self._group

    def get_session_id(self):
        return "group-session"

    def get_platform_name(self):
        return "test"


class _Req:
    def __init__(self):
        self.system_prompt = "base"
        self.extra_user_content_parts = []


class TestGroupInjectionGate:
    @pytest.mark.asyncio
    async def test_group_events_do_not_inject_by_default(self, plugin):
        """P1 回归钉：群聊 + group_inject 未配置 → 画像与记忆都不注入。"""
        plugin.store.upsert_profile("default", "u1", "事实属性", "用户的生日是三月五日")
        await plugin.engine.remember("用户喜欢打篮球", alpha=0.9, scope="default")
        req = _Req()
        await plugin.inject_memories(_Ev(group="g1"), req)
        assert req.system_prompt == "base", "群聊默认不得注入画像到系统提示"
        assert not req.extra_user_content_parts, "群聊默认不得注入记忆块"

    @pytest.mark.asyncio
    async def test_group_inject_enabled_restores_injection(self, plugin):
        """显式开 injection.group_inject=true 时群聊注入恢复（逃生口）。"""
        plugin.config._raw.setdefault("injection", {})["group_inject"] = True
        plugin.store.upsert_profile("default", "u1", "事实属性", "用户的生日是三月五日")
        await plugin.engine.remember("用户喜欢打篮球", alpha=0.9, scope="default")
        req = _Req()
        await plugin.inject_memories(_Ev(group="g1"), req)
        assert "三月五日" in req.system_prompt or any(
            "打篮球" in getattr(p, "text", "") for p in req.extra_user_content_parts
        ), "group_inject=true 时应恢复注入"

    @pytest.mark.asyncio
    async def test_private_events_still_inject(self, plugin):
        """私聊（无 group id）不受门控影响——行为保持。"""
        plugin.store.upsert_profile("default", "u1", "事实属性", "用户的生日是三月五日")
        await plugin.engine.remember("用户喜欢打篮球", alpha=0.9, scope="default")
        req = _Req()
        await plugin.inject_memories(_Ev(group=""), req)
        assert "三月五日" in req.system_prompt, "私聊画像注入必须保持"
        joined = "\n".join(getattr(p, "text", "") for p in req.extra_user_content_parts)
        assert "打篮球" in joined, "私聊记忆注入必须保持"


# ---------------------------------------------------------------- CLI 脚本
class TestScriptGuards:
    def test_cfg_path_none_when_env_unset(self, monkeypatch):
        """ASTRBOT_CMD_CONFIG 未设 → CFG_PATH 为 None（Path("") 会归一化成
        当前目录、.exists() 恒真，友好报错成死代码——审查 D1）。"""
        monkeypatch.delenv("ASTRBOT_CMD_CONFIG", raising=False)
        import importlib

        import scripts.import_export as ie

        importlib.reload(ie)
        assert ie.CFG_PATH is None

    def test_load_provider_lists_options_when_id_empty(self, tmp_path, monkeypatch):
        """provider_id 为空 → 列出配置内可用嵌入提供商并退出（D2），
        不再按作者环境默认 nvidia_embedding 静默外发。"""
        cfg = tmp_path / "cmd_config.json"
        cfg.write_text(json.dumps({
            "provider": [
                {"id": "prov_a", "embedding_api_key": "k1"},
                {"id": "prov_b", "embedding_api_key": "k2"},
                {"id": "chat_only", "api_key": "k3"},
            ]
        }), encoding="utf-8")
        monkeypatch.setenv("ASTRBOT_CMD_CONFIG", str(cfg))
        import importlib

        import scripts.import_export as ie

        importlib.reload(ie)
        with pytest.raises(SystemExit) as ei:
            ie._load_embedding_provider("")
        msg = str(ei.value)
        assert "prov_a" in msg and "prov_b" in msg, "报错应列出可用嵌入提供商"
        assert "chat_only" not in msg, "非嵌入提供商不进入候选清单"

    def test_import_note_scope_defaults_to_default(self, tmp_path):
        """导出 JSON 的笔记缺 scope → 导入回落 'default'（D5），
        与检索/注入的默认域一致，不再静默不可检索。"""
        payload = {
            "memories": [],
            "profiles": [],
            "notes": [{"id": "n1", "content": "迁移的笔记内容", "scope": ""}],
        }
        src = tmp_path / "in.json"
        src.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
        db = tmp_path / "m.db"
        import scripts.import_export as ie

        assert ie.do_import(db, src) == 0
        import sqlite3

        row = sqlite3.connect(str(db)).execute(
            "SELECT scope FROM notes WHERE id='n1'").fetchone()
        assert row and row[0] == "default", f"笔记缺省 scope 应为 default，实际 {row}"
