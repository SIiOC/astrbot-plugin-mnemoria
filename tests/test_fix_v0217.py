"""v0.2.17 scope 隔离回归测试。

覆盖 WebUI 读取与管理端点：账本、回收站、记忆/笔记按 ID CRUD。
管理面板的 scope 是安全边界，不能只靠前端筛选；跨 scope 的 ID 请求必须
看不到目标，也不能修改、软删、恢复或硬删目标。
"""

from __future__ import annotations

import pytest
from quart import Quart

from core import web_api


@pytest.fixture
def quart_app():
    app = Quart(__name__)
    app.config["TESTING"] = True
    return app


class TestWebApiScopeIsolation:
    @pytest.mark.asyncio
    async def test_ledger_list_filters_scope_and_role(self, plugin, quart_app):
        plugin.store.append_ledger("same-session", "assistant", "域A助手", 1.0, scope="scope-a")
        plugin.store.append_ledger("same-session", "assistant", "域B助手", 2.0, scope="scope-b")
        plugin.store.append_ledger("same-session", "user", "域A用户私信", 3.0, scope="scope-a")

        async with quart_app.test_request_context("/?scope=scope-a&limit=20"):
            data = web_api._list_ledger(plugin)
        assert [row["content"] for row in data["items"]] == ["域A助手"]
        assert data["scope"] == "scope-a"

        async with quart_app.test_request_context("/?scope=scope-a&session_id=same-session"):
            data = web_api._list_ledger(plugin)
        assert [row["content"] for row in data["items"]] == ["域A助手"]

        async with quart_app.test_request_context("/?scope=scope-b&session_id=same-session"):
            data = web_api._list_ledger(plugin)
        assert [row["content"] for row in data["items"]] == ["域B助手"]

    @pytest.mark.asyncio
    async def test_trash_list_filters_memory_and_note_scope(self, plugin, quart_app):
        mem_a = plugin.store.add_memory("回收站域A", scope="scope-a")
        mem_b = plugin.store.add_memory("回收站域B", scope="scope-b")
        plugin.store.trash(mem_a)
        plugin.store.trash(mem_b)
        note_a = plugin.store.add_note("笔记域A", scope="scope-a")
        note_b = plugin.store.add_note("笔记域B", scope="scope-b")
        plugin.store.trash_note(note_a)
        plugin.store.trash_note(note_b)

        async with quart_app.test_request_context("/?scope=scope-a"):
            data = web_api._list_trash(plugin)
        assert {row["id"] for row in data["items"]} == {mem_a}
        assert {row["id"] for row in data["notes"]} == {note_a}
        assert data["scope"] == "scope-a"

    @pytest.mark.asyncio
    async def test_memory_id_mutations_cannot_cross_scope(self, plugin, quart_app):
        mid = plugin.store.add_memory("只属于域A", scope="scope-a")

        async with quart_app.test_request_context("/", method="GET", query_string={
            "id": mid, "scope": "scope-b",
        }):
            assert web_api._get_memory(plugin)["memory"] is None

        async with quart_app.test_request_context("/", method="POST", json={
            "id": mid, "scope": "scope-b", "content": "跨域修改",
        }):
            with pytest.raises(ValueError, match="不属于当前 scope"):
                await web_api._update_memory(plugin)
        assert plugin.store.get_memory(mid)["content"] == "只属于域A"

        async with quart_app.test_request_context("/", method="POST", json={
            "id": mid, "scope": "scope-b",
        }):
            with pytest.raises(ValueError, match="不属于当前 scope"):
                await web_api._delete_memory(plugin)
        assert plugin.store.get_memory(mid)["deleted_at"] is None

        async with quart_app.test_request_context("/", method="POST", json={
            "id": mid, "scope": "scope-a",
        }):
            assert (await web_api._delete_memory(plugin))["trashed"] == mid

        async with quart_app.test_request_context("/", method="POST", json={
            "id": mid, "scope": "scope-b",
        }):
            with pytest.raises(ValueError, match="不属于当前 scope"):
                await web_api._restore_memory(plugin)
        assert plugin.store.get_memory(mid)["deleted_at"] is not None

    @pytest.mark.asyncio
    async def test_note_id_mutations_cannot_cross_scope(self, plugin, quart_app):
        nid = plugin.store.add_note("只属于笔记域A", scope="scope-a")

        async with quart_app.test_request_context("/", method="POST", json={
            "id": nid, "scope": "scope-b", "content": "跨域修改笔记",
        }):
            with pytest.raises(ValueError, match="不属于当前 scope"):
                await web_api._update_note(plugin)
        assert plugin.store.get_note(nid)["content"] == "只属于笔记域A"

        async with quart_app.test_request_context("/", method="POST", json={
            "id": nid, "scope": "scope-b",
        }):
            with pytest.raises(ValueError, match="不属于当前 scope"):
                await web_api._delete_note(plugin)
        assert plugin.store.get_note(nid)["deleted_at"] is None

        async with quart_app.test_request_context("/", method="POST", json={
            "id": nid, "scope": "scope-a",
        }):
            assert (await web_api._delete_note(plugin))["trashed"] == nid

        async with quart_app.test_request_context("/", method="POST", json={
            "id": nid, "scope": "scope-b",
        }):
            with pytest.raises(ValueError, match="不属于当前 scope"):
                await web_api._purge_note(plugin)
        assert plugin.store.get_note(nid) is not None
