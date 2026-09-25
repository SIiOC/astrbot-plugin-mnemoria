"""WebAPI 路由集成测试：在真框架（quart）下验证注册与调用约定。

运行（必须用 AstrBot 的 venv）：
  python tests/web_api_smoke.py

覆盖：
  1. 29 条路由全部注册成功且路径带插件名前缀
  2. 每条 GET 路由在带 Quart 请求上下文时可正常返回 (jsonify, 200)
  3. POST 路由带 JSON body 可正常调用
  4. 路由匹配器能按路径+方法命中（模拟 dashboard 侧解析）
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


class FakeContext:
    def __init__(self):
        self.routes = []

    def register_web_api(self, route, handler, methods, desc):
        self.routes.append((route, handler, methods, desc))


def make_plugin(tmp):
    from astrbot.core.star.star_tools import StarTools

    orig = StarTools.get_data_dir
    StarTools.get_data_dir = classmethod(lambda cls, name=None: Path(tmp) / (name or "p"))

    class Ctx(FakeContext):
        class provider_manager:
            @staticmethod
            async def get_provider_by_id(p):
                return None

        def add_llm_tools(self, *t):
            pass

    from astrbot_plugin_mnemoria.main import MnemoriaPlugin
    ctx = Ctx()
    # 第四轮矩阵验证修复：config/save 的 fail-closed 白名单读
    # astrbot_config.schema——生产里 config 是框架 AstrBotConfig（带 schema）。
    # 此前 smoke 传裸 dict 导致该用例自 v0.1.1 起 fail-closed 误挂（smoke
    # 不被 pytest 收集，挂了三轮没被发现）。改用真实 AstrBotConfig 贴近生产。
    import json as _json
    import tempfile as _tempfile
    from astrbot.core.config.astrbot_config import AstrBotConfig
    _schema = _json.loads(
        (Path(__file__).resolve().parents[1] / "_conf_schema.json").read_text(
            encoding="utf-8-sig"))
    _tmp_cfg = Path(_tempfile.mkdtemp()) / "smoke_config.json"
    _tmp_cfg.write_text("{}", encoding="utf-8")
    plugin = MnemoriaPlugin(ctx, config=AstrBotConfig(
        config_path=str(_tmp_cfg), schema=_schema))
    StarTools.get_data_dir = orig
    return plugin, ctx


def run():
    from quart import Quart, request
    import re

    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
        plugin, ctx = make_plugin(tmp)
        routes = ctx.routes
        check("注册了 29 条路由", len(routes) == 29)
        check("路由均带插件名前缀",
              all(r[0].startswith("/astrbot_plugin_mnemoria/") for r in routes))

        # 模拟 dashboard 的路径匹配器
        def route_pattern(route):
            chunks, pos = ["^"], 0
            for m in re.finditer(r"<(?P<n>[^>:]+)(?::(?P<t>[^>]+))?>", route):
                chunks.append(re.escape(route[pos:m.start()]))
                chunks.append(f"(?P<{m.group('n')}>[^/]+)")
                pos = m.end()
            chunks.append(re.escape(route[pos:]) + "$")
            return "".join(chunks)

        app = Quart(__name__)
        app.config["TESTING"] = True

        async def drive():
            async with app.test_request_context("/", method="GET"):
                ges = [r for r in routes if "GET" in r[2]]
                posts = [r for r in routes if "POST" in r[2]]
                check("GET 路由存在", len(ges) >= 8)
                check("POST 路由存在", len(posts) >= 4)

                # 准备一点数据，让 GET 有内容可返回
                plugin.store.upsert_profile("default", "u1", "称呼", "小张")
                await plugin.engine.remember("用户喜欢猫", alpha=0.9, scope="default")
                plugin.engine.record_turn("s1", "user", "我在准备高考", scope="default")

                ok_get = 0
                for route, handler, methods, _ in ges:
                    try:
                        resp = await handler()
                        if isinstance(resp, tuple) and len(resp) == 2 and resp[1] == 200:
                            ok_get += 1
                    except Exception as e:  # noqa: BLE001
                        print("   GET 失败", route, type(e).__name__, e)
                check(f"GET 全部可调用（{ok_get}/{len(ges)}）", ok_get == len(ges))

                # 路由匹配器验证（挑一条带参数的有路径参数吗？——本插件无路径参数，纯静态）
                matched = 0
                for route, handler, methods, _ in routes:
                    for m in methods:
                        if re.fullmatch(route_pattern(route), route):
                            matched += 1
                            break
                check("所有路由可被匹配器命中", matched == len(routes))

            # POST：带 JSON body
            async with app.test_request_context(
                "/", method="POST", json={"content": "用户养了猫"}
            ):
                post_map = {r[0]: r for r in posts}
                mid_route = "/astrbot_plugin_mnemoria/memory/update"
                del_route = "/astrbot_plugin_mnemoria/memory/delete"
                # 先造一条记忆
                m = plugin.store.active_memories("default")[0]
                mid = m["id"]
                async with app.test_request_context(
                    "/", method="POST", json={"id": mid, "content": "改后的内容"}
                ):
                    resp = await post_map[mid_route][1]()
                    check("POST memory/update 可调用", isinstance(resp, tuple) and resp[1] == 200)
                    check("POST 实际改到了数据",
                          plugin.store.get_memory(mid)["content"] == "改后的内容")
                async with app.test_request_context(
                    "/", method="POST", json={"id": mid}
                ):
                    resp = await post_map[del_route][1]()
                    check("POST memory/delete 入回收站",
                          isinstance(resp, tuple) and len(plugin.store.list_trash()) >= 1)
                    # 回收站硬删（v0.2.9）：连带删向量，删后 get_memory 应为 None
                    resp = await post_map["/astrbot_plugin_mnemoria/memory/purge"][1]()
                    check("POST memory/purge 彻底删除",
                          isinstance(resp, tuple) and resp[1] == 200
                          and plugin.store.get_memory(mid) is None)

            # 隔离审核（v0.2.9）：memory/update 接受 quarantined 字段并置脏向量缓存
            await plugin.engine.remember("隔离审核测试记忆", alpha=0.9, scope="default")
            qm = [x for x in plugin.store.active_memories("default")
                  if x["content"] == "隔离审核测试记忆"][0]
            plugin.store.update_memory(qm["id"], quarantined=1)
            plugin.engine._vectors_dirty = False  # 归零后才能断言「审核置脏」
            async with app.test_request_context(
                "/", method="POST", json={"id": qm["id"], "quarantined": False}
            ):
                resp = await post_map[mid_route][1]()
                check("POST memory/update 通过隔离审核",
                      isinstance(resp, tuple) and resp[1] == 200
                      and plugin.store.get_memory(qm["id"])["quarantined"] == 0
                      and plugin.engine._vectors_dirty is True)
            # 设置页模型列表（v0.2.9）：假上下文无 provider_insts，应返回空列表而非报错
            async with app.test_request_context("/", method="GET"):
                resp = await {r[0]: r for r in routes}["/astrbot_plugin_mnemoria/providers"][1]()
                body = await resp[0].get_json()
                check("GET providers 返回 chat/embedding 列表",
                      body["ok"] and body["data"]["chat"] == [] and body["data"]["embedding"] == [])

            # UI 融合新增端点：graph（含 computing 首次态）与 config 读写
            # （前面 delete 测试已把唯一记忆移入回收站，先补一条活的）
            await plugin.engine.remember("星图测试记忆", alpha=0.9, scope="default")
            async with app.test_request_context("/", method="GET"):
                graph_route = "/astrbot_plugin_mnemoria/graph"
                cfg_route = "/astrbot_plugin_mnemoria/config"
                cfg_save_route = "/astrbot_plugin_mnemoria/config/save"
                route_map = {r[0]: r for r in routes}
                resp = await route_map[graph_route][1]()
                body = await resp[0].get_json()
                check("GET graph 返回节点",
                      body["ok"] and any(n["content"] == "星图测试记忆"
                                         for n in body["data"]["nodes"]))
                check("GET graph 首次含 computing 标记或缓存就绪",
                      body["data"]["computing"] is True or body["data"]["edges"] is not None)
                resp = await route_map[cfg_route][1]()
                body = await resp[0].get_json()
                check("GET config 返回扁平配置", body["ok"] and "flat" in body["data"])
            async with app.test_request_context(
                "/", method="POST", json={"updates": {"ledger.group_chats": True}}
            ):
                resp = await route_map[cfg_save_route][1]()
                body = await resp[0].get_json()
                check("POST config/save 就地生效",
                      body["ok"] and "ledger.group_chats" in body["data"]["applied"] and
                      plugin.config.get("ledger.group_chats") is True)

            # 错误路径：缺 id 应被包装为 500 信封（而非异常穿透）
            async with app.test_request_context("/", method="POST", json={}):
                resp = await post_map[mid_route][1]()
                ok_env = (isinstance(resp, tuple) and len(resp) == 2 and resp[1] == 500)
                body = resp[0]
                check("缺 id 时返回 500 信封", ok_env)
                try:
                    payload = await body.get_json()
                    check("错误信封含 ok=false", payload.get("ok") is False)
                except Exception:
                    check("错误信封含 ok=false", False)

        asyncio.run(drive())
        asyncio.run(plugin.terminate())


if __name__ == "__main__":
    run()
    print(f"\n通过 {len(PASS)} / 失败 {len(FAIL)}")
    if FAIL:
        print("失败项：", ", ".join(FAIL))
        sys.exit(1)
