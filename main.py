"""好想记住你（mnemoria）—— AstrBot 长期记忆插件。

提供：对话流水账本 / 自动记忆抽取与衰减 / 四路混合检索注入 / 用户画像 / 主动存取工具。
对标主流记忆插件的**全功能**选择，与其它插件零代码耦合。
"""

from __future__ import annotations

import logging
import os
from datetime import datetime

from astrbot.api.event import AstrMessageEvent, filter
from astrbot.api.star import Context, Star, register
from astrbot.core.star.star_tools import StarTools

try:
    from astrbot.api import logger
except ImportError:
    logger = logging.getLogger(__name__)

from .core import db as dbm
from .core.backup import write_backup
from .core.bridge import Embedder, Reranker
from .core.config import Config
from .core.engine import MemoryEngine
from .core.llm import LLMBridge
from .core.paths import DataPaths, utc_now_ts
from .core.store import MemoryStore
from .core.tasks import TaskRegistry
from .tools import MemoryRecallTool, MemoryRememberTool, NoteCreateTool, ProfileUpdateTool

try:
    from astrbot.api.event.filter import PermissionType
except ImportError:  # 旧框架无权限过滤器：/忘记 在命令内禁用（见 cmd_forget）
    PermissionType = None

PLUGIN_NAME = "astrbot_plugin_mnemoria"
# 插件自身目录（v0.1.9 护栏用：检测落错位置的 AstrBot 主配置副本）
_PLUGIN_DIR = os.path.dirname(os.path.abspath(__file__))


def _metadata_version() -> str:
    """从 metadata.yaml 读版本号（v0.2.6 修复：register 版本与 metadata 同步）。

    此前 @register 里硬编码 "0.2.3"，v0.2.4/v0.2.5 两次发版都忘了改，
    插件面板一直显示旧版本号（审查缺陷）。改为启动时读真值——零依赖
    （不引 yaml，正则抠 version 行即可）；读不到时回落 "0.0.0" 让偏差显式
    暴露，而不是静默显示一个错的旧版本。
    """
    try:
        import re
        text = open(os.path.join(_PLUGIN_DIR, "metadata.yaml"),
                    encoding="utf-8").read()
        m = re.search(r"(?m)^version:\s*[\"']?([0-9][\w.-]*)[\"']?\s*$", text)
        if m:
            return m.group(1)
    except Exception:  # noqa: BLE001
        pass
    return "0.0.0"


@register(
    PLUGIN_NAME,
    "SIiOC",
    "好想记住你：长期记忆系统（账本+记忆衰减+画像+混合检索注入）",
    _metadata_version(),
    "",
)
class MnemoriaPlugin(Star):
    def __init__(self, context: Context, config: dict | None = None):
        super().__init__(context)
        self.logger = logger
        self.ctx = context
        # v0.1.9：插件目录若残留 AstrBot 主配置副本（含 dashboard 口令哈希），
        # 启动时给出可操作告警（详情见方法 docstring）。
        self._warn_stray_runtime_config()

        data_dir = StarTools.get_data_dir(PLUGIN_NAME)
        self.paths = DataPaths(data_dir).ensure()
        # 单一配置真相源：插件属性即引擎所用的同一对象，避免两处引用失同步
        self.config = Config(config or {}, self.paths.meta)
        # 保留框架 AstrBotConfig 对象引用：面板保存配置时用它的 save_config_async
        # 落盘（Config._raw 与它是同一个 dict，就地改即同步）
        self.astrbot_config = config
        self.logger.info("好想记住你数据目录: %s", self.paths.root)

        # 存储与引擎
        self.conn = dbm.connect(self.paths.db)
        old_schema = dbm.current_schema_version(self.conn)
        if dbm.has_existing_schema(self.conn) and old_schema < dbm.SCHEMA_VERSION:
            target = self.paths.schema_backup_path(dbm.SCHEMA_VERSION)
            backed = dbm.backup_before_migration(self.conn, target)
            if backed is None:
                raise RuntimeError(
                    f"schema v{dbm.SCHEMA_VERSION} 升级前备份失败，已停止启动以保护数据"
                )
            self.logger.info("schema 升级前数据库备份完成: %s", backed.name)
        self.fts_ok = dbm.init_schema(self.conn)
        self.store = MemoryStore(self.conn)
        # 桥接对象惰性读取 config 里的 provider_id，配置热更新后无需重建
        self.embedder = Embedder(context, config=self.config)
        self.reranker = Reranker(context, config=self.config)
        self.llm = LLMBridge(context, config=self.config)
        self.engine = MemoryEngine(
            self.store, self.config,
            embedder=self.embedder, reranker=self.reranker, llm=self.llm,
        )
        if not self.llm.enabled:
            self.logger.warning("未配置 provider_id，自动抽取与巩固将停用（工具与账本仍可用）。")

        # 后台任务
        self.tasks = TaskRegistry()
        self._pending_reply: dict[str, tuple[float, str]] = {}  # session -> (捕获时刻, 回复)
        self._extracting_sessions: set[str] = set()  # 防同一会话重复并发抽取
        self._nightly_running = False  # 夜间任务防重入（标记延后写，期间不重复 spawn）

        # 注册工具（NoteCreateTool 无条件注册：其 run() 运行时自查
        # notes.enabled，注册期条件化会让「面板关闭笔记后无法再打开」——
        # 工具注册没有注销 API，双向切换只能靠运行时门控）
        self._tools = [
            MemoryRememberTool(), MemoryRecallTool(), ProfileUpdateTool(), NoteCreateTool(),
        ]
        try:
            context.add_llm_tools(*self._tools)
            self.logger.info("已注册 LLM 工具：%s", "、".join(t.name for t in self._tools))
        except Exception as exc:  # noqa: BLE001
            self.logger.error("注册 LLM 工具失败（插件继续以基础模式运行）: %s", exc)

        # WebUI API
        try:
            from .core.web_api import register_routes
            register_routes(context, self)
        except Exception as exc:  # noqa: BLE001
            self.logger.warning("注册 WebAPI 失败: %s", exc)
        # 热重载场景：同进程新旧实例共用 graph 模块全局，重置上一次
        # terminate 置位的停机标志，否则星图在重载后永久停摆
        try:
            from .core import graph
            graph.reset()
        except Exception:  # noqa: BLE001
            pass

        # 启动后台循环（构造期可能不在事件循环内，故登记惰性启动）
        self._tasks_started = False
        self._ensure_tasks_started()
        self.logger.info("好想记住你插件初始化完成（fts5=%s）", self.fts_ok)

    # ---------------------------------------------------------------- 凭据残留护栏
    def _warn_stray_runtime_config(self) -> None:
        """护栏（v0.1.9）：插件目录内出现 AstrBot 主配置副本时告警。

        机制（2026-09-19 审查实测定位）：执行 `from astrbot.api import ...` 时
        框架会往**当前工作目录**写 `data/cmd_config.json`（含 dashboard 口令
        哈希）。任何“以插件目录为 cwd”的操作（在此目录跑 pytest、脚本、运动
        调试）都会把这份副本落进插件目录，一旦随目录被打包/分享即等于凭据
        外泄（此类事故确实发生过）。
        插件无法阻止框架写它，但可以在启动时把话说明白。
        仅检测与告警，**不自动删除任何文件**。
        """
        try:
            stray = os.path.join(_PLUGIN_DIR, "data", "cmd_config.json")
            if os.path.isfile(stray):
                self.logger.warning(
                    "插件目录内存在 AstrBot 主配置副本 data/cmd_config.json（含 dashboard "
                    "口令哈希）：通常由“以插件目录为工作目录导入框架”产生；打包或分享"
                    "插件目录前请删除该 data/ 目录（不影响 AstrBot 本体配置）。"
                )
        except Exception:  # noqa: BLE001
            pass

    def _ensure_tasks_started(self) -> None:
        """确保后台 tick 已启动（幂等）。构造期无事件循环时，推迟到首个钩子。"""
        if self._tasks_started:
            return
        task = self.tasks.start_periodic(self._tick, interval=60.0, name="mnemoria-tick")
        if task is not None:
            self._tasks_started = True

    # ---------------------------------------------------------------- 配置热更新
    @filter.on_astrbot_loaded()
    async def _startup(self) -> None:
        """框架完全就绪后才启动后台循环（此时一定有事件循环）。"""
        self._ensure_tasks_started()

    def reload_config(self, raw: dict | None) -> None:
        """配置变更时重载。

        桥接对象（embedder/reranker/llm）惰性读取 config，故无需重建；
        这里只需换掉配置对象并让引擎指向新对象。
        astrbot_config 必须同步指向新 dict：面板落盘走它的 save_config_async，
        若仍指向旧对象，保存时会把旧配置整体写回、覆盖掉热更新后的新值。
        """
        self.config = Config(raw if isinstance(raw, dict) else {}, self.paths.meta)
        self.engine.config = self.config
        for bridge in (self.embedder, self.reranker, self.llm):
            bridge.config = self.config
        if isinstance(raw, dict):
            self.astrbot_config = raw
        self.logger.info("好想记住你配置已重载")

    # ---------------------------------------------------------------- 作用域
    def scope_for(self, event: AstrMessageEvent) -> tuple[str, str]:
        """转发到引擎（工具侧通过 event.mnemoria_engine 调用同一实现）。"""
        return self.engine.scope_for(event)

    def identity_for(self, event: AstrMessageEvent) -> tuple[str, str, str]:
        """转发到引擎：返回 (scope, user_key, speaker_key) 并登记身份账本。"""
        return self.engine.identity_for(event)

    def _should_record(self, event: AstrMessageEvent) -> bool:
        if not self.config.get("ledger.enabled", True):
            return False
        try:
            if event.get_group_id() and not self.config.get("ledger.group_chats", False):
                return False
        except Exception:  # noqa: BLE001
            pass
        return True

    # ---------------------------------------------------------------- 钩子
    @filter.on_llm_request(priority=45)
    async def inject_memories(self, event: AstrMessageEvent, request) -> None:
        """注入画像（系统提示末尾）与相关记忆（用户消息前的临时块）。"""
        self._ensure_tasks_started()
        event.mnemoria_engine = self.engine
        scope, user_key, _speaker_key = self.identity_for(event)
        session_id = event.get_session_id()

        # 记录用户这一轮（账本）。带 message_id：ledger 的 UNIQUE(message_id)
        # 幂等靠它生效，同一条消息因 provider 重试/工具循环再次进入钩子时
        # 不会重复记账（NULL 不受唯一约束保护）
        if self._should_record(event):
            try:
                self.engine.record_turn(
                    session_id, "user", event.get_message_str(),
                    scope=scope, ts=utc_now_ts(),
                    message_id=self._event_message_id(event),
                )
            except Exception as exc:  # noqa: BLE001
                self.logger.debug("记录用户轮次失败: %s", exc)

        if not self.config.get("injection.enabled", True):
            return
        # v0.2.16：群聊隐私门控——与 ledger.group_chats=false 的语义对齐
        # （群聊默认不碰）。此前只挡记账不挡注入，群聊消息仍会把发送者的
        # 画像与共享域记忆注入模型上下文（2026-09-26 发布审查 P1，测试钉
        # test_fix_v0216）。需要群聊也注入时显式开 injection.group_inject。
        try:
            is_group = bool(event.get_group_id())
        except Exception:  # noqa: BLE001
            is_group = False
        if is_group and not self.config.get("injection.group_inject", False):
            return
        try:
            # 画像：稳定块，放系统提示末尾（利于前缀缓存）
            prof = self.engine.profile_block(scope, user_key)
            if prof:
                request.system_prompt = (request.system_prompt or "") + "\n" + prof

            # 动态记忆：按节流注入到用户消息
            # fast=True：on_llm_request 是同步等待，注入路径跳过重排、嵌入短超时，
            # 否则远程模型抖动时用户消息会被卡住最多 20 秒（审查缺陷 B）
            if self.engine.should_inject_now(session_id):
                query = event.get_message_str() or ""
                # 查询向量只算一次，记忆与笔记两条检索共享（省一次网络往返）
                qvec = await self.engine.embed_query(query, fast=True)
                cands = await self.engine.recall(
                    query, scope=scope, fast=True, query_vec=qvec,
                    top_k=int(self.config.get("injection.max_items", 8) or 8),
                    token_budget=int(self.config.get("injection.token_budget", 800) or 800),
                    expand_from_session=session_id,
                )
                block = self.engine.memories_block(cands)
                # 笔记知识库：与记忆一同召回（独立表），命中则追加一段
                try:
                    nhits = await self.engine.notes_recall(
                        query, scope=scope, fast=True, query_vec=qvec,
                        top_k=int(self.config.get_num("notes.inject_max_items", 3)),
                    )
                    nblock = self.engine.notes_block(nhits)
                    if nblock:
                        block = (block + "\n" + nblock) if block else nblock
                except Exception as exc:  # noqa: BLE001
                    self.logger.debug("笔记注入失败（已忽略）: %s", exc)
                if block:
                    self._append_temp_part(request, block)
        except Exception as exc:  # noqa: BLE001
            self.logger.warning("记忆注入失败（已忽略）: %s", exc)

    @staticmethod
    def _event_message_id(event: AstrMessageEvent) -> str | None:
        """取平台侧稳定消息 ID（拿不到返回 None，幂等退化为无防护）。"""
        try:
            mid = getattr(getattr(event, "message_obj", None), "message_id", None)
            mid = str(mid) if mid is not None else ""
            return mid or None
        except Exception:  # noqa: BLE001
            return None

    def _append_temp_part(self, request, text: str) -> None:
        try:
            from astrbot.core.agent.message import TextPart
        except Exception:  # noqa: BLE001
            parts = getattr(request, "extra_user_content_parts", None)
            if isinstance(parts, list):
                parts.insert(0, {"type": "text", "text": text})
            return
        part = TextPart(text=text)
        try:
            part.mark_as_temp()
        except Exception:  # noqa: BLE001
            pass
        parts = getattr(request, "extra_user_content_parts", None)
        if isinstance(parts, list):
            parts.insert(0, part)
        else:
            request.extra_user_content_parts = [part]

    @filter.on_llm_response(priority=-100)
    async def capture_reply(self, event: AstrMessageEvent, response) -> None:
        try:
            text = getattr(response, "completion_text", None) or getattr(response, "text", None)
            if isinstance(text, str) and text.strip():
                # 带时间戳：若消息最终未发出（after_sent 不触发），由 _tick 过期清理，
                # 否则该 session 的条目会永久残留（审查缺陷 G1）
                self._pending_reply[event.get_session_id()] = (utc_now_ts(), text.strip())
        except Exception as exc:  # noqa: BLE001
            self.logger.debug("捕获回复失败: %s", exc)

    @filter.after_message_sent(priority=-100)
    async def after_sent(self, event: AstrMessageEvent) -> None:
        session_id = event.get_session_id()
        entry = self._pending_reply.pop(session_id, None)
        reply = entry[1] if entry else ""
        if reply and self._should_record(event):
            scope, _user_key, _speaker_key = self.identity_for(event)
            try:
                self.engine.record_turn(session_id, "assistant", reply, scope=scope, ts=utc_now_ts())
            except Exception as exc:  # noqa: BLE001
                self.logger.debug("记录助手轮次失败: %s", exc)
        if self.engine.should_extract(session_id):
            scope, user_key, _speaker_key = self.identity_for(event)
            self.tasks.spawn(
                self._safe_extract(session_id, scope, user_key),
                name=f"mnemoria-extract-{session_id}",
            )
        if self.engine.should_reflect(session_id):
            self.tasks.spawn(self._safe_reflect(session_id), name=f"mnemoria-reflect-{session_id}")

    async def _safe_extract(self, session_id: str, scope: str, user_key: str) -> None:
        # 同一会话的抽取是有游标的；LLM 慢于 tick 时禁止重复并发，
        # 否则多个任务会同时读取同一游标并重复调用模型（审查缺陷 K）。
        if session_id in self._extracting_sessions:
            return
        self._extracting_sessions.add(session_id)
        try:
            n = await self.engine.extract_session(session_id, scope=scope, user_key=user_key)
            if n:
                self.logger.info("会话 %s 抽取记忆 %d 条", session_id, n)
        except Exception as exc:  # noqa: BLE001
            self.logger.warning("抽取失败: %s", exc)
        finally:
            self._extracting_sessions.discard(session_id)

    async def _safe_reflect(self, session_id: str) -> None:
        try:
            stats = await self.engine.reflect_session(session_id)
            if stats.get("useful") or stats.get("useless"):
                self.logger.info("会话 %s 反思：有用 %d／无用 %d", session_id,
                                 stats.get("useful", 0), stats.get("useless", 0))
        except Exception as exc:  # noqa: BLE001
            self.logger.warning("反思失败: %s", exc)

    # ---------------------------------------------------------------- 用户命令
    # 运维入口：livingmemory 10 个 / mnemosyne 9 个 / persistent-memory 0 个——
    # 补齐三个最常用的，其余走 WebUI 控制台。

    @staticmethod
    def _cmd_arg(event: AstrMessageEvent, *names: str) -> str:
        """从原始消息里取命令后的完整参数（含空格）。

        不用框架注入的 parsed_params：框架 init_handler_md 对有默认值的参数
        存的是「默认值」而非注解，导致 `keyword: str = ""` 只能拿到第一个词，
        其余静默丢弃。这里自己剥命令前缀，多词参数完整保留。
        """
        raw = (event.get_message_str() or "").strip()
        for name in names:
            for prefix in (f"/{name}", name):
                if raw.startswith(prefix):
                    return raw[len(prefix):].strip()
        return raw

    @filter.command("记忆状态", alias={"记忆状况"})
    async def cmd_status(self, event: AstrMessageEvent) -> None:
        """查看好想记住你运行状态：/记忆状态"""
        counts = self.store.count()
        embed = "✅" if getattr(self.embedder, "enabled", False) else "❌"
        rerank = "✅" if getattr(self.reranker, "enabled", False) else "❌"
        llm = "✅" if getattr(self.llm, "enabled", False) else "❌"
        fts = "FTS5" if self.store.fts else "LIKE 降级"
        yield event.plain_result(
            "🧠 好想记住你\n"
            f"记忆：{counts['total']} 条（主动 {counts['active']}）｜回收站 {counts['trash']} 条\n"
            f"通道：向量 {embed}｜重排 {rerank}｜自动抽取 {llm}｜关键词 {fts}\n"
            "管理面板：WebUI → 插件 → 好想记住你"
        )

    @filter.command("记忆搜索", alias={"找记忆"})
    async def cmd_search(self, event: AstrMessageEvent) -> None:
        """搜索长期记忆：/记忆搜索 <关键词>（支持多个词）"""
        kw = self._cmd_arg(event, "记忆搜索", "找记忆")
        if not kw:
            yield event.plain_result("用法：/记忆搜索 <关键词>（可用空格分隔多个词）")
            return
        scope, _ = self.engine.scope_for(event)
        cands = await self.engine.recall(
            kw, scope=scope, top_k=5, token_budget=4000,
            mark_recalled=False, require_match=True,
        )
        if not cands:
            yield event.plain_result(f"没有找到与「{kw}」相关的记忆。")
            return
        lines = [f"🔍 与「{kw}」最相关的记忆："]
        from .core.text import sanitize_for_context
        for i, c in enumerate(cands, 1):
            tag = "【主动】" if c.is_active else ""
            lines.append(f"{i}. {tag}{sanitize_for_context(c.content, 120)}")
        yield event.plain_result("\n".join(lines))

    _admin_only = (
        filter.permission_type(PermissionType.ADMIN)
        if PermissionType is not None
        else (lambda fn: fn)
    )

    @filter.command("忘记")
    @_admin_only
    async def cmd_forget(self, event: AstrMessageEvent) -> None:
        """删除一条记忆（移入回收站，可在面板恢复）：/忘记 <关键词>（仅管理员）"""
        if PermissionType is None:
            # 旧框架没有权限过滤器：删除是破坏性操作，降级语义=禁用命令
            # 而非对所有人放开，精确删除请走 WebUI 面板
            yield event.plain_result(
                "当前框架版本不支持权限过滤，/忘记 已禁用。请在 WebUI 插件面板中删除记忆。"
            )
            return
        kw = self._cmd_arg(event, "忘记")
        if not kw:
            yield event.plain_result("用法：/忘记 <关键词>（移入回收站，30 天内可在面板恢复）")
            return
        scope, _ = self.engine.scope_for(event)
        cands = await self.engine.recall(
            kw, scope=scope, top_k=3, token_budget=4000,
            mark_recalled=False, require_match=True,
        )
        if not cands:
            yield event.plain_result(f"没有找到与「{kw}」相关的记忆。")
            return
        best = cands[0]
        # 唯一命中才动手，多个候选时列出让人去面板删——防误删
        if len(cands) > 1:
            from .core.text import sanitize_for_context
            listing = "\n".join(
                f"{i}. {sanitize_for_context(c.content, 80)}" for i, c in enumerate(cands, 1)
            )
            yield event.plain_result(
                f"找到 {len(cands)} 条相关记忆，为防误删请到 WebUI 面板精确删除：\n{listing}"
            )
            return
        self.store.trash(best.id)
        self.logger.info("用户命令删除记忆 %s", best.id)
        yield event.plain_result(
            f"已忘记：{best.content[:60]}（移入回收站，可在面板恢复）"
        )

    # ---------------------------------------------------------------- 后台循环
    async def _tick(self) -> None:
        """每分钟：空闲抽取检查 + 过期回复清理 + 到点执行衰减/巩固/备份。"""
        # 空闲触发抽取（含反思闭环）
        # v0.1.9：经 engine.active_sessions() 取会话快照，不再直读 engine 的
        # 私有字典（那两个表会被 _evict_session_state 清理，跨对象直读太脆）。
        for session_id, session_user, _session_speaker in self.engine.active_sessions():
            if self.engine.should_extract(session_id):
                scope = str(self.config.get("runtime.default_scope", "default") or "default")
                user_key = session_user
                self.tasks.spawn(self._safe_extract(session_id, scope, user_key), name=f"idle-extract-{session_id}")
            if self.engine.should_reflect(session_id):
                self.tasks.spawn(self._safe_reflect(session_id), name=f"reflect-{session_id}")

        # 过期回复清理：捕获了回复但消息未发出（after_sent 未触发）的残留，
        # 10 分钟后丢弃，防 dict 无限增长（审查缺陷 G1）
        stale = [sid for sid, (ts, _) in self._pending_reply.items()
                 if utc_now_ts() - ts > 600.0]
        for sid in stale:
            self._pending_reply.pop(sid, None)
            self.logger.debug("清理会话 %s 的过期未发送回复", sid)

        now = datetime.now()
        day = now.strftime("%Y%m%d")

        # 衰减：每天一次。日期持久化到 DB（跨重启生效）——若只存内存，
        # 每次重启都会重置标记、同一天重复衰减，把边缘记忆成批扣穿进回收站
        # （2026-09-16 实测：连续两次重启多淘汰 33 条，虽可恢复但属缺陷）。
        # 标记在 decay_sweep 成功后**立即**写入：decay 非幂等（重复跑=重复扣分），
        # 绝不能因后续步骤失败而丢标记；purge_trash 幂等，单独容错即可。
        if self._decay_day() != day and now.hour >= int(self.config.get("memory_behavior.decay_hour", 3) or 3):
            try:
                stats = self.engine.decay_sweep()
            except Exception as exc:  # noqa: BLE001
                self.logger.warning("衰减失败（decay 未执行，下个周期自动重试）: %s", exc)
            else:
                self._set_day_marker("last_decay_day", day)
                self.logger.info("衰减扫描完成：%s", stats)
            try:
                purged = self.engine.purge_trash()
                if purged:
                    self.logger.info("回收站清理 %d 条", purged)
            except Exception as exc:  # noqa: BLE001
                self.logger.warning("回收站清理失败（幂等，明日再清）: %s", exc)

        # 巩固 + 备份：每天一次（配置时刻）。标记由 _nightly 在巩固真正完成后写入，
        # 此处只防重入（后台任务在跑时不重复 spawn）。
        digest_hour = int(self.config.get("memory_behavior.digest_hour", 4) or 4)
        if (self._decay_day("last_digest_day") != day
                and now.hour >= digest_hour
                and not self._nightly_running):
            self._nightly_running = True
            self.tasks.spawn(self._nightly(day), name=f"mnemoria-nightly-{day}")

    # ---------------------------------------------------------------- 日标记
    def _decay_day(self, key: str = "last_decay_day") -> str:
        """读取持久化的「上次执行日期」（跨重启有效，防同日重复执行）。"""
        try:
            return dbm.get_meta(self.conn, key, "") or ""
        except Exception:  # noqa: BLE001
            return ""  # 读不到就当作未执行（最多多跑一次，不阻断）

    def _set_day_marker(self, key: str, day: str) -> None:
        try:
            dbm.set_meta(self.conn, key, day)
        except Exception as exc:  # noqa: BLE001
            self.logger.debug("写入日标记 %s 失败: %s", key, exc)

    async def _nightly(self, day: str) -> None:
        try:
            # 步骤 1：账本归档 + 备份——**不依赖 LLM，先跑且必跑**。
            # （审查缺陷：原结构把它们放在 provider 探测之后，provider_id
            # 配置无效时会随 return 一起停摆，数据安全保障被间接掐断。）
            try:
                days = int(self.config.get("ledger.retention_days", 180) or 180)
                pruned = self.store.prune_ledger(utc_now_ts() - days * 86400.0)
                if pruned:
                    self.logger.info("账本归档 %d 条（超过 %d 天）", pruned, days)
            except Exception as exc:  # noqa: BLE001
                self.logger.warning("账本归档失败: %s", exc)
            if self.config.get("backup.daily_json", True):
                path = write_backup(self.paths, self.conn,
                                    keep=int(self.config.get("backup.keep_copies", 3) or 3))
                if path:
                    self.logger.info("已导出备份: %s", path.name)
            # 归档备份完成即落当日标记——巩固失败不阻断次日周期
            self._set_day_marker("last_digest_day", day)

            # v0.2.0：笔记切片渐进回填——不依赖 LLM，失败下次继续（幂等）
            try:
                done = await self.engine.backfill_note_chunks(
                    limit=int(self.config.get("notes.chunk_backfill_limit", 20) or 20)
                )
                if done:
                    self.logger.info("笔记切片回填 %d 篇", done)
            except Exception as exc:  # noqa: BLE001
                self.logger.warning("笔记切片回填失败（下次重试）: %s", exc)

            # v0.2.14：向量惰性回填——异维向量只在加载点入队，夜间批量
            # 消费；单日上限防换模型后一次性打爆嵌入额度。该步骤不依赖 LLM。
            if self.config.get("retrieval.vector_backfill_enabled", True):
                daily_limit = max(0, int(self.config.get_num(
                    "retrieval.vector_backfill_daily_limit", 200)))
                attempted = 0
                processed = failed = skipped = 0
                while attempted < daily_limit:
                    batch = min(16, daily_limit - attempted)
                    try:
                        result = await self.engine.backfill_vectors(batch=batch)
                    except Exception as exc:  # noqa: BLE001
                        self.logger.warning("向量惰性回填失败（下次夜间继续）: %s", exc)
                        break
                    n = int(result.get("processed", 0) or 0)
                    f = int(result.get("failed", 0) or 0)
                    attempted += n + f
                    processed += n
                    failed += f
                    skipped += int(result.get("skipped", 0) or 0)
                    if not n and not f:
                        break
                    # 失败批次暂不在同一夜重复打 API，避免连续失败快速耗尽 attempts。
                    if f:
                        break
                if processed or failed or skipped:
                    self.logger.info(
                        "向量惰性回填完成：处理 %d，失败 %d，跳过 %d，今日尝试 %d/%d",
                        processed, failed, skipped, attempted, daily_limit,
                    )

            # 步骤 2：巩固 + 淘汰审查——依赖 LLM。provider 未就绪（启动竞态
            # 窗口，首 tick 已延后 60s 基本关闭）只跳过本日巩固，不再 return。
            pid = str(self.config.get("provider_id", "") or "").strip()
            ready = None
            if pid:
                try:
                    ready = await self.ctx.provider_manager.get_provider_by_id(pid)
                except Exception:  # noqa: BLE001
                    ready = None
            if pid and ready is None:
                self.logger.warning("夜间巩固跳过：LLM 提供商 %s 尚未就绪（明日窗口再试）", pid)
            else:
                try:
                    stats = await self.engine.consolidate()
                    if stats.get("merged"):
                        self.logger.info("夜间巩固合并 %d 簇", stats["merged"])
                except Exception as exc:  # noqa: BLE001
                    self.logger.warning("夜间巩固失败: %s", exc)
                # 淘汰审查：LLM 终审冷记忆（delete/keep/promote），失败整批跳过
                if self.config.get("retirement.enabled", True):
                    try:
                        rs = await self.engine.review_retirement(
                            limit=int(self.config.get("retirement.max_candidates", 20) or 20)
                        )
                        if rs.get("candidates"):
                            self.logger.info("淘汰审查完成：%s", rs)
                    except Exception as exc:  # noqa: BLE001
                        self.logger.warning("淘汰审查失败: %s", exc)
        finally:
            # 防重入标志覆盖整个流程（含 finally），只有真正结束才释放，
            # 否则下个 tick 会在旧任务未完成时重复 spawn。
            self._nightly_running = False

    # ---------------------------------------------------------------- 收尾
    async def terminate(self) -> None:
        try:
            await self.tasks.shutdown()
        except Exception as exc:  # noqa: BLE001
            self.logger.debug("任务收尾异常: %s", exc)
        # 先停星图后台线程再关连接：它若还在用共享连接，close 会造成
        # 跨线程访问已关闭连接的崩溃（graph.stop 内部 join，最多等 1 秒）
        try:
            from .core import graph
            graph.stop()
        except Exception as exc:  # noqa: BLE001
            self.logger.debug("星图线程收尾异常: %s", exc)
        try:
            if self.config.get("backup.daily_json", True):
                write_backup(self.paths, self.conn, keep=int(self.config.get("backup.keep_copies", 3) or 3))
        except Exception:  # noqa: BLE001
            pass
        try:
            self.conn.close()
        except Exception:  # noqa: BLE001
            pass
        self.logger.info("好想记住你插件已卸载")
