"""后台任务追踪。

沿用 angel_memory 验证过的模式：所有 asyncio 任务登记在集合中，
terminate 时统一 cancel + gather，杜绝插件卸载后任务泄漏。
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any, Awaitable, Callable

logger = logging.getLogger(__name__)


class TaskRegistry:
    def __init__(self) -> None:
        self._tasks: set[asyncio.Task] = set()
        self._closed = False

    def spawn(self, coro: Awaitable[Any], name: str = "") -> asyncio.Task | None:
        """创建并登记任务；异常在任务内部吞掉并记日志，避免 unhandled。"""
        if self._closed:
            logger.debug("TaskRegistry 已关闭，忽略任务 %s", name)
            return None
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            # 无运行中的事件循环：关闭协程避免 "never awaited" 警告
            logger.debug("当前无事件循环，跳过任务 %s", name)
            close = getattr(coro, "close", None)
            if close:
                close()
            return None
        try:
            task = loop.create_task(coro, name=name or None)
        except RuntimeError as exc:
            logger.warning("创建任务 %s 失败: %s", name, exc)
            return None
        self._tasks.add(task)
        task.add_done_callback(self._on_done)
        return task

    def _on_done(self, task: asyncio.Task) -> None:
        self._tasks.discard(task)
        if task.cancelled():
            return
        exc = task.exception()
        if exc is not None:
            logger.error("后台任务 %s 异常退出: %s", task.get_name(), exc, exc_info=exc)

    def start_periodic(
        self,
        factory: Callable[[], Awaitable[Any]],
        interval: float,
        name: str = "periodic",
    ) -> asyncio.Task | None:
        """按固定间隔重复执行工厂函数产生的协程。

        ⚠️ 先等待一个周期再执行首轮：插件加载期框架尚未注册完 provider
        （实测约 20 秒），首轮 tick 立即执行会让一切依赖 LLM 的后台任务
        （巩固/淘汰/空闲抽取）在空转中烧掉当日执行标记（2026-09-16 启动竞态缺陷）。
        """

        async def _loop() -> None:
            while not self._closed:
                try:
                    await asyncio.sleep(interval)
                except asyncio.CancelledError:
                    raise
                try:
                    await factory()
                except asyncio.CancelledError:
                    raise
                except Exception as exc:  # 单轮失败不终止循环
                    logger.warning("周期任务 %s 单轮失败: %s", name, exc)

        return self.spawn(_loop(), name=name)

    async def shutdown(self) -> None:
        """取消并等待所有任务结束。"""
        self._closed = True
        pending = [t for t in self._tasks if not t.done()]
        for task in pending:
            task.cancel()
        if pending:
            await asyncio.gather(*pending, return_exceptions=True)
        self._tasks.clear()

    @property
    def active_count(self) -> int:
        return len([t for t in self._tasks if not t.done()])
