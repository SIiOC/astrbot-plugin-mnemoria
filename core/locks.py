"""按 key 的异步分区锁，防止并发写导致去重竞态（TOCTOU）。

每个 key 一把 asyncio.Lock。dict 操作只在无 await 间隙的同步段进行
（asyncio 单线程语义下安全），同名 key 串行化、不同 key 并行。
"""

from __future__ import annotations

import asyncio


class PartitionLock:
    def __init__(self) -> None:
        self._locks: dict[str, asyncio.Lock] = {}

    class _Ctx:
        __slots__ = ("owner", "key", "lock")

        def __init__(self, owner: "PartitionLock", key: str, lock: asyncio.Lock) -> None:
            self.owner = owner
            self.key = key
            self.lock = lock

        async def __aenter__(self) -> "PartitionLock._Ctx":
            await self.lock.acquire()
            return self

        async def __aexit__(self, *exc) -> None:
            self.lock.release()

    def hold(self, key: str) -> "PartitionLock._Ctx":
        """用法：`async with plock.hold(key): ...`（自动获取）。"""
        lock = self._locks.get(key)
        if lock is None:
            lock = asyncio.Lock()
            self._locks[key] = lock
        return PartitionLock._Ctx(self, key, lock)
