"""框架桥接：嵌入（embedding）与重排（rerank）。

只依赖 AstrBot 的 provider 抽象，不直连任何 HTTP。所有调用都加超时，
失败一律降级返回 None（检索层自动少一条通道），绝不因外部模型不可用而抛错。
"""

from __future__ import annotations

import asyncio
from astrbot.api import logger



class Embedder:
    def __init__(self, context, provider_id: str | None = None, timeout: float = 12.0,
                 config=None) -> None:
        self.context = context
        self._static_id = (provider_id or "").strip()
        self.config = config          # 传了 config 就惰性读取 provider_id（支持热更新）
        self.timeout = timeout
        self._provider = None
        self._unavailable = False
        self._resolved_id = ""
        self._last_fail_ts = 0.0

    @property
    def provider_id(self) -> str:
        if self.config is not None:
            return str(self.config.embedding_provider_id or "").strip()
        return self._static_id

    @property
    def enabled(self) -> bool:
        if not self.provider_id:
            return False
        if not self._unavailable:
            return True
        # 冷却重试：provider 可能晚于插件加载（启动竞态）或暂时性故障，
        # 60 秒后允许再试一次，避免一次性失败把向量通道锁死整个运行期
        import time as _time
        return (_time.monotonic() - self._last_fail_ts) >= 60.0

    async def _get_provider(self):
        pid = self.provider_id
        # provider 变更时丢弃缓存，重新解析（配置热更新场景）
        if pid != self._resolved_id:
            self._provider = None
            self._unavailable = False
            self._resolved_id = pid
        if self._provider is not None:
            return self._provider
        if not pid:
            return None
        try:
            self._provider = await self.context.provider_manager.get_provider_by_id(pid)
        except Exception as exc:  # noqa: BLE001
            logger.warning("获取嵌入提供商 %s 失败: %s", pid, exc)
            self._provider = None
        if self._provider is None:
            import time as _time
            self._unavailable = True
            self._last_fail_ts = _time.monotonic()
        else:
            self._unavailable = False  # 冷却后重试成功：解除封锁
        return self._provider

    async def embed(self, texts: list[str], timeout: float | None = None) -> list[list[float]] | None:
        """批量嵌入；按 20 条分片（DashScope 单请求上限）。失败返回 None。

        timeout: 单次调用可覆盖默认超时——注入路径用短超时，
        因为 on_llm_request 是同步等待，长超时会直接拖慢用户回复。
        """
        if not self.enabled or not texts:
            return None
        prov = await self._get_provider()
        if prov is None:
            return None
        t = min(self.timeout, timeout) if timeout else self.timeout
        out: list[list[float]] = []
        try:
            for i in range(0, len(texts), 20):
                chunk = texts[i:i + 20]
                vecs = await asyncio.wait_for(_call_embed(prov, chunk), timeout=t)
                if not vecs:
                    return None
                out.extend(vecs)
            return out
        except asyncio.TimeoutError:
            logger.warning("嵌入请求超时（%ss），本轮向量通道停用", t)
            return None
        except Exception as exc:  # noqa: BLE001
            logger.warning("嵌入调用失败: %s", exc)
            return None

    async def embed_one(self, text: str, timeout: float | None = None) -> list[float] | None:
        res = await self.embed([text], timeout=timeout)
        return res[0] if res else None


async def _call_embed(prov, texts: list[str]) -> list[list[float]] | None:
    """兼容不同框架版本的嵌入接口。"""
    for attr in ("get_embeddings", "embed", "get_embedding"):
        fn = getattr(prov, attr, None)
        if fn is None:
            continue
        try:
            res = await fn(texts)
        except TypeError:
            res = await fn(texts, )
        return _coerce_vectors(res)
    logger.warning("嵌入提供商无可用接口")
    return None


def _coerce_vectors(res) -> list[list[float]] | None:
    if res is None:
        return None
    if isinstance(res, tuple):  # 部分实现返回 (vectors, usage)
        res = res[0]
    out: list[list[float]] = []
    for item in res:
        if isinstance(item, (list, tuple)):
            out.append([float(x) for x in item])
        elif hasattr(item, "embedding"):
            out.append([float(x) for x in item.embedding])
        else:
            return None
    return out or None


class Reranker:
    def __init__(self, context, provider_id: str | None = None, timeout: float = 10.0,
                 config=None) -> None:
        self.context = context
        self._static_id = (provider_id or "").strip()
        self.config = config
        self.timeout = timeout
        self._provider = None
        self._unavailable = False
        self._resolved_id = ""
        self._last_fail_ts = 0.0

    @property
    def provider_id(self) -> str:
        if self.config is not None:
            return str(self.config.rerank_provider_id or "").strip()
        return self._static_id

    @property
    def enabled(self) -> bool:
        if not self.provider_id:
            return False
        if not self._unavailable:
            return True
        # 与 Embedder 同款冷却重试（60s），防启动竞态锁死
        import time as _time
        return (_time.monotonic() - self._last_fail_ts) >= 60.0

    async def rerank(self, query: str, docs: list[str]) -> list[int] | None:
        """返回按相关度重排后的索引顺序；失败返回 None（调用方保持原序）。"""
        if not self.enabled or not docs:
            return None
        pid = self.provider_id
        try:
            if pid != self._resolved_id:
                self._provider = None
                self._unavailable = False
                self._resolved_id = pid
            if self._provider is None:
                self._provider = await self.context.provider_manager.get_provider_by_id(pid)
            if self._provider is None:
                import time as _time
                self._unavailable = True
                self._last_fail_ts = _time.monotonic()
                return None
            res = await asyncio.wait_for(
                self._provider.rerank(query=query, documents=docs), timeout=self.timeout
            )
            return _coerce_rerank(res, len(docs))
        except asyncio.TimeoutError:
            logger.warning("重排请求超时，跳过重排")
            return None
        except Exception as exc:  # noqa: BLE001
            logger.debug("重排失败，跳过: %s", exc)
            return None


def _coerce_rerank(res, n: int) -> list[int] | None:
    if not res:
        return None
    if isinstance(res, tuple):
        res = res[0]
    order: list[tuple[int, float]] = []
    for item in res:
        if isinstance(item, dict):
            idx = item.get("index")
            score = item.get("relevance_score", item.get("score", 0.0))
        elif isinstance(item, (list, tuple)) and len(item) >= 2:
            idx, score = item[0], item[1]
        else:
            continue
        if isinstance(idx, int) and 0 <= idx < n:
            order.append((idx, float(score)))
    if not order:
        return None
    order.sort(key=lambda x: x[1], reverse=True)
    return [i for i, _ in order]
