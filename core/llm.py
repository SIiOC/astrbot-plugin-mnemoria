"""框架桥接：后台 LLM 调用（抽取/巩固/画像）。

统一封装 context.llm_generate，强制超时；失败返回 None，调用方自行降级。
所有插件内 LLM 调用必须走这里——框架 SDK 默认超时过长（历史教训：
anthropic SDK 默认 600s，会把会话锁攥死）。
"""

from __future__ import annotations

import asyncio
import json
from astrbot.api import logger
import re
from typing import Any


_JSON_BLOCK = re.compile(r"```(?:json)?\s*(.*?)```", re.S)


class LLMBridge:
    def __init__(self, context, provider_id: str | None = None, timeout: float = 120.0,
                 config=None) -> None:
        self.context = context
        self._static_id = (provider_id or "").strip()
        self.config = config  # 传了则惰性读取 provider_id（支持热更新）
        self.timeout = timeout

    @property
    def provider_id(self) -> str:
        if self.config is not None:
            return str(self.config.provider_id or "").strip()
        return self._static_id

    @property
    def enabled(self) -> bool:
        return bool(self.provider_id)

    async def generate(self, prompt: str, system_prompt: str | None = None) -> str | None:
        pid = self.provider_id
        if not pid:
            return None
        try:
            resp = await asyncio.wait_for(
                self.context.llm_generate(
                    chat_provider_id=pid,
                    prompt=prompt,
                    system_prompt=system_prompt,
                ),
                timeout=self.timeout,
            )
        except asyncio.TimeoutError:
            logger.warning("后台 LLM 调用超时（%ss）", self.timeout)
            return None
        except Exception as exc:  # noqa: BLE001
            logger.warning("后台 LLM 调用失败: %s", exc)
            return None
        return _extract_text(resp)

    async def generate_json(self, prompt: str, system_prompt: str | None = None) -> Any | None:
        text = await self.generate(prompt, system_prompt)
        if not text:
            return None
        return parse_json_loose(text)


def _extract_text(resp) -> str | None:
    if resp is None:
        return None
    for attr in ("completion_text", "text", "content", "result"):
        val = getattr(resp, attr, None)
        if isinstance(val, str) and val.strip():
            return val
    if isinstance(resp, str):
        return resp
    if isinstance(resp, dict):
        for key in ("completion_text", "text", "content"):
            if isinstance(resp.get(key), str):
                return resp[key]
    return None


def parse_json_loose(text: str, *, required: tuple[str, ...] = (),
                     optional: tuple[str, ...] = ()) -> Any | None:
    """尽最大努力从模型输出里抠出 JSON（v0.2.1：多候选 + 字段打分）。

    候选：围栏块 → 全文 → 每个平衡括号片段的扫描（字符串/转义安全）；
    可解析候选按 required/optional 字段命中打分，取最高分（同分取先出现）。
    required/optional 为空时退化为「首个可解析候选胜出」——与旧实现兼容。
    """
    if not text:
        return None
    candidates: list[str] = []
    m = _JSON_BLOCK.search(text)
    if m:
        candidates.append(m.group(1))
    candidates.append(text)
    candidates.extend(_balanced_candidates(text))
    best: Any = None
    best_score = -1
    max_score = 100 * len(required) + 10 * len(optional)
    for cand in candidates:
        cand = cand.strip()
        if not cand:
            continue
        for attempt in (cand, _slice_braces(cand)):
            if not attempt:
                continue
            try:
                obj = json.loads(attempt)
            except (ValueError, TypeError):
                continue
            score = _candidate_score(obj, required, optional)
            if score > best_score:
                best, best_score = obj, score
                if score >= max_score:
                    return obj
            break  # 同一候选只取第一个能解析的形式
    return best


def _candidate_score(obj: Any, required: tuple[str, ...],
                     optional: tuple[str, ...]) -> int:
    """候选打分：必填字段权重 100、可选字段权重 10（非字典记 0）。"""
    if not isinstance(obj, dict) or not (required or optional):
        return 0
    return (100 * sum(1 for k in required if k in obj)
            + 10 * sum(1 for k in optional if k in obj))


def _balanced_candidates(text: str, limit: int = 20) -> list[str]:
    """扫描文本里所有平衡的 {...} / [...] 片段（独立开括号逐个匹配）。"""
    out: list[str] = []
    for start, ch in enumerate(text):
        if len(out) >= limit:
            break
        if ch not in "{[":
            continue
        end = _match_json_end(text, start)
        if end is not None:
            out.append(text[start:end + 1])
    return out


def _match_json_end(text: str, start: int) -> int | None:
    """从 start 处的开括号起找平衡闭合位置；字符串与转义均安全。"""
    pairs = {"}": "{", "]": "["}
    stack: list[str] = []
    in_str = False
    esc = False
    for i in range(start, len(text)):
        ch = text[i]
        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
            continue
        if ch == '"':
            in_str = True
            continue
        if ch in "{[":
            stack.append(ch)
        elif ch in "}]":
            if not stack or stack[-1] != pairs[ch]:
                return None
            stack.pop()
            if not stack:
                return i
    return None


def _slice_braces(text: str) -> str:
    """截取第一个 { 到最后一个 } （或数组）。"""
    starts = [i for i in (text.find("{"), text.find("[")) if i >= 0]
    if not starts:
        return ""
    start = min(starts)
    end = max(text.rfind("}"), text.rfind("]"))
    if end <= start:
        return ""
    return text[start:end + 1]
