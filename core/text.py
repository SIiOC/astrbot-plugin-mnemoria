"""纯函数：文本规范化与指纹。"""

from __future__ import annotations

import hashlib
import re
import unicodedata

_WS = re.compile(r"\s+")
_PUNCT = re.compile(r"[\s，。！？、；：,.!?;:~…—“”\"'()（）\[\]【】<>《》-]+")


def normalize(text: str) -> str:
    """归一化：NFKC + 折叠空白 + 去首尾。用于比较与去重。"""
    if not text:
        return ""
    nfkc = unicodedata.normalize("NFKC", text)
    return _WS.sub(" ", nfkc).strip()


def fold(text: str) -> str:
    """更激进的折叠：去掉空白与标点、转小写。用于指纹比对。"""
    if not text:
        return ""
    nfkc = unicodedata.normalize("NFKC", text).lower()
    return _PUNCT.sub("", nfkc)


def content_hash(text: str) -> str:
    """内容指纹（对 fold 后的文本取 sha256 前 32 位）。"""
    return hashlib.sha256(fold(text).encode("utf-8")).hexdigest()[:32]


def is_mostly_emoji_or_punct(text: str) -> bool:
    """判断文本是否几乎只有表情/标点/空白（无实义内容）。"""
    stripped = fold(text)
    return len(stripped) < 2


def truncate(text: str, limit: int) -> str:
    if limit <= 0 or text is None:
        return ""
    if len(text) <= limit:
        return text
    return text[: max(0, limit - 1)] + "…"


def estimate_tokens(text: str) -> int:
    """粗略 token 估算：中文按 1 字 ≈ 1 token，其余按 4 字符 ≈ 1 token。

    只用于注入预算，不追求精确（宁可略高估）。
    """
    if not text:
        return 0
    cjk = sum(1 for ch in text if "\u4e00" <= ch <= "\u9fff")
    other = len(text) - cjk
    return cjk + (other + 3) // 4


_TAG = re.compile(r"<[^>]{1,80}>")


def sanitize_for_context(text: str, max_chars: int = 200) -> str:
    """注入给模型前的条目消毒（persistent-memory 同款防线）。

    UNTRUSTED 包裹是外层声明，消毒是内层实质：剥 HTML/伪标签
    （防伪装成系统标签的间接注入）、折叠换行（防伪造消息边界）、截断。
    """
    if not text:
        return ""
    t = _TAG.sub("", str(text))
    t = t.replace("\r", " ").replace("\n", "；")
    t = _WS.sub(" ", t).strip()
    return truncate(t, max_chars)
