"""写入门（admission barrier）。

综合多家（MemOS 的 α 门、memoripy 的 admission barrier、persistent-memory 的
噪声过滤）后的判定链。设计的核心是「什么不配被记住」：

1. 噪声过滤：过短、纯表情/标点、纯链接；开头祈使前缀（"记住…""提醒我…"）
   **剥离后按正文入库**（v0.1.9，旧实现整条隔离=白丢记忆）；真正的
   提示注入型元指令（"忽略以上指令""你是…"）仍入隔离区。
2. 秘密过滤：疑似密钥/token/密码的串直接拒收。
3. 防回声：source=assistant 的内容默认不采纳（除非来自用户消息）；
   明确标记为「助手代用户表述」的内容拒收。v0.1.5 起支持分级策略
   assistant_claim_policy（reject_all / allow_relationship / allow_all）：
   陪伴场景下「当X…时我会…」这类互动/约定记忆是高价值内容，
   一刀切拒收会把它们系统性丢掉（对照 angel 的差距主因之一）。
4. 价值门 α：条目自带 alpha 分，低于阈值拒收；套话/寒暄/纯情绪附和直接归零。
5. 去重：向量余弦超过阈值 → 走强化（reinforce）而非新增；
   无向量时以 content_hash 指纹去重。
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from enum import Enum

from .text import content_hash, fold, is_mostly_emoji_or_punct, normalize
from .vector import cosine


class Verdict(str, Enum):
    ACCEPT = "accept"        # 新记忆
    REINFORCE = "reinforce"  # 与既有记忆重复 → 强化
    QUARANTINE = "quarantine"  # 疑似敏感/提示注入 → 入隔离区待人工审（memoripy 同款）
    REJECT = "reject"        # 拒收


@dataclass
class AdmissionResult:
    verdict: Verdict
    reason: str = ""
    target_id: str | None = None  # REINFORCE 时的既有记忆 id
    similarity: float = 0.0
    # 内容改写建议（v0.1.9）：剥离开头祈使前缀后的正文；空串 = 原样使用。
    # 调用方（engine.remember）在 verdict=ACCEPT 时优先采用它落库。
    content: str = ""


# 提示注入型元指令：确属"对 AI 下的命令或人格覆盖"，隔离待人工审。
# v0.1.9 收紧：旧实现只要求子串含"你是"——"你是我的唯一"这类正常情话会被
# 误隔离；现在只认明确的注入/人格覆盖形态。
_INJECTION_PATTERNS = [
    re.compile(r"(系统提示|请你扮演|请你假装|忽略以上|ignore (all )?previous)"),
    re.compile(r"(从现在起|接下来|以后)\s*你(就)?是"),
    re.compile(r"^\s*你是(一个|一名|AI|人工智能|语言模型|助手|机器人)"),
]

# 开头祈使前缀（v0.1.9）："记住…"/"提醒我…"这类句式常见于用户对 AI 的
# 命令，也常见于**抽取产物**的正常记忆正文（如"记住小明喜欢深夜听广播"）。
# 旧实现把两者一律判为元指令送入隔离区——等于白丢一条记忆（误伤面，
# 2026-09-19 审查发现）。现在改为剥离前缀后照常入库。
_LEADING_META_RE = re.compile(
    r"^\s*(?:请你?|麻烦你?)?\s*(?:帮我|替我|给我)?\s*"
    r"(?:记住|记一下|记下|记录下来|别忘了|别忘记|提醒我)"
    r"\s*[:：,，、.。]?\s*"
)
_SECRET_PATTERNS = [
    re.compile(r"\b(sk|pk|ghp|xox[baprs])[-_][A-Za-z0-9\-_]{16,}\b"),  # API keys
    re.compile(r"\b[A-Za-z0-9_\-]{32,}\b"),                            # 长随机串
    re.compile(r"(密码|password|passwd|token|密钥)\s*[:=是]\s*\S+"),
]
_URL_ONLY = re.compile(r"^(https?://\S+\s*)+$")
_FILLERS = (
    "好的", "嗯嗯", "哈哈", "谢谢", "收到", "明白", "在吗", "早", "晚安",
    "哈哈哈", "哦哦", "是的", "好吧", "ok", "OK",
)
_CLICHE_PATTERNS = [
    re.compile(r"(作为(一个)?(AI|人工智能|语言模型|助手))"),
    re.compile(r"(总的来说|综上所述|希望(能)?(对你)?(有所)?帮助|还有什么(可以)?帮(到)?你)"),
]
# allow_relationship 档的互动/关系特征：assistant 记忆带这些信号才放行
# （v0.1.6 增补恋人/交往类标记——亲密关系场景的典型句式）
_RELATIONSHIP_MARKERS = re.compile(
    r"(我会|我们会|约定|承诺|答应|称呼|叫我|叫作|叫做|昵称|"
    r"雷点|忌讳|边界|偏好|喜欢被|讨厌被|当.{1,16}时|"
    r"恋人|情侣|关系是|我们是|交往|对象是)"
)


def looks_like_secret(text: str) -> bool:
    return any(p.search(text) for p in _SECRET_PATTERNS)


def is_meta_instruction(text: str) -> bool:
    """是否含提示注入型元指令（"你是…""忽略以上…"等）。

    v0.1.9：本函数不再判定"记住/提醒我"这类祈使句式——那类文本由
    :func:`strip_meta_prefix` 剥离前缀后照常入库（见模块头与 _LEADING_META_RE）。
    """
    t = normalize(text)
    return any(p.search(t) for p in _INJECTION_PATTERNS)


def strip_meta_prefix(text: str, max_rounds: int = 3) -> tuple[str, bool]:
    """剥离开头的祈使前缀（"记住"/"别忘了提醒我"可叠加，最多剥 3 层）。

    返回 (剥离后文本, 是否发生剥离)。剥完为空串时视为未剥离（保留原文本，
    交由后续噪声门按"无实义短消息"处理），避免把"记住"这类空命令变成空内容。
    """
    cur = str(text or "")
    for _ in range(max(1, int(max_rounds))):
        m = _LEADING_META_RE.match(cur)
        if not m or m.end() == 0:
            break
        rest = cur[m.end():].strip()
        if not rest:
            break
        cur = rest
    return cur, cur != str(text or "")


def is_cliche(text: str) -> bool:
    return any(p.search(text) for p in _CLICHE_PATTERNS)


def _is_filler(text: str) -> bool:
    """寒暄/纯语气词判定。

    注意阈值只取 <=2：中文三字以上（如"我饿了""喜欢猫"）通常带有信息，
    按长度一刀切会误杀正常短句（曾用 <=3 导致"记忆猫"被拒）。
    """
    t = normalize(text)
    if t in _FILLERS:
        return True
    return len(t) <= 2


def _assistant_allowed(text: str, memory_type: str, *, deny: bool, policy: str) -> tuple[bool, str]:
    """分级判定 assistant 来源的记忆是否放行（v0.1.5，P2B）。

    policy 取值：
    - reject_all        全拒（= 旧 deny_assistant_claims=true 的行为）
    - allow_all         全放（= 旧 deny_assistant_claims=false 的行为）
    - allow_relationship 关系/互动类放行：type 为 event/emotional，
      或内容含互动/约定/称呼等特征；纯「AI 代用户立论」的事实仍拒。
    非法/缺失 policy 回落旧开关语义——旧配置（无该键）行为逐字节不变。
    """
    pol = (policy or "").strip().lower()
    if pol not in ("reject_all", "allow_relationship", "allow_all"):
        pol = "reject_all" if deny else "allow_all"
    if pol == "allow_all":
        return True, ""
    if pol == "reject_all":
        return False, "助手代述"
    if (memory_type or "").strip().lower() in ("event", "emotional"):
        return True, ""
    if _RELATIONSHIP_MARKERS.search(text):
        return True, ""
    return False, "assistant 无互动/关系特征"


def assess(
    content: str,
    *,
    alpha: float,
    alpha_threshold: float,
    source: str,
    deny_assistant_claims: bool = True,
    assistant_claim_policy: str = "",
    memory_type: str = "",
) -> AdmissionResult:
    """前置过滤（不含去重）。返回 ACCEPT / REJECT / QUARANTINE。"""
    text = normalize(content)
    if not text:
        return AdmissionResult(Verdict.REJECT, "空内容")
    # v0.1.9：先剥离祈使前缀（"记住小明喜欢猫" → "小明喜欢猫"），
    # 剥离后仍有实义即按普通记忆继续过门，不再整条判元指令。
    text, stripped = strip_meta_prefix(text)
    if is_mostly_emoji_or_punct(text) or _is_filler(text):
        return AdmissionResult(Verdict.REJECT, "无实义短消息")
    if _URL_ONLY.match(text):
        return AdmissionResult(Verdict.REJECT, "纯链接")
    if looks_like_secret(text):
        # 隔离而非丢弃：可能是误判（长订单号/ID），留待人工审（memoripy QUARANTINE 同款）
        return AdmissionResult(Verdict.QUARANTINE, "疑似密钥/隐私串")
    if is_meta_instruction(text):
        return AdmissionResult(Verdict.QUARANTINE, "提示注入/系统指令")
    if source == "assistant":
        allowed, reason = _assistant_allowed(
            text, memory_type, deny=deny_assistant_claims, policy=assistant_claim_policy
        )
        if not allowed:
            return AdmissionResult(Verdict.REJECT, reason)
    if is_cliche(text):
        return AdmissionResult(Verdict.REJECT, "套话")
    if float(alpha) < float(alpha_threshold):
        return AdmissionResult(Verdict.REJECT, f"α={alpha:.2f} 低于门槛")
    return AdmissionResult(
        Verdict.ACCEPT,
        "ok（已剥离祈使前缀）" if stripped else "ok",
        content=text if stripped else "",
    )


def char_ngrams(text: str, n: int = 2) -> set[str]:
    """字符 n-gram 集合（中文按二元组，英文按词）。

    用于文本级相似度二次确认——单靠向量会把「第10条」和「第1条」这类
    字符多重集相近、语义不同的短文本误判为重复（弱嵌入模型下尤其明显）。
    """
    t = fold(text)
    if not t:
        return set()
    grams = {t[i:i + n] for i in range(len(t) - n + 1)}
    if len(t) < n:
        grams = {t}
    return grams


def text_similarity(a: str, b: str) -> float:
    """字符级 Jaccard 相似度，0~1。"""
    ga, gb = char_ngrams(a), char_ngrams(b)
    if not ga or not gb:
        return 0.0
    inter = len(ga & gb)
    union = len(ga | gb)
    return inter / union if union else 0.0


_NUMBER_RE = re.compile(r"\d+(?:\.\d+)?")


def number_signature(text: str) -> tuple[str, ...]:
    """抽出文本中的数字序列（用于识别「仅编号不同」的模板化内容）。"""
    return tuple(_NUMBER_RE.findall(fold(text)))


def differs_only_by_numbers(a: str, b: str) -> bool:
    """判断两段文本是否「只说同一件事、只有数字/编号不同」。

    典型受害场景（模板化抽取内容）：
        「用户的第1条记忆内容」 vs 「用户的第10条记忆内容」
    这类文本字符二元组相似度可达 0.73，向量也可能很高，但它们是不同事实，
    绝不能被当作重复合并——否则记忆会成批丢失（本仓库测试曾实测丢失约 20%）。

    判据：去掉数字后骨架相同，但数字序列不同。
    """
    na, nb = number_signature(a), number_signature(b)
    if not na or not nb or na == nb:
        return False
    skeleton_a = _NUMBER_RE.sub("#", fold(a))
    skeleton_b = _NUMBER_RE.sub("#", fold(b))
    return skeleton_a == skeleton_b


def dedup(
    content: str,
    *,
    scope: str,
    existing: list[tuple[str, str, list[float] | None]],
    new_vec: list[float] | None,
    threshold: float,
    text_confirm: float = 0.7,
    text_dedup: float = 0.80,
    text_scan_cap: int = 2000,
) -> AdmissionResult:
    """去重判定。existing = [(id, content, vec_or_None), ...]（同 scope 的活跃记忆）。

    判定需同时满足：
      1) 向量余弦 >= threshold（无向量时跳过此项）
      2) 文本相似度 >= text_confirm
      3) 二者不是「仅数字不同」的模板化内容（见 differs_only_by_numbers）
    文本指纹完全相同则直接判定重复（最强信号）。

    v0.2.3 新增守卫三（文本级近重复）：向量低于 threshold（弱嵌入/无嵌入）
    时，若与某条既有记忆的字符 Jaccard >= text_dedup（默认 0.80）且不是
    编号模板，同样判重复——「同事实换措辞」（喜欢跑步/爱跑步，实测 0.81）
    靠向量可能只有 0.7+，会漏判成新记忆，造成同一事实多条并存
    （线上跑步兴趣簇 5 条并存即此类）。校准：不同事实的对照对
    （喜欢跑步/喜欢在深夜听广播）实测 0.73，阈值 0.80 有安全边际。
    扫描带长度预过滤与条数上限，控制写入路径开销。
    """
    chash = content_hash(content)
    best_id = None
    best_sim = 0.0
    best_text_id = None
    best_text = 0.0
    scanned = 0
    clen = len(content)
    for mid, ex_content, ex_vec in existing:
        if content_hash(ex_content) == chash:
            return AdmissionResult(Verdict.REINFORCE, "指纹重复", target_id=mid, similarity=1.0)
        if new_vec and ex_vec and len(new_vec) == len(ex_vec):
            sim = cosine(new_vec, ex_vec)
            if sim > best_sim:
                best_sim = sim
                best_id = mid
        # 守卫三的候选扫描：长度差超过 40% 的不可能是同一事实的换措辞
        if scanned < max(0, int(text_scan_cap)):
            scanned += 1
            elen = len(ex_content)
            if abs(clen - elen) <= 0.4 * max(clen, elen, 1):
                ts = text_similarity(content, ex_content)
                if ts > best_text:
                    best_text = ts
                    best_text_id = mid
    if best_id is not None and best_sim >= threshold:
        ex_text = next((c for m, c, _ in existing if m == best_id), "")
        # 守卫一：仅数字不同 → 不同事实，绝不合并
        if differs_only_by_numbers(content, ex_text):
            return AdmissionResult(
                Verdict.ACCEPT, f"向量相近({best_sim:.3f})但仅编号不同，判为不同记忆"
            )
        # 守卫二：文本相似度二次确认，防弱向量误合并
        ts = text_similarity(content, ex_text)
        if ts >= text_confirm:
            return AdmissionResult(
                Verdict.REINFORCE, f"相似 {best_sim:.3f}/文本 {ts:.2f}", target_id=best_id, similarity=best_sim
            )
        return AdmissionResult(
            Verdict.ACCEPT, f"向量相近({best_sim:.3f})但文本差异大({ts:.2f})，判为不同记忆"
        )
    # 守卫三：文本级近重复（向量没到阈值的同事实换措辞）
    if best_text_id is not None and best_text >= float(text_dedup):
        ex_text = next((c for m, c, _ in existing if m == best_text_id), "")
        if not differs_only_by_numbers(content, ex_text):
            return AdmissionResult(
                Verdict.REINFORCE, f"文本近重复 {best_text:.2f}",
                target_id=best_text_id, similarity=best_text,
            )
    return AdmissionResult(Verdict.ACCEPT, "unique")
