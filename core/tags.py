"""检索锚点标签（tags）的规则派生（v0.2.2）。

angel 的 tags 是「主体锚点 + 场合词」：场合词常不出现在正文里，却是查询时
会说的词（正文写「跑步」，查询说「运动/晚上有什么安排」）。mnemoria 自
v0.2.1 起在抽取提示词里用案例教学让模型输出 tags，新记忆覆盖率已达 98%，
但仍有两个口子：
- 模型偶发漏输出 → tags 为空，这条记忆的标签检索通道直接失效；
- v0.2.1 之前写入的历史记忆大量没有 tags（线上实测约 90%）。

本模块用纯规则从 content 派生 tags，两条路径共用同一份逻辑：
- 抽取回退：模型没给 tags 时兜底，零成本、不依赖 LLM；
- 存量回填：scripts/backfill_memory_tags.py（默认 dry-run，先快照）。

派生优先级（达到 limit 即止）：身份锚点 > 「」引用词 > 画像维度 >
拉丁/数字 token（模型名、产品名、ID）> 场合词同义词。
"""

from __future__ import annotations

import re

#: 每条记忆最多派生几个 tag（与抽取提示词的 1~4 对齐，略放宽到 5）
DEFAULT_LIMIT = 5

#: 身份锚点：聊天平台 ID（微信风格 openid / QQ 风格纯数字）
_RE_WECHAT_ID = re.compile(r"[A-Za-z0-9_-]{6,}@im\.[A-Za-z0-9.]+")
_RE_NUMERIC_ID = re.compile(r"(?<!\d)\d{7,12}(?!\d)")
#: 日期形态的数字（20260920）不是身份，排除
_RE_DATE_LIKE = re.compile(r"^(?:19|20)\d{6}$")

#: 「」/『』引用词（模型按 v0.2.1 引号约定写入的原话/专名）
_RE_QUOTED = re.compile(r"[「『]([^」』]{2,16})[」』]")

#: 拉丁/数字 token：模型名、产品名、API 名（glm5.3turbo、qwen-image-3.0-pro）。
#: 允许至多两个空格连接的英文词，把「Star Lantern」「Seed Realtime」
#: 这类多词名称收成单个 tag（拆成 Saving/grace 会稀释锚点）。
_RE_LATIN = re.compile(r"[A-Za-z][A-Za-z0-9._+-]*(?:[ ]+[A-Za-z0-9][A-Za-z0-9._+-]*){0,2}")

#: 画像五维关键词（命中即补一个固定维度 tag，angel 同款）
_DIM_KEYWORDS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("用户别名", ("称呼", "昵称", "别名", "代号", "叫他", "名字是")),
    ("技能树", ("技能", "工具", "模型", "会用", "掌握", "在学习", "配置",
              "部署", "代码", "插件", "api", "sdk")),
    ("关系图谱", ("关系", "互动", "撒娇", "调侃", "亲密", "约定", "承诺",
               "朋友", "群友", "主理人", "反击")),
    ("活跃项目", ("项目", "推进", "训练", "语料", "计划", "正在做", "备考",
               "开发", "作品", "乐团")),
    ("事实属性", ("喜欢", "偏好", "习惯", "作息", "健康", "所在地", "职业",
               "背景", "关注", "需求", "高标准", "风格")),
)

#: 场合词同义词表：触发词在正文里 → 补查询时会说的词。
#: 发布版刻意留空——不应预设任何特定用户的内容；使用者可按自己的
#: 记忆库在此追加（格式：(触发词...), (场合词...))。留空时标签派生
#: 仍由身份锚点/「」引用词/画像维度/拉丁 token 四级完成。
_OCCASION_LEXICON: tuple[tuple[tuple[str, ...], tuple[str, ...]], ...] = (
)

#: 拉丁 token 停用词（无检索价值）
_LATIN_STOP = {"ok", "pc", "app", "http", "https", "www", "com", "cn", "the",
               "and", "json", "api", "sdk", "tts"}


def identities_from_label(label: str) -> list[str]:
    """从「名字（id）」形态的指称里拆出身份锚点候选。

    _user_label 产出形如 「Star Lantern（3141592653）」或「用户（u1）」；
    「用户（u1）」是占位指称（没有真实身份），整体不作为锚点。
    """
    text = str(label or "").strip()
    if not text:
        return []
    out: list[str] = []
    m = re.match(r"^(.*?)[（(]([^）)]+)[）)]$", text)
    if m:
        name, ident = m.group(1).strip(), m.group(2).strip()
        if name.startswith("用户"):
            return []
        out.extend([name, ident])
    elif not text.startswith("用户"):
        out.append(text)
    return [x for x in out if x and len(x) >= 2]


#: tag 最长 40 字：场合词一般 ≤12 字，但平台 openid / 长模型名
#: （qwen-image-3.0-pro）可能到 30+ 字，截断会让身份锚点与产品名失真
_TAG_MAX_LEN = 40


def _norm_tag(tag: str) -> str:
    return re.sub(r"\s+", " ", str(tag or "").strip())[:_TAG_MAX_LEN]


def normalize_tags(tags, *, limit: int = DEFAULT_LIMIT) -> list[str]:
    """去重、去空、截断并限量（保序；与 store 写入侧同一份规则）。"""
    out: list[str] = []
    for t in tags or []:
        s = _norm_tag(t)
        if s and s not in out:
            out.append(s)
        if len(out) >= max(1, int(limit)):
            break
    return out


def derive_tags(content: str, *, memory_type: str = "",
                identities=(), limit: int = DEFAULT_LIMIT) -> list[str]:
    """从记忆正文规则派生 tags（保序去重，达到 limit 即止）。"""
    text = str(content or "")
    if not text.strip():
        return []
    low = text.lower()
    out: list[str] = []

    def push(tag: str) -> bool:
        s = _norm_tag(tag)
        if not s or s in out:
            return False
        out.append(s)
        return len(out) >= max(1, int(limit))

    # 1) 身份锚点：调用方给的（user_ledger/指称拆解）+ 正文里的平台 ID
    for name in identities or []:
        n = str(name or "").strip()
        if len(n) >= 2 and n in text and push(n):
            return out
    for m in _RE_WECHAT_ID.finditer(text):
        if push(m.group(0)):
            return out
    for m in _RE_NUMERIC_ID.finditer(text):
        if _RE_DATE_LIKE.match(m.group(0)):
            continue
        if push(m.group(0)):
            return out

    # 2) 「」引用词（原话/专名，查询时常直接引用）
    for m in _RE_QUOTED.finditer(text):
        if push(m.group(1)):
            return out

    # 3) 画像维度（最多补两个，避免挤占其它锚点）
    dim_hits = 0
    for dim, words in _DIM_KEYWORDS:
        if any(w in low for w in words):
            if push(dim):
                return out
            dim_hits += 1
            if dim_hits >= 2:
                break

    # 4) 拉丁/数字 token（模型名、产品名）。先剔除已收录的 openid
    #    整串及其本地部分/域名——否则一条身份会占掉 3 个 tag 名额
    scan = text
    for ident in out:
        if "@" in ident:
            scan = scan.replace(ident, " ")
            local, _, domain = ident.partition("@")
            scan = scan.replace(local, " ").replace(domain, " ")
    latin_hits = 0
    for m in _RE_LATIN.finditer(scan):
        tok = m.group(0)
        if len(tok) < 3 or tok.lower() in _LATIN_STOP:
            continue
        if push(tok):
            return out
        latin_hits += 1
        if latin_hits >= 3:
            break

    # 5) 场合词同义词（查询语言；正文写了触发词就补）
    for triggers, extras in _OCCASION_LEXICON:
        if any(t in low for t in triggers):
            for extra in extras:
                if push(extra):
                    return out
    return out
