# 好想记住你 (mnemoria)

AstrBot 长期记忆插件：**对话流水账本 + 自动记忆抽取与衰减 + 用户画像 + 四路混合检索注入**。
对标主流记忆插件的**全功能**选择。

零编译依赖（纯标准库 SQLite/FTS5 + 纯 Python 向量余弦），与其它插件零代码耦合。

> **License: AGPL-3.0**（与 AstrBot 框架本体同源）。设计上参考借鉴了多个开源记忆系统
> 的思想（见文末「致谢与借鉴」）；本插件由作者在个人 QQ 机器人环境长期孵化
> （约半年、上千条真实记忆淬炼）后公开发布。
> **发布版不包含任何预设内容**：无用户数据、无个性化标签词典（场合词词典留空，
> 鼓励按自己的记忆库定制）、无个人化示例——一切记忆与画像由你在使用中自行积攒。
> README 的完整迭代史已拆分至 [CHANGELOG.md](CHANGELOG.md)（其中"线上实测"
> 等字样均指作者本人的部署环境）。

---

## 数据处理与隐私

- **全部数据只存在本地**：对话账本、记忆、画像、笔记均保存在 AstrBot 的
  `plugin_data/astrbot_plugin_mnemoria/` 下的单个 SQLite 文件中，不经网络传输到
  作者或任何第三方；插件**无遥测、无统计上报、无任何主动外联**。
- **唯一的网络访问**是调用你在 AstrBot 里自行配置的 LLM / 嵌入 / 重排提供商
  （记忆抽取、写入裁决、反思等功能的必需环节）——发送什么内容、发给谁，
  完全由你的插件配置决定，请自行评估所选提供商的隐私政策。
- **内置防护**：疑似密钥/凭证的内容不入记忆库（进入隔离区待人工审）；
  注入模型的召回内容一律消毒并以 `[UNTRUSTED DATA]` 包裹；群聊记账默认关闭。
- **群聊边界（v1.0.1 起）**：账本（`ledger.group_chats`）与注入
  （`injection.group_inject`）对群聊默认均为关闭——群聊消息不记账、也不注入
  画像/记忆，防止私聊内容在群聊上下文被带出；确需群聊记忆功能时分别开启。
- **命令行脚本的网络访问**：`import_export.py --vectorize`、
  `calibrate_embeddings.py`、`reembed_vectors.py` 会读取 AstrBot 主配置
  （`cmd_config.json`，含各 provider 的 key）并把待嵌入文本发送到你配置的
  嵌入提供商端点——与运行时同一来源，但请知悉这些脚本直接读取配置文件。
- **导出与快照文件是明文**：备份 JSON、迁移导出、维护脚本的快照包含全部
  记忆/画像/笔记明文（含隔离与已删除条目），请像对待聊天记录一样保管，勿外发。

## 功能一览

| 能力 | 说明 |
|---|---|
| **对话流水账本** | 逐轮保存原始对话（SQLite + FTS5 trigram 中文检索），支持「你上次说过…」式回溯 |
| **自动记忆抽取** | 空闲 / 累计轮数触发，后台 LLM 提炼稳定事实，经**写入门**过滤后入库 |
| **四路混合检索** | 向量 + 关键词(BM25) + 标签锚点 + 时间近因，RRF(k=60) 融合，可选重排 |
| **多类型分组限额** | 可选：每类记忆各取 N 条再补满，防单一类型霸榜（`retrieval.per_type_limit`） |
| **用户画像** | 称呼 / 喜好 / 关系 / 雷点等稳定认知，**永不衰减** |
| **热度衰减** | `hotness = sigmoid(log1p(hits)) · 2^(-age/half_life)`，三档 T0/T1/T2，回收站可恢复 |
| **LLM 反思闭环** | 可选：单独调用判「召回的记忆是否真被用到」，据此加分/减分（`reflection`） |
| **LLM 淘汰审查** | 可选：夜间让 LLM 对冷记忆判 删/留/升为长期，失败整批跳过不删（`retirement`） |
| **笔记知识库** | 独立于记忆的知识条目：面板新增、AI 自主整理、导入 `.md` 自动分块；参与检索注入；**v0.2.0 起长笔记按约 500 token 切片检索，命中段落而非整篇**（`notes`） |
| **写入裁决（v0.2.0）** | 去重阈值之下的「同事实换措辞」交给 LLM 判 `add/reinforce/merge/update/noop`；合并/更新不原地覆盖，旧条 `superseded_by` 失效可恢复，全程 `memory_events` 血缘审计（`admission.write_adjudication_*`） |
| **稳定身份账本（v0.2.0）** | 记忆带 `platform:user_id` 稳定身份键，昵称历史入账本（`user_ledger`），改名不断链；旧画像键保持兼容 |
| **重复与噪音守卫（v0.2.3）** | 文本级近重复去重（换措辞同事实不再多条并存）；写入裁决超时/失败时高相似候选保守强化而非裸新增；抽取提示词移植 angel 五问筛选 + 一条记忆一件事；每轮抽取条数代码兜底 |
| **过时记忆治理（v0.2.4）** | 对齐 angel：注入的记忆/画像带相对时间标注（新旧并列可分辨）；检索按记忆年龄温和降权（`_apply_time_decay` 同公式）；写入裁决追加 tag 槽位候选（「改口」型矛盾不再因向量不相似而漏判）；画像更正替换旧片段（不再新旧并存）；`memory_remember` 支持 update/merge 显式更正（可取代永生条目，入回收站可复活）；淘汰审查带创建/召回时间；反思闭环默认开启 |
| **tags 检索锚点（v0.2.2）** | 抽取漏输出 tags 时规则派生兜底（身份锚点/「」引用词/画像维度/场合词同义词）；存量记忆可用 `scripts/backfill_memory_tags.py` 一键回填（默认 dry-run、先快照） |
| **画像固定五维（v0.2.1）** | 借鉴 angel：画像只允许 `用户别名 / 事实属性 / 技能树 / 关系图谱 / 活跃项目` 五维；同维度同 key（旧同义键自动归并）；抽取时喂入已有画像 + 空画像硬底线；tags 场合词案例教学 |
| **主动存取工具** | `memory_remember` / `memory_recall` / `profile_update` / `note_create` |
| **插件页控制台** | 记忆浏览/搜索/新增/编辑/主动切换、笔记管理、画像、账本回看、检索探针、回收站、手动维护 |
| **QQ 命令** | `/记忆状态`、`/记忆搜索 <词>`、`/忘记 <词>`（管理员，入回收站可恢复） |

## 博采众长：融合机制溯源

| 机制 | 来源项目 | 好想记住你实现 |
|---|---|---|
| α 价值准入门 | MemOS `alpha-scorer` | 写入管线第一闸 |
| UNTRUSTED 注入包裹 + 条目消毒 | MemOS / persistent-memory | 注入块双层防线 |
| stable/dynamic 分流保缓存 | TencentDB-Agent-Memory | 画像→system 尾，记忆→user 临时块 |
| 三档衰减 T0/T1/T2 | angel_memory（生态少数实装） | 骨架保留 |
| 热度衰减公式 | OpenViking `memory_lifecycle` | `sigmoid(log1p(hits))·2^(-age/half_life)` |
| 分型 TTL | livingmemory `compute_ttl` | task 2×快消 / knowledge 0.6×耐久（可配） |
| 隔离区 QUARANTINE | memoripy `admission` | 密钥/元指令入隔离区待审，不静默丢弃 |
| 查询扩展 | livingmemory | 短查询拼最近会话上下文进关键词通道 |
| 容量哨兵 | OpenViking | 超软上限后夜间巩固放宽聚类阈值 |
| RRF 相对截断 | Paramecium 垃圾线 | 头名 25% 以下不占 top_k |
| proof_count 增量信念 | hindsight `memory_units` | 重复写入强化而非覆盖 |
| 多类型分组限额 | angel_memory `chained_recall` | 每类先取 N 条，限额挤出的进候补、选完再按 RRF 补满 |
| 分块文档库 | angel_memory 笔记系统 | 笔记库独立于记忆：面板/AI/`.md` 三入口，标题分块 |
| 淘汰审查 delete/keep/promote | angel_memory `memory_retirement_reviewer` | 夜间 LLM 终审冷记忆；输出不可解析则整批跳过不删 |
| 写入阶段操作 ADD/REINFORCE/MERGE/UPDATE/NOOP | Mem0 写入管线 / Letta 记忆编辑 | v0.2.0 写入裁决：低置信/超时/坏输出回退普通新增，合并走双时态失效 |
| 双时态失效 + 来源血缘 | Graphiti edge invalidation | v0.2.0 `memory_events` 审计 + `superseded_by`/`valid_to`，旧条入回收站可恢复 |

## 安装

1. 把整个 `astrbot_plugin_mnemoria` 目录放进 AstrBot 的 `data/plugins/`。
2. 在 WebUI 插件管理中启用（会自动生成配置）。
3. **必填**：在插件配置里选择 `provider_id`（用于记忆抽取的 LLM）。
   - 留空则自动抽取与夜间巩固停用，但**工具与账本仍可用**。
4. 可选：配置 `retrieval.embedding_provider_id` 启用向量检索通道；不配也能用关键词+时间两路。

> 兼容性：`astrbot_version >= 4.5.7`（下界为保守声明）；全部功能在 **AstrBot v4.28** 上
> 开发与实测。作者环境为 Windows，路径处理跨平台，其它平台未经系统测试、欢迎反馈。

完整部署节奏（影子模式 → 开注入 → 替换现有记忆插件）见 [DEPLOY.md](DEPLOY.md)。

## 配置要点

配置面板分 13 组，几个关键项：

- **写入门 (admission)**：`alpha_threshold` 价值门槛（默认 0.4）、`dedup_threshold` 去重阈值（默认 0.92）、
  `deny_assistant_claims` 拒绝「AI 代用户立论」（默认开；v0.1.2 起已接线——抽取要求模型标注 speaker，
  标为 assistant 的条目被拒并记日志）；v0.2.0 新增**写入裁决**：`write_adjudication_enabled`（默认开）、
  `merge_candidate_similarity`（0.78）、`max_similar_candidates`（3）、`adjudication_timeout_seconds`（20）、
  `min_adjudication_confidence`（0.65）。无相似候选时不调用（零开销）；超时/坏输出/低置信时若首位候选足够相似则保守强化（防重复入库），否则回退普通新增——宁可重复也不丢事实。
- **注入 (injection)**：`token_budget` 单次注入预算、`throttle_turns` 每 N 轮才注入一次（省 token / 提升缓存命中）、
  `untrusted_wrap` 用 `[UNTRUSTED DATA]` 包裹召回内容防提示注入。
- **检索 (retrieval)**：`per_type_limit` 同类召回条数上限（默认 0=关，建议 3-5，防单一类型霸榜）。
- **反思 (reflection)**：**v0.2.4 起默认开**（此前默认关）。每 N 轮/空闲做一次独立 LLM 调用，判召回记忆是否有用并加减分
  （`turn_threshold`、`idle_seconds`、`penalty_ratio`）。关闭后记忆只加不减，老旧记忆易长期占据召回位。
- **淘汰 (retirement)**：schema 默认开（v0.2.0 起；保守者可先观察裁决与巩固质量再放开）。
  夜间巩固后让 LLM 终审冷记忆（删/留/升为长期），
  `max_candidates`（20）限输入规模；审查前先落 JSON 快照，删除只进回收站；
  `timeout_seconds`（60）超时或 `min_confidence`（0.7）不足时**整批保留**，不会误删。
- **笔记 (notes)**：笔记库开关、`inject_max_items` 注入条数、`chunk_max_chars` 导入分块上限；
  v0.2.0 切片参数：`chunk_size_chars`（700≈500 token）、`chunk_overlap_chars`（100）、
  `inject_max_chunks_per_note`（2）、`max_chunks_per_note`（8）、`chunk_backfill_limit`（20，夜间回填）。
- **遗忘 (decay_policy)**：`half_life_days` 半衰期、`tier0/1_threshold` 三档阈值、
  `trash_retention_days` 回收站保留天数（默认 30，可恢复）。
- **账本 (ledger)**：`group_chats` 默认**关**（只记私聊，隐私优先）。注入同理：`injection.group_inject` 默认关，群聊消息不注入画像/记忆（v1.0.1 起）。
- **运行 (runtime)**：`default_scope` 记忆隔离域。

> 记忆的手动维护（新增 / 编辑 / 主动↔被动切换 / 删除）全部在**插件页控制台**完成，无需改配置。

## 记忆之外：笔记知识库

笔记与记忆是**两套东西**——记忆是插件从对话里自动提炼的"关于用户的事实"，笔记是你或 AI
主动整理的知识条目（设定、攻略、要点、资料）。笔记有独立数据表，参与检索与注入，
但不受记忆的衰减/淘汰影响。

三种录入方式：
1. **面板手动新增** —— 控制台「笔记」页 → ＋新增笔记。
2. **AI 自主整理** —— 模型可调用 `note_create` 工具沉淀可复用知识（`notes.enabled=false` 时不注册）。
3. **导入 .md 文档** —— 控制台「笔记」页 → 导入 .md，按 `#` 标题自动分块入库，
   记录标题层级路径（如「设定 > 主角」）便于定位来源。超长段落按段落聚合切分，不硬断句子。

笔记与记忆共用同一套回收站生命周期（v0.1.9 起）：删除的笔记进回收站、可在「回收站」页一键恢复，
超过 `decay_policy.trash_retention_days` 后随记忆一起被真正清理。

## 与其它插件共存

- **与 Humanizer**：零代码耦合，可同时启用。两者注入位置不冲突——
  Humanizer 改系统提示与临时内容块，好想记住你把画像加在系统提示末尾、把记忆放在用户消息前的临时块。
  好想记住你的 `on_llm_response` 优先级为 -100（最后执行），因此记账本的是 **Humanizer 改写后的最终回复**。
- **与框架内置「群聊上下文感知」**：若同时开启会出现双份历史注入，建议关闭其中之一。
- **记忆域边界（v1）**：`runtime.default_scope` 指定当前默认记忆域；记忆、笔记、画像、账本检索，以及控制台账本/回收站列表和按 ID 管理操作均按 scope 过滤，不同域不会互相展示或修改。需要跨域迁移时请使用明文导出/导入脚本，并在导入前核对 scope。
- **与 angel_memory**：独立数据库、工具名不冲突（好想记住你统一 `memory_*` / `profile_*` 前缀），
  但两者都做长期记忆时建议只留一个，避免重复注入。

## 测试

测试全部使用 AstrBot 自带 venv（**不要**用系统 Python，缺依赖会误报）。
**v1.0.4 起 pytest 与 `tests/`、`scripts/` 下所有独立脚本都依赖 `astrbot.api` 的
logger**：运行前设置 `ASTRBOT_ROOT` 指向 AstrBot 根目录（`astrbot` 包在源码根
而不在 venv 内），否则脚本会主动报错提示：

```bash
# pytest 套件（推荐）
ASTRBOT_ROOT=/path/to/AstrBot /path/to/AstrBot/venv/Scripts/python -m pytest tests/ -q

# 离线质量评测（可分别指向 v0.1.9 / v0.2.0 插件副本做对比）
python scripts/eval_memory_quality.py --plugin <插件根> --label v0.2.0 --out eval.json

# 独立集成脚本（均需上述 ASTRBOT_ROOT + venv 前置）
python tests/selftest_offline.py    # 核心纯逻辑快检
python tests/integration_smoke.py   # 真框架钩子/工具链路
python tests/web_api_smoke.py       # 29 条 WebAPI 路由
python tests/coexist_smoke.py       # 与 Humanizer 共存
python tests/framework_sim.py       # 真实框架对象仿真（钩子/TextPart/token 隔离）
```

测试覆盖：纯函数、存储层、写入管线、检索融合、注入、抽取、衰减、巩固、并发（TOCTOU）、
边界（超长/Unicode/空值）、配置迁移、性能守门（500~2000 条规模）。

> 测试设计说明：假嵌入刻意用**判别式**向量（不同文本近正交），不用 bag-of-chars——
> 后者会让模板化短句余弦虚高，掩盖真实的去重缺陷。

## 数据与备份

- 数据库：`data/plugin_data/astrbot_plugin_mnemoria/mnemoria.db`（SQLite + WAL）
- 每日 JSON 备份：同目录 `backups/`，默认保留 3 份
- 所有时间戳**统一 UTC 存储**，展示层才转本地时区（规避时区比较类 bug）

## 从 angel_memory 迁移（可选）

**只读保证**：迁移 = 导出副本，不是搬走。脚本以 SQLite 只读模式（`mode=ro`）
打开 angel 数据库，全程不写不改不删；停用 angel 也只是禁用插件，其数据文件原样保留，
随时可重新启用回滚。作者迁移时以 SHA256 前后校验和运动核验过 angel 库逐字节不变
（脚本自身只读；想复核可用 `certutil -hashfile <db> SHA256` 等工具对比）。

```bash
# 1) 只读导出（不动 angel 任何数据）
python scripts/migrate_from_angel.py \
    --angel-db "<AstrBot>/data/plugin_data/astrbot_plugin_angel_memory/memory_center/index/simple_memory.db" \
    --out angel_export.json

# 2) 导入好想记住你（自动按内容指纹去重；angel 内部的完全重复条目会被合并）
python scripts/import_export.py import --db "<好想记住你 mnemoria.db>" --in angel_export.json
```

> 实测记录（2026-09-15，angel v1.6.8，924 条库）：导出 924 条（103 条主动标记）→
> 导入 873 条唯一内容 + 51 条 angel 自身重复被合并。angel 1.6.8 的正文在 `judgment` 列
> （不是 `content`），脚本会自动探测；确认好想记住你运行正常前请勿删除 angel 数据目录。

## 设计取舍（为什么这样做）

| 决策 | 理由 |
|---|---|
| 不用 faiss/numpy | 单机记忆量级（万级）下纯 Python 余弦足够；省去编译依赖与显存占用 |
| trigram FTS5 | 默认分词器不切中文，`高考` 搜不到；trigram 支持中文子串，并保留 LIKE 兜底 |
| 去重加「仅数字不同」守卫 | 实测：字符相似度无法区分同义改写与模板差异（两者区间重叠），必须用结构化判据 |
| 平台型调研结论：保留 T0/T1/T2 | 主流框架普遍**没有**实装衰减（MemOS 为空壳、EverOS/腾讯均无），angel 的三档是少数真跑起来的实现 |
| 容量哨兵而非「做梦式整合」 | mem0 已于 2026-09 全线移除 Dream consolidation（重型后台整合在生产翻车） |
| 注入条目逐条消毒 | persistent-memory 同款防线：剥伪标签/折叠换行/截断——UNTRUSTED 包裹只是声明，消毒才是实质 |
| 显式搜索关兜底 | `/记忆搜索`/`/忘记` 禁用时间近因通道：无匹配就说无匹配，注入路径才保留兜底 |

## 目录结构

```
astrbot_plugin_mnemoria/
├── main.py              # 插件入口：钩子、工具注册、后台循环、收尾
├── metadata.yaml        # 插件元数据（含 pages 声明）
├── _conf_schema.json    # 配置面板 schema（13 组）
├── core/
│   ├── engine.py        # 编排：抽取/检索/注入/衰减/巩固/反思/淘汰/笔记
│   ├── store.py         # SQLite 数据访问（记忆 / 笔记 / 账本 / 画像）
│   ├── db.py            # schema 与迁移（v3：新增 notes 表）
│   ├── admission.py     # 写入门（α 门、防回声、去重+守卫）
│   ├── retrieve.py      # 四路检索 + RRF + 分组限额
│   ├── notes.py         # 笔记库：.md 分块解析 + 混合检索
│   ├── scoring.py       # 热度衰减与分档（纯函数）
│   ├── bridge.py        # embedding / rerank 框架桥接（惰性读配置）
│   ├── llm.py           # 后台 LLM 调用（强制超时）
│   ├── text.py / vector.py  # 纯函数工具
│   ├── web_api.py       # 29 条 WebAPI 路由
│   ├── backup.py        # JSON 备份与轮转
│   ├── config.py        # 配置访问 + 版本迁移
│   ├── locks.py         # 分区锁（防 TOCTOU）
│   ├── tasks.py         # 后台任务追踪
│   └── templates.py     # 提示词模板（抽取/巩固/反思/淘汰）
├── tools/               # 4 个 LLM 工具（含 note_create）
├── pages/console/       # 插件页控制台（纯静态，免构建）
├── scripts/             # 迁移、导入导出、画像归并与离线质量评测
└── tests/               # pytest 套件 + 独立集成脚本
```

## 版本史

完整版本史与孵化期事故记录见 [CHANGELOG.md](CHANGELOG.md)。摘要：

- 1.0.5 — v1.0.4 全量审查修复版（2026-09-28）：脚本运行前置说明补全
  （venv + ASTRBOT_ROOT）、core 版本动态溯源。

- 1.0.4 — 市场审查合规版（2026-09-27）：全插件 logger 统一
  `from astrbot.api import logger`（不再使用内置 logging），CLI 脚本
  导入链适配。

- 1.0.3 — 控制台 scope 边界补全版（2026-09-27）：面板账本/回收站按域
  列出、记忆/笔记按 ID 操作原子域校验、前端 scope 贯穿。

- 1.0.2 — 数据边界与完整性加固版（2026-09-27）：账本检索 scope/role
  隔离、夜间巩固单事务化、裁决回退文本确认、备份状态往返、相似度
  剥离身份前缀、注入消毒补全。

- 1.0.1 — 发布审查修复版（2026-09-26）：群聊注入隐私门控（默认关）、
  CLI 脚本加固（清晰报错/列候选/scope 对齐）、LICENSE 附录与隐私披露补全。

- 1.0.0 — 首个公开发布版（2026-09-24）：发布树经确定性流水线生成，
  零预设内容，与作者内部版本自此分叉；功能快照 = 内部 0.2.15。

- 0.2.20 — v1.0.4 全量审查修复批（脚本运行前置文档、core 版本动态溯源）
- 0.2.19 — 市场审查合规批（logger 统一 `from astrbot.api import logger`，
  CLI 脚本导入链适配）
- 0.2.18 — 控制台 scope 边界补全批（面板账本/回收站按域列出、按 ID 操作
  原子域校验、前端 scope 贯穿）
- 0.2.17 — 数据边界与完整性加固批（账本 scope/role 隔离、巩固事务化、
  裁决回退文本确认、备份状态往返、身份前缀剥离、消毒补全）
- 0.2.16 — 发布审查修复批（群聊注入门控 + CLI 脚本加固 + 文档合规）
- 0.2.15 — v0.2.14 收束审查合并批（undo 护栏血缘方向精化等）
- 0.2.14 — 记忆质量修复：向量混部回填、惰性回填队列、过时记忆治理
- 0.2.12/0.2.13 — 星座星图落地与大数据量降噪
- 0.2.0 ~ 0.2.11 — 写入裁决、双时态血缘审计、tags 检索锚点、画像五维、
  治理与控制台成熟
- 0.1.0 — 首个可用版本

---

## 致谢与借鉴

本项目在设计过程中研究并借鉴了以下开源项目的思想（均只借鉴设计思想与机制，
未复制受版权约束的代码；对应的借鉴点已在代码注释中就地注明）：

- [angel_memory](https://github.com/kawayiYokami/astrbot_plugin_angel_memory) —— 画像固定维度体系、tags 主体锚点 + 场合词、
  淘汰审查的保留规则与「改口」语义、反思闭环的计分思想；
- [mem0](https://github.com/mem0ai/mem0)（Apache-2.0）—— 记忆衰减分带与访问计数封顶的思路；
- livingmemory、memoripy 等 —— 分型 TTL、写入裁决（admission barrier）、隔离区
  （QUARANTINE）等机制的公开设计思想；
- [AstrBot](https://github.com/AstrBotDevs/AstrBot) —— 插件框架、钩子体系与插件页机制。

感谢上述项目的作者与社区。
