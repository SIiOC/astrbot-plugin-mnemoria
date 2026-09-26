# 好想记住你部署 Runbook

> 推荐节奏：先以「影子模式」（只记账不注入）并行观察 1~2 天，确认抽取质量后再
> 开启注入/替换现有记忆插件；任一阶段发现异常都可 30 秒回滚（禁用本插件即恢复原状）。

## 阶段 0：部署（5 分钟）

1. 安装插件（二选一）：
   - AstrBot WebUI → 插件市场 搜索「好想记住你」安装；或
   - 复制本目录到 `<AstrBot>/data/plugins/astrbot_plugin_mnemoria`。
2. 重启 AstrBot（或 WebUI 插件页点重载）。
3. WebUI → 插件 → 好想记住你 → 配置：
   - `provider_id` = 你在 AstrBot 里配置的对话模型 id（抽取/裁决用；**必须用全称
     带前缀的 id**，如 `某提供商/某模型`——短名会解析失败。推理型模型出解 30~60s
     属正常，配合自适应超时基线 `admission.adjudication_timeout_seconds=20` 使用）
   - `retrieval.embedding_provider_id` = 你配置的嵌入模型 id（去重阈值 0.92 与
     文本守卫 0.80 的组合标定于 nemotron 与 dashscope qwen3.7-flash 两个嵌入器；
     **换嵌入器后向量分布会变**——若观察到误去重/漏去重，先跑
     `scripts/calibrate_embeddings.py` 按当前嵌入器重标阈值，再动其它）
   - `retrieval.rerank_provider_id` = 可选的本地/云端重排模型 id（不可用时自动跳过）
   - **`injection.enabled = false`**（阶段 1 只记账不注入，先观察抽取质量）
4. 观察启动日志：应出现 `好想记住你插件初始化完成（fts5=True）` 与 4 个工具注册，无 ERROR。
5. 可选定制：想强化中文检索锚点，可编辑 `core/tags.py` 的场合词同义词表
   （`_OCCASION_LEXICON`），按自己记忆库的高频话题添加「触发词 → 查询词」——
   发布版词典刻意留空是特性而非缺失，标签派生的其余四级（身份锚点/引用词/
   画像维度/拉丁 token）始终自动工作。

## 阶段 0.5：迁移 angel 记忆（若在部署时一起做）

angel 侧只读（`migrate_from_angel.py` 以 mode=ro 打开，动前动后 SHA256 核验）：

```
python scripts/migrate_from_angel.py --angel-db "<AstrBot>/data/plugin_data/astrbot_plugin_angel_memory/memory_center/index/simple_memory.db" --out angel_export.json
python scripts/import_export.py import --db "<AstrBot>/data/plugin_data/astrbot_plugin_mnemoria/mnemoria.db" --in angel_export.json --vectorize
```

- **`--vectorize` 必须加**：引擎没有"检索时按需补建"机制，缺向量的记忆在语义通道不可见；
  该开关只补缺失的行，中断重跑自动续补（读 AstrBot 主配置里嵌入提供商的 key，
  用 `--provider-id` 指定你的嵌入提供商 id）。
- 导入幂等：同内容指纹只强化不重复；angel 同文双版本（主动+被动）合并后升为主动。
- 时间保真默认开：迁移记忆保留原始观察时间（时间近因/衰减锚点不漂移）。
- 正式导入应在好想记住你**首启之前**或停机窗口做（避免与运行中插件的写并发）。

## 阶段 1：影子观察（1~2 天）

- 正常使用 bot（angel 继续负责注入，好想记住你只在后台记账+抽取）。
- 每天看一次控制台（WebUI → 插件页 → 好想记住你）：
  - 「总览」记忆数在涨；
  - 「记忆」页抽查抽取质量：内容是否自包含、有没有把客套记成事实；
  - 「回收站」的隔离条目是否都是该拦的（密钥/元指令），有没有误杀正常句子——
    误杀多就把 `admission.alpha_threshold` 从 0.4 降到 0.3。
- 日志里关注：`好想记住你配置` / `抽取记忆 N 条` / 有无重复 WARNING。

## 阶段 2：开注入（确认质量后）

1. `injection.enabled = true`，`injection.token_budget` 从 400 起步。
2. 与 angel 双注入并存的取舍：
   - **保留期（推荐 3~7 天）**：两个都在注入，模型上下文里会有两份记忆（不同库、内容高度重叠）。
     观察有没有互相矛盾的召回；好想记住你注入块有 UNTRUSTED 包裹，不会被执行。
   - **切换日**：确认好想记住你质量 ≥ angel 后，禁用 angel_memory（WebUI 禁用即可，
     其数据目录原样保留），好想记住你接管。
   - 迁移已在阶段 0.5 做过的话无需重复（导入幂等，重跑也安全）。

## 新功能的启用次序（六项扩展，都可独立开关）

下表按默认值与推荐启用时机列出——基础链路跑通前不必全开，逐项对照即可：

| 功能 | 默认 | 建议启用时机 |
|---|---|---|
| 面板新增记忆 / 主动↔被动切换 | 可用 | 随时（纯手动操作，无额外开销） |
| 笔记知识库（`notes.enabled`） | 开 | 随时；导入 .md 前确认 `notes.chunk_max_chars` 合适 |
| 笔记切片检索 / 回填（v0.2.0） | 开 | 自动；夜间每批回填 `notes.chunk_backfill_limit` 篇旧笔记 |
| 多类型分组限额（`retrieval.per_type_limit`） | 0=关 | 观察期后发现某类记忆霸榜时设为 3-5 |
| 写入裁决（`admission.write_adjudication_enabled`） | 开 | 自动；仅在存在相似候选时调用，无候选零开销 |
| LLM 反思闭环（`reflection.enabled`） | 开（v0.2.4 起） | 每 6 轮/空闲一次 LLM 调用；按需关闭 |
| LLM 淘汰审查（`retirement.enabled`） | 开（v0.2.0 起） | 低置信/超时整批保留，删除先快照、只进回收站；保守者可先观察基础链路再放开 |

> 反思与淘汰的分工：反思是记忆唯一的负反馈通道（关闭后分数只加不减），v0.2.4 起默认开；
> 淘汰审查 v0.2.0 起默认开，但加了快照/低置信保留/超时跳过三重保护，
> 且删除只进回收站，风险可控——线上是否放开由使用者按观察结论决定。

## v0.2.3 升级说明（重复与噪音守卫）

- 去重新增**文本级近重复守卫**：向量没到 0.92 时，字符 Jaccard≥0.80 且非编号模板
  即判重复走强化（换措辞的同事实不再多条并存）；不同事实实测 0.73，有安全边际。
  新配置 `admission.text_dedup_similarity`（默认 0.80）。
- **裁决超时不再裸新增**：首位候选向量≥`admission.conservative_fallback_similarity`
  （默认 0.85）或文本≥0.80 时，保守强化首位候选并留审计事件；达不到才普通新增。
  线上痛点：跑步兴趣曾因反复超时回落新增，5 条并存。
- 抽取提示词移植 angel「五问筛选」（临时状态/低价值噪音/高敏/贬损推测不写）+
  「一条记忆一件事」；每轮抽取条数按 alpha 降序兜底截断（`memory_behavior.max_extract_per_turn`，默认 6）。

## v0.2.2 升级说明（tags 检索锚点补全）

- 抽取时模型漏输出 tags，不再落空数组：按「身份锚点 > 「」引用词 > 画像维度 >
  拉丁/数字 token（模型名/产品名）> 场合词同义词」规则派生兜底（零成本、不依赖 LLM）。
- 存量历史记忆（v0.2.1 之前写入、大量没有 tags）可一键回填，标签检索通道随之恢复：
  ```
  python scripts/backfill_memory_tags.py --db <plugin_data>/astrbot_plugin_mnemoria/mnemoria.db
  python scripts/backfill_memory_tags.py --db <...> --apply
  ```
- 回填只填 tags 为空的记忆（已有 tags 一律不碰），先落 JSON 快照、单事务提交；
  纯噪声/无锚点可派的记忆会跳过（线上实测约 4%）。

## v0.2.1 升级说明（画像固定五维）

- 新写入的画像自动归一到五个固定维度（用户别名/事实属性/技能树/关系图谱/活跃项目）；
  注入与面板展示也按维度聚合，历史漂移键（喜好/兴趣/使用习惯…）会与同维度合并显示。
- 存量画像想一次性收敛：先预览、再执行（执行前自动快照到 `backups/`）：
  ```
  python scripts/migrate_profile_taxonomy.py --db <plugin_data>/astrbot_plugin_mnemoria/mnemoria.db
  python scripts/migrate_profile_taxonomy.py --db <...> --apply
  ```
- 只重写 `profiles` 表，不动记忆/笔记/账本；未识别的自定义键原样保留。
- **写入语义**（与迁移后的存量数据一致）：用户别名=覆盖写（称呼会改）；
  事实属性/技能树/关系图谱/活跃项目=合并追加去重（单维度上限 4000 字，按片段边界取舍）——
  模型每轮只需写新增片段，不会把旧事实冲掉；未知键（含助手人设历史占用键「名字」）
  与面板手动编辑保持显式覆盖。

## v0.2.0 升级说明（schema v5）

- 旧库首次启动会自动迁移：**先**在 `plugin_data/astrbot_plugin_mnemoria/backups/`
  生成 `pre-schema-v5-*.db` 一致性备份，**再**补列建表；备份失败会拒绝启动（保护数据）。
- 迁移只新增（`speaker_key` 列、`memory_events`、`note_chunks`/FTS、`user_ledger`、`group_ledger`），
  不改写旧记忆/画像/笔记；旧 `speaker` 与画像键原样保留。
- 回滚到 v0.1.9：停插件、覆盖旧代码即可；新表对新代码不可见，不影响旧版读写。

## 回滚（任一阶段，30 秒）

WebUI → 插件 → 好想记住你 → 禁用。angel 从未动过，零损失。
好想记住你数据留在 `data/plugin_data/astrbot_plugin_mnemoria/`，重启后不丢。

## 已知观察点（试运行重点盯的）

| 观察项 | 异常判据 | 处置 |
|---|---|---|
| 抽取延迟 | 回复明显变慢 >10s（注入路径嵌入超时上限 3s+兜底） | 清空 embedding_provider_id 降级双通道 |
| α 失真 | 隔离区出现大量正常句子 | 降 alpha_threshold 或调 EXTRACT_PROMPT 措辞 |
| 记忆膨胀 | 阶段 2 一周后总记忆 >2000 | 调小 half_life_days（默认 7） |
| 群聊误记 | 账本出现群消息 | 确认 ledger.group_chats=false（默认已关） |
| 与 Humanizer 冲突 | 回复风格异常/日志改写链报错 | 好想记住你钩子 priority=45/-100 已避开；出现则把 injection.enabled=false 先回影子态 |

## 部署前最后一件事

- 运行实例 `data/plugins/` 下若已存在同名目录（旧版本/手动复制过），先备份
  其数据目录（`plugin_data/astrbot_plugin_mnemoria/`）再覆盖，避免新旧文件混杂；
- 若同时启用其它记忆插件（angel_memory / livingmemory 等），建议先在 WebUI
  禁用其一——多个记忆插件同时注入会造成重复上下文
  （见 README「与其它插件共存」）。
