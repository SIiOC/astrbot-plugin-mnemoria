/* 好想记住你 · 功能设置（v5 重设计的生产版）。
 *
 * 13 分组对齐插件 _conf_schema.json；读取走 GET config（当前生效值），
 * 保存走 POST config/save（就地写 + 框架落盘 + 热生效——桥接对象惰性读配置）。
 */
(function () {
  "use strict";

  var MN = window.MN;
  if (!MN) return;

  var GROUPS = [
    { id: "g_provider", name: "抽取模型", tag: "provider_id",
      desc: "决定「什么对话值得被记住」的抽取 LLM。留空则关闭自动抽取，仅保留工具主动存取与账本。",
      fields: [
        { k: "provider_id", t: "select", opts: "auto", hint: "建议使用具备思考能力的模型" },
        { k: "admission.min_message_length", t: "range", min: 2, max: 40, hint: "总计文本短于此则跳过抽取" }
      ] },
    { id: "g_admission", name: "写入门", tag: "admission",
      desc: "控制「什么才配被记下来」——防垃圾记忆与自我回声的关键闸门。",
      fields: [
        { k: "admission.alpha_threshold", t: "range", min: 0, max: 1, step: 0.05, hint: "记忆价值 α 低于此值拒收" },
        { k: "admission.dedup_threshold", t: "range", min: 0.8, max: 1, step: 0.01, hint: "近重复阈值（向量+文本二次确认）" },
        { k: "admission.deny_assistant_claims", t: "bool", hint: "拒绝把 AI 自己的话当用户事实" },
        { k: "admission.quarantine_max", t: "range", min: 0, max: 1000, step: 10, hint: "隔离区上限，超限丢最旧" }
      ] },
    { id: "g_retrieval", name: "检索", tag: "retrieval",
      desc: "「向量 + 关键词(BM25) + 标签锚点 + 时间近因」四路 RRF 融合，任一通道命中即可召回。",
      fields: [
        { k: "retrieval.embedding_provider_id", t: "select", opts: "auto", hint: "嵌入模型（换库需重嵌）" },
        { k: "retrieval.rerank_provider_id", t: "select", opts: "auto", hint: "留空=不重排" },
        { k: "retrieval.candidate_pool", t: "range", min: 16, max: 256, step: 8, hint: "重排前候选数" },
        { k: "retrieval.rrf_k", t: "range", min: 10, max: 120, step: 5, hint: "RRF 平滑常数" },
        { k: "retrieval.query_expansion", t: "bool", hint: "短查询用近期对话扩展" },
        { k: "retrieval.per_type_limit", t: "range", min: 0, max: 10, hint: "0=不限；分类型配额" }
      ] },
    { id: "g_injection", name: "注入", tag: "injection",
      desc: "稳定画像走系统提示末尾（利于前缀缓存），逐轮动态记忆走用户消息前缀。",
      fields: [
        { k: "injection.enabled", t: "bool", warn: true, badge: "影子模式", hint: "关闭=只记账不注入" },
        { k: "injection.token_budget", t: "range", min: 100, max: 2000, step: 50 },
        { k: "injection.max_items", t: "range", min: 1, max: 30, hint: "每轮注入条数上限" },
        { k: "injection.throttle_turns", t: "range", min: 0, max: 10, hint: "每 N 轮才注入一次" },
        { k: "injection.untrusted_wrap", t: "bool", hint: "用 UNTRUSTED 包裹记忆条目" }
      ] },
    { id: "g_decay", name: "衰减", tag: "decay_policy",
      desc: "三档模型：T0 自然遗忘 / T1 召回判定 / T2 长期保留。",
      fields: [
        { k: "decay_policy.enabled", t: "bool" },
        { k: "decay_policy.half_life_days", t: "range", min: 3, max: 180, hint: "半衰期（天）" },
        { k: "decay_policy.tier0_threshold", t: "range", min: 0, max: 1, step: 0.05 },
        { k: "decay_policy.tier1_threshold", t: "range", min: 0, max: 1, step: 0.05 },
        { k: "decay_policy.trash_retention_days", t: "range", min: 7, max: 120, hint: "回收站保留天数" }
      ] },
    { id: "g_behavior", name: "记忆行为", tag: "memory_behavior",
      desc: "空闲抽取触发、容量哨兵、夜间巩固与定时任务节奏。",
      fields: [
        { k: "memory_behavior.trigger_turns", t: "range", min: 2, max: 30, hint: "累计 N 轮触发抽取" },
        { k: "memory_behavior.idle_seconds", t: "range", min: 0, max: 1800, step: 30, hint: "空闲 N 秒触发；0=禁用" },
        { k: "memory_behavior.capacity_soft", t: "range", min: 200, max: 5000, step: 100, hint: "超容则聚类更激进" },
        { k: "memory_behavior.consolidate_max_calls", t: "range", min: 1, max: 100, hint: "夜间巩固 LLM 上限" },
        { k: "memory_behavior.decay_hour", t: "range", min: 0, max: 23, hint: "衰减扫描整点" },
        { k: "memory_behavior.digest_hour", t: "range", min: 0, max: 23, hint: "夜间巩固整点" }
      ] },
    { id: "g_reflection", name: "反思闭环", tag: "reflection",
      desc: "独立于抽取的 LLM 调用：判近期召回的记忆有用/无用，反馈强化或扣分。",
      fields: [
        { k: "reflection.enabled", t: "bool", warn: true },
        { k: "reflection.turn_threshold", t: "range", min: 2, max: 30 },
        { k: "reflection.idle_seconds", t: "range", min: 60, max: 3600, step: 60 },
        { k: "reflection.penalty_ratio", t: "range", min: 0, max: 1, step: 0.05, hint: "无用记忆扣分比例" }
      ] },
    { id: "g_retirement", name: "淘汰审查", tag: "retirement",
      desc: "夜间巩固后把「已冷下去」的记忆交给 LLM 终审：删 / 留 / 升为长期。",
      fields: [
        { k: "retirement.enabled", t: "bool", warn: true },
        { k: "retirement.max_candidates", t: "range", min: 1, max: 200, hint: "每晚审查上限" }
      ] },
    { id: "g_notes", name: "笔记库", tag: "notes",
      desc: "与「记忆」分开的知识条目：手动新增、AI 自主整理、.md 导入分块。",
      fields: [
        { k: "notes.enabled", t: "bool" },
        { k: "notes.inject_max_items", t: "range", min: 0, max: 20 },
        { k: "notes.chunk_max_chars", t: "range", min: 200, max: 3000, step: 50, hint: ".md 分块上限" },
        { k: "notes.candidate_pool", t: "range", min: 8, max: 128, step: 8 }
      ] },
    { id: "g_ledger", name: "对话账本", tag: "ledger",
      desc: "逐轮保存原始对话，支持「你上次说过…」式回溯，并为夜间巩固提供素材。",
      fields: [
        { k: "ledger.enabled", t: "bool" },
        { k: "ledger.retention_days", t: "range", min: 7, max: 365, hint: "保留天数" },
        { k: "ledger.group_chats", t: "bool", hint: "是否记录群聊" }
      ] },
    { id: "g_profile", name: "用户画像", tag: "profile",
      desc: "从对话中提炼的稳定认知（称呼/喜好/关系/雷点），永不衰减。",
      fields: [
        { k: "profile.enabled", t: "bool" },
        { k: "profile.inject_max_items", t: "range", min: 1, max: 20 },
        { k: "profile.auto_extract", t: "bool" }
      ] },
    { id: "g_runtime", name: "运行时", tag: "runtime",
      desc: "scope 隔离不同记忆域（人格/账号），同 scope 内记忆才互相可见。",
      fields: [
        { k: "runtime.default_scope", t: "text", hint: "默认隔离域" }
      ] },
    { id: "g_backup", name: "备份", tag: "backup",
      desc: "每日导出 JSON 快照，保留若干份。",
      fields: [
        { k: "backup.daily_json", t: "bool" },
        { k: "backup.keep_copies", t: "range", min: 1, max: 30 }
      ] }
  ];

  function $(id) { return document.getElementById(id); }
  var flat = {}, dirty = {}, bound = false;

  /* 字段中文名：键名只在标签下方小字保留（定位用），主标签给用户读 */
  var LABELS = {
    "provider_id": "抽取模型",
    "admission.min_message_length": "最短消息长度",
    "admission.alpha_threshold": "记忆价值阈值 α",
    "admission.dedup_threshold": "近重复阈值",
    "admission.deny_assistant_claims": "拒绝 AI 自述",
    "admission.quarantine_max": "隔离区上限",
    "retrieval.embedding_provider_id": "嵌入模型",
    "retrieval.rerank_provider_id": "重排模型",
    "retrieval.candidate_pool": "重排候选数",
    "retrieval.rrf_k": "RRF 平滑常数",
    "retrieval.query_expansion": "短查询扩展",
    "retrieval.per_type_limit": "分类型配额",
    "injection.enabled": "启用注入",
    "injection.token_budget": "注入 token 预算",
    "injection.max_items": "每轮注入上限",
    "injection.throttle_turns": "注入节流轮数",
    "injection.untrusted_wrap": "UNTRUSTED 包裹",
    "decay_policy.enabled": "启用衰减",
    "decay_policy.half_life_days": "半衰期（天）",
    "decay_policy.tier0_threshold": "T0 遗忘阈值",
    "decay_policy.tier1_threshold": "T1 召回阈值",
    "decay_policy.trash_retention_days": "回收站保留天数",
    "memory_behavior.trigger_turns": "抽取触发轮数",
    "memory_behavior.idle_seconds": "空闲触发秒数",
    "memory_behavior.capacity_soft": "容量软上限",
    "memory_behavior.consolidate_max_calls": "夜间巩固 LLM 上限",
    "memory_behavior.decay_hour": "衰减扫描整点",
    "memory_behavior.digest_hour": "夜间巩固整点",
    "reflection.enabled": "启用反思",
    "reflection.turn_threshold": "反思触发轮数",
    "reflection.idle_seconds": "反思空闲秒数",
    "reflection.penalty_ratio": "无用扣分比例",
    "retirement.enabled": "启用淘汰审查",
    "retirement.max_candidates": "每晚审查上限",
    "notes.enabled": "启用笔记库",
    "notes.inject_max_items": "笔记注入上限",
    "notes.chunk_max_chars": ".md 分块上限",
    "notes.candidate_pool": "笔记候选数",
    "ledger.enabled": "启用账本",
    "ledger.retention_days": "账本保留天数",
    "ledger.group_chats": "记录群聊",
    "profile.enabled": "启用画像",
    "profile.inject_max_items": "画像注入上限",
    "profile.auto_extract": "自动提炼画像",
    "runtime.default_scope": "默认隔离域",
    "backup.daily_json": "每日 JSON 备份",
    "backup.keep_copies": "备份保留份数"
  };

  function val(k) { return flat[k]; }

  function renderField(f) {
    var v = val(f.k);
    var ctl = "";
    if (f.t === "bool") {
      var on = v === true || v === "true" || v === 1 || v === "1";
      ctl = '<span class="cfg-sw' + (on ? " on" : "") + (f.warn ? " warn" : "") + '" data-k="' + f.k + '"></span>';
      if (f.badge) ctl += '<span class="cfg-badge">' + f.badge + "</span>";
    } else if (f.t === "range") {
      var step = f.step || 1;
      var disp = step < 1 ? Number(v == null ? 0 : v).toFixed(2) : (v == null ? 0 : v);
      ctl = '<input type="range" min="' + f.min + '" max="' + f.max + '" step="' + step
        + '" value="' + (v == null ? 0 : v) + '" data-k="' + f.k + '">'
        + '<span class="cfg-val">' + disp + "</span>";
    } else if (f.t === "select") {
      /* provider 下拉：枚举到真实模型列表就用下拉；枚举失败（旧版宿主/接口不可用）
         退化为文本输入，不把人困在空下拉里 */
      var plist = providerOptions(f.k);
      if (plist.length) {
        ctl = '<select data-k="' + MN.esc(f.k) + '"></select>';
      } else {
        ctl = '<input type="text" value="' + MN.esc(v) + '" data-k="' + MN.esc(f.k)
          + '" placeholder="模型 ID（宿主模型列表不可用，请手输）">';
      }
    } else {
      ctl = '<input type="text" value="' + MN.esc(v) + '" data-k="' + MN.esc(f.k) + '">';
    }
    var lab = LABELS[f.k] || f.k;
    return '<div class="cfg-field"><div class="cfg-lab"><b>' + lab + '</b><code class="cfg-key">' + f.k + "</code>"
      + (f.hint ? '<span class="hint">' + f.hint + "</span>" : "") + '</div><div class="cfg-ctl">' + ctl + "</div></div>";
  }

  function render() {
    var sec = "", nav = "";
    GROUPS.forEach(function (g, gi) {
      nav += '<a href="#' + g.id + '" data-g="' + g.id + '"' + (gi === 0 ? ' class="on"' : "") + ">" + g.name + "</a>";
      sec += '<div class="cfg-card" id="' + g.id + '"><h2>' + g.name + '<span class="tag">' + g.tag + "</span></h2>";
      if (g.desc) sec += '<div class="cfg-desc">' + g.desc + "</div>";
      g.fields.forEach(function (f) { sec += renderField(f); });
      sec += "</div>";
    });
    $("cfgNav").innerHTML = nav;
    $("cfgSections").innerHTML = sec;
    bindFields();
  }

  function markDirty(k, v) {
    dirty[k] = v;
    var btn = $("cfgSaveBtn");
    if (btn) {
      var n = Object.keys(dirty).length;
      btn.textContent = n ? "保存配置（" + n + " 项改动）" : "保存配置";
    }
  }

  function bindFields() {
    document.querySelectorAll("#cfgSections .cfg-sw").forEach(function (sw) {
      sw.addEventListener("click", function () {
        sw.classList.toggle("on");
        var on = sw.classList.contains("on");
        markDirty(sw.dataset.k, on);
      });
    });
    document.querySelectorAll('#cfgSections input[type="range"]').forEach(function (r) {
      r.addEventListener("input", function () {
        var step = parseFloat(r.step) || 1;
        var out = r.nextElementSibling;
        if (out) out.textContent = step < 1 ? (+r.value).toFixed(2) : r.value;
        markDirty(r.dataset.k, +r.value);
      });
    });
    document.querySelectorAll("#cfgSections input[type=text]").forEach(function (t) {
      t.addEventListener("change", function () { markDirty(t.dataset.k, t.value.trim()); });
    });
    /* select 的选项：未知值放首位 */
    document.querySelectorAll("#cfgSections select").forEach(function (s) {
      var k = s.dataset.k;
      var cur = val(k);
      var opts = [""].concat(providerOptions(k));
      if (cur && opts.indexOf(cur) < 0) opts.unshift(cur);
      s.innerHTML = opts.map(function (o) {
        var label = o === "" ? "（留空）" : o;
        return '<option value="' + MN.esc(o) + '"' + (o == cur ? " selected" : "") + ">" + MN.esc(label) + "</option>";
      }).join("");
      s.addEventListener("change", function () { markDirty(k, s.value); });
    });
  }

  var providersCache = { chat: [], embedding: [] };
  function providerOptions(k) {
    /* 嵌入字段只给嵌入模型，其余（抽取/重排）给对话模型 */
    if (k === "retrieval.embedding_provider_id") return providersCache.embedding;
    return providersCache.chat;
  }
  async function loadProviders() {
    /* 真实模型列表走后端 providers 端点（枚举 AstrBot provider_manager）；
       失败则保持空列表，renderField 会把 select 退化为文本输入 */
    try {
      var d = await MN.api("providers");
      providersCache = { chat: d.chat || [], embedding: d.embedding || [] };
    } catch (e) {
      providersCache = { chat: [], embedding: [] };
    }
  }

  async function save() {
    if (!Object.keys(dirty).length) { MN.toast("没有改动"); return; }
    var btn = $("cfgSaveBtn");
    btn.disabled = true;
    try {
      var d = await MN.api("config/save", null, { updates: dirty }, "POST");
      MN.toast("已保存 " + (d.applied || []).length + " 项（热生效，已落盘）");
      Object.keys(dirty).forEach(function (k) { flat[k] = dirty[k]; });
      dirty = {};
      btn.textContent = "保存配置";
      MN.loadOverview();
    } catch (e) {
      MN.toast(e.message, true);
    }
    btn.disabled = false;
  }

  window.MNConfig = {
    load: async function () {
      if (!bound) {
        bound = true;
        $("cfgSaveBtn").addEventListener("click", save);
        await loadProviders();
      }
      $("cfgSections").innerHTML = '<div class="cfg-desc">加载配置中…</div>';
      try {
        var d = await MN.api("config");
        flat = d.flat || {};
        dirty = {};
        var btn = $("cfgSaveBtn"); if (btn) btn.textContent = "保存配置";
        render();
      } catch (e) {
        $("cfgSections").innerHTML = '<div class="cfg-desc">配置加载失败：' + MN.esc(e.message) + "</div>";
      }
    }
  };
})();
