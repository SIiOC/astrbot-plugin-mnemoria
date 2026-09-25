/* 好想记住你控制台前端逻辑。通过宿主注入的 AstrBotPluginPage bridge 访问后端 API。 */
(function () {
  "use strict";

  // bridge 脚本由宿主注入在本脚本之后，顶层缓存会拿到 undefined，必须现取
  var state = { view: "overview", scope: "default", memActiveOnly: false };

  // bridge 的 postMessage 请求没有原生超时，宿主不应答时会永远挂起，必须有超时兜底
  var READY_TIMEOUT_MS = 4000;
  var API_TIMEOUT_MS = 20000;
  var PROBE_TIMEOUT_MS = 90000;     // 检索探针走完整嵌入+重排链路，最坏 30s 级
  var IMPORT_TIMEOUT_MS = 600000;   // .md 导入逐块串行嵌入，大文件可达分钟级

  /* ------------------------------------------------------------ 基础工具 */
  function $(id) { return document.getElementById(id); }

  function toast(msg, isErr, undo) {
    /* isErr 可传撤销回调（第二参数为 function 时视为 undo） */
    if (typeof isErr === "function") { undo = isErr; isErr = false; }
    var el = $("toast");
    el.textContent = msg;
    if (undo) {
      el.textContent = "";
      var span = document.createElement("span");
      span.textContent = msg;
      var u = document.createElement("span");
      u.className = "undo"; u.textContent = "撤销";
      u.addEventListener("click", function () { el.className = "toast"; undo(); });
      el.appendChild(span); el.appendChild(u);
    }
    el.className = "toast show" + (isErr ? " err" : "");
    clearTimeout(toast._t);
    toast._t = setTimeout(function () { el.className = "toast" + (isErr ? " err" : ""); }, undo ? 5000 : 2400);
  }

  function esc(s) {
    return String(s == null ? "" : s)
      .replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;")
      .replace(/"/g, "&quot;");
  }

  // 把可能永远挂起的 Promise 转为限时：超时落 reject（结果晚到则丢弃）
  function withTimeout(promise, ms, label) {
    return new Promise(function (resolve, reject) {
      var settled = false;
      var timer = setTimeout(function () {
        if (settled) return;
        settled = true;
        reject(new Error(label));
      }, ms);
      promise.then(function (v) {
        if (settled) return;
        settled = true;
        clearTimeout(timer);
        resolve(v);
      }, function (e) {
        if (settled) return;
        settled = true;
        clearTimeout(timer);
        reject(e);
      });
    });
  }

  // 统一调用：解析 {ok,data} 信封；bridge 不可用时给出可读错误
  async function api(endpoint, params, body, method, timeoutMs) {
    var bridge = window.AstrBotPluginPage;
    if (!bridge) {
      throw new Error("未检测到插件页 bridge（请从 AstrBot 插件页入口打开）");
    }
    if (window.parent === window) {
      throw new Error("独立标签页没有宿主应答（请从 AstrBot WebUI 的插件页入口打开）");
    }
    var ms = timeoutMs || API_TIMEOUT_MS;
    var res;
    try {
      var req = method === "POST"
        ? bridge.apiPost(endpoint + qs(params), body || {})
        : bridge.apiGet(endpoint, params || {});
      res = await withTimeout(req, ms, "宿主 " + Math.round(ms / 1000) + "s 内无应答（可刷新重试）");
    } catch (e) {
      throw new Error("请求失败：" + (e && e.message ? e.message : e));
    }
    // bridge 可能返回信封或裸数据
    var payload = res && typeof res === "object" && "ok" in res ? res : { ok: true, data: res };
    if (!payload.ok) {
      throw new Error(payload.error || "未知错误");
    }
    return payload.data;
  }

  function qs(params) {
    if (!params) return "";
    var parts = [];
    for (var k in params) {
      if (params[k] === undefined || params[k] === null || params[k] === "") continue;
      parts.push(encodeURIComponent(k) + "=" + encodeURIComponent(params[k]));
    }
    return parts.length ? "?" + parts.join("&") : "";
  }

  /* ------------------------------------------------------------ 视图切换 */
  function switchView(name) {
    state.view = name;
    var tabs = document.querySelectorAll(".tab");
    for (var i = 0; i < tabs.length; i++) {
      tabs[i].classList.toggle("active", tabs[i].getAttribute("data-view") === name);
    }
    var views = document.querySelectorAll(".view");
    for (var j = 0; j < views.length; j++) {
      views[j].classList.toggle("active", views[j].id === "view-" + name);
    }
    loadView(name);
  }

  function loadView(name) {
    if (name === "overview") {
      loadOverview();
      // 总览与星图合并：总览页下半部就是记忆星图
      if (window.MNStarmap) window.MNStarmap.load();
    }
    else if (name === "memories") {
      // 记忆库（v6 重设计）：独立模块，缺失时回退不可用提示
      if (window.MNLibrary) window.MNLibrary.load();
      else loadMemoriesLegacy();
    }
    else if (name === "profiles") loadProfiles();
    else if (name === "notes") loadNotes();
    else if (name === "ledger") loadLedger();
    else if (name === "trash") loadTrash();
    else if (name === "settings") loadBackups();
    else if (name === "config" && window.MNConfig) window.MNConfig.load();
  }

  /* ------------------------------------------------------------ 总览 */
  async function loadOverview() {
    var box = $("statCards");
    try {
      var d = await api("overview");
      var c = d.counts || {};
      var cards = [
        { k: "记忆总数", v: c.total || 0, cls: "" },
        { k: "主动记忆", v: c.active || 0, cls: "on" },
        { k: "回收站", v: c.trash || 0, cls: "" },
        { k: "向量检索", v: d.embedding ? "已启用" : "未启用", cls: d.embedding ? "on" : "off" },
        { k: "重排", v: d.rerank ? "已启用" : "未启用", cls: d.rerank ? "on" : "off" },
        { k: "自动抽取", v: d.llm ? "已启用" : "未启用", cls: d.llm ? "on" : "off" },
        { k: "关键词索引", v: d.fts5 ? "FTS5" : "LIKE 降级", cls: d.fts5 ? "on" : "off" }
      ];
      box.innerHTML = cards.map(function (x) {
        return '<div class="card"><div class="k">' + esc(x.k) + '</div><div class="v ' + x.cls + '">' + esc(x.v) + "</div></div>";
      }).join("");
    } catch (e) {
      box.innerHTML = '<div class="card"><div class="k">加载失败</div><div class="v off">' + esc(e.message) + "</div></div>";
    }
  }

  /* ------------------------------------------------------------ 记忆库（v6 重设计） */
  /* 交互与渲染已拆到 library.js（分页/筛选/主动切换/就地编辑/删除撤销/新增），
     这里只保留加载入口与异常回退；CRUD 走与旧版相同的后端端点。 */
  function loadMemoriesLegacy() {
    var box = document.querySelector("#view-memories .lib-list");
    if (box) box.innerHTML = '<div class="empty">记忆库模块未加载（library.js 缺失）</div>';
  }


  /* ------------------------------------------------------------ 画像 */
  var profEditing = null; // {uid,key} 编辑锁位；null=新增

  async function loadProfiles() {
    var tbody = $("profRows");
    tbody.innerHTML = '<tr><td colspan="6" class="empty">加载中…</td></tr>';
    try {
      var user = $("profUser").value.trim();
      var d = await api("profiles", { scope: state.scope, user_key: user });
      var items = d.items || [];
      /* 未按用户过滤时，把出现过的用户 ID 收成 datalist 候选，下次输入可联想 */
      if (!user) {
        var seen = {}, ulist = [];
        items.forEach(function (p) {
          if (p.user_key && !seen[p.user_key]) { seen[p.user_key] = 1; ulist.push(p.user_key); }
        });
        var dl = $("profUserList");
        if (dl) {
          dl.innerHTML = ulist.map(function (u) { return '<option value="' + esc(u) + '"></option>'; }).join("");
        }
      }
      if (!items.length) {
        tbody.innerHTML = '<tr><td colspan="6" class="empty">暂无画像</td></tr>';
        return;
      }
      tbody.innerHTML = items.map(function (p) {
        return "<tr>" +
          "<td>" + esc(p.user_key) + "</td>" +
          "<td>" + esc(p.key) + "</td>" +
          '<td class="content-cell">' + esc(p.value) + "</td>" +
          "<td>" + esc(p.confidence == null ? "" : Number(p.confidence).toFixed(2)) + "</td>" +
          "<td>" + esc(p.updated_at_local || "") + "</td>" +
          '<td><button class="btn mini" data-pedit="' + esc(p.user_key) + "\u0001" + esc(p.key) + '">编辑</button> ' +
          '<button class="btn mini danger" data-uid="' + esc(p.user_key) + '" data-key="' + esc(p.key) + '">删除</button></td>' +
          "</tr>";
      }).join("");
      bindProfileActions(tbody);
    } catch (e) {
      tbody.innerHTML = '<tr><td colspan="6" class="empty">' + esc(e.message) + "</td></tr>";
    }
  }

  function bindProfileActions(tbody) {
    var dels = tbody.querySelectorAll("button[data-key]");
    for (var i = 0; i < dels.length; i++) {
      dels[i].addEventListener("click", async function () {
        if (!(await confirmBox("删除这条画像条目？"))) return;
        try {
          await api("profile/delete", null, {
            scope: state.scope,
            user_key: this.getAttribute("data-uid"),
            key: this.getAttribute("data-key")
          }, "POST");
          toast("已删除");
          loadProfiles();
        } catch (e) { toast(e.message, true); }
      });
    }
    var eds = tbody.querySelectorAll("button[data-pedit]");
    for (var j = 0; j < eds.length; j++) {
      eds[j].addEventListener("click", function () {
        var parts = this.getAttribute("data-pedit").split("\u0001");
        profEditing = { uid: parts[0], key: parts[1] };
        $("profFormTitle").textContent = "编辑画像条目";
        $("profFUid").value = parts[0]; $("profFUid").disabled = true;
        $("profFKey").value = parts[1]; $("profFKey").disabled = true;
        var tr = this.closest("tr");
        $("profFValue").value = tr.children[2].textContent;
        $("profFConf").value = tr.children[3].textContent;
        toggleProfPanel(true);
      });
    }
  }

  function toggleProfPanel(show) {
    var panel = $("profCreatePanel");
    if (!panel) return;
    var willShow = show === undefined ? panel.style.display === "none" : show;
    panel.style.display = willShow ? "block" : "none";
    if (willShow) { $("profFUid").focus(); }
  }

  function profFormReset() {
    profEditing = null;
    $("profFormTitle").textContent = "新增画像条目";
    $("profFUid").disabled = false; $("profFKey").disabled = false;
    $("profFUid").value = ""; $("profFKey").value = ""; $("profFValue").value = ""; $("profFConf").value = "1";
  }

  async function submitProf() {
    var uid = $("profFUid").value.trim();
    var key = $("profFKey").value.trim();
    var value = $("profFValue").value.trim();
    if (!uid || !key || !value) { toast("用户 ID、维度、取值都不能为空", true); return; }
    try {
      await api("profile/save", null, {
        scope: state.scope, user_key: uid, key: key, value: value,
        confidence: Number($("profFConf").value || 1)
      }, "POST");
      toast(profEditing ? "已更新画像" : "已新增画像");
      toggleProfPanel(false);
      profFormReset();
      loadProfiles();
    } catch (e) { toast(e.message, true); }
  }

  /* ------------------------------------------------------------ 笔记 */
  async function loadNotes() {
    var tbody = $("noteRows");
    tbody.innerHTML = '<tr><td colspan="6" class="empty">加载中…</td></tr>';
    var q = $("noteSearch").value.trim();
    try {
      var d = await api("notes", { scope: state.scope, q: q || "", limit: 200 });
      var items = d.items || [];
      $("noteCount").textContent = items.length + " 条";
      if (!items.length) {
        tbody.innerHTML = '<tr><td colspan="6" class="empty">没有笔记</td></tr>';
        return;
      }
      var srcLabel = { manual: "手动", ai: "AI", file: "文档" };
      tbody.innerHTML = items.map(function (n) {
        /* 标签：建库时收了却一直没展示——渲染成可点小片，点击即以该标签搜索
           （LIKE 通道明确覆盖 tags 列；FTS 通道退化为内容匹配，仍可接受） */
        var tags = String(n.tags || "").split(/[,，]/).map(function (t) { return t.trim(); })
          .filter(function (t) { return !!t; });
        var tagsHtml = tags.length
          ? '<div class="note-tags">' + tags.map(function (t) {
              return '<span class="note-tag" data-ntag="' + esc(t) + '">' + esc(t) + "</span>";
            }).join("") + "</div>"
          : "";
        return "<tr>" +
          "<td>" + esc(n.title || "—") + tagsHtml + "</td>" +
          '<td class="content-cell">' + esc(n.content) + "</td>" +
          "<td>" + esc(srcLabel[n.source] || n.source || "") + "</td>" +
          "<td>" + esc(n.heading || n.file_name || "") + "</td>" +
          "<td>" + esc(n.updated_at_local || "") + "</td>" +
          "<td>" +
          '<button class="btn mini" data-nedit="' + esc(n.id) + '">编辑</button> ' +
          '<button class="btn mini" data-ndel="' + esc(n.id) + '">删除</button>' +
          "</td></tr>";
      }).join("");
      bindNoteActions(tbody, items);
      var tagEls = tbody.querySelectorAll("[data-ntag]");
      for (var ti = 0; ti < tagEls.length; ti++) {
        tagEls[ti].addEventListener("click", function () {
          $("noteSearch").value = this.getAttribute("data-ntag");
          loadNotes();
        });
      }
    } catch (e) {
      tbody.innerHTML = '<tr><td colspan="6" class="empty">' + esc(e.message) + "</td></tr>";
    }
  }

  function bindNoteActions(tbody, items) {
    var byId = {};
    items.forEach(function (n) { byId[n.id] = n; });
    var dels = tbody.querySelectorAll("button[data-ndel]");
    for (var i = 0; i < dels.length; i++) {
      dels[i].addEventListener("click", async function () {
        if (!(await confirmBox("删除这条笔记？"))) return;
        try {
          await api("note/delete", null, { id: this.getAttribute("data-ndel") }, "POST");
          toast("已删除"); loadNotes();
        } catch (e) { toast(e.message, true); }
      });
    }
    var eds = tbody.querySelectorAll("button[data-nedit]");
    for (var j = 0; j < eds.length; j++) {
      eds[j].addEventListener("click", async function () {
        var id = this.getAttribute("data-nedit");
        var cur = byId[id] || {};
        var text = await promptBox("编辑笔记内容：", cur.content || "");
        if (text === null) return;
        text = text.trim();
        if (!text || text === cur.content) return;
        try {
          await api("note/update", null, { id: id, content: text }, "POST");
          toast("已保存"); loadNotes();
        } catch (e) { toast(e.message, true); }
      });
    }
  }

  function toggleNoteCreate(show) {
    var panel = $("noteCreatePanel");
    if (!panel) return;
    var willShow = show === undefined ? panel.style.display === "none" : show;
    panel.style.display = willShow ? "block" : "none";
    if (willShow) $("noteNewTitle").focus();
  }

  async function submitNote() {
    var content = $("noteNewContent").value.trim();
    if (!content) { toast("请输入内容", true); return; }
    try {
      await api("note/create", null, {
        content: content,
        title: $("noteNewTitle").value.trim(),
        tags: $("noteNewTags").value.trim(),
        scope: state.scope
      }, "POST");
      toast("已保存笔记");
      $("noteNewTitle").value = ""; $("noteNewContent").value = ""; $("noteNewTags").value = "";
      toggleNoteCreate(false);
      loadNotes();
    } catch (e) { toast(e.message, true); }
  }

  function importNoteFile() {
    $("noteFile").click();
  }

  function onNoteFileChosen() {
    var f = $("noteFile").files && $("noteFile").files[0];
    if (!f) return;
    var reader = new FileReader();
    reader.onload = async function () {
      try {
        var d = await api("note/import", null, {
          text: String(reader.result || ""),
          file_name: f.name,
          scope: state.scope
        }, "POST", IMPORT_TIMEOUT_MS);
        toast("已导入 " + (d.imported || 0) + " 块");
        loadNotes();
      } catch (e) { toast(e.message, true); }
      $("noteFile").value = "";
    };
    reader.onerror = function () { toast("读取文件失败", true); };
    reader.readAsText(f, "utf-8");
  }

  /* ------------------------------------------------------------ 账本 */
  async function loadLedger() {
    var box = $("ledRows");
    box.innerHTML = '<div class="empty">加载中…</div>';
    var q = $("ledSearch").value.trim();
    try {
      var d = q
        ? await api("ledger/search", { q: q, limit: 60 })
        : await api("ledger", { limit: 60 });
      var items = d.items || [];
      $("ledCount").textContent = items.length + " 条";
      if (!items.length) { box.innerHTML = '<div class="empty">暂无记录</div>'; return; }
      /* 按日分组：日期变化时插入分隔行，长账本不再是一条无尽的流水 */
      var html = "", lastDay = "";
      items.forEach(function (m) {
        var day = (m.ts_local || "").split(" ")[0] || "未知日期";
        if (day !== lastDay) {
          html += '<div class="led-day">' + esc(day) + "</div>";
          lastDay = day;
        }
        var who = m.role === "user" ? "用户" : "助手";
        html += '<div class="msg ' + esc(m.role) + '">' +
          '<span class="who">' + who + "</span>" +
          '<span class="txt">' + esc(m.content) + "</span>" +
          '<span class="when">' + esc((m.ts_local || "").split(" ").slice(1).join(" ") || m.ts_local || "") + "</span>" +
          "</div>";
      });
      box.innerHTML = html;
    } catch (e) {
      box.innerHTML = '<div class="empty">' + esc(e.message) + "</div>";
    }
  }

  /* ------------------------------------------------------------ 检索探针 */
  async function runProbe() {
    var box = $("probeRows");
    var q = $("probeQuery").value.trim();
    if (!q) { toast("请输入内容"); return; }
    box.innerHTML = '<div class="empty">检索中…</div>';
    try {
      var d = await api("recall", { q: q, scope: state.scope, limit: 10 }, null, null, PROBE_TIMEOUT_MS);
      var items = d.items || [];
      if (!items.length) { box.innerHTML = '<div class="empty">没有召回任何记忆</div>'; return; }
      box.innerHTML = items.map(function (c, i) {
        return '<div class="probe-item"><div>' + (i + 1) + ". " + esc(c.content) + "</div>" +
          '<div class="receipt">' + esc(c.receipt || "") + "</div></div>";
      }).join("");
    } catch (e) {
      box.innerHTML = '<div class="empty">' + esc(e.message) + "</div>";
    }
  }

  /* ------------------------------------------------------------ 回收站 */
  var TRASH_TYPES = {
    fact: "事实", knowledge: "知识", event: "事件",
    skill: "技能", emotional: "情感", task: "任务"
  };
  async function loadTrash() {
    var tbody = $("trashRows");
    tbody.innerHTML = '<tr><td colspan="6" class="empty">加载中…</td></tr>';
    try {
      var d = await api("trash");
      var items = d.items || [];
      /* v0.1.9：回收站同时列笔记（此前笔记软删后面板看不到也恢复不了） */
      var notes = d.notes || [];
      if (!items.length && !notes.length) { tbody.innerHTML = '<tr><td colspan="6" class="empty">回收站为空</td></tr>'; return; }
      var rows = items.map(function (m) {
        /* v0.2.11：隔离条目带「隔离待审」徽标——它的「恢复」就是人工审核
           （恢复即清 deleted_at+quarantined 转正），与普通删除的恢复区分开 */
        var typ = m.quarantined
          ? '<span class="tq-badge" title="写入门隔离的待审条目；点「恢复」即通过审核转正">隔离待审</span>'
          : (m.superseded_by
              ? '<span class="tq-badge" title="已被新说法取代；点「彻底恢复」才清除血缘并重新参与检索">已被取代</span>'
              : esc(TRASH_TYPES[m.memory_type] || m.memory_type || ""));
        return "<tr>" +
          '<td class="content-cell">' + esc(m.content) + "</td>" +
          "<td>" + typ + "</td>" +
          "<td>" + esc(m.strength == null ? "" : Number(m.strength).toFixed(1)) + "</td>" +
          "<td>" + esc(m.created_at_local || "") + "</td>" +
          "<td>" + esc(m.deleted_at_local || "") + "</td>" +
          "<td>" +
          /* v0.2.14：已被取代的旧说法普通「恢复」是无效操作（血缘保留、条目
             仍留在回收站）——不给这个按钮，只提供「彻底恢复」；普通软删/
             隔离条目才显示「恢复」。 */
          (m.superseded_by
            ? '<button class="btn mini" data-restore-full="' + esc(m.id) + '" title="清除取代血缘并重新进入检索">彻底恢复</button> '
            : '<button class="btn mini" data-restore="' + esc(m.id) + '">恢复</button> ') +
          '<button class="btn mini danger" data-purge="' + esc(m.id) + '">彻底删除</button></td>' +
          "</tr>";
      });
      rows = rows.concat(notes.map(function (n) {
        var label = n.title ? (n.title + "：" + n.content) : n.content;
        return "<tr>" +
          '<td class="content-cell">' + esc(label) + "</td>" +
          "<td>笔记</td>" +
          "<td>—</td>" +
          "<td>" + esc(n.created_at_local || "") + "</td>" +
          "<td>" + esc(n.deleted_at_local || "") + "</td>" +
          '<td><button class="btn mini" data-restore-note="' + esc(n.id) + '">恢复</button> ' +
          '<button class="btn mini danger" data-purge-note="' + esc(n.id) + '">彻底删除</button></td>' +
          "</tr>";
      }));
      tbody.innerHTML = rows.join("");
      var btns = tbody.querySelectorAll("button[data-restore]");
      for (var i = 0; i < btns.length; i++) {
        btns[i].addEventListener("click", async function () {
          try {
            await api("memory/restore", null, { id: this.getAttribute("data-restore") }, "POST");
            toast("已恢复");
            loadTrash();
          } catch (e) { toast(e.message, true); }
        });
      }
      var fullBtns = tbody.querySelectorAll("button[data-restore-full]");
      for (var f = 0; f < fullBtns.length; f++) {
        fullBtns[f].addEventListener("click", async function () {
          if (!(await confirmBox("这会清除取代血缘并让旧说法重新参与检索，确定？"))) return;
          try {
            await api("memory/restore", null, {
              id: this.getAttribute("data-restore-full"), clear_superseded: true
            }, "POST");
            toast("已彻底恢复并重新加入检索");
            loadTrash();
          } catch (e) { toast(e.message, true); }
        });
      }
      var nbtns = tbody.querySelectorAll("button[data-restore-note]");
      for (var j = 0; j < nbtns.length; j++) {
        nbtns[j].addEventListener("click", async function () {
          try {
            await api("note/restore", null, { id: this.getAttribute("data-restore-note") }, "POST");
            toast("笔记已恢复");
            loadTrash();
          } catch (e) { toast(e.message, true); }
        });
      }
      /* 彻底删除：硬删（连带向量/切片），不可恢复，二次确认文案必须说清 */
      var pbtns = tbody.querySelectorAll("button[data-purge]");
      for (var k = 0; k < pbtns.length; k++) {
        pbtns[k].addEventListener("click", async function () {
          if (!(await confirmBox("彻底删除后无法恢复（连带删除向量），确定？"))) return;
          try {
            await api("memory/purge", null, { id: this.getAttribute("data-purge") }, "POST");
            toast("已彻底删除");
            loadTrash();
          } catch (e) { toast(e.message, true); }
        });
      }
      var pnbtns = tbody.querySelectorAll("button[data-purge-note]");
      for (var m2 = 0; m2 < pnbtns.length; m2++) {
        pnbtns[m2].addEventListener("click", async function () {
          if (!(await confirmBox("彻底删除后无法恢复（连带删除切片），确定？"))) return;
          try {
            await api("note/purge", null, { id: this.getAttribute("data-purge-note") }, "POST");
            toast("笔记已彻底删除");
            loadTrash();
          } catch (e) { toast(e.message, true); }
        });
      }
    } catch (e) {
      tbody.innerHTML = '<tr><td colspan="6" class="empty">' + esc(e.message) + "</td></tr>";
    }
  }

  /* ------------------------------------------------------------ 维护 */
  async function loadBackups() {
    try {
      var d = await api("backups");
      var items = d.items || [];
      $("backupList").innerHTML = items.length
        ? items.map(function (b) {
            return '<div class="row"><span>' + esc(b.name) + "</span><span>" + esc(b.mtime) + "</span></div>";
          }).join("")
        : '<div class="empty">暂无备份</div>';
    } catch (e) {
      $("backupList").innerHTML = '<div class="empty">' + esc(e.message) + "</div>";
    }
  }

  async function runAction(endpoint, label) {
    var out = $("maintResult");
    out.textContent = label + " 执行中…";
    try {
      var d = await api(endpoint, null, {}, "POST");
      out.textContent = label + " 完成：\n" + JSON.stringify(d, null, 2);
      toast(label + "完成");
      loadBackups();
    } catch (e) {
      out.textContent = label + " 失败：" + e.message;
      toast(e.message, true);
    }
  }

  /* ------------------------------------------------------------ 初始化 */
  function bind() {
    var tabs = document.querySelectorAll(".tab");
    for (var i = 0; i < tabs.length; i++) {
      tabs[i].addEventListener("click", function () { switchView(this.getAttribute("data-view")); });
    }
    // 记忆库视图的事件绑定由 library.js 自管（加载时挂接）
    $("profLoad").addEventListener("click", loadProfiles);
    $("profUser").addEventListener("keydown", function (e) { if (e.key === "Enter") loadProfiles(); });
    $("profAddBtn").addEventListener("click", function () { profFormReset(); toggleProfPanel(); });
    $("profSave").addEventListener("click", submitProf);
    $("profCancel").addEventListener("click", function () { toggleProfPanel(false); profFormReset(); });
    $("noteSearchBtn").addEventListener("click", loadNotes);
    $("noteSearch").addEventListener("keydown", function (e) { if (e.key === "Enter") loadNotes(); });
    $("noteNewBtn").addEventListener("click", function () { toggleNoteCreate(); });
    $("noteNewSave").addEventListener("click", submitNote);
    $("noteNewCancel").addEventListener("click", function () { toggleNoteCreate(false); });
    $("noteImportBtn").addEventListener("click", importNoteFile);
    $("noteFile").addEventListener("change", onNoteFileChosen);
    $("ledSearchBtn").addEventListener("click", loadLedger);
    $("ledSearch").addEventListener("keydown", function (e) { if (e.key === "Enter") loadLedger(); });
    $("probeBtn").addEventListener("click", runProbe);
    $("probeQuery").addEventListener("keydown", function (e) { if (e.key === "Enter") runProbe(); });
    $("trashReload").addEventListener("click", loadTrash);
    $("runDecay").addEventListener("click", function () { runAction("decay/run", "衰减扫描"); });
    $("runBackup").addEventListener("click", function () { runAction("backup/run", "导出备份"); });
    $("reloadStats").addEventListener("click", loadOverview);
    $("scopeInput").addEventListener("change", function () {
      state.scope = this.value.trim() || "default";
      loadView(state.view);
    });
  }

  async function start() {
    bind();
    var bridge = window.AstrBotPluginPage;
    if (bridge && bridge.ready) {
      // 主题跟随不等 ready：context 任何时候到达都能应用
      if (bridge.onContext) {
        bridge.onContext(function (c) {
          document.documentElement.setAttribute("data-theme", c && c.isDark ? "dark" : "light");
        });
      }
      var ctx = bridge.getContext && bridge.getContext();
      if (ctx && ctx.isDark) document.documentElement.setAttribute("data-theme", "dark");
      try {
        await withTimeout(bridge.ready(), READY_TIMEOUT_MS, "bridge ready 超时");
      } catch (e) { /* 宿主未应答 context：不阻塞数据加载 */ }
    }
    loadView("overview");
  }

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", start);
  } else {
    start();
  }

  /* ------------------------------------------------------------ 自绘弹窗
     宿主把插件页放在沙箱 iframe 里，sandbox="allow-scripts allow-forms allow-downloads"
     **缺 allow-modals** —— 浏览器会静默屏蔽原生 confirm()/prompt()/alert()
     （confirm 直接返回 false，不弹任何窗），导致所有确认类操作"点了没反应"。
     因此自绘弹窗替代：confirmBox/promptBox 均返回 Promise。 */
  function _dialog(opts) {
    return new Promise(function (resolve) {
      var old = document.getElementById("mnDialogMask");
      if (old) old.parentNode.removeChild(old);
      var mask = document.createElement("div");
      mask.id = "mnDialogMask";
      mask.className = "mn-mask";
      var box = document.createElement("div");
      box.className = "mn-dialog";
      var msg = document.createElement("div");
      msg.className = "mn-dialog-msg";
      msg.textContent = opts.message;
      box.appendChild(msg);
      var input = null;
      if (opts.input) {
        input = document.createElement("textarea");
        input.className = "mn-dialog-input";
        input.value = opts.defaultValue || "";
        box.appendChild(input);
      }
      var row = document.createElement("div");
      row.className = "mn-dialog-actions";
      var cancel = document.createElement("button");
      cancel.className = "btn";
      cancel.textContent = "取消";
      var ok = document.createElement("button");
      ok.className = "btn primary";
      ok.textContent = opts.okText || "确定";
      row.appendChild(cancel);
      row.appendChild(ok);
      box.appendChild(row);
      mask.appendChild(box);
      document.body.appendChild(mask);

      var settled = false;
      function done(val) {
        if (settled) return;
        settled = true;
        document.removeEventListener("keydown", onKey);
        if (mask.parentNode) mask.parentNode.removeChild(mask);
        resolve(val);
      }
      function onKey(e) {
        if (e.key === "Escape") done(opts.input ? null : false);
        else if (e.key === "Enter" && opts.input && !e.shiftKey) done(input.value);
      }
      cancel.addEventListener("click", function () { done(opts.input ? null : false); });
      ok.addEventListener("click", function () { done(opts.input ? input.value : true); });
      mask.addEventListener("click", function (e) {
        if (e.target === mask) done(opts.input ? null : false);
      });
      document.addEventListener("keydown", onKey);
      if (input) { input.focus(); input.select(); } else { ok.focus(); }
    });
  }

  function confirmBox(message, okText) {
    return _dialog({ message: message, okText: okText });
  }

  function promptBox(message, defaultValue) {
    return _dialog({ message: message, input: true, defaultValue: defaultValue });
  }

  /* 分模块视图（星图/记忆库/设置）共享的公共接口 */
  window.MN = {
    api: api, toast: toast, esc: esc, $: $,
    state: state, switchView: switchView, loadOverview: loadOverview,
    confirm: confirmBox, prompt: promptBox,
  };
})();
