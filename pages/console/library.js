/* 好想记住你 · 记忆库（v6 重设计的生产版）。
 *
 * 存放全部记忆，支持：搜索（语义）/ 状态与类型过滤 / 排序 / 分页（每页 20 条），
 * 逐条操作：双击就地编辑正文、类型下拉改型、主动↔被动开关、删除（回收站+撤销）、
 * 顶部新增（走引擎写入门，重复即强化）。所有操作走后端既有端点。
 */
(function () {
  "use strict";

  var MN = window.MN;
  if (!MN) return;

  var TYPES = {
    fact:      { label: "事实", color: "#6fd3e8" },
    knowledge: { label: "知识", color: "#b39dff" },
    event:     { label: "事件", color: "#ffb86b" },
    skill:     { label: "技能", color: "#7ee2a8" },
    emotional: { label: "情感", color: "#ff9db8" },
    task:      { label: "任务", color: "#ffe08a" }
  };
  var PAGE_SIZE = 20;
  var state = { q: "", active: "all", typeOff: {}, sort: "time", page: 1 };
  var items = [];
  var lastScope = null, bound = false, loadedOnce = false;

  function $(id) { return document.getElementById(id); }
  function fmtDate(ts) {
    if (!ts) return "—";
    var d = new Date(ts * 1000);
    return (d.getMonth() + 1) + "月" + d.getDate() + "日";
  }

  function passes(m) {
    if (state.q && m.content.indexOf(state.q) < 0 && state._semanticMode !== true) return false;
    if (state.active === "active" && !m.is_active) return false;
    if (state.active === "passive" && m.is_active) return false;
    if (state.typeOff[m.memory_type]) return false;
    return true;
  }
  function sortFn(a, b) {
    if (state.sort === "strength") return (b.strength || 0) - (a.strength || 0);
    if (state.sort === "heat") return (b.hit_count || 0) - (a.hit_count || 0);
    return (b.t || 0) - (a.t || 0);
  }

  /* ---------------- 渲染 ---------------- */
  function render() {
    var rows = items.filter(passes).sort(sortFn);
    var total = rows.length;
    var pages = Math.max(1, Math.ceil(total / PAGE_SIZE));
    if (state.page > pages) state.page = pages;
    if (state.page < 1) state.page = 1;
    var start = (state.page - 1) * PAGE_SIZE;
    var pageRows = rows.slice(start, start + PAGE_SIZE);

    var list = $("libList");
    if (!total) {
      list.innerHTML = '<div class="lib-empty">没有匹配的记忆</div>';
    } else {
      list.innerHTML = pageRows.map(rowHtml).join("");
      bindRows();
    }
    renderPager(total, pages, start, pageRows.length);

    var act = items.filter(function (m) { return m.is_active; }).length;
    var qua = items.filter(function (m) { return m.quarantined; }).length;
    $("libStats").innerHTML =
      "本域共 <b>" + items.length + "</b> 条记忆<br/>主动 <b>" + act + "</b> · 被动 <b>" + (items.length - act) +
      "</b><br/>隔离待审 <b>" + qua + "</b> · 筛选后 <b>" + total + "</b>";
  }

  function rowHtml(m) {
    var T = TYPES[m.memory_type] || TYPES.fact;
    var opts = "";
    for (var k in TYPES) {
      opts += '<option value="' + k + '"' + (k === m.memory_type ? " selected" : "") + ">" + TYPES[k].label + "</option>";
    }
    return '<div class="lib-mem' + (m.is_active ? " active" : "") + (m.quarantined ? " quarantined" : "") + '" data-id="' + MN.esc(m.id) + '">'
      + '<div class="lib-body">'
      + '<div class="lib-content" data-edit>' + MN.esc(m.content) + "</div>"
      + '<div class="lib-meta">'
      + '<select class="lib-type" data-type>' + opts + "</select>"
      + '<span>记住于 ' + fmtDate(m.t) + "</span>"
      + "<span>召回 " + (m.hit_count || 0) + " 次</span>"
      + "<span>证据 " + (m.proof_count || 1) + " 条</span>"
      + "</div></div>"
      + '<div class="lib-side2">'
      + '<div class="lib-str"><b>' + Math.round(m.strength || 0) + "</b><span>强度</span>"
      + '<div class="lib-strbar"><i style="width:' + Math.min(100, m.strength || 0) + '%"></i></div></div>'
      + '<div class="lib-act"><span class="lib-sw' + (m.is_active ? " on" : "") + '" data-act></span>'
      + '<span class="lbl">' + (m.is_active ? "主动" : "被动") + "</span></div>"
      /* v0.2.11：隔离审核按钮已移除——库列表的 WHERE deleted_at IS NULL 天然
         排除出生即 deleted_at 的隔离条目，按钮永远不渲染（09-23 审查发现的
         死代码）；隔离条目的真实审核入口=回收站「恢复」（恢复即清
         deleted_at+quarantined 转正），回收站行内有「隔离待审」徽标提示。 */
      + '<button class="lib-del" data-del title="删除（移入回收站，30 天可恢复）">🗑</button>'
      + "</div></div>";
  }

  function byId(id) {
    for (var i = 0; i < items.length; i++) if (items[i].id === id) return items[i];
    return null;
  }

  function bindRows() {
    $("libList").querySelectorAll(".lib-mem").forEach(function (el) {
      var m = byId(el.dataset.id);
      if (!m) return;
      el.querySelector("[data-act]").addEventListener("click", async function () {
        var next = !m.is_active;
        var tip = next ? "设为主动记忆？它将永不衰减。" : "切回被动记忆？它将重新参与衰减。";
        if (!(await MN.confirm(tip))) return;
        try {
          await MN.api("memory/update", null, {
            id: m.id, scope: MN.state.scope, is_active: next
          }, "POST");
          m.is_active = next;
          MN.toast(next ? "已设为主动（永不衰减）" : "已设为被动");
          render();
        } catch (e) { MN.toast(e.message, true); }
      });
      el.querySelector("[data-type]").addEventListener("change", async function (e) {
        var t = e.target.value;
        try {
          await MN.api("memory/update", null, {
            id: m.id, scope: MN.state.scope, memory_type: t
          }, "POST");
          m.memory_type = t;
          MN.toast("类型已改为「" + (TYPES[t] || { label: t }).label + "」");
          render();
        } catch (err) { MN.toast(err.message, true); }
      });
      el.querySelector("[data-del]").addEventListener("click", async function () {
        if (!(await MN.confirm("确定删除这条记忆？会先移入回收站，30 天内可恢复。"))) return;
        var idx = items.indexOf(m);
        try {
          await MN.api("memory/delete", null, {
            id: m.id, scope: MN.state.scope
          }, "POST");
          items.splice(idx, 1);
          render();
          MN.toast("已移入回收站（30 天可恢复）", function () {
              MN.api("memory/restore", null, {
                id: m.id, scope: MN.state.scope
              }, "POST").then(function () {

              items.splice(Math.min(idx, items.length), 0, m);
              render(); MN.toast("已撤销恢复");
            }).catch(function (e2) { MN.toast(e2.message, true); });
          });
        } catch (e) { MN.toast(e.message, true); }
      });
      var c = el.querySelector("[data-edit]");
      c.addEventListener("dblclick", function () {
        c.setAttribute("contenteditable", "true");
        c.focus();
        /* document.execCommand 已弃用，改用 Selection/Range 全选 */
        var range = document.createRange();
        range.selectNodeContents(c);
        var sel = window.getSelection();
        sel.removeAllRanges();
        sel.addRange(range);
        var org = m.content;
        c.addEventListener("blur", async function () {
          c.removeAttribute("contenteditable");
          var nv = c.textContent.trim();
          if (!nv || nv === org) { render(); return; }
          try {
            await MN.api("memory/update", null, {
              id: m.id, scope: MN.state.scope, content: nv
            }, "POST");
            m.content = nv;
            MN.toast("记忆内容已更新（旧向量已清，检索稍后自动补嵌）");
          } catch (e) { MN.toast(e.message, true); }
          render();
        }, { once: true });
      });
    });
  }

  function renderPager(total, pages, start, shown) {
    var pg = $("libPager");
    if (!pg) return;
    if (total <= PAGE_SIZE) {
      pg.innerHTML = total ? '<span class="lib-pinfo">共 ' + total + " 条</span>" : "";
      return;
    }
    var cur = state.page, h = "";
    h += '<button ' + (cur <= 1 ? "disabled" : "") + ' data-go="' + (cur - 1) + '">‹ 上一页</button>';
    var nums = [];
    for (var i = 1; i <= pages; i++) {
      if (i === 1 || i === pages || Math.abs(i - cur) <= 1) nums.push(i);
      else if (nums[nums.length - 1] !== "…") nums.push("…");
    }
    nums.forEach(function (n) {
      if (n === "…") h += '<span class="lib-dots">…</span>';
      else h += '<button class="' + (n === cur ? "on" : "") + '" data-go="' + n + '">' + n + "</button>";
    });
    h += '<button ' + (cur >= pages ? "disabled" : "") + ' data-go="' + (cur + 1) + '">下一页 ›</button>';
    h += '<span class="lib-pinfo">第 ' + (start + 1) + "–" + (start + shown) + " 条 / 共 " + total + " 条 · 每页 " + PAGE_SIZE + "</span>";
    pg.innerHTML = h;
    pg.querySelectorAll("button[data-go]").forEach(function (b) {
      b.addEventListener("click", function () {
        state.page = +b.dataset.go; render();
        var view = document.getElementById("view-memories");
        if (view) view.scrollTop = 0;
      });
    });
  }

  /* ---------------- 数据加载 ---------------- */
  async function fetchItems() {
    var d = await MN.api("memories", { scope: MN.state.scope, q: state.q || "", limit: 2000 });
    items = d.items || [];
    state._semanticMode = !!state.q;
    render();
  }

  /* ---------------- 新增 ---------------- */
  var newActive = false;
  async function submitCreate() {
    var content = $("libNewContent").value.trim();
    if (!content) { $("libNewContent").focus(); return; }
    try {
      var d = await MN.api("memory/create", null, {
        content: content,
        memory_type: $("libNewType").value,
        is_active: newActive,
        scope: MN.state.scope
      }, "POST");
      if (d && d.created) MN.toast("已新增记忆");
      else if (d && d.reinforced) MN.toast("内容已存在，已强化既有记忆");
      else MN.toast((d && d.reason) || "未写入");
      $("libNewContent").value = "";
      newActive = false;
      var sw = $("libNewActive"); if (sw) sw.classList.remove("on");
      $("libAddPanel").classList.remove("show");
      state.page = 1; state.q = ""; $("libSearch").value = "";
      await fetchItems();
      MN.loadOverview();
    } catch (e) { MN.toast(e.message, true); }
  }

  /* ---------------- 绑定 ---------------- */
  function bind() {
    var tch = $("libTypeChips");
    var html = "";
    for (var k in TYPES) {
      html += '<span class="lib-tchip on" data-t="' + k + '"><span class="dot" style="background:' + TYPES[k].color + '"></span>' + TYPES[k].label + "</span>";
    }
    tch.innerHTML = html;
    tch.querySelectorAll(".lib-tchip").forEach(function (c) {
      c.addEventListener("click", function () {
        var t = c.dataset.t;
        state.typeOff[t] = !state.typeOff[t];
        c.classList.toggle("on", !state.typeOff[t]);
        c.classList.toggle("off", !!state.typeOff[t]);
        state.page = 1; render();
      });
    });
    $("libSearch").addEventListener("input", function () {
      state.q = this.value.trim();
      state.page = 1;
      clearTimeout(state._t);
      state._t = setTimeout(fetchItems, 350);   /* 搜索走语义通道，防抖 */
    });
    $("libSeg").querySelectorAll("button").forEach(function (b) {
      b.addEventListener("click", function () {
        $("libSeg").querySelectorAll("button").forEach(function (x) { x.classList.remove("on"); });
        b.classList.add("on");
        state.active = b.dataset.v;
        state.page = 1; render();
      });
    });
    $("libSort").addEventListener("change", function () { state.sort = this.value; state.page = 1; render(); });
    $("libAddToggle").addEventListener("click", function () {
      var p = $("libAddPanel");
      p.classList.toggle("show");
      if (p.classList.contains("show")) $("libNewContent").focus();
    });
    $("libAddCancel").addEventListener("click", function () {
      $("libAddPanel").classList.remove("show");
      $("libNewContent").value = "";
    });
    $("libNewActive").addEventListener("click", function () {
      newActive = !newActive; this.classList.toggle("on", newActive);
    });
    $("libAddSave").addEventListener("click", submitCreate);
    var sel = $("libNewType");
    for (var k2 in TYPES) sel.innerHTML += '<option value="' + k2 + '">' + TYPES[k2].label + "</option>";
  }

  window.MNLibrary = {
    load: async function () {
      if (!bound) { bind(); bound = true; }
      if (!loadedOnce || lastScope !== MN.state.scope) {
        loadedOnce = true; lastScope = MN.state.scope;
        $("libList").innerHTML = '<div class="lib-empty">加载中…</div>';
        try { await fetchItems(); }
        catch (e) { $("libList").innerHTML = '<div class="lib-empty">' + MN.esc(e.message) + "</div>"; }
      } else {
        render();
      }
    },
    /* 跨视图联动入口（星图详情卡「在记忆库中打开」）：
       以正文为查询走语义搜索强制刷新，调用前需先 switchView("memories") */
    openWith: function (q) {
      if (!bound) { bind(); bound = true; }
      state.q = (q || "").trim();
      state.page = 1;
      var s = $("libSearch"); if (s) s.value = state.q;
      loadedOnce = true; lastScope = MN.state.scope;
      $("libList").innerHTML = '<div class="lib-empty">检索中…</div>';
      fetchItems().catch(function (e) {
        $("libList").innerHTML = '<div class="lib-empty">' + MN.esc(e.message) + "</div>";
      });
    }
  };
})();
