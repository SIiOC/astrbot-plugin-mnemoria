/* 好想记住你 · 记忆星图（A++ 星座星图，v0.2.12 落地）
 *
 * 设计稿 v7_constellation.html 的生产移植：语义聚簇布局 + 星云/星座线/星座名 +
 * 语义缩放 LOD（大星=星座，点击飞入成员视图）+ sprite 预渲染 + 视口剔除 +
 * 真实时间轴（一颗星=一条记忆）+ Esc 分层退出 / 双击空白复位 + Pointer Events。
 *
 * 数据来自后端 /graph（节点=全部记忆，骨干边=向量相似度 top-3 带余弦 w，演化链=
 * superseded_by）；星座由前端对骨干边跑自适应阈值并查集（记忆越多阈值越放宽，
   星座总数收敛到 40 座以内，单簇上限 24）——v7 设计稿里
 * 「core/graph.py 同趟产出簇标签」的落点仍是后续优化，当前命名取代表记忆前缀。
 * 边计算在后台线程 + 磁盘缓存，未就绪时先渲染星点并轮询补线（补线后重聚簇）。
 * 探针走生产 /recall（三路 RRF，与模型看到的同源）。
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
  var GOLD = { r: 240, g: 198, b: 116 };
  var MEMC = {};
  (function () { for (var k in TYPES) MEMC[k] = hexToRgb(TYPES[k].color); })();

  var MEMS = [], EDGES = [], SUPS = [];
  var byId = {};
  var CLIST = [], CLUSTERS = {};
  var BACKBONE = {}, BRIDGES = {};
  var supNext = {}, supPrev = {};
  var loaded = false, edgesPolling = null;

  /* 星座聚簇参数：相似度阈值自适应——从 HI 起步逐档放宽（步长 0.02、下限 LO），
     直到 ≥2 成员的星座数 ≤ TARGET。记忆越多阈值越低、簇并得越大，
     否则 998 条会裂成 100+ 座把屏幕糊满（线上实测 0.7 固定阈值 → 102 座）。
     CLUSTER_MAX 防一条链滚成巨球。 */
  var CLUSTER_SIM_HI = 0.72, CLUSTER_SIM_LO = 0.56, CLUSTER_TARGET = 40, CLUSTER_MAX = 24;

  /* ---------------- 工具 ---------------- */
  function hashStr(s) {
    var h = 2166136261;
    for (var i = 0; i < s.length; i++) { h ^= s.charCodeAt(i); h = Math.imul(h, 16777619); }
    return h >>> 0;
  }
  function hexToRgb(h) {
    return { r: parseInt(h.slice(1, 3), 16), g: parseInt(h.slice(3, 5), 16), b: parseInt(h.slice(5, 7), 16) };
  }
  function rgba(c, a) { return "rgba(" + c.r + "," + c.g + "," + c.b + "," + a + ")"; }
  function mix(c1, c2, t) {
    return { r: c1.r + (c2.r - c1.r) * t | 0, g: c1.g + (c2.g - c1.g) * t | 0, b: c1.b + (c2.b - c1.b) * t | 0 };
  }
  function fmtDate(ts) {
    var d = new Date(ts * 1000);
    return (d.getMonth() + 1) + "月" + d.getDate() + "日";
  }

  /* ---------------- 布局：角向=星座扇区，径向=时间环（v7 同构） ---------------- */
  var RINGS = 5, CORE = 70, OUTER = 400, RZ = 190, ZBASE = 16, ZCORE = 95;
  var SIZE_K = 1, T0 = 0, T1 = 1;
  var MONTHS = [];

  function computeMonths() {
    MONTHS = [];
    var d = new Date(T0 * 1000), idx = 0;
    d = new Date(d.getFullYear(), d.getMonth() + 1, 1);
    while (d.getTime() / 1000 <= T1 && idx < 30) {
      MONTHS.push([(d.getMonth() + 1) + "月", (d.getTime() / 1000 - T0) / (T1 - T0), idx]);
      d = new Date(d.getFullYear(), d.getMonth() + 1, 1); idx++;
    }
  }

  /* 并查集聚簇：相近记忆收成星座；≥3 成员的簇参与 LOD 聚合。
     阈值自适应：高阈值起跑，星座太多就逐档放宽重跑（并查集 O(E·α)，
     E≈3n，重跑几趟代价可忽略），让小簇并成大星座而不是铺满屏幕。 */
  function computeClusters() {
    var sim = CLUSTER_SIM_HI, named = 0;
    for (;;) {
      buildClusters(sim);
      named = 0;
      for (var ck in CLUSTERS) if (CLUSTERS[ck].members.length >= 2) named++;
      if (named <= CLUSTER_TARGET || sim <= CLUSTER_SIM_LO) break;
      sim = Math.max(CLUSTER_SIM_LO, sim - 0.02);
    }
    CLIST = [];
    for (var ck2 in CLUSTERS) {
      var cl = CLUSTERS[ck2];
      cl.members.sort(function (a, b) { return a.t - b.t; });
      cl.meanT = cl.members.reduce(function (s, m) { return s + m.t; }, 0) / cl.members.length;
      cl.rep = cl.members.slice().sort(function (a, b) { return b.strength - a.strength; })[0];
      cl.big = cl.members.length >= 3;
      /* 命名过渡方案：取最强成员正文前 5 字；剥掉「名字（聊天ID）」式前缀——
         否则线上数据的名字全变成 "小明（z0xw…" 这种残句（簇标签的正式
         落点是 graph.py 同趟产出，见 设计说明.md 生产落地路径） */
      var nm = cl.rep.content.replace(/^用户/, "");
      nm = nm.replace(/^[^（）]{1,16}（[^）]*）/, "").replace(/^[的，。：:、\s]+/, "");
      cl.name = cl.members.length >= 2 ? (nm.slice(0, 5) || cl.rep.content.slice(0, 5)) : null;
      CLIST.push(cl);
    }
    CLIST.sort(function (a, b) { return a.meanT - b.meanT; });
  }
  function buildClusters(sim) {
    var uf = {}, ufSize = {};
    function find(x) { while (uf[x] !== x) { uf[x] = uf[uf[x]]; x = uf[x]; } return x; }
    MEMS.forEach(function (m) { uf[m.id] = m.id; ufSize[m.id] = 1; });
    EDGES.slice().sort(function (a, b) { return b.w - a.w; }).forEach(function (e) {
      if (e.w < sim) return;
      var ra = find(e.a.id), rb = find(e.b.id);
      if (ra === rb || ufSize[ra] + ufSize[rb] > CLUSTER_MAX) return;
      uf[rb] = ra; ufSize[ra] += ufSize[rb];
    });
    CLUSTERS = {};
    MEMS.forEach(function (m) {
      var c = find(m.id);
      m._cid = c;
      (CLUSTERS[c] = CLUSTERS[c] || { id: c, members: [] }).members.push(m);
    });
  }

  function layout() {
    var ts = MEMS.map(function (m) { return m.t; });
    T0 = Math.min.apply(null, ts); T1 = Math.max.apply(null, ts);
    if (T1 - T0 < 1) { T1 = T0 + 1; }
    computeMonths();
    /* 星盘半径与节点尺寸随总数自适应（密度向 LOD 让位） */
    OUTER = 400 * (1 + Math.min(0.6, Math.log(Math.max(MEMS.length, 34) / 34) / Math.LN10 * 0.28));
    SIZE_K = Math.max(0.45, Math.min(1, 140 / Math.max(MEMS.length, 1)));

    byId = {};
    MEMS.forEach(function (m) { byId[m.id] = m; });
    /* 边/演化链入的是 id 字符串，先建 byId 再映射成节点引用 */
    EDGES = EDGES.map(function (e) {
      return { a: byId[e.a], b: byId[e.b], w: e.w, key: e.a + "|" + e.b };
    }).filter(function (e) { return e.a && e.b; });
    SUPS = SUPS.map(function (s) {
      return { from: byId[s.from], to: byId[s.to] };
    }).filter(function (s) { return s.from && s.to; });

    computeClusters();

    /* 骨干边集合：每节点最强的 3 条（常态仅骨干线可见） */
    BACKBONE = {};
    MEMS.forEach(function (m) {
      EDGES.filter(function (e) { return e.a === m || e.b === m; })
        .sort(function (x, y) { return y.w - x.w; })
        .slice(0, 3).forEach(function (e) { BACKBONE[e.a.id + "|" + e.b.id] = 1; });
    });
    supNext = {}; supPrev = {};
    SUPS.forEach(function (s) { supNext[s.from.id] = s.to.id; supPrev[s.to.id] = s.from.id; });

    /* 星座视图的跨簇桥：每座大星座只保留最强的 2 条对外连线。
       线上数据边相似度普遍 0.9+（p50=0.909），固定 w 门槛拦不住——
       1594 条跨簇骨干边里 w≥0.95 的仍有 151 条，不按簇配额就是一张白网 */
    BRIDGES = {};
    var perCl = {};
    EDGES.forEach(function (e) {
      if (e.a._cid === e.b._cid) return;
      var ca = CLUSTERS[e.a._cid], cb = CLUSTERS[e.b._cid];
      if (!ca || !cb || !ca.big || !cb.big) return;
      (perCl[ca.id] = perCl[ca.id] || []).push(e);
      (perCl[cb.id] = perCl[cb.id] || []).push(e);
    });
    for (var bk in perCl) {
      perCl[bk].sort(function (x, y) { return y.w - x.w; }).slice(0, 2)
        .forEach(function (e) { BRIDGES[e.a.id + "|" + e.b.id] = 1; });
    }

    /* 星座扇区：按平均时间排序瓜分圆周，簇间留缝 */
    var GAP = Math.min(0.12, 1.8 / CLIST.length), PER_NODE = 0.30, MIN_SPAN = 0.22;
    var totalSpan = 0;
    CLIST.forEach(function (cl) {
      cl.span = Math.max(MIN_SPAN, Math.sqrt(cl.members.length) * PER_NODE);
      totalSpan += cl.span;
    });
    var scale = (Math.PI * 2 - GAP * CLIST.length) / totalSpan;
    var cursor = -Math.PI / 2;
    CLIST.forEach(function (cl) {
      var span = cl.span * scale, n = cl.members.length;
      cl.a0 = cursor; cl.a1 = cursor + span; cl.amid = cursor + span / 2;
      cl.members.forEach(function (m, i) {
        var j = ((hashStr(m.id + "aj") % 1000) / 1000 - 0.5);
        m._ang = cl.a0 + ((i + 0.5) / n + j * 0.35 / n) * span;
      });
      cursor = cl.a1 + GAP;
    });

    MEMS.forEach(function (m) {
      byId[m.id] = m;
      var u = (m.t - T0) / (T1 - T0);
      var ring = Math.min(RINGS - 1, Math.floor(u * RINGS));
      var inner = ring === 0 ? CORE * 0.55 : CORE + ring * (OUTER - CORE) / RINGS;
      var outer = CORE + (ring + 1) * (OUTER - CORE) / RINGS;
      var h = (hashStr(m.id) % 1000) / 1000;
      m._r = inner + (0.15 + 0.7 * h) * (outer - inner);
      m._ring = ring; m._u = u;
      m._c = MEMC[m.type] || MEMC.fact;
      m._sup = m.superseded_by || null;
      m._x = Math.cos(m._ang) * m._r;
      m._y = Math.sin(m._ang) * m._r;
      var g = ((hashStr(m.id + "z1") % 1000) / 1000 + (hashStr(m.id + "z2") % 1000) / 1000
        + (hashStr(m.id + "z3") % 1000) / 1000 - 1.5);
      m._z = g * (ZCORE * Math.exp(-m._r / RZ) + ZBASE);
      m._pulse = (hashStr(m.id) % 100) / 100;
    });

    /* 星座质心/色/半径（星云与代表星用） */
    CLIST.forEach(function (cl) {
      var sx = 0, sy = 0, sw = 0, cr = 0, cg = 0, cb = 0;
      cl.members.forEach(function (m) {
        var w = 1 + m.strength / 60;
        sx += m._x * w; sy += m._y * w; sw += w;
        cr += m._c.r * w; cg += m._c.g * w; cb += m._c.b * w;
      });
      cl.cx = sx / sw; cl.cy = sy / sw;
      cl.color = { r: cr / sw | 0, g: cg / sw | 0, b: cb / sw | 0 };
      cl.radius = 30;
      cl.members.forEach(function (m) {
        cl.radius = Math.max(cl.radius, Math.hypot(m._x - cl.cx, m._y - cl.cy) + 26);
      });
    });
  }
  /* ---------------- 相机与投影 ---------------- */
  var cv, ctx, DPR = Math.min(window.devicePixelRatio || 1, 2), W = 0, H = 0;
  var cam = {
    yaw: 0.6, pitch: 0.6, zoom: 1.0,
    vyaw: 0.0003, vpitch: 0,
    tyaw: null, tpitch: null, _tz: 1.06
  };
  var FOV = 850;
  function cy() { return (H - 70) / 2; }
  function project(x, y, z) {
    var cy_ = Math.cos(cam.yaw), sy_ = Math.sin(cam.yaw);
    var x1 = x * cy_ - y * sy_;
    var y1 = x * sy_ + y * cy_;
    var cp = Math.cos(cam.pitch), sp = Math.sin(cam.pitch);
    var y2 = y1 * cp - z * sp;
    var z2 = y1 * sp + z * cp;
    var s = FOV / (FOV - z2 * cam.zoom);
    return { x: W / 2 + x1 * cam.zoom * s, y: cy() + y2 * cam.zoom * s, s: s, depth: z2 };
  }
  function resize() {
    var sec = document.getElementById("smStage");
    if (!sec || !cv) return;
    var r = sec.getBoundingClientRect();
    W = Math.max(320, Math.floor(r.width));
    H = Math.max(240, Math.floor(r.height));
    cv.width = W * DPR; cv.height = H * DPR;
    cv.style.width = W + "px"; cv.style.height = H + "px";
    if (loaded && !userZoomed) fitView();
  }

  /* ---------------- 状态与 LOD ----------------
     memberAlpha：0=纯星座视图（代表亮星+星云+名字），1=纯成员视图。
     由 cam.zoom 相对开屏自适应缩放（fitZoom）的倍数平滑过渡。 */
  var reveal = 1, selected = null, hovered = null, hoveredCl = null;
  var probe = null, tAnim = 0, focusDim = 0;
  var fitZoom = 1, memberAlpha = 1, userZoomed = false;
  function lodUpdate() {
    var lk = cam.zoom / Math.max(fitZoom, 0.05);
    var t = (lk - 1.15) / 0.9;
    memberAlpha = t <= 0 ? 0 : t >= 1 ? 1 : t * t * (3 - 2 * t);
    if (memberAlpha >= 0.5) hoveredCl = null;
  }
  function visible(m) { return m._u <= reveal + 1e-6; }
  function nodeR(m) {
    var base = 2.2 + (m.strength || 10) / 100 * 4.2;
    var age = Math.max(0, Math.min(1, (1 - m._u) * 0.9));
    return base * (1 - age * 0.35) * (m.active ? 1.25 : 1) * SIZE_K;
  }
  var adjSet = {};
  function computeFocusSet() {
    adjSet = {};
    var f = hovered || selected;
    if (!f) return null;
    adjSet[f.id] = 1;
    EDGES.forEach(function (e) {
      if (e.a === f) adjSet[e.b.id] = 1;
      if (e.b === f) adjSet[e.a.id] = 1;
    });
    /* 演化链整链点亮（v7：悬停任一环，前后链全亮） */
    var cur = f;
    while (supPrev[cur.id]) { cur = byId[supPrev[cur.id]]; if (!cur) break; adjSet[cur.id] = 1; }
    cur = f;
    while (supNext[cur.id]) { cur = byId[supNext[cur.id]]; if (!cur) break; adjSet[cur.id] = 1; }
    return f;
  }

  /* ---------------- 绘制：星云 ---------------- */
  function drawNebulas(focusCid) {
    CLIST.forEach(function (cl) {
      /* 星座视图下小簇不铺星云（几十团雾叠在一起比星点更显乱），
         放大到成员视图后 2 成员的小簇仍有自己的雾 */
      if (cl.members.length < (memberAlpha > 0.5 ? 2 : 4)) return;
      if (!cl.members.some(visible)) return;
      var p = project(cl.cx, cl.cy, 0);
      var R = cl.radius * cam.zoom * p.s;
      if (R < 8) return;
      var boost = (focusCid && focusCid === cl.id) ? 2.0 : 1;
      var a = 0.085 * boost;
      var g = ctx.createRadialGradient(p.x, p.y, R * 0.08, p.x, p.y, R);
      g.addColorStop(0, rgba(mix(cl.color, { r: 255, g: 255, b: 255 }, 0.15), a));
      g.addColorStop(0.7, rgba(cl.color, a * 0.5));
      g.addColorStop(1, rgba(cl.color, 0));
      ctx.beginPath(); ctx.arc(p.x, p.y, R, 0, Math.PI * 2);
      ctx.fillStyle = g; ctx.fill();
    });
  }

  /* ---------------- 绘制：星座线（簇内按时间折线，成员视图） ---------------- */
  function drawConstellations(focusActive, focusCid) {
    if (memberAlpha < 0.02) return;
    ctx.save();
    CLIST.forEach(function (cl) {
      if (cl.members.length < 2) return;
      var dim = focusActive && focusCid !== cl.id;
      var started = false;
      ctx.beginPath();
      cl.members.forEach(function (m) {
        if (!visible(m)) return;
        var p = project(m._x, m._y, m._z);
        started ? ctx.lineTo(p.x, p.y) : ctx.moveTo(p.x, p.y);
        started = true;
      });
      ctx.strokeStyle = rgba(mix(cl.color, { r: 255, g: 255, b: 255 }, 0.45),
        (dim ? 0.04 : 0.12) * memberAlpha);
      ctx.lineWidth = 0.7;
      ctx.stroke();
    });
    ctx.restore();
  }

  /* ---------------- 绘制：环参考线 + 月份 + 盘外缘星座名 + 银心 ---------------- */
  function drawRingGuides(focusActive, focusCid) {
    ctx.save();
    for (var i = 0; i <= RINGS; i++) {
      var r = (i === 0 ? CORE * 0.55 : CORE + i * (OUTER - CORE) / RINGS);
      ctx.beginPath();
      for (var a = 0; a <= 64; a++) {
        var ang = a / 64 * Math.PI * 2;
        var p = project(Math.cos(ang) * r, Math.sin(ang) * r, 0);
        a ? ctx.lineTo(p.x, p.y) : ctx.moveTo(p.x, p.y);
      }
      ctx.strokeStyle = "rgba(140,170,220," + (0.07 - i * 0.008) + ")";
      ctx.setLineDash(i === RINGS ? [4, 6] : [2, 6]);
      ctx.lineWidth = 1;
      ctx.stroke();
    }
    ctx.setLineDash([]);
    /* 月份标签：半径=该月所在时间环外缘，角度错开防叠字 */
    ctx.font = "10px 'Segoe UI'";
    ctx.textAlign = "center";
    MONTHS.forEach(function (mo) {
      var r = CORE + Math.min(RINGS, Math.ceil(mo[1] * RINGS)) * (OUTER - CORE) / RINGS + 16;
      var ang = -Math.PI / 2 + mo[2] * 0.8;
      var p = project(Math.cos(ang) * r, Math.sin(ang) * r, 0);
      ctx.fillStyle = "rgba(124,138,165,0.55)";
      ctx.fillText(mo[0], p.x, p.y);
    });
    /* 盘外缘星座名（成员视图；星座视图的名字在代表星旁） */
    if (memberAlpha > 0.02) CLIST.forEach(function (cl) {
      if (!cl.name) return;
      var r = OUTER + 38;
      var p = project(Math.cos(cl.amid) * r, Math.sin(cl.amid) * r, 0);
      if (p.x < -40 || p.x > W + 40 || p.y < -20 || p.y > H + 20) return;
      var on = focusCid === cl.id;
      ctx.font = (on ? "600 " : "") + "12px 'Microsoft YaHei'";
      try { ctx.letterSpacing = "3px"; } catch (e) {}
      ctx.fillStyle = rgba(mix(cl.color, { r: 255, g: 255, b: 255 }, 0.35),
        (focusActive ? (on ? 0.9 : 0.13) : 0.5) * memberAlpha);
      ctx.fillText(cl.name, p.x, p.y);
      try { ctx.letterSpacing = "0px"; } catch (e) {}
    });
    ctx.textAlign = "start";
    /* 银心微光 */
    var c0 = project(0, 0, 0);
    var g = ctx.createRadialGradient(c0.x, c0.y, 2, c0.x, c0.y, 90 * cam.zoom);
    g.addColorStop(0, "rgba(240,198,116,0.10)");
    g.addColorStop(1, "rgba(240,198,116,0)");
    ctx.beginPath(); ctx.arc(c0.x, c0.y, 90 * cam.zoom, 0, Math.PI * 2);
    ctx.fillStyle = g; ctx.fill();
    ctx.restore();
  }

  /* ---------------- 绘制：边（常态弱骨干线，光带只属于焦点） ---------------- */
  function drawEdges(focusActive) {
    var i, e, A, B;
    var clusterMode = memberAlpha < 0.5;
    ctx.save();
    for (i = 0; i < EDGES.length; i++) {
      e = EDGES[i];
      if (!visible(e.a) || !visible(e.b)) continue;
      /* 星座视图：簇内边由星云/代表星概括；跨簇只画每座星座最强的 2 条桥
         （BRIDGES 白名单），否则上千条高相似骨干边织成一张白网 */
      var key = e.a.id + "|" + e.b.id;
      if (clusterMode) {
        if (e.a._cid === e.b._cid) { if (CLUSTERS[e.a._cid].big) continue; }
        else if (!BRIDGES[key]) continue;
      }
      var isFocus = focusActive && (adjSet[e.a.id] && adjSet[e.b.id]
        && (e.a === hovered || e.a === selected || e.b === hovered || e.b === selected));
      if (!isFocus && !BACKBONE[key]) continue;
      A = project(e.a._x, e.a._y, e.a._z);
      B = project(e.b._x, e.b._y, e.b._z);
      if (isFocus) {
        var col = mix(e.a._c, e.b._c, 0.5);
        var g = ctx.createLinearGradient(A.x, A.y, B.x, B.y);
        g.addColorStop(0, rgba(e.a._c, 0.75));
        g.addColorStop(1, rgba(e.b._c, 0.75));
        ctx.beginPath(); ctx.moveTo(A.x, A.y); ctx.lineTo(B.x, B.y);
        ctx.strokeStyle = rgba(mix(col, { r: 255, g: 255, b: 255 }, 0.55), 0.18);
        ctx.lineWidth = 5; ctx.stroke();
        ctx.beginPath(); ctx.moveTo(A.x, A.y); ctx.lineTo(B.x, B.y);
        ctx.strokeStyle = g; ctx.lineWidth = 1.6; ctx.stroke();
        var tt = (tAnim * 0.16 + e.a._pulse) % 1;
        var P = project(e.a._x + (e.b._x - e.a._x) * tt, e.a._y + (e.b._y - e.a._y) * tt,
          e.a._z + (e.b._z - e.a._z) * tt);
        ctx.beginPath(); ctx.arc(P.x, P.y, 1.8 * P.s * cam.zoom, 0, Math.PI * 2);
        ctx.fillStyle = rgba(mix(col, { r: 255, g: 255, b: 255 }, 0.6), 0.9); ctx.fill();
      } else {
        var avgD = (A.depth + B.depth) / 2;
        var fog = Math.max(0.35, Math.min(1, 0.55 + avgD / 220 * 0.45));
        ctx.beginPath(); ctx.moveTo(A.x, A.y); ctx.lineTo(B.x, B.y);
        ctx.strokeStyle = focusActive ? "rgba(150,175,215,0.04)"
          : "rgba(150,175,215," + (0.12 + (e.w || 0.5) * 0.07) * fog + ")";
        ctx.lineWidth = 0.5 + (e.w || 0.5) * 0.5; ctx.stroke();
      }
    }
    /* 演化链：金色（星座视图下两端任一位于大簇内则省略，由代表星概括） */
    for (i = 0; i < SUPS.length; i++) {
      var s = SUPS[i];
      if (!visible(s.from) || !visible(s.to)) continue;
      if (clusterMode) {
        var ca = CLUSTERS[s.from._cid], cb = CLUSTERS[s.to._cid];
        if ((ca && ca.big) || (cb && cb.big)) continue;
      }
      A = project(s.from._x, s.from._y, s.from._z);
      B = project(s.to._x, s.to._y, s.to._z);
      var dimS = focusActive && !(adjSet[s.from.id] && adjSet[s.to.id]);
      ctx.beginPath(); ctx.moveTo(A.x, A.y); ctx.lineTo(B.x, B.y);
      ctx.setLineDash([2, 4]);
      ctx.strokeStyle = rgba(GOLD, dimS ? 0.12 : 0.55);
      ctx.lineWidth = 1.3; ctx.stroke(); ctx.setLineDash([]);
      var tt2 = (tAnim * 0.3 + s.from._pulse) % 1;
      var Q = project(s.from._x + (s.to._x - s.from._x) * tt2,
        s.from._y + (s.to._y - s.from._y) * tt2, s.from._z + (s.to._z - s.from._z) * tt2);
      ctx.beginPath(); ctx.arc(Q.x, Q.y, 2 * Q.s, 0, Math.PI * 2);
      ctx.fillStyle = rgba(GOLD, dimS ? 0.3 : 0.95); ctx.fill();
    }
    ctx.restore();
  }

  /* ---------------- 绘制：节点 sprite 预渲染（类型色×半径档一次画成） ---------------- */
  var SPRITES = {}, SSK = 6, GLOW_K = 2.9;
  function spriteFor(m) {
    var rb = Math.round(nodeR(m) * 2) / 2;
    var key = m.type + "|" + rb;
    var sp = SPRITES[key];
    if (sp) return sp;
    var size = Math.ceil(rb * GLOW_K * 2 * SSK);
    var c = document.createElement("canvas"); c.width = c.height = size;
    var g2 = c.getContext("2d");
    var ccx = size / 2, rr = rb * SSK;
    var gg = g2.createRadialGradient(ccx, ccx, rr * 0.5, ccx, ccx, rr * GLOW_K);
    gg.addColorStop(0, rgba(m._c, 0.17));
    gg.addColorStop(1, rgba(m._c, 0));
    g2.beginPath(); g2.arc(ccx, ccx, rr * GLOW_K, 0, Math.PI * 2); g2.fillStyle = gg; g2.fill();
    var bg = g2.createRadialGradient(ccx - rr * 0.35, ccx - rr * 0.35, rr * 0.1, ccx, ccx, rr);
    bg.addColorStop(0, rgba(mix(m._c, { r: 255, g: 255, b: 255 }, 0.55), 1));
    bg.addColorStop(1, rgba(m._c, 0.85));
    g2.beginPath(); g2.arc(ccx, ccx, rr, 0, Math.PI * 2); g2.fillStyle = bg; g2.fill();
    g2.beginPath(); g2.arc(ccx - rr * 0.3, ccx - rr * 0.3, rr * 0.28, 0, Math.PI * 2);
    g2.fillStyle = "rgba(255,255,255,0.85)"; g2.fill();
    sp = { cv: c, rb: rb };
    SPRITES[key] = sp;
    return sp;
  }

  /* ---------------- 绘制：星座代表星（星座视图的"亮星"） ---------------- */
  function drawClusterLayer(dmin, dmax) {
    if (memberAlpha > 0.995) return;
    var clA = 1 - memberAlpha;
    ctx.save();
    /* 标注总预算：按簇规模排序，缩略态只给最大的 12 座星座配名字，
       其余星座要等放大过渡（或悬停）才显名——星座一多，全开标签必然糊屏 */
    var labelAllow = {};
    CLIST.filter(function (c) { return c.big && c.name; })
      .sort(function (a, b) { return b.members.length - a.members.length; })
      .forEach(function (c, i) { labelAllow[c.id] = i < 12 ? 0.35 : 0.8; });
    /* 甜甜圈式标签车道：只取规模最大的 12 座（悬停星座例外），同名片前缀
       去重（线上大量同前缀记忆会裂出一排同名星座，只给最大那座配名），
       然后按屏幕方位角排序，同车道角距不足一个标签宽就逐层外移（最多 3 条
       车道），从根上消灭中央密集区标签互相叠字（用上一帧的 _sx/_sy，滞后一帧无感） */
    var spots = [];
    CLIST.forEach(function (cl) {
      cl._lane = -1;
      if (!cl.big || !cl.name || cl._sx == null) return;
      if ((labelAllow[cl.id] === 0.35 && clA > 0.35) || hoveredCl === cl) spots.push(cl);
    });
    spots.sort(function (a, b) { return b.members.length - a.members.length; });
    var seenName = {};
    spots = spots.filter(function (cl) {
      if (hoveredCl === cl) return true;
      if (seenName[cl.name]) return false;
      seenName[cl.name] = 1;
      return true;
    });
    spots.sort(function (a, b) {
      return Math.atan2(a._sy - cy(), a._sx - W / 2) - Math.atan2(b._sy - cy(), b._sx - W / 2);
    });
    var lanes = [];
    spots.forEach(function (cl) {
      var ang = Math.atan2(cl._sy - cy(), cl._sx - W / 2);
      for (var lane = 0; lane < 3; lane++) {
        var need = 84 / (150 + lane * 120);
        if (lanes[lane] === undefined || ang - lanes[lane] > need) {
          lanes[lane] = ang; cl._lane = lane; break;
        }
      }
    });
    CLIST.forEach(function (cl) {
      if (!cl.big) return;
      var rep = cl.rep;
      if (!visible(rep)) return;
      var p = project(rep._x, rep._y, rep._z);
      if (p.x < -120 || p.x > W + 120 || p.y < -120 || p.y > H + 120) { cl._sx = null; return; }
      var sp = spriteFor(rep);
      var scaleUp = 1.3 + Math.log(cl.members.length) * 0.28;
      var r = sp.rb * scaleUp * p.s * cam.zoom;
      cl._sx = p.x; cl._sy = p.y; cl._sr = Math.max(r * 2.2, 13);
      var norm = (dmax - dmin) > 1 ? (p.depth - dmin) / (dmax - dmin) : 0.5;
      var alpha = clA * (0.55 + 0.45 * norm);
      var probeHit = probe && cl.members.some(function (m) { return probe.set[m.id]; });
      if (probe && !probeHit) alpha *= 0.35;
      /* 星座底色辉光：让代表星在远处也读得出"亮星" */
      var bg2 = ctx.createRadialGradient(p.x, p.y, r * 0.3, p.x, p.y, r * 4.2);
      bg2.addColorStop(0, rgba(mix(cl.color, { r: 255, g: 255, b: 255 }, 0.25), 0.30 * alpha));
      bg2.addColorStop(1, rgba(cl.color, 0));
      ctx.beginPath(); ctx.arc(p.x, p.y, r * 4.2, 0, Math.PI * 2); ctx.fillStyle = bg2; ctx.fill();
      /* 悬停星座：额外亮环 */
      if (hoveredCl === cl) {
        ctx.beginPath(); ctx.arc(p.x, p.y, r * 1.9, 0, Math.PI * 2);
        ctx.strokeStyle = rgba(mix(cl.color, { r: 255, g: 255, b: 255 }, 0.5), 0.55 * clA);
        ctx.lineWidth = 1.2; ctx.stroke();
      }
      var ds = sp.rb * scaleUp * GLOW_K * 2 * p.s * cam.zoom;
      ctx.globalAlpha = Math.min(1, alpha);
      ctx.drawImage(sp.cv, p.x - ds / 2, p.y - ds / 2, ds, ds);
      ctx.globalAlpha = 1;
      /* 簇内有探针命中：金环提示"答案在这座星座里" */
      if (probeHit) {
        ctx.beginPath(); ctx.arc(p.x, p.y, r * 2.4, 0, Math.PI * 2);
        ctx.strokeStyle = "rgba(240,198,116,0.85)"; ctx.lineWidth = 1.2;
        ctx.setLineDash([3, 3]); ctx.stroke(); ctx.setLineDash([]);
      }
      /* 名字 + 条数：沿"远离盘心"方向摆放，车道由上面的甜甜圈布局分配，
         车道每外移一层再退 20px，同方位角的标签不再叠字 */
      if (cl.name && cl._lane >= 0) {
        var lx = p.x - W / 2, ly = p.y - cy();
        var ll = Math.hypot(lx, ly) || 1;
        var off = r + 14 + cl._lane * 20;
        var tx = p.x + lx / ll * off, ty = p.y + ly / ll * off;
        ctx.textAlign = "center";
        ctx.font = "600 12px 'Microsoft YaHei'";
        ctx.fillStyle = rgba(mix(cl.color, { r: 255, g: 255, b: 255 }, 0.4), Math.min(1, alpha * 1.3));
        ctx.fillText(cl.name, tx, ty - 4);
        ctx.font = "10px 'Segoe UI'";
        ctx.fillStyle = rgba({ r: 124, g: 138, b: 165 }, Math.min(1, alpha * 1.1));
        ctx.fillText(cl.members.length + " 条", tx, ty + 9);
      }
    });
    ctx.textAlign = "start";
    ctx.restore();
  }

  /* ---------------- 绘制：成员节点 ---------------- */
  function drawNode(m, focusActive, dmin, dmax) {
    if (!visible(m)) return;
    var cl = CLUSTERS[m._cid];
    var aggregated = cl && cl.big;
    if (aggregated && memberAlpha < 0.02) { m._sx = null; return; }
    var p = m._proj;
    if (p.x < -80 || p.x > W + 80 || p.y < -80 || p.y > H + 80) { m._sx = null; return; }
    var sp = spriteFor(m);
    var r = sp.rb * p.s * cam.zoom;
    m._sx = p.x; m._sy = p.y; m._sr = r; m._depth = p.depth;
    var isHi = (m === hovered || m === selected);
    var inFocus = focusActive && adjSet[m.id];
    var dimmed = focusActive && !inFocus;
    var norm = (dmax - dmin) > 1 ? (p.depth - dmin) / (dmax - dmin) : 0.5;
    var alpha = (dimmed ? 0.28 : 1) * (0.55 + 0.45 * norm);
    if (aggregated) alpha *= memberAlpha;
    /* 星座视图里孤星/小簇成员压暗到背景层：星座结构才能从星海里浮出来，
       随 memberAlpha 升高平滑恢复全亮 */
    if (!aggregated && memberAlpha < 0.5) alpha *= 0.22 + 1.56 * memberAlpha;
    if (probe && !probe.set[m.id] && !isHi) alpha *= 0.35;

    /* 回放诞生闪光 */
    var born = reveal - m._u;
    if (born >= 0 && born < 0.035) {
      var bw = (born / 0.035);
      ctx.beginPath(); ctx.arc(p.x, p.y, r * (1 + (1 - bw) * 7), 0, Math.PI * 2);
      ctx.strokeStyle = rgba(m._c, (1 - bw) * 0.7);
      ctx.lineWidth = 1.5; ctx.stroke();
    }
    /* 主动恒星脉动光环 */
    if (m.active && !dimmed) {
      var pulse = 1 + Math.sin(tAnim * 2 + m._pulse * 10) * 0.15;
      ctx.beginPath(); ctx.arc(p.x, p.y, r * 2.4 * pulse, 0, Math.PI * 2);
      ctx.strokeStyle = rgba(m._c, 0.3 * alpha); ctx.lineWidth = 1; ctx.stroke();
    }
    /* 隔离待审：虚线圈 */
    if (m.quarantined) {
      ctx.beginPath(); ctx.arc(p.x, p.y, r * 2.2, 0, Math.PI * 2);
      ctx.setLineDash([2, 3]); ctx.strokeStyle = "rgba(200,120,120," + 0.4 * alpha + ")";
      ctx.stroke(); ctx.setLineDash([]);
    }
    /* 珠子本体 + 辉光：一次 drawImage */
    var ds = sp.rb * GLOW_K * 2 * p.s * cam.zoom;
    ctx.globalAlpha = Math.min(1, alpha);
    ctx.drawImage(sp.cv, p.x - ds / 2, p.y - ds / 2, ds, ds);
    ctx.globalAlpha = 1;
    /* 悬停显影辉光 */
    if (isHi) {
      var glowR = r * 4.8;
      var hg = ctx.createRadialGradient(p.x, p.y, r * 0.5, p.x, p.y, glowR);
      hg.addColorStop(0, rgba(m._c, 0.30 * alpha));
      hg.addColorStop(1, rgba(m._c, 0));
      ctx.beginPath(); ctx.arc(p.x, p.y, glowR, 0, Math.PI * 2); ctx.fillStyle = hg; ctx.fill();
    }
    /* 探针命中光环 + 排名徽标 */
    if (probe && probe.set[m.id]) {
      var col = probe.colors[m.id] || "#f0c674";
      ctx.beginPath(); ctx.arc(p.x, p.y, r * 3.2, 0, Math.PI * 2);
      ctx.strokeStyle = col; ctx.lineWidth = 1.2; ctx.setLineDash([3, 3]); ctx.stroke(); ctx.setLineDash([]);
      var rank = probe.order.indexOf(m.id) + 1;
      if (rank > 0) {
        ctx.beginPath(); ctx.arc(p.x + r * 2.6, p.y - r * 2.6, 8, 0, Math.PI * 2);
        ctx.fillStyle = col; ctx.fill();
        ctx.fillStyle = "#06080f"; ctx.font = "bold 9px sans-serif";
        ctx.textAlign = "center"; ctx.textBaseline = "middle";
        ctx.fillText(rank, p.x + r * 2.6, p.y - r * 2.6 + 0.5);
        ctx.textAlign = "start"; ctx.textBaseline = "alphabetic";
      }
    }
    /* 选中环 */
    if (m === selected) {
      ctx.beginPath(); ctx.arc(p.x, p.y, r * 2.0, 0, Math.PI * 2);
      ctx.strokeStyle = "rgba(255,255,255,0.7)"; ctx.lineWidth = 1; ctx.stroke();
    }
    /* 悬停正文短标签 */
    if (isHi && m !== selected) {
      ctx.font = "11px 'Microsoft YaHei'";
      ctx.fillStyle = "rgba(216,226,242,0.92)";
      var label = m.content.length > 14 ? m.content.slice(0, 14) + "…" : m.content;
      ctx.fillText(label, p.x + r * 2.6, p.y + 4);
    }
  }

  /* ---------------- 主循环 ---------------- */
  function frame() {
    /* 可见性守卫：切换到其他视图时 canvas 只是被 CSS 隐藏（仍在 DOM），
       若无此检查，大量节点×每帧渐变的绘制会在后台永久空转烧 CPU
       （rAF 空转本身极轻，跳过重绘即可）。 */
    if (!ctx || !cv.isConnected || cv.offsetParent === null) {
      requestAnimationFrame(frame);
      return;
    }
    tAnim += 0.016;
    if (cam.tyaw != null) {
      cam.yaw += (cam.tyaw - cam.yaw) * 0.06;
      if (Math.abs(cam.tyaw - cam.yaw) < 0.01) { cam.tyaw = null; cam.vyaw = 0; }
    } else {
      if (selected) { cam.vyaw *= 0.8; }
      else { cam.yaw += cam.vyaw; cam.vyaw += (0.0003 - cam.vyaw) * 0.008; }
    }
    if (cam.tpitch != null) {
      cam.pitch += (cam.tpitch - cam.pitch) * 0.08;
      if (Math.abs(cam.tpitch - cam.pitch) < 0.01) cam.tpitch = null;
    }
    cam.pitch += cam.vpitch; cam.vpitch *= 0.94;
    if (cam.pitch > 1.52) { cam.pitch = 1.52; cam.vpitch = 0; }
    if (cam.pitch < -1.52) { cam.pitch = -1.52; cam.vpitch = 0; }
    if (cam._tz != null) { cam.zoom += (cam._tz - cam.zoom) * 0.12; }
    lodUpdate();

    ctx.clearRect(0, 0, cv.width, cv.height);
    ctx.save(); ctx.scale(DPR, DPR);
    /* 星尘背景 */
    ctx.fillStyle = "rgba(180,200,235,0.18)";
    for (var i = 0; i < 70; i++) {
      var hx = (hashStr("d" + i) % 1000) / 1000 * W, hy = (hashStr("d" + i + "b") % 1000) / 1000 * H;
      var tw = 0.5 + 0.5 * Math.sin(tAnim * 0.7 + i);
      ctx.globalAlpha = 0.05 + 0.10 * tw;
      ctx.fillRect(hx, hy, 1, 1);
    }
    ctx.globalAlpha = 1;
    var focusNode = computeFocusSet();
    var focusCid = focusNode ? focusNode._cid : (hoveredCl ? hoveredCl.id : null);
    focusDim += ((focusNode ? 1 : 0) - focusDim) * 0.15;
    var focusActive = focusDim > 0.5;
    /* 统一投影 → 深度归一 → 分层绘制（星云在下，节点在上） */
    var vis = MEMS.filter(visible);
    var dmin = 1e9, dmax = -1e9;
    vis.forEach(function (m) {
      m._proj = project(m._x, m._y, m._z);
      if (m._proj.depth < dmin) dmin = m._proj.depth;
      if (m._proj.depth > dmax) dmax = m._proj.depth;
    });
    drawNebulas(focusCid);
    drawRingGuides(focusActive, focusCid);
    drawConstellations(focusActive, focusCid);
    drawEdges(focusActive);
    drawClusterLayer(dmin, dmax);
    vis.sort(function (a, b) { return a._proj.depth - b._proj.depth; })
      .forEach(function (m) { drawNode(m, focusActive, dmin, dmax); });
    ctx.restore();
    requestAnimationFrame(frame);
  }

  /* ---------------- 开屏自适应 ---------------- */
  function fitView() {
    var padX = Math.min(80, W * 0.08), padTop = Math.min(140, H * 0.18), padBot = Math.min(120, H * 0.16);
    var availX = W / 2 - padX;
    var availY = Math.min(cy() - padTop, (H - cy()) - padBot);
    if (availX < 120) availX = 120;
    if (availY < 120) availY = 120;
    var z = cam.zoom || 1;
    for (var it = 0; it < 7; it++) {
      cam.zoom = z;
      var mx = 0, my = 0;
      MEMS.forEach(function (m) {
        var p = project(m._x, m._y, m._z);
        var dx = Math.abs(p.x - W / 2) + nodeR(m) * p.s * cam.zoom;
        var dy = Math.abs(p.y - cy()) + nodeR(m) * p.s * cam.zoom;
        if (dx > mx) mx = dx; if (dy > my) my = dy;
      });
      if (mx < 1 || my < 1) break;
      z *= Math.min(availX / mx, availY / my);
    }
    cam.zoom = Math.max(0.2, Math.min(1.5, z));
    cam._tz = cam.zoom;
    fitZoom = cam.zoom;
  }
  function resetView() {
    userZoomed = false; fitView();
    cam.tyaw = 0.6; cam.tpitch = 0.6;
  }

  /* ---------------- 数据加载 ---------------- */
  async function fetchData() {
    var d = await MN.api("graph", { scope: MN.state.scope });
    MEMS = d.nodes || [];
    var hadEdges = EDGES.length;
    if (d.computing) {
      EDGES = []; SUPS = [];
    } else {
      EDGES = d.edges || []; SUPS = d.superseded || [];
    }
    layout();
    fitView();
    loaded = true;
    drawTimelineBar();
    renderLegend();
    updateStatus();
    if (d.computing) {
      setStatus("星图连线计算中（每颗星取最强关联）…数秒后自动出现");
      if (!edgesPolling) {
        edgesPolling = setInterval(async function () {
          try {
            var d2 = await MN.api("graph", { scope: MN.state.scope });
            if (!d2.computing) {
              EDGES = d2.edges || []; SUPS = d2.superseded || [];
              layout();
              clearInterval(edgesPolling); edgesPolling = null;
              setStatus("");
            }
          } catch (e) { /* 轮询失败下次再试 */ }
        }, 4000);
      }
    } else if (!hadEdges && EDGES.length) {
      setStatus("");
    }
  }

  function setStatus(msg) {
    var el = MN.$("smStatus");
    if (el) el.textContent = msg;
  }
  function updateStatus() {
    var act = MEMS.filter(function (m) { return m.active; }).length;
    var cons = CLIST.filter(function (c) { return c.members.length >= 2; }).length;
    var el = MN.$("smStats");
    if (el) el.innerHTML =
      "<div class='s'><b>" + MEMS.length + "</b><span>记忆</span></div>" +
      "<div class='s'><b>" + cons + "</b><span>星座</span></div>" +
      "<div class='s'><b class='g'>" + act + "</b><span>主动恒星</span></div>" +
      "<div class='s'><b>" + SUPS.length + "</b><span>演化链</span></div>";
  }

  /* ---------------- 图例 / 时间轴 / 详情 ---------------- */
  function renderLegend() {
    var el = MN.$("smLegend");
    if (!el) return;
    var html = '<div class="t">星等 · 类型</div>';
    for (var k in TYPES) {
      html += '<div class="li"><span class="dot" style="background:' + TYPES[k].color + '"></span>' + TYPES[k].label + "</div>";
    }
    html += '<div class="li dim"><span class="dot" style="background:#f0c674"></span>记忆演化（被更新）</div>';
    html += '<div class="li dim"><span class="dot" style="background:rgba(200,120,120,.5)"></span>隔离待审</div>';
    html += '<div class="t" style="margin-top:8px">大星 = 星座（点击放大）<br/>悬停星星显关联</div>';
    el.innerHTML = html;
  }

  /* 时间轴：一颗星 = 一条真实记忆（颜色=类型，y 抖动防重叠） */
  var tlCv, tlCtx;
  function drawTimelineBar() {
    if (!tlCv) return;
    var w = tlCv.clientWidth, h = tlCv.clientHeight;
    if (!w || !h) return;
    tlCv.width = w * DPR; tlCv.height = h * DPR;
    tlCtx.setTransform(DPR, 0, 0, DPR, 0, 0);
    tlCtx.clearRect(0, 0, w, h);
    MEMS.forEach(function (m) {
      var x = m._u * w;
      var y = 7 + (hashStr(m.id + "ty") % 100) / 100 * (h - 14);
      var lit = m._u <= reveal;
      tlCtx.globalAlpha = lit ? 0.7 : 0.12;
      tlCtx.fillStyle = TYPES[m.type] ? TYPES[m.type].color : TYPES.fact.color;
      tlCtx.beginPath();
      tlCtx.arc(x, y, 0.9 + (m.strength || 10) / 100 * 1.2, 0, Math.PI * 2);
      tlCtx.fill();
    });
    tlCtx.globalAlpha = 1;
  }
  function setReveal(v) {
    reveal = Math.max(0, Math.min(1, v));
    var range = MN.$("smReveal");
    if (range) {
      range.value = Math.round(reveal * 1000);
      range.style.setProperty("--fill", (reveal * 100) + "%");
    }
    var knob = MN.$("smCursor");
    if (knob) knob.style.left = (reveal * 100) + "%";
    var d = new Date((T0 + reveal * (T1 - T0)) * 1000);
    var dl = MN.$("smDate");
    if (dl) { dl.style.left = (reveal * 100) + "%"; dl.textContent = (d.getMonth() + 1) + "月" + d.getDate() + "日"; }
    drawTimelineBar();
  }

  function renderDetail() {
    var d = MN.$("smDetail");
    if (!d) return;
    if (!selected) { d.className = "sm-detail"; return; }
    var m = selected;
    var T = TYPES[m.type] || TYPES.fact;
    var rels = EDGES.filter(function (e) { return e.a === m || e.b === m; })
      .sort(function (a, b) { return b.w - a.w; }).slice(0, 5);
    var chainPrev = MEMS.filter(function (x) { return x.superseded_by === m.id; })[0];
    var cl = CLUSTERS[m._cid];
    var html = '<span class="sm-close" data-smclose>✕</span>'
      + '<span class="sm-chip" style="color:' + T.color + ';border-color:' + T.color + '55">'
      + '<span class="dot" style="width:7px;height:7px;border-radius:50%;background:' + T.color + ';display:inline-block"></span>'
      + T.label + (m.active ? " · 主动记忆" : "") + "</span>"
      + (cl && cl.name ? '<div style="font-size:11px;color:#7c8aa5;margin:-2px 0 8px;letter-spacing:.1em">✦ '
        + MN.esc(cl.name) + ' 星座</div>' : "")
      + '<div class="sm-content">' + MN.esc(m.content) + "</div>"
      + '<div class="sm-meta"><div>强度 <b>' + Math.round(m.strength || 0) + "</b></div>"
      + "<div>被召回 <b>" + (m.heat || 0) + " 次</b></div>"
      + "<div>记住于 <b>" + fmtDate(m.t) + "</b></div>"
      + "<div>状态 <b>" + (m.quarantined ? "隔离待审" : "生效中") + "</b></div></div>"
      + '<button class="sm-openlib" data-openlib>在记忆库中打开 ›</button>';
    if (m._sup && byId[m._sup]) {
      html += '<h4>记忆演化</h4><div class="sm-chain">已更新为：<br/>' + MN.esc(byId[m._sup].content) + "</div>";
    } else if (chainPrev) {
      html += '<h4>记忆演化</h4><div class="sm-chain">由「' + MN.esc(chainPrev.content) + "」演化而来</div>";
    }
    if (rels.length) {
      html += "<h4>相关记忆（语义骨干）</h4>";
      rels.forEach(function (e) {
        var o = e.a === m ? e.b : e.a;
        html += '<div class="sm-rel" data-smjump="' + o.id + '"><span class="dot" style="width:6px;height:6px;border-radius:50%;background:'
          + (TYPES[o.type] || TYPES.fact).color + '"></span>' + MN.esc(o.content) + "</div>";
      });
    }
    d.innerHTML = html;
    d.className = "sm-detail show";
    var close = d.querySelector("[data-smclose]");
    if (close) close.addEventListener("click", function () { selected = null; renderDetail(); });
    var openlib = d.querySelector("[data-openlib]");
    if (openlib) openlib.addEventListener("click", function () {
      /* 跳到记忆库并以正文走语义搜索定位本条（openWith 会强制刷新列表） */
      MN.switchView("memories");
      if (window.MNLibrary && window.MNLibrary.openWith) window.MNLibrary.openWith(m.content);
    });
    d.querySelectorAll("[data-smjump]").forEach(function (r) {
      r.addEventListener("click", function () {
        focusNode(r.getAttribute("data-smjump"));
      });
    });
  }

  /* 选中某节点：它若在大簇内且当前是星座视图，先放大到成员视图再转过去 */
  function focusNode(id) {
    var target = byId[id];
    if (!target) return;
    selected = target; renderDetail();
    var cl = CLUSTERS[target._cid];
    if (cl && cl.big && memberAlpha < 0.5) {
      userZoomed = true;
      cam._tz = Math.max(fitZoom * 2.4, cam._tz || cam.zoom);
    }
    cam.tyaw = -target._ang + Math.PI / 2; cam.vyaw = 0;
  }

  /* ---------------- 探针（真实三路 RRF） ---------------- */
  async function runProbe() {
    var q = MN.$("smProbe").value.trim();
    if (!q) { probe = null; var box0 = MN.$("smProbeResult"); box0.className = "sm-probe-result"; return; }
    var box = MN.$("smProbeResult");
    box.innerHTML = '<div class="row">检索中…</div>';
    box.className = "sm-probe-result show";
    try {
      var d = await MN.api("recall", { q: q, scope: MN.state.scope, limit: 6 }, null, null, 90000);
      var items = d.items || [];
      var set = {}, order = [], colors = {};
      var RC = { semantic: ["向量", "#6fd3e8"], lexical: ["关键词", "#ffd27a"], recency: ["时间", "#c39dff"] };
      items.forEach(function (c, i) {
        if (!byId[c.id]) return;
        set[c.id] = 1; order.push(c.id);
        var ch = c.channels || {};
        var pick = ch.semantic ? "semantic" : ch.lexical ? "lexical" : "recency";
        colors[c.id] = RC[pick][1];
      });
      probe = { set: set, order: order, colors: colors };
      if (!order.length) {
        box.innerHTML = '<div class="row">没有召回任何记忆</div>';
      } else {
        box.innerHTML = items.map(function (c, i) {
          var ch = c.channels || {};
          var chips = Object.keys(ch).map(function (k) {
            var rc = RC[k] || [k, "#888"];
            return '<span class="sm-chip-route" style="color:' + rc[1] + ';border-color:' + rc[1] + '66">' + rc[0] + "#" + ch[k] + "</span>";
          }).join("");
          return '<div class="row" data-smjump="' + c.id + '">'
            + '<span class="sm-rank" style="background:' + (colors[c.id] || "#f0c674") + '">' + (i + 1) + "</span>"
            + "<span>" + MN.esc(c.content.slice(0, 22)) + "…</span>"
            + '<span class="sm-routes">' + chips + "</span></div>";
        }).join("");
        box.querySelectorAll("[data-smjump]").forEach(function (r) {
          r.addEventListener("click", function () {
            focusNode(r.getAttribute("data-smjump"));
          });
        });
      }
      selected = order.length ? byId[order[0]] : null;
      renderDetail();
    } catch (e) {
      box.innerHTML = '<div class="row">' + MN.esc(e.message) + "</div>";
    }
  }

  /* ---------------- 交互（Pointer Events：鼠标/触屏同一套） ---------------- */
  var tip, drag = null, tlDrag = false;
  function bindOnce() {
    cv = MN.$("smCanvas");
    if (!cv) return false;
    ctx = cv.getContext("2d");
    tip = MN.$("smTip");
    tlCv = MN.$("smTlCanvas");
    tlCtx = tlCv ? tlCv.getContext("2d") : null;

    cv.addEventListener("pointerdown", function (ev) {
      drag = { sx: ev.clientX, sy: ev.clientY, yaw: cam.yaw, pitch: cam.pitch, moved: false };
      cam.vyaw = 0; cam.vpitch = 0; cam.tyaw = null; cam.tpitch = null;
    });
    cv.addEventListener("pointermove", function (ev) {
      if (drag) {
        var dx = ev.clientX - drag.sx, dy = ev.clientY - drag.sy;
        if (Math.abs(dx) + Math.abs(dy) > 4) drag.moved = true;
        cam.yaw = drag.yaw + dx * 0.006;
        var pt = drag.pitch + dy * 0.005;
        if (pt > 1.52) pt = 1.52;
        if (pt < -1.52) pt = -1.52;
        cam.pitch = pt;
        cam.vyaw = (ev.movementX || 0) * 0.00018;
        cam.vpitch = (ev.movementY || 0) * 0.00012;
        return;
      }
      var rect = cv.getBoundingClientRect();
      var x = ev.clientX - rect.left, y = ev.clientY - rect.top;
      /* 星座视图：先测代表星（命中即提示"点击放大"） */
      if (memberAlpha < 0.5) {
        var bc = null, bd2 = 1e9;
        CLIST.forEach(function (cl) {
          if (!cl.big || cl._sx == null) return;
          var dd = Math.hypot(x - cl._sx, y - cl._sy);
          if (dd < cl._sr && dd < bd2) { bd2 = dd; bc = cl; }
        });
        hoveredCl = bc;
        if (bc) {
          tip.style.display = "block";
          tip.style.left = Math.min(x + 16, W - 276) + "px";
          tip.style.top = Math.min(y + 12, H - 90) + "px";
          tip.innerHTML = "<div>✦ " + MN.esc(bc.name || "未名星座") + "</div><div class='m'>"
            + bc.members.length + " 条记忆 · 点击放大查看</div>";
          cv.style.cursor = "pointer"; hovered = null;
          return;
        }
      } else {
        hoveredCl = null;
      }
      var best = null, bd = 1e9;
      MEMS.forEach(function (m) {
        if (m._sx == null || !visible(m)) return;
        var d = Math.hypot(x - m._sx, y - m._sy);
        if (d < Math.max(m._sr * 2.4, 9) && d < bd) { bd = d; best = m; }
      });
      hovered = best;
      if (best && tip) {
        tip.style.display = "block";
        tip.style.left = Math.min(x + 16, W - 276) + "px";
        tip.style.top = Math.min(y + 12, H - 90) + "px";
        tip.innerHTML = "<div>" + MN.esc(best.content) + "</div><div class='m'>"
          + (TYPES[best.type] || TYPES.fact).label + " · 强度 " + Math.round(best.strength || 0)
          + " · 召回 " + (best.heat || 0) + " 次 · " + fmtDate(best.t) + (best.active ? " · 主动" : "") + "</div>";
        cv.style.cursor = "pointer";
      } else { if (tip) tip.style.display = "none"; cv.style.cursor = "grab"; }
    });
    window.addEventListener("pointerup", function () {
      if (drag && !drag.moved) {
        if (hoveredCl && memberAlpha < 0.5) {
          /* 点击星座：转向它并放大到成员视图 */
          userZoomed = true;
          cam._tz = Math.max(fitZoom * 2.4, cam._tz || cam.zoom);
          cam.tyaw = -hoveredCl.amid + Math.PI / 2;
          cam.tpitch = 0.55;
          hoveredCl = null;
          tip.style.display = "none";
        } else {
          selected = (hovered && visible(hovered)) ? hovered : null;
          renderDetail();
        }
      }
      drag = null;
    });
    /* 双击空白：复位视角（双击在星/星座上不复位） */
    cv.addEventListener("dblclick", function () {
      if (hovered || hoveredCl) return;
      resetView();
    });
    /* Esc：先清探针 → 再取消选中 → 最后复位视图 */
    window.addEventListener("keydown", function (ev) {
      if (ev.key !== "Escape") return;
      var pi = MN.$("smProbe");
      if (pi && pi.value) {
        pi.value = ""; probe = null;
        var pb = MN.$("smProbeResult");
        if (pb) pb.className = "sm-probe-result";
      }
      else if (selected) { selected = null; renderDetail(); }
      else if (userZoomed) { resetView(); }
      hovered = null; hoveredCl = null;
      if (tip) tip.style.display = "none";
    });
    cv.addEventListener("wheel", function (ev) {
      ev.preventDefault();
      userZoomed = true;
      cam._tz = Math.max(0.2, Math.min(4.5, (cam._tz || cam.zoom) * (ev.deltaY > 0 ? 0.9 : 1.12)));
    }, { passive: false });

    /* 时间轴：轨道拖拽 + 滑杆 + 播放（一颗星=一条真实记忆） */
    function tlScrub(ev) {
      var r = MN.$("smTrack").getBoundingClientRect();
      setReveal((ev.clientX - r.left) / r.width); stopPlay();
    }
    MN.$("smTrack").addEventListener("pointerdown", function (ev) { tlDrag = true; tlScrub(ev); });
    window.addEventListener("pointermove", function (ev) { if (tlDrag) tlScrub(ev); });
    window.addEventListener("pointerup", function () { tlDrag = false; });
    var range = MN.$("smReveal");
    range.addEventListener("input", function () { setReveal(+range.value / 1000); stopPlay(); });
    MN.$("smPlay").addEventListener("click", function () {
      playing = !playing;
      this.textContent = playing ? "❚❚" : "▶";
      if (playing) {
        if (reveal >= 1) setReveal(0);
        (function step() {
          if (!playing) return;
          setReveal(reveal + 0.003);
          if (reveal >= 1) { stopPlay(); return; }
          requestAnimationFrame(step);
        })();
      }
    });
    MN.$("smProbe").addEventListener("keydown", function (ev) { if (ev.key === "Enter") runProbe(); });
    window.addEventListener("resize", function () { resize(); drawTimelineBar(); });
    return true;
  }
  var playing = false;
  function stopPlay() { playing = false; MN.$("smPlay").textContent = "▶"; }

  /* ---------------- 入口 ---------------- */
  var bound = false;
  window.MNStarmap = {
    load: async function () {
      if (!bound) { bound = bindOnce(); if (!bound) return; }
      resize();
      if (!loaded) {
        setReveal(1);
        setStatus("加载星图…");
        try { await fetchData(); }
        catch (e) { setStatus("加载失败：" + e.message); return; }
        setReveal(1);
        requestAnimationFrame(frame);
      } else {
        resize();
      }
    },
    reload: function () {
      loaded = false; MEMS = []; EDGES = []; SUPS = [];
      byId = {}; CLIST = []; CLUSTERS = {}; BACKBONE = {};
      probe = null; selected = null; hovered = null; hoveredCl = null;
      userZoomed = false;
    }
  };
})();
