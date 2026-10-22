/* n-cartpole dashboard — served by the ESP32-S3 controller (firmware/web/).

   Vanilla JS, no build step, no libraries. Talks to the device's API
   (firmware/docs/design.md): GET /api/info, WebSocket /ws (binary telemetry
   frames + status JSON at 2 Hz), POST /api/cmd, and the uploads.

   - The telemetry layout is ncp_telemetry.h (232-byte packed little-endian
     frames behind a 4-byte header); TLM below decodes those exact offsets.
   - The model blob is ncp_model.h (.ncpm: 512-byte header, then float32/float16
     weights in forward order with the RunningNorm folded in). The page fetches
     /api/model and runs the same network on every streamed observation to draw
     its activations. The forward pass, scene, plots, goal pictograms and network
     diagram are ported from the personal site's /controls/swingup/ page
     (personal-site/assets/js/swingup.js: createPolicy, drawScene, drawPlot,
     drawNet, goalIcon), generalised to the blob's dims.

   firmware/tools/mock_device.py serves this page with a simulated plant. */
(function () {
  'use strict';

  /* ================================================================
     Formats: ncp_telemetry.h and ncp_model.h
  ================================================================ */
  var MAX_LINKS = 4, MAX_STATE = 2 + 2 * MAX_LINKS, MAX_POS = 1 + MAX_LINKS, MAX_HIST = 8;
  var MAX_OBS = 2 + 3 * MAX_LINKS + MAX_LINKS + MAX_HIST;
  var TLM = {
    VERSION: 1, HDR: 4, FRAME: 232,
    tick: 0, run: 4, mode: 5, goal: 6, flags: 7,
    est: 8, meas: 8 + 4 * MAX_STATE, truth: 8 + 4 * (MAX_STATE + MAX_POS),
    obs: 8 + 4 * (2 * MAX_STATE + MAX_POS),
    cmd: 212, applied: 216, x_ref: 220, loop_us: 224, policy_us: 226, est_us: 228, period_us: 230
  };
  var RUN = ['idle', 'armed', 'fault', 'maint'];
  var FLAG = { SETTLED: 1, RANDOM: 2, SIM: 4, TRUTH: 8 };
  var MODE = ['ai', 'manual'];
  var NCP = { HEADER: 512, GOAL: 1, DELTA: 2, EKF: 4 };

  function decodeFrame(dv, o) {
    function f32s(off, n) {
      var a = new Float32Array(n);
      for (var i = 0; i < n; i++) a[i] = dv.getFloat32(o + off + 4 * i, true);
      return a;
    }
    return {
      tick: dv.getUint32(o + TLM.tick, true),
      run: dv.getUint8(o + TLM.run), mode: dv.getUint8(o + TLM.mode),
      goal: dv.getUint8(o + TLM.goal), flags: dv.getUint8(o + TLM.flags),
      est: f32s(TLM.est, MAX_STATE), meas: f32s(TLM.meas, MAX_POS),
      truth: f32s(TLM.truth, MAX_STATE), obs: f32s(TLM.obs, MAX_OBS),
      cmd: dv.getFloat32(o + TLM.cmd, true), applied: dv.getFloat32(o + TLM.applied, true),
      x_ref: dv.getFloat32(o + TLM.x_ref, true),
      loop_us: dv.getUint16(o + TLM.loop_us, true), policy_us: dv.getUint16(o + TLM.policy_us, true),
      est_us: dv.getUint16(o + TLM.est_us, true), period_us: dv.getUint16(o + TLM.period_us, true)
    };
  }

  // ncp_header_t, byte for byte.
  function parseNcpm(buf) {
    if (buf.byteLength < NCP.HEADER) throw new Error('shorter than the 512-byte header');
    var dv = new DataView(buf, 0, NCP.HEADER), u8 = new Uint8Array(buf, 0, NCP.HEADER);
    if (String.fromCharCode(u8[0], u8[1], u8[2], u8[3]) !== 'NCPM') throw new Error('not an .ncpm model (magic)');
    function d(off) { return dv.getFloat64(off, true); }
    function arr(off, n) { var a = []; for (var i = 0; i < n; i++) a.push(d(off + 8 * i)); return a; }
    var h = {
      version: dv.getUint16(4, true), header_size: dv.getUint16(6, true),
      total_size: dv.getUint32(8, true), crc32: dv.getUint32(12, true), train_step: dv.getUint32(16, true),
      n_links: u8[20], n_blocks: u8[21], expand: u8[22], dtype: u8[23],
      hidden: dv.getUint16(24, true), obs_dim: dv.getUint16(26, true),
      flags: u8[28], action_history: u8[29], delay_steps: u8[30], n_goals: u8[31],
      goal_settle_steps: dv.getUint16(32, true),
      dt: d(40), force_max: d(48), force_slew: d(56), x_lim: d(64),
      M: d(72), g: d(80), b: d(88)
    };
    var n = Math.min(h.n_links, MAX_LINKS);
    h.masses = arr(120, n); h.lengths = arr(152, n); h.com = arr(184, n);
    h.goal_tol_angle = d(336); h.goal_tol_vel = d(344);
    var lab = '';
    for (var i = 352; i < 416 && u8[i]; i++) lab += String.fromCharCode(u8[i]);
    try { lab = decodeURIComponent(escape(lab)); } catch (e) { /* not UTF-8: keep bytes */ }
    h.label = lab;
    if (h.version !== 1) throw new Error('blob version ' + h.version + ' (this page reads 1)');
    if (h.n_links < 1 || h.n_links > MAX_LINKS || !h.hidden || !h.obs_dim || h.n_blocks > 4) throw new Error('dims out of range');
    h.goal = !!(h.flags & NCP.GOAL); h.delta = !!(h.flags & NCP.DELTA); h.ekf = !!(h.flags & NCP.EKF);
    h.params = weightCount(h);
    h.expected_size = NCP.HEADER + h.params * (h.dtype ? 2 : 4);
    return h;
  }

  function weightCount(h) {
    var H = h.hidden, E = H * h.expand, D = h.obs_dim;
    return H * D + H + h.n_blocks * (2 * H + E * H + E + H * E + H) + 2 * H + H + 1;
  }

  var CRC_TABLE = null;
  function crc32(u8, skipFrom, skipTo) {   // zlib.crc32, reading [skipFrom, skipTo) as zeros
    if (!CRC_TABLE) {
      CRC_TABLE = new Uint32Array(256);
      for (var n = 0; n < 256; n++) {
        var c = n;
        for (var k = 0; k < 8; k++) c = c & 1 ? 0xEDB88320 ^ (c >>> 1) : c >>> 1;
        CRC_TABLE[n] = c >>> 0;
      }
    }
    var crc = 0xFFFFFFFF;
    for (var i = 0; i < u8.length; i++) {
      var b = i >= skipFrom && i < skipTo ? 0 : u8[i];
      crc = CRC_TABLE[(crc ^ b) & 0xFF] ^ (crc >>> 8);
    }
    return (crc ^ 0xFFFFFFFF) >>> 0;
  }

  function halfToFloat(buffer, offset, count) {
    var dv = new DataView(buffer, offset), out = new Float32Array(count);
    for (var i = 0; i < count; i++) {
      var h = dv.getUint16(2 * i, true);
      var sign = h & 0x8000 ? -1 : 1, e = (h >> 10) & 0x1f, f = h & 0x3ff;
      out[i] = e === 0 ? sign * f * 5.960464477539063e-8
             : e === 31 ? (f ? NaN : sign * Infinity)
             : sign * (1 + f / 1024) * Math.pow(2, e - 15);
    }
    return out;
  }

  /* The SimBa actor's deterministic head (ncp_policy.h; swingup.js createPolicy):
     in_proj → blocks × [h += fc2(relu(fc1(LN(h))))] → LN → mean; a = tanh(mean)·Fmax. */
  function createPolicy(h, buffer) {
    var count = h.params, all;
    if (h.dtype === 0) {
      all = new Float32Array(buffer.slice(NCP.HEADER, NCP.HEADER + 4 * count));
    } else {
      all = halfToFloat(buffer, NCP.HEADER, count);
    }
    var H = h.hidden, E = H * h.expand, D = h.obs_dim, B = h.n_blocks, pos = 0;
    function take(n) { var t = all.subarray(pos, pos + n); pos += n; return t; }
    var T = { in_w: take(H * D), in_b: take(H), blocks: [] };
    for (var b = 0; b < B; b++) {
      T.blocks.push({ ln_w: take(H), ln_b: take(H), fc1_w: take(E * H), fc1_b: take(E), fc2_w: take(H * E), fc2_b: take(H) });
    }
    T.fl_w = take(H); T.fl_b = take(H); T.out_w = take(H); T.out_b = take(1);
    var x = new Float32Array(H), t = new Float32Array(H), u = new Float32Array(E);
    var trace = { obs: new Float32Array(D), h: [new Float32Array(H)], u: [], mean: 0, action: 0 };
    for (b = 0; b < B; b++) { trace.u.push(new Float32Array(E)); trace.h.push(new Float32Array(H)); }

    function linear(W, bias, inp, inDim, outDim, out) {
      for (var o = 0; o < outDim; o++) {
        var s = bias[o], row = o * inDim;
        for (var i = 0; i < inDim; i++) s += W[row + i] * inp[i];
        out[o] = s;
      }
    }
    function layerNorm(v, w, bias, out) {
      var mean = 0, va = 0, i;
      for (i = 0; i < H; i++) mean += v[i];
      mean /= H;
      for (i = 0; i < H; i++) { var dd = v[i] - mean; va += dd * dd; }
      var inv = 1 / Math.sqrt(va / H + 1e-5);
      for (i = 0; i < H; i++) out[i] = (v[i] - mean) * inv * w[i] + bias[i];
    }
    function act(obs) {
      for (var i = 0; i < D; i++) trace.obs[i] = obs[i] || 0;
      linear(T.in_w, T.in_b, trace.obs, D, H, x);
      trace.h[0].set(x);
      for (var bi = 0; bi < B; bi++) {
        var P = T.blocks[bi];
        layerNorm(x, P.ln_w, P.ln_b, t);
        linear(P.fc1_w, P.fc1_b, t, H, E, u);
        for (i = 0; i < E; i++) if (u[i] < 0) u[i] = 0;
        for (var o = 0; o < H; o++) {
          var s = P.fc2_b[o], row = o * E;
          for (var j = 0; j < E; j++) s += P.fc2_w[row + j] * u[j];
          x[o] += s;
        }
        trace.u[bi].set(u); trace.h[bi + 1].set(x);
      }
      layerNorm(x, T.fl_w, T.fl_b, t);
      var m = T.out_b[0];
      for (var k = 0; k < H; k++) m += T.out_w[k] * t[k];
      trace.mean = m; trace.action = Math.tanh(m) * h.force_max;
      return trace.action;
    }
    return { act: act, trace: trace };
  }

  // Goal index ↔ label (n_cartpole.env.goals.goal_configs): bit k set = link k down; labels base link first.
  function goalLabel(n, g) { var s = ''; for (var k = 0; k < n; k++) s += (g >> k) & 1 ? 'D' : 'U'; return s; }
  function goalIndex(n, label) { var g = 0; for (var k = 0; k < n; k++) if (label[k] === 'D') g |= 1 << k; return g; }
  function goalTarget(n, g) { var t = []; for (var k = 0; k < n; k++) t.push((g >> k) & 1 ? Math.PI : 0); return t; }
  function wrapAngle(a) { a = (a + Math.PI) % (2 * Math.PI); if (a < 0) a += 2 * Math.PI; return a - Math.PI; }
  // Degrees in [−90°, 270°): up (0°) and down (180°) both away from the wrap, so neither flickers in sign.
  function lane(a) { return wrapAngle(a - Math.PI / 2) * 180 / Math.PI + 90; }

  // Labels of the network's inputs, from the blob's dims (ncp_ctrl.h step 2).
  function obsLayout(h) {
    var sub = ['₁', '₂', '₃', '₄'], L = [{ l: 'x − xref', s: h.x_lim || 0.5 }, { l: 'ẋ', s: 2 }];
    for (var k = 0; k < h.n_links; k++) {
      L.push({ l: 'cos θ' + sub[k], s: 1 }, { l: 'sin θ' + sub[k], s: 1 }, { l: 'θ̇' + sub[k], s: 10 });
    }
    if (h.goal) for (k = 0; k < h.n_links; k++) L.push({ l: 'cos goal' + sub[k], s: 1, goal: true });
    h.hist_at = L.length;
    for (k = 0; k < h.action_history; k++) L.push({ l: 'F[t−' + (k + 1) + ']/Fmax', s: 1 });
    while (L.length < h.obs_dim) L.push({ l: 'o[' + L.length + ']', s: 1 });
    return L.slice(0, h.obs_dim);
  }

  /* ================================================================
     Page
  ================================================================ */
  var $ = function (id) { return document.getElementById(id); };
  var root = document.documentElement;
  function cssVar(name) { return getComputedStyle(root).getPropertyValue(name).trim(); }
  function clamp(v, lo, hi) { return Math.min(hi, Math.max(lo, v)); }
  function get(o, path, dflt) {
    var p = path.split('.');
    for (var i = 0; i < p.length; i++) { if (o == null) return dflt; o = o[p[i]]; }
    return o == null ? dflt : o;
  }
  function setText(id, v) { var e = $(id); if (e && e.textContent !== String(v)) e.textContent = v; }
  function esc(s) { return String(s).replace(/[&<>"']/g, function (c) { return '&#' + c.charCodeAt(0) + ';'; }); }
  function minus(s) { return String(s).replace(/^-/, '−'); }
  function num(v, d) { var s = (+v).toFixed(d); if (/^-0\.?0*$/.test(s)) s = s.slice(1); return minus(s); }
  function signed(v, d) { var s = num(v, d); return s.charAt(0) === '−' ? s : '+' + s; }
  function us(v) { return v == null ? '—' : v >= 10000 ? (v / 1000).toFixed(1) + ' ms' : Math.round(v) + ' µs'; }
  function bytes(n) {
    if (n == null) return '—';
    if (n >= 1048576) return (n / 1048576).toFixed(2) + ' MB';
    if (n >= 1024) return (n / 1024).toFixed(n >= 102400 ? 0 : 1) + ' KB';
    return n + ' B';
  }
  function duration(s) {
    if (s == null) return '—';
    s = Math.max(0, Math.floor(s));
    var d = Math.floor(s / 86400), h = Math.floor(s / 3600) % 24, m = Math.floor(s / 60) % 60, x = s % 60;
    if (d) return d + 'd ' + h + 'h';
    if (h) return h + 'h ' + (m < 10 ? '0' : '') + m + 'm';
    if (m) return m + 'm ' + (x < 10 ? '0' : '') + x + 's';
    return x + 's';
  }
  function plural(n, w) { return n + ' ' + w + (n === 1 ? '' : 's'); }

  var S = {
    info: null, status: null, statusAt: 0,
    hdr: null, policy: null, layout: null,
    model: { state: 'idle', got: 0, total: 0, crc: null, error: '' },
    conn: 'connecting', everLive: false, lastMsg: 0,
    last: null, hist: [], frameKey: '',
    run: null, mode: null, goal: null, random: false, xref: 0, fault: '', backend: null,
    seg: null, pendingGoal: null, pendingUntil: 0,
    parity: null, netMs: 0, netFps: 0,
    rebooting: null, lastFrameSeen: 0,
    link: { frames: 0, fps: null, gaps: 0, lost: 0 },  // telemetry as this page receives it
    misses: [], missArm: null, statusRun: null,         // deadline misses over time, at arming
    events: { seq: -1, list: [], loading: false, seenBad: 0 },
    parityN: 0, parityAt: 0, stale: false
  };

  // Geometry and limits: from the blob header when it's loaded, else /api/info, else the rig defaults.
  function geom() {
    var h = S.hdr, m = get(S.info, 'model', null) || {};
    var n = (h && h.n_links) || m.n_links || 2;
    var L = h && h.lengths && h.lengths.length === n ? h.lengths : null;
    if (!L) { L = []; for (var k = 0; k < n; k++) L.push(0.25); }
    return {
      n: n, lengths: L, masses: (h && h.masses) || L.map(function () { return 0.12; }),
      x_lim: (h && h.x_lim) || m.x_lim || 0.5, dt: (h && h.dt) || m.dt || 0.005,
      fmax: (h && h.force_max) || m.force_max || 15, slew: (h && h.force_slew) || m.force_slew || 0,
      settle: (h && h.goal_settle_steps) || 50,
      tolA: (h && h.goal_tol_angle) || 0.3, tolV: (h && h.goal_tol_vel) || 1.5
    };
  }
  function goalLabels() {
    var n = geom().n, out = [];
    for (var g = 0; g < 1 << n; g++) out.push(goalLabel(n, g));
    return out;
  }
  // The pose the plant is resting in (within the "reached" tolerances), or null.
  function poseOf(st) {
    var G = geom();
    for (var g = 0; g < 1 << G.n; g++) {
      var t = goalTarget(G.n, g), ok = true;
      for (var k = 0; k < G.n && ok; k++) {
        if (!(Math.abs(wrapAngle(st[2 + 2 * k] - t[k])) < G.tolA) || !(Math.abs(st[3 + 2 * k]) < G.tolV)) ok = false;
      }
      if (ok) return goalLabel(G.n, g);
    }
    return null;
  }

  /* ---- Theme toggle (the site's site.js) ---- */
  (function () {
    var btn = document.querySelector('.theme-toggle');
    function isDark() {
      var t = root.getAttribute('data-theme');
      return t ? t === 'dark' : window.matchMedia('(prefers-color-scheme: dark)').matches;
    }
    btn.setAttribute('aria-checked', isDark() ? 'true' : 'false');
    btn.addEventListener('click', function () {
      var next = isDark() ? 'light' : 'dark';
      root.setAttribute('data-theme', next);
      btn.setAttribute('aria-checked', next === 'dark' ? 'true' : 'false');
      try { localStorage.setItem('theme', next); } catch (e) {}
      rgbCache = {};
    });
  })();

  /* ---- HTTP helpers ---- */
  function fetchJSON(url, opts, timeoutMs) {
    opts = opts || {};
    var ctl = typeof AbortController !== 'undefined' ? new AbortController() : null, timer = null;
    if (ctl) { opts.signal = ctl.signal; timer = setTimeout(function () { ctl.abort(); }, timeoutMs || 5000); }
    opts.cache = 'no-store';
    return fetch(url, opts).then(function (r) {
      return r.text().then(function (t) {
        var j = null;
        try { j = t ? JSON.parse(t) : null; } catch (e) { j = null; }
        if (!r.ok) {
          var err = new Error((j && j.error) || ('HTTP ' + r.status));
          err.status = r.status;
          throw err;
        }
        return j;
      });
    }).finally(function () { if (timer) clearTimeout(timer); });
  }

  var msgTimer = null;
  function flash(id, text, kind, ms) {
    var e = $(id);
    if (!e) return;
    e.textContent = text;
    e.className = 'panel-msg' + (kind ? ' ' + kind : '');
    clearTimeout(msgTimer);
    if (ms !== 0) msgTimer = setTimeout(function () { e.textContent = ''; e.className = 'panel-msg'; }, ms || 4000);
  }

  function send(cmd) {
    return fetchJSON('api/cmd', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(cmd) }, 3000)
      .catch(function (e) {
        var k = Object.keys(cmd)[0];
        flash('cmd-msg', k + ': ' + (e.name === 'AbortError' ? 'no answer from the device' : e.message), 'bad', 5000);
        throw e;
      });
  }

  /* ---- Connection: WebSocket with backoff, info on every (re)connect ---- */
  var ws = null, wsTimer = null, backoff = 400;
  function setConn(state) {
    S.conn = state;
    renderStale();
    var c = $('conn');
    c.dataset.state = state;
    setText('conn-text', { live: 'live', connecting: 'connecting', reconnecting: 'reconnecting', rebooting: 'rebooting', offline: 'offline' }[state] || state);
    renderControls(true);
  }
  function connect() {
    clearTimeout(wsTimer);
    if (ws) { try { ws.onclose = null; ws.close(); } catch (e) {} ws = null; }
    if (S.conn !== 'rebooting') setConn(S.everLive ? 'reconnecting' : 'connecting');
    var sock;
    try {
      sock = new WebSocket((location.protocol === 'https:' ? 'wss://' : 'ws://') + location.host + '/ws');
    } catch (e) { scheduleReconnect(); return; }
    ws = sock;
    sock.binaryType = 'arraybuffer';
    sock.onopen = function () {
      backoff = 400;
      S.lastMsg = performance.now();
      loadInfo();
    };
    sock.onmessage = function (ev) {
      S.lastMsg = performance.now();
      if (S.conn !== 'live' && !(S.rebooting && performance.now() - S.rebooting.t0 < 2500)) { S.everLive = true; setConn('live'); }
      if (typeof ev.data === 'string') {
        var s = null;
        try { s = JSON.parse(ev.data); } catch (e) { return; }
        onStatus(s);
      } else {
        onTelemetry(ev.data);
      }
    };
    sock.onclose = function () {
      if (ws !== sock) return;
      ws = null;
      if (S.conn !== 'rebooting') setConn(S.everLive ? 'reconnecting' : 'connecting');
      scheduleReconnect();
    };
    sock.onerror = function () { try { sock.close(); } catch (e) {} };
  }
  function scheduleReconnect() {
    clearTimeout(wsTimer);
    wsTimer = setTimeout(connect, backoff);
    backoff = Math.min(Math.round(backoff * 1.6), 6000);
  }
  // A socket that stays open but goes quiet (Wi-Fi drop without a FIN) is dead too.
  setInterval(function () {
    if (ws && ws.readyState === 1 && performance.now() - S.lastMsg > 4000) { try { ws.close(); } catch (e) {} }
    S.link.fps = S.conn === 'live' ? S.link.frames : null;
    S.link.frames = 0;
    renderLink();
    renderStale();
    renderParity();
  }, 1000);

  /* Stale: the connection dropped, or the device stopped sending. Everything that shows a
     live value fades (CSS .stale) and a banner says since when. */
  function renderStale() {
    var now = performance.now();
    var quietFrames = S.last && now - S.lastFrameSeen > 1500, quietStatus = S.statusAt && now - S.statusAt > 3000;
    var stale = S.everLive && (S.conn !== 'live' || !!quietFrames || !!quietStatus);
    if (stale !== S.stale) { S.stale = stale; document.body.classList.toggle('stale', stale); }
    if (S.conn === 'live') {   // the socket is open but nothing arrives
      $('conn').dataset.state = stale ? 'reconnecting' : 'live';
      setText('conn-text', stale ? 'stale' : 'live');
    }
    $('stale-banner').hidden = !stale;
    if (stale) {
      var since = Math.max(S.lastFrameSeen || 0, S.statusAt || 0), ago = since ? (now - since) / 1000 : null;
      setText('stale-text', (S.conn === 'rebooting' ? 'The device is rebooting' : S.conn !== 'live' ? 'Connection lost' : 'The device stopped sending') +
        (ago != null ? ' ' + duration(ago) + ' ago' : '') + ': the values below are frozen.');
    }
  }

  function loadInfo() {
    return fetchJSON('api/info', null, 5000).then(function (info) {
      var prev = S.info;
      S.info = info || {};
      renderInfo();
      var m = get(S.info, 'model', null);
      var crc = m ? m.crc32 || m.label : null;
      if (!m) { if (S.model.state !== 'none') { S.policy = null; S.hdr = null; S.model = { state: 'none', got: 0, total: 0, crc: null, error: '' }; renderModel(); } }
      else if (crc !== S.model.crc || S.model.state === 'error' || S.model.state === 'none') loadModel(crc, m.bytes);
      if (S.rebooting && prev) rebootDone(prev);
      if (!releasesLoaded) loadReleases();
      return info;
    }).catch(function () { /* the socket's reconnect loop retries */ });
  }

  /* ---- Event log: fetched when status.events_seq moves ---- */
  function loadEvents() {
    if (S.events.loading) return;
    S.events.loading = true;
    fetchJSON('api/events', null, 4000).then(function (j) {
      S.events.loading = false;
      if (!j) return;
      var wasSeq = S.events.seq;
      S.events.seq = j.seq; S.events.list = j.events || []; S.events.boot = j.boot; S.events.uptimeMs = j.uptime_ms; S.events.at = performance.now();
      // Errors that arrived while Health was not on screen mark its tab.
      var bad = S.events.list.filter(function (e) { return e.level === 'error'; });
      var newest = bad.length ? bad[bad.length - 1].seq : 0;
      if (wasSeq < 0) S.events.seenBad = newest;   // the backlog at page load is not news
      if (S.tab === 'health') S.events.seenBad = newest;
      S.events.newBad = newest > S.events.seenBad;
      renderEvents();
    }).catch(function () { S.events.loading = false; });
  }

  // Misses per minute over the last minute of status messages.
  function missRate() {
    var mh = S.misses;
    if (mh.length < 2) return null;
    var a = mh[0], b = mh[mh.length - 1], dt = (b.t - a.t) / 60000;
    return dt > 0.05 ? (b.n - a.n) / dt : null;
  }

  /* ---- Model: download the device's blob, with progress ---- */
  function loadModel(crc, size) {
    S.model = { state: 'loading', got: 0, total: size || 0, crc: crc, error: '' };
    S.policy = null;
    renderModel();
    fetch('api/model', { cache: 'no-store' }).then(function (r) {
      if (r.status === 404) { S.model.state = 'none'; return null; }
      if (!r.ok) throw new Error('HTTP ' + r.status);
      S.model.total = +r.headers.get('Content-Length') || S.model.total;
      if (!r.body || !r.body.getReader) return r.arrayBuffer();
      var reader = r.body.getReader(), chunks = [], got = 0;
      function pump() {
        return reader.read().then(function (res) {
          if (res.done) {
            var all = new Uint8Array(got), o = 0;
            chunks.forEach(function (c) { all.set(c, o); o += c.length; });
            return all.buffer;
          }
          chunks.push(res.value); got += res.value.length; S.model.got = got;
          return pump();
        });
      }
      return pump();
    }).then(function (buf) {
      if (!buf) { renderModel(); return; }
      var h = parseNcpm(buf);
      if (buf.byteLength < h.expected_size) throw new Error('truncated download');
      S.hdr = h;
      S.layout = obsLayout(h);
      S.policy = createPolicy(h, buf);
      S.parity = null; S.parityN = 0;
      S.model.state = 'ready';
      renderModel();
      buildGoalButtons();
      renderControls(true);
    }).catch(function (e) {
      S.model.state = 'error'; S.model.error = e.message || String(e);
      renderModel();
    });
  }

  /* ---- Telemetry ---- */
  var tlmWarned = false;
  function onTelemetry(buf) {
    if (buf.byteLength < TLM.HDR) return;
    var dv = new DataView(buf), ver = dv.getUint8(0), count = dv.getUint8(1), fs = dv.getUint16(2, true);
    if (ver !== TLM.VERSION || fs < TLM.FRAME) {
      if (!tlmWarned) { tlmWarned = true; flash('cmd-msg', 'Telemetry v' + ver + ' (' + fs + ' B frames): this page reads v1 — reload it after a firmware update.', 'bad', 0); }
      if (fs < TLM.FRAME) return;
    }
    for (var i = 0; i < count; i++) {
      var off = TLM.HDR + i * fs;
      if (off + TLM.FRAME > buf.byteLength) break;
      pushFrame(decodeFrame(dv, off));
    }
  }

  var HIST_S = 6;
  function pushFrame(f) {
    var G = geom(), last = S.last;
    if (last && f.tick < last.tick) { S.hist = []; S.seg = null; }   // the device rebooted
    S.link.frames++;
    if (last && f.tick > last.tick) {
      var rate = get(S.info, 'telemetry.rate_hz', 0), step = rate ? Math.max(1, Math.round(1 / (G.dt * rate))) : 0;
      var missed = step ? Math.round((f.tick - last.tick) / step) - 1 : 0;
      if (missed > 0) { S.link.gaps++; S.link.lost += missed; }
    }
    var prevRun = S.run;
    S.last = f; S.lastFrameSeen = performance.now();
    S.run = RUN[f.run] || 'idle'; S.mode = MODE[f.mode] || 'ai';
    S.goal = goalLabel(G.n, f.goal); S.random = !!(f.flags & FLAG.RANDOM); S.xref = f.x_ref;
    S.backend = f.flags & FLAG.SIM ? 'sim' : 'rig';
    if (S.run !== 'fault') S.fault = '';
    if (S.pendingGoal && (S.pendingGoal === S.goal || performance.now() > S.pendingUntil)) S.pendingGoal = null;

    // Time to target: clock restarts on a new goal, on arming, and on a pose reset.
    if (!S.seg || S.seg.goal !== f.goal || (S.run === 'armed' && prevRun !== 'armed') || S.seg.restart) {
      S.seg = { goal: f.goal, tick0: f.tick, from: poseOf(f.est), reach: null, was: false };
    }
    if (f.flags & FLAG.SETTLED) {
      if (S.seg.reach === null && S.run === 'armed') S.seg.reach = Math.max(0, (f.tick - S.seg.tick0 - G.settle) * G.dt);
      S.seg.was = true;
    }

    var th = [];
    for (var k = 0; k < G.n; k++) th.push(lane(f.est[2 + 2 * k]));
    S.hist.push({ tick: f.tick, x: f.est[0], xr: f.x_ref, th: th, cmd: f.cmd, applied: f.applied, loop: f.loop_us, period: f.period_us });
    var keep = HIST_S / G.dt;
    while (S.hist.length && S.hist[0].tick < f.tick - keep) S.hist.shift();
    if (S.hist.length > 2000) S.hist.shift();
    renderControls(false);
  }

  /* ---- Status (2 Hz) ---- */
  function onStatus(s) {
    S.status = s; S.statusAt = performance.now();
    var L = s.loop || {}, now = performance.now();
    if (L.misses != null) {
      var mh = S.misses;
      if (mh.length && L.misses < mh[mh.length - 1].n) { mh.length = 0; S.missArm = null; }   // rebooted
      mh.push({ t: now, n: L.misses });
      while (mh.length > 2 && now - mh[0].t > 60000) mh.shift();
      if (s.run === 'armed' && S.statusRun !== 'armed') S.missArm = L.misses;
    }
    S.statusRun = s.run;
    if (s.events_seq != null && s.events_seq !== S.events.seq) loadEvents();
    if (!S.last || performance.now() - S.lastFrameSeen > 1000) {
      if (s.run) S.run = s.run;
      if (s.mode) S.mode = s.mode;
      if (s.goal) S.goal = s.goal;
      if (s.random != null) S.random = !!s.random;
      if (s.x_ref != null) S.xref = s.x_ref;
    }
    S.fault = s.run === 'fault' ? s.fault || '' : '';
    if (s.backend) S.backend = s.backend;
    renderStatus();
    renderControls(true);
  }

  /* ---- Rendering: header, info cards ---- */
  function kv(id, rows) {
    var e = $(id);
    if (!e) return;
    e.innerHTML = rows.filter(Boolean).map(function (r) {
      if (r.head) return '<span class="kv-head">' + esc(r.head) + '</span>';
      return '<dt>' + esc(r[0]) + '</dt><dd' + (r[2] ? ' class="' + r[2] + '"' : '') + (r[3] ? ' title="' + esc(r[3]) + '"' : '') + '>' + (r.html ? r[1] : esc(r[1] == null ? '—' : r[1])) + '</dd>';
    }).join('');
  }

  function renderInfo() {
    var I = S.info || {}, fw = I.fw || {}, chip = I.chip || {}, m = I.model, ota = I.ota || {};
    setText('hostname', I.hostname || location.hostname || 'cartpole');
    document.title = (I.hostname || 'cartpole') + ' · live controller';
    var be = I.backend || S.backend || '';
    $('backend').dataset.backend = be;
    setText('backend', be ? be.toUpperCase() : '—');
    $('backend').title = be === 'rig' ? 'Real hardware: commands go to the motor drive' : be === 'sim' ? 'Simulated plant on the chip' : 'Plant backend';
    setText('model-label', m ? m.label || 'unnamed' : 'none');
    setText('foot-ver', fw.version ? 'v' + fw.version : '');
    setText('foot-chip', [chip.model, chip.mhz && chip.mhz + ' MHz', chip.cores && plural(chip.cores, 'core'), fw.idf && 'ESP-IDF ' + fw.idf].filter(Boolean).join(' · '));
    var dt = (m && m.dt) || geom().dt;
    setText('fact-rate', Math.round(1 / dt) + ' Hz');
    setText('eyebrow', [chip.model ? chip.model.toLowerCase() : 'esp32-s3', be === 'rig' ? 'real rig' : be === 'sim' ? 'simulated plant' : null, geom().n === 2 ? 'double pendulum on a cart' : plural(geom().n, 'link') + ' on a cart'].filter(Boolean).join(' · '));
    $('sim-tools').hidden = be !== 'sim';
    $('rig-tools').hidden = be !== 'rig';

    kv('sys-kv', [
      { head: 'chip' },
      ['model', [chip.model, chip.revision != null ? 'rev ' + chip.revision : null].filter(Boolean).join(' ') || null],
      ['cpu', chip.mhz ? chip.cores + ' × ' + chip.mhz + ' MHz' : null],
      ['flash / psram', chip.flash_mb != null ? chip.flash_mb + ' MB / ' + (chip.psram_mb || 0) + ' MB' : null],
      { head: 'firmware' },
      ['version', fw.version ? 'v' + fw.version : null, 'accent'],
      ['backend', be ? (be === 'rig' ? 'rig (motor drive)' : 'sim (plant on the chip)') : null],
      ['esp-idf', fw.idf],
      ['built', fw.built],
      ['telemetry', I.telemetry ? 'v' + I.telemetry.version + ' · ' + I.telemetry.frame_size + ' B · ' + I.telemetry.rate_hz + ' Hz' : null],
      { head: 'this boot' },
      ['boot', I.boot != null ? '#' + I.boot : null],
      ['last reset', I.reset_reason, /crash|watchdog|brownout/.test(I.reset_reason || '') ? 'bad' : '', 'Why the chip last restarted']
    ]);
    var hn = $('host-name');
    if (I.hostname && document.activeElement !== hn) { hn.placeholder = I.hostname; if (!hn.value) hn.value = ''; }
    var sysTag = $('sys-tag');
    sysTag.textContent = be ? be + ' build' : '—';
    sysTag.className = 'tag' + (be === 'rig' ? ' ok' : '');
    kv('ota-kv', [
      ['running', fw.version ? 'v' + fw.version + ' on ' + (ota.running || '?') : ota.running, 'accent'],
      ['next slot', ota.next],
      ['image state', ota.state, ota.state === 'valid' ? 'good' : ota.state ? 'accent' : '', 'pending: rolls back unless it runs healthy for 10 s'],
      ['project', fw.project]
    ]);
    var tag = $('ota-tag');
    tag.textContent = ota.state === 'pending' ? 'new image: pending' : ota.state ? 'image ' + ota.state : '—';
    tag.className = 'tag' + (ota.state === 'valid' ? ' ok' : ota.state === 'pending' ? ' warn' : '');
    renderModel();
    renderStatus();
    renderReleases();
  }

  function renderModel() {
    var m = get(S.info, 'model', null), h = S.hdr, st = S.model.state;
    $('model-dl').hidden = !m;
    var why = get(S.info, 'model_error', '');
    if (!m && st !== 'loading') {
      kv('model-kv', [['model', 'none stored', 'bad'], why ? ['rejected', why, 'bad', 'The stored blob failed validation at boot'] : null,
        { head: 'to add one' }, ['export', 'task export-firmware'], ['then', 'drop the .ncpm on Update']]);
    } else {
      m = m || {};
      var G = geom(), dtype = m.dtype || (h ? (h.dtype ? 'f16' : 'f32') : null);
      var goals = m.goals || (h && h.goal ? goalLabels() : null);
      kv('model-kv', [
        ['label', m.label || (h && h.label), 'accent'],
        ['trained', m.step != null ? (m.step / 1e6).toFixed(m.step % 1e6 ? 2 : 0) + ' M steps' : null],
        ['links', (m.n_links || G.n) + (goals && goals.length ? ' · goals ' + goals.join(' ') : '')],
        { head: 'network' },
        ['shape', (m.hidden || (h && h.hidden)) + ' × ' + (m.blocks != null ? m.blocks : h && h.n_blocks) + ' blocks · ×' + (m.expand || (h && h.expand) || 4)],
        ['inputs', m.obs_dim || (h && h.obs_dim)],
        ['params', m.params != null ? (m.params / 1e6).toFixed(2) + ' M' : h ? (h.params / 1e6).toFixed(2) + ' M' : null],
        ['size', (m.bytes != null ? bytes(m.bytes) : h ? bytes(h.total_size) : '—') + (dtype ? ' · ' + dtype : '')],
        ['weights in', m.weights_in ? (m.weights_in === 'psram' ? 'PSRAM' : 'internal SRAM') : null],
        { head: 'control' },
        ['estimator', m.estimator ? (m.estimator === 'ekf' ? 'EKF' : 'differenced') : h ? (h.ekf ? 'EKF' : 'differenced') : null],
        ['action', m.action_mode ? m.action_mode + (m.action_history ? ' · ' + m.action_history + ' past cmds' : '') : null],
        ['delay', m.delay_steps != null ? plural(m.delay_steps, 'step') : null],
        ['dt', G.dt ? (G.dt * 1000).toFixed(G.dt < 0.01 ? 1 : 0) + ' ms · ' + Math.round(1 / G.dt) + ' Hz' : null],
        ['force', '±' + num(G.fmax, 2) + ' N' + (G.slew ? ' · ' + num(G.slew, 0) + ' N/s' : '')],
        ['rail', '±' + num(G.x_lim, 2) + ' m'],
        ['crc32', m.crc32]
      ]);
    }
    if (h) {
      var xl = Math.round(h.x_lim / 2 * 100) / 100, sl = $('xref');
      sl.min = -xl; sl.max = xl;
      setText('xref-min', minus((-xl).toFixed(2))); setText('xref-max', '+' + xl.toFixed(2));
    }
    renderNetMeta();
  }

  function renderNetMeta() {
    var e = $('nn-meta'), h = S.hdr, st = S.model.state;
    if (st === 'loading') {
      var pct = S.model.total ? Math.round(100 * S.model.got / S.model.total) : null;
      e.textContent = 'downloading weights' + (pct != null ? ' · ' + pct + ' %' : '') + ' · ' + bytes(S.model.got) + (S.model.total ? ' / ' + bytes(S.model.total) : '');
      return;
    }
    if (!h) {
      var why = get(S.info, 'model_error', '');
      e.textContent = st === 'none' ? 'no model on the device' + (why ? ' · stored one rejected: ' + why : '') : st === 'error' ? 'could not load the model: ' + S.model.error : '';
      return;
    }
    var parts = [h.hidden + ' × ' + h.n_blocks + ' blocks', (h.params / 1e6).toFixed(2) + ' M params', h.dtype ? 'f16' : 'f32'];
    if (S.netMs) parts.push('forward ' + S.netMs.toFixed(1) + ' ms');
    var html = esc(parts.join(' · '));
    if (S.parity != null) html += ' · <span class="' + (S.parity < 0.05 ? 'ok' : '') + '" title="Mean |command the browser predicts − command the device sent|">browser vs device |ΔF| ' + S.parity.toFixed(3) + ' N</span>';
    e.innerHTML = html;
  }

  function renderStatus() {
    var s = S.status || {}, L = s.loop || {}, period = L.period_us || geom().dt * 1e6;
    // Health
    var ex = L.exec_avg_us, mx = L.exec_max_us, pol = L.policy_avg_us, est = L.est_avg_us;
    setText('loop-exec', us(ex)); setText('loop-period', us(period));
    setText('fact-exec', ex != null ? Math.round(100 * ex / period) + ' %' : '—');
    var rate = missRate(), sinceArm = S.missArm != null && L.misses != null && s.run === 'armed' ? L.misses - S.missArm : null;
    setText('fact-miss', rate != null ? num(rate, rate < 10 ? 1 : 0) : '—');
    $('fact-miss').className = rate ? 'bad' : '';
    function pct(v) { return clamp(100 * (v || 0) / period, 0, 100); }
    var pp = pct(pol), pe = pct(est), pr = Math.max(0, pct(ex) - pp - pe);
    $('seg-pol').style.width = pp + '%';
    $('seg-est').style.left = pp + '%'; $('seg-est').style.width = pe + '%';
    $('seg-rest').style.left = (pp + pe) + '%'; $('seg-rest').style.width = pr + '%';
    $('loop-max').style.left = pct(mx) + '%';
    $('loop-meter').classList.toggle('over', ex != null && ex > 0.8 * period);
    setText('h-pol', us(pol)); setText('h-est', us(est));
    setText('h-max', mx != null ? us(mx) + ' (' + Math.round(100 * mx / period) + ' %)' : '—');
    setText('h-miss', L.misses == null ? '—' : (rate != null ? num(rate, 1) + ' /min' : '…') + (sinceArm != null ? ' · ' + sinceArm + ' since armed' : '') + ' · ' + L.misses + ' total');
    $('h-miss').className = rate ? 'bad' : L.misses ? 'accent' : '';
    $('h-miss').title = 'Rate over the last minute; the total counts since boot';
    var rt = L.rt_factor;
    setText('h-rt', rt != null ? '×' + rt.toFixed(2) + (rt < 0.995 ? ' slow motion' : '') : '—');
    var bt = $('budget-tag'), ok = L.budget_ok;
    if (ex == null) { bt.textContent = '—'; bt.className = 'tag'; }
    else if (ok === false) { bt.textContent = 'over budget'; bt.className = 'tag bad'; }
    else if (rate) { bt.textContent = 'missing deadlines'; bt.className = 'tag warn'; }
    else { bt.textContent = 'within budget'; bt.className = 'tag ok'; }
    S.healthWarn = (ex != null && ok === false) || !!rate;
    renderIO(s.io, s.calib);
    healthAlert();
    setText('h-note', ok === false
      ? (S.backend === 'rig' ? 'Arming is refused while the loop runs over 80 % of its period: pick a smaller model.' : 'Over budget: the simulator stretches its period instead (slow motion), so the model can still be tried.')
      : 'The loop runs the estimator and the network every tick, armed or not, so its timing is known before arming.');

    // Cores & tasks
    var cores = s.cores || [];
    [0, 1].forEach(function (id) {
      var c = cores.filter(function (x) { return x.id === id; })[0];
      setText('core' + id + '-load', c ? c.load.toFixed(1) + ' %' : '—');
      $('core' + id + '-fill').style.width = c ? clamp(c.load, 0, 100) + '%' : '0';
    });
    renderTasks(s.tasks);

    // Wi-Fi (Settings) and memory + link (Health)
    var w = s.wifi || {};
    setText('up-tag', s.uptime_s != null ? 'up ' + duration(s.uptime_s) : '—');
    renderLink();
    var bars = '', r = w.rssi;
    if (r != null) {
      var lvl = r >= -55 ? 4 : r >= -65 ? 3 : r >= -75 ? 2 : 1;
      bars = '<span class="rssi" aria-hidden="true">' + [4, 6, 8, 11].map(function (hh, i) { return '<i class="' + (i < lvl ? 'on' : '') + '" style="height:' + hh + 'px"></i>'; }).join('') + '</span>';
    }
    kv('wifi-kv', [
      ['mode', w.mode ? (w.mode === 'ap' ? 'access point' : 'station') : null, 'accent'],
      [w.mode === 'ap' ? 'network' : 'joined', w.ssid],
      ['ip', w.ip],
      ['mdns', (get(S.info, 'hostname', null) || 'cartpole') + '.local'],
      w.mode === 'ap' ? null : ['signal', r != null ? bars + esc(r + ' dBm') : '—']
    ].map(function (row) { if (row && row[0] === 'signal') row.html = true; return row; }));
    var wt = $('wifi-tag');
    wt.textContent = w.mode === 'ap' ? 'AP · setup' : w.mode ? 'STA' : '—';
    wt.className = 'tag' + (w.mode === 'ap' ? ' warn' : w.mode ? ' ok' : '');
    S.wifiMode = w.mode;
    setTabAlert('settings', w.mode === 'ap' ? 'warn' : null);
    if (S.tabAuto) { S.tabAuto = false; if (w.mode === 'ap') showTab('settings'); }
    trackJoin(w.join);
    var ssid = $('wifi-ssid');
    if (w.ssid && !ssid.value && document.activeElement !== ssid && w.mode !== 'ap') ssid.placeholder = w.ssid;
  }

  var IDLE_TASKS = /^(IDLE\d?|ipc\d)$/;
  function renderTasks(tasks) {
    var body = $('task-rows'), check = $('layout-check'), tag = $('layout-tag');
    if (!tasks || !tasks.length) return;
    var bad = [];
    tasks.forEach(function (t) {
      if (t.name === 'ctrl' && t.core !== 1) bad.push('ctrl is not pinned to core 1');
      else if (t.name !== 'ctrl' && t.core === 1 && !IDLE_TASKS.test(t.name)) bad.push(t.name + ' shares core 1 with the control loop');
      else if (t.name !== 'ctrl' && t.core === -1 && !IDLE_TASKS.test(t.name)) bad.push(t.name + ' may run on core 1 (unpinned)');
    });
    var hasCtrl = tasks.some(function (t) { return t.name === 'ctrl'; });
    if (!hasCtrl) bad.push('no ctrl task');
    var sorted = tasks.slice().sort(function (a, b) {
      var ca = a.core === -1 ? 2 : a.core, cb = b.core === -1 ? 2 : b.core;
      if (cb !== ca) return ca === 1 ? -1 : cb === 1 ? 1 : ca - cb;
      var ia = IDLE_TASKS.test(a.name), ib = IDLE_TASKS.test(b.name);
      return ia !== ib ? (ia ? 1 : -1) : (b.cpu || 0) - (a.cpu || 0);
    });
    body.innerHTML = sorted.map(function (t) {
      var misplaced = (t.name === 'ctrl' && t.core !== 1) || (t.name !== 'ctrl' && t.core !== 0 && !IDLE_TASKS.test(t.name));
      var cls = (t.name === 'ctrl' ? 'hl' : IDLE_TASKS.test(t.name) ? 'idle' : '') + (misplaced ? ' misplaced' : '');
      var core = t.core === -1 ? 'any' : t.core;
      var cpu = t.cpu == null ? '—' : t.cpu.toFixed(1) + ' %';
      return '<tr class="' + cls + '"><td>' + esc(t.name) + '</td><td><span class="core-badge c' + esc(core) + '">' + esc(core) + '</span></td><td>' + esc(t.prio != null ? t.prio : '—') +
        '</td><td><span class="cpu-cell">' + cpu + '<span class="cpu-bar"><i style="width:' + clamp(t.cpu || 0, 0, 100) + '%"></i></span></span></td><td>' + (t.stack_free != null ? esc(t.stack_free) + ' B' : '—') + '</td></tr>';
    }).join('');
    if (bad.length) {
      check.innerHTML = '<span class="bad">✕</span> ' + esc(bad.join('; '));
      tag.textContent = 'check pinning'; tag.className = 'tag bad';
    } else {
      check.innerHTML = '<span class="ok">✓</span> ctrl runs alone on core 1; Wi-Fi, lwIP, httpd and telemetry stay on core 0.';
      tag.textContent = 'pinning ok'; tag.className = 'tag ok';
    }
  }

  // Health → Memory & link: the status heap figures plus the telemetry this page receives.
  function renderLink() {
    var s = S.status || {}, hp = s.heap || {}, w = s.wifi || {}, L = S.link, live = S.conn === 'live';
    var rate = get(S.info, 'telemetry.rate_hz', null), age = S.statusAt ? (performance.now() - S.statusAt) / 1000 : null;
    var low = live && L.fps != null && rate && L.fps < 0.8 * rate;
    setText('fact-link', live && L.fps != null ? L.fps + ' fps' : '—');
    kv('dev-kv', [
      { head: 'memory' },
      ['internal free', hp.internal_free != null ? bytes(hp.internal_free) : null, '', 'heap_caps_get_free_size(MALLOC_CAP_INTERNAL)'],
      ['lowest since boot', hp.internal_min != null ? bytes(hp.internal_min) : null, hp.internal_min != null && hp.internal_min < 32768 ? 'bad' : ''],
      ['psram free', hp.psram_free != null ? bytes(hp.psram_free) : null],
      { head: 'link' },
      ['wi-fi', w.mode === 'ap' ? 'access point (setup)' : w.rssi != null ? w.rssi + ' dBm' + (w.ssid ? ' · ' + w.ssid : '') : null, w.mode === 'ap' ? 'accent' : w.rssi != null && w.rssi < -75 ? 'bad' : ''],
      ['telemetry', live && L.fps != null ? L.fps + (rate ? ' / ' + rate : '') + ' fps' : live ? '…' : 'not connected', low ? 'bad' : live ? 'good' : ''],
      ['frames lost', live ? L.lost + (L.gaps ? ' in ' + plural(L.gaps, 'gap') : '') : null, L.lost ? 'accent' : ''],
      ['status age', age != null && live ? age.toFixed(1) + ' s' : null, age != null && age > 2 ? 'bad' : '']
    ]);
  }

  // Health → Sensors & drive (rig): encoders with the AS5048A diagnostics, e-stop, drive, zero refs.
  // Also the calibration line under Live → Calibrate and the "last fault" line over the scene.
  var ioPrevBad = null;
  function renderIO(io, cal) {
    var body = $('io-body'), tag = $('io-tag'), s = S.status || {}, f = s.faults || {};
    // Last fault (any backend), while not in one
    var ago = f.ago_s != null ? f.ago_s + (performance.now() - S.statusAt) / 1000 : null;
    setText('run-last', S.conn === 'live' && f.count && S.run !== 'fault' ? 'last fault: ' + (f.last || '?') + (ago != null ? ' · ' + duration(ago) + ' ago' : '') + (f.count > 1 ? ' · ' + f.count + ' since boot' : '') : '');
    if (!io) return;
    if (!io.rig) {
      if (body.dataset.mode !== 'sim') {
        body.dataset.mode = 'sim';
        body.innerHTML = '<p class="panel-hint">The simulated plant has no encoders, e-stop or drive. On the rig backend this card shows the joint encoders (with the magnet diagnostics), the cart encoder, the e-stop input, the drive and the zero references.</p>';
      }
      tag.textContent = 'sim'; tag.className = 'tag';
      S.ioBad = false;
      return;
    }
    body.dataset.mode = 'rig';
    var problems = [], warns = [], enc = io.enc || [], names = ['joint 1 · θ₁', 'joint 2 · φ = θ₂ − θ₁'];
    if (io.estop) problems.push('e-stop pressed');
    if (io.read_errors) problems.push(io.read_errors + ' bad reads in a row');
    var rows = enc.map(function (e, i) {
      var mag = 'not read yet', cls = '';
      if (e.agc != null) {
        var issues = [];
        if (e.comp_high) issues.push('field too weak (magnet too far)');
        if (e.comp_low) issues.push('field too strong (magnet too close)');
        if (e.cof) issues.push('CORDIC overflow');
        if (e.ocf === false) issues.push('offset compensation not finished');
        mag = issues.length ? issues.join(', ') : 'ok';
        cls = issues.length ? 'bad' : e.agc > 220 || e.agc < 20 ? 'accent' : 'good';
        mag += ' · AGC ' + e.agc;
        if (issues.length) problems.push('encoder ' + (i + 1) + ': ' + issues.join(', '));
        else if (cls === 'accent') warns.push('encoder ' + (i + 1) + ': AGC ' + e.agc + ' near its limit');
      }
      var grew = ioPrevBad && e.bad > ioPrevBad[i];
      if (grew) warns.push('encoder ' + (i + 1) + ': bad reads rising');
      return '<tr><td>' + esc(names[i] || 'enc ' + (i + 1)) + '</td><td>' + e.raw + '</td><td>' + num(e.raw * 360 / 16384, 1) + '°</td>' +
        '<td class="' + cls + '">' + esc(mag) + '</td><td class="' + (grew ? 'bad' : e.bad ? 'accent' : '') + '">' + e.bad + '</td></tr>';
    });
    ioPrevBad = enc.map(function (e) { return e.bad; });
    cal = cal || {};
    if (!cal.angles) problems.push('angles not zeroed');
    if (!cal.cart) problems.push('cart not zeroed');
    body.innerHTML =
      '<div class="table-wrap"><table class="tasks io-table" aria-label="Joint encoders"><thead><tr><th scope="col">encoder</th><th scope="col">raw</th><th scope="col">raw angle</th><th scope="col">magnet</th><th scope="col">bad reads</th></tr></thead><tbody>' + rows.join('') + '</tbody></table></div>' +
      '<dl class="kv">' + [
        ['e-stop', io.estop ? 'PRESSED' : 'released', io.estop ? 'bad' : 'good'],
        ['drive', io.drive ? 'enabled · ' + Math.round(100 * (io.duty || 0)) + ' % ' + (io.dir < 0 ? '−x' : '+x') : 'disabled (EN low)', io.drive ? 'accent' : ''],
        ['cart encoder', io.cart_count + ' counts'],
        { head: 'zero references' },
        ['angles', cal.angles ? 'set' + (cal.angles_boot ? ' on boot #' + cal.angles_boot : '') + (cal.off1 != null ? ' · ' + num(cal.off1, 4) + ', ' + num(cal.off2, 4) + ' rad' : '') : 'not set', cal.angles ? 'good' : 'bad'],
        ['cart', cal.cart ? 'set' + (cal.cart_boot ? ' on boot #' + cal.cart_boot : '') + (cal.x_zero != null ? ' · ' + num(cal.x_zero, 4) + ' m' : '') : 'not set', cal.cart ? 'good' : 'bad']
      ].map(function (r) {
        return r.head ? '<span class="kv-head">' + esc(r.head) + '</span>' : '<dt>' + esc(r[0]) + '</dt><dd class="' + (r[2] || '') + '">' + esc(r[1]) + '</dd>';
      }).join('') + '</dl>' +
      (problems.length || warns.length ? '<p class="check-line"><span class="' + (problems.length ? 'bad' : 'warn') + '">' + (problems.length ? '✕' : '!') + '</span> ' + esc(problems.concat(warns).join('; ')) + '</p>'
        : '<p class="check-line"><span class="ok">✓</span> Encoders read cleanly, the magnets are in range, and both zero references are set.</p>');
    tag.textContent = io.estop ? 'e-stop' : problems.length ? (cal.angles && cal.cart ? 'check sensors' : 'not zeroed') : warns.length ? 'watch' : 'ok';
    tag.className = 'tag ' + (problems.length ? 'bad' : warns.length ? 'warn' : 'ok');
    S.ioBad = problems.length > 0;
    // Live → Calibrate
    var cl = $('cal-line');
    cl.textContent = cal.angles && cal.cart ? 'Zeroed: angles' + (cal.angles_boot ? ' on boot #' + cal.angles_boot : '') + ', cart' + (cal.cart_boot ? ' on boot #' + cal.cart_boot : '') + '. Re-zero after moving a magnet or the belt.'
      : (!cal.angles && !cal.cart ? 'Neither zero reference is set' : !cal.angles ? 'The angles are not zeroed' : 'The cart is not zeroed') + ': arming is refused until both are.';
    cl.className = 'panel-hint' + (cal.angles && cal.cart ? '' : ' bad');
  }

  function healthAlert() {
    setTabAlert('health', S.events.newBad || S.ioBad ? 'bad' : S.healthWarn ? 'warn' : null);
  }

  function clock(ms) {
    var t = Math.max(0, Math.floor(ms / 1000)), h = Math.floor(t / 3600), m = Math.floor(t / 60) % 60, x = t % 60;
    return (h ? h + ':' + (m < 10 ? '0' : '') : '') + m + ':' + (x < 10 ? '0' : '') + x;
  }
  function renderEvents() {
    var E = S.events, list = $('events');
    if (!E.list.length) { list.innerHTML = '<li class="muted">Nothing yet.</li>'; return; }
    // Newest first, grouped by boot: entries from before a reset (crash, watchdog, brownout,
    // reboot) survive in the device's RTC memory and explain what happened.
    var html = [], prevBoot = null, multi = E.list.some(function (e) { return e.boot != null && e.boot !== E.boot; });
    E.list.slice().reverse().forEach(function (e) {
      if (multi && e.boot !== prevBoot) {
        html.push('<li class="ev-sep">' + (e.boot === E.boot ? 'this boot' : 'before the reset') + (e.boot != null ? ' · boot #' + e.boot : '') + '</li>');
        prevBoot = e.boot;
      }
      html.push('<li class="ev ev-' + esc(e.level) + (multi && e.boot !== E.boot ? ' ev-prev' : '') + '"><time title="since that boot">' + clock(e.t_ms) + '</time><span>' + esc(e.msg) + '</span></li>');
    });
    list.innerHTML = html.join('');
    var nBad = E.list.filter(function (e) { return e.level === 'error'; }).length, nWarn = E.list.filter(function (e) { return e.level === 'warn'; }).length;
    var tag = $('log-tag');
    tag.textContent = nBad ? plural(nBad, 'error') : nWarn ? plural(nWarn, 'warning') : 'no errors';
    tag.className = 'tag ' + (nBad ? 'bad' : nWarn ? 'warn' : 'ok');
    healthAlert();
  }

  // Settings → Model: does this page's forward pass of the device's blob give the device's command?
  function renderParity() {
    var e = $('model-check');
    if (!S.policy) { e.innerHTML = ''; return; }
    if (S.parity == null) {
      e.innerHTML = '<span class="muted">○</span> Browser check: waiting. Arm in AI (no excitation) and this page compares its own forward pass of the blob with the command the device sends.';
      return;
    }
    var fresh = performance.now() - S.parityAt < 2000, ok = S.parity < 0.05;
    e.innerHTML = (ok ? '<span class="ok">✓</span> The browser\'s forward pass matches the device' : '<span class="bad">✕</span> The browser and the device disagree (blob, observation layout or action pipeline)') +
      ': mean <span class="nowrap">|ΔF| ' + S.parity.toFixed(3) + ' N</span> over ' + S.parityN.toLocaleString('en-US') + ' ticks' + (fresh ? '.' : ' (last armed run).');
  }

  // Live → time to target: only while the AI drives; counts up until the goal is reached.
  function renderReach() {
    var live = S.conn === 'live', seg = S.seg, f = S.last, G = geom();
    var show = live && S.run === 'armed' && S.mode === 'ai' && seg && f;
    $('reach').hidden = !show;
    if (!show) return;
    var settled = f.flags & FLAG.SETTLED, val, lbl, state, running = false;
    if (seg.from && seg.from === S.goal) {
      val = '—'; lbl = 'holding at ' + S.goal; state = settled ? 'holding' : 'recovering';
    } else if (seg.reach != null) {
      val = seg.reach.toFixed(2); lbl = (seg.from ? seg.from + ' ' : '') + '→ ' + S.goal + ' (s)'; state = settled ? 'holding' : 'knocked out · recovering';
    } else {
      val = ((f.tick - seg.tick0) * G.dt).toFixed(1); lbl = (seg.from ? seg.from + ' ' : '') + '→ ' + S.goal + ' (s)'; state = 'on its way'; running = true;
    }
    setText('reach-val', val); setText('reach-lbl', lbl); setText('reach-state', state);
    $('reach-val').classList.toggle('running', running);
  }

  // Live → State: estimate / encoder readings / true state, refreshed ~10 Hz.
  var roKey = '';
  function renderReadout() {
    var f = S.last, G = geom(), n = G.n, truth = !!(f && (f.flags & FLAG.TRUTH)), body = $('ro-rows');
    var key = n + '|' + truth;
    if (key !== roKey) {
      roKey = key;
      var sub = '₁₂₃₄', rows = [['x', 'm'], ['ẋ', 'm/s']];
      for (var k = 0; k < n; k++) rows.push(['θ' + sub[k], '°'], ['θ̇' + sub[k], '°/s']);
      body.innerHTML = rows.map(function (r) {
        return '<tr><th scope="row">' + r[0] + ' <span>' + r[1] + '</span></th><td></td><td></td><td class="ro-true"></td></tr>';
      }).join('');
      $('ro-true-h').hidden = !truth;
      body.querySelectorAll('.ro-true').forEach(function (td) { td.hidden = !truth; });
    }
    var cells = body.querySelectorAll('td'), D = 180 / Math.PI;
    function put(i, v, d) { var c = cells[i]; if (c) { var t = v == null || !isFinite(v) ? '—' : num(v, d); if (c.textContent !== t) c.textContent = t; } }
    // Rows: x, ẋ, then θₖ (°, as the plot lanes) and θ̇ₖ (°/s); meas holds positions only: [x, θ₁, …].
    function val(v, r) { return r < 2 ? v : r % 2 === 0 ? lane(v) : v * D; }
    for (var r = 0; r < 2 + 2 * n; r++) {
      var ang = r >= 2 && r % 2 === 0, d = r === 0 ? 3 : r === 1 ? 2 : ang ? 1 : 0;
      var mi = r === 0 ? 0 : ang ? 1 + (r - 2) / 2 : -1;
      put(3 * r, f ? val(f.est[r], r) : null, d);
      put(3 * r + 1, f && mi >= 0 ? val(f.meas[mi], r) : null, d);
      put(3 * r + 2, f && truth ? val(f.truth[r], r) : null, d);
    }
    setText('ro-tick', f ? 'tick ' + f.tick : '');
    var margin = f ? G.x_lim - Math.abs(f.est[0]) : null;
    kv('ro-kv', [
      ['force cmd', f ? signed(f.cmd, 2) + ' N' : null],
      ['force applied', f ? signed(f.applied, 2) + ' N' : null],
      ['rail margin', margin != null ? num(margin, 3) + ' m' : null, margin != null && margin < 0.1 * G.x_lim ? 'bad' : ''],
      ['settled', f ? (f.flags & FLAG.SETTLED ? 'yes · at ' + (S.goal || '?') : 'no') : null, f && (f.flags & FLAG.SETTLED) ? 'good' : '']
    ]);
  }

  /* ---- Controls ---- */
  // Pictogram of a target pose (swingup.js goalIcon): cart on a rail, link 1 then link 2.
  function goalIcon(label) {
    var cx = 16, cy = 26, L = 12, x = cx, y = cy, tilt = -0.1;
    var out = '<svg viewBox="0 0 32 52" focusable="false">' +
      '<line class="gi-rail" x1="2" y1="' + (cy + 3) + '" x2="30" y2="' + (cy + 3) + '"/>' +
      '<rect class="gi-cart" x="11" y="' + (cy - 2.5) + '" width="10" height="5" rx="1.5"/>';
    var Lk = L * 2 / Math.max(2, label.length);
    for (var k = 0; k < label.length; k++) {
      if (k > 0) tilt += label[k] === label[k - 1] ? 0.4 : 0.8;
      var th = (label[k] === 'D' ? Math.PI : 0) + tilt;
      var nx = x + Lk * Math.sin(th), ny = y - Lk * Math.cos(th);
      out += '<line class="gi-rod" x1="' + x.toFixed(1) + '" y1="' + y.toFixed(1) + '" x2="' + nx.toFixed(1) + '" y2="' + ny.toFixed(1) + '"/>' +
             '<circle class="gi-bob-' + (k % 2) + '" cx="' + nx.toFixed(1) + '" cy="' + ny.toFixed(1) + '" r="3"/>';
      x = nx; y = ny;
    }
    return out + '</svg>';
  }
  var GOAL_TITLES = { UU: 'Both up', UD: 'θ₁ up, θ₂ down', DU: 'θ₁ down, θ₂ up', DD: 'Both down' };
  var goalKey = '';
  function buildGoalButtons() {
    var labels = goalLabels(), key = labels.join(',');
    if (key === goalKey) return;
    goalKey = key;
    // 2 links: the site's 2 × 2 (rows: link 1 up, down; columns: link 2 up, down).
    var order = labels.length === 4 ? ['UU', 'UD', 'DU', 'DD'] : labels;
    var grid = $('goal-grid'), rnd = $('goal-random');
    Array.prototype.slice.call(grid.querySelectorAll('[data-goal]')).forEach(function (b) { b.remove(); });
    order.forEach(function (l) {
      var b = document.createElement('button');
      b.type = 'button'; b.className = 'goal-btn'; b.dataset.goal = l;
      b.title = GOAL_TITLES[l] || l; b.setAttribute('aria-label', l + ': ' + (GOAL_TITLES[l] || l)); b.setAttribute('aria-pressed', 'false');
      b.innerHTML = '<span class="goal-icon" aria-hidden="true">' + goalIcon(l) + '</span><span class="goal-label">' + l + '</span>';
      b.addEventListener('click', function () {
        S.pendingGoal = l; S.pendingUntil = performance.now() + 1500; renderControls(true);
        send({ goal: l }).catch(function () { S.pendingGoal = null; renderControls(true); });
      });
      grid.insertBefore(b, rnd);
    });
    // Sim: restart the plant at any pose (hanging first).
    var row = $('pose-row');
    row.innerHTML = '';
    var poses = labels.slice().sort(function (a, b) { return (b.split('D').length) - (a.split('D').length); });
    poses.forEach(function (l) {
      var b = document.createElement('button');
      b.type = 'button'; b.className = 'goal-btn'; b.dataset.pose = l;
      b.title = 'Restart the simulated plant at ' + l + (l.indexOf('U') < 0 ? ' (hanging)' : '');
      b.innerHTML = '<span class="goal-icon" aria-hidden="true">' + goalIcon(l) + '</span><span class="goal-label">' + (l.indexOf('U') < 0 ? 'hang' : l) + '</span>';
      b.addEventListener('click', function () {
        send({ reset: l }).then(function () { if (S.seg) S.seg.restart = true; flash('cmd-msg', 'Plant reset to ' + l + '.', 'good', 2000); }).catch(function () {});
      });
      row.appendChild(b);
    });
  }

  var ctrlKey = '';
  function renderControls(force) {
    var live = S.conn === 'live', run = S.run, armed = run === 'armed';
    var noModel = S.model.state === 'none' || get(S.info, 'model', 1) === null;
    var key = [live, run, S.mode, S.goal, S.random, S.fault, S.pendingGoal, noModel, S.backend, S.model.state, S.seg && S.seg.reach, S.last && (S.last.flags & FLAG.SETTLED)].join('|');
    if (!force && key === ctrlKey) return;
    ctrlKey = key;

    // Arm (Live and Record carry one each)
    document.querySelectorAll('[data-arm]').forEach(function (arm) {
      arm.classList.toggle('armed', armed);
      arm.classList.toggle('fault', run === 'fault');
      arm.setAttribute('aria-pressed', armed ? 'true' : 'false');
      arm.disabled = !live || run === 'maint' || run == null;
      arm.querySelector('.arm-label').textContent = armed ? 'Armed · disarm' : run === 'fault' ? 'Re-arm' : run === 'maint' ? 'Writing flash…' : 'Arm';
    });
    // Header: the controller state, on every tab
    var rc = $('run-chip');
    rc.dataset.run = live ? run || '' : '';
    setText('run-chip-text', !live ? '—' : (run || '—').toUpperCase() + (run === 'fault' && S.fault ? ' · ' + S.fault : ''));
    rc.title = armed ? 'Armed: click (or Esc) to disarm' : run === 'fault' ? 'Fault: ' + (S.fault || 'tripped') + '. Re-arm on Live.' : 'Controller state. Esc disarms from any tab.';
    setText('rec-ctrl-note', !live ? '' : run === 'fault' ? 'fault · ' + (S.fault || 'tripped') : (S.mode === 'manual' ? 'Manual · PD holds the cart at ' + signed(S.xref, 2) + ' m' : 'AI → ' + (S.goal || '?')) + (armed ? '' : ' · not driving'));

    // Mode
    document.querySelectorAll('.mode-btn[data-mode]').forEach(function (b) {
      var on = b.dataset.mode === S.mode;
      b.classList.toggle('active', on); b.setAttribute('aria-pressed', on ? 'true' : 'false');
      b.disabled = !live || (b.dataset.mode === 'ai' && noModel);
      b.title = b.dataset.mode === 'ai' && noModel ? 'No model on the device' : '';
    });
    $('mode-hint').innerHTML = S.mode === 'manual'
      ? '<strong>Manual:</strong> a PD loop (≤ 10 N) drives the cart to the slider position — jog it, or nudge the links.'
      : '<strong>AI:</strong> the network drives to the target and balances around the slider position.';

    // Goals
    document.querySelectorAll('.goal-btn[data-goal]').forEach(function (b) {
      var on = b.dataset.goal === S.goal;
      b.classList.toggle('active', on && !S.pendingGoal);
      b.classList.toggle('pending', b.dataset.goal === S.pendingGoal);
      b.setAttribute('aria-pressed', on ? 'true' : 'false');
      b.disabled = !live;
    });
    var rnd = $('goal-random');
    rnd.classList.toggle('active', !!S.random); rnd.setAttribute('aria-pressed', S.random ? 'true' : 'false'); rnd.disabled = !live;
    document.querySelectorAll('.goal-btn[data-pose]').forEach(function (b) { b.disabled = !live; });
    $('zero-angles').disabled = $('zero-cart').disabled = !live || armed;
    $('xref').disabled = !live;
    if (!dragging) syncSlider();

    if (pending && !uploading && !pending.done) renderPending();
    dz.classList.toggle('locked', armed);
    renderRecord();

    // Overlay: run state + time to target
    var badge = $('run-badge');
    badge.dataset.run = live ? run || 'idle' : 'offline';
    setText('run-text', live ? (run || '—').toUpperCase() : S.conn === 'rebooting' ? 'REBOOTING' : S.everLive ? 'OFFLINE' : 'CONNECTING');
    setText('run-reason', live && run === 'fault' ? ' · ' + (S.fault || 'tripped') : live && run === 'maint' ? ' · writing flash' : '');
    var sub = '';
    if (live && S.goal) {
      sub = S.mode === 'manual' ? 'Manual → cart to ' + signed(S.xref, 2) + ' m' : 'AI → ' + esc(S.goal) + (Math.abs(S.xref) > 0.004 ? ' at ' + signed(S.xref, 2) + ' m' : '');
      if (S.random && S.mode !== 'manual') sub += ' · random';
      if (!armed) sub += ' <span class="muted">(not driving)</span>';
    }
    var se = $('run-sub');
    if (se.innerHTML !== sub) se.innerHTML = sub;
    renderReach();
  }

  // Arm / Disarm. Esc disarms from anywhere on the page.
  document.querySelectorAll('[data-arm]').forEach(function (b) {
    b.addEventListener('click', function () { send({ arm: S.run !== 'armed' }).catch(function () {}); });
  });
  $('run-chip').addEventListener('click', function () {
    if (S.run === 'armed') { send({ arm: false }).catch(function () {}); flash('cmd-msg', 'Disarmed.', 'good', 2000); }
    else if (S.tab !== 'live') location.hash = 'live';
  });
  document.addEventListener('keydown', function (e) {
    if (e.key === 'Escape' && S.run === 'armed') { send({ arm: false }).catch(function () {}); flash('cmd-msg', 'Disarmed (Esc).', 'good', 2000); }
  });
  document.querySelectorAll('.mode-btn[data-mode]').forEach(function (b) {
    b.addEventListener('click', function () { if (b.dataset.mode !== S.mode) send({ mode: b.dataset.mode }).catch(function () {}); });
  });
  $('goal-random').addEventListener('click', function () { send({ random: !S.random }).catch(function () {}); });

  // x_ref: throttled to ~20 Hz while dragging, the final value on release; double-click centres.
  var slider = $('xref'), dragging = false, lastSent = 0, sendTimer = null, releaseTimer = null;
  function syncSlider() {
    if (S.xref == null) return;
    slider.value = S.xref;
    setText('xref-val', signed(S.xref, 2) + ' m');
  }
  function sendXref(final) {
    var v = +(+slider.value).toFixed(2), now = performance.now();
    setText('xref-val', signed(v, 2) + ' m');
    clearTimeout(sendTimer);
    if (final || now - lastSent >= 50) { lastSent = now; send({ x_ref: v }).catch(function () {}); }
    else sendTimer = setTimeout(function () { sendXref(false); }, 50 - (now - lastSent));
  }
  slider.addEventListener('input', function () { dragging = true; clearTimeout(releaseTimer); sendXref(false); });
  slider.addEventListener('change', function () {
    sendXref(true);
    clearTimeout(releaseTimer);
    releaseTimer = setTimeout(function () { dragging = false; }, 400);   // let the device echo it back first
  });
  slider.addEventListener('dblclick', function () { slider.value = 0; dragging = true; sendXref(true); releaseTimer = setTimeout(function () { dragging = false; }, 400); });

  function zero(what, question) {
    if (S.run === 'armed') { flash('cmd-msg', 'Disarm first.', 'bad'); return; }
    if (!window.confirm(question)) return;
    send({ zero: what }).then(function () { flash('cmd-msg', what === 'angles' ? 'Angles zeroed: θ₁ = π, φ = 0.' : 'Cart zeroed: x = 0 here.', 'good'); }).catch(function () {});
  }
  $('zero-angles').addEventListener('click', function () {
    zero('angles', 'Zero the angle encoders?\n\nBoth links must hang straight down and be at rest: this sets θ₁ = π and the joint-2 angle φ = 0.');
  });
  $('zero-cart').addEventListener('click', function () {
    zero('cart', 'Zero the cart encoder?\n\nThe cart\'s current position becomes x = 0 (the centre of the rail): move it to the middle first.');
  });

  /* ================================================================
     Drawing (swingup.js drawScene / drawPlot / drawNet)
  ================================================================ */
  if (!CanvasRenderingContext2D.prototype.roundRect) {
    CanvasRenderingContext2D.prototype.roundRect = function (x, y, w, h, r) {
      var rad = Math.min(r, Math.min(w, h) / 2);
      this.moveTo(x + rad, y);
      this.arcTo(x + w, y, x + w, y + h, rad); this.arcTo(x + w, y + h, x, y + h, rad);
      this.arcTo(x, y + h, x, y, rad); this.arcTo(x, y, x + w, y, rad);
      this.closePath();
    };
  }
  // The CSS box sets each canvas's size; only the pixel buffer is sized here (never style.height).
  function sizeCanvas(cv) {
    var dpr = Math.min(window.devicePixelRatio || 1, 3);
    var W = cv.clientWidth || 300, H = cv.clientHeight || 150;
    if (cv.width !== Math.round(W * dpr) || cv.height !== Math.round(H * dpr)) { cv.width = Math.round(W * dpr); cv.height = Math.round(H * dpr); }
    var ctx = cv.getContext('2d');
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    return { ctx: ctx, W: W, H: H };
  }
  var rgbCache = {};
  function rgb(name) {
    var c = cssVar(name);
    if (!rgbCache[c]) {
      var m = /^#([0-9a-f]{6})$/i.exec(c), n = m ? parseInt(m[1], 16) : 0x888888;
      rgbCache[c] = [n >> 16, (n >> 8) & 255, n & 255];
    }
    return rgbCache[c];
  }
  function mix(a, b, f) {
    return 'rgb(' + Math.round(a[0] + (b[0] - a[0]) * f) + ',' + Math.round(a[1] + (b[1] - a[1]) * f) + ',' + Math.round(a[2] + (b[2] - a[2]) * f) + ')';
  }
  function isDark() {
    var t = root.getAttribute('data-theme');
    return t ? t === 'dark' : window.matchMedia('(prefers-color-scheme: dark)').matches;
  }

  // State/units panel, lower right (personal-site statebox.js): decimal points stacked.
  function drawStateBox(ctx, W, H, rows) {
    var LH = 14, PAD_X = 8, GAP = 6;
    ctx.save();
    ctx.font = '12px ' + cssVar('--mono');
    var parts = rows.map(function (r) {
      var s = num(r.value, 1), dot = s.indexOf('.');
      return { label: r.label, int: dot < 0 ? s : s.slice(0, dot), frac: dot < 0 ? '' : s.slice(dot), unit: r.unit || '' };
    });
    function widest(k) { return Math.max.apply(null, parts.map(function (p) { return ctx.measureText(p[k]).width; })); }
    var intW = ctx.measureText('−').width + Math.max.apply(null, parts.map(function (p) { return ctx.measureText(p.int.replace('−', '')).width; }));
    var labelW = widest('label'), fracW = widest('frac'), unitW = widest('unit');
    var bw = Math.ceil(PAD_X * 2 + labelW + GAP + intW + fracW + (unitW ? GAP + unitW : 0)), bh = rows.length * LH + 10;
    var bx = W - bw - 10, by = H - bh - 10;
    ctx.fillStyle = isDark() ? 'rgba(11,15,20,0.72)' : 'rgba(245,246,248,0.82)';
    ctx.beginPath(); ctx.roundRect(bx, by, bw, bh, 3); ctx.fill();
    ctx.strokeStyle = cssVar('--border'); ctx.lineWidth = 1; ctx.stroke();
    var xL = bx + PAD_X, xP = xL + labelW + GAP + intW, xU = xP + fracW + GAP;
    ctx.textBaseline = 'top'; ctx.fillStyle = cssVar('--text-muted');
    parts.forEach(function (p, i) {
      var y = by + 6 + i * LH;
      ctx.textAlign = 'left'; ctx.fillText(p.label, xL, y);
      ctx.textAlign = 'right'; ctx.fillText(p.int, xP, y);
      ctx.textAlign = 'left'; ctx.fillText(p.frac, xP, y);
      if (p.unit) ctx.fillText(p.unit, xU, y);
    });
    ctx.restore();
  }

  var sceneCv = $('scene'), plotCv = $('plot'), netCv = $('net'), sparkCv = $('spark');

  function drawScene() {
    var c = sizeCanvas(sceneCv), ctx = c.ctx, W = c.W, H = c.H;
    var bg = cssVar('--bg-soft'), border = cssVar('--border'), strong = cssVar('--border-strong');
    var muted = cssVar('--text-muted'), mono = cssVar('--mono'), font = cssVar('--font'), lime = cssVar('--lime');
    var col1 = cssVar('--pend-1'), col2 = cssVar('--pend-2'), colF = cssVar('--viz-4'), colCart = cssVar('--pend-cart'), colRod = cssVar('--pend-rod');
    ctx.fillStyle = bg; ctx.fillRect(0, 0, W, H);
    var f = S.last, G = geom();
    if (!f) {
      ctx.fillStyle = muted; ctx.font = '13px ' + font; ctx.textAlign = 'center'; ctx.textBaseline = 'middle';
      ctx.fillText(S.conn === 'rebooting' ? 'Rebooting…' : S.conn === 'live' ? 'Waiting for telemetry…' : 'Connecting to the device…', W / 2, H / 2);
      return;
    }
    var stale = performance.now() - S.lastFrameSeen > 1500 || S.conn !== 'live';
    var reach = G.lengths.reduce(function (a, b) { return a + b; }, 0), cartHalf = 0.08;
    var scale = Math.min((W - 40) / (2 * (G.x_lim + cartHalf) + 0.1), (H - 70) / (2 * reach + 0.04));
    var cx = W / 2, trackY = Math.round(H / 2 + 14);
    function sx(xm) { return cx + xm * scale; }
    if (stale) ctx.globalAlpha = 0.45;

    // Rail with hard stops where the cart centre reaches ±x_lim, ticks every 0.25 m.
    var railL = sx(-G.x_lim - cartHalf), railR = sx(G.x_lim + cartHalf);
    ctx.strokeStyle = strong; ctx.lineWidth = 4; ctx.lineCap = 'round';
    ctx.beginPath(); ctx.moveTo(railL, trackY); ctx.lineTo(railR, trackY); ctx.stroke();
    ctx.lineCap = 'butt'; ctx.fillStyle = muted;
    [railL, railR].forEach(function (xp) { ctx.fillRect(xp - 3, trackY - 14, 6, 28); });
    ctx.font = '10px ' + mono; ctx.textAlign = 'center'; ctx.textBaseline = 'alphabetic';
    for (var m = -Math.floor(G.x_lim / 0.25) * 0.25; m <= G.x_lim + 1e-9; m += 0.25) {
      ctx.strokeStyle = border; ctx.lineWidth = 1;
      ctx.beginPath(); ctx.moveTo(sx(m), trackY + 8); ctx.lineTo(sx(m), trackY + 14); ctx.stroke();
      ctx.fillStyle = muted; ctx.globalAlpha = (stale ? 0.45 : 1) * 0.7;
      ctx.fillText(minus(m.toFixed(2)), sx(m), trackY + 26);
      ctx.globalAlpha = stale ? 0.45 : 1;
    }

    // x_ref: a lime marker on the rail (where Manual drives the cart, or the AI balances).
    var xr = sx(clamp(f.x_ref, -G.x_lim, G.x_lim));
    ctx.strokeStyle = lime; ctx.fillStyle = lime; ctx.lineWidth = 1.5;
    ctx.setLineDash([3, 3]); ctx.globalAlpha = (stale ? 0.45 : 1) * 0.55;
    ctx.beginPath(); ctx.moveTo(xr, trackY - 4); ctx.lineTo(xr, trackY - 22); ctx.stroke();
    ctx.setLineDash([]); ctx.globalAlpha = stale ? 0.45 : 1;
    ctx.beginPath(); ctx.moveTo(xr, trackY + 3); ctx.lineTo(xr - 6, trackY + 12); ctx.lineTo(xr + 6, trackY + 12); ctx.closePath(); ctx.fill();
    ctx.font = '10px ' + mono; ctx.fillStyle = cssVar('--lime-text');
    ctx.fillText('x_ref', xr, trackY - 26);

    var st = f.est, cartX = sx(st[0]);
    function bobR(k) { return Math.max(6, 0.028 * scale * Math.sqrt((G.masses[k] || 0.15) / 0.15)); }

    // The simulator's true state: a faint ghost behind the estimate.
    if (f.flags & FLAG.TRUTH) {
      var tx = sx(f.truth[0]), ty = trackY;
      ctx.globalAlpha = (stale ? 0.45 : 1) * 0.22;
      ctx.fillStyle = colCart;
      ctx.beginPath(); ctx.roundRect(tx - cartHalf * scale, trackY - Math.max(16, 0.07 * scale) / 2, 2 * cartHalf * scale, Math.max(16, 0.07 * scale), 3); ctx.fill();
      for (var q = 0; q < G.n; q++) {
        var th0 = f.truth[2 + 2 * q], nx = tx + G.lengths[q] * scale * Math.sin(th0), ny = ty - G.lengths[q] * scale * Math.cos(th0);
        ctx.strokeStyle = colRod; ctx.lineWidth = 3; ctx.lineCap = 'round';
        ctx.beginPath(); ctx.moveTo(tx, ty); ctx.lineTo(nx, ny); ctx.stroke();
        ctx.fillStyle = q % 2 ? col2 : col1;
        ctx.beginPath(); ctx.arc(nx, ny, bobR(q), 0, 2 * Math.PI); ctx.fill();
        tx = nx; ty = ny;
      }
      ctx.lineCap = 'butt'; ctx.globalAlpha = stale ? 0.45 : 1;
    }

    // Target pose, hung from the cart: dashed rods and hollow bobs.
    var target = goalTarget(G.n, f.goal), gx = cartX, gy = trackY;
    ctx.globalAlpha = (stale ? 0.45 : 1) * 0.45; ctx.setLineDash([4, 4]); ctx.lineWidth = 1.5;
    for (var gk = 0; gk < G.n; gk++) {
      var gnx = gx + G.lengths[gk] * scale * Math.sin(target[gk]), gny = gy - G.lengths[gk] * scale * Math.cos(target[gk]);
      ctx.strokeStyle = colRod;
      ctx.beginPath(); ctx.moveTo(gx, gy); ctx.lineTo(gnx, gny); ctx.stroke();
      ctx.setLineDash([]); ctx.strokeStyle = gk % 2 ? col2 : col1;
      ctx.beginPath(); ctx.arc(gnx, gny, bobR(gk) + 2, 0, 2 * Math.PI); ctx.stroke();
      ctx.setLineDash([4, 4]);
      gx = gnx; gy = gny;
    }
    ctx.setLineDash([]); ctx.globalAlpha = stale ? 0.45 : 1;

    // Cart
    var cw = 2 * cartHalf * scale, ch = Math.max(16, 0.07 * scale);
    ctx.fillStyle = colCart;
    ctx.beginPath(); ctx.roundRect(cartX - cw / 2, trackY - ch / 2, cw, ch, 3); ctx.fill();

    // Links (bob x = x + l sin θ, height = l cos θ)
    var px = cartX, py = trackY;
    for (var k = 0; k < G.n; k++) {
      var th = st[2 + 2 * k], len = G.lengths[k] * scale;
      var qx = px + len * Math.sin(th), qy = py - len * Math.cos(th);
      ctx.strokeStyle = colRod; ctx.lineWidth = 4; ctx.lineCap = 'round';
      ctx.beginPath(); ctx.moveTo(px, py); ctx.lineTo(qx, qy); ctx.stroke();
      ctx.fillStyle = k % 2 ? col2 : col1;
      ctx.beginPath(); ctx.arc(qx, qy, bobR(k), 0, 2 * Math.PI); ctx.fill();
      ctx.fillStyle = bg;
      ctx.beginPath(); ctx.arc(px, py, 3, 0, 2 * Math.PI); ctx.fill();
      px = qx; py = qy;
    }
    ctx.lineCap = 'butt';

    // Applied force
    if (Math.abs(f.applied) > 0.2) {
      var fl = f.applied / G.fmax * 60, ay = trackY + 40, fs = fl >= 0 ? 1 : -1;
      ctx.strokeStyle = colF; ctx.fillStyle = colF; ctx.lineWidth = 3;
      ctx.beginPath(); ctx.moveTo(cartX, ay); ctx.lineTo(cartX + fl, ay); ctx.stroke();
      ctx.beginPath(); ctx.moveTo(cartX + fl + 3 * fs, ay); ctx.lineTo(cartX + fl - 6 * fs, ay - 5); ctx.lineTo(cartX + fl - 6 * fs, ay + 5); ctx.closePath(); ctx.fill();
    }
    ctx.globalAlpha = 1;

    // Legend, lower left (wide canvases only: on a phone the state box needs the corner)
    if (W >= 600) {
    ctx.font = '10px ' + mono; ctx.textAlign = 'left'; ctx.textBaseline = 'alphabetic'; ctx.fillStyle = muted;
    var ly = H - 14, lx = 14;
    function key(draw, text) { draw(lx, ly - 4); lx += 22; ctx.fillStyle = muted; ctx.fillText(text, lx, ly); lx += ctx.measureText(text).width + 16; }
    key(function (x, y) { ctx.fillStyle = col1; ctx.beginPath(); ctx.arc(x + 8, y, 4, 0, 2 * Math.PI); ctx.fill(); }, 'estimate');
    key(function (x, y) { ctx.strokeStyle = col1; ctx.globalAlpha = 0.6; ctx.lineWidth = 1.5; ctx.beginPath(); ctx.arc(x + 8, y, 5, 0, 2 * Math.PI); ctx.stroke(); ctx.globalAlpha = 1; }, 'target');
    if (f.flags & FLAG.TRUTH) key(function (x, y) { ctx.fillStyle = col1; ctx.globalAlpha = 0.28; ctx.beginPath(); ctx.arc(x + 8, y, 4, 0, 2 * Math.PI); ctx.fill(); ctx.globalAlpha = 1; }, 'true state');
    if (stale) { ctx.fillStyle = cssVar('--bad'); ctx.fillText('stale', lx, ly); }
    }

    var rows = [{ label: 'x', value: st[0], unit: 'm' }, { label: 'ẋ', value: st[1], unit: 'm/s' }];
    for (var li = 0; li < G.n; li++) rows.push({ label: 'θ' + (li + 1), value: lane(st[2 + 2 * li]), unit: '°' });
    rows.push({ label: 'F', value: f.applied, unit: 'N' });
    drawStateBox(ctx, W, H, rows);
  }

  // Hover: a crosshair snaps to the nearest sample; the margin values follow it.
  var recPlotCv = $('rec-plot');
  [plotCv, recPlotCv].forEach(function (cv) {
    cv.hoverX = null;
    cv.addEventListener('pointermove', function (e) { cv.hoverX = e.clientX - cv.getBoundingClientRect().left; });
    cv.addEventListener('pointerleave', function () { cv.hoverX = null; });
  });

  // Live: x, every θ and F. Record (compact): x and F, what an excitation does to the cart.
  function drawPlot(cv, compact) {
    var c = sizeCanvas(cv), ctx = c.ctx, W = c.W, H = c.H, G = geom(), plotHoverX = cv.hoverX;
    var bg = cssVar('--bg'), border = cssVar('--border'), muted = cssVar('--text-muted'), text = cssVar('--text'), mono = cssVar('--mono');
    var colors = [cssVar('--viz-1'), cssVar('--viz-2'), cssVar('--viz-3'), cssVar('--viz-4')];
    ctx.fillStyle = bg; ctx.fillRect(0, 0, W, H);
    var lanes = [{ label: 'x (m)', lo: -G.x_lim, hi: G.x_lim, d: 2, get: function (s) { return s.x; }, ref: function (s) { return s.xr; } }];
    for (var k = 0; k < G.n && !compact; k++) (function (k) { lanes.push({ label: 'θ' + '₁₂₃₄'[k] + ' (°)', lo: -90, hi: 270, d: 0, wrap: true, get: function (s) { return s.th[k]; } }); })(k);
    lanes.push({ label: 'F (N)', lo: -G.fmax, hi: G.fmax, d: 1, get: function (s) { return s.cmd; }, second: function (s) { return s.applied; } });
    var PAD_L = 74, PAD_R = 10, PAD_T = 4, PAD_B = 16;
    var rowH = (H - PAD_T - PAD_B) / lanes.length, pw = W - PAD_L - PAD_R;
    var hist = S.hist, n = hist.length, lastTick = n ? hist[n - 1].tick : 0, span = HIST_S / G.dt;
    function xPx(tick) { return PAD_L + pw * (1 - (lastTick - tick) / span); }
    var hk = -1;
    if (plotHoverX !== null && plotHoverX >= PAD_L && plotHoverX <= PAD_L + pw && n) {
      var tk = lastTick - (1 - (plotHoverX - PAD_L) / pw) * span, best = Infinity;
      for (var i = 0; i < n; i++) { var dd = Math.abs(hist[i].tick - tk); if (dd < best) { best = dd; hk = i; } }
    }
    var dots = [];
    lanes.forEach(function (s, si) {
      var yTop = PAD_T + si * rowH, mid = yTop + rowH / 2, col = colors[si % colors.length];
      if (s.label.charAt(0) === 'F') col = colors[3];
      function yPx(v) { return yTop + rowH * (0.08 + 0.84 * (s.hi - clamp(v, s.lo, s.hi)) / (s.hi - s.lo)); }
      ctx.strokeStyle = border; ctx.lineWidth = 0.5;
      if (si > 0) { ctx.beginPath(); ctx.moveTo(PAD_L, yTop); ctx.lineTo(W - PAD_R, yTop); ctx.stroke(); }
      ctx.beginPath(); ctx.moveTo(PAD_L, yPx(0)); ctx.lineTo(W - PAD_R, yPx(0)); ctx.stroke();
      if (s.wrap) {   // up (0°) solid, down (180°) dotted, each named at the right edge
        ctx.setLineDash([2, 3]); ctx.beginPath(); ctx.moveTo(PAD_L, yPx(180)); ctx.lineTo(W - PAD_R, yPx(180)); ctx.stroke(); ctx.setLineDash([]);
        ctx.font = '9px ' + mono; ctx.fillStyle = muted; ctx.textAlign = 'right'; ctx.textBaseline = 'bottom';
        ctx.fillText('up', W - PAD_R - 3, yPx(0) - 1); ctx.fillText('down', W - PAD_R - 3, yPx(180) - 1);
      }
      function trace(fn) {
        ctx.beginPath();
        var prev = null;
        for (var i = 0; i < n; i++) {
          var v = fn(hist[i]), xp = xPx(hist[i].tick), yp = yPx(v);
          var gapped = i > 0 && hist[i].tick - hist[i - 1].tick > 0.25 / G.dt;
          if (i === 0 || gapped || (s.wrap && Math.abs(v - prev) > 180)) ctx.moveTo(xp, yp); else ctx.lineTo(xp, yp);
          prev = v;
        }
        ctx.stroke();
      }
      if (s.ref) { ctx.strokeStyle = cssVar('--lime'); ctx.globalAlpha = 0.7; ctx.lineWidth = 1; ctx.setLineDash([3, 3]); trace(s.ref); ctx.setLineDash([]); ctx.globalAlpha = 1; }
      if (s.second) { ctx.strokeStyle = col; ctx.globalAlpha = 0.55; ctx.lineWidth = 1.2; ctx.setLineDash([4, 3]); trace(s.second); ctx.setLineDash([]); ctx.globalAlpha = 1; }
      ctx.strokeStyle = col; ctx.lineWidth = 1.4; trace(s.get);

      var hovered = hk >= 0, cur = n ? s.get(hist[hovered ? hk : n - 1]) : 0;
      if (hovered) dots.push({ y: yPx(cur), color: col });
      ctx.font = '10px ' + mono; ctx.textAlign = 'right'; ctx.textBaseline = 'middle';
      var lw = ctx.measureText(s.label).width;
      ctx.fillStyle = col; ctx.fillRect(PAD_L - 6 - lw - 12, mid - 8.5, 8, 3);
      ctx.fillStyle = muted; ctx.fillText(s.label, PAD_L - 6, mid - 7);
      ctx.fillStyle = text; ctx.fillText(num(cur, s.d), PAD_L - 6, mid + 7);
      if (s.second && n) {
        ctx.fillStyle = muted; ctx.textAlign = 'right';
        ctx.fillText('cmd solid · applied dashed', W - PAD_R - 4, yTop + 9);
      }
      if (s.ref && n) { ctx.fillStyle = cssVar('--lime-text'); ctx.textAlign = 'right'; ctx.fillText('x_ref dashed', W - PAD_R - 4, yTop + 9); }
    });
    // time axis
    ctx.font = '10px ' + mono; ctx.fillStyle = muted; ctx.textBaseline = 'alphabetic';
    for (var t = 0; t <= HIST_S; t += 1) {
      var xp = PAD_L + pw * (1 - t / HIST_S);
      ctx.textAlign = t === 0 ? 'right' : 'center';
      ctx.fillText(t === 0 ? 'now' : minus('-' + t + ' s'), xp, H - 4);
    }
    if (hk >= 0 && dots.length) {
      var hx = xPx(hist[hk].tick);
      ctx.strokeStyle = muted; ctx.globalAlpha = 0.6; ctx.lineWidth = 1;
      ctx.beginPath(); ctx.moveTo(hx, PAD_T); ctx.lineTo(hx, H - PAD_B); ctx.stroke(); ctx.globalAlpha = 1;
      dots.forEach(function (d) {
        ctx.fillStyle = bg; ctx.beginPath(); ctx.arc(hx, d.y, 5, 0, 2 * Math.PI); ctx.fill();
        ctx.fillStyle = d.color; ctx.beginPath(); ctx.arc(hx, d.y, 3.5, 0, 2 * Math.PI); ctx.fill();
      });
    }
    if (!n) {
      ctx.fillStyle = muted; ctx.font = '12px ' + cssVar('--font'); ctx.textAlign = 'center'; ctx.textBaseline = 'middle';
      ctx.fillText('No telemetry yet', PAD_L + pw / 2, H / 2);
    }
  }

  // Execution time of each streamed tick against the period and the 80 % budget.
  function drawSpark() {
    var c = sizeCanvas(sparkCv), ctx = c.ctx, W = c.W, H = c.H, G = geom();
    var period = get(S.status, 'loop.period_us', G.dt * 1e6), hist = S.hist, n = hist.length;
    ctx.fillStyle = cssVar('--bg'); ctx.fillRect(0, 0, W, H);
    var top = 10, bot = H - 6, lim = period * 1.1;
    function y(v) { return bot - clamp(v / lim, 0, 1.05) * (bot - top); }
    ctx.strokeStyle = cssVar('--border-strong'); ctx.lineWidth = 1;
    ctx.beginPath(); ctx.moveTo(0, y(period)); ctx.lineTo(W, y(period)); ctx.stroke();
    ctx.strokeStyle = cssVar('--lime'); ctx.setLineDash([4, 3]);
    ctx.beginPath(); ctx.moveTo(0, y(0.8 * period)); ctx.lineTo(W, y(0.8 * period)); ctx.stroke(); ctx.setLineDash([]);
    ctx.font = '9px ' + cssVar('--mono'); ctx.fillStyle = cssVar('--text-muted'); ctx.textAlign = 'left'; ctx.textBaseline = 'bottom';
    ctx.fillText('period ' + us(period), 6, y(period) - 1);
    ctx.fillStyle = cssVar('--lime-text'); ctx.fillText('80 %', 6, y(0.8 * period) - 1);
    if (!n) return;
    var lastTick = hist[n - 1].tick, span = HIST_S / G.dt;
    ctx.strokeStyle = cssVar('--viz-1'); ctx.lineWidth = 1.2; ctx.beginPath();
    for (var i = 0; i < n; i++) {
      var xp = W * (1 - (lastTick - hist[i].tick) / span), yp = y(hist[i].loop);
      if (i === 0) ctx.moveTo(xp, yp); else ctx.lineTo(xp, yp);
    }
    ctx.stroke();
    ctx.fillStyle = cssVar('--bad');
    for (i = 0; i < n; i++) if (hist[i].loop > 0.8 * period) { ctx.beginPath(); ctx.arc(W * (1 - (lastTick - hist[i].tick) / span), y(hist[i].loop), 2, 0, 2 * Math.PI); ctx.fill(); }
    ctx.fillStyle = cssVar('--text-muted'); ctx.textAlign = 'right';
    ctx.fillText('this tick ' + us(hist[n - 1].loop), W - 6, top + 2);
  }

  /* ---- Network diagram: the SimBa actor with live activations (swingup.js drawNet) ---- */
  function drawNet() {
    var c = sizeCanvas(netCv), ctx = c.ctx, W = c.W, H = c.H;
    var bg = cssVar('--bg-soft'), text = cssVar('--text'), muted = cssVar('--text-muted');
    var border = cssVar('--border-strong'), mono = cssVar('--mono'), font = cssVar('--font');
    var neg = rgb('--nn-act-neg'), zero = rgb('--nn-act-zero'), pos = rgb('--nn-act-pos');
    ctx.fillStyle = bg; ctx.fillRect(0, 0, W, H);
    ctx.textAlign = 'center'; ctx.textBaseline = 'alphabetic';
    var msgX = Math.min(W / 2, 200);   // stays in view when the canvas scrolls sideways on a phone
    if (!S.policy) {
      ctx.fillStyle = muted; ctx.font = '13px ' + font;
      var st = S.model.state;
      if (st === 'loading') {
        var fr = S.model.total ? S.model.got / S.model.total : 0;
        ctx.fillText('Loading the network…', msgX, H / 2 - 14);
        ctx.strokeStyle = border; ctx.lineWidth = 1; ctx.strokeRect(msgX - 110.5, H / 2 - 0.5, 221, 7);
        ctx.fillStyle = cssVar('--lime'); ctx.fillRect(msgX - 110, H / 2, 220 * fr, 6);
        ctx.fillStyle = muted; ctx.font = '11px ' + mono;
        ctx.fillText(bytes(S.model.got) + (S.model.total ? ' / ' + bytes(S.model.total) : ''), msgX, H / 2 + 24);
      } else if (st === 'none') {
        ctx.fillText('No model on the device', msgX, H / 2 - 6);
        ctx.font = '11px ' + font; ctx.fillText('Export one with task export-firmware and drop the .ncpm on Settings → Update.', msgX, H / 2 + 14);
      } else if (st === 'error') {
        ctx.fillText('Could not load the network: ' + S.model.error, msgX, H / 2);
      } else {
        ctx.fillText(S.conn === 'live' ? 'Waiting for the model…' : 'Connecting…', msgX, H / 2);
      }
      return;
    }
    var h = S.hdr, tr = S.policy.trace, B = h.n_blocks, D = h.obs_dim, L = S.layout;
    function dims(nu) { var cols = Math.ceil(Math.sqrt(nu)); return { cols: cols, rows: Math.ceil(nu / cols) }; }
    var dh = dims(h.hidden), du = dims(h.hidden * h.expand);
    var cellH = Math.max(3, Math.min(6, Math.floor(100 / dh.cols))), cellU = Math.max(2, Math.min(4, Math.floor(132 / du.cols)));
    var hs = dh.cols * cellH, hsy = dh.rows * cellH, usx = du.cols * cellU, usy = du.rows * cellU;
    var gap = 40, rowH = Math.min(22, 236 / D), ix = 10, iy = 24;
    var x0 = 162, DW = x0 + hs + B * (gap + usx + gap + hs) + gap + 120, DH = Math.max(300, iy + D * rowH + 44);
    var sc = Math.min(W / DW, H / DH);
    ctx.save();
    ctx.translate((W - DW * sc) / 2, (H - DH * sc) / 2);
    ctx.scale(sc, sc);
    var gridY = Math.max(70, (DH - hsy) / 2 - 20);
    function grid(vals, bx, by, d, cell, signed, label, sub) {
      var m = 1e-6, i;
      for (i = 0; i < vals.length; i++) m = Math.max(m, Math.abs(vals[i]));
      for (i = 0; i < vals.length; i++) {
        var f = vals[i] / m;
        ctx.fillStyle = signed ? (f < 0 ? mix(zero, neg, -f) : mix(zero, pos, f)) : mix(zero, pos, Math.max(0, f));
        ctx.fillRect(bx + (i % d.cols) * cell, by + Math.floor(i / d.cols) * cell, cell, cell);
      }
      ctx.strokeStyle = border; ctx.lineWidth = 1; ctx.strokeRect(bx - .5, by - .5, d.cols * cell + 1, d.rows * cell + 1);
      ctx.fillStyle = text; ctx.font = '600 12px ' + font;
      ctx.fillText(label, bx + d.cols * cell / 2, by + d.rows * cell + 20);
      ctx.fillStyle = muted; ctx.font = '11px ' + font;
      ctx.fillText(sub, bx + d.cols * cell / 2, by + d.rows * cell + 35);
    }
    function arrow(a, b, y) {
      ctx.strokeStyle = muted; ctx.fillStyle = muted; ctx.lineWidth = 1.2;
      ctx.beginPath(); ctx.moveTo(a, y); ctx.lineTo(b, y); ctx.stroke();
      ctx.beginPath(); ctx.moveTo(b, y); ctx.lineTo(b - 5, y - 3.5); ctx.lineTo(b - 5, y + 3.5); ctx.closePath(); ctx.fill();
    }
    // Inputs: one labelled bar each.
    ctx.font = '11px ' + mono;
    var full = [mix(zero, neg, 1), mix(zero, pos, 1)], zeroC = mix(zero, zero, 0);
    for (var k = 0; k < D; k++) {
      var cy = iy + k * rowH, v = clamp(tr.obs[k] / (L[k] ? L[k].s : 1), -1, 1), bh = Math.min(12, rowH - 4);
      ctx.textAlign = 'right'; ctx.fillStyle = L[k] && L[k].goal ? text : muted; ctx.fillText(L[k] ? L[k].l : '', ix + 78, cy + bh / 2 + 7);
      ctx.fillStyle = zeroC; ctx.fillRect(ix + 84, cy + 3, 40, bh);
      ctx.fillStyle = v < 0 ? full[0] : full[1];
      ctx.fillRect(ix + 104 + (v < 0 ? v * 20 : 0), cy + 3, Math.abs(v) * 20, bh);
      ctx.fillStyle = border; ctx.fillRect(ix + 104, cy + 1, 1, bh + 4);
    }
    ctx.textAlign = 'center'; ctx.fillStyle = text; ctx.font = '600 12px ' + font;
    ctx.fillText('Input', ix + 76, iy + D * rowH + 16);
    ctx.fillStyle = muted; ctx.font = '11px ' + font;
    ctx.fillText(D + ' numbers', ix + 76, iy + D * rowH + 31);
    var midY = gridY + hsy / 2;
    arrow(ix + 130, x0 - 4, midY);

    grid(tr.h[0], x0, gridY, dh, cellH, true, 'Embed', h.hidden + ' units');
    var cx = x0 + hs;
    for (var b = 0; b < B; b++) {
      arrow(cx + 4, cx + gap - 4, midY);
      var ux = cx + gap, uy = gridY - (usy - hsy) / 2;
      grid(tr.u[b], ux, uy, du, cellU, false, 'Block ' + (b + 1) + ' · ReLU', h.hidden * h.expand + ' units');
      arrow(ux + usx + 4, ux + usx + gap - 4, midY);
      var hx2 = ux + usx + gap;
      grid(tr.h[b + 1], hx2, gridY, dh, cellH, true, 'Sum', 'h + fc2(…)');
      ctx.strokeStyle = muted; ctx.setLineDash([3, 3]); ctx.lineWidth = 1;
      var sx0 = cx - hs / 2 + (b > 0 ? 14 : 0), sx1 = hx2 + hs / 2 - (b < B - 1 ? 14 : 0), top = gridY - 14 - Math.max(0, (usy - hsy) / 2);
      ctx.beginPath(); ctx.moveTo(sx0, gridY - 2); ctx.lineTo(sx0, top); ctx.lineTo(sx1, top); ctx.lineTo(sx1, gridY - 2); ctx.stroke();
      ctx.setLineDash([]);
      ctx.fillStyle = muted;
      ctx.beginPath(); ctx.moveTo(sx1, gridY - 1); ctx.lineTo(sx1 - 3.5, gridY - 7); ctx.lineTo(sx1 + 3.5, gridY - 7); ctx.closePath(); ctx.fill();
      ctx.font = '11px ' + font; ctx.textAlign = 'center';
      ctx.fillText('skip', (sx0 + sx1) / 2, top - 4);
      cx = hx2 + hs;
    }
    arrow(cx + 4, cx + gap - 4, midY);

    // Output: one number, squashed by tanh. Delta mode: a rate of change of the command.
    var ox = cx + gap + 8, f = Math.tanh(tr.mean), unit = h.delta ? ' N/s' : ' N', scaleOut = h.delta ? h.force_slew : h.force_max;
    ctx.fillStyle = zeroC; ctx.fillRect(ox, midY - 48, 16, 96);
    ctx.fillStyle = f < 0 ? full[0] : full[1];
    ctx.fillRect(ox, f < 0 ? midY : midY - f * 48, 16, Math.abs(f) * 48);
    ctx.fillStyle = border; ctx.fillRect(ox - 3, midY, 22, 1);
    ctx.fillStyle = text; ctx.font = '600 13px ' + mono; ctx.textAlign = 'left';
    var out = f * scaleOut, val = num(out, h.delta ? 0 : 1);
    if (val.charAt(0) !== '−') val = ' ' + val;
    ctx.fillText(val + unit, ox + 24, midY + 4);
    ctx.textAlign = 'center'; ctx.font = '600 12px ' + font;
    ctx.fillText(h.delta ? 'Force rate' : 'Force', ox + 8, gridY + hsy + 20);
    ctx.fillStyle = muted; ctx.font = '11px ' + font;
    ctx.fillText('tanh · ' + num(scaleOut, h.delta ? 0 : 1) + unit, ox + 8, gridY + hsy + 35);
    ctx.restore();
  }

  // One forward pass per new frame (≤ 25 per second), then the parity check against the device's command.
  var lastNetTick = -1, lastNetAt = 0, netCount = 0, netWindow = 0;
  function runNet(now) {
    if (!S.policy || !S.last || S.last.tick === lastNetTick || now - lastNetAt < 38) return false;
    var f = S.last, h = S.hdr, G = geom(), t0 = performance.now();
    var a = S.policy.act(f.obs);
    S.netMs = S.netMs ? 0.9 * S.netMs + 0.1 * (performance.now() - t0) : performance.now() - t0;
    lastNetTick = f.tick; lastNetAt = now; netCount++;
    if (f.run === 1 && f.mode === 0 && !get(S.status, 'excite.active', false)) {   // an excitation is added to cmd: no parity then
      var pred;
      if (h.delta) {
        if (!h.action_history || !G.slew) pred = null;
        else pred = clamp(f.obs[h.hist_at] * h.force_max + clamp(a / h.force_max, -1, 1) * h.force_slew * h.dt, -h.force_max, h.force_max);
      } else pred = clamp(a, -h.force_max, h.force_max);
      if (pred != null) {
        var e = Math.abs(pred - f.cmd);
        S.parity = S.parity == null ? e : 0.95 * S.parity + 0.05 * e;
        S.parityN++; S.parityAt = performance.now();
      }
    }
    if (now - netWindow > 1000) { S.netFps = netCount; netCount = 0; netWindow = now; renderNetMeta(); }
    return true;
  }

  var lastStatusRender = 0, lastReadout = 0;
  function frame(now) {
    var ran = runNet(now);   // every tab: the parity check in Settings → Model needs it
    if (S.tab === 'live') {
      drawScene();
      drawPlot(plotCv, false);
      if (now - lastReadout > 100) { lastReadout = now; renderReadout(); renderReach(); }
      if (ran || !S.policy || now - lastNetAt > 500) drawNet();
    }
    if (S.tab === 'record') drawPlot(recPlotCv, true);
    if (S.tab === 'health') drawSpark();
    if (S.model.state === 'loading' && now - lastStatusRender > 200) { lastStatusRender = now; renderNetMeta(); }
    requestAnimationFrame(frame);
  }

  /* ================================================================
     Update: one drop zone for firmware images (0xE9) and model blobs ("NCPM")
  ================================================================ */
  var dz = $('dropzone'), fileInput = $('file-input'), panel = $('file-panel'), pending = null, uploading = false;
  var ARMED_WHY = 'Disarm first: writing flash disables the cache, which stalls both cores — the control loop would miss its deadlines with the drive enabled.';

  function cstr(u8, at, n) {
    var s = '';
    for (var i = at; i < at + n && i < u8.length && u8[i]; i++) s += String.fromCharCode(u8[i]);
    return s;
  }
  // esp_app_desc_t sits right after the image header (24 B) and the first segment header (8 B).
  function parseAppImage(u8) {
    var dv = new DataView(u8.buffer, u8.byteOffset, Math.min(u8.length, 256));
    if (u8.length < 256 || dv.getUint32(32, true) !== 0xABCD5432) return null;
    return { version: cstr(u8, 48, 32), project: cstr(u8, 80, 32), time: cstr(u8, 112, 16), date: cstr(u8, 128, 16), idf: cstr(u8, 144, 32) };
  }

  function identify(file, buf) {
    var u8 = new Uint8Array(buf), out = { file: file, buf: buf, kind: null, rows: [], problems: [], warnings: [] };
    if (u8.length >= 4 && cstr(u8, 0, 4) === 'NCPM') {
      out.kind = 'model';
      var h;
      try { h = parseNcpm(buf); } catch (e) { out.problems.push(e.message); return out; }
      out.hdr = h;
      var crc = crc32(u8, 12, 16), crcOk = crc === h.crc32;
      if (h.total_size !== u8.length) out.problems.push('size: the header says ' + bytes(h.total_size) + ', the file is ' + bytes(u8.length) + ' (truncated?)');
      else if (h.expected_size !== u8.length) out.problems.push('size disagrees with the dims (' + bytes(h.expected_size) + ' expected)');
      if (!crcOk) out.problems.push('CRC mismatch: the file is corrupted');
      var cur = get(S.info, 'model', null), n = geom().n;
      if (S.backend === 'rig' && h.n_links !== n) out.problems.push('this rig has ' + plural(n, 'link') + '; the model was trained for ' + h.n_links);
      else if (h.n_links !== n) out.warnings.push('the current model has ' + plural(n, 'link') + '; this one ' + h.n_links);
      if (cur && cur.crc32 && parseInt(cur.crc32, 16) === h.crc32) out.warnings.push('this exact model is already on the device');
      var dt = h.dt, period = Math.round(dt * 1e6);
      out.rows = [
        ['label', h.label || '(none)', 'accent'],
        ['trained', (h.train_step / 1e6).toFixed(2) + ' M steps'],
        ['links', h.n_links + (h.goal ? ' · goal-conditioned' : '')],
        ['network', h.hidden + ' × ' + h.n_blocks + ' blocks · ×' + h.expand + ' · ' + (h.params / 1e6).toFixed(2) + ' M params'],
        ['inputs', h.obs_dim + (h.action_history ? ' (' + h.action_history + ' past cmds)' : '')],
        ['weights', bytes(u8.length) + ' · ' + (h.dtype ? 'f16' : 'f32')],
        ['action', (h.delta ? 'delta' : 'absolute') + ' · ' + (h.ekf ? 'EKF' : 'differenced') + ' · delay ' + h.delay_steps],
        ['rate', Math.round(1 / dt) + ' Hz (' + period + ' µs period)'],
        ['limits', '±' + num(h.force_max, 2) + ' N · ' + num(h.force_slew, 0) + ' N/s · rail ±' + num(h.x_lim, 2) + ' m'],
        ['crc32', ('0000000' + h.crc32.toString(16)).slice(-8) + (crcOk ? ' ✓' : ' ✕'), crcOk ? 'good' : 'bad']
      ];
    } else if (u8.length && u8[0] === 0xE9) {
      out.kind = 'firmware';
      var d = parseAppImage(u8);
      if (/-factory\.bin$/i.test(file.name) || !d) {
        out.problems.push(d ? 'this is the USB factory image (bootloader + partition table + app at 0x0): use n-cartpole-fw-<ver>.bin for an update over the air'
          : 'no app descriptor: is this the factory image or a bootloader? Over the air takes n-cartpole-fw-<ver>.bin');
      }
      if (d && d.project !== 'n_cartpole_fw') out.problems.push('project "' + d.project + '" — this device only takes n_cartpole_fw images');
      if (!/^n-cartpole-fw-.*\.bin$/i.test(file.name) && !out.problems.length) out.warnings.push('unusual file name (releases are n-cartpole-fw-<ver>.bin)');
      out.desc = d;
      var running = get(S.info, 'fw.version', null);
      if (d && running && d.version === running) out.warnings.push('this version is already running');
      if (u8.length > 3 * 1024 * 1024) out.problems.push('larger than the 3 MB app slot');
      out.rows = d ? [
        ['project', d.project, d.project === 'n_cartpole_fw' ? 'good' : 'bad'],
        ['version', d.version, 'accent'],
        ['running now', running ? running : '—'],
        ['built', (d.date + ' ' + d.time).trim()],
        ['esp-idf', d.idf],
        ['size', bytes(u8.length)],
        ['installs to', get(S.info, 'ota.next', 'the other slot')]
      ] : [['size', bytes(u8.length)]];
    } else {
      out.problems.push('neither a firmware image (first byte 0xE9) nor a model blob ("NCPM")');
    }
    return out;
  }

  function renderPending(note, noteKind) {
    if (!pending) { panel.hidden = true; return; }
    var p = pending, armed = S.run === 'armed', bad = p.problems.length > 0;
    var title = p.kind === 'model' ? 'Model blob' : p.kind === 'firmware' ? 'Firmware image' : 'Unknown file';
    var html = '<div class="file-title">' + title + ' <span class="mono">' + esc(p.file.name) + '</span></div>';
    if (p.rows.length) {
      html += '<dl class="kv">' + p.rows.map(function (r) { return '<dt>' + esc(r[0]) + '</dt><dd' + (r[2] ? ' class="' + r[2] + '"' : '') + '>' + esc(r[1]) + '</dd>'; }).join('') + '</dl>';
    }
    p.problems.forEach(function (m) { html += '<p class="notice bad">' + esc(m) + '</p>'; });
    p.warnings.forEach(function (m) { html += '<p class="notice">' + esc(m) + '</p>'; });
    if (!bad && armed) html += '<p class="notice bad">' + esc(ARMED_WHY) + '</p>';
    if (!bad && p.kind === 'firmware') html += '<p class="panel-hint">The device writes the image to <code>' + esc(get(S.info, 'ota.next', 'the other slot')) + '</code>, reboots into it, and keeps it only if it runs healthy for 10 s — otherwise it rolls back.</p>';
    if (!bad && p.kind === 'model') html += '<p class="panel-hint">Stored in the model partition and hot-loaded; no reboot. The controller is held in MAINT while flash is written.</p>';
    if (note) html += '<p class="notice ' + (noteKind || '') + '">' + note + '</p>';
    html += '<div class="progress-row" id="up-progress" hidden><span class="progress"><i id="up-bar"></i></span><span id="up-pct">0 %</span></div>';
    if (!p.done) {
      html += '<div class="btn-row"><button class="sim-btn primary" type="button" id="up-go"' + (bad || armed || uploading || S.conn !== 'live' ? ' disabled' : '') + '>' +
        (p.kind === 'firmware' && p.desc ? 'Install v' + esc(p.desc.version) + ' &amp; reboot' : p.kind === 'model' ? 'Upload model' : 'Upload') +
        '</button><button class="sim-btn" type="button" id="up-cancel"' + (uploading ? ' disabled' : '') + '>Cancel</button></div>';
    } else {
      html += '<div class="btn-row"><button class="sim-btn" type="button" id="up-cancel">Done</button></div>';
    }
    panel.innerHTML = html;
    panel.className = 'file-panel' + (bad ? ' bad' : '');
    panel.hidden = false;
    var go = $('up-go'), cancel = $('up-cancel');
    if (go) go.addEventListener('click', upload);
    if (cancel) cancel.addEventListener('click', function () { pending = null; renderPending(); });
  }

  function choose(file) {
    if (!file) return;
    if (uploading) return;
    file.arrayBuffer().then(function (buf) { pending = identify(file, buf); renderPending(); })
      .catch(function () { pending = { file: file, kind: null, rows: [], problems: ['could not read the file'], warnings: [] }; renderPending(); });
  }

  function upload() {
    var p = pending;
    if (!p || uploading) return;
    if (S.run === 'armed') { renderPending(esc(ARMED_WHY), 'bad'); return; }
    if (p.kind === 'firmware' && !window.confirm('Install firmware v' + p.desc.version + '?\n\nThe device reboots into it (the page reconnects by itself). If it doesn\'t run healthy for 10 s it rolls back to v' + get(S.info, 'fw.version', 'the current one') + '.')) return;
    uploading = true;
    renderPending();
    var bar = $('up-bar'), pct = $('up-pct'), row = $('up-progress');
    row.hidden = false;
    var xhr = new XMLHttpRequest();
    xhr.open('POST', p.kind === 'firmware' ? 'api/ota' : 'api/model');
    xhr.setRequestHeader('Content-Type', 'application/octet-stream');
    xhr.upload.onprogress = function (e) {
      if (!e.lengthComputable) return;
      var f = e.loaded / e.total;
      bar.style.width = (100 * f).toFixed(1) + '%';
      pct.textContent = f < 1 ? Math.round(100 * f) + ' %' : 'writing flash…';
    };
    xhr.onload = function () {
      uploading = false;
      var j = null;
      try { j = JSON.parse(xhr.responseText); } catch (e) {}
      if (xhr.status >= 200 && xhr.status < 300 && j && j.ok !== false) {
        p.done = true;
        if (p.kind === 'firmware') {
          var from = get(S.info, 'fw.version', '?');
          S.rebooting = { from: from, to: j.version || p.desc.version, t0: performance.now() };
          setConn('rebooting');
          renderPending('Installed <b>v' + esc(S.rebooting.to) + '</b>. Rebooting — this page reconnects by itself…', 'good');
          waitForReboot();
        } else {
          renderPending('Model <b>' + esc(j.label || p.hdr.label) + '</b> stored and loaded.', 'good');
          loadInfo();
        }
      } else {
        var why = (j && j.error) || ('HTTP ' + xhr.status);
        if (xhr.status === 409) why += ' — ' + ARMED_WHY;
        renderPending('Upload refused: ' + esc(why), 'bad');
      }
    };
    xhr.onerror = function () { uploading = false; renderPending('The upload failed: the connection dropped.', 'bad'); };
    xhr.send(p.buf);
  }

  function waitForReboot() {
    // The socket drops when the device restarts; poll /api/info until it answers again.
    var tries = 0;
    (function poll() {
      tries++;
      if (!S.rebooting) return;
      fetchJSON('api/info', null, 2000).then(function (info) {
        if (performance.now() - S.rebooting.t0 < 2500) throw new Error('still the old boot');
        if (get(info, 'fw.version', null) === S.rebooting.from && S.rebooting.from !== S.rebooting.to && tries < 40) throw new Error('not yet');
        S.info = info; rebootDone(null); renderInfo(); connect();
      }).catch(function () { if (tries < 90) setTimeout(poll, 1500); else { S.rebooting = null; setConn('offline'); renderPending('The device has not come back after 2 minutes. Check its serial console.', 'bad'); } });
    })();
  }
  function rebootDone() {
    var r = S.rebooting;
    if (!r) return;
    S.rebooting = null;
    var now = get(S.info, 'fw.version', '?'), state = get(S.info, 'ota.state', '');
    if (pending && pending.kind === 'firmware') {
      renderPending(now === r.to
        ? 'Back online on <b>v' + esc(now) + '</b> (was v' + esc(r.from) + ')' + (state === 'pending' ? ' · it marks itself valid after 10 s of healthy running.' : '.')
        : 'Back online, but still on v' + esc(now) + ' — the new image rolled back. Check the serial log.', now === r.to ? 'good' : 'bad');
    }
    if (S.conn === 'rebooting') setConn('reconnecting');
  }

  dz.addEventListener('click', function () { if (!uploading) fileInput.click(); });
  dz.addEventListener('keydown', function (e) { if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); fileInput.click(); } });
  fileInput.addEventListener('change', function () { choose(fileInput.files[0]); fileInput.value = ''; });
  ['dragenter', 'dragover'].forEach(function (ev) { dz.addEventListener(ev, function (e) { e.preventDefault(); dz.classList.add('over'); }); });
  ['dragleave', 'drop'].forEach(function (ev) { dz.addEventListener(ev, function (e) { e.preventDefault(); dz.classList.remove('over'); }); });
  dz.addEventListener('drop', function (e) { choose(e.dataTransfer && e.dataTransfer.files[0]); });
  // A file dropped anywhere else on the page shouldn't navigate away from the dashboard.
  window.addEventListener('dragover', function (e) { e.preventDefault(); });
  window.addEventListener('drop', function (e) { e.preventDefault(); });

  /* ---- Releases: the latest fw-v* tags, when the internet is reachable (quietly nothing offline) ---- */
  var releasesLoaded = false, releases = null;
  function semver(v) { return String(v).replace(/^fw-v|^v/, '').split(/[.-]/).map(function (x) { return parseInt(x, 10) || 0; }); }
  function newer(a, b) { a = semver(a); b = semver(b); for (var i = 0; i < 3; i++) { if ((a[i] || 0) !== (b[i] || 0)) return (a[i] || 0) > (b[i] || 0); } return false; }
  function loadReleases() {
    releasesLoaded = true;
    var repo = get(S.info, 'fw.repo', 'Tsmorz/n-cartpole');
    fetchJSON('https://api.github.com/repos/' + repo + '/releases?per_page=12', { headers: { Accept: 'application/vnd.github+json' } }, 5000)
      .then(function (list) {
        releases = (list || []).filter(function (r) { return /^fw-v/.test(r.tag_name) && !r.draft; }).slice(0, 4);
        renderReleases();
      }).catch(function () { /* offline (e.g. on the device's access point): keep the static hint */ });
  }
  function renderReleases() {
    if (!releases || !releases.length) return;
    var running = get(S.info, 'fw.version', null);
    $('releases').innerHTML = releases.map(function (r) {
      var asset = (r.assets || []).filter(function (a) { return /^n-cartpole-fw-.*\.bin$/.test(a.name) && !/factory/.test(a.name); })[0];
      var v = r.tag_name.replace(/^fw-v/, ''), badge = '';
      if (running && v === running) badge = '<span class="rel-badge">running</span>';
      else if (running && newer(v, running)) badge = '<span class="rel-badge">newer</span>';
      return '<li><a class="rel-tag" href="' + esc(r.html_url) + '" target="_blank" rel="noopener">' + esc(r.tag_name) + '</a>' + badge +
        '<span class="rel-date">' + esc((r.published_at || '').slice(0, 10)) + '</span>' +
        (asset ? '<a class="rel-get" href="' + esc(asset.browser_download_url) + '">' + esc(bytes(asset.size)) + ' .bin</a>' : '') + '</li>';
    }).join('');
  }

  /* ---- Wi-Fi and reboot ---- */
  // Progress of a join this page started: the device reports it in status.wifi.join. When the
  // device leaves its setup access point (or switches channel) this page loses it, so a timer
  // covers the case where the result never arrives.
  function trackJoin(j) {
    if (!S.joinWatch || !j) return;
    var host = (get(S.info, 'hostname', null) || 'cartpole') + '.local';
    if (j.state === 'connecting') {
      flash('wifi-msg', 'Joining "' + S.joinWatch + '"… (up to 20 s)', '', 0);
    } else if (j.state === 'ok') {
      flash('wifi-msg', 'Joined "' + j.ssid + '" at ' + j.ip + '. Put this computer back on that network, then open http://' + host + '/ or http://' + j.ip + '/', 'good', 0);
      S.joinWatch = null;
      $('wifi-save').disabled = false;
    } else if (j.state === 'failed') {
      flash('wifi-msg', 'Could not join "' + S.joinWatch + '": ' + (j.error || 'unknown error') + '. Nothing was saved.', 'bad', 0);
      S.joinWatch = null;
      $('wifi-save').disabled = false;
    }
  }
  $('wifi-form').addEventListener('submit', function (e) {
    e.preventDefault();
    var ssid = $('wifi-ssid').value.trim(), pass = $('wifi-pass').value;
    if (!ssid) return;
    if (S.run === 'armed') { flash('wifi-msg', 'Disarm first: saving the network writes flash.', 'bad'); return; }
    if (S.wifiMode !== 'ap' && !window.confirm('Join "' + ssid + '"?\n\nIf it works the device moves to that network and this page loses it; if not, it stays on the current one.')) return;
    fetchJSON('api/wifi', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ ssid: ssid, password: pass }) }, 5000)
      .then(function () {
        S.joinWatch = ssid;
        $('wifi-save').disabled = true;
        $('wifi-pass').value = '';
        flash('wifi-msg', 'Joining "' + ssid + '"…', '', 0);
        setTimeout(function () {
          if (S.joinWatch !== ssid) return;
          S.joinWatch = null;
          $('wifi-save').disabled = false;
          flash('wifi-msg', 'Lost contact with the device — it probably joined "' + ssid + '". Put this computer back on that network and open http://' + ((get(S.info, 'hostname', null) || 'cartpole') + '.local') + '/ (or look up its IP in your router).', 'good', 0);
        }, 35000);
      }).catch(function (err) { flash('wifi-msg', err.message, 'bad'); });
  });
  // Hostname: saved to NVS and applied to mDNS at once (this page stays on its current address).
  $('host-form').addEventListener('submit', function (e) {
    e.preventDefault();
    var name = $('host-name').value.trim().toLowerCase().replace(/\.local$/, '');
    if (!/^[a-z0-9]([a-z0-9-]{0,30}[a-z0-9])?$/.test(name)) { flash('host-msg', '1–32 of a–z, 0–9 and -, not starting or ending with -.', 'bad'); return; }
    fetchJSON('api/hostname', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ hostname: name }) }, 5000)
      .then(function (j) {
        $('host-name').value = '';
        flash('host-msg', 'Renamed: http://' + (j && j.hostname || name) + '.local/ from now on (this page keeps working where it is).', 'good', 8000);
        loadInfo();
      }).catch(function (err) { flash('host-msg', err.message, 'bad'); });
  });
  // Forget: the device erases the network and reboots into its setup access point.
  $('wifi-forget').addEventListener('click', function () {
    if (S.run === 'armed') { flash('host-msg', 'Disarm first.', 'bad'); return; }
    var ap = 'cartpole-XXXX';
    if (!window.confirm('Forget the saved Wi-Fi network?\n\nThe device reboots into its setup access point (' + ap + ', http://192.168.4.1/) and this page loses it. Join that access point to set a new network.')) return;
    fetchJSON('api/wifi/forget', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: '{}' }, 4000).then(function () {
      flash('host-msg', 'Forgotten. The device is rebooting into its access point ' + ap + ': join it and open http://192.168.4.1/.', 'good', 0);
      S.rebooting = { from: get(S.info, 'fw.version', ''), to: get(S.info, 'fw.version', ''), t0: performance.now() };
      setConn('rebooting');
      waitForReboot();
    }).catch(function (err) { flash('host-msg', err.message, 'bad'); });
  });

  $('reboot').addEventListener('click', function () {
    if (S.run === 'armed') { flash('cmd-msg', 'Disarm before rebooting.', 'bad'); return; }
    if (!window.confirm('Reboot the device?')) return;
    fetchJSON('api/reboot', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: '{}' }, 4000).then(function () {
      S.rebooting = { from: get(S.info, 'fw.version', ''), to: get(S.info, 'fw.version', ''), t0: performance.now() };
      setConn('rebooting');
      waitForReboot();
    }).catch(function (err) { flash('cmd-msg', 'reboot: ' + err.message, 'bad'); });
  });

  /* ================================================================
     Record: logs for offline system identification (ncp_record.h, ncp_excite.h)
  ================================================================ */
  var PRESETS = [
    { id: 'ring', name: 'Ring-down', armed: false, mode: null, kind: 'none', seconds: 30,
      setup: 'Disarmed. Lift both links by hand, start the recording and let go: the free swing gives the pendulum periods and how fast they decay.',
      ids: 'joint friction, link COM and inertia' },
    { id: 'chirp', name: 'Cart chirp', armed: true, mode: 'manual', kind: 'chirp', amp: 3, f0: 0.2, f1: 6, duration: 40, seconds: 42,
      setup: 'Arm in Manual: the PD holds the cart at the slider while a 3 N chirp sweeps 0.2 → 6 Hz on top.',
      ids: 'cart mass, viscous + Coulomb rail friction, force gain, link coupling' },
    { id: 'prbs', name: 'PRBS', armed: true, mode: 'manual', kind: 'prbs', amp: 3, period: 0.05, duration: 30, seconds: 32, seed: 1,
      setup: 'Arm in Manual: ±3 N, a new random sign every 50 ms, on top of the PD.',
      ids: 'the same as the chirp, broadband; the actuator delay' },
    { id: 'steps', name: 'Steps', armed: true, mode: 'manual', kind: 'steps', amp: 4, period: 0.5, duration: 20, seconds: 22,
      setup: 'Arm in Manual: +4 N, 0, −4 N, 0 …, each held 0.5 s, on top of the PD.',
      ids: 'delay, slew limit, force gain' },
    { id: 'loop', name: 'Closed loop', armed: true, mode: 'ai', kind: 'none', seconds: 30,
      setup: 'Arm in AI and let the policy work: pick targets or Random. No excitation.',
      ids: 'validation of the fitted model under the policy' }
  ];
  var preset = PRESETS[0], recDownloads = 0, recKey = '';
  var recEl = { seconds: $('rec-seconds'), kind: $('rec-kind'), amp: $('rec-amp'), f0: $('rec-f0'), f1: $('rec-f1'), period: $('rec-period'), duration: $('rec-duration'), seed: $('rec-seed') };
  function pickPreset(p) {
    preset = p;
    recEl.seconds.value = p.seconds; recEl.kind.value = p.kind;
    recEl.amp.value = p.amp != null ? p.amp : 3; recEl.f0.value = p.f0 != null ? p.f0 : 0.2; recEl.f1.value = p.f1 != null ? p.f1 : 6;
    recEl.period.value = p.period != null ? p.period : 0.05; recEl.duration.value = p.duration != null ? p.duration : p.seconds;
    recEl.seed.value = p.seed != null ? p.seed : 1;
    document.querySelectorAll('#rec-presets .mode-btn').forEach(function (b) {
      var on = b.dataset.preset === p.id; b.classList.toggle('active', on); b.setAttribute('aria-pressed', on ? 'true' : 'false');
    });
    setText('rec-setup', p.setup);
    $('rec-ids').innerHTML = '<strong>Identifies:</strong> ' + esc(p.ids) + '. Then offline: <code>task sysid -- recordings/*.ncpr --config config/rig.toml</code>.';
    kindFields(); renderRecord(true);
  }
  function kindFields() {
    var k = recEl.kind.value;
    document.querySelectorAll('.rec-fields [data-for]').forEach(function (f) { f.hidden = f.dataset.for.split(' ').indexOf(k) < 0; });
    setText('rec-period-l', k === 'steps' ? 'hold (s)' : 'bit (s)');
  }
  PRESETS.forEach(function (p) {
    var b = document.createElement('button');
    b.type = 'button'; b.className = 'mode-btn'; b.dataset.preset = p.id; b.textContent = p.name;
    b.addEventListener('click', function () { pickPreset(p); });
    $('rec-presets').appendChild(b);
  });
  recEl.kind.addEventListener('change', function () { kindFields(); renderRecord(true); });
  function excitation() {
    var k = recEl.kind.value;
    if (k === 'none') return null;
    function v(e, d) { var x = parseFloat(e.value); return isFinite(x) ? x : d; }
    return { kind: k, amp: clamp(v(recEl.amp, 3), 0, 10), f0: v(recEl.f0, 0.2), f1: v(recEl.f1, 6), period: v(recEl.period, 0.05),
             duration: v(recEl.duration, 30), seed: Math.max(0, Math.round(v(recEl.seed, 1))) };
  }
  function renderRecord(force) {
    var r = get(S.status, 'record', null), x = get(S.status, 'excite', null), live = S.conn === 'live';
    var st = r ? r.state : null, recording = st === 'recording';
    var key = [force ? Math.random() : '', live, st, r && r.samples, x && x.active, x && x.t, S.run, S.mode, preset.id].join('|');
    if (key === recKey) return;
    recKey = key;
    // Header indicator
    $('rec-chip').hidden = !recording;
    if (recording) setText('rec-chip-t', (r.seconds || 0).toFixed(1) + ' s');
    var tag = $('rec-tag');
    tag.textContent = !r ? '—' : recording ? '● recording' : st === 'done' ? 'ready to download' : 'idle';
    tag.className = 'tag' + (recording ? ' bad' : st === 'done' ? ' ok' : '');
    // Setup check
    var w = '';
    if (live && S.run) {
      if (preset.armed && S.run !== 'armed') w = 'Needs the device armed' + (preset.mode ? ' in ' + (preset.mode === 'ai' ? 'AI' : 'Manual') : '') + ': the excitation only reaches the drive while armed.';
      else if (preset.armed && preset.mode && S.mode !== preset.mode) w = 'Switch to ' + (preset.mode === 'ai' ? 'AI' : 'Manual') + ' first' + (preset.mode === 'manual' ? ': the PD keeps the cart near the slider while the signal shakes it.' : '.');
      else if (!preset.armed && S.run === 'armed') w = 'Disarm first: a ring-down is the links swinging freely, with the drive off.';
    }
    var we = $('rec-warn'); we.hidden = !w; we.textContent = w;
    // Progress
    var cap = r && r.capacity_s ? r.capacity_s : 0, secs = r ? r.seconds || 0 : 0;
    setText('rec-state', !r ? 'no recording yet' : recording ? 'recording' + (r.note ? ' · ' + r.note : '') : st === 'done' ? 'last recording' + (r.note ? ' · ' + r.note : '') : 'idle');
    setText('rec-secs', r && (recording || st === 'done') ? secs.toFixed(1) + ' / ' + num(cap, 0) + ' s' : '—');
    var bar = $('rec-bar');
    bar.style.width = cap ? clamp(100 * secs / cap, 0, 100) + '%' : '0';
    bar.classList.toggle('rec-on', recording);
    var exc = x && x.kind && x.kind !== 'none' ? x.kind + (x.active ? ' · active' : ' · done') + ' · t ' + num(x.t || 0, 1) + ' s' : 'none';
    kv('rec-kv', [
      ['samples', r && r.samples != null ? r.samples.toLocaleString('en-US') : '—'],
      ['capacity', cap ? num(cap, 0) + ' s · ' + bytes(Math.round(cap / geom().dt) * 128) : '—'],
      ['dropped', r && r.dropped != null ? r.dropped : '—', r && r.dropped ? 'bad' : ''],
      ['excitation', exc, x && x.active ? 'accent' : '']
    ]);
    $('rec-start').disabled = !live || recording;
    $('rec-stop').disabled = !live || !(recording || (x && x.active));
    setText('rec-stop', recording ? 'Stop' : x && x.active ? 'Stop excitation' : 'Stop');
    $('rec-try').disabled = !live || recording || recEl.kind.value === 'none';
    $('rec-dl').disabled = !live || st !== 'done';
  }
  $('rec-start').addEventListener('click', function () {
    var secs = clamp(parseFloat(recEl.seconds.value) || preset.seconds, 1, 600);
    send({ record: { action: 'start', seconds: secs, note: $('rec-note').value.slice(0, 63), excite: excitation() } })
      .then(function () { flash('rec-msg', 'Recording ' + secs + ' s…', 'good'); }).catch(function (e) { flash('rec-msg', e.message, 'bad'); });
  });
  $('rec-stop').addEventListener('click', function () {
    var recording = get(S.status, 'record.state', '') === 'recording';
    send(recording ? { record: { action: 'stop' } } : { excite: null }).catch(function (e) { flash('rec-msg', e.message, 'bad'); });
  });
  $('rec-try').addEventListener('click', function () {
    var e = excitation();
    if (e) send({ excite: e }).then(function () { flash('rec-msg', 'Excitation on, not recording.', 'good'); }).catch(function (er) { flash('rec-msg', er.message, 'bad'); });
  });
  $('rec-dl').addEventListener('click', function () {
    fetch('api/record', { cache: 'no-store' }).then(function (r) {
      if (!r.ok) return r.json().catch(function () { return {}; }).then(function (j) { throw new Error(j.error || 'HTTP ' + r.status); });
      var cd = r.headers.get('Content-Disposition') || '', m = /filename="?([^";]+)"?/.exec(cd);
      return r.blob().then(function (b) { return { blob: b, name: m ? m[1] : null }; });
    }).then(function (d) {
      recDownloads++;
      var name = d.name || ((get(S.info, 'hostname', null) || 'cartpole') + '-' + recDownloads + '.ncpr');
      var a = document.createElement('a'), url = URL.createObjectURL(d.blob);
      a.href = url; a.download = name; document.body.appendChild(a); a.click(); a.remove();
      setTimeout(function () { URL.revokeObjectURL(url); }, 5000);
      flash('rec-msg', 'Saved ' + name + ' (' + bytes(d.blob.size) + ').', 'good', 6000);
    }).catch(function (e) { flash('rec-msg', 'download: ' + e.message, 'bad'); });
  });
  pickPreset(PRESETS[0]);

  /* ---- Tabs (Live / Health / Record / Settings) ----
     The URL hash selects a tab (#settings), the last choice is remembered, and a device in setup
     mode opens Settings on first load so the Wi-Fi form is the first thing seen. */
  var TABS = ['live', 'health', 'record', 'settings'];
  function tabFromHash() {
    var h = (location.hash || '').replace('#', '');
    return TABS.indexOf(h) >= 0 ? h : null;
  }
  function showTab(name) {
    if (TABS.indexOf(name) < 0) name = 'live';
    S.tab = name;
    TABS.forEach(function (t) {
      var on = t === name, b = $('tab-btn-' + t);
      b.setAttribute('aria-selected', on ? 'true' : 'false');
      b.tabIndex = on ? 0 : -1;
      $('tab-' + t).hidden = !on;
    });
    try { localStorage.setItem('tab', name); } catch (e) {}
    if (name === 'health' && S.events.newBad) {
      var bad = S.events.list.filter(function (e) { return e.level === 'error'; });
      S.events.seenBad = bad.length ? bad[bad.length - 1].seq : 0;
      S.events.newBad = false;
      healthAlert();
    }
  }
  function setTabAlert(name, kind) {
    var b = $('tab-btn-' + name);
    if (!b) return;
    if (kind) b.setAttribute('data-alert', kind); else b.removeAttribute('data-alert');
  }
  TABS.forEach(function (t, i) {
    var b = $('tab-btn-' + t);
    b.addEventListener('click', function () { if (location.hash !== '#' + t) location.hash = t; else showTab(t); });
    b.addEventListener('keydown', function (e) {
      var j = e.key === 'ArrowRight' ? i + 1 : e.key === 'ArrowLeft' ? i - 1 : e.key === 'Home' ? 0 : e.key === 'End' ? TABS.length - 1 : -1;
      if (j < 0) return;
      e.preventDefault();
      j = (j + TABS.length) % TABS.length;
      location.hash = TABS[j];
      $('tab-btn-' + TABS[j]).focus();
    });
  });
  window.addEventListener('hashchange', function () { var t = tabFromHash(); if (t) showTab(t); });
  (function () {
    var t = tabFromHash(), stored = null;
    if (!t) { try { stored = localStorage.getItem('tab'); } catch (e) {} }
    S.tabAuto = !t && TABS.indexOf(stored) < 0;
    showTab(t || (TABS.indexOf(stored) >= 0 ? stored : 'live'));
  })();

  /* ---- Boot ---- */
  buildGoalButtons();
  renderModel();
  renderControls(true);
  connect();
  requestAnimationFrame(frame);
})();
