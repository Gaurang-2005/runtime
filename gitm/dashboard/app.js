"use strict";
// gitm dashboard. Polls the server's API every 2 s; every value shown comes from it.
// Server data is only ever inserted as text (never innerHTML).

const $ = (id) => document.getElementById(id);
const store = {
  get(k) { try { return localStorage.getItem(k); } catch { return null; } },
  set(k, v) { try { localStorage.setItem(k, v); } catch { /* private window */ } },
};
let base = store.get("gitm.url") || "http://127.0.0.1:7070";
let token = store.get("gitm.token") || "";
// A link printed by `gitm run --server` carries #server=...&token=... . The
// fragment never leaves the browser; it is read once, stored, and cleared.
if (location.hash.length > 1) {
  const h = new URLSearchParams(location.hash.slice(1));
  if (h.get("server")) base = h.get("server");
  if (h.get("token")) token = h.get("token");
  store.set("gitm.url", base); store.set("gitm.token", token);
  history.replaceState(null, "", location.pathname);
}
let selected = null;
let chosenMode = null;        // what the operator clicked, before starting a session
let currentMode = "manual";   // what the server is in
let lastPreview = 0;
let timer = null;
const hist = { tps: [], ttft: [], rps: [] };
const HIST_N = 300; // 10 minutes at 2 s

function el(tag, attrs = {}, ...kids) {
  const e = document.createElement(tag);
  for (const [k, v] of Object.entries(attrs)) {
    if (k === "class") e.className = v;
    else if (k.startsWith("on")) e.addEventListener(k.slice(2), v);
    else e.setAttribute(k, v);
  }
  for (const k of kids) e.append(k instanceof Node ? k : document.createTextNode(k ?? ""));
  return e;
}
function fill(tbody, rows) { tbody.replaceChildren(...rows); }

async function api(method, path, body) {
  const r = await fetch(base.replace(/\/$/, "") + path, {
    method, headers: { Authorization: `Bearer ${token}`, "Content-Type": "application/json" },
    body: body === undefined ? undefined : JSON.stringify(body),
  });
  const data = await r.json().catch(() => ({}));
  if (!r.ok) throw new Error(data.detail || `HTTP ${r.status}`);
  return data;
}

const fmt = {
  n(v, d = 1) { return v == null || Number.isNaN(v) ? "–" : Number(v).toFixed(d); },
  ms(s) { return s == null ? "–" : `${(s * 1000).toFixed(0)}`; },
  pct(v) { return v == null ? "–" : `${v >= 0 ? "+" : ""}${(v * 100).toFixed(1)}%`; },
  t(ts) { return ts ? new Date(ts * 1000).toLocaleTimeString() : "–"; },
};

function pillFor(status) {
  const good = ["improved", "kept", "promoted", "done", "pass", "approved"];
  const bad = ["regressed", "invalid", "failed_to_execute", "reverted", "rejected", "fail", "declined"];
  const cls = good.includes(status) ? "good" : bad.includes(status) ? "bad" : "warn";
  return el("span", { class: `pill ${cls}` }, status);
}

function chart(svgId, valueId, series, unit) {
  const svg = $(svgId);
  const vals = series.filter((v) => v != null);
  $(valueId).textContent = vals.length ? `${fmt.n(vals[vals.length - 1])} ${unit}` : "";
  if (vals.length < 2) { svg.replaceChildren(); return; }
  const max = Math.max(...vals) * 1.1 || 1;
  const pts = series.map((v, i) => v == null ? null :
    `${(i / (HIST_N - 1)) * 300},${90 - (v / max) * 84 - 3}`).filter(Boolean).join(" ");
  const ns = "http://www.w3.org/2000/svg";
  const line = document.createElementNS(ns, "polyline");
  line.setAttribute("points", pts);
  line.setAttribute("fill", "none");
  line.setAttribute("stroke", getComputedStyle(svg).getPropertyValue(`--${svgId === "c1" ? "chart1" : svgId === "c2" ? "chart2" : "chart3"}`));
  line.setAttribute("stroke-width", "2");
  line.setAttribute("vector-effect", "non-scaling-stroke");
  svg.replaceChildren(line);
}

function push(arr, v) { arr.push(v); while (arr.length > HIST_N) arr.shift(); }

async function refresh() {
  try {
    const [status, stats, ivs, promos, events, sims] = await Promise.all([
      api("GET", "/status"), api("GET", "/stats?window_s=30"), api("GET", "/interventions"),
      api("GET", "/promotions"), api("GET", "/events"),
      // Optional panel: a server without a simulator should not blank the page.
      api("GET", "/traffic").catch(() => []),
    ]);
    $("connstate").textContent = "connected";
    $("connstate").className = "muted";
    $("model").textContent = `${status.model} · ${status.endpoint}`;
    renderStats(stats, status);
    renderEngines(status);
    renderInterventions(ivs);
    renderPromotions(promos);
    renderEvents(events);
    renderSims(sims);
    renderMode(status);
    renderTraffic(status.traffic);
    renderQueue(status.queue || []);
    if (Date.now() - lastPreview > 10000) { lastPreview = Date.now(); loadPreview(); loadProfile(); }
    $("pause").textContent = status.paused ? "Resume" : "Pause";
    $("revert").disabled = !status.probation;
    if (selected) renderDetail(await api("GET", `/interventions/${selected}`));
  } catch (e) {
    $("connstate").textContent = `not connected: ${e.message}`;
    $("connstate").className = "err";
  }
}

function renderStats(s, status) {
  $("k-tps").textContent = fmt.n(s.output_tokens_per_s);
  $("k-rps").textContent = fmt.n(s.requests_per_s, 2);
  $("k-ttft").textContent = `${fmt.ms(s.ttft_p50_s)} / ${fmt.ms(s.ttft_p99_s)} ms`;
  $("k-lat").textContent = `${fmt.n(s.latency_p50_s, 2)} / ${fmt.n(s.latency_p99_s, 2)} s`;
  $("k-err").textContent = String(s.errors);
  $("k-rec").textContent = String(status.proxy?.recorded ?? "–");
  push(hist.tps, s.output_tokens_per_s);
  push(hist.ttft, s.ttft_p99_s == null ? null : s.ttft_p99_s * 1000);
  push(hist.rps, s.requests_per_s);
  chart("c1", "c1v", hist.tps, "tok/s");
  chart("c2", "c2v", hist.ttft, "ms");
  chart("c3", "c3v", hist.rps, "req/s");
}

function renderEngines(status) {
  const cards = Object.values(status.engines).map((e) => el("div", { class: "engine" },
    el("div", { class: "row" }, el("span", { class: "role" }, e.role),
      el("span", { class: `pill ${e.alive ? "good" : "bad"}` }, e.alive ? "up" : "down"),
      el("span", { class: "muted" }, e.url)),
    el("div", { class: "mono" }, (e.serve_args || []).join(" ") || "(vLLM defaults)"),
    el("div", { class: "muted" }, `GPUs ${e.gpus ?? "all"} · mem ${e.mem_fraction ?? "default"} · ` +
      `start ${e.startup_s ?? "–"} s`)));
  $("engines").replaceChildren(...cards);
  const p = status.probation;
  $("probation").replaceChildren(p ? el("div", {},
    el("strong", {}, `Probation: ${p.intervention}`),
    el("span", { class: "muted" }, ` ends ${fmt.t(p.ends)} · old engine decoded ` +
      `${fmt.n(p.baseline_decode_tok_s)} tok/s per request · ` +
      `last: ${fmt.n(p.observations.at(-1)?.decode_tok_s)} tok/s, ` +
      `${p.observations.at(-1)?.requests ?? 0} requests`)) : "");
}

function renderInterventions(ivs) {
  fill($("ivs"), ivs.slice().reverse().map((iv) => el("tr", { class: "click", onclick: () => select(iv.id) },
    el("td", { class: "mono" }, iv.id), el("td", { class: "mono" }, JSON.stringify(iv.knobs)),
    el("td", {}, pillFor(iv.status)), el("td", { class: "num" }, fmt.pct(iv.effect)),
    el("td", { class: "num" }, iv.interval ? `${fmt.pct(iv.interval[0])} … ${fmt.pct(iv.interval[1])}` : "–"),
    el("td", { class: "num" }, String(iv.samples)),
    el("td", { class: "mono muted" }, iv.traffic || "–"))));
}

function select(id) { selected = id; $("detail").hidden = false; refresh(); }

function renderDetail(iv) {
  $("dtitle").textContent = `Intervention ${iv.id} · ${JSON.stringify(iv.knobs)}`;
  $("ddetail").textContent = `${iv.status}${iv.detail ? " — " + iv.detail : ""}`;
  fill($("dsamples"), (iv.progress || []).map((p) => el("tr", {},
    el("td", { class: "num" }, String(p.index)), el("td", {}, p.arm),
    el("td", { class: "num" }, fmt.n(p.mean_tpot_ms, 2)), el("td", { class: "num" }, fmt.n(p.output_throughput)),
    el("td", { class: "num" }, fmt.n(p.p99_ttft_ms)), el("td", { class: "num" }, fmt.n(p.p99_itl_ms)),
    el("td", { class: "num" }, fmt.n(p.correctness, 3)), el("td", { class: "num" }, fmt.n(p.stable_rps, 2)))));
  fill($("drules"), (iv.rules || []).map((r) => el("tr", {},
    el("td", { class: "mono" }, r.name), el("td", {}, pillFor(r.passed ? "pass" : "fail")),
    el("td", { class: "muted" }, r.detail))));
  $("dpromote").disabled = iv.status !== "awaiting_approval";
  $("ddecline").disabled = !["awaiting_approval", "approved"].includes(iv.status);
  $("dstop").disabled = !["queued", "running"].includes(iv.status);
}

function renderPromotions(p) {
  fill($("promos"), p.promotions.slice().reverse().map((x) => el("tr", {},
    el("td", {}, fmt.t(x.t)), el("td", { class: "mono" }, JSON.stringify(x.knobs)),
    el("td", {}, x.outcome))));
}

function renderEvents(evs) {
  fill($("events"), evs.slice(-60).reverse().map((e) => {
    const { t, kind, ...rest } = e;
    delete rest.argv; delete rest.serve_args;
    return el("tr", {}, el("td", { class: "muted" }, fmt.t(t)), el("td", { class: "mono" }, kind),
      el("td", { class: "mono" }, JSON.stringify(rest)));
  }));
}

function renderSims(sims) {
  fill($("sims"), sims.slice().reverse().map((s) => el("tr", {},
    el("td", { class: "mono" }, s.id), el("td", {}, s.preset), el("td", { class: "num" }, fmt.n(s.rate_rps, 2)),
    el("td", { class: "num" }, `${s.sent} / ${s.planned}`), el("td", { class: "num" }, String(s.ok)),
    el("td", { class: "num" }, String(s.failed)), el("td", {}, pillFor(s.status)),
    el("td", {}, s.status === "running" ? el("button", { class: "ghost",
      onclick: () => api("POST", `/traffic/${s.id}/stop`).then(refresh) }, "Stop") : ""))));
}

const MODE_TEXT = {
  manual: "Tests only what you queue. A win waits for your approval.",
  assisted: "gitm also picks what to test, within the limits below. Every win waits for your approval.",
  autonomous: "gitm picks, tests and promotes by itself, within the limits below and every promotion rule.",
};

function renderMode(st) {
  const mode = st.mode || "manual", sess = st.session;
  currentMode = mode;
  const shown = sess ? mode : (chosenMode || mode);
  for (const b of document.querySelectorAll("#modeseg button")) b.classList.toggle("on", b.dataset.mode === shown);
  $("modenow").textContent = sess ? `${mode} · running` : "manual";
  $("modenow").className = `pill ${sess ? "good" : ""}`;
  $("modedesc").textContent = MODE_TEXT[shown];
  $("sessform").hidden = !!sess || shown === "manual";
  $("s-start").textContent = `Start ${shown}`;
  $("sessstops").hidden = !sess;
  if (!sess) { $("sessstat").replaceChildren(); return; }
  const left = sess.remaining_s == null ? "no time limit" : `${Math.floor(sess.remaining_s / 3600)}h ${Math.floor(sess.remaining_s % 3600 / 60)}m left`;
  const tests = `${sess.tests_started}${sess.max_tests ? " / " + sess.max_tests : ""} tests started`;
  const sched = sess.schedule.length ? `tests only ${sess.schedule.map((w) => w.join("–")).join(", ")} h` +
    (sess.in_schedule ? " (now: inside)" : " (now: outside, waiting)") : "any time";
  const scope = [sess.allow_knobs.length ? `only ${sess.allow_knobs.join(", ")}` : "",
                 sess.deny_knobs.length ? `never ${sess.deny_knobs.join(", ")}` : "",
                 sess.tiers ? `tiers ${sess.tiers.join(", ")}` : ""].filter(Boolean).join(" · ");
  $("sessstat").replaceChildren(el("div", {}, `${left} · ${tests} · ${sched}`),
    scope ? el("div", { class: "muted" }, scope) : "");
}

function renderTraffic(t) {
  if (!t || !t.current) { return; }
  const row = (label, k, fmtf = (v) => String(v)) => [el("span", { class: "h" }, label),
    el("span", {}, t.validated_on ? fmtf(t.validated_on[k]) : "–"), el("span", {}, fmtf(t.current[k]))];
  $("traffickv").replaceChildren(
    el("span", { class: "h" }, ""), el("span", { class: "h" }, "stable validated on"), el("span", { class: "h" }, "now"),
    ...row("requests / s", "rate_rps", (v) => fmt.n(v, 2)), ...row("prompt p50 (tokens)", "input_p50"),
    ...row("answer p50 (tokens)", "output_p50"), ...row("prompt : answer", "io_ratio", (v) => fmt.n(v, 1)),
    ...row("burstiness", "burstiness", (v) => fmt.n(v, 1)));
  const session = currentMode !== "manual";
  const settle = t.settling ? ` Watching for it to hold steady (${t.settling.windows} of ${t.settling.of} windows).` : "";
  $("trafficbanner").replaceChildren(t.changed ? el("div", { class: "banner" },
    `Traffic changed at ${fmt.t(t.changed_at)}: results measured on the old traffic no longer count as tried. ` +
    (session ? "gitm is re-testing for the new traffic: it promotes anything better, or marks stable as validated on it once nothing beats it."
             : "Start Assisted or Autonomous to re-test stable on it.") + settle) : "");
  $("trafficdist").textContent = t.distance == null ? "" :
    `change detection: distance ${fmt.n(t.distance, 2)} from the reference (a change above ${fmt.n(t.threshold, 1)}; 1.0 = some axis 2×)`;
  const vd = t.validated_distance;
  $("trafficvalid").textContent = t.validated_how ?
    `Stable validated: ${t.validated_how}, ${fmt.t(t.validated_at)}` +
    (vd != null && vd > t.threshold ? ` · current traffic differs from it (distance ${fmt.n(vd, 2)}): not yet validated` : "") : "";
}

function renderQueue(q) {
  if (!q.length) { fill($("queue"), [el("tr", {}, el("td", { colspan: "6", class: "muted" }, "empty"))]); return; }
  const act = (label, fn, title) => el("button", { class: "ghost mini", title, onclick: fn }, label);
  fill($("queue"), q.map((item, i) => el("tr", {},
    el("td", { class: "num" }, String(i + 1)), el("td", { class: "mono" }, item.id),
    el("td", { class: "mono" }, JSON.stringify(item.knobs)), el("td", {}, item.submitter),
    el("td", { class: "muted" }, item.waits || ""),
    el("td", {}, el("span", { class: "row", style: "margin:0;flex-wrap:nowrap" },
      act("↑", () => moveTo(item.id, Math.max(0, item.position - 1)), "Move up"),
      act("↓", () => moveTo(item.id, item.position + 1), "Move down"),
      act("Run next", () => api("POST", `/queue/${item.id}/front`).then(refresh), "Pin to the front"),
      act("Remove", () => api("POST", `/interventions/${item.id}/stop`).then(refresh), "Remove from the queue"))))));
}

async function moveTo(id, position) {
  await api("POST", `/queue/${id}/move`, { position }).catch(() => {});
  refresh();
}

async function loadPreview() {
  try {
    const p = await api("GET", "/explore/preview?n=6");
    $("preview").textContent = p.length ? p.map((c) => JSON.stringify(c.knobs)).join("  ·  ") :
      "nothing left to try for this setup and traffic";
  } catch (e) { $("preview").textContent = e.message; }
}

async function loadProfile() {
  let v;
  try { v = await api("GET", "/profile"); } catch (e) { $("profstate").textContent = e.message; return; }
  const p = v.profile, a = p && p.analysis;
  const last = v.last_attempt;
  let state = "no profile yet";
  if (v.running) state = `profiling now (${v.running}): the canary is traced, tests wait`;
  else if (v.requested) state = `profile requested: runs before the next test`;
  else if (p) state = `profiled ${fmt.t(p.finished)} · ${a.kernels.toLocaleString()} kernels · ` +
    (v.stale ? `not in use: ${v.stale}` : "in use: the explorer ranks by it");
  if (last && last.status === "failed" && !v.running) state += ` · last attempt failed: ${last.detail}`;
  $("profstate").textContent = state;
  $("profnow").disabled = Boolean(v.running || v.requested);
  if (!a) {
    $("profops").replaceChildren(); $("profbasis").textContent = ""; $("profcauses").textContent = "–";
  } else {
    const top = a.ops.slice(0, 8), max = Math.max(...top.map((o) => o.share), 0.0001);
    $("profops").replaceChildren(...top.flatMap((o) => [
      el("span", { class: "mono" }, o.name),
      el("span", { class: "track" }, el("span", { class: "fill", style: `width:${(100 * o.share / max).toFixed(1)}%;display:block` })),
      el("span", { class: "v" }, `${(o.share * 100).toFixed(1)}%`)]));
    const g = a.graph;
    $("profbasis").textContent = `${(a.labelled_share * 100).toFixed(0)}% of GPU time named by layer · batch ${g.batch} (${g.batch_source}) · ` +
      (g.floors_priced ? `floors priced for ${g.hardware}` : `floors not priced: ${g.not_priced_because}, so nothing is ruled out for being at its floor`);
    $("profcauses").replaceChildren(...(a.causes.length ? a.causes.map((c) =>
      el("div", { style: "margin-bottom:4px" }, el("span", { class: "pill warn" }, c.signal), " ", c.note))
      : [document.createTextNode("no scheduler pressure in the window")]));
  }
  fill($("profnext"), (v.next || []).filter((c) => c.knobs).map((c, i) => el("tr", {},
    el("td", { class: "num" }, String(i + 1)), el("td", { class: "mono" }, JSON.stringify(c.knobs)),
    el("td", { class: "muted" }, c.why || (a && !v.stale ? "not ranked by the profile; catalogue order" : "catalogue order")))));
  fill($("profout"), (v.ruled_out || []).map((c) => el("tr", {},
    el("td", { class: "mono" }, JSON.stringify(c.knobs)), el("td", { class: "muted" }, c.reason))));
}

$("profnow").addEventListener("click", async () => {
  $("profnow").disabled = true;
  try { await api("POST", "/profile"); } catch (e) { $("profstate").textContent = e.message; }
  loadProfile();
});

function showTestSettings(p) {
  $("t-reps").value = p.repetitions; $("t-settle").value = p.settle_s; $("t-speed").value = p.replay_speed;
  // One test: 2 arms x samples, each = settle + replay (assume ~40 s of recorded
  // traffic at real speed) + ~3 s probe, plus ~5 canary restarts of ~35 s.
  const per = p.settle_s + 40 / p.replay_speed + 3;
  $("t-est").textContent = `≈ ${Math.round((2 * p.repetitions * per + 5 * 35) / 60)} min per test on a ~40 s recording`;
  api("GET", "/policy/warnings").then((w) => { $("t-warn").textContent = w.join(" "); }).catch(() => {});
}

async function loadPresets() {
  const presets = await api("GET", "/traffic/presets");
  $("preset").replaceChildren(...Object.keys(presets).map((k) => el("option", { value: k }, k)));
  const describe = () => {
    const p = presets[$("preset").value];
    $("presetnote").textContent = `${p.system_words} shared words · user ${p.user_words.join("–")} words · ` +
      `answers ${p.max_tokens.join("–")} tokens · default ${p.rate_rps} req/s, burst ${p.burstiness}`;
  };
  $("preset").onchange = describe;
  describe();
}

$("conn").addEventListener("submit", (ev) => {
  ev.preventDefault();
  base = $("url").value.trim() || base;
  token = $("token").value.trim();
  store.set("gitm.url", base); store.set("gitm.token", token);
  start();
});
$("simform").addEventListener("submit", async (ev) => {
  ev.preventDefault();
  const body = { preset: $("preset").value, duration_s: Number($("dur").value || 120) };
  if ($("rate").value) body.rate_rps = Number($("rate").value);
  if ($("burst").value) body.burstiness = Number($("burst").value);
  try { await api("POST", "/traffic/simulate", body); } catch (e) { $("presetnote").textContent = e.message; }
  refresh();
});
$("ivform").addEventListener("submit", async (ev) => {
  ev.preventDefault();
  $("iverr").textContent = "";
  let knobs;
  try { knobs = JSON.parse($("knobs").value); } catch { $("iverr").textContent = "knobs must be JSON"; return; }
  try {
    const iv = await api("POST", "/interventions", { knobs, note: $("note").value, submitter: "dashboard" });
    select(iv.id);
  } catch (e) { $("iverr").textContent = e.message; }
});
$("pause").addEventListener("click", async () => {
  const paused = $("pause").textContent === "Resume";
  await api("POST", paused ? "/resume" : "/pause").catch(() => {});
  refresh();
});
$("revert").addEventListener("click", async () => {
  if (!$("revert").dataset.armed) {   // two clicks, no confirm() dialog
    $("revert").dataset.armed = "1"; $("revert").textContent = "Click again to revert";
    setTimeout(() => { delete $("revert").dataset.armed; $("revert").textContent = "Revert promotion"; }, 4000);
    return;
  }
  await api("POST", "/revert", { reason: "reverted from the dashboard" }).catch((e) => { $("ctlnote").textContent = e.message; });
  refresh();
});
$("approval").addEventListener("change", async () => {
  await api("PUT", "/policy", { require_approval: $("approval").checked }).catch(() => {});
});
for (const b of document.querySelectorAll("#modeseg button")) {
  b.addEventListener("click", async () => {
    const st = await api("GET", "/autonomy").catch(() => null);
    if (st && st.session && b.dataset.mode !== st.mode) {
      $("modedesc").textContent = b.dataset.mode === "manual" ? "Use Stop below to go back to manual." :
        `A ${st.mode} session is running. Stop it first to start ${b.dataset.mode}.`;
      return;
    }
    chosenMode = b.dataset.mode;
    refresh();
  });
}
$("sessform").addEventListener("submit", async (ev) => {
  ev.preventDefault();
  $("s-err").textContent = "";
  const num = (id) => ($(id).value.trim() === "" ? null : Number($(id).value));
  const globs = (id) => $(id).value.split(",").map((x) => x.trim()).filter(Boolean);
  const from = num("s-from"), to = num("s-to");
  const tiers = [$("s-low").checked && "low_risk", $("s-mod").checked && "moderate"].filter(Boolean);
  const body = { mode: chosenMode, budget_h: num("s-hours"), max_tests: num("s-tests"),
    schedule: from != null && to != null ? [[from, to]] : null,
    allow_knobs: globs("s-allow"), deny_knobs: globs("s-deny"), tiers };
  try { await api("POST", "/autonomy/start", body); chosenMode = null; } catch (e) { $("s-err").textContent = e.message; }
  refresh();
});
$("s-stop").addEventListener("click", () => api("POST", "/autonomy/stop", { cancel_current: false }).then(refresh));
$("s-stopnow").addEventListener("click", () => api("POST", "/autonomy/stop", { cancel_current: true }).then(refresh));
$("testform").addEventListener("submit", async (ev) => {
  ev.preventDefault();
  $("t-err").textContent = "";
  try {
    const p = await api("PUT", "/policy", { repetitions: Number($("t-reps").value),
      settle_s: Number($("t-settle").value), replay_speed: Number($("t-speed").value) });
    showTestSettings(p);
  } catch (e) { $("t-err").textContent = e.message; }
});

$("dpromote").addEventListener("click", async () => {
  await api("POST", `/interventions/${selected}/promote`, { by: "dashboard" })
    .catch((e) => { $("ddetail").textContent = e.message; });
  refresh();
});
$("ddecline").addEventListener("click", async () => {
  await api("POST", `/interventions/${selected}/decline`, { by: "dashboard" })
    .catch((e) => { $("ddetail").textContent = e.message; });
  refresh();
});
$("dstop").addEventListener("click", async () => {
  await api("POST", `/interventions/${selected}/stop`).catch(() => {});
  refresh();
});

async function start() {
  $("url").value = base; $("token").value = token;
  if (!token) { $("connstate").textContent = "enter the server URL and api token"; return; }
  try {
    await loadPresets();
    const policy = await api("GET", "/policy");
    $("approval").checked = !!policy.require_approval;
    showTestSettings(policy);
  } catch { /* refresh() reports the connection state */ }
  clearInterval(timer);
  refresh();
  timer = setInterval(refresh, 2000);
}
start();
