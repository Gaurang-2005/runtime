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
let selected = null;
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
  const good = ["improved", "kept", "promoted", "done", "pass"];
  const bad = ["regressed", "invalid", "failed_to_execute", "reverted", "rejected", "fail"];
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
      api("GET", "/promotions"), api("GET", "/events"), api("GET", "/traffic"),
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
    el("td", { class: "num" }, String(iv.samples)))));
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
$("dpromote").addEventListener("click", async () => {
  await api("POST", `/interventions/${selected}/promote`, { by: "dashboard" })
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
  } catch { /* refresh() reports the connection state */ }
  clearInterval(timer);
  refresh();
  timer = setInterval(refresh, 2000);
}
start();
