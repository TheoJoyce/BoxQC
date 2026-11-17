// Training page: start and follow training runs on this computer, and manage models.
// Training runs inside the studio process (see boxqc/studio/training/jobs.py); the
// device is only involved at the end, when the finished model is deployed to it.
"use strict";

const POLL_MS = 2000;
const ACTIVE = ["queued", "preparing", "training", "exporting", "evaluating", "deploying"];
const STATE_TEXT = {
  queued: "starting", preparing: "preparing the dataset", training: "training",
  exporting: "exporting the model", evaluating: "evaluating on validation images",
  deploying: "deploying to the device", done: "done", failed: "failed", cancelled: "cancelled",
};
const SERIES = ["var(--series-1)", "var(--series-2)"];

// The page polls GET /api/training every POLL_MS and re-renders from the answer
// (`overview`): this computer, the trainers, the active run and the recent runs.
let overview = null;
let selectedId = null;      // null = follow the active / latest run
let formFilled = false;
const finishedCache = new Map();

const $ = (id) => document.getElementById(id);
const fmt = (v, d = 3) => (v === undefined || v === null ? "–" : (+v).toFixed(d));
const isActive = (job) => job && ACTIVE.includes(job.state);

// CSS classes for a run-state badge (colours reuse the dataset status badges).
function badgeClass(state) {
  if (state === "done") return "badge labeled";
  if (state === "failed") return "badge defect";
  if (state === "cancelled") return "badge skip";
  return "badge unlabeled";
}

// Seconds as "42s", "3m 10s" or "1h 05m"-style text.
function duration(s) {
  s = Math.max(0, Math.round(s));
  if (s < 60) return `${s}s`;
  const m = Math.floor(s / 60);
  return m < 60 ? `${m}m ${s % 60}s` : `${Math.floor(m / 60)}h ${m % 60}m`;
}

// ---- this computer and its trainers ----------------------------------------
// A trainer is a training backend (overview.trainers, from the studio's registry).
function currentTrainer() {
  const name = $("train-form").trainer.value || overview.default_trainer;
  return overview.trainers.find((t) => t.name === name) || overview.trainers[0] || null;
}

// The "This computer" card: can it train, and with what.
function renderHost() {
  const o = overview, h = o.host || {}, t = currentTrainer();
  const ready = Boolean(t && t.available) && !h.error;
  $("host-state").textContent = ready ? "ready to train" : "cannot train yet";
  $("host-state").className = ready ? "badge labeled" : "badge defect";
  $("host-help").hidden = !t || t.available;
  const rows = [
    ["Computer", h.hostname],
    ["Trainer", t ? t.label : null],
    ["Compute device", h.device],
    ["GPU", h.gpu],
    ["PyTorch", h.torch],
    ["System", h.platform],
    ["Problem", h.error || (t && t.unavailable_reason)],
  ].filter(([, v]) => v);
  $("host").replaceChildren(...rows.map(([k, v]) => el("tr", {}, el("th", {}, k), el("td", {}, String(v)))));
}

// ---- form ----------------------------------------------------------------
function setOptions(select, options, value) {
  select.replaceChildren(...options.map((o) => el("option", { value: o.value }, o.label)));
  if (value !== undefined && value !== null && !options.some((o) => o.value === String(value))) {
    select.append(el("option", { value }, value));   // a value from boxqc.yaml the trainer doesn't list
  }
  if (value !== undefined && value !== null) select.value = String(value);
}

// Base models and compute devices depend on the chosen trainer.
function fillTrainerOptions(keep = true) {
  const f = $("train-form"), d = overview.defaults, t = currentTrainer();
  if (!t) return;
  const isDefault = t.name === overview.default_trainer;
  setOptions(f.base_model, t.base_models, keep && isDefault ? d.base_model : (t.base_models[0] || {}).value);
  setOptions(f.device, t.devices, keep && isDefault && t.devices.some((o) => o.value === d.device) ? d.device : "auto");
}

// Fill the form once with the defaults from boxqc.yaml; later polls leave the user's edits alone.
function fillForm() {
  if (formFilled) return;
  const f = $("train-form"), d = overview.defaults;
  setOptions(f.trainer, overview.trainers.map((t) => ({ value: t.name, label: t.label })), overview.default_trainer);
  $("trainer-field").hidden = overview.trainers.length < 2;
  f.epochs.value = d.epochs;
  f.batch.value = d.batch;
  if (![...f.imgsz.options].some((o) => o.value === String(d.imgsz))) f.imgsz.append(el("option", {}, d.imgsz));
  f.imgsz.value = String(d.imgsz);
  fillTrainerOptions();
  formFilled = true;
}
$("train-form").trainer.addEventListener("change", () => { fillTrainerOptions(false); renderHost(); renderStartButton(); });
$("train-form").deploy.addEventListener("change", (e) => { $("train-form").activate.disabled = !e.target.checked; });

// Enable "Start training" only when a run can start, and say why not otherwise.
function renderStartButton() {
  const o = overview, t = currentTrainer();
  let note = "";
  if (!t) note = "No trainer is installed.";
  else if (!t.available) note = t.unavailable_reason;
  else if (o.active) note = "A run is in progress.";
  else if (datasetEmpty) note = "The chosen dataset has no labeled images.";
  $("start").disabled = Boolean(note);
  $("start-note").textContent = note;
}

// Training uses the labeled images of one named dataset, or every labeled image.
let datasetSummary = null;
const labeledCount = (st) => st.labeled_clean + st.labeled_defective;

async function loadDatasetStats() {
  datasetSummary = await api("/api/datasets");
  const sel = $("train-dataset");
  fillDatasetSelect(sel, datasetSummary, {
    value: sel.value || datasetSummary.capture_dataset || "", all: true, allLabel: "All labeled images",
    count: (st) => `${labeledCount(st)} labeled` });
  renderDatasetStats();
}

function renderDatasetStats() {
  const name = $("train-dataset").value;
  const st = name ? datasetSummary.datasets.find((d) => d.name === name).stats : datasetSummary.all;
  $("dataset-stats").replaceChildren(
    el("div", { class: "stat" }, el("b", {}, st.labeled_defective), el("span", {}, "labeled defective")),
    el("div", { class: "stat" }, el("b", {}, st.labeled_clean), el("span", {}, "labeled clean")),
    el("div", { class: "stat" }, el("b", {}, st.by_status.unlabeled), el("span", {}, "unlabeled (not used)")),
  );
  const n = labeledCount(st);
  $("dataset-note").textContent = !n ? "No labeled images in this dataset yet."
    : name ? `Only images in "${name}" are used; everything else is left out.`
      : `Every labeled image is used, including ${datasetSummary.unassigned.total ? "images in no dataset" : "all datasets"}.`;
  datasetEmpty = n === 0;
  if (overview) renderStartButton();
}
let datasetEmpty = false;
$("train-dataset").addEventListener("change", renderDatasetStats);

$("train-form").addEventListener("submit", async (e) => {
  e.preventDefault();
  const f = e.target;
  const body = {
    epochs: +f.epochs.value, imgsz: +f.imgsz.value, batch: +f.batch.value,
    base_model: f.base_model.value, device: f.device.value, name: f.name.value.trim(),
    trainer: f.trainer.value, deploy: f.deploy.checked, activate: f.deploy.checked && f.activate.checked,
    dataset: $("train-dataset").value,
  };
  $("start").disabled = true;
  try {
    const job = await api("/api/training/jobs", { method: "POST", json: body });
    selectedId = null;
    toast(`Started run ${job.id}`, "ok");
    await poll();
  } catch (err) {
    toast(err.message, "error");
    renderStartButton();
  }
});

// ---- runs list -------------------------------------------------------------
function renderJobs() {
  const jobs = overview.jobs;
  if (!jobs.length) {
    $("jobs").replaceChildren(el("tr", {}, el("td", { class: "muted" }, "None yet.")));
    return;
  }
  const shown = currentId();
  $("jobs").replaceChildren(...jobs.map((j) => el("tr", {
    class: "clickable" + (j.id === shown ? " on" : ""),
    onclick: () => { selectedId = isActive(j) ? null : j.id; poll(); },
  },
  el("td", { class: "small" }, j.created.slice(5, 16).replace("T", " ")),
  el("td", { class: "small" }, j.params.name || `${j.params.epochs} ep · ${j.params.base_model.replace(".pt", "")}`,
    el("div", { class: "muted" }, j.params.dataset || "all labeled")),
  el("td", {}, el("span", { class: badgeClass(j.state) }, j.state)),
  el("td", { class: "small muted" }, j.mAP50 !== null && j.mAP50 !== undefined ? `mAP50 ${fmt(j.mAP50, 2)}` : ""))));
}

// The run to show: the one clicked in the list, else the active one, else the latest.
function currentId() {
  if (selectedId) return selectedId;
  if (overview.active) return overview.active.id;
  return overview.jobs.length ? overview.jobs[0].id : null;
}

// Full details of a run. The active run comes with the overview; finished runs never
// change again, so they are fetched once and cached.
async function loadJob(id) {
  if (overview.active && overview.active.id === id) return overview.active;
  if (finishedCache.has(id)) return finishedCache.get(id);
  const job = await api(`/api/training/jobs/${id}`);
  if (!isActive(job)) finishedCache.set(id, job);
  return job;
}

// ---- run view ------------------------------------------------------------
// The run panel: title, parameters, progress bar with time estimate, stop buttons,
// then result, charts and log.
function renderRun(job) {
  $("run-panel").hidden = !job;
  if (!job) return;
  const p = job.params;
  $("run-title").textContent = p.name || `Run ${job.id}`;
  $("run-state").textContent = job.stop_request ? `stopping (${job.stop_request})` : job.state;
  $("run-state").className = badgeClass(job.state);
  $("run-params").textContent = [`dataset: ${p.dataset || "all labeled images"}`, `${p.epochs} epochs`, p.base_model, `${p.imgsz}px`,
    `batch ${p.batch}`, `device ${p.device}`, job.host ? `on ${job.host}` : null,
    !p.deploy ? "keep on this computer" : p.activate ? "deploy & activate when done" : "deploy when done",
    `started ${(job.started || job.created).slice(11, 19)}`]
    .filter(Boolean).join(" · ");

  // progress
  const epochs = job.epochs || p.epochs;
  const last = job.history[job.history.length - 1];
  let pct = 0;
  if (job.state === "training") pct = (job.epoch / epochs) * 100;
  else if (["exporting", "evaluating", "deploying", "done"].includes(job.state)) pct = 100;
  else if (!isActive(job)) pct = (job.epoch / epochs) * 100;
  $("progress-bar").style.width = `${pct.toFixed(1)}%`;
  $("progress-bar").className = "progress-bar " + (job.state === "failed" ? "bad" : job.state === "done" ? "good" : "");
  const epochText = job.epoch ? `Epoch ${job.epoch} / ${epochs}` : `0 / ${epochs} epochs`;
  $("progress-text").textContent = `${epochText} · ${STATE_TEXT[job.state] || job.state}`;
  let eta = "";
  if (job.state === "training" && last && last.elapsed_s && job.epoch < epochs) {
    eta = `~${duration((last.elapsed_s / job.epoch) * (epochs - job.epoch))} left · ${duration(last.elapsed_s)} elapsed`;
  } else if (last && last.elapsed_s) {
    eta = `${duration(last.elapsed_s)} of training`;
  }
  $("progress-eta").textContent = eta;

  // controls
  const active = isActive(job);
  $("run-controls").hidden = !active;
  $("stop-early").hidden = job.state !== "training";
  $("stop-early").disabled = $("cancel").disabled = Boolean(job.stop_request);
  $("stop-early").onclick = () => stopRun(job.id, "early");
  $("cancel").onclick = () => {
    if (confirm("Cancel this run? No model will be kept.")) stopRun(job.id, "cancel");
  };
  $("run-error").hidden = !job.error;
  $("run-error").textContent = job.error || "";

  renderResult(job);
  renderCharts(job, epochs);
  renderLog(job);
}

async function stopRun(id, mode) {
  try {
    await api(`/api/training/jobs/${id}/stop`, { method: "POST", json: { mode } });
    toast(mode === "early" ? "Stopping after this epoch…" : "Cancelling…", "ok");
    poll();
  } catch (e) { toast(e.message, "error"); }
}

// Result of a finished run: where the model ended up, its scores and the threshold table.
function renderResult(job) {
  const r = job.result;
  $("result").hidden = !r;
  if (!r) return;
  const m = r.metrics || {}, ev = r.evaluation || {}, at = ev.at_recommended;
  const ds = r.dataset || {};
  // `r.active` and `r.deployed` are what was true when the run finished. Since then the
  // device may have been switched to another model, so prefer the live model list.
  const name = r.device_model || r.model;
  const live = deviceOnline ? models.find((x) => x.name === name) : null;
  const active = live ? live.active : deviceOnline ? false : r.active;
  const deployed = live ? live.on_device : deviceOnline ? false : r.deployed;
  const where = active ? "active on the device" : deployed ? "on the device (not active)"
    : r.deploy_error ? "on this computer (deploy failed)" : "on this computer (not deployed)";
  const stats = [
    [modelLabel(name) + (active ? " ✓" : ""), where],
    [fmt(m.mAP50), "mAP50"],
    [fmt(m.recall, 2), "recall"],
    [r.recommended_threshold ?? "–", "recommended threshold"],
  ];
  if (at) stats.push([`${at.fn}/${at.fn + at.tp}`, "defective missed (val)"], [`${at.fp}/${at.fp + at.tn}`, "false alarms (val)"]);
  if (ds.train) stats.push([`${ds.train.images} + ${ds.val.images}`, "train + val images"]);
  $("result-stats").replaceChildren(...stats.map(([v, k]) =>
    el("div", { class: "stat" }, el("b", { class: String(v).length > 10 ? "long" : "" }, v), el("span", {}, k))));

  const btn = $("apply-thr");
  btn.hidden = !r.recommended_threshold || !active;     // never offer it for a model that is not inspecting
  btn.textContent = `Use threshold ${r.recommended_threshold} on the device`;
  btn.onclick = async () => {
    try {
      await api(`${DEVICE}/settings`, { method: "PUT", json: { conf_threshold: r.recommended_threshold } });
      toast(`Threshold set to ${r.recommended_threshold}`, "ok");
    } catch (e) { toast(e.message, "error"); }
    loadModels();
  };
  $("eval-table").replaceChildren(...(ev.table || []).map((row) =>
    el("tr", { class: at && row.threshold === at.threshold ? "on" : "" },
      el("td", {}, row.threshold.toFixed(2) + (at && row.threshold === at.threshold ? " (recommended)" : "")),
      el("td", {}, `${row.fn}/${row.tp + row.fn}`),
      el("td", {}, `${row.fp}/${row.fp + row.tn}`),
      el("td", {}, `${(row.accuracy * 100).toFixed(1)}%`))));
}

// The log, kept scrolled to the bottom unless the user has scrolled up to read.
function renderLog(job) {
  const pre = $("log");
  const atBottom = pre.scrollHeight - pre.scrollTop - pre.clientHeight < 30;
  const text = job.log.join("\n");
  if (pre.textContent !== text) {
    pre.textContent = text || "(no output yet)";
    if (atBottom) pre.scrollTop = pre.scrollHeight;
  }
}

// ---- charts --------------------------------------------------------------
function renderCharts(job, epochs) {
  const h = job.history;
  $("charts").hidden = $("epoch-details").hidden = !h.length;
  if (!h.length) return;
  lineChart($("chart-map"), h, epochs, [
    { key: "mAP50", label: "mAP50" },
    { key: "mAP50-95", label: "mAP50-95" },
  ], { run: job.id, yMax: 1, digits: 3 });
  lineChart($("chart-loss"), h, epochs, [
    { key: "box_loss", label: "box" },
    { key: "cls_loss", label: "class" },
  ], { run: job.id, digits: 3 });
  const rows = [...h].reverse().map((e) => el("tr", {},
    el("td", {}, e.epoch), el("td", {}, fmt(e.mAP50)), el("td", {}, fmt(e["mAP50-95"])),
    el("td", {}, fmt(e.precision)), el("td", {}, fmt(e.recall)),
    el("td", {}, fmt(e.box_loss)), el("td", {}, fmt(e.cls_loss))));
  $("epoch-table").replaceChildren(...rows);
}

const SVGNS = "http://www.w3.org/2000/svg";
function svg(tag, attrs = {}) {
  const n = document.createElementNS(SVGNS, tag);
  for (const [k, v] of Object.entries(attrs)) n.setAttribute(k, v);
  return n;
}

// A round axis maximum at or above v (1, 2, 2.5, 5 or 10 times a power of ten).
function niceMax(v) {
  if (v <= 0) return 1;
  const p = 10 ** Math.floor(Math.log10(v));
  return [1, 2, 2.5, 5, 10].map((m) => m * p).find((m) => m >= v);
}

// Single-axis line chart with a legend, end-of-line labels and a hover crosshair.
// It is redrawn only when something it shows has changed: the run (`run`, its id), the
// width, or the number of epochs so far. Two runs can have the same number of epochs.
function lineChart(box, data, epochs, series, { run = "", yMax, digits = 3 } = {}) {
  const width = box.clientWidth || 400, height = 190;
  const key = `${run}|${width}|${data.length}|${epochs}`;
  if (box.dataset.key === key) return;
  box.dataset.key = key;

  const pts = series.map((s) => data.filter((d) => typeof d[s.key] === "number").map((d) => [d.epoch, d[s.key]]));
  const top = yMax ?? niceMax(Math.max(...pts.flat().map((p) => p[1]), 0) * 1.05);
  const m = { l: 38, r: 78, t: 8, b: 26 };
  const pw = width - m.l - m.r, ph = height - m.t - m.b;
  const xMax = Math.max(epochs, 2);
  const X = (e) => m.l + ((e - 1) / (xMax - 1)) * pw;
  const Y = (v) => m.t + ph - (Math.min(v, top) / top) * ph;

  const root = svg("svg", { width, height, viewBox: `0 0 ${width} ${height}`, role: "img",
    "aria-label": `${series.map((s) => s.label).join(" and ")} by epoch` });
  // recessive grid + axis labels
  for (let i = 0; i <= 4; i++) {
    const v = (top / 4) * i, y = Y(v);
    root.append(svg("line", { x1: m.l, x2: m.l + pw, y1: y, y2: y, class: i ? "grid" : "axis" }));
    const t = svg("text", { x: m.l - 6, y: y + 4, "text-anchor": "end", class: "tick" });
    t.textContent = top <= 1 ? v.toFixed(2) : +v.toFixed(2);
    root.append(t);
  }
  const xticks = [...new Set([1, Math.round(xMax / 4), Math.round(xMax / 2), Math.round((3 * xMax) / 4), xMax])].filter((e) => e >= 1);
  for (const e of xticks) {
    const t = svg("text", { x: X(e), y: height - 8, "text-anchor": "middle", class: "tick" });
    t.textContent = e;
    root.append(t);
  }
  // lines + direct labels at the latest point
  const ends = [];
  pts.forEach((p, i) => {
    if (!p.length) return;
    root.append(svg("polyline", { points: p.map(([e, v]) => `${X(e)},${Y(v)}`).join(" "),
      fill: "none", stroke: SERIES[i], "stroke-width": 2, "stroke-linejoin": "round", "stroke-linecap": "round" }));
    const [e, v] = p[p.length - 1];
    root.append(svg("circle", { cx: X(e), cy: Y(v), r: 4, fill: SERIES[i], class: "ring" }));
    ends.push({ i, x: X(e), y: Y(v) });
  });
  ends.sort((a, b) => a.y - b.y);
  for (let k = 1; k < ends.length; k++) ends[k].y = Math.max(ends[k].y, ends[k - 1].y + 14);  // no overlap
  for (const end of ends) {
    const t = svg("text", { x: m.l + pw + 8, y: end.y + 4, class: "direct" });
    t.textContent = series[end.i].label;
    root.append(t);
  }

  // hover crosshair + tooltip
  const cross = svg("line", { y1: m.t, y2: m.t + ph, class: "cross", visibility: "hidden" });
  const dots = series.map((_, i) => svg("circle", { r: 4, fill: SERIES[i], class: "ring", visibility: "hidden" }));
  const hit = svg("rect", { x: m.l, y: 0, width: pw, height: m.t + ph, fill: "transparent" });
  root.append(cross, ...dots, hit);
  const tip = el("div", { class: "chart-tip", hidden: true });
  hit.addEventListener("pointermove", (ev) => {
    const r = root.getBoundingClientRect();
    const epoch = 1 + ((ev.clientX - r.left - m.l) / pw) * (xMax - 1);
    const d = data.reduce((a, b) => (Math.abs(b.epoch - epoch) < Math.abs(a.epoch - epoch) ? b : a));
    const x = X(d.epoch);
    cross.setAttribute("x1", x); cross.setAttribute("x2", x); cross.setAttribute("visibility", "visible");
    series.forEach((s, i) => {
      const ok = typeof d[s.key] === "number";
      dots[i].setAttribute("visibility", ok ? "visible" : "hidden");
      if (ok) { dots[i].setAttribute("cx", x); dots[i].setAttribute("cy", Y(d[s.key])); }
    });
    tip.replaceChildren(el("b", {}, `Epoch ${d.epoch}`), ...series.map((s, i) =>
      el("div", {}, el("span", { class: "swatch", style: `background:${SERIES[i]}` }), `${s.label} ${fmt(d[s.key], digits)}`)));
    tip.hidden = false;
    const left = x + 12 + 140 > width ? x - 12 - 140 : x + 12;
    tip.style.left = `${Math.max(0, left)}px`;
    tip.style.top = `${m.t}px`;
  });
  hit.addEventListener("pointerleave", () => {
    tip.hidden = true;
    cross.setAttribute("visibility", "hidden");
    dots.forEach((d) => d.setAttribute("visibility", "hidden"));
  });

  const legend = el("div", { class: "legend small" }, ...series.map((s, i) =>
    el("span", {}, el("span", { class: "swatch", style: `background:${SERIES[i]}` }), s.label)));
  box.replaceChildren(legend, root, tip);
}

// ---- models card ---------------------------------------------------------
// One list for both places a model can be: this computer (where it was trained) and
// the device (where it runs). /api/models merges them by file name.
let models = [];
let studioStatus = null;
let deviceOnline = false;
let selectedModel = null;   // null = the active model

const sameList = (a, b) => Array.isArray(a) && Array.isArray(b) && a.length === b.length
  && a.every((v, i) => (typeof v === "number" ? Math.abs(v - b[i]) < 1e-6 : v === b[i]));
const dt = (iso) => (iso ? iso.slice(0, 16).replace("T", " ") : null);
const pct = (v) => (typeof v === "number" ? `${(v * 100).toFixed(1)}%` : "–");

async function loadModels() {
  try {
    const [list, status] = await Promise.all([api("/api/models"), api("/api/status")]);
    models = list.models;
    deviceOnline = list.device_online;
    studioStatus = status;
  } catch (e) {
    $("model-detail").replaceChildren(el("p", { class: "error-text" }, e.message));
    return;
  }
  if (selectedModel && !models.some((m) => m.name === selectedModel)) selectedModel = null;
  renderModels();
}

// The model whose details are shown: the clicked one, else the active one, else the newest.
function shownModel() {
  return models.find((m) => m.name === selectedModel) || models.find((m) => m.active) || models[0];
}

function renderModels() {
  const onDevice = models.filter((m) => m.on_device).length;
  $("models-count").textContent = !models.length ? ""
    : deviceOnline ? `${onDevice} on the device · ${models.filter((m) => m.local).length} on this computer`
      : "device offline: showing this computer only";
  const shown = shownModel();
  $("model-chips").replaceChildren(...models.map((m) => el("button", {
    type: "button", role: "tab", class: "chip" + (m === shown ? " on" : ""),
    "aria-selected": m === shown ? "true" : "false", tabindex: m === shown ? "0" : "-1",
    title: m.name + (m.on_device ? "" : " (not on the device)"),
    onclick: () => { selectedModel = m.name; renderModels(); },
  }, m.active ? el("span", { class: "dot", title: "active" }) : null, modelLabel(m.name))));
  if (!shown) {
    $("model-detail").replaceChildren(el("p", { class: "muted small" },
      "No models yet. Start a training run to create one."));
    return;
  }
  $("model-detail").replaceChildren(modelDetail(shown));
}

$("model-chips").addEventListener("keydown", (e) => {
  if (!["ArrowLeft", "ArrowRight"].includes(e.key) || !models.length) return;
  const i = models.indexOf(shownModel());
  selectedModel = models[(i + (e.key === "ArrowRight" ? 1 : -1) + models.length) % models.length].name;
  renderModels();
  $("model-chips").querySelector(".chip.on")?.focus();
});

// A titled two-column table from [label, value] rows; rows without a value are left out.
function kvTable(title, rows) {
  rows = rows.filter(([, v]) => v !== null && v !== undefined && v !== "");
  if (!rows.length) return null;
  return el("div", { class: "kv" }, el("h3", {}, title),
    el("table", {}, el("tbody", {}, ...rows.map(([k, v]) =>
      el("tr", {}, el("th", {}, k), el("td", {}, v instanceof Node ? v : String(v)))))));
}

// Everything about one model: actions, warnings, scores and its training provenance.
// `m` is an entry of GET /api/models; `m.info` is the model's metadata sidecar.
function modelDetail(m) {
  const info = m.info || {}, met = info.metrics || {}, ds = info.dataset || {}, u = m.usage;
  const ev = info.evaluation || {}, at = ev.at_recommended;
  const device = studioStatus.device || {};     // {} while the device is unreachable
  const settings = device.settings || {};
  const hasInfo = Object.keys(info).length > 0;

  // things that would make this model misbehave on the device right now
  const warnings = [];
  if (info.classes && !sameList(info.classes, studioStatus.classes)) {
    warnings.push(`Trained for classes ${info.classes.join(", ")}, but boxqc.yaml now lists ${studioStatus.classes.join(", ")}.`);
  }
  if (info.roi && settings.roi && !sameList(info.roi, settings.roi)) {
    warnings.push("The ROI has changed since this model was trained. Retrain, or detections may be off.");
  }
  if (m.active && device.model_error) warnings.push(`Load error: ${device.model_error}`);
  if (!hasInfo) warnings.push("This model has no metadata file, so its training details are unknown.");
  if (!m.on_device && deviceOnline) warnings.push("Not on the device yet. Deploy it to use it for inspections.");
  if (m.device_copy_differs) {
    warnings.push("The device has a different model stored under this name, and the details below describe "
      + "the copy on this computer. Deploy this version to upload it; the device will give it a new name.");
  }

  const thrNow = settings.conf_threshold;
  const rec = info.recommended_threshold;
  const header = el("div", { class: "row model-head" },
    el("h3", { class: "grow model-name" }, m.name),
    m.active ? el("span", { class: "badge labeled" }, "active") : null,
    m.on_device && !m.active ? el("button", { class: "primary", onclick: () => activateModel(m.name) }, "Make active") : null,
    !m.on_device || m.device_copy_differs ? el("button", {
      class: "primary", disabled: deviceOnline ? null : "disabled",
      title: deviceOnline ? "Upload to the device and make it the active model" : "The device is not reachable",
      onclick: () => deployModel(m.name),
    }, m.device_copy_differs ? "Deploy this version & activate" : "Deploy & activate") : null,
    m.active && rec && Math.abs(rec - thrNow) > 1e-6
      ? el("button", { onclick: () => applyThreshold(rec) }, `Use threshold ${rec}`) : null,
    m.on_device ? el("button", {
      class: "fail", disabled: m.active ? "disabled" : null,
      title: m.active ? "Activate another model first" : "Delete this model from the device",
      onclick: () => deleteModel(m.name, "device"),
    }, "Delete from device") : null,
    m.local ? el("button", {
      class: "fail", title: "Delete the copy kept on this computer",
      onclick: () => deleteModel(m.name, "local"),
    }, "Delete local copy") : null);

  const stat = (v, label, extra) => el("div", { class: "stat" }, el("b", {}, v), el("span", {}, label),
    extra ? el("span", { class: "stat-extra" }, extra) : null);
  const stats = el("div", { class: "stats" },
    stat(fmt(met.mAP50), "mAP50"),
    stat(fmt(met["mAP50-95"]), "mAP50-95"),
    stat(fmt(met.precision, 2), "precision"),
    stat(fmt(met.recall, 2), "recall"),
    stat(rec ?? "–", "recommended threshold", m.active ? `device uses ${thrNow}` : null),
    stat(u && u.inference_ms ? `${Math.round(u.inference_ms)} ms` : "–", "inference on the device (median)"));

  const split = (d) => (d ? `${d.images} images · ${d.defective} defective, ${d.clean} clean · ${d.boxes} boxes` : null);
  const epochsText = info.epochs
    ? (info.epochs_completed && info.epochs_completed !== info.epochs
      ? `${info.epochs_completed} of ${info.epochs} (stopped early)` : `${info.epochs}`) : null;
  const on = info.trained_on;
  const tables = el("div", { class: "kv-grid" },
    kvTable("Training", [
      ["Trained", dt(info.trained_at)],
      ["Trainer", info.trainer],
      ["On", on ? `${on.hostname} (${on.device})` : null],
      ["Duration", info.train_seconds ? duration(info.train_seconds) : null],
      ["Base model", info.base_model],
      ["Image size", info.imgsz ? `${info.imgsz} px` : null],
      ["Batch", info.batch],
      ["Epochs", epochsText],
      ["Best epoch", info.best_epoch],
      ["Early-stop patience", info.patience],
    ]),
    kvTable("Dataset", [
      ["Trained on", info.dataset_name ? info.dataset_name
        : "dataset_name" in info ? "All labeled images" : (hasInfo ? "All labeled images (before datasets existed)" : null)],
      ["Train", split(ds.train)],
      ["Validation", split(ds.val)],
      ["Classes", info.classes ? info.classes.join(", ") : null],
      ["ROI", info.roi ? info.roi.map((v) => v.toFixed(2)).join(", ")
        + (!settings.roi ? "" : sameList(info.roi, settings.roi) ? " (same as now)" : " (differs from now)") : null],
      ["Validation at thr " + (at ? at.threshold : ""), at
        ? `${at.fn}/${at.fn + at.tp} defective missed · ${at.fp}/${at.fp + at.tn} false alarms · ${pct(at.accuracy)} correct` : null],
    ]),
    kvTable("Where", [
      ["Stored", [m.local ? "this computer" : null, m.on_device ? "the device" : null].filter(Boolean).join(" and ")],
      ["Inspections", !m.on_device ? null : u ? `${u.inspections} (${u.PASS} PASS, ${u.FAIL} FAIL)` : "none yet"],
      ["Last used", u ? dt(u.last_used) : null],
      ["File", `${m.size_mb} MB · ${m.modified}`],
      ["Format", info.format],
      ["Run folder", info.run_dir],
    ]));

  const parts = [header];
  if (warnings.length) parts.push(el("ul", { class: "warn-box small" }, ...warnings.map((w) => el("li", {}, w))));
  parts.push(stats, tables);
  if (ev.table && ev.table.length) {
    parts.push(el("details", {},
      el("summary", { class: "small muted" }, "PASS/FAIL on validation images at each threshold"),
      el("table", { style: "margin-top:.4rem" },
        el("thead", {}, el("tr", {}, ...["Threshold", "Missed defective", "False alarms", "Accuracy"].map((h) => el("th", {}, h)))),
        el("tbody", {}, ...ev.table.map((row) => el("tr", { class: at && row.threshold === at.threshold ? "on" : "" },
          el("td", {}, row.threshold.toFixed(2) + (at && row.threshold === at.threshold ? " (recommended)" : "")),
          el("td", {}, `${row.fn}/${row.tp + row.fn}`), el("td", {}, `${row.fp}/${row.fp + row.tn}`),
          el("td", {}, pct(row.accuracy))))))));
  }
  return el("div", { class: "model-detail" }, ...parts);
}

async function applyThreshold(t) {
  try {
    await api(`${DEVICE}/settings`, { method: "PUT", json: { conf_threshold: t } });
    toast(`Threshold set to ${t}`, "ok");
  } catch (e) { toast(e.message, "error"); }
  loadModels();
}

// where: "device" removes it from the device, "local" removes this computer's copy
async function deleteModel(name, where) {
  const place = where === "device" ? "the device" : "this computer";
  if (!confirm(`Delete ${name} from ${place}? This cannot be undone.`)) return;
  try {
    const base = where === "device" ? `${DEVICE}/models` : "/api/models";
    await api(`${base}/${encodeURIComponent(name)}`, { method: "DELETE" });
    toast(`Deleted ${name} from ${place}`, "ok");
  } catch (e) { toast(e.message, "error"); }
  loadModels();
}

async function deployModel(name) {
  try {
    const r = await api(`/api/models/${encodeURIComponent(name)}/deploy`, { method: "POST", json: { activate: true } });
    toast(`Deployed ${r.name}${r.active ? " and made it active" : ""}`, "ok");
  } catch (e) { toast(e.message, "error"); }
  await refreshPills();
  loadModels();
}

// activateModel() lives in common.js and fires this event (also used by the top-bar menu)
document.addEventListener("boxqc:model-changed", loadModels);

// ---- polling -------------------------------------------------------------
// One refresh of the whole page. Also called right after an action for instant feedback.
let lastActiveId = null;
async function poll() {
  try {
    overview = await api("/api/training");
  } catch (e) {
    $("host-state").textContent = "studio unreachable";
    return;
  }
  fillForm();
  renderHost();
  renderStartButton();
  renderJobs();
  let job = null;
  try {
    const id = currentId();
    job = id ? await loadJob(id) : null;
  } catch {
    selectedId = null;  // the run is gone (e.g. the studio restarted); follow the latest again
  }
  renderRun(job);
  // a finished run has produced (and maybe deployed) a model
  const activeId = overview.active ? overview.active.id : null;
  if (lastActiveId && !activeId) { loadDatasetStats(); loadModels(); refreshPills(); }
  lastActiveId = activeId;
}

window.addEventListener("resize", () => { document.querySelectorAll(".chart").forEach((c) => { c.dataset.key = ""; }); poll(); });
loadDatasetStats();
loadModels();
poll();
setInterval(poll, POLL_MS);
