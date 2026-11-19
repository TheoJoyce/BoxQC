// Inspection page: run, visualise detections at an adjustable threshold, review history.
// Inspections happen on the device; this page drives it through the /api/device proxy.
"use strict";

// An inspection record holds every detection down to a low reporting floor, not just
// those above the threshold. That lets this page redraw a result and recompute the
// verdict for any threshold on the slider without asking the device to run again.
const canvas = document.getElementById("result");
const ctx = canvas.getContext("2d");
const thr = document.getElementById("thr");
const photo = new Image();
let current = null;     // current inspection record
let savedThreshold = 0.5;
let busy = false;

function viewThreshold() { return parseFloat(thr.value); }

// The verdict a record would have had at threshold `t` (same rule as the device).
function verdictAt(record, t) {
  if (!record.model) return "NO_MODEL";
  return record.detections.some((d) => d.conf >= t) ? "FAIL" : "PASS";
}

// Paint the inspected photo with the ROI outline and the boxes at the slider's threshold.
function drawResult() {
  if (!current || !photo.naturalWidth) return;
  const wrap = document.getElementById("viewer");
  const scale = Math.min(wrap.clientWidth / photo.naturalWidth, (window.innerHeight * 0.72) / photo.naturalHeight, 1);
  canvas.width = Math.round(photo.naturalWidth * scale);
  canvas.height = Math.round(photo.naturalHeight * scale);
  ctx.drawImage(photo, 0, 0, canvas.width, canvas.height);
  const sx = canvas.width / current.image_size[0];
  const sy = canvas.height / current.image_size[1];

  const [rx0, ry0, rx1, ry1] = current.roi || [0, 0, 1, 1];
  ctx.strokeStyle = "#999";
  ctx.lineWidth = 1;
  ctx.setLineDash([4, 4]);
  ctx.strokeRect(rx0 * canvas.width, ry0 * canvas.height, (rx1 - rx0) * canvas.width, (ry1 - ry0) * canvas.height);

  const t = viewThreshold();
  const showLow = document.getElementById("show-low").checked;
  for (const d of current.detections) {
    const above = d.conf >= t;
    if (!above && !showLow) continue;
    const [x0, y0, x1, y1] = d.box;
    const color = classColor(d.cls_id);
    ctx.setLineDash(above ? [] : [5, 4]);
    ctx.lineWidth = above ? 3 : 1.5;
    ctx.strokeStyle = color;
    ctx.globalAlpha = above ? 1 : 0.7;
    ctx.strokeRect(x0 * sx, y0 * sy, (x1 - x0) * sx, (y1 - y0) * sy);
    const label = `${d.name} ${d.conf.toFixed(2)}`;
    ctx.font = "13px system-ui";
    const tw = ctx.measureText(label).width + 6;
    ctx.fillStyle = color;
    ctx.fillRect(x0 * sx, Math.max(0, y0 * sy - 17), tw, 17);
    ctx.fillStyle = "#000";
    ctx.fillText(label, x0 * sx + 3, Math.max(13, y0 * sy - 4));
    ctx.globalAlpha = 1;
  }
  ctx.setLineDash([]);
  canvas.hidden = false;
  document.getElementById("result-empty").hidden = true;
}

// Verdict banner, detection table and result details for the current record and threshold.
function renderPanel() {
  const t = viewThreshold();
  document.getElementById("thr-val").textContent = t.toFixed(2);
  document.getElementById("thr-save").disabled = Math.abs(t - savedThreshold) < 1e-6;
  if (!current) return;

  const v = verdictAt(current, t);
  const vEl = document.getElementById("verdict");
  vEl.textContent = v.replace("_", " ");
  vEl.className = `verdict ${v}`;
  const note = document.getElementById("thr-note");
  note.textContent = v !== current.verdict
    ? `Recorded verdict was ${current.verdict} at ${current.threshold.toFixed(2)}; at ${t.toFixed(2)} it would be ${v}.`
    : "Any detection at or above this confidence is a FAIL.";

  const tbody = document.getElementById("dets");
  if (!current.model) {
    tbody.replaceChildren(el("tr", {}, el("td", { colspan: 3, class: "muted" }, "No model on the device. Train one on the Training page, or upload one in System.")));
  } else if (!current.detections.length) {
    tbody.replaceChildren(el("tr", {}, el("td", { colspan: 3, class: "muted" }, "No detections above the reporting floor.")));
  } else {
    tbody.replaceChildren(...current.detections.map((d) =>
      el("tr", { class: d.conf >= t ? "" : "dim" },
        el("td", {}, el("span", { class: "swatch", style: `background:${classColor(d.cls_id)}` }), d.name),
        el("td", {}, d.conf.toFixed(3)),
        el("td", {}, `${Math.round(d.box[2] - d.box[0])}×${Math.round(d.box[3] - d.box[1])}`))));
  }
  const tm = current.timing_ms || {};
  document.getElementById("result-meta").textContent =
    `${current.id} · ${current.model || "no model"} · inference ${tm.inference ?? "–"} ms · total ${tm.total ?? "–"} ms`;
  document.getElementById("to-dataset").disabled = false;
  const link = document.getElementById("open-annotated");
  link.href = `${DEVICE}/inspections/${current.id}/image.jpg?annotated=true`;
  link.hidden = false;
}

// Make `record` the one on display and load its photo from the device.
function showRecord(record) {
  current = record;
  renderPanel();
  photo.onload = drawResult;
  photo.onerror = () => toast("Inspection image not stored (inference.save_inspections is off?)", "error");
  photo.src = `${DEVICE}/inspections/${record.id}/image.jpg`;
}

// ---- sound in the browser ----
// Optional and off by default: play the verdict's buzzer pattern on this computer too.
// The patterns (beep count and length, the gap) come from the device's status, so this
// sounds like the buzzer; the built-in values below are only for a device too old to say.
const browserSound = document.getElementById("browser-sound");
let soundPatterns = {
  PASS: { beeps: 1, beep_seconds: 0.12, gap_seconds: 0.10 },
  FAIL: { beeps: 3, beep_seconds: 0.33, gap_seconds: 0.20 },
  ERROR: { beeps: 6, beep_seconds: 0.08, gap_seconds: 0.08 },
};
let audio = null;       // AudioContext, created on a click (browsers block sound before one)

async function loadSoundPatterns() {
  try {
    const hw = (await api(`${DEVICE}/status`)).hardware;
    if (hw && hw.patterns) soundPatterns = hw.patterns;
  } catch { /* keep what we have */ }
}
// Unlock audio inside a user gesture; later, after an `await`, browsers may refuse.
function unlockAudio() {
  if (!browserSound.checked) return;
  try {
    audio ||= new (window.AudioContext || window.webkitAudioContext)();
    if (audio.state === "suspended") audio.resume();
  } catch { audio = null; }
}
function playVerdict(verdict) {
  if (!browserSound.checked || !audio) return;
  const p = soundPatterns[verdict] || soundPatterns.ERROR;
  const osc = audio.createOscillator();
  const gain = audio.createGain();
  osc.type = "square";
  osc.frequency.value = 2700;        // about the pitch of a typical buzzer
  gain.gain.value = 0;
  osc.connect(gain).connect(audio.destination);
  let t = audio.currentTime + 0.02;
  for (let i = 0; i < p.beeps; i++) {
    gain.gain.setValueAtTime(0, t);
    gain.gain.linearRampToValueAtTime(0.06, t + 0.005);          // short ramps: no clicks
    gain.gain.setValueAtTime(0.06, t + p.beep_seconds - 0.005);
    gain.gain.linearRampToValueAtTime(0, t + p.beep_seconds);
    t += p.beep_seconds + p.gap_seconds;
  }
  osc.start();
  osc.stop(t);
}
browserSound.addEventListener("change", () => {
  try { localStorage.setItem("boxqc.browserSound", browserSound.checked ? "1" : ""); } catch { /* private mode */ }
  if (browserSound.checked) { unlockAudio(); loadSoundPatterns(); }
});
try { browserSound.checked = localStorage.getItem("boxqc.browserSound") === "1"; } catch { /* off */ }
if (browserSound.checked) loadSoundPatterns();

// Ask the device to inspect now (captures, runs the model, sounds the buzzer).
async function runInspect() {
  if (busy) return;
  busy = true;
  unlockAudio();
  const btn = document.getElementById("inspect");
  btn.disabled = true;
  btn.textContent = "Inspecting…";
  try {
    const buzz = document.getElementById("buzz").checked;
    const record = await api(`${DEVICE}/inspect?buzz=${buzz}`, { method: "POST" });
    playVerdict(record.verdict);
    showRecord(record);
    loadHistory().catch(() => {});
  } catch (e) {
    playVerdict("ERROR");
    const vEl = document.getElementById("verdict");
    vEl.textContent = "ERROR";
    vEl.className = "verdict ERROR";
    toast(e.message, "error");
  } finally {
    busy = false;
    btn.disabled = false;
    btn.replaceChildren("Inspect ", el("kbd", {}, "Space"));
  }
}

// Recent inspections from the device's log; clicking a row shows it again.
async function loadHistory() {
  const items = await api(`${DEVICE}/inspections?limit=30`);
  document.getElementById("history").replaceChildren(...items.map((r) => {
    const top = r.detections.reduce((a, d) => (!a || d.conf > a.conf ? d : a), null);
    return el("tr", { class: "clickable", onclick: () => showRecord(r) },
      el("td", {}, r.time.replace("T", " ").slice(0, 19)),
      el("td", {}, el("span", { class: `badge ${r.verdict === "PASS" ? "labeled" : r.verdict === "FAIL" ? "defect" : "unlabeled"}` }, r.verdict)),
      el("td", {}, r.source || ""),
      el("td", {}, top ? `${top.name} ${top.conf.toFixed(2)}` : "–"),
      el("td", { class: "muted small" }, r.model || "–"),
      el("td", {}, r.timing_ms?.total ?? ""));
  }));
}

document.getElementById("inspect").onclick = runInspect;
document.getElementById("show-low").onchange = drawResult;
thr.addEventListener("input", () => { renderPanel(); drawResult(); });
window.addEventListener("resize", drawResult);

document.getElementById("thr-save").onclick = async () => {
  try {
    const s = await api(`${DEVICE}/settings`, { method: "PUT", json: { conf_threshold: viewThreshold() } });
    savedThreshold = s.conf_threshold;
    renderPanel();
    toast(`Default threshold set to ${savedThreshold.toFixed(2)}`, "ok");
  } catch (e) { toast(e.message, "error"); }
};

// "Send to dataset" adds the image to the capture dataset (chosen on the Capture or Dataset page).
let captureDataset = null;
async function loadCaptureDataset() {
  try { captureDataset = (await api("/api/datasets")).capture_dataset; } catch { return; }
  document.getElementById("to-dataset").textContent =
    captureDataset ? `Send to "${captureDataset}" for labeling` : "Send to dataset for labeling";
  const note = document.getElementById("to-dataset-target");
  note.replaceChildren(captureDataset
    ? el("span", {}, "Goes into the capture dataset. ", el("a", { href: "/capture" }, "Change"))
    : el("span", {}, "No capture dataset is set, so it won't be in any dataset. ", el("a", { href: "/capture" }, "Choose one")));
}

document.getElementById("to-dataset").onclick = async () => {
  if (!current) return;
  try {
    const ann = await api(`/api/inspections/${current.id}/to-dataset?min_conf=${viewThreshold()}`, { method: "POST" });
    const ds = ann.datasets[0] || NO_DATASET;
    toast(`Added to ${ann.datasets[0] ? `"${ann.datasets[0]}"` : "the dataset"}, opening the annotator`, "ok");
    setTimeout(() => { location.href = `/annotate/${ann.id}?dataset=${encodeURIComponent(ds)}`; }, 600);
  } catch (e) { toast(e.message, "error"); }
};
loadCaptureDataset();

document.getElementById("upload").addEventListener("change", async (e) => {
  const file = e.target.files[0];
  if (!file) return;
  const form = new FormData();
  form.append("file", file);
  try {
    showRecord(await api(`${DEVICE}/inspect/upload`, { method: "POST", form }));
    loadHistory().catch(() => {});
  } catch (err) { toast(err.message, "error"); }
  e.target.value = "";
});

document.addEventListener("keydown", (e) => {
  if (notAHotkey(e) || e.repeat) return;
  if (e.key === " ") { e.preventDefault(); runInspect(); }
});

(async function init() {
  try {
    const s = await api(`${DEVICE}/settings`);
    savedThreshold = s.conf_threshold;
    thr.value = savedThreshold;
    renderPanel();
    await loadHistory();
  } catch (e) {
    thr.value = savedThreshold;
    renderPanel();
    toast(e.message, "error");
  }
})();
