// Bounding-box annotator (the /annotate/<id> page).
//
// Boxes are stored normalized (0..1) in full-image coordinates, exactly as the server
// keeps them (see boxqc/studio/dataset.py), so nothing here depends on the size the
// image happens to be displayed at. The canvas is redrawn from scratch on every change.
//
// State: `boxes` is the working copy of the image's boxes, `selected` the index of the
// box being edited (-1 = none), `drag` the pointer gesture in progress, and `dirty`
// whether there are unsaved changes. Nothing is sent to the server until Save.
//
// `suggestions` are a model's proposals for this image (see the "labeling assistant"
// section further down). They are drawn dashed and are not boxes: one becomes a box
// when the user adds it, and the others are never saved.
"use strict";

const CLASSES = window.BOXQC.classes;
const canvas = document.getElementById("canvas");
const ctx = canvas.getContext("2d");
const img = new Image();
const HANDLE = 9;       // px (display) grab radius for corners, with a mouse or pen
const HANDLE_TOUCH = 24;  // ...and with a finger, which covers far more than a cursor tip
let grab = HANDLE;      // the radius for the pointer in use (set on pointerdown)
const MIN_SIZE = 6;     // px (display) minimum new box size

let ann = null;
let roi = [0, 0, 1, 1];
let boxes = [];
let selected = -1;
let currentClass = 0;
let drag = null;        // {mode: draw|move|resize, ...}
let dirty = false;
let suggestions = [];   // [{cls, conf, x0, y0, x1, y1, gone}]; gone: "added" | "dismissed" once dealt with
let hotSuggestion = null;   // the suggestion under the pointer (or under it in the side panel)
// Navigation (newer/older, next unlabeled) stays inside the dataset the page was opened from.
const DATASET = new URLSearchParams(location.search).get("dataset") || "";
const dsQuery = DATASET ? `?dataset=${encodeURIComponent(DATASET)}` : "";
const datasetPage = `/dataset${dsQuery}`;

// ---- geometry helpers ---------------------------------------------------
const W = () => canvas.width;
const H = () => canvas.height;

// Pointer event -> normalized image coordinates, clamped to the image.
function toNorm(e) {
  const r = canvas.getBoundingClientRect();
  return {
    x: Math.min(1, Math.max(0, (e.clientX - r.left) / r.width)),
    y: Math.min(1, Math.max(0, (e.clientY - r.top) / r.height)),
  };
}
// A box with its corners in order (x0 <= x1, y0 <= y1), whichever way it was dragged.
function normBox(b) {
  return { cls: b.cls, x0: Math.min(b.x0, b.x1), y0: Math.min(b.y0, b.y1), x1: Math.max(b.x0, b.x1), y1: Math.max(b.y0, b.y1) };
}
// Distance between two normalized points in displayed pixels (for grab radii).
function pxDist(a, b) {
  const r = canvas.getBoundingClientRect();
  return Math.hypot((a.x - b.x) * r.width, (a.y - b.y) * r.height);
}
// Index of the box under a point, or -1.
function hitBox(p) {
  // smallest box containing the point wins (so nested boxes stay selectable)
  let best = -1, bestArea = Infinity;
  boxes.forEach((b, i) => {
    if (p.x >= b.x0 && p.x <= b.x1 && p.y >= b.y0 && p.y <= b.y1) {
      const area = (b.x1 - b.x0) * (b.y1 - b.y0);
      if (area < bestArea) { best = i; bestArea = area; }
    }
  });
  return best;
}
const boxArea = (b) => Math.abs((b.x1 - b.x0) * (b.y1 - b.y0));
// Overlap of two boxes as intersection over union: 0 = apart, 1 = identical.
function iou(a, b) {
  const w = Math.min(a.x1, b.x1) - Math.max(a.x0, b.x0), h = Math.min(a.y1, b.y1) - Math.max(a.y0, b.y0);
  if (w <= 0 || h <= 0) return 0;
  return (w * h) / (boxArea(a) + boxArea(b) - w * h);
}
// The suggestion on display under a point (the smallest, like hitBox), or null.
function hitSuggestion(p) {
  let best = null;
  for (const s of shownSuggestions()) {
    if (p.x >= s.x0 && p.x <= s.x1 && p.y >= s.y0 && p.y <= s.y1 && (!best || boxArea(s) < boxArea(best))) best = s;
  }
  return best;
}
// What a press at a point acts on: {corner} of the selected box, else {box} (an index)
// or {suggestion}, whichever of the two is smaller, so that a small suggestion inside a
// large box (or the other way round) can still be reached. {} = empty image.
function targetAt(p) {
  const corner = hitCorner(p);
  if (corner) return { corner };
  const box = hitBox(p), suggestion = hitSuggestion(p);
  if (box >= 0 && !(suggestion && boxArea(suggestion) < boxArea(boxes[box]))) return { box };
  return suggestion ? { suggestion } : {};
}
// The corner of the selected box under a point, as the pair of keys it moves
// ({kx: "x0", ky: "y1"}), or null. Only the selected box has grab handles.
function hitCorner(p) {
  if (selected < 0) return null;
  const b = boxes[selected];
  const corners = [["x0", "y0"], ["x1", "y0"], ["x0", "y1"], ["x1", "y1"]];
  for (const [kx, ky] of corners) {
    if (pxDist(p, { x: b[kx], y: b[ky] }) <= grab) return { kx, ky };
  }
  return null;
}

// ---- rendering ----------------------------------------------------------
// Size the canvas to the image, scaled down to fit the column and 75% of the window height.
function fitCanvas() {
  if (!img.naturalWidth) return;
  const wrap = document.getElementById("wrap");
  const maxW = wrap.clientWidth;
  const maxH = window.innerHeight * 0.75;
  const scale = Math.min(maxW / img.naturalWidth, maxH / img.naturalHeight, 1);
  canvas.width = Math.round(img.naturalWidth * scale);
  canvas.height = Math.round(img.naturalHeight * scale);
  draw();
}

// One suggestion: a dashed outline and a hollow "+ class NN%" tag. Boxes are solid with
// a filled tag, so what will be saved and what is only proposed cannot be confused.
function drawSuggestion(s) {
  const color = classColor(CLASSES.indexOf(s.cls));
  const hot = s === hotSuggestion;
  const x = s.x0 * W(), y = s.y0 * H(), w = (s.x1 - s.x0) * W(), h = (s.y1 - s.y0) * H();
  ctx.save();
  if (hot) { ctx.fillStyle = color + "33"; ctx.fillRect(x, y, w, h); }
  ctx.globalAlpha = hot ? 1 : 0.85;
  ctx.strokeStyle = color;
  ctx.lineWidth = hot ? 2.5 : 1.5;
  ctx.setLineDash([3, 3]);
  ctx.strokeRect(x, y, w, h);
  ctx.setLineDash([]);
  ctx.font = "12px system-ui";
  const label = `+ ${s.cls} ${Math.round(s.conf * 100)}%`;
  const tw = ctx.measureText(label).width + 6, ty = Math.max(0, y - 16);
  ctx.fillStyle = "rgba(0,0,0,0.75)";
  ctx.fillRect(x, ty, tw, 16);
  ctx.lineWidth = 1;
  ctx.strokeRect(x + 0.5, ty + 0.5, tw - 1, 15);
  ctx.fillStyle = color;
  ctx.fillText(label, x + 3, ty + 12);
  ctx.restore();
}

// Paint the image, the dimmed area outside the ROI, the suggestions, and every box
// (plus the one being drawn) on top.
function draw() {
  ctx.clearRect(0, 0, W(), H());
  ctx.drawImage(img, 0, 0, W(), H());
  if (document.getElementById("show-roi").checked) {
    const [rx0, ry0, rx1, ry1] = roi;
    ctx.save();
    ctx.fillStyle = "rgba(0,0,0,0.45)";
    ctx.beginPath();
    ctx.rect(0, 0, W(), H());
    ctx.rect(rx0 * W(), ry0 * H(), (rx1 - rx0) * W(), (ry1 - ry0) * H());
    ctx.fill("evenodd");
    ctx.restore();
  }
  shownSuggestions().forEach(drawSuggestion);
  const all = drag && drag.mode === "draw" ? [...boxes, normBox(drag.box)] : boxes;
  all.forEach((b, i) => {
    const ci = Math.max(0, CLASSES.indexOf(b.cls));
    const color = classColor(ci);
    const x = b.x0 * W(), y = b.y0 * H(), w = (b.x1 - b.x0) * W(), h = (b.y1 - b.y0) * H();
    ctx.lineWidth = i === selected ? 3 : 2;
    ctx.strokeStyle = color;
    // dashed = boxes suggested by the model (image sent from Inspect), not yet confirmed
    ctx.setLineDash(ann && ann.status === "unlabeled" && ann.source === "inspection" ? [6, 4] : []);
    ctx.strokeRect(x, y, w, h);
    ctx.setLineDash([]);
    if (i === selected) {
      ctx.fillStyle = color + "22";
      ctx.fillRect(x, y, w, h);
      ctx.fillStyle = color;
      for (const [cx, cy] of [[x, y], [x + w, y], [x, y + h], [x + w, y + h]]) ctx.fillRect(cx - 4, cy - 4, 8, 8);
    }
    ctx.font = "12px system-ui";
    const label = b.cls;
    const tw = ctx.measureText(label).width + 6;
    ctx.fillStyle = color;
    ctx.fillRect(x, Math.max(0, y - 16), tw, 16);
    ctx.fillStyle = "#000";
    ctx.fillText(label, x + 3, Math.max(12, y - 4));
  });
}

// Class picker, the list of boxes and the status line.
function renderSidebar() {
  document.getElementById("classbar").replaceChildren(...CLASSES.map((c, i) =>
    el("button", { class: i === currentClass ? "on" : "", onclick: () => setClass(i) },
      el("span", { class: "swatch", style: `background:${classColor(i)}` }), c,
      el("span", { class: "grow" }), el("kbd", {}, i + 1))));
  document.getElementById("box-count").textContent = `(${boxes.length})`;
  document.getElementById("box-list").replaceChildren(...boxes.map((b, i) =>
    el("tr", { class: "clickable" + (i === selected ? " on" : ""), onclick: () => { selected = i; update(); } },
      el("td", {}, el("span", { class: "swatch", style: `background:${classColor(CLASSES.indexOf(b.cls))}` }), b.cls),
      el("td", { class: "muted" }, `${Math.round((b.x1 - b.x0) * ann.width)}×${Math.round((b.y1 - b.y0) * ann.height)}`),
      el("td", {}, el("button", { onclick: (e) => { e.stopPropagation(); removeBox(i); } }, "✕")))));
  const s = ann ? `Status: ${ann.status}${dirty ? " · unsaved changes" : ""}` : "";
  document.getElementById("status-line").textContent = s;
  renderAssistant();
}

function update() { draw(); renderSidebar(); }
function markDirty() { dirty = true; update(); }

// ---- editing ------------------------------------------------------------
// Pick the class for new boxes; if a box is selected, change its class too.
function setClass(i) {
  currentClass = i;
  if (selected >= 0) { boxes[selected].cls = CLASSES[i]; markDirty(); } else update();
}
function removeBox(i) {
  const [gone] = boxes.splice(i, 1);
  const s = addedFrom.get(gone);
  if (s) s.gone = null;     // deleting a box that was a suggestion puts the suggestion back
  selected = -1;
  markDirty();
}

// A press starts one of three gestures, tried in this order: resize (on a corner of
// the selected box), move (inside a box, which also selects it), draw (anywhere else).
// A press on a suggestion starts a draw as well: released without dragging it was a
// click, which adds the suggestion (see endDrag); dragged, it is a new box as usual.
canvas.addEventListener("pointerdown", (e) => {
  if (!ann || e.button !== 0) return;     // the right button dismisses suggestions (contextmenu, below)
  grab = e.pointerType === "touch" ? HANDLE_TOUCH : HANDLE;
  canvas.setPointerCapture(e.pointerId);
  const p = toNorm(e);
  const target = targetAt(p);
  if (target.corner) {
    drag = { mode: "resize", ...target.corner };
    return;
  }
  if (target.box !== undefined) {
    selected = target.box;
    const b = boxes[selected];
    currentClass = Math.max(0, CLASSES.indexOf(b.cls));
    drag = { mode: "move", start: p, orig: { ...b } };
    update();
    return;
  }
  selected = -1;
  drag = { mode: "draw", start: p, suggestion: target.suggestion,
    box: { cls: CLASSES[currentClass], x0: p.x, y0: p.y, x1: p.x, y1: p.y } };
  update();
});

canvas.addEventListener("pointermove", (e) => {
  if (!drag) {
    const target = targetAt(toNorm(e));
    canvas.style.cursor = target.corner ? "nwse-resize" : target.box !== undefined ? "move"
      : target.suggestion ? "pointer" : "crosshair";
    highlightSuggestion(target.suggestion || null);
    return;
  }
  const p = toNorm(e);
  if (drag.mode === "draw") {
    drag.box.x1 = p.x; drag.box.y1 = p.y;
  } else if (drag.mode === "move") {
    const o = drag.orig, w = o.x1 - o.x0, h = o.y1 - o.y0;
    const nx0 = Math.min(1 - w, Math.max(0, o.x0 + p.x - drag.start.x));
    const ny0 = Math.min(1 - h, Math.max(0, o.y0 + p.y - drag.start.y));
    Object.assign(boxes[selected], { x0: nx0, y0: ny0, x1: nx0 + w, y1: ny0 + h });
    dirty = true;
  } else if (drag.mode === "resize") {
    boxes[selected][drag.kx] = p.x;
    boxes[selected][drag.ky] = p.y;
    dirty = true;
  }
  draw();
});

// Finish the gesture: keep a drawn box only if it is big enough to be deliberate.
function endDrag() {
  if (!drag) return;
  if (drag.mode === "draw") {
    const b = normBox(drag.box);
    const r = canvas.getBoundingClientRect();
    if ((b.x1 - b.x0) * r.width >= MIN_SIZE && (b.y1 - b.y0) * r.height >= MIN_SIZE) {
      boxes.push(b);
      selected = boxes.length - 1;
      dirty = true;
    } else if (drag.suggestion) {
      addSuggestion(drag.suggestion);     // a click, not a drag
    }
  } else if (drag.mode === "resize" && selected >= 0) {
    boxes[selected] = normBox(boxes[selected]);
  }
  drag = null;
  update();
}
canvas.addEventListener("pointerup", endDrag);
canvas.addEventListener("pointercancel", endDrag);
canvas.addEventListener("pointerleave", () => highlightSuggestion(null));
canvas.addEventListener("contextmenu", (e) => {
  const s = hitSuggestion(toNorm(e));
  if (!s) return;           // elsewhere the browser's own menu ("Save image as…") stays
  e.preventDefault();
  drag = null;              // Ctrl+click on a Mac is a left press too: it must not add what it dismisses
  dismissSuggestion(s);
});

// ---- persistence --------------------------------------------------------
// Send the boxes and notes to the server; `status` ("labeled" | "skip") is optional.
// `clear` saves the image without any box ("clean"). The boxes on screen are only
// dropped once the server has accepted that, so a failed save loses nothing.
async function save(status, { clear = false } = {}) {
  const body = { boxes: clear ? [] : boxes, notes: document.getElementById("notes").value };
  if (status) body.status = status;
  ann = await api(`/api/images/${ann.id}`, { method: "PUT", json: body });
  if (clear) { boxes = []; selected = -1; }
  dirty = false;
  update();
  return ann;
}

// Run a save action, then go to a neighbour of this image (`target` is a key of the
// server's nav object, e.g. "next_unlabeled"), or back to the dataset when there is none.
async function withNav(fn, target) {
  try {
    await fn();
    const nav = (await api(`/api/images/${window.ITEM_ID}${dsQuery}`)).nav;
    const next = nav[target];
    if (next) location.href = `/annotate/${next}${dsQuery}`;
    else {
      toast(`Saved. No more unlabeled images${DATASET ? ` in "${datasetLabel(DATASET)}"` : ""} 🎉`, "ok");
      setTimeout(() => { location.href = datasetPage; }, 1200);
    }
  } catch (e) { toast(e.message, "error"); }
}

async function saveAndToast(status) {
  try { await save(status); toast("Saved", "ok"); } catch (e) { toast(e.message, "error"); }
}

document.getElementById("save").onclick = () => saveAndToast("labeled");
document.getElementById("save-next").onclick = () => withNav(() => save("labeled"), "next_unlabeled");
document.getElementById("clean").onclick = () => withNav(() => save("labeled", { clear: true }), "next_unlabeled");
document.getElementById("skip").onclick = () => withNav(() => save("skip"), "next_unlabeled");
document.getElementById("delete").onclick = async () => {
  if (!confirm("Delete this image permanently?")) return;
  try {
    const nav = (await api(`/api/images/${ann.id}${dsQuery}`)).nav;
    await api(`/api/images/${ann.id}`, { method: "DELETE" });
    dirty = false;
    const next = nav.next || nav.prev;
    location.href = next ? `/annotate/${next}${dsQuery}` : datasetPage;
  } catch (e) { toast(e.message, "error"); }
};
document.getElementById("notes").addEventListener("input", () => { dirty = true; renderSidebar(); });
document.getElementById("show-roi").addEventListener("change", draw);
window.addEventListener("resize", fitCanvas);
window.addEventListener("beforeunload", (e) => { if (dirty) { e.preventDefault(); e.returnValue = ""; } });

document.addEventListener("keydown", (e) => {
  if (isTyping(e)) return;
  const k = e.key, lower = k.toLowerCase();
  // Ctrl/Cmd+S saves. Every other key with a modifier belongs to the browser: Cmd+C is
  // "copy", not "C = clean" (which would erase the boxes), Cmd+X is not "skip".
  if (hasModifier(e)) {
    if (lower === "s" && (e.ctrlKey || e.metaKey) && !e.altKey) {
      e.preventDefault();
      if (!e.repeat) document.getElementById("save").click();
    }
    return;
  }
  if (/^[1-9]$/.test(k) && +k <= CLASSES.length) { setClass(+k - 1); return; }
  if ((k === "Delete" || k === "Backspace") && selected >= 0) { e.preventDefault(); removeBox(selected); return; }
  if (k === "Escape") { selected = -1; drag = null; update(); return; }
  if (e.repeat) return;     // a key held down saves or moves on once, not once per repeat
  if (lower === "s") document.getElementById("save").click();
  else if (lower === "n") document.getElementById("save-next").click();
  else if (lower === "c") document.getElementById("clean").click();
  else if (lower === "x") document.getElementById("skip").click();
  else if (lower === "a") addShownSuggestions();
  else if (lower === "h") document.getElementById("assist-on").click();
  else if (k === "ArrowLeft") document.getElementById("nav-prev").click();
  else if (k === "ArrowRight") document.getElementById("nav-next").click();
});

// ---- load ---------------------------------------------------------------
(async function init() {
  try {
    const [data, settings] = await Promise.all([api(`/api/images/${window.ITEM_ID}${dsQuery}`), api("/api/settings")]);
    ann = data;
    roi = settings.roi;
    boxes = ann.boxes.map((b) => ({ ...b }));
    document.getElementById("notes").value = ann.notes || "";
    document.getElementById("info").textContent =
      `${ann.id} · ${ann.width}×${ann.height} · ${ann.source}${ann.tag ? " · tagged " + ann.tag : ""}`;
    const prev = document.getElementById("nav-prev"), next = document.getElementById("nav-next");
    if (data.nav.prev) prev.href = `/annotate/${data.nav.prev}${dsQuery}`; else prev.style.visibility = "hidden";
    if (data.nav.next) next.href = `/annotate/${data.nav.next}${dsQuery}`; else next.style.visibility = "hidden";
    img.onload = fitCanvas;
    img.src = `/api/images/${ann.id}/file`;
    renderSidebar();
    loadDatasets();
    initAssistant();
  } catch (e) {
    toast(e.message, "error");
  }
})();

// ---- labeling assistant ----------------------------------------------------
// A model on the studio machine proposes boxes (GET /api/images/<id>/suggestions, see
// boxqc/studio/assistant.py). The page's job is to make picking the good ones cheap:
//
//   * click a suggestion to add it, right-click to dismiss it;
//   * the confidence slider hides the weaker ones, and "Add all shown" (A) takes the rest;
//   * a suggestion that is never added is never saved, so bad ones need no action at all.
//
// A suggestion is hidden once it was added or dismissed, and also while a box already
// covers the same spot (so re-running on a labeled image only shows what is missing).
const DUPLICATE_IOU = 0.5;          // overlap above which a suggestion repeats an existing box
const ASSIST_PREFS = "boxqc.assistant";
const addedFrom = new WeakMap();    // box -> the suggestion it was added from

// What the user chose last time, kept in this browser: suggest at all, the slider, the model.
const assist = { on: true, minConf: 0.25, model: "" };
try {
  const saved = JSON.parse(localStorage.getItem(ASSIST_PREFS) || "{}");
  if (typeof saved.on === "boolean") assist.on = saved.on;
  if (saved.minConf >= 0.05 && saved.minConf <= 0.95) assist.minConf = saved.minConf;
  if (typeof saved.model === "string") assist.model = saved.model;
} catch { /* private mode or a corrupt value: use the defaults */ }
function saveAssistPrefs() {
  try { localStorage.setItem(ASSIST_PREFS, JSON.stringify(assist)); } catch { /* not remembered, still works */ }
}

// idle: not run for this image · loading · ready: `assistRun` is the server's answer ·
// none: there is no model to ask · error: `assistError` says why
let assistState = "idle";
let assistRun = null;
let assistError = "";
let assistCalls = 0;        // to drop the answer of a run that a newer one has replaced

// The suggestions to draw and list: switched on, sure enough for the slider, not dealt
// with yet, and not a repeat of a box that is already there.
function shownSuggestions() {
  if (!assist.on || !suggestions.length) return [];
  const taken = boxes.map(normBox);
  return suggestions.filter((s) => !s.gone && s.conf >= assist.minConf && !taken.some((b) => iou(b, s) >= DUPLICATE_IOU));
}

function highlightSuggestion(s) {
  if (s === hotSuggestion) return;
  hotSuggestion = s;
  draw();
}

// Turn a suggestion into a box of its own. It is left selected, so 1-9 changes its
// class, its corners can be dragged tighter and Del takes it back out.
function addSuggestion(s) {
  const box = { cls: s.cls, x0: s.x0, y0: s.y0, x1: s.x1, y1: s.y1 };
  addedFrom.set(box, s);
  s.gone = "added";
  boxes.push(box);
  selected = boxes.length - 1;
  currentClass = Math.max(0, CLASSES.indexOf(s.cls));
  hotSuggestion = null;
  dirty = true;
}

function addShownSuggestions() {
  const shown = shownSuggestions();
  if (!shown.length) return;
  shown.forEach(addSuggestion);
  selected = -1;
  update();
}

function dismissSuggestion(s) {
  s.gone = "dismissed";
  hotSuggestion = null;
  update();
}

// Ask the server what the model sees in this image.
async function runAssistant() {
  const call = ++assistCalls;
  assistState = "loading";
  suggestions = [];
  hotSuggestion = null;
  update();
  try {
    const q = assist.model ? `?model=${encodeURIComponent(assist.model)}` : "";
    const run = await api(`/api/images/${ann.id}/suggestions${q}`);
    if (call !== assistCalls) return;
    assistRun = run;
    suggestions = run.suggestions;
    assistState = "ready";
  } catch (e) {
    if (call !== assistCalls) return;
    assistError = e.message;
    assistState = "error";
  }
  update();
}

// Suggestions appear by themselves on images that still need labeling. On a labeled or
// skipped image they would only be noise (everything the user decided not to box), so
// there the user asks for them with the button.
function autoSuggest() {
  if (assist.on && assistState === "idle" && ann.status === "unlabeled") runAssistant();
}

async function initAssistant() {
  const select = document.getElementById("assist-model");
  try {
    const info = await api("/api/assistant");
    if (!info.models.length) { assistState = "none"; update(); return; }
    if (!info.models.some((m) => m.name === assist.model)) assist.model = "";   // deleted since
    select.replaceChildren(
      el("option", { value: "" }, `Newest (${modelLabel(info.default)})`),
      ...info.models.map((m) => el("option", { value: m.name },
        modelLabel(m.name) + (m.dataset_name ? ` · ${m.dataset_name}` : ""))));
    select.value = assist.model;
    document.getElementById("assist-model-wrap").hidden = false;
  } catch (e) {
    assistError = e.message;
    assistState = "error";
  }
  update();
  autoSuggest();
}

// The Assistant panel: what the model found, the slider, and one row per suggestion on display.
function renderAssistant() {
  const $ = (id) => document.getElementById(id);
  const shown = shownSuggestions();
  const ready = assist.on && assistState === "ready";
  $("assist-on").checked = assist.on;
  $("assist-run").hidden = !(ann && assist.on && assistState === "idle");
  $("assist-body").hidden = !(ready && suggestions.length);
  $("assist-conf").value = Math.round(assist.minConf * 100);
  $("assist-conf-val").textContent = `${Math.round(assist.minConf * 100)}%`;
  $("assist-add-all").disabled = !shown.length;
  $("assist-add-all").replaceChildren(
    shown.length ? `Add all ${shown.length} shown` : "Nothing to add", " ", el("kbd", {}, "A"));

  let status;
  if (!assist.on) status = "Off. Turn it on to have a trained model propose boxes.";
  else if (assistState === "none") {
    status = ["No trained model on this computer yet. Label a first batch by hand, ",
      el("a", { href: "/training" }, "train a model"), ", and it will propose boxes for the rest."];
  } else if (assistState === "loading") status = "Looking for defects…";
  else if (assistState === "error") status = assistError;
  else if (assistState === "idle") {
    status = !ann ? "" : ann.status === "unlabeled" ? "" : `This image is already ${ann.status === "skip" ? "skipped" : "labeled"}.`;
  } else {
    const waiting = suggestions.filter((s) => !s.gone).length;
    const ignored = assistRun.ignored_classes.length
      ? ` It also found ${assistRun.ignored_classes.join(", ")}, which boxqc.yaml does not list.` : "";
    const from = modelLabel(assistRun.model);
    status = (!suggestions.length ? `${from} found nothing in this image.`
      : !waiting ? `Every suggestion from ${from} is added or dismissed.`
        : !shown.length ? `Nothing left to show. ${from} has ${waiting} more below the slider or under boxes that are already there.`
          : `${shown.length} of ${waiting} suggestions shown, from ${from}.`) + ignored;
  }
  $("assist-status").replaceChildren(...[status].flat());
  $("assist-status").className = "small " + (assist.on && assistState === "error" ? "error-text" : "muted");

  $("assist-list").replaceChildren(...(ready ? shown : []).map((s) => {
    const row = el("tr", { class: "clickable", title: "Add this box", onclick: () => { addSuggestion(s); update(); } },
      el("td", {}, el("span", { class: "swatch", style: `background:${classColor(CLASSES.indexOf(s.cls))}` }), s.cls),
      el("td", { class: "muted" }, `${Math.round(s.conf * 100)}%`),
      el("td", {}, el("button", { title: "Dismiss this suggestion",
        onclick: (e) => { e.stopPropagation(); dismissSuggestion(s); } }, "✕")));
    row.addEventListener("pointerenter", () => highlightSuggestion(s));
    row.addEventListener("pointerleave", () => highlightSuggestion(null));
    return row;
  }));
}

document.getElementById("assist-on").addEventListener("change", (e) => {
  assist.on = e.target.checked;
  saveAssistPrefs();
  update();
  autoSuggest();
});
document.getElementById("assist-conf").addEventListener("input", (e) => {
  assist.minConf = e.target.value / 100;
  saveAssistPrefs();
  update();
});
document.getElementById("assist-model").addEventListener("change", (e) => {
  assist.model = e.target.value;
  saveAssistPrefs();
  if (assist.on && assistState !== "idle") runAssistant();     // show the other model's view of this image
});
document.getElementById("assist-run").onclick = runAssistant;
document.getElementById("assist-add-all").onclick = addShownSuggestions;

// ---- dataset membership of this image --------------------------------------
async function loadDatasets() {
  const summary = await api("/api/datasets");
  const ctx = document.getElementById("ds-context");
  ctx.replaceChildren(DATASET
    ? el("span", {}, "Browsing ", el("a", { href: datasetPage }, datasetLabel(DATASET)),
      ": newer/older and \"next unlabeled\" stay inside it.")
    : "Browsing all images.");
  const list = document.getElementById("ds-list");
  if (!summary.datasets.length) {
    list.replaceChildren(el("span", { class: "small muted" }, "No datasets yet."));
    return;
  }
  list.replaceChildren(...summary.datasets.map((d) => {
    const box = el("input", { type: "checkbox" });
    box.checked = ann.datasets.includes(d.name);
    box.addEventListener("change", async () => {
      try {
        await api(`/api/datasets/${encodeURIComponent(d.name)}/items`, {
          method: "POST", json: box.checked ? { add: [ann.id] } : { remove: [ann.id] } });
        ann.datasets = box.checked ? [...ann.datasets, d.name] : ann.datasets.filter((x) => x !== d.name);
        toast(box.checked ? `Added to "${d.name}"` : `Removed from "${d.name}"`, "ok");
      } catch (e) {
        box.checked = !box.checked;
        toast(e.message, "error");
      }
    });
    return el("label", {}, box, d.name,
      summary.capture_dataset === d.name ? el("span", { class: "badge ds" }, "capture") : null);
  }));
}

document.getElementById("ds-new-here").onclick = async () => {
  const name = await newDatasetDialog({ capture: false });
  if (!name) return;
  try {
    await api(`/api/datasets/${encodeURIComponent(name)}/items`, { method: "POST", json: { add: [ann.id] } });
    ann.datasets = [...ann.datasets, name];
  } catch (e) { toast(e.message, "error"); }
  loadDatasets();
};
