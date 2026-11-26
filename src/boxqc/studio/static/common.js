// Shared helpers for all pages.
//
// The pages talk to the studio server only. Paths under /api/device/ are forwarded by
// the studio to the inspection device's API (/api/v1/...); everything else under /api/
// is the studio's own (dataset, models, training).
"use strict";

const DEVICE = "/api/device";

// Every page is server-rendered from templates/base.html, which loads this file first
// and sets window.BOXQC = { classes, deviceUrl }. There is no framework and no build
// step: each page has one script that fetches JSON with api() and builds DOM with el().

// Box colours per class index. Same order as _PALETTE in boxqc/imaging.py (which is BGR),
// so a class has the same colour in the browser and in images drawn by the server.
const PALETTE = ["#ffc800", "#c850ff", "#00c8ff", "#50ff50", "#ff3c3c", "#3c78ff"];
const classColor = (i) => PALETTE[((i % PALETTE.length) + PALETTE.length) % PALETTE.length];

// Call the studio's API. `json` sends a JSON body, `form` a FormData (file uploads).
// Resolves to the parsed response; rejects with an Error whose message is the server's
// `detail` text, ready to show in a toast.
async function api(path, { method = "GET", json, form } = {}) {
  const opts = { method, headers: {} };
  if (json !== undefined) {
    opts.headers["Content-Type"] = "application/json";
    opts.body = JSON.stringify(json);
  } else if (form !== undefined) {
    opts.body = form;
  }
  const res = await fetch(path, opts);
  const text = await res.text();
  let data = null;
  try { data = text ? JSON.parse(text) : null; } catch { data = text; }
  if (res.status === 401 && data && data.access) {
    // Serve mode, and this browser has no (or an outdated) access key: ask for it.
    location.href = `${data.access}?next=${encodeURIComponent(location.pathname + location.search)}`;
    throw new Error("This studio needs its access key.");
  }
  if (!res.ok) {
    const msg = (data && data.detail) ? data.detail : `${res.status} ${res.statusText}`;
    throw new Error(typeof msg === "string" ? msg : JSON.stringify(msg));
  }
  return data;
}

// Show a short message at the bottom of the page. kind: "info" | "ok" | "error".
let toastTimer;
function toast(msg, kind = "info") {
  const el = document.getElementById("toast");
  el.textContent = msg;
  el.className = `toast ${kind}`;
  el.hidden = false;
  clearTimeout(toastTimer);
  toastTimer = setTimeout(() => { el.hidden = true; }, kind === "error" ? 6000 : 2500);
}

// Create a DOM element: el("button", { class: "primary", onclick: fn }, "Save").
// `class` sets className, `on*` adds an event listener, other attributes are set unless
// null/undefined/false. Children may be nodes, strings, numbers or (nested) arrays;
// null and undefined are skipped. Text is never parsed as HTML, so untrusted strings
// (dataset names, notes) are safe to pass.
function el(tag, attrs = {}, ...children) {
  const node = document.createElement(tag);
  for (const [k, v] of Object.entries(attrs)) {
    if (k === "class") node.className = v;
    else if (k.startsWith("on")) node.addEventListener(k.slice(2), v);
    else if (v !== null && v !== undefined && v !== false) node.setAttribute(k, v);
  }
  for (const c of children.flat()) {
    if (c === null || c === undefined) continue;
    node.append(c instanceof Node ? c : document.createTextNode(String(c)));
  }
  return node;
}

// ---- hotkeys ---------------------------------------------------------------
// A page's keydown handler starts with `if (notAHotkey(e)) return;`. (The labeling page
// uses the two halves separately, because it also handles Ctrl/Cmd+S.)

// Inputs that take no text. A checkbox or slider keeps the keyboard focus after a click,
// and that must not switch the page's hotkeys off.
const NON_TEXT_INPUTS = new Set(["checkbox", "radio", "range", "button", "submit", "reset", "file", "color", "image"]);
const SLIDER_KEYS = new Set(["ArrowLeft", "ArrowRight", "ArrowUp", "ArrowDown", "Home", "End", "PageUp", "PageDown"]);

// Is the key going into a field (or a dialog) rather than to the page?
function isTyping(e) {
  if (document.querySelector("dialog[open]")) return true;
  const t = e.target;
  if (!t || !t.tagName) return false;
  if (t.tagName === "INPUT") {
    if (t.type === "range") return SLIDER_KEYS.has(e.key);     // these keys move the slider
    return !NON_TEXT_INPUTS.has(t.type);
  }
  return t.tagName === "TEXTAREA" || t.tagName === "SELECT" || t.isContentEditable;
}

// Ctrl, Cmd or Alt with a key is a browser or system shortcut (copy, cut, bookmark, ...),
// never one of ours. Without this check Cmd+C on the labeling page would run "C = clean".
const hasModifier = (e) => e.ctrlKey || e.metaKey || e.altKey;

const notAHotkey = (e) => isTyping(e) || hasModifier(e);

// A dropdown keeps the focus after a choice, and typing in it selects options, so the
// hotkeys would stay dead until the user clicked elsewhere. Give the focus back.
document.addEventListener("change", (e) => {
  if (e.target instanceof HTMLSelectElement) e.target.blur();
});

const modelLabel = (name) => name.replace(/\.[A-Za-z0-9]+$/, "");

// --topbar-h: how much of the window the sticky top bar covers (0 where it scrolls
// away, on phones), so that other sticky bars (.selbar) stop just below it whatever its
// height: one row on a desktop, two on a tablet with finger-sized links.
document.addEventListener("DOMContentLoaded", () => {
  const bar = document.querySelector(".topbar");
  if (!bar || !window.ResizeObserver) return;
  const measure = () => document.documentElement.style.setProperty(
    "--topbar-h", getComputedStyle(bar).position === "sticky" ? `${bar.offsetHeight}px` : "0px");
  new ResizeObserver(measure).observe(bar);
  window.addEventListener("resize", measure);
});

// Top-bar pills. Returns the studio status; its `device` field is the device's own
// status, or null (with `device_error`) while the device cannot be reached.
async function refreshPills() {
  try {
    const s = await api("/api/status");
    const d = s.device;
    const cam = document.getElementById("pill-camera");
    cam.textContent = !d ? "device offline" : d.camera ? `camera: ${d.camera}` : "camera: offline";
    cam.className = "pill " + (!d || !d.camera ? "bad" : d.camera === "mock" ? "warn" : "good");
    cam.title = d ? `Device ${d.hostname} at ${s.device_url}` : `${s.device_error || "Device unreachable"}`;
    const mod = document.getElementById("pill-model");
    mod.replaceChildren(
      el("span", { class: "pill-text" }, !d ? "model: –" : d.model ? `model: ${modelLabel(d.model)}` : "no model"),
      el("span", { class: "caret", "aria-hidden": "true" }));
    mod.className = "pill " + (d && d.model ? "good" : "warn");
    mod.title = (d && d.model_error) || "Switch the device's active model";
    return s;
  } catch {
    return null;
  }
}

document.addEventListener("DOMContentLoaded", refreshPills);

// ---- model switcher (the model pill in the top bar) -------------------------
// A dropdown listing the models on the device; choosing one makes it the active model.
// Pages that show model details listen for the "boxqc:model-changed" event it fires.
const fmtPct = (v) => (typeof v === "number" ? v.toFixed(2) : null);

function modelMenuItem(m) {
  const info = m.info || {};
  const bits = [
    info.trained_at ? info.trained_at.slice(0, 16).replace("T", " ") : m.modified,
    fmtPct(info.metrics && info.metrics.mAP50) && `mAP50 ${fmtPct(info.metrics.mAP50)}`,
    info.recommended_threshold && `thr ${info.recommended_threshold}`,
    info.imgsz && `${info.imgsz}px`,
    info.dataset_name && `data: ${info.dataset_name}`,
  ].filter(Boolean);
  return el("button", {
    type: "button", role: "option", class: "menu-item" + (m.active ? " active" : ""),
    "aria-selected": m.active ? "true" : "false", "data-name": m.name,
  },
  el("span", { class: "menu-check", "aria-hidden": "true" }, m.active ? "✓" : ""),
  el("span", { class: "menu-text" },
    el("span", { class: "menu-title" }, modelLabel(m.name)),
    el("span", { class: "menu-sub" }, bits.join(" · "))));
}

async function openModelMenu() {
  const btn = document.getElementById("pill-model"), list = document.getElementById("model-list");
  list.replaceChildren(el("div", { class: "menu-note" }, "Loading models…"));
  list.hidden = false;
  btn.setAttribute("aria-expanded", "true");
  let models;
  try { models = await api(`${DEVICE}/models`); } catch (e) {
    list.replaceChildren(el("div", { class: "menu-note" }, e.message));
    return;
  }
  list.replaceChildren(
    el("div", { class: "menu-head" }, "Active model"),
    ...(models.length ? models.map(modelMenuItem) : [el("div", { class: "menu-note" }, "No models on the device yet.")]),
    el("a", { class: "menu-foot", href: "/training#models" }, "Model details →"));
  (list.querySelector(".menu-item.active") || list.querySelector(".menu-item"))?.focus();
}

function closeModelMenu(focusButton = false) {
  const btn = document.getElementById("pill-model"), list = document.getElementById("model-list");
  if (!list || list.hidden) return;
  list.hidden = true;
  btn.setAttribute("aria-expanded", "false");
  if (focusButton) btn.focus();
}

async function activateModel(name) {
  closeModelMenu(true);
  try {
    await api(`${DEVICE}/models/${encodeURIComponent(name)}/activate`, { method: "POST" });
    toast(`Active model: ${name}`, "ok");
  } catch (e) { toast(e.message, "error"); }
  await refreshPills();
  document.dispatchEvent(new CustomEvent("boxqc:model-changed", { detail: { name } }));
}

document.addEventListener("DOMContentLoaded", () => {
  const btn = document.getElementById("pill-model"), list = document.getElementById("model-list");
  if (!btn) return;
  btn.addEventListener("click", () => (list.hidden ? openModelMenu() : closeModelMenu()));
  list.addEventListener("click", (e) => {
    const item = e.target.closest(".menu-item");
    if (!item) return;
    if (item.classList.contains("active")) closeModelMenu(true);
    else activateModel(item.dataset.name);
  });
  list.addEventListener("keydown", (e) => {
    const items = [...list.querySelectorAll(".menu-item")];
    const i = items.indexOf(document.activeElement);
    if (e.key === "ArrowDown" || e.key === "ArrowUp") {
      e.preventDefault();
      const next = items[(i + (e.key === "ArrowDown" ? 1 : -1) + items.length) % items.length];
      next?.focus();
    } else if (e.key === "Home" || e.key === "End") {
      e.preventDefault();
      items[e.key === "Home" ? 0 : items.length - 1]?.focus();
    }
  });
  document.addEventListener("keydown", (e) => {
    if (e.key === "Escape" && !list.hidden) { e.stopPropagation(); closeModelMenu(true); }
  }, true);
  document.addEventListener("pointerdown", (e) => {
    if (!list.hidden && !e.target.closest(".model-menu")) closeModelMenu();
  });
});

// ---- named datasets (Capture, Dataset, Annotate, Inspect and Training pages) --
const NO_DATASET = "__none__";   // filter value: images in no dataset

function datasetLabel(name) {
  return name === NO_DATASET ? "Not in any dataset" : name || "All images";
}

// Fill a <select> from /api/datasets. opts: all / none (extra entries), value, count(stats) -> text
function fillDatasetSelect(select, summary, { value = "", all = false, allLabel = "All images",
  none = false, noneLabel = "Not in any dataset", count = (st) => st.total, extra = [] } = {}) {
  const opt = (v, text, st) => el("option", { value: v }, st ? `${text} (${count(st)})` : text);
  select.replaceChildren(
    ...(all ? [opt("", allLabel, summary.all)] : []),
    ...summary.datasets.map((d) => opt(d.name, d.name, d.stats)),
    ...(none ? [opt(NO_DATASET, noneLabel, summary.unassigned)] : []),
    ...extra.map(([v, text]) => opt(v, text)));
  const values = [...select.options].map((o) => o.value);
  select.value = values.includes(value) ? value : values[0] ?? "";
}

async function setCaptureDataset(name) {
  const r = await api("/api/datasets/capture", { method: "PUT", json: { name: name || null } });
  toast(r.capture_dataset ? `New captures go to "${r.capture_dataset}"` : "New captures go to no dataset", "ok");
  return r.capture_dataset;
}

// Shows the New dataset dialog; resolves to the created dataset's name, or null if cancelled.
function newDatasetDialog({ capture = true } = {}) {
  const dlg = document.getElementById("new-dataset"), form = document.getElementById("new-dataset-form");
  const err = document.getElementById("new-dataset-error");
  form.reset();
  form.capture.checked = capture;
  err.hidden = true;
  dlg.showModal();
  form.name.focus();
  return new Promise((resolve) => {
    const done = (value) => {
      form.onsubmit = null;
      document.getElementById("new-dataset-cancel").onclick = null;
      dlg.onclose = null;
      if (dlg.open) dlg.close();
      resolve(value);
    };
    document.getElementById("new-dataset-cancel").onclick = () => done(null);
    dlg.onclose = () => done(null);
    form.onsubmit = async (e) => {
      e.preventDefault();
      try {
        const d = await api("/api/datasets", { method: "POST", json: {
          name: form.name.value.trim(), description: form.description.value.trim(), capture: form.capture.checked } });
        toast(`Created dataset "${d.name}"`, "ok");
        done(d.name);
      } catch (ex) {
        err.textContent = ex.message;
        err.hidden = false;
      }
    };
  });
}
