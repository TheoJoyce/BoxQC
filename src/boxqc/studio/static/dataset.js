// Dataset page: browse images by named dataset, manage datasets, add/remove images in bulk.
"use strict";

// State: `current` is the dataset shown (also kept in the URL so links and reloads
// work), `filter` the status tab, `items` the images listed, `selected` the ticked ids.
const $ = (id) => document.getElementById(id);
let current = new URLSearchParams(location.search).get("dataset") || "";  // "" = all, name, or NO_DATASET
let filter = "";            // status filter
let summary = null;
let items = [];
const selected = new Set();
let lastClicked = -1;

const isNamed = () => Boolean(current) && current !== NO_DATASET;
const query = (extra = {}) => {
  const q = new URLSearchParams(extra);
  if (current) q.set("dataset", current);
  return q.toString() ? `?${q}` : "";
};
const annotateHref = (id) => `/annotate/${id}${query()}`;

// Statistics of whatever is shown: all images, images in no dataset, or one dataset.
function currentStats() {
  if (!current) return summary.all;
  if (current === NO_DATASET) return summary.unassigned;
  return (summary.datasets.find((d) => d.name === current) || {}).stats || summary.all;
}

// ---- dataset bar -----------------------------------------------------------
// Fetch the datasets and their statistics, then fill the selectors, description and counters.
async function loadSummary() {
  summary = await api("/api/datasets");
  if (isNamed() && !summary.datasets.some((d) => d.name === current)) current = "";
  history.replaceState(null, "", `/dataset${query()}`);
  fillDatasetSelect($("ds-select"), summary, { value: current, all: true, none: true });
  const prevTarget = $("sel-target").value;  // keep the last pick, but prefer a real dataset over "+ New"
  fillDatasetSelect($("sel-target"), summary, { value: prevTarget === "__new__" ? "" : prevTarget,
    extra: [["__new__", "+ New dataset…"]] });

  const d = summary.datasets.find((x) => x.name === current);
  $("ds-rename").hidden = $("ds-delete").hidden = $("ds-capture-wrap").hidden = !d;
  $("ds-capture").checked = Boolean(d) && summary.capture_dataset === current;
  const cap = summary.capture_dataset;
  let desc;
  if (d) {
    desc = [d.description, `created ${d.created.slice(0, 10)}`,
      cap === d.name ? "new captures are saved here" : null].filter(Boolean).join(" · ");
  } else if (current === NO_DATASET) {
    desc = "Images that aren't in any dataset, such as everything captured before datasets existed.";
  } else {
    desc = `Every image. ${cap ? `New captures are saved to "${cap}".` : "New captures aren't added to any dataset."}`;
  }
  $("ds-desc").textContent = desc;

  const st = currentStats();
  const classStats = Object.entries(st.boxes_by_class).map(([k, v]) =>
    el("div", { class: "stat" }, el("b", {}, v), el("span", {}, `${k} boxes`)));
  $("stats").replaceChildren(
    el("div", { class: "stat" }, el("b", {}, st.total), el("span", {}, "images")),
    el("div", { class: "stat" }, el("b", {}, st.by_status.unlabeled), el("span", {}, "unlabeled")),
    el("div", { class: "stat" }, el("b", {}, st.labeled_clean), el("span", {}, "labeled clean")),
    el("div", { class: "stat" }, el("b", {}, st.labeled_defective), el("span", {}, "labeled defective")),
    ...classStats,
  );
}

$("ds-select").addEventListener("change", (e) => {
  current = e.target.value;
  selected.clear();
  refresh();
});

$("ds-new").onclick = async () => {
  const name = await newDatasetDialog({ capture: true });
  if (!name) return;
  current = name;
  selected.clear();
  refresh();
};

$("ds-rename").onclick = async () => {
  const name = prompt(`New name for "${current}"`, current);
  if (!name || name.trim() === current) return;
  try {
    const d = await api(`/api/datasets/${encodeURIComponent(current)}`, { method: "PATCH", json: { name: name.trim() } });
    current = d.name;
    toast(`Renamed to "${d.name}"`, "ok");
  } catch (e) { toast(e.message, "error"); }
  refresh();
};

$("ds-delete").onclick = async () => {
  const n = currentStats().total;
  if (!confirm(`Delete dataset "${current}"?\n\nIts ${n} image(s) are kept; they are only removed from this dataset.`)) return;
  try {
    await api(`/api/datasets/${encodeURIComponent(current)}`, { method: "DELETE" });
    toast(`Deleted dataset "${current}"`, "ok");
    current = "";
  } catch (e) { toast(e.message, "error"); }
  refresh();
};

$("ds-capture").addEventListener("change", async (e) => {
  try { await setCaptureDataset(e.target.checked ? current : null); } catch (err) { toast(err.message, "error"); }
  loadSummary();
});

// ---- grid -----------------------------------------------------------------
// One image tile: thumbnail (links to the annotator), tick box and status badges.
function card(a, index) {
  const defects = a.boxes.length;
  const others = a.datasets.filter((d) => d !== current);
  const pick = el("input", { type: "checkbox", "aria-label": `Select ${a.id}` });
  pick.checked = selected.has(a.id);
  pick.addEventListener("click", (e) => { e.stopPropagation(); toggle(index, e.shiftKey); });
  const link = el("a", { href: annotateHref(a.id) },
    el("img", { src: `/api/images/${a.id}/thumb`, loading: "lazy", alt: a.id }));
  link.addEventListener("click", (e) => {  // while selecting, clicking an image selects it
    if (selected.size) { e.preventDefault(); toggle(index, e.shiftKey); }
  });
  return el("div", { class: "card" + (selected.has(a.id) ? " picked" : "") },
    el("label", { class: "pick", title: "Select (shift-click for a range)" }, pick),
    link,
    el("div", { class: "meta" },
      el("span", { class: `badge ${a.status}` }, a.status),
      a.status === "labeled" ? el("span", { class: "badge" + (defects ? " defect" : "") }, defects ? `${defects} box${defects > 1 ? "es" : ""}` : "clean") : null,
      a.status === "unlabeled" && defects ? el("span", { class: "badge" }, `${defects} suggested`) : null,
      a.tag ? el("span", { class: "badge" }, a.tag) : null,
      a.source !== "capture" ? el("span", { class: "badge" }, a.source) : null,
      ...others.slice(0, 2).map((d) => el("span", { class: "badge ds", title: "dataset" }, d)),
      others.length > 2 ? el("span", { class: "badge ds", title: others.slice(2).join(", ") }, `+${others.length - 2}`) : null,
      el("span", { class: "muted", style: "margin-left:auto" }, a.id.slice(0, 15)),
    ));
}

// Tick or untick an image; with `range` (shift-click) everything since the last click too.
function toggle(index, range) {
  const id = items[index].id;
  const on = !selected.has(id);
  const [from, to] = range && lastClicked >= 0 ? [Math.min(lastClicked, index), Math.max(lastClicked, index)] : [index, index];
  for (let i = from; i <= to; i++) on ? selected.add(items[i].id) : selected.delete(items[i].id);
  lastClicked = index;
  renderGrid();
}

function renderGrid() {
  $("grid").replaceChildren(...items.map(card));
  const empty = $("empty");
  empty.hidden = items.length > 0;
  empty.replaceChildren(isNamed() && !filter
    ? `"${current}" has no images yet. Capture with "Save new captures here" on, or select images under "All images" and add them.`
    : "No images here.");
  $("selbar").hidden = selected.size === 0;
  $("sel-count").textContent = `${selected.size} selected`;
  $("sel-remove").hidden = !isNamed();
  $("sel-remove").textContent = `Remove from "${current}"`;
}

// Fetch the images for the current dataset and status tab, and point the "Label next
// unlabeled" button at the first unlabeled one.
async function loadItems() {
  items = await api(`/api/images${query(filter ? { status: filter } : {})}`);
  for (const id of [...selected]) if (!items.some((a) => a.id === id)) selected.delete(id);
  lastClicked = -1;
  renderGrid();
  const unl = filter === "unlabeled" ? items : await api(`/api/images${query({ status: "unlabeled" })}`);
  const btn = $("start-labeling");
  btn.href = unl.length ? annotateHref(unl[0].id) : "#";
  btn.style.visibility = unl.length ? "visible" : "hidden";
}

async function refresh() {
  try {
    await loadSummary();
    await loadItems();
  } catch (e) { toast(e.message, "error"); }
}

$("tabs").addEventListener("click", (e) => {
  const b = e.target.closest("button");
  if (!b) return;
  filter = b.dataset.status;
  document.querySelectorAll("#tabs button").forEach((x) => x.classList.toggle("on", x === b));
  loadItems();
});

// ---- selection actions ----------------------------------------------------
$("sel-all").onclick = () => { items.forEach((a) => selected.add(a.id)); renderGrid(); };
$("sel-clear").onclick = () => { selected.clear(); renderGrid(); };

$("sel-add").onclick = async () => {
  let target = $("sel-target").value;
  if (target === "__new__") {
    target = await newDatasetDialog({ capture: false });
    if (!target) return;
  }
  try {
    const r = await api(`/api/datasets/${encodeURIComponent(target)}/items`, { method: "POST", json: { add: [...selected] } });
    toast(`Added ${r.changed} image(s) to "${target}"${r.changed < selected.size ? " (the rest were already in it)" : ""}`, "ok");
    selected.clear();
  } catch (e) { toast(e.message, "error"); }
  refresh();
};

$("sel-remove").onclick = async () => {
  try {
    const r = await api(`/api/datasets/${encodeURIComponent(current)}/items`, { method: "POST", json: { remove: [...selected] } });
    toast(`Removed ${r.changed} image(s) from "${current}". The images themselves are kept.`, "ok");
    selected.clear();
  } catch (e) { toast(e.message, "error"); }
  refresh();
};

document.addEventListener("keydown", (e) => {
  if (notAHotkey(e)) return;
  if (e.key === "Escape" && selected.size) { selected.clear(); renderGrid(); }
});

// ---- upload -----------------------------------------------------------------
$("upload").addEventListener("change", async (e) => {
  const files = [...e.target.files];
  if (!files.length) return;
  const form = new FormData();
  files.forEach((f) => form.append("files", f));
  if (isNamed()) form.append("dataset", current);
  try {
    const r = await api("/api/images/upload", { method: "POST", form });
    toast(`Uploaded ${r.added.length} image(s)`, "ok");
  } catch (err) { toast(err.message, "error"); }
  e.target.value = "";
  refresh();
});

refresh();
