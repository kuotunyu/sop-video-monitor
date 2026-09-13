"""Single-page review UI served by :mod:`sop_monitor.review` (no external assets)."""

PAGE = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>SOP deviation review</title>
<style>
  :root { --bg:#f6f5f2; --panel:#fff; --ink:#1d1d1b; --muted:#6b6a66; --line:#e2e0da;
          --accept:#1f7a4d; --reject:#b3261e; --skip:#6b6a66; --focus:#2457c5; }
  * { box-sizing:border-box; }
  body { margin:0; font:14px/1.45 system-ui, -apple-system, "Segoe UI", sans-serif; color:var(--ink); background:var(--bg); }
  header { display:flex; gap:16px; align-items:baseline; padding:10px 16px; border-bottom:1px solid var(--line); background:var(--panel); }
  header h1 { font-size:16px; margin:0; }
  header .note { color:var(--muted); font-size:12px; }
  #summary { margin-left:auto; font-size:12px; color:var(--muted); }
  main { display:grid; grid-template-columns: 340px 1fr; height: calc(100vh - 45px); }
  aside { border-right:1px solid var(--line); overflow:auto; background:var(--panel); }
  .filters { display:flex; gap:6px; padding:8px; border-bottom:1px solid var(--line); position:sticky; top:0; background:var(--panel); }
  .filters select { flex:1; font:inherit; padding:3px; }
  .item { padding:8px 10px; border-bottom:1px solid var(--line); cursor:pointer; }
  .item:hover { background:#f0efe9; }
  .item.sel { outline:2px solid var(--focus); outline-offset:-2px; }
  .item .top { display:flex; justify-content:space-between; gap:8px; }
  .kind { font-size:11px; text-transform:uppercase; letter-spacing:.04em; color:var(--muted); }
  .badge { font-size:11px; padding:0 6px; border-radius:8px; color:#fff; }
  .badge.accept { background:var(--accept); } .badge.reject { background:var(--reject); } .badge.skip { background:var(--skip); }
  section { overflow:auto; padding:12px 16px; }
  .videos { display:grid; grid-template-columns: repeat(3, 1fr); gap:8px; }
  .videos figure { margin:0; }
  .videos video { width:100%; background:#000; border-radius:4px; }
  .videos figcaption { font-size:12px; color:var(--muted); }
  .controls { display:flex; gap:8px; align-items:center; margin:8px 0 12px; flex-wrap:wrap; }
  button { font:inherit; padding:6px 12px; border-radius:6px; border:1px solid var(--line); background:var(--panel); cursor:pointer; }
  button.accept { border-color:var(--accept); color:var(--accept); } button.reject { border-color:var(--reject); color:var(--reject); }
  button:focus-visible, select:focus-visible, textarea:focus-visible { outline:2px solid var(--focus); outline-offset:1px; }
  .detail h2 { font-size:15px; margin:4px 0; }
  .detail dl { display:grid; grid-template-columns: 150px 1fr; gap:2px 12px; margin:8px 0; }
  .detail dt { color:var(--muted); }
  textarea { width:100%; min-height:56px; font:inherit; padding:6px; }
  .disclaimer { font-size:12px; color:var(--muted); margin-top:12px; }
  kbd { font-size:11px; border:1px solid var(--line); border-radius:3px; padding:0 4px; background:#fafaf8; }
</style>
</head>
<body>
<header>
  <h1>SOP deviation review</h1>
  <span class="note">development data (HA-ViD val) · suggestions for human review, not compliance decisions</span>
  <span id="summary"></span>
</header>
<main>
  <aside>
    <div class="filters">
      <select id="fKind" aria-label="Filter by kind"><option value="">all kinds</option></select>
      <select id="fStatus" aria-label="Filter by status">
        <option value="">all</option><option value="open">open</option>
        <option value="accept">accepted</option><option value="reject">rejected</option><option value="skip">skipped</option>
      </select>
      <select id="fRec" aria-label="Filter by recording"><option value="">all recordings</option></select>
    </div>
    <div id="list" role="listbox" aria-label="Deviation queue"></div>
  </aside>
  <section>
    <div id="empty">Select an item.</div>
    <div id="detail" class="detail" hidden>
      <div class="videos" id="videos"></div>
      <div class="controls">
        <button id="replay">Replay window <kbd>space</kbd></button>
        <button class="accept" data-d="accept">Accept deviation <kbd>a</kbd></button>
        <button class="reject" data-d="reject">Reject (not a deviation) <kbd>r</kbd></button>
        <button data-d="skip">Skip <kbd>s</kbd></button>
        <span class="note">next <kbd>j</kbd> · previous <kbd>k</kbd></span>
      </div>
      <h2 id="title"></h2>
      <dl id="facts"></dl>
      <label for="note">Note (optional)</label>
      <textarea id="note"></textarea>
      <p class="disclaimer">Accepting records that a human confirmed the deviation in the video. Neither the queue nor the decision is a safety or compliance guarantee.</p>
    </div>
  </section>
</main>
<script>
let items = [], shown = [], current = null;
const $ = (id) => document.getElementById(id);
const esc = (s) => String(s ?? "").replace(/[&<>"']/g, (c) => ({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;","'":"&#39;"}[c]));
const fmt = (s) => s == null ? "—" : `${Math.floor(s/60)}:${(s%60).toFixed(1).padStart(4,"0")}`;

async function load() {
  const r = await fetch("/api/items"); const data = await r.json();
  items = data.items; renderSummary(data.summary);
  const kinds = [...new Set(items.map(i => i.kind))], recs = [...new Set(items.map(i => i.recording))].sort();
  for (const k of kinds) $("fKind").add(new Option(k, k));
  for (const r of recs) $("fRec").add(new Option(r, r));
  renderList();
}
function renderSummary(s) {
  $("summary").textContent = Object.entries(s).map(([k, v]) =>
    `${k}: ${v.accepted + v.rejected + v.skipped}/${v.queued} reviewed` +
    (v.precision_reviewed == null ? "" : `, ${(100*v.precision_reviewed).toFixed(0)} % accepted`)).join(" · ");
}
function status(i) { return i.decision ? i.decision.decision : "open"; }
function renderList() {
  const k = $("fKind").value, st = $("fStatus").value, rec = $("fRec").value;
  shown = items.filter(i => (!k || i.kind === k) && (!rec || i.recording === rec) && (!st || status(i) === st));
  const list = $("list"); list.innerHTML = "";
  for (const i of shown) {
    const div = document.createElement("div");
    div.className = "item" + (current && current.id === i.id ? " sel" : "");
    div.setAttribute("role", "option");
    const st = status(i);
    div.innerHTML = `<div class="top"><span class="kind">${i.kind}</span>${st === "open" ? "" : `<span class="badge ${st}">${st}</span>`}</div>
      <div>${esc(i.step_description)}</div><div class="kind">${esc(i.recording)} · ${esc(i.plate)} · ${fmt(i.start_s)}</div>`;
    div.onclick = () => select(i);
    list.appendChild(div);
  }
}
function select(i) {
  current = i; $("empty").hidden = true; $("detail").hidden = false;
  $("title").textContent = `${i.kind}: ${i.step_description}`;
  const ev = i.evidence, facts = [
    ["recording", `${i.recording} (subject ${i.subject}, ${i.plate} plate)`],
    ["step label", i.step], ["time", i.start_s == null ? "whole recording" : `${fmt(i.start_s)} – ${fmt(i.end_s)}`],
    ["rule", ev.rule], ["source", i.source === "pred" ? "recogniser output" : "ground-truth annotation"],
  ];
  if (ev.missing_predecessors) facts.push(["missing before it", ev.missing_predecessors.map(m => m.description).join("; ")]);
  if (ev.window_s) facts.push(["duration", `${ev.duration_s} s (${ev.kind}); window ${ev.window_s[0]}–${ev.window_s[1]} s`]);
  $("facts").innerHTML = facts.map(([a, b]) => `<dt>${esc(a)}</dt><dd>${esc(b)}</dd>`).join("");
  $("note").value = i.decision ? i.decision.note : "";
  const box = $("videos"); box.innerHTML = "";
  for (const [view, vid] of Object.entries(i.videos)) {
    const fig = document.createElement("figure");
    fig.innerHTML = `<video muted playsinline preload="metadata" src="/video/${encodeURIComponent(vid)}.mp4"></video><figcaption>${esc(view)} · ${esc(vid)}</figcaption>`;
    box.appendChild(fig);
  }
  replay(); renderList();
}
function replay() {
  if (!current) return;
  const vids = [...document.querySelectorAll("#videos video")];
  const start = Math.max(0, (current.start_s ?? 0) - 2), end = current.end_s == null ? null : current.end_s + 2;
  for (const v of vids) {
    const go = () => { v.currentTime = start; v.play().catch(() => {}); };
    if (v.readyState >= 1) go(); else v.addEventListener("loadedmetadata", go, { once: true });
    v.ontimeupdate = end == null ? null : () => { if (v.currentTime > end) v.currentTime = start; };
  }
}
async function decide(d) {
  if (!current) return;
  const r = await fetch("/api/decisions", { method: "POST", headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ id: current.id, decision: d, note: $("note").value }) });
  const data = await r.json(); if (!r.ok) { alert(data.error); return; }
  current.decision = data.decision; renderSummary(data.summary); step(1);
}
function step(delta) {
  const idx = shown.findIndex(i => current && i.id === current.id);
  const next = shown[Math.min(shown.length - 1, Math.max(0, idx + delta))];
  if (next) select(next); else renderList();
}
document.querySelectorAll("button[data-d]").forEach(b => b.onclick = () => decide(b.dataset.d));
$("replay").onclick = replay;
["fKind", "fStatus", "fRec"].forEach(id => $(id).onchange = renderList);
document.addEventListener("keydown", (e) => {
  if (e.target.tagName === "TEXTAREA" || e.target.tagName === "SELECT") return;
  if (e.key === "a") decide("accept"); else if (e.key === "r") decide("reject"); else if (e.key === "s") decide("skip");
  else if (e.key === "j") step(1); else if (e.key === "k") step(-1);
  else if (e.key === " ") { e.preventDefault(); replay(); }
});
load();
</script>
</body>
</html>
"""
