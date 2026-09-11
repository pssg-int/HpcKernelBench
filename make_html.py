#!/usr/bin/env python3
"""
Generate the self-contained interactive table: output/index.html

Reads output/included.json. Everything (data, CSS, JS) is inlined; open the file
directly in a browser. Charts follow the dataviz skill: single sequential hue for
magnitude bars, direct labels, hover tooltips, light/dark via prefers-color-scheme.
"""

import json
import os
from collections import Counter

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "output")

CAT_LABELS = {
    "dense_la": "Dense LA",
    "sparse_la": "Sparse LA",
    "stencil_pde": "Stencil/PDE",
    "fft_spectral": "FFT/Spectral",
    "tensor": "Tensor",
    "graph": "Graph",
    "primitives": "Primitives",
    "nbody_md": "N-body/MD",
    "unstructured_amr": "Mesh/AMR/PIC",
    "solver_components": "Solvers",
    "compression": "Compression",
    "ml_kernels": "ML kernels",
}
CAT_FULL = {
    "dense_la": "Dense linear algebra (GEMM, factorizations, eigensolvers)",
    "sparse_la": "Sparse linear algebra (SpMV, SpMM, SpGEMM, SpTRSV, formats)",
    "stencil_pde": "Stencil / structured-grid PDE / lattice Boltzmann",
    "fft_spectral": "FFT / NTT / spectral transforms",
    "tensor": "Tensor algebra (contraction, MTTKRP, tensor networks)",
    "graph": "Graph kernels (BFS, PageRank, triangle counting, mining)",
    "primitives": "Parallel primitives (sort, scan, hash, string/sequence)",
    "nbody_md": "N-body / molecular dynamics / FMM / particle-mesh",
    "unstructured_amr": "Unstructured mesh / AMR / particle-in-cell",
    "solver_components": "Iterative-solver components (Krylov, AMG, preconditioners)",
    "compression": "Compression kernels (error-bounded lossy, lossless)",
    "ml_kernels": "ML kernels crossing into HPC (tensor-core GEMM, GNN, conv)",
}


def main():
    with open(os.path.join(OUT, "included.json")) as f:
        papers = json.load(f)

    rows = []
    for p in papers:
        rows.append({
            "t": p["title"],
            "v": p["venue"],
            "y": p["year"],
            "c": p["categories"],
            "k": "; ".join(p["kernels"])[:160],
            "pl": p["platform"],
            "ap": p["approach"] or "",
            "ci": p["cited_by"],
            "d": ("https://doi.org/" + p["doi"]) if p["doi"] else p["dblp_url"],
            "o": p["one_liner"],
            "cf": p["confidence"],
            "as": p.get("artifact_status", "none"),
            "au": p.get("artifact_url", ""),
            "st": p.get("artifact_stars", 0),
        })

    cat_counts = Counter(c for p in papers for c in p["categories"])
    year_counts = Counter(p["year"] for p in papers)
    venue_counts = Counter(p["venue"] for p in papers)

    data_js = json.dumps(rows, ensure_ascii=False)
    cats_js = json.dumps([
        {"id": c, "label": CAT_LABELS[c], "full": CAT_FULL[c],
         "n": cat_counts.get(c, 0)}
        for c in sorted(CAT_LABELS, key=lambda c: -cat_counts.get(c, 0))
    ], ensure_ascii=False)
    years_js = json.dumps(sorted(year_counts.items()))
    venues_js = json.dumps(venue_counts.most_common())

    html = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>HPC Kernel-Optimization Papers 2020+</title>
<style>
:root {
  --surface: #fcfcfb; --ink: #1d1d1c; --ink-2: #55554f; --ink-3: #8a8a82;
  --card: #ffffff; --line: #e6e6e1; --accent: #256abf; --bar: #3987e5;
  --chip: #eef2f7; --chip-ink: #3a4a5e; --hover: #f3f6fa;
}
@media (prefers-color-scheme: dark) {
  :root {
    --surface: #1a1a19; --ink: #ececea; --ink-2: #b0b0a8; --ink-3: #7d7d75;
    --card: #232322; --line: #3a3a37; --accent: #86b6ef; --bar: #5598e7;
    --chip: #2d3743; --chip-ink: #b9c8da; --hover: #2a2a28;
  }
}
* { box-sizing: border-box; margin: 0; }
body { background: var(--surface); color: var(--ink);
  font: 15px/1.5 -apple-system, "Segoe UI", Roboto, "Noto Sans", sans-serif;
  padding: 28px 4vw 80px; }
h1 { font-size: 26px; letter-spacing: -0.02em; }
.sub { color: var(--ink-2); margin: 6px 0 26px; max-width: 70ch; }
.tiles { display: flex; flex-wrap: wrap; gap: 14px; margin-bottom: 30px; }
.tile { background: var(--card); border: 1px solid var(--line); border-radius: 10px;
  padding: 14px 20px; min-width: 150px; }
.tile b { display: block; font-size: 26px; font-variant-numeric: tabular-nums; }
.tile span { color: var(--ink-2); font-size: 13px; }
.charts { display: grid; grid-template-columns: minmax(300px,1.3fr) minmax(260px,1fr);
  gap: 18px; margin-bottom: 34px; }
@media (max-width: 900px){ .charts { grid-template-columns: 1fr; } }
.card { background: var(--card); border: 1px solid var(--line); border-radius: 10px;
  padding: 18px 20px; }
.card h2 { font-size: 15px; margin-bottom: 14px; color: var(--ink); }
.hbar-row { display: grid; grid-template-columns: 110px 1fr 40px; align-items: center;
  gap: 8px; margin: 3px 0; cursor: pointer; }
.hbar-row .lbl { font-size: 13px; color: var(--ink-2); text-align: right;
  overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
.hbar-track { height: 16px; }
.hbar { height: 16px; background: var(--bar); border-radius: 0 4px 4px 0;
  min-width: 2px; }
.hbar-row:hover .hbar { filter: brightness(1.12); }
.hbar-row.active .hbar { outline: 2px solid var(--accent); outline-offset: 1px; }
.hbar-row .num { font-size: 12.5px; color: var(--ink-2);
  font-variant-numeric: tabular-nums; }
.vbars { display: flex; align-items: flex-end; gap: 6px; height: 150px;
  padding-top: 18px; }
.vbar-w { flex: 1; display: flex; flex-direction: column; align-items: center;
  gap: 4px; height: 100%; justify-content: flex-end; }
.vbar { width: 100%; max-width: 46px; background: var(--bar);
  border-radius: 4px 4px 0 0; min-height: 2px; }
.vbar-w:hover .vbar { filter: brightness(1.12); }
.vbar-w .n { font-size: 11.5px; color: var(--ink-2);
  font-variant-numeric: tabular-nums; }
.vbar-w .yl { font-size: 11.5px; color: var(--ink-3); }
.filters { display: flex; flex-wrap: wrap; gap: 10px; margin-bottom: 14px;
  align-items: center; }
.filters input, .filters select { background: var(--card); color: var(--ink);
  border: 1px solid var(--line); border-radius: 8px; padding: 7px 10px;
  font-size: 14px; }
.filters input { width: 260px; }
.count-note { color: var(--ink-2); font-size: 13.5px; margin-left: auto; }
table { width: 100%; border-collapse: collapse; background: var(--card);
  border: 1px solid var(--line); border-radius: 10px; overflow: hidden; }
thead th { text-align: left; font-size: 12.5px; text-transform: uppercase;
  letter-spacing: 0.04em; color: var(--ink-3); padding: 10px 12px;
  border-bottom: 1px solid var(--line); cursor: pointer; user-select: none;
  white-space: nowrap; background: var(--card); position: sticky; top: 0; }
thead th.sorted { color: var(--accent); }
tbody td { padding: 10px 12px; border-bottom: 1px solid var(--line);
  vertical-align: top; font-size: 14px; }
tbody tr:hover { background: var(--hover); }
td.title a { color: var(--ink); text-decoration: none; font-weight: 600; }
td.title a:hover { color: var(--accent); text-decoration: underline; }
td.title .ol { display: block; color: var(--ink-2); font-size: 13px;
  font-weight: 400; margin-top: 2px; max-width: 62ch; }
.chip { display: inline-block; background: var(--chip); color: var(--chip-ink);
  border-radius: 6px; padding: 1px 7px; font-size: 12px; margin: 1px 2px 1px 0;
  white-space: nowrap; }
td.num { font-variant-numeric: tabular-nums; text-align: right; }
.venue-year { white-space: nowrap; color: var(--ink-2); }
a.art { white-space: nowrap; text-decoration: none; font-size: 13px;
  border: 1px solid var(--line); border-radius: 6px; padding: 1px 7px; }
a.art.verified { color: var(--accent); border-color: var(--accent); }
a.art.likely { color: var(--ink-2); }
a.art:hover { background: var(--hover); }
.stars { color: var(--ink-3); font-size: 12px; white-space: nowrap; }
.kern { color: var(--ink-2); font-size: 13px; max-width: 26ch; }
.footer { margin-top: 26px; color: var(--ink-3); font-size: 12.5px;
  max-width: 90ch; }
.wrap-scroll { overflow-x: auto; }
</style>
</head>
<body>
<h1>HPC Kernel-Optimization Papers, 2020&ndash;2026</h1>
<p class="sub">Main-track papers whose primary contribution is an optimized
computational-kernel implementation (or kernel-producing code generator), mined
from SC, IPDPS, ICS, HPDC, ASPLOS, PPoPP, CGO, USENIX ATC, DAC and TPDS via DBLP
+ OpenAlex, classified with LLM assistance and a verification pass. Click a
category bar to filter the table.</p>

<div class="tiles" id="tiles"></div>

<div class="charts">
  <div class="card"><h2>Papers per kernel category (a paper may count in several)</h2>
    <div id="catbars"></div></div>
  <div class="card"><h2>Papers per year</h2><div class="vbars" id="yearbars"></div>
    <h2 style="margin-top:22px">Papers per venue</h2><div id="venuebars"></div></div>
</div>

<div class="filters">
  <input id="q" type="search" placeholder="Search title / kernel / summary&hellip;">
  <select id="fcat"><option value="">All categories</option></select>
  <select id="fven"><option value="">All venues</option></select>
  <select id="fyear"><option value="">All years</option></select>
  <select id="fplat"><option value="">All platforms</option>
    <option>nvidia-gpu</option><option>amd-gpu</option><option>intel-gpu</option>
    <option>cpu</option><option>arm-cpu</option><option>fpga</option>
    <option>distributed</option></select>
  <select id="fappr"><option value="">All approaches</option>
    <option>manual</option><option>codegen</option><option>autotuning</option>
    <option>library</option><option>algorithmic</option></select>
  <select id="fart"><option value="">Artifact: any</option>
    <option value="has">has artifact</option>
    <option value="verified">verified only</option>
    <option value="likely">likely only</option>
    <option value="none">no artifact</option></select>
  <span class="count-note" id="cnt"></span>
</div>

<div class="wrap-scroll">
<table>
  <thead><tr>
    <th data-k="t">Title</th>
    <th data-k="vy">Venue</th>
    <th data-k="c">Categories</th>
    <th data-k="k">Kernels</th>
    <th data-k="pl">Platform</th>
    <th data-k="ap">Approach</th>
    <th data-k="ci" class="num">Cites</th>
    <th data-k="st">Artifact</th>
  </tr></thead>
  <tbody id="tb"></tbody>
</table>
</div>

<p class="footer">Built by Claude Code from public metadata (DBLP, OpenAlex,
Semantic Scholar, Crossref, USENIX). Classification is LLM-assisted from
title+abstract and may contain errors; verdicts marked low-confidence were
re-checked in a second pass. Categories: a paper may belong to several.
Citation counts are OpenAlex snapshots and lag Google Scholar.</p>

<script>
const DATA = __DATA__;
const CATS = __CATS__;
const YEARS = __YEARS__;
const VENUES = __VENUES__;

const el = id => document.getElementById(id);
const catLabel = Object.fromEntries(CATS.map(c => [c.id, c.label]));

// stat tiles
function tiles() {
  const nArt = DATA.filter(r => r.as !== 'none').length;
  el('tiles').innerHTML = [
    [DATA.length, 'kernel papers included'],
    [nArt, 'with open-source artifact'],
    ['7029', 'main-track papers scanned'],
    ['10', 'venues'],
  ].map(([b, s]) => `<div class="tile"><b>${b}</b><span>${s}</span></div>`).join('');
}
tiles();

// category bars (click to filter)
const maxCat = Math.max(...CATS.map(c => c.n));
el('catbars').innerHTML = CATS.map(c =>
  `<div class="hbar-row" data-cat="${c.id}" title="${c.full}: ${c.n} papers">
     <span class="lbl">${c.label}</span>
     <span class="hbar-track"><span class="hbar" style="width:${100 * c.n / maxCat}%"></span></span>
     <span class="num">${c.n}</span></div>`).join('');

// year bars
const maxY = Math.max(...YEARS.map(([, n]) => n));
el('yearbars').innerHTML = YEARS.map(([y, n]) =>
  `<div class="vbar-w" title="${y}: ${n} papers"><span class="n">${n}</span>
     <span class="vbar" style="height:${Math.max(2, 100 * n / maxY)}%"></span>
     <span class="yl">${String(y).slice(2)}</span></div>`).join('');

// venue bars
const maxV = Math.max(...VENUES.map(([, n]) => n));
el('venuebars').innerHTML = VENUES.map(([v, n]) =>
  `<div class="hbar-row" data-ven="${v}" title="${v}: ${n} papers">
     <span class="lbl">${v}</span>
     <span class="hbar-track"><span class="hbar" style="width:${100 * n / maxV}%"></span></span>
     <span class="num">${n}</span></div>`).join('');

// filter selects
el('fcat').innerHTML += CATS.map(c =>
  `<option value="${c.id}">${c.label} (${c.n})</option>`).join('');
el('fven').innerHTML += VENUES.map(([v, n]) =>
  `<option value="${v}">${v} (${n})</option>`).join('');
el('fyear').innerHTML += YEARS.map(([y]) => `<option>${y}</option>`).join('');

let sortKey = 'ci', sortDir = -1;

function esc(s) { return s.replace(/&/g, '&amp;').replace(/</g, '&lt;'); }

function render() {
  const q = el('q').value.toLowerCase();
  const fc = el('fcat').value, fv = el('fven').value,
        fy = el('fyear').value, fp = el('fplat').value, fa = el('fappr').value,
        fr = el('fart').value;
  let rows = DATA.filter(r =>
    (!q || (r.t + ' ' + r.k + ' ' + r.o).toLowerCase().includes(q)) &&
    (!fc || r.c.includes(fc)) && (!fv || r.v === fv) &&
    (!fy || String(r.y) === fy) && (!fp || r.pl.includes(fp)) &&
    (!fa || r.ap === fa) &&
    (!fr || (fr === 'has' ? r.as !== 'none' : r.as === fr)));
  rows.sort((a, b) => {
    let x, y;
    if (sortKey === 'vy') { x = a.v + a.y; y = b.v + b.y; }
    else if (sortKey === 'c') { x = a.c[0] || ''; y = b.c[0] || ''; }
    else { x = a[sortKey]; y = b[sortKey]; }
    return (x < y ? -1 : x > y ? 1 : 0) * sortDir;
  });
  el('cnt').textContent = `${rows.length} / ${DATA.length} papers`;
  el('tb').innerHTML = rows.map(r => `<tr>
    <td class="title"><a href="${r.d}" target="_blank" rel="noopener">${esc(r.t)}</a>
      <span class="ol">${esc(r.o)}</span></td>
    <td class="venue-year">${r.v} '${String(r.y).slice(2)}</td>
    <td>${r.c.map(c => `<span class="chip">${catLabel[c] || c}</span>`).join('')}</td>
    <td class="kern">${esc(r.k)}</td>
    <td>${r.pl.map(p => `<span class="chip">${p}</span>`).join('')}</td>
    <td>${r.ap}</td>
    <td class="num">${r.ci}</td>
    <td>${r.au ? `<a class="art ${r.as}" href="${r.au}" target="_blank" rel="noopener">${
        r.as === 'verified' ? '&#10004;' : '&#8776;'} code</a>${
        r.st ? ` <span class="stars">&#9733;${r.st}</span>` : ''}` : ''}</td></tr>`).join('');
  document.querySelectorAll('.hbar-row[data-cat]').forEach(rw =>
    rw.classList.toggle('active', rw.dataset.cat === fc));
}

['q', 'fcat', 'fven', 'fyear', 'fplat', 'fappr', 'fart'].forEach(id =>
  el(id).addEventListener('input', render));
document.querySelectorAll('.hbar-row[data-cat]').forEach(rw =>
  rw.addEventListener('click', () => {
    el('fcat').value = el('fcat').value === rw.dataset.cat ? '' : rw.dataset.cat;
    render();
  }));
document.querySelectorAll('.hbar-row[data-ven]').forEach(rw =>
  rw.addEventListener('click', () => {
    el('fven').value = el('fven').value === rw.dataset.ven ? '' : rw.dataset.ven;
    render();
  }));
document.querySelectorAll('thead th').forEach(th =>
  th.addEventListener('click', () => {
    const k = th.dataset.k;
    if (sortKey === k) sortDir *= -1; else { sortKey = k; sortDir = k === 'ci' ? -1 : 1; }
    document.querySelectorAll('thead th').forEach(t =>
      t.classList.toggle('sorted', t === th));
    render();
  }));
render();
</script>
</body>
</html>
"""
    html = html.replace("__DATA__", data_js).replace("__CATS__", cats_js)
    html = html.replace("__YEARS__", years_js).replace("__VENUES__", venues_js)
    path = os.path.join(OUT, "index.html")
    with open(path, "w") as f:
        f.write(html)
    print(f"Wrote {path} ({os.path.getsize(path)//1024} KB, {len(papers)} papers)")


if __name__ == "__main__":
    main()
