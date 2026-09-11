#!/usr/bin/env python3
"""
Aggregate benchspecs/*/spec.yaml into:
  output/specbook.html   — browsable spec book (one card per kernel track,
                            variants expandable, cross-cutting fairness rules)
  output/specbook.json   — machine-readable index of all variants
  output/specbook.md     — flat markdown summary table

Reports schema problems rather than hiding them.
"""

import json
import os
import glob
from collections import Counter

import yaml

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "output")

DOMAIN = {
    "sparse LA": ["spmv", "spmm", "sddmm", "spgemm", "sptrsv", "spmspv",
                  "sparse-format-conversion", "sparse-factorization",
                  "sparse-attention-kernel"],
    "dense LA": ["gemm", "batched-gemm", "quantized-gemm", "gemv", "trsm", "lu",
                 "cholesky", "qr", "eigensolver", "svd", "matrix-inversion",
                 "blas-level1-2", "math-functions", "matrix-function-evaluation",
                 "homotopy-continuation"],
    "stencil / PDE": ["stencil", "lattice-boltzmann", "fdtd-seismic",
                      "climate-kernel", "cellular-automata"],
    "spectral": ["fft", "ntt"],
    "tensor": ["mttkrp", "tensor-contraction", "sparse-tensor-contraction",
               "tucker", "tensor-network", "tensor-train-decomposition"],
    "graph": ["bfs", "sssp", "pagerank", "triangle-counting",
              "graph-pattern-mining", "connected-components",
              "community-detection", "graph-coloring", "betweenness",
              "kcore-kclique", "graph-partitioning", "dynamic-graph-kernel",
              "random-walk", "mst", "maximal-independent-set", "vertex-cover",
              "bipartite-matching", "lca-bridge-finding", "graph-layout",
              "signed-graph-balancing", "spectral-sparsification",
              "knn-graph-construction", "astar-search"],
    "primitives": ["sort", "scan-reduction", "hash-table", "set-intersection",
                   "topk-selection", "string-regex-matching",
                   "sequence-alignment", "json-parsing", "spatial-join",
                   "hausdorff-distance", "membership-filter",
                   "dp-dynamic-programming", "morphology-image-kernel"],
    "N-body / MD": ["md-force", "fmm", "nbody", "particle-mesh"],
    "mesh / AMR": ["unstructured-mesh-kernel", "amr-kernel", "particle-in-cell"],
    "solvers": ["cg-krylov", "multigrid", "preconditioner", "tridiagonal-solve",
                "iterative-refinement", "mixed-precision-solver"],
    "compression": ["lossy-compression", "lossless-compression",
                    "content-defined-chunking", "erasure-coding"],
    "ML kernels": ["convolution", "attention-kernel", "gnn-aggregation",
                   "embedding-ops", "moe-kernel", "layernorm-softmax-fused",
                   "transformer-inference-fused", "dnn-operator-fusion",
                   "winograd", "ann-search", "spiking-neural-network"],
    "codegen / tuning": ["autotuner-multi-kernel"],
}
SLUG_DOMAIN = {s: d for d, ss in DOMAIN.items() for s in ss}
DOMAIN_ORDER = list(DOMAIN) + ["other"]


def esc(s):
    return str(s).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def render_block(obj, depth=0):
    """Render nested dict/list as compact definition lists."""
    pad = "margin-left:%dpx" % (depth * 12)
    if isinstance(obj, dict):
        rows = []
        for k, v in obj.items():
            if isinstance(v, (dict, list)):
                rows.append(f'<div style="{pad}"><span class="k">{esc(k)}</span>'
                            f'{render_block(v, depth + 1)}</div>')
            else:
                rows.append(f'<div style="{pad}"><span class="k">{esc(k)}</span>'
                            f'<span class="v">{esc(v)}</span></div>')
        return "".join(rows)
    if isinstance(obj, list):
        if all(not isinstance(x, (dict, list)) for x in obj):
            return ('<span class="v list">' +
                    ", ".join(esc(x) for x in obj) + "</span>")
        return "".join(f'<div style="{pad}">{render_block(x, depth + 1)}</div>'
                       for x in obj)
    return f'<span class="v">{esc(obj)}</span>'


def main():
    specs, problems = {}, []
    for sf in sorted(glob.glob(os.path.join(HERE, "benchspecs", "*", "spec.yaml"))):
        slug = os.path.basename(os.path.dirname(sf))
        try:
            with open(sf) as f:
                d = yaml.safe_load(f)
        except Exception as e:
            problems.append(f"{slug}: YAML parse error: {e}")
            continue
        if not isinstance(d, dict) or "variants" not in d:
            problems.append(f"{slug}: missing 'variants'")
            continue
        d["_slug"] = slug
        d["_has_survey"] = os.path.exists(
            os.path.join(os.path.dirname(sf), "survey.md"))
        specs[slug] = d

    groups = {}
    if os.path.exists(os.path.join(OUT, "benchmark_groups.json")):
        with open(os.path.join(OUT, "benchmark_groups.json")) as f:
            raw = json.load(f)
        alias = {"other:ann-search": "ann-search", "other:mst": "mst",
                 "other:knn-search": "ann-search",
                 "other:tensor-train-decomposition": "tensor-train-decomposition"}
        for k, v in raw.items():
            groups[alias.get(k, k.replace("other:", ""))] = len(v)

    n_var = sum(len(d["variants"]) for d in specs.values())
    by_domain = {}
    for slug, d in specs.items():
        by_domain.setdefault(SLUG_DOMAIN.get(slug, "other"), []).append((slug, d))
    for dm in by_domain:
        by_domain[dm].sort(key=lambda kv: -groups.get(kv[0], 0))

    # ---------- json ----------
    index = {slug: {
        "summary": d.get("summary", ""),
        "operation": d.get("operation", ""),
        "n_papers": groups.get(slug, 0),
        "variants": [{"id": v.get("id"), "claim": v.get("claim")}
                     for v in d["variants"]],
    } for slug, d in specs.items()}
    with open(os.path.join(OUT, "specbook.json"), "w") as f:
        json.dump(index, f, ensure_ascii=False, indent=1)

    # ---------- markdown ----------
    with open(os.path.join(OUT, "specbook.md"), "w") as f:
        f.write("# HPC KernelBench — benchmark specifications\n\n")
        f.write(f"{len(specs)} kernel tracks, {n_var} benchmark variants, "
                "derived from how each track's own papers evaluate.\n\n")
        f.write("| Kernel | Papers | Variants | What is measured |\n|---|---|---|---|\n")
        for dm in DOMAIN_ORDER:
            for slug, d in by_domain.get(dm, []):
                ids = "<br>".join(f"`{v.get('id')}`" for v in d["variants"])
                f.write(f"| **{slug}** | {groups.get(slug, 0)} | {ids} | "
                        f"{str(d.get('summary','')).strip().splitlines()[0][:110]} |\n")

    # ---------- html ----------
    sections = []
    for dm in DOMAIN_ORDER:
        if dm not in by_domain:
            continue
        cards = []
        for slug, d in by_domain[dm]:
            vs = []
            for v in d["variants"]:
                vid = esc(v.get("id", "?"))
                claim = esc(v.get("claim", ""))
                body = render_block({k: val for k, val in v.items()
                                     if k not in ("id", "claim")})
                vs.append(f'<details class="variant"><summary>'
                          f'<code>{vid}</code><span class="claim">{claim}</span>'
                          f'</summary><div class="vbody">{body}</div></details>')
            notes = d.get("notes_on_fairness") or []
            openq = d.get("open_questions") or []
            extra = ""
            if notes:
                extra += ('<div class="notes"><b>Fairness notes</b><ul>' +
                          "".join(f"<li>{esc(n)}</li>" for n in notes) + "</ul></div>")
            if openq:
                extra += ('<div class="notes open"><b>Open questions</b><ul>' +
                          "".join(f"<li>{esc(n)}</li>" for n in openq) + "</ul></div>")
            op = esc(d.get("operation", ""))
            cards.append(
                f'<details class="track" id="{esc(slug)}"><summary>'
                f'<b>{esc(slug)}</b>'
                f'<span class="np">{groups.get(slug, 0)} papers</span>'
                f'<span class="nv">{len(d["variants"])} variants</span></summary>'
                f'<p class="sum">{esc(str(d.get("summary","")).strip())}</p>'
                + (f'<p class="op"><code>{op}</code></p>' if op else "")
                + "".join(vs) + extra + "</details>")
        sections.append(f'<section><h2>{esc(dm)}</h2>{"".join(cards)}</section>')

    prob_html = ""
    if problems:
        prob_html = ('<div class="notes open"><b>Schema problems</b><ul>' +
                     "".join(f"<li>{esc(p)}</li>" for p in problems) + "</ul></div>")

    html = f"""<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>HPC KernelBench — benchmark specifications</title>
<style>
:root {{ --surface:#fcfcfb; --ink:#1d1d1c; --ink-2:#55554f; --ink-3:#8a8a82;
  --card:#fff; --line:#e6e6e1; --accent:#256abf; --chip:#eef2f7; --hover:#f3f6fa; }}
@media (prefers-color-scheme: dark) {{ :root {{ --surface:#1a1a19; --ink:#ececea;
  --ink-2:#b0b0a8; --ink-3:#7d7d75; --card:#232322; --line:#3a3a37;
  --accent:#86b6ef; --chip:#2d3743; --hover:#2a2a28; }} }}
*{{box-sizing:border-box;margin:0}}
body{{background:var(--surface);color:var(--ink);
  font:15px/1.55 -apple-system,"Segoe UI",Roboto,"Noto Sans",sans-serif;
  padding:28px 4vw 80px;max-width:1080px;margin:0 auto}}
h1{{font-size:24px;letter-spacing:-.02em}}
.sub{{color:var(--ink-2);margin:6px 0 22px;max-width:78ch}}
h2{{font-size:15px;margin:22px 0 8px;border-bottom:1px solid var(--line);
  padding-bottom:5px;text-transform:capitalize}}
.tiles{{display:flex;flex-wrap:wrap;gap:12px;margin-bottom:20px}}
.tile{{background:var(--card);border:1px solid var(--line);border-radius:10px;
  padding:12px 18px}} .tile b{{display:block;font-size:23px}}
.tile span{{color:var(--ink-2);font-size:13px}}
.track{{background:var(--card);border:1px solid var(--line);border-radius:9px;
  margin:6px 0}}
.track>summary{{cursor:pointer;padding:9px 14px;display:flex;gap:10px;
  align-items:center;user-select:none}}
summary::-webkit-details-marker{{display:none}}
.np,.nv{{color:var(--ink-3);font-size:12.5px}} .np{{margin-left:auto}}
.nv{{color:var(--accent)}}
.sum{{padding:2px 16px 8px;color:var(--ink-2);font-size:13.5px;max-width:85ch}}
.op{{padding:0 16px 8px}} .op code{{background:var(--chip);border-radius:5px;
  padding:2px 7px;font-size:12.5px}}
.variant{{margin:4px 12px;border:1px solid var(--line);border-radius:7px}}
.variant>summary{{cursor:pointer;padding:7px 11px;display:flex;gap:10px;
  align-items:baseline;flex-wrap:wrap}}
.variant code{{color:var(--accent);font-size:13px;font-weight:600}}
.claim{{color:var(--ink-2);font-size:12.5px}}
.vbody{{padding:4px 12px 11px;font-size:13px;border-top:1px solid var(--line)}}
.vbody .k{{color:var(--ink-3);display:inline-block;min-width:130px;
  vertical-align:top}}
.vbody .v{{color:var(--ink)}} .vbody .v.list{{color:var(--ink-2)}}
.notes{{margin:8px 12px 12px;padding:9px 12px;background:var(--hover);
  border-radius:7px;font-size:13px}}
.notes b{{font-size:12.5px;color:var(--ink-2)}}
.notes ul{{margin:5px 0 0 16px;color:var(--ink-2)}}
.notes.open{{border-left:3px solid var(--accent)}}
.footer{{margin-top:26px;color:var(--ink-3);font-size:12.5px;max-width:85ch}}
</style></head><body>
<h1>HPC KernelBench — benchmark specifications</h1>
<p class="sub">One specification per kernel track, derived from how that track's
own papers actually evaluate (arXiv fulltext + artifact benchmark scripts).
Each variant fixes the input suite, the timing protocol (warmup, repetitions,
statistic, timer), what the timing window includes (preprocessing / format
conversion / tuning search), the correctness gate, and the reported metric.
Where the literature's common practice is unfair or unreproducible, the spec
diverges and says so under <i>Fairness notes</i>.</p>
<div class="tiles">
  <div class="tile"><b>{len(specs)}</b><span>kernel tracks</span></div>
  <div class="tile"><b>{n_var}</b><span>benchmark variants</span></div>
  <div class="tile"><b>310</b><span>papers with artifacts surveyed</span></div>
</div>
{prob_html}
{"".join(sections)}
<p class="footer">Built by Claude Code. Specs are grounded in per-paper surveys
(<code>benchspecs/&lt;kernel&gt;/survey.md</code>); each spec's
<code>evidence</code> field maps claims back to individual papers. Numbers here
describe intended measurement methodology — nothing has been executed yet.</p>
</body></html>"""
    path = os.path.join(OUT, "specbook.html")
    with open(path, "w") as f:
        f.write(html)

    print(f"{len(specs)} tracks, {n_var} variants -> specbook.html/.json/.md")
    if problems:
        print("PROBLEMS:")
        for p in problems:
            print("  " + p)
    missing_survey = [s for s, d in specs.items() if not d["_has_survey"]]
    if missing_survey:
        print("missing survey.md:", missing_survey)


if __name__ == "__main__":
    main()
