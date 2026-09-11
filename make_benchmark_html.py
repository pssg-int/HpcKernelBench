#!/usr/bin/env python3
"""
Render output/benchmark_candidates.html from output/benchmark_groups.json:
kernel tracks grouped by domain, each listing its papers with artifact links.
Same styling family as index.html; single sequential hue for magnitude bars.
"""

import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "output")

DOMAIN_ORDER = ["sparse LA", "dense LA", "stencil / PDE", "spectral", "tensor",
                "graph", "primitives", "N-body / MD", "mesh / AMR", "solvers",
                "compression", "ML kernels", "codegen / misc", "other"]


def main():
    with open(os.path.join(OUT, "benchmark_groups.json")) as f:
        groups = json.load(f)
    from benchmark_groups import SLUG_DOMAIN

    by_domain = {}
    for slug, ps in groups.items():
        d = SLUG_DOMAIN.get(slug, "other")
        by_domain.setdefault(d, []).append((slug, ps))
    for d in by_domain:
        by_domain[d].sort(key=lambda kv: -len(kv[1]))

    total = len({p["key"] for ps in groups.values() for p in ps})
    strong = sum(1 for ps in groups.values() if len(ps) >= 3)

    def esc(s):
        return s.replace("&", "&amp;").replace("<", "&lt;")

    sections = []
    for d in DOMAIN_ORDER:
        if d not in by_domain:
            continue
        cards = []
        for slug, ps in by_domain[d]:
            n = len(ps)
            cls = "strong" if n >= 3 else ("pair" if n == 2 else "single")
            rows = "".join(
                f'<li><a href="https://doi.org/{esc(p["doi"])}" target="_blank" '
                f'rel="noopener" class="pt">{esc(p["title"])}</a>'
                f'<span class="pm">{p["venue"]} \'{str(p["year"])[2:]}'
                f'{" · " + ", ".join(p["platform"][:2]) if p["platform"] else ""}'
                f'{" · " + p["approach"] if p["approach"] else ""}</span>'
                f'<a class="repo {p["status"]}" href="{esc(p["url"])}" '
                f'target="_blank" rel="noopener">'
                f'{"&#10004;" if p["status"] == "verified" else "&#8776;"} code</a>'
                f'<span class="ol">{esc(p["one_liner"])}</span></li>'
                for p in ps)
            cards.append(
                f'<details class="track {cls}"{" open" if n >= 6 else ""}>'
                f'<summary><b>{esc(slug)}</b>'
                f'<span class="cnt">{n} paper{"s" if n > 1 else ""}</span>'
                f'</summary><ul>{rows}</ul></details>')
        cards_html = "".join(cards)
        sections.append(f'<section><h2>{esc(d)}</h2>{cards_html}</section>')
    sections_html = "".join(sections)

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>HPC KernelBench — benchmark candidate tracks</title>
<style>
:root {{
  --surface:#fcfcfb; --ink:#1d1d1c; --ink-2:#55554f; --ink-3:#8a8a82;
  --card:#ffffff; --line:#e6e6e1; --accent:#256abf; --chip:#eef2f7;
  --hover:#f3f6fa;
}}
@media (prefers-color-scheme: dark) {{
  :root {{
    --surface:#1a1a19; --ink:#ececea; --ink-2:#b0b0a8; --ink-3:#7d7d75;
    --card:#232322; --line:#3a3a37; --accent:#86b6ef; --chip:#2d3743;
    --hover:#2a2a28;
  }}
}}
* {{ box-sizing:border-box; margin:0; }}
body {{ background:var(--surface); color:var(--ink);
  font:15px/1.5 -apple-system,"Segoe UI",Roboto,"Noto Sans",sans-serif;
  padding:28px 4vw 80px; max-width:1100px; margin:0 auto; }}
h1 {{ font-size:24px; letter-spacing:-0.02em; }}
.sub {{ color:var(--ink-2); margin:6px 0 24px; max-width:75ch; }}
section {{ margin-bottom:26px; }}
h2 {{ font-size:16px; margin:18px 0 8px; color:var(--ink);
  border-bottom:1px solid var(--line); padding-bottom:5px;
  text-transform:capitalize; }}
.track {{ background:var(--card); border:1px solid var(--line);
  border-radius:9px; margin:6px 0; }}
.track summary {{ cursor:pointer; padding:9px 14px; display:flex;
  align-items:center; gap:10px; user-select:none; }}
.track summary::-webkit-details-marker {{ display:none; }}
.track summary b {{ font-size:14.5px; }}
.track.single summary b {{ color:var(--ink-2); font-weight:500; }}
.cnt {{ margin-left:auto; color:var(--ink-3); font-size:13px;
  font-variant-numeric:tabular-nums; }}
.track.strong summary .cnt {{ color:var(--accent); font-weight:600; }}
.track ul {{ list-style:none; padding:2px 14px 12px; }}
.track li {{ padding:7px 0; border-top:1px solid var(--line);
  display:grid; grid-template-columns:1fr auto auto; gap:2px 12px;
  align-items:baseline; }}
.pt {{ color:var(--ink); text-decoration:none; font-weight:600;
  font-size:14px; }}
.pt:hover {{ color:var(--accent); text-decoration:underline; }}
.pm {{ color:var(--ink-3); font-size:12.5px; white-space:nowrap; }}
a.repo {{ white-space:nowrap; text-decoration:none; font-size:12.5px;
  border:1px solid var(--line); border-radius:6px; padding:0 7px; }}
a.repo.verified {{ color:var(--accent); border-color:var(--accent); }}
a.repo.likely {{ color:var(--ink-2); }}
.ol {{ grid-column:1 / -1; color:var(--ink-2); font-size:13px;
  max-width:90ch; }}
.tiles {{ display:flex; flex-wrap:wrap; gap:12px; margin-bottom:22px; }}
.tile {{ background:var(--card); border:1px solid var(--line);
  border-radius:10px; padding:12px 18px; }}
.tile b {{ display:block; font-size:24px; }}
.tile span {{ color:var(--ink-2); font-size:13px; }}
.footer {{ margin-top:24px; color:var(--ink-3); font-size:12.5px; }}
</style>
</head>
<body>
<h1>HPC KernelBench — benchmark candidate tracks</h1>
<p class="sub">The {total} kernel-optimization papers (2020+) with a confirmed
open-source artifact, grouped by the concrete kernel they optimize. A track
with several competing implementations of the same kernel is a natural
benchmark unit. &#10004; = artifact verified, &#8776; = likely.
Companion to <code>index.html</code> (full 917-paper table).</p>
<div class="tiles">
  <div class="tile"><b>{total}</b><span>papers with artifacts</span></div>
  <div class="tile"><b>{len(groups)}</b><span>kernel groups</span></div>
  <div class="tile"><b>{strong}</b><span>tracks with &ge;3 papers</span></div>
</div>
{sections_html}
<p class="footer">Built by Claude Code. Kernel-name normalization is
LLM-assisted; a paper appears in every group it contributes to (1&ndash;3).
Groups with &ge;6 papers are expanded by default.</p>
</body>
</html>
"""
    path = os.path.join(OUT, "benchmark_candidates.html")
    with open(path, "w") as f:
        f.write(html)
    print(f"Wrote {path} ({os.path.getsize(path)//1024} KB)")


if __name__ == "__main__":
    main()
