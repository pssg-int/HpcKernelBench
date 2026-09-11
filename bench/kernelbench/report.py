"""
Turn result JSON documents into a readable comparison.

    python -m kernelbench.report results/*.json
    python -m kernelbench.report --html results/report.html results/*.json
"""

from __future__ import annotations

import argparse
import glob
import json
import os


def load(paths: list[str]) -> list[dict]:
    docs = []
    for p in paths:
        for f in sorted(glob.glob(p)):
            with open(f) as fh:
                d = json.load(fh)
            d["_file"] = os.path.basename(f)
            docs.append(d)
    return docs


def text_table(docs: list[dict]) -> str:
    lines = []
    for d in docs:
        head = f"{d['kernel']} / {d['variant']}"
        tag = "CONFORMING" if d.get("conforming") else "non-conforming"
        lines.append(f"\n=== {head}   [{tag}]")
        if not d.get("conforming"):
            for r in d.get("nonconformance_reasons", []):
                lines.append(f"      ! {r}")
        envd = d.get("environment", {})
        lines.append(f"    host {envd.get('host','?')} | "
                     f"{envd.get('cpu',{}).get('model','?')[:44]} | "
                     f"gpu {envd.get('gpu',{}).get('name','none')}")
        unit = next((r["metrics"].get("throughput_unit", "GFLOP/s")
                     for r in d["runs"] if r["valid"]), "GFLOP/s")
        lines.append(f"    {'impl':<26} {'workload':<18} {'dim':>5} "
                     f"{'median ms':>10} {unit:>10} {'stdev%':>7} "
                     f"{'prep ms':>9}  gate")
        for r in d["runs"]:
            dim = r["params"].get("N") or r["params"].get("K") or ""
            s, m, c = r["stats_ms"], r["metrics"], r["correctness"]
            # `tolerance` is legitimately None for CORRECTNESS_MODE="exact"
            # (bit-exact gates, e.g. lossless-compression's roundtrip check)
            tol_txt = f"{c['tolerance']:.0e}" if c["tolerance"] is not None else "exact"
            if not r["valid"]:
                lines.append(f"    {r['implementation']:<26} "
                             f"{r['matrix']['name']:<18} {str(dim):>5} "
                             f"{'—':>10} {'—':>10} {'—':>7} "
                             f"{r['preprocessing_ms']:>9.2f}  FAIL "
                             f"({c['metric']}={c['value']:.2e} > {tol_txt})")
                continue
            stdev_pct = 100 * s["stdev"] / s["median"] if s["median"] else 0
            lines.append(f"    {r['implementation']:<26} "
                         f"{r['matrix']['name']:<18} {str(dim):>5} "
                         f"{s['median']:>10.3f} "
                         f"{m.get('throughput', m.get('gflops', 0.0)):>10.2f} "
                         f"{stdev_pct:>6.1f}% {r['preprocessing_ms']:>9.2f}  "
                         f"pass ({c['value']:.1e}<={tol_txt})")
    return "\n".join(lines)


def leaderboard(docs: list[dict]) -> str:
    """Per (kernel, variant, matrix, dim): rank implementations by GFLOP/s."""
    from collections import defaultdict
    groups = defaultdict(list)
    units: dict[str, str] = {}          # kernel -> its throughput_unit, e.g. "GB/s"
    for d in docs:
        for r in d["runs"]:
            if not r["valid"]:
                continue
            dim = r["params"].get("N") or r["params"].get("K") or 1
            # `direction` (e.g. compress vs decompress) is a distinct
            # operation, not a variant of the same one -- never rank them
            # against each other (see compression.py's module docstring).
            direction = r["params"].get("direction")
            key = (r["kernel"], r["variant"], r["matrix"]["name"], dim, direction)
            groups[key].append((r["metrics"]["gflops"], r["implementation"],
                                r["stats_ms"]["median"]))
            units.setdefault(r["kernel"], r["metrics"].get("throughput_unit", "GFLOP/s"))
    out = ["\n=== leaderboard (higher is better)"]
    for key in sorted(groups, key=lambda k: tuple(str(x) for x in k)):
        rows = sorted(groups[key], reverse=True)
        if len(rows) < 2:
            continue
        k, v, mat, dim, direction = key
        tag = f"  [{direction}]" if direction else ""
        out.append(f"\n  {k}/{v}  {mat}  dim={dim}{tag}")
        best = rows[0][0]
        unit = units.get(k, "GFLOP/s")
        for gf, impl, ms in rows:
            rel = gf / best if best else float("nan")
            out.append(f"    {gf:8.2f} {unit:<9} {ms:9.3f} ms  "
                       f"{rel:5.2f}x  {impl}")
    return "\n".join(out) if len(out) > 1 else ""


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("paths", nargs="+")
    ap.add_argument("--html", help="also write an HTML report here")
    args = ap.parse_args()
    docs = load(args.paths)
    if not docs:
        print("no result documents matched")
        return 1
    txt = text_table(docs) + "\n" + leaderboard(docs)
    print(txt)
    if args.html:
        body = txt.replace("&", "&amp;").replace("<", "&lt;")
        with open(args.html, "w") as f:
            f.write(f"""<!DOCTYPE html><html><head><meta charset="utf-8">
<title>KernelBench results</title><style>
body{{background:#fcfcfb;color:#1d1d1c;font:13px/1.5 ui-monospace,Menlo,monospace;padding:26px}}
@media (prefers-color-scheme:dark){{body{{background:#1a1a19;color:#ececea}}}}
pre{{white-space:pre-wrap}}h1{{font:600 18px system-ui}}
</style></head><body><h1>HPC KernelBench — results</h1><pre>{body}</pre></body></html>""")
        print(f"\nwrote {args.html}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
