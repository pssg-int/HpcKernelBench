"""
Turn result JSON documents into a readable comparison.

A domain may declare BASELINE_IMPLS = {kernel: [impl, ...]} to get a
"speedup vs baseline" column in the leaderboard.

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


def _row_name(matrix: dict) -> str:
    """Workload name, plus the shape actually run when an adapter substituted one."""
    name, shape = matrix.get("name", "?"), matrix.get("shape")
    return f"{name} (ran {shape})" if shape and shape not in name else name


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
                             f"{_row_name(r['matrix']):<18} {str(dim):>5} "
                             f"{'—':>10} {'—':>10} {'—':>7} "
                             f"{r['preprocessing_ms']:>9.2f}  FAIL "
                             f"({c['metric']}={c['value']:.2e} > {tol_txt})")
                continue
            stdev_pct = 100 * s["stdev"] / s["median"] if s["median"] else 0
            lines.append(f"    {r['implementation']:<26} "
                         f"{_row_name(r['matrix']):<18} {str(dim):>5} "
                         f"{s['median']:>10.3f} "
                         f"{m.get('throughput', m.get('gflops', 0.0)):>10.2f} "
                         f"{stdev_pct:>6.1f}% {r['preprocessing_ms']:>9.2f}  "
                         f"pass ({c['value']:.1e}<={tol_txt})")
            lines.extend(_native_lines(m.get("paper_native")))
        unsupported = d.get("unsupported", [])
        if unsupported:
            lines.append(f"    unsupported ({len(unsupported)}): "
                         "implementation declined the workload, not a failure")
            for u in unsupported:
                reason = str(u.get("reason", "")).strip().splitlines()
                lines.append(f"      {u.get('impl', '?'):<26} "
                             f"{str(u.get('workload', '?')):<18} "
                             f"{(reason[0] if reason else '')[:110]}")
        if not d["runs"] and not unsupported:
            lines.append("    (no runs recorded)")
    return "\n".join(lines)


def _native_lines(native: dict | None) -> list[str]:
    """The implementation's own paper's metric(s), indented under its row."""
    if not native:
        return []
    p = native["primary"]
    out = [f"      paper metric [{native['paper']}]: "
           f"{p['name']} = {p['value']:.4g}"]
    for m in native.get("metrics", []):
        tag = " (derived)" if m.get("class") == "derived" else ""
        out.append(f"        {m['name']} = {m['value']:.4g} {m['unit']}{tag}")
    skipped = [n["name"] for n in native.get("not_collected", [])]
    if skipped:
        out.append(f"        not collected: {'; '.join(skipped)}")
    return out


def _device(doc: dict, run: dict) -> str:
    """The processor a run executed on: GPU name for device runs, CPU model else."""
    envd = doc.get("environment", {})
    if run.get("platform") == "cpu":
        return envd.get("cpu", {}).get("model", "?") or "?"
    return envd.get("gpu", {}).get("name", "?") or "?"


def _workload_sig(matrix: dict) -> tuple:
    """
    What was actually computed, not just the requested name. An adapter may
    run something other than the name it was handed (the stencil SPIDER
    adapter always runs its own radius-7 box for one step), and the same name
    can be loaded at different sizes, so shape/grid/steps are part of the key
    whenever a workload records them.
    """
    grid = matrix.get("grid_shape")
    return (matrix.get("name", "?"), matrix.get("shape"),
            tuple(grid) if grid else None, matrix.get("timesteps_per_call"))


def _workload_label(sig: tuple) -> str:
    name, shape, grid, steps = sig
    parts = [name]
    if shape and shape not in name:
        parts.append(f"(ran {shape})")
    if grid:
        parts.append("x".join(str(g) for g in grid))
    if steps is not None:
        parts.append(f"T={steps}")
    return "  ".join(parts)


def _baselines(kernel: str) -> list[str]:
    """Reference implementations a kernel's domain names for speedup columns."""
    try:
        from . import domains
        mod = domains.load(kernel)
    except Exception:          # report must work without the domain's deps
        return []
    return list(getattr(mod, "BASELINE_IMPLS", {}).get(kernel, []))


def leaderboard(docs: list[dict]) -> str:
    """
    Rank implementations of the same computation, on the same device, at the
    same precision. Results from different GPUs/CPUs or precisions are never
    ranked against each other. Single-result groups are shown too (marked),
    so real runs are not hidden just because nothing else ran them.
    """
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
            key = (r["kernel"], r["variant"], _workload_sig(r["matrix"]), dim,
                   direction, _device(d, r), r.get("precision", "?"))
            groups[key].append((r["metrics"]["gflops"], r["implementation"],
                                r["stats_ms"]["median"]))
            units.setdefault(r["kernel"], r["metrics"].get("throughput_unit", "GFLOP/s"))
    out = ["\n=== leaderboard (higher is better; ranked only within one device "
           "and precision)"]
    base_cache: dict[str, list[str]] = {}
    for key in sorted(groups, key=lambda k: tuple(str(x) for x in k)):
        rows = sorted(groups[key], reverse=True)
        k, v, sig, dim, direction, device, prec = key
        tag = f"  [{direction}]" if direction else ""
        solo = "  (only result)" if len(rows) == 1 else ""
        out.append(f"\n  {k}/{v}  {_workload_label(sig)}  dim={dim}{tag}")
        out.append(f"    on {device}, {prec}{solo}")
        best = rows[0][0]
        unit = units.get(k, "GFLOP/s")
        if k not in base_cache:
            base_cache[k] = _baselines(k)
        base = next(((gf, impl) for pref in base_cache[k]
                     for gf, impl, _ in rows if impl == pref), None)
        for gf, impl, ms in rows:
            rel = gf / best if best else float("nan")
            if base is None:
                sp = "      —"
            else:
                sp = f"{gf / base[0]:6.2f}x" if base[0] else "      —"
            out.append(f"    {gf:8.2f} {unit:<9} {ms:9.3f} ms  "
                       f"{rel:5.2f}x of best  {sp} vs baseline  {impl}")
        if base is not None:
            out.append(f"    baseline: {base[1]}")
        elif base_cache[k]:
            out.append(f"    baseline: none of {', '.join(base_cache[k])} ran "
                       "in this group")
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
