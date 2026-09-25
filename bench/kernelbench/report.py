"""
Turn result JSON documents into a readable comparison.

A domain may declare BASELINE_IMPLS = {kernel: [impl, ...]} to get a
"speedup vs baseline" column in the leaderboard, and PAPER_BASELINES =
{kernel: {impl: {"paper", "baselines": [(name, impl or None, reason)]}}} to
get each paper compared against the baselines its own evaluation used.

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
        errors = d.get("errors", [])
        if errors:
            lines.append(f"    crashed ({len(errors)}, recorded by --keep-going)")
            for e in errors:
                msg = str(e.get("error", "")).strip().splitlines()
                lines.append(f"      {e.get('impl', '?'):<26} "
                             f"{str(e.get('workload', '?')):<18} "
                             f"{(msg[0] if msg else '')[:110]}")
        if not d["runs"] and not unsupported and not errors:
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
    whenever a workload records them. When all three are recorded they ARE
    the computation, and the name is dropped from the key, so `star2d1r` and
    `star2d1r@16384x16384:T=1000` land in the same group.
    """
    grid = matrix.get("grid_shape")
    shape, steps = matrix.get("shape"), matrix.get("timesteps_per_call")
    if grid and shape and steps is not None:
        return (None, shape, tuple(grid), steps)
    return (matrix.get("name", "?"), shape,
            tuple(grid) if grid else None, steps)


def _workload_label(sig: tuple) -> str:
    name, shape, grid, steps = sig
    parts = [name or shape]
    if name and shape and shape not in name:
        parts.append(f"(ran {shape})")
    if grid:
        parts.append("x".join(str(g) for g in grid))
    if steps is not None:
        parts.append(f"T={steps}")
    return "  ".join(parts)


def _domain_attr(kernel: str, attr: str) -> dict:
    """A kernel's entry in one of its domain module's tables, or {}."""
    try:
        from . import domains
        mod = domains.load(kernel)
    except Exception:          # report must work without the domain's deps
        return {}
    return getattr(mod, attr, {}).get(kernel, {})


def _baselines(kernel: str) -> list[str]:
    """Reference implementations a kernel's domain names for speedup columns."""
    return list(_domain_attr(kernel, "BASELINE_IMPLS"))


def paper_baselines(docs: list[dict]) -> str:
    """
    Each paper against the baselines ITS OWN evaluation used (the domain's
    PAPER_BASELINES), measured on the same device, precision and workload.
    Baselines that are not integrated are listed with the reason, so a
    missing comparison is visible rather than silently absent. A paper may
    also declare `derived_vs`: the comparison its own figures make across
    precisions (SPIDER: fp16 result / 4 x 7/r vs fp64 baselines); those
    ratios use the derived values from the paper's metric block and are
    labeled derived, never ranked.
    """
    from collections import defaultdict
    runs = [(d, r) for d in docs for r in d["runs"] if r["valid"]]
    if not runs:
        return ""
    by_key = defaultdict(dict)   # (kernel, sig, device, precision) -> {impl: run}
    for d, r in runs:
        key = (r["kernel"], _workload_sig(r["matrix"]), _device(d, r), r.get("precision"))
        prev = by_key[key].get(r["implementation"])
        if prev is None or r["metrics"]["gflops"] > prev["metrics"]["gflops"]:
            by_key[key][r["implementation"]] = r
    out = ["\n=== each paper vs the baselines its own evaluation used "
           "(same device, precision and workload)"]
    for kernel in sorted({r["kernel"] for _, r in runs}):
        table = _domain_attr(kernel, "PAPER_BASELINES")
        unit = next(r["metrics"].get("throughput_unit", "GFLOP/s")
                    for _, r in runs if r["kernel"] == kernel)
        for impl, entry in table.items():
            mine = sorted(((k, g[impl]) for k, g in by_key.items()
                           if k[0] == kernel and impl in g),
                          key=lambda kv: tuple(str(x) for x in kv[0]))
            if not mine:
                continue
            out.append(f"\n  {entry['paper']}  [{impl}]")
            missing = [(n, why) for n, i, why in entry["baselines"] if i is None]
            for n, why in missing:
                out.append(f"    not integrated: {n} -- {why}")
            integrated = [(n, i) for n, i, _ in entry["baselines"] if i is not None]
            for key, r in mine:
                _, sig, device, prec = key
                tp = r["metrics"]["gflops"]
                out.append(f"    {_workload_label(sig)}  on {device}, {prec}: "
                           f"{tp:.2f} {unit}")
                group = by_key[key]
                for n, bimpl in integrated:
                    b = group.get(bimpl)
                    if b is None:
                        if entry.get("derived_vs"):
                            continue   # compared through the derived block below
                        out.append(f"      vs {n:<30} (no {bimpl} run on this workload)")
                    else:
                        bt = b["metrics"]["gflops"]
                        sp = f"{tp / bt:6.2f}x" if bt else "     —"
                        out.append(f"      vs {n:<30} {sp}   ({bt:.2f} {unit}, {bimpl})")
                out.extend(_derived_vs(entry, integrated, r, device, by_key, kernel))
    return "\n".join(out) if len(out) > 1 else ""


def _derived_vs(entry, integrated, run, device, by_key, kernel) -> list[str]:
    """A paper's own cross-precision comparison from its derived metrics."""
    import re
    dv = entry.get("derived_vs")
    native = run["metrics"].get("paper_native") or {}
    if not dv or not native:
        return []
    lines = []
    for m in native.get("metrics", []):
        mo = re.search(r"-\dD(\d+)R-equivalent", m["name"])
        if m.get("class") != "derived" or not mo:
            continue
        r = int(mo.group(1))
        shape = dv["shape"].format(r=r)
        sig = (None, shape, tuple(dv["grid"]), dv["timesteps"])
        group = by_key.get((kernel, sig, device, dv["precision"]), {})
        label = f"{shape} {'x'.join(map(str, dv['grid']))} T={dv['timesteps']}"
        lines.append(f"      derived (paper normalization, not a measurement): "
                     f"{m['name']} = {m['value']:.2f} vs {dv['precision']} runs of {label}")
        for n, bimpl in integrated:
            b = group.get(bimpl)
            if b is None:
                lines.append(f"        vs {n:<28} (no {bimpl} run)")
            else:
                bt = b["metrics"]["gflops"]
                sp = f"{m['value'] / bt:6.2f}x" if bt else "     —"
                lines.append(f"        vs {n:<28} {sp}   ({bt:.2f}, {bimpl})")
    return lines


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
    txt = text_table(docs) + "\n" + leaderboard(docs) + "\n" + paper_baselines(docs)
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
