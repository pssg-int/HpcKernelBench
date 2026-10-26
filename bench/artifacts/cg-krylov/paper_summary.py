"""
Paper-faithful cg-krylov micro-benchmark tables (bench/CG_PAPER_MICROBENCH.md).

Reads the runner's result JSONs and prints, per paper, the comparison that
paper's own claim is about, computed from each code's OWN internal timer
(never the subprocess wall clock, which includes process start-up):

  PERKS          ms/step: perks-cg-cached vs perks-cg-baseline (PERKS's own
                 non-persistent version) vs cusparse-cg; 10,000 steps
  Mille-feuille  ms/step: millefeuille-cg-mixed (solve kernel only) vs
                 cusparse-cg vs millefeuille-cg-reduce; 100 steps; relres
  SPCG           sparsified (sp, per ratio) vs SPCG's own non-sparsified
                 driver (nonsp): PCG-loop and overall time, iterations
  BootCMatchGX   setup / solve / total vs AmgX, iterations

Files are found by name under the results directory:
  paper_perks_<m>.json  paper_mf_<m>.json  paper_spcg_<m>_<tag>.json
  paper_bcmg_<m>.json

  python paper_summary.py <bench/results dir>   # prints Markdown
"""
from __future__ import annotations

import glob
import json
import math
import os
import sys
from collections import defaultdict


def _runs(pattern):
    """{matrix: {impl: params-with-extras}} over all files matching pattern."""
    out = defaultdict(dict)
    for f in sorted(glob.glob(pattern)):
        d = json.load(open(f))
        for r in d.get("runs", []):
            p = dict(r.get("params", {}))
            p["_valid"] = r.get("valid")
            p["_nnz"] = (r.get("matrix") or {}).get("nnz")
            p["_rows"] = (r.get("matrix") or {}).get("rows")
            name = (r.get("matrix") or {}).get("name")
            key = r["implementation"]
            if key == "spcg-sp-ilu0-cg":
                key = f"spcg-sp@{p.get('spcg_ratio')}"
            out[name][key] = p
        for e in d.get("errors", []):
            m = e.get("workload")
            out[m][e["impl"]] = {"_error": str(e.get("error", ""))[:80]}
        for u in d.get("unsupported", []):
            m = u.get("workload")
            out[m][u["impl"]] = {"_error": "UNSUPPORTED: " + str(u.get("reason", ""))[:70]}
    return out


def _per_step(p, ms_key):
    if not p or ms_key not in p or not p.get("iterations_actual"):
        return None
    return p[ms_key] / p["iterations_actual"]


def _fmt(x, nd=4):
    if x is None:
        return "—"
    if isinstance(x, float) and (math.isnan(x) or math.isinf(x)):
        return str(x)
    return f"{x:.{nd}g}"


def _cell(p, ms_key):
    if p is None:
        return "not run"
    if "_error" in p:
        return p["_error"]
    v = _per_step(p, ms_key)
    flag = "" if p.get("_valid") else " (gate FAIL)"
    return _fmt(v) + flag


def _ratio(a, b):
    return b / a if a and b else None


def _geomean(xs):
    xs = [x for x in xs if x and x > 0]
    return math.exp(sum(math.log(x) for x in xs) / len(xs)) if xs else None


def perks(d):
    print("\n## PERKS (ICS'23): ms per CG step, 10,000 steps, fp64, b = ones\n")
    print("| matrix | nnz | CSR MB | PERKS cached | PERKS baseline | cuSPARSE | cached vs baseline | cached vs cuSPARSE |")
    print("|---|---|---|---|---|---|---|---|")
    s1, s2 = [], []
    for m, impls in d.items():
        c, b, u = impls.get("perks-cg-cached"), impls.get("perks-cg-baseline"), impls.get("cusparse-cg")
        pc = _per_step(c, "perks_reported_ms") if c and "_error" not in c else None
        pb = _per_step(b, "perks_reported_ms") if b and "_error" not in b else None
        pu = _per_step(u, "cusparse_solve_ms_internal") if u and "_error" not in u else None
        nnz = next((p.get("_nnz") for p in impls.values() if p.get("_nnz")), None)
        rows = next((p.get("_rows") for p in impls.values() if p.get("_rows")), None)
        mb = (nnz * 12 + (rows or 0) * 4) / 1e6 if nnz else None
        r1, r2 = _ratio(pc, pb), _ratio(pc, pu)
        s1.append(r1); s2.append(r2)
        print(f"| {m} | {nnz} | {_fmt(mb, 3)} | {_cell(c, 'perks_reported_ms')} | "
              f"{_cell(b, 'perks_reported_ms')} | {_cell(u, 'cusparse_solve_ms_internal')} | "
              f"{_fmt(r1, 3)}x | {_fmt(r2, 3)}x |")
    print(f"\ngeomean speedup: cached vs PERKS baseline {_fmt(_geomean(s1), 3)}x, "
          f"cached vs cuSPARSE {_fmt(_geomean(s2), 3)}x")


def millefeuille(d):
    print("\n## Mille-feuille (SC'24): ms per CG step, 100 steps, b = A*ones\n")
    print("| matrix | mixed (kernel) | mixed relres | cuSPARSE fp64 | cuSPARSE relres | MF plain fp64 | mixed vs cuSPARSE |")
    print("|---|---|---|---|---|---|---|")
    s = []
    for m, impls in d.items():
        x, u, r = impls.get("millefeuille-cg-mixed"), impls.get("cusparse-cg"), impls.get("millefeuille-cg-reduce")
        px = x.get("millefeuille_reported_ms_per_iter") if x and "_error" not in x else None
        pu = _per_step(u, "cusparse_solve_ms_internal") if u and "_error" not in u else None
        sp = _ratio(px, pu)
        s.append(sp)
        print(f"| {m} | {_cell(x, 'millefeuille_reported_ms') if x else 'not run'} | "
              f"{_fmt((x or {}).get('relres_achieved'), 3)} | {_cell(u, 'cusparse_solve_ms_internal')} | "
              f"{_fmt((u or {}).get('relres_achieved'), 3)} | {_cell(r, 'millefeuille_reported_ms')} | "
              f"{_fmt(sp, 3)}x |")
    print(f"\ngeomean speedup, mixed vs cuSPARSE: {_fmt(_geomean(s), 3)}x  (paper: 3-5x)")


def spcg(d):
    print("\n## SPCG (SC'25): sparsified (sp) vs SPCG's own non-sparsified driver (nonsp), fp32, b = ones\n")
    print("Times are SPCG's own CUDA events. iter = PCG-loop speedup, e2e = overall (setup + loop) speedup.\n")
    print("| matrix | nonsp loop ms (it) | ratio | sp loop ms (it) | iter speedup | e2e speedup | sp relres |")
    print("|---|---|---|---|---|---|---|")
    best_it, best_e2e = [], []
    per_ratio = defaultdict(lambda: ([], []))
    for m, impls in d.items():
        ns = impls.get("spcg-ilu0-cg")
        if not ns or "_error" in ns or "spcg_pcg_ms" not in ns:
            print(f"| {m} | {(ns or {}).get('_error', 'not run')} | | | | | |")
            continue
        bi = be = None
        for key in sorted(k for k in impls if k.startswith("spcg-sp@")):
            sp = impls[key]
            ratio = key.split("@", 1)[1]
            if "_error" in sp or "spcg_pcg_ms" not in sp:
                print(f"| {m} | | {ratio} | {sp.get('_error', '—')} | | | |")
                continue
            it = _ratio(sp["spcg_pcg_ms"], ns["spcg_pcg_ms"])
            e2e = _ratio(sp["spcg_overall_ms_internal"], ns["spcg_overall_ms_internal"])
            per_ratio[ratio][0].append(it); per_ratio[ratio][1].append(e2e)
            if sp.get("_valid"):
                bi = max(bi or 0, it or 0); be = max(be or 0, e2e or 0)
            print(f"| {m} | {_fmt(ns['spcg_pcg_ms'])} ({ns.get('iterations_actual')}) | {ratio} | "
                  f"{_fmt(sp['spcg_pcg_ms'])} ({sp.get('iterations_actual')}) | {_fmt(it, 3)}x | "
                  f"{_fmt(e2e, 3)}x | {_fmt(sp.get('relres_achieved'), 3)}"
                  f"{'' if sp.get('_valid') else ' (gate FAIL)'} |")
        best_it.append(bi); best_e2e.append(be)
    for ratio, (its, e2es) in sorted(per_ratio.items()):
        print(f"\nratio {ratio}: geomean iter {_fmt(_geomean(its), 3)}x, e2e {_fmt(_geomean(e2es), 3)}x")
    print(f"\nbest valid ratio per matrix (oracle, upper bound on SPCG's selector): geomean iter "
          f"{_fmt(_geomean(best_it), 3)}x, e2e {_fmt(_geomean(best_e2e), 3)}x  "
          "(paper: 1.23x / 1.65x iterative, 1.68x / 3.73x end-to-end)")


def bcmg(d):
    print("\n## BootCMatchGX (TPDS'23) vs AmgX: 3D Poisson, fp64, rtol 1e-6, b = ones\n")
    print("| grid | BCMG setup / solve / total ms (it) | AmgX setup / solve / total ms (it) | solve speedup | total speedup |")
    print("|---|---|---|---|---|")
    for m, impls in d.items():
        b, a = impls.get("bootcmatchgx-cg-fcg-bcmg"), impls.get("amgx-pcg-amg")

        def trip(p, s, v):
            if not p:
                return "not run", None, None
            if "_error" in p:
                return p["_error"], None, None
            su, so = p.get(s), p.get(v)
            tot = su + so if su is not None and so is not None else None
            flag = "" if p.get("_valid") else " (gate FAIL)"
            return (f"{_fmt(su)} / {_fmt(so)} / {_fmt(tot)} ({p.get('iterations_actual')}){flag}", so, tot)

        tb, sb, totb = trip(b, "bcmgx_setup_ms_internal", "bcmgx_solve_ms_internal")
        ta, sa, tota = trip(a, "amgx_setup_ms", "amgx_solve_ms")
        print(f"| {m} | {tb} | {ta} | {_fmt(_ratio(sb, sa), 3)}x | {_fmt(_ratio(totb, tota), 3)}x |")
    print("\n(paper: up to 2x over AmgX; speedup > 1 means BootCMatchGX is faster)")


def main(root):
    perks(_runs(os.path.join(root, "paper_perks_*.json")))
    millefeuille(_runs(os.path.join(root, "paper_mf_*.json")))
    spcg(_runs(os.path.join(root, "paper_spcg_*.json")))
    bcmg(_runs(os.path.join(root, "paper_bcmg_*.json")))


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "results")
