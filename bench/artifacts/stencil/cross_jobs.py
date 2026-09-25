#!/usr/bin/env python3
"""
Run the stencil track's shared-configuration matrix: every workload of every
sweep in kernelbench/domains/stencil.py CROSS_PLAN, one GPU job per workload
(all of that sweep's implementations in one runner call, so the ones gated
under the same boundary convention share the cached CPU reference).

    $PY bench/artifacts/stencil/cross_jobs.py                  # print the jobs
    $PY bench/artifacts/stencil/cross_jobs.py --run            # run them, in order
    $PY bench/artifacts/stencil/cross_jobs.py --run --sweep cross-tcu-2d
    $PY bench/artifacts/stencil/cross_jobs.py --run --gres a100 --only star2d3r

Run from anywhere on the LOGIN node: each job goes through bench/gpu_run.sh
(one blocking srun). Results land in bench/results/stencil_<sweep>_
<workload>.json; a job whose result file already exists is skipped, so an
interrupted pass resumes where it stopped (--force re-runs).
"""
from __future__ import annotations

import argparse
import os
import re
import subprocess
import sys

BENCH = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
sys.path.insert(0, BENCH)

from kernelbench.domains import stencil  # noqa: E402


def jobs(sweeps: list[str] | None, only: str | None):
    for plan in stencil.CROSS_PLAN:
        if sweeps and plan["sweep"] not in sweeps:
            continue
        for wl in stencil.SWEEPS[plan["sweep"]]:
            if only and not wl.startswith(only):
                continue
            tag = re.sub(r"[^A-Za-z0-9]+", "_", wl).strip("_")
            out = f"results/stencil_{plan['sweep']}_{tag}.json"
            cmd = (f"$PY -m kernelbench.runner --kernel stencil --variant {plan['variant']} "
                   f"--precision {plan['precision']} --impl {','.join(plan['impls'])} "
                   f"--matrices {wl} --keep-going --out {out}")
            yield plan, wl, out, cmd


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--run", action="store_true", help="submit the jobs (default: print)")
    ap.add_argument("--sweep", action="append", help="limit to this sweep id (repeatable)")
    ap.add_argument("--only", help="limit to workloads starting with this prefix")
    ap.add_argument("--gres", default="a100", help="gpu_run.sh -g value (default a100)")
    ap.add_argument("--force", action="store_true", help="re-run jobs whose result exists")
    args = ap.parse_args()

    gpu_run = os.path.join(BENCH, "gpu_run.sh")
    todo = list(jobs(args.sweep, args.only))
    for i, (plan, wl, out, cmd) in enumerate(todo, 1):
        line = [gpu_run, "-g", args.gres, "-t", str(plan["minutes"]), "-m", plan["mem"], "--", cmd]
        shown = f"{gpu_run} -g {args.gres} -t {plan['minutes']} -m {plan['mem']} -- '{cmd}'"
        done = os.path.exists(os.path.join(BENCH, out))
        print(f"[{i}/{len(todo)}] {plan['sweep']} {wl}" + ("  (done, skipped)" if done and not args.force else ""))
        print(f"    {shown}", flush=True)
        if not args.run or (done and not args.force):
            continue
        rc = subprocess.call(line, cwd=BENCH)
        print(f"    -> exit {rc}{'' if os.path.exists(os.path.join(BENCH, out)) else ' (no result file)'}",
              flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
