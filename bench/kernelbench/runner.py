"""
CLI: run a spec variant over a matrix set with one or more implementations.

    python -m kernelbench.runner --kernel spmm --variant spmm-cpu-kernel-f32 \
        --impl scipy-csr-spmm --matrices cant,pdb1HYS --dims 32,128

    python -m kernelbench.runner --kernel spmv --list
    python -m kernelbench.runner --kernel sddmm --smoke      # synthetic, tiny

Results are written as one JSON document per invocation under bench/results/.
"""

from __future__ import annotations

import argparse
import copy
import inspect
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from kernelbench import artifact_registry, domains, env, harness, matrices, spec, workload  # noqa: E402

RESULTS = os.path.normpath(os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "results"))

SMOKE_MATRICES = [
    ("smoke-uniform", dict(rows=4000, cols=4000, nnz_per_row=24, pattern="uniform")),
    ("smoke-banded", dict(rows=4000, cols=4000, nnz_per_row=24, pattern="banded")),
    ("smoke-powerlaw", dict(rows=4000, cols=4000, nnz_per_row=12, pattern="powerlaw")),
]


def _release_device_memory() -> None:
    """After a crashed run (--keep-going), return cached device memory so the
    next implementation does not inherit a full allocator."""
    torch = sys.modules.get("torch")
    if torch is not None:
        try:
            torch.cuda.empty_cache()
        except Exception:  # noqa: BLE001
            pass


def describe_variant(sp: spec.Spec, vid: str) -> None:
    v = sp.variant(vid)
    p = v.protocol
    print(f"\n{sp.kernel} / {v.id}")
    print(f"  claim      : {v.claim[:160]}")
    print(f"  warmup     : {p.warmup}  ({p.provenance.get('warmup')})")
    print(f"  reps       : {p.reps}  ({p.provenance.get('reps')})")
    print(f"  statistic  : {p.statistic}")
    print(f"  timer      : {p.timer[:90]}")
    print(f"  scope      : {p.timing_scope[:110]}")
    print(f"  tolerance  : {v.tolerance}  ({v.tolerance_provenance})")
    print(f"  metric     : {v.metric}")
    dims = v.dense_dims()
    if dims:
        print(f"  dense dims : {dims}")
    subset = v.recommended_subset()
    if subset:
        print(f"  matrices   : {len(subset)} recommended, first 8: {subset[:8]}")
    if p.unparsed:
        print(f"  UNPARSED   : {p.unparsed}")


def main() -> int:
    ap = argparse.ArgumentParser(description="HPC KernelBench runner")
    ap.add_argument("--kernel", required=True,
                    help="kernel slug; see --list-kernels")
    ap.add_argument("--list-kernels", action="store_true",
                    help="show every kernel and whether its domain is implemented")
    ap.add_argument("--variant")
    ap.add_argument("--impl", help="comma-separated implementation names")
    ap.add_argument("--matrices", help="comma-separated SuiteSparse names")
    ap.add_argument("--dims", help="comma-separated N (spmm) or K (sddmm) values")
    ap.add_argument("--precision", default=None)
    ap.add_argument("--smoke", action="store_true",
                    help="synthetic tiny matrices + reduced protocol (NOT spec-conforming)")
    ap.add_argument("--list", action="store_true", help="describe the spec's variants")
    ap.add_argument("--warmup", type=int, help="override protocol warmup")
    ap.add_argument("--reps", type=int, help="override protocol reps")
    ap.add_argument("--out", help="result json path")
    ap.add_argument("--keep-going", action="store_true",
                    help="record an implementation's crash on a workload (e.g. CUDA "
                         "out of memory) under the result's `errors` and continue, "
                         "instead of aborting the whole invocation")
    args = ap.parse_args()

    # Pin CUDA_HOME/PATH to the toolkit matching torch's CUDA major BEFORE any
    # domain module, artifact adapter or runtime-JIT DSL is imported (see
    # env.pin_cuda_toolchain for the 2026-09-04 default-module drift that
    # motivated this). Printed so a log always shows which toolkit ran.
    print(f"  [env] toolchain: {env.pin_cuda_toolchain()}")

    if args.list_kernels:
        status = domains.load_all()
        print("domain modules:")
        for mod, st in status.items():
            print(f"  {mod:14s} {st}")
        print("\nkernels:")
        for k in domains.kernels():
            mod = domains.OWNER[k]
            ok = status.get(mod) == "ok"
            impl = ""
            if ok:
                m = domains.load(k)
                impl = "implemented" if k in getattr(m, "KERNELS", []) else "planned"
            print(f"  {k:22s} domain={mod:12s} {impl if ok else status.get(mod)}")
        return 0

    domain = domains.load(args.kernel)
    if args.kernel not in getattr(domain, "KERNELS", []):
        print(f"kernel {args.kernel!r} is owned by domain {domains.OWNER[args.kernel]!r} "
              f"but not implemented there yet (PLANNED={getattr(domain,'PLANNED',[])})")
        return 2
    sp = spec.load(args.kernel)

    if args.list:
        print(f"=== {args.kernel} — {sp.summary[:200]}")
        for vid in sp.variant_ids:
            describe_variant(sp, vid)
        print(f"\nimplementations: cpu={sorted(domain.CPU_IMPLS.get(args.kernel, {}))}")
        try:
            print(f"                 cuda={sorted(domain.cuda_impls().get(args.kernel, {}))}")
        except Exception as e:
            print(f"                 cuda=unavailable ({type(e).__name__})")
        rows = artifact_registry.discover_status(args.kernel)
        if rows:
            print("paper artifacts:")
            for r in rows:
                mark = "ok " if r["available"] else "-- "
                print(f"  {mark}{r['name']:<16} {r['impl']:<24} {r['reason']}")
        return 0

    if not args.variant:
        ap.error("--variant is required (use --list to see them)")
    variant = sp.variant(args.variant)

    # precision: spec-driven, overridable
    precision = args.precision
    if precision is None:
        text = (variant.protocol.precision + " " + variant.id).lower()
        precision = ("fp64" if "fp64" in text else
                     ("fp32" if "fp32" in text else
                      domain.DEFAULT_PRECISION.get(args.kernel, "fp32")))

    # dims
    if args.dims:
        dims = [int(d) for d in args.dims.split(",")]
    else:
        dims = variant.dense_dims() or [128]
    if not getattr(domain, "DIM_KEY", {}).get(args.kernel):
        dims = [1]

    # implementations
    cpu_table = domain.CPU_IMPLS.get(args.kernel, {})
    try:
        cuda_table = domain.cuda_impls().get(args.kernel, {})
    except Exception:
        cuda_table = {}
    artifact_table = {n: meta["factory"]
                      for n, meta in artifact_registry.discover(args.kernel).items()}
    all_impls = {**cpu_table, **cuda_table, **artifact_table}
    impl_names = args.impl.split(",") if args.impl else sorted(cpu_table)[:1]
    for n in impl_names:
        if n not in all_impls:
            ap.error(f"unknown impl {n!r} for {args.kernel}; "
                     f"cpu={sorted(cpu_table)} cuda={sorted(cuda_table)} "
                     f"artifacts={sorted(artifact_table)}")

    # matrices
    mats = []
    if args.smoke:
        # Domains whose kernels share one workload type (sparse.py: spmv/
        # spmm/sddmm all consume a Matrix) can return a single fixed smoke
        # list from a zero-arg smoke_workloads(). Domains whose kernels have
        # mutually incompatible shapes (dense.py: a Cholesky SPD matrix is
        # not a valid GEMM or BLAS-1-vector input) need to know which kernel
        # is being smoked; smoke_workloads(kernel=...) opts into that. This
        # keeps sparse.py's original zero-arg contract working unchanged.
        # Same idea one level down: a single kernel whose VARIANTS have
        # mutually incompatible shapes for a given impl (annsearch.py: a
        # D=3 point-cloud workload is not a valid input for a high-
        # dimensional recall-variant impl, and vice versa) opts in via
        # smoke_workloads(variant=...) -- again purely additive, every
        # other domain's signature is untouched.
        sig_params = inspect.signature(domain.smoke_workloads).parameters
        smoke_kwargs = {}
        if "kernel" in sig_params:
            smoke_kwargs["kernel"] = args.kernel
        if "variant" in sig_params:
            smoke_kwargs["variant"] = args.variant
        mats = list(domain.smoke_workloads(**smoke_kwargs))
    else:
        names = args.matrices.split(",") if args.matrices else \
            variant.recommended_subset()[:3]
        # Optional domain hooks, both additive: expand_workloads() turns a
        # named sweep (e.g. stencil's "sweep:spider-2d-scaling") into its
        # workload list, and a load_workload() that accepts `variant` sizes
        # each workload per the variant being run (stencil variant 2 is
        # 10240^2 x T=10240, not variant 1's 16384^2 x T=1000).
        expand = getattr(domain, "expand_workloads", None)
        if expand:
            names = expand(names)
        load_kwargs = ({"variant": variant.id} if "variant" in
                       inspect.signature(domain.load_workload).parameters else {})
        for n in names:
            print(f"  loading {n} ...", flush=True)
            mats.append(domain.load_workload(n, **load_kwargs))

    # Optional domain hook: a variant may need to transform every workload
    # before anything (reference or impl) sees it -- e.g. spmm-binary-
    # adjacency-kernel forces the sparse operand's values to 1.0 so weighted
    # and pattern-only kernels are gated on the exact same binarized input
    # (kernelbench.domains.sparse.variant_transform). Applies to both the
    # --smoke and --matrices branches above, since both funnel into `mats`
    # here. Purely additive: domains without this hook (`getattr` default
    # None) run exactly as before.
    xf = getattr(domain, "variant_transform", None)
    if xf:
        mats = [xf(args.kernel, variant.id, m) for m in mats]

    environment = env.capture()
    env_warnings = env.warn_if_unsuitable(environment)
    for w in env_warnings:
        print(f"  [env] {w}")

    warmup = args.warmup if args.warmup is not None else (5 if args.smoke else None)
    reps = args.reps if args.reps is not None else (20 if args.smoke else None)

    # Optional domain opt-in: give every implementation its own deep copy of
    # each workload. Some stencil adapters rewrite the workload they are
    # handed in prepare() (SPIDER swaps in its radius-7 box, FlashFFTStencil
    # forces T=1, AN5D its own coefficients); with one shared object that
    # rewrite leaked into every implementation listed after it. Only for
    # domains whose workloads are small descriptors -- a deep copy of a
    # SuiteSparse matrix per implementation would not be.
    fresh_copies = getattr(domain, "FRESH_WORKLOAD_PER_IMPL", {}).get(args.kernel, False)

    records = []
    unsupported = []   # (impl, workload) pairs an impl declined, with its reason
    errors = []        # --keep-going: (impl, workload) pairs that crashed
    for impl_name in impl_names:
        try:
            impl = all_impls[impl_name](precision)
        except NotImplementedError as e:
            # an implementation that does not support the requested precision
            # at all (e.g. an fp16-only tensor-core kernel asked for fp32) is a
            # coverage fact, recorded like an unsupported workload -- not a crash
            reason = str(e).strip().splitlines()[0] if str(e).strip() else "unsupported precision"
            print(f"  {impl_name}: UNSUPPORTED at precision {precision}: {reason[:160]}")
            unsupported.append({"impl": impl_name, "workload": "*", "precision": precision,
                                "reason": str(e)})
            continue
        for m_shared in mats:
            for d in dims:
                m = copy.deepcopy(m_shared) if fresh_copies else m_shared
                params = {"seed": 42, "precision": precision}
                dim_key = getattr(domain, "DIM_KEY", {}).get(args.kernel)
                if dim_key:
                    params[dim_key] = d
                label = f"{impl_name} {m.name} " + \
                        (f"dim={d}" if dim_key else "")
                print(f"  running {label} ...", end="", flush=True)
                t0 = time.perf_counter()
                try:
                    r = harness.run_variant(
                        impl, m, variant, params,
                        reference=domain.REFERENCES[args.kernel],
                        correctness_mode=domain.CORRECTNESS_MODE[args.kernel],
                        reference_name=getattr(domain, "REFERENCE_NAME", {})
                            .get(args.kernel, "scipy fp64 CSR"),
                        warmup_override=warmup, reps_override=reps,
                        # optional domain hook: a cheaper stand-in for the
                        # correctness gate (stencil: same grid, fewer steps)
                        gate_matrix=(domain.gate_workload(args.kernel, m)
                                     if hasattr(domain, "gate_workload") else None),
                        # workload-derived correctness bound (e.g. a lossy
                        # compressor's per-run error bound); None for every
                        # domain that doesn't set this, so behavior is unchanged
                        tolerance_override=getattr(m, "correctness_tolerance", None))
                except NotImplementedError as e:
                    # An implementation may legitimately not cover a workload
                    # (paper artifacts are often compiled for fixed shapes,
                    # head dims, radices, ...). ARTIFACT_GUIDE asks adapters to
                    # raise NotImplementedError from prepare() naming the
                    # constraint; that is a per-(impl, workload) fact to
                    # record, not a reason to abort the whole sweep and lose
                    # the workloads the impl does support (2026-09-05, fft/
                    # turbofft: mixed-radix smoke shapes killed the run).
                    reason = str(e).strip().splitlines()[0] if str(e).strip() else "unsupported"
                    print(f" UNSUPPORTED: {reason[:160]}")
                    unsupported.append({"impl": impl_name, "workload": m.name,
                                        **({dim_key: d} if dim_key else {}),
                                        "reason": str(e)})
                    continue
                except Exception as e:  # noqa: BLE001
                    if not args.keep_going:
                        raise
                    import traceback
                    traceback.print_exc()
                    reason = f"{type(e).__name__}: {e}".strip().splitlines()[0]
                    print(f" ERROR (--keep-going): {reason[:160]}")
                    errors.append({"impl": impl_name, "workload": m.name,
                                   **({dim_key: d} if dim_key else {}),
                                   "error": f"{type(e).__name__}: {e}"})
                    _release_device_memory()
                    continue
                dt = time.perf_counter() - t0
                if r.valid:
                    unit = r.metrics.get("throughput_unit", "GFLOP/s")
                    print(f" {r.stats_ms['median']:.3f} ms  "
                          f"{r.metrics['gflops']:.2f} {unit}  "
                          f"(err {r.correctness.value:.2e} <= {r.correctness.tolerance})"
                          f"  [{dt:.1f}s]")
                    native = r.metrics.get("paper_native")
                    if native:
                        p = native["primary"]
                        print(f"      paper metric: {p['value']:.4g} {p['name']}")
                else:
                    print(f" INVALID: {r.correctness.metric}="
                          f"{r.correctness.value:.3e} vs tol {r.correctness.tolerance}")
                records.append(r.to_dict())

    doc = {
        "schema": "kernelbench/result/v1",
        "kernel": args.kernel,
        "variant": variant.id,
        "spec_path": os.path.relpath(sp.path),
        "spec_claim": variant.claim,
        "spec_summary": sp.summary,
        "spec_notes_on_fairness": sp.notes_on_fairness,
        "conforming": bool(not args.smoke and warmup is None and reps is None
                           and not env_warnings
                           and not any(r["warnings"] for r in records)),
        "nonconformance_reasons": (
            (["synthetic smoke matrices; reduced protocol"] if args.smoke else [])
            + (["protocol overridden on the command line"]
               if (args.warmup is not None or args.reps is not None) else [])
            + env_warnings
            + sorted({w for r in records for w in r["warnings"]})),
        "environment": environment,
        "runs": records,
        # workloads an impl declined (NotImplementedError from prepare()); not
        # failures, not successes -- coverage facts, kept out of `runs`
        "unsupported": unsupported,
        # --keep-going only: crashes (e.g. device out of memory), with the error
        "errors": errors,
    }
    os.makedirs(RESULTS, exist_ok=True)
    out = args.out or os.path.join(
        RESULTS, f"{args.kernel}_{variant.id}_{int(time.time())}.json")
    with open(out, "w") as f:
        json.dump(doc, f, indent=1)
    n_valid = sum(1 for r in records if r["valid"])
    extra = f" ({len(unsupported)} unsupported)" if unsupported else ""
    if errors:
        extra += f" ({len(errors)} crashed, see `errors`)"
    print(f"\n{n_valid}/{len(records)} runs valid{extra} -> {out}")
    print(f"conforming: {doc['conforming']}"
          + ("" if doc["conforming"] else f"  ({'; '.join(doc['nonconformance_reasons'])})"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
