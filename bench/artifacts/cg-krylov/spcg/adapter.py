"""
SPCG adapter for the cg-krylov track's `cg-e2e-ilu0-to-convergence` variant.

Paper: "Sparsified Preconditioned Conjugate Gradient Solver on GPUs" (SC'25).
PAPER_KEY = "conf/sc/MaACSH25" (title-matched in ../../../output/included.json).
Artifact: https://github.com/SwiftWare-Lab/SPCG, commit
e667f3acb6e6c3bb219d4a8b520fea10e0987ff3.

What is wrapped
----------------
`gpu_src/ilu0_gpu/nonsp/main.cpp` (built by build.sh into the
`conjugateGradientPrecond` binary) is a single end-to-end CLI driver: load a
.mtx file -> build a cuSPARSE ILU(0) factorization -> run PCG to its OWN
internal convergence criterion -> print/CSV-dump results -> exit. There is no
separate library entry point, so per ARTIFACT_GUIDE.md rule 1 ("if the
artifact only ships an end-to-end binary that loads matrices itself, wrap at
the finest boundary available and document the contamination") this adapter
wraps the COMPILED BINARY as a subprocess, one fresh process per solve.

This is not merely the easiest option -- it is the *correct* one for this
variant. The spec's own cg-e2e-ilu0-to-convergence protocol requires "5
independent runs (fresh factorization each run, to capture any
nondeterminism in parallel ILU factorization ordering on GPU)"; a fresh OS
process trivially satisfies "fresh factorization," with zero risk of
leftover CUDA-context/cuSPARSE-handle state leaking between reps (which an
in-process re-run would have to work much harder to guarantee, given the
artifact's own `cudaDeviceReset()` at the end of main()).

Two minimal patches to source/gpu_src/ilu0_gpu/nonsp/main.cpp (see STATUS.md
for the exact diff; ARTIFACT_GUIDE.md rule 3: build/IO changes are fine,
kernel code is not, and NEITHER patch touches a single cuBLAS/cuSPARSE call):
  1. rhs[] is read from env var SPCG_RHS_FILE (raw little-endian float32, N
     values) if set, else the artifact's own rhs=1.0 default is untouched.
     Lets this adapter honor the spec's `vector_operand` convention (b dense
     U(-1,1), seed=42) instead of SPCG's own all-ones default.
  2. after the solve, x is dumped to SPCG_X_OUT_FILE (raw little-endian
     float32, N values) if that env var is set -- purely additive, added
     after `cudaMemcpy(x, d_x, ...)`, never read back by the solver itself.
     This is the ONLY way to get the solution vector out of this artifact at
     all (upstream never writes x to a file); without it, to_host() would
     have nothing to gate on.

Correctness gate discipline (mirrors solvers.py's ScipyCG.to_host() in
"to-convergence" mode EXACTLY, so this adapter's pass/fail decision is
apples-to-apples comparable with the domain's own reference impl): to_host()
independently recomputes, in fp64, relres = ||b - A@x|| / ||b|| using the x
read back from the dump file and the SAME b that was actually written to
SPCG_RHS_FILE -- never trusting SPCG's own internal sqrt(r1)/tol/k bookkeeping,
which is deliberately NOT what governs the gate here (see notes_on_fairness
in benchspecs/cg-krylov/spec.yaml: SPCG's own tol=1e-12 is an ABSOLUTE
unnormalized threshold, not comparable across matrices/right-hand-sides of
different scale). gate_tol = max(10*rtol, 1e-9), rtol=1e-6; passed iff finite
and relres <= gate_tol. Returns np.array([1.0]) / np.array([0.0]) to match
solvers.reference_cg()'s sentinel exactly, so CORRECTNESS_MODE="exact"
(np.array_equal) works.

Timing-boundary caveat (documented prominently here AND in STATUS.md, per
the task brief): timer() returns the plain CPU-wall-clock kernelbench.harness.
Timer, NOT a CUDA-event timer -- the solve runs in a separate subprocess, so
there is no shared CUDA context to bracket with events from this process.
This means the harness's own canonical per-rep number includes process
startup, CUDA context init/teardown, and Matrix-Market text parsing on top of
the actual ILU(0)+PCG work -- coarser than SPCG's own in-process CUDA-event
brackets. Those SPCG-internal numbers (parsed from results_summary_float.csv,
written fresh by each subprocess invocation into a scratch cwd) are ALSO
captured into params for audit (`spcg_precond_ms`, `spcg_pcg_ms`,
`spcg_final_residual`, `spcg_overall_ms_internal`) -- but the harness's
subprocess wall-clock is what actually gets timed and reported as the
metric, which is the honest "contamination" ARTIFACT_GUIDE.md rule 1 asks to
be recorded rather than hidden.
"""

from __future__ import annotations

import csv
import os
import re
import shutil
import subprocess
import tempfile

KERNEL = "cg-krylov"
IMPL_NAME = "spcg-ilu0-cg"
PAPER_KEY = "conf/sc/MaACSH25"
PRECISIONS = ["fp32"]

_HERE = os.path.dirname(os.path.abspath(__file__))
_SOURCE = os.path.join(_HERE, "source")
_BINARY = os.path.join(
    _SOURCE, "gpu_src", "ilu0_gpu", "nonsp", "build", "conjugateGradientPrecond")


def available() -> tuple[bool, str]:
    if not os.path.exists(_BINARY):
        return False, f"binary not built: {_BINARY} missing (run build.sh)"
    if not os.access(_BINARY, os.X_OK):
        return False, f"binary present but not executable: {_BINARY}"
    return True, ""


def create(precision: str):
    return SpcgIlu0Cg(precision)


def _last_csv_row(path: str) -> dict | None:
    """Read a results_summary_float.csv (appended once per subprocess
    invocation) and return the LAST row as a dict, or None if the file is
    missing/empty. Structured CSV parsing (not stdout regex) since the
    artifact's own file already has stable, named columns."""
    if not os.path.exists(path):
        return None
    with open(path, newline="") as f:
        rows = list(csv.DictReader(f))
    return rows[-1] if rows else None


class SpcgIlu0Cg:
    """
    One `run()` call = one fresh `conjugateGradientPrecond` subprocess = one
    complete ILU(0) factorization + PCG-to-convergence solve, satisfying the
    spec's "fresh factorization each run" requirement automatically (see
    module docstring). `prepare()` writes the workload's CSR to a .mtx file
    and the spec's U(-1,1) rhs to a raw binary file once; `run()` launches
    the subprocess with those paths (plus SPCG_X_OUT_FILE for the dump) and
    records SPCG's own internal timing/residual/iteration-count breakdown
    into `params` for audit; `to_host()` performs the independent fp64
    correctness gate (see module docstring for exactly why and how).
    """

    name = IMPL_NAME
    platform = "cuda"

    def __init__(self, precision: str = "fp32"):
        # SPCG's own kernel code is float32-only throughout (cuBLAS S*
        # calls, cuSPARSE CUDA_R_32F descriptors, `const float tol`) -- not
        # something this adapter patches (ARTIFACT_GUIDE.md rule 3: build-
        # system fixes are fine, kernel numerics are not). Separately: the
        # runner's own precision-text heuristic (runner.py checks "fp64" in
        # variant.protocol.precision BEFORE "fp32", and this variant's spec
        # text legitimately says "fp32 (primary)... fp64 (secondary)") means
        # the default `--variant cg-e2e-ilu0-to-convergence` invocation
        # WITHOUT an explicit --precision flag resolves to "fp64" --
        # confirmed by loading the spec directly while building this
        # adapter. Rather than hard-failing on that mismatch (which would
        # break the task's own documented test invocation), this class
        # clamps to fp32 unconditionally and records the mismatch in params,
        # matching solvers.ScipyCG's own non-raising
        # `np.float64 if precision == "fp64" else np.float32` fallback style.
        self.requested_precision = precision
        self.precision = "fp32"
        self._workdir = None
        self._params = None
        self._A64 = None
        self._b64 = None
        self._mtx_path = None
        self._rhs_path = None
        self._x_out_path = None
        self._results_csv = None

    def prepare(self, matrix, params: dict):
        import numpy as np
        import scipy.io as sio

        params["precision"] = "fp32"  # see __init__ note: the cost model
        # (workload._cost_cg's ITEMSIZE lookup) must reflect the precision
        # actually executed, not whatever the runner's heuristic requested.
        if self.requested_precision != "fp32":
            params["spcg_precision_note"] = (
                f"requested precision {self.requested_precision!r} != fp32; "
                "SPCG's own kernel is float32-only (unpatched), clamped to fp32")

        n = matrix.shape[0]
        rng = np.random.default_rng(params.get("seed", 42))
        # spec's vector_operand: b dense, U(-1,1), fixed seed=42 -- same RNG
        # call as solvers.ScipyCG.prepare() and reference_cg's implicit
        # convention, so this artifact is gated against the identical b any
        # other cg-krylov implementation in this harness would see.
        b64 = rng.uniform(-1.0, 1.0, size=n)
        self._A64 = matrix.csr.astype(np.float64)
        self._b64 = b64

        self._workdir = tempfile.mkdtemp(prefix="spcg_")
        safe_name = re.sub(r"[^A-Za-z0-9_.-]+", "_", matrix.name) or "matrix"
        self._mtx_path = os.path.join(self._workdir, f"{safe_name}.mtx")
        # Artifact's own format-conversion-equivalent step (it only reads
        # .mtx): timed as preprocessing by the harness (ARTIFACT_GUIDE.md
        # rule 2). symmetry="general" avoids scipy's own symmetric-detection
        # path entirely -- matches what loadMMSparseMatrix produces either
        # way (it re-expands "symmetric" typecode matrices back to general
        # internally), and keeps nnz reported by the binary equal to
        # matrix.nnz exactly, with no ambiguity for the caller.
        sio.mmwrite(self._mtx_path, matrix.csr.tocoo(), symmetry="general")

        self._rhs_path = os.path.join(self._workdir, "rhs.bin")
        b64.astype("<f4").tofile(self._rhs_path)  # Patch 1: SPCG_RHS_FILE

        self._x_out_path = os.path.join(self._workdir, "x_out.bin")
        self._results_csv = os.path.join(self._workdir, "results_summary_float.csv")

        # resolved knobs written back so they are explicit, auditable fields
        # in RunResult.params -- same discipline as solvers.ScipyCG.prepare()
        params["mode"] = "e2e-ilu0-to-convergence"
        params["rtol"] = float(params.get("rtol", 1e-6))
        # SPCG's OWN internal loop-termination constants (main.cpp: `const
        # float tol = 1e-12f`, `int max_iter = 1000`) -- recorded for audit,
        # NEVER adopted as this adapter's own gate (see module docstring /
        # spec notes_on_fairness: SPCG's tol is absolute+unnormalized, ours
        # is a relative-residual gate). Not patched, per the task brief.
        params["spcg_internal_tol_abs_r1"] = 1e-12
        params["spcg_internal_max_iter"] = 1000
        params["spec_max_iter"] = 2000  # our spec's own cap (informational
        # only -- SPCG's own loop is never told about it; it always runs its
        # own hardcoded 1000-iteration cap regardless of what this says).

        self._params = params
        return {"mtx": self._mtx_path}

    def run(self, h):
        env = dict(os.environ)
        env["SPCG_RHS_FILE"] = self._rhs_path
        env["SPCG_X_OUT_FILE"] = self._x_out_path
        if os.path.exists(self._x_out_path):
            # avoid mistaking a stale dump (e.g. from a crashed prior call)
            # for evidence that this run actually produced fresh output
            os.remove(self._x_out_path)

        proc = subprocess.run(
            [_BINARY, self._mtx_path], cwd=self._workdir, env=env,
            capture_output=True, text=True, timeout=900,
        )

        p = self._params
        p["spcg_subprocess_returncode"] = proc.returncode
        if not os.path.exists(self._x_out_path):
            # SPCG's own post-hoc `qaerr1/qaerr2` sanity check (unrelated to
            # our gate) CAN make the process exit(EXIT_FAILURE) even after a
            # perfectly good solve -- that alone is not fatal here (the x
            # dump, our Patch 2, happens before that check runs). Only a
            # MISSING dump file is a real failure: it means the process
            # crashed/exited before even reaching the solve's end.
            raise RuntimeError(
                f"spcg subprocess produced no x dump (returncode={proc.returncode}); "
                f"stdout tail:\n{proc.stdout[-2000:]}\n"
                f"stderr tail:\n{proc.stderr[-2000:]}")

        row = _last_csv_row(self._results_csv)
        if row is not None:
            # SPCG's own in-process CUDA-event breakdown -- audit-only, see
            # module docstring's "Timing-boundary caveat". Updated on EVERY
            # run() call (including warmup/measured reps, matching
            # solvers.ScipyCG.run()'s own per-call params update), so by the
            # end of the protocol these reflect the LAST measured rep, same
            # as the iteration count the cost rule (_cost_cg) actually uses.
            p["iterations_actual"] = int(row["Iterations Spent"])
            p["spcg_final_residual"] = float(row["Final Residual"])
            p["spcg_precond_ms"] = float(row["Preconditioning Time (ms)"])
            p["spcg_pcg_ms"] = float(row["PCG Time (ms)"])
            p["spcg_overall_ms_internal"] = float(row["Overall Time (ms)"])

        return self._x_out_path

    def to_host(self, out):
        import numpy as np

        n = self._A64.shape[0]
        x = np.fromfile(out, dtype="<f4").astype(np.float64)
        if x.size != n:
            relres = float("inf")
            finite = False
        else:
            b_norm = max(float(np.linalg.norm(self._b64)), 1e-300)
            relres = float(np.linalg.norm(self._b64 - self._A64 @ x) / b_norm)
            finite = bool(np.isfinite(relres))

        p = self._params
        rtol = float(p.get("rtol", 1e-6))
        # mirrors solvers.ScipyCG.to_host()'s "to-convergence" gate EXACTLY
        # (same formula, same 10x slack), so this artifact's pass/fail
        # decision is directly comparable to the domain's own reference impl.
        gate_tol = max(10.0 * rtol, 1e-9)
        passed = finite and relres <= gate_tol
        p["relres_achieved"] = relres
        p["gate"] = f"relres<={gate_tol:.3e} (to-convergence, 10x slack over rtol)"
        p["gate_passed"] = bool(passed)
        return np.array([1.0 if passed else 0.0])

    def timer(self):
        # plain CPU wall-clock, NOT CudaEventTimer: the solve runs in a
        # separate subprocess with its own CUDA context, so there is nothing
        # for an in-process CUDA event to bracket. See module docstring's
        # "Timing-boundary caveat".
        from kernelbench.harness import Timer
        return Timer()

    def free(self, h):
        if self._workdir and os.path.isdir(self._workdir):
            shutil.rmtree(self._workdir, ignore_errors=True)
        self._workdir = None
