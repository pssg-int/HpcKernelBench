"""
Mille-feuille adapter for the cg-krylov track.

Paper: "Mille-feuille: A Tile-Grained Mixed Precision Single-Kernel
Conjugate Gradient Solver on GPUs" (SC'24).
`PAPER_KEY = "conf/sc/YangZNJS0T024"` (title-matched in
`../../../output/included.json`).
Artifact: https://github.com/SuperScientificSoftwareLaboratory/Mille-feuille

`source/src/main-cg.cu` is a single-file CLI driver (`./main-cg matrix.mtx`,
no separate library API) containing three alternative CG solvers dispatched
by nnz: `cg_solve_inc` (nnz<10000), `cg_solve_sync` (10000<=nnz<100000),
`cg_solve_reduce` (nnz>=100000, the paper's actual single-kernel tile-grained
mixed-precision contribution). Per ARTIFACT_GUIDE.md rule 1 ("wrap at the
finest boundary available"), this adapter wraps the process boundary: each
`run()` launches a fresh `main-cg` subprocess for ONE full solve. `PRECISIONS
= ["fp64"]` -- `MAT_VAL_TYPE` (the interface/accumulation type, and the type
of everything this adapter observes: RHS, x, residual) is hardcoded `double`
in `src/common.h`. The paper's actual contribution is an INTERNAL fp32
companion array (`Val_Low`, `MAT_VAL_LOW_TYPE=float`) used for on-chip
tile-wise dynamic precision downcasting during the SpMV -- that mixed-
precision behavior happens unconditionally inside `cg_solve_reduce` and is
not a precision this adapter can select or turn off; it is documented here
and in STATUS.md, not modeled as a second PRECISIONS entry.

Three source patches were applied (see STATUS.md for the full diff and
narrative) -- summarized here since they change what this adapter can rely
on:
  Patch 1 (bug fix): `cg_solve_reduce`'s loop was hardcoded
    `while (iterations < 10000)`, ignoring the `maxiter` argument the
    function already receives (a real, independently documented artifact
    bug -- see benchspecs/cg-krylov/spec.yaml's own text and survey.md's
    Divergences section). Fixed to `while (iterations < maxiter)`. The
    reported time was also divided by a literal `100` (not the true
    iteration count); fixed by printing the RAW loop time plus the true
    `iterations` count on stdout, so the per-iteration division happens
    here in Python where it can be audited exactly.
  Patch 2: `main()` now dumps the final host `X` array (the solution) to a
    file when `MF_X_OUT_FILE` is set -- this adapter's only way to read
    back the solved x independently of the artifact's own (mis-terminated-
    until-Patch-1) internal residual bookkeeping.
  Patch 3: `main()` now optionally overrides the manufactured RHS
    (`Y_golden = A @ ones(n)`) with an externally supplied vector when
    `MF_RHS_FILE` is set. This adapter always sets it, generating
    b ~ U(-1,1) seed=42 to match this track's spec convention (rather than
    the artifact's own manufactured RHS) -- and, as a side effect, avoids a
    separate pre-existing artifact quirk: `main()` truncates `n` down to a
    multiple of `BLOCK_SIZE=16` before computing `Y_golden`, but the
    truncation-unaware `Y_golden[i] += Val[j]*X[ColIdx[j]]` loop can index
    `X` (sized to the TRUNCATED n) with untruncated `ColIdx` values, an
    out-of-bounds host read for matrices whose dimension is not a multiple
    of 16. Supplying Y_golden directly sidesteps this entirely.
  Neither `cg_solve_inc` nor `cg_solve_sync` shares Patch 1's bug (they are
  single persistent-kernel launches with a device-side iteration budget, no
  `while (iterations < LITERAL)` host loop at all) -- confirmed by reading
  both functions; not patched since only `cg_solve_reduce` is this paper's
  contribution and the survey's cited bug.

Independent correctness check (`to_host()`): reads the dumped x, and
recomputes ||b - A_trunc @ x|| / ||b|| in fp64 against A_trunc (see
BLOCK_SIZE-truncation note above -- A_trunc is A's top-left
n_truncated x n_truncated principal submatrix, the best available
approximation of what the tile format actually represents once truncated).
Never trusts the artifact's own internal residual bookkeeping. Mirrors
`kernelbench.domains.solvers.ScipyCG.to_host()`'s fixed-iter gate exactly:
`passed = finite and relres < 1.0` (genuinely made progress from x0=0,
NOT full convergence -- a fixed iteration budget never promises that).

`iterations_actual` (read back from the binary's own Patch-1-fixed stdout,
never assumed) and `millefeuille_reported_ms` (the artifact's own timer,
labeled clearly as NOT the harness's canonical number) are written into
`params` for audit, mirroring solvers.py's own "params as a side channel"
convention.

Timing boundary (documented, not hidden): `timer()` is a plain CPU
wall-clock `Timer` around one subprocess launch. Because main-cg has no
persistent state or library entry point, EVERY timed `run()` (warmup and
measured reps alike) re-executes: process startup, `.mtx` re-parsing
(biio), and the artifact's own CSR->tile-format construction (`time_format`
inside the C++ code) -- none of that is separated out by OUR
`preprocessing_ms` (which covers only this adapter's own `prepare()`:
writing the `.mtx`/RHS files once). This is a materially larger timing-
boundary contamination than a typical subprocess-wrapped adapter and is
flagged prominently in STATUS.md; it is an unavoidable consequence of the
artifact shipping no separable "solve" entry point, per ARTIFACT_GUIDE.md
rule 1's "document the contamination" escape hatch.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import tempfile

KERNEL = "cg-krylov"
IMPL_NAME = "millefeuille-cg-reduce"
PAPER_KEY = "conf/sc/YangZNJS0T024"
PRECISIONS = ["fp64"]

_HERE = os.path.dirname(os.path.abspath(__file__))
_SOURCE = os.path.join(_HERE, "source")
_BINARY = os.path.join(_SOURCE, "main-cg")
_BLOCK_SIZE = 16  # matches src/common.h's #define BLOCK_SIZE 16


def available() -> tuple[bool, str]:
    if not os.path.exists(_BINARY):
        return False, "source/main-cg missing -- run build.sh first"
    if not os.access(_BINARY, os.X_OK):
        return False, "source/main-cg exists but is not executable"
    try:
        r = subprocess.run(["nvidia-smi", "-L"], capture_output=True,
                            text=True, timeout=10)
        if r.returncode != 0 or "GPU" not in r.stdout:
            return False, "nvidia-smi did not report a GPU"
    except Exception as e:
        return False, f"nvidia-smi check failed: {type(e).__name__}: {e}"
    return True, ""


def create(precision: str):
    return MilleFeuilleCG(precision)


class MilleFeuilleCG:
    """Subprocess-per-call wrapper around Mille-feuille's `main-cg` binary.
    See module docstring for the full architecture/patch narrative."""

    name = IMPL_NAME
    platform = "cuda"

    def __init__(self, precision: str = "fp64"):
        if precision != "fp64":
            raise NotImplementedError(
                f"{IMPL_NAME} only wired for fp64 (MAT_VAL_TYPE=double is "
                f"hardcoded in the artifact's src/common.h; requested {precision})")
        self.precision = precision
        self._tmpdir = None
        self._last_stdout = ""

    def prepare(self, matrix, params: dict):
        import numpy as np
        from scipy.io import mmwrite

        A = matrix.csr.astype(np.float64)
        n = A.shape[0]
        n_trunc = (n // _BLOCK_SIZE) * _BLOCK_SIZE
        if n_trunc <= 0:
            n_trunc = n  # degenerate guard; not expected for this track's matrices

        maxiter = int(params.get("maxiter", 50))
        # resolved knobs written back so they are explicit, auditable fields
        # in RunResult.params -- mirrors ScipyCG.prepare()'s own convention
        params["mode"] = "fixed-iter"  # the only mode this artifact implements
        params["maxiter"] = maxiter
        params["millefeuille_n_original"] = int(n)
        params["millefeuille_n_truncated"] = int(n_trunc)
        params["millefeuille_nnz_original"] = int(A.nnz)
        if n_trunc != n:
            params["millefeuille_truncation_note"] = (
                f"main-cg truncates n to a multiple of BLOCK_SIZE={_BLOCK_SIZE} "
                f"internally ({n} -> {n_trunc}, {n - n_trunc} trailing rows/cols "
                "dropped); this adapter's correctness check uses A's top-left "
                "n_truncated x n_truncated principal submatrix accordingly")
        nnz_full = A.nnz
        params["millefeuille_code_path"] = (
            "cg_solve_reduce" if nnz_full >= 100000 else
            "cg_solve_sync" if nnz_full >= 10000 else "cg_solve_inc")

        rng = np.random.default_rng(params.get("seed", 42))
        b = rng.uniform(-1.0, 1.0, size=n_trunc).astype(np.float64)

        # --- artifact's own input format: a Matrix Market file -----------
        # writing it out (once, here) IS the artifact's format-conversion
        # step under the harness contract; timed as preprocessing.
        tmpdir = tempfile.mkdtemp(prefix="millefeuille_")
        mtx_path = os.path.join(tmpdir, "matrix.mtx")
        mmwrite(mtx_path, A, symmetry="general")
        rhs_path = os.path.join(tmpdir, "rhs.bin")
        b.tofile(rhs_path)
        x_out_path = os.path.join(tmpdir, "x_out.bin")

        self._tmpdir = tmpdir
        self._params = params
        self._A_trunc = A[:n_trunc, :n_trunc].tocsr()
        self._b = b
        self._n_trunc = n_trunc
        self._maxiter = maxiter

        return {"mtx": mtx_path, "rhs": rhs_path, "x_out": x_out_path,
                "maxiter": maxiter}

    def run(self, h):
        env = dict(os.environ)
        env["MF_MAXITER"] = str(h["maxiter"])
        env["MF_RHS_FILE"] = h["rhs"]
        env["MF_X_OUT_FILE"] = h["x_out"]
        proc = subprocess.run(
            [_BINARY, h["mtx"]], cwd=_SOURCE, env=env,
            capture_output=True, text=True, timeout=1800)
        self._last_stdout = proc.stdout
        if proc.returncode != 0:
            raise RuntimeError(
                f"{IMPL_NAME}: main-cg exited {proc.returncode}\n"
                f"stdout(tail)={proc.stdout[-1000:]}\nstderr(tail)={proc.stderr[-1000:]}")
        return h["x_out"]

    def to_host(self, out):
        import numpy as np

        reported_ms = None
        reported_iters = None
        for line in self._last_stdout.splitlines():
            if line.startswith("time_cg="):
                # "time_cg=<float> ms iterations=<int>" (post-Patch-1 format)
                try:
                    tokens = line.replace("time_cg=", "").split()
                    reported_ms = float(tokens[0])
                    for tok in tokens:
                        if tok.startswith("iterations="):
                            reported_iters = int(tok.split("=", 1)[1])
                except Exception:
                    pass

        n_trunc = self._n_trunc
        x = np.fromfile(out, dtype=np.float64)
        p = self._params
        if x.size != n_trunc:
            p["millefeuille_x_dump_size_mismatch"] = (
                f"expected {n_trunc} doubles, got {x.size}")
            x = np.zeros(n_trunc) if x.size == 0 else np.resize(x, n_trunc)

        A = self._A_trunc
        b = self._b
        b_norm = max(float(np.linalg.norm(b)), 1e-300)
        relres = float(np.linalg.norm(b - A @ x) / b_norm)
        finite = bool(np.isfinite(relres))

        p["relres_achieved"] = relres
        # the exact quantity this track's survey flags as the artifact's own
        # bug (see module docstring); read back from the Patch-1-fixed
        # binary's own stdout, falling back to the configured maxiter only
        # if parsing somehow failed (never silently assumed correct)
        p["iterations_actual"] = reported_iters if reported_iters is not None else self._maxiter
        if reported_iters is not None and reported_iters != self._maxiter:
            p["millefeuille_iterations_mismatch"] = (
                f"binary reported {reported_iters} actual iterations vs "
                f"configured maxiter={self._maxiter}")
        if reported_ms is not None:
            # audit-only: the artifact's OWN timer, NOT the harness's
            # canonical number (that's the subprocess wall-clock Timer below)
            p["millefeuille_reported_ms"] = reported_ms
            iters_for_div = reported_iters or self._maxiter
            if iters_for_div:
                p["millefeuille_reported_ms_per_iter"] = reported_ms / iters_for_div

        # mirrors ScipyCG.to_host()'s fixed-iter gate exactly (see
        # kernelbench/domains/solvers.py): finite and genuinely decreased
        # from x0=0 (relres0==1.0 exactly), NOT full convergence.
        passed = finite and relres < 1.0
        p["gate"] = ("finite-and-decreased (fixed-iter; full convergence not "
                     "required) -- mirrors ScipyCG.to_host()")
        p["gate_passed"] = bool(passed)
        return np.array([1.0 if passed else 0.0])

    def timer(self):
        from kernelbench.harness import Timer
        return Timer()

    def free(self, h):
        if self._tmpdir:
            shutil.rmtree(self._tmpdir, ignore_errors=True)
            self._tmpdir = None
