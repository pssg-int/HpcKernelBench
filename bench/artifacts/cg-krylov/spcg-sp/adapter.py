"""
SPCG (SC'25), the paper's actual method: the SPARSIFIED ILU(0) PCG driver
`gpu_src/ilu0_gpu/sp` (../spcg wraps `ilu0_gpu/nonsp`, which is SPCG's own
NON-sparsified baseline -- the thing the paper's speedups are measured
against).

Paper: "Sparsified Preconditioned Conjugate Gradient Solver on GPUs".
PAPER_KEY = conf/sc/MaACSH25. Artifact: https://github.com/SwiftWare-Lab/SPCG
@ e667f3a (source/ is a symlink to ../spcg/source, same clone).

What the `sp` driver does (read from the source, not changed):
  - argv: <A.mtx> <ratio>. It reads A, and a second, already-sparsified
    matrix from <dir>/<name>_<ratio %.2f>.mtx. ILU(0) is built on the
    sparsified matrix; CG runs on the original A (SpMV with A, triangular
    solves with the sparsified factors).
  - fp32, rhs = 1, x0 = 0, stops when ||r|| < 1e-12 (absolute, on the
    recurrence residual) or after 1000 iterations -- identical to `nonsp`.
  - Both drivers time with CUDA events: "Preconditioning Time" (ILU(0)
    analysis + factorization + SpSV analyses) and "Overall Time" (that plus
    the PCG loop), written to results_summary.csv. Both record an event and
    write a CSV line every iteration inside the timed loop, so that cost is
    the same on both sides of the paper's comparison.
  - Its matrix reader (sp/util.h readMatrix + make_full) only accepts a
    real "symmetric" Matrix Market file holding the LOWER triangle in
    column-major order with every column present -- the layout of the
    SuiteSparse downloads and of MATLAB mmwrite output the paper's pipeline
    produces. prepare() writes A and the sparsified matrix in exactly that
    layout.

The sparsified matrix: SPCG builds it offline with MATLAB
(script_src/matlab_scripts/matrix_sparsification.m). _sparsify() below is a
line-for-line port: sort |a_ij| over all stored nonzeros of the full
matrix, threshold = sorted[max(1, floor(nnz * ratio))] (1-based), drop
every OFF-diagonal entry with |a_ij| <= threshold. Ratios 0.01 / 0.05 / 0.10
are the ones the paper's scripts generate. The paper picks one ratio per
matrix with script_src/python_scripts/algorithm/alg_ilu0.py from MATLAB-
computed condition-number/norm/wavefront statistics; that selector is NOT
ported. Run each ratio (KB_SPCG_RATIO) and report all three; the best of
the three is an upper bound on what the selector could pick.

mixed-in patch (sp.patch, applied by build.sh; I/O only, same as ../spcg's
patch on nonsp): SPCG_RHS_FILE (external b, float32), SPCG_X_OUT_FILE
(dump of the solution).

Gate: identical to ../spcg (to-convergence, relres = ||b - A x|| / ||b||
recomputed in fp64, pass iff finite and <= max(10 * rtol, 1e-9), rtol 1e-6).
Timing: subprocess wall clock (harness) + SPCG's own CUDA-event times in
params: spcg_precond_ms, spcg_pcg_ms (= Overall - Preconditioning),
spcg_overall_ms_internal -- the numbers the paper's speedups use.
"""
from __future__ import annotations

import csv
import os
import re
import shutil
import subprocess
import tempfile

KERNEL = "cg-krylov"
IMPL_NAME = "spcg-sp-ilu0-cg"
PAPER_KEY = "conf/sc/MaACSH25"
PRECISIONS = ["fp32"]

_HERE = os.path.dirname(os.path.abspath(__file__))
_SOURCE = os.path.join(_HERE, "source")   # symlink -> ../spcg/source
_BINARY = os.path.join(_SOURCE, "gpu_src", "ilu0_gpu", "sp", "build",
                       "conjugateGradientPrecond")
_RATIOS = (0.01, 0.05, 0.1)   # matrix_sparsification.m removal_percentages


def available() -> tuple[bool, str]:
    if not os.path.exists(_BINARY):
        return False, f"binary not built: {_BINARY} missing (run build.sh)"
    if not os.access(_BINARY, os.X_OK):
        return False, f"binary present but not executable: {_BINARY}"
    return True, ""


def create(precision: str):
    return SpcgSparsified(precision)


def _sparsify(A, ratio):
    """Port of matrix_sparsification.m: returns (A_sparsified, n_removed)."""
    import numpy as np
    import scipy.sparse as sp

    coo = A.tocoo()
    vals = np.abs(coo.data)
    s = np.sort(vals)
    idx = max(1, int(np.floor(s.size * ratio)))      # MATLAB 1-based index
    thr = s[idx - 1]
    drop = (coo.row != coo.col) & (vals <= thr)
    keep = ~drop
    As = sp.csr_matrix((coo.data[keep], (coo.row[keep], coo.col[keep])), shape=A.shape)
    As.sort_indices()
    return As, int(drop.sum()), float(thr)


def _write_lower_colmajor(path, A):
    """Matrix Market 'symmetric' file, lower triangle, column-major order --
    the layout sp/util.h readMatrix() requires (see module docstring)."""
    import numpy as np
    import scipy.sparse as sp

    L = sp.tril(A, format="csc")
    L.sort_indices()
    n = A.shape[0]
    cols_present = np.diff(L.indptr) > 0
    if not cols_present.all():
        raise NotImplementedError(
            f"{IMPL_NAME}: column {int(np.argmin(cols_present))} of the lower triangle is "
            "empty; SPCG's sp reader cannot represent that (needs a stored diagonal)")
    col = np.repeat(np.arange(n), np.diff(L.indptr))
    with open(path, "w") as f:
        f.write("%%MatrixMarket matrix coordinate real symmetric\n")
        f.write(f"{n} {n} {L.nnz}\n")
        np.savetxt(f, np.column_stack([L.indices + 1, col + 1, L.data]),
                   fmt=["%d", "%d", "%.17g"])


def _last_csv_row(path):
    if not os.path.exists(path):
        return None
    with open(path, newline="") as f:
        rows = list(csv.DictReader(f))
    return rows[-1] if rows else None


class SpcgSparsified:
    name = IMPL_NAME
    platform = "cuda"

    def __init__(self, precision: str = "fp32"):
        # float32-only kernel, same as ../spcg: clamp and record
        self.requested_precision = precision
        self.precision = "fp32"
        self._workdir = None

    def prepare(self, matrix, params: dict):
        import numpy as np
        from kernelbench.domains.solvers import cg_rhs

        ratio = float(params.get("spcg_ratio", os.environ.get("KB_SPCG_RATIO", 0.05)))
        if not any(abs(ratio - r) < 1e-9 for r in _RATIOS):
            raise ValueError(f"KB_SPCG_RATIO={ratio}: SPCG's scripts generate only {_RATIOS}")
        params["precision"] = "fp32"
        if self.requested_precision != "fp32":
            params["spcg_precision_note"] = (
                f"requested {self.requested_precision!r}; SPCG is float32-only, clamped")

        A = matrix.csr.tocsr().astype(np.float64)
        A.sum_duplicates()
        self._A64 = A
        self._b64 = cg_rhs(A, params)   # KB_CG_RHS (the paper: ones)

        As, n_removed, thr = _sparsify(A, ratio)
        params["spcg_ratio"] = ratio
        params["spcg_threshold"] = thr
        params["spcg_offdiag_removed"] = n_removed
        params["spcg_offdiag_removed_frac"] = n_removed / max(A.nnz, 1)

        self._workdir = tempfile.mkdtemp(prefix="spcg_sp_")
        safe = re.sub(r"[^A-Za-z0-9_]+", "_", matrix.name) or "matrix"
        self._mtx = os.path.join(self._workdir, f"{safe}.mtx")
        _write_lower_colmajor(self._mtx, A)
        # the driver derives this name itself: <dir>/<name>_<ratio %.2f>.mtx
        _write_lower_colmajor(os.path.join(self._workdir, f"{safe}_{ratio:.2f}.mtx"), As)
        self._rhs = os.path.join(self._workdir, "rhs.bin")
        self._b64.astype("<f4").tofile(self._rhs)
        self._x_out = os.path.join(self._workdir, "x_out.bin")
        self._csv = os.path.join(self._workdir, "results_summary.csv")
        self._ratio = ratio

        params["mode"] = "e2e-sparsified-ilu0-to-convergence"
        params["rtol"] = float(params.get("rtol", 1e-6))
        params["spcg_internal_tol_abs"] = 1e-12
        params["spcg_internal_max_iter"] = 1000
        self._params = params
        return {"mtx": self._mtx}

    def run(self, h):
        env = dict(os.environ)
        env["SPCG_RHS_FILE"] = self._rhs
        env["SPCG_X_OUT_FILE"] = self._x_out
        if os.path.exists(self._x_out):
            os.remove(self._x_out)
        proc = subprocess.run([_BINARY, self._mtx, f"{self._ratio}"], cwd=self._workdir,
                              env=env, capture_output=True, text=True, timeout=900)
        p = self._params
        p["spcg_subprocess_returncode"] = proc.returncode
        if not os.path.exists(self._x_out):
            raise RuntimeError(
                f"{IMPL_NAME}: no x dump (returncode={proc.returncode});\n"
                f"stdout tail:\n{proc.stdout[-2000:]}\nstderr tail:\n{proc.stderr[-2000:]}")
        row = _last_csv_row(self._csv)
        if row is not None:
            overall = float(row["Overall Time (ms)"])
            precond = float(row["Preconditioning Time (ms)"])
            p["iterations_actual"] = int(row["Iterations Spent"])
            p["spcg_final_residual"] = float(row["Final Residual"])
            p["spcg_precond_ms"] = precond
            p["spcg_pcg_ms"] = overall - precond
            p["spcg_overall_ms_internal"] = overall
        return self._x_out

    def to_host(self, out):
        import numpy as np

        n = self._A64.shape[0]
        x = np.fromfile(out, dtype="<f4").astype(np.float64)
        p = self._params
        if x.size != n:
            relres, finite = float("inf"), False
        else:
            bn = max(float(np.linalg.norm(self._b64)), 1e-300)
            relres = float(np.linalg.norm(self._b64 - self._A64 @ x) / bn)
            finite = bool(np.isfinite(relres))
        gate_tol = max(10.0 * p["rtol"], 1e-9)
        passed = finite and relres <= gate_tol
        p["relres_achieved"] = relres
        p["gate"] = f"relres<={gate_tol:.3e} (to-convergence, 10x slack over rtol)"
        p["gate_passed"] = bool(passed)
        return np.array([1.0 if passed else 0.0])

    def timer(self):
        from kernelbench.harness import Timer
        return Timer()

    def free(self, h):
        if self._workdir and os.path.isdir(self._workdir):
            shutil.rmtree(self._workdir, ignore_errors=True)
        self._workdir = None
