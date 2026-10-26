"""
cuSPARSE conjugate gradient -- the vendor-library BASELINE for the
fixed-iteration group (`cg-kernel-fixed-iter`): the comparator PERKS and
Mille-feuille should both beat.

Why this baseline: Mille-feuille (SC'24) claims 3-5x over cuSPARSE on A100,
and PERKS (ICS'23) has no vendor-library comparator in our harness yet.
bridge.cu ports NVIDIA's CUDA samples conjugateGradient /
conjugateGradientPrecond (SPCG's own driver starts from the latter); it ports those
samples' call sequence (cusparseSpMV + cuBLAS dot/axpy/scal/copy, host
pointer mode); see its header for exactly what is kept and what differs.
This is a library baseline, not a paper's contribution; PAPER_KEY names the
paper whose claim it is the comparator for.

Protocol (mirrors solvers.ScipyCG "fixed-iter" and the perks/millefeuille
adapters, so the rows are directly comparable):
  - b = U(-1,1), seed 42; x0 = 0.
  - run() = exactly `maxiter` (default 50) unpreconditioned CG steps, no
    early exit. The matrix and vectors are uploaded once in prepare().
  - Timed IN-PROCESS with CUDA events (CudaEventTimer): only the solve,
    no process start-up, file parsing or H2D copy. The perks/millefeuille
    adapters run a subprocess per call and are timed with a wall clock that
    includes all of those, so compare against their own internal times
    (`perks_reported_ms`, `millefeuille_reported_ms_per_iter`) as well.
  - Gate: fixed-iter rule, finite and relres < 1 (made progress from x0=0),
    relres recomputed independently in fp64 from the returned x.

The ILU(0)-preconditioned variant lives in ../cusparse-ilu0-cg/ and reuses
this directory's bridge.so and the CusparseCG class below.
"""
from __future__ import annotations

import ctypes
import os

import numpy as np

KERNEL = "cg-krylov"
IMPL_NAME = "cusparse-cg"
PAPER_KEY = "conf/sc/YangZNJS0T024"   # Mille-feuille: "3-5x over cuSPARSE"
ROLE = "baseline"
PRECISIONS = ["fp64", "fp32"]

_HERE = os.path.dirname(os.path.abspath(__file__))
SO_PATH = os.path.join(_HERE, "bridge.so")
_PREC = {"fp64": (0, np.float64), "fp32": (1, np.float32)}


def available() -> tuple[bool, str]:
    try:
        if not os.path.exists(SO_PATH):
            return False, f"not built: {SO_PATH} missing (run build.sh)"
        return True, ""
    except Exception as e:  # available() must never raise
        return False, f"{type(e).__name__}: {e}"


def load_lib():
    lib = ctypes.CDLL(SO_PATH)
    vp, ip, dp = ctypes.c_void_p, ctypes.c_int, ctypes.c_double
    lib.kbcg_prepare.argtypes = [ip, ip, vp, vp, vp, vp, ip, ip]
    lib.kbcg_prepare.restype = vp
    lib.kbcg_setup.argtypes = [vp]
    lib.kbcg_setup.restype = ip
    lib.kbcg_solve.argtypes = [vp, ip, dp, ip]
    lib.kbcg_solve.restype = ip
    lib.kbcg_copy_x.argtypes = [vp, vp]
    lib.kbcg_copy_x.restype = ip
    lib.kbcg_iters.argtypes = [vp]
    lib.kbcg_iters.restype = ip
    lib.kbcg_relres_recurrence.argtypes = [vp]
    lib.kbcg_relres_recurrence.restype = dp
    lib.kbcg_setup_ms.argtypes = [vp]
    lib.kbcg_setup_ms.restype = ctypes.c_float
    lib.kbcg_solve_ms.argtypes = [vp]
    lib.kbcg_solve_ms.restype = ctypes.c_float
    lib.kbcg_free.argtypes = [vp]
    lib.kbcg_free.restype = None
    lib.kbcg_last_error.restype = ctypes.c_char_p
    return lib


class CusparseCG:
    name = IMPL_NAME
    platform = "cuda"
    PRECOND = 0            # 0 none, 1 ILU(0)
    MODE = "fixed-iter"    # or "to-convergence"
    DEFAULT_MAXITER = 50   # spec: k=50 CG iterations per trial

    def __init__(self, precision: str = "fp64"):
        if precision not in _PREC:
            raise NotImplementedError(
                f"{self.name} is built for {sorted(_PREC)}; requested {precision!r}")
        self.precision = precision
        self.lib = load_lib()
        self._params = None
        self._A64 = None
        self._b64 = None
        self._n = 0

    def _error(self) -> str:
        return (self.lib.kbcg_last_error() or b"").decode(errors="replace")

    def prepare(self, matrix, params: dict):
        csr = matrix.csr.tocsr().astype(np.float64)
        csr.sum_duplicates()
        csr.sort_indices()
        n = csr.shape[0]
        if csr.shape[0] != csr.shape[1]:
            raise NotImplementedError(f"{self.name}: matrix is not square {csr.shape}")
        if np.any(csr.diagonal() == 0):
            # ILU(0) and the legacy samples both need every diagonal stored
            raise NotImplementedError(f"{self.name}: matrix has zero/missing diagonal entries")
        from kernelbench.domains.solvers import cg_rhs, cg_steps
        b64 = cg_rhs(csr, params)   # same b as every cg-krylov impl (KB_CG_RHS)
        self._A64, self._b64, self._n = csr, b64, n

        rowptr = np.ascontiguousarray(csr.indptr, dtype=np.int32)
        colind = np.ascontiguousarray(csr.indices, dtype=np.int32)
        val = np.ascontiguousarray(csr.data, dtype=np.float64)
        code, _ = _PREC[self.precision]
        handle = self.lib.kbcg_prepare(n, int(csr.nnz), rowptr.ctypes.data, colind.ctypes.data,
                                       val.ctypes.data, b64.ctypes.data, code, self.PRECOND)
        if not handle:
            raise RuntimeError(f"{self.name} prepare: {self._error()}")

        params["mode"] = self.MODE
        if self.MODE == "fixed-iter":
            params["maxiter"] = cg_steps(params, self.DEFAULT_MAXITER)   # KB_CG_STEPS
            params["rtol"] = 0.0
        else:
            params["maxiter"] = int(params.get("maxiter", 2000))
            params["rtol"] = float(params.get("rtol", 1e-6))
        params["precond"] = "ilu0" if self.PRECOND else "none"
        params["timing_scope"] = ("in-process CUDA events; solve only" if not self.PRECOND else
                                  "in-process CUDA events; ILU(0) setup + solve")
        self._params = params
        return {"handle": handle}

    def run(self, h):
        p = self._params
        if self.PRECOND and self.lib.kbcg_setup(h["handle"]):
            raise RuntimeError(f"{self.name} setup: {self._error()}")
        fixed = 1 if self.MODE == "fixed-iter" else 0
        if self.lib.kbcg_solve(h["handle"], p["maxiter"], p["rtol"], fixed):
            raise RuntimeError(f"{self.name} solve: {self._error()}")
        # last call's numbers; the runner stores params after the measured reps
        p["iterations_actual"] = int(self.lib.kbcg_iters(h["handle"]))
        p["cusparse_solve_ms_internal"] = float(self.lib.kbcg_solve_ms(h["handle"]))
        p["cusparse_setup_ms_internal"] = float(self.lib.kbcg_setup_ms(h["handle"]))
        p["relres_recurrence"] = float(self.lib.kbcg_relres_recurrence(h["handle"]))
        return h

    def to_host(self, out):
        x = np.empty(self._n, dtype=np.float64)
        if self.lib.kbcg_copy_x(out["handle"], x.ctypes.data):
            raise RuntimeError(f"{self.name} copy_x: {self._error()}")
        b_norm = max(float(np.linalg.norm(self._b64)), 1e-300)
        relres = float(np.linalg.norm(self._b64 - self._A64 @ x) / b_norm)
        finite = bool(np.isfinite(relres))
        p = self._params
        p["relres_achieved"] = relres
        if self.MODE == "fixed-iter":
            passed = finite and relres < 1.0
            p["gate"] = "finite-and-decreased (fixed-iter; full convergence not required)"
        else:
            gate_tol = max(10.0 * p["rtol"], 1e-9)
            passed = finite and relres <= gate_tol
            p["gate"] = f"relres<={gate_tol:.3e} (to-convergence, 10x slack over rtol)"
            p["converged"] = bool(p.get("relres_recurrence", np.inf) < p["rtol"])
        p["gate_passed"] = bool(passed)
        return np.array([1.0 if passed else 0.0])

    def timer(self):
        from kernelbench.impls.gpu_cuda import CudaEventTimer
        return CudaEventTimer()

    def free(self, h):
        if h.get("handle"):
            self.lib.kbcg_free(h["handle"])
        h.clear()


def create(precision: str):
    return CusparseCG(precision)
