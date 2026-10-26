"""
AmgX AMG-preconditioned CG -- the library BASELINE for BootCMatchGX (TPDS'23)
in the to-convergence group (`cg-e2e-ilu0-to-convergence`). BootCMatchGX's
paper claims "up to 2x vs AmgX" (it also compares against hypre).

What runs: NVIDIA AmgX v2.5.0 (source/, see source.provenance), single GPU,
no MPI, driven in-process through bridge.so. Config =
pcg_aggregation_jacobi.json, which is AmgX's own shipped
src/configs/PCG_AGGREGATION_JACOBI.json (PCG + aggregation AMG, pairwise
SIZE_2 selector, block-Jacobi smoother, V-cycle) with only printing turned
off and max_iters raised to the spec's 2000 cap. Its stop rule is already
the spec's: RELATIVE_INI L2 (||r_k|| / ||r_0|| < 1e-6, and r_0 = b since
x0 = 0). Aggregation AMG is the closest AmgX family to BootCMatchGX's own
matching-based aggregation; BootCMatchGX's paper does not say which AmgX
config it used, so NVIDIA's shipped one is used unchanged.

Protocol (mirrors the bootcmatchgx/spcg adapters' gate exactly):
  - b = U(-1,1), seed 42; x0 = 0. Matrix and b uploaded once in prepare().
  - run() = fresh AMG hierarchy setup + solve; CUDA-event timed in-process
    (no process start-up or file parsing). amgx_setup_ms / amgx_solve_ms
    record the split, like BootCMatchGX's own setup/solve timers.
  - Gate: relres = ||b - A x|| / ||b|| recomputed in fp64 from the returned
    x, pass iff finite and <= max(10*rtol, 1e-9).
fp64 is primary (BootCMatchGX is fp64-only); fp32 (AmgX dFFI mode) exists.
"""
from __future__ import annotations

import ctypes
import json
import os

import numpy as np

KERNEL = "cg-krylov"
IMPL_NAME = "amgx-pcg-amg"
PAPER_KEY = "journals/tpds/BernaschiCVD23"   # BootCMatchGX: "up to 2x vs AmgX"
ROLE = "baseline"
PRECISIONS = ["fp64", "fp32"]

_HERE = os.path.dirname(os.path.abspath(__file__))
_SO_PATH = os.path.join(_HERE, "bridge.so")
_CONFIG = os.path.join(_HERE, "pcg_aggregation_jacobi.json")
_PREC = {"fp64": 0, "fp32": 1}
_STATUS = {0: "success", 1: "failed", 2: "diverged", 3: "not_converged"}


def available() -> tuple[bool, str]:
    try:
        if not os.path.exists(_SO_PATH):
            return False, f"not built: {_SO_PATH} missing (run build.sh)"
        return True, ""
    except Exception as e:  # available() must never raise
        return False, f"{type(e).__name__}: {e}"


def _load_lib():
    lib = ctypes.CDLL(_SO_PATH)
    vp, ip, dp = ctypes.c_void_p, ctypes.c_int, ctypes.c_double
    lib.kbamgx_prepare.argtypes = [ip, ip, vp, vp, vp, vp, ip, ctypes.c_char_p]
    lib.kbamgx_prepare.restype = vp
    lib.kbamgx_run.argtypes = [vp]
    lib.kbamgx_run.restype = ip
    lib.kbamgx_copy_x.argtypes = [vp, vp]
    lib.kbamgx_copy_x.restype = ip
    for f in ("kbamgx_iters", "kbamgx_status"):
        getattr(lib, f).argtypes = [vp]
        getattr(lib, f).restype = ip
    for f in ("kbamgx_setup_ms", "kbamgx_solve_ms"):
        getattr(lib, f).argtypes = [vp]
        getattr(lib, f).restype = dp
    lib.kbamgx_free.argtypes = [vp]
    lib.kbamgx_free.restype = None
    lib.kbamgx_last_error.restype = ctypes.c_char_p
    return lib


class AmgxPcgAmg:
    name = IMPL_NAME
    platform = "cuda"

    def __init__(self, precision: str = "fp64"):
        if precision not in _PREC:
            raise NotImplementedError(
                f"{IMPL_NAME} is built for {sorted(_PREC)}; requested {precision!r}")
        self.precision = precision
        self.lib = _load_lib()
        self._params = None
        self._A64 = None
        self._b64 = None
        self._n = 0

    def _error(self) -> str:
        return (self.lib.kbamgx_last_error() or b"").decode(errors="replace")

    def prepare(self, matrix, params: dict):
        csr = matrix.csr.tocsr().astype(np.float64)
        csr.sum_duplicates()
        csr.sort_indices()
        n = csr.shape[0]
        if csr.shape[0] != csr.shape[1]:
            raise NotImplementedError(f"{IMPL_NAME}: matrix is not square {csr.shape}")
        from kernelbench.domains.solvers import cg_rhs
        b64 = cg_rhs(csr, params)   # same b as every cg-krylov impl (KB_CG_RHS)
        self._A64, self._b64, self._n = csr, b64, n

        params["mode"] = "to-convergence"
        params["maxiter"] = int(params.get("maxiter", 2000))
        params["rtol"] = float(params.get("rtol", 1e-6))
        with open(_CONFIG) as f:
            cfg = json.load(f)
        cfg["solver"]["max_iters"] = params["maxiter"]
        cfg["solver"]["tolerance"] = params["rtol"]
        params["amgx_config"] = os.path.basename(_CONFIG)
        params["timing_scope"] = "in-process CUDA events; AMG setup + solve"

        rowptr = np.ascontiguousarray(csr.indptr, dtype=np.int32)
        colind = np.ascontiguousarray(csr.indices, dtype=np.int32)
        val = np.ascontiguousarray(csr.data, dtype=np.float64)
        handle = self.lib.kbamgx_prepare(n, int(csr.nnz), rowptr.ctypes.data, colind.ctypes.data,
                                         val.ctypes.data, b64.ctypes.data, _PREC[self.precision],
                                         json.dumps(cfg).encode())
        if not handle:
            raise RuntimeError(f"{IMPL_NAME} prepare: {self._error()}")
        self._params = params
        return {"handle": handle}

    def run(self, h):
        if self.lib.kbamgx_run(h["handle"]):
            raise RuntimeError(f"{IMPL_NAME} run: {self._error()}")
        p = self._params
        p["iterations_actual"] = int(self.lib.kbamgx_iters(h["handle"]))
        status = int(self.lib.kbamgx_status(h["handle"]))
        p["amgx_status"] = _STATUS.get(status, str(status))
        p["converged"] = status == 0
        p["amgx_setup_ms"] = float(self.lib.kbamgx_setup_ms(h["handle"]))
        p["amgx_solve_ms"] = float(self.lib.kbamgx_solve_ms(h["handle"]))
        return h

    def to_host(self, out):
        x = np.empty(self._n, dtype=np.float64)
        if self.lib.kbamgx_copy_x(out["handle"], x.ctypes.data):
            raise RuntimeError(f"{IMPL_NAME} copy_x: {self._error()}")
        b_norm = max(float(np.linalg.norm(self._b64)), 1e-300)
        relres = float(np.linalg.norm(self._b64 - self._A64 @ x) / b_norm)
        finite = bool(np.isfinite(relres))
        p = self._params
        gate_tol = max(10.0 * p["rtol"], 1e-9)
        passed = finite and relres <= gate_tol
        p["relres_achieved"] = relres
        p["gate"] = f"relres<={gate_tol:.3e} (to-convergence, 10x slack over rtol)"
        p["gate_passed"] = bool(passed)
        return np.array([1.0 if passed else 0.0])

    def timer(self):
        from kernelbench.impls.gpu_cuda import CudaEventTimer
        return CudaEventTimer()

    def free(self, h):
        if h.get("handle"):
            self.lib.kbamgx_free(h["handle"])
        h.clear()


def create(precision: str):
    return AmgxPcgAmg(precision)
