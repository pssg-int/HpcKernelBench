"""
AmgT (HYPRE BoomerAMG) adapter for the multigrid track's
`mg-gpu-solve-kernel-fixed-iter` variant.

Paper: "AmgT: Algebraic Multigrid Solver on Tensor Cores", SC'24.
PAPER_KEY = "conf/sc/LuZWFLCY0C024" (same paper as ../../spgemm/amgt/,
already verified there).
Artifact: https://github.com/SuperScientificSoftwareLaboratory/AmgT -- a
fork of HYPRE. This directory does NOT re-clone it: `source/` here is a
SYMLINK to ../../spgemm/amgt/source (see build.sh/STATUS.md), and this
adapter wraps a DIFFERENT part of the SAME already-built libHYPRE.so than
the spgemm track's adapter does -- HYPRE's own public BoomerAMG solver API
(HYPRE_BoomerAMGSetup/Solve), not AmgT's bespoke spgemm_amgT_fp64 kernel.
No rebuild of HYPRE was needed or performed (verified: `nm -D libHYPRE.so`
already exports every BoomerAMG/IJ symbol this wrapper calls -- see
build.sh's own symbol-presence check).

Kernel entry point wrapped (ARTIFACT_GUIDE.md rule 1): HYPRE's own
HYPRE_BoomerAMGSetup / HYPRE_BoomerAMGSolve, called through mg_wrapper.cu's
extern-"C" glue (this directory, NEW code, touches zero lines of source/).
This is HYPRE's PUBLIC API, used exactly as AmgT_test/test_new.c itself
uses it (test_new.c:392-411: BoomerAMGCreate+configure, then
HYPRE_BoomerAMGSetup(precond, A, b, x) and HYPRE_BoomerAMGSolve(precond, A,
b, x) called DIRECTLY as a standalone solver -- NOT nested inside a PCG
loop, confirmed by reading test_new.c directly: it creates a
HYPRE_ParCSRPCGCreate solver object but never calls
HYPRE_ParCSRPCGSetPrecond/Setup/Solve on it -- dead code in the artifact's
own driver). This adapter reproduces that same standalone-solver usage.

Configuration (mg_wrapper.cu's amgmg_boomeramg_create): copies test_new.c's
own PCG-preconditioner block VERBATIM (PMIS coarsening, strong threshold
0.25, max 7 levels, V-cycle, 3/3 pre-/post-sweeps, l1-Jacobi relaxation,
extended+i interpolation) EXCEPT NumFunctions=1 instead of AmgT's hardcoded
3 -- this wrapper's inputs (poisson-3d, suitesparse-real SPD structural
matrices) are scalar problems, and NumFunctions=3 would misinterpret which
unknowns belong to the same physical node (see mg_wrapper.cu's header
comment for the full reasoning). This is the ONLY configuration value not
copied verbatim from AmgT's own code.

THE KNOWN TRAP (task brief, spec.yaml notes_on_fairness): AmgT's own driver
sets `HYPRE_BoomerAMGSetTol(precond, 1e-20)` so every timed run always
executes the full fixed iteration count regardless of convergence. This
adapter does the SAME for THIS variant deliberately (max_iter=50, tol=1e-20
-- passed as explicit arguments to amgmg_boomeramg_create, not hardcoded in
the .cu file, see its docstring) because that is exactly what
mg-gpu-solve-kernel-fixed-iter's own protocol asks for: a fixed amount of
work, comparable across implementations, independent of convergence
quality. This is NOT the e2e-pcg variant's honest-convergence protocol
(1e-6 tol, hard fail if not reached) -- that variant is NOT wired by this
adapter (see STATUS.md "Not done").

Setup/solve split (ARTIFACT_GUIDE.md rule 2, matching
solvers.AmgSolveFixedIterCPU's own discipline exactly so the two are
apples-to-apples comparable): prepare() builds the IJMatrix/ParCSR view of
the workload's CSR (device Unified-Memory staging, mirroring
AmgT_test/test_new.c's own construction sequence -- see mg_wrapper.cu),
builds the RHS vector b (same U(-1,1) seed=42 convention as the CPU
reference), creates+configures the BoomerAMG solver, and calls
HYPRE_BoomerAMGSetup ONCE -- all timed together as ONE preprocessing_ms by
the harness, matching the variant's timing_scope ("solve kernel only,
hierarchy already resident"). run() zeroes x (fresh x0=0, matching the CPU
class's "fresh x0=0 EVERY call" discipline) then calls
HYPRE_BoomerAMGSolve -- the ONLY thing inside the timed per-rep window.

Correctness gate: to_host() D2H-copies x and independently recomputes
relres = ||b-Ax||/||b|| in fp64 using scipy (never HYPRE's own internal
`HYPRE_BoomerAMGGetFinalRelativeResidualNorm`, which is recorded into
params for audit only) -- the SAME finite-x + finite-independently-
recomputed-residual gate as solvers.AmgSolveFixedIterCPU.to_host(), so a
pass/fail here is directly comparable to the CPU reference's own gate.
CORRECTNESS_MODE is "exact" (see kernelbench.domains.solvers module
docstring's "Correctness philosophy" -- the harness's REFERENCES["multigrid"]
sentinel-array pattern is reused here unchanged, imported directly from
that module rather than re-implemented, since it genuinely is the same
formula/convention, not a competing one).
"""

from __future__ import annotations

import ctypes
import os

import numpy as np

KERNEL = "multigrid"
IMPL_NAME = "amgt-boomeramg-fixed-iter"
PAPER_KEY = "conf/sc/LuZWFLCY0C024"
PRECISIONS = ["fp64"]

_HERE = os.path.dirname(os.path.abspath(__file__))
_LIB_PATH = os.path.join(_HERE, "libamgtmg_wrapper.so")

_lib = None
_inited = False


def _load_lib():
    global _lib
    if _lib is not None:
        return _lib
    if not os.path.exists(_LIB_PATH):
        raise FileNotFoundError(f"{_LIB_PATH} not built -- run build.sh")
    lib = ctypes.CDLL(_LIB_PATH)
    p = ctypes.c_void_p
    pi = ctypes.POINTER(ctypes.c_int)
    pd = ctypes.POINTER(ctypes.c_double)

    lib.amgmg_init.argtypes = []
    lib.amgmg_init.restype = ctypes.c_int

    lib.amgmg_build_matrix.argtypes = [ctypes.c_int, ctypes.c_int, pi, pi, pd, p]
    lib.amgmg_build_matrix.restype = p

    lib.amgmg_build_vector.argtypes = [ctypes.c_int, pd, p]
    lib.amgmg_build_vector.restype = p

    lib.amgmg_parvector_zero.argtypes = [p, ctypes.c_int]
    lib.amgmg_parvector_zero.restype = None

    lib.amgmg_parvector_read.argtypes = [p, ctypes.c_int, pd]
    lib.amgmg_parvector_read.restype = None

    lib.amgmg_boomeramg_create.argtypes = [ctypes.c_int, ctypes.c_double]
    lib.amgmg_boomeramg_create.restype = p

    lib.amgmg_boomeramg_setup.argtypes = [p, p, p, p]
    lib.amgmg_boomeramg_setup.restype = ctypes.c_int

    lib.amgmg_boomeramg_solve.argtypes = [p, p, p, p]
    lib.amgmg_boomeramg_solve.restype = ctypes.c_int

    lib.amgmg_get_num_iterations.argtypes = [p]
    lib.amgmg_get_num_iterations.restype = ctypes.c_int

    lib.amgmg_get_final_residual.argtypes = [p]
    lib.amgmg_get_final_residual.restype = ctypes.c_double

    lib.amgmg_boomeramg_destroy.argtypes = [p]
    lib.amgmg_boomeramg_destroy.restype = None

    lib.amgmg_matrix_destroy.argtypes = [p]
    lib.amgmg_matrix_destroy.restype = None

    lib.amgmg_vector_destroy.argtypes = [p]
    lib.amgmg_vector_destroy.restype = None

    _lib = lib
    return _lib


def _ensure_init():
    global _inited
    if _inited:
        return
    # Same Cray-MPICH GPU-transport-layer quirk as ../../spgemm/amgt/adapter.py
    # -- this adapter's MPI usage is also a single-process HYPRE_Init()
    # singleton with zero GPU-direct MPI transfers, so disabling
    # GPU-aware MPI support is safe and avoids an extra link dependency.
    os.environ["MPICH_GPU_SUPPORT_ENABLED"] = "0"
    lib = _load_lib()
    rc = lib.amgmg_init()
    if rc != 0:
        raise RuntimeError(f"amgmg_init() failed, rc={rc}")
    _inited = True


def available() -> tuple[bool, str]:
    if not os.path.exists(_LIB_PATH):
        return False, "libamgtmg_wrapper.so not built -- run build.sh"
    try:
        import torch
    except Exception as e:
        return False, f"torch import failed: {type(e).__name__}: {e}"
    if not torch.cuda.is_available():
        return False, "CUDA not available on this host"
    try:
        _load_lib()
    except Exception as e:
        return False, f"ctypes load failed: {type(e).__name__}: {e}"
    return True, ""


def create(precision: str):
    return AmgtBoomerAmgFixedIter(precision)


class AmgtBoomerAmgFixedIter:
    """See module docstring. One prepare() = one HYPRE_BoomerAMGSetup call
    (timed as preprocessing); one run() = zero x, then one
    HYPRE_BoomerAMGSolve call with maxiter=50, tol=1e-20 (AmgT's own fixed-
    iteration protocol, see module docstring's "THE KNOWN TRAP")."""

    name = IMPL_NAME
    platform = "cuda"

    def __init__(self, precision: str = "fp64"):
        if precision != "fp64":
            raise NotImplementedError(
                f"{IMPL_NAME} only wired for fp64 -- HYPRE's BoomerAMG in this "
                f"build runs fp64 throughout (HYPRE_Real=double, no mixed-"
                f"precision cascade wired here); requested {precision}")
        self.precision = precision
        self._A64 = None
        self._b64 = None
        self._A_ij = None
        self._A_par = None
        self._b_ij = None
        self._b_par = None
        self._x_ij = None
        self._x_par = None
        self._precond = None
        self._n = 0

    def prepare(self, matrix, params: dict):
        _ensure_init()
        lib = _load_lib()

        A = matrix.csr.astype(np.float64)
        A.sort_indices()
        n = A.shape[0]
        rng = np.random.default_rng(params.get("seed", 42))
        b64 = rng.uniform(-1.0, 1.0, size=n)

        params["mg_phase"] = "solve"
        params["maxiter"] = int(params.get("maxiter", 50))
        params["boomeramg_tol"] = float(params.get("boomeramg_tol", 1e-20))

        rowptr = np.ascontiguousarray(A.indptr, dtype=np.int32)
        colidx = np.ascontiguousarray(A.indices, dtype=np.int32)
        vals = np.ascontiguousarray(A.data, dtype=np.float64)

        parcsr_out = ctypes.c_void_p()
        A_ij = lib.amgmg_build_matrix(
            ctypes.c_int(n), ctypes.c_int(A.nnz),
            rowptr.ctypes.data_as(ctypes.POINTER(ctypes.c_int)),
            colidx.ctypes.data_as(ctypes.POINTER(ctypes.c_int)),
            vals.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
            ctypes.byref(parcsr_out))
        if not A_ij:
            raise RuntimeError("amgmg_build_matrix returned NULL")

        b_par_out = ctypes.c_void_p()
        b_ij = lib.amgmg_build_vector(
            ctypes.c_int(n), b64.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
            ctypes.byref(b_par_out))
        x_par_out = ctypes.c_void_p()
        x_ij = lib.amgmg_build_vector(ctypes.c_int(n), None, ctypes.byref(x_par_out))

        precond = lib.amgmg_boomeramg_create(
            ctypes.c_int(params["maxiter"]), ctypes.c_double(params["boomeramg_tol"]))
        if not precond:
            raise RuntimeError("amgmg_boomeramg_create returned NULL")

        # --- the actual setup call: timed ONCE by the harness as
        # preprocessing_ms, per mg-gpu-solve-kernel-fixed-iter's own
        # timing_scope (solve kernel only, hierarchy already resident).
        rc = lib.amgmg_boomeramg_setup(precond, parcsr_out, b_par_out, x_par_out)
        if rc != 0:
            raise RuntimeError(f"HYPRE_BoomerAMGSetup failed, rc={rc}")

        self._A64, self._b64, self._n = A, b64, n
        self._A_ij, self._A_par = A_ij, parcsr_out
        self._b_ij, self._b_par = b_ij, b_par_out
        self._x_ij, self._x_par = x_ij, x_par_out
        self._precond = precond
        self._params = params
        return {}

    def run(self, h):
        lib = _load_lib()
        # fresh x0=0 EVERY call -- matches solvers.AmgSolveFixedIterCPU's
        # own discipline (see module docstring).
        lib.amgmg_parvector_zero(self._x_par, ctypes.c_int(self._n))
        rc = lib.amgmg_boomeramg_solve(self._precond, self._A_par, self._b_par, self._x_par)
        if rc != 0:
            raise RuntimeError(f"HYPRE_BoomerAMGSolve failed, rc={rc}")

        p = self._params
        p["hypre_num_iterations"] = lib.amgmg_get_num_iterations(self._precond)
        p["hypre_final_residual_internal"] = lib.amgmg_get_final_residual(self._precond)
        return self._x_par

    def to_host(self, out):
        lib = _load_lib()
        x = np.empty(self._n, dtype=np.float64)
        lib.amgmg_parvector_read(out, ctypes.c_int(self._n),
                                 x.ctypes.data_as(ctypes.POINTER(ctypes.c_double)))

        b_norm = max(float(np.linalg.norm(self._b64)), 1e-300)
        relres = float(np.linalg.norm(self._b64 - self._A64 @ x) / b_norm)
        finite = bool(np.all(np.isfinite(x))) and bool(np.isfinite(relres))

        p = self._params
        p["relres_after_fixed_iters"] = relres
        p["gate"] = ("finite x + finite independently-recomputed residual "
                    "(fixed-iter; convergence not required)")
        p["gate_passed"] = bool(finite)
        return np.array([1.0 if finite else 0.0])

    def timer(self):
        from kernelbench.impls.gpu_cuda import CudaEventTimer
        return CudaEventTimer()

    def free(self, h):
        lib = _load_lib()
        if self._precond:
            lib.amgmg_boomeramg_destroy(self._precond)
            self._precond = None
        if self._x_ij:
            lib.amgmg_vector_destroy(self._x_ij)
            self._x_ij = None
        if self._b_ij:
            lib.amgmg_vector_destroy(self._b_ij)
            self._b_ij = None
        if self._A_ij:
            lib.amgmg_matrix_destroy(self._A_ij)
            self._A_ij = None
        if isinstance(h, dict):
            h.clear()
