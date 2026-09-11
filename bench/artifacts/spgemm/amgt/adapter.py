"""
AmgT adapter for the spgemm track.

Paper: "AmgT: Algebraic Multigrid Solver on Tensor Cores", SC'24.
`PAPER_KEY = conf/sc/LuZWFLCY0C024` (already verified against
`../../output/included.json`'s `artifact_url` field per this integration's
task brief -- not re-derived here).
Artifact: https://github.com/SuperScientificSoftwareLaboratory/AmgT -- a fork
of HYPRE with AmgT's own SpGEMM/SpMV kernels patched into
`AmgT_HYPRE/src/seq_mv/`.

Kernel entry point wrapped (ARTIFACT_GUIDE.md rule 1 -- wrap the kernel, not
the paper's benchmark script): `spgemm_amgT_fp64` (source/AmgT_HYPRE/src/
seq_mv/csr_spgemm_device.c:1527), a standalone C function computing
`*C_ptr = A @ B` for two arbitrary `hypre_CSRMatrix*` objects via AmgT's own
mBSR tensor-core-friendly block format. This is genuinely separable from
AmgT's AMG solver: no AMG setup/solve runs anywhere in this adapter, only
this one SpGEMM kernel call on two hand-built device `hypre_CSRMatrix`
objects -- confirmed NOT the same thing as `AmgT_test/test_new.c`'s own
end-to-end AMG-solve benchmark driver, which this adapter does not use or
link (see wrapper.cu's header comment for the exact reasoning and where the
function lives relative to the AMG hierarchy).

prepare() (ARTIFACT_GUIDE.md rule 2, the artifact's own format conversion,
timed as preprocessing): our workload's CSR (`indptr`/`indices`/`data`) is
H2D-copied into a device `hypre_CSRMatrix`, then converted ONCE via AmgT's
own `CSR2BSR_GPU` (source/AmgT_HYPRE/src/seq_mv/csr_matvec_device.c:1198),
which is idempotent -- guarded internally by `if (!hypre_BSRTAG(A))`
(confirmed by reading the function body directly). `run()` then calls
`spgemm_amgT_fp64(A, A, &C)` (see aliasing note below); that function's OWN
internal `CSR2BSR_GPU(A); CSR2BSR_GPU(B);` calls become free no-ops because
the tag is already set, so the harness's per-call timed window covers only
symbolic pass + `numeric_spgemm_hybrid` kernel + C's device allocation +
BSR2CSR-back-to-CSR -- matching `spgemm-square-kernel-f64`'s `timing_scope`
(benchspecs/spgemm/spec.yaml: format conversion excluded and reported once;
symbolic/numeric/allocation included).

Aliasing A == B for the self-product C = A @ A (this track's `operation`
field): this adapter builds ONE device `hypre_CSRMatrix` and passes it as
both operands. Verified safe by reading `spgemm_amgT_fp64`'s body directly
(see wrapper.cu's header comment for the full trace): `dmatA`/`dmatB` are
local by-value copies of the bsrMAT struct taken once at entry, never
written back into `*A`/`*B` -- only C's own freshly-allocated buffers are
written, so there is no read/write aliasing hazard.

PRECISIONS = ["fp64"] -- `spgemm_amgT_fp64` is AmgT's fp64 kernel path,
selected at BUILD time via the artifact's own documented `seq_mv.h` header
swap (see build.sh / STATUS.md: `cp config_files/AmgT_FP64.h
AmgT_HYPRE/src/seq_mv/seq_mv.h`, matching the artifact's own README:
"Select the compilation version by changing the execuative in compile.sh").
The mixed-precision path (`spgemm_amgT_fp32`/`spgemm_amgT_fp16`,
`AmgT_Mixed.h`) is out of scope for this task.

Correctness gate: `to_host()` D2H-copies C's CSR arrays, builds a
`scipy.sparse.csr_matrix`, and reindexes onto the canonical `|A|@|A|`
pattern via `kernelbench.impls.cpu_ref._canonical_pattern` /
`_reindex_to_pattern` -- the IDENTICAL convention
`kernelbench.impls.gpu_cuda.TorchSpGEMM.to_host()` uses for the cuSPARSE
baseline, just applied to raw D2H-copied arrays instead of a torch CSR
tensor.

Per-call leak note: `spgemm_amgT_fp64` allocates a FRESH `C` via
`hypre_CSRMatrixCreate` on every call (the artifact's own allocation
strategy -- hoisting it out would mean patching kernel code, which
ARTIFACT_GUIDE.md rule 3 forbids). `harness.run_variant()` calls
`impl.run()` repeatedly (1 gate check + warmup + measured reps) without
freeing `out` in between, so this adapter frees the PREVIOUS iteration's C
at the START of the next `run()` call (`self._prev_c`); `free()` cleans up
whatever is left (the last C, plus A/B) at the end. `hypre_CSRMatrixDestroy`
(the artifact's own, unmodified) is used for every free -- see wrapper.cu's
header comment for why it is sufficient on its own (it already frees the
attached mBSR side-structure via its own `if (hypre_BSRTAG(matrix)==1)`
branch, csr_matrix.c).

Timer/stream note: every AmgT kernel launch in the two files above omits an
explicit CUDA stream argument (confirmed by grep: no `cudaStreamCreate` /
`cudaStream_t` anywhere in either file's device code), i.e. everything runs
on the default stream -- the same stream torch's CudaEventTimer records its
start/stop events on, so device-event timing across `run()` is valid with no
cross-stream gap.
"""

from __future__ import annotations

import ctypes
import os

import numpy as np

KERNEL = "spgemm"
IMPL_NAME = "amgt-spgemm-mbsr"
PAPER_KEY = "conf/sc/LuZWFLCY0C024"
PRECISIONS = ["fp64"]

_HERE = os.path.dirname(os.path.abspath(__file__))
_LIB_PATH = os.path.join(_HERE, "libamgt_wrapper.so")

_lib = None
_inited = False


def _load_lib():
    global _lib, _inited
    if _lib is not None:
        return _lib
    if not os.path.exists(_LIB_PATH):
        raise FileNotFoundError(f"{_LIB_PATH} not built -- run build.sh")
    lib = ctypes.CDLL(_LIB_PATH)
    p = ctypes.c_void_p
    pi = ctypes.POINTER(ctypes.c_int)
    pd = ctypes.POINTER(ctypes.c_double)

    lib.amgt_init.argtypes = []
    lib.amgt_init.restype = ctypes.c_int

    lib.amgt_build_and_convert.argtypes = [
        ctypes.c_int, ctypes.c_int, ctypes.c_int, pi, pi, pd]
    lib.amgt_build_and_convert.restype = p

    lib.amgt_spgemm_run.argtypes = [p, p]
    lib.amgt_spgemm_run.restype = p

    lib.amgt_csr_sizes.argtypes = [p, pi, pi, pi]
    lib.amgt_csr_sizes.restype = None

    lib.amgt_csr_copy_out.argtypes = [p, pi, pi, pd]
    lib.amgt_csr_copy_out.restype = None

    lib.amgt_csr_destroy.argtypes = [p]
    lib.amgt_csr_destroy.restype = None

    _lib = lib
    return _lib


def _ensure_init():
    global _inited
    if _inited:
        return
    # Cray-MPICH quirk (this machine, not an AmgT bug): the login node's
    # module environment sets MPICH_GPU_SUPPORT_ENABLED=1 unconditionally,
    # which makes MPI_Init() abort with "GPU_SUPPORT_ENABLED is requested,
    # but GTL library is not linked" unless the GPU-transport-layer lib
    # (-lmpi_gtl_cuda, a separate Cray package) is also linked. This
    # adapter's MPI usage is a single-process "singleton" HYPRE_Init() --
    # no GPU-direct MPI transfer ever happens -- so disabling GPU-aware MPI
    # support for this one call is safe and avoids the extra link
    # dependency. Set here (not via build.sh's link flags) so the adapter
    # works regardless of the caller's shell environment.
    os.environ["MPICH_GPU_SUPPORT_ENABLED"] = "0"  # force, not setdefault --
    # the module environment sets it to "1" unconditionally, so setdefault
    # would be a no-op.
    lib = _load_lib()
    rc = lib.amgt_init()
    if rc != 0:
        raise RuntimeError(f"amgt_init() failed, rc={rc}")
    _inited = True


def available() -> tuple[bool, str]:
    if not os.path.exists(_LIB_PATH):
        return False, "libamgt_wrapper.so not built -- run build.sh"
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
    return AmgtSpGEMM(precision)


class AmgtSpGEMM:
    name = IMPL_NAME
    platform = "cuda"

    def __init__(self, precision: str = "fp64"):
        if precision != "fp64":
            raise NotImplementedError(
                f"{IMPL_NAME} only wired for fp64 -- spgemm_amgT_fp64 is "
                f"AmgT's fp64 kernel path selected at build time via the "
                f"seq_mv.h header swap (see STATUS.md); the mixed-precision "
                f"path (AmgT_Mixed.h) is out of scope. requested {precision}")
        self.precision = precision
        self._matrix = None
        self._prev_c = None

    def prepare(self, matrix, params: dict):
        _ensure_init()
        lib = _load_lib()

        A = matrix.csr.astype(np.float64)
        A.sort_indices()
        nrows, ncols = A.shape
        nnz = A.nnz

        rowptr = np.ascontiguousarray(A.indptr, dtype=np.int32)
        colidx = np.ascontiguousarray(A.indices, dtype=np.int32)
        vals = np.ascontiguousarray(A.data, dtype=np.float64)

        # --- artifact's own format conversion (CSR -> mBSR), timed once as
        # preprocessing by the harness's wall-clock wrapper around
        # impl.prepare() -- see wrapper.cu's amgt_build_and_convert() header
        # comment for the exact call sequence (H2D copy + CSR2BSR_GPU).
        A_dev = lib.amgt_build_and_convert(
            ctypes.c_int(nrows), ctypes.c_int(ncols), ctypes.c_int(nnz),
            rowptr.ctypes.data_as(ctypes.POINTER(ctypes.c_int)),
            colidx.ctypes.data_as(ctypes.POINTER(ctypes.c_int)),
            vals.ctypes.data_as(ctypes.POINTER(ctypes.c_double)))
        if not A_dev:
            raise RuntimeError("amgt_build_and_convert returned NULL")

        self._matrix = matrix
        self._prev_c = None
        return {"A": A_dev}

    def run(self, h):
        lib = _load_lib()
        # free the PREVIOUS call's C now (safe: the harness has already
        # consumed it via to_host() if it was the gate-check call, or never
        # touched it if it was a prior warmup/measured rep -- see this
        # module's docstring "Per-call leak note").
        if self._prev_c:
            lib.amgt_csr_destroy(self._prev_c)
            self._prev_c = None

        # ONE SpGEMM kernel call: C = A @ A (A aliased for both operands,
        # see docstring's aliasing note).
        C = lib.amgt_spgemm_run(h["A"], h["A"])
        if not C:
            raise RuntimeError(
                "amgt_spgemm_run failed (see stderr for the CUDA error "
                "wrapper.cu's amgt_spgemm_run prints)")
        self._prev_c = C
        return C

    def to_host(self, out):
        lib = _load_lib()
        nrows_c, ncols_c, nnz_c = ctypes.c_int(), ctypes.c_int(), ctypes.c_int()
        lib.amgt_csr_sizes(out, ctypes.byref(nrows_c), ctypes.byref(ncols_c),
                           ctypes.byref(nnz_c))
        nrows, ncols, nnz = nrows_c.value, ncols_c.value, nnz_c.value

        rowptr = np.empty(nrows + 1, dtype=np.int32)
        colidx = np.empty(max(nnz, 1), dtype=np.int32)
        vals = np.empty(max(nnz, 1), dtype=np.float64)
        lib.amgt_csr_copy_out(
            out,
            rowptr.ctypes.data_as(ctypes.POINTER(ctypes.c_int)),
            colidx.ctypes.data_as(ctypes.POINTER(ctypes.c_int)),
            vals.ctypes.data_as(ctypes.POINTER(ctypes.c_double)))
        colidx = colidx[:nnz]
        vals = vals[:nnz]

        import scipy.sparse as sp
        from kernelbench.impls.cpu_ref import _canonical_pattern, _reindex_to_pattern
        C = sp.csr_matrix((vals, colidx, rowptr), shape=(nrows, ncols))
        A64 = self._matrix.csr.astype(np.float64)
        A64.sort_indices()
        pattern = _canonical_pattern(A64)
        return _reindex_to_pattern(C, pattern)

    def timer(self):
        from kernelbench.impls.gpu_cuda import CudaEventTimer
        return CudaEventTimer()

    def free(self, h):
        lib = _load_lib()
        if self._prev_c:
            lib.amgt_csr_destroy(self._prev_c)
            self._prev_c = None
        a = h.get("A") if isinstance(h, dict) else None
        if a:
            lib.amgt_csr_destroy(a)
        if isinstance(h, dict):
            h.clear()
