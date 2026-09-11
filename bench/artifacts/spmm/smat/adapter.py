"""
SMaT adapter for the spmm track.

Paper: "SMaT: (S)parse (Ma)trix Matrix (T)ensor Core-accelerated library",
SC'24. `PAPER_KEY = conf/sc/OkanovicKLBVH24`. Artifact:
https://github.com/spcl/smat, cloned `--depth 1` at `057e44a` (see
`source.provenance`).

SMaT converts an unstructured sparse matrix into a block-CSR (BCSR) of
16x16 (`MMA_M x MMA_K`) tiles -- storing every tile that has >=1 nonzero
in full dense form, skipping all-zero tiles -- then runs a raw
`mma.sync`-PTX (`ldmatrix`+`mma.sync.aligned.m16n8k16`) Tensor-Core kernel
over that block format (`source/src/cuda_hgemm/src/mma/mmaCBT.cu`'s
`mmaCBTKernelSparse`, launched by `mmaCBTKernel` -- the ONLY sparse kernel
`main.cu`'s benchmark actually calls; sibling `mmaB`/`mmaBT`/`mmaNaive`
kernels are commented out there).

Kernel wrapped (ARTIFACT_GUIDE.md rule 1): `mmaCBTKernel`/
`mmaCBTKernelSparse`, compiled completely unmodified into
`libsmat_wrapper.so` together with this directory's `wrapper.cu`. See
`wrapper.cu`'s header comment and `build.sh` for why the gflags/CMake build
ARTIFACT_GUIDE.md anticipated needing a vendored gflags turned out to be
avoidable: `SparseMatrix` (SMaT's own preprocessing class,
`source/.../common/matrix.h`) is header-only with zero dependency on
gflags/OpenMP/cublas -- only `main.cu`'s CLI-flag parsing needs those, and
this adapter never links `main.cu` at all.

Preprocessing (rule 2, timed in `prepare()`): `SparseMatrix`'s constructor
does SMaT's real preprocessing -- reads the `.mtx` file, builds CSR on the
host, converts CSR -> BCSR (block reordering / tiling, the paper's own
contribution) and uploads everything to the device. `wrapper.cu::smat_prepare`
calls this constructor completely unmodified.

Dense operand B: numpy `default_rng` matching `cpu_ref.reference_spmm`'s
`_dense_operand` exactly (see rode/insum/mp-spmm adapters' docstrings for
why this matters), cast to `half` -- SMaT's kernel is fp16-in/fp16-out
Tensor-Core MMA, no fp32 path.

## IMPLEMENTATION FINDING: large correctness-gate errors on real, large-
## magnitude matrices -- reproduced independently from the gate machinery
## (see STATUS.md for the full investigation)

`mmaCBTKernelSparse`'s accumulator (`uint32_t RC[2]`, fed to
`HMMA16816`/`mma.sync.aligned.m16n8k16`) accumulates entirely in `half`
(fp16), never promoted to fp32 -- confirmed by reading the kernel body
(`mmaCBT.cu`): no `float`-typed accumulator exists anywhere in it. On a
real SuiteSparse matrix with large-magnitude entries (`cant`: values up to
~881), a dot product over ~64 nonzeros per row routinely lands output
magnitudes in the low thousands, well inside fp16's *representable* range
(max ~65504) but past the point where fp16's 10-bit mantissa preserves the
several-hundred-term running sum precisely -- `max_scaled_err` was measured
at 1.8e4 on `cant` (`--dims 128`), far beyond ordinary fp16 unit-roundoff
(~1e-3). This is a genuine numerical property of an all-`half`-accumulate
Tensor-Core kernel applied to large-magnitude real data, not a wrapper bug
(cross-checked against the synthetic smoke set's small `U(-1,1)` values,
where the same kernel is much closer to the fp16-roundoff floor -- see
STATUS.md for the actual numbers). Gate reported as a real FAIL per
ARTIFACT_GUIDE.md rule 4; not worked around.
"""

from __future__ import annotations

import ctypes
import os

import numpy as np

KERNEL = "spmm"
IMPL_NAME = "smat-mmacbt"
PAPER_KEY = "conf/sc/OkanovicKLBVH24"
PRECISIONS = ["fp16"]

_HERE = os.path.dirname(os.path.abspath(__file__))
_LIB_PATH = os.path.join(_HERE, "libsmat_wrapper.so")
_WORK = os.path.join(_HERE, "work")

_lib = None


def _load_lib():
    global _lib
    if _lib is not None:
        return _lib
    if not os.path.exists(_LIB_PATH):
        raise FileNotFoundError(f"{_LIB_PATH} not built -- run build.sh")
    lib = ctypes.CDLL(_LIB_PATH)
    p = ctypes.c_void_p
    ul = ctypes.c_ulong
    lib.smat_prepare.argtypes = [ctypes.c_char_p]
    lib.smat_prepare.restype = p
    lib.smat_get_row.argtypes = [p]
    lib.smat_get_row.restype = ul
    lib.smat_get_col.argtypes = [p]
    lib.smat_get_col.restype = ul
    lib.smat_get_nnz.argtypes = [p]
    lib.smat_get_nnz.restype = ul
    lib.smat_get_nonzero_blocks.argtypes = [p]
    lib.smat_get_nonzero_blocks.restype = ul
    lib.smat_run.argtypes = [p, p, p, ul]
    lib.smat_run.restype = None
    lib.smat_free.argtypes = [p]
    lib.smat_free.restype = None
    _lib = lib
    return _lib


def available() -> tuple[bool, str]:
    if not os.path.exists(_LIB_PATH):
        return False, "libsmat_wrapper.so not built -- run build.sh"
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
    return SMaTSpMM(precision)


class SMaTSpMM:
    name = IMPL_NAME
    platform = "cuda"

    def __init__(self, precision: str = "fp16"):
        if precision != "fp16":
            raise NotImplementedError(
                f"{IMPL_NAME}: SMaT's mmaCBTKernel is fp16-in/fp16-out "
                f"Tensor-Core MMA only (see module docstring); requested "
                f"{precision}")
        self.precision = precision
        os.makedirs(_WORK, exist_ok=True)

    def prepare(self, matrix, params: dict):
        import torch
        from scipy.io import mmwrite

        lib = _load_lib()
        N = int(params["N"])

        # --- write our workload as a standard MatrixMarket file (SMaT's own
        #     mmio_highlevel.h reader's input format) ---
        mtx_path = os.path.join(_WORK, f"{matrix.name}.mtx")
        mmwrite(mtx_path, matrix.csr.astype(np.float64), field="real", symmetry="general")

        # --- SMaT's own preprocessing: SparseMatrix's constructor reads the
        #     .mtx file, builds CSR, converts to BCSR, uploads to device
        #     (timed as preprocessing, ARTIFACT_GUIDE.md rule 2) ---
        handle = lib.smat_prepare(mtx_path.encode())
        if not handle:
            raise RuntimeError(f"{IMPL_NAME}: smat_prepare returned NULL for {mtx_path}")

        M_padded = int(lib.smat_get_row(handle))
        K_padded = int(lib.smat_get_col(handle))
        M, K = matrix.csr.shape
        # BCSR pads M/K up to a multiple of MMA_M=MMA_K=16; padded rows/cols
        # are structurally zero (no nonzero block ever created for them, see
        # matrix.h::csrToBcsr), so the extra region contributes nothing --
        # slice it back off below rather than exposing padded output.

        rng = np.random.default_rng(params.get("seed", 42))
        B_np = rng.uniform(-1.0, 1.0, size=(K_padded, N)).astype(np.float32)
        B_np[K:, :] = 0.0  # padding columns of A are always-zero blocks; keep B's padding rows inert too
        # mmaCBTKernelSparse indexes B as `B[i*MMA_K + warp_col*K]` (see
        # mmaCBT.cu) -- i.e. column-major over the logical (K, N) operand
        # (equivalently, row-major (N, K), B^T). Confirmed empirically while
        # building this adapter: feeding B row-major (K, N) -- the harness's
        # own convention, and what cpu_ref.reference_spmm's reference is
        # computed against -- produced max_scaled_err ~2-3 even on tiny,
        # well-conditioned synthetic smoke matrices (nowhere near fp16
        # roundoff), which stopped the moment B was transposed to match this
        # kernel's actual expected layout (see STATUS.md). The GATE
        # reference still uses `B_np` (K, N) unchanged; only the buffer
        # physically handed to the kernel is the transposed, made-
        # contiguous copy.
        B_kernel_np = np.ascontiguousarray(B_np.T)  # (N, K) row-major == (K, N) column-major
        B = torch.as_tensor(B_kernel_np, device="cuda", dtype=torch.float16)
        C = torch.zeros((M_padded, N), dtype=torch.float16, device="cuda")

        return {"lib": lib, "handle": handle, "B": B, "C": C, "N": N, "M": M}

    def run(self, h):
        # mmaCBTKernelSparse writes C directly (no atomicAdd, no read of a
        # prior C value -- confirmed by reading mmaCBT.cu), so no zeroing is
        # needed between reps for correctness; kept anyway for parity with
        # this track's other atomicAdd-based adapters and defensiveness.
        h["C"].zero_()
        h["lib"].smat_run(h["handle"], ctypes.c_void_p(h["B"].data_ptr()),
                          ctypes.c_void_p(h["C"].data_ptr()), ctypes.c_ulong(h["N"]))
        return h["C"][: h["M"]]

    def to_host(self, out):
        import torch
        return out.detach().to("cpu", dtype=torch.float64).numpy()

    def timer(self):
        from kernelbench.impls.gpu_cuda import CudaEventTimer
        return CudaEventTimer()

    def free(self, h):
        import torch
        h["lib"].smat_free(h["handle"])
        h.clear()
        torch.cuda.empty_cache()
