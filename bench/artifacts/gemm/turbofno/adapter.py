"""
TurboFNO adapter for the gemm track.

Paper: "TurboFNO: High-Performance Fourier Neural Operator with Fused
FFT-GEMM-iFFT on GPU", SC'25. `PAPER_KEY = conf/sc/WuZDZHC25`.
Artifact: https://github.com/shixun404/TurboFNO -- the repo's own README
states its contribution includes "Custom high-performance FFT and GEMM
kernels, matching or exceeding cuFFT and cuBLAS performance", so the GEMM
wrapped here is a genuine paper-authored kernel, not a cuBLAS passthrough.

Kernel wrapped (ARTIFACT_GUIDE.md rule 1: wrap the kernel, not a benchmark
script): `cgemm`, a hand-written shared-memory-tiled complex GEMM
(`source/fusion_variants/1D_E_baseline/cgemm.cuh`), unmodified. It is the
finest available kernel boundary -- see wrapper.cu's header comment for why
it is compiled standalone rather than via the repo's fused FFT+GEMM+iFFT
benchmark target (`fused.cu`), which is a bigger fused unit than this
track's plain gemm workload calls for.

PRECISION / real-vs-complex decision (documented per the task brief, see
STATUS.md for the full writeup): TurboFNO has no real-valued GEMM
microkernel anywhere in the repo (checked TurboFFT/ and utils/ -- both are
FFT-only or macro/header utilities, no separate real GEMM). `cgemm` is
inherently complex (`float2` operands throughout). This adapter feeds it
operands with the real part drawn from the SAME seeded-RNG recipe
dense.py's `reference_gemm`/`_rng_operand` uses
(`np.random.default_rng(seed).uniform(-1, 1, size=...).astype(fp32)`) and an
imaginary part of exactly zero; complex multiply-add then reduces
algebraically to the real product for both real and imaginary output
components ((a+0i)(c+0i) = ac + 0i, and summing such terms keeps the
imaginary part exactly 0 in IEEE arithmetic since every partial product's
imaginary term is a*0 or 0*c), so the real part of `cgemm`'s output IS the
correct real GEMM result and is gated against `reference_gemm`'s fp64 answer
via `to_host()` returning only that real component. This is a single,
unmodified invocation of the artifact's own compiled kernel (ARTIFACT_GUIDE
rule 1), not a kernel-code patch. STRUCTURAL CAVEAT (recorded here and in
STATUS.md, not hidden): `cgemm` still performs full complex arithmetic (4
real multiplies + 2 adds per MAC, see cgemm.cuh's `c[i][j].x +=
a[...].x*b[...].x - a[...].y*b[...].y; c[i][j].y += ...`) versus 1 multiply +
1 add for a real GEMM, so it does strictly more raw arithmetic than the
`work_count`/GFLOP/s this harness's cost model credits it for (dense.py's
`_cost_gemm` counts `2*M*N*K*batch`, the REAL-GEMM flop count) -- this
adapter's reported GFLOP/s is therefore pessimistic relative to a true
complex-FLOP accounting. The cost model itself is out of scope to change.

LAYOUT: cgemm.cuh indexes A/B/C column-major (cuBLAS NN convention, see
wrapper.cu's header comment for the exact index derivation cross-checked
against fused.cu's own `cublasCgemm(..., dA, M, dB, K, ..., dC_ref, M)`
call). Row-major numpy operands are transposed into C-contiguous buffers of
the *transposed* shape before upload -- `np.ascontiguousarray(X.T)` on an
(M,K) row-major X yields a (K,M) C-contiguous buffer whose flat bytes are
bit-identical to X stored column-major (M,K) -- so no separate "transpose"
kernel or extra device-side work is introduced; it is a host-side numpy
reshape done once in prepare(), the same spirit as insum/inferfast's own
prepare()-time layout conversions.

NO BATCHING in the kernel itself: `cgemm` has no batch parameter, so for
batch>1 shapes (the smoke batched-gemm workload) `run()` loops the SAME
unmodified launcher once per batch element on a distinct pointer offset --
not a new fused/batched kernel, matching ARTIFACT_GUIDE rule 1's "one kernel
invocation" per logical unit of work (here: one invocation per batch item,
since the artifact defines no wider unit).

ALIGNMENT CONSTRAINT (inherited from the artifact, see wrapper.cu): cgemm.cuh
has no boundary/tail handling, so M must be a multiple of THREADBLOCK_M=64,
N a multiple of THREADBLOCK_N=64, K a multiple of THREADBLOCK_K=8. Checked
explicitly in prepare() (raises rather than silently corrupting memory).
All three --smoke gemm shapes satisfy this; no padding was implemented
(out of this integration's budget -- see STATUS.md "Not done").
"""

from __future__ import annotations

import ctypes
import os

import numpy as np

KERNEL = "gemm"
IMPL_NAME = "turbofno-cgemm"
PAPER_KEY = "conf/sc/WuZDZHC25"
PRECISIONS = ["fp32"]

_HERE = os.path.dirname(os.path.abspath(__file__))
_LIB_PATH = os.path.join(_HERE, "libturbofno_wrapper.so")

# matches kernelbench/domains/dense.py's own _B_OFFSET exactly, so B draws
# the same stream as the harness's own reference_gemm.
_B_OFFSET = 1_000_003
# artifact's own baseline tile sizes (source/utils/TurboFNO.h) -- the
# alignment requirement wrapper.cu's cgemm launch inherits, not invented here.
_TB_M, _TB_N, _TB_K = 64, 64, 8

_lib = None


def _load_lib():
    global _lib
    if _lib is not None:
        return _lib
    if not os.path.exists(_LIB_PATH):
        raise FileNotFoundError(f"{_LIB_PATH} not built -- run build.sh")
    lib = ctypes.CDLL(_LIB_PATH)
    lib.turbofno_cgemm_launch.argtypes = [
        ctypes.c_int, ctypes.c_int, ctypes.c_int,
        ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
        ctypes.c_float, ctypes.c_float, ctypes.c_float, ctypes.c_float,
    ]
    lib.turbofno_cgemm_launch.restype = ctypes.c_int  # cudaError_t
    _lib = lib
    return _lib


def available() -> tuple[bool, str]:
    if not os.path.exists(_LIB_PATH):
        return False, "libturbofno_wrapper.so not built -- run build.sh"
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
    return TurboFNOGemm(precision)


def _rng_operand(rows: int, cols: int, seed: int, dtype) -> np.ndarray:
    """Identical recipe to kernelbench/domains/dense.py's own _rng_operand
    (private to that module, so replicated verbatim here rather than
    importing an underscore-prefixed symbol across modules)."""
    rng = np.random.default_rng(seed)
    return rng.uniform(-1.0, 1.0, size=(rows, cols)).astype(dtype)


class TurboFNOGemm:
    name = IMPL_NAME
    platform = "cuda"

    def __init__(self, precision: str = "fp32"):
        if precision != "fp32":
            raise NotImplementedError(
                f"{IMPL_NAME} only wired for fp32 real / float2 complex "
                f"operands (cgemm.cuh's `float2` is fp32-component "
                f"throughout); requested {precision}")
        self.precision = precision

    def prepare(self, workload, params: dict):
        import torch
        lib = _load_lib()

        w = workload
        M, N, K, batch = int(w.M), int(w.N), int(w.K), int(w.batch)
        if M % _TB_M or N % _TB_N or K % _TB_K:
            raise NotImplementedError(
                f"turbofno-cgemm: shape M={M},N={N},K={K} not aligned to "
                f"the kernel's own tile sizes ({_TB_M}/{_TB_N}/{_TB_K}) -- "
                f"cgemm.cuh has no boundary handling (see wrapper.cu); "
                f"padding not implemented for this integration")

        seed = params.get("seed", w.seed)
        dtype = np.float32  # PRECISIONS = ["fp32"]; see class docstring

        A_list, B_list = [], []
        for b in range(batch):
            A_re = _rng_operand(M, K, seed + b, dtype)             # (M,K)
            B_re = _rng_operand(K, N, seed + _B_OFFSET + b, dtype)  # (K,N)
            # column-major (M,K)/(K,N) bytes == C-contiguous (K,M)/(N,K)
            # bytes of the transpose -- see module docstring's LAYOUT note.
            A_list.append(np.ascontiguousarray(A_re.T))  # (K,M)
            B_list.append(np.ascontiguousarray(B_re.T))  # (N,K)

        # real-valued-in-complex-buffer: imaginary part left at exactly 0
        # (np.zeros-initialized complex64 array, then real component filled
        # in-place) -- see module docstring's precision-decision note.
        A_np = np.zeros((batch, K, M), dtype=np.complex64)
        B_np = np.zeros((batch, N, K), dtype=np.complex64)
        for b in range(batch):
            A_np[b].real = A_list[b]
            B_np[b].real = B_list[b]
        C_np = np.zeros((batch, N, M), dtype=np.complex64)  # column-major (M,N)

        A_dev = torch.from_numpy(A_np).cuda()
        B_dev = torch.from_numpy(B_np).cuda()
        C_dev = torch.from_numpy(C_np).cuda()

        self._M, self._N, self._batch = M, N, batch
        return {"A": A_dev, "B": B_dev, "C": C_dev, "M": M, "N": N, "K": K, "batch": batch}

    def run(self, h):
        lib = _load_lib()
        h["C"].zero_()
        M, N, K, batch = h["M"], h["N"], h["K"], h["batch"]
        A, B, C = h["A"], h["B"], h["C"]
        for b in range(batch):
            err = lib.turbofno_cgemm_launch(
                ctypes.c_int(M), ctypes.c_int(N), ctypes.c_int(K),
                ctypes.c_void_p(A[b].data_ptr()),
                ctypes.c_void_p(B[b].data_ptr()),
                ctypes.c_void_p(C[b].data_ptr()),
                ctypes.c_float(1.0), ctypes.c_float(0.0),   # alpha = 1+0i
                ctypes.c_float(0.0), ctypes.c_float(0.0))   # beta  = 0+0i
            if err != 0:
                raise RuntimeError(f"turbofno_cgemm_launch returned cudaError_t={err}")
        return C

    def to_host(self, out):
        import torch
        M, N, batch = self._M, self._N, self._batch
        # out: (batch, N, M) complex64, column-major-(M,N) bytes viewed as
        # C-order (N,M) -- see module docstring's LAYOUT note. Take the real
        # component only (imaginary part is exactly 0 by construction, see
        # the precision-decision note) and transpose back to (M,N).
        real = out.detach().to("cpu").real.to(torch.float64)  # (batch, N, M)
        real = real.transpose(-1, -2).contiguous().numpy()    # (batch, M, N)
        return real[0] if batch == 1 else real

    def timer(self):
        from kernelbench.impls.gpu_cuda import CudaEventTimer
        return CudaEventTimer()

    def free(self, h):
        import torch
        h.clear()
        torch.cuda.empty_cache()
