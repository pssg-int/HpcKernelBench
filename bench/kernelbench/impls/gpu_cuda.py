"""
CUDA implementations.

Two families:
  * `Torch*`   — vendor/library baselines (cuSPARSE via torch.sparse), the
                 "what you get for free" reference every custom kernel must beat.
  * `Custom*`  — the hand-written kernels in ../../csrc/kernels.cu, loaded through
                 ctypes with torch supplying device pointers, so nothing is
                 allocated inside a timed region.

Timing uses CUDA events, one pair per iteration, per the specs. Note the specs'
own finding: RoDe / GE-SpMM / SMaT wrap a whole loop in ONE event pair and report
the mean, which cannot expose variance — the harness deliberately does not.
"""

from __future__ import annotations

import ctypes
import os

import numpy as np

# ---------------------------------------------------------------------------
# Shared OPERAND generators, imported from the CPU reference / domain modules.
#
# BUG THIS FIXES: every built-in CUDA impl below used to draw its dense
# operands with torch.Generator(...).manual_seed(seed) + torch.rand/randn,
# while every reference function (and every CPU impl, and every artifact
# adapter) draws them with numpy.random.default_rng(seed) -- two RNG
# algorithms that do not produce the same stream from the same seed. That
# mismatch fed the GPU kernel a genuinely DIFFERENT B/x/Q than the one the
# reference checks against, so every built-in CUDA impl failed its
# correctness gate before computing a single wrong FLOP.
#
# The fix below: import the SAME numpy operand-generator function each
# reference already calls, build the operand as a numpy array with it, then
# move it to the GPU with torch.as_tensor(...).to(device, dtype). Where a
# reference inlines its own copy of the formula with no separate function to
# import (reference_spmv, reference_fft's real/complex branches already
# covered by spectral._fft_operand -- see below), the exact numpy recipe is
# replicated locally as a small helper instead.
#
# This is allowed under DOMAIN_GUIDE.md's reference-independence rule: that
# rule forbids a REFERENCE from sharing COMPUTATION with any impl it gates,
# not sharing the OPERAND ARRAY every impl (CPU and GPU alike) is handed to
# begin with -- see TorchSpGEMM's to_host() below (already imports
# cpu_ref._canonical_pattern/_reindex_to_pattern) and every domain module's
# own reference function, which already imports and calls these same
# operand generators itself (e.g. ml.reference_conv calls
# ml._make_conv_operands; dense.reference_gemm calls dense._rng_operand).
# ---------------------------------------------------------------------------
from .cpu_ref import _dense_operand
from ..domains.dense import _rng_operand, _rng_vector, _spd_matrix, _B_OFFSET
from ..domains.spectral import _fft_operand
from ..domains.ml import _make_conv_operands, _qkv, _qkv_sparse, _make_activation
from ..domains.tensor import _make_factors

_LIB_PATH = os.path.normpath(os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "..", "csrc", "libkernelbench.so"))

_lib = None


def _load_lib():
    global _lib
    if _lib is None:
        if not os.path.exists(_LIB_PATH):
            raise FileNotFoundError(
                f"{_LIB_PATH} not built — run `make` in bench/csrc")
        _lib = ctypes.CDLL(_LIB_PATH)
        p = ctypes.c_void_p
        for fn, argtypes in {
            "spmv_csr_warp_f32": [ctypes.c_int, p, p, p, p, p, p],
            "spmv_csr_warp_f64": [ctypes.c_int, p, p, p, p, p, p],
            "spmm_csr_warp_f32": [ctypes.c_int, ctypes.c_int, p, p, p, p, p, p],
            "sddmm_csr_warp_f32": [ctypes.c_int, ctypes.c_int, p, p, p, p, p, p, p],
        }.items():
            getattr(_lib, fn).argtypes = argtypes
            getattr(_lib, fn).restype = None
        _lib.kernels_last_error.restype = ctypes.c_int
    return _lib


class CudaEventTimer:
    """One event pair per iteration, as the specs require."""

    def __init__(self):
        import torch
        self.torch = torch
        self.start = torch.cuda.Event(enable_timing=True)
        self.stop = torch.cuda.Event(enable_timing=True)

    def __enter__(self):
        self.start.record()
        return self

    def __exit__(self, *exc):
        self.stop.record()
        self.stop.synchronize()
        self.seconds = self.start.elapsed_time(self.stop) / 1e3
        return False


def _torch_dtype(precision: str):
    import torch
    return {"fp64": torch.float64, "fp32": torch.float32,
            "fp16": torch.float16, "bf16": torch.bfloat16}[precision]


def _np_dtype_of(torch_dtype):
    """torch dtype -> numpy dtype, for feeding a torch dtype into a numpy RNG
    recipe. bf16 has no native numpy type; alias to fp32, the same
    convention every domain module's own `_dtype()`/`_NP_DTYPE` helper uses
    (numpy has no meaningful bf16 rounding to offer without real tensor-core
    hardware anyway)."""
    import torch
    return {torch.float64: np.float64, torch.float32: np.float32,
            torch.float16: np.float16, torch.bfloat16: np.float32}[torch_dtype]


def _dense(rows, dim, seed, dtype, device):
    """Dense SpMM/SDDMM operand -- cpu_ref._dense_operand's numpy recipe
    (the SAME operand generator reference_spmm/reference_sddmm themselves
    call), moved to the GPU. See the module-level comment above for why
    importing this OPERAND generator does not violate reference
    independence."""
    import torch
    arr = _dense_operand(rows, dim, seed, _np_dtype_of(dtype))
    return torch.as_tensor(arr).to(device=device, dtype=dtype)


def _spmv_vector(n, seed, dtype):
    """SpMV right-hand-side vector -- replicates cpu_ref.reference_spmv's
    inline numpy recipe verbatim (that reference has no separate function to
    import: `rng.uniform(-1.0, 1.0, size=A.shape[1]).astype(dtype)`). Kept
    as a plain numpy array; callers move it to the GPU themselves."""
    rng = np.random.default_rng(seed)
    return rng.uniform(-1.0, 1.0, size=n).astype(dtype)


class _CudaBase:
    platform = "cuda"

    def __init__(self, precision: str = "fp32"):
        self.precision = precision

    def to_host(self, out):
        return out.detach().to("cpu", dtype=__import__("torch").float64).numpy()

    def timer(self):
        return CudaEventTimer()

    def free(self, h):
        import torch
        h.clear()
        torch.cuda.empty_cache()


# ------------------------------------------------------------ library baselines
class TorchSpMV(_CudaBase):
    name = "cusparse-csr-spmv (torch)"

    def prepare(self, matrix, params):
        import torch
        dt = _torch_dtype(self.precision)
        A = matrix.csr
        Ad = torch.sparse_csr_tensor(
            torch.as_tensor(A.indptr, dtype=torch.int32),
            torch.as_tensor(A.indices, dtype=torch.int32),
            torch.as_tensor(A.data, dtype=dt),
            size=A.shape, device="cuda")
        x_np = _spmv_vector(A.shape[1], params.get("seed", 42), _np_dtype_of(dt))
        x = torch.as_tensor(x_np)
        return {"A": Ad, "x": x.to("cuda", dt)}

    def run(self, h):
        return h["A"] @ h["x"]


class TorchSpMM(_CudaBase):
    name = "cusparse-csr-spmm (torch)"

    def prepare(self, matrix, params):
        import torch
        dt = _torch_dtype(self.precision)
        A = matrix.csr
        Ad = torch.sparse_csr_tensor(
            torch.as_tensor(A.indptr, dtype=torch.int32),
            torch.as_tensor(A.indices, dtype=torch.int32),
            torch.as_tensor(A.data, dtype=dt),
            size=A.shape, device="cuda")
        B = _dense(A.shape[1], int(params["N"]), params.get("seed", 42), dt, "cuda")
        return {"A": Ad, "B": B}

    def run(self, h):
        return h["A"] @ h["B"]


class TorchSpGEMM(_CudaBase):
    """
    C = A @ A via torch's sparse CSR-CSR matmul (cuSPARSE underneath) -- the
    GPU library baseline the spgemm track's paper artifacts are compared
    against, analogous to TorchSpMV/TorchSpMM above. `to_host()` reindexes
    the result onto the same |A|@|A| structural pattern the CPU reference
    and every other spgemm impl use (see cpu_ref._canonical_pattern /
    _reindex_to_pattern) so cuSPARSE's own output order does not matter.
    """

    name = "cusparse-csr-spgemm (torch)"

    def prepare(self, matrix, params):
        import torch
        dt = _torch_dtype(self.precision)
        A = matrix.csr
        A.sort_indices()
        Ad = torch.sparse_csr_tensor(
            torch.as_tensor(A.indptr, dtype=torch.int64),
            torch.as_tensor(A.indices, dtype=torch.int64),
            torch.as_tensor(A.data, dtype=dt),
            size=A.shape, device="cuda")
        self._matrix = matrix
        return {"A": Ad}

    def run(self, h):
        return h["A"] @ h["A"]

    def to_host(self, out):
        from ..impls.cpu_ref import _canonical_pattern, _reindex_to_pattern
        import scipy.sparse as sp
        crow = out.crow_indices().detach().cpu().numpy()
        col = out.col_indices().detach().cpu().numpy()
        val = out.values().detach().cpu().to(dtype=__import__("torch").float64).numpy()
        C = sp.csr_matrix((val, col, crow), shape=tuple(out.shape))
        A64 = self._matrix.csr.astype(np.float64)
        A64.sort_indices()
        pattern = _canonical_pattern(A64)
        return _reindex_to_pattern(C, pattern)


class TorchSDDMM(_CudaBase):
    name = "torch-sampled-addmm-sddmm"

    def prepare(self, matrix, params):
        import torch
        dt = _torch_dtype(self.precision)
        S = matrix.csr
        K = int(params["K"])
        Sd = torch.sparse_csr_tensor(
            torch.as_tensor(S.indptr, dtype=torch.int32),
            torch.as_tensor(S.indices, dtype=torch.int32),
            torch.as_tensor(S.data, dtype=dt),
            size=S.shape, device="cuda")
        A = _dense(S.shape[0], K, params.get("seed", 42), dt, "cuda")
        B = _dense(S.shape[1], K, params.get("seed", 42) + 1, dt, "cuda")
        return {"S": Sd, "A": A, "Bt": B.t().contiguous(),
                "Sdata": torch.as_tensor(S.data, dtype=dt).to("cuda")}

    def run(self, h):
        import torch
        # sampled_addmm computes S_pattern * (A @ Bt); multiply by S values after
        out = torch.sparse.sampled_addmm(h["S"], h["A"], h["Bt"], beta=0.0, alpha=1.0)
        return out.values() * h["Sdata"]

    def to_host(self, out):
        import torch
        return out.detach().to("cpu", dtype=torch.float64).numpy()


# ------------------------------------------------------ dense library baselines
class TorchGemm(_CudaBase):
    """C = A @ B via torch (cuBLAS/rocBLAS underneath) -- the dense
    counterpart to TorchSpMV/TorchSpMM above. Functional-only smoke-gated on
    the login node (2026-09-06): 3/3 valid against reference_gemm."""

    name = "torch-matmul"

    def prepare(self, workload, params):
        import torch
        dt = _torch_dtype(self.precision)
        seed = params.get("seed", workload.seed)
        w = workload
        np_dtype = _np_dtype_of(dt)
        # dense._rng_operand / dense._B_OFFSET -- the SAME operand generator
        # and batch offset reference_gemm itself calls.
        if w.batch > 1:
            A = torch.stack([torch.as_tensor(_rng_operand(w.M, w.K, seed + b, np_dtype)).to("cuda", dt)
                              for b in range(w.batch)])
            B = torch.stack([torch.as_tensor(_rng_operand(w.K, w.N, seed + _B_OFFSET + b, np_dtype)).to("cuda", dt)
                              for b in range(w.batch)])
        else:
            A = torch.as_tensor(_rng_operand(w.M, w.K, seed, np_dtype)).to("cuda", dt)
            B = torch.as_tensor(_rng_operand(w.K, w.N, seed + _B_OFFSET, np_dtype)).to("cuda", dt)
        return {"A": A, "B": B}

    def run(self, h):
        return h["A"] @ h["B"]


class TorchGemv(_CudaBase):
    """y = A @ x via torch. GEMV modeled as GEMM-with-N=1, matching
    kernelbench/domains/dense.py's NumpyGemv. Functional-only smoke-gated on
    the login node (2026-09-06): 3/3 valid against reference_gemv."""

    name = "torch-mv"

    def prepare(self, workload, params):
        import torch
        dt = _torch_dtype(self.precision)
        seed = params.get("seed", workload.seed)
        np_dtype = _np_dtype_of(dt)
        # dense._rng_operand/_rng_vector/_B_OFFSET -- the SAME generators
        # reference_gemv itself calls.
        A = torch.as_tensor(_rng_operand(workload.M, workload.K, seed, np_dtype)).to("cuda", dt)
        x = torch.as_tensor(_rng_vector(workload.K, seed + _B_OFFSET, np_dtype)).to("cuda", dt)
        return {"A": A, "x": x}

    def run(self, h):
        import torch
        return torch.mv(h["A"], h["x"])


class TorchCholesky(_CudaBase):
    """L = torch.linalg.cholesky(A) (cuSOLVER underneath). SPD operand IS
    dense._spd_matrix's own numpy array (the SAME generator reference_
    cholesky calls: M @ M.T + n*I, built once on the CPU at the run's own
    precision, then moved to the GPU verbatim -- no GPU-side matmul
    reconstruction of M @ M.T, which would risk a different rounding path).

    Functional-only smoke gate attempted on the login node (2026-09-06):
    blocked by an ENVIRONMENT defect, not a code bug -- torch.linalg.
    cholesky lazily dlopen's libtorch_cuda_linalg.so, which this venv's
    torch install does not ship; reproduces with a bare
    torch.linalg.cholesky(torch.eye(4, device='cuda')) call, independent of
    this class's own code."""

    name = "torch-cholesky"

    def prepare(self, workload, params):
        import torch
        dt = _torch_dtype(self.precision)
        seed = params.get("seed", workload.seed)
        n = workload.n
        # dense._spd_matrix -- the SAME operand generator reference_cholesky
        # itself calls (built entirely on the CPU at the run's own
        # precision, then moved to the GPU as-is -- no GPU-side M@M.T
        # reconstruction, which would risk a different rounding path).
        A = torch.as_tensor(_spd_matrix(n, seed, _np_dtype_of(dt))).to("cuda", dt)
        return {"A": A}

    def run(self, h):
        import torch
        return torch.linalg.cholesky(h["A"])


# --------------------------------------------------------------- custom kernels
class CustomSpMV(_CudaBase):
    name = "custom-warp-csr-spmv"

    def prepare(self, matrix, params):
        import torch
        lib = _load_lib()
        dt = _torch_dtype(self.precision)
        A = matrix.csr
        h = {
            "lib": lib,
            "rows": A.shape[0],
            "indptr": torch.as_tensor(A.indptr, dtype=torch.int32).cuda(),
            "indices": torch.as_tensor(A.indices, dtype=torch.int32).cuda(),
            "data": torch.as_tensor(A.data, dtype=dt).cuda(),
        }
        x_np = _spmv_vector(A.shape[1], params.get("seed", 42), _np_dtype_of(dt))
        h["x"] = torch.as_tensor(x_np).to("cuda", dt)
        h["y"] = torch.zeros(A.shape[0], dtype=dt, device="cuda")
        return h

    def run(self, h):
        fn = h["lib"].spmv_csr_warp_f32 if self.precision == "fp32" \
            else h["lib"].spmv_csr_warp_f64
        fn(h["rows"],
           ctypes.c_void_p(h["indptr"].data_ptr()),
           ctypes.c_void_p(h["indices"].data_ptr()),
           ctypes.c_void_p(h["data"].data_ptr()),
           ctypes.c_void_p(h["x"].data_ptr()),
           ctypes.c_void_p(h["y"].data_ptr()),
           ctypes.c_void_p(0))
        return h["y"]


class CustomSpMM(_CudaBase):
    name = "custom-warp-csr-spmm"

    def prepare(self, matrix, params):
        if self.precision != "fp32":
            raise NotImplementedError(
                "custom-warp-csr-spmm only ships an fp32 CUDA symbol; "
                f"requested {self.precision}")
        import torch
        lib = _load_lib()
        dt = _torch_dtype(self.precision)
        A = matrix.csr
        N = int(params["N"])
        h = {
            "lib": lib, "rows": A.shape[0], "N": N,
            "indptr": torch.as_tensor(A.indptr, dtype=torch.int32).cuda(),
            "indices": torch.as_tensor(A.indices, dtype=torch.int32).cuda(),
            "data": torch.as_tensor(A.data, dtype=dt).cuda(),
            "B": _dense(A.shape[1], N, params.get("seed", 42), dt, "cuda"),
        }
        h["C"] = torch.zeros((A.shape[0], N), dtype=dt, device="cuda")
        return h

    def run(self, h):
        h["lib"].spmm_csr_warp_f32(
            h["rows"], h["N"],
            ctypes.c_void_p(h["indptr"].data_ptr()),
            ctypes.c_void_p(h["indices"].data_ptr()),
            ctypes.c_void_p(h["data"].data_ptr()),
            ctypes.c_void_p(h["B"].data_ptr()),
            ctypes.c_void_p(h["C"].data_ptr()),
            ctypes.c_void_p(0))
        return h["C"]


# ------------------------------------------------------------------ stencil
class TorchStencilConv(_CudaBase):
    """
    Double-buffered stencil sweep via torch's conv1d/2d/3d with circular
    ('wrap') padding -- the library-baseline GPU path, analogous to TorchSpMV
    et al. above (a hand-tuned custom kernel, e.g. with shared-memory spatial
    or temporal blocking, would be the CustomSpMV-style counterpart).

    conv1d/2d/3d compute CROSS-CORRELATION (no kernel flip), so the kernel is
    built with `workload.dense_kernel(flip=False)` -- see that method's
    docstring in domains/stencil.py for why this differs from the flip=True
    layout scipy.ndimage.convolve needs for the identical stencil.

    Functional-only smoke gate attempted on the login node (2026-09-06):
    blocked by an ENVIRONMENT defect, not a code bug -- torch's conv1d/2d/3d
    route through cuDNN, and this venv's nvidia-cudnn-cu12 install is missing
    the engine-plugin shared libraries (libcudnn_engines_precompiled.so.9 and
    friends; only libcudnn.so.9/libcudnn_graph.so.9 are present), so every
    cuDNN convolution call raises CUDNN_STATUS_SUBLIBRARY_LOADING_FAILED --
    reproduces with a bare torch.nn.functional.conv2d call, independent of
    this class's own code. See TorchConv2d's docstring below for the same
    finding (first hit there).
    """

    name = "torch-conv-stencil"

    def prepare(self, workload, params):
        import torch
        import torch.nn.functional as F  # noqa: F401  (imported here to fail fast if missing)
        dt = _torch_dtype(self.precision)
        dims = workload.dims
        if dims > 3:
            raise NotImplementedError(
                "torch conv1d/2d/3d supports dims<=3 only; stencil dims="
                f"{dims} has no torch-conv GPU path")
        u0_np = workload.initial_field(dtype=np.float64)
        kernel_np = workload.dense_kernel(dtype=np.float64, flip=False)
        u0 = torch.as_tensor(u0_np, dtype=dt, device="cuda")
        weight = torch.as_tensor(kernel_np, dtype=dt, device="cuda").reshape(
            1, 1, *kernel_np.shape)
        bufs = [torch.empty_like(u0), torch.empty_like(u0)]
        return {
            "u0": u0, "bufs": bufs, "weight": weight, "dims": dims,
            "radius": workload.radius,
            "timesteps": int(params.get("timesteps", workload.timesteps)),
        }

    def run(self, h):
        import torch.nn.functional as F
        bufs, weight, dims, r = h["bufs"], h["weight"], h["dims"], h["radius"]
        conv = {1: F.conv1d, 2: F.conv2d, 3: F.conv3d}[dims]
        bufs[0].copy_(h["u0"])
        cur, nxt = 0, 1
        for _ in range(h["timesteps"]):
            src = bufs[cur].unsqueeze(0).unsqueeze(0)
            padded = F.pad(src, (r, r) * dims, mode="circular")
            bufs[nxt] = conv(padded, weight).squeeze(0).squeeze(0)
            cur, nxt = nxt, cur
        return bufs[cur]


# ------------------------------------------------------------------- FFT
class TorchFFT(_CudaBase):
    """
    torch.fft GPU baseline (cuFFT-backed), timed with CudaEventTimer per the
    specs -- one event pair per iteration, matching every other CUDA impl in
    this file. Shares spectral._fft_operand with NumpyFFT/ScipyFFT/
    reference_fft (see domains/spectral.py) for an FFTWorkload's operand.

    DOMAIN_GUIDE.md is explicit that GPU TIMING must not run on this shared
    login node -- no timing numbers from this class are ever published from
    here. Functional-only correctness smoke-gated on the login node
    (2026-09-06): 8/8 valid against reference_fft, forward/inverse and
    real/complex branches alike.
    """

    name = "torch-fft"

    def prepare(self, workload, params):
        import torch
        np_real = np.float32 if self.precision == "fp32" else np.float64
        np_complex = np.complex64 if self.precision == "fp32" else np.complex128
        axes = tuple(range(-workload.ndim, 0))
        # spectral._fft_operand -- the SAME operand generator NumpyFFT/
        # ScipyFFT (and reference_fft, for the forward/complex branches)
        # already build their input with; the R2C inverse branch's
        # Hermitian-packed input uses numpy's own rfftn here (matching
        # NumpyFFT/ScipyFFT's own callback), not torch.fft's, so the CPU-
        # built operand is identical byte-for-byte before the .to("cuda")
        # transfer below.
        x_np = _fft_operand(workload, np_real, np_complex,
                             params.get("seed", workload.seed),
                             lambda a, ax: np.fft.rfftn(a, axes=ax))
        x = torch.as_tensor(x_np).to("cuda")
        return {"x": x, "axes": axes, "workload": workload}

    def run(self, h):
        import torch
        w, x, axes = h["workload"], h["x"], h["axes"]
        if w.real_input:
            if w.direction == "forward":
                return torch.fft.rfftn(x, dim=axes)
            return torch.fft.irfftn(x, s=w.dims, dim=axes)
        return torch.fft.fftn(x, dim=axes) if w.direction == "forward" \
            else torch.fft.ifftn(x, dim=axes)

    def to_host(self, out):
        import torch
        out = out.detach().to("cpu")
        if out.is_complex():
            out = torch.view_as_real(out)
        return out.to(torch.float64).numpy()


class CustomSDDMM(_CudaBase):
    name = "custom-warp-csr-sddmm"

    def prepare(self, matrix, params):
        if self.precision != "fp32":
            raise NotImplementedError(
                "custom-warp-csr-sddmm only ships an fp32 CUDA symbol; "
                f"requested {self.precision}")
        import torch
        lib = _load_lib()
        dt = _torch_dtype(self.precision)
        S = matrix.csr
        K = int(params["K"])
        h = {
            "lib": lib, "rows": S.shape[0], "K": K,
            "indptr": torch.as_tensor(S.indptr, dtype=torch.int32).cuda(),
            "indices": torch.as_tensor(S.indices, dtype=torch.int32).cuda(),
            "data": torch.as_tensor(S.data, dtype=dt).cuda(),
            "A": _dense(S.shape[0], K, params.get("seed", 42), dt, "cuda"),
            "B": _dense(S.shape[1], K, params.get("seed", 42) + 1, dt, "cuda"),
        }
        h["P"] = torch.zeros(S.nnz, dtype=dt, device="cuda")
        return h

    def run(self, h):
        h["lib"].sddmm_csr_warp_f32(
            h["rows"], h["K"],
            ctypes.c_void_p(h["indptr"].data_ptr()),
            ctypes.c_void_p(h["indices"].data_ptr()),
            ctypes.c_void_p(h["data"].data_ptr()),
            ctypes.c_void_p(h["A"].data_ptr()),
            ctypes.c_void_p(h["B"].data_ptr()),
            ctypes.c_void_p(h["P"].data_ptr()),
            ctypes.c_void_p(0))
        return h["P"]


# --------------------------------------------------------------------- ml
class TorchConv2d(_CudaBase):
    """
    cuDNN via torch.nn.functional.conv2d -- the GPU library baseline every
    custom conv kernel in the convolution track (Tetris, cgo, TC-BNN, ...) is
    compared against, analogous to TorchGemm/TorchSpMM above. Operand
    generation imports kernelbench/domains/ml.py's own `_make_conv_operands`
    directly -- the SAME operand generator reference_conv itself calls (X
    then W, U(-1,1)) -- so a GPU rerun of the same workload/seed draws the
    bit-identical operand array reference_conv widens to fp64.

    Functional-only smoke gate attempted on the login node (2026-09-06):
    blocked by an ENVIRONMENT defect, not a code bug -- this venv's
    nvidia-cudnn-cu12 install ships only libcudnn.so.9/libcudnn_graph.so.9,
    missing the engine-plugin shared libraries (libcudnn_engines_precompiled
    .so.9 and friends) every cuDNN convolution call needs, so
    F.conv2d(...) raises CUDNN_STATUS_SUBLIBRARY_LOADING_FAILED --
    reproduces with a bare torch.nn.functional.conv2d call on this venv,
    independent of this class's own code (see TorchStencilConv above for the
    same finding via conv1d/2d/3d).
    """

    name = "cudnn-conv2d (torch)"

    def prepare(self, workload, params):
        import torch
        dt = _torch_dtype(self.precision)
        w = workload
        N = int(params.get("N", w.N))
        # ml._make_conv_operands -- the SAME operand generator
        # reference_conv itself calls (X then W, U(-1,1)).
        X_np, W_np = _make_conv_operands(w, params, N, _np_dtype_of(dt))
        X = torch.as_tensor(X_np).to("cuda", dt)
        W = torch.as_tensor(W_np).to("cuda", dt)
        return {"X": X, "W": W, "stride": w.stride,
                "padding": (w.pad_h, w.pad_w), "groups": w.groups}

    def run(self, h):
        import torch.nn.functional as F
        return F.conv2d(h["X"], h["W"], stride=h["stride"],
                        padding=h["padding"], groups=h["groups"])


class TorchAttention(_CudaBase):
    """
    torch.nn.functional.scaled_dot_product_attention -- dispatches to
    FlashAttention/memory-efficient/math backends depending on hardware and
    dtype; the GPU library baseline every survey kernel in the
    attention-kernel track (RedFuser, Hexcute, MetaAttention, ...) is
    compared against. Handles GQA natively (Hkv<H) via `enable_gqa`, and
    both prefill (Sq==Sk) and decode (Sq=1) shapes via the same call, since
    kernelbench/domains/ml.py's AttentionWorkload models both uniformly.

    Functional-only smoke-gated on the login node (2026-09-06):
    3/4 valid against reference_attention (fp16, attn-prefill-kernel-
    fp16bf16). smoke-attn-decode-gqa (Sq=1, Sk=48, mask=causal) FAILS with
    max_scaled_err ~4.4 (tol 0.01) -- a SEPARATE, genuine bug, not the RNG
    mismatch this file's other fixes address, and NOT fixed here per this
    task's "investigate briefly and report; do not loosen gates" instruction.
    Root cause, isolated directly (bypassing this class, same seeded
    operands): torch's `is_causal=True` uses TOP-LEFT causal alignment when
    Sq != Sk (mask[q,k] = k<=q, so the single decode query can only see
    key 0), while this project's reference (and real KV-cache decode
    semantics) use BOTTOM-RIGHT alignment (mask[q,k] = k <= q+(Sk-Sq), so
    the decode query sees the full cache) -- confirmed by probing
    `F.scaled_dot_product_attention(is_causal=True)` against explicit
    top-left/bottom-right boolean masks on a toy (Sq=1,Sk=5) case. Every
    prefill shape (Sq==Sk) is unaffected since the two conventions coincide
    there, which is why smoke-attn-prefill-{causal,bidir,gqa} all pass.
    FIXED 2026-09-06: prepare() now passes an explicit bottom-right boolean
    attn_mask whenever Sq != Sk (see the comment there); is_causal is kept for
    Sq == Sk so prefill still reaches the flash backend.
    """

    name = "torch-sdpa (flash/mem-efficient backend)"

    def prepare(self, workload, params):
        import torch
        dt = _torch_dtype(self.precision)
        w = workload
        # ml._qkv -- the SAME operand generator reference_attention itself
        # calls (N(0,1), draw order Q, K, V).
        Q_np, K_np, V_np = _qkv(w, params, _np_dtype_of(dt))
        Q = torch.as_tensor(Q_np).to("cuda", dt)
        K = torch.as_tensor(K_np).to("cuda", dt)
        V = torch.as_tensor(V_np).to("cuda", dt)
        h = {"Q": Q, "K": K, "V": V, "is_causal": w.mask == "causal",
             "enable_gqa": w.Hkv < w.H, "attn_mask": None}
        if w.mask == "causal" and w.Sq != w.Sk:
            # torch's is_causal=True is TOP-LEFT aligned (mask[q,k] = k <= q);
            # this track's reference (ml._causal_mask) is BOTTOM-RIGHT aligned
            # (k <= q + (Sk - Sq)), the KV-cache-continuation semantics under
            # which a decode query (Sq=1) sees its whole history. Pass the
            # explicit boolean mask for Sq != Sk (True = attend); the two
            # conventions coincide for Sq == Sk so prefill keeps is_causal
            # and the flash backend. Fixed 2026-09-06 (was err 4.4 on
            # smoke-attn-decode-gqa).
            offset = w.Sk - w.Sq
            i = torch.arange(w.Sq, device="cuda")[:, None]
            j = torch.arange(w.Sk, device="cuda")[None, :]
            h["attn_mask"] = (j <= i + offset)
            h["is_causal"] = False
        return h

    def run(self, h):
        import torch.nn.functional as F
        return F.scaled_dot_product_attention(
            h["Q"], h["K"], h["V"], attn_mask=h["attn_mask"], is_causal=h["is_causal"],
            enable_gqa=h["enable_gqa"])


class TorchSparseAttention(_CudaBase):
    """
    torch.nn.functional.scaled_dot_product_attention with an explicit
    BOOLEAN attn_mask -- the "what you get for free" dense-fallback GPU
    baseline for sparse-attention-kernel, analogous to TorchAttention above
    for the dense attention-kernel track. torch's SDPA still computes the
    FULL (S,S) score matrix internally when given an arbitrary bool mask (no
    sparsity is exploited by this path); the REAL GPU competitors for this
    track are the paper artifacts under bench/artifacts/
    sparse-attention-kernel/ (native sparse formats -- ILP-scheduled
    adjacency tiles, fused mask-aware kernels, COO/CSR explicit kernels),
    not this dense fallback. Kept only as the same "vendor library, for
    comparison" wiring point TorchConv2d/TorchAttention are.

    Operand generation imports kernelbench/domains/ml.py's own `_qkv_sparse`
    directly -- the SAME operand generator reference_sparse_attention itself
    calls (draw order Q, K, V, N(0,1)) -- so this GPU baseline draws the
    bit-identical array the reference widens to fp64. The mask itself is
    read directly from the workload's own `.mask` (SparseAttentionWorkload's
    cached, checksummed array -- see ml.py) and cast to torch bool, so this
    GPU baseline consumes the EXACT SAME mask array every CPU impl and the
    reference do.

    Functional-only smoke-gated on the login node (2026-09-06): 7/7 valid
    against reference_sparse_attention (fp16, sparse-attn-structured-mask-
    kernel-fp16). Every smoke workload here is prefill-shaped (Sq==Sk), so
    TorchAttention's decode-only is_causal alignment bug (see its docstring
    above) does not apply -- this class never passes is_causal at all, only
    an explicit attn_mask.
    """

    name = "torch-sdpa-masked (dense fallback, torch bool attn_mask)"

    def prepare(self, workload, params):
        import torch
        dt = _torch_dtype(self.precision)
        w = workload
        # ml._qkv_sparse -- the SAME operand generator reference_sparse_
        # attention itself calls (N(0,1), draw order Q, K, V).
        Q_np, K_np, V_np = _qkv_sparse(w, params, _np_dtype_of(dt))
        Q = torch.as_tensor(Q_np).to("cuda", dt)
        K = torch.as_tensor(K_np).to("cuda", dt)
        V = torch.as_tensor(V_np).to("cuda", dt)
        mask = torch.as_tensor(w.mask, dtype=torch.bool, device="cuda")
        return {"Q": Q, "K": K, "V": V, "mask": mask}

    def run(self, h):
        import torch.nn.functional as F
        return F.scaled_dot_product_attention(h["Q"], h["K"], h["V"], attn_mask=h["mask"])


class TorchDequantGemm(_CudaBase):
    """
    Naive/explicit dequant-then-dense-GEMM baseline -- the EXACT baseline
    named in benchspecs/quantized-gemm/spec.yaml's own `operation` field:
    dequantize B to a dense fp16 B' via a SEPARATE pre-pass, then run a
    standard dense GEMM ("the naive/explicit baseline every fused kernel is
    compared against"). Dequantization happens in prepare() -- this IS the
    artifact's own preprocessing per ARTIFACT_GUIDE's "format conversion
    goes in prepare()" convention, timed once and reported separately, not
    folded into the timed GEMM call.

    Consumes the workload's SHARED `w.quantize()` codes/scale -- the
    fairness point every quantized-gemm impl must honor, see
    kernelbench/domains/ml.py's QuantGemmWorkload docstring -- and
    dequantizes them INLINE here (`codes_t * scale_t`, a fresh expression,
    not a call to reference_qgemm's or NumpyDequantGemm's own dequantize
    line) so this GPU baseline's dequantize arithmetic stays independent of
    the CPU reference's, per DOMAIN_GUIDE's audit ruling.

    Functional-only smoke-gated on the login node (2026-09-06): 3/3 valid
    against reference_qgemm (qgemm-w4a16-decode-kernel).
    """

    name = "torch-dequant-then-cublas-gemm"

    def prepare(self, workload, params):
        import torch
        dt = _torch_dtype(self.precision)
        w = workload
        # ml._make_activation -- the SAME operand generator reference_qgemm
        # itself calls for A (N(0,1) at the run's own precision).
        A_np = _make_activation(w, params, _np_dtype_of(dt))
        A = torch.as_tensor(A_np).to("cuda", dt)
        codes, scale, group_idx = w.quantize()
        codes_t = torch.as_tensor(codes, dtype=torch.float32, device="cuda")
        scale_t = torch.as_tensor(scale[group_idx], dtype=torch.float32, device="cuda")
        W_dequant = (codes_t * scale_t).to(dt)   # inline dequantize -- see class docstring
        return {"A": A, "W": W_dequant}

    def run(self, h):
        return h["A"] @ h["W"]


# ------------------------------------------------------------ tensor algebra
class TorchMTTKRP(_CudaBase):
    """
    torch.index_add_-based scatter-accumulate MTTKRP -- the GPU counterpart
    of kernelbench/domains/tensor.py's numpy-mttkrp (np.add.at). Builds the
    per-nonzero Khatri-Rao row product via gather + elementwise multiply
    (advanced indexing on GPU tensors), then scatter-adds into the output
    row with index_add_, which -- like np.add.at -- correctly accumulates
    repeated mode indices.

    Functional-only smoke-gated on the login node (2026-09-06): 3/3 valid
    against reference_mttkrp (mttkrp-general-kernel-fp64), after also fixing
    a pre-existing missing `import torch` in run() below (see comment there)
    -- an unconditional NameError never caught before because this class was
    never actually run.
    """

    name = "torch-mttkrp"

    def prepare(self, w, params):
        import torch
        dt = _torch_dtype(self.precision)
        R = int(params.get("R", 32))
        mode = int(params.get("mode", 0)) % w.order
        seed = params.get("seed", w.seed)
        # tensor._make_factors -- the SAME operand generator reference_
        # mttkrp/NumpyMTTKRP themselves call (per-mode U(-1,1), seeded
        # seed+m -- see _make_factors' docstring).
        factors_np = _make_factors(w.shape, R, seed, _np_dtype_of(dt))
        factors = [torch.as_tensor(f).to("cuda", dt) for f in factors_np]
        indices = torch.as_tensor(w.indices, dtype=torch.int64, device="cuda")
        values = torch.as_tensor(w.values, dtype=dt, device="cuda")
        return {"shape": w.shape, "indices": indices, "values": values,
                "factors": factors, "mode": mode, "order": w.order, "R": R}

    def run(self, h):
        # PRE-EXISTING bug, unrelated to the RNG fix above, found only now that
        # this impl is actually gated for the first time: `run()` used
        # `torch.zeros` with no `import torch` in scope, an unconditional
        # NameError on first call. Never caught before because this class was
        # WIRING ONLY (imported for its compile/import check, never run).
        import torch
        indices, values, factors = h["indices"], h["values"], h["factors"]
        mode, order, R = h["mode"], h["order"], h["R"]
        kr = values.unsqueeze(1).expand(-1, R).clone()
        for m in range(order):
            if m == mode:
                continue
            kr = kr * factors[m][indices[:, m]]
        out = torch.zeros((h["shape"][mode], R), dtype=kr.dtype, device="cuda")
        out.index_add_(0, indices[:, mode], kr)
        return out


class TorchEinsumContraction(_CudaBase):
    """
    torch.einsum -- the cuTENSOR/cuBLAS-backed "what you get for free"
    baseline that kernelbench/domains/tensor.py's explicit-transpose
    numpy-tensordot-contraction (and its GPU counterpart below) is
    contrasted against: torch fuses permutation into one opaque call here,
    exactly like the numpy version does.

    Functional-only smoke-gated on the login node (2026-09-06): 3/3 valid
    against reference_tensor_contraction (tccg-tree-kernel-fp32).
    """

    name = "torch-einsum-contraction"

    def prepare(self, w, params):
        import torch
        dt = _torch_dtype(self.precision)
        # ContractionWorkload.operands -- the SAME operand generator
        # reference_tensor_contraction itself calls (rngA at self.seed,
        # rngB at self.seed+1).
        A_np, B_np = w.operands(dtype=_np_dtype_of(dt))
        A = torch.as_tensor(A_np).to("cuda", dt)
        B = torch.as_tensor(B_np).to("cuda", dt)
        return {"equation": w.equation, "A": A, "B": B}

    def run(self, h):
        import torch
        return torch.einsum(h["equation"], h["A"], h["B"])


class TorchTensordotContraction(_CudaBase):
    """
    GPU counterpart of numpy-tensordot-contraction: the same explicit
    permute-to-(free,contracted) / reshape-to-2D / matmul / reshape-back /
    permute-to-output pipeline, in torch. Always includes the permutation
    in-kernel (transpose_in_timing's untimed-hoist branch is not
    replicated here).

    Functional-only smoke-gated on the login node (2026-09-06): 3/3 valid
    against reference_tensor_contraction (tccg-tree-kernel-fp32).
    """

    name = "torch-tensordot-contraction"

    def prepare(self, w, params):
        import torch
        dt = _torch_dtype(self.precision)
        # ContractionWorkload.operands -- same shared generator as
        # TorchEinsumContraction above.
        A_np, B_np = w.operands(dtype=_np_dtype_of(dt))
        A = torch.as_tensor(A_np).to("cuda", dt)
        B = torch.as_tensor(B_np).to("cuda", dt)
        in_a, in_b = w.input_specs
        contracted = w.contracted_letters
        free_a = [l for l in in_a if l not in contracted]
        free_b = [l for l in in_b if l not in contracted]
        axes_a = [in_a.index(l) for l in free_a] + [in_a.index(l) for l in contracted]
        axes_b = [in_b.index(l) for l in contracted] + [in_b.index(l) for l in free_b]
        m = 1
        for l in free_a:
            m *= A.shape[in_a.index(l)]
        n = 1
        for l in free_b:
            n *= B.shape[in_b.index(l)]
        k = 1
        for l in contracted:
            k *= A.shape[in_a.index(l)]
        natural_order = free_a + free_b
        natural_shape = [A.shape[in_a.index(l)] for l in free_a] + \
            [B.shape[in_b.index(l)] for l in free_b]
        out_perm = [natural_order.index(l) for l in w.output_spec]
        return {"A": A, "B": B, "axes_a": axes_a, "axes_b": axes_b,
                "m": m, "n": n, "k": k, "natural_shape": natural_shape,
                "out_perm": out_perm}

    def run(self, h):
        A2d = h["A"].permute(*h["axes_a"]).reshape(h["m"], h["k"])
        B2d = h["B"].permute(*h["axes_b"]).reshape(h["k"], h["n"])
        out = (A2d @ B2d).reshape(*h["natural_shape"])
        return out.permute(*h["out_perm"])
