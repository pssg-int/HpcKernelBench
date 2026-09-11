"""
InferFast adapter for the spmm track.

Paper: "InferFast: Bridging the Gap Between Unstructured LLM Sparsity and
Practical GPU Throughput", ICS'26. `PAPER_KEY = conf/ics/ShenBSCH26`
(matched by title in `../../output/included.json`).
Artifact: https://github.com/MLsys-HPC/InferFast (a Flash-LLM/SpInfer-lineage
bitmap-tiled Tensor-Core SpMM kernel for unstructured-sparse fp16 weights).

Build: see build.sh + wrapper.cu. `source/build/libSpMM_API.so` is the
artifact's own unmodified library (`InferFast_SpMM_SplitK_API`,
`InferFast_InitSparseMatrixA`, `reorder_matrices` -- all its own code,
untouched). `libinferfast_wrapper.so` (this directory's `wrapper.cu`, NOT
under source/) only adds extern-"C" linkage and explicit host<->device
buffer-size plumbing so ctypes can call it; it re-links the artifact's
already-compiled symbols, it does not reimplement them.

Kernel entry point wrapped (ARTIFACT_GUIDE.md rule 1: wrap the kernel, not
a benchmark script): `InferFast_SpMM_SplitK_API`, called with split_k=1 --
ONE SpMM kernel launch, no reduction pass.

prepare() does the artifact's own format conversion (rule 2, timed as
preprocessing): our workload's CSR is densified (the artifact's compression
function -- InferFast_InitSparseMatrixA -- takes a dense host matrix and
finds its own nonzero/bitmap structure; there is no CSR-input path in this
artifact) and zero-padded to the artifact's fixed tile grid (M % 128 == 0,
K % 64 == 0 -- InferFast_InitSparseMatrixA silently drops any non-tile-
aligned tail via integer division, see wrapper.cu's header comment), then
`reorder_matrices` + `InferFast_InitSparseMatrixA` (both the artifact's own
C++, unmodified) compress it into its bitmap/tile-offset/reorder format,
which is then H2D-copied.

PRECISION: fp16 only -- the artifact's kernel operates on `half` throughout
(Tensor Core MMA), there is no fp32 path to wrap honestly, so
`PRECISIONS = ["fp16"]` (ARTIFACT_GUIDE.md: "what the artifact actually
supports"), not the fp32 the ARTIFACT_GUIDE.md example line and the task's
generic verification template default to. The spmm spec's variants are all
fp32/int (`spmm-gpu-kernel-f32`, `...-e2e-preproc-f32`, `...-cpu-kernel-f32`,
`...-gpu-quantized-int`) -- none is an fp16 variant -- so gating this adapter
requires `--precision fp16` on the runner CLI alongside
`--variant spmm-gpu-kernel-f32` (the CLI's `--precision` flag overrides the
variant-name-derived default and this is not a spec violation, just a
precision selection the spec's variant set doesn't happen to enumerate).
See STATUS.md for the actual gate result at fp16.
"""

from __future__ import annotations

import ctypes
import os

import numpy as np

KERNEL = "spmm"
IMPL_NAME = "inferfast-spmm-splitk"
PAPER_KEY = "conf/ics/ShenBSCH26"
PRECISIONS = ["fp16"]

_HERE = os.path.dirname(os.path.abspath(__file__))
_LIB_PATH = os.path.join(_HERE, "libinferfast_wrapper.so")

_lib = None


def _load_lib():
    global _lib
    if _lib is not None:
        return _lib
    if not os.path.exists(_LIB_PATH):
        raise FileNotFoundError(f"{_LIB_PATH} not built -- run build.sh")
    lib = ctypes.CDLL(_LIB_PATH)
    p = ctypes.c_void_p
    lib.inferfast_compress.argtypes = [p, ctypes.c_int, ctypes.c_int]
    lib.inferfast_compress.restype = p
    lib.inferfast_compress_sizes.argtypes = [p] + [ctypes.POINTER(ctypes.c_int)] * 5
    lib.inferfast_compress_sizes.restype = None
    lib.inferfast_copy_out.argtypes = [p, p, p, p, p, p]
    lib.inferfast_copy_out.restype = None
    lib.inferfast_free_compress.argtypes = [p]
    lib.inferfast_free_compress.restype = None
    lib.inferfast_run.argtypes = [p, p, p, p, p, p, ctypes.c_int, p, p,
                                   ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int]
    lib.inferfast_run.restype = ctypes.c_int  # cudaError_t
    _lib = lib
    return _lib


def available() -> tuple[bool, str]:
    if not os.path.exists(_LIB_PATH):
        return False, "libinferfast_wrapper.so not built -- run build.sh"
    if not os.path.exists(os.path.join(_HERE, "source", "build", "libSpMM_API.so")):
        return False, "source/build/libSpMM_API.so not built -- run build.sh"
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
    return InferFastSpMM(precision)


def _round_up(x: int, m: int) -> int:
    return ((x + m - 1) // m) * m


class InferFastSpMM:
    name = IMPL_NAME
    platform = "cuda"

    def __init__(self, precision: str = "fp16"):
        if precision != "fp16":
            raise NotImplementedError(
                f"{IMPL_NAME} only wired for fp16 (Tensor Core `half` kernel, "
                f"no fp32 path in the artifact); requested {precision}")
        self.precision = precision
        self.split_k = 1

    def prepare(self, matrix, params: dict):
        import torch
        lib = _load_lib()

        M, K = matrix.csr.shape
        N = int(params["N"])
        Mp = _round_up(M, 128)
        Kp = _round_up(K, 64)

        # --- artifact's own format conversion, timed as preprocessing ---
        # 1) densify + zero-pad to the artifact's fixed tile grid (its own
        #    InitSparseMatrixA has no CSR-input path and silently drops any
        #    non-tile-aligned tail -- padding here, not inside the artifact,
        #    keeps that behavior visible instead of hidden).
        A_dense = np.zeros((Mp, Kp), dtype=np.float16)
        csr = matrix.csr
        for i in range(M):
            lo, hi = csr.indptr[i], csr.indptr[i + 1]
            A_dense[i, csr.indices[lo:hi]] = csr.data[lo:hi].astype(np.float16)

        # 2) the artifact's own host-side compression (reorder_matrices +
        #    InferFast_InitSparseMatrixA, both unmodified -- see wrapper.cu).
        A_ptr = A_dense.ctypes.data_as(ctypes.c_void_p)
        handle = lib.inferfast_compress(A_ptr, ctypes.c_int(Mp), ctypes.c_int(Kp))
        try:
            num_gtiles = ctypes.c_int()
            num_ltiles = ctypes.c_int()
            val_count = ctypes.c_int()
            max_nnz_intile = ctypes.c_int()
            reorder_len = ctypes.c_int()
            lib.inferfast_compress_sizes(
                handle, ctypes.byref(num_gtiles), ctypes.byref(num_ltiles),
                ctypes.byref(val_count), ctypes.byref(max_nnz_intile),
                ctypes.byref(reorder_len))

            compressed_val = np.empty(val_count.value, dtype=np.float16)
            tile_off_median = np.empty(num_ltiles.value, dtype=np.uint16)
            tile_off_global = np.empty(num_gtiles.value + 1, dtype=np.int32)
            bitmap = np.empty(num_ltiles.value, dtype=np.uint32)
            reorder = np.empty(reorder_len.value, dtype=np.uint16)
            lib.inferfast_copy_out(
                handle,
                compressed_val.ctypes.data_as(ctypes.c_void_p),
                tile_off_median.ctypes.data_as(ctypes.c_void_p),
                tile_off_global.ctypes.data_as(ctypes.c_void_p),
                bitmap.ctypes.data_as(ctypes.c_void_p),
                reorder.ctypes.data_as(ctypes.c_void_p))
        finally:
            lib.inferfast_free_compress(handle)

        # 3) H2D upload -- torch tensors so ctypes can hand raw device
        #    pointers to inferfast_run(), same convention as
        #    kernelbench/impls/gpu_cuda.py's CustomSpMM.
        # torch has no unsigned 16/32-bit dtype; .view() bit-reinterprets
        # (zero-copy, no value cast) rather than converting, so the raw
        # uint16_t/uint32_t bit patterns InferFast's own compression wrote
        # reach the device unchanged -- the kernel reads them back as
        # unsigned on the C++ side regardless of torch's signed dtype label.
        compressed_val_dev = torch.from_numpy(compressed_val).cuda()
        tile_off_median_dev = torch.from_numpy(tile_off_median.view(np.int16)).cuda()
        tile_off_global_dev = torch.from_numpy(tile_off_global).cuda()
        bitmap_dev = torch.from_numpy(bitmap.view(np.int32)).cuda()
        reorder_dev = torch.from_numpy(reorder.view(np.int16)).cuda()
        max_nnz_dev = torch.tensor([max_nnz_intile.value], dtype=torch.int32, device="cuda")

        # B: numpy RNG matching kernelbench.impls.cpu_ref.reference_spmm's
        # _dense_operand exactly (see insum/adapter.py's docstring for why
        # this matters: gpu_cuda.py's torch-RNG `_dense` helper draws a
        # DIFFERENT B than the numpy-RNG reference the correctness gate
        # compares against -- reproduced independently on the harness's own
        # built-in cusparse-csr-spmm while building the insum adapter).
        # Real K rows get the reference draw; the K..Kp-1 padding rows are
        # zero (they only ever multiply against A's zero-padded columns, so
        # their value is mathematically irrelevant -- zero for cleanliness).
        rng = np.random.default_rng(params.get("seed", 42))
        B_np = np.zeros((Kp, N), dtype=np.float16)
        B_np[:K, :] = rng.uniform(-1.0, 1.0, size=(K, N)).astype(np.float16)
        B_dev = torch.from_numpy(B_np).cuda()

        C_dev = torch.zeros(Mp * N, dtype=torch.float16, device="cuda")
        # to_host() needs (Mp, N, M) but the harness's Implementation
        # protocol only passes `out` (the tensor) to to_host(), not the
        # handle -- stash it as instance state, set here (prepare() is the
        # only place these are known) and read there. `impl` is the same
        # object across prepare()/run()/to_host() within one run_variant()
        # call, so this is safe (not shared across concurrent runs).
        self._shape = (Mp, N, M)

        return {
            "compressed_val": compressed_val_dev, "tile_off_median": tile_off_median_dev,
            "tile_off_global": tile_off_global_dev, "bitmap": bitmap_dev,
            "reorder": reorder_dev, "max_nnz_dev": max_nnz_dev,
            "max_nnz_cpu": max_nnz_intile.value, "B": B_dev, "C": C_dev,
            "M": M, "Mp": Mp, "N": N, "K": K, "Kp": Kp,
        }

    def run(self, h):
        lib = _load_lib()
        h["C"].zero_()
        err = lib.inferfast_run(
            ctypes.c_void_p(h["compressed_val"].data_ptr()),
            ctypes.c_void_p(h["tile_off_global"].data_ptr()),
            ctypes.c_void_p(h["tile_off_median"].data_ptr()),
            ctypes.c_void_p(h["reorder"].data_ptr()),
            ctypes.c_void_p(h["bitmap"].data_ptr()),
            ctypes.c_void_p(h["max_nnz_dev"].data_ptr()),
            ctypes.c_int(h["max_nnz_cpu"]),
            ctypes.c_void_p(h["B"].data_ptr()),
            ctypes.c_void_p(h["C"].data_ptr()),
            ctypes.c_int(h["Mp"]), ctypes.c_int(h["N"]), ctypes.c_int(h["Kp"]),
            ctypes.c_int(self.split_k))
        if err != 0:
            raise RuntimeError(f"InferFast_SpMM_SplitK_API returned cudaError_t={err}")
        return h["C"]

    def to_host(self, out):
        import torch
        # InferFast writes C COLUMN-MAJOR (Mp, N) -- see wrapper.cu's
        # inferfast_run docstring, traced to
        # source/csrc/SpMM_Kernel.cuh's `BlockGlobalPTR[j + i * M_Global]`
        # store. A flat (Mp*N,) buffer in that layout is exactly a
        # row-major (N, Mp) array, so view+transpose, then crop the tile
        # padding (Mp -> true M) that prepare() added.
        Mp, N, M = self._shape
        mat = out.detach().to("cpu", dtype=torch.float64).view(N, Mp).t().contiguous()
        return mat[:M, :].numpy()

    def timer(self):
        from kernelbench.impls.gpu_cuda import CudaEventTimer
        return CudaEventTimer()

    def free(self, h):
        import torch
        h.clear()
        torch.cuda.empty_cache()
