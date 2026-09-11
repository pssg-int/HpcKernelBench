"""
FlashSparse adapter for the sddmm track.

Paper: "FlashSparse: Minimizing Computation Redundancy for Fast Sparse
Matrix Multiplications on Tensor Cores", PPoPP'25. `PAPER_KEY =
conf/ppopp/ShiLXFWW25`. Artifact: https://github.com/ParCIS/FlashSparse.
`source/` here is a SYMLINK to ../../spmm/flashsparse/source (read-only
reuse of the spmm track's already-cloned checkout, not a re-clone -- see
source.provenance).

Same recency-tiebreak selection rationale as the spmm track's flashsparse
adapter (see its STATUS.md/adapter.py docstring): the most recent (PPoPP'25)
entry in the RoDe/DTC-SpMM/FlashSparse tensor-core-kernel line this track's
`sddmm-tensorcore-blocked-fp16` variant is built around.

## What was wrapped (rule 1)

`FS_SDDMM.forward_gen_fp16_gnn` (source/FlashSparse/SDDMM/src/
{benchmark.cpp,sddmmKernel.cu} -- `sddmm_gen_forward_gat_gnn` ->
`sddmm_gen_forward_cuda_kernel_gat` / `SDDMMKernel4Block`-equivalent tensor-
core dot-product kernel, fp16 compute / fp32 MMA accumulate) is the general
(two-operand: row-indexed `lhs_matrix` + col-indexed `rhs_matrix`) SDDMM
entry point -- NOT the "_gat"-only (self-attention Q.Q^T, single operand)
variant that this same file also exports. Both entry points call the
IDENTICAL compiled `__global__` kernel and Store() logic (confirmed by
reading `sddmm_gen_forward_cuda_gat_gnn`'s body: it launches the exact same
`sddmm_gen_forward_cuda_kernel_gat<<<...>>>` the "_gat" host wrapper does,
just passing two distinct matrices instead of the same one twice) -- so this
is a genuine general SDDMM, not a restricted GAT-only kernel.

Its structural preprocessing, `FS_Block.blockProcess_sddmm_balance`
(source/FlashSparse/Block/example.cpp :: `blockProcess_sddmm`, CPU/OpenMP),
is the ONLY preprocessing path in the artifact that returns the 4 arrays
(row_offsets, col_indices, values, t_window_row) FS_SDDMM's pybind functions
actually take -- `FS_Block_gpu.preprocess_gpu_fs` (used by the spmm track's
adapter) returns only 3 and is not usable here. Both `FS_SDDMM` and
`FS_Block` are built fresh by this directory's own `fs_sddmm_setup.py` (the
spmm adapter's `fs_setup.py` built neither -- see its own header comment).

## The real engineering problem: FS_SDDMM's output is NOT in CSR nnz order

Unlike RoDe's SDDMM kernel (`../rode/adapter.py`), which writes each result
directly at its own CSR position, FS_SDDMM's tensor-core kernel writes into
a position determined by its own HMMA-fragment/row-window tiling -- a
genuine layout-decode problem, same category as `../fused3s/adapter.py`'s
TC-block reindex. `layout_decode.py` in this directory replays FS_SDDMM's
`Store()` index arithmetic (traced line-by-line from unmodified artifact
source -- see that file's module docstring for the full worked derivation,
including the non-obvious residue-tile sub-cases) to build a permutation
from the kernel's raw output layout back to the caller's CSR nnz order.
**The correctness gate passing (see STATUS.md) is the actual evidence this
decode is right** -- a wrong derivation would show up as a large, not a
~1e-2, fp16 error, exactly fused3s/adapter.py's own evidentiary standard.

## Verified: the task's flagged `values=1.0` concern is a non-issue here

See `layout_decode.py`'s module docstring, section "Verified: FS_Block_gpu's
values=1.0-hardcoded limitation is a NON-ISSUE here" for the full trace.
Summary: FS_SDDMM's kernel never multiplies by a caller-supplied edge
weight at all -- `values` (from either FS_Block_gpu OR FS_Block's own
preprocessing) is purely a 0/1 structural validity mask gating which
(row_local, col_local) slots get written, exactly like
`kernelbench.impls.gpu_cuda.TorchSDDMM`'s `torch.sparse.sampled_addmm`
(also unweighted) and fused3s/adapter.py's kernel. This adapter multiplies
the kernel's raw dot-product output by `S[i,j]` itself, in Python, after the
CUDA call returns and after `layout_decode`'s reindex -- same established
pattern.

## Precision

fp16 compute, fp32 MMA accumulate, fp16-stored output (`torch::kFloat16`
allocation in benchmark.cpp) -- intrinsic tensor-core arithmetic, not a
"requested precision" toggle. `PRECISIONS = ["fp16"]`; gated under
`sddmm-tensorcore-blocked-fp16` (1e-2 relative/scaled-error tolerance),
`--precision fp16` explicit (same reason as fused3s/adapter.py: runner.py's
implicit precision resolution would otherwise silently pick fp32 for this
variant id and fail `create()`).

## Shape / dimension constraints (rule 8)

`FS_Block.blockProcess_sddmm` computes `rowsNew = rows/window` via integer
division (`window=8`) -- M is padded to a multiple of 8 here (same
discipline as the spmm track's flashsparse adapter and dtcspmm/inferfast).
No square-matrix constraint (unlike fused3s): `lhs_matrix`/`rhs_matrix` are
independently shaped (M_pad x K) / (N x K), col_indices holding real
(unpadded) column ids 0..N-1 directly.

Dense operands: A (row-indexed, M x K) seed 42, B (col-indexed, N x K) seed
43, generated in float64 first (matching `cpu_ref.reference_sddmm`'s
`_dense_operand` exactly) then cast to fp16 for the CUDA call -- the gate
therefore measures both fp16 quantization of the inputs AND fp16-stored
accumulation rounding against the true fp64 reference, same as fused3s.
"""

from __future__ import annotations

import os
import sys

import numpy as np

KERNEL = "sddmm"
IMPL_NAME = "flashsparse-sddmm"
PAPER_KEY = "conf/ppopp/ShiLXFWW25"
PRECISIONS = ["fp16"]

_HERE = os.path.dirname(os.path.abspath(__file__))
_BUILD_LIB = os.path.join(_HERE, "build", "lib")

BLK_H = 8      # window size (FS_Block.blockProcess_sddmm's "window")
BLK_W = 16     # column tile width ("wide")
PART = 32      # load-balance split threshold ("partSize_t"/"maxPart") --
               # matches the artifact's own test convention (SDDMM/test/
               # mgcn16/mdataset_fp16.py: blockProcess_sddmm_balance(...,8,16,32))

_fs_sddmm = None
_fs_block = None


def _load_modules():
    global _fs_sddmm, _fs_block
    if _fs_sddmm is not None:
        return _fs_sddmm, _fs_block
    if not os.path.isdir(_BUILD_LIB):
        raise FileNotFoundError(f"{_BUILD_LIB} not built -- run build.sh")
    if _BUILD_LIB not in sys.path:
        sys.path.insert(0, _BUILD_LIB)
    import FS_SDDMM as _sddmm
    import FS_Block as _block
    _fs_sddmm, _fs_block = _sddmm, _block
    return _fs_sddmm, _fs_block


def available() -> tuple[bool, str]:
    if not os.path.isdir(_BUILD_LIB):
        return False, "build/lib not built -- run build.sh"
    try:
        import torch
    except Exception as e:
        return False, f"torch import failed: {type(e).__name__}: {e}"
    if not torch.cuda.is_available():
        return False, "CUDA not available on this host"
    try:
        _load_modules()
    except Exception as e:
        return False, f"FS_SDDMM/FS_Block import failed: {type(e).__name__}: {e}"
    return True, ""


def create(precision: str):
    return FlashSparseSDDMM(precision)


class FlashSparseSDDMM:
    name = IMPL_NAME
    platform = "cuda"

    def __init__(self, precision: str = "fp16"):
        if precision != "fp16":
            raise NotImplementedError(
                f"{IMPL_NAME} is wired for FS_SDDMM's fp16 HMMA kernel only "
                f"(a forward_gen_tf32_gnn entry point also exists in the "
                f"artifact but is not wired here); requested {precision}")
        self.precision = precision

    def prepare(self, matrix, params: dict):
        import torch
        # adapter.py is loaded standalone (importlib.util.spec_from_file_
        # location, see artifact_registry.py) with no package context, so
        # layout_decode is imported by path via sys.path -- same pattern
        # fused3s/adapter.py uses for its F3S extension module.
        if _HERE not in sys.path:
            sys.path.insert(0, _HERE)
        import layout_decode as ld

        fs_sddmm, fs_block = _load_modules()

        S = matrix.csr
        M, N = S.shape
        K = int(params["K"])

        m_pad = ((M + BLK_H - 1) // BLK_H) * BLK_H
        indptr_pad = np.empty(m_pad + 1, dtype=np.int32)
        indptr_pad[: M + 1] = S.indptr
        indptr_pad[M + 1:] = S.indptr[-1]  # empty trailing (padding) rows

        row_ptr = torch.from_numpy(indptr_pad)
        col_idx = torch.from_numpy(np.ascontiguousarray(S.indices, dtype=np.int32))

        # --- artifact's own structural preprocessing (CPU/OpenMP): CSR ->
        # FlashSparse's 8-row-window / 16-wide-tile load-balanced SDDMM
        # format, timed as preprocessing per the harness contract. ---
        row_offsets_t, col_indices_t, values_t, t_window_row_t = fs_block.blockProcess_sddmm_balance(
            row_ptr, col_idx, BLK_H, BLK_W, PART)

        # Our own decode: replay FS_SDDMM's Store() index arithmetic to find
        # where every one of S's real nonzeros will land in the kernel's
        # raw (non-CSR-ordered) output -- see layout_decode.py. Purely
        # structural (pattern-only), independent of any dense-operand
        # values, so it is computed once here alongside the artifact's own
        # preprocessing.
        row_offsets_np = row_offsets_t.numpy()
        col_indices_np = col_indices_t.numpy()
        t_window_row_np = t_window_row_t.numpy()
        perm = ld.build_csr_to_output_perm(
            row_offsets_np, col_indices_np, t_window_row_np,
            S.indptr, S.indices, N, output_size=int(values_t.numel()))

        # numpy RNG matching cpu_ref.reference_sddmm's _dense_operand
        # exactly: A row-indexed (M,K) seed, B col-indexed (N,K) seed+1.
        seed = params.get("seed", 42)
        rng_a = np.random.default_rng(seed)
        rng_b = np.random.default_rng(seed + 1)
        A_np = rng_a.uniform(-1.0, 1.0, size=(m_pad, K)).astype(np.float64)
        B_np = rng_b.uniform(-1.0, 1.0, size=(N, K)).astype(np.float64)
        A = torch.as_tensor(A_np, dtype=torch.float16, device="cuda").contiguous()
        B = torch.as_tensor(B_np, dtype=torch.float16, device="cuda").contiguous()

        # to_host() below only receives run()'s return value, not this
        # handle, so stash perm + S's values on self here; prepare()/run()/
        # to_host() are always called on the same instance within one
        # run_variant() call (matches the fused3s/rassm adapters' pattern).
        self._last_perm = perm
        self._last_s_values = np.ascontiguousarray(S.data, dtype=np.float64)

        return {
            "row_offsets": row_offsets_t.to("cuda", dtype=torch.int32).contiguous(),
            "col_indices": col_indices_t.to("cuda", dtype=torch.int32).contiguous(),
            "values": values_t.to("cuda", dtype=torch.int32).contiguous(),
            "t_window_row": t_window_row_t.to("cuda", dtype=torch.int32).contiguous(),
            "A": A, "B": B, "K": K,
        }

    def run(self, h):
        fs_sddmm, _ = _load_modules()
        out, = fs_sddmm.forward_gen_fp16_gnn(
            h["K"], h["row_offsets"], h["col_indices"], h["values"],
            h["t_window_row"], h["A"], h["B"], PART)
        return out

    def to_host(self, out):
        import torch
        # out: raw fp16 kernel output, flat (values.size(0),) tensor, in
        # FS_SDDMM's own row-window/tile layout (see layout_decode.py).
        # Reindex to the caller's original CSR nnz order via the perm
        # computed in prepare(), then apply S[i,j] (module docstring).
        raw = out.detach().to("cpu", dtype=torch.float64).numpy()
        gathered = raw[self._last_perm]
        return gathered * self._last_s_values

    def free(self, h):
        h.clear()

    def timer(self):
        from kernelbench.impls.gpu_cuda import CudaEventTimer
        return CudaEventTimer()
