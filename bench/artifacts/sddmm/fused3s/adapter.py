"""
Adapter for Fused3S (Fused3S: Fast Sparse Attention on Tensor Cores,
ICS'25, HPCForge/Fused3S) — GPU tensor-core SDDMM stage of a fused sparse-
attention pipeline.

Fused3S fuses SDDMM (Q.K^T at the sparsity pattern) + softmax + SpMM (.V)
into one kernel family for graph/sparse attention. The paper's ablation
kernel `f3s_1tb1tcb` (source/src/F3S_kernel.cu: f3sKernel1tb1tcb, exposed to
Python as F3S.f3s_1tb1tcb in source/src/F3S.cpp) is the ONE variant that can
optionally write out the raw SDDMM stage result (`saveSddmmResult=True`)
before softmax/SpMM run — every other, more-heavily-optimized kernel variant
in this repo (f3s_1tb1rw*) has that same instrumentation commented out for
performance. So this adapter wraps f3s_1tb1tcb with saveSddmmResult=True,
applySoftmax=False, and returns ONLY the sddmmResult tensor — the SpMM/
softmax stages still run inside the same CUDA call (this kernel is a single
fused launch, no separable sub-kernel boundary exists below the CUDA-call
level) but their outputs are simply discarded. This is the boundary
documented in STATUS.md ("wrap the SDDMM STAGE if separable, else record
what boundary you wrapped").

Two adapter-side responsibilities beyond "call the kernel":

1. Fused3S's own accuracy test (source/scripts/tests/test_f3s_accuracy.py)
   rounds the sparsity pattern's nonzero values to 1 before comparing
   (`A_csr_h.data = np.ceil(A_csr_h.data, dtype=np.float32)`) — i.e. the
   kernel computes an UNWEIGHTED masked dot product dot(Q_i,K_j) at edges,
   never multiplying by an edge weight. This is the same situation as
   kernelbench.impls.gpu_cuda.TorchSDDMM (torch.sparse.sampled_addmm is
   also unweighted, `* h["Sdata"]` afterward) — so this adapter multiplies
   the kernel's raw dot-product output by S[i,j] itself, in Python, after
   the CUDA call returns, exactly mirroring that established pattern.

2. sddmmResult is laid out in the kernel's own tensor-core-fragment order,
   NOT the caller's CSR nnz order — see _decode_layout()'s docstring for the
   full derivation (cross-checked from utils.cu's bitmap WRITE side and
   F3S_kernel.cu's addPartialSums/saveSddmmResult READ side).

The kernel only accepts fp16 Q/K (torch::Half) with fp32 accumulation — that
is the tensor-core arithmetic path, not a "requested precision" toggle — so
this adapter only services precision="fp16" and is meant to be run against
the sddmm-tensorcore-blocked-fp16 spec variant (1e-2 relative-error gate),
NOT the fp32-CSR variant's 1e-4 gate. See STATUS.md.

Fused3S's preprocess_gpu/kernel are written for SQUARE adjacency-style
sparsity patterns (one `numNodes` controls both Q's and K's row count) — this
adapter only supports square (M==N) matrices; prepare() raises otherwise.
"""

from __future__ import annotations

import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
SRC_DIR = os.path.join(HERE, "source", "src")

KERNEL = "sddmm"
IMPL_NAME = "fused3s-1tb1tcb-sddmm"
PAPER_KEY = "conf/ics/LiC25"
# The kernel's tensor-core arithmetic is fp16-in/fp32-accumulate; there is no
# separate fp32 (or fp64) compute path in the artifact, so only "fp16" is
# serviced (see module docstring).
PRECISIONS = ["fp16"]

BLK_H = 16   # config.h
BLK_N = 8    # config.h BLK_W / F3S_kernel.cu BLK_N (same tile geometry)

# NOTE on a workaround that is NOT here: an earlier version of build.sh let
# distutils bake `-rpath .../nersc-python/lib` into the extension, which
# shadows the system libstdc++.so.6 with an old one missing a GCC-8+ symbol
# the extension needs. Working around that at import time by dlopen'ing the
# correct libstdc++ with RTLD_GLOBAL (an LD_PRELOAD equivalent, since
# os.environ["LD_PRELOAD"] set from an already-running interpreter is a
# no-op) SEGFAULTS -- two different libstdc++ copies loaded into one process
# is an ABI hazard, not a fix. The real fix lives in build.sh: it overrides
# LDSHARED so the link step never bakes in that rpath in the first place, so
# nothing special is needed here.

_F3S = None
_IMPORT_ERROR = ""


def _load():
    global _F3S, _IMPORT_ERROR
    if _F3S is not None:
        return _F3S
    if _IMPORT_ERROR:
        raise RuntimeError(_IMPORT_ERROR)
    try:
        import sys
        if SRC_DIR not in sys.path:
            sys.path.insert(0, SRC_DIR)
        import F3S  # noqa: N814 -- matches the extension's actual module name
        _F3S = F3S
        return _F3S
    except Exception as e:
        _IMPORT_ERROR = f"{type(e).__name__}: {e}"
        raise


def available() -> tuple[bool, str]:
    so_present = any(f.startswith("F3S.cpython") and f.endswith(".so")
                      for f in os.listdir(SRC_DIR)) if os.path.isdir(SRC_DIR) else False
    if not so_present:
        return False, "F3S*.so not built — run build.sh"
    try:
        _load()
        import torch
        if not torch.cuda.is_available():
            return False, "no CUDA device visible"
        return True, ""
    except Exception as e:
        msg = f"{type(e).__name__}: {e}"
        if "CXXABI" in msg or "GLIBCXX" in msg:
            msg = ("libstdc++ ABI mismatch — run with "
                   "LD_PRELOAD=/usr/lib64/libstdc++.so.6 (see STATUS.md)")
        return False, msg


def _dense_operand(rows: int, dim: int, seed: int, dtype) -> np.ndarray:
    # identical formula to kernelbench.impls.cpu_ref._dense_operand — kept
    # as a separate copy (not a shared code object) per the
    # reference-independence rule in DOMAIN_GUIDE.md.
    rng = np.random.default_rng(seed)
    return rng.uniform(-1.0, 1.0, size=(rows, dim)).astype(dtype)


def _decode_layout(row_window_offset_np, sparse_a_to_x_index_np, tcb_bitmap_np, n_tcb):
    """
    Map every flat position of sddmmResult (length n_tcb*BLK_H*BLK_N) to
    (row, col, valid).

    Derivation (cross-checking the preprocessing WRITE side against the
    kernel READ side, both read from unmodified artifact source):

    WRITE side — utils.cu's update_bitmap(bitmap, tcblock_id,
    n_sub_blocks_per_tcblock=2, blockSize_h, blockSize_w, row_local,
    col_local), called as
      update_bitmap(tcblock_bit_map, tcblock_id, 2, blockSize_w, blockSize_h,
                    row_local, col_local)
    -- note blockSize_w/blockSize_h are SWAPPED at the call site relative to
    the function's own parameter names, which (given blockSize_h=16,
    blockSize_w=8 in config.h) works out to: sub_block_row_id = row_local//8
    (0 or 1), sub_block_col_id = 0 always (col_local<8), so word index =
    tcblock_id*2 + (row_local//8), bit = 63 - ((row_local%8)*8 + col_local).
    Also (from the sibling line in generate_tcoffset_id_atob_kernel):
      sparse_AToB[tcblock_id * blockSize_w + col_local] = <real column>
    i.e. SparseAToXindex[tcb*BLK_N + col_local] is the real column for slot
    (tcb, col_local), independent of row_local.

    READ side — F3S_kernel.cu's addPartialSums(sum, tcbId, ...) walks flat
    index `ind` in [0,256) for a PAIR of TC blocks (tcbId, tcbId+1) and reads
    TCblock_bit_map[tcbId*2 + ind/64], bit 63-(ind%64); saveSddmmResult
    writes sddmm_result[tcbId*128 + i*128 + j*64 + laneId*2 + k] using the
    exact same i/j/laneId/k decomposition, confirming the *local* index
    within ONE TC block (0..127) is: word_off = local//64, bit_off =
    local%64, i.e. IDENTICAL to the write side's (row_local//8, (row_local%8
    )*8+col_local). Inverting: row_local = (local//64)*8 + (local%64)//8,
    col_local = local%8.

    The row-window id for a TC block `t` is NOT emitted anywhere for
    f3s1tb1tcb (its grid is `dim3(nRowWindow,1,1)`, so `bid=blockIdx.x` IS
    the row window, and the kernel just walks
    rowWindowOffset[bid]:rowWindowOffset[bid+1]) — recovered here the same
    way the kernel does, via searchsorted on rowWindowOffset (a standard CSR
    row-pointer lookup).
    """
    local = np.arange(BLK_H * BLK_N, dtype=np.int64)              # 0..127
    row_local = (local // 64) * 8 + (local % 64) // 8              # (128,)
    col_local = (local % 8)                                        # (128,)
    word_off = (local // 64).astype(np.int64)                      # (128,) in {0,1}
    bit_off = (local % 64).astype(np.int64)                        # (128,)
    bitmask = (np.uint64(1) << (63 - bit_off).astype(np.uint64))   # (128,)

    tcb = np.arange(n_tcb, dtype=np.int64)
    bid = np.searchsorted(row_window_offset_np, tcb, side="right").astype(np.int64) - 1

    row = (bid[:, None] * BLK_H + row_local[None, :]).reshape(-1)
    col_per_tcb = sparse_a_to_x_index_np.reshape(n_tcb, BLK_N).astype(np.int64)
    col = col_per_tcb[:, col_local].reshape(-1)

    word_idx = (tcb[:, None] * 2 + word_off[None, :]).reshape(-1)
    words = tcb_bitmap_np[word_idx]
    valid = (words & np.tile(bitmask, n_tcb)) != 0

    return row, col, valid


class Fused3SSDDMM:
    name = "fused3s-1tb1tcb-sddmm"
    platform = "cuda"

    def __init__(self, precision: str = "fp16"):
        if precision != "fp16":
            raise NotImplementedError(
                f"fused3s-1tb1tcb-sddmm only computes in fp16/fp32-accumulate "
                f"tensor-core arithmetic; requested precision {precision!r} "
                "is not serviced (run with --precision fp16 against the "
                "sddmm-tensorcore-blocked-fp16 variant, see STATUS.md)")
        self.precision = precision

    def prepare(self, matrix, params):
        import torch
        F3S = _load()
        S = matrix.csr
        M, N = S.shape
        if M != N:
            raise NotImplementedError(
                "Fused3S's preprocess_gpu/kernel are written for square "
                f"adjacency-style patterns (one numNodes); got {M}x{N}")
        K = int(params["K"])
        seed = params.get("seed", 42)

        # true (unrounded) operands, bit-identical to what
        # kernelbench.impls.cpu_ref.reference_sddmm uses for precision!="fp32"
        # (float64, no rounding) -- this adapter rounds its OWN copies to
        # fp16 for the tensor-core call, so the gate measures fp16
        # quantization + fp16 MMA rounding against the true fp64 values,
        # which is what the tensorcore-blocked-fp16 variant's tolerance is
        # calibrated for.
        A = _dense_operand(M, K, seed, np.float64)       # row-indexed ("Q")
        B = _dense_operand(N, K, seed + 1, np.float64)    # col-indexed ("K")
        Q_half = torch.as_tensor(A, dtype=torch.float16, device="cuda").contiguous()
        K_half = torch.as_tensor(B, dtype=torch.float16, device="cuda").contiguous()
        V_half = torch.zeros((N, K), dtype=torch.float16, device="cuda")  # unused by the SDDMM stage

        num_row_windows = (M + BLK_H - 1) // BLK_H
        indices = torch.as_tensor(S.indices, dtype=torch.int32, device="cuda").contiguous()
        indptr = torch.as_tensor(S.indptr, dtype=torch.int32, device="cuda").contiguous()
        block_partition = torch.zeros(num_row_windows, dtype=torch.int32, device="cuda")
        edge_to_column = torch.zeros(int(S.nnz), dtype=torch.int32, device="cuda")
        edge_to_row = torch.zeros(int(S.nnz), dtype=torch.int32, device="cuda")

        (row_window_offset, _sorted_row_windows, _tcblock_rowid, _tcblocktile_id,
         _tcblock_offset, sparse_a_to_x_index, tcb_bitmap, _n_tcb_reported) = F3S.preprocess_gpu(
            indices, indptr, M, BLK_H, BLK_N, block_partition, edge_to_column, edge_to_row)

        n_tcb = sparse_a_to_x_index.shape[0] // BLK_N

        # structural (value-independent) reindex from the kernel's TC-block
        # layout back to S's original nnz order -- computed once here since
        # it only depends on the sparsity pattern, same as RASSM's adapter.
        row_np, col_np, valid_np = _decode_layout(
            row_window_offset.cpu().numpy().astype(np.int64),
            sparse_a_to_x_index.cpu().numpy(),
            tcb_bitmap.cpu().numpy(),
            n_tcb)
        valid_idx = np.nonzero(valid_np)[0]
        valid_key = row_np[valid_idx].astype(np.int64) * np.int64(N) + col_np[valid_idx].astype(np.int64)

        row_idx = np.repeat(np.arange(M, dtype=np.int64), np.diff(S.indptr))
        orig_key = row_idx * np.int64(N) + S.indices.astype(np.int64)

        order = np.argsort(valid_key, kind="stable")
        sorted_key = valid_key[order]
        pos = np.searchsorted(sorted_key, orig_key)
        if len(orig_key) and (pos >= len(sorted_key)).any():
            raise RuntimeError("Fused3S layout decode: reindex out of range — pattern mismatch")
        perm = valid_idx[order[pos]]
        if len(orig_key) and not np.array_equal(sorted_key[pos], orig_key):
            raise RuntimeError(
                "Fused3S layout decode: (row,col) pattern mismatch between "
                "the decoded TC-block layout and the input CSR — reindexing "
                "would silently misalign the correctness gate")

        # to_host() below only receives run()'s return value, not this
        # handle, so stash perm + S's values on self here; prepare()/run()/
        # to_host() are always called on the same instance within one
        # run_variant() call (matches the rassm adapter's pattern).
        self._last_perm = perm
        self._last_s_values = np.ascontiguousarray(S.data, dtype=np.float64)

        return {
            "F3S": F3S, "row_window_offset": row_window_offset,
            "sparse_a_to_x_index": sparse_a_to_x_index, "tcb_bitmap": tcb_bitmap,
            "M": M, "Q": Q_half, "K": K_half, "V": V_half,
        }

    def run(self, h):
        F3S = h["F3S"]
        time_t, output, sddmm_result = F3S.f3s_1tb1tcb(
            h["row_window_offset"], h["sparse_a_to_x_index"], h["tcb_bitmap"],
            h["M"], h["Q"], h["K"], h["V"],
            False,   # applySoftmax -- we only want the raw SDDMM stage
            True)    # saveSddmmResult
        del output, time_t  # softmax+SpMM output/time not part of this track's kernel
        return sddmm_result

    def to_host(self, out):
        import torch
        # out: sddmmResult, flat (n_tcb*128,) float32 CUDA tensor, in the
        # kernel's TC-block layout. Reindex to the caller's original CSR nnz
        # order, then apply S[i,j] (see module docstring point 1).
        raw = out.detach().to("cpu", dtype=torch.float64).numpy()
        gathered = raw[self._last_perm]
        return gathered * self._last_s_values

    def timer(self):
        from kernelbench.impls.gpu_cuda import CudaEventTimer
        return CudaEventTimer()

    def free(self, h):
        import torch
        h.clear()
        torch.cuda.empty_cache()


def create(precision: str):
    return Fused3SSDDMM(precision)
