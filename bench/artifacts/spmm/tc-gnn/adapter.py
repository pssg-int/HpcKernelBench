"""
Adapter for TC-GNN (TC-GNN: Bridging Sparse GNN Computation and Dense Tensor
Cores on GPUs, USENIX ATC'23, YukeWang96/TC-GNN_ATC23) -- spmm track.

REUSE, NOT A REBUILD (task brief): this artifact is already built for the
gnn-aggregation track at `../../gnn-aggregation/tc-gnn/` (`build.sh`
compiled `source/TCGNN_conv/{TCGNN.cpp,TCGNN_kernel.cu}`, unmodified, into
that directory's `build/TCGNN*.so`). See that STATUS.md for the full kernel
read-through; this adapter reuses the SAME entry point (`forward_AGNN`) and
the SAME two real, confirmed artifact bugs it documents (guarded here
identically, not re-derived) -- only the reference differs: spmm's own
reference is plain `C = A @ B` with the workload's own arbitrary CSR
values, no GCN normalization step.

## What boundary was wrapped (same choice as gnn-aggregation, same reason)

`TCGNN.forward` (`spmm_forward`, `source/TCGNN_conv/TCGNN.cpp`) is
UNWEIGHTED -- every populated cell of its shared-memory sparse tile is
hardcoded to `1` in `TCGNN_kernel.cu`'s `spmm_forward_cuda_kernel`
(confirmed by reading the kernel: no value array anywhere in
`spmm_forward`'s signature at all). This track's kernel boundary is a
GENUINE weighted SpMM (`spmm-tensorcore-fp16`'s / `spmm-gpu-kernel-f32`'s
own reference, `reference_spmm`, uses an arbitrary random CSR `A`, not a
0/1 pattern) -- so this adapter uses `forward_AGNN` instead, TC-GNN's own
weighted WMMA SpMM entry point that DOES accept a real per-edge value
array (`edgeAttention`, confirmed:
`sparse_A[row_local*BLK_W+col_local] = edgeAttention[eIdx];`), fed here
with the workload's own `matrix.csr.data` directly (cast to fp32) rather
than a GCN-normalized value. This is the SAME entry point, SAME
preprocessing pipeline (`TCGNN.preprocess`), and SAME two confirmed bugs
(guarded, not patched) as
`../../gnn-aggregation/tc-gnn/adapter.py` -- reproduced here rather than
imported cross-directory, per this task's "each artifact directory is
self-contained" convention (same policy `../../sddmm/rode/adapter.py`
states for its own `wrapper.cu` port).

One difference from the `spmm-binary-adjacency-kernel` variant
(`kernelbench.domains.sparse.variant_transform`, added by a concurrent
integration in this same session): that hook forces `workload.csr.data[:] =
1` BEFORE this adapter's `prepare()` ever sees the workload, so gating this
SAME adapter under that variant automatically exercises TC-GNN's kernel on
a genuine 0/1 adjacency pattern with no adapter-side special-casing needed
-- `forward_AGNN`'s per-edge `edgeAttention` array is simply all-ones in
that case, mathematically identical to (though not literally routed
through) TC-GNN's own unweighted `forward` path.

## Two real, confirmed artifact findings (reproduced from the gnn-aggregation adapter, not re-derived)

**1. Silent tail-drop for F not a multiple of 16** (`spmmAGNN_forward_cuda_
kernel`'s host wrapper sizes shared memory with CEILING division but the
kernel body recomputes the same name with FLOOR division) -- guarded:
`prepare()` raises `NotImplementedError` for `N % 16 != 0`.

**2. Confirmed one-element out-of-bounds WRITE (heap corruption) whenever
`num_nodes` (this track's `M`, since `TCGNN.preprocess()` requires a square
adjacency) is an exact multiple of `BLK_H=16`** (`TCGNN.cpp`'s `preprocess()`
row-window loop is off-by-one, `for (iter=0; iter<num_nodes+1; iter+=16)`)
-- guarded: `prepare()` raises `NotImplementedError` for `M % 16 == 0`, not
merely a "low confidence" flag, since this is confirmed memory corruption
(see the gnn-aggregation STATUS.md's canary-buffer reproduction), not a
silently-wrong-answer case.

**3. Confirmed silent zero-fill for F > 128** (fixed `WARPperBlock=WPB=8`,
no outer loop over further embedding-dimension tiles) -- guarded:
`prepare()` raises `NotImplementedError` for `N > 128`.

All three constraints, thresholds, and citations are IDENTICAL to
`../../gnn-aggregation/tc-gnn/adapter.py`'s -- this is the same compiled
kernel, so the same bugs apply regardless of which track calls it.

## Compute precision

`TCGNN_kernel.cu` uses `wmma::precision::tf32` fragments throughout --
TF32 tensor-core compute, fp32 accumulate. `spmm` (unlike
`gnn-aggregation`) HAS a dedicated laxer tensor-core variant,
`spmm-tensorcore-fp16` (tolerance `1e-2`, vs. the fp32 CSR variant's
`1e-4`) -- exactly the home this precision class belongs in (same
rationale as `dtcspmm`/`flashsparse`/`sspmm` in this same track).
`PRECISIONS = ["fp16"]`, matching `spmm-tensorcore-fp16`'s CLI convention
(`--precision fp16` selects the tensor-core-appropriate tolerance row, even
though the actual compute is TF32 -- same convention as
`artifacts/quantized-gemm/marlin`, `artifacts/sddmm/fused3s`).
"""

from __future__ import annotations

import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
_GNNAGG_TCGNN = os.path.join(HERE, "..", "..", "gnn-aggregation", "tc-gnn")
_BUILD_DIR = os.path.join(_GNNAGG_TCGNN, "build")   # reused, not local

KERNEL = "spmm"
IMPL_NAME = "tcgnn-agnn-spmm-f16"
PAPER_KEY = "conf/usenix/WangFWHD23"
# TCGNN_kernel.cu's WMMA fragments are torch::kFloat (fp32) tensors computed
# via TF32 tensor cores internally; gated as spmm's tensor-core-class
# precision row (see module docstring's "Compute precision").
PRECISIONS = ["fp16"]

BLK_H = 16
WPB = 8  # TCGNN_kernel.cu's #define WPB 8 -- fixed warps-per-block
MAX_F = WPB * BLK_H  # 128: the widest embedding_dim a single launch actually computes

_TCGNN = None
_IMPORT_ERROR = ""


def _load():
    global _TCGNN, _IMPORT_ERROR
    if _TCGNN is not None:
        return _TCGNN
    if _IMPORT_ERROR:
        raise RuntimeError(_IMPORT_ERROR)
    try:
        import torch  # noqa: F401 -- must be imported before the extension so its shared libs are resolvable
        if _BUILD_DIR not in sys.path:
            sys.path.insert(0, _BUILD_DIR)
        import TCGNN  # noqa: N814 -- matches the extension's actual module name
        _TCGNN = TCGNN
        return _TCGNN
    except Exception as e:
        _IMPORT_ERROR = f"{type(e).__name__}: {e}"
        raise


def available() -> tuple[bool, str]:
    so_present = (os.path.isdir(_BUILD_DIR)
                  and any(f.startswith("TCGNN") and f.endswith(".so") for f in os.listdir(_BUILD_DIR)))
    if not so_present:
        return False, (f"reused build/ not found at {_BUILD_DIR} -- run "
                        f"../../gnn-aggregation/tc-gnn/build.sh (this "
                        f"track's own build.sh only symlinks source/, it "
                        f"does not compile) then this artifact's build.sh")
    try:
        _load()
        import torch
        if not torch.cuda.is_available():
            return False, "no CUDA device visible"
        return True, ""
    except Exception as e:
        msg = f"{type(e).__name__}: {e}"
        if "CXXABI" in msg or "GLIBCXX" in msg:
            msg = ("libstdc++ ABI mismatch -- run with "
                   "LD_PRELOAD=/usr/lib64/libstdc++.so.6 (see STATUS.md)")
        return False, msg


def _dense_operand(rows: int, dim: int, seed: int, dtype) -> np.ndarray:
    # identical formula to kernelbench.impls.cpu_ref._dense_operand -- kept
    # as a separate copy per the reference-independence rule.
    rng = np.random.default_rng(seed)
    return rng.uniform(-1.0, 1.0, size=(rows, dim)).astype(dtype)


class TCGNNSpMM:
    name = IMPL_NAME
    platform = "cuda"

    def __init__(self, precision: str = "fp16"):
        if precision != "fp16":
            raise NotImplementedError(
                f"{IMPL_NAME}: TCGNN_kernel.cu's WMMA fragments are "
                f"hardcoded torch::kFloat (fp32) tensors (internally TF32 "
                f"tensor-core compute); requested precision {precision!r} "
                "is not serviced (run with --precision fp16 against "
                "spmm-tensorcore-fp16, see STATUS.md)")
        self.precision = precision

    def prepare(self, matrix, params: dict):
        import torch
        TCGNN = _load()
        A = matrix.csr
        M, K = A.shape
        if M != K:
            raise NotImplementedError(
                f"{IMPL_NAME}: TC-GNN's preprocess()/forward_AGNN() treat "
                f"nodePointer's length-1 as a single shared num_nodes for "
                f"both the row and column dimension of the sparse operand "
                f"(see TC-GNN's main_tcgnn.py); got a {M}x{K} non-square "
                f"matrix, not serviced")
        if M % BLK_H == 0:
            raise NotImplementedError(
                f"{IMPL_NAME}: TCGNN.cpp's preprocess() writes "
                f"blockPartition[num_nodes/{BLK_H}] one element past the "
                f"end of the num_row_windows-sized tensor whenever "
                f"num_nodes is an exact multiple of BLK_H={BLK_H} "
                f"(confirmed with a canary-buffer test -- see "
                f"../../gnn-aggregation/tc-gnn/STATUS.md finding #2); this "
                f"corrupts heap memory rather than just returning a wrong "
                f"answer. M={M} is a multiple of {BLK_H}, not serviced.")
        N = int(params["N"])
        if N % BLK_H != 0:
            raise NotImplementedError(
                f"{IMPL_NAME}: spmmAGNN_forward_cuda_kernel computes "
                f"dimTileNum = embedding_dim // BLK_H ({BLK_H}) with FLOOR "
                f"division, silently dropping any output columns at index "
                f">= floor(N/{BLK_H})*{BLK_H} -- N={N} is not a multiple of "
                f"{BLK_H}, not serviced (see "
                f"../../gnn-aggregation/tc-gnn/STATUS.md finding #1).")
        if N > MAX_F:
            raise NotImplementedError(
                f"{IMPL_NAME}: spmmAGNN_forward_cuda_kernel launches with a "
                f"fixed WARPperBlock={WPB} and selects the embedding-"
                f"dimension tile purely via threadIdx.y (range [0,{WPB})), "
                f"with no outer loop over further tiles -- output columns "
                f"at index >= {MAX_F} are never written by any warp and "
                f"silently stay zero (see "
                f"../../gnn-aggregation/tc-gnn/STATUS.md finding #3). "
                f"N={N} > {MAX_F}, not serviced.")
        seed = params.get("seed", 42)

        num_nodes = M
        num_edges = int(A.nnz)
        num_row_windows = (num_nodes + BLK_H - 1) // BLK_H

        row_pointers = torch.tensor(np.ascontiguousarray(A.indptr, dtype=np.int32))
        column_index = torch.tensor(np.ascontiguousarray(A.indices, dtype=np.int32))
        # edgeAttention: shape [n_head=1, n_e], the workload's OWN CSR
        # values fed directly (no normalization -- unlike the
        # gnn-aggregation adapter, spmm's reference has none) -- one value
        # per CSR nonzero in the SAME order as column_index (preprocess()
        # never reorders edgeList, see the gnn-aggregation adapter's
        # docstring for the derivation, reproduced not re-run here).
        edge_values = torch.tensor(A.data.astype(np.float32)).reshape(1, -1)

        blockPartition = torch.zeros(num_row_windows, dtype=torch.int32)
        edgeToColumn = torch.zeros(num_edges, dtype=torch.int32)
        edgeToRow = torch.zeros(num_edges, dtype=torch.int32)

        TCGNN.preprocess(column_index, row_pointers, num_nodes,
                          BLK_H, 8, blockPartition, edgeToColumn, edgeToRow)

        X = _dense_operand(num_nodes, N, seed, np.float32)

        h = {
            "TCGNN": TCGNN,
            "row_pointers": row_pointers.cuda(),
            "column_index": column_index.cuda(),
            "edge_values": edge_values.cuda(),
            "blockPartition": blockPartition.cuda(),
            "edgeToColumn": edgeToColumn.cuda(),
            "edgeToRow": edgeToRow.cuda(),
            "X": torch.as_tensor(X, device="cuda"),
        }
        return h

    def run(self, h):
        return h["TCGNN"].forward_AGNN(
            h["X"], h["row_pointers"], h["column_index"], h["edge_values"],
            h["blockPartition"], h["edgeToColumn"], h["edgeToRow"])[0]

    def to_host(self, out):
        import torch
        return out.detach().to("cpu", dtype=torch.float64).numpy()

    def timer(self):
        from kernelbench.impls.gpu_cuda import CudaEventTimer
        return CudaEventTimer()

    def free(self, h):
        import torch
        h.clear()
        torch.cuda.empty_cache()


def create(precision: str):
    return TCGNNSpMM(precision)
