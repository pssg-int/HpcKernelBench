"""
Adapter for TC-GNN (TC-GNN: Bridging Sparse GNN Computation and Dense Tensor
Cores on GPUs, USENIX ATC'23, YukeWang96/TC-GNN_ATC23) -- GPU tensor-core
neighbor-aggregation kernel.

## What boundary was wrapped

`source/TCGNN_conv/{TCGNN.cpp,TCGNN_kernel.cu}` is a torch CUDA extension
(pybind module `TCGNN`, built UNMODIFIED by build.sh) exposing several entry
points: `preprocess` (CPU, structural block-partition), `forward` (WMMA
SpMM, UNWEIGHTED -- every populated cell of its shared-memory sparse tile is
hardcoded to `1` in TCGNN_kernel.cu's `spmm_forward_cuda_kernel`, confirmed
by reading the kernel: `sparse_A[row_local*BLK_W+col_local] = 1;`, no value
array anywhere in `spmm_forward`'s signature), `forward_ef` (SDDMM), and
`forward_AGNN` (WMMA SpMM that DOES take a real per-edge value array,
`edgeAttention`, confirmed: `sparse_A[row_local*BLK_W+col_local] =
edgeAttention[eIdx];`). This adapter calls `forward_AGNN` directly -- the
only one of TC-GNN's own, unmodified entry points that computes a genuine
weighted sparse-times-dense product, `Y = A_hat @ X` for an ARBITRARY
`A_hat`, exactly this track's kernel boundary (ARTIFACT_GUIDE.md rule 1).
`forward` (the paper's own "isolated single-kernel SpMM" benchmark path,
`2_tcgnn_single_kernel.py` -> `gnn_conv.py:SAG`) is intentionally NOT used
here: it can only represent a binary/unweighted adjacency (no way to pass
`D^-1/2(A+I)D^-1/2`'s real-valued entries through it at all), and this
domain's ONE reference (`kernelbench.impls.cpu_ref.reference_gnn_aggregation`,
per DOMAIN_GUIDE.md's one-CPU_IMPLS-entry-per-kernel contract) is the GCN
SYMMETRIC-NORMALIZED aggregation, not plain A@X. `forward_AGNN` is still the
identical WMMA/tensor-core compute kernel and identical `preprocess`
structural pipeline as `forward` -- same TC-block tiling, same tensor-core
MMA -- it merely also accepts real edge values, which the paper's own
`AGNNConv`/`TCGNNFunction_AGNN` (`gnn_conv.py`) already calls in exactly
this shape (`edgeAttention` shape `[n_head=1, n_e]`, flat length `num_edges`,
one value per CSR nonzero in `column_index`/`edgeList` order) for its own
attention-weighted aggregation. This is squarely "the artifact's own kernel",
not a new capability invented for this benchmark.

## Preprocessing (prepare(), timed once): reuses the artifact's own code, unmodified

`TCGNN.preprocess(column_index, row_pointers, num_nodes, BLK_H, BLK_W,
blockPartition, edgeToColumn, edgeToRow)` -- called here EXACTLY as
`main_tcgnn.py` calls it -- is TC-GNN's own tensor-core-tile block
partitioning (16x8 TC blocks, `edgeToColumn`/`edgeToRow` maps). Read in
full (`TCGNN.cpp`'s `preprocess`): it is purely STRUCTURAL (built only from
`column_index`/`row_pointers`, an OpenMP CPU routine using
`.accessor<int,1>()` -- no CUDA, no touch of any value array at all), and it
never reorders `column_index`/`edgeList` itself (the internal
`neighbor_window` used for per-row-window dedup/sort is a throwaway copy
used only to build a `col -> compressed_col` map; `edgeToColumn[e_index]` is
written back at the ORIGINAL `e_index`) -- confirmed by reading `preprocess`
line by line, not assumed. This means the per-edge `edgeAttention` array we
build (`A_hat.data`, in the SAME order as `A_hat.indices`/`column_index`)
stays index-aligned with the structural maps `preprocess` produces, with no
value permutation needed. This IS "the artifact's own format conversion"
(ARTIFACT_GUIDE.md rule 2), so it lives entirely in `prepare()`, timed once
as preprocessing.

Adjacency normalization (`A_hat = D^-1/2 (A+I) D^-1/2`) is computed
independently here (`_gcn_normalize` below, own code, not calling
`kernelbench.impls.cpu_ref`'s helpers or `artifacts/gnn-aggregation/
stragcn/adapter.py`'s copy) -- same formula, separate code object, per the
reference-independence audit ruling in DOMAIN_GUIDE.md.

## Two real, confirmed artifact findings

**1. Confirmed bug: `spmmAGNN_forward_cuda_kernel` silently drops the tail
of the feature dimension for F not a multiple of BLK_H=16.** The C++ host
wrapper `spmmAGNN_forward_cuda` (`TCGNN_kernel.cu`) sizes its dynamic shared
memory with CEILING division, `dimTileNum = (embedding_dim + BLK_H - 1) /
BLK_H`, but the `__global__` kernel body itself
(`spmmAGNN_forward_cuda_kernel`) recomputes the SAME name with FLOOR
division instead: `const unsigned dimTileNum = embedding_dim / BLK_H;`
(unsigned integer division truncates). For F a multiple of 16 these agree;
for F NOT a multiple of 16, the kernel only iterates
`floor(F/16)` dimension tiles, silently leaving every output column at
index `>= floor(F/16)*16` at its `torch::zeros_like(input)` initial value
(0), rather than computing it -- a silent truncation, not a crash. This
track's spec only sweeps `F` (called `N` here) over `{32, 128, 256, 512}`,
all multiples of 16, so gate verification below never exercises this path,
but `prepare()` still asserts `F % 16 == 0` and raises `NotImplementedError`
with this explanation for any other F, per ARTIFACT_GUIDE.md rule 8, rather
than silently returning a partially-zero answer.

**2. Confirmed bug: `TCGNN.preprocess()` writes ONE ELEMENT PAST THE END of
the `blockPartition` tensor whenever `num_nodes` is an exact multiple of
`BLK_H=16`.** `preprocess()`'s row-window loop (`TCGNN.cpp`) is
`for (unsigned iter = 0; iter < num_nodes + 1; iter += blockSize_h)`, off by
one against the correct bound (compare `fill_edgeToRow`'s own loop two lines
above it in the same file, correctly `nid < num_nodes`, no `+1`). When
`num_nodes % 16 == 0`, this loop's LAST iteration lands at
`iter == num_nodes`, computing `windowId = iter/16 == num_row_windows`
(one past the valid `[0, num_row_windows)` range of the `blockPartition`
tensor `preprocess()` was handed) and writes there. Confirmed empirically
(not just from reading the loop bound): allocating `blockPartition` as a
size-1 VIEW into a larger canary tensor `[-777,-777,-777,-777]` with
`num_nodes=16` and calling `preprocess()` leaves the canary buffer as
`[2, 1, -777, -777]` -- index 1, one past the valid size-1 view, was
overwritten (with `1`, a real blockPartition value for a DIFFERENT,
nonexistent row window). This out-of-bounds write corrupts whatever heap
memory follows the tensor's storage allocation; in this integration it
surfaced as a DELAYED `free(): invalid pointer` / glibc malloc assertion
crash inside an unrelated `torch.Tensor` deallocation later in the same
process (confirmed via `gdb` backtrace: the abort happens in
`THPVariable_subclass_clear` -> `free()`, not inside TC-GNN's own code at
all) -- exactly the kind of crash that "sometimes reproduces, sometimes
doesn't" depending on incidental heap layout, which is what made this bug
hard to pin down. All three of this domain's SMOKE_WORKLOADS matrices use
`rows=4000` (`4000 % 16 == 0`), so `--smoke` hits this on every workload;
`cora` (2708 nodes, `2708 % 16 == 4`, not a multiple) does not. Not patched
(rule 3: kernel-code changes are out of scope) -- `prepare()` instead
refuses any workload whose `num_nodes % 16 == 0` with `NotImplementedError`
naming this exact constraint, since running it does not "silently return a
wrong answer" (rule 8's usual case) but corrupts process memory outright.

**3. Confirmed bug: the kernel silently computes only the first 128 feature
columns for F > 128, leaving the rest at zero.** Both `spmm_forward_cuda`
and `spmmAGNN_forward_cuda` launch with a FIXED block shape
`dim3 block(WARP_SIZE=32, WARPperBlock=WPB=8, 1)` (`#define WPB 8`), and
inside the `__global__` kernel body the embedding-dimension tiling is
selected purely by `wid = threadIdx.y` (range `[0, WARPperBlock) = [0,8)`)
via `if (wid < dimTileNum) { ... }` -- there is no outer loop chunking
`dimTileNum` across multiple launches or multiple values of `wid` per
thread. For `dimTileNum = embedding_dim / BLK_H <= 8` (i.e. `F <= 128`)
every tile has a warp; for `F > 128`, tiles `8..dimTileNum-1` (every output
column at index `>= 128`) are never assigned to any warp and are never
written, silently staying at the kernel's `torch::zeros_like(input)` initial
value. Confirmed empirically on `cora` (2708 nodes, safe from finding #2's
bug): `F=32` and `F=128` gate at `max_scaled_err` in the `~7e-4` range
(TF32-level precision noise, see "Compute precision" below); `F=256` and
`F=512` gate at `max_scaled_err = 1.000` (the entire tail beyond column 128
is exactly zero against a genuinely nonzero reference, driving the
scaled error to its ceiling). This track's spec sweeps `N` (this adapter's
`F`) over `{32, 128, 256, 512}` -- `prepare()` raises `NotImplementedError`
for `F > 128`, so only 32 and 128 are gated below; this matches TC-GNN's own
paper convention (every benchmark script here uses `--hidden 16`, an order
of magnitude under this limit) rather than a limit the paper's own
evaluation ever tests against.

**4. Confirmed resource leak (not a correctness bug): `spmmAGNN_forward_cuda`
calls `cudaStreamCreate` once per invocation (`new cudaStream_t[num_attention]`
+ `cudaStreamCreate(&streams[att_idx])` in a loop) with no matching
`cudaStreamDestroy`/`delete[]` anywhere in the function or file. Harmless
for the handful of gate-only launches this integration performs (rule 5:
login-node build + minimal correctness check), but a full spec-conforming
timed run (100-200 reps) would leak one CUDA stream handle per rep. Not
patched -- ARTIFACT_GUIDE.md rule 3 puts kernel-code changes out of scope --
flagged here for whoever runs the compute-node timing sweep later.

## Compute precision

`TCGNN_kernel.cu` uses `wmma::fragment<..., wmma::precision::tf32, ...>` /
`wmma::fragment<wmma::accumulator, ..., float>` throughout (both
`spmm_forward_cuda_kernel` and `spmmAGNN_forward_cuda_kernel`) -- TF32
tensor-core compute, fp32 accumulate, identical precision family to
`artifacts/spmm/dtcspmm`. Unlike the `spmm` track, `gnn-aggregation`'s
`gnn-agg-kernel-f32` variant has no separate tf32 row to gate against --
`PRECISIONS = ["fp32"]` here describes the tensor dtype at the Python
boundary (input/output/edge-value tensors are all `torch.float32`; TC-GNN
ships no fp64/fp16 path), and the measured `max_scaled_err` below is
reported and gated as-is against the spec's 1e-4 tolerance rather than
silently relabeled to a laxer bound.
"""

from __future__ import annotations

import os
import sys

import numpy as np
import scipy.sparse as sp

HERE = os.path.dirname(os.path.abspath(__file__))
SRC_DIR = os.path.join(HERE, "source")
BUILD_DIR = os.path.join(HERE, "build")

KERNEL = "gnn-aggregation"
IMPL_NAME = "tcgnn-agnn-spmm-gcnnorm"
PAPER_KEY = "conf/usenix/WangFWHD23"
# TCGNN_kernel.cu's WMMA fragments are torch::kFloat (fp32) tensors computed
# via TF32 tensor cores internally; no fp64/fp16 path exists. See module
# docstring's "Compute precision" section.
PRECISIONS = ["fp32"]

BLK_H = 16
BLK_W = 8
WPB = 8  # TCGNN_kernel.cu's #define WPB 8 -- fixed warps-per-block, see module docstring finding #3
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
        if BUILD_DIR not in sys.path:
            sys.path.insert(0, BUILD_DIR)
        import TCGNN  # noqa: N814 -- matches the extension's actual module name
        _TCGNN = TCGNN
        return _TCGNN
    except Exception as e:
        _IMPORT_ERROR = f"{type(e).__name__}: {e}"
        raise


def available() -> tuple[bool, str]:
    so_present = any(f.startswith("TCGNN") and f.endswith(".so")
                      for f in os.listdir(BUILD_DIR)) if os.path.isdir(BUILD_DIR) else False
    if not so_present:
        return False, "TCGNN*.so not built -- run build.sh"
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
    # as a separate copy (not a shared code object) per the
    # reference-independence rule in DOMAIN_GUIDE.md.
    rng = np.random.default_rng(seed)
    return rng.uniform(-1.0, 1.0, size=(rows, dim)).astype(dtype)


def _gcn_normalize(A: sp.csr_matrix) -> sp.csr_matrix:
    """A_hat = D^-1/2 (A+I) D^-1/2, computed independently of
    kernelbench.impls.cpu_ref and of the stragcn adapter's own copy -- see
    module docstring's reference-independence note."""
    n = A.shape[0]
    A64 = A.astype(np.float64)
    A_tilde = (A64 + sp.identity(n, format="csr", dtype=np.float64)).tocsr()
    A_tilde.sort_indices()
    deg_row = np.asarray(A_tilde.sum(axis=1)).ravel()
    deg_col = np.asarray(A_tilde.sum(axis=0)).ravel()
    dinv_row = np.zeros_like(deg_row)
    dinv_col = np.zeros_like(deg_col)
    nzr, nzc = deg_row > 0, deg_col > 0
    dinv_row[nzr] = deg_row[nzr] ** -0.5
    dinv_col[nzc] = deg_col[nzc] ** -0.5
    A_hat = (sp.diags(dinv_row) @ A_tilde @ sp.diags(dinv_col)).tocsr()
    A_hat.sort_indices()
    return A_hat


class TCGNNAggregation:
    name = IMPL_NAME
    platform = "cuda"
    normalization = "gcn-sym: D^-1/2 (A+I) D^-1/2 (independent copy, see module docstring)"

    def __init__(self, precision: str = "fp32"):
        if precision != "fp32":
            raise NotImplementedError(
                f"TCGNN_kernel.cu's WMMA fragments are hardcoded torch::kFloat "
                f"(fp32) tensors (internally TF32 tensor-core compute, see "
                f"module docstring); requested precision {precision!r} is not "
                "serviced")
        self.precision = precision

    def prepare(self, matrix, params):
        import torch
        TCGNN = _load()
        A = matrix.csr
        M, N = A.shape
        if M != N:
            raise NotImplementedError(
                f"TC-GNN's preprocess()/forward_AGNN() treat nodePointer's "
                f"length-1 as a single shared num_nodes for both the row and "
                f"column dimension of the adjacency (see main_tcgnn.py); "
                f"got a {M}x{N} non-square matrix, not serviced")
        if M % BLK_H == 0:
            raise NotImplementedError(
                f"TCGNN.cpp's preprocess() writes blockPartition[num_nodes/"
                f"BLK_H] one element past the end of the num_row_windows-"
                f"sized tensor whenever num_nodes is an exact multiple of "
                f"BLK_H={BLK_H} (confirmed with a canary-buffer test -- see "
                f"module docstring finding #2); this corrupts heap memory "
                f"rather than just returning a wrong answer. num_nodes={M} "
                f"is a multiple of {BLK_H}, not serviced.")
        F = int(params["N"])
        if F % BLK_H != 0:
            raise NotImplementedError(
                f"spmmAGNN_forward_cuda_kernel computes "
                f"dimTileNum = embedding_dim // BLK_H (16) with FLOOR "
                f"division (confirmed in TCGNN_kernel.cu), silently dropping "
                f"any output columns at index >= floor(F/16)*16 -- F={F} is "
                f"not a multiple of 16, not serviced (see module docstring's "
                "finding #1). This track's spec only sweeps F in "
                "{32,128,256,512}, all multiples of 16, so this never blocks "
                "a spec-conforming run.")
        if F > MAX_F:
            raise NotImplementedError(
                f"spmmAGNN_forward_cuda_kernel launches with a fixed "
                f"WARPperBlock={WPB} and selects the embedding-dimension "
                f"tile purely via threadIdx.y (range [0,{WPB})), with no "
                f"outer loop over further tiles -- output columns at index "
                f">= {MAX_F} are never written by any warp and silently stay "
                f"zero (confirmed empirically: F=256/512 on cora gate at "
                f"max_scaled_err=1.0; see module docstring finding #3). "
                f"F={F} > {MAX_F}, not serviced.")
        seed = params.get("seed", 42)

        A_hat = _gcn_normalize(A)
        params["gnn_agg_nnz"] = int(A_hat.nnz)
        params["gnn_agg_normalization"] = self.normalization

        num_nodes = M
        num_edges = int(A_hat.nnz)
        num_row_windows = (num_nodes + BLK_H - 1) // BLK_H

        row_pointers = torch.tensor(A_hat.indptr, dtype=torch.int32)
        column_index = torch.tensor(A_hat.indices, dtype=torch.int32)
        # edgeAttention: shape [n_head=1, n_e], matching gnn_conv.py's own
        # AGNNConv convention (n_heads=1 module-level constant there) -- flat
        # memory layout is a plain length-num_edges array, one value per CSR
        # nonzero in the SAME order as column_index (see module docstring:
        # preprocess() never reorders edgeList).
        edge_values = torch.tensor(A_hat.data, dtype=torch.float32).reshape(1, -1)

        blockPartition = torch.zeros(num_row_windows, dtype=torch.int32)
        edgeToColumn = torch.zeros(num_edges, dtype=torch.int32)
        edgeToRow = torch.zeros(num_edges, dtype=torch.int32)

        # TC-GNN's own structural preprocessing (CPU, OMP) -- see module
        # docstring: never touches edge values, so it is safe to run before
        # or after building edge_values above.
        TCGNN.preprocess(column_index, row_pointers, num_nodes,
                          BLK_H, BLK_W, blockPartition, edgeToColumn, edgeToRow)

        X = _dense_operand(num_nodes, F, seed, np.float32)

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
    return TCGNNAggregation(precision)
