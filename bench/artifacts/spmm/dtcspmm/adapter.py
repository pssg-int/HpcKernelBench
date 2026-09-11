"""
DTC-SpMM adapter for the spmm track.

Paper: "DTC-SpMM: Bridging the Gap in Accelerating General Sparse Matrix
Multiplication with Tensor Cores", ASPLOS'24. `PAPER_KEY =
conf/asplos/Fan0024` (matches `benchspecs/spmm/spec.yaml`'s own `evidence`
entry). Artifact: https://github.com/HPMLL/DTC-SpMM_ASPLOS24

Selected as a core general-GPU-SpMM baseline under the revised
kernel-centrality rule (2026-09-05): this is the tensor-core baseline every
later paper in the line (FlashSparse, the sibling agent's Voltrix/SSpMM/
MP-SpMM) cites and compares against, and the spec's own `spmm-gpu-kernel-f32`
`claim` names it explicitly ("DTC-SpMM's inner kernel").

## What the artifact actually is

`source/DTC-SpMM/{DTCSpMM.cpp,DTCSpMM_kernel.cu}` is a torch CUDA extension.
`preprocess_gpu` builds DTC-SpMM's ME-TCF compressed tensor-core format
(16x8 tiles, bitmap-encoded, row-window offsets) from CSR; `run_DTCSpMM`
runs the actual WMMA SpMM kernel against it -- both called directly,
unmodified (ARTIFACT_GUIDE.md rule 1), via this directory's own
`dtc_setup.py` build (see build.sh / STATUS.md for why the artifact's own
`source/DTC-SpMM/setup.py` -- which wants a prebuilt Sputnik+Glog -- was not
used verbatim).

**Compute precision (confirmed by reading the kernel, not assumed):**
`wmma::fragment<..., wmma::precision::tf32, ...>` /
`wmma::fragment<wmma::accumulator, ..., float>` (DTCSpMM_kernel.cu lines
~1606-1684 and every other `spmm_forward_cuda_kernel_improved_ptx_1684_*`
variant) -- TF32 Tensor-Core compute, fp32 accumulate. This is
`spmm-tensorcore-fp16`'s documented "tf32 secondary" precision row, not the
`fp16 primary` row -- `PRECISIONS = ["tf32"]`, gated with `--precision tf32`
(the variant explicitly supports both as separate, never-averaged rows).

## IMPORTANT FINDING: the artifact's exposed Python API is structurally
## unweighted (binary-pattern) SpMM, not general real-valued SpMM

Traced every SpMM code path reachable from `DTCSpMM.cpp`'s `PYBIND11_MODULE`
bindings (`run_DTCSpMM`, `run_DTCSpMM_balance`, `DTCSpMM_gcn`, and their
`_clock`-instrumented siblings): EVERY one of them builds its Tensor-Core
value operand as `auto val = torch::ones({TCblocktile_id.size(0)}, options);
float *valuesA = val.data<float>();` INSIDE the C++ wrapper -- there is no
parameter anywhere in the exposed API for the caller to supply real nonzero
values (`grep -rn "torch::Tensor value" DTC-SpMM/` -> zero hits). The
underlying CUDA kernels ARE templated to accept an arbitrary `valuesA`
device pointer (their `_with_value_` naming is literal -- the numeric
multiply-accumulate genuinely uses whatever `valuesA` contains), but the
compressed ME-TCF preprocessing pipeline (`preprocess_gpu` /
`seg_sort_dequ`) tracks only STRUCTURE (which cells of each 16x8 tile are
populated + which dense-matrix row each compressed column maps to), never a
value permutation -- consistent with the paper's own evaluation suite (spec's
own `evidence` line for this key: "8-graph GNN suite (origin+TCA-reordered)")
being entirely GNN-adjacency graphs, which are inherently unweighted.
Recovering a value permutation ourselves would require reverse-engineering
`seg_sort_dequ`'s internal sort/dedup order well beyond this integration's
budget, and -- more importantly -- would no longer be "wrapping the kernel
the artifact ships" (rule 1) but a new capability neither the paper nor the
released code demonstrates.

**Consequence, disclosed rather than hidden (rule 4: a genuine gate failure
IS a result)**: this adapter computes `A_pattern @ B` (every stored nonzero
counted as 1.0), which equals the spec's `A @ B` (with A's REAL values) only
when A already contains no values other than 1 -- true for this track's
GNN-graph inputs (cora/citeseer/pubmed via this codebase's own Planetoid
loader, which explicitly sets `A.data[:] = 1.0`; likely also the 8-graph
GNN suite's own pattern-only SuiteSparse matrices) but false for general
numerically-valued matrices (cant, consph, pdb1HYS, ...) and for this
track's synthetic smoke matrices (`kernelbench.matrices.synthetic` draws
`U(-1,1)` values, never 1.0). See STATUS.md for both gate results:
smoke (expected FAIL, evidencing the above) and cora (PASS, showing the
ME-TCF tiling + WMMA kernel mechanism is numerically correct on the class of
input the artifact's own released API actually supports).

`num_nodes` is a single scalar in `preprocess_gpu`'s signature, used for
BOTH matrix dimensions -- DTC-SpMM's own API assumes a square (GNN-adjacency
-style) matrix; a rectangular workload raises `NotImplementedError` (rule 8).
"""

from __future__ import annotations

import os
import sys

import numpy as np

KERNEL = "spmm"
IMPL_NAME = "dtcspmm-tcf-spmm"
PAPER_KEY = "conf/asplos/Fan0024"
PRECISIONS = ["tf32"]

_HERE = os.path.dirname(os.path.abspath(__file__))
_BUILD_LIB = os.path.join(_HERE, "build", "lib")

BLK_H = 16  # source/DTC-SpMM/config.h
BLK_W = 8

_dtc = None


def _load_module():
    global _dtc
    if _dtc is not None:
        return _dtc
    if not os.path.isdir(_BUILD_LIB):
        raise FileNotFoundError(f"{_BUILD_LIB} not built -- run build.sh")
    if _BUILD_LIB not in sys.path:
        sys.path.insert(0, _BUILD_LIB)
    import DTCSpMM as _m
    _dtc = _m
    return _dtc


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
        _load_module()
    except Exception as e:
        return False, f"DTCSpMM extension import failed: {type(e).__name__}: {e}"
    return True, ""


def create(precision: str):
    return DTCSpMMImpl(precision)


class DTCSpMMImpl:
    name = IMPL_NAME
    platform = "cuda"

    def __init__(self, precision: str = "tf32"):
        if precision != "tf32":
            raise NotImplementedError(
                f"{IMPL_NAME} only ships a TF32-compute/fp32-accumulate WMMA "
                f"kernel (wmma::precision::tf32, confirmed by reading "
                f"DTCSpMM_kernel.cu -- no fp16/fp32 fragment path exists); "
                f"requested {precision}")
        self.precision = precision

    def prepare(self, matrix, params: dict):
        import torch
        dtc = _load_module()

        A = matrix.csr
        M, K = A.shape
        if M != K:
            raise NotImplementedError(
                f"{IMPL_NAME}: DTC-SpMM's preprocess_gpu() takes a single "
                f"num_nodes for both matrix dimensions (assumes a square "
                f"GNN-adjacency-style matrix); requested shape ({M}, {K}) "
                f"is rectangular")
        N = int(params["N"])
        if N % BLK_H != 0:
            raise NotImplementedError(
                f"{IMPL_NAME}: dense width N must be a multiple of "
                f"BLK_H={BLK_H} (WARPperBlock = N/BLK_H launch-config "
                f"divisor in DTCSpMM_kernel.cu); requested N={N}")

        nnz = int(A.nnz)
        num_row_windows = (M + BLK_H - 1) // BLK_H

        col_idx = torch.as_tensor(A.indices.astype(np.int32), device="cuda")
        row_ptr = torch.as_tensor(A.indptr.astype(np.int32), device="cuda")
        block_partition = torch.zeros(num_row_windows, dtype=torch.int32, device="cuda")
        edge_to_column = torch.zeros(nnz, dtype=torch.int32, device="cuda")
        edge_to_row = torch.zeros(nnz, dtype=torch.int32, device="cuda")

        # --- artifact's own format conversion: CSR -> ME-TCF compressed
        # tensor-core format (16x8 tiles, bitmap-encoded), timed as
        # preprocessing per the harness contract. NOTE: takes no values
        # array -- see module docstring's IMPORTANT FINDING. ---
        (row_window_offset, _tcblock_rowid, tcblocktile_id,
         tcblock_offset, sparse_atox_idx, _block_count) = dtc.preprocess_gpu(
            col_idx, row_ptr, M, BLK_H, BLK_W,
            block_partition, edge_to_column, edge_to_row)

        # numpy RNG matching cpu_ref.reference_spmm's _dense_operand exactly
        # (see insum/adapter.py's docstring for why this matters).
        rng = np.random.default_rng(params.get("seed", 42))
        B_np = rng.uniform(-1.0, 1.0, size=(K, N)).astype(np.float32)
        B = torch.as_tensor(B_np, device="cuda")

        return {
            "row_window_offset": row_window_offset,
            "tcblocktile_id": tcblocktile_id,
            "tcblock_offset": tcblock_offset,
            "sparse_atox_idx": sparse_atox_idx,
            "M": M, "nnz": nnz, "B": B,
        }

    def run(self, h):
        dtc = _load_module()
        # run_DTCSpMM allocates a fresh zeros_like(input) output internally
        # (DTCSpMM_kernel.cu:830) -- no accumulation across calls, no
        # manual zeroing needed here (unlike RoDe/Insum's atomicAdd kernels).
        # "float_nonsplit" is the simplest of DTC-SpMM's 5 autotuned exeplan
        # code paths (float/float2/float4 x split/nonsplit) and the only one
        # valid for every N in this spec's dense-dims sweep without an
        # additional N%32==0 (float2/float4) constraint; the paper's own
        # eval autotunes per (dataset, N) instead (see
        # scripts/DTCSpMM/run_DTC_SpMM.py's ExecutionPlan table) -- not
        # reproduced here, a simplification disclosed in STATUS.md.
        out = dtc.run_DTCSpMM(
            h["B"], h["row_window_offset"], h["tcblocktile_id"],
            h["tcblock_offset"], h["sparse_atox_idx"],
            h["M"], h["nnz"], "float_nonsplit")[0]
        return out

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
