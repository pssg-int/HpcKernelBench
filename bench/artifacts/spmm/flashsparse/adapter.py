"""
FlashSparse adapter for the spmm track.

Paper: "FlashSparse: Minimizing Computation Redundancy for Fast Sparse
Matrix Multiplications on Tensor Cores", PPoPP'25. `PAPER_KEY =
conf/ppopp/ShiLXFWW25` (matched by title in `../../../output/included.json`;
this exact key also appears in `benchspecs/spmm/spec.yaml`'s
`notes_on_fairness`/`open_questions` as one of the papers presumed -- but not
independently verified there -- to reuse DTC-SpMM's 8-graph GNN suite).
Artifact: https://github.com/ParCIS/FlashSparse -- a swap-and-transpose
tensor-core SpMM/SDDMM kernel (MMA m16n8k16-class), fp16/tf32.

Selected as a core general-GPU-SpMM baseline under the revised
kernel-centrality rule (2026-09-05): the most recent (PPoPP'25) entry in the
RoDe/DTC-SpMM tensor-core-SpMM line this spmm track's `spmm-gpu-kernel-f32`
and `spmm-tensorcore-fp16` variants are built around.

## What was wrapped (rule 1)

`FS_Block_gpu.preprocess_gpu_fs` (CSR -> FlashSparse's own GPU-built
compressed tensor-core format: row-window offsets + a per-window
"sparse-A-to-X" column-index array + a value array, 8x8-tiled) and
`FS_SpMM.forward_fp16` (the actual swap-and-transpose WMMA SpMM kernel
against that format). Both are `source/FlashSparse/{Block_gpu,SpMM}/*`
symbols, compiled unmodified via this directory's own `fs_setup.py` (see
build.sh / STATUS.md -- only 2 of the artifact's 4 bundled extensions are
built; FS_SDDMM is a different track, FS_Block is an alternate CPU
preprocessing path not used here).

**Timing-scope caveat, disclosed per rule 1's explicit allowance ("wrap at
the finest boundary available and document the contamination")**: unlike
RoDe/DTC-SpMM/InferFast (all of which keep every buffer GPU-resident and
expose a device-pointer kernel entry point), FlashSparse's OWN pybind
bindings (`spmm_forward_fp16` in `SpMM/src/benchmark.cpp`) take CPU-resident
torch tensors and do their own `cudaMalloc`/H2D-copy/kernel-launch/D2H-copy/
`cudaFree` INSIDE the bound function -- there is no separately-exposed
device-pointer entry point in the compiled `.so` (`spmm_forward_cuda_fp16`,
the raw-pointer function that actually launches the kernel, is not
pybind-exposed). Every `run()` call below therefore also times FlashSparse's
own H2D/D2H copies and allocation, not kernel-only time. This is
acceptable for the login-node GATE check this integration performs (rule 5:
no timing sweeps here anyway); a compute-node timing run would need a
custom ctypes wrapper around `spmm_forward_cuda_fp16` directly (same
strategy as `rode/wrapper.cu`) to get kernel-only timing -- not built here,
out of this integration's budget.
`spmm_forward_cuda_fp16` (`SpMM/src/spmmKernel.cu`) additionally
hardcodes its own 10-iteration internal warmup loop before the `epoches`
-times timed loop -- called here with `epoches=1`, so every `run()` still
pays that fixed 10-launch warmup regardless (further disclosed
contamination, not a correctness issue).

## IMPORTANT FINDING: same binary-pattern-only limitation as DTC-SpMM

Traced `FS_Block_gpu.preprocess_gpu_fs`'s value-generation path
(`block_kernel.cu::generate_tcoffset_id_atob_fs`, called from
`seg_sort_dequ_fs`): the returned `values` tensor is allocated as
`torch::zeros(...)` and then populated ONLY via
`values_[...] = __float2half(1.0);` for every structurally-occupied
position -- there is no parameter anywhere in `preprocess_gpu_fs`'s
signature for a caller to supply real nonzero values
(`grep -rn "torch::Tensor value" Block_gpu/` in the upstream source --
zero hits for an INPUT values parameter; only the internally-synthesized
output is named `values`). A second, CPU-side preprocessing path exists
(`FS_Block.blockProcess_fp16`, used by `SpMM/test/mgcn/test.py`) that DOES
take a caller-supplied tensor called `dd` ("degree"), but tracing its own
correctness check in that same test file
(`value.append(dd[i]*dd[column_index1[j]])`) shows it computes a RANK-1
`degree[row] * degree[col]` GCN-style symmetric-normalization weight, not
an arbitrary per-edge value -- structurally incapable of representing a
general nonzero-value matrix either (a rank-1 outer product can only ever
express `outer(d, d)`-shaped weights, not this track's arbitrary SuiteSparse
magnitudes). Both of FlashSparse's own preprocessing paths are therefore
fundamentally for GNN-adjacency-style (or degree-normalized) aggregation,
consistent with the whole TC-GNN/DTC-SpMM lineage this paper extends (see
`dtcspmm/STATUS.md`'s identical finding for the direct predecessor this
paper cites as a baseline) -- not a bug in this integration.

**Consequence (rule 4)**: this adapter computes `A_pattern @ B`. Gated
against the spec's real-valued fp64 reference, this passes exactly when A's
real values are already all 1 (cora/citeseer/pubmed via this codebase's
Planetoid loader) and fails by construction for general-valued matrices and
this track's `U(-1,1)` synthetic smoke set -- see STATUS.md for both
results.

## Precision

`FS_SpMM.forward_fp16` -> `spmm_forward_cuda_kernel_fp16` (WMMA, half
compute). `PRECISIONS = ["fp16"]`, gated under `spmm-tensorcore-fp16` with
`--precision fp16` per this task's instructions (FlashSparse also ships a
`forward_tf32` entry point, not wired here -- fp16 is the paper's headline
"swap-and-transpose" configuration and the simplest single kernel call to
verify within this integration's budget).
"""

from __future__ import annotations

import os
import sys

import numpy as np

KERNEL = "spmm"
IMPL_NAME = "flashsparse-spmm"
PAPER_KEY = "conf/ppopp/ShiLXFWW25"
PRECISIONS = ["fp16"]

_HERE = os.path.dirname(os.path.abspath(__file__))
_BUILD_LIB = os.path.join(_HERE, "build", "lib")

BLK_H = 8  # window size ("window"/"wide" in the artifact's own test.py)
BLK_W = 8

_fs_spmm = None
_fs_block_gpu = None


def _load_modules():
    global _fs_spmm, _fs_block_gpu
    if _fs_spmm is not None:
        return _fs_spmm, _fs_block_gpu
    if not os.path.isdir(_BUILD_LIB):
        raise FileNotFoundError(f"{_BUILD_LIB} not built -- run build.sh")
    if _BUILD_LIB not in sys.path:
        sys.path.insert(0, _BUILD_LIB)
    import FS_SpMM as _spmm
    import FS_Block_gpu as _block
    _fs_spmm, _fs_block_gpu = _spmm, _block
    return _fs_spmm, _fs_block_gpu


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
        return False, f"FS_SpMM/FS_Block_gpu import failed: {type(e).__name__}: {e}"
    return True, ""


def create(precision: str):
    return FlashSparseSpMM(precision)


class FlashSparseSpMM:
    name = IMPL_NAME
    platform = "cuda"

    def __init__(self, precision: str = "fp16"):
        if precision != "fp16":
            raise NotImplementedError(
                f"{IMPL_NAME} is wired for the fp16 WMMA kernel "
                f"(FS_SpMM.forward_fp16) only; a forward_tf32 entry point "
                f"also exists in the artifact but is not wired here; "
                f"requested {precision}")
        self.precision = precision

    def prepare(self, matrix, params: dict):
        import torch
        fs_spmm, fs_block = _load_modules()

        A = matrix.csr
        M, K = A.shape
        if M != K:
            raise NotImplementedError(
                f"{IMPL_NAME}: FS_Block_gpu.preprocess_gpu_fs() takes a "
                f"single num_nodes for both matrix dimensions (assumes a "
                f"square GNN-adjacency-style matrix); requested shape "
                f"({M}, {K}) is rectangular")
        N = int(params["N"])
        nnz = int(A.nnz)

        # Pad M to a multiple of BLK_H=8 (window size) -- preprocess_gpu_fs
        # computes window_num = num_nodes / blockSize_h via integer
        # division, so a non-multiple silently truncates the tail window.
        m_pad = ((M + BLK_H - 1) // BLK_H) * BLK_H
        indptr_pad = np.empty(m_pad + 1, dtype=np.int32)
        indptr_pad[: M + 1] = A.indptr
        indptr_pad[M + 1:] = A.indptr[-1]  # empty trailing (padding) rows

        # preprocess_gpu_fs / forward_fp16 both take CPU-resident tensors
        # and do their own H2D/D2H internally (see module docstring).
        row_ptr = torch.from_numpy(indptr_pad)
        col_idx = torch.from_numpy(np.ascontiguousarray(A.indices, dtype=np.int32))

        # --- artifact's own format conversion: CSR -> FlashSparse's
        # compressed tensor-core format (8x8 windows), timed as
        # preprocessing per the harness contract. NOTE: takes no values
        # array -- see module docstring's IMPORTANT FINDING. ---
        row_window_offset, sparse_atox_idx, values, _prep_ms = fs_block.preprocess_gpu_fs(
            row_ptr, col_idx, m_pad, nnz, BLK_H, BLK_W)

        # numpy RNG matching cpu_ref.reference_spmm's _dense_operand exactly
        # (see insum/adapter.py's docstring for why this matters), cast to
        # fp16 CPU tensor (forward_fp16's own expected dtype).
        rng = np.random.default_rng(params.get("seed", 42))
        B_np = rng.uniform(-1.0, 1.0, size=(K, N)).astype(np.float16)
        rhs = torch.from_numpy(B_np)

        return {
            "row_window_offset": row_window_offset,
            "sparse_atox_idx": sparse_atox_idx,
            "values": values,
            "rhs": rhs,
            "m_pad": m_pad, "N": N, "M": M,
        }

    def run(self, h):
        fs_spmm, _ = _load_modules()
        # epoches=1: the artifact's own internal timed loop count (it also
        # hardcodes a separate, always-on 10-iteration warmup -- see module
        # docstring). Returns [output_cpu_tensor, avg_ms_tensor]; we only
        # need the output for the harness's own (per-run() CudaEventTimer)
        # timing and correctness gate.
        out, _avg_ms = fs_spmm.forward_fp16(
            h["row_window_offset"], h["sparse_atox_idx"], h["values"],
            h["rhs"], h["m_pad"], h["N"], h["M"], 1)
        return out

    def to_host(self, out):
        # Already a CPU tensor (forward_fp16 copies D2H internally).
        return out.detach().to(dtype=__import__("torch").float64).numpy()

    def timer(self):
        from kernelbench.impls.gpu_cuda import CudaEventTimer
        return CudaEventTimer()

    def free(self, h):
        h.clear()
