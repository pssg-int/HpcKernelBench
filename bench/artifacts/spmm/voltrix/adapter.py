"""
Voltrix adapter for the spmm track.

Paper: "Voltrix: Sparse Matrix-Matrix Multiplication on Tensor Cores with
Asynchronous and Balanced Kernel Optimization", USENIX ATC'25.
`PAPER_KEY = conf/usenix/XiaWY0C25`. Artifact:
https://github.com/YaqiXia/Voltrix-SpMM -- a Hopper-only (TMA/mbarrier)
tensor-core SpMM kernel, JIT-compiled at runtime (own `voltrix.jit`
framework, same env-var-driven design as DeepGEMM/DeepEP; see
voltrix/jit/compiler.py).

Written 2026-09-09 when this integration was moved from DEFERRED-HARDWARE
to an actual H100 (sm_90a) build+gate -- see STATUS.md's REQUIRES_GPU /
"Reproduction on zaratan" section for the empirical sm_80 failure this
supersedes and the hardware evidence (hardcoded
`-gencode=arch=compute_90a,code=sm_90a`, genuine `cp.async.bulk`/`mbarrier`
PTX) that made DEFERRED-HARDWARE (not SKIPPED) the right original ruling
per ARTIFACT_GUIDE.md rule 9.

## What was wrapped (rule 1)

Voltrix's own top-level Python API, unmodified: `voltrix.csr_preprocess`
(`voltrix/spmm/spmm.py` -- CSR -> Voltrix's own DTC-SpMM-lineage compressed
tensor-core format: 16x8-tiled row windows, bit-packed `hspa_packed`,
`hind` column index; this is genuinely the SAME `BLK_H=16`/`BLK_W=8`
row-window scheme as `spmm/dtcspmm`'s ME-TCF format -- Voltrix vendors
DTC-SpMM as `third-party/DTC-SpMM` and its preprocessing kernels
(`preprocess_kernel`/`hmat_gen_kernel`/`hmat_packed_swizzle_kernel`) are
this same lineage's own from-scratch JIT reimplementation, not a call into
the vendored copy) and `voltrix.spmm` (`voltrix/jit_kernels/spmm.py` ->
`voltrix::voltrix_spmm_forward_cuda`, the actual TMA/mbarrier
warp-specialized async-copy Tensor-Core SpMM kernel -- the paper's own
contribution). Both are called exactly as `tests/test_spmm.py` (the
artifact's own smoke test) calls them -- no ctypes shim needed, Voltrix's
Python layer already IS the kernel entry point.

## IMPORTANT FINDING: same structural binary-adjacency limitation as
## DTC-SpMM/FlashSparse/GeneralSparse (this track's `spmm-binary-adjacency-
## kernel` variant exists for exactly this)

`voltrix.csr_preprocess(indptr, indices, num_nodes)` takes NO values
parameter (confirmed by reading `voltrix/spmm/spmm.py` in full -- its
signature is `(indptr, indices, num_nodes)`, and the packed `hspa`/
`hspa_packed` buffers it builds via `hmat_gen_kernel`/
`hmat_packed_swizzle_kernel` are populated purely from `edge_to_column`/
`edge_to_row`/structure, never from a caller-supplied value array). This is
the exact same finding `dtcspmm/adapter.py` and `flashsparse/adapter.py`
document for their own artifacts (Voltrix vendors DTC-SpMM directly, see
above -- unsurprising it inherits the same structural limitation). This
adapter therefore computes `A_pattern @ B`, gated under
`spmm-binary-adjacency-kernel` (whose harness-level
`kernelbench.domains.sparse.variant_transform` hook binarizes the
REFERENCE too, for every implementation, before either sees it -- see
`dtcspmm/STATUS.md`'s "2026-09-06: re-gated under
spmm-binary-adjacency-kernel" section for the precedent this adapter
follows directly), not `spmm-gpu-kernel-f32`/`spmm-tensorcore-fp16` (which
require real-valued A and this kernel structurally cannot supply that).

`num_nodes` is a single scalar used for both matrix dimensions (square,
GNN-adjacency-style) -- same constraint as DTC-SpMM/FlashSparse; a
rectangular workload raises `NotImplementedError` (rule 8).

## Precision

`spmm_kernel`'s own asserts require `input.dtype == output.dtype ==
torch.float` (`voltrix/jit_kernels/spmm.py`) -- fp32 storage/API
throughout (the Tensor-Core compute precision inside the compiled kernel
itself was not independently verified from the `.cuh` source within this
integration's budget; only the Python-visible dtype contract is asserted
here). `PRECISIONS = ["fp32"]`, gated with `--precision fp32` against this
variant's `< 1e-4` fp32 tolerance.

## Preprocessing (rule 2, timed in prepare())

`voltrix.csr_preprocess` -- Voltrix's own CSR -> compressed-tensor-core-
format conversion (`preprocess_kernel` + `hmat_gen_kernel` +
`hmat_packed_swizzle_kernel`, each independently JIT-compiled+autotuned on
first call, cached under `VOLTRIX_CACHE_DIR` thereafter) -- called exactly
as the artifact's own `tests/test_spmm.py` calls it, timed as
preprocessing per the harness contract.

Dense operand B: numpy `default_rng` matching `cpu_ref.reference_spmm`'s
`_dense_operand` exactly (see `spmm/insum/adapter.py`'s docstring for why
this matters).
"""

from __future__ import annotations

import os
import sys

import numpy as np

KERNEL = "spmm"
IMPL_NAME = "voltrix-spmm"
PAPER_KEY = "conf/usenix/XiaWY0C25"
PRECISIONS = ["fp32"]

_HERE = os.path.dirname(os.path.abspath(__file__))
_SRC = os.path.join(_HERE, "source")

_voltrix = None


def _ensure_env():
    """sys.path + Voltrix's own JIT-cache-location env knob, both needed
    before `import voltrix` -- matches build.sh's identical default so a
    runner process (which does not source build.sh) still reuses the same
    warm JIT cache build.sh populated. Does NOT compile anything (rule 10):
    `import voltrix` only imports function/constant DEFINITIONS -- see
    voltrix/__init__.py + voltrix/jit_kernels/__init__.py -- nothing at
    module scope calls compile_and_tune()."""
    if _SRC not in sys.path:
        sys.path.insert(0, _SRC)
    os.environ.setdefault("VOLTRIX_CACHE_DIR", os.path.join(_HERE, "cache"))


def _load_module():
    global _voltrix
    if _voltrix is not None:
        return _voltrix
    if not os.path.isdir(_SRC):
        raise FileNotFoundError(f"{_SRC} not found -- run build.sh (clone missing?)")
    _ensure_env()
    import voltrix as _m
    if not (hasattr(_m, "csr_preprocess") and hasattr(_m, "spmm")):
        raise ImportError(
            f"voltrix resolved to '{_m.__file__}' but is missing csr_preprocess/spmm")
    _voltrix = _m
    return _voltrix


def available() -> tuple[bool, str]:
    if not os.path.isdir(_SRC):
        return False, f"{_SRC} not found -- run build.sh"
    try:
        import torch
    except Exception as e:
        return False, f"torch import failed: {type(e).__name__}: {e}"
    if not torch.cuda.is_available():
        return False, "CUDA not available on this host"
    cap = torch.cuda.get_device_capability(0)
    if cap < (9, 0):
        return False, (
            f"Voltrix needs Hopper-or-newer (sm_90a: TMA + mbarrier, "
            f"hardcoded -gencode=arch=compute_90a in voltrix/jit/compiler.py); "
            f"this device reports compute capability {cap[0]}.{cap[1]}")
    try:
        _load_module()
    except Exception as e:
        return False, f"voltrix import failed: {type(e).__name__}: {e}"
    return True, ""


def create(precision: str):
    return VoltrixSpMM(precision)


class VoltrixSpMM:
    name = IMPL_NAME
    platform = "cuda"

    def __init__(self, precision: str = "fp32"):
        if precision != "fp32":
            raise NotImplementedError(
                f"{IMPL_NAME}: spmm_kernel's own asserts require "
                f"input/output dtype == torch.float (voltrix/jit_kernels/"
                f"spmm.py) -- fp32 only; requested {precision}")
        self.precision = precision

    def prepare(self, matrix, params: dict):
        import torch
        voltrix = _load_module()

        A = matrix.csr
        M, K = A.shape
        if M != K:
            raise NotImplementedError(
                f"{IMPL_NAME}: voltrix.csr_preprocess() takes a single "
                f"num_nodes for both matrix dimensions (assumes a square "
                f"GNN-adjacency-style matrix, inherited from the vendored "
                f"DTC-SpMM preprocessing lineage -- see module docstring); "
                f"requested shape ({M}, {K}) is rectangular")
        N = int(params["N"])
        nnz = int(A.nnz)

        indptr = torch.from_numpy(np.ascontiguousarray(A.indptr, dtype=np.int32))
        indices = torch.from_numpy(np.ascontiguousarray(A.indices, dtype=np.int32))

        # --- artifact's own format conversion: CSR -> Voltrix's compressed
        # tensor-core format (16x8 row windows), timed as preprocessing per
        # the harness contract. NOTE: takes no values array -- see module
        # docstring's IMPORTANT FINDING. ---
        blk_offsets, hspa_packed, hind = voltrix.csr_preprocess(indptr, indices, M)
        # Stable, run-specific hash_tag (see spmm.py::feature_hash): avoids
        # the artifact's own "hash_tag not set, falling back to memory
        # address" perf warning and gives the JIT autotuner's per-kernel
        # cache a meaningful, reproducible key instead of a raw pointer.
        hspa_packed.hash_tag = f"kb_{IMPL_NAME}_{M}_{nnz}"

        # numpy RNG matching cpu_ref.reference_spmm's _dense_operand exactly
        # (see spmm/insum/adapter.py's docstring for why this matters).
        rng = np.random.default_rng(params.get("seed", 42))
        B_np = rng.uniform(-1.0, 1.0, size=(K, N)).astype(np.float32)
        feat = torch.as_tensor(B_np, device="cuda")

        return {
            "blk_offsets": blk_offsets, "hspa_packed": hspa_packed, "hind": hind,
            "M": M, "num_edges": nnz, "feat": feat,
        }

    def run(self, h):
        voltrix = _load_module()
        # voltrix.spmm() allocates a fresh torch.empty(...) output every
        # call (voltrix/spmm/spmm.py) -- no accumulation across calls, no
        # manual zeroing needed (same convention as DTC-SpMM's run_DTCSpMM).
        return voltrix.spmm(
            h["blk_offsets"], h["hspa_packed"], h["hind"],
            num_nodes=h["M"], num_edges=h["num_edges"], feat=h["feat"])

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
