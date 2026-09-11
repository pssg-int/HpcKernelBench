"""
NM-SpMM adapter for the spmm track.

Paper: "NM-SpMM: Efficient N:M Sparsity for Deep Learning" (IPDPS'25).
`PAPER_KEY = conf/ipps/MaWDCH0ZWZCDWFP25`. Artifact:
https://github.com/CMa-X/NM-SpMM -- a hierarchically-blocked N:M-structured
GPU GEMM kernel: A[M,K] (dense) @ Wsparse[K,N] (N:M-pruned along K), fp32
CUDA-core (cp.async-based, no wmma/mma.sync -- confirmed by grepping
source/src/*.cu and source/include/ptx.h for "half"/"wmma"/"mma": none
found; every buffer is `float*`). This is a real, DIFFERENT precision family
from what the task brief anticipated ("fp16 tensor cores -> fp16 bound
1e-2"): NM-SpMM is plain fp32, so it is gated under `spmm-gpu-kernel-f32`
(tolerance ~1e-4), not `spmm-tensorcore-fp16`.

## Regime: PARTIAL (needs an already N:M-structured sparse operand)

`spmm-gpu-quantized-int` (this track's variant for "quantized low-precision
SpMM on Tensor Cores") names NM-SpMM in its claim text alongside Magicube/
InferFast/GeneralSparse, but that variant's protocol -- packed int4/int8
values, EXACT integer-match correctness (no tolerance: "this is integer
arithmetic, not floating point") -- does not fit this artifact at all: NM-
SpMM computes real fp32 arithmetic on `float*` buffers, never touches an
integer MAC path. Checked and ruled out (ARTIFACT_GUIDE.md rule 8's "check
whether an existing variant fits" step): no existing benchspecs/spmm/
spec.yaml variant is actually a home for "fp32 GEMM against an N:M-pruned
K x N weight" other than spmm-gpu-kernel-f32's own tolerance family, so this
adapter targets that variant's correctness bound but cannot consume that
variant's own `recommended_subset` (general SuiteSparse matrices) -- see
below.

NM-SpMM's kernel does not take arbitrary CSR. Tracing `source/tests/
test_nmspmm.cu` + `source/include/utils.h::init_data` (the artifact's ONLY
data-generation path -- it ships no general-CSR-to-NM-format converter, same
situation as SSpMM's vector-sparse format, see ../sspmm/adapter.py) shows the
sparse operand's own structural contract: for a GEMM `C[M,N] = A[M,K] @
Wsparse[K,N]`, group the N (output-column) axis into chunks of
`VEC_LEN=32`; for EVERY 32-column chunk, and for every contiguous
`pruning_M`-wide block of the K (contraction) axis, EXACTLY `pruning_N` of
the `pruning_M` rows survive pruning, and -- critically -- the survivING
row-SET is shared identically across ALL 32 columns in that chunk (values
differ per column, but the zero/nonzero PATTERN does not: `init_data`'s
`DT[(k+u)*Q+a] = tmp_index[u]` is indexed only by (compressed-row, column-
chunk `a`), never by an individual column within the chunk). This is a
coarser, "vector-wise" N:M sparsity, not the fine-grained per-column
independent pattern (e.g. NVIDIA's 2:4) most "N:M sparsity" literature
means -- see the module-level NOTE below for the exact ratios this build
wires and what the artifact supports overall.

This track's spec puts the sparse operand on the LEFT (`C[M,N] =
A[M,K](sparse) * B[K,N](dense)`); NM-SpMM puts its pruned operand on the
RIGHT (`C[M,N] = A[M,K](dense) * Wsparse[K,N]`). This adapter reconciles the
two via the standard transpose identity `A @ B = (B^T @ A^T)^T`: given the
harness's `matrix.csr` (shape Mh x Kh, the "sparse operand") and its dense
`B` (Kh x Nh, built by `cpu_ref._dense_operand`, matched exactly here), it
computes NM-SpMM's own GEMM with `M_nm=Nh, K_nm=Kh, N_nm=Mh`,
`A_nm = B^T` (dense, col-major) and `Wsparse_nm = matrix.csr^T`
(K_nm x N_nm = Kh x Mh, N:M-pruned along K_nm), producing `C_nm = C^T`,
transposed back to `(Mh, Nh)` in `to_host()`. Concretely this means: in the
ORIGINAL matrix.csr's own axes, groups of 32 CONSECUTIVE ROWS must share an
identical nonzero-column PATTERN within every `pruning_M`-wide column block
(values may differ per row) -- a structural property essentially no general
SuiteSparse matrix has (each row's sparsity pattern is independent), hence
`prepare()` VERIFIES this and raises `NotImplementedError` naming the
constraint when it does not hold, per ARTIFACT_GUIDE.md rule 8. The
verified, PASSING gate instead runs on a standalone synthetic N:M workload
built to satisfy the constraint exactly (`gate_synthetic_nm.py`, this
directory -- precedent: `../../spmv/diaq/STATUS.md`'s substitute gate),
checked against `kernelbench.impls.cpu_ref.reference_spmm` on the SAME
matrix -- the harness's own independent fp64 reference, never a loosened
tolerance.

## What NM-SpMM's N:M configs actually are (survey finding, not "2:4")

`source/tests/test_nmspmm.cu` hardcodes exactly 4 supported `sparsity`
values, each with a FIXED compile-time block width (not runtime-tunable --
`pruning_M` is passed to the CLI but never forwarded into any kernel launch;
the real block width is the template constant `Ks` baked into each
`nmGEMM_*_{low,high}_sparsity` dispatcher in `source/src/kernel_*.cu`):

| sparsity | kept : block (Ks) | reduced ratio | preprocessing path |
|---|---|---|---|
| 0.5   | 16 : 32 | 1:2  | low  (`PreProcessing_low_sparsity` only) |
| 0.625 | 12 : 32 | 3:8  | low  (`PreProcessing_low_sparsity` only) |
| 0.75  | 16 : 64 | 1:4  | high (`transIndex` + `PreProcessing_high_sparsity` + `column_info`) |
| 0.875 |  8 : 64 | 1:8  | high (same as above) |

None of these is the fine-grained "2:4" (block-of-4) pattern most N:M
literature targets -- NM-SpMM's own contribution is exactly this coarser,
hardware-friendlier block width (32 or 64) that amortizes metadata cost
across a whole vector of output columns, per its README's "hierarchical
blocking" claim.

**This adapter wires ONLY sparsity=0.5 (16:32, the `low` path) on the
smallest kernel tile (`kernel_32x32_4x4`, `Ns=32` -- picked because it makes
the `PreProcessing_low_sparsity` layout swizzle degenerate to `Qs=1`, the
simplest and least failure-prone case to port faithfully)** -- see
`wrapper.cu`'s header comment and STATUS.md's "Not done" section for the
0.625/0.75/0.875 configs and the 3 larger kernel tiles that exist in the
artifact but were not wired here (login-node integration budget).
"""

from __future__ import annotations

import ctypes
import os

import numpy as np

KERNEL = "spmm"
IMPL_NAME = "nmspmm-32x32-n2m1to2"
PAPER_KEY = "conf/ipps/MaWDCH0ZWZCDWFP25"
PRECISIONS = ["fp32"]

VEC_LEN = 32          # fixed by the artifact (NM-SpMM.h's VEC_LEN macro)
PRUNING_M = 32         # block width wired here (the "low sparsity", Ks=32 path)
PRUNING_N = 16         # kept per block at sparsity=0.5 (16:32 == 1:2)
SPARSITY = 0.5
NS_TILE = 32           # kernel_32x32_4x4's own N-tile (Qs = Ns/VEC_LEN = 1)

_HERE = os.path.dirname(os.path.abspath(__file__))
_LIB_PATH = os.path.join(_HERE, "libnmspmm_wrapper.so")

_lib = None


def _load_lib():
    global _lib
    if _lib is not None:
        return _lib
    if not os.path.exists(_LIB_PATH):
        raise FileNotFoundError(f"{_LIB_PATH} not built -- run build.sh")
    lib = ctypes.CDLL(_LIB_PATH)
    p = ctypes.c_void_p
    lib.nm_preprocess_low.argtypes = [p, ctypes.c_int, ctypes.c_int, ctypes.c_int]
    lib.nm_preprocess_low.restype = None
    lib.nm_gemm_32x32_low.argtypes = [p, p, p, p, ctypes.c_int, ctypes.c_int,
                                      ctypes.c_int, ctypes.c_int,
                                      ctypes.c_float, ctypes.c_int]
    lib.nm_gemm_32x32_low.restype = ctypes.c_int
    _lib = lib
    return _lib


def available() -> tuple[bool, str]:
    if not os.path.exists(_LIB_PATH):
        return False, "libnmspmm_wrapper.so not built -- run build.sh"
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
    return NMSpMM(precision)


def verify_and_extract_nm_pattern(csr, pruning_M: int = PRUNING_M,
                                  pruning_N: int = PRUNING_N,
                                  vec_len: int = VEC_LEN):
    """
    Verify `csr` (Mh x Kh) satisfies NM-SpMM's structural contract (see
    module docstring) and, if so, extract the per-(row-group, col-block)
    shared column pattern + per-row values needed to build the artifact's
    compressed inputs.

    Returns (patterns, values) where:
      patterns[g][kb] = sorted array of `pruning_N` LOCAL column offsets
                        (0..pruning_M-1) shared by every row in row-group g
                        within column-block kb.
      values[g][kb]   = (vec_len, pruning_N) array, values[r, u] = csr's
                        value at (row=g*vec_len+r, col=kb*pruning_M+patterns[g][kb][u]).

    Raises NotImplementedError (naming the constraint, per ARTIFACT_GUIDE.md
    rule 8) if ANY row-group/col-block fails to have exactly `pruning_N`
    nonzeros with a pattern identical across all `vec_len` rows in the group.
    """
    Mh, Kh = csr.shape
    if Mh % vec_len != 0:
        raise NotImplementedError(
            f"{IMPL_NAME}: N:M structured sparsity required; matrix.csr's "
            f"row count ({Mh}) must be a multiple of VEC_LEN={vec_len} "
            f"(NM-SpMM shares one N:M pattern across every {vec_len}-row "
            f"group) -- general SuiteSparse matrices do not satisfy this")
    if Kh % pruning_M != 0:
        raise NotImplementedError(
            f"{IMPL_NAME}: N:M structured sparsity required; matrix.csr's "
            f"column count ({Kh}) must be a multiple of pruning_M="
            f"{pruning_M} -- general SuiteSparse matrices do not satisfy this")

    n_groups = Mh // vec_len
    n_blocks = Kh // pruning_M
    indptr, indices, data = csr.indptr, csr.indices, csr.data
    patterns = [[None] * n_blocks for _ in range(n_groups)]
    values = [[None] * n_blocks for _ in range(n_groups)]

    for g in range(n_groups):
        row0 = g * vec_len
        # per-row {local_col_in_block: value} maps, split by block, for this group
        row_block_maps = []
        for r in range(vec_len):
            row = row0 + r
            lo, hi = indptr[row], indptr[row + 1]
            cols = indices[lo:hi]
            vals = data[lo:hi]
            blocks = [dict() for _ in range(n_blocks)]
            for c, v in zip(cols.tolist(), vals.tolist()):
                kb, local = divmod(int(c), pruning_M)
                blocks[kb][local] = v
            row_block_maps.append(blocks)

        for kb in range(n_blocks):
            ref_cols = frozenset(row_block_maps[0][kb].keys())
            if len(ref_cols) != pruning_N:
                raise NotImplementedError(
                    f"{IMPL_NAME}: N:M structured sparsity required; "
                    f"general SuiteSparse matrices do not satisfy it -- "
                    f"row-group {g} (rows {row0}..{row0+vec_len-1}), "
                    f"column block {kb} (cols {kb*pruning_M}.."
                    f"{kb*pruning_M+pruning_M-1}): row {row0} has "
                    f"{len(ref_cols)} nonzeros in this block, expected "
                    f"exactly pruning_N={pruning_N} (sparsity={SPARSITY})")
            for r in range(1, vec_len):
                cols_r = frozenset(row_block_maps[r][kb].keys())
                if cols_r != ref_cols:
                    raise NotImplementedError(
                        f"{IMPL_NAME}: N:M structured sparsity required; "
                        f"general SuiteSparse matrices do not satisfy it -- "
                        f"row-group {g}, column block {kb}: row {row0+r}'s "
                        f"nonzero-column pattern {sorted(cols_r)} differs "
                        f"from row {row0}'s {sorted(ref_cols)} (NM-SpMM's "
                        f"vector-sparse format requires ALL {vec_len} rows "
                        f"in a group to share one pattern per block)")
            sorted_cols = sorted(ref_cols)
            patterns[g][kb] = np.array(sorted_cols, dtype=np.int32)
            block = np.empty((vec_len, pruning_N), dtype=np.float64)
            for r in range(vec_len):
                m = row_block_maps[r][kb]
                for u, c in enumerate(sorted_cols):
                    block[r, u] = m[c]
            values[g][kb] = block
    return patterns, values


class NMSpMM:
    name = IMPL_NAME
    platform = "cuda"
    _last_shape = None   # (Nh, Mh) = (M_nm, N_nm)

    def __init__(self, precision: str = "fp32"):
        if precision != "fp32":
            raise NotImplementedError(
                f"{IMPL_NAME} only wired for fp32 (every NM-SpMM kernel "
                f"buffer is `float*`; no half/tensor-core path exists in "
                f"the artifact -- grepped source/src/*.cu and "
                f"source/include/ptx.h for wmma/mma.sync/half, none found); "
                f"requested {precision}")
        self.precision = precision

    def prepare(self, matrix, params: dict):
        import torch
        lib = _load_lib()

        Nh = int(params["N"])
        Mh, Kh = matrix.csr.shape
        # kernel_32x32_4x4's tile: Ms=Ns=32. M_nm=Nh must be a multiple of
        # Ms=32; N_nm=Mh must be a multiple of Ns=32 (this is also required
        # by the VEC_LEN=32 pruning-group constraint below, so no new
        # restriction beyond what verify_and_extract_nm_pattern needs).
        if Nh % 32 != 0:
            raise NotImplementedError(
                f"{IMPL_NAME}: kernel_32x32_4x4's M-tile is 32 "
                f"(dimGrid uses M_nm/32 where M_nm=N, the dense operand "
                f"width); N={Nh} is not a multiple of 32")

        # --- verify + extract the N:M pattern (this IS the structural gate
        # rule 8 asks for: raise NotImplementedError naming the constraint
        # rather than silently computing garbage on an unstructured input) ---
        patterns, values = verify_and_extract_nm_pattern(matrix.csr)
        n_groups = Mh // VEC_LEN
        n_blocks = Kh // PRUNING_M
        W = Kh // 2   # compressed K dimension at sparsity=0.5

        # --- B_compressed: row-major (W, Mh) float32, matches the artifact's
        # "BT" (row-major dense weight values) exactly: BT[k*N_nm+j] holds
        # the SAME numbers as the artifact's col-major B[k+j*W] -- a
        # relabeling of the same values, not a transpose (see module
        # docstring / utils.h's own BT[i*N+j] = B[i+j*W] construction). ---
        B_compressed = np.zeros((W, Mh), dtype=np.float32)
        # --- DT_raw: row-major (W, n_groups) int32, LOCAL block-relative
        # offsets (0..PRUNING_M-1), exactly init_data()'s
        # DT[(k+u)*Q+a] = tmp_index[u] construction, before the artifact's
        # own PreProcessing_low_sparsity layout swizzle is applied. ---
        DT_raw = np.zeros((W, n_groups), dtype=np.int32)

        for g in range(n_groups):
            for kb in range(n_blocks):
                base_w = kb * PRUNING_N
                cols_local = patterns[g][kb]          # (PRUNING_N,) local offsets, sorted
                block_vals = values[g][kb]             # (VEC_LEN, PRUNING_N)
                for u in range(PRUNING_N):
                    DT_raw[base_w + u, g] = int(cols_local[u])
                    B_compressed[base_w + u, g * VEC_LEN:(g + 1) * VEC_LEN] = \
                        block_vals[:, u].astype(np.float32)

        # --- artifact's OWN preprocessing (unmodified C function, called
        # via ctypes -- ARTIFACT_GUIDE.md rule 2: this IS the artifact's
        # format conversion, ported nowhere, just invoked in-place) ---
        DT_raw = np.ascontiguousarray(DT_raw)
        lib.nm_preprocess_low(
            DT_raw.ctypes.data_as(ctypes.c_void_p), W, n_groups, NS_TILE)

        # --- A_nm = B_dense^T, col-major (Fortran order), shape (Nh, Kh) --
        # matches cpu_ref.reference_spmm's _dense_operand(K,N,seed) exactly
        # (see rode/sspmm adapters' docstrings for why this RNG match matters)
        rng = np.random.default_rng(params.get("seed", 42))
        B_dense = rng.uniform(-1.0, 1.0, size=(Kh, Nh)).astype(np.float32)
        A_nm = np.asfortranarray(B_dense.T)   # (Nh, Kh), col-major

        A_dev = torch.from_numpy(A_nm.copy(order="F")).cuda()
        B_dev = torch.from_numpy(B_compressed).cuda()
        D_dev = torch.from_numpy(DT_raw).cuda()
        C_dev = torch.zeros(Nh * Mh, dtype=torch.float32, device="cuda")

        self._last_shape = (Nh, Mh)

        return {
            "A": A_dev, "B": B_dev, "D": D_dev, "C": C_dev,
            "M_nm": Nh, "N_nm": Mh, "K_nm": Kh, "W": W,
        }

    def run(self, h):
        lib = _load_lib()
        h["C"].zero_()
        ret = lib.nm_gemm_32x32_low(
            ctypes.c_void_p(h["A"].data_ptr()),
            ctypes.c_void_p(h["B"].data_ptr()),
            ctypes.c_void_p(h["D"].data_ptr()),
            ctypes.c_void_p(h["C"].data_ptr()),
            h["M_nm"], h["N_nm"], h["K_nm"], h["W"],
            ctypes.c_float(SPARSITY), 1)
        if ret != 0:
            raise RuntimeError(f"{IMPL_NAME}: nm_gemm_32x32_low returned "
                               f"cudaError_t={ret}")
        return h["C"]

    def to_host(self, out):
        import torch
        Nh, Mh = self._last_shape
        # C_nm is row-major (M_nm=Nh, N_nm=Mh); the harness wants (Mh, Nh) =
        # C_nm^T (see module docstring's transpose identity).
        mat = out.detach().to("cpu", dtype=torch.float64).view(Nh, Mh)
        return mat.t().contiguous().numpy()

    def timer(self):
        from kernelbench.impls.gpu_cuda import CudaEventTimer
        return CudaEventTimer()

    def free(self, h):
        import torch
        h.clear()
        torch.cuda.empty_cache()
