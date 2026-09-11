"""
SSpMM adapter for the spmm track.

Paper: "SSpMM: A Scalable and Efficient Tensor-Core-Based SpMM Library
Across GPU Generations", TPDS'25. `PAPER_KEY = journals/tpds/XueWYTLFSCSL25`.
Artifact: https://github.com/xuezy-mmi/SSpMM (a vectorSparse-lineage
Tensor-Core SpMM: `source/SSpMM/` is the paper's own library; `vectorSparse`,
`magicube`, `tSparse` sibling directories in the same repo are baselines the
paper compares against, NOT wrapped here).

Kernel entry point wrapped (ARTIFACT_GUIDE.md rule 1): `spmm::SSpMM(...)`
(`source/SSpMM/include/sspmm.cuh` / `src/sspmm.cu`) -- the paper's own
kernel ("Ro-SpMM" in the artifact's own test driver, `sspmm_test.cpp`, as
opposed to `spmm::SpMM`, "mma884", the vendored vectorSparse baseline this
adapter deliberately does NOT wrap). Called via `libsspmm_wrapper.so`
(this directory's wrapper.cu, NOT under source/), which links unmodified
against `sspmm.o` (also compiled unmodified by build.sh) and only adds
`extern "C"` linkage -- see wrapper.cu.

## Input format ("vector-sparse" CSR) and why prepare() must build it itself

`spmm::SSpMM` does not take plain CSR. It takes a variant where rows are
grouped into fixed-size blocks of `vec_length` (8 here -- see below), and
EVERY row within a group shares the SAME set of nonzero columns: one
`column_indices` entry per "vector-column" `j` names a column shared by the
whole group, and `values[j*vec_length + v]` gives the group's `v`-th row's
value at that column (0.0 if that particular row didn't actually have a
nonzero there). This is the standard "vectorized"/blocked-ELL-style sparse
format vectorSparse/Magicube-lineage kernels use; the artifact ships no
general CSR->vector-sparse converter of its own -- its own benchmark driver
(`sspmm_test.cpp`) only ever reads ALREADY-vectorized DLMC text files off
disk (`../benchmark/dlmc-v8/...`). Converting an arbitrary CSR into this
format is therefore something THIS adapter has to do (ARTIFACT_GUIDE.md
rule 2's "artifact's own format conversion goes in prepare()" -- here there
is no artifact code to call for it, so this is the standard technique
applied to arbitrary input rather than a reimplementation of anything the
kernel or its preprocessing does): for each group of `vec_length` rows, take
the UNION of their nonzero columns and explicitly store a 0.0 value for any
row in the group missing a nonzero at a column another row in the group
has. This padding is always an EXACT literal zero -- it can only ADD
explicit zero terms to the sum, never drop or approximate a real nonzero,
so (unlike MP-SpMM's 2:4 pruning) it never changes the true product and the
gate compares against the ORIGINAL unpadded matrix with no special handling
needed.

`vec_length` is fixed to 8 in this adapter (the finer of the two values
`spmm::SSpMM`'s own `switch(vec_length)` supports -- 8 or 16, see
wrapper.cu -- chosen because a finer group size means less column-union
padding overhead on irregular/unstructured matrices). M is padded with
all-zero rows up to a multiple of 8 if needed (also exact -- cropped back
out in to_host()).

`row_indices` (a row-swizzle/reordering array in the artifact's own test
driver) is passed through to `spmm::SSpMM` but is NEVER dereferenced inside
ANY kernel in `src/sspmm.cu` -- every one of the 5 `__ldg(row_indices + ...)`
call sites in that file is commented out in the artifact's own source. It is
therefore a dead parameter as compiled; this adapter passes an identity
array (matching the "IdentityRowSwizzle" path the artifact's own driver also
defaults to) purely for documentation honesty, not because it affects the
result.

PRECISION: fp16 compute / fp32 accumulate (`spmm::SSpMM`'s mixed-precision
overload) -- gated under `spmm-tensorcore-fp16` per the task, `--precision
fp16`.
"""

from __future__ import annotations

import ctypes
import os

import numpy as np

KERNEL = "spmm"
IMPL_NAME = "sspmm-rospmm-v8"
PAPER_KEY = "journals/tpds/XueWYTLFSCSL25"
PRECISIONS = ["fp16"]

VEC_LENGTH = 8   # see module docstring for why 8 was chosen over 16

_HERE = os.path.dirname(os.path.abspath(__file__))
_LIB_PATH = os.path.join(_HERE, "libsspmm_wrapper.so")

_lib = None


def _load_lib():
    global _lib
    if _lib is not None:
        return _lib
    if not os.path.exists(_LIB_PATH):
        raise FileNotFoundError(f"{_LIB_PATH} not built -- run build.sh")
    lib = ctypes.CDLL(_LIB_PATH)
    p = ctypes.c_void_p
    lib.sspmm_run.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int,
                              p, p, p, p, p, p]
    lib.sspmm_run.restype = ctypes.c_int
    _lib = lib
    return _lib


def available() -> tuple[bool, str]:
    if not os.path.exists(_LIB_PATH):
        return False, "libsspmm_wrapper.so not built -- run build.sh"
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
    return SSpMM(precision)


def _vectorize_csr(csr, vec_length: int):
    """
    Convert an arbitrary CSR matrix into SSpMM's "vector-sparse" format (see
    module docstring). Returns (m_vec, row_offsets[int32], column_indices
    [int32], values[m_vec_groups... flattened, float16], Mp, padded_vec_nnz).

    Padding is always an exact literal 0.0 (never a dropped/approximated
    nonzero) -- see module docstring for why this keeps the gate valid
    against the ORIGINAL matrix.
    """
    M, K = csr.shape
    Mp = ((M + vec_length - 1) // vec_length) * vec_length
    m_vec = Mp // vec_length

    indptr = csr.indptr
    indices = csr.indices
    data = csr.data

    row_offsets = np.zeros(m_vec + 1, dtype=np.int64)
    col_groups: list[np.ndarray] = []
    val_groups: list[np.ndarray] = []  # each shape (nnz_vec_g, vec_length)

    for g in range(m_vec):
        cols_union: set[int] = set()
        row_maps = []
        for v in range(vec_length):
            r = g * vec_length + v
            if r < M:
                lo, hi = indptr[r], indptr[r + 1]
                rc = indices[lo:hi]
                rv = data[lo:hi]
                row_maps.append(dict(zip(rc.tolist(), rv.tolist())))
                cols_union.update(rc.tolist())
            else:
                row_maps.append({})  # zero-padded phantom row (M..Mp-1)
        cols_sorted = np.array(sorted(cols_union), dtype=np.int32)
        nnz_vec_g = len(cols_sorted)
        row_offsets[g + 1] = row_offsets[g] + nnz_vec_g
        col_groups.append(cols_sorted)
        if nnz_vec_g:
            block = np.zeros((nnz_vec_g, vec_length), dtype=np.float32)
            for v in range(vec_length):
                rm = row_maps[v]
                if rm:
                    for j, c in enumerate(cols_sorted.tolist()):
                        val = rm.get(c)
                        if val is not None:
                            block[j, v] = val
            val_groups.append(block)

    column_indices = (np.concatenate(col_groups) if col_groups else
                      np.zeros(0, dtype=np.int32))
    values = (np.concatenate(val_groups, axis=0) if val_groups else
             np.zeros((0, vec_length), dtype=np.float32)).astype(np.float16)
    padded_vec_nnz = int(row_offsets[-1])
    return (m_vec, row_offsets.astype(np.int32), column_indices,
            values.reshape(-1), Mp, padded_vec_nnz)


class SSpMM:
    name = IMPL_NAME
    platform = "cuda"
    _last_shape = None   # (Mp, N, M_orig)

    def __init__(self, precision: str = "fp16"):
        if precision != "fp16":
            raise NotImplementedError(
                f"{IMPL_NAME} only wired for fp16 (spmm::SSpMM's mixed "
                f"half*half->fp32-accumulate overload; the fp32*fp32 "
                f"overload in the artifact just prints \"doesn't support "
                f"float input\" and returns without computing anything); "
                f"requested {precision}")
        self.precision = precision

    def prepare(self, matrix, params: dict):
        import torch
        lib = _load_lib()

        N = int(params["N"])
        if N % 64 != 0:
            # spmm::SSpMM's grid is ceil(N / Tile_N) with Tile_N=64 hardcoded
            # in SSpMM_ex<float4,int,1,32,64,32> (src/sspmm.cu); N not a
            # multiple of 64 would either under-cover the output or read
            # past the intended tile -- not exercised by the paper's own
            # benchmarks (N in {32,64,128,...}), so this adapter refuses it
            # explicitly rather than guessing at padding semantics the
            # artifact itself never defines.
            raise NotImplementedError(
                f"{IMPL_NAME}: spmm::SSpMM's output tile is Tile_N=64 "
                f"(hardcoded in SSpMM_ex<...,64,...>); N must be a multiple "
                f"of 64, got N={N}")

        M_orig, K = matrix.csr.shape
        m_vec, row_offsets, column_indices, values, Mp, padded_vec_nnz = \
            _vectorize_csr(matrix.csr, VEC_LENGTH)

        row_indices = np.arange(m_vec, dtype=np.int32)  # dead parameter, see docstring

        rng = np.random.default_rng(params.get("seed", 42))
        B_np = rng.uniform(-1.0, 1.0, size=(K, N)).astype(np.float32).astype(np.float16)

        row_offsets_dev = torch.from_numpy(row_offsets).cuda()
        column_indices_dev = torch.from_numpy(column_indices).cuda()
        row_indices_dev = torch.from_numpy(row_indices).cuda()
        values_dev = torch.from_numpy(values).cuda()
        B_dev = torch.from_numpy(B_np).cuda()
        C_dev = torch.zeros(Mp * N, dtype=torch.float32, device="cuda")

        self._last_shape = (Mp, N, M_orig)

        return {
            "m_vec": m_vec, "N": N, "K": K,
            "row_indices": row_indices_dev, "row_offsets": row_offsets_dev,
            "column_indices": column_indices_dev, "values": values_dev,
            "B": B_dev, "C": C_dev,
            "Mp": Mp, "M_orig": M_orig,
            "padded_vec_nnz": padded_vec_nnz,  # diagnostic: see STATUS.md
        }

    def run(self, h):
        lib = _load_lib()
        h["C"].zero_()
        ret = lib.sspmm_run(
            h["m_vec"], VEC_LENGTH, h["N"], h["K"],
            ctypes.c_void_p(h["row_indices"].data_ptr()),
            ctypes.c_void_p(h["row_offsets"].data_ptr()),
            ctypes.c_void_p(h["column_indices"].data_ptr()),
            ctypes.c_void_p(h["values"].data_ptr()),
            ctypes.c_void_p(h["B"].data_ptr()),
            ctypes.c_void_p(h["C"].data_ptr()))
        if ret != 0:
            raise RuntimeError(f"{IMPL_NAME}: sspmm_run returned {ret} "
                               f"(cudaError_t, or -2 for unsupported "
                               f"vec_length -- should be unreachable)")
        return h["C"]

    def to_host(self, out):
        import torch
        Mp, N, M_orig = self._last_shape
        mat = out.detach().to("cpu", dtype=torch.float64).view(Mp, N)
        return mat[:M_orig, :].numpy()

    def timer(self):
        from kernelbench.impls.gpu_cuda import CudaEventTimer
        return CudaEventTimer()

    def free(self, h):
        import torch
        h.clear()
        torch.cuda.empty_cache()
