"""
Adapter for GE-SpMM (GE-SpMM: General-Purpose Sparse Matrix-Matrix
Multiplication on GPUs for Graph Neural Networks, SC'20, hgyhungry/ge-spmm)
-- the classic GNN aggregation SpMM kernel.

## What boundary was wrapped

`source/pytorch-custom/{spmm.cpp,spmm_kernel.cu}` is the artifact's own
torch CUDA extension (its `op.py` JIT-compiles it via
`torch.utils.cpp_extension.load(name='spmm', sources=['spmm.cpp',
'spmm_kernel.cu'])`); `csr_spmm(rowptr, colind, values, dense)` -- exposed
via `PYBIND11_MODULE(spmm, m)` -- is a genuine WEIGHTED CSR SpMM
(`spmm_test0`/`spmm_test1`/`spmm_test2` in `spmm_kernel.cu`, dispatched on
`dense.size(1)`, plain fp32 accumulate, no tensor cores): `A_csrVal[ptr] *
B_dnVal[offset]`, summed per output row -- exactly `Y = A @ X` for an
ARBITRARY real-valued `A`, taking the value array directly. This adapter
calls `csr_spmm` with our own `A_hat = D^-1/2(A+I)D^-1/2` fed straight in as
the CSR value array -- no sandwich trick, no per-edge-attention workaround
needed (unlike TC-GNN): this kernel's own native interface already IS
exactly this domain's kernel boundary. (`spmm_cuda_no_edge_value` /
`csr_spmm_no_edge_value`, the artifact's plain-adjacency variant used
without a `values` tensor, was read but not wrapped -- `csr_spmm` alone
already covers this domain's one reference exactly, with no extra
capability needed from the unweighted path.)

Note this is a DIFFERENT (and cleaner) boundary than the artifact's other
published example, `pytorch-custom/op.py`'s `GCNConv`, which reproduces GCN
symmetric normalization via the classic "pre/post row-scale the dense
operand around an UNWEIGHTED aggregation" trick
(`x = x*out_deg_norm; aggr_out = SPMMFunction.apply(...); aggr_out =
aggr_out*in_deg_norm`) -- exactly because `SPMMFunction.forward`'s
`edge_weight_csr=None` default path calls `csr_spmm_no_edge_value`, not
`csr_spmm`. Since `csr_spmm` (the weighted path) is directly available and
already computes precisely what this domain needs without any operand
sandwiching, this adapter uses it directly rather than replicating the
artifact's own `GCNConv` scaling trick.

## Preprocessing (prepare(), timed once)

Adjacency normalization (`A_hat = D^-1/2(A+I)D^-1/2`) is computed
independently here (`_gcn_normalize`, own code, not calling
`kernelbench.impls.cpu_ref` or any other adapter's copy, per the
reference-independence rule in DOMAIN_GUIDE.md). No other format conversion
is needed: `csr_spmm` consumes plain CSR (`rowptr`, `colind`, `values`)
directly, the SAME layout `matrix.csr` already is -- casting to
int32/float32 and moving to the GPU is the only "conversion" here.

## Build

`spmm.cpp`'s `csr2cscKernel` calls `cusparseCsr2cscEx2*` (used only by the
`csr2csc` pybind entry point, unrelated to `csr_spmm`/this adapter, but
compiled as part of the same translation unit); `build.sh` adds
`-lcusparse` since torch's `cpp_extension.load()` does not link cuSPARSE by
default the way it links cuBLAS/cudart. `source/pytorch-custom/op.py`'s own
`load(name='spmm', ...)` call is what fixes the compiled module's INTERNAL
name to `spmm` (from `PYBIND11_MODULE(spmm, m)` in the unmodified
`spmm.cpp`) -- build.sh's own `cpp_ext.load()` call uses the SAME
`name="spmm"` for exactly this reason (a different name here would build
fine but fail torch's own post-build import-verification step with
`ImportError: dynamic module does not define module export function`,
confirmed empirically while writing this adapter). `available()`/`_load()`
below therefore load the compiled `build/spmm.so` via
`importlib.util.spec_from_file_location("spmm", so_path)` rather than a
plain `sys.path.insert + import spmm` statement, to avoid ever registering
a module named the generic `"spmm"` into the shared process-wide
`sys.modules` (a real, if currently unrealized, collision risk if another
future artifact under this registry also happens to compile a module
literally named `spmm`).
"""

from __future__ import annotations

import importlib.util
import os

import numpy as np
import scipy.sparse as sp

HERE = os.path.dirname(os.path.abspath(__file__))
BUILD_DIR = os.path.join(HERE, "build")

KERNEL = "gnn-aggregation"
IMPL_NAME = "gespmm-csrspmm-gcnnorm"
PAPER_KEY = "conf/sc/HuangD0Y20"
# spmm_kernel.cu's spmm_test0/1/2 are plain float accumulate, no tensor cores.
PRECISIONS = ["fp32"]

_MOD = None
_IMPORT_ERROR = ""


def _load():
    global _MOD, _IMPORT_ERROR
    if _MOD is not None:
        return _MOD
    if _IMPORT_ERROR:
        raise RuntimeError(_IMPORT_ERROR)
    try:
        import torch  # noqa: F401 -- must be imported before the extension so its shared libs are resolvable
        so_path = os.path.join(BUILD_DIR, "spmm.so")
        if not os.path.exists(so_path):
            raise FileNotFoundError(so_path)
        # load via file path with the module's ACTUAL compiled-in name
        # ("spmm", from spmm.cpp's PYBIND11_MODULE(spmm, m)) -- see module
        # docstring's "Build" section for why a plain `import spmm` after a
        # sys.path insert is avoided here.
        spec = importlib.util.spec_from_file_location("spmm", so_path)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        _MOD = mod
        return _MOD
    except Exception as e:
        _IMPORT_ERROR = f"{type(e).__name__}: {e}"
        raise


def available() -> tuple[bool, str]:
    if not os.path.exists(os.path.join(BUILD_DIR, "spmm.so")):
        return False, "spmm.so not built -- run build.sh"
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
    kernelbench.impls.cpu_ref and every other adapter's own copy -- see
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


class GESpMMAggregation:
    name = IMPL_NAME
    platform = "cuda"
    normalization = "gcn-sym: D^-1/2 (A+I) D^-1/2 (independent copy, see module docstring)"

    def __init__(self, precision: str = "fp32"):
        if precision != "fp32":
            raise NotImplementedError(
                f"spmm_kernel.cu's spmm_test0/1/2 are hardcoded float "
                f"(fp32) buffers (torch::kFloat32 asserted in spmm.cpp); "
                f"requested precision {precision!r} is not serviced")
        self.precision = precision

    def prepare(self, matrix, params):
        import torch
        mod = _load()
        A = matrix.csr
        M, N = A.shape
        if M != N:
            raise NotImplementedError(
                f"GCN symmetric normalization (this domain's reference) "
                f"assumes a square adjacency; got a {M}x{N} matrix, not "
                f"serviced")
        F = int(params["N"])
        seed = params.get("seed", 42)

        A_hat = _gcn_normalize(A)
        params["gnn_agg_nnz"] = int(A_hat.nnz)
        params["gnn_agg_normalization"] = self.normalization

        rowptr = torch.tensor(A_hat.indptr, dtype=torch.int32, device="cuda")
        colind = torch.tensor(A_hat.indices, dtype=torch.int32, device="cuda")
        values = torch.tensor(A_hat.data, dtype=torch.float32, device="cuda")
        X = _dense_operand(M, F, seed, np.float32)
        dense = torch.as_tensor(X, device="cuda").contiguous()

        return {"mod": mod, "rowptr": rowptr, "colind": colind,
                "values": values, "dense": dense}

    def run(self, h):
        return h["mod"].csr_spmm(h["rowptr"], h["colind"], h["values"], h["dense"])

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
    return GESpMMAggregation(precision)
