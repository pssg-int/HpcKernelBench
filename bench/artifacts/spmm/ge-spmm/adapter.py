"""
Adapter for GE-SpMM (GE-SpMM: General-Purpose Sparse Matrix-Matrix
Multiplication on GPUs for Graph Neural Networks, SC'20, hgyhungry/ge-spmm)
-- spmm track.

REUSE, NOT A REBUILD (task brief): this artifact is already built for the
gnn-aggregation track at `../../gnn-aggregation/ge-spmm/` (`build.sh`
compiled `source/pytorch-custom/{spmm.cpp,spmm_kernel.cu}`, unmodified, via
`torch.utils.cpp_extension.load()`, into that directory's `build/spmm.so`).
GE-SpMM's compiled extension has no notion of "gnn-aggregation" vs "spmm" --
`csr_spmm(rowptr, colind, values, dense)` is a genuine WEIGHTED CSR SpMM
(`spmm_test0`/`spmm_test1`/`spmm_test2` in spmm_kernel.cu, dispatched on
`dense.size(1)`, plain fp32 multiply-accumulate, no tensor cores):
`A_csrVal[ptr] * B_dnVal[offset]`, summed per output row -- exactly
`C = A @ B` for an ARBITRARY real-valued CSR `A`, this track's own reference
(`kernelbench.impls.cpu_ref.reference_spmm` / `ScipySpMM`) with no operand
sandwiching or normalization step needed at all -- markedly SIMPLER than the
gnn-aggregation adapter for this same kernel, which additionally has to
compute `A_hat = D^-1/2(A+I)D^-1/2` before feeding `csr_spmm` (see
`../../gnn-aggregation/ge-spmm/adapter.py`'s docstring). This adapter's
`build.sh` therefore does NOT compile anything: it symlinks `source/` to the
gnn-aggregation track's clone (same commit) and imports the compiled module
directly from `../../gnn-aggregation/ge-spmm/build/spmm.so` (no local
`build/` here at all).

## N-sweep (DIM_KEY note, task brief)

`spmm_cuda`'s own dispatch (spmm_kernel.cu:425-454) is a fully general
3-tier ladder on `k = dense.size(1)` (`k<32` -> `spmm_test0`, `k<64` ->
`spmm_test1`, else -> `spmm_test2`), each tier computing ceiling-divided
tiles of the K dimension with in-kernel bounds checks -- there is no N
value the dispatch itself refuses. Checked empirically (see STATUS.md)
against every N this track sweeps ({32,128,256,512} for
`spmm-gpu-kernel-f32`) plus {32,128} for `spmm-tensorcore-fp16`'s N=32 edge
(not gated here, see below): all pass the correctness gate at ordinary fp32
accumulation error. `prepare()` still asserts N>0 defensively but otherwise
does not reject any N in this domain's registry -- unlike RoDe (N in
{32,128} only) or TC-GNN (`../../spmm/tc-gnn/`, F<=128 only), GE-SpMM's own
paper claim ("general-purpose... for GNNs", arbitrary hidden width) is
reflected in its dispatch code, not just its abstract.

## Compute precision

`spmm_test0/1/2` operate on `float*` throughout (`torch::kFloat32` asserted
in `spmm.cpp`); no tensor-core or reduced-precision path exists.
`PRECISIONS = ["fp32"]`, matching `spmm-gpu-kernel-f32`'s own default
precision (`DEFAULT_PRECISION["spmm"] == "fp32"` already -- no explicit
`--precision` flag needed).
"""

from __future__ import annotations

import importlib.util
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
_GNNAGG_GESPMM = os.path.join(HERE, "..", "..", "gnn-aggregation", "ge-spmm")
_BUILD_DIR = os.path.join(_GNNAGG_GESPMM, "build")   # reused, not local

KERNEL = "spmm"
IMPL_NAME = "gespmm-spmm-f32"
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
        so_path = os.path.join(_BUILD_DIR, "spmm.so")
        if not os.path.exists(so_path):
            raise FileNotFoundError(so_path)
        # loaded via file path with the module's ACTUAL compiled-in name
        # ("spmm", from spmm.cpp's PYBIND11_MODULE(spmm, m)) -- same reason
        # ../../gnn-aggregation/ge-spmm/adapter.py's _load() avoids a plain
        # `sys.path.insert + import spmm` (would register the generic name
        # "spmm" process-wide, a collision risk with a future artifact).
        spec = importlib.util.spec_from_file_location("spmm", so_path)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        _MOD = mod
        return _MOD
    except Exception as e:
        _IMPORT_ERROR = f"{type(e).__name__}: {e}"
        raise


def available() -> tuple[bool, str]:
    if not os.path.exists(os.path.join(_BUILD_DIR, "spmm.so")):
        return False, (f"reused build/ not found at {_BUILD_DIR} -- run "
                        f"../../gnn-aggregation/ge-spmm/build.sh (this "
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
    # as a separate copy (not a shared code object) per the
    # reference-independence rule in DOMAIN_GUIDE.md.
    rng = np.random.default_rng(seed)
    return rng.uniform(-1.0, 1.0, size=(rows, dim)).astype(dtype)


class GESpMM:
    name = IMPL_NAME
    platform = "cuda"

    def __init__(self, precision: str = "fp32"):
        if precision != "fp32":
            raise NotImplementedError(
                f"{IMPL_NAME}: spmm_kernel.cu's spmm_test0/1/2 are "
                f"hardcoded float (fp32) buffers (torch::kFloat32 asserted "
                f"in spmm.cpp); requested precision {precision!r} is not "
                f"serviced")
        self.precision = precision

    def prepare(self, matrix, params: dict):
        import torch
        mod = _load()

        A = matrix.csr
        M, K = A.shape
        N = int(params["N"])
        if N <= 0:
            raise NotImplementedError(f"{IMPL_NAME}: N={N} <= 0, not serviced")
        seed = params.get("seed", 42)

        rowptr = torch.tensor(np.ascontiguousarray(A.indptr, dtype=np.int32), device="cuda")
        colind = torch.tensor(np.ascontiguousarray(A.indices, dtype=np.int32), device="cuda")
        values = torch.tensor(np.ascontiguousarray(A.data, dtype=np.float32), device="cuda")

        B_np = _dense_operand(K, N, seed, np.float32)
        dense = torch.as_tensor(B_np, device="cuda").contiguous()

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
    return GESpMM(precision)
