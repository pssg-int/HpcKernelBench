"""
Domain modules. Importing one registers its kernels' cost rules, workload
loaders and implementations. `load(kernel)` imports whichever domain owns a
kernel, so the runner never needs to know the mapping by hand.
"""

from __future__ import annotations

import importlib

# kernel -> domain module name. Extend when a domain module is added.
OWNER = {
    # sparse linear algebra
    "spmv": "sparse", "spmm": "sparse", "sddmm": "sparse",
    "spgemm": "sparse", "sptrsv": "sparse", "spmspv": "sparse",
    "gnn-aggregation": "sparse",
    # dense linear algebra
    "gemm": "dense", "batched-gemm": "dense", "gemv": "dense",
    "blas-level1-2": "dense", "trsm": "dense", "cholesky": "dense",
    "lu": "dense", "qr": "dense", "svd": "dense", "eigensolver": "dense",
    # stencil / PDE
    "stencil": "stencil", "lattice-boltzmann": "stencil",
    "fdtd-seismic": "stencil",
    # spectral
    "fft": "spectral", "ntt": "spectral",
    # graph
    "bfs": "graph", "sssp": "graph", "pagerank": "graph",
    "triangle-counting": "graph", "connected-components": "graph",
    "graph-pattern-mining": "graph",
    # compression
    "lossy-compression": "compression", "lossless-compression": "compression",
    # parallel primitives
    "sort": "primitives", "scan-reduction": "primitives",
    "topk-selection": "primitives", "hash-table": "primitives",
    "set-intersection": "primitives",
    # tensor algebra
    "mttkrp": "tensor", "tensor-contraction": "tensor",
    # iterative solvers
    "cg-krylov": "solvers", "multigrid": "solvers", "preconditioner": "solvers",
    # ML kernels
    "convolution": "ml", "attention-kernel": "ml", "quantized-gemm": "ml",
    "winograd": "ml", "sparse-attention-kernel": "ml",
    # nearest-neighbor / vector search
    "ann-search": "annsearch",
    "sequence-alignment": "alignment", "string-regex-matching": "automata",
}

_loaded: set[str] = set()


def load(kernel: str):
    mod = OWNER.get(kernel)
    if mod is None:
        raise KeyError(f"no domain owns kernel {kernel!r}; known: {sorted(OWNER)}")
    m = importlib.import_module(f"{__name__}.{mod}")
    _loaded.add(mod)
    return m


def load_all() -> dict[str, str]:
    """Import every domain module that exists; report which are missing."""
    status = {}
    for mod in sorted(set(OWNER.values())):
        try:
            importlib.import_module(f"{__name__}.{mod}")
            _loaded.add(mod)
            status[mod] = "ok"
        except ModuleNotFoundError:
            status[mod] = "not implemented"
        except Exception as e:  # a broken domain must not hide the others
            status[mod] = f"error: {type(e).__name__}: {e}"
    return status


def kernels(implemented_only: bool = False) -> list[str]:
    if not implemented_only:
        return sorted(OWNER)
    st = load_all()
    return sorted(k for k, m in OWNER.items() if st.get(m) == "ok")
