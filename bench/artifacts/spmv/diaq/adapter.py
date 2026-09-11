"""
Adapter for DiaQ (ICS 2026, "Diagonal-Budgeted Trotterization for Efficient
Quantum Hamiltonian Simulation", conf/ics/ChunduryBLSM26).

The paper's given artifact URL is diaq_for_hamsim, but that repo is a
benchmark/application suite that only *uses* DiaQ as an external dependency
(its README literally says "Install DiaQ first" pointing at
github.com/srikarchundury/diaq) -- it contains no kernel of its own. The
actual SpMV kernel (src/spMV.cpp, exposed to Python via pybind11 as
`diaq.spMV`) lives in that sibling repo, cloned here as ./source; see
build.sh's docstring and STATUS.md for the provenance chain.

DiaQ's format ("diagonal-major": only active diagonals stored, packed
contiguously, SoA real/imag layout) is a genuinely general one -- `dq.spMV`
runs correctly on ANY matrix built via `dq.from_numpy`, not just physically
valid quantum Hamiltonians (see source/python_tests/test_spmv.py, which
feeds it a plain `np.random.rand(n, n)` dense matrix). So it qualifies as a
"reusable general SpMV routine" per the integration brief.

BUT: the only Python-exposed constructor, `dq.from_numpy`, requires the
FULLY MATERIALIZED DENSE M x N array as input (see source/src/conversions.cpp
`from_numpy`, which does `fi = mat_pos[0]*shape[1] + mat_pos[1]` against a
flat `vals` buffer of length M*N) -- there is no sparse/CSR/COO constructor
in the public API. For general SuiteSparse test matrices this is infeasible:
'cant' (62451x62451) would need a ~62 GB complex128 buffer just as adapter
input, before DiaQ even starts scanning diagonals. This is not an oversight
in our wrapping -- it mirrors exactly how the paper's own microbenchmark
(source_paper_repo_hamsim/benchmark_apps/microbench_spmv.py) uses DiaQ: it
always builds a dense `H = generate_sparse_band_matrix(...)` array first,
keeping qubit counts modest specifically to keep that dense buffer small.

`prepare()` therefore refuses (raises, with a clear message) any matrix
whose dense element count exceeds MAX_DENSE_ELEMENTS, rather than letting a
huge allocation hang/OOM the (shared) login node. See STATUS.md for what
gate check was actually run in place of the mandated `--matrices cant` command.
"""

from __future__ import annotations

import os
import sys

import numpy as np

KERNEL = "spmv"
IMPL_NAME = "diaq-spmv"
PAPER_KEY = "conf/ics/ChunduryBLSM26"
PRECISIONS = ["fp64"]  # DiaQ's valType is `double` (src/config.hpp); Python API is complex128-only

_HERE = os.path.dirname(os.path.abspath(__file__))
_SO_DIR = os.path.join(_HERE, "source", "cpu_build", "lib")

# Safety cap on the dense M*N element count `from_numpy` must materialize.
# 5e7 complex128 elements = 800 MB (plus transient per-diagonal buffers
# inside from_numpy, worst case a few x that) -- generous for the harness's
# smoke matrices (4000x4000 = 1.6e7) but far below any real SuiteSparse
# matrix in the spec's recommended_subset (smallest is O(1e5) rows/cols,
# i.e. >= 1e10 dense elements).
MAX_DENSE_ELEMENTS = 5 * 10**7


def available() -> tuple[bool, str]:
    try:
        if not os.path.isdir(_SO_DIR):
            return False, f"not built: {_SO_DIR} missing (run build.sh)"
        so = [f for f in os.listdir(_SO_DIR) if f.startswith("diaq.cpython") and f.endswith(".so")]
        if not so:
            return False, f"not built: no diaq.cpython-*.so under {_SO_DIR}"
        return True, ""
    except Exception as e:  # available() must never raise
        return False, f"{type(e).__name__}: {e}"


def _import_diaq():
    if _SO_DIR not in sys.path:
        sys.path.insert(0, _SO_DIR)
    import diaq as dq  # noqa: F401 (imported for its side effect + returned below)
    return dq


class DiaqSpMV:
    name = "diaq-spmv"
    platform = "cpu"

    def __init__(self, precision: str = "fp64"):
        if precision != "fp64":
            raise NotImplementedError(
                f"DiaQ's Python bindings are complex128 (double) only; requested {precision}")
        self.precision = precision
        self.dq = _import_diaq()

    def prepare(self, matrix, params: dict):
        from kernelbench.harness import Timer  # noqa: F401 (re-exported via timer())
        A = matrix.csr
        rows, cols = A.shape
        n_elem = rows * cols
        if n_elem > MAX_DENSE_ELEMENTS:
            raise RuntimeError(
                f"diaq-spmv: matrix {matrix.name} is {rows}x{cols} ({n_elem:.3e} dense "
                f"elements); DiaQ's from_numpy requires a fully materialized dense "
                f"array and this exceeds the MAX_DENSE_ELEMENTS={MAX_DENSE_ELEMENTS:.0e} "
                "safety cap (see adapter.py docstring) -- refusing rather than risking "
                "an OOM/hang on a shared machine.")
        # This IS the artifact's own (only) format-construction path -- timed
        # once as preprocessing, per the contract.
        dense = np.asarray(A.todense(), dtype=np.complex128)
        A_fmt = self.dq.from_numpy(dense)

        rng = np.random.default_rng(params.get("seed", 42))
        x = rng.uniform(-1.0, 1.0, size=cols).astype(np.float64)
        x_fmt = self.dq.from_numpy_vector(x.astype(np.complex128))
        return {"A_fmt": A_fmt, "x_fmt": x_fmt}

    def run(self, h):
        return self.dq.spMV(h["A_fmt"], h["x_fmt"])

    def to_host(self, out) -> np.ndarray:
        y = np.array(self.dq.to_numpy_vector(out))
        # input x had zero imaginary part and A is the exact embedding of a
        # real matrix into DiaQ's complex storage, so y's imaginary part is
        # 0 up to floating-point noise; the harness's gate wants a real array.
        return np.ascontiguousarray(y.real, dtype=np.float64)

    def timer(self):
        from kernelbench.harness import Timer
        return Timer()

    def free(self, h):
        h.clear()


def create(precision: str):
    return DiaqSpMV(precision)
