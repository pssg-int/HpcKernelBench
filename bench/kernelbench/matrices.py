"""
Matrix acquisition and loading.

Real inputs come from the SuiteSparse Matrix Collection; a matrix is fetched
once into the cache directory and reused. Synthetic matrices exist only for the
login-node smoke test and are labelled as such in the result record so they can
never be mistaken for a spec-conforming run.
"""

from __future__ import annotations

import hashlib
import os
import tarfile
import time
import urllib.request
from dataclasses import dataclass

import numpy as np
import scipy.sparse as sp

CACHE = os.environ.get(
    "KERNELBENCH_MATRIX_CACHE",
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "matrices"),
)
CACHE = os.path.normpath(CACHE)
SS_INDEX_URL = "https://sparse.tamu.edu/files/ssstats.csv"
SS_MM_URL = "https://suitesparse-collection-website.herokuapp.com/MM/{group}/{name}.tar.gz"


@dataclass
class Matrix:
    name: str
    csr: sp.csr_matrix
    source: str            # 'suitesparse' | 'synthetic'
    group: str = ""
    load_seconds: float = 0.0

    @property
    def shape(self):
        return self.csr.shape

    @property
    def nnz(self) -> int:
        return int(self.csr.nnz)

    def describe(self) -> dict:
        m, n = self.shape
        return {
            "name": self.name,
            "source": self.source,
            "group": self.group,
            "rows": int(m),
            "cols": int(n),
            "nnz": self.nnz,
            "density": self.nnz / (m * n) if m and n else 0.0,
            "avg_nnz_per_row": self.nnz / m if m else 0.0,
            "max_nnz_per_row": int(np.diff(self.csr.indptr).max()) if m else 0,
            "value_checksum": hashlib.sha256(
                np.ascontiguousarray(self.csr.data).tobytes()).hexdigest()[:16],
            "load_seconds": round(self.load_seconds, 4),
        }


# ---------------------------------------------------------------- suitesparse
def _resolve_group(name: str) -> str | None:
    """Look up a matrix's group in the cached SuiteSparse index."""
    os.makedirs(CACHE, exist_ok=True)
    idx = os.path.join(CACHE, "ssstats.csv")
    if not os.path.exists(idx):
        urllib.request.urlretrieve(SS_INDEX_URL, idx)
    with open(idx) as f:
        lines = f.read().splitlines()
    # ssstats.csv: first two lines are counts/date, then "group,name,rows,cols,nnz,..."
    for line in lines[2:]:
        parts = line.split(",")
        if len(parts) > 2 and parts[1] == name:
            return parts[0]
    return None


def fetch_suitesparse(name: str, group: str | None = None) -> str:
    """Download+extract a matrix, return the path to its .mtx. Cached."""
    os.makedirs(CACHE, exist_ok=True)
    mtx = os.path.join(CACHE, f"{name}.mtx")
    if os.path.exists(mtx):
        return mtx
    group = group or _resolve_group(name)
    if group is None:
        raise LookupError(f"{name!r} not found in the SuiteSparse index")
    tgz = os.path.join(CACHE, f"{name}.tar.gz")
    if not os.path.exists(tgz):
        urllib.request.urlretrieve(SS_MM_URL.format(group=group, name=name), tgz)
    with tarfile.open(tgz) as tf:
        member = next(m for m in tf.getmembers() if m.name.endswith(f"{name}.mtx"))
        member.name = os.path.basename(member.name)
        tf.extract(member, CACHE)
    return mtx


def load_matrix(name: str, group: str | None = None) -> Matrix:
    t0 = time.perf_counter()
    path = fetch_suitesparse(name, group)
    from scipy.io import mmread
    coo = mmread(path)
    if np.iscomplexobj(coo):
        # casting complex to float64 silently drops imaginary parts, and the
        # reference would inherit the same corruption, so the gate could never
        # catch it. The sparse specs restrict inputs to real matrices anyway.
        raise ValueError(
            f"{name} is complex-valued; the sparse specs require real matrices "
            "(pick a different matrix or extend the loader deliberately)")
    csr = sp.csr_matrix(coo, dtype=np.float64)
    csr.sort_indices()
    return Matrix(name=name, csr=csr, source="suitesparse", group=group or "",
                  load_seconds=time.perf_counter() - t0)


# ---------------------------------------------------------------- synthetic
def synthetic(name: str, rows: int, cols: int, nnz_per_row: int,
              pattern: str = "banded", seed: int = 20260806) -> Matrix:
    """
    Smoke-test matrices only. `pattern` controls row-length regularity, which is
    the property SpMV/SpMM kernels are most sensitive to:
      uniform  — every row has exactly nnz_per_row entries (best case)
      banded   — entries clustered near the diagonal (cache-friendly)
      powerlaw — row lengths follow a power law (load-imbalance stress)
    """
    t0 = time.perf_counter()
    rng = np.random.default_rng(seed)
    if pattern == "powerlaw":
        lengths = np.clip(
            (rng.pareto(1.5, rows) + 1) * nnz_per_row, 1, min(cols, 50 * nnz_per_row)
        ).astype(np.int64)
    else:
        lengths = np.full(rows, min(nnz_per_row, cols), dtype=np.int64)
    indptr = np.zeros(rows + 1, dtype=np.int64)
    np.cumsum(lengths, out=indptr[1:])
    total = int(indptr[-1])
    indices = np.empty(total, dtype=np.int64)
    for i in range(rows):
        lo, hi = indptr[i], indptr[i + 1]
        k = hi - lo
        if pattern == "banded":
            start = max(0, min(cols - k, i - k // 2))
            indices[lo:hi] = np.arange(start, start + k)
        else:
            indices[lo:hi] = np.sort(rng.choice(cols, size=k, replace=False))
    data = rng.uniform(-1.0, 1.0, size=total)
    csr = sp.csr_matrix((data, indices, indptr), shape=(rows, cols))
    csr.sort_indices()
    return Matrix(name=name, csr=csr, source="synthetic",
                  group=pattern, load_seconds=time.perf_counter() - t0)
