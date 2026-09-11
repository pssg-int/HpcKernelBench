"""
Flop and byte accounting, one place, so every implementation is scored the same
way. The counts follow each spec's `metric` field literally.
"""

from __future__ import annotations

ITEMSIZE = {"fp64": 8, "fp32": 4, "fp16": 2, "tf32": 4, "int8": 1, "int4": 0.5}


def flops(kernel: str, *, nnz: int, dim: int = 1, unfolded_nnz: int | None = None) -> int:
    """
    spmv   : 2*nnz                (beta=0, no y accumulate)
    spmm   : 2*nnz*N
    sddmm  : 2*nnz*K
    A symmetric-storage SpMV must pass unfolded_nnz = 2*nnz_stored - diag_nnz,
    which the spec mandates as an anti-gaming rule.
    """
    n = unfolded_nnz if unfolded_nnz is not None else nnz
    if kernel == "spmv":
        return 2 * n
    if kernel in ("spmm", "sddmm"):
        return 2 * n * dim
    raise ValueError(f"no flop rule for {kernel!r}")


def bytes_moved(kernel: str, *, nnz: int, rows: int, cols: int, dim: int = 1,
                value: str = "fp64", index: str = "int32") -> int:
    """
    Compulsory-traffic model (each datum counted once). Used for the GB/s
    secondary metric and for a roofline sanity bound; it is a lower bound on
    real traffic, and labelled as such in the result record.
    """
    vb = ITEMSIZE[value]
    ib = 4 if index == "int32" else 8
    if kernel == "spmv":
        return int(nnz * (vb + ib) + (rows + 1) * ib + cols * vb + rows * vb)
    if kernel == "spmm":
        # A (nnz vals+idx, rowptr) + B (cols x dim) + C (rows x dim)
        return int(nnz * (vb + ib) + (rows + 1) * ib + cols * dim * vb + rows * dim * vb)
    if kernel == "sddmm":
        # S pattern + A (rows x dim) + B (cols x dim) + P values out
        return int(nnz * ib + (rows + 1) * ib + rows * dim * vb + cols * dim * vb + nnz * vb)
    raise ValueError(f"no byte rule for {kernel!r}")


def throughput(flop_count: int, seconds: float) -> float:
    """GFLOP/s."""
    return flop_count / seconds / 1e9 if seconds > 0 else float("nan")


def bandwidth(byte_count: int, seconds: float) -> float:
    """GB/s."""
    return byte_count / seconds / 1e9 if seconds > 0 else float("nan")
