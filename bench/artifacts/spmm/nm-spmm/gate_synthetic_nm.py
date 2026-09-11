#!/usr/bin/env python3
"""
Standalone substitute gate for NM-SpMM (precedent:
../../spmv/diaq/STATUS.md's substitute gate).

`spmm-gpu-kernel-f32`'s mandated `recommended_subset` (general SuiteSparse
matrices) cannot be fed to this artifact: NM-SpMM's kernel requires the
sparse operand to already satisfy a specific N:M block-vector structure
(see adapter.py's module docstring) that no general SuiteSparse matrix has.
This script instead builds a SYNTHETIC matrix that DOES satisfy the exact
constraint NM-SpMM's own kernel_32x32_4x4/sparsity=0.5 path requires (16 of
every 32 contiguous columns nonzero, pattern shared across every 32-row
group), then runs the SAME gate code path the runner CLI would
(`kernelbench.harness.run_variant`) against `spmm-gpu-kernel-f32`'s own
correctness bound and `kernelbench.impls.cpu_ref.reference_spmm` -- the
harness's independent fp64 CPU reference, never a loosened tolerance.

Usage: $PY gate_synthetic_nm.py
"""

from __future__ import annotations

import os
import sys

_BENCH = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                       "..", "..", ".."))
sys.path.insert(0, _BENCH)
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)

import numpy as np
import scipy.sparse as sp

from kernelbench import domains, env, harness, matrices, spec
from kernelbench.impls import cpu_ref

import adapter as nm_adapter

domains.load("spmm")   # registers the spmm cost rule (workload.register_cost)

VEC_LEN = nm_adapter.VEC_LEN
PRUNING_M = nm_adapter.PRUNING_M
PRUNING_N = nm_adapter.PRUNING_N


def build_nm_matrix(Mh: int, Kh: int, seed: int = 20260906) -> sp.csr_matrix:
    """
    A synthetic Mh x Kh matrix satisfying NM-SpMM's OWN structural contract
    at sparsity=0.5 (pruning_M=32, pruning_N=16): for every group of
    VEC_LEN=32 consecutive rows and every pruning_M=32-wide column block,
    exactly 16 columns are kept, with the SAME 16 positions shared by all
    32 rows in the group (values differ per row) -- see adapter.py's module
    docstring for why this is the exact property
    verify_and_extract_nm_pattern() checks for, mirroring
    source/include/utils.h::init_data()'s own random-N:M-pattern generation
    (independently re-derived here in Python -- construction only, no
    kernel/preprocessing code duplicated; the artifact's OWN preprocessing.cu
    function is still what adapter.py calls in prepare()).
    """
    assert Mh % VEC_LEN == 0 and Kh % PRUNING_M == 0
    rng = np.random.default_rng(seed)
    n_groups = Mh // VEC_LEN
    n_blocks = Kh // PRUNING_M

    rows, cols, vals = [], [], []
    for g in range(n_groups):
        for kb in range(n_blocks):
            chosen = np.sort(rng.choice(PRUNING_M, size=PRUNING_N, replace=False))
            global_cols = kb * PRUNING_M + chosen
            for r in range(VEC_LEN):
                row = g * VEC_LEN + r
                row_vals = rng.uniform(-1.0, 1.0, size=PRUNING_N)
                rows.extend([row] * PRUNING_N)
                cols.extend(global_cols.tolist())
                vals.extend(row_vals.tolist())
    csr = sp.csr_matrix((vals, (rows, cols)), shape=(Mh, Kh), dtype=np.float64)
    csr.sort_indices()
    return csr


def main() -> int:
    print(f"[env] toolchain: {env.pin_cuda_toolchain()}")

    ok, reason = nm_adapter.available()
    if not ok:
        print(f"NM-SpMM adapter not available: {reason}")
        return 1

    sp_spec = spec.load("spmm")
    variant = sp_spec.variant("spmm-gpu-kernel-f32")
    print(f"variant: {variant.id}  tolerance: {variant.tolerance} "
          f"({variant.tolerance_provenance})")

    Mh, Kh = 128, 128   # 4 row-groups x 4 column-blocks; small, fast gate
    csr = build_nm_matrix(Mh, Kh)
    mat = matrices.Matrix(name="synthetic-nm-16of32-block32", csr=csr,
                          source="synthetic", group="nm-spmm-gate")

    # sanity: the matrix we built actually satisfies the constraint (a
    # failure here would mean this gate script's own construction is wrong,
    # not the artifact)
    nm_adapter.verify_and_extract_nm_pattern(csr)
    print(f"matrix: {Mh}x{Kh}, nnz={csr.nnz} "
          f"(density={csr.nnz/(Mh*Kh):.3f}, expect 0.500)")

    all_ok = True
    for Nh in (32, 128, 256, 512):
        impl = nm_adapter.create("fp32")
        params = {"seed": 42, "precision": "fp32", "N": Nh}
        r = harness.run_variant(
            impl, mat, variant, params,
            reference=cpu_ref.reference_spmm,
            correctness_mode="max_scaled_err",
            reference_name="scipy fp64 CSR",
            warmup_override=1, reps_override=2)
        c = r.correctness
        status = "PASS" if r.valid else "FAIL"
        print(f"  N={Nh:4d}  valid={r.valid}  {status}  "
              f"max_scaled_err={c.value:.3e}  tol={c.tolerance}  "
              f"gflops={r.metrics.get('gflops', float('nan')):.2f}"
              if r.valid else
              f"  N={Nh:4d}  valid={r.valid}  {status}  "
              f"max_scaled_err={c.value:.3e}  tol={c.tolerance}")
        all_ok = all_ok and r.valid

    print()
    print("ALL PASS" if all_ok else "AT LEAST ONE FAILURE")
    return 0 if all_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
