#!/usr/bin/env python
"""
Standalone pruned-weight-matched correctness check for tetris-sparse-conv.

Why this exists (see STATUS.md's "Correctness" section): the harness's
built-in `conv-dense-kernel-fp32` reference (kernelbench.domains.ml.
reference_conv) draws its own FRESH, unpruned W from the same seed -- it has
no way to receive this adapter's pruned W, so the formal harness gate is
expected to (and does) report a mismatch for tetris-sparse-conv, and that
mismatch is not a kernel bug. This script runs the methodologically correct
check instead: a dense fp64 reference (kernelbench.domains.ml.
_direct_conv_fp64) computed on the EXACT pruned W the kernel was fed (zeros
and all), compared with the same max_scaled_err convention
harness.check_correctness uses.

Usage:
    $PY check_pruned_correctness.py   # PY from bench/env.sh
"""
import os
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_BENCH = os.path.normpath(os.path.join(_HERE, "..", "..", ".."))
sys.path.insert(0, _BENCH)
sys.path.insert(0, _HERE)

import adapter  # noqa: E402
from kernelbench.domains import ml  # noqa: E402

TOLERANCE = 1e-3  # benchspecs/convolution/spec.yaml's conv-sparse-pruned-kernel-fp32 bound


def main() -> int:
    ok, reason = adapter.available()
    print("available:", ok, reason)
    if not ok:
        return 1

    impl = adapter.create("fp32")
    w = ml.load_workload("resnet18_1")
    print("workload:", w.describe())

    h = impl.prepare(w, {"N": 1})
    sparsity = float((h["W_pruned"] == 0).mean())
    print(f"prepared. W_pruned sparsity: {sparsity:.4f}")

    out = impl.run(h)
    got = impl.to_host(out)
    print("output shape:", got.shape, "dtype:", got.dtype)

    # dense fp64 reference on the SAME pruned W the kernel actually multiplied
    X64 = h["X"].astype(np.float64)
    W64 = h["W_pruned"].astype(np.float64)
    ref = ml._direct_conv_fp64(X64, W64, w)
    scale = ml._direct_conv_fp64(np.abs(X64), np.abs(W64), w)

    diff = np.abs(got - ref)
    max_scaled_err = float((diff / np.maximum(scale, 1e-300)).max())
    print(f"max_scaled_err (pruned-weight-matched) = {max_scaled_err:.6e}")
    print(f"tolerance = {TOLERANCE:.0e}")
    passed = max_scaled_err <= TOLERANCE
    print("PASS" if passed else "FAIL")

    impl.free(h)
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
