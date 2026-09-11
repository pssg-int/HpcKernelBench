"""
Check #3 (GATE SEMANTICS, cholesky's residual-smuggled-through-to_host
trick): verify the residual gate is numerically sound --
  (a) a CORRECT factorization passes well under the spec's <30 threshold.
  (b) a WRONG "factorization" (corrupted L, or a structurally-different-but-
      plausible-looking matrix) is REJECTED (residual >> 30), i.e. the gate
      cannot be fooled by a bad implementation returning garbage that merely
      has the right shape/dtype.

Drives dense.py's actual ScipyCholesky/reference_cholesky pair + harness.py's
real check_correctness function -- not a reimplementation -- so this exercises
the exact code path a real run would use.
"""
import sys
sys.path.insert(0, "/pscratch/sd/c/cunyang/msu/hpc-bench/kernel-papers/bench")

import numpy as np
from kernelbench.domains import dense
from kernelbench import harness

w = dense.DenseShape(name="probe-cholesky-n512", kernel="cholesky",
                      variant="cholesky-dense-single-node-fp64-kernel",
                      shape_class="square", M=512, K=512, n=512,
                      precision="fp64", seed=42)
params = {"seed": 42}

ref, scale = dense.reference_cholesky(w, params)
tol = 30.0  # spec's literal threshold, cholesky-dense-single-node-fp64-kernel

def run_gate(label, L_bad_fn=None):
    impl = dense.ScipyCholesky(precision="fp64")
    h = impl.prepare(w, params)
    L = impl.run(h)
    if L_bad_fn is not None:
        L = L_bad_fn(L.copy())
    got = impl.to_host(L)  # computes ||L L^T - A||_F using the SAME self._A
    corr = harness.check_correctness(got, ref, tol, "cholesky-probe",
                                      mode="max_scaled_err", scale=scale)
    print(f"{label:42s} residual={corr.value:12.4f}  tol={tol}  passed={corr.passed}")
    return corr

print("=== cholesky residual gate soundness (dense.py + harness.py, real code path) ===")
c_ok = run_gate("correct factorization (scipy potrf)")
assert c_ok.passed, "correct factorization should pass!"

# (a) a plausible-looking WRONG L: a different valid Cholesky factor for a
#     DIFFERENT SPD matrix of the same shape (e.g. cholesky of A+shift) --
#     the kind of bug where an implementation silently used stale/wrong input
c_wrong_diff_input = run_gate(
    "L of a DIFFERENT SPD matrix (shifted A)",
    L_bad_fn=lambda L: np.linalg.cholesky(
        (w.M and dense._spd_matrix(w.n, w.seed + 999, np.float64)))
)
assert not c_wrong_diff_input.passed, "wrong-input factor should NOT pass the gate!"

# (b) corrupted L: flip the sign of one off-diagonal entry (small, localized
#     corruption -- the kind of single-element indexing bug a real kernel bug
#     might produce)
def corrupt_one_entry(L):
    L = L.copy()
    L[300, 100] += 1.0  # perturb one entry outside the zero upper triangle... wait use lower
    return L

def corrupt_lower_entry(L):
    L = L.copy()
    L[100, 30] *= -1.0  # flip sign of one lower-triangular entry
    return L

c_corrupt = run_gate("single-entry-corrupted L (sign flip)", L_bad_fn=corrupt_lower_entry)
assert not c_corrupt.passed, "corrupted factor should NOT pass the gate!"

# (c) all-zeros L (degenerate/garbage output of the right shape/dtype)
c_zero = run_gate("all-zero L (garbage output, right shape)",
                   L_bad_fn=lambda L: np.zeros_like(L))
assert not c_zero.passed, "zero output should NOT pass the gate!"

# (d) sensitivity check: how much perturbation DOES still pass? tiny
#     random fp64-noise-level perturbation (simulating legitimate rounding
#     differences between two valid factorization algorithms/orderings)
rng = np.random.default_rng(0)
def tiny_noise(L):
    noise = rng.normal(scale=1e-13 * np.abs(L).max(), size=L.shape)
    return L + noise

c_tiny = run_gate("tiny (1e-13-scale) noise on L -- simulates a different"
                   " valid algorithm's rounding", L_bad_fn=tiny_noise)

print("\nAll assertions passed: correct factor passes; wrong-input, corrupted,"
      " and all-zero factors are all correctly rejected.")
