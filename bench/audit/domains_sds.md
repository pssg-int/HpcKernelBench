# Adversarial audit: domain modules vs. specs

Scope: `kernelbench/domains/{sparse,dense,stencil,spectral}.py` against
`../benchspecs/<kernel>/spec.yaml`. Read fully, cross-checked against
`harness.py`, `spec.py`, `workload.py`, `metrics.py`, `runner.py`, `report.py`,
`impls/cpu_ref.py`, `impls/gpu_cuda.py`. Empirical checks run with
`/pscratch/sd/c/cunyang/gnn/plexus_env/bin/python`, CPU only, no CUDA
execution. Scripts used are kept at
`/tmp/claude-106793/-pscratch-sd-c-cunyang-msu-hpc-bench/bf2b92db-5313-4854-9693-d7aa8869eb58/scratchpad/check{1..9}_*.py`
for reproducibility; their output is quoted inline below.

**Tally: 7 BUG, 3 FAIRNESS, 4 NIT.** Two additional items were specifically
checked and found *not* to be defects (Cholesky's residual gate; the
scipy-convolve/naive-loop reference-independent implementations) — reported
under "Verified sound" so the audit's negative results are on the record too.

---

## 1. REFERENCE INDEPENDENCE — the systemic finding (BUG, all four modules)

**Claim under test:** is the correctness reference computed by a genuinely
different code path than the implementation under test?

**Result: no, not at each domain's own default/fp64 precision.** Every
domain's *primary* CPU implementation and its "independent" fp64 reference
turn out to call the *same* underlying library routine, so the gate proves
only that a library is deterministic with itself — not that the
implementation is correct. Verified empirically (checks #2, #8, #9), using
the real `prepare()/run()/to_host()` code paths, not reimplementations:

| module | impl (default precision) | vs. reference | result |
|---|---|---|---|
| sparse.py | `scipy-csr-spmv` (fp64) | `reference_spmv` (`cpu_ref.py:149`) | **bit-identical**, `max\|out-ref\|=0.0` |
| sparse.py | `scipy-csr-spmm` (forced fp64) | `reference_spmm` (`cpu_ref.py:158`) | **bit-identical** |
| sparse.py | `naive-csr-spmv` (hand Python loop, `cpu_ref.py:51-63`) | same reference | differs at `8.9e-16` (fp64 ulp) — genuinely independent |
| dense.py | `numpy-gemm` (fp64) | `reference_gemm` (`dense.py:486`) | **bit-identical** |
| dense.py | `numpy-gemv` (fp64) | `reference_gemv` (`dense.py:506`) | **bit-identical** |
| dense.py | `numpy-blas12`/dot (fp64) | `reference_blas12` (`dense.py:549`) | **bit-identical** |
| stencil.py | `numpy-stencil` (fp64) | `reference_stencil` (`stencil.py:346`) | **bit-identical** — both call the same module-level `_sweep()` (`stencil.py:317`) |
| stencil.py | `scipy-convolve-stencil` (`scipy.ndimage.convolve`) | same reference | differs at `4.2e-17` — genuinely independent |
| spectral.py | `numpy-fft` (fp64) | `reference_fft` (`spectral.py:277`) | **bit-identical** |
| spectral.py | `scipy-fft` (fp64, nominally "a different library") | same reference | **also bit-identical** |
| spectral.py | `numpy-fft` vs `scipy-fft` directly | — | **also bit-identical to each other** |

The spectral.py result is the sharpest case: **both** of the domain's shipped
CPU implementations (`numpy-fft`, `scipy-fft`) are bit-for-bit identical to
the reference *and to each other* at fp64 — `numpy.fft` and `scipy.fft` both
wrap the same vendored pocketfft core on this build, so spectral.py currently
ships zero implementations that are numerically independent of its own
reference for the `fft` kernel. A bug in the shared pocketfft core, or a
shared mistake in how both wrapper functions are called (wrong axis, wrong
sign convention, etc.), would be invisible to this gate.

Root cause: every domain module regenerates operands with the same seeded RNG
"generate at target precision, widen to fp64" discipline the code comments
advertise (this part is done correctly — verified directly by the
bit-identical results *only* appearing at matched precision, and by
`fp32-impl vs fp64-reference` gaps in check #2 being nonzero at the expected
~1e-7/1e-4 magnitude), but the reference function is written as "call the
same vendor routine, just at fp64" rather than as an independently-derived
computation (a different algorithm, a brute-force loop, a different library
with a genuinely different implementation). At default precision (fp64 for
spmv/gemm/gemv/blas12/stencil; fp32 for spmm/sddmm/fft, where the check is at
least a real rounding check, see table), the gate collapses to "does scipy
agree with scipy" / "does numpy agree with numpy".

**Not broken everywhere**: `naive-csr-spmv` (sparse.py), `scipy-convolve-stencil`
(stencil.py), and Cholesky's whole gate design (dense.py, see "Verified
sound" below) are all genuine, working counterexamples in this same codebase
— so the fix pattern already exists locally, it's just not applied to the
other CPU baseline implementations.

- **File refs:** `kernelbench/domains/sparse.py:84-102`,
  `kernelbench/impls/cpu_ref.py:23-93,149-163`;
  `kernelbench/domains/dense.py:486-568,591-745`;
  `kernelbench/domains/stencil.py:317-342,346-368,377-425`;
  `kernelbench/domains/spectral.py:148-275,277-315`.
- **Classification: BUG** (4 instances, one per module — counted separately
  since each needs an independent fix in a different file, tallied as B1-B4
  below).

---

## 2. Per-finding list

### B1 (sparse.py) — reference-in-disguise for scipy-csr-spmv/spmm at fp64
See section 1. `sparse.py:84-102` (`REFERENCES`, `CPU_IMPLS`,
`DEFAULT_PRECISION`), `impls/cpu_ref.py:23-48,67-93,149-163`.
Fix direction: add a genuinely independent fp64 CPU reference (e.g. a dense
`A.toarray() @ x`/`@ B` computed via a *different* BLAS call, or route the
correctness check for the scipy-backed CPU impls through the
already-present `naive-csr-spmv`-style independent path instead of
`cpu_ref.reference_spmv/reference_spmm`).

### B2 (stencil.py) — reference-in-disguise for numpy-stencil at fp64
See section 1. `stencil.py:317-342` (`_sweep`, shared by both), `:346-368`
(`reference_stencil`), `:377-425` (`NumpyStencil`). `scipy-convolve-stencil`
already demonstrates the fix pattern (genuinely different code path,
`stencil.py:427-473`) — the reference should be built that way, not by
calling `_sweep()` a second time.

### B3 (spectral.py) — numpy-fft AND scipy-fft both identical to reference
See section 1, the sharpest instance: **neither** shipped `fft` CPU
implementation is independent of the reference or of each other at fp64.
`spectral.py:148-208` (`NumpyFFT`), `:210-275` (`ScipyFFT`), `:277-315`
(`reference_fft`). A genuinely independent fp64 reference here would need a
different transform algorithm (e.g. a direct O(N^2) DFT for small N, mirroring
`spectral.py`'s own `_schoolbook_ntt` pattern already used for NTT's
independent-method check at `spectral.py:565-591`) rather than another
pocketfft wrapper.

### B4 (dense.py) — reference-in-disguise for numpy-gemm/gemv/blas12 at fp64
See section 1 and check #9. `dense.py:486-514,549-568` (references),
`:591-657,699-745` (impls). Same fix direction as B1/B2/B3. (Cholesky is the
one dense.py kernel that already avoids this — see "Verified sound" below.)

### B5 (sparse.py) — spmv-symmetric-kernel's anti-gaming flop rule is unreachable
`benchspecs/spmv/spec.yaml:216-222` mandates, for the `spmv-symmetric-kernel`
variant, the *unfolded* flop count `2*(2*nnz_stored - diag_nnz)` — explicitly
so half-storage can't masquerade as a 2x speedup — and this exact example is
`DOMAIN_GUIDE.md`'s own flagship illustration of "cost rule must follow the
spec literally." `_cost_spmv` (`sparse.py:48-55`) does implement the hook
(`params.get("unfolded_nnz")`), but grep across `kernelbench/` confirms
**nothing ever sets it**: no CPU or CUDA implementation in `sparse.py`'s
`CPU_IMPLS`/`cuda_impls()` (`sparse.py:97-114`) constructs a half-storage/
triangular matrix, and `runner.py` never threads an `unfolded_nnz` param
through. Concretely: running
`--kernel spmv --variant spmv-symmetric-kernel --impl scipy-csr-spmv ...`
today silently computes GFLOP/s from the *full* matrix's stored nnz (the
`spmv-csr-kernel` formula), not the mandated unfolded/half-storage one, and
also skips the spec's separately-mandated race-freedom check
(`spmv/spec.yaml:224-230`) entirely. The kernel is listed as fully
implemented (`spmv` is in `KERNELS`, not `PLANNED`), so this variant gap is
silent, not disclosed.
**Classification: BUG.**

### B6 (dense.py) — gemv/blas-level1-2 primary-metric unit silently swapped
`benchspecs/gemv/spec.yaml:118-127` and
`benchspecs/blas-level1-2/spec.yaml:129-137` both state explicitly
`primary: GB/s ...; secondary: GFLOP/s`/`GOps/s` (both kernels are
memory-bandwidth-bound by the specs' own framing). But
`dense.py:479-482` registers:
```
workload.register_cost("gemv", _cost_gemv, "GFLOP/s")            # dense.py:480
workload.register_cost("blas-level1-2", _cost_blas12, "GOps/s")  # dense.py:482
```
i.e. the flop/ops count — the spec's stated *secondary* metric — is wired as
the harness's one "primary throughput" slot
(`met["throughput"]`/`met["throughput_unit"]`, `harness.py:273-284`), and
that field is exactly what `runner.py:198-203` prints as the headline number
and what `report.py`'s `leaderboard()` (`report.py:68-100`, ranks by
`metrics["gflops"]`) uses to rank implementations. The spec's real primary
metric (GB/s) is computed correctly (the byte formulas in `_cost_gemv`,
`dense.py:430-438`, and `_cost_blas12`, `dense.py:470-476`, both match their
specs' formulas literally — verified by inspection) but ends up filed under
`gbytes_per_s_lower_bound`, a field the label ("lower bound") and every
downstream consumer treat as a secondary/informational number, never used
for ranking. `_cost_gemv`'s own comment (`dense.py:433-435`) shows the author
knew ("`byts` below... is the metric that actually matters, per
gemv-dense-kernel's own primary=GB/s metric") but the registration was never
fixed to match, and this gap is not listed among the module's own disclosed
"Known, disclosed simplifications" (`dense.py:96-104`).
**Classification: BUG** (silent unit/primary-metric mismatch, exactly the
class of defect `DOMAIN_GUIDE.md` calls "must follow the spec literally").

### B7 (dense.py) — blas-l1-vector-kernel tolerance mis-parsed; demonstrated to accept an out-of-spec result
`benchspecs/blas-level1-2/spec.yaml:145-160`'s correctness text is
`|result - reference| < flteps, ... flteps = 1e-4 for fp32, 1e-6 for fp64`.
`dense.py`'s own module docstring (`dense.py:40-62`, point 2) asserts:
> "Verified directly: `spec.load("blas-level1-2").variant("blas-l1-vector-kernel").tolerance` is `None` ("unparsed")... Of blas-level1-2's four variants, only `blas-l2-gemv-symv-kernel` carries a tolerance the parser can extract."

**Both of those specific claims are false as of the current `spec.py`.**
Check #5 (`spec.load("blas-level1-2").variant("blas-l1-vector-kernel")`):
```
tolerance=0.0001  provenance='parsed-assignment'
```
`spec.py`'s `_tolerance()` has a second regex path beyond the "number right
after `<`" one the docstring describes — a named-assignment fallback
(`_ASSIGN_RE`, `spec.py:42-44`, tried at `spec.py:88-90`) that matches
`flteps = 1e-4` directly. It resolves to **1e-4 — the fp32 number**, not
fp64's 1e-6, because the fp32 clause appears first in the sentence.

Consequence, demonstrated in check #6 using the real `harness.check_correctness`
path: `dense.py:586-587` sets `DEFAULT_PRECISION["blas-level1-2"] = "fp64"`,
justified by the same docstring's point 3 ("the one precision for which the
parsed number is guaranteed to match the spec's own intent") — also false
here, since the actually-parsed number is the fp32 one. Injecting a `5e-5`
absolute error into an otherwise-correct fp64 `dot` result:
```
gate using harness's ACTUAL tolerance (0.0001): passed=True   (value=5.00e-05)
gate using spec's STATED fp64 bound (1e-6):      passed=False (value=5.00e-05)
```
A fp64 `blas-level1-2` result **50x worse than the spec's own explicit fp64
bound is silently accepted** by the harness as it stands today. (Check #7
confirms this is an isolated defect, not a codebase-wide pattern: every other
sampled variant across all four modules — `spmv-csr-kernel`,
`spmm-gpu-kernel-f32`, `sddmm-csr-kernel-f32`, `gemm-square-kernel`,
`gemv-dense-kernel`, `stencil-cpu-gpu-kernel-fp64`, `fft-1d-batched-kernel`
— parses the tolerance matching its own domain module's default precision
correctly.)
**Classification: BUG** — both a stale/incorrect code comment and a live,
demonstrated gate-leniency defect for `blas-level1-2`'s default-precision runs.

---

## 3. FAIRNESS

### F1 (sparse.py) — spmv byte formula for `x` doesn't match the spec's literal wording
`benchspecs/spmv/spec.yaml:88-94`'s GB/s formula literally reads:
`... + num_rows*sizeof(value) [x, read once per row...] + num_rows*sizeof(value) [y, write once]`
— but the operation is `y[M] = A[M,K] * x[K]` (`spmv/spec.yaml:10`), so `x`
actually has `K` (= `cols`) entries, not `M` (= `rows`). `_cost_spmv`
(`sparse.py:48-55`) uses `cols * vb` for the `x` term — which is the
*mathematically correct* byte count for `x`, but does not match the spec
text's literal (apparently typo'd) wording. The code silently does the right
thing rather than reproducing the spec's own error; worth a note back to the
spec rather than a code fix, since fidelity-to-literal-text and
fidelity-to-actual-semantics disagree here and the code chose semantics.
**Classification: FAIRNESS** (a literal-fidelity gap between code and spec
text, in the direction that doesn't hurt anyone, but is exactly the kind of
silent divergence the audit is asked to surface).

### F2 (spectral.py) — FFT C2R inverse operand construction is not bit-derived from the reference's operand
For a real-input inverse (C2R) transform, `_fft_operand()`
(`spectral.py:121-145`) builds the frequency-domain input by
forward-transforming the ALREADY-PRECISION-ROUNDED real signal through the
implementation's *own* backend (`spectral.py:140-142`); `reference_fft()`
(`spectral.py:277-315`) independently forward-transforms an *un-rounded fp64*
real signal built from the same seed. This deviates from the "round once at
target precision, then widen the SAME rounded array to fp64" discipline every
other operand-construction path in this codebase (and this module's own
forward-direction case) follows. Check #4 confirms the two real signals
diverge (`2.9e-8` at fp32) before either FFT is even run, and quantifies the
knock-on effect on the actual correctness-gate quantity (`max|out-ref|`) at
`1.4e-7` — three orders of magnitude under the fp32 gate's `1e-4` tolerance,
and confirmed to vanish entirely at fp64 (`0.0`, same seed, same precision).
So this does not currently flip any pass/fail outcome, but it is a real,
demonstrable deviation from the codebase's own stated precision-discipline
convention for one specific direction/kind combination.
**Classification: FAIRNESS** (methodologically inconsistent, not proven to
break a gate at the tolerances currently in force).

### F3 (dense.py) — module docstring's precision-tolerance rationale (points 2-3) is stale/incorrect
Beyond B7's concrete consequence, the module docstring's own stated
*reasoning* for `DEFAULT_PRECISION = "fp64"` everywhere (`dense.py:63-75`,
point 3: "every spec in this track states its tolerance as a table keyed by
precision... the parser returns exactly one number: the FIRST one following a
comparison operator, which for every variant in this module happens to be
the fp64 entry") is empirically false for `blas-l1-vector-kernel` specifically
(check #5/#7: the parsed number there is the fp32 entry, reached via a
different regex path the docstring doesn't account for). The design decision
(fp64 default) is still reasonable on its own merits, but the written
justification for it no longer matches `spec.py`'s actual behavior and should
be corrected so a future reader doesn't re-trust a false "verified directly"
claim.
**Classification: FAIRNESS** (documentation-vs-behavior drift with a real
downstream effect via B7, listed separately since the fix is "correct the
docstring," distinct from B7's "fix the tolerance resolution/import path").

---

## 4. NIT

### N1 (stencil.py) — buffer reset sits inside the timed region, but is empirically negligible at spec scale
`NumpyStencil.run()` (`stencil.py:407-415`) and `ScipyConvolveStencil.run()`
(`stencil.py:456-463`) both start with `np.copyto(bufs[0], h["u0"])` inside
the function that `harness.py` times as a whole
(`with impl.timer() as t: impl.run(handle)`, `harness.py:257-262`) — but that
reset is not one of the "T sweeps"
`benchspecs/stencil/spec.yaml:87-92`'s `timing_scope` describes as the timed
unit ("Each trial times all T sweeps as ONE elapsed interval"). Check #1
measured the reset/full-run time fraction directly (median of 5-10 reps) at
several grid sizes and timestep counts and extrapolated linearly to the
spec's own `DEFAULT_TIMESTEPS=1000` (`stencil.py:57`, reset cost is O(cells),
independent of T, so the fraction scales as ~1/T):
```
smoke-star2d1r (T=5, AS-IS)  reset/run= 1.38%  (implied @T=1000: ~0.007%)
star2d1r 256x256   T=200     reset/run= 0.03%  (implied @T=1000: ~0.005%)
star2d1r 1024x1024 T=100     reset/run= 0.02%  (implied @T=1000: ~0.002%)
```
At the module's own `--smoke` config (T=5) the contamination is ~1.4% — but
`--smoke` runs are already unconditionally flagged non-spec-conforming by
`runner.py:217-223`, so this doesn't silently corrupt a claimed-conforming
number. At spec-conforming T=1000 the contamination is two to three orders
of magnitude below what any of this suite's own `>25%-of-median` variance
warnings (`harness.py:291-294`) would even flag. **Verified empirically
negligible in practice**, though technically outside the spec's literal
timing_scope — kept as a NIT rather than a BUG.

### N2 (sparse.py) — `metrics.py`'s `flops()`/`bytes_moved()` are dead, duplicate code
`kernelbench/metrics.py:11-44` reimplements `sparse.py`'s `_cost_spmv`/
`_cost_spmm`/`_cost_sddmm` formulas verbatim (confirmed identical by
inspection) but is never called: `harness.py:30` only ever uses
`metrics.throughput`/`metrics.bandwidth` (`harness.py:275-277`); grep across
the whole `kernelbench/` tree finds no other caller of `metrics.flops` or
`metrics.bytes_moved`. Harmless today (the two copies still agree), but a
silent-drift risk if `sparse.py`'s cost rules are ever edited without
noticing the shadow copy — no test would catch the divergence since nothing
exercises the dead code.
**Classification: NIT.**

### N3 (dense.py/spec.py boundary) — stencil's `DEFAULT_TIMESTEPS` cannot be spec-derived at all
Not really `stencil.py`'s fault, but worth recording: `benchspecs/stencil/spec.yaml`'s
per-variant timestep count lives under a key (`iteration_count`,
`stencil/spec.yaml:65-71`) that `spec.py`'s generic `Protocol` dataclass
(`spec.py:103-116`) has no field for and doesn't even preserve into
`unparsed` (unlike `warmup`/`reps`, which fail into `unparsed` when
unparseable). `stencil.py:57`'s `DEFAULT_TIMESTEPS = 1000` is therefore an
unavoidable hardcoded constant, not a `spec.load(...)`-derived value — in
tension with `DOMAIN_GUIDE.md`'s non-negotiable #1 ("never hardcode
warmup/reps/tolerance in a domain module"), even though the module's own
docstring (`stencil.py:149-156`) is honest about exactly this. Fixing it
would mean extending `spec.py`'s `Protocol` to carry a generic
`iteration_count`/T field, which is out of this audit's stated scope
(domain modules, not `spec.py`) — recorded here so it isn't lost.
**Classification: NIT.**

### N4 (spectral.py) — NTT `run()`'s per-call output buffer is not hoisted
`NumpyNTT.run()` (`spectral.py:717-722`) allocates `y = np.empty_like(x)`
fresh inside the timed `run()` call every rep, rather than once in
`prepare()` alongside the twiddle tables (which correctly are hoisted, see
`spectral.py:690-715`). `benchspecs/ntt/spec.yaml`'s `timing_scope`
(`ntt/spec.yaml:78-83`) doesn't explicitly forbid this (it's about
twiddle-table generation, not output-buffer allocation), and for a "NTT/s"
metric at real problem sizes (`N=2**16, L up to 48`) an
`np.empty_like` allocation is negligible next to the O(N log N) work — but
it is a small, easily-fixed inconsistency with the "everything hoistable
must be hoisted" principle `DOMAIN_GUIDE.md` states plainly.
**Classification: NIT.**

---

## 5. Verified sound (checked, no defect found)

### V1 (dense.py) — Cholesky's residual-smuggled-through-`to_host` gate
This is the one place in the four audited modules that deliberately avoids
the reference-in-disguise trap (see section 1): `reference_cholesky`
(`dense.py:517-546`) returns `(0, ||A||_F*N*eps)` instead of a factor,
and `ScipyCholesky.to_host` (`dense.py:683-690`) computes the *actual*
backward residual `||L_computed @ L_computed.T - A||_F` from whatever `L`
`run()` really produced. Check #3 drives the real `dense.py` + `harness.py`
code path (not a reimplementation) and confirms it is numerically sound
against the exact failure mode the audit asked about — "verify it cannot
pass a WRONG factorization":
```
correct factorization (scipy potrf)          residual=      0.0011  tol=30.0  passed=True
L of a DIFFERENT SPD matrix (shifted A)       residual=3.02e12       tol=30.0  passed=False
single-entry-corrupted L (sign flip)          residual=2.87e9        tol=30.0  passed=False
all-zero L (garbage output, right shape)      residual=8.80e12       tol=30.0  passed=False
tiny (1e-13-scale) noise on L (rounding-level) residual=     27.65    tol=30.0  passed=True
```
A correct factorization passes with large margin (residual ≈0.001 ≪ 30); a
factor of the wrong matrix, a single corrupted entry, and an all-zero
garbage output are all firmly rejected (residuals 8-13 orders of magnitude
over threshold); and a legitimate rounding-level perturbation (simulating a
different valid algorithm/pivoting order) still passes at 27.65, just under
30 — showing the threshold is reasonably tight, not trivially loose. No
defect found.

### V2 — the two genuinely-independent CPU implementations already in this codebase
`naive-csr-spmv` (`cpu_ref.py:51-63`, sparse.py) and `scipy-convolve-stencil`
(`stencil.py:427-473`) both differ from their shared reference at exactly the
fp64-ulp scale expected of a *different but equally correct* algorithm
(`8.9e-16` and `4.2e-17` respectively, checks #2 and #8) — i.e. they
demonstrate the reference-independence discipline working correctly when it
is actually applied. These are the templates B1-B4's fixes should follow.

---

## Reproduction

Scripts are preserved under `bench/audit/scripts/` (all CPU-only, drive the
real `kernelbench` code paths, no reimplementation, no CUDA):

```
PY=/pscratch/sd/c/cunyang/gnn/plexus_env/bin/python
cd /pscratch/sd/c/cunyang/msu/hpc-bench/kernel-papers/bench
$PY audit/scripts/check2_reference_independence.py       # B1 (sparse.py scipy-vs-reference bit-identity)
$PY audit/scripts/check3_cholesky_gate.py                # V1 (cholesky residual gate soundness)
$PY audit/scripts/check8_stencil_and_fft_reference_sharing.py  # B2, B3 (stencil.py, spectral.py)
$PY audit/scripts/check9_dense_reference_sharing.py       # B4 (dense.py)
$PY audit/scripts/check5_tolerance_parsing.py             # B7/F3 (tolerance parse survey)
$PY audit/scripts/check6_blas12_wrong_tolerance_demo.py   # B7 (concrete gate-leniency demo)
$PY audit/scripts/check7_tolerance_survey.py              # confirms B7 is isolated, not systemic
$PY audit/scripts/check1_stencil_reset_cost.py            # N1 (reset/run timing fraction)
$PY audit/scripts/check4_fft_c2r_precision.py             # F2 (FFT C2R operand divergence)
```
