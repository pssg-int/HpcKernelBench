# Domain-module audit: tensor.py / ml.py / compression.py vs their specs

Method: read `kernelbench/domains/{tensor,ml,compression}.py` in full against
`kernelbench/harness.py`, `kernelbench/runner.py`, `kernelbench/spec.py`, and
`benchspecs/{mttkrp,tensor-contraction,convolution,attention-kernel,
lossy-compression,lossless-compression}/spec.yaml`. Top suspicions were
verified empirically with `/pscratch/sd/c/cunyang/gnn/plexus_env/bin/python`
(CPU only) via targeted mutation tests (inject a plausible bug into shared
code, check whether the correctness gate still passes) and one live CLI run.
All test scripts are in
`/tmp/claude-106793/-pscratch-sd-c-cunyang-msu-hpc-bench/bf2b92db-5313-4854-9693-d7aa8869eb58/scratchpad/test_*.py`
and are reproducible.

Verdict key: **BUG** = the gate/module does not do what the spec or the
module's own docstring claims, confirmed by inspection and/or reproduction.
**FAIRNESS** = works as coded, but the choice diverges from the spec or
creates a systematic bias/comparability gap. **NIT** = latent risk, low
severity, or a documentation/robustness gap.

---

## 1. Reference independence (highest priority)

### 1.1 mttkrp — BUG (critical): reference and the only CPU impl call the *same function*

`kernelbench/domains/tensor.py:196-217` defines `_mttkrp_coo(...)`, the
Khatri-Rao-row-product + `np.add.at` scatter-accumulate. This single function
is called from **both**:

- `reference_mttkrp` (tensor.py:255-267, specifically line 264:
  `out = _mttkrp_coo(w.shape, w.indices, values, factors, mode, R)`)
- `NumpyMTTKRP.run` (tensor.py:296-298:
  `return _mttkrp_coo(h["shape"], h["indices"], h["values"], h["factors"], h["mode"], h["R"])`)

At `DEFAULT_PRECISION["mttkrp"] = "fp64"` (tensor.py:758, the spec-conforming
default), both call sites run identical fp64 arithmetic through the identical
function — not "the same algorithm independently coded," but literally one
function invoked twice. The shipped result
(`bench/results/mttkrp_mttkrp-general-kernel-fp64_1786071821.json`) shows
`correctness.value: 0.0` exactly for every run, confirming this — the
tensor-domain agent's report ("err=0.00e+00... since both reference and impl
compute identically at fp64") understates the problem: it isn't merely
same-algorithm/same-precision, it is the same code object.

**This means the gate cannot catch any bug in `_mttkrp_coo` itself** —
wrong mode excluded from the Khatri-Rao product, wrong scatter axis, sign
error, wrong accumulation order — none of it is observable, at any precision,
because whatever `_mttkrp_coo` computes (right or wrong) is what both sides
of the comparison produce.

**Empirical confirmation** (`test_mttkrp_mutation.py`): monkeypatched
`_mttkrp_coo` to scatter into `indices[:, (mode+1) % order]` instead of
`indices[:, mode]` — a classic "wrong axis" transcription bug, very plausible
in a hand-written MTTKRP kernel. Result:

```
BASELINE (unmodified _mttkrp_coo): 0.0 passed= True
AFTER INJECTING A WRONG-SCATTER-AXIS BUG into the shared _mttkrp_coo:
  gate value: 0.0  passed= True
  actual error vs. the INDEPENDENT ground truth: 8.28 / scale 3.36
```

An independent triple-nested-loop reference (no `np.add.at`, no shared
helper, built for this test) shows the mutated output is wrong by >200% of
scale — yet the shipped gate reports `err=0.0`, `passed=True`.

**Minimal fix** (per the task's own suggestion): make `reference_mttkrp`
genuinely independent — e.g. an explicit Python loop over nonzeros building
the Khatri-Rao row with a `for m in range(order)` product written out longhand
(no `np.add.at`, no shared function with the vectorized impl), or a
dense-tensor `np.einsum`-based reference for small/synthetic shapes. Either
removes the shared-function hazard; a loop-vs-vectorized split is the same
pattern `tensor.py`'s own docstring already uses to describe
`numpy-tensordot-contraction` vs `numpy-einsum-contraction`'s intended
contrast (permutation exposed vs. opaque), so the module already knows the
right shape for this fix, it just wasn't applied to mttkrp.

### 1.2 attention-kernel — BUG: `numpy-attention`'s mask gate is blind because it shares `_causal_mask()` with the reference

`ml.py:498-511` defines `_causal_mask(Sq, Sk)`. It is called from:

- `reference_attention` (ml.py:741-743)
- `NumpyAttention.prepare` (ml.py:783)

`NumpyFlashAttention` does **not** call it — it recomputes the identical
inequality inline, per-tile (ml.py:867-870: `j = np.arange(k0, k1)...; valid
= j <= (i_idx + offset)`), which is real independence for that impl only.

**Empirical confirmation** (`test_attn_mask_mutation2.py`): patched
`ml._causal_mask` to leak one extra (future) key per query —
`j <= (i + offset + 1)` instead of `j <= (i + offset)` — a plausible
off-by-one that violates the causal contract without degenerating any row to
all-masked:

```
BASELINE numpy-attention:       1.538e-07  passed=True
BASELINE numpy-flash-attention: 1.490e-07  passed=True

AFTER a 1-future-token-leak bug in the SHARED _causal_mask:
  numpy-attention gate value:       1.471e-07  passed=True   (unchanged!)
  numpy-flash-attention gate value: 3.09       passed=False  (correctly caught)
```

`numpy-attention`'s gate is completely insensitive to this causal-masking bug
(value barely moves, from float noise) because both sides of the comparison
apply the identically-broken mask. `numpy-flash-attention` catches it only
because it happens not to call the shared helper — that independence looks
incidental (both were presumably meant to implement "the same, obviously
correct" formula), not a designed control.

The rest of `NumpyAttention`'s pipeline (QK^T, softmax, @V — ml.py:786-794)
*is* separately written from `reference_attention` (ml.py:740-748), just
structurally identical line-by-line, so a bug isolated to arithmetic outside
masking (e.g. forgetting the `1/sqrt(d)` scale, wrong einsum subscript) would
be caught — verified by inspection, not retested, since the mask sharing is
the actual defect.

**Minimal fix**: give `_causal_mask` a second, differently-derived
implementation for the reference only (e.g. build the boolean mask from
`np.tril`/`np.triu` with the same semantics but a different formula/orientation
than the `i`/`j`/`offset` broadcast the impls share), or simply inline the
mask construction independently inside `reference_attention` instead of
importing the module-level helper.

### 1.3 tensor-contraction — mixed: `numpy-tensordot-contraction` is genuinely independent; `numpy-einsum-contraction` has little independent logic to be wrong

`reference_tensor_contraction` (tensor.py:531-538) computes
`np.einsum(w.equation, A, B)` (fp64). `NumpyTensordotContraction` implements
an explicit transpose+reshape+GEMM pipeline (tensor.py:610-655) — a
genuinely different code path.

**Empirical confirmation** (`test_tensordot_mutation.py`): swapped the
free/contracted axis order when building `axes_a`/`axes_b`
(a plausible "which axis comes first" slip in a hand-written contraction
kernel):

```
BASELINE numpy-tensordot-contraction: 1.56e-07  passed=True
AFTER swapping axes_a/axes_b free/contracted order: 5.71  passed=False
```

Correctly caught — `numpy-tensordot-contraction`'s independence is solid.

`NumpyEinsumContraction` (tensor.py:541-573), by contrast, is
`np.einsum(h["equation"], h["A"], h["B"], optimize=h["path"])` — i.e. it
calls the exact same vendored primitive the reference uses, over the exact
same equation string derived from the same `ContractionWorkload`. There is
essentially no domain-module-owned logic left for a "wrong-but-plausible
implementation bug" to live in (numpy's einsum contraction engine itself is
out of scope — same trust level as using scipy as ground truth elsewhere in
this codebase). This isn't a defect so much as a fact worth stating plainly:
this impl's correctness gate mostly tests "did numpy compute what its own
manual computed," which is a much weaker claim than what the
im2col-vs-scipy or naive-vs-flash pairs in the other two domains test.
Classified NIT rather than BUG since there's no realistic implementation
error the current design could hide that mttkrp's/attention's sharing
*actually* hides (the shared einsum path is a trusted library call, not
domain-owned arithmetic).

### 1.4 convolution — OK: reference is genuinely independent of both impls

`reference_conv` → `_direct_conv_fp64` (ml.py:307-328) is a hand-written
`Kh×Kw`-tap loop with `einsum('ncij,oc->noij', ...)` per tap — shares no code
with `NumpyIm2colConv` (slice-gather into a preallocated im2col buffer +
BLAS `@`, ml.py:391-405) or `ScipyDirectConv` (`scipy.ndimage.correlate`,
ml.py:465-485). All three are structurally different implementations of the
same math. Not retested by mutation (inspection is conclusive here — no
shared function, no shared helper), but flagged as the one solid baseline in
this file for what "independent" is supposed to look like.

### 1.5 compression — OK, and not really the same category of question

`reference_compression` (compression.py:257-260) returns the original data
untouched; the correctness gate is a genuine roundtrip test (lossless:
`Decompress(Compress(x)) == x` bit-exact; lossy: `|Decompress(Compress(x)) -
x| <= eb`) against ground truth that is independent of any codec's own
internal logic by construction. No reference-independence gap here — see
§3 for the mechanics.

---

## 2. Cost-rule fidelity

### 2.1 convolution flops_ref — correct, matches spec literally

Spec (`benchspecs/convolution/spec.yaml:14`): `flops_ref = 2 * N * Cout *
(Cin/groups) * Kh * Kw * Hout * Wout`. Module (`ml.py:281-300`,
`_cost_convolution`): `flops = 2 * N * w.Cout * w.cin_per_group * w.Kh *
w.Kw * w.Hout * w.Wout`. Exact match. Confirmed the formula uses only
`Hout`/`Wout` (never `Hin`/`Win`), so the `_infer_in_dim` inverse-derivation
choice (`ml.py:115-131`, solving `Hin` from `Hout`/stride/pad) affects only
what gets *multiplied* (operand shapes), never what gets *reported* as work —
exactly as the module's own docstring claims. No issue.

### 2.2 attention causal-factor convention — correct, matches spec literally

Prefill (spec `attention-kernel/spec.yaml:74-76`): `flops = (mask==bidirectional
? 4 : 2) * B*H*S^2*d`. Module (`ml.py:686-690`): `(2 if causal else 4) *
B*H*Sq*Sk*d` for `Sq==Sk>1` — same formula, causal/bidirectional inverted
correctly (causal→2, bidirectional→4 both places). Decode (spec
`attn-decode-kv-cache-kernel`'s secondary metric, spec.yaml:139): `flops =
4*B*H*L*d`. Module: `4 * B*H*Sq*Sk*d` for the decode-shaped branch, and with
`Sq=1, Sk=L` this is exactly `4*B*H*L*d`. Byte formula also checked against
spec's `bytes = B*2(K+V)*L*Hkv*d*sizeof + Q,O` (spec.yaml:129-131) — module's
`q+k+v+o` byte sum matches term-for-term. No issue.

### 2.3 mttkrp `order*nnz*R` vs spec's `2*R*nnz` — FAIRNESS: real, disclosed in code, but NOT disclosed in the output record

`tensor.py:220-249` (`_cost_mttkrp`) computes `flops = w.order * w.nnz * R`,
the literal instruction count for what `_mttkrp_coo` actually executes
((order-1) multiplies + 1 add per nonzero per rank = `order` R-wide ops).
The spec (`mttkrp/spec.yaml:55-56`) states `2*R*nnz(X)` — the
SpMV/SpMM-style "one multiply + one accumulate per nonzero" convention that
does *not* separately charge for the extra `(order-2)` Khatri-Rao chain
multiplies needed once more than 2 modes are involved. The module's own
docstring (`tensor.py:220-238`) discloses this honestly and the spec's own
`open_questions` (spec.yaml:174-178) admits the "2R vs R" ambiguity was never
checked against any paper's internal counter — so this is a defensible,
disclosed choice, not silent gaming.

**But the disclosure doesn't survive into the actual data product.** Checked
the shipped result `mttkrp_mttkrp-general-kernel-fp64_1786071821.json`:
`spec_notes_on_fairness` (copied verbatim from the spec) says nothing about
this; `metrics`/`params` carry no note either. Concretely, for the shipped
smoke case (order=3, nnz=2000, R=32): `work_count: 192000` = `3*2000*32`
(module's convention) vs. `128000` = `2*32*2000` (spec's literal convention)
— **the reported GFLOP/s number is 1.5x higher than what the spec's own
formula would produce for identical wall-clock time**, and for the order-4
smoke case the ratio is 2x (`4R` vs `2R`). A reader of the JSON result
records (the actual downstream artifact for Phase 2/3) comparing this
module's mttkrp GFLOP/s against a paper's number computed under the spec's
literal 2R convention has no way to know from the record alone that they
aren't comparable. Fix: echo the chosen flop convention into `params` or
`metrics` (e.g. `params["flop_convention"] = "order*nnz*R (see tensor.py
docstring); spec literal is 2*R*nnz"`) so it travels with the number.

### 2.4 compression `work = original bytes`, both directions — correct

`_cost_bytes` (`compression.py:227-249`) returns `orig = w.data.nbytes`
unconditionally, regardless of `direction`, for both `lossy-compression` and
`lossless-compression` (registered identically at compression.py:252-253).
Matches both specs' explicit "numerator is ALWAYS original (uncompressed)
byte count" requirement for compress AND decompress. No issue.

---

## 3. Compression-specific (all four checked; no bugs found)

### 3.1 ratio/PSNR/max_err computed outside the timed region — confirmed by tracing harness.py's actual order

`harness.run_variant` (`harness.py:190-305`) order is: `prepare()` →
untimed correctness-check `run()` → `warmup` × `run()` (discarded) →
`reps` × (`timer()` + `run()`) → **`finally: impl.free(handle)`** → stats/cost
computed from `params` *after* the `try/finally` block exits.

- `_LosslessCodec.free` (`compression.py:348-354`) records
  `achieved_compressed_bytes`/`compression_ratio` from `h["last_out"]` (the
  *last measured rep's* compressed output) — called only after every timed
  rep has completed. Confirmed outside the timed window.
- `_QuantizeZlibCodec.free` (`compression.py:481-498`) computes
  `achieved_max_abs_error`/`psnr_db` the same way, also strictly after
  timing. Confirmed.
- For `-decompress`-direction impls, the untimed one-shot `_compress()` call
  needed to produce the input happens in `prepare()` (before timing even
  starts), and ratio bookkeeping for that direction happens there too
  (`_LosslessCodec.prepare`, compression.py:318-328) — also correctly outside
  the timed window, just at the other end.

No timing leakage in either direction, for either codec family.

### 3.2 lossy gate uses the workload's own `eb`, not a spec-parsed number — verified live

Traced the parse: `spec._tolerance` (`spec.py:65-100`) on
`lossy-compression/spec.yaml`'s correctness text ("...`<= eb`...") finds no
digit after the comparison operator (`eb` isn't numeric) and no structural
marker, so it correctly returns `(None, "unparsed")` — `variant.tolerance`
is `None` for `lossy-comp-kernel-cpu-ebound`. `runner.py:196` passes
`tolerance_override=getattr(m, "correctness_tolerance", None)` where `m` is
the `Field` workload, whose `correctness_tolerance` is the resolved absolute
bound from `_resolve_eb` (compression.py:168-182).

Ran it for real (not just inspected):

```
$ python -m kernelbench.runner --kernel lossy-compression \
    --variant lossy-comp-kernel-cpu-ebound \
    --impl quantize-zlib-lossy-compress --smoke
  running quantize-zlib-lossy-compress smoke-smooth-3d ... (err 3.42e-03 <= 0.0034191445382311943)
  running quantize-zlib-lossy-compress smoke-turbulent-3d ... (err 1.29e-03 <= 0.0012902792096138001)
  running quantize-zlib-lossy-compress smoke-multiscale-3d ... (err 3.81e-03 <= 0.003808632005006075)
3/3 runs valid
```

The tolerances printed (`0.0034191...`, `0.0012902...`, `0.0038086...`) are
each field's own resolved rel-eb bound (`DEFAULT_REL_EB=1e-3` × that field's
own value range), not a single literal spec number, and they differ per
field as expected. The `tolerance_override` path works end-to-end. No issue.

### 3.3 `direction` param recorded — confirmed

Both `_LosslessCodec.prepare` (compression.py:321) and
`_QuantizeZlibCodec.prepare` (compression.py:456) stamp
`params["direction"] = self.direction` unconditionally, before anything is
timed. Confirmed present in the shipped result JSONs (e.g.
`quantize-zlib-lossy-compress ... 'direction': 'compress'` in
`lossy-compression_lossy-comp-kernel-cpu-ebound_...json`). No issue.

### 3.4 lossless roundtrip gate truly bit-exact — confirmed, with one caveat noted (not a bug)

`CORRECTNESS_MODE["lossless-compression"] = "exact"` routes to
`check_correctness`'s `mode == "exact"` branch (`harness.py:156-158`):
`np.array_equal(out, ref)` where both sides are `.astype(np.float64)`
(`_LosslessCodec.to_host`, compression.py:337-343; `reference_compression`,
compression.py:257-260). Since the underlying data is fp32 (default
precision for this kernel) and fp32→fp64 upcast is lossless/deterministic
for finite values, `np.array_equal` after the upcast is equivalent to
comparing the raw fp32 bytes directly — genuinely bit-exact, not a numeric
tolerance in disguise. Caveat noted for completeness, not a bug: this would
give a false *mismatch* if the field ever contained NaN (`NaN != NaN`), but
none of the three synthetic generators (`_smooth_field`/`_turbulent_field`/
`_multiscale_field`) can produce NaN, so this is unreachable in practice.

One more thing worth noting while here: `runner.py`'s
`tolerance_override=getattr(m, "correctness_tolerance", None)` fires for
*lossless*-compression runs too (the `Field` dataclass carries a resolved
`correctness_tolerance` regardless of kernel), so a spurious `eb` value
does get computed and passed through — but `check_correctness`'s `exact`
branch ignores the `tolerance` argument entirely and hardcodes
`tolerance=None` in the returned record (harness.py:158). Confirmed harmless
by reading the branch; NIT at most (mildly confusing that a lossless run's
call site computes a number that's then silently discarded, but it's dead
code, not a live defect).

---

## 4. Attention-specific

### 4.1 flash-attention numerically equivalent to naive — confirmed empirically, including at scale

Beyond the shipped smoke result (`numpy-attention` err ≈1.5e-7 vs tolerance
1e-2), ran three additional shapes not in the smoke/recommended set
(`test_flash_scale.py`) to stress edge cases the shipped smoke run doesn't
reach:

```
long causal prefill (GQA, ragged tail, Sq=Sk=4100, not a multiple of BLOCK=64):
  NumpyAttention       err=1.575e-07  passed=True
  NumpyFlashAttention  err=1.707e-07  passed=True
long decode (Sq=1, Sk=16384, GQA):
  NumpyAttention       err=1.085e-08  passed=True
  NumpyFlashAttention  err=1.097e-08  passed=True
long bidirectional prefill (Sq=Sk=3000):
  NumpyAttention       err=8.919e-08  passed=True
  NumpyFlashAttention  err=8.916e-08  passed=True
```

Tiled online-softmax (`NumpyFlashAttention`, `ml.py:854-878`) tracks the
materialized naive softmax to the same ~1e-7 precision as `NumpyAttention`
across a ragged last-tile case (4100 not a multiple of `BLOCK=64`), a
long-KV decode case, and a bidirectional case — no numerical blow-up, no
degradation with sequence length. Genuinely equivalent. No issue (aside from
the shared-mask blind spot in §1.2, which affects `numpy-attention`'s gate,
not this equivalence question).

### 4.2 Mask handling identical between impls and reference — see §1.2 (BUG)

The masking *formula* is identical everywhere (verified: `_causal_mask`'s
`j <= i + offset` and `NumpyFlashAttention`'s inline `j <= i_idx + offset`
are the same inequality). The problem isn't disagreement, it's that
`numpy-attention`'s copy is share-called rather than independently derived
— see §1.2 for the mutation-test evidence.

### 4.3 GQA head-mapping — correct convention, but duplicated (not shared) code across all 3 sites

`Kh = np.repeat(K, g, axis=1)` (and the `Vh` equivalent) appears verbatim at
`reference_attention` (ml.py:738-739), `NumpyAttention.prepare`
(ml.py:781-782), and `NumpyFlashAttention.prepare` (ml.py:840-841).
`np.repeat`'s elementwise-repeat semantics (`[kv0,kv0,...,kv1,kv1,...]`,
query head `h` → kv head `h // group_size`) matches the standard GQA
convention (equivalent to HF transformers' `repeat_kv`). Confirmed correct.
Classified NIT rather than BUG: this is copy-pasted code, not a shared
function call, so an edit to one site wouldn't silently propagate to the
others the way `_mttkrp_coo`/`_causal_mask` do — but a *systemic*
misunderstanding of the convention (e.g. all three having used `np.tile`
instead of `np.repeat`) would still have gone undetected by this gate, since
all three sites encode the same choice. Worth knowing, not worth fixing
given the convention used is actually correct.

---

## 5. Preprocessing placement, determinism, and one more conformance gap

Preprocessing placement reviewed across all three files and found correct
in every case checked: factor-matrix/operand generation and format
conversion are hoisted into `prepare()` (`NumpyMTTKRP.prepare`
tensor.py:282-294; `NumpyEinsumContraction.prepare`'s `np.einsum_path`
search tensor.py:558-561; `NumpyIm2colConv.prepare`'s buffer allocation
ml.py:377-389; compression's untimed one-shot compress-for-decompress calls,
§3.1). All random generation uses `np.random.default_rng(seed)` with an
explicit, params-derived seed shared between reference and every impl
(`_make_factors`, `_make_conv_operands`, `_qkv`, `operands()`) — no
nondeterminism found anywhere in these three modules.

### 5.1 `NumpyTensordotContraction(transpose_in_timing=False)` — FAIRNESS: a real non-conformance mode the harness can't flag

Per spec (`tensor-contraction/spec.yaml` notes_on_fairness: "never strip
[permutation] out of a timed kernel or amortize it separately"), the
`transpose_in_timing=False` path (tensor.py:640-646) is explicitly
non-spec-conforming by the module's own docstring (tensor.py:589-595,
"hoists BOTH input permutes into prepare() instead"). `runner.py`'s
`conforming` field (`runner.py:217-223`) is computed *only* from
`--smoke`/`--warmup`/`--reps`/environment warnings — it has no hook for a
domain module to report its own params-level non-conformance. Currently
unreachable via the documented CLI (no `--transpose-in-timing` flag exists;
`runner.py`'s `params` dict never sets this key, so the default `True` — the
spec-conforming path — is always what a CLI run gets). But a direct-API
caller who sets `params["transpose_in_timing"] = False` would get a result
JSON that still says `"conforming": true` with no trace of the deviation
anywhere in the record. Latent, not currently exploitable through the
runner, but worth a one-line fix (thread a `nonconformance_reasons` hook
through `params` the same way `direction`/`achieved_compressed_bytes` are
threaded).

### 5.2 `algorithm_class` disclosure — BUG: mandatory spec metadata never reaches the result record

`convolution/spec.yaml:110-111` states plainly: "`algorithm_class` field
(direct | im2col-implicit-gemm | winograd | fft) is **mandatory metadata on
every submitted result**." `NumpyIm2colConv`/`ScipyDirectConv` both set
`algorithm_class` as a class attribute (ml.py:371, ml.py:450). But
`harness.Implementation`'s protocol (harness.py:34-45) and `RunResult`
(harness.py:74-111) have no field for it, and `harness.run_variant` never
reads `impl.algorithm_class` anywhere. Confirmed empirically — checked the
shipped `convolution_conv-dense-kernel-fp32_...json`: the string
`"algorithm_class"` does not appear anywhere in the record (top-level keys
are `['correctness','implementation','kernel','matrix','metrics','params',
'platform','precision','preprocessing_ms','protocol_used','stats_ms',
'times_ms','valid','variant','warnings']`). A reader of the result record
has to already know which class name maps to which algorithm — the mandatory
metadata the spec asks for simply isn't in the JSON. Since `runner.py`
labels this exact record `"conforming": true` (no `--smoke`, no protocol
override, no env warnings triggered in a hypothetical clean run), the
record would be self-certifying as spec-conforming while actually missing
mandatory spec-required content. Minimal fix: add `algorithm_class:
getattr(impl, "algorithm_class", None)` to `RunResult`/`to_dict` in
harness.py, populated whenever the implementation sets it (harmless
`None` for kernels where it doesn't apply).

### 5.3 `ITEMSIZE` missing `bf16` in tensor.py — NIT

`tensor.py:52`: `ITEMSIZE = {"fp64": 8, "fp32": 4, "fp16": 2, "tf32": 4}` —
no `bf16` entry, unlike `ml.py:102`'s `ITEMSIZE` which does include it. Not
currently reachable (`DEFAULT_PRECISION` for both mttkrp and
tensor-contraction is fp64/fp32, and the CLI's precision auto-detection only
special-cases fp64/fp32 text), but the CUDA-side `TorchMTTKRP`/
`TorchEinsumContraction`/`TorchTensordotContraction` in `impls/gpu_cuda.py`
accept an arbitrary `precision` string, so a future `--precision bf16` GPU
run would `KeyError` inside `_cost_mttkrp`/`_cost_tensor_contraction`. Cheap
fix: add `"bf16": 2` to `tensor.py`'s `ITEMSIZE` for parity with `ml.py`.

---

## Summary of verdicts

| # | Location | Verdict | One-line |
|---|---|---|---|
| 1.1 | tensor.py:196-217,255-267,296-298 | **BUG** | mttkrp reference and impl call the literal same function; empirically 0 err survives a real bug |
| 1.2 | ml.py:498-511,741-743,783 | **BUG** | numpy-attention shares `_causal_mask` with the reference; empirically a causal-leak bug is invisible |
| 5.2 | ml.py:371,450 + harness.py (RunResult) | **BUG** | spec-mandatory `algorithm_class` metadata never reaches the result JSON |
| 2.3 | tensor.py:220-249 | **FAIRNESS** | order*nnz*R vs spec's 2*R*nnz (1.5-2x); disclosed in code, not in the output record |
| 5.1 | tensor.py:589-595,640-646 | **FAIRNESS** | transpose_in_timing=False is a real non-conformance mode the harness can't flag (currently CLI-unreachable) |
| 1.3 | tensor.py:541-573 | **NIT** | numpy-einsum-contraction is a thin pass-through to the same primitive as the reference; little independent logic to be wrong |
| 4.3 | ml.py:738-739,781-782,840-841 | **NIT** | GQA `np.repeat` convention correct but copy-pasted identically 3x, not shared, but still systemic-bug-blind |
| 3.4 | compression.py + runner.py | **NIT** | lossless runs compute a spurious tolerance_override that's silently discarded (dead code, harmless) |
| 5.3 | tensor.py:52 | **NIT** | `ITEMSIZE` missing `bf16`, latent KeyError risk for a future GPU bf16 run |

Solid, no issues found: convolution's cost rule (§2.1) and reference
independence (§1.4), attention's cost rule (§2.2) and flash-vs-naive
numerical equivalence at scale (§4.1), compression's cost rule (§2.4) and
all four compression-specific mechanics (§3.1-3.4), and tensordot-contraction's
reference independence (§1.3, positive half).
