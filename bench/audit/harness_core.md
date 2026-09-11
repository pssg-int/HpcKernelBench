# Harness core audit — correctness/fairness/reproducibility

Scope: `kernelbench/{harness,spec,runner,report,metrics,workload,matrices,env}.py`,
`csrc/kernels.cu`, `smoke_all.sh`, cross-referenced against `DOMAIN_GUIDE.md` and
`kernelbench/domains/sparse.py`, and against the real specs in `../benchspecs/`.

All findings below were verified by direct execution (CPU only, no CUDA
launched) with `/pscratch/sd/c/cunyang/gnn/plexus_env/bin/python`. Reproducer
scripts live in the sandbox scratchpad; the relevant output is inlined under
each finding so this file is self-contained.

Classification: **BUG** = a wrong or self-inconsistent number/flag.
**FAIRNESS** = numerically defensible but misleading, or a documented
guarantee that doesn't actually hold. **NIT** = real but narrow/cosmetic,
not currently reachable by this project's actual inputs.

---

## BUGS

### B1. `statistic_used` in the result record can name a statistic that was never actually used

`harness.py:267-268`:
```python
stats = _stats(times_ms)
sec = stats[proto.statistic if proto.statistic in stats else "median"] / 1e3
```
`_stats()` (harness.py:114-125) only produces the keys `median, min, max, mean,
stdev, p10, p90, n`. `proto.statistic` comes from `spec.py`'s normalization
(`spec.py:235-237`):
```python
stat_norm = "median" if "median" in stat.lower() else \
            ("mean" if "mean" in stat.lower() else stat.lower().split()[0])
```
Any spec statistic text that isn't literally "median"/"mean" as a substring
falls through to **the first word of the raw text**, lowercased — e.g. "p99",
"minimum", "maximum". None of those are keys in `_stats()`'s dict (`min`/`max`
exist, but not `minimum`/`maximum`; `p10`/`p90` exist, but not `p99`). The
fallback silently substitutes `median` — **and nothing downstream is told**:
`met["statistic_used"] = proto.statistic` (harness.py:281) still records the
*original, unused* string.

Reproduced directly:
```
proto.statistic='p99'      -> actually used 'median' (value=1.0250 ms) <-- MISMATCH
proto.statistic='minimum'  -> actually used 'median' (value=1.0250 ms) <-- MISMATCH
proto.statistic='maximum'  -> actually used 'median' (value=1.0250 ms) <-- MISMATCH
proto.statistic='min'      -> actually used 'min'    (value=0.9000 ms)   (OK, key matches)
proto.statistic='max'      -> actually used 'max'    (value=50.0000 ms)  (OK, key matches)
```
A result record can therefore claim `"statistic_used": "p99"` while the
`throughput`/`gflops` figure was actually computed from the **median** —
silently, with no warning appended (compare to the explicit
`warnings.append("spec ... unparsable; harness default ... used")` pattern
used elsewhere in the same function for warmup/reps). The record is
self-inconsistent in a way that defeats the file's own stated audit purpose
("Every constant used is copied into the result record together with where it
came from, so a result can be audited against its spec without rerunning it" —
harness.py:17-18).

**Fix direction**: either raise/warn when `proto.statistic` isn't a `_stats()`
key, or (better) extend `_stats()` to compute arbitrary percentiles from the
raw statistic string, and always write the *key actually indexed* to
`statistic_used`, not the raw spec string.

---

### B2. `doc["conforming"]` in `runner.py` doesn't see a run's own internal non-conformance

`runner.py:170-171` computes local `warmup`/`reps` purely from the CLI:
```python
warmup = args.warmup if args.warmup is not None else (5 if args.smoke else None)
reps   = args.reps   if args.reps   is not None else (20 if args.smoke else None)
...
doc = {
    ...
    "conforming": bool(not args.smoke and warmup is None and reps is None
                       and not env_warnings),
    ...
}
```
This only checks "did the *user* override the protocol on the command line."
It never looks at what `harness.run_variant()` actually did internally. But
`run_variant()` (harness.py:213-224) has its own, *independent* fallback: if
the spec's own `warmup`/`reps` text is unparsable (`proto.warmup`/`proto.reps`
is `None`), it silently substitutes hardcoded defaults (10 / 100) and appends
a per-run warning — and that warning **never propagates** to
`doc["nonconformance_reasons"]`; `runner.py`'s `main()` only does
`records.append(r.to_dict())`, nothing aggregates `r.warnings` upward.

Reproduced by calling `harness.run_variant()` with a `Protocol(warmup=None,
reps=None, ...)` (simulating an unparsable spec field) exactly as
`run_variant` would receive it from a real `--kernel ... --variant ...` CLI
invocation with no `--warmup`/`--reps` flags:
```
run-level warnings: ['spec warmup unparsable; harness default 10 used',
                      'spec reps unparsable; harness default 100 used']
protocol_used: {'warmup': 10, 'reps': 100, ...}
runner.py-level doc['conforming'] = True
```
`report.py`'s `text_table()` (report.py:31) shows a bare `[CONFORMING]` tag
driven only by `doc.get("conforming")`. So a document can display
`[CONFORMING]` while every run inside it actually used non-spec, harness-
invented protocol constants — exactly the failure mode `README.md` promises
the harness prevents ("A result that deviates from its spec says so, in the
record").

**Fix direction**: fold `any("harness default" in w for r in records for w in
r["warnings"])` (or a dedicated boolean on `RunResult`) into
`doc["conforming"]`/`nonconformance_reasons`.

---

### B3. `spec.py`'s `_first_int` truncates a multi-value `reps` sweep to its first number

`benchspecs/spmv/spec.yaml:162` (real spec text, `spmv-sequence-krylov`):
```yaml
reps: "k in {2, 4, 8}; report each k separately (not just the largest)"
```
`spec.py:23,27-37`:
```python
_INT_RE = re.compile(r"(\d+)")
...
m = _INT_RE.search(value)
if m:
    return int(m.group(1)), "parsed"
```
`.search()` returns the *first* match only. Reproduced directly against the
real, loaded spec:
```
sp = spec.load("spmv"); v = sp.variant("spmv-sequence-krylov")
v.protocol.reps == 2          # not {2, 4, 8}
v.protocol.provenance['reps'] == 'parsed'   # looks confidently parsed, not flagged
```
The spec explicitly wants three separate runs (k=2, 4, 8), each reported
separately — the whole point of the variant is to show how per-call
throughput changes across a Krylov sequence length. The harness silently
collapses this to a single `reps=2` with `provenance="parsed"` (i.e. it looks
successfully parsed, not defaulted, so nothing flags it for review). Whoever
wires up `spmv-sequence-krylov` (not yet implemented per `sparse.py`'s
`PLANNED`) would get exactly one k value tested, not three, unless they know
to special-case this variant id.

This is the counterpart to the "k=100 post-conversion SpMM calls" case the
task asked me to check for correctness: that one **is** correct (verified —
`spmm-gpu-e2e-preproc-f32`'s `reps` text is `"k=100 post-conversion SpMM
calls; ..."`, single number, parses to 100, which *is* the spec's real k).
`_first_int` is a "grab-the-first-integer" heuristic that happens to be right
whenever the text has exactly one number and wrong whenever it has more than
one — `spmv-sequence-krylov` is a real, present-in-the-repo case of the
latter.

---

### B4. `CustomSpMM`/`CustomSDDMM` silently run the fp32 CUDA kernel on fp64 buffers

`kernelbench/impls/gpu_cuda.py`:
- `CustomSpMV.run()` (line 259-261) correctly branches:
  ```python
  fn = h["lib"].spmv_csr_warp_f32 if self.precision == "fp32" \
      else h["lib"].spmv_csr_warp_f64
  ```
- `CustomSpMM.run()` (line 291-292) does **not** branch:
  ```python
  h["lib"].spmm_csr_warp_f32(h["rows"], h["N"], ..., h["B"].data_ptr(), ...)
  ```
- `CustomSDDMM.run()` (line 429-430) same pattern, hardcoded `_f32`.

`prepare()` in both classes builds device tensors with
`dt = _torch_dtype(self.precision)` (unrestricted — `fp64` is accepted with
no guard), so `--precision fp64 --impl custom-warp-csr-spmm` (or `sddmm`)
builds real `float64` buffers and then unconditionally calls the kernel whose
C signature is `const float* data, const float* B, float* C`. `csrc/kernels.cu`
confirms there is no `spmm_csr_warp_f64`/`sddmm_csr_warp_f64` symbol at all
(only `spmv_csr_warp_f64` exists — grep confirms `spmm_csr_warp_f32` and
`sddmm_csr_warp_f32` are the *only* wrapper symbols exported for those two
kernels), and `gpu_cuda.py`'s `_load_lib()` function table (line 37-44) never
registers such symbols either — so this isn't a "not implemented yet, raises
clearly" situation, it's a silent type mismatch: an 8-byte-per-element device
buffer gets reinterpreted as 4-byte floats inside the kernel.

Not executed here (CUDA), but this is a pure source-level type mismatch,
confirmable by inspection alone. It would very likely fail the pre-timing
correctness gate (garbage values) rather than publish a silently-wrong
throughput number, but it means `custom-warp-csr-spmm`/`-sddmm` simply cannot
be run at fp64 at all today, with no error message pointing at why —
`self.precision` is accepted and threaded through `prepare()`, then silently
ignored in `run()`.

**Fix direction**: mirror `CustomSpMV`'s branch, or raise
`NotImplementedError` in `__init__`/`prepare()` when `precision != "fp32"`
until `spmm_csr_warp_f64`/`sddmm_csr_warp_f64` exist.

---

### B5. SuiteSparse complex matrices are silently corrupted, undetectably, before any spec check runs

`matrices.py:101-109`:
```python
def load_matrix(name, group=None):
    ...
    coo = mmread(path)
    csr = sp.csr_matrix(coo, dtype=np.float64)
    ...
```
`spmv`'s spec (`benchspecs/spmv/spec.yaml`, `spmv-csr-kernel.inputs.selection`)
explicitly requires "real (non-complex)" matrices — but nothing in
`matrices.py` checks `coo.dtype.kind` or rejects a complex source file.
Verified directly (write a small complex `.mtx`, read it back exactly the way
`load_matrix` does):
```
mmread dtype for a complex .mtx: complex128
sp.csr_matrix(..., dtype=float64) on complex input -> 1 python warning raised:
    ComplexWarning: Casting complex values to real discards the imaginary part
original complex diagonal : [1.+2.j 0.+5.j 3.+0.j 2.-1.j]
after forced float64 cast : [1.    0.    3.    2. ]
```
A purely-imaginary entry (`0+5j`) becomes an exact `0.0` — not an
approximation, a silent value change baked into `matrix.csr` before anything
else touches it. Critically: **this can never be caught by the correctness
gate**, because the reference computation (`impls/cpu_ref.py`'s
`reference_spmv` etc.) is computed from `matrix.csr` — the *same*
already-truncated real matrix — not from the original complex data. Both
`impl.run()` and the reference agree with each other perfectly (they're
computing the identical, wrong-input problem), so `check_correctness` reports
a clean pass. The benchmark would run, pass its gate, and publish a real-
looking throughput number for a matrix that isn't the one SuiteSparse
actually ships.

(Checked the adjacent worry too — MatrixMarket **pattern** matrices are fine:
`mmread` fills them with `1.0` automatically, `astype(float64)` is a no-op:
`mmread dtype for a MatrixMarket 'pattern' file: float64, values=[1. 1. 1.]`.)

**Fix direction**: `load_matrix()` should check `np.iscomplexobj(coo)` (or
`coo.dtype.kind == 'c'`) and raise, not silently downcast; the spec's
`selection` text already says matrices should be real, this just needs to be
enforced in code rather than only in prose.

---

## FAIRNESS

### F1. `gflops_amortized_over_reps` amortizes preprocessing over `reps`, a protocol constant that usually has nothing to do with the spec's own amortization horizon `k`

`harness.py:285-289`:
```python
if preprocessing_ms > 0:
    amortized_sec = sec + (preprocessing_ms / 1e3) / reps
    met["gflops_amortized_over_reps"] = metrics.throughput(fl, amortized_sec)
    met["amortization_k"] = reps
```
This field is computed **unconditionally** whenever `preprocessing_ms > 0`
(true for essentially every run — `prepare()` always takes measurable wall
time). Checked against the two real spmm variants side by side:

- `spmm-gpu-kernel-f32`: `reps=100`, `metric = {'primary': 'GFLOP/s
  (2*nnz*N flops)', 'secondary': 'ms'}` — **no amortized metric defined at
  all**. The spec's own text: `format: "CSR (given, conversion not timed;
  matches all GPU papers surveyed)"`, `preprocessing_reported: "separately,
  once, following RoDe's csv-per-baseline practice"`. `reps=100` here exists
  purely so `_stats()` has enough samples for a stable median/stdev — it is a
  *statistical* repeat count with zero real-world "amortize preprocessing
  over N reuses" meaning.
- `spmm-gpu-e2e-preproc-f32`: `reps=100`, `metric = {'primary': 'GFLOP/s
  (amortized)', ...}`. Here `reps` **is** the spec's real `k` — but only
  because the spec's prose happens to spell it `"k=100 post-conversion SpMM
  calls; report both amortized ... and pure-kernel time"`, and
  `_first_int` extracted the 100 out of that same sentence.

Both runs get `amortization_k = 100` and a `gflops_amortized_over_reps` value
in the JSON, computed by the identical formula — but only the second one's
number corresponds to anything the spec actually defines. For the first, the
harness manufactures a metric the spec explicitly keeps separate ("reported
separately... never amortized" is the whole point of that variant existing
distinctly from the e2e one), using a number (`reps`) whose only real meaning
for that variant is "how many timestamps we sampled for the median."

This is not currently surfaced by `report.py` (grepped — `gflops_amortized_
over_reps`/`amortization_k` aren't read anywhere in `report.py`), so it isn't
yet visibly distorting a leaderboard. But it sits in every result JSON,
correctly formatted and named suggestively, and would mislead anyone who
consumes the raw records directly (or a future report.py column) into
thinking it reflects the variant's own claimed amortization behavior.

**Fix direction**: only compute this field for variants whose `metric.primary`
text says "amortized" (or better: have the domain module pass an explicit `k`
alongside `tolerance_override`, the way lossy-compression already threads a
workload-derived bound through `run_variant`), and stop reusing `reps` as a
stand-in for a concept it usually isn't.

### F2. CPU `Timer`/context-manager dispatch overhead lands *inside* the measured interval, not outside it

Measured directly (20,000 iterations, a near-zero-cost `run()` that just
increments an int — representative of what a fast, small-matrix scipy SpMV
call would look like relative to Python overhead):
```
raw impl.run() cost, no instrumentation :     192.3 ns/call
time.perf_counter()-measured 'seconds'  :     490.8 ns/call   <- what becomes
                                                                  the throughput
                                                                  denominator
harness-loop overhead beyond the bare call: ~807.3 ns/iteration (wall clock)
```
The reported `t.seconds` (490.8 ns) is ~2.5x the actual `run()` cost (192.3
ns). This isn't a logic bug — `impl.timer()` is constructed *before*
`__enter__` (correctly outside the window), and the `with` statement does
bracket only `impl.run(handle)`. But the Python-level cost of *dispatching*
`__enter__`/`__exit__` (bound-method lookup + call for each) happens between
the two `time.perf_counter()` calls, so it's arithmetically inside the
measured interval whether or not it's "inside the `with:` body." For any CPU
kernel whose true cost is at or below a few hundred nanoseconds to a couple
of microseconds, this instrumentation overhead is a non-trivial fraction of
the reported time, which will read as inflated `ms` / deflated `GFLOP/s`.

This does **not** bias implementation-vs-implementation rankings measured the
same way (every CPU impl pays the same fixed `Timer()` dispatch tax), so
leaderboards stay internally consistent. But the *absolute* throughput number
the harness reports — the number a paper's own claimed GFLOP/s gets compared
against — is measurably not "what `run()` alone costs" for fast kernels. Real
spmv/spmm/sddmm variants target `nnz >= 1e4`, which likely keeps most
production runs above this noise floor, but the CPU smoke matrices (4000
rows) and any small-matrix sweep sit closer to it.

(CUDA event timing is unaffected by this specific issue — `CudaEventTimer`
measures device-side timestamps, so host-side Python dispatch overhead around
`start.record()`/kernel-launch/`stop.record()` isn't counted; see harness.py's
own `impls/gpu_cuda.py` docstring for that design intent. `CudaEventTimer.
__init__` doing `torch.cuda.Event(...)` twice per iteration, outside the
timed window, is real per-iteration Python/driver overhead too, but it's a
throughput-of-the-harness-itself cost, not something that leaks into the
recorded interval — not flagged as a numeric issue here.)

### F3. Precision auto-detection in `runner.py` is dead code for the sparse domain

`runner.py:116-121`:
```python
text = (variant.protocol.precision + " " + variant.id).lower()
precision = ("fp64" if "fp64" in text else
             ("fp32" if "fp32" in text else
              domain.DEFAULT_PRECISION.get(args.kernel, "fp32")))
```
Every real `spmv`/`spmm`/`sddmm` variant id uses the suffix `-f32`/`-f64`
(missing the "p" — `spmm-gpu-kernel-f32`, `sddmm-csr-kernel-f32`), and none of
those variants set a `protocol.precision` field either. Verified directly
against the loaded specs:
```
spmm/spmm-gpu-kernel-f32:   'fp64' in text -> False   'fp32' in text -> False
sddmm/sddmm-csr-kernel-f32: 'fp64' in text -> False   'fp32' in text -> False
```
So for the sparse domain this whole substring check finds nothing and always
falls through to `domain.DEFAULT_PRECISION` — currently harmless only because
`DEFAULT_PRECISION = {"spmv": "fp64", "spmm": "fp32", "sddmm": "fp32"}`
happens to already be the right answer for the *only* variants that exist
today. It is not a general mechanism that would correctly infer precision
from a future `-f64` spmm/sddmm variant id if one were added, despite reading
like one.

Separately, `spmv-csr-kernel`'s `protocol.precision` text *does* contain both
substrings (`"fp64 (primary): reference computed in fp64. fp32 (secondary):
..."`), and the check tests `"fp64" in text` first, so it always resolves to
`fp64` — meaning the spec's own "fp32 (secondary)" claim can never be
auto-selected; a caller must know to pass `--precision fp32` explicitly. Also
currently harmless (fp64 is the correct default primary), just not doing what
its shape suggests.

### F4. `clocks_locked` can never be true; env.py has no way to detect a shared/busy GPU on an otherwise-exclusive node

`env.py:64`:
```python
"clocks_locked": None,   # set by the runner if it locks clocks
```
Grepped the entire `bench/` tree (source + all 29 committed result JSONs
under `results/`): `clocks_locked` is **only ever** the literal `None`/`null`
— nothing in `runner.py` or anywhere else ever sets it to `True`. The comment
describes a feature (`the runner ... locks clocks`) that doesn't exist yet.
Consequence: `env.warn_if_unsuitable()` (env.py:74-85) unconditionally
appends "GPU clocks not locked" to every GPU run's warnings, regardless of
whether the operator actually locked clocks out-of-band (`nvidia-smi -lgc ...`
before invoking the runner) — there's no channel for that fact to reach the
record. Every GPU run is therefore structurally incapable of reporting
`conforming: true` today, which somewhat defeats the purpose of the flag
(it stops discriminating between "someone locked clocks and ran cleanly" and
"nobody did anything").

Separately: `warn_if_unsuitable` checks `node_type == "login"`, missing
SLURM env, and `clocks_locked` — but never checks whether the GPU is
*currently* contended (e.g. `nvidia-smi --query-compute-apps=pid,used_memory`
showing another process, or utilization already non-zero before the run
starts) even on an exclusive-looking compute-node allocation with a clean
SLURM env. A misconfigured job, an MPS-shared GPU, or a stray leftover
process would go completely undetected.

### F5. `spec.py`'s "same as" cross-reference resolution is a single pass, so it's silently order-dependent for multi-hop chains

`_resolve_cross_references`/`_resolve_protocol_references` (spec.py:170-215)
each do one loop over `self._variants.values()`, resolving `"same tolerance/
protocol as <target>"` by copying whatever the target already has *at that
point in the single pass*. Built a synthetic 3-hop chain
`A ("same as B") -> B ("same as C") -> C (literal 1e-7)` and loaded it two
ways:
```
YAML order A, B, C:
  variant-c (literal):     tolerance=1e-07
  variant-b (refs c):      tolerance=1e-07   (resolved: C was already literal)
  variant-a (refs b, c transitively): tolerance=None   prov="unresolved reference to 'variant-b'"

YAML order C, B, A (same logical spec, different declaration order):
  variant-a: tolerance=1e-07   prov='inherited from variant-b'   (fully resolved)
```
Identical spec content, different variant declaration order in the YAML,
different outcome — `A` stays permanently unresolved in one ordering and
fully resolves in the other, purely because Python dict iteration follows
insertion order and the loop never revisits an already-processed variant once
its target later becomes resolved. This fails *safe* (an unresolved tolerance
makes the gate `None` → the run correctly reports as uncertifiable, per
`check_correctness`'s `"no tolerance parsed from spec; gate cannot pass"`
note) rather than silently wrong, and it's a documented limitation ("one hop,
no cycles" in the docstring) rather than a total surprise — but "one hop" is
undersold: even a 2-hop chain's success depends on which variant happens to
be declared first in the YAML, which isn't obviously connected to "hops" from
a spec author's perspective. Grepped current `benchspecs/*/spec.yaml` for
"same ... as" chains — didn't find one that's actually multi-hop today
(single-hop references dominate, e.g. `spmm-gpu-e2e-preproc-f32 -> spmm-gpu-
kernel-f32` directly), so this is presently latent rather than actively
biting a real spec.

---

## NIT

- **`metrics.py`'s `flops()`/`bytes_moved()` are dead code.** Grepped for
  callers: only `metrics.throughput`/`metrics.bandwidth` are actually called
  by `harness.py`; the flop/byte *counting* rules live independently inside
  each domain module (e.g. `sparse.py`'s `_cost_spmv/_cost_spmm/_cost_sddmm`,
  which duplicate — currently correctly — the same formulas). The module
  docstring says "one place, so every implementation is scored the same way,"
  but the actual formulas are re-implemented per domain; `metrics.py`'s
  copies aren't that "one place" for anything currently registered.
- **SDDMM's compulsory-byte formula doesn't charge for reading `S`'s own
  stored values**, only for the pattern (`nnz*ib`) and the output (`nnz*vb`)
  (`metrics.py:41-43`, `sparse.py`'s `_cost_sddmm`), even though
  `kernels.cu`'s `sddmm_csr_warp_kernel` reads `data[k]` once per nonzero into
  a *separate* output buffer `P[k]`. Acceptable under the explicitly-labeled
  "lower bound" contract (`workload.py:35-37`), but worth confirming it's
  deliberate rather than an oversight, since it's a same-order-of-magnitude
  term being dropped, not a rounding-level one, for small `K`.
- **`_stats([])` crashes (`IndexError`/`StatisticsError`)** if `times_ms` is
  empty, reachable if `reps` is ever exactly `0` — `run_variant`'s guard only
  checks `if reps is None:` (harness.py:218), not `reps <= 0`. No current spec
  literally has `reps: 0`, but `--reps 0` on the CLI, or a future spec whose
  prose happens to put a `0` first (e.g. "0 warmup, N reps"-style phrasing
  feeding the wrong field), would crash rather than fail with a clear message.
- **`int32` indptr/indices, both in `kernels.cu`'s kernel signatures and in
  `gpu_cuda.py`'s `torch.as_tensor(A.indptr, dtype=torch.int32)`.** Confirmed
  the cast silently wraps on overflow rather than raising:
  ```
  int64 source : [0, 2000000000, 2147483647, 2147483648, 3000000000]
  int32 result : [0, 2000000000, 2147483647, -2147483648, -1294967296]
  ```
  Not reachable by this project's actual matrix set (`spmv`/`spmm` specs cap
  `nnz <= 2e8`, well under `2^31-1`), and any real corruption here would very
  likely either crash the kernel (illegal device memory access from a
  negative/huge offset) or fail the pre-timing correctness gate outright
  rather than silently publish a wrong number — but there's no assertion
  anywhere documenting the limit.
- **`csrc/kernels.cu`'s SDDMM launch uses `gridDim.x = rows` directly**
  (`sddmm_csr_warp_kernel<<<rows, threads, ...>>>`, vs spmv/spmm's `blocks =
  ceil(rows/warps_per_block)`), bound by CUDA's max grid-dim-x
  (`2^31-1` on the target sm_80). Not reachable by any real SuiteSparse row
  count. Reviewed the warp-reduction/partial-warp logic in all three kernels
  for races — every `__shfl_down_sync` is reached by a full, uniformly-taken
  warp in all three kernels (the `if (warp_id >= rows) return`/`if (row >=
  rows) return` guards are uniform across every lane of the same warp/block
  by construction), so no partial-warp reduction hazard found.
- **`check_correctness`'s `"exact"` mode casts through `float64` before
  comparing** (`harness.py:151-158`: `out = np.asarray(out, dtype=np.float64)`
  happens unconditionally, before the `mode == "exact"` branch). For
  integer-valued "exact" gates (counts, labels, bit-exact roundtrips) with
  values beyond `2^53`, this could silently lose precision and flip a
  pass/fail. Not practically reachable given this project's domain sizes.
- **`report.py`'s dim/grouping key uses `params.get("N") or params.get("K") or
  1/""`** (report.py:46, 77) — a legitimate `N=0`/`K=0` would be treated as
  "missing" by the falsy-`or` chain. No current kernel spec sweeps N/K=0, so
  latent only.

---

## Verification scripts

All reproducers above are self-contained Python scripts run against the real
`kernelbench` package (`sys.path.insert(0, ".../bench")`) with
`/pscratch/sd/c/cunyang/gnn/plexus_env/bin/python`, CPU only — no CUDA
kernels were launched. They live in the sandbox scratchpad used for this
audit and are not part of the repo; the output captured above is the full
verification trail. Notably `torch` (CPU-mode, `import torch` without any
`.cuda()` call) was used only to check `torch.as_tensor(..., dtype=torch.
int32)`'s overflow semantics (finding under NIT), not to run any kernel.
