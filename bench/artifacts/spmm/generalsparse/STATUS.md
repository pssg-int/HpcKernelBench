# generalsparse (GeneralSparse) — spmm

**Status: BUILT+GATED on spmm-binary-adjacency-kernel (its actual regime —
fp16, unweighted 0/1-adjacency SpMM) since 2026-09-06; general weighted
variants (`spmm-tensorcore-fp16`): gate FAILS (values hardcoded to 1.0, see
finding below)**

## 2026-09-06: re-gated under spmm-binary-adjacency-kernel

`benchspecs/spmm/spec.yaml` gained a dedicated `spmm-binary-adjacency-kernel`
variant (see `../README.md`'s "Pattern-only SpMM variant" section) whose
harness-level `kernelbench.domains.sparse.variant_transform` hook forces the
sparse operand's stored values to 1.0 for EVERY implementation, including
the fp64 CSR reference, before either sees it -- the exact regime the
IMPLEMENTATION FINDING below shows this artifact's own `.mtx` reader
collapses every input to (independent of whether the *harness* additionally
binarizes it). No adapter code change was needed.

```
LD_PRELOAD=/usr/lib64/libstdc++.so.6 $PY -m kernelbench.runner --kernel spmm \
    --variant spmm-binary-adjacency-kernel --impl generalsparse-warpbitmap \
    --precision fp16 --smoke --dims 128 --warmup 1 --reps 3
```
Result: **3/3 valid** (3 synthetic smoke matrices, N=128 -- dims restricted
to one value here since each (matrix, N) pair costs its own `gs_emit` +
`nvcc` compile, ~15-20s), `max_scaled_err` 1.71e-03 – 1.92e-03, all
`<= tol 0.01` -- the SAME smoke matrices that were 0/9 INVALID under
`spmm-tensorcore-fp16` (err 2.96–5.67) now pass cleanly.

```
LD_PRELOAD=/usr/lib64/libstdc++.so.6 $PY -m kernelbench.runner --kernel spmm \
    --variant spmm-binary-adjacency-kernel --impl generalsparse-warpbitmap \
    --precision fp16 --matrices cant --dims 128 --warmup 1 --reps 3
```
Result: **1/1 valid**, `max_scaled_err = 2.20e-03 <= tol 0.01` (~230s wall,
almost entirely the `gs_emit`+`nvcc` compile). This is the FIRST correctness
PASS demonstrated for this adapter on a real (non-synthetic) matrix -- under
`spmm-tensorcore-fp16` this SAME matrix (`cant`, real-valued) gave
`max_scaled_err = 2.114`, a FAIL, because the reference used `cant`'s real
weights while the kernel structurally could not; binarizing the reference
alongside the kernel closes that gap. Fills in the "Not done" item below
("A matrix demonstrating a correctness PASS ... was not found").

```
LD_PRELOAD=/usr/lib64/libstdc++.so.6 $PY -m kernelbench.runner --kernel spmm \
    --variant spmm-binary-adjacency-kernel --impl generalsparse-warpbitmap \
    --precision fp16 --matrices cora --dims 128 --warmup 1 --reps 3
```
Result: **UNSUPPORTED** (unchanged from the `spmm-tensorcore-fp16` result
below) -- `warp_bitmap`'s own `is_valid_according_to_metadata()` padding-rate
assertion rejects `cora`'s row-length distribution regardless of variant;
this is a structural limitation of using one fixed operator composition
(see module docstring), orthogonal to the values-hardcoding finding that
motivates this variant, and is NOT fixed by binarization.

Reduced-protocol, non-conforming numbers only (warmup=1, reps=3, shared
login-node GPU) -- no timing sweep. The `spmm-tensorcore-fp16` gate result
immediately below is UNCHANGED and kept as the on-record general-weighted
-variant failure this variant exists to explain, per the user's 2026-09-06
decision (never delete a documented FAIL, add the correct home alongside
it).

- Paper: "GeneralSparse: Bridging the Gap in SpMM for Pruned Large Language
  Model Inference on GPUs", ATC'25. `PAPER_KEY = conf/usenix/WangGXCT25`.
- Artifact: https://github.com/Wangyaoyuu/GeneralSparse, cloned `--depth 1` at
  `824d9a8ad749631f56c1250cd3dc329784b51611` (see `source.provenance`).
- Toolchain: host compiler `/opt/cray/pe/gcc-native/14/bin/g++` (GCC 14.3.0,
  `-std=c++11`) for GeneralSparse's own C++ code generator; `nvcc` 12.9
  (`-ccbin` the same g++, `-std=c++17`, `-gencode arch=compute_80,code=sm_80`)
  for the per-matrix shared-library shim this adapter compiles at
  `prepare()`-time. Python `/pscratch/sd/c/cunyang/gnn/plexus_env/bin/python`,
  torch `2.8.0+cu128`.

## Selection rationale

Rated `core` / regime `matches` for spmm in `output/kernel_centrality.json`
(the paper's own README additionally benchmarks the SuiteSparse collection,
this track's default input regime, alongside pruned-LLM weight matrices).

## What the artifact actually is (rule 1: wrap the kernel, not the script)

GeneralSparse is not a fixed kernel binary; it is a ~330KB host-side C++ CODE
GENERATOR (`source/code_generator.cc`, `source/code_builder.cc`) that, per
input matrix, composes a chain of box-division/thread-blocking/reduction
"operators" into a bespoke `__global__ void kernel_<id>(...)` CUDA kernel and
emits it as `kernel_file.cu`. The artifact's own driver,
`source/token_test.cc::main()`, hardcodes a **benchmark SCRIPT**: it tries
~30 different such operator compositions per matrix, compiling+running each
via `nvcc`+`system()` and keeping whichever is fastest (its own README:
`token_test` -> `data_source/<id>/a.out` per candidate -> `obtain_result.py`
greps all of them for the max GFLOP/s). Running that full search on every
gate call would be running the paper's benchmark script, not its kernel.

**What was built instead**: `gs_emit.cc` (this directory, not under
`source/`) reproduces exactly ONE of those ~30 compositions —
`test_spmm_warp_bitmap`'s operator sequence, copied verbatim from
`token_test.cc:1253-1308` (fixed column-direction thread blocking + a
warp-bitmap reduction — a real, representative composition, not a
simplified stand-in for the paper's method) — and stops at
`code_generator::generate_final_program(1)`, which only **writes**
`kernel_file.cu` (repeat_num=1: exactly one, un-repeated kernel launch
statement) without compiling or running it. `gs_transform.py` (this
directory) then turns that generated file into `shim.cu`: the
`__global__ void kernel_<id>(...)` body is copied byte-for-byte (never
touched); the host setup code (reads GeneralSparse's own box-division
metadata arrays + H2D copies — its real preprocessing output, rule 2) is
lifted into `extern "C" void kb_init()`; the one kernel-launch statement is
lifted into `extern "C" void kb_run(void* c_ptr, void* b_ptr)`, rebound to
harness-supplied C/B device pointers instead of GeneralSparse's own generated
host arrays. See both modules' docstrings for the exact, empirically-verified
text-split rule (confirmed against a real generated file while building
this, not assumed from reading the generator source alone).

`build.sh` builds `source/token_test`'s `.o` files (needed as link
dependencies) and links `gs_emit` against all of them except `token_test.o`
(which carries the 30-strategy `main()`). `adapter.py::prepare()` calls
`gs_emit <mtx> <N>` as a subprocess (timed as preprocessing), text-transforms
the result, compiles `shim.cu` via `nvcc` (also timed as preprocessing), and
`ctypes.CDLL`-loads it — the same established pattern as
`rode`/`insum`/`mp-spmm` (torch CUDA tensors' `.data_ptr()` passed straight
into the wrapped kernel launch).

## IMPLEMENTATION FINDING: the artifact's own .mtx reader discards real
## nonzero VALUES (independently confirmed in source, not inferred)

`source/struct.cc::get_matrix_index_and_val_from_file` (both the
`graph_flag` branch and the plain branch, ~lines 140 and ~178) parses
`val = atof(sv[2].c_str())` from each input line but **never uses it**:
every stored value is instead hardcoded via `float_val_vec.push_back(1)` /
`double_val_vec.push_back(1)`. This happens at host-side matrix ingestion,
before any GPU code exists — independent of the `HALF` precision config.
Effect: the sparse operand GeneralSparse's own kernel entry point ever
computes with is the **binary pattern** of whatever matrix is fed in, never
its real weights. This is the same CLASS of finding already documented for
`dtcspmm`/`flashsparse` in `../README.md` ("Cross-artifact finding"), though
a different root cause here (dead code in the file reader, not deliberate
GNN-adjacency-only preprocessing). A trivial one-line-per-branch fix exists
(`push_back(val)`) but was deliberately **not** applied, for the same reason
dtcspmm/flashsparse's discard was left alone: consistent, disclosed treatment
of the same class of artifact behavior across this track's baselines rather
than silently patching one of three.

The DENSE operand (`x_arr`/B) is a separate, non-data-driven matter: the
generator hardcodes it to all-ones too (`x_arr[i] = 1;` in
`generate_matrix_format_read_code`), but this adapter's `kb_run` never emits
or uses that generated code at all — B and C are real harness-owned CUDA
tensors (`np.random.default_rng` matching `cpu_ref.reference_spmm`'s
`_dense_operand` exactly, same convention as rode/insum/mp-spmm). So this
adapter DOES exercise the real kernel arithmetic against a genuine
`U(-1,1)` dense operand; only the sparse side is forced to binary.

**Consequence for the gate**: PASS is expected only on matrices whose real
values already equal 1 (unweighted/binary graphs); FAIL/mismatch is expected
on any matrix with real nonzero weights. Confirmed empirically below — this
is a real result, the gate is not loosened (rule 4).

## IMPLEMENTATION FINDING: a genuine double-accumulation bug in the
## artifact's own generated correctness check (found while building the
## shim, independent of anything this adapter does)

Every generated `kernel_file.cu` launches `kernel_<id>` TWICE: an unused
"warm-up" launch right after setup, then `cudaMemcpy(y_arr, d_y_arr, ...,
D2H)` of THAT warm-up result into the host buffer later compared against the
reference, THEN the timing preamble, THEN the "timed" launch (again
`atomicAdd`-ing into the SAME never-re-zeroed `d_y_arr`), then `gettimeofday`
stops — with **no further D2H copy**. So `check_result()` verifies the
warm-up run's output, not the timed run's (whose device buffer, after two
un-zeroed accumulations, is never inspected again). This adapter's `run()`
zeroes C before every call specifically to avoid inheriting this bug;
verified empirically that repeated calls with a zeroed C return
bit-identical output (see below).

## Gate verification (login node, functional check + real matrices)

```
PY=/pscratch/sd/c/cunyang/gnn/plexus_env/bin/python
LD_PRELOAD=/usr/lib64/libstdc++.so.6 $PY -m kernelbench.runner --kernel spmm \
    --variant spmm-tensorcore-fp16 --impl generalsparse-warpbitmap \
    --precision fp16 --smoke --warmup 1 --reps 3
```
Result: **0/9 valid** on the synthetic smoke set (signed `U(-1,1)` values,
not binary) — `max_scaled_err` 2.96 – 5.67, matching the values-discarded
finding above (same order of magnitude as dtcspmm/flashsparse's own smoke
failure, independent confirmation of the same class of bug).

```
LD_PRELOAD=/usr/lib64/libstdc++.so.6 $PY -m kernelbench.runner --kernel spmm \
    --variant spmm-tensorcore-fp16 --impl generalsparse-warpbitmap \
    --precision fp16 --matrices cant --dims 128 --warmup 1 --reps 3
```
Result: **0/1 valid** — `cant` (real-valued SuiteSparse matrix, 62451x62451,
nnz=4,007,383): kernel launches and runs cleanly (no crash), `max_scaled_err
= 2.114` — again the values-discarded finding, not a crash/build failure.

```
LD_PRELOAD=/usr/lib64/libstdc++.so.6 $PY -m kernelbench.runner --kernel spmm \
    --variant spmm-tensorcore-fp16 --impl generalsparse-warpbitmap \
    --precision fp16 --matrices cora,citeseer,pubmed,web-BerkStan --dims 128 \
    --warmup 1 --reps 3
```
Result: all four **UNSUPPORTED** — GeneralSparse's own
`fixed_interval_col_direction_thread_blocking_operator::is_valid_according_to_metadata()`
assertion fires (observed message: `"current padding rate is 4.00047, higher
than 4"` for `cora`) inside `gs_emit`, i.e. this ONE fixed operator
composition's own internal validity check rejects these matrices' row-length
distributions. GeneralSparse's own benchmark script would fall back to one
of the other ~29 compositions here (its adaptive-search contribution); this
adapter deliberately does not run that search (rule 1). Reported as
`NotImplementedError` (rule 8), not a crash. This is a genuine limitation of
restricting the integration to one fixed strategy, disclosed rather than
worked around, and it is notable that it is exactly the GNN-benchmark-style,
skewed-degree graphs (the recommended_subset's Planetoid/webgraph entries)
that trip it, while the "regular" scientific SuiteSparse matrix (`cant`)
does not — some empirical support for the paper's own framing that its
search exists precisely to adapt to the graphs' row-length skew.

Reduced-protocol numbers only (warmup=1, reps=3, shared login-node GPU;
`conforming: False` in every run above), per ARTIFACT_GUIDE.md rule 5 — no
timing sweep was run.

## Not done

- The full ~30-strategy autotuning search (this adapter fixes one
  composition; see above) — GeneralSparse's own top-level contribution, out
  of scope for a login-node build+gate integration.
- A matrix demonstrating a correctness PASS (i.e. one whose real values
  already equal 1 AND survives `warp_bitmap`'s padding-rate bound) was not
  found among the matrices tried (cora/citeseer/pubmed/web-BerkStan all hit
  the padding-rate assertion instead); not pursued further given the
  session's time budget. `cant` demonstrates the kernel launches and the
  FAIL path is real and disclosed either way.
- No sweep across the full `recommended_subset` or N in {256, 512}; not
  exercised (`grid_block_operator`'s dims were only verified at N=128).
- `PADDING_RATE_UP_BOUND=4` (global_config.json) was left at its shipped
  default rather than raised to make `warp_bitmap` accept more matrices —
  changing the artifact's own tuning knob to force a pass would be closer to
  loosening the gate than reporting an honest limitation (rule 4's spirit).
- `token_test`'s own binary is built (a link dependency of `gs_emit`) but
  never executed by this integration — see rule-1 discussion above.

## Files (rule 3: files we write live outside source/)

- `gs_emit.cc` / `gs_emit` (built binary) — our driver, reproduces one fixed
  GeneralSparse operator composition and stops at code emission.
- `gs_transform.py` — kernel_file.cu -> shim.cu text surgery.
- `adapter.py`, `build.sh`, `STATUS.md`, `source.provenance`, `source.patch`
  (config path fix only — see below).
- `work/` — per-matrix `.mtx` inputs, generated `shim_<id>.cu`/`.so`
  (scratch, not versioned).

`source.patch` records `global_config.json`'s `ROOT_PATH_STR`/
`spmv_header_file` fix (this clone's absolute path instead of the original
author's `/home/wangyaoyu/GeneralSparse`) — a build-system path fix
(rule 3), applied idempotently by `build.sh` via a small inline Python
script rather than a static patch (the correct path is clone-location-
specific). Note: `global_config.json`'s `DENSE_MATRIX_SIZE`/`VECTOR_WIDTH`
fields also appear changed in `source.patch` — this is incidental, not part
of the intentional patch: `source/config.cc::set_config()` writes its
argument straight back to this SAME on-disk JSON file on every call (by
design, confirmed by reading config.cc), so those two fields simply reflect
whatever `N` a prior `gs_emit` invocation last ran with. Harmless (each
`gs_emit` call sets `DENSE_MATRIX_SIZE` from its own CLI argument before
generating), but worth recording so a future diff of this file isn't
mistaken for an additional intentional patch.

## Reproduction on zaratan (2026-09-09)

- Machine: UMD zaratan, A100 MIG 1g.5gb slice (sm_80), login-node build.
- Toolchain: nvcc 12.8.93 (conda-forge CUDA 12.8), g++ 13.4.0 (conda-forge,
  `$CXX`/`$KB_CXX`), torch 2.8.0+cu128, Python 3.12.14. `token_test`/`gs_emit`
  build with the same g++ throughout (no gcc12-host requirement here).
- Build: OK, no errors (only pre-existing `-Wreturn-type` warnings in
  `source/kernel_generator.h`). Build-system changes: none beyond what
  `build.sh` already does (the `ROOT_PATH_STR`/`spmv_header_file` config
  rewrite ran idempotently).
- Adapter fix: `adapter.py`'s `_HOST_COMPILER` fell back straight to the
  Perlmutter path `/opt/cray/pe/gcc-native/14/bin/g++` (via an unset
  `HOST_COMPILER` env var) when neither `HOST_COMPILER` nor a machine-neutral
  knob was set — that path doesn't exist here, so `prepare()`'s `nvcc -ccbin`
  call for the per-matrix shim compile would have failed. Changed the
  fallback chain to `HOST_COMPILER` env, then `CXX` env (already exported
  machine-correctly by `bench/env.sh`/`toolchain.sh` on every machine), then
  the old Perlmutter path as the last resort (rule 8).
- Gotcha (not a code change, a gate-invocation fix): STATUS.md's recorded
  commands above prefix every runner call with
  `LD_PRELOAD=/usr/lib64/libstdc++.so.6` — a Perlmutter-specific override
  (that machine's venv python otherwise drags in an old libstdc++). On
  zaratan `gs_emit` links against conda-forge gcc 13's libstdc++
  (`GLIBCXX_3.4.29`+); forcing the *system* `libstdc++.so.6` instead broke
  every `gs_emit` invocation (`version 'GLIBCXX_3.4.29' not found`), which
  surfaced as every workload going UNSUPPORTED (including ones that should
  pass) rather than as a crash. `bench/env.sh` already exports the correct
  `LD_PRELOAD` (`KB_LD_PRELOAD`, conda gcc 13's libstdc++) for this machine,
  so the gate commands below omit the explicit override and let it stand.
- Data: `cora`/`citeseer`/`pubmed` (Planetoid) are not in the SuiteSparse
  index; pre-fetched on the login node via
  `kernelbench.domains.sparse.load_workload("citeseer"/"pubmed")` before the
  gate (`cora` and `cant`/`web-BerkStan` were already cached from an earlier
  step) — the GPU compute node has no outbound internet and a first-touch
  Planetoid download inside a gate run times out (`ConnectionTimeoutError`
  to `github.com/kimiyoung/planetoid`), observed on a first attempt before
  the pre-fetch.
- Gate (`spmm-binary-adjacency-kernel`, fp16, N=128, warmup=1, reps=3):
  smoke-uniform+bin PASS (err 1.79e-03 <= 0.01), smoke-banded+bin PASS
  (err 1.71e-03 <= 0.01), smoke-powerlaw+bin PASS (err 1.92e-03 <= 0.01) —
  3/3 valid; `cant` PASS (err 2.20e-03 <= 0.01) — 1/1 valid; `cora`
  UNSUPPORTED (padding-rate assertion) — 0/0 valid (1 unsupported).
- Gate (`spmm-tensorcore-fp16`, fp16, warmup=1, reps=3, kept as the
  on-record general-weighted-variant failure): smoke set (dims 128/256/512
  x uniform/banded/powerlaw) all INVALID, `max_scaled_err` 3.107e+00 -
  5.672e+00 vs tol 0.01 — 0/9 valid; `cant` (N=128) INVALID, err
  2.114e+00 — 0/1 valid; `cora`/`citeseer`/`pubmed`/`web-BerkStan` (N=128)
  all UNSUPPORTED (padding-rate assertion) — 0/0 valid (4 unsupported).
- Deviation from the recorded ruling: none. Every PASS/FAIL/UNSUPPORTED
  outcome and every error magnitude reproduces the ruling above (the
  `spmm-tensorcore-fp16` smoke error range, 3.107e+00-5.672e+00 here vs
  2.96e+00-5.67e+00 recorded, differs slightly — plausible RNG-stream /
  driver-version noise on already-badly-wrong outputs — but both are firmly
  INVALID for the same disclosed values-discarded reason, so this does not
  change the ruling).
- Verdict here: BUILT+GATED on `spmm-binary-adjacency-kernel` (PASS);
  `spmm-tensorcore-fp16` gate FAILS as recorded — same as the recorded
  ruling.
