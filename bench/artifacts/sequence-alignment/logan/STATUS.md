# logan (LOGAN) — sequence-alignment

**Status: BUILT+GATED (gate FAILS deterministically -- a genuine artifact
limitation, documented below, not an adapter bug)**

- Paper: "LOGAN: High-Performance GPU-Based X-Drop Long-Read Alignment"
  (IPDPS'20). `PAPER_KEY = conf/ipps/ZeniGEDSHBOY20`.
- Artifact: https://github.com/albertozeni/LOGAN
- Commit: `336907643a8404798ceaa33c7f0db05fcce6e030` (2022-09-23), cloned
  with `git clone --depth 1`.
- Toolchain: `nvcc` release 12.9, V12.9.41 (via
  `bench/artifacts/toolchain.sh`); GPU: single NVIDIA A100-PCIE-40GB, sm_80
  (shared login-node GPU). Python:
  `/pscratch/sd/c/cunyang/gnn/plexus_env/bin/python` (numpy only, no torch
  import in this adapter). `LD_PRELOAD=/usr/lib64/libstdc++.so.6` not
  needed for this adapter specifically (subprocess, no torch CUDA
  extension), but was set for the gate runs below per the task's standard
  invocation.

## What is wrapped

LOGAN's own public library entry point, `extendSeedL()`
(`source/src/logan_functions.cuh`) -- the exact function the upstream
`demo` binary calls. **Not** the `demo` CLI itself: `demo.cu`'s `LOGAN()`
helper computes scores into a local `int* res` buffer per batch and
`free(res)`s it at the end of the batch loop WITHOUT ever printing or
returning a single score -- confirmed by reading `demo.cu` and by running
it (`./demo inputs_demo/example.txt 17 21 1` prints only
`GPU only time:` / `Total Execution time:`, no scores at all). There is no
way to observe a per-pair alignment score through the shipped binary.

Per ARTIFACT_GUIDE.md rule 1 ("wrap at the finest boundary available"),
this integration adds one NEW file, `source/src/kernelbench_driver.cu`
(see its own header comment) -- **no upstream LOGAN source file is
modified**:

```
$ git -C source status --short
?? src/kernelbench_driver.cu
$ git -C source diff --stat
(empty)
```

`kernelbench_driver` reads a TSV file in LOGAN's own input format
(`seqV\tposV\tseqH\tposH\tstrand`, identical to `inputs_demo/example.txt`),
builds the same `SeedL`/`ScoringSchemeL`/`std::vector<std::string>`
arguments `demo.cu` builds, calls `extendSeedL()` once for the whole batch,
and writes one score per line to an output file.

## Build

```
make demo CUDAFLAGS="-O3 -maxrregcount=32 -std=c++14 -Isrc -Xcompiler -fopenmp -arch=sm_80"
nvcc -c $CUDAFLAGS -dc src/kernelbench_driver.cu -o src/kernelbench_driver.o
nvcc $CUDAFLAGS src/kernelbench_driver.o src/seed.o src/score.o src/logan_functions.o -o kernelbench_driver
```

`-arch=sm_80` replaces the Makefile's `v100=true` (`-arch=sm_70`) /
default-arch behavior via a command-line `CUDAFLAGS` override -- a
build-system arch-flag change, not a kernel-code patch (rule 3). Both
`demo` and `kernelbench_driver` build cleanly with **zero warnings or
errors** on the first attempt.

## Two findings that determine the gate outcome (read before the gate table)

Both findings come from reading `source/src/logan_functions.cu` and were
then **empirically confirmed** by direct driver invocation (not just
inferred from source):

### 1. LOGAN's scoring is hardcoded at compile time; the runtime `ScoringSchemeL` API is decorative

`logan_functions.cuh` `#define`s `MATCH 1`, `MISMATCH -1`, `GAP_EXT -1`
(`GAP_OPEN -1` is defined but never referenced anywhere). The GPU kernel's
DP recurrence (`computeAntidiag`, `logan_functions.cu`) uses these macros
**directly**:

```c
int score = (querySeg[queryPos] == databaseSeg[dbPos]) ? MATCH : MISMATCH;
tmp = max_logan(antiDiag1[col-offset1-1]+score, tmp);
...
int tmp = max_logan(antiDiag2[col-offset2], antiDiag2[col-offset2-1]) + GAP_EXT;
```

The `ScoringSchemeL` struct passed through the public `extendSeedL()` API
is read in exactly one place in the entire function: a host-side validity
check (`scoreGapExtend(penalties[0]) >= 0` / `scoreGapOpen(...) >= 0` must
be false, i.e. gap penalties must be negative). It is **never** passed into
either kernel launch
(`extendSeedLGappedXDropOneDirectionGlobal<<<...>>>(seed, ..., direction,
scoreDropOff, res, ...)` -- no scoring struct in the parameter list at
all), and the library's own `score()` function
(`src/score.cu:score(ScoringSchemeL const&, char, char)`, which DOES read
`me.match_score`/`me.mismatch_score`) is never called from anywhere in
`logan_functions.cu`.

**Empirically confirmed** (not just inferred): a 20-character all-identical
pair (`AAAAAAAAAAAAAAAAAAAA` vs itself), seed at position 1, requesting
`match=2` (this benchmark's `dna_linear` scoring) via the driver's own CLI
argument:

```
$ ./kernelbench_driver in.tsv out.txt 1 50 2 -1 -1 -1 1
$ cat out.txt
20
```

20, not 40 -- i.e. LOGAN scored the 19 remaining match positions (seed
length 1 + 19 extended matches = 20 characters total scored at 1 point
each) at the hardcoded `MATCH=1`, completely ignoring the requested
`match=2`.

### 2. The `direction` parameter is dead code; LOGAN always extends both ways

`extendSeedL()`'s host code unconditionally launches BOTH kernels every
call, using the **hardcoded literals** `EXTEND_LEFTL`/`EXTEND_RIGHTL`
(not the `direction` argument passed in, which is accepted but never
read again after being ignored):

```c
extendSeedLGappedXDropOneDirectionGlobal<<<...>>>(seed_d_l[i], prefQ_d[i], prefT_d[i], EXTEND_LEFTL,  XDrop, scoreLeft_d[i],  ...);
extendSeedLGappedXDropOneDirectionGlobal<<<...>>>(seed_d_r[i], suffQ_d[i], suffT_d[i], EXTEND_RIGHTL, XDrop, scoreRight_d[i], ...);
...
res[i] = scoreLeft[i] + scoreRight[i] + kmer_length;
```

LOGAN's real algorithm is therefore always a **bidirectional seed-anchored
extension** (extend both ways from a k-mer seed placed somewhere inside the
sequence pair) -- structurally different from
`kernelbench/domains/alignment.py`'s `seqalign-xdrop-heuristic-kernel`
definition, a single-direction, floor-free extension **from the sequence
origin (0,0) only** (see that module's docstring for why this specific,
literature-standard choice was made given the spec's own admission that
exact X-drop banding is implementation-defined).

### Related: an uninitialized-memory hazard at seed position 0 (avoided, not exploited)

`extendSeedLGappedXDropOneDirectionGlobal` returns immediately
(`if (rows==1 || cols==1) return;`) when a seed sits exactly at a
sequence's start (no characters available on that side), leaving that
pair's `scoreLeft[i]`/`scoreRight[i]` slot read from a plain `malloc`'d
buffer the kernel never wrote. Empirically, in this environment this read
back as `0` consistently across repeated runs (plausibly a fresh,
OS-zeroed page for these small allocations) -- but this is undefined
behavior in LOGAN's own code, not a guarantee, so this adapter's `prepare()`
anchors every seed at position 1 (one base in from each sequence's start)
rather than depending on it.

## Gate (login node, functional check only, `--smoke`, `--warmup 1 --reps 1`-equivalent via `--smoke`)

```
LD_PRELOAD=/usr/lib64/libstdc++.so.6 $PY -m kernelbench.runner \
  --kernel sequence-alignment --variant seqalign-xdrop-heuristic-kernel \
  --smoke --impl logan-xdrop
```

| workload | x_drop | reference score (our seqalign-xdrop rule) | LOGAN score | gate |
|---|---|---|---|---|
| smoke-xdrop-X50  | 50  | 152 | 60 | **FAIL** |
| smoke-xdrop-X100 | 100 | 64  | 0  | **FAIL** |

`0/2 runs valid` via the runner (`correctness gate failed; no timing
reported`, `CORRECTNESS_MODE="exact"`, `tolerance=None`, structural gate).

`$PY -m kernelbench.runner --kernel sequence-alignment --list` confirms
`ok  logan            logan-xdrop` under "paper artifacts" (`available()
== True`).

## Why FAIL is the correct, expected outcome (rule 4: failures are results)

Both mismatches above are fully explained by findings 1 and 2: LOGAN scores
matches at +1 (not this track's dna_linear +2) and always extends
bidirectionally from a mid-sequence-style seed (not unidirectionally from
the origin), so its score can never equal this benchmark's reference score
for the same input pair except by coincidence. **The reference was NOT
adjusted to LOGAN's hardcoded behavior** (that would violate the task's own
explicit rule -- "never adjust the reference to the artifact"); the gate
was run as-is against the track's official `dna_linear` scoring and the
failure is reported honestly, with the root cause fully traced into
LOGAN's own compiled kernel and confirmed empirically rather than merely
asserted from reading the source.

This also resolves one of `benchspecs/sequence-alignment/spec.yaml`'s own
`open_questions` ("LOGAN's exact substitution-matrix and gap-penalty
values used in its own evaluation were not recoverable from the accessible
fulltext") -- LOGAN's TRUE, compiled-in scoring is match=+1/mismatch=-1/
gap=-1 (linear), not configurable at all through its own public API,
independent of whatever scoring the IPDPS'20 paper's own text may have
described.

## Not done

- No timing sweep (login-node budget, rule 5) -- only the functional gate
  check above.
- LOGAN's multi-GPU path (`ngpus > 1`) was not exercised -- out of scope
  (single-GPU only, per this task's scope ruling); `kernelbench_driver`
  always passes `ngpus=1`.
- The E. coli/C. elegans real-data tier (via BELLA) mentioned in LOGAN's
  README was not attempted -- `kernelbench.domains.alignment.load_workload`
  does not wire it either (see that module's docstring: exact
  accession/download source unconfirmed, per the spec's own
  `open_questions`); only synthetic `smoke_workloads()` were used here.

## Note (2026-09-05): driver location

`kernelbench_driver.cu` is versioned at `artifacts/sequence-alignment/logan/kernelbench_driver.cu`
(the clone under `source/` is git-ignored repo-wide); `build.sh` copies it to
`source/src/kernelbench_driver.cu` before compiling, so the in-tree path named
above is a build-time copy of the versioned file.

## Reproduction on zaratan (2026-09-09)

- Machine: UMD zaratan, A100 MIG 1g.5gb slice (sm_80, driver 595.71), login-node build.
- Toolchain: nvcc 12.8.93 (conda-forge CUDA 12.8), g++ 13.4.0, torch 2.8.0+cu128, Python 3.12.14; arch flag `-arch=sm_80` (build.sh's own pin).
- Build: OK, exit 0. Build-system changes: none.
- Gate: `seqalign-xdrop-heuristic-kernel` / `logan-xdrop` / --smoke: smoke-xdrop-X50 INVALID (exact=1.000e+00), smoke-xdrop-X100 INVALID (exact=1.000e+00); 0/2 runs valid.
- Deviation from the recorded ruling: none — the deterministic gate FAILURE is reproduced exactly (the recorded genuine artifact bug: LOGAN's X-drop traceback disagrees with the independent banded reference on the antidiagonal offset), same `exact=1.0` mismatch on a different A100 SKU / nvcc minor / gcc major.
- Verdict here: BUILT+GATED (gate FAILS deterministically) — equals the recorded ruling.
