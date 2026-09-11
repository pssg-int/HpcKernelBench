# TurboFFT — STATUS

**Outcome: BUILT+GATED (second pass) — ran the artifact's own code generator
(the first pass evidently never did); all 3 of the 8 fft-domain smoke
workloads TurboFFT's kernel set structurally supports (1D, power-of-two,
C2C) PASS the correctness gate.**

Paper: "TurboFFT: Co-Designed High-Performance and Fault-Tolerant Fast
Fourier Transform on GPUs", PPoPP 2025.
PAPER_KEY = `conf/ppopp/WuZ0HJDDCC25`.
Repo: `https://github.com/shixun404/TurboFFT`
(unchanged from first pass: `main` branch, HEAD
`918179c826f332967097fb977094ae7f54d36ab6`).

## First-pass finding (2026-08-08) — kept for history

BUILD-FAILED: `include/TurboFFT.h`'s `#if ARCH_SM == 80` block `#include`s
one generated `.cuh` kernel file per `logN` (1..25), but the repo as cloned
only shipped `logN in {7,8,9,10}`, and even those defined a stale,
differently-named/-signed kernel (`fft_10(float2*, float2*, int)`, 3 args)
than the one the header actually calls
(`fft_radix_2<float2, N, dim_id, if_thread_ft, if_ft, if_err_injection>`,
6 args). The pass correctly identified TurboFFT as code-generated but did
not run the generator.

## Second pass: running the generator

TurboFFT's own `readme.md` and `run_A100.sh` describe the intended flow as
Environment Setup -> **Code Generation** -> Compilation -> Benchmarking ->
Plotting. `run_A100.sh` as shipped has ONLY these two lines active (every
other codegen/build/benchmark line, including the base
`fft_codegen.py` calls, is commented out):

```bash
cd TurboFFT/include/code_gen/scripts
python fft_codegen_stride.py --gpu $gpu --datatype float2
python fft_codegen_stride_output.py --gpu $gpu --datatype float2
```

These write `*_stride.cuh` files (a DIFFERENT naming suffix, used by
TurboFNO's fusion variants, not by `TurboFFT.h`'s `#include` list) — running
`run_A100.sh` verbatim therefore never touches the actual base
`fft_radix_2_logN_N_upload_0.cuh` files `TurboFFT.h` needs, which explains
why the repo ships only the 4 stale ones checked in from an old commit
("TurboFNO_dev update" per `git log`) and not the rest.

The actual generator for `TurboFFT.h`'s files is
`include/code_gen/scripts/fft_codegen.py` (present but never invoked by
`run_A100.sh`'s active lines). Ran it directly:

```bash
cd TurboFFT/include/code_gen/scripts
$PY fft_codegen.py --gpu A100 --datatype float2                          # default (no FT)
$PY fft_codegen.py --gpu A100 --datatype float2 --if_thread_ft 1
$PY fft_codegen.py --gpu A100 --datatype float2 --if_ft 1
$PY fft_codegen.py --gpu A100 --datatype float2 --if_ft 1 --if_err_injection 1
$PY fft_codegen.py --gpu A100 --datatype double2   # + the same 3 FT variants
```

(the 8 invocations `run_A100.sh`'s own comments enumerate but never run).
Each writes `../generated/{float2,double2}/fft_radix_{2}_logN_{N}_upload_{i}.cuh`
for every row of `include/param/A100/param_{float2,double2}.csv` (logN
1..30) — `save_generated_code()`'s own write-mode logic (`'w'`/truncate
when `if_thread_ft==0 and if_ft==0`, else `'a'`/append) means running the
plain call FIRST and the 3 FT variants after correctly ACCUMULATES all 4
`if_thread_ft`/`if_ft`/`if_err_injection` combinations `TurboFFT.h`'s
`ARCH_SM==80` block unconditionally instantiates
(`TurboFFT_Kernel_Entry<float2/double2, {0,0,0}/{1,0,0}/{0,1,0}/{0,1,1}, 80>`,
8 struct specializations total, `include/TurboFFT.h:520-820`) into a single
file per logN, matching what the header actually `#include`s.

**Full-header probe** (proves BOTH first-pass problems are fixed): after
running all 8 codegen invocations above, compiled the complete, UNMODIFIED
`include/TurboFFT.h` standalone (`#include "include/TurboFFT.h"` + empty
`main()`, `nvcc -DARCH_SM=80 -gencode arch=compute_80,code=sm_80 -I.
-I../Common -std=c++17 -c`):

```
real  4m8.086s
[0 errors; only #550-D/#177-D unused-variable warnings in the huge
 logN=23..30 composite kernels]
```

This is decisive: the exact header the first pass could not compile past
`TurboFFT.h:479`'s missing include now compiles clean, end to end, with
every one of the 200 (`25 logN x 4 FT-variants x 2 dtypes`) generated
kernel files it references.

## Adapter scope (this shim, not the full header)

Per ARTIFACT_GUIDE.md rule 1 ("wrap the kernel, not the paper's benchmark
script"), the shipped adapter does NOT build the full header above (that
compile is 4+ minutes and pulls in cuFFT/VkFFT comparison code from
`main.cu` that has nothing to do with TurboFFT's own kernel). Instead:

- `turbofft_shim.cu` (this dir, NOT part of the artifact) `#include`s only
  the plain (`if_thread_ft=0, if_ft=0, if_err_injection=0`), `float2`,
  `logN=1..13` (`N=2..8192`) generated files — TurboFFT's own
  SINGLE-kernel-launch param-table rows (`param_float2.csv` rows 1-13,
  `row[1]==1` "num decomposition stages"; rows 14+ decompose one big 1D
  transform into 2-3 sequential launches, out of scope for this shim, see
  `turbofft_shim.cu`'s docstring) — and exposes one `extern "C"` launcher,
  `turbofft_forward_c2c_f32`, whose grid/block/shared-mem formula and
  kernel-argument order are copied verbatim from TurboFFT's own
  `main.cu::test_turbofft<>`'s `kernel_launch_times==1` case.
- `build.sh` runs ONLY the plain-flags `fft_codegen.py` invocation (always
  `'w'`/truncate mode, so idempotent regardless of what a prior
  `--if_ft`/`--if_thread_ft` sweep run against the same `generated/` tree
  left behind — e.g. the full-header probe above), then compiles the shim
  with `nvcc -shared`.
- `adapter.py` wraps that `.so` via `ctypes` (same pattern as
  `artifacts/bfs/efg/adapter.py`), generating operands with
  `np.random.default_rng(seed)` (matching `reference_fft`/`NumpyFFT`
  EXACTLY — NOT `torch`'s own RNG family, which this domain's existing
  `torch-fft` CUDA baseline uses and which would silently compare FFTs of
  DIFFERENT random data against the reference; noted here as an
  observation about `kernelbench/impls/gpu_cuda.py::TorchFFT`, not fixed,
  out of this task's scope).

Direction: TurboFFT's codegen'd kernel set has NO inverse-FFT path anywhere
(confirmed by grep: no `ifft_radix_2` symbol in `TurboFFT.h`/
`TurboFFT_radix_2_template.h`). `direction="inverse"` is implemented via the
standard, kernel-code-untouched identity `IFFT(x) = conj(FFT(conj(x))) / N`
— pure host-side pre/post-processing (`torch.conj_physical`) around the SAME
unmodified forward kernel.

Layout: `"in-place"` is honored the same way this domain's own
`NumpyFFT`/`ScipyFFT` already do (see their docstrings in
`kernelbench/domains/spectral.py`) — run out-of-place into a scratch buffer,
copy the result back into the operand buffer, honoring the byte-movement
convention without claiming an allocation saving the kernel doesn't
actually provide.

## Direct kernel-correctness check (before wiring into the harness)

```
N=1024 bs=4    rc=0  max_rel_err=3.164e-06   (forward)
N=512  bs=1    rc=0  max_rel_err=1.322e-06   (forward)
N=256  bs=8    rc=0  max_rel_err=1.407e-06   (forward)
N=8192 bs=1    rc=0  max_rel_err=7.213e-06   (forward)
N=2    bs=32   rc=0  max_rel_err=3.434e-08   (forward)
N=1024 bs=1    rc=0  max_rel_err=2.990e-06   (inverse, via conj trick)
```
vs. `numpy.fft.fft`/`ifft` at fp64, all well inside fp32 FFT rounding
expectations.

## Gate run

Coverage against this domain's 8-workload smoke set
(`kernelbench/domains/spectral.py::_FFT_SMOKE`): TurboFFT's kernels are
radix-2 (power-of-two N), C2C only, 1D only. 3 of 8 smoke workloads match:
`smoke-1d-pow2-c2c-fwd-oop` (N=1024,batch=4,fwd,oop),
`smoke-1d-pow2-c2c-fwd-inplace` (N=512,batch=1,fwd,in-place),
`smoke-1d-pow2-c2c-inverse-oop` (N=1024,batch=1,inverse,oop). The other 5
(mixed-radix-56, R2C forward/inverse, 2D, 3D) correctly raise
`NotImplementedError` from `prepare()` with a clear, specific message.

**The mandated command as written crashes partway through** — this is a
pre-existing `runner.py`/`spectral.py` interaction, not an adapter defect:
`smoke_workloads(kernel="fft")` is NOT variant-aware (unlike e.g.
`annsearch.py`'s opt-in `smoke_workloads(variant=...)`), so `--smoke`
always builds the FULL 8-workload list regardless of `--variant`, and
`harness.run_variant`'s `impl.prepare()` call is not wrapped in a
try/except anywhere in `runner.py`'s per-workload loop:

```
$ LD_PRELOAD=/usr/lib64/libstdc++.so.6 $PY -m kernelbench.runner \
    --kernel fft --variant fft-1d-batched-kernel --smoke --impl turbofft

  running turbofft smoke-1d-pow2-c2c-fwd-oop  ... 0.020 ms  10.45 GFLOP/s  (err 4.53e-06 <= 0.0001)  [0.3s]
  running turbofft smoke-1d-mixedradix56-c2c-fwd-oop  ...Traceback (most recent call last):
  ...
  File ".../kernelbench/harness.py", line 228, in run_variant
    handle = impl.prepare(matrix, params)
  File ".../artifacts/fft/turbofft/adapter.py", line 175, in prepare
    raise NotImplementedError(
NotImplementedError: turbofft: N=56 is outside this shim's wrapped range ...
```

Note the FIRST workload (the one the mandated command's own smoke ordering
happens to try first) already PASSES: `err=4.53e-06 <= 1e-4`. Per the exact
precedent in `artifacts/spmv/diaq/STATUS.md` (mandated `--matrices cant`
also could not run as specified; substituted a direct `harness.run_variant`
script), a substitute gate script
(`/tmp/.../scratchpad/turbofft_gate.py`, not checked into the repo) calls
`harness.run_variant` directly — the SAME code path the runner CLI uses —
restricted to the 3 workloads this artifact structurally supports:

```
smoke-1d-pow2-c2c-fwd-oop           SUPPORTED  PASS  err=4.532e-06 <= 0.0001  (0.0084 ms)
smoke-1d-mixedradix56-c2c-fwd-oop   skip (out of scope: N=56 not power-of-two)
smoke-1d-pow2-c2c-fwd-inplace       SUPPORTED  PASS  err=2.010e-06 <= 0.0001  (0.0499 ms)
smoke-1d-pow2-c2c-inverse-oop       SUPPORTED  PASS  err=4.060e-09 <= 0.0001  (0.0784 ms)
smoke-1d-r2c-fwd-oop                skip (out of scope: real_input, no R2C codegen path)
smoke-1d-r2c-inverse-oop            skip (out of scope: real_input, no R2C codegen path)
smoke-2d-c2c-fwd-oop                skip (out of scope: 2D, not a spatial-grid transform)
smoke-3d-c2c-fwd-oop                skip (out of scope: 3D, not a spatial-grid transform)

ALL SUPPORTED WORKLOADS PASS
```

(warmup/reps overridden to 1/1 for the smoke-substitute script, same
`warnings` field the harness itself always emits for overridden protocol —
not a spec-conforming timed run, matching every other `--smoke` result in
this project.)

`prepare()` rejections for the 5 out-of-scope workloads were also verified
individually to raise `NotImplementedError` (not a crash/wrong-answer) with
specific messages (`/tmp/.../scratchpad/turbofft_reject_test.py`).

This is NOT touched in `kernelbench/`: the smoke-set/variant-filtering gap
is real but fixing it is a cross-track harness design change (every domain
module would need auditing for the same interaction), outside this task's
"do not modify kernelbench/ unless a genuine harness bug is found (minimal
fix)" + "never touch other tracks" scope. Flagging it here for whoever picks
up the next narrow GPU-kernel adapter in this domain.

## Provenance

- Repo commit: `918179c826f332967097fb977094ae7f54d36ab6` (`main` branch,
  unchanged from first pass).
- Toolchain: `nvcc` 12.9 (`/opt/nvidia/hpc_sdk/Linux_x86_64/25.5/cuda/12.9`,
  via `bench/artifacts/toolchain.sh`), host compiler `g++-12`, GPU arch
  `sm_80` (A100), Python `/pscratch/sd/c/cunyang/gnn/plexus_env` (torch
  2.8.0+cu128).
- Patches: NONE to any file inside `source/` — only ran the artifact's own
  `fft_codegen.py` (a documented, intended step of its own build flow) to
  populate `include/code_gen/generated/{float2,double2}/*.cuh`, which are
  build ARTIFACTS (git-tracked in the upstream repo as a build convenience,
  not source we authored or edited) rather than source we patched.

## Verdict

`turbofft: BUILT+GATED (second pass -- ran the artifact's own
fft_codegen.py, which fft.dev/run_A100.sh describes but never actually
invokes for the base kernel files; this alone fixes BOTH problems the first
pass found: missing logN files AND the stale fft_10 name/signature mismatch,
confirmed by a clean 4m8s zero-error compile of the full unmodified
TurboFFT.h. Shipped adapter wraps only logN 1..13 float2 forward via a thin
extern "C" shim, turbofft_shim.cu; the 3 of 8 fft-domain smoke workloads it
structurally supports (1D power-of-two C2C fwd-oop/fwd-inplace/inverse-oop)
ALL PASS the correctness gate, err 4e-9..5e-6 <= 1e-4; the other 5 correctly
raise NotImplementedError. The literal mandated --smoke CLI command crashes
on the 2nd (mixed-radix, out-of-scope) workload due to a pre-existing,
not-variant-aware smoke_workloads()/uncaught-NotImplementedError interaction
in runner.py+spectral.py, not an adapter defect -- substitute gate script
demonstrates all 3 supported points pass, same precedent as
artifacts/spmv/diaq/STATUS.md)`

## Reproduction on zaratan (2026-09-09)

- Machine: UMD zaratan, A100 MIG 1g.5gb slice (sm_80, on an
  A100-SXM4-40GB physical card, driver 595.71.05, node gpu-b11-6),
  login-node build.
- Toolchain: nvcc 12.8.93 (conda-forge CUDA 12.8), g++ 12.4.0 (conda-forge
  `kb-gcc12`, via `KB_GXX12` -- `build.sh`'s `CXX="${KB_GXX12:-g++-12}"`
  default), torch 2.8.0+cu128, Python 3.12.14, `-gencode
  arch=compute_80,code=sm_80` unchanged. `build.sh` needed zero edits: its
  `PY` fallback is already read from the exported `KB_PY`/`PY` knob before
  its own Perlmutter default line evaluates (rule 8).
- Build: OK. Step 1 (`fft_codegen.py --gpu A100 --datatype float2`)
  regenerated the 13 single-specialization `logN 1..13` base `.cuh` files
  cleanly; step 2 compiled `turbofft_shim.so` (1.1 MB) with only harmless
  `#550-D`/`#177-D` unused-variable warnings in the generated kernels
  (same as the original pass). Build-system changes: none.
  Machine gotcha hit and worked around (not a build-system edit): step 1
  (`fft_codegen.py`, which imports torch/numpy) hit this login node's
  `RLIMIT_NPROC=256` session-wide cap via OpenBLAS's default 64-thread
  pool (`OpenBLAS blas_thread_init: pthread_create failed ... RLIMIT_NPROC
  256 current`), the same class of resource-contention failure as the
  documented `make -j$(nproc)` trap, just triggered by numpy/BLAS thread
  spawn instead of parallel compiler processes. `bench/env.sh` was updated
  (by a concurrent session working elsewhere in this same integration
  pass) to export `OPENBLAS_NUM_THREADS=4`/`OMP_NUM_THREADS=4`/
  `MKL_NUM_THREADS=4`/`NUMEXPR_NUM_THREADS=4` by default for exactly this
  reason; re-sourcing `bench/env.sh` after that change fixed the build
  with no further action.
- Gate: mandated command run verbatim, `--kernel fft --variant
  fft-1d-batched-kernel --smoke --impl turbofft`. Result: **3/3 runs valid
  (5 unsupported)**, `smoke-1d-pow2-c2c-fwd-oop` err 4.53e-06,
  `smoke-1d-pow2-c2c-fwd-inplace` err 2.01e-06, `smoke-1d-pow2-c2c-
  inverse-oop` err 4.06e-09, all `<= tol 1e-4` -- matching the original
  pass's numbers to the shown precision. The other 5 workloads printed a
  clean `UNSUPPORTED: ...` line and did not crash the run.
- Deviation from the recorded ruling: **the mandated `--smoke` CLI no
  longer crashes.** The original pass recorded runner.py's per-workload
  loop as uncatching `NotImplementedError` from `impl.prepare()`, crashing
  on the 2nd (mixed-radix) workload; `bench/kernelbench/runner.py` on this
  checkout now wraps that call in `except NotImplementedError as e:` (line
  ~245, prints `UNSUPPORTED: ...` and continues) -- a genuine harness fix
  made sometime between the original pass and now, not something changed
  in this session, and not specific to zaratan. Consequence: the
  substitute `harness.run_variant` script this STATUS.md's "Gate run"
  section used to work around the crash was not needed here -- the
  mandated command alone now reproduces all 3 supported-workload PASS
  results directly, with the same error magnitudes. (Attempted to run the
  substitute script anyway for a second, independent check via
  `gpu_run.sh`; it failed with "No such file or directory" because the
  compute node's `/tmp` is not shared with the login node's `/tmp` where
  the session scratchpad lives -- a copy was staged briefly under this
  artifact's own directory to work around it, then removed; not needed in
  the end since the mandated command already passed cleanly.)
- Verdict here: BUILT+GATED, and slightly BETTER than the recorded ruling
  (the mandated command now runs end-to-end without the harness-side
  workaround the original pass needed) -- same underlying kernel-
  correctness result (3 of 8 smoke workloads structurally supported, all
  PASS at fp32-FFT-appropriate error magnitudes; 5 correctly rejected).
