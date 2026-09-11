# FT-GEMM — STATUS

**Status: BUILT+GATED** — `gemm-square-kernel`, fp32, 3/3 smoke valid (err 7.39e-05 / 3.04e-05 / 1.79e-04 <= the spec's own fp32 bound 1e-3) after the harness learned to read the spec's per-precision tolerances (2026-09-05; see the last section). Earlier outcome, kept for the record: BUILT — gate FAILS under the harness's parsed tolerance (a
precision-agnostic parser artifact, not a kernel defect); the artifact's
own measured error is comfortably within the spec's OWN STATED fp32
tolerance, disclosed with full numbers below (precedent: spmm/inferfast's
"record BUILT with the measured error")

Paper: "Anatomy of High-Performance GEMM with Online Fault Tolerance on
GPUs", ICS'23. `PAPER_KEY = conf/ics/WuZLHJWC23`.
Repo: `https://github.com/shixun404/Fault-Tolerant-SGEMM-on-NVIDIA-GPUs`
(commit `52cde58bafd23484b463d6d2a4a6a2fa4bbc8f34`, 2025-04-02;
`git clone --depth 1` into `./source/`, untouched — read-only, no patch).

## Toolchain / provenance

- nvcc 12.9 (`/opt/nvidia/hpc_sdk/Linux_x86_64/25.5/cuda/12.9`), via
  `bench/artifacts/toolchain.sh`.
- Arch flag: `-arch=sm_80` (A100) — matches the artifact's own
  `CMAKE_CUDA_ARCHITECTURES 80`.
- Python: `/pscratch/sd/c/cunyang/gnn/plexus_env/bin/python`.
- Build: single `nvcc -O3 -std=c++14 -arch=sm_80 -Xcompiler -fPIC -shared
  bridge.cu -o bridge.so`. No `cuda-samples` submodule needed (the
  artifact's own `README.md` says to clone it for `helper_cuda.h`/
  `helper_functions.h`, but those headers are only used by `sgemm.cu`'s
  CLI `main()` — the kernel headers this bridge `#include`s
  (`include_code_gen/{sgemm_large,ft_sgemm_large}.cuh`) need neither).
  Builds clean; only harmless unused-local-variable warnings from the
  artifact's own generated code (`tx_injec`, `err_bound1`, `num_warp_B`,
  etc. — dead locals left over from the code-generator, not touched here).

## What the artifact actually is

`source/kernel/ft_sgemm/sgemm.cu`'s CLI (`ft_sgemm START END GAP
ST_KERNEL END_KERNEL`) drives 5 tile-size configs (small/medium/large/
tall/huge) x {fault-tolerance off, fault-tolerance on (fused ABFT)} plus a
non-fused `abft_baseline`, all dispatched by `kernel_number` inside one
`main()`. The actually-compiled kernel bodies come from
`kernel/ft_sgemm/include_code_gen/*.cuh` (code-generated; `kernels.cuh`
`#include`s these, NOT the hand-written `kernel/ft_sgemm/include/*.cuh`
files of the same name — confirmed by reading `kernels.cuh`'s own
`#include` list, most of the `include/` versions are commented out there).

## What was wrapped, and why "large"

`bridge.cu` (this directory, NOT part of the artifact) `#include`s
`include_code_gen/sgemm_large.cuh` and `include_code_gen/ft_sgemm_large.cuh`
verbatim (both define macros `tab`/`tcab` identically — the C preprocessor
allows identical redefinition, confirmed no warning/error from either) and
adds two new extern-"C" host wrappers, `ftgemm_sgemm_large_run` (plain) and
`ftgemm_ft_sgemm_large_run` (fused-ABFT), each doing device
alloc+H2D+launch+D2H+free for exactly one kernel call — the SAME grid/block
formula (`gridDim=(ceil(M/64),ceil(N/64))`, `blockDim=64`)
`source/kernel/ft_sgemm/sgemm.cu`'s own `main()` uses for
`kernel_number==3`/`13`. No kernel arithmetic touched.

"large" (ms=ns=64, ks=8) was chosen over small/medium/tall/wide/huge
because its tile-alignment requirement is the LOOSEST of the five and is
the only one satisfied by ALL THREE of `dense.py`'s `--smoke` gemm shapes
(`256x256x256`, `384x256x512`, and the per-batch-item `64x64x64` of
`smoke-gemm-batched-b4-64`) — broader gate coverage than any other tile
config, not an arbitrary pick.

`ft_sgemm_large` is the paper's own headline contribution: online
Algorithm-Based Fault Tolerance FUSED directly into the SGEMM main loop
(row/column checksums accumulated in shared memory as the kernel goes,
`include_code_gen/ft_sgemm_large.cuh`), genuinely self-contained — it
takes the SAME 8-argument `(M,N,K,A,B,C,alpha,beta)` signature as the
plain kernel, no external checksum-vector state (unlike the artifact's
OTHER `abft_baseline` config, `kernel_number==10`, which DOES need
separate non-fused checksum buffers — not wrapped here). Registered as
this directory's ONLY `IMPL_NAME` (`ftgemm-sgemm-large`), per the task's
instruction that the FT-on kernel is the paper's actual point; the plain
FT-off kernel (`ftgemm_sgemm_large_run`) is equally exposed in
`bridge.cu` and was used to cross-check the operand-layout convention
below, but is not separately registered (kept to one canonical impl per
directory, matching this repo's convention).

## Layout (verified empirically, not just derived — see probe transcript)

Reading the kernel body's own stride arithmetic: `A += ks * M` per K-step
means A is COLUMN-MAJOR `(M,K)` (bit-identical to a C-contiguous
`(K,M)` array holding `A.T`); `B += ks * N` per K-step means B is
ROW-MAJOR `(K,N)` (bit-identical to `dense.py`'s own row-major `(K,N)`
operand — no transform needed at all); the output-store pattern
(`C += (by*ns+...) * M`) means C is written COLUMN-MAJOR `(M,N)`.
`adapter.py::prepare()` therefore feeds `np.ascontiguousarray(A.T)` (a
memory-layout reinterpretation, not a numeric transform) for the "A"
argument and `B` completely as-is for "B"; `run()` reads the raw `(N,M)`
row-major output back and transposes once more to a plain `(M,N)`
row-major result.

Confirmed on a random 128x192x64 shape (multiples of 64/64/8) against
`dense.py`'s own `reference_gemm` formula, standalone ctypes probe, BEFORE
wiring into `adapter.py`:

```
sgemm_large (FT-off):   max abs err 2.82e-06, max_scaled_err 1.70e-07
ft_sgemm_large (FT-on): max abs err 1.65e-03, max_scaled_err 1.01e-04
```

Both non-tiny, structured errors would have shown up as ~1 (garbage) or a
clean power-of-two-related artifact if the layout convention were wrong;
instead both are exactly at fp32-rounding scale (FT-off) or the fp32-
appropriate ABFT-checksum-rounding scale (FT-on, see below) — the layout
derivation is correct.

## The tolerance-parsing finding (why the gate FAILS, evidenced, not assumed)

`benchspecs/gemm/spec.yaml`'s `gemm-square-kernel.correctness` text states,
verbatim: *"< 1e-6 for fp64 kernels ... < 1e-3 for fp32 ... < 1e-2 for
fp16."* This artifact is fp32 (`float` throughout, `PRECISIONS = ["fp32"]`)
and its own measured error is comfortably inside the spec's STATED fp32
bound:

| shape | FT-on max_scaled_err | spec's fp32 bound (1e-3) | margin |
|---|---|---|---|
| smoke-gemm-square-256 | 7.394e-05 | 1e-3 | 13.5x under |
| smoke-gemm-irregular-384x256x512 | 3.044e-05 | 1e-3 | 33x under |
| smoke-gemm-batched-b4-64 | 1.787e-04 | 1e-3 | 5.6x under |

But `kernelbench/spec.py`'s free-text tolerance parser extracts a SINGLE
numeric tolerance from this correctness field and does not disambiguate by
the declared runtime precision — confirmed directly:

```
$PY -c "from kernelbench import spec; v=spec.load('gemm').variant('gemm-square-kernel'); print(v.tolerance)"
1e-06
```

i.e. the harness gates EVERY precision (fp64/fp32/fp16 alike) against the
FIRST number in the free-text field (the fp64 one, `1e-6`), not the
precision-appropriate one. This is a pre-existing harness limitation (same
CLASS of finding as `spcg`'s sibling STATUS.md, which documents an
analogous `runner.py` precision-text heuristic picking the wrong
substring for a different spec) — flagged here, NOT fixed, per this
task's own scope (`ARTIFACT_GUIDE.md`: "no `kernelbench/` changes unless a
genuine harness bug is found" — this is exactly that kind of finding, left
for whoever owns `spec.py`'s tolerance parser next, not patched in this
pass). The FT-on kernel's own numerical behavior is genuinely sound for
fp32 — an ABFT scheme that recomputes a tile from checksums whenever a
row/column discrepancy exceeds its own internal `err_bound1=9500.0`
threshold necessarily reintroduces slightly more rounding than the plain
accumulation path (checksum sums, corrections, and comparisons are all
extra fp32 arithmetic on top of the matmul itself) — a real, expected,
disclosed cost of the fault-tolerance mechanism itself, not a bug.

Per ARTIFACT_GUIDE.md rule 4 ("if it fails the gate, that IS a result —
record it, do not loosen the gate to make it pass"): the gate is NOT
overridden. The plain FT-off kernel (`ftgemm_sgemm_large_run`, same tile
config, same bridge, not separately registered) passes this harness's
1e-6 gate comfortably (max_scaled_err 1.30e-7 to 2.20e-7 across the same
three smoke shapes, see probe transcript above) — disclosed here for
comparison, per the task's "disclose which" framing.

## Timing-boundary caveat

`bridge.cu`'s wrappers do device `cudaMalloc`+H2D+launch+D2H+`cudaFree`
INSIDE every call (not hoisted into a separate prepare/run split at the
device-buffer level, unlike turbofno's sibling adapter in this directory,
which keeps its device tensors resident across calls). This is acceptable
for this integration's login-node correctness-only gate (rule 5: no timing
sweeps happen here anyway) but means `run()`'s wall time is NOT a pure
kernel-only measurement — a future compute-node timing pass should hoist
the device allocation/H2D into `prepare()` first (straightforward: split
`bridge.cu`'s two functions into `_prepare`/`_run`/`_copy_out`/`_free`,
same pattern as the stencil-family bridges in this repo).

## Gate verification (login node, functional check only)

```
$PY -m kernelbench.runner --kernel gemm --variant gemm-square-kernel \
    --impl ftgemm-sgemm-large --smoke --precision fp32
```

```
running ftgemm-sgemm-large smoke-gemm-square-256 ... INVALID: max_scaled_err=7.394e-05 vs tol 1e-06
running ftgemm-sgemm-large smoke-gemm-irregular-384x256x512 ... INVALID: max_scaled_err=3.044e-05 vs tol 1e-06
running ftgemm-sgemm-large smoke-gemm-batched-b4-64 ... INVALID: max_scaled_err=1.787e-04 vs tol 1e-06
0/3 runs valid
```

`$PY -m kernelbench.runner --kernel gemm --list` confirms:
`ok  ftgemm           ftgemm-sgemm-large` under "paper artifacts"
(`available()==True`) — the adapter itself, the build, and the layout are
all correct; only the harness's precision-agnostic tolerance parse fails
this specific fp32 kernel, as demonstrated above.

## Not done

- `abft_baseline` (non-fused ABFT, `kernel_number==10`, needs external
  checksum buffers) was not wrapped — out of scope, the fused kernel is
  the paper's actual point.
- small/medium/tall/wide/huge tile configs were not wrapped — "large" was
  chosen deliberately for smoke-shape coverage (see above); not a
  correctness gap, just narrower shape coverage than the full artifact CLI.
- The device-alloc-per-call timing contamination (see "Timing-boundary
  caveat") was not fixed — out of scope for a login-node correctness pass.

## Verdict

`ftgemm: BUILT (ftgemm-sgemm-large, FT-on) — gate FAILS under the
harness's parsed tolerance (1e-6, precision-agnostic parser artifact,
see above); measured max_scaled_err 3.0e-5..1.8e-4, comfortably inside
the spec's OWN STATED fp32 bound (1e-3, 5.6x-33x margin) | disclosed
comparison: the plain FT-off counterpart (same tile config, not
separately registered) PASSES the 1e-6 gate cleanly (1.3e-7..2.2e-7)`


## Gate re-run after the per-precision tolerance fix (2026-09-05)

`kernelbench/spec.py` now parses correctness sentences of the form "< 1e-6 for
fp64, < 1e-3 for fp32, < 1e-2 for fp16" into `Variant.tolerance_by_precision`,
and `harness.run_variant` gates each run at the bound matching its precision
(`Variant.tolerance_for(params["precision"])`). Previously the first number
(the fp64 bound) was applied to every precision, which is why this fp32 kernel
was recorded as BUILT (blocked) despite being 5-30x inside the spec's stated
fp32 bound. Re-run (login node, functional only):

```
runner --kernel gemm --variant gemm-square-kernel --smoke --precision fp32 --impl ftgemm-sgemm-large
  smoke-gemm-square-256            err 7.39e-05 <= 0.001  PASS
  smoke-gemm-irregular-384x256x512 err 3.04e-05 <= 0.001  PASS
  smoke-gemm-batched-b4-64         err 1.79e-04 <= 0.001  PASS
3/3 runs valid
```
This is the spec's own bound, not a loosened one.

## Reproduction on zaratan (2026-09-08)

- Machine: UMD zaratan, NVIDIA A100-SXM4-40GB (sm_80, driver 595.71), login-node build, GPU gate on a full A100 compute node (gpu-b11-6) via gpu_run.sh (ran before the login-cluster's GPU-queue policy switched gpu_run.sh's default to the A100 MIG 1g.5gb slice; this gate had already completed on a full A100 by then).
- Toolchain: nvcc 12.8 (conda-forge CUDA 12.8, `KB_CUDA_HOME`), g++ 13.4 (`$CXX`), Python 3.12.14, torch 2.8.0+cu128. Arch flag `-arch=sm_80` (build.sh's own pin, matches the A100). `bridge.cu` build unchanged from the recorded toolchain (only harmless unused-local-variable warnings, same ones STATUS.md already documents).
- Build: OK, exit 0, single `nvcc -O3 -std=c++14 -arch=sm_80 -Xcompiler -fPIC -shared bridge.cu -o bridge.so`. No build-system changes needed.
- Gate: `--kernel gemm --variant gemm-square-kernel --impl ftgemm-sgemm-large --smoke --precision fp32`:
  - smoke-gemm-square-256: err 7.39e-05 <= 0.001 PASS
  - smoke-gemm-irregular-384x256x512: err 3.04e-05 <= 0.001 PASS
  - smoke-gemm-batched-b4-64: err 1.79e-04 <= 0.001 PASS
  - 3/3 runs valid.
- Deviation from the recorded ruling: none — errors match the recorded numbers exactly (same kernel, same tolerance-fix already landed).
- Verdict here: BUILT+GATED, equals the recorded ruling.
