# Fused3S — sddmm — STATUS: BUILT+GATED

- Paper: Fused3S: Fast Sparse Attention on Tensor Cores (ICS'25).
  `PAPER_KEY = conf/ics/LiC25`
- Repo: https://github.com/HPCForge/Fused3S, commit
  `65d43a9cf212d6f18d8931de8e1c66a728756d7d` (2025-10-13). Cloned fresh into
  `source/` (`git clone --depth 1`).
- Build: torch CUDA extension (`source/src/setup.py`, unmodified), nvcc 12.9,
  `sm_80` (A100, `TORCH_CUDA_ARCH_LIST=8.0`), host compiler g++ 14.3.0, python
  env `/pscratch/sd/c/cunyang/gnn/plexus_env` (torch 2.8/cu128).

## What boundary was wrapped

Fused3S fuses SDDMM (`Q.K^T` at the sparsity pattern) + softmax + SpMM
(`.V`) into single CUDA kernel launches — there is no separable sub-kernel
boundary below the CUDA-call level for the paper's fastest variants
(`f3s_1tb1rw*`). Reading `F3S_kernel.cu` shows every one of those faster
variants has its `saveSddmmResult`-equivalent instrumentation **commented
out** (e.g. `f3sKernel1tb1rw`'s `// {//save sddmm result ... }`), i.e. they
physically cannot report the isolated SDDMM stage. The one kernel that can
is `f3s1tb1tcb` (`f3sKernel1tb1tcb`, exposed as `F3S.f3s_1tb1tcb`) — one of
the paper's own ablation baselines, using the same TC-block sparse-attention
formulation, just less warp-scheduling-optimized than the headline variants.

`adapter.py` calls `F3S.f3s_1tb1tcb(..., applySoftmax=False,
saveSddmmResult=True)` and returns only `sddmmResult`, discarding `output`
(the softmax+SpMM result) and `time`. This is the boundary: **the SDDMM
stage of the fused pipeline, isolated via the artifact's own
ablation-kernel flag**, not the paper's fastest end-to-end number. Per
ARTIFACT_GUIDE rule 1, the contamination is stated here rather than hidden:
the CUDA call itself always launches the full 1tb1tcb kernel (SDDMM +
mask + partial-sum machinery); only softmax and SpMM are skipped/discarded,
which the kernel's own `applySoftmax` flag genuinely gates (no extra work
happens when `False`), so no softmax/SpMM compute leaks into the timed
region even though the wrapped call is not a standalone SDDMM-only kernel
launch in the source.

## Two things the adapter does beyond "call the kernel"

**1. Unweighted mask, not weighted SDDMM.** Fused3S's own accuracy test
(`scripts/tests/test_f3s_accuracy.py`) rounds the sparsity pattern's nonzero
values to 1 before comparing (`A_csr_h.data = np.ceil(...)`) — confirmed by
reading `f3sKernel1tb1tcb`'s SDDMM section (`F3S_kernel.cu` ~L714-740): it
computes `dot(Q_i,K_j)` via `HMMA16816`, masks non-edge slots to 0 via the
TC-block bitmap, and never multiplies by an edge value. This is the same
situation `kernelbench.impls.gpu_cuda.TorchSDDMM` already handles for
`torch.sparse.sampled_addmm` (also unweighted) — `adapter.py` mirrors that
established pattern: multiply the kernel's raw dot-product output by
`S[i,j]` in Python, after the CUDA call returns.

**2. TC-block layout decode.** `sddmmResult` is laid out in the kernel's
own tensor-core-fragment order, not the caller's CSR nnz order.
`adapter.py::_decode_layout()`'s docstring has the full derivation,
cross-checked from BOTH sides of unmodified artifact source:
- WRITE side: `utils.cu`'s `update_bitmap()` (called with `blockSize_h`/
  `blockSize_w` swapped at the call site relative to the function's own
  parameter names) and the sibling `sparse_AToB[tcblock_id*blockSize_w +
  col_local] = <real column>` line.
- READ side: `F3S_kernel.cu`'s `addPartialSums()` (bitmap bit lookup) and
  `saveSddmmResult()` (the `i/j/laneId/k` decomposition that writes
  `sddmm_result`), shown to use an *identical* local-index encoding to the
  write side.
- The per-TC-block row-window id isn't returned by `preprocess_gpu` in a
  form `f3s1tb1tcb` uses (its grid is `dim3(nRowWindow,1,1)`, i.e.
  `bid=blockIdx.x` directly) — recovered the same way the kernel does, via
  `searchsorted` on `row_window_offset`.

`adapter.py::prepare()` uses this decode to build a structural
`(row,col)->position` permutation once per matrix (same pattern as the
rassm adapter), with a hard `RuntimeError` on any key mismatch rather than
a silent misalignment. **The correctness gate passing (see below) is the
actual evidence this decode is right** — a wrong bit/index derivation here
would show up as a large, not a ~1e-4, error.

## Precision — ran the fp16 tensor-core variant, not fp32-CSR

The kernel only accepts `torch::Half` (fp16) `Q`/`K` with fp32 MMA
accumulation — this is intrinsic tensor-core arithmetic, not a "requested
precision" toggle software could redirect. `PRECISIONS = ["fp16"]`;
`create()` raises `NotImplementedError` for anything else. Per the task's
own guidance for tensor-core artifacts, verification below uses
`--variant sddmm-tensorcore-blocked-fp16` (1e-2 relative-error gate) with
`--precision fp16` explicitly passed — **not** the literal
`sddmm-csr-kernel-f32` template command, because the fp32-CSR variant's
1e-4 gate is the wrong bound for fp16 tensor-core arithmetic and the
literal command's implicit precision resolution (`runner.py` only special-
cases `"fp32"`/`"fp64"` substrings in the variant id, so
`sddmm-tensorcore-blocked-fp16` alone would silently resolve to `fp32` and
fail `create()`) — this is not a gate loosened, it is the correct spec
variant for this artifact's actual arithmetic.

The dense operand values themselves are generated identically to
`cpu_ref.reference_sddmm`'s non-fp32 branch (float64, unrounded — since the
requested label is `"fp16"`, not `"fp32"`), then separately cast to
`torch.float16` for the CUDA call — so the gate measures both fp16
quantization of the inputs AND fp16 MMA accumulation rounding against the
true fp64 values, which is what the `1e-2` tolerance is calibrated for.

## Machine-specific build workaround (not an artifact patch)

This machine's python (`plexus_env`, layered on a NERSC conda env) bakes
`CXX = "g++ -pthread -B .../compiler_compat"` into distutils — that `-B`
flag makes g++ silently pick up an ancient (GCC 7-era) `cc1plus`, which then
fails torch's `#if __GNUC__ < 9` check. `build.sh` works around this the
same way any CUDA-extension build on this machine would need to: point
`CXX`/`CC` at the real system compiler and override `LDSHARED` with a clean
link line (this venv's own `torch/lib` rpath, no `-B`, no NERSC-conda rpath)
so the extension doesn't end up linked against a stale `libstdc++.so.6`
missing a GCC-8+ symbol (`__cxa_call_terminate`) the extension needs.

**Residual, unfixable-from-the-adapter issue:** even with a clean build, the
`python3.11` executable itself has an old-style `DT_RPATH` (higher priority
than `LD_LIBRARY_PATH`) pointing at that same stale `libstdc++.so.6`, baked
in at the interpreter binary level — so *any* process using this
interpreter has the old libstdc++ already resident before any of our code
runs, regardless of what the extension's own rpath says (confirmed:
`LD_LIBRARY_PATH=/usr/lib64:...` set before the harness starts does NOT
help; a same-process `ctypes.CDLL(..., RTLD_GLOBAL)` workaround was tried
and **segfaults** — two different libstdc++.so.6 copies loaded into one
process is an ABI hazard, not a fix, so that code was removed from
`adapter.py`). The only working fix is `LD_PRELOAD` set at the shell level
**before the python process starts** (`os.environ["LD_PRELOAD"]` set from
inside an already-running interpreter is a no-op — the dynamic linker only
reads it once, at exec time). This is a property of this machine's python
install, not of the artifact, but it does mean:

**Running this adapter requires `LD_PRELOAD=/usr/lib64/libstdc++.so.6` in
the invoking shell.** `available()` will report `False` (import error)
without it.

## Verification (login node, correctness gate only, reduced protocol)

```
PY=/pscratch/sd/c/cunyang/gnn/plexus_env/bin/python
export LD_PRELOAD=/usr/lib64/libstdc++.so.6
$PY -m kernelbench.runner --kernel sddmm --variant sddmm-tensorcore-blocked-fp16 \
    --precision fp16 --impl fused3s-1tb1tcb-sddmm --matrices cant --dims 128 \
    --warmup 1 --reps 3
```

Result: **1/1 valid**, `max_scaled_err = 1.70e-04` (tolerance `1e-2`) — a
clean pass with margin consistent with genuine fp16 quantization/MMA
rounding noise (not the O(1) garbage a wrong layout-decode would produce).
Not spec-conforming (login node, reduced warmup/reps, shared GPU) — timing
numbers from this run must not be published, only the correctness result.

## Limitations recorded, not worked around

- Fused3S's `preprocess_gpu`/kernel are written for **square** adjacency-
  style sparsity patterns (one `numNodes` controls both Q's and K's row
  count); `adapter.py::prepare()` raises `NotImplementedError` for
  rectangular matrices. `cant` (62451x62451) is square, so this did not
  block verification, but it is a real scope limit of this integration.

## Provenance

- Artifact commit: `65d43a9cf212d6f18d8931de8e1c66a728756d7d`
- nvcc: 12.9 (`/opt/nvidia/hpc_sdk/Linux_x86_64/25.5/cuda/12.9`)
- Host compiler: g++ 14.3.0 (`/opt/cray/pe/gcc-native/14/bin/g++`)
- GPU arch: `sm_80` (A100, `TORCH_CUDA_ARCH_LIST=8.0`)
- Python/torch: `/pscratch/sd/c/cunyang/gnn/plexus_env` — torch 2.8.0+cu128

## Reproduction on zaratan (2026-09-09)

- Machine: UMD zaratan, A100 MIG 1g.5gb slice (sm_80; job ran on
  gpu-b11-6.zaratan.umd.edu, physical card NVIDIA A100-SXM4-40GB), login-node
  build.
- Toolchain: nvcc 12.8.93 (conda-forge CUDA 12.8), g++ 13.4.0 (conda-forge),
  torch 2.8.0+cu128, Python 3.12.14; no cmake (torch `setup.py build_ext`).
  Arch flags: `TORCH_CUDA_ARCH_LIST=8.0` (sm_80), unchanged from the recorded
  build.
- Build: OK (`build.sh` found the extension already built —
  `F3S.cpython-312-x86_64-linux-gnu.so`, built this session — and skipped
  recompilation; idempotent per the script's own design). Build-system
  changes: none — `build.sh`'s existing `${PY:-...}`/`${CUDA_HOME:-...}`/
  `${CXX:-...}`/`${CC:-...}` fallbacks already read this machine's
  env.sh-exported values first, so no edit was needed.
- Gate: `sddmm-tensorcore-blocked-fp16` (fp16, `--matrices cant --dims 128
  --warmup 1 --reps 3`): PASS, `max_scaled_err = 1.70e-04 <= tol 1e-2`, 1/1
  runs valid.
- Deviation from the recorded ruling: none — same variant, same matrix
  (`cant`), same error to 3 significant figures (1.70e-04 here vs. 1.70e-04
  on Perlmutter).
- Verdict here: BUILT+GATED — same as the recorded ruling.
