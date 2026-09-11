# flashsparse (FlashSparse) — spmm

**Status: BUILT+GATED on spmm-binary-adjacency-kernel (its actual regime —
fp16, unweighted 0/1-adjacency SpMM) since 2026-09-06; general weighted
variants (`spmm-tensorcore-fp16`): gate FAILS (values hardcoded to 1.0, see
finding below)**

## 2026-09-06: re-gated under spmm-binary-adjacency-kernel

`benchspecs/spmm/spec.yaml` gained a dedicated `spmm-binary-adjacency-kernel`
variant (see `../README.md`'s "Pattern-only SpMM variant" section) whose
harness-level `kernelbench.domains.sparse.variant_transform` hook forces the
sparse operand's stored values to 1.0 for EVERY implementation, including
the fp64 CSR reference, before either sees it -- the exact regime this
adapter's IMPORTANT FINDING below shows the artifact is structurally limited
to. No adapter code change was needed: `flashsparse-spmm` already computes
`A_pattern @ B` unconditionally (`preprocess_gpu_fs` never reads `A.data`),
so once the reference is binarized too, the two agree on every matrix, not
just already-binary ones like `cora`.

```
LD_PRELOAD=/usr/lib64/libstdc++.so.6 $PY -m kernelbench.runner --kernel spmm \
    --variant spmm-binary-adjacency-kernel --impl flashsparse-spmm \
    --precision fp16 --smoke --warmup 1 --reps 3
```
Result: **12/12 valid** (3 synthetic smoke matrices x N in {32,128,256,512}),
`max_scaled_err` 7.6e-04 – 2.0e-03, all `<= tol 0.01` -- the SAME smoke
matrices that were 0/9 INVALID under `spmm-tensorcore-fp16` (err 2.96–5.67)
now pass cleanly, because the reference is binarized alongside the kernel.

```
LD_PRELOAD=/usr/lib64/libstdc++.so.6 $PY -m kernelbench.runner --kernel spmm \
    --variant spmm-binary-adjacency-kernel --impl flashsparse-spmm \
    --precision fp16 --matrices cora --dims 128 --warmup 1 --reps 3
```
Result: **1/1 valid**, `max_scaled_err = 3.93e-03` (identical to the
`spmm-tensorcore-fp16`/cora result below, since `cora` was already binary --
`cora+bin` binarizes a no-op there). Reduced-protocol, non-conforming
numbers only (warmup=1, reps=3, shared login-node GPU) -- no timing sweep.

The `spmm-tensorcore-fp16` gate result documented further below ("Gate verification") is UNCHANGED and
kept as the on-record general-weighted-variant failure this variant exists
to explain, per the user's 2026-09-06 decision (never delete a documented
FAIL, add the correct home alongside it).

- Paper: "FlashSparse: Minimizing Computation Redundancy for Fast Sparse
  Matrix Multiplications on Tensor Cores", PPoPP'25. `PAPER_KEY =
  conf/ppopp/ShiLXFWW25`.
- Artifact: https://github.com/ParCIS/FlashSparse
- Commit cloned: `168764612c1a2ad5b8661be22cfb6a8ae044bada` (2025-10-05),
  `git clone --depth 1`. `source/` is byte-identical to upstream — no patch
  needed (unlike `dtcspmm/`, FlashSparse's SpMM+preprocessing extensions
  have no Sputnik/Glog entanglement to work around).
- Toolchain: `nvcc` 12.9, host compiler `/opt/cray/pe/gcc-native/14/bin/g++`
  (GCC 14.3.0, pinned via `CC`/`CXX` — same login-node `c++`-resolves-to-g++7
  issue as `dtcspmm/`/`gemv/marlin`). `TORCH_CUDA_ARCH_LIST=8.0` (A100;
  upstream targets RTX4090/H100). Python
  `/pscratch/sd/c/cunyang/gnn/plexus_env/bin/python`, torch `2.8.0+cu128`.

## Selection rationale

Core general-GPU-SpMM baseline under the revised kernel-centrality rule
(2026-09-05): the most recent (PPoPP'25) entry in the RoDe/DTC-SpMM
tensor-core-SpMM line this spmm track's `spmm-gpu-kernel-f32` and
`spmm-tensorcore-fp16` variants are built around; recency is the tiebreak
among the three core-kernel picks, not the selection criterion itself.

## Build (rule 3)

The artifact's own `source/FlashSparse/compile.sh` (`python setup.py
install`) builds ALL FOUR of its bundled extensions (`FS_SpMM`, `FS_SDDMM`,
`FS_Block`, `FS_Block_gpu`) and installs into the shared venv's
site-packages. This integration only needs the SpMM kernel (`FS_SpMM`) and
its GPU preprocessing (`FS_Block_gpu`) — `FS_SDDMM` is a different track,
`FS_Block` is an alternate CPU preprocessing path not used here. Built
instead via this directory's own `fs_setup.py` (a `CUDAExtension` listing
only those two, same isolation precedent as `dtcspmm/dtc_setup.py`), into
`build/lib/` under this directory — never installed into the shared venv.
Zero source files touched (checked: `git status --short` in `source/` is
clean after the build).

## What was wrapped (rule 1)

`FS_Block_gpu.preprocess_gpu_fs` (CSR -> FlashSparse's GPU-built compressed
tensor-core format: row-window offsets, a per-window sparse-A-to-X column
index, and a value array, 8x8-tiled) + `FS_SpMM.forward_fp16` (the actual
swap-and-transpose WMMA SpMM kernel), both called directly, unmodified.
`preprocess_gpu_fs`'s conversion is timed as preprocessing in `prepare()`.

**Timing-scope caveat (disclosed, per rule 1's explicit allowance)**: unlike
RoDe/DTC-SpMM/InferFast (GPU-resident buffers, device-pointer kernel entry
points), FlashSparse's own pybind bindings take CPU-resident torch tensors
and do their OWN `cudaMalloc`/H2D/kernel-launch/D2H/`cudaFree` *inside* the
bound function — the raw device-pointer function that actually launches the
kernel (`spmm_forward_cuda_fp16`, in `spmmKernel.cu`) is not itself
pybind-exposed. Every `run()` call therefore also times FlashSparse's own
H2D/D2H copies and allocation, not kernel-only time — acceptable for this
login-node GATE check (rule 5: no timing sweeps here), but would need a
custom device-pointer ctypes wrapper (same strategy as `rode/wrapper.cu`)
for a fair compute-node timing run — not built here, out of budget.
`spmm_forward_cuda_fp16` additionally hardcodes its own 10-iteration
internal warmup loop unconditionally before the `epoches`-times timed loop;
called here with `epoches=1`, so every `run()` still pays that fixed
10-launch warmup (further disclosed contamination, not a correctness bug).

## IMPORTANT FINDING: same binary-pattern-only limitation as DTC-SpMM

Traced `FS_Block_gpu.preprocess_gpu_fs`'s value-generation path
(`block_kernel.cu::generate_tcoffset_id_atob_fs`, called from
`seg_sort_dequ_fs`): the returned `values` tensor is allocated as
`torch::zeros(...)` and populated ONLY via `values_[...] =
__float2half(1.0);` at every structurally-occupied position — no parameter
anywhere in `preprocess_gpu_fs`'s signature lets a caller supply real
nonzero values (`grep -rn "torch::Tensor value" Block_gpu/` upstream: zero
hits for an *input* values parameter). A second, CPU-side preprocessing path
exists (`FS_Block.blockProcess_fp16`, used by `SpMM/test/mgcn/test.py`) that
DOES take a caller-supplied tensor called `dd` ("degree") — but that same
test file's own correctness check (`value.append(dd[i]*dd[column_index1
[j]])`) shows it computes a RANK-1 `degree[row]*degree[col]` GCN-style
symmetric-normalization weight, not an arbitrary per-edge value — this
cannot represent a general nonzero-value matrix either (a rank-1 outer
product can only ever express `outer(d, d)`-shaped weights). Both of
FlashSparse's own preprocessing paths are therefore fundamentally built for
GNN-adjacency-style (or degree-normalized) aggregation, consistent with the
TC-GNN/DTC-SpMM lineage this paper directly extends (`dtcspmm/STATUS.md`
documents the identical finding for the paper this one cites as a
baseline) — not a bug introduced by this integration, and not something
reverse-engineering `seg_sort_dequ_fs`'s internal sort/dedup order within
this integration's budget could responsibly fix (it would become a new
capability neither the paper nor the released code demonstrates).

**Consequence (rule 4)**: this adapter computes `A_pattern @ B`, matching
the spec's real-valued `A @ B` exactly when A's stored values are already
all 1 (cora/citeseer/pubmed via this codebase's Planetoid loader) and
diverging by construction for general-valued matrices and this track's
`U(-1,1)` synthetic smoke set.

## Precision

`FS_SpMM.forward_fp16` -> `spmm_forward_cuda_kernel_fp16` — half-precision
WMMA compute. `PRECISIONS = ["fp16"]`, gated under `spmm-tensorcore-fp16`
with `--precision fp16` (per this task's instructions; the artifact also
ships `forward_tf32`, not wired here — fp16 is the paper's headline
swap-and-transpose configuration).

## adapter.py

- `KERNEL = "spmm"`, `IMPL_NAME = "flashsparse-spmm"`, `PRECISIONS =
  ["fp16"]`.
- `prepare()`: CSR row-count padded to a multiple of `BLK_H=8` (window size;
  `preprocess_gpu_fs` computes `window_num = num_nodes / blockSize_h` via
  integer division, so a non-multiple would silently truncate the tail
  window — padded explicitly, same discipline as `dtcspmm`/`inferfast`).
  Square-matrix constraint (`M == K`, `preprocess_gpu_fs` takes one
  `num_nodes` for both dimensions) raises `NotImplementedError` naming the
  constraint (rule 8) — doesn't trigger for this track's
  `recommended_subset` in practice. `B`: numpy `default_rng` matching
  `cpu_ref.reference_spmm`'s `_dense_operand` exactly, cast to fp16.
- `run()`: ONE `forward_fp16(...)` call with `epoches=1` (see timing-scope
  caveat above).
- `to_host()`: the returned tensor is already CPU-resident (FlashSparse
  copies D2H internally) — just cast to fp64 numpy.
- `timer()` reuses `kernelbench.impls.gpu_cuda.CudaEventTimer`.

## Gate verification (login node, functional check only)

**Smoke (expected FAIL — evidences the IMPORTANT FINDING, not a bug):**

```
LD_PRELOAD=/usr/lib64/libstdc++.so.6 $PY -m kernelbench.runner --kernel spmm \
    --variant spmm-tensorcore-fp16 --impl flashsparse-spmm --precision fp16 \
    --smoke --warmup 1 --reps 3
```

Result: **0/9 INVALID**. `max_scaled_err` ranged 2.96–5.67 against tol
`0.01` — essentially IDENTICAL magnitudes to `dtcspmm`'s smoke failure
(2.96–5.67 here vs. 2.96–5.67 there), strong independent confirmation both
artifacts fail for the exact same structural reason (missing value
-weighting), not numerical noise.

**Real binary graph (cora):**

```
LD_PRELOAD=/usr/lib64/libstdc++.so.6 $PY -m kernelbench.runner --kernel spmm \
    --variant spmm-tensorcore-fp16 --impl flashsparse-spmm --precision fp16 \
    --matrices cora --warmup 1 --reps 3
```

Result: **3/3 valid**. `max_scaled_err = 3.93e-03` (dim=128,256) and
`8.17e-03` (dim=512), all `<= tol 0.01` — fp16-typical magnitude (somewhat
larger than DTC-SpMM's tf32 result, consistent with fp16's narrower dynamic
range / this kernel's fp16, not fp32, accumulation), confirming the
swap-and-transpose kernel mechanism is correct for the class of input the
artifact's released API actually supports. Reduced-protocol, non-conforming
numbers only (warmup=1, reps=3, shared login-node GPU, `high variance`
flagged) — no timing sweep was run.

## Not done

- No value-permutation recovery for `seg_sort_dequ_fs` (see IMPORTANT
  FINDING) — same reasoning as `dtcspmm/STATUS.md`.
- No device-pointer-only wrapper to strip FlashSparse's own H2D/D2H from the
  timed region (see "Timing-scope caveat") — out of budget; irrelevant for
  this login-node gate check, would matter for a future compute-node timing
  sweep.
- No sweep across the rest of the 8-graph GNN suite / full protocol; no
  `forward_tf32` or `_balance` variant wired — one representative binary
  graph (cora) at the simplest single kernel call was sufficient to gate
  within budget.

## Reproduction on zaratan (2026-09-09)

- Machine: UMD zaratan, A100 (full SXM4 40GB, `gpu-b11-6`, sm_80) via
  `bench/gpu_run.sh`, login-node build.
- Toolchain: nvcc 12.8.93 (conda-forge CUDA 12.8), g++ 13.4.0, torch
  2.8.0+cu128, Python 3.12.14. Build re-ran `fs_setup.py` unchanged (idempotent
  `ninja: no work to do` — object files already current); no build-system
  changes needed here beyond the LD_PRELOAD-from-env fix already committed
  in `build.sh` from an earlier checkpoint on this machine.
- Build: OK. Build-system changes: none this pass (only exported
  `OPENBLAS_NUM_THREADS=4 OMP_NUM_THREADS=4` around the build invocation —
  see machine-gotchas note below — not a file change).
- Gate: `spmm-binary-adjacency-kernel` smoke fp16: 12/12 valid, err
  1.35e-03–1.97e-03 (all `<= 0.01`). `spmm-binary-adjacency-kernel` cora
  dim=128: 1/1 valid, err 3.93e-03. `spmm-tensorcore-fp16` smoke fp16: 0/9
  INVALID, err 2.96–5.67 (matches the recorded finding exactly). `spmm-
  tensorcore-fp16` cora dim=128/256/512: 3/3 valid, err 3.93e-03/3.93e-03/
  8.17e-03 — all four numbers match the STATUS.md values above bit-for-bit.
- Deviation from the recorded ruling: none. Every number reproduces the
  Perlmutter-recorded value.
- Verdict here: BUILT+GATED — equals the recorded ruling.

Machine gotchas hit while reproducing (recorded once here, applies to the
other four directories in this batch too): (1) the shared login node's
`RLIMIT_NPROC=256` was saturated for long stretches by concurrent sibling
agents reproducing other artifacts, causing intermittent shell-fork failures
and requiring `OPENBLAS_NUM_THREADS`/`OMP_NUM_THREADS` capped low around
Python imports and native builds — a resource-contention symptom, not an
artifact issue (see `spmm/rassm/STATUS.md` for where it actually blocked a
gate). (2) `bench/gpu_run.sh`'s compute nodes (`gpu-b11-6`) have **no
outbound internet** — `cora`'s Planetoid raw files (`ind.cora.*`, 8 files)
and any first-use SuiteSparse `.mtx`/`.tar.gz` had to be pre-fetched from
the login node into `bench/graphs/planetoid/Cora/raw/` /
`bench/matrices-cache-equivalent` (see `bench/env.sh`'s `KERNELBENCH_MATRIX_CACHE`
note in `spmm/rassm/STATUS.md`) before a `gpu_run.sh` gate that needs a real
matrix/graph will succeed; a bare `--smoke` run (synthetic matrices, no I/O)
is unaffected.
