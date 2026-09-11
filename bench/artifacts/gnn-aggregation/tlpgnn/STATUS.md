# TLPGNN — gnn-aggregation — STATUS: BUILT (correctness gate FAILS -- structural normalization mismatch, not a numerical issue)

- Paper: "TLPGNN: A Lightweight Two-Level Parallelism Paradigm for Graph
  Neural Network Computation on GPU" (HPDC'22). `PAPER_KEY =
  conf/hpdc/FuJH22`. Selected as a **core**, regime-**matches** baseline
  under the revised kernel-centrality rule (2026-09-05) --
  `output/kernel_centrality.json`'s `gnn-aggregation|conf/hpdc/FuJH22`
  entry: "TLPGNN's own benchmark (gcn/test_kernel.py) is the narrowest,
  purest kernel-only scope surveyed in this track."
- Repo: https://github.com/charlifu/TLPGNN, commit
  `4b4b4ea37665702714ec28c88bdbeff8c7710686` (2022-12-05). Cloned fresh into
  `source/` (`git clone --depth 1`), unmodified.
- Build: `source/gcn/naive_kernel.cu` (byte-identical to the clone), nvcc
  12.9, `sm_80` (A100), host compiler g++ 14.3.0, python env
  `/pscratch/sd/c/cunyang/gnn/plexus_env` (torch 2.8/cu128). The artifact's
  own `test_kernel.py` JIT-compiles this via
  `torch.utils.cpp_extension.load_inline` at run time; `build.sh` instead
  drives the file-based, ahead-of-time `torch.utils.cpp_extension.load()`
  against `naive_kernel.cu` UNMODIFIED plus `./gcn_glue.cpp` (a
  byte-for-byte copy of `test_kernel.py`'s inline `cpp_source` string, one
  necessary `#include <torch/extension.h>` added since `load()` -- unlike
  `load_inline()` -- does not auto-prepend it; this same include had to be
  force-added to `naive_kernel.cu`'s OWN compile via `-include
  torch/extension.h` in `extra_cuda_cflags`, since that file also relies on
  `load_inline()`'s auto-injected include and would not compile stand-alone
  otherwise -- a build-flag fix, `naive_kernel.cu` itself is untouched).
  Both files live outside `source/` per ARTIFACT_GUIDE.md rule 3.

## What boundary was wrapped

`source/gcn/naive_kernel.cu`'s `gcn_conv_cuda_forward` -- the exact kernel
the artifact's own README instructs a user to run
(`cd gcn/ && python test_kernel.py --dataset citeseer --size 32`). A
SEPARATE kernel, `source/gcn/atomic_kernel.cu` (dynamic atomic-counter
vertex scheduling instead of static block assignment; exercised only by an
undocumented `test_atomic_kernel.py` hardcoding a personal absolute dataset
path), computes the IDENTICAL per-vertex aggregation formula -- only the
work-distribution strategy differs. This adapter wraps `naive_kernel.cu`
because it is the artifact's own documented entry point; the finding below
applies identically to `atomic_kernel.cu`.

## CRITICAL FINDING: the kernel computes row-mean aggregation, not this domain's reference, and ignores edge weights entirely

`gcn_conv_cuda_forward_kernel` computes, for destination vertex `des_v`,
`sum(features[rows[i]] for i in [col_starts[des_v], col_starts[des_v+1]))
* (1.0/(e_pos-s_pos))` -- **D^-1 A @ X** (unweighted row-mean neighbor
aggregation), derived purely from the CSC structure (`col_starts`, `rows`).
**There is no per-edge value parameter anywhere in `forward`'s signature.**
Two consequences:

1. This domain's ONE reference (`kernelbench.impls.cpu_ref.
   reference_gnn_aggregation`) is `D^-1/2(A+I)D^-1/2 @ X` (GCN symmetric
   normalization, +self-loop). `D^-1 A` and `D^-1/2(A+I)D^-1/2` are NOT
   related by any diagonal pre/post-scaling of the dense operand in
   general (unlike TC-GNN's `forward_AGNN`, TLPGNN's kernel exposes no
   value channel to inject the correct per-edge weights through at all).
   There is no way to make this kernel compute the domain's reference
   without editing kernel code, out of scope per rule 3.
2. Edge weights, where present in the input, are structurally ignored --
   every structural neighbor contributes `1/degree` regardless of the
   actual value of `A[i,j]`.

**This is treated as a real, expected result (ARTIFACT_GUIDE.md rule 4),
not a SKIP**: the kernel genuinely implements a well-known sparse-adjacency
aggregation (same `nnz*N` flop shape `_cost_gnn_agg` charges for) -- it is
just a different normalization than the one this domain's single reference
checks. `prepare()` feeds the kernel exactly what `test_kernel.py` itself
would (raw CSC, no self-loop -- the artifact's own script adds none
either), and the gate is left to fail honestly.

## Two additional real, confirmed constraints (not patched)

- **Division by zero for isolated vertices.** `deg = 1.0/(e_pos-s_pos)` is
  computed unconditionally; any vertex with zero in-neighbors (an isolated
  CSC column) divides by zero. `prepare()` raises `NotImplementedError` if
  any column of the CSC has zero nonzeros (checked directly, cheap).
- Empty adjacency (`nnz==0`) is refused for the same reason.
- Neither guard was actually exercised by the smoke/`cora` matrices below
  (no isolated vertices in either), so they never gated a real run; they
  are defensive, evidence-based guards against a real unhandled kernel
  constraint (rule 8), not something observed to fail here.

## Compute precision

`AT_DISPATCH_FLOATING_TYPES` dispatches on the input tensor's dtype and
genuinely supports both `float` and `double` (no tensor-core reduced-
precision path, unlike TC-GNN) -- `PRECISIONS = ["fp32"]` only because this
domain implements no fp64 gnn-aggregation variant to gate fp64 against.

## Verification (login node, correctness gate only, reduced protocol)

```
PY=/pscratch/sd/c/cunyang/gnn/plexus_env/bin/python
export LD_PRELOAD=/usr/lib64/libstdc++.so.6
cd bench
$PY -m kernelbench.runner --kernel gnn-aggregation --variant gnn-agg-kernel-f32 \
    --impl tlpgnn-naive-mean-agg --smoke
$PY -m kernelbench.runner --kernel gnn-aggregation --variant gnn-agg-kernel-f32 \
    --impl tlpgnn-naive-mean-agg --matrices cora --dims 32,128,256,512 --warmup 2 --reps 3
```

Results (all INVALID, all crash-free, all consistent with the structural
finding above -- errors are O(1)-O(20), not a precision artifact, and DO
NOT shrink with a "closer" precision the way a numerical bug would):

| workload    | F   | max_scaled_err | tolerance |
|-------------|-----|-----------------|-----------|
| smoke-uniform  | 32-512 | 1.71 - 2.17 | 1e-4 |
| smoke-banded   | 32-512 | 0.78 - 1.06 | 1e-4 |
| smoke-powerlaw | 32-512 | 2.21 - 3.05 | 1e-4 |
| cora           | 32-512 | 16.7 - 17.3 | 1e-4 |

**0/16 pass the gate** -- expected, by construction, per the normalization
mismatch above; not a bug in the kernel, not loosened. Not spec-conforming
(login node, reduced warmup/reps, shared GPU) -- timing numbers must not be
published.

## Not done

- No compute-node timing sweep (out of scope, rule 5).
- `atomic_kernel.cu` (the dynamic-scheduling variant) was read but not
  separately wrapped -- it computes the identical formula, so it would
  produce the identical gate outcome.
- The isolated-vertex/empty-adjacency guards are defensive (rule 8), not
  exercised by any matrix used above.

## Provenance

- Artifact commit: `4b4b4ea37665702714ec28c88bdbeff8c7710686`
- nvcc: 12.9 (`/opt/nvidia/hpc_sdk/Linux_x86_64/25.5/cuda/12.9`)
- Host compiler: g++ 14.3.0 (`/opt/cray/pe/gcc-native/14/bin/g++`)
- GPU arch: `sm_80` (A100, `TORCH_CUDA_ARCH_LIST=8.0`)
- Python/torch: `/pscratch/sd/c/cunyang/gnn/plexus_env` -- torch 2.8.0+cu128

## Reproduction on zaratan (2026-09-09)

- Machine: UMD zaratan, A100-SXM4-40GB (sm_80, full card, partition `gpu`),
  login-node build.
- Toolchain: nvcc 12.8.93 (conda-forge CUDA 12.8), g++ 13.4.0, torch
  2.8.0+cu128, Python 3.12.14. `TORCH_CUDA_ARCH_LIST=8.0` (unchanged),
  matches this GPU.
- Build: OK. No build-system changes needed (`build.sh` already reads
  `${CC:-...}`/`${CXX:-...}`/`${CUDA_HOME:-...}`/`${LD_PRELOAD:-...}`
  correctly — all exported before this script runs). Forced a rebuild
  (`FORCE_REBUILD=1`) from the pre-existing `build/tlpgnn_gcn*.so`; ninja
  found both object files already up to date and only re-linked,
  confirming the earlier build already used this exact toolchain.
- Gate (`gnn-agg-kernel-f32`/fp32, exact commands quoted in this file —
  `--smoke`'s full 12-shape set plus the 4 `cora` dims): every one of the
  16 runs is INVALID, `max_scaled_err` reproducing the recorded ranges to
  3-4 significant figures: smoke-uniform 1.711-2.174 (recorded 1.71-2.17);
  smoke-banded 0.778-1.060 (recorded 0.78-1.06); smoke-powerlaw 2.213-3.048
  (recorded 2.21-3.05); cora 16.73-17.29 (recorded 16.7-17.3). 0/16 valid,
  same as recorded. Errors are O(1)-O(20) as before, not shrinking with
  precision — consistent with the documented structural normalization
  mismatch, not a numerical regression.
- Deviation from the recorded ruling: none.
- Verdict here: BUILT (gate FAILS — structural normalization mismatch, not
  a numerical issue) — equals the recorded ruling.
