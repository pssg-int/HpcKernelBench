# H100 (sm_90) baseline data — old code on new hardware

Goal (user, 2026-09-10): run every baseline on an H100, not just the two
sm_90a-specific artifacts, to see how A100-era (sm_80) HPC-kernel code behaves
on Hopper — both the functional gate AND, for cleanly-gating artifacts,
indicative throughput. Two layers emerged and both are reported here:

* **Layer A — the A100 binary as-is on H100.** An artifact built for sm_80
  runs on an H100 *only if it embedded PTX* (`-arch=sm_80` / `compute_80`),
  which JIT-forwards to sm_90. An artifact built **SASS-only** (`code=sm_80`,
  the common case) fails at the first kernel launch with
  `cudaErrorNoKernelImageForDevice` — it does not run at all without a recompile.
* **Layer B — recompiled for sm_90.** Every adapter build.sh takes `KB_SM`
  (=90 on the H100 node, via env.sh); a genuine clean rebuild targets sm_90 and
  the kernel then runs. Several build systems had to be taught to pass the arch
  through and to actually recompile (they were incremental / had skip-guards);
  those were build-system fixes only, committed, no kernel code touched.

**Method.** Functional gates run through the batch drivers under
`shared/kernel-bench/` (`h100_gate_batch*.sbatch`, `h100_gate_batch_force.sbatch`
for forced sm_90 rebuilds, `h100_gate_realwl.sbatch` for fixed-shape adapters
that need their real workload rather than the synthetic `--smoke` shapes). Raw
lines in `h100_results.tsv`, per-artifact logs in `h100_logs/`. Gates are
non-conforming (shared node, GPU clocks not lockable here) — correctness only,
not spec timing.

## Headline findings

1. **sm_90a-specific:** quantized-gemm/mxblas **PASS** (own FP8 tests, 432.9
   TFLOPS); spmm/voltrix builds+runs but **gate-fails** (0/12, TF32-roundoff
   2.9e-4 vs the adapter's declared fp32 tol 1e-4 — not loosened).
2. **Most PTX-embedding sm_80 kernels run correctly on H100 unchanged** (JIT
   forward): ~40 artifacts pass with errors matching their A100 record.
3. **SASS-only sm_80 binaries do NOT run on H100** — they fail with
   `cudaErrorNoKernelImageForDevice`. Recompiling for sm_90 fixes them: proven
   on gnn-aggregation/ge-spmm, spmm/ge-spmm, gnn-aggregation/stragcn,
   spmm/inferfast, sparse-attention-kernel/gpa, sddmm/tc-gnn — all move from
   no-kernel-image to a clean PASS after a forced sm_90 rebuild.
4. **H100-specific regressions** (passed on A100, fail on H100 even rebuilt):
   attention-kernel/metaattention — TileLang JIT aborts with an internal
   layout-inference error (`Get different layout for K_shared`) on sm_90;
   stencil/convstencil — `illegal memory access`; spmv/cb-spmv — one banded
   fp64 case now wrong (2/3).
5. **Reproduced A100 gate-fails (NOT H100 regressions):** e.g. tlpgnn (0/12,
   documented structural-normalization mismatch), attention-kernel/et
   (documented binding bug), spmm/mp-spmm, sequence-alignment/logan — the H100
   result matches the recorded A100 verdict.

## Functional-gate results on H100 (99 artifacts, 60 PASS)

`PASS` = ran & gate passed on H100; `PASS*` = passed after a forced sm_90
rebuild (was a no-kernel-image failure as the A100 sm_80 binary); `NKI` =
`cudaErrorNoKernelImageForDevice` (SASS-only sm_80 — needs recompile, some via
a heavy dependency/DSL rebuild not yet done); `gate-fail`/`unsupported`/
`timeout` as recorded (mostly consistent with the A100 record).

| artifact | H100 gate | err | class |
|---|---|---|---|
| ann-search/clover | 2/2 runs valid | err 1.31e-07 <= | PASS |
| ann-search/pathweaver | 2/2 runs valid | err 9.38e-03 <= | PASS |
| attention-kernel/bytetransformer | 1/1 runs valid | err 2.23e-03 <= | PASS* |
| attention-kernel/et | 0/3 runs valid |  | gate-fail |
| attention-kernel/flashattention-t | sm_90 CUTLASS rebuild >2.5h (in flight |  | sm_90 build too slow (deferred) |
| attention-kernel/metaattention | [15:30:07] : Fatal: Get different layo |  | **H100 regression** |
| attention-kernel/pat | GATE_RESULT valid=True max_scaled_err= |  | PASS* |
| bfs/blest | 2/2 runs valid | err 0.00e+00 <= | PASS |
| bfs/efg | 2/2 runs valid | err 0.00e+00 <= | PASS |
| cg-krylov/bootcmatchgx | NO_VALID_LINE(rc=124) | err 0.00e+00 <= | timeout/hang |
| cg-krylov/millefeuille | 3/3 runs valid | err 0.00e+00 <= | PASS |
| cg-krylov/perks | 3/3 runs valid | err 0.00e+00 <= | PASS |
| cg-krylov/spcg | 3/3 runs valid | err 0.00e+00 <= | PASS |
| cholesky/exageostat | 2/2 runs valid | err 0.00e+00 <= | PASS |
| cholesky/hicma-x | 2/2 runs valid | err 4.11e-03 <= | PASS |
| connected-components/ecl-scc | 2/2 runs valid | err 0.00e+00 <= | PASS |
| connected-components/yacclab | 3/3 runs valid | err 0.00e+00 <= | PASS |
| convolution/hidet | NO_VALID_LINE(rc=1) |  | error |
| convolution/tetris | 0/1 runs valid (2 unsupported) |  | gate-fail |
| fft/turbofft | 3/3 runs valid (5 unsupported) | err 4.53e-06 <= | PASS |
| gemm/ftgemm | 3/3 runs valid | err 7.39e-05 <= | PASS |
| gemm/hexcute | NO_VALID_LINE(rc=1) |  | error |
| gemm/moonpoly | 3/3 runs valid | err 1.21e-04 <= | PASS |
| gemm/turbofno | 3/3 runs valid | err 1.76e-07 <= | PASS |
| gemv/marlin | NO_VALID_LINE(rc=1) |  | error |
| gemv/packkv | 3/3 runs valid | err 2.69e-04 <= | PASS |
| gnn-aggregation/ge-spmm | 12/12 runs valid | err 1.93e-07 <= | PASS* |
| gnn-aggregation/stragcn | 12/12 runs valid | err 5.99e-07 <= | PASS* |
| gnn-aggregation/tc-gnn | 0/2 runs valid (2 unsupported) |  | gate-fail |
| gnn-aggregation/tlpgnn | 0/12 runs valid |  | gate-fail |
| graph-pattern-mining/fringe-sgc | 2/2 runs valid | err 0.00e+00 <= | PASS |
| graph-pattern-mining/glumin | 2/2 runs valid | err 0.00e+00 <= | PASS |
| graph-pattern-mining/graphfold | 2/2 runs valid | err 0.00e+00 <= | PASS |
| graph-pattern-mining/graphset | 2/2 runs valid | err 0.00e+00 <= | PASS |
| graph-pattern-mining/stmatch | 0/2 runs valid |  | gate-fail |
| lossless-compression/gpulz | 3/3 runs valid | err 0.00e+00 <= | PASS |
| lossless-compression/lscomp | 3/3 runs valid | err 0.00e+00 <= | PASS |
| lossless-compression/mans | NO_VALID_LINE(rc=124) |  | timeout/hang |
| lossless-compression/opthuffdec | 3/3 runs valid | err 0.00e+00 <= | PASS |
| lossless-compression/zipserv | 0/3 runs valid |  | gate-fail |
| lossy-compression/cuszp | 3/3 runs valid | err 3.42e-03 <= | PASS |
| lossy-compression/cuszp-v1 | 3/3 runs valid | err 3.42e-03 <= | PASS |
| lossy-compression/fzgpu | 3/3 runs valid | err 3.42e-03 <= | PASS |
| lossy-compression/pfpl | 3/3 runs valid | err 3.42e-03 <= | PASS |
| mttkrp/blco | 3/3 runs valid | err 2.65e-16 <= | PASS |
| multigrid/amgt | NO_VALID_LINE(rc=134) |  | error |
| multigrid/bootcmatchgx | 3/3 runs valid | err 0.00e+00 <= | PASS* |
| quantized-gemm/fp6llm | 0/3 runs valid |  | gate-fail |
| quantized-gemm/marlin | NO_VALID_LINE(rc=124) |  | timeout/hang |
| quantized-gemm/qfactory | NO_VALID_LINE(rc=124) |  | timeout/hang |
| quantized-gemm/tilus | 3/3 runs valid | err 1.74e-04 <= | PASS* |
| sddmm/flashsparse | ran sm_90; KNOWN residue-tile overflow |  | gate-fail |
| sddmm/fused3s | NO_VALID_LINE(brc=0,rc=255) |  | NKI (recompile) |
| sddmm/rassm | 0/6 runs valid |  | gate-fail |
| sddmm/rode | 6/6 runs valid | err 1.05e-07 <= | PASS |
| sddmm/sputnik | 6/6 runs valid | err 1.17e-07 <= | PASS |
| sddmm/tc-gnn | 2/2 runs valid | err 2.67e-04 <= | PASS* |
| sequence-alignment/logan | 0/2 runs valid |  | gate-fail |
| sparse-attention-kernel/fused3s | 1/1 runs valid (6 unsupported) | err 3.35e-04 <= | PASS* |
| sparse-attention-kernel/gpa | 7/7 runs valid | err 2.42e-03 <= | PASS* |
| sparse-attention-kernel/sparse-transformer | 0/0 runs valid (7 unsupported) |  | unsupported (needs real workload) |
| sparse-attention-kernel/vit-sparse | 0/0 runs valid (7 unsupported) |  | unsupported (needs real workload) |
| spgemm/amgt | NO_VALID_LINE(rc=134) |  | error |
| spgemm/ocean | 3/3 runs valid | err 2.21e-16 <= | PASS |
| spgemm/tilespgemm | 0/3 runs valid |  | gate-fail |
| spmm/dtcspmm | 12/12 runs valid | err 2.31e-04 <= | PASS* |
| spmm/flashsparse | 12/12 runs valid | err 1.35e-03 <= | PASS* |
| spmm/ge-spmm | 12/12 runs valid | err 2.24e-07 <= | PASS* |
| spmm/generalsparse | 12/12 runs valid | err 1.46e-03 <= | PASS* |
| spmm/inferfast | 12/12 runs valid | err 3.88e-04 <= | PASS* |
| spmm/insum | NO_VALID_LINE(rc=124) |  | timeout/hang |
| spmm/mp-spmm | 0/3 runs valid (6 unsupported) |  | gate-fail |
| spmm/nm-spmm | 0/0 runs valid (12 unsupported) |  | unsupported (needs real workload) |
| spmm/rassm | 0/12 runs valid |  | gate-fail |
| spmm/rode | 6/6 runs valid (6 unsupported) | err 2.24e-07 <= | PASS |
| spmm/smat | 0/9 runs valid |  | gate-fail |
| spmm/sputnik | 12/12 runs valid | err 2.24e-07 <= | PASS |
| spmm/sspmm | 9/9 runs valid | err 3.24e-04 <= | PASS |
| spmm/tc-gnn | 0/0 runs valid (9 unsupported) |  | unsupported (needs real workload) |
| spmm/voltrix | 0/12 runs valid |  | gate-fail |
| spmv/adaptive-spmv | 3/3 runs valid | err 3.27e-16 <= | PASS |
| spmv/cb-spmv | 2/3 runs valid | err 3.57e-16 <= | PASS |
| spmv/diaq | 3/3 runs valid | err 4.16e-16 <= | PASS |
| spmv/spmv-acc | 3/3 runs valid | err 0.00e+00 <= | PASS |
| spmv/sspmv | 3/3 runs valid | err 3.68e-16 <= | PASS |
| spmv/tilespmv | 3/3 runs valid | err 3.41e-16 <= | PASS |
| sptrsv/split-sptrsv | NO_VALID_LINE(rc=124) |  | timeout/hang |
| sptrsv/yysptrsv | NO_VALID_LINE(rc=124) |  | timeout/hang |
| stencil/an5d | 3/3 runs valid | err 9.54e-09 <= | PASS |
| stencil/convstencil | CUDA error 700: illegal memory access |  | **H100 regression** |
| stencil/flashfftstencil | 0/0 runs valid (1 unsupported) |  | unsupported (needs real workload) |
| stencil/lorastencil | needs star2d3r real workload (bespoke  |  | unsupported (needs real workload) |
| stencil/spider | 1/1 runs valid (2 unsupported) | err 2.90e-03 <= | PASS |
| string-regex-matching/gpunfa | 2/2 runs valid | err 0.00e+00 <= | PASS |
| string-regex-matching/ngap | 2/2 runs valid | err 0.00e+00 <= | PASS |
| tensor-contraction/fastkron | NO_VALID_LINE(rc=139) |  | error |
| topk-selection/gpu-topk-study | NO_VALID_LINE(rc=1) |  | error |
| triangle-counting/tc-compare | 2/2 runs valid | err 0.00e+00 <= | PASS |
| triangle-counting/tot | 2/2 runs valid | err 0.00e+00 <= | PASS |


### Tally

| class | n |
|---|---|
| PASS (incl. 14 after sm_90 rebuild) | 60 |
| H100-specific regression | 2 |
| no-kernel-image / build-error (SASS-only or heavy rebuild pending) | 8 |
| gate-fail, ran (mostly reproduced A100 verdicts) | 15 |
| unsupported on generic --smoke (fixed-shape; needs real workload) | 6 |
| timeout / hang (some reproduce A100 hangs: mans, yysptrsv) | 7 |

## Remaining tail (documented, not chased to a PASS)

* **Fixed-shape kernels needing a bespoke standalone gate** (their A100 record
  gates via `harness.run_variant` with specific shapes, not `--smoke`):
  stencil/lorastencil (star2d3r), spmm/nm-spmm, sparse-attention-kernel/
  {sparse-transformer,vit-sparse}. These PASS on A100 with their real shapes;
  running them on H100 needs the same bespoke shapes. (attention-kernel/pat and
  bytetransformer are now PASS via their real workloads; et reproduces its A100
  binding-bug gate-fail.)
* **flashattention-t — sm_90 build too slow:** the CUTLASS forward template for
  head_dim {64,128} x {fp16,bf16} does not finish compiling for sm_90 even
  in >2.5 h; deferred. Its A100 sm_80 PASS (max_scaled_err 3.0e-3) stands.
* **Heavy sm_90 rebuilds of a large dependency / DSL framework:**
  convolution/hidet & gemm/hexcute (DSL compilers), multigrid/amgt +
  spgemm/amgt (reuse a big sm_80 HYPRE build — needs HYPRE recompiled for
  sm_90). Their sm_80 binary fails on H100 with no-kernel-image; a full sm_90
  rebuild is the fix but was out of scope for this pass.
* **sddmm/fused3s edge case:** recompiled for sm_90 (gencode compute_90, build
  OK) and sparse-attention-kernel/fused3s — which reuses the same .so — now
  PASSES, but the sddmm gate path still hits no-kernel-image on the F3S SDDMM
  kernel (a template config apparently not instantiated for sm_90). One artifact.
* **Known A100-side blocks reproduced:** topk-selection/gpu-topk-study (prebuilt
  .so needs glibc 2.34 > RHEL8's 2.28), lossless-compression/mans (hang).

## Phase 2 — indicative throughput on H100 (40-artifact representative subset)

Per the user's choice: a representative subset (≤2 passing artifacts per kernel
track, spanning all tracks) timed on a full H100 (`--gres=gpu:h100:1`) on the
SAME gate/smoke inputs, longer protocol (warmup 5, reps 20). **Indicative only,
NOT spec-conforming** — GPU clocks are not lockable here (no root), and the
`--smoke` inputs are small, so memory-bound / iterative / graph kernels report
near-zero or no FLOP-rate (their real throughput needs the real datasets —
SuiteSparse / graphs / SIFT1M — a follow-up if wanted). Compute-bound kernels
(GEMM, SpMM, SDDMM, FFT, quantized-GEMM) give meaningful figures even on smoke.
Raw per-workload rates in `shared/kernel-bench/h100_phase2_timing.tsv` and
`h100_logs/<short>.h100.timing.log`.

| kernel | artifact | arch | H100 throughput (indicative) | gate |
|---|---|---|---|---|
| ann-search | ann-search/clover | sm_80 PTX→JIT | — | 2/2 runs valid |
| ann-search | ann-search/pathweaver | sm_80 PTX→JIT | — | 2/2 runs valid |
| attention-kernel | attention-kernel/bytetransformer | sm_90 | n/a | 0/0 runs valid (4 unsupporte |
| attention-kernel | attention-kernel/pat | sm_90 | n/a | 0/0 runs valid (4 unsupporte |
| bfs | bfs/blest | sm_80 PTX→JIT | — | 2/2 runs valid |
| bfs | bfs/efg | sm_80 PTX→JIT | — | 2/2 runs valid |
| cg-krylov | cg-krylov/millefeuille | sm_80 PTX→JIT | 0.01 GFLOP/s | 3/3 runs valid |
| cg-krylov | cg-krylov/perks | sm_80 PTX→JIT | 0.01 GFLOP/s | 3/3 runs valid |
| cholesky | cholesky/exageostat | sm_80 PTX→JIT | 0.02 GFLOP/s | 2/2 runs valid |
| cholesky | cholesky/hicma-x | sm_80 PTX→JIT | 0.01 GFLOP/s | 2/2 runs valid |
| connected-components | connected-components/ecl-scc | sm_80 PTX→JIT | — | 2/2 runs valid |
| connected-components | connected-components/yacclab | sm_80 PTX→JIT | — | 3/3 runs valid |
| fft | fft/turbofft | sm_80 PTX→JIT | 18.82 GFLOP/s | 3/3 runs valid (5 unsupporte |
| gemm | gemm/ftgemm | sm_80 PTX→JIT | 166.07 GFLOP/s | 3/3 runs valid |
| gemm | gemm/moonpoly | sm_80 PTX→JIT | 3647.22 GFLOP/s | 3/3 runs valid |
| gemv | gemv/packkv | sm_80 PTX→JIT | 5.06 GB/s | 3/3 runs valid |
| gnn-aggregation | gnn-aggregation/ge-spmm | sm_90 | 4185.29 GFLOP/s | 12/12 runs valid |
| gnn-aggregation | gnn-aggregation/stragcn | sm_90 | 856.16 GFLOP/s | 12/12 runs valid |
| graph-pattern-mining | graph-pattern-mining/fringe-sgc | sm_80 PTX→JIT | — | 2/2 runs valid |
| graph-pattern-mining | graph-pattern-mining/glumin | sm_80 PTX→JIT | — | 2/2 runs valid |
| lossless-compression | lossless-compression/gpulz | sm_80 PTX→JIT | 0.95 GB/s | 3/3 runs valid |
| lossless-compression | lossless-compression/lscomp | sm_80 PTX→JIT | 0.20 GB/s | 3/3 runs valid |
| lossy-compression | lossy-compression/cuszp | sm_80 PTX→JIT | 0.27 GB/s | 3/3 runs valid |
| lossy-compression | lossy-compression/cuszp-v1 | sm_80 PTX→JIT | 0.04 GB/s | 3/3 runs valid |
| mttkrp | mttkrp/blco | sm_80 PTX→JIT | 8.05 GFLOP/s | 3/3 runs valid |
| multigrid | multigrid/bootcmatchgx | sm_90 | 0.00 GFLOP/s | 3/3 runs valid |
| quantized-gemm | quantized-gemm/tilus | sm_90 | 385.88 GFLOP/s | 3/3 runs valid |
| sddmm | sddmm/rode | sm_80 PTX→JIT | 1545.27 GFLOP/s | 6/6 runs valid |
| sddmm | sddmm/tc-gnn | sm_90 | n/a | 0/0 runs valid (6 unsupporte |
| sparse-attention-kernel | sparse-attention-kernel/gpa | sm_90 | 2.19 GFLOP/s | 7/7 runs valid |
| spgemm | spgemm/ocean | sm_80 PTX→JIT | 10.76 GFLOP/s | 3/3 runs valid |
| spmm | spmm/dtcspmm | sm_90 | 4.96 GFLOP/s | 12/12 runs valid |
| spmm | spmm/flashsparse | sm_90 | 62.27 GFLOP/s | 12/12 runs valid |
| spmv | spmv/adaptive-spmv | sm_80 PTX→JIT | 21.35 GFLOP/s | 3/3 runs valid |
| spmv | spmv/cb-spmv | sm_80 PTX→JIT | 21.52 GFLOP/s | 2/3 runs valid |
| stencil | stencil/an5d | sm_80 PTX→JIT | 4.76 GCUP/s | 3/3 runs valid |
| string-regex-matching | string-regex-matching/gpunfa | sm_80 PTX→JIT | — | 2/2 runs valid |
| string-regex-matching | string-regex-matching/ngap | sm_80 PTX→JIT | — | 2/2 runs valid |
| triangle-counting | triangle-counting/tc-compare | sm_80 PTX→JIT | — | 2/2 runs valid |
| triangle-counting | triangle-counting/tot | sm_80 PTX→JIT | — | 2/2 runs valid |

Highlights (indicative, smoke inputs): gnn-aggregation/ge-spmm ~4.2 TFLOP/s and
gemm/moonpoly ~3.6 TFLOP/s lead; sddmm/rode ~1.5 TFLOP/s; stragcn ~0.86 TFLOP/s;
quantized-gemm/tilus ~0.39 TFLOP/s; several are sm_90 recompiles (ge-spmm,
stragcn, dtcspmm, flashsparse, tilus, gpa, bootcmatchgx) now producing H100
throughput where the A100 sm_80 binary would not have run at all. Fixed-shape
adapters (bytetransformer/pat/tc-gnn) show unsupported here because Phase 2 used
`--smoke`; their throughput needs the real workload.

## Phase 2b — REAL-dataset throughput on H100 (18 sparse/graph artifacts)

Follow-up (user: "换成真实数据集"): the sparse/graph subset re-timed on their
spec `recommended_subset` **real** inputs — SuiteSparse matrices, Planetoid
citation graphs, and SNAP graphs (SpMV/SpMM/SDDMM/SpGEMM/CG/GNN + triangle-
counting, graph-pattern-mining, connected-components) — instead of `--smoke`.
Graph exact-count kernels (TC/GPM/CC) validate a count, so the runner reports
correctness, not a FLOP-rate (throughput shown as —). Inputs are pre-fetched on the login
node into the shared cache (`KERNELBENCH_MATRIX_CACHE`; compute nodes have no
internet — an SSL/certifi fix in env.sh unblocked the download). Same H100,
warmup 5 / reps 20, no rebuild (sm_90 `.so` on disk). Indicative (clocks not
lockable). Real matrices used: bcsstk17/18/25 (CG), 192bit/wv2010/xenon2 (SDDMM),
mip1/shipsec1/pdb1HYS (SpMM), audikw_1 (77M nnz)/ldoor (SpMV), consph/cant/pdb1HYS
(SpGEMM), cora/citeseer/pubmed (GNN/SDDMM-TC). Raw: `h100_realdata_timing.tsv`.

| kernel | artifact | real matrices/graphs | H100 throughput | gate on real data |
|---|---|---|---|---|
| ann-search | ann-search/pathweaver | sift1m | 898659.87 QPS | 1/1 runs valid |
| cg-krylov | cg-krylov/millefeuille | bcsstk17,bcsstk18,bcsstk25 | — | 0/3 runs valid ⚠ |
| cg-krylov | cg-krylov/perks | bcsstk17,bcsstk18,bcsstk25 | — | 0/3 runs valid ⚠ |
| connected-components | connected-components/ecl-scc | cage14,soc-LiveJournal1,com-Youtube | — | 3/3 runs valid |
| gnn-aggregation | gnn-aggregation/ge-spmm | cora,citeseer,pubmed | 1669.96 GFLOP/s | 12/12 runs valid |
| gnn-aggregation | gnn-aggregation/stragcn | cora,citeseer,pubmed | 166.38 GFLOP/s | 5/12 runs valid |
| graph-pattern-mining | graph-pattern-mining/fringe-sgc | email-Enron,wiki-Vote | — | 2/2 runs valid |
| graph-pattern-mining | graph-pattern-mining/glumin | email-Enron,wiki-Vote | — | 2/2 runs valid |
| mttkrp | mttkrp/blco | nips,uber,chicago-crime | 1657.81 GFLOP/s | 3/3 runs valid |
| sddmm | sddmm/rode | 192bit,wv2010,xenon2 | 6311.03 GFLOP/s | 6/6 runs valid |
| sddmm | sddmm/tc-gnn | cora,citeseer,pubmed | 31.17 GFLOP/s | 6/6 runs valid |
| spgemm | spgemm/ocean | consph,cant,pdb1HYS | 148.83 GFLOP/s | 3/3 runs valid |
| spmm | spmm/dtcspmm | mip1,shipsec1,pdb1HYS | 0.06 GFLOP/s | rc=124 ⚠ |
| spmm | spmm/flashsparse | mip1,shipsec1,pdb1HYS | 359.61 GFLOP/s | 12/12 runs valid |
| spmv | spmv/adaptive-spmv | audikw_1,ldoor | 438.61 GFLOP/s | 2/2 runs valid |
| spmv | spmv/cb-spmv | audikw_1,ldoor | — | 0/2 runs valid ⚠ |
| triangle-counting | triangle-counting/tc-compare | as-caida,p2p-Gnutella31,email-EuAll | — | 3/3 runs valid |
| triangle-counting | triangle-counting/tot | as-caida,p2p-Gnutella31,email-EuAll | — | 3/3 runs valid |

**Real data is far more informative than smoke, and exposes issues smoke hid:**
* Throughput jumps to realistic levels on real matrices — sddmm/rode ~6.3 TFLOP/s,
  gnn/ge-spmm ~1.7 TFLOP/s, mttkrp/blco ~1.66 TFLOP/s (real FROSTT nips/uber/
  chicago-crime tensors), spmv/adaptive-spmv ~0.44 TFLOP/s (on 77M-nnz audikw_1),
  spmm/flashsparse ~0.36 TFLOP/s — vs the tiny smoke figures. ann-search/pathweaver
  hits ~899k QPS on real SIFT1M (recall@10 within the 0.2 floor). Graph exact-count
  kernels (TC/GPM/CC) pass on real SNAP graphs.
* **Gates that pass on smoke FAIL on real data (⚠):** cg-krylov/millefeuille &
  perks go 3/3→0/3 (fixed-iteration CG doesn't reach the reference tolerance on
  ill-conditioned bcsstk matrices — smoke's easy matrices masked it);
  spmv/cb-spmv 0/2 (its fp64 error, already 2/3 on smoke, fails outright on the
  large real matrices); spmm/dtcspmm times out (sm_90, real matrices much larger);
  gnn/stragcn 5/12 (partial). These are indicative-run observations, not
  necessarily bugs to fix — but they show real inputs are the meaningful test.

## Remaining real-dataset work (needs manual data staging)

Domains whose `recommended_subset` are large/special datasets not auto-fetchable
by SuiteSparse name: lossless/lossy-compression (SDRBench HACC/Nyx/Hurricane/CESM),
bfs (billion-edge SNAP com-Friendster), string-regex (ANMLZoo/AutomataZoo corpora),
connected-components/yacclab (image datasets). (ann-search/pathweaver SIFT1M and
mttkrp FROSTT nips/uber/chicago-crime are now staged and DONE — see table above.)
Dense/shape domains (gemm, fft,
stencil, cholesky, quantized-gemm, attention) have no external dataset — their
"real" input is problem size, already covered by Phase 2. Staging the special
datasets is the remaining follow-up if wanted.

_Generated from `h100_results.tsv` + `h100_phase2_timing.tsv` +
`h100_realdata_timing.tsv` by `shared/kernel-bench/gen_h100_doc.py`; re-run to refresh._
