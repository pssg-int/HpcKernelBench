# spmm — artifact index

`benchspecs/spmm/spec.yaml` variants: `spmm-gpu-kernel-f32` (steady-state GPU
CSR SpMM, fp32, format given), `spmm-gpu-e2e-preproc-f32` (conversion timed,
amortized over k=100 calls), `spmm-cpu-kernel-f32`, `spmm-gpu-quantized-int`
(int4/int8 Tensor-Core), `spmm-tensorcore-fp16` (fp16/tf32 compute, fp32
accumulate — added 2026-08-07 specifically to give Tensor-Core-`half`/`tf32`
kernels a fair, disclosed home instead of failing the fp32 variant's 1e-4
tolerance by construction), `spmm-binary-adjacency-kernel` (0/1
unweighted-adjacency SpMM — added 2026-09-06, see "Pattern-only SpMM
variant" below).

## Selection rule (revised 2026-09-05)

Earlier integrations in this track (`insum`, `inferfast`, `rassm`) were
picked by recency alone. The user's revised ruling: **baselines must be core
kernel papers evaluated on the spec's own input regime** — a hand-optimized
kernel implementation (not a DSL/compiler demo, not a narrow
application-specific side-case) targeting this track's actual default
claim (general, unstructured-sparsity, single-GPU-card SpMM, fp32 or its
natural Tensor-Core counterpart) — with recency used **only** as a tiebreak
among otherwise-equally-central candidates. Concretely:

- **core** — the paper's primary contribution IS a general-purpose SpMM
  kernel for exactly this track's default regime (unstructured sparsity,
  single NVIDIA GPU, fp32 CUDA-core or fp16/tf32 Tensor-Core), and the
  artifact is cited as a baseline by later papers in the same line.
- **component** — a real, legitimately-measured kernel implementation, but
  for a narrower or auxiliary regime within the track (structured/quantized
  sparsity, a compiler/DSL-generated kernel, CPU-only, or an
  application-specific weight-sparsity kernel) rather than the track's
  line-defining general case.
- **off-regime** — does not run, or does not implement this track's kernel
  at all, on the hardware/regime this integration targets (e.g. a
  next-generation-GPU-only kernel that hard-errors on this machine's A100).

## Artifacts

| short | paper / venue | centrality | status | precision / variant |
|---|---|---|---|---|
| **rode** | RoDe, PPoPP'24 (`conf/ppopp/PangFQZL24`) | core | BUILT+GATED, clean pass | fp32 / `spmm-gpu-kernel-f32` |
| **dtcspmm** | DTC-SpMM, ASPLOS'24 (`conf/asplos/Fan0024`) | core | BUILT+GATED on `spmm-binary-adjacency-kernel` (its actual regime, clean pass, 2026-09-06); `spmm-tensorcore-fp16` (general/weighted): gate FAILS by construction, disclosed | tf32 / `spmm-binary-adjacency-kernel` (general variant: `spmm-tensorcore-fp16`) |
| **flashsparse** | FlashSparse, PPoPP'25 (`conf/ppopp/ShiLXFWW25`) | core | BUILT+GATED on `spmm-binary-adjacency-kernel` (its actual regime, clean pass, 2026-09-06); `spmm-tensorcore-fp16` (general/weighted): gate FAILS by construction, disclosed | fp16 / `spmm-binary-adjacency-kernel` (general variant: `spmm-tensorcore-fp16`) |
| **sspmm** | Ro-SpMM/SSpMM, TPDS'25 | core | BUILT+GATED, clean pass (severe padding overhead on unstructured graphs noted) | fp16 / `spmm-tensorcore-fp16` |
| **mp-spmm** | (2:4 structured sparsity), SC'25 | component | BUILT+GATED — **confirms a real N=128 accuracy bug** in the artifact's own kernel | fp16-class / `spmm-tensorcore-fp16` |
| **insum** | Insum / IndirectEinsum, ASPLOS'26 | component | BUILT+GATED | fp32 / `spmm-gpu-kernel-f32` |
| **inferfast** | InferFast, ICS'26 | component | BUILT (gate fails: fp16 kernel vs. fp32-calibrated variant — expected, see STATUS.md) | fp16 / `spmm-gpu-kernel-f32` (no dedicated fp16 variant existed when this was integrated) |
| **rassm** | RASSM, ASPLOS'25 | component | BUILT+GATED (CPU) | fp32 / `spmm-cpu-kernel-f32` |
| **voltrix** | Voltrix, ATC'25 | core / matches | **DEFERRED-HARDWARE (needs sm_90a / Hopper; see `voltrix/REQUIRES_GPU`)** — hardcoded `compute_90a`/`sm_90a`, genuine Hopper-only TMA/`mbarrier` PTX; `cudaErrorNoKernelImageForDevice` on this machine's A100 even in preprocessing | n/a |
| **smat** | SMaT, SC'24 | core / matches | BUILT — **gate FAILS (0/9 smoke, 0/1 cant)**; confirms a real B-operand indexing bug** in `mmaCBTKernelSparse` for any K>16, masked by the artifact's own hardcoded-all-ones B (dead random-fill code) | fp16 / `spmm-tensorcore-fp16` |
| **generalsparse** | GeneralSparse, ATC'25 (`conf/usenix/WangGXCT25`) | core / matches | BUILT+GATED on `spmm-binary-adjacency-kernel` (its actual regime, clean pass incl. real matrix `cant`, 2026-09-06; one fixed operator composition out of the artifact's own ~30-strategy search, `cora` still UNSUPPORTED there — unrelated to values); `spmm-tensorcore-fp16` (general/weighted): gate FAILS by construction, disclosed | fp16 / `spmm-binary-adjacency-kernel` (general variant: `spmm-tensorcore-fp16`) |
| **liteform** | LiteForm, HPDC'25 (`conf/hpdc/PengTPK25`) | core / matches | BUILD-FAILED — needs SparseTIR, a from-source TVM fork (no pip wheel; CUDA-11-era codegen, unverified against this machine's 12.9); deferred, not attempted, see STATUS.md | n/a |
| **nm-spmm** | NM-SpMM, IPDPS'25 (`conf/ipps/MaWDCH0ZWZCDWFP25`) | core / partial (needs an already N:M-structured sparse operand; no general SuiteSparse matrix satisfies it) | BUILT+GATED — wraps `nmGEMM_small_matrices_low_sparsity` (`kernel_32x32_4x4`, sparsity=0.5 i.e. 16-of-32, the artifact's own `Ks=32` block width, NOT the fine-grained "2:4" most N:M literature means) + the artifact's own `PreProcessing_low_sparsity`. 4/4 valid on a standalone synthetic N:M workload built to satisfy the structural contract (precedent: `../spmv/diaq/STATUS.md`'s substitute gate); `max_scaled_err` 1.5-2.1e-7 (tol 1e-4). General SuiteSparse/smoke matrices correctly raise `NotImplementedError` (verified against `smoke-uniform`). Not Hopper-only (sm_80 builds fine) — not DEFERRED-HARDWARE. Real precision finding: the task brief expected fp16 Tensor Cores; the artifact is plain fp32 CUDA-core (no wmma/mma.sync anywhere in source). | fp32 / `spmm-gpu-kernel-f32` (substitute gate, see STATUS.md) |
| **sputnik** | Sputnik (Gale et al.), SC'20 (`conf/sc/GaleZYE20`) | core / mismatch (own regime is DNN-pruned weights at moderate sparsity, not general SuiteSparse) | BUILT+GATED, clean pass — wraps `sputnik::CudaSpmm` (fp32 CSR x dense, `bias=nullptr`) + a byte-for-byte port of `SortedRowSwizzle` preprocessing (Glog/Abseil-free port; `matrix_utils.cu.cc`'s other code needs both, unvendored here, for unrelated test-data generation). **Provenance ruling**: the paper table's listed URL (`luckylsk34/Sparse-Kernels`) is a stripped, renamed, single-commit fork of `google-research/sputnik` with Apache-2.0 license notices removed — cloned the canonical repo instead (pinned commit `bbf5840`), documented in `source.provenance`/STATUS.md. 12/12 smoke + `cant` 1/1 + `cora` 1/1 valid, `max_scaled_err` ~2-4e-7 (tol 1e-4). Every N in {32,128,256,512} works directly (Sputnik's own n%4/n%2 dispatch), no dense-width restriction unlike RoDe/SSpMM/NM-SpMM. Also wired for sddmm (`sputnik::CudaSddmm`) — see `../sddmm/README.md`. | fp32 / `spmm-gpu-kernel-f32` |
| **tc-gnn** | TC-GNN, USENIX ATC'23 (`conf/usenix/WangFWHD23`) | core / matches | BUILT+GATED (reused build) — `source/`/compiled `.so` REUSED from `../../gnn-aggregation/tc-gnn/` (no second clone or compile); wraps the weighted `forward_AGNN` WMMA SpMM kernel, fed this track's own arbitrary CSR values (no normalization, unlike the gnn-aggregation adapter). 1/1 valid on `cora` N=128 (`max_scaled_err=4.87e-04`, tol `1e-2`) under BOTH `spmm-tensorcore-fp16` and the concurrently-added `spmm-binary-adjacency-kernel` (per-precision tolerance `tolerance_for("fp16")==1e-2`); every `--smoke` workload and every N>128 UNSUPPORTED via two confirmed guards reproduced from the gnn-aggregation build (`M%16==0` heap corruption; `N>128` silent zero-fill). | fp16 (tf32 compute) / `spmm-tensorcore-fp16`, also gated under `spmm-binary-adjacency-kernel` |
| **ge-spmm** | GE-SpMM, SC'20 (`conf/sc/HuangD0Y20`) | core / matches | BUILT+GATED (reused build) — `source/`/compiled `build/spmm.so` REUSED from `../../gnn-aggregation/ge-spmm/` (no second clone or compile); wraps `csr_spmm(rowptr,colind,values,dense)`, a genuine weighted fp32 CSR SpMM, values consumed directly (no normalization needed — spmm's own reference is plain `C=A@B`). 20/20 valid across smoke (12) + `cant`,`cora` (8) at N in {32,128,256,512}, `max_scaled_err` 1.45e-07 – 4.19e-07. No N restriction found (checked all four dims against the artifact's own general 3-tier dispatch, unlike RoDe's N in {32,128} or TC-GNN's N<=128). | fp32 / `spmm-gpu-kernel-f32` |

`rode`/`dtcspmm`/`flashsparse` were integrated in this session as the
canonical general-GPU-SpMM line under the revised rule (see each
`STATUS.md` for full provenance, patches, and gate evidence).
`sspmm`/`mp-spmm`/`voltrix`/`smat` (surveyed) were integrated in parallel by
a sibling agent in this same session (see each `STATUS.md`).
`generalsparse`/`liteform`/`smat` (built) were completed in a follow-up
session (see each `STATUS.md` for full provenance, patches, and gate
evidence) — `smat`'s CMake/gflags blocker noted above turned out to be
avoidable by wrapping at a finer boundary (see `smat/STATUS.md`).

## Pattern-only SpMM variant (added 2026-09-06)

`dtcspmm`, `flashsparse`, and `generalsparse` (see the cross-artifact finding
below) each hardcode the sparse operand's values to 1.0 in their released
code, so none of them can ever pass a general weighted-SpMM gate
(`spmm-gpu-kernel-f32`, `spmm-tensorcore-fp16`) no matter how correct their
kernel is — they compute `A_pattern @ B`, not `A @ B`. That is nonetheless a
real, GNN-relevant regime (message-passing aggregation on a plain
0/1 adjacency matrix; all three papers' own evaluations use exactly such
graphs), so `benchspecs/spmm/spec.yaml` gained a dedicated
`spmm-binary-adjacency-kernel` variant: the harness
(`kernelbench.domains.sparse.variant_transform`, wired into
`kernelbench/runner.py`) forces every workload's stored values to 1.0 for
**every** implementation gated under this variant, including the fp64 CSR
reference, before either sees it. Weighted and pattern-only kernels
therefore compete on the exact same (binarized) input here.

**Who competes**: the three pattern-only kernels above, plus any weighted
GPU SpMM kernel that also has an adapter in this track — `rode-spmm`,
`sspmm-rospmm-v8`, `inferfast-spmm-splitk` were verified to pass this
variant's gate cleanly (smoke + `cora`); the built-in `cusparse-csr-spmm`/
`custom-warp-csr-spmm` currently do NOT (see "Known limitation" below — a
pre-existing bug unrelated to this variant).

**Fairness rule**: numbers measured under `spmm-binary-adjacency-kernel` are
commensurable ONLY with each other. They must never be placed next to
`spmm-gpu-kernel-f32`/`spmm-tensorcore-fp16` numbers for the same
implementation — those variants require real-valued `A`, this one
deliberately does not. A weighted kernel that reads values it structurally
does not need on this variant is simply slower here, which is a fair
outcome, not a mismatch (`spec.yaml`'s `notes_on_fairness`).

**Known limitation (pre-existing, not introduced by this variant)**: the
built-in `TorchSpMM`/`CustomSpMM` implementations
(`kernelbench/impls/gpu_cuda.py`) generate their dense operand `B` with
`torch.Generator`, while the CPU reference (`cpu_ref._dense_operand`) uses
`numpy.random.default_rng` — the two RNGs produce different `B` matrices for
the same seed, so `cusparse-csr-spmm`/`custom-warp-csr-spmm` fail the
correctness gate on EVERY spmm variant (confirmed: also reproduces under the
unmodified `spmm-gpu-kernel-f32`), not specifically because of the new
binary-adjacency variant or its `variant_transform` hook. Left unfixed here
as out of this task's scope (a `gpu_cuda.py` change would affect every spmm/
spmv/sddmm variant, not just this one); recorded so the two built-ins'
absence from `spmm-binary-adjacency-kernel`'s passing set is not mistaken
for a variant-specific problem.

## Cross-artifact finding: the TC-GNN/DTC-SpMM lineage is structurally binary-pattern-only

Both `dtcspmm` and `flashsparse` (independently, by construction — traced in
each artifact's own preprocessing code, not inferred from the gate numbers
alone) ship Python-exposed SpMM APIs that discard real nonzero VALUES and
substitute a hardcoded 1.0 for every stored position. Both were gated at
`spmm-tensorcore-fp16`: PASS on unweighted/binary graphs already in this
track's own recommended matrix set (cora, err ~4-8e-3 for both, right at
fp16/tf32 unit-roundoff), FAIL by construction on general real-valued
matrices and this track's synthetic smoke set (`max_scaled_err` in the
2.9-5.7 range for both — essentially identical magnitudes, independent
confirmation of the same root cause). This traces to their shared TC-GNN
ancestry (both papers' preprocessing pipelines were built for GNN-adjacency
aggregation, never generalized to arbitrary values at the API level despite
each paper's title claiming "general" SpMM) and is disclosed in full in
each artifact's own STATUS.md rather than papered over with a loosened
gate (ARTIFACT_GUIDE.md rule 4).

`generalsparse` independently exhibits the SAME class of behavior via a
DIFFERENT root cause (dead code in `source/struct.cc`'s `.mtx` reader
hardcodes every value to 1 rather than using the parsed field — not a
deliberate GNN-adjacency design choice like dtcspmm/flashsparse's) — a
third, independently-arrived-at confirmation that this track's "general
SpMM" claims recur to unweighted-graph-only implementations more often than
their abstracts suggest. See `generalsparse/STATUS.md` for the exact
file:line evidence. Deliberately left unpatched for the same reason: a
trivial one-line fix exists, but applying it to only one of three artifacts
sharing this bug class would make cross-artifact comparison less honest,
not more.

`smat` adds a distinct, more serious finding: not a preprocessing/API
choice but a genuine indexing bug in the kernel itself (`mmaCBTKernelSparse`
reads the wrong slice of the dense operand B for any matrix wider than one
16-column tile), invisible under the artifact's own evaluation only because
its dense-operand generator's random-fill code is commented out in favor of
a hardcoded all-ones fill. See `smat/STATUS.md` for the full derivation and
an isolated, gate-independent reproduction.
