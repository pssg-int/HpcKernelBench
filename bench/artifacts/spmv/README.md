# spmv — artifact index

`benchspecs/spmv/spec.yaml` variants: `spmv-csr-kernel` (steady-state GPU/CPU
CSR SpMV, fp64 primary / fp32 secondary, format given/constructed once),
`spmv-e2e-preproc` (conversion timed, amortized over k=100 calls),
`spmv-sequence-krylov` (bounded sequence of related SpMV calls),
`spmv-symmetric-kernel` (half-storage symmetric SpMV, full unfolded-nnz
flop count).

## Selection rule (revised 2026-09-05)

"Newest 3 open-source artifacts per track" is retired. Baselines are now
chosen from `kernel-papers/output/baseline_selection.md`
(`select_baselines.py`, driven by per-(paper,track) ratings in
`output/kernel_centrality.json`):

1. **kernel centrality** `core` (the kernel IS the paper's headline
   contribution, evaluated at kernel level) > `component` (a part of a
   bigger system/pipeline) — `tangential` is never a baseline;
2. **regime match** with the track spec's inputs (general SuiteSparse,
   dense x, single NVIDIA GPU, fp64/fp32): `matches` > `partial` >
   `mismatch`;
3. **single-NVIDIA-GPU path** required (current scope);
4. **recency** only as the tiebreak; up to 5 per track.

Already-integrated adapters the rule would not have picked (`sspmv`,
`diaq`) stay in the registry as competitors under the same gate, but are
labelled `component`/off-regime here and in their own `STATUS.md` rather
than treated as the track's "human SOTA" reference.

For spmv, `output/baseline_selection.md` names 3 papers to integrate this
session — `cb-spmv` (already integrated, prior session), `tilespmv`,
`spmv-acc` — plus `journals/tpds/LiAY21` (Adaptive SpMV/SpMSpV, `core`,
regime `partial`) as the next-ranked eligible candidate, integrated in this
session as `adaptive-spmv` (dense-vector path only, per its own centrality
note: "emphasis on varied input-vector sparsity is narrower than the
spec's plain dense-vector SpMV variant").

## Artifacts

| short | paper / venue | centrality / regime | status | precision / variant |
|---|---|---|---|---|
| **cb-spmv** | CB-SpMV, ICS'25 (`conf/ics/CongSC00Q25`) | core / matches | BUILT+GATED, clean pass | fp64 / `spmv-csr-kernel` |
| **tilespmv** | TileSpMV, IPDPS'21 (`conf/ipps/NiuLDJ0T21`) | core / matches | BUILT+GATED (partial) — exact on structured inputs (err=3.2e-16); wrong-answer on wide-column-fanout synthetic inputs; **crashes (heap corruption in the artifact's own `Tile_create`)** on both real matrices tried (`webbase-1M`, `cant`) | fp64 / `spmv-csr-kernel` |
| **adaptive-spmv** | Adaptive SpMV/SpMSpV, TPDS'21 (`journals/tpds/LiAY21`) | core / partial | BUILT+GATED, clean pass (dense-vector `HolaSpmv` path only — SpMSpV and the paper's ML selector are out of scope) | fp64 / `spmv-csr-kernel` |
| **spmv-acc** | spmv-acc, HPDC'23 (`conf/hpdc/ChuHDDC0WH23`) | core / matches | BUILD-FAILED — real CUDA path exists (`cuda-hipcc` CMake preset, zero AMD-only deps in the kernel library) but this machine's only HIP toolchain (`hip/5.5.1`) is compile-incompatible with every CUDA toolkit available here (see `STATUS.md`) | n/a |
| **sspmv** | SSpMV (LeSpMV), DAC'25 (`conf/dac/LinLDZLY25`) | core / matches, **no single-GPU path** (CPU-only) | BUILT+GATED, clean pass — kept as competitor, not SOTA baseline | fp64 / `spmv-csr-kernel` (CPU) |
| **diaq** | DiaQ (for quantum Hamiltonian sim), ICS'26 (`conf/ics/ChunduryBLSM26`) | component / mismatch | BUILT (gate blocked on the mandated `cant` matrix; substitute gate PASSES) — kept as competitor, not SOTA baseline | fp64 |
| **rassm** | RASSM, ASPLOS'25 (`conf/asplos/JainGC25`) | tangential / mismatch | SKIPPED — public artifact implements SpMM/SDDMM only, zero SpMV in the codebase (`grep -ril spmv .` from repo root: no hits) | n/a |
| **bit-graphblas** | Bit-GraphBLAS, IPDPS'22 (`conf/ipps/ChenSSTBL22`) | core / mismatch | SKIPPED (cheap, no clone) — 1-bit/binary-only B2SR format; `benchspecs/spmv/survey.md` explicitly excludes this 521-matrix binary domain from the general fp SpMV variant (1-bit storage changes the byte-accounting model) | n/a |

`tilespmv`, `adaptive-spmv`, `spmv-acc` were integrated in this session
under the revised rule (full provenance, patches, and gate evidence in
each `STATUS.md`). `cb-spmv`/`sspmv`/`diaq`/`rassm` were integrated in an
earlier session (see each `STATUS.md`); `bit-graphblas` was never cloned
(regime mismatch established directly from the spec survey).

## Cross-artifact findings from this session

**TileSpMV: a genuine artifact bug, found by direct measurement, corrected
mid-integration.** TileSpMV vendors a copy of Weifeng Liu's CSR5 SpMV to
handle nonzeros deferred from "very sparse" 16x16 tiles — but those same
nonzeros are ALSO retained in the tile's own in-tile COO storage, handled
by the main tile kernel. Calling both (as the artifact's own warmup loop
does) double-counts; the artifact's actual MEASURED loop already avoids
this by commenting out the CSR5 call, which turned out to be the correct
behavior, not a missing correction as first assumed. Full account,
including the exact `y = 2*y_ref` measurement that disproved the initial
assumption, in `tilespmv/STATUS.md` and `tilespmv/bridge.cu`. Separately,
TileSpMV's own `Tile_create` format-conversion code has a real
heap-corruption bug that crashes on both real SuiteSparse matrices tried
on this machine (CUDA 12.9 / current glibc) — reported as a gate failure,
not patched (rule 3: no kernel-code patches).

**spmv-acc: a real CUDA path blocked by a HIP-toolchain version wall, not
by the artifact.** The repo is HIP-first but genuinely supports a
single-NVIDIA-GPU build via `hipcc`'s NVIDIA backend, with zero AMD-only
library dependencies in the kernel code itself. This machine's only
available HIP module (5.5.1) fails to compile against every CUDA toolkit
available here — CUDA 12.9 (this repo's usual pin) because HIP 5.5.1's own
vendored NVIDIA shim header calls CUDA-Graph APIs with signatures CUDA 12.x
changed, and CUDA 11.7 (the version HIP 5.5.1 was actually built against)
because its `nvcc` predates this machine's system glibc/GCC 14. Neither
failure is inside any spmv-acc source file. Full evidence in
`spmv-acc/STATUS.md`.
