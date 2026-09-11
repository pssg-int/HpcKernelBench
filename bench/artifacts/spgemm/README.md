# spgemm — artifact index

`benchspecs/spgemm/spec.yaml` variants: `spgemm-square-kernel-f64`
(steady-state device SpGEMM for C=A^2, fp64, symbolic+numeric+C-alloc
timed, format conversion excluded/reported separately),
`spgemm-e2e-preproc-f64` (conversion amortized over k=100 calls),
`spgemm-tensorcore-mixed-precision` (fp16/mixed-precision variant),
`spgemm-distributed-strong-scaling-f64` (multi-GPU, out of this scope's
single-card requirement).

## Selection rule (revised 2026-09-05)

"Newest 3 open-source artifacts per track" is retired. Baselines are now
chosen from `kernel-papers/output/baseline_selection.md`
(`select_baselines.py`, driven by per-(paper,track) ratings in
`output/kernel_centrality.json`):

1. **kernel centrality** `core` (the kernel IS the paper's headline
   contribution, evaluated at kernel level) > `component` (a part of a
   bigger system/pipeline) — `tangential` is never a baseline;
2. **regime match** with the track spec's inputs (general SuiteSparse
   A^2, fp64, single NVIDIA GPU): `matches` > `partial` > `mismatch`;
3. **single-NVIDIA-GPU path** required (current scope);
4. **recency** only as the tiebreak; up to 5 per track.

For spgemm, `output/baseline_selection.md` lists 6 eligible papers, 2
named "to integrate": `ocean` (already integrated, prior session) and
`TileSpGEMM` (`core`/`matches` — "one of the deeply-surveyed papers that
directly defines the spec's square-kernel variant"), integrated in this
session as `tilespgemm`. The other 4 eligible entries (Bit-GraphBLAS —
regime mismatch/binary-only format out of scope; ToT — component/partial,
triangle-counting-via-tensor-cores not general SpGEMM; Popcorn — listed
`[integrated]` in `baseline_selection.md` but its own `STATUS.md` found no
actual SpGEMM kernel in the artifact, SKIPPED; `amgt` — demoted,
component/mismatch, kept as a competitor) were not newly touched this
session.

## Artifacts

| short | paper / venue | centrality / regime | status | precision / variant |
|---|---|---|---|---|
| **ocean** | Ocean-SpGEMM, ICS'26 (`conf/ics/LiG26`) | core / matches | BUILT+GATED, clean pass (3/3 smoke, ~1e-16 err) — 1 real bug found+fixed at the wrapper boundary (missing CSR row-offset sentinel) | fp64 / `spgemm-square-kernel-f64` |
| **tilespgemm** | TileSpGEMM, PPoPP'22 (`conf/ppopp/NiuLJS0022`) | core / matches | BUILT+GATED (partial) — mechanical wrapping verified correct (dims/nnz/near-perfect symbolic pattern), but the artifact's own numeric kernel produces wrong values (synthetic) and out-of-superset column indices (2 real matrices: `cant`, `pdb1HYS`) | fp64 / `spgemm-square-kernel-f64` |
| **amgt** | AmgT, SC'24 (`conf/sc/LuZWFLCY0C024`) | component / mismatch, demoted | BUILT+GATED — kept as competitor, not a SOTA baseline | fp64/mixed / `spgemm-tensorcore-mixed-precision` |
| **popcorn** | Popcorn, PPoPP'25 (`conf/ppopp/BellavitaPMVG25`) | component / mismatch | SKIPPED — artifact reformulates Kernel K-means as sparse-times-DENSE (cuSPARSE SpMM/SpMV), not SpGEMM (sparse x sparse); no working SpGEMM kernel exists in the codebase | n/a |

`tilespgemm` was integrated in this session (full provenance, patches,
and gate evidence in `tilespgemm/STATUS.md`); `ocean`/`amgt`/`popcorn`
were integrated/assessed in an earlier session (see each `STATUS.md`).

## Cross-artifact findings from this session

**TileSpGEMM: a genuine, reproducible numeric/indexing bug, isolated by
elimination.** The bridge and build were verified mechanically correct by
ruling out every plausible confounder one at a time: identical wrong output
at `-O3` and `-O0` (not a compiler-optimization artifact); identical wrong
output with `OMP_NUM_THREADS=1` and with OpenMP entirely compiled out (not
a threading race); the bridge's A/B host-array aliasing was proven safe by
reading `csr2tile_row_major`/`csr2tile_col_major`'s full bodies line by
line (both only READ the shared CSR fields); deterministic across every one
of the above configurations. On real SuiteSparse matrices (`cant`,
`pdb1HYS`), the artifact's own kernel produces output at column positions
**structurally outside `|A|@|A|`'s mathematically guaranteed superset** —
not a formatting mismatch, a genuine indexing bug, clustered at tile-column
boundaries (`BLOCK_SIZE=16`) in both the real-matrix and a small synthetic
control test. No timing number was obtained. Full account, including the
abandoned raw-CLI cross-check attempt (segfaults on this login node for
reasons unrelated to TileSpGEMM's own logic), in `tilespgemm/STATUS.md`.

No `source/`-tracked file needed a line-level patch for `tilespgemm` (2
build-system/compat fixes only: `__shfl`→`__shfl_sync` shim, forcing the
system `cuda_fp16.h` ahead of the artifact's own vendored CUDA-11-era
copy).
