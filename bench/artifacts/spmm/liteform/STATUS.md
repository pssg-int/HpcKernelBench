# liteform (LiteForm) — spmm

**Status: BUILD-FAILED (dependency unbuildable within this session's budget) — cloned and surveyed only**

- Paper: "LiteForm: Lightweight and Automatic Format Composition for Sparse
  Matrix-Matrix Multiplication on GPUs", HPDC'25.
  `PAPER_KEY = conf/hpdc/PengTPK25`.
- Artifact: https://github.com/johnpzh/liteform_ae, cloned `--depth 1` (see
  `source.provenance`).
- Rated `core` / regime `matches` for spmm (`output/kernel_centrality.json`),
  so it remains a wanted baseline under the revised selection rule.

## What the artifact actually is

`source/playground_pipline/drive05.spmm.end-to-end.sh` runs
`proc05.spmm.end-to-end.py`, which for each matrix predicts (via two
pretrained `sklearn` `RandomForestClassifier`s shipped as `.joblib` files)
whether to use the paper's own composable "CELL" format or a plain BSR
fallback, then times the resulting kernel. Both paths — `build_hyb_format`/
`bench_hyb_with_config`/`search_bucket_config` (CELL) and `bench_bsrmm`
(BSR) — are implemented entirely in `source/playground_pipline/
format_algos.py` on top of `tvm.sparse` (`FormatRewriteRule`,
`lower_sparse_buffer`, `lower_sparse_iter`, `column_part_hyb`,
`format_decompose`, `sch.get_sparse_iteration`, `sch.tensorize(...,
"wmma_sync")`) plus `from sparsetir_artifact import profile_tvm_ms`.
`format_algos.py`'s `import tvm.sparse` line and every one of those calls
require **SparseTIR** — a from-source FORK of Apache TVM
(github.com/uwsampl/sparsetir) that adds sparse-tensor TIR scheduling
primitives — not vanilla/mainline Apache TVM, which has no `tvm.sparse`
module. There is no code path in this artifact that avoids it: the paper's
whole contribution (automatic format composition/selection = the model
picking CELL vs BSR, both TVM-sparse-scheduled) is inseparable from
SparseTIR.

## Why this was not built (2026-09-05)

1. **No prebuilt distribution.** `pip index versions sparsetir` /
   `sparsetir-artifact` -> no matching distribution (checked directly, not
   assumed). `apache-tvm` (mainline, PyPI) does NOT ship `tvm.sparse` —
   confirmed by precedent in this same repo,
   `artifacts/gnn-aggregation/featgraph/STATUS.md`, which found the PyPI
   wheel's API had diverged too far from a much OLDER, much SMALLER
   TVM-version mismatch (v0.7 vs. a renamed module) to be worth patching
   around; SparseTIR is a structurally different, non-mainline API surface
   (a fork that ADDS a whole scheduling subsystem), not a renamed-function
   version drift, so the same class of problem is strictly worse here.
2. **Full from-source build required**, confirmed via SparseTIR's own
   documentation (`sampl.cs.washington.edu/SparseTIR/install.html`):
   "a fork of Apache TVM" built as a C++ shared library (CMake 3.18+, LLVM
   10+, **CUDA Toolkit 11+**) then a separate Python-bindings install step —
   no prebuilt wheel, no Docker image offered. CMake 3.28 and LLVM modules
   (18/20/21/22 via `module spider llvm`) ARE available on this machine, so
   the missing piece is not raw tooling — it is (a) the multi-hour,
   open-ended build+debug time a fork of a project the size of TVM
   realistically takes on a machine it was never built on before, most of
   which would be spent diagnosing whatever breaks first, and (b) a real,
   unverified compatibility risk: SparseTIR targets "CUDA Toolkit 11+" and
   was last actively developed years before this machine's CUDA 12.9 /
   sm_80 driver stack; TIR->CUDA codegen forks of this era commonly hardcode
   PTX intrinsics or `nvrtc`/driver-API call patterns that break silently or
   loudly on newer toolkits (the exact failure mode already hit twice
   elsewhere in this integration session for CUDA-version-sensitive
   research code: `bench/artifacts/toolchain.sh`'s own header, and
   `RoDe`/`dtcspmm`'s toolkit-pin build-system fixes).
3. **Disproportionate to this task's remaining budget** relative to the
   other two assigned artifacts (`generalsparse`, `smat`) in the same
   session — a bounded per-artifact decision, not a judgment that LiteForm
   is unimportant (it is `core`/`matches` and stays a wanted baseline).

This mirrors this track's own `smat` precedent (`../smat/STATUS.md`):
survey the real blocker precisely, record it with evidence, and defer rather
than force a rushed, undocumented attempt.

## Next attempt

Build SparseTIR from source on a machine/session with a larger time budget
specifically earmarked for it (LLVM + CMake are already available here via
`module load llvm/<ver>`): `git clone https://github.com/uwsampl/sparsetir`,
follow `sampl.cs.washington.edu/SparseTIR/install.html`'s CMake + Python-
bindings steps, verify the CUDA-11-era codegen paths against this machine's
CUDA 12.9 / sm_80 before trusting any numeric output, THEN wrap
`format_algos.py`'s format prediction (the two `.joblib` RandomForest calls)
as `prepare()`'s format-selection step and the resulting `bench_hyb_with_config`
/ `bench_bsrmm` TVM-compiled kernel launch as `run()` — exactly the split
this task specified, unimplementable only because the TVM-fork dependency
itself could not be stood up in this session.
