# MP-SpMM — STATUS

**Status: BUILT+GATED — reproduces a real accuracy bug in the artifact's
own N=128 kernel; N=32 (the artifact's only other supported size) is
correct.**

- Paper: "Bridging the Gap between Unstructured SpMM and Structured Sparse
  Tensor Cores", SC'25. `PAPER_KEY = conf/sc/DongS0LXHZ025`.
- Artifact: https://github.com/CGCL-codes/MP-SpMM_SC25 — this repo contains
  only a `README.md` pointing at the paper's Zenodo AD/AE release
  (`doi:10.5281/zenodo.16933452`, `MP-SpMM_SC25.zip`, md5
  `7aacfbc60cdc0c535bf666538cbe2046`, 113 MB, v3/2025-08-23). Fetched and
  verified against the published md5; see `source.provenance` (not a git
  clone — this is a code-drop release, `source/.git` does not exist).
- Toolchain: `nvcc` 12.9, host compiler `g++-12` (SUSE 12.3.0) —
  `-gencode arch=compute_80,code=sm_80`; preprocessing tool compiled with
  plain `g++-12 -std=c++17`. Python
  `/pscratch/sd/c/cunyang/gnn/plexus_env/bin/python`, torch `2.8.0+cu128`.
- Selection rationale: core general-GPU-SpMM baseline under the revised
  kernel-centrality rule (matching+padding to convert unstructured SpMM into
  Ampere's 2:4 structured sparse-tensor-core format, evaluated on the spec's
  own SuiteSparse/GNN input regime) — and the user's own group had already
  flagged a suspected N=128 accuracy bug in this artifact, which this
  integration exists in part to confirm or refute independently.

## What the artifact actually is

`source/mpspmm/preprocessing/impl-iterative-2-4.cpp` runs a greedy row-pair
MATCHING over the sparsity graph (pair rows whose union of nonzero columns
fits the Ampere 2:4 constraint within each 16-wide tile) and zero-PADS the
result into a `(opdA, metadata, tcblocks_offset, old_col_all)` layout.
`source/mpspmm/SpMM/kernels.cu` then runs one of two hand-written
`mma.sp.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32` (fp16 compute, fp32
accumulate) kernels against that layout, selected by N:
`sparse_mma_kernel_base_Bhalf2_Cfloat4` for N=32,
`sparse_mma_kernel_base_Buint64_Cfloat4` for N=128. The artifact's own
`main()` (`source/mpspmm/SpMM/spmm_sp_new.cu`) has **no kernel for any other
N** — `if (N==32) ... else if (N==128) ...` with no `else`, so any other N
would silently launch nothing in the original code.

## What was wrapped / built (ARTIFACT_GUIDE.md rules 1-3)

- **Preprocessing**: `mpspmm_preprocess`, compiled unmodified from
  `source/mpspmm/preprocessing/impl-iterative-2-4.cpp` by `build.sh`.
  `adapter.py`'s `prepare()` writes our workload's CSR to a Matrix-Market
  file at the exact path the tool expects (`get_config_value("dataset_path")`
  + a hardcoded `"dataset"` directory + `<name>.mtx` — a **flat** layout,
  not `<name>/<name>.mtx` as first guessed; corrected from the tool's own
  `"File ... does not exist"` error during gating, see commit history of
  this file), then runs it as a subprocess from a working directory two
  levels below a `path.txt` this adapter writes into its own `work/` scratch
  tree (never under `source/`) — timed as part of `prepare()`, per rule 2.
  This is the paper's real matching+padding contribution; nothing here
  reimplements it.
- **Kernel**: `libmpspmm_wrapper.so` (`wrapper.cu`, this directory, NOT under
  `source/`). The artifact ships no library — `spmm_sp_new.cu` is a
  standalone `main()` reading a file from disk — so `wrapper.cu` `#include`s
  `source/mpspmm/SpMM/kernels.cu` **unmodified** (exactly as `spmm_sp_new.cu`
  itself does) to get the two `__global__` kernels, and reproduces `main()`'s
  exact grid/block/`cudaFuncSetAttribute` launch configuration in an
  `extern "C" mpspmm_run(...)` entry point. It also carries a **byte-for-byte
  copy** of `spmm_sp_new.cu`'s inline `valueToStorage`/`storeArrayInUint32`/
  `storeArrayInUint32ForLargeMatrix` metadata-packing functions (only the
  `std::vector<int>` parameter was changed to a raw pointer+length pair for
  ctypes callability — no logic changed) — these functions are defined
  inline in the same file as the artifact's own `main()` and cannot be
  linked against directly, the same situation `inferfast/wrapper.cu` handled
  by re-declaring `reorder_matrices` against an already-compiled symbol; here
  there is no separately-compiled object to link against, so the function
  bodies had to be copied rather than merely re-declared. Full reasoning is
  in `wrapper.cu`'s header comment.
- `adapter.py`'s `prepare()` also does the artifact's own NaN->0 (`opdA`) and
  `-1`->`0` (`old_col_all`) cleanup identically to `spmm_sp_new.cu main()`,
  then H2D-uploads everything. `B` is drawn with
  `numpy.random.default_rng(seed).uniform(-1,1,(K,N))`, matching
  `cpu_ref.reference_spmm`'s `_dense_operand` exactly (see `insum`/
  `inferfast` STATUS.md for why this specific RNG match matters).
- **PRECISIONS = ["fp16"]**; gated under `spmm-tensorcore-fp16`
  (`--precision fp16`), the variant this task specifies for tensor-core
  artifacts (tolerance `0.01`, parsed cleanly per `--list`).

## Gate result (login node, functional check only)

```
source bench/artifacts/toolchain.sh
LD_PRELOAD=/usr/lib64/libstdc++.so.6 $PY -m kernelbench.runner --kernel spmm \
    --variant spmm-tensorcore-fp16 --impl mpspmm-2-4-iterative \
    --precision fp16 --smoke --warmup 1 --reps 2
```

**N=128 (the variant's default sweep includes N=128, from `dense_dims:
[128, 256, 512]`): FAILS on all 3 smoke matrices**, by 200-300x over the
gate:

| matrix | max_scaled_err | tol |
|---|---|---|
| smoke-uniform | 2.225e+00 | 0.01 |
| smoke-banded | 2.245e+00 | 0.01 |
| smoke-powerlaw | 2.747e+00 | 0.01 |

An error of O(1-3) against a `|A|*|B|` componentwise-backward-error
denominator is not fp16 rounding noise (fp16 unit roundoff is `~5e-4`,
confirmed below) — this is a real, wrong-output bug in
`sparse_mma_kernel_base_Buint64_Cfloat4`, independently reproducing the
prior finding from the user's own group.

**N=256 and N=512 (also in the variant's sweep): correctly `UNSUPPORTED`**
(`prepare()` raises `NotImplementedError`, per rule 8 — the artifact has no
kernel for these N; the runner records them as unsupported, not failed).

**N=32 (the artifact's ONLY other supported size, outside this variant's
dim list — checked manually per the task's instruction to "check other N"),
`--dims 32`: PASSES cleanly, fp16-typical magnitude**:

| matrix | max_scaled_err | tol | throughput |
|---|---|---|---|
| smoke-uniform | 3.12e-04 | 0.01 | 18.64 GFLOP/s |
| smoke-banded | 3.02e-04 | 0.01 | 19.64 GFLOP/s |
| smoke-powerlaw | 3.94e-04 | 0.01 | 22.96 GFLOP/s |

(reduced-protocol numbers only — warmup=1, reps=2, shared login-node GPU,
non-conforming, not a timing result per rule 5.)

### Smallest failing case

Isolated with a standalone script calling `adapter.py` directly on hand-built
tiny matrices (bypassing the harness's matrix loader, same reference formula
`check_correctness` uses):

| M | K | nnz/row | N | max_scaled_err | verdict |
|---|---|---|---|---|---|
| 16 | 16 | 4 | 128 | 5.291e+00 | **FAIL** |
| 16 | 16 | 4 | 32 | 6.745e-04 | ok |
| 16 | 32 | 4 | 128 | 1.309e+01 | FAIL |
| 32 | 32 | 4 | 128 | 1.768e+01 | FAIL |

**M=16, K=16 (a single 16x16 tile — the smallest possible input the kernel
accepts at all, since M is padded to a multiple of 16), nnz_per_row=4, N=128
already fails at `max_scaled_err=5.29`, while the IDENTICAL matrix at N=32
passes at `6.7e-04`.** This isolates the bug precisely to
`sparse_mma_kernel_base_Buint64_Cfloat4` itself (the N=128 kernel) — not the
matching/padding preprocessing (shared unchanged between both N values and
demonstrably correct when paired with the N=32 kernel), not matrix size, and
not a preprocessing/data-layout mismatch in this adapter. A single tile is
already enough to trigger it, which also means the bug is very unlikely to
be a rare corner case at scale — every N=128 run over any real matrix in the
spec's recommended_subset should be expected to fail the same way (not
verified at scale, out of the login-node budget, but the single-tile isolation
makes that the natural expectation).

**Per ARTIFACT_GUIDE.md rule 4 ("if it fails the gate, that IS a result —
record it, do not loosen the gate to make it pass") and the task's explicit
instruction: the gate was NOT loosened.** `mpspmm-2-4-iterative` is a valid,
correctly-wired competitor that produces a genuinely wrong answer at N=128;
its N=32 result is a legitimate, correct data point outside this variant's
default sweep.

## Not done

- No sweep across the recommended_subset matrices or across N=32 timing
  (out of scope per the task's login-node budget: gate-only checks).
- The C/C++ preprocessing tool's other two matching strategies (adjacent-
  matching, max-matching) and the CUDA-preprocessing variants were not
  wired — iterative-matching was sufficient to reproduce and isolate the
  bug, which is the priority this integration was scoped around.
- Did not attempt to patch the N=128 kernel — ARTIFACT_GUIDE.md rule 3
  prohibits touching kernel code, and the point of this integration is to
  record the artifact's real, as-shipped behavior, not a corrected version
  of it.

## Reproduction on zaratan (2026-09-09)

- Machine: UMD zaratan, A100 MIG 1g.5gb slice (sm_80), login-node build.
- Toolchain: nvcc 12.8.93 (conda-forge CUDA 12.8), g++ 12.4.0 (conda-forge
  `kb-gcc12` env, via `KB_GXX12`, for both the preprocessing tool and the
  wrapper's `-ccbin`), torch 2.8.0+cu128, Python 3.12.14, arch
  `compute_80,code=sm_80` as recorded.
- Build: OK (only advisory `ptxas` notes about `.sp::ordered_metadata` vs
  `.sp`, not errors). `source/` was fetched fresh from the recorded Zenodo
  URL (md5-verified) via the `env -u LD_LIBRARY_PATH -u LD_PRELOAD curl`
  block build.sh already carries (an uncommitted addition from an earlier,
  interrupted pass — reviewed here: machine-neutral, no hardcoded paths,
  ran idempotently). No further build-system changes were needed.
- Gate: `spmm-tensorcore-fp16` (fp16, smoke, default dims sweep
  128/256/512, warmup=1, reps=2) — N=128 INVALID on all 3 smoke matrices,
  `max_scaled_err` 2.225e+00 (uniform) / 2.245e+00 (banded) / 2.747e+00
  (powerlaw) vs tol 0.01 — bit-for-bit identical to the recorded table;
  N=256/N=512 UNSUPPORTED (6 total) as recorded. `--dims 32`: 3/3 valid,
  `max_scaled_err` 3.12e-04 / 3.02e-04 / 3.94e-04 <= 0.01 — again identical
  to the recorded table.
- Deviation from the recorded ruling: none — the N=128 accuracy bug
  reproduces exactly (same error magnitudes to 3 significant figures) on
  this machine's different toolchain/GPU, confirming it is a genuine
  artifact bug in `sparse_mma_kernel_base_Buint64_Cfloat4`, not a
  Perlmutter-specific numerical quirk.
- Verdict here: BUILT+GATED (gate reproduces the N=128 bug, N=32 correct)
  — same as the recorded ruling.
