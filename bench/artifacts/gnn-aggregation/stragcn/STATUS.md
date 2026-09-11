# StraGCN — gnn-aggregation — STATUS: BUILT+GATED

- Paper: StraGCN: GPU-Accelerated Strassen's Sparse-Dense Matrix
  Multiplication for GCN Training (SC'25). `PAPER_KEY = conf/sc/HeLDL0M025`
  (confirmed via `output/included.json`, matched by `artifact_url`).
- Repo: https://github.com/CGCL-codes/StraGCN, commit
  `fe1bd688c55a75640110392cbbb1043711d3a065` (2025-07-08). Cloned fresh into
  `source/` (`git clone --depth 1`), unmodified.
- Build: `source/StraGCN/GNN_strassen.cpp` + `source/StraGCN/strassen.cu`
  (both byte-identical to the clone), nvcc 12.9, `sm_80` (A100,
  `TORCH_CUDA_ARCH_LIST=8.0`), host compiler g++ 14.3.0, python env
  `/pscratch/sd/c/cunyang/gnn/plexus_env` (torch 2.8/cu128).

## What boundary was wrapped

`GCN_ST` (the artifact's own pybind11 torch extension) exposes three entry
points: `forward(A, X, W)` (full GCN layer: `spmm_strassen(A, X@W)`),
`backward`, and `spmmstra(A_rowPtr, A_colIdx, A_values, offset, X)` —
`spmm_strassen(A, X)` alone, with no bundled dense weight GEMM. This adapter
calls `spmmstra` directly: it is exactly this track's kernel boundary,
`Y = A_hat @ X`, with no contamination from a second (GEMM) kernel.

## Build-system substitution, not a kernel patch

`source/StraGCN/setup.py` hardcodes `os.environ["CC"] = "gcc-10"` /
`"g++-10"` at import time — neither exists on this machine, and since the
script sets these itself, no shell-level override could win against it. This
adapter never executes that `setup.py`; `build.sh` instead drives
`torch.utils.cpp_extension.load()` directly against the identical, unmodified
`.cpp`/`.cu` sources, with `CXX`/`CC` pointed at the system g++ 14 (same
sysconfig `-B .../compiler_compat` issue `bench/artifacts/sddmm/fused3s`'s
STATUS.md documents in full). Per ARTIFACT_GUIDE.md rule 3 ("build-system
fixes ... are fine; touching kernel code is not") this is a build-driver
substitution, not a patch to the artifact's own code — `source/` is untouched.

**Residual runtime requirement (same as fused3s, same root cause):** this
venv's `python3.11` executable itself has a `DT_RPATH` pointing at a stale
`libstdc++.so.6` (missing `CXXABI_1.3.15`), baked in ahead of anything a
shell-level `LD_LIBRARY_PATH` can override. **Running this adapter (and
`build.sh`'s own import-verification step) requires
`LD_PRELOAD=/usr/lib64/libstdc++.so.6` in the invoking shell.**

## Preprocessing (prepare(), timed once): reuses the artifact's own code

`spmmstra` does not take a plain CSR triple — it expects `A_hat` pre-split
via Strassen's classic 2x2-block, 7-multiply decomposition (`M1..M7`, each
`half x half`, `half = ceil(V/2)`), concatenated into flat
`(rowPtr, colIdx, values)` arrays plus an `offset` array. `adapter.py`
imports `split_CSR`/`preAdd` directly from `source/StraGCN/dataset.py`
**unmodified** and does the flat-array packing `GCN.py` itself does inline —
this genuinely is "the artifact's own format conversion"
(ARTIFACT_GUIDE.md rule 2), so it lives entirely in `prepare()`, timed once.
`split_CSR` is a pure-Python `O(nnz)` loop (not vectorized) — a real,
un-worked-around cost of this artifact's own preprocessing, not something
this adapter speeds up or hides.

Adjacency normalization (`A_hat = D^-1/2 (A+I) D^-1/2`) is computed
independently in `adapter.py::_gcn_normalize` (own code, not calling
`kernelbench.impls.cpu_ref`'s helpers) — confirmed to be StraGCN's own
convention too: `source/StraGCN/dataset.py`'s `__main__` block computes
`deg_inv = degree**-0.5; values = deg_inv[row]*deg_inv[col]` on the raw
adjacency, the same GCN symmetric normalization this domain's reference uses
(this domain's own choice of A+I self-loops, since StraGCN's script doesn't
add them explicitly, is used here so the adapter's output clears THIS
domain's actual reference gate).

## Verification (login node, correctness gate only, reduced protocol)

```
PY=/pscratch/sd/c/cunyang/gnn/plexus_env/bin/python
export LD_PRELOAD=/usr/lib64/libstdc++.so.6
cd bench
$PY -m kernelbench.runner --kernel gnn-aggregation --variant gnn-agg-kernel-f32 \
    --impl stragcn-spmmstra-gcnnorm --matrices ca-HepPh --dims 32,128 --warmup 2 --reps 3
$PY -m kernelbench.runner --kernel gnn-aggregation --variant gnn-agg-kernel-f32 \
    --impl stragcn-spmmstra-gcnnorm --matrices cora --dims 256,512 --warmup 2 --reps 3
```

Results, all **valid**:

| matrix   | F (dense N) | max_scaled_err | tolerance |
|----------|-------------|-----------------|-----------|
| ca-HepPh | 32          | 3.55e-05        | 1e-4      |
| ca-HepPh | 128         | 8.33e-05        | 1e-4      |
| cora     | 256         | 1.14e-05        | 1e-4      |
| cora     | 512         | 5.54e-05        | 1e-4      |

Not spec-conforming (login node, reduced warmup/reps, shared GPU) — timing
numbers from this run must not be published, only the correctness result.

## Two real artifact findings

**1. Confirmed bug: silently wrong output for ODD feature width `F`.**
`Add_total_s`/`Add_finish_s` (`strassen.cu`) hardcode the dense operand's row
stride as `NN = 2*halfn = 2*ceil(F/2)`. A PyTorch tensor of shape `(V, F)`
has row stride `F`; these are equal only when `F` is **even**. For odd `F`,
every row past the first is read/written at a systematically wrong offset —
confirmed empirically with small hand-built cases (independent of this
adapter, calling `GCN_ST.spmmstra` directly): `F=4` (even) gives max abs err
`~5e-8`; `F=5` (odd) gives max abs err `~1.75` on an output whose true
values are `O(1)` — silently wrong, no crash, no exception raised. (A
separate, more benign effect: odd `V` — even `F` reads/writes a few floats
past the end of the X/output buffers' last row-chunk; empirically harmless
on this allocator, since the stray access lands in unused allocation padding
and never touches a valid output row — both `V=5` and `V=7` cases with even
`F` gave correct results in the same hand test.) This track's spec only
sweeps even `F` (`N in {32, 128, 256, 512}`), so `prepare()` asserts `F` is
even and raises `NotImplementedError` with this explanation for odd `F`
rather than silently returning a wrong answer — verification above never
exercises the odd-`F` path, so it never blocks a spec-conforming run, but
it's a real, load-bearing limit of the kernel as shipped.

**2. NOT a StraGCN bug — a smoke-workload mismatch, discovered and worth
recording:** running this adapter against `smoke_workloads()`'s synthetic
matrices (`--smoke`) reports `0/12 valid` with `max_scaled_err` in the
`1e294`–`1e295` range. Root cause, fully traced (not a kernel defect):
`matrices.synthetic()` draws **signed** `U(-1,1)` edge weights (fine for
spmv/spmm/sddmm, which don't care about sign). For `gnn-aggregation`'s GCN
normalization, this makes a meaningful fraction of rows have a "degree"
`(A+I).sum(axis=1)` that is exactly zero *or negative* — this domain's
reference and this adapter both correctly zero that row's normalization
factor (`D^-1/2` of a non-positive number is undefined), producing a
**structurally exact-zero** row of `A_hat` in both. A **direct** SpMM
(scipy, or a plain warp-per-row CUDA kernel) computes `0 * anything = 0`
exactly at such a row — no floating-point residue. Strassen's algorithm
computes that same structurally-zero row via `M1+M4-M5+M7`-style
sums/differences of otherwise-nonzero intermediate blocks; a well-known
numerical property of Strassen's scheme is that it is **not** exact-zero
preserving under catastrophic cancellation the way direct multiplication is,
so it lands a `~1e-8` fp32 rounding residue instead of a bit-exact `0.0`.
`harness.check_correctness`'s `max_scaled_err` divides by the reference
`scale = |A_hat|@|X|`, which is exactly `0.0` at these rows (floored to
`1e-300`) — so a `~1e-8` numerator over a `1e-300` floor explodes to
`~1e292`, independent of how small the true error actually is. On every
**real** graph in `recommended_subset` (SNAP/citation/protein adjacency,
non-negative weights by construction), `(A+I).sum(axis=1) >= 1` always, this
degeneracy cannot occur, and — as the verification table above shows — the
gate passes cleanly with ordinary `1e-5`-scale fp32 error. Recorded here
rather than worked around: `smoke_workloads()`'s signed synthetic patterns
are appropriate for spmv/spmm/sddmm but are not a meaningful correctness
smoke test for this artifact's Strassen-based `gnn-aggregation` path
specifically; verification instead uses the real-graph invocations above.

## Provenance

- Artifact commit: `fe1bd688c55a75640110392cbbb1043711d3a065`
- nvcc: 12.9 (`/opt/nvidia/hpc_sdk/Linux_x86_64/25.5/cuda/12.9`)
- Host compiler: g++ 14.3.0 (`/opt/cray/pe/gcc-native/14/bin/g++`)
- GPU arch: `sm_80` (A100, `TORCH_CUDA_ARCH_LIST=8.0`)
- Python/torch: `/pscratch/sd/c/cunyang/gnn/plexus_env` — torch 2.8.0+cu128

## Reproduction on zaratan (2026-09-09)

- Machine: UMD zaratan, A100-SXM4-40GB (sm_80, full card, partition `gpu`),
  login-node build.
- Toolchain: nvcc 12.8.93 (conda-forge CUDA 12.8), g++ 13.4.0, torch
  2.8.0+cu128, Python 3.12.14. `TORCH_CUDA_ARCH_LIST=8.0` (unchanged),
  matches this GPU.
- Build: OK (`build.sh` already reads `${CC:-...}`/`${CXX:-...}`/
  `${CUDA_HOME:-...}`/`${LD_PRELOAD:-...}` correctly — every one of those
  is exported before this build.sh runs, by `toolchain.sh`/`env.sh`, so
  no build-system change was needed here). Forced a rebuild
  (`FORCE_REBUILD=1`) to confirm from a clean state rather than trusting
  the pre-existing `build/GCN_ST*.so`; ninja found the two object files
  already up to date and only re-linked, confirming the earlier build
  already used this exact toolchain.
- Gate (`gnn-agg-kernel-f32`/fp32, exact commands quoted in this file):
  ca-HepPh N=32 PASS (err 3.55e-05 <= 1e-4); ca-HepPh N=128 PASS (err
  8.33e-05 <= 1e-4); cora N=256 PASS (err 1.14e-05 <= 1e-4); cora N=512
  PASS (err 5.54e-05 <= 1e-4). 4/4 valid. All four error values match the
  recorded numbers exactly.
- Data note (not a build-system or kernel change, recorded for the next
  run on this machine): this login node's `urllib.request.urlretrieve`
  hung indefinitely fetching both the SuiteSparse index/matrix and the
  Planetoid/Cora raw files (`curl` to the same URLs succeeded in
  well under a second, so this is a Python-urllib-specific stall on this
  machine, not a network-reachability problem) — worked around by
  fetching `ssstats.csv`, `ca-HepPh.tar.gz` (extracted to
  `bench/matrices/ca-HepPh.mtx`), and the 8
  `ind.cora.{x,tx,allx,y,ty,ally,graph,test.index}` files with `curl -L`
  directly into `bench/matrices/` and `bench/graphs/planetoid/Cora/raw/`
  (the exact paths/filenames `kernelbench/matrices.py` and
  `torch_geometric.datasets.Planetoid` already expect) before running the
  gate; no code was changed, both caches are git-ignored as before.
- Deviation from the recorded ruling: none.
- Verdict here: BUILT+GATED — equals the recorded ruling.
