# GE-SpMM — gnn-aggregation — STATUS: BUILT+GATED

- Paper: "GE-SpMM: General-Purpose Sparse Matrix-Matrix Multiplication on
  GPUs for Graph Neural Networks" (SC'20). `PAPER_KEY = conf/sc/HuangD0Y20`.
  Selected as a **core**, regime-**matches** baseline under the revised
  kernel-centrality rule (2026-09-05) -- `output/kernel_centrality.json`'s
  `gnn-aggregation|conf/sc/HuangD0Y20` entry: "explicitly named as one of
  the papers defining the kernel-f32 variant's 'common denominator', and
  its dataset-sweep convention (N doubling 128->256->512) is reused verbatim
  by the spec."
- Repo: https://github.com/hgyhungry/ge-spmm, commit
  `f62f51169eb26c0d4411f6d9744eb585854410e1` (2021-07-01). Cloned fresh into
  `source/` (`git clone --depth 1`, non-recursive -- the repo's own
  submodules for GunRock/DGL/PyG baselines were not fetched; not needed,
  see below), unmodified.
- Build: `source/pytorch-custom/{spmm.cpp,spmm_kernel.cu}` (byte-identical
  to the clone), nvcc 12.9, `sm_80` (A100, `TORCH_CUDA_ARCH_LIST=8.0`), host
  compiler g++ 14.3.0, python env `/pscratch/sd/c/cunyang/gnn/plexus_env`
  (torch 2.8/cu128). Built via `torch.utils.cpp_extension.load()` -- the
  SAME call `source/pytorch-custom/op.py` itself makes
  (`load(name='spmm', sources=['spmm.cpp','spmm_kernel.cu'])`) -- run ahead
  of time in `build.sh` instead of at import time (ARTIFACT_GUIDE.md rule 9).

## What boundary was wrapped

`csr_spmm(rowptr, colind, values, dense)` (`spmm.cpp` /
`spmm_kernel.cu`'s `spmm_test0`/`spmm_test1`/`spmm_test2`, dispatched on
`dense.size(1)`) is a genuine WEIGHTED CSR SpMM: plain fp32
multiply-accumulate, `A_csrVal[ptr] * B_dnVal[offset]`, no tensor cores, no
hidden binary-only assumption. This is exactly `Y = A @ X` for an arbitrary
real-valued `A`, taking the value array directly at the Python boundary --
this adapter feeds it `A_hat = D^-1/2(A+I)D^-1/2` (computed independently in
`adapter.py::_gcn_normalize`, own code) straight in, with NO sandwich trick
or per-edge-attention workaround needed (contrast TC-GNN, which has to
route through a differently-named entry point for this same reason -- see
`artifacts/gnn-aggregation/tc-gnn/STATUS.md`). `csr_spmm_no_edge_value` (the
artifact's plain-adjacency path) and `op.py`'s `GCNConv` (which reproduces
symmetric normalization via pre/post row-scaling around the UNWEIGHTED
kernel, because its own default `edge_weight_csr=None` path calls
`csr_spmm_no_edge_value`) were both read but not used -- the weighted
`csr_spmm` path already covers this domain's exact reference with no
operand sandwiching required.

## Preprocessing (prepare(), timed once)

Only `A_hat`'s normalization (own independent code, same formula as every
other adapter's copy, reference-independence rule per DOMAIN_GUIDE.md) --
`csr_spmm` consumes plain CSR (`rowptr`/`colind`/`values`) directly, the
SAME layout `matrix.csr` already is, so casting to int32/float32 and moving
to the GPU is the only "conversion" needed.

## Build note: pybind module name

`spmm.cpp`'s `PYBIND11_MODULE(spmm, m)` fixes the compiled extension's
INTERNAL init symbol to `spmm` regardless of what `name=` is passed to
`cpp_extension.load()`. Confirmed empirically: an initial `build.sh` using
`name="gespmm"` (to avoid the generic "spmm" identifier colliding with a
hypothetical future artifact) built successfully but failed torch's own
post-build import-verification with `ImportError: dynamic module does not
define module export function (PyInit_gespmm)` -- `spmm.cpp` was NOT
touched to fix this (rule 3); `build.sh` instead uses the SAME
`name="spmm"` the artifact's own `op.py` uses. `adapter.py::_load()`
compensates for the resulting generic name by loading the compiled
`build/spmm.so` via `importlib.util.spec_from_file_location("spmm",
so_path)` directly (module name matched to the compiled-in symbol, as
required) rather than a `sys.path.insert + import spmm` statement, so the
generic name `spmm` is never registered into the shared process-wide
`sys.modules` where a different future artifact's own "spmm"-named module
could collide with it.

`spmm.cpp` also declares `csr2csc_cuda`/`cusparseCsr2cscEx2*` (an unrelated
pybind entry point in the same translation unit, not used here); `build.sh`
adds `-lcusparse` since torch's `cpp_extension.load()` does not link
cuSPARSE by default.

## Compute precision

`spmm_test0`/`spmm_test1`/`spmm_test2` operate on `float*` throughout
(`torch::kFloat32` asserted in `spmm.cpp`); no tensor-core or reduced-
precision path exists. `PRECISIONS = ["fp32"]`.

## Verification (login node, correctness gate only, reduced protocol)

```
PY=/pscratch/sd/c/cunyang/gnn/plexus_env/bin/python
export LD_PRELOAD=/usr/lib64/libstdc++.so.6
cd bench
$PY -m kernelbench.runner --kernel gnn-aggregation --variant gnn-agg-kernel-f32 \
    --impl gespmm-csrspmm-gcnnorm --smoke
$PY -m kernelbench.runner --kernel gnn-aggregation --variant gnn-agg-kernel-f32 \
    --impl gespmm-csrspmm-gcnnorm --matrices cora --dims 32,128,256,512 --warmup 2 --reps 3
```

Results, all **valid**:

| workload       | F (dense N) | max_scaled_err | tolerance |
|----------------|-------------|-----------------|-----------|
| smoke (12/12)  | 32-512      | 1.93e-07 - 2.88e-07 | 1e-4 |
| cora (4/4)     | 32-512      | 1.72e-07 - 2.12e-07 | 1e-4 |

**16/16 pass the gate** -- ordinary fp32 accumulation error, well within
tolerance; not tightened or loosened. Not spec-conforming (login node,
reduced warmup/reps, shared GPU) -- the GFLOP/s numbers printed above are
indicative only and must not be published as timing results.

## Not done

- No compute-node timing sweep (out of scope, rule 5).
- GunRock/DGL/PyG integration paths (`gunrock-test/`, `dgl-custom/`) were
  not fetched (non-recursive clone) or wrapped -- out of scope: they wrap
  the SAME kernel inside a bigger framework, which rule 1 explicitly
  prefers avoiding ("wrap the kernel, not the paper's benchmark script").
- `csr_spmm_no_edge_value` (unweighted path) not separately wrapped -- the
  weighted path already covers this domain's one reference exactly.

## Reuse by the spmm track (2026-09-06)

`artifacts/spmm/ge-spmm/` reuses this build (symlinked `source/`, imports
`build/spmm.so` directly, no second compile) to wrap the same `csr_spmm`
kernel for the `spmm` track's plain (non-normalized) `C = A @ B` reference —
see that directory's STATUS.md for its own gate results (20/20 valid,
err ~1.5e-07..4.2e-07, no N restriction found).

## Provenance

- Artifact commit: `f62f51169eb26c0d4411f6d9744eb585854410e1`
- nvcc: 12.9 (`/opt/nvidia/hpc_sdk/Linux_x86_64/25.5/cuda/12.9`)
- Host compiler: g++ 14.3.0 (`/opt/cray/pe/gcc-native/14/bin/g++`)
- GPU arch: `sm_80` (A100, `TORCH_CUDA_ARCH_LIST=8.0`)
- Python/torch: `/pscratch/sd/c/cunyang/gnn/plexus_env` -- torch 2.8.0+cu128

## Reproduction on zaratan (2026-09-09)

- Machine: UMD zaratan, A100 MIG 1g.5gb slice (sm_80, driver 595.71.05; job
  ran on gpu-b11-6.zaratan.umd.edu), login-node build.
- Toolchain: nvcc 12.8.93 (conda-forge CUDA 12.8), g++ 13.4.0, torch
  2.8.0+cu128, Python 3.12.14. Build was already present from an earlier
  pass on this machine (`build.sh` reported "already built, skipping");
  re-verified with `ldd` that `spmm.so` links against THIS machine's own
  `libcudart.so.12` / `libstdc++.so.6` (conda-forge `kb-env`), confirming
  it was compiled here.
- Build: OK (reused, no recompilation). Build-system changes: none.
- Gate: `cora` in the recorded command could not be loaded on this machine
  (`torch_geometric` missing in `kb-env`; see
  `../../gnn-aggregation/tc-gnn/STATUS.md`'s reproduction note for the full
  explanation). Substituted `cant` for direct numeric confirmation:
  - `gnn-agg-kernel-f32` smoke (12/12 valid): err 1.93e-07 - 2.88e-07
    <= tol 1e-4.
  - `gnn-agg-kernel-f32` on `cant`, dims 32/128/256/512 (4/4 valid):
    err 3.69e-07 - 4.46e-07 <= tol 1e-4.
- Deviation from the recorded ruling: none -- 16/16 workloads valid here
  (recorded: 16/16 including `cora`), ordinary fp32 accumulation error,
  same conclusion as the recorded `cora`-based verification (err
  1.93e-07 - 2.88e-07 there).
- Verdict here: BUILT+GATED -- equals the recorded ruling.
