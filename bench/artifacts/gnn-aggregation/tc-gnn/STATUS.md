# TC-GNN — gnn-aggregation — STATUS: BUILT (correctness gate FAILS at measured tf32-level error; two workload-crashing bugs found and guarded, not patched)

- Paper: "TC-GNN: Bridging Sparse GNN Computation and Dense Tensor Cores on
  GPUs" (USENIX ATC'23). `PAPER_KEY = conf/usenix/WangFWHD23`. Selected as a
  **core**, regime-**matches** baseline under the revised kernel-centrality
  rule (2026-09-05) -- `output/kernel_centrality.json`'s
  `gnn-aggregation|conf/usenix/WangFWHD23` entry: "one of the few papers in
  this track that ships both an isolated kernel-only benchmark AND an
  epoch-level model benchmark, and its 14-graph suite overlaps heavily with
  the spec's recommended_subset".
- Repo: https://github.com/YukeWang96/TC-GNN_ATC23, commit
  `0ff60b2f0acbce25bc0b137d3909c41a520da0fa` (2023-10-15). Cloned fresh into
  `source/` (`git clone --depth 1`), unmodified.
- Build: `source/TCGNN_conv/{TCGNN.cpp,TCGNN_kernel.cu}` (byte-identical to
  the clone), nvcc 12.9, `sm_80` (A100, `TORCH_CUDA_ARCH_LIST=8.0`, required
  by `wmma::precision::tf32` fragments), host compiler g++ 14.3.0, python
  env `/pscratch/sd/c/cunyang/gnn/plexus_env` (torch 2.8/cu128). Built via
  `torch.utils.cpp_extension.load()` (same pattern as
  `artifacts/gnn-aggregation/stragcn/build.sh`) rather than
  `source/0_build_tcgnn.sh`'s `python setup.py install`, so the compiled
  `.so` lands under this directory's own `build/`, never inside `source/`
  (ARTIFACT_GUIDE.md rule 3) -- `source/TCGNN_conv/setup.py` itself is a
  plain, unmodified `CUDAExtension`, no build-driver substitution was
  actually needed, this is purely about where the artifact lands.

## What boundary was wrapped

TC-GNN's `TCGNN` torch extension exposes `forward` (WMMA SpMM, but
UNWEIGHTED -- every populated cell is hardcoded to `1` in
`spmm_forward_cuda_kernel`, confirmed by reading the kernel; no value array
in `spmm_forward`'s signature at all) and `forward_AGNN` (the SAME WMMA/
tensor-core compute and structural preprocessing, but taking a genuine
per-edge `edgeAttention` array -- `sparse_A[...] = edgeAttention[eIdx]`).
This adapter calls `forward_AGNN` directly: it is the only one of TC-GNN's
own, unmodified entry points that can compute this domain's actual
reference (`Y = D^-1/2(A+I)D^-1/2 @ X`, real-valued, not binary), and it is
exactly TC-GNN's own kernel/format -- `gnn_conv.py`'s `AGNNConv`/
`TCGNNFunction_AGNN` already calls it in precisely this shape
(`edgeAttention` shape `[n_head=1, n_e]`) for the paper's own attention
mechanism. `TCGNN.preprocess()` (CPU, OMP, purely structural -- read in
full, never touches edge values or reorders `column_index`) builds the
tensor-core row-window/block-partition metadata exactly as
`main_tcgnn.py` does; this genuinely is "the artifact's own format
conversion" (ARTIFACT_GUIDE.md rule 2), timed once in `prepare()`.
Normalization (`A_hat = D^-1/2(A+I)D^-1/2`) is computed independently in
`adapter.py::_gcn_normalize` (own code, not calling
`kernelbench.impls.cpu_ref` or the stragcn adapter's copy, per the
reference-independence rule).

## Four real, confirmed artifact findings (none patched -- rule 3)

**1. Silent tail-drop for F not a multiple of 16.** The C++ host wrapper
sizes shared memory with CEILING division (`dimTileNum =
(embedding_dim + BLK_H - 1) / BLK_H`) but the `__global__` kernel body
recomputes the same name with FLOOR division
(`dimTileNum = embedding_dim / BLK_H`) -- for F not a multiple of 16 the
kernel processes fewer tiles than were sized for, silently leaving the tail
columns at their zero-initialized value. Guarded: `prepare()` raises
`NotImplementedError` for `F % 16 != 0`.

**2. Confirmed one-element out-of-bounds WRITE (heap corruption) whenever
`num_nodes` is an exact multiple of `BLK_H=16`.** `TCGNN.cpp`'s
`preprocess()` row-window loop is `for (iter = 0; iter < num_nodes + 1;
iter += 16)` -- off by one against the correct bound (contrast
`fill_edgeToRow`'s own loop two lines above, correctly `nid < num_nodes`,
no `+1`, in the SAME file). When `num_nodes % 16 == 0` the loop's last
iteration computes `windowId = num_nodes/16 == num_row_windows`, one past
`blockPartition`'s valid `[0, num_row_windows)` range, and writes there.
**Confirmed empirically, not just from reading the code**: with
`num_nodes=16` and `blockPartition` allocated as a size-1 view into a
4-element canary tensor `[-777,-777,-777,-777]`, calling `preprocess()`
leaves the canary as `[2, 1, -777, -777]` -- index 1 (one past the valid
view) was overwritten. This corrupts whatever heap memory follows the
tensor's storage; in this integration it surfaced as a DELAYED
`free(): invalid pointer` / glibc malloc assertion abort inside an UNRELATED
`torch.Tensor` deallocation much later in the same process --
confirmed via `gdb` backtrace (`THPVariable_subclass_clear` -> `free()`,
nowhere near TC-GNN's own code), which is exactly why the crash appeared to
"sometimes reproduce, sometimes not" across differently-shaped reproduction
scripts before this was traced to its root cause. **All three of this
domain's `SMOKE` matrices use `rows=4000` (`4000 % 16 == 0`)**, so
`--smoke` hits this on every workload; `cora` (2708 nodes,
`2708 % 16 == 4`) does not. Guarded: `prepare()` raises
`NotImplementedError` for `num_nodes % 16 == 0` -- this is NOT the usual
"silently wrong answer" rule-8 case, it is confirmed memory corruption, so
refusing the workload outright (rather than merely flagging low confidence)
is the only safe response available without touching the artifact's code.

**3. Confirmed silent zero-fill for F > 128.** Both `spmm_forward_cuda` and
`spmmAGNN_forward_cuda` launch with a FIXED block shape
`dim3 block(32, WPB=8, 1)`; the kernel selects its embedding-dimension tile
purely via `threadIdx.y in [0, WPB)`, with no outer loop over further
tiles. For `dimTileNum = F/16 > WPB = 8` (i.e. `F > 128`), output columns at
index `>= 128` are never assigned to any warp and stay at the kernel's
`torch::zeros_like(input)` initial value. **Confirmed empirically** on
`cora` (safe from finding #2): `F=256` and `F=512` both gate at
`max_scaled_err = 1.000` (the entire tail beyond column 128 is exactly zero
against a genuinely nonzero reference). This track's spec sweeps `F` over
`{32, 128, 256, 512}`; `prepare()` raises `NotImplementedError` for
`F > 128`. Matches TC-GNN's own paper convention -- every one of its own
benchmark scripts uses `--hidden 16`, well under this limit; the paper's own
evaluation never exercises `F > 128` on this kernel either.

**4. Confirmed resource leak (not a correctness bug):**
`spmmAGNN_forward_cuda` calls `cudaStreamCreate` once per invocation
(`new cudaStream_t[num_attention]` in a loop) with no matching
`cudaStreamDestroy`/`delete[]` anywhere in the function or file. Harmless
for the handful of gate-only launches this integration performs; a full
compute-node timed run (100-200 reps) would leak one stream handle per rep.

## Compute precision

`TCGNN_kernel.cu` uses `wmma::fragment<..., wmma::precision::tf32, ...>` /
`wmma::fragment<wmma::accumulator, ..., float>` throughout -- TF32
tensor-core compute, fp32 accumulate (same precision family as
`artifacts/spmm/dtcspmm`). `gnn-aggregation`'s `gnn-agg-kernel-f32` variant
has no separate tf32 row to gate against (unlike `spmm`), so
`PRECISIONS = ["fp32"]` describes the Python-boundary tensor dtype; the
measured error below is TF32-level (~7e-4), reported and gated as-is
against the spec's 1e-4 tolerance rather than relabeled to a laxer bound.

## Verification (login node, correctness gate only, reduced protocol)

```
PY=/pscratch/sd/c/cunyang/gnn/plexus_env/bin/python
export LD_PRELOAD=/usr/lib64/libstdc++.so.6
cd bench
$PY -m kernelbench.runner --kernel gnn-aggregation --variant gnn-agg-kernel-f32 \
    --impl tcgnn-agnn-spmm-gcnnorm --smoke
$PY -m kernelbench.runner --kernel gnn-aggregation --variant gnn-agg-kernel-f32 \
    --impl tcgnn-agnn-spmm-gcnnorm --matrices cora --dims 32,128,256,512 --warmup 2 --reps 3
```

Results:

| workload           | F (dense N) | outcome      | max_scaled_err | tolerance |
|--------------------|-------------|--------------|-----------------|-----------|
| smoke (all 3x4=12) | 32-512      | UNSUPPORTED  | --              | --        |
| cora               | 32          | **INVALID**  | 6.983e-04       | 1e-4      |
| cora               | 128         | **INVALID**  | 7.122e-04       | 1e-4      |
| cora               | 256         | UNSUPPORTED  | --              | --        |
| cora               | 512         | UNSUPPORTED  | --              | --        |

**0/2 serviceable (F<=128, num_nodes%16!=0) workloads pass the gate.**
The measured error is a real, unforced TF32-tensor-core precision result
(~7x the 1e-4 tolerance) -- not loosened, not worked around. Per
ARTIFACT_GUIDE.md rule 4, this is a genuine result: TC-GNN's `forward_AGNN`
kernel, at its own native compute precision, does not clear this track's
1e-4 correctness bound on this input. Not spec-conforming (login node,
reduced warmup/reps, shared GPU) -- timing numbers must not be published.

## Checked: is there a laxer/tf32-appropriate variant to gate against instead?

`$PY -m kernelbench.runner --kernel gnn-aggregation --list` enumerates all
four of this track's variants (`gnn-agg-kernel-f32`,
`gnn-agg-e2e-preproc-f32`, `gnn-agg-layer-forward-real-features`,
`gnn-agg-epoch-training-e2e`); reading `benchspecs/gnn-aggregation/
spec.yaml`'s `correctness` field for each: `gnn-agg-kernel-f32` sets
`< 1e-4`, `gnn-agg-e2e-preproc-f32` states "same tolerance as
gnn-agg-kernel-f32", `gnn-agg-layer-forward-real-features` sets
`1e-4 max relative error`, and `gnn-agg-epoch-training-e2e` uses a
different check entirely (final validation accuracy within 1pp, not a
per-call numerical gate, and not applicable to an isolated-kernel
adapter). Unlike `spmm` (which has a dedicated `spmm-tensorcore-fp16`
variant with a separate, explicitly laxer tf32 tolerance row --
`artifacts/spmm/dtcspmm` gates against that), **this track has no
tf32-appropriate variant anywhere** -- every numerically-gated variant here
uses the SAME fixed `1e-4` bound regardless of what precision the kernel
under test actually computes at. There is therefore no legitimate way to
gate TC-GNN's tf32 kernel except at this fixed bound; per this task's own
instruction ("if the track has no fp16 variant, record BUILT with the
measured error and say so"), the outcome above stands as `BUILT` with the
measured error stated plainly rather than silently relabeled to a laxer
bound that does not exist in this domain.

## Not done

- No compute-node timing sweep (out of scope for this integration per
  ARTIFACT_GUIDE.md rule 5).
- The `forward` (unweighted) entry point and the paper's own epoch-training
  pipeline (`main_tcgnn.py`, `gnn_conv.py`'s `GCNConv`/`GINConv`) were read
  but not wrapped -- out of this track's kernel-only scope
  (`gnn-agg-kernel-f32`).
- Findings #1-#3 above are workload-level `NotImplementedError` guards, not
  patches to `source/` (rule 3); `source/` is byte-identical to the clone.

## Reuse by the spmm and sddmm tracks (2026-09-06)

`artifacts/spmm/tc-gnn/` reuses this build (symlinked `source/`, imports
`build/TCGNN.so` directly, no second compile) to wrap the SAME
`forward_AGNN` entry point for the `spmm` track's plain (non-normalized)
`C = A @ B` reference, reproducing the same three guards above — 2/2
serviceable (`cora`, N=128) workloads pass at `4.87e-04` under both
`spmm-tensorcore-fp16` and the concurrently-added
`spmm-binary-adjacency-kernel`. `artifacts/sddmm/tc-gnn/` reuses it again
to wrap `forward_ef` (the SDDMM entry point). See each directory's own
STATUS.md for full gate results.

## Provenance

- Artifact commit: `0ff60b2f0acbce25bc0b137d3909c41a520da0fa`
- nvcc: 12.9 (`/opt/nvidia/hpc_sdk/Linux_x86_64/25.5/cuda/12.9`)
- Host compiler: g++ 14.3.0 (`/opt/cray/pe/gcc-native/14/bin/g++`)
- GPU arch: `sm_80` (A100, `TORCH_CUDA_ARCH_LIST=8.0`)
- Python/torch: `/pscratch/sd/c/cunyang/gnn/plexus_env` -- torch 2.8.0+cu128

## Reproduction on zaratan (2026-09-09)

- Machine: UMD zaratan, A100 MIG 1g.5gb slice (sm_80, driver 595.71.05; job
  ran on gpu-b11-6.zaratan.umd.edu), login-node build.
- Toolchain: nvcc 12.8.93 (conda-forge CUDA 12.8), g++ 13.4.0 (conda-forge),
  torch 2.8.0+cu128, Python 3.12.14, `TORCH_CUDA_ARCH_LIST=8.0` (unchanged).
- Build: OK. `build.sh` found `build/TCGNN.so` already present (from an
  earlier pass on this machine) and skipped recompilation
  (`compgen -G .../TCGNN*.so` idempotence check); re-verified with `ldd`
  that the `.so` links against THIS machine's own `libcudart.so.12` /
  `libstdc++.so.6` (conda-forge `kb-env`), confirming it was actually
  compiled here, not carried over from Perlmutter. Build-system changes:
  none -- `build.sh` already reads `CUDA_HOME`/`CC`/`CXX`/`PY` via
  `${VAR:-<Perlmutter default>}`, and `toolchain.sh`/`env.sh` populate those
  with this machine's values before `build.sh` runs, so the existing
  ${VAR:-default} pattern was already machine-neutral.
- Gate:
  - `gnn-agg-kernel-f32` smoke (12 workloads, `rows=4000`,
    `4000 % 16 == 0`): 0/0 valid, 12/12 UNSUPPORTED via guard #2
    (`num_nodes % 16 == 0`) -- matches recorded.
  - `gnn-agg-kernel-f32` on `cora`: **could not run** -- this machine's
    `kb-env` Python environment does not have `torch_geometric` installed,
    so `kernelbench.domains.sparse._load_planetoid` raises
    `ModuleNotFoundError: No module named 'torch_geometric'`. The module
    docstring's claim that torch_geometric is "already present in this
    machine's venv" was true of the Perlmutter `plexus_env` this artifact
    was originally verified against, but is not true of zaratan's `kb-env`.
    This is a harness-level Python-environment gap (affects every
    Planetoid-backed workload on this machine, not just this artifact) --
    out of scope for a per-artifact build-system fix: fixing it would
    require either installing into the shared `kb-env` (disallowed) or
    modifying `kernelbench/domains/sparse.py` (disallowed). Substituted
    `cant` (SuiteSparse, 62451x62451, `62451 % 16 == 3`, same
    guard-boundary safety class as `cora`'s `2708 % 16 == 4`) for direct
    numeric confirmation instead:
  - `gnn-agg-kernel-f32` on `cant`, dim=32: **INVALID**,
    max_scaled_err=4.850e-04 vs tol 1e-4.
  - `gnn-agg-kernel-f32` on `cant`, dim=128: **INVALID**,
    max_scaled_err=4.833e-04 vs tol 1e-4.
  - `gnn-agg-kernel-f32` on `cant`, dim=256, 512: UNSUPPORTED via guard #3
    (`F > 128`) -- matches recorded.
- Deviation from the recorded ruling: none in substance. The exact `cora`
  command quoted above could not be executed (torch_geometric missing on
  this machine, see above), but the `cant` substitute reproduces the same
  qualitative result the recorded ruling describes: 0/2 serviceable
  (F<=128, num_nodes%16!=0) workloads pass the 1e-4 gate, at a TF32-level
  error (~4.8e-4 here vs ~7e-4 on `cora` on Perlmutter) several times the
  tolerance; all three guards (F%16!=0, num_nodes%16==0, F>128) fire
  exactly where documented.
- Verdict here: BUILT (gate fails: TF32-level correctness error, ~4.8e-4
  vs 1e-4 tolerance) -- equals the recorded ruling.
