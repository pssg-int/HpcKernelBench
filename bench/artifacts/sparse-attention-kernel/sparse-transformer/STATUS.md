# sparse-transformer (STOF) — sparse-attention-kernel — STATUS: BUILT+GATED (with a confirmed artifact bug)

- Paper: "Accelerating Sparse Transformer Inference on GPU", PPoPP'26.
  `PAPER_KEY = conf/ppopp/DaiDRYLL0CS26` (matched by `artifact_url` in
  `../../../../output/included.json`; arXiv:2506.06095).
- Artifact: https://github.com/HeyDavid633/PPoPP26-pap161-AE.
- Commit cloned: `6f2d0cf18f89b8e6343ff7cf9d6689b74002a51f`, `git clone --depth 1`.
- Toolchain: python `/pscratch/sd/c/cunyang/gnn/plexus_env/bin/python`, torch
  `2.8.0+cu128`, nvcc `12.9` (`/opt/nvidia/hpc_sdk/Linux_x86_64/25.5/cuda/12.9`),
  host compiler `/opt/cray/pe/gcc-native/14/bin/g++` (GCC 14), GPU arch
  `sm_80` (A100, `-gencode=arch=compute_80,code=sm_80`, matching this repo's
  own `src/setup.py`'s default `cuda_arch="80"`).

## What was built, and why not via the repo's own setup.py directly

The repo's real "Our Kernel" path (confirmed the one `src/benchmk_attn_
unified.py` actually times against 8 baselines, per the track's own
survey.md) is the `binding_attn` torch extension
(`src/ops/src/binding_attn.cpp` + `binding_attn_cuda.cu`, a CUTLASS/
FlashAttention-family fused kernel). `src/setup.py` globs `*.cpp`+`*_cuda.cu`
under `ops/src/` and would build BOTH `binding_attn` AND a second, separate,
~6300-line extension `block_attn_mask` (`block_attn_mask_cuda.cu`) in one
`build_ext --inplace` invocation — `block_attn_mask` is not called anywhere
in `benchmk_attn_unified.py`'s actual timed comparison (its Python wrapper,
`Block_Attn_Mask.forward` in `src/util/masks.py`, appears to be dead/earlier
code — it calls `block_attn_mask.forward()` for side effects and then
returns `q` UNCHANGED, discarding whatever the call computed). Building it
would cost significant extra compile time/risk (6300 lines of raw CUDA) for
no benefit to what this integration measures, so `build.sh` invokes torch's
`CUDAExtension`/`BuildExtension` API directly, in a private throwaway
`setup.py` under `_build_binding_attn/` (NOT part of the artifact's own
source tree), using EXACTLY `src/setup.py`'s own `extra_compile_args`/
architecture flags, restricted to `binding_attn`'s 2 source files. **No
artifact source file was edited.** Same two machine-specific compiler
workarounds as `bench/artifacts/attention-kernel/pat/build.sh` and
`bench/artifacts/sddmm/fused3s/build.sh` (CXX/CC -> system g++14, LDSHARED
rpath override) — see those files for the full derivation. `MAX_JOBS=1`
(same OOM-avoidance reasoning as pat's build).

Build succeeded in ~1 minute (much faster than PAT's ~16 min — this
extension only has 2 source files, no 4-way explicit head-dim
instantiation).

## What the adapter calls

`adapter.py::run()` calls `binding_attn.forward(...)` — the compiled
extension's C++ entry point — **directly**, not through
`src/ops/package_op.py`'s `binding_attn_func`/`BindingAttnFunc` autograd
wrapper. Reason: `package_op.py` unconditionally does `import
block_attn_mask` at module import time (the OTHER, un-built extension
above), so importing it would fail without also building that unrelated
6300-line kernel. `package_op.py`'s own `_binding_attn_forward()` (read in
full, unmodified) is a thin, autograd-irrelevant pass-through to this exact
`binding_attn.forward(...)` call for a forward-only benchmark — this
adapter's `run()` uses the literal same positional argument order,
confirmed by reading `_binding_attn_forward`'s source.

`prepare()` converts the workload's fixed `(S,S)` mask into STOF's own
full/part/inner-bitmap block-sparse format via `src/util/masks.py::
get_OuterTile_storage()` (unmodified) — a general, lossless block
classification of ANY `[B,N,N]` 0/1 mask (each `BLOCK_M x BLOCK_N` outer
tile is classified full/empty/mixed; mixed tiles additionally record an
8x8-sub-tile bitmap) — legitimate ARTIFACT_GUIDE-rule-2 preprocessing, timed
once. `BLOCK_M=BLOCK_N=64`, `benchmk_attn_unified.py`'s own CLI default.
Precision/shape are hard-constrained by `binding_attn.cpp`'s own
`run_mha_fwd()` (unmodified, read in full): `AT_ERROR` on `d != 64` and on
bf16 — `PRECISIONS = ["fp16"]`, `prepare()` raises `NotImplementedError` for
`d != 64`.

## Real artifact bug found: the inner-bitmap fine mask is dead code

**`inner_bitmaps` — the data structure that is supposed to encode exact
sub-tile masking for "part" (mixed-density) blocks — is threaded through
every function signature in the call chain
(`binding_attn.cpp::flashattn_binding_gpu` -> `run_mha_fwd` ->
`flash_fwd_launch_template.h::run_flash_fwd`/`bind_fwd_kernel` ->
`bind_fwd_kernel.h::compute_mask_attn`) but is NEVER DEREFERENCED anywhere
in the actual kernel body.** Confirmed two ways, both against unmodified
source:
1. `grep -rn "inner_bitmap" src/ops/src/include/` finds it **only** in
   function signatures/call-forwarding sites (`flash.h`,
   `flash_fwd_launch_template.h`, `bind_fwd_kernel.h`'s signature line) —
   zero occurrences inside any function BODY.
2. `bind_fwd_kernel.h::compute_mask_attn()`'s "part" block loop (the loop
   that iterates `part_col_idx`/`part_row_ptr` — the mixed-tile case
   `inner_bitmaps` exists to handle) calls only
   `mask.template apply_mask<Is_causal, Is_even_MN>(acc_s, ...)`
   (`src/ops/src/include/mask.h`'s `Mask::apply_mask`, read in full) — which
   applies ONLY the generic causal/local windowing mask (from the
   `Is_causal`/`window_size_left`/`window_size_right` params), never reading
   any bitmap. `mask.h` contains **zero** occurrences of "bitmap" anywhere
   in the file.

**Consequence**: for any mask whose sparsity pattern does not align exactly
to `BLOCK_M x BLOCK_N` (64x64) tile boundaries — i.e., produces at least one
"part" (mixed) tile — the kernel silently computes attention as if those
mixed tiles were **fully dense** (no fine-grained masking at all), which is
NOT the mask the caller specified. `get_OuterTile_storage`'s bitmap
computation (`get_InnerTile_bitmap`, correctly implemented on the Python
side) is therefore pure dead weight for `binding_attn` — a real,
reproducible correctness bug in the released kernel, not an integration
mistake on this adapter's side (the same argument order/layout/scale, tested
below on a mask with **zero** mixed tiles, gates cleanly with error
consistent with ordinary fp16 rounding).

**Empirical confirmation** (both runs below use the identical adapter code,
differing only in the workload's mask pattern):
- `block_local(block=64, n_global_blocks=0)` at `S=512` — every 64x64 outer
  tile is either exactly all-1 ("full") or all-0 ("skip"), **zero** "part"
  tiles by construction (block boundaries align exactly to `BLOCK_M`) ->
  **`max_scaled_err = 3.44e-4`, PASSES** (tolerance `1e-2`).
- `sliding_window(window=64)` at `S=512` — a diagonal band produces mostly
  "part" (mixed) tiles at this tile granularity -> **`max_scaled_err =
  6.83`, FAILS** (`max_pointwise_rel_err = 1.9e5`, `l2_rel_err = 1.55` — the
  output is essentially uncorrelated with the correct masked result, exactly
  what "silently computed as dense" predicts).

This is precisely the kind of bug the track's own survey flagged as a
structural risk: STOF's own correctness script (`src/correct_verify_attn.
py`) is a separate file from the perf benchmark and only prints
`max_diff`/`mean_diff` with no assert — this project's mandatory,
same-run correctness gate (`benchspecs/sparse-attention-kernel/spec.yaml`)
is what actually caught this.

## Gate verification (login node, correctness gate only, reduced protocol)

```python
import sys
sys.path.insert(0, "/pscratch/sd/c/cunyang/msu/hpc-bench/kernel-papers/bench")
from kernelbench import domains, harness, spec
from kernelbench.artifact_registry import discover
from kernelbench.domains.ml import SparseAttentionWorkload

domain = domains.load("sparse-attention-kernel")
sp = spec.load("sparse-attention-kernel")
variant = sp.variant("sparse-attn-structured-mask-kernel-fp16")
impl = discover("sparse-attention-kernel")["stof-binding-attn"]["factory"]("fp16")

w = SparseAttentionWorkload(
    name="stof-gate-blockdiag", variant="sparse-attn-structured-mask-kernel-fp16",
    B=1, H=2, d=64, S=512, pattern="block_local",
    pattern_params={"block": 64, "n_global_blocks": 0}, sf_target=None)
params = {"seed": 42, "precision": "fp16"}
r = harness.run_variant(impl, w, variant, params,
                         reference=domain.REFERENCES["sparse-attention-kernel"],
                         correctness_mode=domain.CORRECTNESS_MODE["sparse-attention-kernel"],
                         warmup_override=1, reps_override=3)
print(r.to_dict())
```
Run as: `LD_PRELOAD=/usr/lib64/libstdc++.so.6 $PY gate_stof.py`.

Result on the `block_local` (full/skip-tile-only) workload: **valid: true**,
`max_scaled_err = 3.44e-4 <= tol 1e-2` (secondary views:
`max_pointwise_rel_err=2.99`, `l2_rel_err=2.63e-4`). Reduced-protocol numbers
only (`warmup=1, reps=3`, shared login-node GPU), non-conforming per
ARTIFACT_GUIDE rule 5.

The `sliding_window` workload's **failing** run (`max_scaled_err=6.83`,
`valid: false`) is recorded above as the evidence for the bug section — per
ARTIFACT_GUIDE rule 4, this failure is itself a result and the gate was
**not** loosened or worked around to make it pass.

`discover_status("sparse-attention-kernel")` confirms `stof-binding-attn` is
discovered `available: True` (with `LD_PRELOAD` set).

## Mask/batch semantics note (checked, not a bug for this integration)

`get_OuterTile_storage`'s own per-batch loop (`for b in range(batch_size):
...`) appends block-pointer entries for every `(batch, row-block)`
combination, producing arrays of length `batch_size*num_m_block + 1`.
Reading `flash_fwd_launch_template.h::run_flash_fwd`'s grid launch (`dim3
grid(num_m_block, params.b, params.h)`) and `bind_fwd_kernel.h`'s
`compute_mask_attn` (`m_block = blockIdx.x`, indexed `[0, num_m_block)` with
NO batch-dependent offset added before indexing into `full_row_ptr`/
`part_row_ptr`) shows the kernel only ever reads the **first**
`num_m_block+1` entries of these arrays for EVERY batch element — i.e., the
per-batch block-pointer construction `get_OuterTile_storage` produces is
inconsistent with how the compiled kernel actually indexes them for
`batch_size > 1` with a per-batch-VARYING mask. This did not affect this
adapter's own gate (the workload's mask is, by this track's own convention,
identical across every batch element, so batch 0's block structure equals
every other batch's), but is flagged here as a related, deeper limitation of
the artifact's mask-format-to-kernel pipeline for anyone reusing this
adapter with a genuinely per-batch-varying mask.

## Provenance

- Artifact commit: `6f2d0cf18f89b8e6343ff7cf9d6689b74002a51f`
- nvcc: 12.9 (`/opt/nvidia/hpc_sdk/Linux_x86_64/25.5/cuda/12.9`)
- Host compiler: g++ 14.3.0 (`/opt/cray/pe/gcc-native/14/bin/g++`)
- GPU arch: `sm_80` (A100)
- Python/torch: `/pscratch/sd/c/cunyang/gnn/plexus_env` — torch 2.8.0+cu128

## Not done

- `block_attn_mask` (the second, ~6300-line extension) not built/exercised
  — confirmed dead/unused by the paper's own actual benchmark script.
- No sweep across the spec's `recommended_subset` shapes or STOF's other 4
  named mask types (causal/longformer/bigbird/dilated) — those would ALL be
  expected to hit the same "part"-tile bitmap bug documented above (they all
  produce mixed tiles at 64x64 granularity), not independently re-verified
  per mask type given the login-node budget and the root cause already being
  conclusively localized to source code, not to a specific mask shape.

## Reproduction on zaratan (2026-09-09)

- Machine: UMD zaratan, A100 MIG 1g.5gb slice (sm_80, driver 595.71.05; job
  ran on gpu-b11-6.zaratan.umd.edu, physical card NVIDIA A100-SXM4-40GB),
  login-node build.
- Toolchain: nvcc 12.8.93 (conda-forge CUDA 12.8), g++ 13.4.0 (conda-forge),
  torch 2.8.0+cu128, Python 3.12.14; no cmake (private throwaway
  `_build_binding_attn/setup_binding_attn_only.py`, torch's
  `CUDAExtension`/`BuildExtension`, ninja backend). Arch flags:
  `-gencode=arch=compute_80,code=sm_80` (sm_80), same as recorded.
- Build: OK, after 3 build-system fixes (none touch `source/`, all in
  `build.sh` / `_build_binding_attn/setup_binding_attn_only.py` / `adapter.py`
  — files this integration itself owns):
  1. **Idempotency check never matched.** `build.sh` tested for
     `binding_attn.cpython-*.so`, but `BuildExtension.with_options(...,
     no_python_abi_suffix=True)` produces a plain `binding_attn.so` — the
     glob never matched, so every run silently did a full CUTLASS rebuild
     instead of skipping. Fixed to test for the literal
     `$OPS_SRC/binding_attn.so`.
  2. **Missing runtime rpath — ninja bypasses `$LDSHARED`.** `build.sh`
     exports `LDSHARED` with an explicit `-Wl,-rpath,<torch>/lib` (documented
     fix for this machine's conda-python distutils quirk — see build.sh's
     header, same technique as `sddmm/fused3s`/`gpa`). `BuildExtension`'s
     `use_ninja=True` path generates its own link command in `build.ninja`
     and does **not** consult `$LDSHARED`, so the built `.so` linked fine
     (via `-L<torch>/lib` at link time) but had NO rpath to `<torch>/lib`
     at all (confirmed: `readelf -d` showed only the conda env's own
     `lib`), so `import binding_attn` failed at runtime with `ImportError:
     libc10.so: cannot open shared object file`, even though compilation
     succeeded. Fixed by passing `extra_link_args=["-Wl,-rpath,<torch_lib>"]`
     explicitly in the `CUDAExtension(...)` call (computed from
     `torch.__file__`, not hardcoded) — `extra_link_args` IS honored by the
     ninja path. Re-verified with `readelf -d`: rpath now includes
     `<torch>/lib`.
  3. **Missing runtime dependency: `matplotlib`.** `source/src/util/masks.py`
     (unmodified artifact code, imported by `adapter.py` for
     `get_OuterTile_storage`) does `import matplotlib.pyplot as plt` at
     MODULE level — unused by `get_OuterTile_storage` itself, but the whole
     module fails to import without it. Not present in this machine's shared
     Python env (was on the `plexus_env` used for the recorded run).
     Vendored via `pip install --target=<artifact>/pylibs matplotlib` (same
     per-artifact-pylibs pattern as `attention-kernel/pat/build.sh`), added
     to `build.sh` (runs before the idempotency check, so it re-verifies on
     every invocation, not just a fresh build) and to `adapter.py`'s
     `_ensure_paths()`. `PYLIBS` is **appended** (not prepended) to
     `sys.path` — `pip --target` pulled in its own numpy as a matplotlib
     dependency, and appending (rather than the `sys.path.insert(0, ...)`
     pat/adapter.py uses) ensures `import numpy` still resolves to the
     shared env's numpy 2.3.3 (the one torch was actually built against),
     not the vendored 2.5.3; verified directly (`import numpy` after
     appending PYLIBS still reports the shared env's numpy 2.3.3).
  4. First build attempt hit the login-node RLIMIT_NPROC/OpenBLAS
     thread-creation contention described in this file's own "MACHINE
     GOTCHAS" section (transient, from concurrent activity on the login
     node, not this artifact) — resolved by retrying with
     `OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1` in the shell, no file change.
- Gate (same reduced-protocol Python harness script as recorded, `warmup=1,
  reps=3`, `--precision fp16`):
  - `block_local(block=64, n_global_blocks=0)`, S=512 (the "part"-tile-free
    workload — the recorded PASS case): PASS, `max_scaled_err = 3.44e-04 <=
    tol 1e-2` — matches the recorded value exactly (3.44e-4).
  - `sliding_window(window=64)`, S=512 (the recorded bug-evidence FAIL case):
    FAIL, `max_scaled_err = 3.96` (`max_pointwise_rel_err = 5.46e4`,
    `l2_rel_err = 1.59`) — same qualitative failure as recorded
    (`max_scaled_err = 6.83`, `max_pointwise_rel_err = 1.9e5`): output
    essentially uncorrelated with the correct masked result, consistent
    with the documented "part tiles computed as fully dense" bug. The
    absolute error magnitude differs from the Perlmutter run (3.96 vs.
    6.83) but both are >2 orders of magnitude past the 1e-2 tolerance and
    both exhibit the same signature (huge `max_pointwise_rel_err`); not
    re-investigated further since the root cause was already conclusively
    localized to source code (see "Real artifact bug found" above), and a
    silently-dense-computed kernel's exact error is expected to vary with
    the specific mask/QKV values, not be bit-reproducible across machines.
- Deviation from the recorded ruling: none in outcome (same PASS/FAIL split
  on the same two patterns, same bug). Three build-system fixes were needed
  to get there on this machine (see above) — none touch `source/` or the
  kernel algorithm.
- Verdict here: BUILT+GATED (with a confirmed artifact bug) — same as the
  recorded ruling.
