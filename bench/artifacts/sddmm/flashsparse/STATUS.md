# flashsparse (FlashSparse) — sddmm

**Status: BUILT** (both extensions compile and load; correctness gate
**INCOMPLETE** — a genuine, evidenced data-loss bug was found in the
compiled kernel's own residue-tile Store() logic, documented below rather
than papered over. Not GATED.)

- Paper: "FlashSparse: Minimizing Computation Redundancy for Fast Sparse
  Matrix Multiplications on Tensor Cores", PPoPP'25. `PAPER_KEY =
  conf/ppopp/ShiLXFWW25`.
- Artifact: https://github.com/ParCIS/FlashSparse. `source/` here is a
  SYMLINK to `../../spmm/flashsparse/source` (read-only reuse, not a
  re-clone) — same commit as the spmm track's adapter:
  `168764612c1a2ad5b8661be22cfb6a8ae044bada` (2025-10-05).
- Built fresh here (the spmm adapter's `fs_setup.py` builds neither of
  these): `FS_SDDMM` (`source/FlashSparse/SDDMM/src/{benchmark.cpp,
  sddmmKernel.cu}`) + `FS_Block` (`source/FlashSparse/Block/example.cpp`,
  CPU/OpenMP), via this directory's own `fs_sddmm_setup.py`. Zero source
  files touched (`source/` byte-identical to the spmm symlink target).

## Selection rationale

Same recency-tiebreak rationale as the spmm track's flashsparse adapter:
most recent (PPoPP'25) entry in the RoDe/FlashSparse tensor-core line this
track's `sddmm-tensorcore-blocked-fp16` variant is built around.

## What was wrapped (rule 1)

`FS_SDDMM.forward_gen_fp16_gnn` (`sddmm_gen_forward_gat_gnn` ->
`sddmm_gen_forward_cuda_kernel_gat`, the SAME compiled `__global__` kernel
the artifact's self-attention-only "_gat" entry point also calls — confirmed
by reading both host wrappers: `_gnn` just passes two distinct matrices
instead of the same one twice) is the general (row-operand + col-operand)
SDDMM entry point, called completely unmodified. Its structural
preprocessing, `FS_Block.blockProcess_sddmm_balance` (the ONLY preprocessing
path in the artifact returning the 4 arrays `forward_gen_fp16_gnn` actually
takes — `FS_Block_gpu.preprocess_gpu_fs`, used by the spmm adapter, returns
only 3 and is not usable here), is likewise called unmodified. Both are
CPU-resident-array-free at the kernel-call boundary: `forward_gen_fp16_gnn`
takes CUDA tensors directly (no internal H2D copy, unlike the `_gat`
non-`_gnn` wrapper) — a cleaner boundary than the spmm track's FS_SpMM.

## Verified: the task's flagged `values=1.0` concern is a non-issue for SDDMM

Traced `FS_SDDMM`'s `Store()` (`output_tile.h`) and `FS_Block::
blockProcess_sddmm`'s value-generation loop directly: `values` (an `int`
array) is a pure 0/1 structural VALIDITY MASK gating which (row_local,
col_local) slots the kernel writes at all (`if (*(values_+...) != 0)
*(output_matrix_+...) = output_fragment_[...]`) — the kernel NEVER
multiplies by a caller-supplied edge weight, from EITHER preprocessing path.
This is the identical situation `kernelbench.impls.gpu_cuda.TorchSDDMM`
(`torch.sparse.sampled_addmm`) and `../fused3s/adapter.py` already handle:
this adapter applies `S[i,j]` in Python, in `to_host()`, after the CUDA call
and the layout reindex below. So whether `FS_Block_gpu`'s SEPARATE (unused
here) preprocessing path hardcodes `1.0` is moot — `FS_SDDMM`'s kernel was
never going to consume real edge weights from any preprocessing path.

## The real engineering problem, and what was found

FS_SDDMM's output tensor is laid out in the kernel's own tensor-core
fragment / row-window-tile order, not the caller's CSR nnz order (same
category of problem as `../fused3s/adapter.py`'s TC-block reindex).
`layout_decode.py` replays `Store()`'s index arithmetic (traced line-by-line
from unmodified artifact source — see that file's module docstring for the
full derivation) to map every position the kernel writes back to
`(real_row, real_col)`.

**Full 16-wide tiles** (`(id+1)*16 <= v_size`, the majority of tiles for any
window with >=16 merged columns) and **small residue tiles**
(`residue = v_size mod 16 < 8`) were derived and INDEPENDENTLY
cross-verified two ways: (1) the mask array's own write-side layout
(`FS_Block::blockProcess_sddmm`) and the kernel's read-side pointer
arithmetic land on the exact same formula by construction, and (2) direct
differential testing against the compiled kernel (synthetic 8-row windows,
`v_size` in {16,17,20,24,25,32,33,40,48,64,100}, all-ones dense operands) —
the kernel's actual nonzero-output count matched the expected real-edge
count exactly in every one of these cases, with the predicted `(row,col)`
set consistent.

**IMPORTANT FINDING (evidenced, not a decode bug on this integration's
side): the "8 <= residue < 16" residue-tile sub-case has a genuine data-loss
bug in the compiled kernel.** The same differential test at
`v_size in {9, 15, 31, 63}` (all reduce to `residue in {9, 15}` in the
final, partial tile) showed the kernel's actual nonzero-output count
UNDER-counting the true edge count (e.g. `v_size=9`: 8 written vs. 9 real
edges; `v_size=15`: 12 vs 15; `v_size=31`: 26 vs 31; `v_size=63`: 58 vs 63),
with extra nonzero entries appearing at positions the mask marks invalid —
i.e. two or more distinct `(row_local, col_local)` slots are being written
to the SAME physical output position by `Store()`'s own pointer arithmetic,
clobbering each other, for at least `residue == 15` (reproduced in every
trial) and inconsistently for `residue == 9` depending on whether it is the
window's only tile (`v_size=9`, id=0: collides) or a later tile in a
multi-tile window (`v_size=25`, id=1, residue=9: does NOT collide) — a
genuinely input-shape-dependent artifact bug, not a fixed offset error this
integration mis-derived (the SAME formula, independently verified correct
for every full-tile and `residue<8` case above, is what produces the
collision here; hand-tracing `Store()`'s sequential, mutating
`output_matrix_ += ...` statements for the `col1`-block case shows the
computed offset can exceed the entry's own `v_size*8`-element allocated
region entirely, e.g. `residue=15, warpin_id%4=3` -> offset ~161 against a
120-element budget).

Per ARTIFACT_GUIDE.md rule 3 (never patch kernel logic), this is not
something to "fix" from the adapter side — the compiled kernel is used
completely unmodified throughout. `layout_decode.build_csr_to_output_perm`
now takes the kernel's actual output tensor size and raises a clearly-named
`RuntimeError` (not a bare `IndexError`) the moment its own replay of
`Store()`'s arithmetic would address outside that region, rather than
silently mis-mapping or crashing uninformatively.

## Consequence for gating (rule 4 — a real result, not a workaround)

Any real-world graph will have SOME row-windows whose final tile lands at
`residue in [8,16)` (roughly half of all non-multiple-of-16 window sizes),
so this affects a nontrivial, input-dependent fraction of every practical
matrix — including this track's own smoke set and every SuiteSparse/GNN
matrix tried. `prepare()` therefore reliably raises the `RuntimeError` above
for `sddmm-tensorcore-blocked-fp16 --impl flashsparse-sddmm` on real inputs.
This is disclosed here as a genuine artifact-kernel finding, not resolved by
loosening the gate or guessing at a "fixed" offset without kernel-side
evidence to justify it.

## Gate verification (login node)

```
PY=/pscratch/sd/c/cunyang/gnn/plexus_env/bin/python
LD_PRELOAD=/usr/lib64/libstdc++.so.6 $PY -m kernelbench.runner --kernel sddmm \
    --variant sddmm-tensorcore-blocked-fp16 --impl flashsparse-sddmm \
    --precision fp16 --smoke --warmup 1 --reps 3
```

Result: `prepare()` raises the named `RuntimeError` on `smoke-uniform
dim=32` (first workload) — a clean, informative failure (not a crash),
consistent with the finding above. **Not marked UNSUPPORTED** (rule 8 is for
a workload SHAPE the artifact structurally cannot handle, e.g. fixed head
dims; this is a data-loss bug triggered by the sparsity pattern's own
row-window residue distribution, present in essentially every real input) —
recorded here as a failed/incomplete gate instead.

`--kernel sddmm --list` and `--variant sddmm-csr-kernel-f32` (the DEFAULT
variant `smoke_all.sh` exercises, no `--impl` filter) both resolve this
adapter's precision to `"fp32"`, which `FlashSparseSDDMM.__init__` correctly
declines via `NotImplementedError` (this adapter is fp16-only) — so this
finding does **not** break `smoke_all.sh`'s green/red count; it only
surfaces under the fp16 tensor-core variant this adapter is actually built
for.

## Not done

- No general fix for the residue-collision case — would require either (a)
  independently reverse-engineering a DIFFERENT, non-colliding preprocessing
  parameterization the compiled kernel's hardcoded `16`/`64`/`8` constants
  would tolerate (risky, unverified, and arguably reimplementing behavior
  the released kernel doesn't actually provide), or (b) patching the kernel
  itself (prohibited by rule 3). Neither was attempted.
- No `forward_gen_tf32_gnn` (tf32 path) — out of scope, same fp16-only
  choice as the spmm track's flashsparse adapter.
- No sweep across the 20-graph `recommended_subset` or the full
  warmup=3/reps=10 protocol — moot until/unless the finding above is
  resolved upstream.

## Reproduction on zaratan (2026-09-09)

- Machine: UMD zaratan, A100 (full SXM4 40GB, `gpu-b11-6`, sm_80) via
  `bench/gpu_run.sh`, login-node build.
- Toolchain: nvcc 12.8.93 (conda-forge CUDA 12.8), g++ 13.4.0, torch
  2.8.0+cu128, Python 3.12.14. `fs_sddmm_setup.py` build succeeded cleanly
  (both `FS_SDDMM` and `FS_Block` extensions), no build-system changes.
- Build: OK. Build-system changes: none.
- Gate: `sddmm-tensorcore-blocked-fp16` smoke fp16, first workload
  (`smoke-uniform dim=32`): `prepare()` raised
  `RuntimeError: FlashSparse SDDMM layout decode: computed output position
  752159 >= kernel output size 752112 -- this is the KNOWN residue-tile
  collision/overflow issue documented in STATUS.md ...` — the exact same
  message and same category of failure recorded above (same "col1" residue
  [8,16) collision), reproduced on different hardware (A100 SXM4 vs.
  Perlmutter's A100 PCIe) and a different toolchain (conda-forge CUDA 12.8/
  gcc13 vs. NVIDIA HPC SDK CUDA 12.9/gcc14) — strong independent evidence
  this is a genuine compiled-kernel bug, not a toolchain- or
  hardware-specific artifact of the original run.
- Deviation from the recorded ruling: none.
- Verdict here: BUILT (gate incomplete: same documented residue-tile
  data-loss bug) — equals the recorded ruling.
