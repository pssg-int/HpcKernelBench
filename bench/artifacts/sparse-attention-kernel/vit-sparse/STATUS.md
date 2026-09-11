# vit-sparse (ipdps-26_vit) — sparse-attention-kernel — STATUS: BUILT+GATED

- Paper: "Achieving Low Latency Inference on High Resolution Images by
  Exploiting Sparsity in Vision Transformers", IPDPS'26.
  `PAPER_KEY = conf/ipps/LiK26` (matched by `artifact_url` in
  `../../../../output/included.json`; no dblp/arXiv id exists for this paper
  yet, per the track's own survey.md).
- Artifact: https://github.com/KLab-AI3/ipdps-26_vit.
- Commit cloned: `170fa82da20a1236885a78f6fb9a5d51dfe7ccc7` (2026-02-18),
  `git clone --depth 1`.
- Toolchain: python `/pscratch/sd/c/cunyang/gnn/plexus_env/bin/python`,
  torch `2.8.0+cu128`. **No CUDA compilation** — this artifact has no
  compiled extension anywhere in the repo (confirmed: no `setup.py`/`*.cu`/
  `*.cpp` outside the vendored `models/DynamicVit|RegionVit|VisionLongformer`
  subtrees, which this adapter does not touch); its actual kernel path is a
  pure-Python composition around `torch.nn.functional.
  scaled_dot_product_attention`.

## What the artifact actually is, and the wrapping boundary chosen

The artifact's documented pipeline is T1 (adjacency construction) -> T2
(reorder + block extraction) -> T3 (tile profiling) -> T4 (Gurobi ILP
scheduling) -> T5 (benchmarking). Reading `src/run_bench.py`'s **actual**
`argparse` (the file "full file read", per the track's own survey.md) shows
its real CLI is `--adj`/`--blocks`/`--device`/`--dtype`/`--heads`/
`--head_dim`/`--warmup`/`--iters`/`--assume_dense_blocks`/`--max_blocks`/
`--seed`/`--out`/`--out_json` — there is **no `--schedule` argument**, even
though the README's stage-by-stage doc claims T5 is invoked as
`run_bench.py --schedule results/schedules/schedule_ic.json ...`. This is a
real README-vs-released-code mismatch (the same species of finding the
track's own survey.md documented for STOF's "100 iterations" claim): T5, as
released, runs directly off T2's `blocks.json` output and never consumes
T4's ILP schedule at all. **Consequence for this integration: the Gurobi
ILP solver (`src/solve_ilp_gurobi.py`) is not needed to exercise the
artifact's actual per-block attention kernel**, and this adapter does not
attempt to install/license Gurobi.

This adapter therefore wraps:
1. **Preprocessing** (`prepare()`, timed once, per ARTIFACT_GUIDE rule 2):
   `src/extract_blocks.py::extract_blocks()` (unmodified) — density-aware
   greedy block extraction from the workload's own fixed mask, converted to
   a `scipy.sparse.csr_matrix`.
2. **The kernel** (`run()`, timed per iteration): `src/run_bench.py::
   attn_block()` (unmodified) — one `scaled_dot_product_attention` call per
   extracted block per batch element, with the block's true adjacency
   submask applied as an additive bias via `build_block_mask_from_adj()`
   (also unmodified).

Neither file was edited. `--assume_dense_blocks` (the artifact's own
speed/accuracy toggle) is never set — the true per-block submask is always
applied, matching the survey's own note that this is the "more correct"
(default) path.

## Finding: attn_block()'s per-block softmax has no cross-block merge

`run_bench.py` (full file read) never threads any running-max/running-sum
state between per-block `attn_block()` calls — each block's output is an
**independently normalized** softmax over that block's own `[BQ,BK]` score
submatrix. This is mathematically correct as a wrapper of the FULL `SxS`
masked-attention kernel (the operation this track's fp64 reference computes)
**only when every query row's entire mask support is contained inside
exactly one extracted block** — otherwise two blocks would each normalize a
different partial softmax independently, which is not equal to a joint
per-row softmax over the row's full support (adding the missing online-
softmax merge logic would be touching the artifact's own kernel/algorithm
code, which ARTIFACT_GUIDE rule 3 rules out).

**Empirically checked** (see this adapter's development notes / the checks
below, reproducible from `kernelbench.domains.ml.SparseAttentionWorkload`):
for a `sliding_window` mask (S=1024, window=102, i.e. the spec's own
`sliding-s1024-sf01` shape) with tile sizes 64/128/256, **61%–99% of rows
have their mask support split across more than one extracted block** — the
greedy row-major rectangle-growing extraction does not align with a diagonal
band's natural "staircase" shape. The one pattern family in this module for
which containment holds **exactly** is `block_local` with
`n_global_blocks=0` (pure block-diagonal, no CLS-style global block) and a
tile size equal to the pattern's own `block` parameter: verified
`S=1024, block=64` -> exactly 16 blocks, `0/1024` rows split, full nnz
coverage (`sum(block.nnz) == A.nnz`).

`adapter.py::_check_single_block_containment()` therefore **validates** this
property in `prepare()` from the workload's own extracted blocks and raises
`NotImplementedError` (the sanctioned "shape-constrained artifact" behavior,
same as `bench/artifacts/attention-kernel/pat/adapter.py` and
`bench/artifacts/sddmm/fused3s/adapter.py`) for any workload/tile-size
combination that fails it, rather than silently producing a
softmax-normalization-wrong number. **Gate workload used below is
`block_local(n_global_blocks=0)`**, the one family proven exact.

## Real artifact bug found: fp16 mask-bias dtype mismatch

`attn_block()` (`src/run_bench.py` L101-136) hardcodes its additive mask
bias tensor as `torch.zeros((...), dtype=torch.float32)` regardless of the
query/key/value tensors' own dtype. Calling it with fp16 Q/K/V (the
artifact's own advertised default — `run_bench.py --dtype fp16` is the
CLI default, and the README's FlashAttention note implies fp16 is the
intended precision) raises, reproducibly:
```
RuntimeError: invalid dtype for bias - should match query's dtype
```
from torch's own `scaled_dot_product_attention`. This is a genuine bug in
the released artifact, not an integration mistake on this adapter's side —
confirmed by tracing the exact `attn_bias` construction line in
unmodified source. Per ARTIFACT_GUIDE rule 3 (do not touch kernel code),
this adapter does **not** patch `attn_block()` to fix the dtype; instead
`PRECISIONS = ["fp32"]` only. bf16 was not separately tested (would hit the
identical bug).

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
impl = discover("sparse-attention-kernel")["vit-sparse-block-tiled-attn"]["factory"]("fp32")

w = SparseAttentionWorkload(
    name="vitsparse-gate-blocklocal", variant="sparse-attn-structured-mask-kernel-fp16",
    B=2, H=4, d=32, S=1024, pattern="block_local",
    pattern_params={"block": 64, "n_global_blocks": 0}, sf_target=None)
params = {"seed": 42, "precision": "fp32", "block_tile": 64, "rho_min": 0.99}
r = harness.run_variant(impl, w, variant, params,
                         reference=domain.REFERENCES["sparse-attention-kernel"],
                         correctness_mode=domain.CORRECTNESS_MODE["sparse-attention-kernel"],
                         warmup_override=1, reps_override=3)
print(r.to_dict())
```
Run as: `LD_PRELOAD=/usr/lib64/libstdc++.so.6 $PY gate_vitsparse.py`.

Result: **valid: true**. `max_scaled_err = 1.47e-06 <= tol 1e-2` (secondary
views: `max_pointwise_rel_err=0.196`, `l2_rel_err=6.45e-07`) — a clean pass
with wide margin. Preprocessing (block extraction, 16 blocks over 1024x1024,
6.25% density) took ~1.6s (dominated by Python-level per-row CSR slicing in
the artifact's own `extract_blocks.py`, not this adapter). Reduced-protocol
numbers only (`warmup=1, reps=3`, shared login-node GPU), explicitly
non-conforming per ARTIFACT_GUIDE rule 5 — not a timing result.
`discover_status("sparse-attention-kernel")` confirms
`vit-sparse-block-tiled-attn` is discovered `available: True`.

A negative check (`sliding_window` pattern, which fails single-block
containment) was also run and confirmed `prepare()` raises
`NotImplementedError` cleanly, per the "Finding" section above — no silent
wrong-number path exists.

## Mask/pattern provenance

`block_local(n_global_blocks=0, block=64)` is a synthetic proxy this module
builds (`kernelbench.domains.ml._block_local_mask`) — the closest analogue
to RegionViT's fixed regional block-sparse pattern this project can honestly
construct without a trained ViT checkpoint (see `ml.py`'s own docstring on
this). It is **not** one of the 3 real ViT-model-derived masks
(DynamicViT/RegionViT/VisionLongformer) the paper's own headline results
use — this integration gates the artifact's real kernel code against a
structurally-similar but synthetic block-sparse workload, documented per
the task's "generate our workload mask within that family and document"
guidance (the artifact's own `gen_adj.py --mode load` DOES accept an
arbitrary externally-supplied `.npz`/`.npy`/`.pt` adjacency, so nothing about
the artifact itself restricts which mask family can be fed in — the
restriction here is purely the single-block-containment property `attn_block
()`'s missing cross-block merge requires, independent of mask origin).

## Not done

- No sweep across tile sizes / larger S (S=16384 shapes) — a single small
  gate check only, per the task's login-node budget.
- bf16 not tested (would hit the same fp16-class dtype bug as fp16).
- The Gurobi ILP scheduler (T4) and structural reordering (T2's hc/ic/rcm)
  are not exercised at all — confirmed unnecessary for T5's actual (as-
  released) execution path, but this does mean the paper's own headline
  "up to 2.1x over FlashAttention fixed tiling" claim (which does depend on
  ILP-optimized tile assignment) is not what this integration measures;
  this integration measures the artifact's real per-block kernel under a
  simple density-threshold block partition instead.

## Reproduction on zaratan (2026-09-09)

- Machine: UMD zaratan, A100 MIG 1g.5gb slice (sm_80, driver 595.71.05; job
  ran on gpu-b11-6.zaratan.umd.edu, physical card NVIDIA A100-SXM4-40GB),
  login-node build.
- Toolchain: torch 2.8.0+cu128, Python 3.12.14; no nvcc/cmake/host compiler
  involved (pure-Python artifact, no compiled extension).
- Build: OK (`build.sh`'s import/smoke check passed: `extract_blocks`,
  `build_block_mask_from_adj`, `attn_block` all importable). Build-system
  changes: none.
- Gate (same reduced-protocol Python harness script as recorded, `warmup=1,
  reps=3`, `--precision fp32`):
  - `block_local(block=64, n_global_blocks=0)`, B=2,H=4,d=32,S=1024 (the
    proven single-block-containment case): PASS, `max_scaled_err =
    1.47e-06 <= tol 1e-2` — matches the recorded value exactly (1.47e-06).
  - `sliding_window(window=102)`, S=1024 (negative containment check):
    `prepare()` raised `NotImplementedError` cleanly, as recorded — no
    silent wrong-number path.
- Deviation from the recorded ruling: none — same PASS value to 3
  significant figures, same clean `NotImplementedError` on the
  containment-violating pattern.
- Verdict here: BUILT+GATED — same as the recorded ruling.
