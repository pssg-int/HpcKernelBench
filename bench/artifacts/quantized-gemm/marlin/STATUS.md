# marlin (MARLIN) — quantized-gemm

**Status: BUILT+GATED (reused build) err=5.18e-05..3.65e-04 at decode/M<=64
shapes (all <= tol 1e-3); a REAL precision-degradation bug found in
marlin's own "par>1" batched dispatch path at M in {128,512,2048} (see
below) — not gated around, honestly reported.**

- Paper: "MARLIN: Mixed-Precision Auto-Regressive Parallel Inference on
  Large Language Models", PPoPP'25. `PAPER_KEY = conf/ppopp/FrantarCCHA25`.
- Artifact: https://github.com/IST-DASLab/marlin
- Commit: `1f25790bdd49fba53106164a24666dade68d7c90` (identical to
  `artifacts/gemv/marlin` — see "Reuse" below, no second clone).
- Toolchain: `nvcc` 12.9, GPU `sm_80` (A100-PCIE-40GB). Python
  `/pscratch/sd/c/cunyang/gnn/plexus_env/bin/python`, torch `2.8.0+cu128`.
  `LD_PRELOAD=/usr/lib64/libstdc++.so.6` required (CXXABI mismatch, same as
  every other torch-extension artifact in this project).

## Selection rationale (2026-09-05 baseline-selection revision)

**Core baseline, per `output/baseline_selection.md`'s quantized-gemm
section**: `kernel_centrality.json`'s `quantized-gemm|conf/ppopp/
FrantarCCHA25` entry rates centrality `core`, regime `matches` — "MARLIN is
the canonical hand-written kernel this spec's variant is built around — its
exact per-layer LLaMA/Falcon-180B shapes are copied verbatim into the
spec's recommended_subset." This is the primary reason `tilus`/`qfactory`
were demoted to competitors: MARLIN (and FP6-LLM, MXBLAS below) are the
core, kernel-level, regime-matching papers the revised rule now requires
ahead of DSL/compiler component papers.

## Reuse, not a rebuild (task brief's explicit instruction)

This artifact is **already built for the gemv track**
(`artifacts/gemv/marlin/`: `build.sh` compiled marlin's real ahead-of-time
CUDA extension `marlin_cuda` via `pip install --target=vendor`). Marlin's
compiled kernel has no notion of "gemv" vs "quantized-gemm" — it is the
identical `.so` either way; only each track's own adapter performs a
different shape mapping. This artifact's `build.sh` therefore does
**not** compile anything:
1. symlinks `source/` -> `../../gemv/marlin/source` (same commit, one clone);
2. imports `marlin`/`marlin_cuda` directly from
   `../../gemv/marlin/vendor/` — **no local `vendor/` exists here at all**.

`source.provenance` documents the reuse pointer instead of a `reclone`
command (there is nothing to re-clone independently — see that file).

## Shape mapping — simpler than gemv's (no transpose trick)

`ml.py`'s `QuantGemmWorkload` already models `C[M,N] = A[M,K] @
dequantize(W[K,N])` — i.e. A is the activation, W is the weight — exactly
marlin's own `C[m,n] = A_act[m,k] @ B_weight[k,n]` convention, with
`M<->m` (batch), `K<->k` (infeatures), `N<->n` (outfeatures). Unlike the
gemv adapter (which had to role-swap onto marlin's `m=1` slot because
gemv's own `A` plays the *weight* role there), this track's `M` maps
directly onto marlin's own batch dimension `m` — and per
`marlin_cuda_kernel.cu`'s `marlin_cuda()`, `m` has **no tile-alignment
constraint at all** (the kernel internally chunks any `m` into blocks of
16 rows, see its own `for (i = 0; i < tot_m_blocks; i += 4)` loop) — so
**no M-padding is ever needed** in this adapter, unlike gemv's.

## Hard constraints (marlin's `Layer.__init__`, unmodified)

`infeatures % 128 == 0`, `outfeatures % 256 == 0`, `groupsize in {-1,
128}`. Checked exhaustively against `ml.py`'s `_QGEMM_SHAPES` (all 23
real `recommended_subset` entries): every K is already a multiple of 128
and every N a multiple of 256 — **zero padding ever fires on the real
registry**; padding logic (zero-pad K up to 128, N up to 256, exact for
the gate since padded rows/cols are zero-weight) only exercises on two of
the three `--smoke` shapes (`K=96`->128, `N=384`->512).
`group_size_eff` must be exactly 128 (marlin's grouped mode) or exactly
`K` (the domain's per-column fallback, mapped to marlin's `groupsize=-1`);
any other effective group size (the `smoke-qgemm-w3-g64-batched` shape,
`group_size_eff=64`) is not representable by marlin's 2-value enum ->
`NotImplementedError`. `bits` must be exactly 4 (marlin hard-codes
`maxq=2**4-1`) -> `NotImplementedError` for INT3.

## Fairness

Same `w.quantize()` codes/scale as `reference_qgemm` and every other impl
(`tilus`, `qfactory`, `numpy-dequant-gemm`) — fed to marlin's own
`Layer.pack(linear, scales)`, which takes a DEQUANTIZED fp16 weight +
matching scale and internally re-derives `round(w/s)`, lossless here since
`w = codes*scale` divided by `s` recovers `codes` exactly (identical
technique to `gemv/marlin/adapter.py` and `tilus/adapter.py`).

## Gate results (login node, functional check only)

```
PY=/pscratch/sd/c/cunyang/gnn/plexus_env/bin/python
cd /pscratch/sd/c/cunyang/msu/hpc-bench/kernel-papers/bench
LD_PRELOAD=/usr/lib64/libstdc++.so.6 $PY -m kernelbench.runner \
    --kernel quantized-gemm --variant qgemm-w4a16-decode-kernel \
    --impl marlin-w4a16-gemm --smoke
```

Result: **2/2 valid, 1 correctly UNSUPPORTED** (bits=3, a genuine marlin
capability limit, not an adapter gap):

| smoke shape | bits/group | max_scaled_err | note |
|---|---|---|---|
| smoke-qgemm-w4-g128-decode (M=8,K=256,N=384) | INT4 g=128 | 2.14e-04 | N padded 384->512 |
| smoke-qgemm-w4-percol-decode (M=4,K=96,N=128) | INT4 g=96 (per-column) | 3.65e-04 | K padded 96->128, groupsize=-1 |
| smoke-qgemm-w3-g64-batched (M=64,K=192,N=256) | INT3 g=64 | -- | UNSUPPORTED: bits=3 (marlin is INT4-only) |

Real `recommended_subset` shapes, `qgemm-w4a16-decode-kernel` (tol 1e-3
parsed):

| shape | max_scaled_err |
|---|---|
| qgemm-llama-2-7b-qkv_proj-decode-m1 (M=1,K=4096,N=12288) | 5.18e-05 |
| qgemm-gpt-neox-20b-lm_head-decode-m8 (M=8,K=6144,N=50432) | 5.79e-05 |

All comfortably within tolerance, same order of magnitude as
`numpy-dequant-gemm`'s own ~1e-4 baseline (see `ml.py`'s docstring).

## REAL BUG FOUND: marlin's "par>1" batched-M dispatch has a precision
regression, reproducible, not an adapter artifact

`qgemm-w4a16-batched-kernel`'s own correctness text does NOT parse via
`spec.py` (documented gap, same as `tilus`/`qfactory`: "identical gate...
as", not "same... as" -- tolerance shows as `None`/unparsed in `--list`).
Held informally against the decode variant's own 1e-3 (which the spec text
says should be identical), the SAME real shape
(`qgemm-llama-2-7b-qkv_proj`, K=4096, N=12288) shows a striking,
non-monotonic jump in `max_scaled_err` as M grows into marlin's own
"parallel batch-of-64" dispatch path:

| M | max_scaled_err | marlin's own kernel dispatch (`marlin_cuda()` in `marlin_cuda_kernel.cu`) |
|---|---|---|
| 1 (decode) | 5.18e-05 | `prob_m<=16`: thread_k=128/thread_n=128, `par=1` |
| 8 (decode) | ~2-6e-05 | same small-batch path, `par=1` |
| 64 (batched) | 7.43e-05 | `thread_m_blocks==4` exactly (not `>4`): `par=1` still, no parallel-batch split |
| 128 (batched) | **1.51e-02** | `thread_m_blocks=8>4` -> `par=2` (parallel batch-of-64 split engages) |
| 512 (batched) | **5.84e-03** | `par=8` |
| 2048 (batched) | **3.34e-03** | `par` capped at `max_par=16`, dispatched as two internal launches (par=16 then par=15) |

Every M<=64 case (where `marlin_cuda()`'s own `if (thread_m_blocks > 4)`
branch — the one that sets `par>1` and reshapes the problem into parallel
64-row batches — never fires) lands at ~5e-5-7e-5, matching the decode
regime and the `numpy-dequant-gemm` baseline. Every M>64 case (where that
branch DOES fire) lands 1-2 orders of magnitude higher, and NON-
monotonically in M (128 is worse than 512, which is worse than 2048) —
inconsistent with a benign "more output elements -> larger order-
statistic max" explanation (which would predict roughly monotonic growth
with element count, not a peak at M=128). This points at a genuine
correctness/precision issue specific to marlin's own `par>1` code path,
not this adapter (no adapter-side padding or reshaping differs between
these M values — K=4096 and N=12288 are both already tile-aligned for
every M tested here, so the ONLY thing changing is `prob_m` itself, which
flows straight into marlin's own kernel). Circumstantially consistent with
a comment already IN marlin's own source
(`marlin_cuda_kernel.cu`, inside the `par>1` branch): *"Note that parallel
> 1 currently only works for inputs without any padding"* — an explicit,
upstream-acknowledged caveat about this exact code path's fragility, even
though the specific shapes tested here have no padding by this adapter's
own accounting (K, N already tile-aligned) and marlin still shows elevated
error, suggesting the caveat's scope is broader than just "external"
padding.

**Not further root-caused** (would require reading/instrumenting
`marlin_cuda_kernel.cu`'s split-K/parallel-batch reduction path, out of
this integration's login-node/time budget) and **not gated around**: this
is reported as-is, per ARTIFACT_GUIDE rule 4. Since
`qgemm-w4a16-batched-kernel`'s tolerance is unparsed track-wide (pre-
existing gap, not introduced here), no run at ANY M currently produces a
"valid" result for this variant regardless of this finding — but a reader
comparing against the decode variant's stated-identical 1e-3 should know
M in {128, 512, 2048} would fail it while M in {1, 8, 64} would not.

## Real memory constraint found (login-node scope, not a marlin bug)

The single largest `recommended_subset` shape,
`qgemm-falcon-180b-fused_fc2-decode` (K=74240, N=14848 — MARLIN's own
extreme-K benchmarked shape per the spec), was killed by the OS
(`exit 137`, SIGKILL) when attempted at any M on this login node.
`reference_qgemm` and this adapter each materialize a full
`(K,N)` fp64 dequantized-weight array (~8.8 GB) plus several fp64
temporaries (dequant, zero-padded copy, fp16-cast staging before the GPU
transfer) — a transient footprint high enough to trigger the shared login
node's process-level OOM protection even though `free -g` reports ~380 GB
available system-wide (this machine evidently caps single-process memory
below that, or another concurrent job's usage pushed the shared node over
threshold at the time — not independently confirmed which, out of scope
to chase further). This is a property of `reference_qgemm`'s / this
adapter's own fp64 host-side dequantization at this ONE extreme shape, not
of marlin's kernel or GPU memory (the actual `.cuda()` tensors involved,
~1 GB combined, are trivial for an A100-40GB) — every OTHER
`recommended_subset` shape (including the second-largest, `Falcon-180B
fused_attn_fc`, K=14848 N=75264, and `GPT-NeoX-20B lm_head`, tested above)
ran without issue. Belongs on a compute-node allocation per
DOMAIN_GUIDE.md's own "no compute-node jobs" login-node scope for this
integration pass; not investigated further here.

## Not done (out of this integration's login-node budget)

- No sweep across the full 23-shape x 13-M-value `recommended_subset`
  registry -- a handful of representative shapes at both decode and
  batched M values (above), per ARTIFACT_GUIDE rule 5.
- The `par>1` precision bug above was characterized empirically (5 M
  values on one shape) but not root-caused in marlin's CUDA source.
- The Falcon-180B `fused_fc2` extreme-K/N shape was not gated at any M
  (login-node OOM, see above) -- belongs on a compute-node allocation.
- `qgemm-w8a8-mx` does not apply to marlin (weight-only W4A16, not
  both-operand quantization) -- see `mxblas/STATUS.md`.

## Reproduction on zaratan (2026-09-09)

- Machine: UMD zaratan, A100 MIG 1g.5gb slice (sm_80, 14 SMs, ~5 GB; host GPU
  reports as NVIDIA A100-SXM4-40GB via nvidia-smi under the MIG partition),
  login-node build (reused, not rebuilt -- see below).
- Toolchain: nvcc 12.8.93 (conda-forge CUDA 12.8), g++ 13.4.0 (conda-forge,
  env.sh's `$CC`/`$CXX`), torch 2.8.0+cu128, Python 3.12.14. Commit
  unchanged: `1f25790bdd49fba53106164a24666dade68d7c90` (identical to
  `gemv/marlin`, reused vendor/).
- Build: OK, after one real fix, shared with `gemv/marlin`: this script's
  own `cd "$(dirname "${BASH_SOURCE[0]}")"` ran BEFORE `source
  ".../toolchain.sh"` (also resolved via `$(dirname "${BASH_SOURCE[0]}")`),
  so once cwd had already changed, the toolchain.sh path doubled and failed
  with "No such file or directory" -- the identical invocation-order bug
  documented in `gemv/marlin/STATUS.md`'s "Reproduction on zaratan" section
  (present since the original baseline commit, latent only because the
  reference machine invoked build.sh by absolute path). Fixed the same way:
  `source toolchain.sh` moved before the `cd`. `gemv/marlin/build.sh` was
  built first (see that STATUS.md) so this script's own reuse-check
  (`gemv/marlin/vendor/marlin` must exist) passed and no compilation
  happened here, exactly as designed. No change to `source/` (a symlink to
  `gemv/marlin/source`) or marlin's own code.
- Gate: `--variant qgemm-w4a16-decode-kernel --impl marlin-w4a16-gemm
  --smoke`: **2/2 valid, 1 correctly UNSUPPORTED** -- smoke-qgemm-w4-g128-
  decode err 2.14e-04, smoke-qgemm-w4-percol-decode err 3.65e-04 (tol
  `1e-3`), smoke-qgemm-w3-g64-batched UNSUPPORTED (bits=3, marlin is
  INT4-only) -- identical errors to the recorded run.
- Deviation from the recorded ruling: none for the `--smoke` gate reproduced
  here (2/2 valid, 1 unsupported, same errors). The recorded ruling's
  additional `recommended_subset`/`par>1`-precision-bug/OOM findings (a
  broader exploratory sweep beyond the quoted `--smoke` runner command)
  were not re-run in this reproduction pass -- out of scope for a
  build+gate reproduction of the quoted command; not contradicted by
  anything observed here.
- Verdict here: BUILT+GATED -- equals the recorded ruling.
