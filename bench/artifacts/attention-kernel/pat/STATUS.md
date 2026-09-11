# pat (PAT: Prefix-Aware Attention) — attention-kernel

**Status: BUILT+GATED**

- Paper: "PAT: Accelerating LLM Decoding via Prefix-Aware Attention with
  Resource Efficient Multi-Tile Kernel", ASPLOS'26.
  `PAPER_KEY = conf/asplos/YiZHYSWZZLL26` (matched by title in
  `../../../output/included.json`).
- Artifact: https://github.com/flashserve/PAT.
- Commit cloned: `8cb067f0a31dabe800958f657093f17ac5d7366d` (`main`/`ae`
  branches both point here), `git clone --depth 1`.
- Vendored dependency: `source/third_party/cutlass` — NVIDIA/cutlass,
  commit `dcf215af68a2d08d305076c152a06f201728cd53`, `git clone --depth 1`
  (header-only; PAT's own `setup.py` requires `CUTLASS_ROOT` pointing at a
  checkout with `include/` and `tools/util/include/` — no cutlass build
  step of its own).
- Toolchain: python `/pscratch/sd/c/cunyang/gnn/plexus_env/bin/python`,
  torch `2.8.0+cu128`, nvcc `12.9`
  (`/opt/nvidia/hpc_sdk/Linux_x86_64/25.5/cuda/12.9`), host compiler
  `/opt/cray/pe/gcc-native/14/bin/g++` (GCC 14), GPU arch `sm_80` (A100,
  `-arch=compute_80`, PAT's own `setup.py` hardcodes this — matches our
  hardware with no arch-flag change needed).

## What the artifact actually is, and the wrapping boundary chosen

PAT is a CUTLASS-based decode-attention kernel: for a batch of paged-KV-cache
sequences it identifies shared prefixes across the batch and schedules them
into dedicated CTAs to cut redundant KV-cache reads. `csrc/` holds the CUDA
kernel (`pat_fwd_kernel.h`, `pat_fwd_launch_template.h`, 4 explicit
instantiation `.cu` files — `pat_fwd_split_hdim{64,128}_{fp16,bf16}_sm80.cu`,
i.e. **only head_dim 64 and 128 are compiled**, no hdim=16). `setup.py` is a
standard, unmodified `torch.utils.cpp_extension.CUDAExtension` named
`prefix_attn._prefix_attn`. `prefix_attn/` (the Python package,
`__init__.py`/`data_class.py`/`block_scheduler.py`/`utils.py`) is PAT's own
public API. This adapter calls PAT's own
`prefix_attn.prefix_attn_with_kvcache(q, k_cache_paged, v_cache_paged, tree,
softmax_scale, out)` — the exact pattern `source/test/test.py::
tree_benchmark` uses — **not** PAT's own benchmark scripts
(`benchmark/`, `test/test.py`), per ARTIFACT_GUIDE rule 1. **No artifact
source file was edited.**

`prepare()` does two things beyond generating operands, both legitimate
preprocessing under ARTIFACT_GUIDE rule 2 (the artifact's own format
conversion, timed as preprocessing):
1. Builds PAT's paged KV-cache layout — `k_cache_paged`/`v_cache_paged` of
   shape `(num_blocks, block_size, Hkv, d)` plus a per-sequence
   `block_table` — from the harness's flat `(B,Hkv,Sk,d)` K/V. Every
   sequence in our workload has the SAME `Sk` (no per-sequence length
   variation modeled), so the simplest correct choice was used: each
   sequence gets its own contiguous block range (no prefix sharing is
   invented, since our workload has none — a legitimate, PAT-supported
   "independent sequences" input, exercised verbatim by PAT's own
   `test/test.py::test_tree_attn_manual`). `block_size=16` matches
   `benchspecs/attention-kernel/spec.yaml`'s decode variant's own
   `layout: "paged, block_size=16"` convention.
2. Builds PAT's own scheduling metadata (`SeqGroup`/`PrefixTreeCPP` via
   `build_radix_tree` + `pack_schedule` + `kernel_info.to_gpu`) — the
   artifact's own required step to tell its kernel which CTA processes
   which (query, KV-block) pairs.

**Layout gotcha**, confirmed and handled: PAT's `q`/`out` tensors are
`(B, Sq, H, d)` — head dim is axis 2, NOT axis 1 like this harness's
`(B,H,Sq,d)` convention (confirmed from `prefix_attn/_prefix_attn.pyi`'s
tensor comments and `test/test.py`'s
`q = torch.randn(len(seq_group), 1, nheads_q, head_dim, ...)`).
`to_host()` transposes axes 1,2 back before the harness compares.

**RNG discipline** (the trap documented in
`bench/artifacts/spmm/insum/STATUS.md`'s "Finding, not an Insum bug"
postmortem): Q/K/V are generated with the exact `np.random.default_rng`
formula `kernelbench.domains.ml._qkv` uses — same seed, same shapes, same
call order — so this adapter's operands are bit-identical to the fp64
reference (and every other impl under test). PAT's own
`prefix_attn.generate_random_kv_cache` (`torch.random.manual_seed` +
`torch.randn`) is deliberately NOT used — a different RNG, different bit
sequence from the same integer seed.

## Build (`build.sh`)

Runs PAT's own unmodified `setup.py build_ext --inplace` (not `pip install`
into the shared venv). Two machine-specific compiler workarounds (identical
situation/fix as `bench/artifacts/sddmm/fused3s/build.sh`): `CXX`/`CC`
pointed at `/opt/cray/pe/gcc-native/14/bin/g++`/`gcc` (bypasses the venv
distutils's broken `-B .../compiler_compat` CXX, which otherwise picks up an
ancient GCC-7-era `cc1plus`); `LDSHARED` overridden to a clean link line
(rpaths to this venv's own `torch/lib`, no NERSC-python rpath baked in).
`einops` (a runtime import of `prefix_attn/utils.py`, not present in the
shared venv) installed into an isolated `pip install --target=pylibs/`
directory rather than the shared venv, per the task's environment
instructions.

**`MAX_JOBS=1`, deliberately**: a first build attempt with default (parallel,
`os.cpu_count()`-wide) ninja concurrency had every non-first `.cu` translation
unit SIGKILLed (`cicc died due to signal 9`) — consistent with a
memory-pressure OOM kill on this shared login node (other artifact builds
were running concurrently at the time), not a code error. Each of PAT's 4
CUTLASS `.cu` files is a memory-heavy compile on its own; `MAX_JOBS=1`
serializes them and is a build-concurrency knob, not a kernel-code change.
**Consequence: the full build takes ~16 minutes wall-clock** on this shared
machine (4 CUTLASS `.cu` files + `bindings.cpp`, each several minutes of
`cicc` template instantiation) — well over the nominal per-artifact budget,
but the build did complete successfully and is idempotent (re-running
`build.sh` is a no-op once built), so this is a one-time cost.

## Finding: the same libstdc++ ABI trap as flashattention-t

Immediately after building, `import prefix_attn` failed:
```
ImportError: .../nersc-python/lib/libstdc++.so.6: version `CXXABI_1.3.15' not found
(required by .../prefix_attn/_prefix_attn.cpython-311-x86_64-linux-gnu.so)
```
This is the **identical** root cause documented in detail in
`bench/artifacts/attention-kernel/flashattention-t/STATUS.md`'s "Finding: a
second, distinct libstdc++ ABI trap" section (read there for the full
derivation) — summary: this venv's python is a symlink to a NERSC-provided
interpreter that transitively loads an OLD bundled `libstdc++.so.6`
(missing `CXXABI_1.3.15`) as soon as `torch` is imported anywhere in the
process, before PAT's own `LDSHARED`-baked `RPATH` (which correctly points
at `torch/lib`, confirmed via `readelf -d`) gets a chance to matter — once a
library with a given SONAME is resident, the loader does not load a second
copy, `RPATH` or not. The system's `/usr/lib64/libstdc++.so.6.0.34` **does**
have `CXXABI_1.3.15`. Confirmed fix, identical to flashattention-t's:
`LD_PRELOAD=/usr/lib64/libstdc++.so.6` set in the shell environment
**before** the python process starts (verified: `import torch` then
`from prefix_attn import prefix_attn_with_kvcache, PrefixTreeCPP` succeeds
cleanly with this set; an `os.environ`-based fix from inside an
already-running interpreter is a documented no-op).

**Environment requirement, like CUTLASS_ROOT for the build itself**: every
invocation of this adapter (gate checks, future timed runs) must be prefixed
with `LD_PRELOAD=/usr/lib64/libstdc++.so.6`.

## Shape coverage

PAT's `csrc/` only instantiates head_dim in `{64, 128}` — none of
`kernelbench.domains.ml.smoke_workloads(kernel="attention-kernel")`'s 4
synthetic shapes (all `d=16`) are supported. PAT also only implements decode
(`Sq=1`, unconditionally causal over the full KV history — see
`ml.py`'s `_causal_mask` docstring on why decode causal masking removes
nothing). `prepare()` raises `NotImplementedError` cleanly for anything else
(wrong head_dim, `Sq!=1`, non-causal mask) — the sanctioned
"shape-constrained artifacts" behavior.

## Gate verification

Per the task's own instruction for decode-oriented artifacts ("gate on the
decode-GQA workload, note their home variant in STATUS.md") and because
`--smoke` forces all 4 `d=16` synthetic shapes regardless of `--variant`
(with no per-run try/except in `runner.py`'s loop, so the first
unsupported shape crashes the whole invocation before a result file is
written), gated with a small standalone script exercising ONE real,
supported decode-GQA shape via `kernelbench.harness.run_variant` directly
against this artifact's home variant, `attn-decode-kv-cache-kernel`:

```python
import sys
sys.path.insert(0, "/pscratch/sd/c/cunyang/msu/hpc-bench/kernel-papers/bench")
from kernelbench import domains, harness, spec
from kernelbench.artifact_registry import discover

domain = domains.load("attention-kernel")
sp = spec.load("attention-kernel")
variant = sp.variant("attn-decode-kv-cache-kernel")
impls = discover("attention-kernel")
factory = impls["pat-decode-attn"]["factory"]
impl = factory("fp16")

from kernelbench.domains.ml import AttentionWorkload
w = AttentionWorkload(name="pat-gate-tiny", variant="attn-decode-kv-cache-kernel",
                       variant_kind="decode", B=4, H=8, Hkv=2, d=128, Sq=1, Sk=256, mask="causal")

params = {"seed": 42, "precision": "fp16"}
r = harness.run_variant(impl, w, variant, params,
                         reference=domain.REFERENCES["attention-kernel"],
                         correctness_mode=domain.CORRECTNESS_MODE["attention-kernel"],
                         warmup_override=1, reps_override=3)
print(r.to_dict())
```
Run as: `LD_PRELOAD=/usr/lib64/libstdc++.so.6 $PY gate_pat.py`.

Result: **valid: true**. `max_scaled_err = 1.632e-04 <= tol 1e-2`
(secondary views also recorded: `max_pointwise_rel_err=0.298`,
`l2_rel_err=2.57e-04`). `0.078 ms` median — reduced-protocol numbers only
(`warmup=1, reps=3`, shared login-node GPU), explicitly non-conforming per
ARTIFACT_GUIDE rule 5, not a timing result. `--list`/`discover_status`
confirms `pat-decode-attn` is discovered as `available: True` for the
`attention-kernel` track (with `LD_PRELOAD` set).

## Not done

- No GQA-ratio sweep or prefix-sharing (tree-structured) workload exercised
  — our `AttentionWorkload` doesn't model correlated/shared prefix lengths
  (see `ml.py`'s own note on this), and PAT's prefix-sharing optimization
  (its actual paper contribution) is therefore untested here; only the
  "independent sequences" path was gated.
- No bf16 gate run (fp16 only) — bf16 template instantiations exist and are
  compiled, listed in `PRECISIONS`, but not exercised in this gate.
- No sweep across the spec's `recommended_subset` decode shapes (e.g. the
  saturating `B=1134` configs) — a single small gate check only, per the
  task's login-node budget.

## Reproduction on zaratan (2026-09-09)

- Machine: UMD zaratan, NVIDIA A100-SXM4-40GB (sm_80, driver 595.71.05), gpu
  partition (`bench/gpu_run.sh`), login-node build.
- Toolchain: nvcc 12.8.93 (conda-forge CUDA 12.8), g++ 13.4.0 (conda-forge,
  `$KB_CXX`/`$KB_CC`), torch 2.8.0+cu128, Python 3.12.14. CUTLASS: same
  pinned commit `dcf215af68a2d08d305076c152a06f201728cd53`
  (`source.provenance`/`build.sh`'s `CUTLASS_SHA`, unchanged). Arch flag:
  `-arch=compute_80` (PAT's own `setup.py`, unchanged).
- Build: OK — extension was already built from an earlier pass on this
  machine (`source/prefix_attn/_prefix_attn.cpython-312-*.so` present,
  `build.sh`'s idempotency check short-circuited). `MAX_JOBS=1` (already in
  `build.sh`, machine-neutral default) was in effect; no build-system
  changes needed. The Perlmutter-era libstdc++ ABI trap (`LD_PRELOAD`
  needed before the python process starts) did not need a manual
  workaround here — `bench/env.sh`'s `KB_LD_PRELOAD` default already
  points at `$KB_CUDA_HOME/lib/libstdc++.so.6` and is exported
  automatically by every `gpu_run.sh` invocation.
- Gate: `attn-decode-kv-cache-kernel`, `pat-decode-attn`, fp16,
  `B=4,H=8,Hkv=2,d=128,Sq=1,Sk=256,causal` (same reduced-protocol
  standalone script as recorded, `warmup=1, reps=3`): **PASS**,
  `max_scaled_err=1.632e-04 <= tol 1e-2` (`max_pointwise_rel_err=0.298`,
  `l2_rel_err=2.57e-04` — identical to the recorded run to the last
  printed digit).
- Deviation from the recorded ruling: none.
- Verdict: BUILT+GATED — same as recorded ruling.
