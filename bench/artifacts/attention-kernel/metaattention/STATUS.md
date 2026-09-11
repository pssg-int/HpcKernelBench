# metaattention (MetaAttention) — attention-kernel

**Status: BUILT+GATED**

- Paper: "MetaAttention: A Unified and Performant Attention Framework across
  Hardware Backends", PPoPP'26. `PAPER_KEY = conf/ppopp/ChenC0XMM0X00W026`
  (matched by title in `../../../output/included.json`).
- Artifact: https://github.com/SJTU-IPADS/MetaAttention, checked out on the
  `PPoPP_AE` branch — the repo's main-branch README explicitly points there
  ("The PPoPP'26 Artifact Evaluation is available on the `PPoPP_AE` branch"),
  and that branch has its own AE-specific README distinct from the generic
  main-branch quickstart.
- Commit cloned: `52575e61ef4202fe844aa857cc4ed843a81437d1` (`git fetch
  --depth 1 origin PPoPP_AE`, checked out as a local branch).
- Toolchain: python `/pscratch/sd/c/cunyang/gnn/plexus_env/bin/python`, torch
  `2.8.0+cu128`, `tilelang==0.1.13` (PyPI wheel — the AE's own
  `pyproject.toml` pins a specific git rev of `tile-ai/tilelang`, but the
  PyPI release was sufficient to compile and run correctly; not independently
  diffed against that exact rev, see "Not done"). No nvcc/g++ invocation of
  our own — TileLang JIT-compiles its own generated kernels internally.

## What the artifact actually is

`attention_engine/` (importable as `attn_engine`/`core`/`autotuner` once
`source/attention_engine/` is on `sys.path`) is a Python DSL + code-generation
framework: an attention variant (score function, mask predicate, online-
softmax recurrence) is described with a small set of framework classes
(`AttentionEngine`, `OnlineFunc`, `CustomIO`, `SymbolScalar`, `Var`,
`meta_tensor`), and `AttentionEngine(...)` lowers it — via **TileLang** as
the runtime backend (confirmed: `pyproject.toml`'s
`description = "MetaAttention with TileLang as the default runtime
backend"`) — into a callable `mod(q, k, v) -> out` GPU kernel, JIT-compiled
per exact `(B,H,S,D,DV)` shape.

The repo's own README states hardware requirements of "1x NVIDIA Hopper GPU,
or 1x AMD MI200 Series GPU" for **reproducing the paper's own figures** — but
the framework's arch registry (`attention_engine/autotuner/arch/A100.py`)
already has a first-class A100 entry (`compute_capability = "80"`), and this
integration **empirically confirmed** (not merely inferred from the arch
registry) that the framework compiles and runs correctly on our A100 —
`get_attn_device()` resolves to `A100` at runtime and `tune=False` uses a
built-in default schedule with no dependency on a pre-tuned
`tuned_config/A100/*.json` file (which doesn't exist in this repo — only
`H100/` and `MI250/` tuned-config dirs are present).

## Wrapping boundary (ARTIFACT_GUIDE rule 1)

`source/examples/mha.py::causal_softmax_attention` is the artifact's own
demonstration of exactly this call pattern, but only wires the CAUSAL case
with a fixed `H` (no separate `Hkv`) — there is no ready-made bidirectional
or GQA factory to import directly for this track's other required shapes.
`adapter.py`'s `_build_softmax_attention()` is a small, NEW, adapter-local
helper (analogous to the RASSM/FlashAttention-T wrapper precedent in this
repo) that calls the artifact's SAME public API
(`AttentionEngine`/`OnlineFunc`/`CustomIO`/`SymbolScalar`/`Var`/
`meta_tensor`) with the mask predicate parameterized — every field name and
the entire online-softmax recurrence (`online_fwd`/`online_fwd_epilogue`/
`forward`/`backward`) is copied verbatim from `examples/mha.py` /
`examples/mha_v2.py`; nothing here is invented attention math, only the mask
closure is a parameter instead of hardcoded. **No artifact source file was
edited.**

`prepare()` (1) COMPILES the kernel for the workload's exact shape by calling
`AttentionEngine(...)` — MetaAttention's kernels are shape-specialized
(B/H/S/D are compile-time constants baked into the TileLang lowering), so
this genuinely is the artifact's own "compile for this shape" preprocessing
step, timed as such (confirmed empirically: 2–12 seconds per shape,
including unused backward-pass kernels the framework compiles alongside
forward — `tune_bwd=False` does not skip generating them); and (2) permutes
operands from this harness's `(B,H,S,d)` layout to the artifact's own
`(B,S,H,d)` calling convention (confirmed from
`source/testing/test.py::test_softmaxattention`'s
`query = torch.randn(B, S, H, D, ...)` / `o = attention_module(q,k,v)` call
pattern) — a required layout adaptation, not a math change.

## Dependency install — isolated, not the shared venv

`build.sh` installs `tilelang einops jinja2 ninja sympy termcolor` via
`pip install --target=pylibs/ ...` into
`bench/artifacts/attention-kernel/metaattention/pylibs/`, **not** into the
shared `/pscratch/sd/c/cunyang/gnn/plexus_env` venv. Reason: that venv
already has `numpy==2.3.3` installed and is used by many unrelated projects
on this machine; MetaAttention's own `pyproject.toml` pins `numpy<2`, and
`tilelang`'s dependency tree (`apache-tvm-ffi`, `z3-solver`, `cloudpickle`,
`ml_dtypes`, ...) is non-trivial — a bare `pip install` risked silently
downgrading numpy and breaking every other project sharing that venv.
`adapter.py` `sys.path.insert`s `pylibs/` (and `source/attention_engine/`,
`source/`) at import time instead. The `3rd_parties/cutlass*` git submodules
were **not** initialized — grep-confirmed (`grep -rl 3rd_parties
attention_engine/attn_engine attention_engine/core`) that only the legacy,
unused `core/template/cute_template*/` Hopper-specific paths reference them;
the TileLang lowering path `examples/mha.py` uses does not need them.

## Shape coverage — and an honestly-reported failure, not a hidden one

Gated against `kernelbench.domains.ml.smoke_workloads(kernel=
"attention-kernel")` (4 tiny `d=16` synthetic shapes) directly via
`kernelbench.harness.run_variant`, since `--smoke`'s lack of per-shape
try/except (see Gate verification below) means all 4 must be tried:

| smoke shape | shape | result |
|---|---|---|
| `smoke-attn-prefill-causal` | B=2,H=4,Hkv=4,d=16,Sq=Sk=24, causal | **PASS**, `max_scaled_err = 5.16e-4 <= 1e-2` |
| `smoke-attn-prefill-bidir` | B=2,H=4,Hkv=4,d=16,Sq=Sk=20, bidirectional | raises `NotImplementedError` (see below) |
| `smoke-attn-prefill-gqa` | B=2,H=8,Hkv=2,d=16,Sq=Sk=24, causal | raises `NotImplementedError` (GQA not wired) |
| `smoke-attn-decode-gqa` | B=3,H=8,Hkv=2,d=16,Sq=1,Sk=48, causal | raises `NotImplementedError` (GQA/decode not wired) |

The causal-prefill PASS is notable: unlike PAT and FlashAttention-T (hand-
tuned CUDA templates hard-restricted to `head_dim in {64,128}`), this
TileLang-codegen artifact was **verified to compile and run correctly at
`d=16`** — not merely assumed to, since the framework's own test suite never
exercises `D<64`. This was actually tried and passed against the harness's
real fp64 reference, not just "ran without crashing."

**Bidirectional was attempted and found to produce wrong numbers, not just
skipped for convenience.** An always-true `mask_mod` predicate
(`lambda b,h,q_idx,kv_idx: q_idx >= 0`) was tried with both
`infer_mask=True` and `infer_mask=False`; both COMPILE and RUN without error
but produce a numerically wrong result — verified with an independent fp64
bidirectional reference computed by hand (plain `einsum`+`softmax`, not
reusing any harness code): `max_abs_err ≈ 1.277` (vs `≈ 4e-4` for the causal
path's own max-abs-err at the same shape) — i.e. essentially unrelated
output, not a small numerical discrepancy. Both `infer_mask` settings gave
the IDENTICAL wrong result, ruling out the block-mask-skip optimization as
the specific cause; the actual root cause (a bad assumption elsewhere in the
codegen for a non-causal mask, or a subtlety in how this adapter's predicate
gets symbolically traced) was not chased further within budget.
**`prepare()` raises `NotImplementedError` for `mask != "causal"` rather than
shipping a silently-wrong result** — per ARTIFACT_GUIDE rule 4 ("correctness
gate is non-negotiable... do not loosen the gate to make it pass") applied
proactively: this integration would rather under-report coverage than emit
a result that fails the gate (or worse, one whose failure a future reader
might attribute to the artifact rather than to this adapter's own mask
formulation).

GQA (`Hkv<H`): `examples/mha.py`/`mha_v2.py`'s single-`H` factory builds
Q,K,V all with the same head count. A GQA-capable factory
(`examples/sparse_gqa_decode.py`) exists in the repo but is a block-SPARSE
decode kernel (takes a `BLOCK`/sparsity-pattern argument), not a drop-in
dense-GQA replacement — adapting it was out of budget.

Decode (`Sq=1`): not wired; prefill was the priority given the time
available.

## Gate verification

The literal command
```
$PY -m kernelbench.runner --kernel attention-kernel --variant attn-prefill-kernel-fp16bf16 \
    --impl metaattention-softmax --smoke --warmup 1 --reps 3
```
was tried first (this artifact was the one candidate of the three where it
had a real chance of working, since TileLang codegen is not head-dim-
restricted like the other two). It still cannot be used as-is: `--smoke`
forces all 4 smoke shapes regardless of `--variant`, `runner.py`'s per-run
loop has no try/except, and 3 of the 4 shapes raise `NotImplementedError` by
design (see table above) — the first one reached (`smoke-attn-prefill-bidir`,
second in iteration order) crashes the whole invocation. Gated instead with
a standalone script calling `kernelbench.harness.run_variant` directly per
smoke shape:

```python
import sys
sys.path.insert(0, "/pscratch/sd/c/cunyang/msu/hpc-bench/kernel-papers/bench")
from kernelbench import domains, harness, spec
from kernelbench.artifact_registry import discover

domain = domains.load("attention-kernel")
sp = spec.load("attention-kernel")
variant = sp.variant("attn-prefill-kernel-fp16bf16")
impls = discover("attention-kernel")
factory = impls["metaattention-softmax"]["factory"]

workloads = {w.name: w for w in domain.smoke_workloads(kernel="attention-kernel")}
for name in ["smoke-attn-prefill-causal", "smoke-attn-prefill-bidir",
             "smoke-attn-prefill-gqa", "smoke-attn-decode-gqa"]:
    impl = factory("fp16")
    w = workloads[name]
    params = {"seed": 42, "precision": "fp16"}
    r = harness.run_variant(impl, w, variant, params,
                             reference=domain.REFERENCES["attention-kernel"],
                             correctness_mode=domain.CORRECTNESS_MODE["attention-kernel"],
                             warmup_override=1, reps_override=3)
    print(name, r.to_dict()["valid"], r.to_dict()["correctness"]["value"])
```

Result for `smoke-attn-prefill-causal`: **valid: true**,
`max_scaled_err = 5.161e-04 <= tol 1e-2`. `0.104-0.110 ms` median — reduced-
protocol numbers only (`warmup=1, reps=3`, shared login-node GPU), explicitly
non-conforming per ARTIFACT_GUIDE rule 5, not a timing result. The other 3
shapes raise `NotImplementedError` as designed (confirmed via direct
`impl.prepare()` calls, exception messages quoted in the coverage table's
notes above). `--list`/`discover_status` confirms `metaattention-softmax` is
discovered as `available: True` for the `attention-kernel` track.

## Not done

- No independent diff against the exact `tile-ai/tilelang` git rev
  (`61f968b6c5c00c8986b9331d599f95dfee7a60cd`) the AE's own `pyproject.toml`
  pins — the PyPI `tilelang==0.1.13` wheel was used instead (much faster to
  install, and sufficient to compile/run/pass the gate); if PyPI's release
  has drifted from that exact rev in some way that matters, it wasn't
  caught by anything short of the gate itself passing, which it did.
- Bidirectional, GQA, decode not wired — see Shape coverage; bidirectional
  specifically was tried and found INCORRECT (not merely "not attempted"),
  documented above rather than silently dropped.
- No sweep across the spec's `recommended_subset` shapes — a single gate
  check only, per the task's login-node budget.

## Reproduction on zaratan (2026-09-09)

- Machine: UMD zaratan, NVIDIA A100-SXM4-40GB (sm_80, driver 595.71.05), gpu
  partition (`bench/gpu_run.sh`), login-node build.
- Toolchain: python `$KB_PY` (conda-forge, Python 3.12.14), torch
  2.8.0+cu128 in the shared env, nvcc 12.8.93 not directly invoked (TileLang
  JIT-compiles its own generated kernels). `build.sh`'s isolated
  `pip install --target=pylibs/ tilelang einops jinja2 ninja sympy
  termcolor` (no version pin in `build.sh`, same as recorded) resolved to
  **`tilelang==0.1.14`** here (PyPI's latest at build time), one point
  release newer than the `0.1.13` recorded on Perlmutter — not
  independently diffed for behavior changes, same caveat as the untested
  git-rev delta already noted above; the gate passed regardless.
- Toolchain finding, not a bug: `adapter.py::_ensure_paths()` puts
  `pylibs/` at the **front** of `sys.path` before `available()`'s first
  `import torch`. Because `tilelang`'s own dependency resolution required a
  torch newer than the shared env's 2.8.0, `pip install --target=pylibs`
  pulled a **separate, full torch 2.14.0 (+ its own `nvidia-*-cu13` wheel
  stack)** into `pylibs/` (visible in `build.sh`'s pip output). Every
  import of `torch` inside this adapter's process (gate included) therefore
  resolves to **pylibs' torch 2.14.0**, not the shared env's 2.8.0+cu128 —
  confirmed harmless (kernel compiles and the gate passes against the fp64
  reference) but flagged here since it's a real toolchain divergence from
  every other artifact in this repo, driven by an unpinned `pip install
  --target` picking up whatever dependency graph the isolated target
  resolves to on this machine/date. No source or build-system fix applied
  (not a bug to fix, just a fact to record per rule 4).
- Build: OK. No build-system changes.
- Gate: `attn-prefill-kernel-fp16bf16`, `metaattention-softmax`, fp16, all 4
  `ml.smoke_workloads(kernel="attention-kernel")` shapes (same standalone
  script as recorded):
  `smoke-attn-prefill-causal` **PASS**, `max_scaled_err=5.161e-04 <= tol
  1e-2` (matches the recorded value to the last printed digit);
  `smoke-attn-prefill-bidir`/`smoke-attn-prefill-gqa`/`smoke-attn-decode-gqa`
  raise `NotImplementedError` as designed (same messages as recorded).
- Deviation from the recorded ruling: none in outcome; the pylibs-torch
  version divergence above is a toolchain fact worth flagging for future
  runs, not a deviation in the gate result.
- Verdict: BUILT+GATED (causal PASS, bidir/GQA/decode cleanly unsupported)
  — same as recorded ruling.
