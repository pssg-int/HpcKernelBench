# Adding a domain module

Contract every `kernelbench/domains/<name>.py` must satisfy. `sparse.py` is the
worked reference — read it first, then mirror its shape.

The harness (`harness.py`) never learns anything kernel-specific: it calls
`prepare → gate → warmup → timed reps → statistic` and asks the workload registry
for the cost model. A domain module supplies the rest.

## Required module attributes

```python
KERNELS = ["gemm", "gemv"]          # kernels this module implements NOW
PLANNED = ["trsm"]                  # owned but not yet implemented (honest --list)

def smoke_workloads() -> list[Workload]:
    """Small, synthetic, runs anywhere in seconds. NOT spec-conforming."""

def load_workload(name: str) -> Workload:
    """Build/fetch a real workload by the name the spec's recommended_subset uses."""

REFERENCES = {"gemm": reference_gemm, ...}
CORRECTNESS_MODE = {"gemm": "max_scaled_err", ...}
DEFAULT_PRECISION = {"gemm": "fp32", ...}
CPU_IMPLS  = {"gemm": {"impl-name": ImplClass, ...}, ...}
def cuda_impls() -> dict: ...        # imported lazily; may raise if no CUDA
```

Plus, at import time, one `workload.register_cost(kernel, rule, unit)` per kernel.

## Workload-loading hooks (optional)

`runner.py` looks for two more optional pieces, both additive:

- `load_workload(name, *, variant=...)`: if the domain's `load_workload`
  accepts a `variant` keyword, the runner passes the variant id, so one
  workload name can be sized per variant (stencil: variant 1 16384^2 x
  T=1000, variant 2 10240^2 x T=10240).
- `expand_workloads(names) -> names`: rewrites the `--matrices` list before
  loading, e.g. stencil expands `sweep:spider-2d-scaling` into the 60
  workloads of SPIDER's Figure 11.

## variant_transform (optional hook)

```python
def variant_transform(kernel: str, variant_id: str, workload) -> Workload:
    """Called by runner.py on every workload (smoke or loaded) right after
    it is built, before the reference or any implementation sees it. Return
    the workload UNCHANGED for every (kernel, variant) this hook does not
    apply to -- this is the default, and every domain without the hook at
    all is unaffected (`getattr(domain, "variant_transform", None)`)."""
```

Use this when a *variant* (not the kernel in general) needs every
implementation gated under it -- including the reference -- to see a
transformed input, rather than each implementation transforming its own
copy independently (which would silently let implementations disagree on
what they are even computing). The alternative, teaching each impl/reference
to special-case the variant id, does not scale and risks exactly the kind of
inconsistency this hook prevents.

Worked example: `spmm-binary-adjacency-kernel`
(`benchspecs/spmm/spec.yaml`) exists because three integrated Tensor-Core
SpMM artifacts (`dtcspmm`, `flashsparse`, `generalsparse`) hardcode the
sparse operand's values to 1.0 in their released code -- they compute
`A_pattern @ B`, a real GNN-adjacency regime, but can never pass a
general weighted-SpMM gate. `kernelbench/domains/sparse.py::variant_transform`
returns a COPY of the `Matrix` (`csr.data[:] = 1`, dtype preserved, `name`
suffixed `+bin`, an extra `binary_pattern = True` attribute) for exactly
that (kernel, variant) pair, leaving every other kernel/variant combination
untouched. Because `reference_spmm` and every spmm implementation read
`workload.csr` directly (never a private cached copy), binarizing the
workload ONCE here is enough to binarize the input for the whole run --
weighted and pattern-only kernels then compete on the identical input.
`runner.py` wires it in generically, right after `mats` is built (covering
both the `--smoke` and `--matrices` branches):

```python
xf = getattr(domain, "variant_transform", None)
if xf:
    mats = [xf(args.kernel, variant.id, m) for m in mats]
```

Never mutate the workload the domain module's own cache holds (e.g.
`matrices.load_matrix`'s result, reused across `--dims` sweeps and other
variants/impls in the same run) -- always return an independent copy.

## A Workload

Any object with `.name` and `.describe() -> dict`. The describe dict lands in the
result record and must identify the input completely enough to reproduce it
(sizes, generation seed or dataset provenance, a checksum where cheap).

## A cost rule

```python
def _cost_gemm(w, params) -> tuple[int, int]:
    """returns (work_count, compulsory_bytes)"""
```

`work_count` must follow the spec's `metric` field **literally**, including its
anti-gaming provisions. Two live examples from the specs:

* `spmv-symmetric-kernel` requires the **unfolded** flop count
  `2*(2*nnz_stored - diag)`, not the stored-nnz count, so half-storage cannot
  masquerade as a 2x speedup.
* `stencil` refuses GFLOP/s entirely (FLOP conventions differ per paper) and uses
  **GCell-updates/s** — pass `unit="GCUP/s"` and count cell updates as the work.

`bytes` is the compulsory-traffic lower bound (each datum once). It is labelled
as a lower bound in the record; never present it as measured traffic.

## An implementation

```python
class MyImpl:
    name = "my-impl"; platform = "cpu"      # or "cuda"
    def __init__(self, precision: str): ...
    def prepare(self, workload, params) -> handle   # TIMED ONCE as preprocessing
    def run(self, handle) -> output                 # TIMED per iteration
    def to_host(self, out) -> np.ndarray            # for the correctness gate
    def timer(self) -> Timer                        # Timer() cpu / CudaEventTimer
    def free(self, handle) -> None
```

**Everything that can be hoisted out of `run()` must be**: allocation, operand
generation, format conversion, plan creation. If it legitimately belongs to the
per-call cost, keep it in `run()` and say so in the impl docstring. This split is
the whole point of the preprocessing discipline the specs enforce.

## A reference

```python
def reference_gemm(workload, params) -> tuple[np.ndarray, np.ndarray]:
    """returns (result_fp64, scale) — scale = magnitude of the computation"""
```

`scale` is what makes the correctness gate cancellation-robust: for a product it
is `|A| @ |B|` (the componentwise backward-error denominator). Returning a bare
array is allowed, and then the gate falls back to a global scale — weaker, so
prefer the pair. For kernels whose output is exact (integer/counting kernels),
use `CORRECTNESS_MODE = "exact"` and no tolerance.

## An implementation-selected convention (params override)

A spec sometimes allows more than one input/boundary/format CONVENTION for
the same kernel, and different implementations legitimately pick different
ones — the gate must compare each implementation against a reference that
computed the SAME convention, not silently pick one convention for
everybody and call every other implementation wrong. The pattern (already
used by `dense.py`'s `reference_gemv` via `params["quant_bits"]`, `ml.py`'s
`reference_qgemm` via `params["dequantized_W_override"]`, and
`stencil.py`'s `params["boundary"]`): an implementation's own `prepare()`
sets a key on the SAME `params` dict the harness passes, unmodified, to the
reference right afterward (`ref_out = reference(matrix, params)` in
`harness.run_variant()` — `prepare()` always runs first); the reference
reads it with `params.get("the_key", <workload's own default>)`, so every
existing caller that never sets the key is completely unaffected.
`stencil.py` itself ended up needing all three conventions the pattern
anticipates: `"periodic"` (wrap-around, the workload default), `"fixed"`
(an in-array band frozen to its t=0 value and never recomputed, AN5D's
convention), and `"zero-halo"` (a zero-valued EXTERNAL halo, every cell
recomputed every sweep reading 0 for out-of-domain neighbors, SPIDER's
convention) — see its module docstring's "Boundary convention hook" for
the full definitions. Two non-negotiables specific to this pattern: (1) the
gate must still compare the FULL output under whichever convention was
selected — never crop or patch the comparison region to manufacture a
pass; an implementation whose kernel does not actually compute the
convention it (or the domain's default) claims must still fail, honestly,
exactly as `stencil.py`'s `spider-box2d7r-sptc` adapter did while its own
zero-Dirichlet halo matched neither of the domain's first two conventions
— its gate was left failing rather than papered over with cropping, until
a genuinely matching THIRD convention (`"zero-halo"`) was added to the
domain itself to model what SPIDER actually computes (see
`artifacts/stencil/spider/STATUS.md`); the correct fix for a real
convention mismatch is to model the convention faithfully, never to
crop/patch the comparison to force a pass; (2) document the convention the
params key selects in the domain module's docstring, not just in the one
adapter that happens to use it, since the next implementation to need a
different convention should find the hook already generalized rather than
reinvent it per-adapter.

## Non-negotiables

1. Read the protocol from the spec (`spec.load(kernel).variant(vid)`) — never
   hardcode warmup/reps/tolerance in a domain module.
2. The correctness gate runs before timing. If a kernel has no meaningful
   numeric reference (e.g. a counting kernel), the gate is an exact comparison
   against an independently computed answer, not a skip.
3. Anything that makes a run non-spec-conforming (synthetic input, reduced
   protocol, shared machine) must surface — the runner already does this; do not
   work around it.
4. Comments and code in English.

## Verifying your module

```bash
PY=/pscratch/sd/c/cunyang/gnn/plexus_env/bin/python
$PY -m kernelbench.runner --kernel <k> --list          # spec parses, impls found
$PY -m kernelbench.runner --kernel <k> --variant <v> --impl <i> --smoke
$PY -m kernelbench.report "results/*.json"
```

A domain is done when every kernel in `KERNELS` runs green under `--smoke` with
its gate passing, and CUDA code (if any) compiles via `make -C csrc check`.
Do NOT run GPU timing on the login node — compile only.

## Reference independence — audit ruling (2026-08-07)

The deep audit proved that a reference sharing its compute path with the
implementation under test is vacuous (a mutated mttkrp scatter axis and a
leaked attention mask both passed such gates at err=0). Rules going forward:

1. A reference may NEVER call the same function/helper as any implementation
   it gates. Same *formula* re-typed inline is acceptable; same *code object*
   is not.
2. Library-backed CPU impls (scipy/numpy BLAS, pocketfft) are bit-identical to
   library-backed references by construction. That gate verifies nothing for
   THOSE impls — it exists for independent implementations (custom CUDA, paper
   artifacts, agent submissions), which are the benchmark's actual subjects.
   Domain modules must say this in their docstring rather than imply otherwise.
3. Where feasible, add a small-size independent cross-check in prepare()
   (NTT's O(N^2) schoolbook assertion is the pattern).
4. This rule protects the REFERENCE's computation only, not the operand it is
   handed: every IMPLEMENTATION (CPU or GPU) must regenerate its dense/random
   operands with the exact same `numpy.random.default_rng(seed)` recipe the
   reference uses, since a mismatched RNG silently feeds the kernel under
   test a different input than the one the reference checks against (found
   2026-09-06: `kernelbench/impls/gpu_cuda.py`'s built-in CUDA baselines were
   drawing operands with `torch.Generator`+`torch.rand`/`randn` instead,
   failing every gate they were ever actually run under); those built-ins now
   import each domain's own operand-generator function directly (e.g.
   `cpu_ref._dense_operand`, `ml._qkv`, `ml._make_conv_operands`,
   `dense._rng_operand`, `spectral._fft_operand`, `tensor._make_factors`) —
   sharing that OPERAND generator is exactly the "same formula" case rule 1
   allows, not the "same code object" computation-sharing it forbids.

## NERSC Filesystem Safety (REQUIRED — admin warning received 2026-08-07)

Never recursively traverse `/`, `/global`, `/global/cfs`, `/global/homes`,
`/pscratch`, `/opt`, `/usr`, `/cvmfs`, or any other shared top-level directory.
This covers: `find`, `bfs`, `fd`, `tree`, recursive `du`, `rg --files`,
recursive `grep`, recursive `ls`, globstar expansion, and recursive traversal
in Python or any other language.

Before searching, identify a bounded root inside the current workspace or a
known project or data directory. Constrain depth and filename patterns.

To locate software use `command -v`, `type -a`, `module spider`, package
metadata, or known environment prefixes. Do not search mounted filesystems for
executables or libraries. A compute allocation is not permission for an
unbounded traversal of a shared filesystem.
