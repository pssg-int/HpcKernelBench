# cuke (cuKE) — tensor-contraction — STATUS

**Outcome: SKIPPED — every one of cuKE's own generated KGE score-function
kernels is either scalar-output or batched over edges; this harness's
`tensor-contraction` domain (`kernelbench.domains.tensor.ContractionWorkload`)
structurally cannot represent either shape**

Paper: "cuKE: An Efficient Code Generator for Score Function Computation in
Knowledge Graph Embedding" (IPDPS'24). `PAPER_KEY = conf/ipps/HuLJ24`
(matched by title + `artifact_url` in `../../../output/included.json`).
Artifact: https://github.com/parallel-group/cuke (commit
`8973dd8221322b0de4bc9047155a63faecef1735`, `git clone --depth 1` into
`./source/`).

## What the artifact actually is

`parallel-group/cuke` is the general, post-publication evolution of the
paper's code generator: a source-to-source ASG->IR->CUDA/C++ compiler
(`source/pycuke1/`). The paper-specific contribution lives in
`source/apps/kge/kge.py`: five named functions — `transE`, `transH`,
`transR`, `transF`, `RESCAL` — that build a cuKE ASG for each of this
track's own `kge-batched-fused-kernel-fp32` variant's 5 named KGE score
functions, then call `gpu.print_cuda(gen_ir(res))` to CODEGEN a fused CUDA
kernel for that score function (the runtime "index deduplication" inspector
optimization lives in `source/apps/kge/inspection/inspector.cu`, invoked
from `test_cuke.py`). This is a real, working code generator for a real,
paper-relevant kernel family.

## Why SKIPPED (evidence)

Per the task brief's own escape hatch ("SKIP with evidence if its generated
kernels are all KGE-score-specific and none maps to the spec's
contraction"): all 5 generated kernels were inspected in `kge.py` and every
one of them is structurally incompatible with
`kernelbench.domains.tensor.ContractionWorkload` — the harness's ONLY
tensor-contraction workload type, the one every `--variant`/`--smoke` run
actually constructs and gates against, regardless of which spec variant ID
is passed on the command line (`runner.py` always calls
`domain.smoke_workloads(kernel="tensor-contraction")` /
`domain.load_workload(name)`, both fixed to `ContractionWorkload`'s 2-operand,
non-batched, non-scalar-output shape) — for two INDEPENDENT, both fatal,
reasons:

1. **Scalar output.** `bvv(a, b) = apply(lambda x, y: einsum('i,i->', x, y),
   ...)` (a per-edge dot product) underlies `transH`, `transF`, and `RESCAL`
   (`RESCAL = bvv(bvm(vh, mr), vt)`). `ContractionWorkload.__post_init__`'s
   equation parser (`_EQ_RE = r"^...->([a-zA-Z]+)\s*$"`) requires **at least
   one** output-index letter — an einsum equation with an empty (scalar)
   right-hand side, `"i,i->"`, does not even parse. There is no way to
   construct a `ContractionWorkload` for a dot-product contraction at all.
2. **Batched over edges.** ALL FIVE score functions gather per-edge operands
   via `Eemb[h]`, `Eemb[t]`, `Remb[r]`, `Proj[r]` where `h`/`t`/`r` are
   length-`batch_size` index tensors (`h = Tensor((batch_size,),
   dtype='int', name='h')`), then combine them via cuKE's `apply()` operator
   — a per-batch-row map, confirmed against `cuke`'s own README example
   ("`cond` has the same size as `a`" for `a.apply(...)`). Every generated
   kernel's true shape therefore carries a batch index shared by both
   operands AND the output. `ContractionWorkload.__post_init__` explicitly
   rejects exactly this shape with `NotImplementedError` ("index(es) ...
   appear in both operands AND the output (a batched/shared free
   dimension) ... none of this kernel's recommended TCCG cases need one").

Both failure modes are structural properties of the domain module's own
`__post_init__` validation (`kernelbench/domains/tensor.py` lines ~341,
~401-417), not something a different choice of score function or batch size
could route around: `TransR`/`bvm` alone (`einsum('i,ij->j', ...)`, per-edge)
is shape-valid ONLY at `batch_size == 1`, which would defeat the entire
point of the `kge-batched-fused-kernel-fp32` variant (its own `claim` field:
"steady-state throughput of small, **batched**, irregular fused tensor
contractions ... including the in-kernel runtime index-deduplication cost"
— the paper's actual contribution is amortizing the dedup cost over a real
batch; a batch of 1 tests nothing cuKE-specific).

A generic, non-KGE contraction COULD in principle be built directly from
cuKE's underlying `einsum` ASG primitive (used internally by `bvv`/`bvm`
above) to match `ContractionWorkload`'s shape — but doing so would exercise
none of the artifact's actual contribution (its loop-fusion/tiling/shared-
memory/runtime-index-deduplication passes are all keyed off the KGE op
names `bvv`/`bvm`/`bsv`/`bov`/`bsm`, per `kge.py`'s own `fuse_rule`/`tiler`/
`smem` transform classes) and would not be testing "cuKE" in any sense tied
to the paper this track credits; per ARTIFACT_GUIDE.md rule 1 ("wrap the
kernel, not the paper's benchmark script") this would misrepresent what is
being measured. `source/apps/examples/examples.py` (a generic CPU-codegen
elementwise-add/indexing demo, using an apparently stale `import transform`
/ `from asg import ...` style inconsistent with the rest of the repo's
`pycuke1.*` package layout) was also inspected and confirmed to be an
unrelated compiler smoke test, not a paper-relevant kernel.

No build was attempted (`ARTIFACT_GUIDE.md`'s cheap-skip rule).

## Verdict

`cuke: SKIPPED (all 5 of cuKE's own KGE score-function kernels are either
scalar-output [rejected by ContractionWorkload's einsum-equation parser] or
batched-over-edges [rejected by ContractionWorkload's shared-batch-index
check]; the harness's tensor-contraction domain has no workload type either
shape can populate without trivializing away the artifact's actual
contribution). No build attempted.`
