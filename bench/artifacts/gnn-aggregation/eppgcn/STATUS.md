# EPPGCN — gnn-aggregation — STATUS: SKIPPED (evidence below)

**Outcome: SKIPPED — no genuinely PLAIN (unfused) forward-aggregation entry
point exists, and the one degree convention traceable through the real
benchmark pipeline does not match this domain's GCN normalization even
approximately.**

Paper: "Accelerating Backward Aggregation in GCN Training With Execution
Path Preparing on GPUs" (TPDS'22). `PAPER_KEY = journals/tpds/XuSYLJ22`.
Selected as **core**, regime-**matches** under the revised kernel-centrality
rule (2026-09-05) -- `output/kernel_centrality.json`'s
`gnn-aggregation|journals/tpds/XuSYLJ22` entry: "the only surveyed paper
reporting both a with- and without-preprocessing end-to-end number side by
side alongside a kernel-only figure." Selected per the task's own wording
("wrap only if a forward/plain aggregation entry point exists, else SKIP
with evidence") -- investigated in full below; the condition is not met.
Repo: https://github.com/Catriminal/EPPGCN, commit
`af3a5161181d1ad22b87932f1dc79ba252bb011f` (2022-01-27). Cloned into
`source/` (`git clone --depth 1`); no build attempted (cheap skip once the
API-boundary evidence below was established, per ARTIFACT_GUIDE.md's scope
ruling: a genuine boundary-identity issue, not a toolchain issue).

## Why SKIPPED (evidence)

**1. Every sparse-touching pybind entry point unconditionally fuses a dense
GEMM -- there is no separate aggregation-only wrapper the way TC-GNN ships.**
`EPPGCN/GCNConv/EPPGCN.cpp`'s `PYBIND11_MODULE` exposes `forward`
(`spmm_forward` -> `spmm_forward_cuda`), `backward`
(`spmm_backward` -> `spmm_backward_cuda`), and `ours_backward`
(`ours_backward_cuda`) -- EVERY one of these, read in full, computes
`torch::mm(input, weight)` (or the backward-pass equivalent) as its literal
FIRST operation, unconditionally, with no parameter or code path to skip it:

```cpp
// EPPGCN_kernel.cu, spmm_forward_cuda -- literally the first two lines:
cudaEventRecord(start, 0);
auto tmp = torch::mm(input, weight);
```

Contrast `artifacts/gnn-aggregation/tc-gnn/`: TC-GNN's own `gnn_conv.py`
ships a SEPARATE `TCGNNFunction_SAG`/`SAG` class that calls
`TCGNN.forward(X, row_pointers, ...)` DIRECTLY, with no GEMM anywhere in
that call path -- a genuinely "plain aggregation" entry point, which is
exactly what let that adapter isolate the aggregation kernel cleanly. EPPGCN
ships no analogous wrapper (confirmed:
`grep -rln "single_kernel\|kernel_only\|forward_only\|SAG\b"` over
`EPPGCN/*.py` returns nothing). The only way to isolate aggregation from
`spmm_forward` would be passing `weight = Identity(F)` so `tmp = input` --
a workaround that still runs a real (if numerically inert) `F x F` dense
GEMM inside the SAME timed C++ call every invocation, which the task's own
"plain aggregation entry point" condition is written to rule out.

**2. The one degree-normalization convention traceable through the real
benchmark pipeline does not implement Kipf & Welling's `D^-1/2` scaling --
it implements its reciprocal.** `EPPGCN_kernel.cu`'s
`spmm_forward_cuda_kernel` computes, per edge `(srcId, nid)`:

```cpp
float degree_norm_inv = __fmaf_rn(src_norm, degrees[nid], 0);   // = degrees[srcId] * degrees[nid]
...
partial_results[...] += __fmaf_rn(degree_norm_inv, input[nid][d], 0);  // MULTIPLIES, not divides
```

i.e. each neighbor's contribution is scaled by `degrees[srcId] *
degrees[nid]` (a product, used as a multiplier). Tracing where `degrees`
itself comes from in the REAL pipeline (not a stale/dead code path):
`gcn_main.py:84` -- `degrees = dataset.degrees` -- and
`dataset.py:118-119`:

```python
degrees = (self.row_pointers[1:] - self.row_pointers[:-1]).tolist()  # raw degree, NOT inverse
self.degrees = torch.sqrt(torch.FloatTensor(list(map(func, degrees)))).cuda()  # func(x)=x if x>0 else 1
```

`self.degrees[i] = sqrt(deg[i])` -- the RAW degree's square root, with no
reciprocal anywhere in this path. Combined with finding above, each edge is
scaled by `sqrt(deg[src]) * sqrt(deg[dst])`, which AMPLIFIES high-degree
nodes' contributions -- the opposite of `D^-1/2(A+I)D^-1/2`'s normalizing
effect (this domain's one reference). This is not a simple sign/reciprocal
fix we can apply on our side without unverified assumptions about intent:
GNNAdvisor-lineage papers (which EPPGCN's forward path is inherited from,
essentially unmodified -- see finding #3) are known to sometimes ship
performance-benchmark-only feature values where the numerical CORRECTNESS
of the demo run is not the point, only the kernel's execution pattern over
a realistic sparsity structure; safely resolving which is the case here
would require deeper archaeology than this integration's remaining budget
allows.

**3. The paper's own novelty (backward aggregation) is not even exercised
by wrapping the forward path.** Per the paper's title and this repo's own
`README.md` ("the C++/CUDA source code ... for GCN sparse computation
kernel and graph compaction"), EPPGCN's actual contribution is
`ours_backward`/`build_back_part`/`compact_back_edge`/`split_back_part` --
all backward-pass-only entry points (this track's forward-only
`gnn-agg-kernel-f32` variant cannot exercise them at all, and they are
symmetric to finding #1: `ours_backward_cuda` also fuses `torch::mm` calls
for the weight gradient). `spmm_forward` (what a hypothetical wrap would
use) is functionally the GNNAdvisor BASELINE this paper compares against,
not EPPGCN's own contribution -- wrapping it would not actually benchmark
this paper's headline kernel even if findings #1-#2 were resolved.

## Not done

- No build attempted (`EPPGCN/GCNConv/setup.py`'s `GNNAdvisor` CUDAExtension
  was not compiled) -- the boundary-identity issues above are visible from
  reading the source, and ARTIFACT_GUIDE.md's scope ruling treats a genuine
  kernel-identity/boundary mismatch as a cheap skip, no build attempt
  required (same class of ruling as `artifacts/gnn-aggregation/fasten`'s
  SKIP).
- `EPPGCN/GCNConv/build_part`'s exact partition-CSR layout was read at the
  pybind-signature level (`EPPGCN.cpp`) but not exercised, since findings
  #1-#2 already rule out a faithful wrap regardless of how that
  preprocessing is reproduced.
