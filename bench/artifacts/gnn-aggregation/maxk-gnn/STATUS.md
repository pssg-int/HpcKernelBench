# MaxK-GNN — gnn-aggregation — STATUS

**Outcome: SKIPPED — the differentiated kernel has no Python entry point
anywhere in the repo (not even the paper's own "trained-model pipeline"
reaches it), and would compete on an axis this spec explicitly excludes.**

Paper: "MaxK-GNN: Extremely Fast GPU Kernel Design for Accelerating Graph
Neural Networks Training" (ASPLOS'24). `PAPER_KEY = conf/asplos/PengXSHZHKKD24`
(confirmed via `output/included.json`, matched by `artifact_url`).
Repo: `https://github.com/xiexi51/MaxK-GNN`, commit
`b8536d1ac4e70d04a62d9c9171f9850940b610ac` (2024-03-04; `git clone --depth 1`
into `./source/`).

## Why SKIPPED (evidence)

**1. No Python binding exists for the paper's own kernel, anywhere in the
repo.** `kernels/spmm_maxk.cu` (forward row-wise-product SpGEMM,
`spmm_kernel_opt2_sparse_v3`) and `kernels/spmm_maxk_backward.cu` (backward
outer-product SSpMM) are plain CUDA `__global__` functions launched only
from `kernels/main.cu`, a standalone C++ **CMake benchmark binary**
(`maxk_kernel_test`) — `grep -rl "pybind11\|cpp_extension\|PYBIND11_MODULE\|load_inline" .`
over the whole repository returns **zero matches**. There is no `setup.py`,
no torch extension, no ctypes-exportable `extern "C"` symbol table — nothing
callable from Python without writing a new binding from scratch (unlike
StraGCN, which ships its own `pybind11` module, or Fused3S, same).

**2. The kernel that main.cu benchmarks is unreachable even from the
"trained-model pipeline."** The task's own guidance anticipated "if the
kernel only accepts its trained-model pipeline, SKIP with evidence" — the
actual situation here is the mirror image and stronger: `maxk_gnn_dgl.py`
(the real training entry point, `scripts_train/*.sh`) imports
`dgl.nn as dglnn` and builds its `SAGE`/`GCN`/`GIN` models
(`utils/models.py`) entirely from **standard DGL layers**. The custom
`MaxK` `torch.autograd.Function` in `utils/models.py` implements only the
**top-k nonlinearity** (masking a dense feature tensor) — it never calls
into `spmm_maxk.cu`'s kernel. The paper's differentiated SpGEMM/SSpMM
kernels are reachable from **neither** the Python training pipeline **nor**
any other Python entry point — only from the standalone C++ benchmark.

**3. Reaching the kernel requires a hardcoded, non-parameterized
preprocessing file dependency.** `SPMM_MAXK::do_test()`
(`kernels/spmm_maxk.cu`) reads warp-partition metadata via a **hardcoded
relative path**, `"../w12_nz64_warp_4/" + graph + ".warp4"` — generated
offline by `kernels/generate_meta.py` from `.indptr`/`.indices` files it
expects under a fixed `./graphs/` directory. The graphs themselves
(`kernels/README`'s "24 graphs") are distributed only via a Google Drive
link, not a scriptable URL. None of this is parameterizable from a call
site; it is baked into the C++ benchmark's own file layout.

**4. Even a hypothetical binding would compete on an axis this spec
explicitly excludes from the pooled metric.** MaxK's forward kernel computes
aggregation over a **k-sparse-per-row** feature matrix (only `k` of `F`
columns nonzero per node, via the MaxK top-k nonlinearity from the
*previous* layer) — a structurally different arithmetic from the dense-`F`
aggregation `gnn-agg-kernel-f32` measures. `benchspecs/gnn-aggregation/spec.yaml`'s
own `notes_on_fairness`, written during the Phase-2 survey before this
integration task began, already rules this out:

> "MaxK-GNN's backward SSpMM (structured top-k sparsity) and QGTC's
> sub-8-bit quantized aggregation are NOT pooled into the fp32 dense-N
> metric above -- they measure genuinely different arithmetic (fewer
> nonzeros per row by construction; integer MACs instead of FMAs) and
> reporting them on the same GFLOP/s axis as e.g. GE-SpMM would imply a
> false apples-to-apples comparison."

Given (1)-(3) — no reachable Python entry point anywhere, requiring
substantial new glue code AND a reimplementation of an undocumented
offline-metadata generator AND a Google-Drive-gated dataset just to attempt
a call — combined with (4) — even a successful binding would be gated
against a reference (dense-`F` GCN aggregation) it is not designed to match,
a mismatch this spec already flags as out of scope rather than a genuine
correctness result — the cost of building a from-scratch binding is not
justified by a result this spec's own fairness notes already say would not
belong on `gnn-agg-kernel-f32`'s leaderboard. No build attempted.

## Provenance

- Source kept at `./source/` (`git clone --depth 1`) for provenance; no
  `build.sh`/`adapter.py` (nothing wrappable without new glue code the
  reasoning above argues isn't worth writing).
