# Survey — hausdorff-distance (singleton track: 1 paper)

## conf/ics/GengYLWZ26 — X-HD: Fast Hausdorff Distance Computation with Ray Tracing (ICS 2026)

- **What is measured**: Hausdorff Distance (HD) between two point sets A, B — the standard
  worst-case similarity measure `HD(A,B) = max(h(A,B), h(B,A))` where the directed distance
  `h(A,B) = max_{a in A} min_{b in B} dist(a,b)`. X-HD accelerates the k-NN search that HD's inner
  `min_{b in B} dist(a,b)` reduces to, using GPU RT cores (same OptiX/BVH hardware-acceleration
  approach as this same research group's RayJoin, `benchspecs/spatial-join`). Three specific
  optimizations beyond naive RT-kNN: (1) a spatial GRID to reduce BVH traversal intensity, (2) HD
  ESTIMATORS to prune points that provably cannot contribute to the final max-min result (branch-
  and-bound-style early termination), (3) offloading distance computation from RT shaders to a
  dedicated CUDA kernel specifically to fix a load-imbalance problem the grid introduces.
- **workloads/inputs**: the binary's CLI (`hd_exec`) exposes `-input_type image/wkt/ply` — i.e.
  THREE structurally different domains are explicitly first-class in this artifact: **image**
  (MRI medical-imaging point sets — HD is a standard medical-image-segmentation-quality metric),
  **wkt** (Well-Known-Text geospatial point/polygon data), **ply** (Polygon-file-format 3D graphics
  point clouds/meshes). `-n_dims` is a required parameter, meaning dimensionality (2D geospatial vs.
  3D graphics/medical-volume) is explicit, not assumed. This 3-domain breadth is itself a notable
  workload-suite requirement distinct from most single-domain kernel papers in this project.
- **timing protocol/scope**: not visible beyond the bare CLI usage example in the fetched README
  (`./bin/hd_exec -input1 ... -input2 ... -n_dims ... -input_type ... -variant ... -execution ...
  -v=1`); "You may find more options in flags.cc" points to the actual flag definitions, which were
  fetched but returned only the compiled/renamed CLI usage text (no visible warmup/repeat/timer
  constants in the portion retrieved) — i.e. protocol details are NOT recoverable from the sources
  fetched in this survey and must be specified from field convention plus the paper's abstract.
- **precision & correctness**: `-variant eb/nn/itk/rt` names FOUR implementation strategies openly
  exposed as CLI choices: **eb** (likely "exact brute-force" or an exact-bound method — the
  artifact's own presumed ground-truth/reference variant), **nn** (a generic RT-accelerated k-NN
  library used naively, matching the paper's own stated baseline: "naively using an RT-accelerated
  k-NN library yields poor performance due to lack of domain-specific optimizations"), **itk**
  (Insight Toolkit, the paper's named industrial/standard baseline — a well-established medical-
  imaging library), **rt** (X-HD's own optimized algorithm, the paper's contribution). `-execution
  cpu/gpu` is a SEPARATE axis — the README states "RT only has the GPU implementation" (i.e. `rt`
  variant + `-execution cpu` is not a valid combination; `eb`/`nn`/`itk` presumably support both
  cpu and gpu).
- **metric**: the abstract reports X-HD outperforms ITK by 5.3x on average, and outperforms "a
  GPU-optimized HD solution" (likely the `nn` variant, or a distinct named prior RT-kNN-based HD
  work) by up to 6.4x.
- **baselines**: ITK (industrial-standard medical-imaging library, CPU), a GPU-optimized HD
  solution (RT-accelerated k-NN library used naively, or a distinct prior work — not disambiguated
  in the fetched abstract).
- **source**: abstract (`output/included.json`); repo README
  (`gh api repos/pwrliang/X-HD/contents/README.md`) and `flags.cc`
  (`gh api .../contents/flags.cc`, returned only compiled usage text, not the raw flag-definition
  source).

## Divergences

- Single-paper track: no cross-paper divergence. The main gap is a data gap, not a disagreement:
  the artifact's actual timing protocol (warmup/reps/statistic/timer) was not recoverable from the
  sources fetched in this survey (the README is a single terse usage block; `flags.cc`'s raw source
  was not obtained). This spec's protocol section is therefore explicitly specified from this
  project's general convention rather than lifted from the artifact, flagged in open_questions.
- The paper spans three structurally different input domains (medical imaging, geospatial, 3D
  graphics) under one CLI; this spec treats all three as REQUIRED sub-cases of one variant (rather
  than three separate variants) since the underlying HD algorithm and its correctness/timing
  protocol are domain-agnostic — only the input format and typical point-set size differ.
