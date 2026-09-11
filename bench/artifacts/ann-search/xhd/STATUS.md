# xhd (X-HD: Fast Hausdorff Distance Computation with Ray Tracing) — STATUS

**Outcome: SKIPPED — scope mismatch (Hausdorff distance is not a k-NN search
implementation, per this track's own spec ruling) compounded by a build
disproportionate to the marginal benefit of extracting its internal k-NN
sub-routine**

- Paper: "X-HD: Fast Hausdorff Distance Computation with Ray Tracing"
  (ICS'26). `PAPER_KEY = conf/ics/GengYLWZ26`.
- Artifact: https://github.com/pwrliang/X-HD
- Commit cloned: `7bf41c8442d059c94f4178355c6d5a10571d9658`, `git clone --depth 1`.
- No build attempted (evidence below is from reading `CMakeLists.txt`,
  `vcpkg.json`, and the relevant `src/hd_impl/*.h` headers; a "cheap skip,
  no build attempt" per this track's own scope-ruling posture for a clearly
  out-of-scope/disproportionate candidate).

## 1. Scope ruling (already made by benchspecs/ann-search/spec.yaml)

`spec.yaml`'s `notes_on_fairness` is explicit and was read again before this
decision: *"X-HD's own end-to-end deliverable (a Hausdorff distance VALUE
between two point sets) is explicitly OUT of scope for exact-spatial-knn-
kernel — only its internal RT-accelerated k-NN routine is a valid candidate
submission, since the end-to-end HD computation is a different operation (a
max-min reduction over k-NN results), not itself a k-NN search
implementation choice."* This integration therefore could only ever target
`src/hd_impl/hausdorff_distance_nearest_neighbor_search.h`'s
`HausdorffDistanceNearestNeighborSearch<COORD_T, N_DIMS>` class (or the RT
variant, `hausdorff_distance_rt.h`), never the `hd_exec` end-to-end binary
`expr/run_fig5.sh` drives.

## 2. Why extracting that sub-routine is disproportionate here

Unlike CLOVER (`bench/artifacts/ann-search/clover/`) and RTNN
(`bench/artifacts/ann-search/rtnn/`), both of which expose a genuinely
standalone free-function k-NN entry point (`bitonic_hubs::C_and_Q`,
`optixNSearch`'s search path) that a small shim can wrap directly, X-HD's
internal k-NN classes are deeply embedded in its own framework:

- `HausdorffDistanceNearestNeighborSearch` **inherits**
  `HausdorffDistance<COORD_T, N_DIMS>` (`hausdorff_distance.h`) — X-HD's own
  Hausdorff-distance class hierarchy, not a standalone k-NN interface.
- It depends directly on **RMM** (`rmm/cuda_stream_view.hpp`,
  `rmm/device_scalar.hpp` — RAPIDS Memory Manager), **glog**, and **cukd**
  (a third vendored/fetched CUDA k-d tree library, `cukd/builder.h`,
  `cukd/fcp.h`) — three more moving parts than CLOVER's header-only,
  dependency-free `include/bitonic-hubs.cuh` or RTNN's OptiX-plus-CUDA-only
  `optixNSearch`.
- `src/clover/` inside this SAME repo is literally CLOVER's own source
  (`bitonic-hubs.cuh` et al., vendored) — X-HD uses CLOVER as one of its own
  internal baselines (`hausdorff_distance_clover.h`), confirming survey.md's
  cross-reference finding; wrapping X-HD's OWN clover-baseline path would
  duplicate the `bench/artifacts/ann-search/clover/` integration already
  done directly against the upstream repo, not add a new candidate.

## 3. Build weight (why not attempted even to confirm it links)

`CMakeLists.txt` requires, in addition to a working OptiX (fetched via
`FetchContent` from `https://github.com/NVIDIA/optix-dev.git`, tag `v9.0.0`
— confirming, same as the RTNN finding, that OptiX headers themselves are
not the scarce resource here):

- **RMM**, fetched via `FetchContent` from `rapidsai/rmm` at configure time
  (a git clone of a RAPIDS C++ library, itself with its own dependency
  chain).
- **vcpkg** dependencies (`vcpkg.json`): `hdr-histogram`, `nlohmann-json`,
  `gflags`, `glog`, and **ITK** — the Insight Toolkit, a large biomedical-
  imaging library (X-HD's `run_mri_datasets` baseline target) that routinely
  takes 30-60+ minutes to build from source via vcpkg, entirely
  disproportionate to a login-node integration pass for a component
  (medical image I/O) this integration would never even exercise (only the
  point-set k-NN path is in scope, per §1).

## Conclusion

Given (1) the spec's own explicit scope ruling already excludes X-HD's
headline deliverable, (2) the one in-scope sub-routine is not a standalone,
easily-wrappable API the way CLOVER's and RTNN's are, and (3) reaching it at
all requires a build (RMM FetchContent + vcpkg/ITK) far heavier than this
track's other three candidates combined, this artifact is SKIPPED rather
than attempting a multi-hour vcpkg/ITK build for a component whose own
paper's spec-mandated evaluation scope this repo cannot use in the first
place.
