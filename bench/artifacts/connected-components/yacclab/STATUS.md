# YACCLAB (BUF) — STATUS

**Outcome: BUILT+GATED — exact match PASSES on all 3 smoke workloads**

Paper: "Optimized Block-Based Algorithms to Label Connected Components on
GPUs", TPDS 2020. PAPER_KEY = `journals/tpds/AllegrettiBG20`. Repo:
`https://github.com/prittt/YACCLAB`
(commit `54e43543a02ff9966124c4bc41a5eda568aa8a06`, 2023-07-14;
`git clone --depth 1` into `./source/`).

## Regime: images, not graphs — and why this is a WRAP, not a SKIP

`benchspecs/connected-components/spec.yaml` scopes 3 sparse-adjacency
variants (2 BCC, 1 SCC) PLUS one genuinely different sub-problem,
`cc-image-ccl-2d3d`: binary-image 2D/3D connected-component LABELING on a
regular pixel grid, 8-connectivity fixed, with its own dataset suite,
protocol, and correctness definition. Its own `evidence` entry for this
paper says plainly: **"DOMAIN MISMATCH: 2D/3D binary-image pixel/voxel CCL
on regular grid, not sparse graphs"** — this survey-level judgment is
correct as a statement about the GENERAL "connected-components" track
regime (3 of its 4 variants are sparse-graph problems this paper's kernel
cannot and does not address), but the spec ALSO carved out a DEDICATED
variant for exactly this sub-problem, grounded in this paper's own richer
test taxonomy (its own `evidence` entry: "7 independently-toggleable test
categories (richest taxonomy in either track"). Per this integration
task's explicit branching instruction ("if it has an image-CCL variant,
wrap the GPU BKE/BUF algorithm for that variant ... If the spec is
graph-only, record SKIP") — the spec is NOT graph-only, so this is a WRAP,
not a SKIP. **No centrality-rating correction is being reported for this
paper**: `output/kernel_centrality.json`'s "regime: matches" verdict is
consistent with the track having a real, spec-defined variant this paper's
kernel matches, even though 3/4 of the track's OTHER variants would not
match it. (Contrast with the SKIP branch's hypothetical, which this
artifact did not need.)

## Which algorithm: BUF (not BKE)

The repo's `cuda/src/` contains both of this paper's contributions:
`labeling_allegretti_2019_BUF.cu` (Block-based Union-Find) and
`labeling_allegretti_2019_BKE.cu` (Block-based Komura-Equivalence) — the
spec's own `algorithm_family_disclosure` field names both explicitly. This
integration wraps **BUF** only; BKE is architecturally identical (same
kernel-launch shape, same OpenCV-type dependency) and could be added the
same way as follow-up work (see "Not done").

## Why a shim, not real OpenCV (ARTIFACT_GUIDE's explicit fallback used)

YACCLAB's CMake build requires OpenCV (`core;imgcodecs;imgproc`, plus the
CUDA-only `cudafeatures2d` contrib module the GPU algorithms themselves
`#include`). Checked directly, per ARTIFACT_GUIDE's instruction ("only if
importable/installable artifact-locally without building OpenCV from
source"):

```
$PY -m pip install --target <tmp> opencv-python-headless --only-binary=:all:
```
**succeeds** — a prebuilt wheel exists — but it is a CPU-only OpenCV build
with no CUDA support whatsoever, let alone the CUDA-enabled
`cudafeatures2d` contrib module `labeling_allegretti_2019_BUF.cu` itself
`#include`s (`opencv2/cudafeatures2d.hpp`). That module requires building
OpenCV+opencv_contrib FROM SOURCE with CUDA enabled — explicitly out of
scope. Per the guide's own fallback: **wrap the CUDA kernels directly with
a thin shim.**

`shim/{opencv2/cudafeatures2d.hpp, labeling_algorithms.h, register.h}`
(sibling of this directory) provide minimal stand-ins for exactly the
identifiers the kernel file itself references — `cv::cuda::PtrStepSz<T>`
(the small POD device-pointer+pitch+shape struct the 4 kernels take as
parameters), a `cv::cuda::GpuMat`-equivalent (create/upload/download +
implicit conversion to `PtrStepSz<T>`, exactly how real OpenCV lets a
`GpuMat` be passed where a kernel expects a `PtrStepSz<T>`), and a
`GpuLabeling2D<Connectivity2D::CONN_8>` base class providing the member
names (`d_img_`, `d_img_labels_`, `img_`, `img_labels_`, `perf_`) `BUF`'s
own code references. **None of this replaces or edits the kernel code
itself** — see each shim file's own docstring for exactly which real
OpenCV entry point it stands in for and why. `bridge_yacclab_buf.cu`
`#include`s `labeling_allegretti_2019_BUF.cu` VERBATIM (`git -C source
diff` is empty) — the 4 `__global__` kernels (InitLabeling, Merge,
Compression, FinalLabeling) and the `BUF` class orchestrating them are
100% unmodified upstream code, byte-identical to the tracked file.

## A real correctness bug this shim work surfaced (and fixed, in the shim, not the kernel)

First build+gate attempt (`debug_buf.py`, kept in this directory as
evidence) produced garbage output — CUDA errors "misaligned address"
during BOTH `cudaMalloc` (root cause) and every subsequent CUDA call in
the same process (CUDA's sticky-error semantics: once a device fault
occurs, the context is poisoned until reset, so every later call reports
the SAME cached error regardless of its own inputs — this poisoning, not a
separate bug, is why unrelated-looking later trials also failed).

Root cause: the first version of the shim's `GpuMat::create()` used a
flat, UNPADDED allocation (`step = cols * elem_size`, plain `cudaMalloc`).
`Merge` (the kernel — unmodified, `labeling_allegretti_2019_BUF.cu` line
~119) reads pairs of pixels via a 16-bit-aligned access:
`*(reinterpret_cast<int16_t*>(img.data + img_index))`. For an ODD `cols`,
a flat `step = cols` is odd, so odd-numbered rows start at an ODD byte
offset from the (well-aligned) base allocation — an actual misaligned
16-bit read, a real device fault. Real OpenCV's `GpuMat` never hits this
because it always allocates via `cudaMallocPitch`, which pads every row to
a hardware-aligned pitch specifically to make this exact access pattern
safe — this integration's shim was simply missing that padding on its
first attempt. **Fixed by switching the shim's `create()` from
`cudaMalloc` to `cudaMallocPitch`** (this directory's `shim/opencv2/
cudafeatures2d.hpp`) — confirmed via `debug_buf.py`: 5/5 random-shape
trials (including odd widths) now round-trip exactly against an
independent scipy `ndimage.label` (8-connectivity) reference. This is a
fix to OUR OWN shim (a third-party-library stand-in we wrote), not a
patch to `source/` — the kernel's own `int16_t`-aligned access pattern is
correct and was never touched; it simply assumes (as real OpenCV always
guarantees) a properly padded/pitched device buffer.

## Domain change (small, additive — per this task's explicit allowance)

`kernelbench/domains/graph.py` gained:
- `Image2D` workload dataclass + `_smoke_binary_image`/`_image_smoke_workloads`
  (3 small synthetic multi-blob binary images — blurred-random-field,
  thresholded, NOT iid noise, which would give one giant percolating
  component under 8-connectivity and be a poor smoke test)
- `_load_image_workload` (synthetic surrogates for the spec's 7 named
  YACCLAB datasets, matched by name substring — same "no real corpus
  download" posture `compression.py`'s `load_workload` already takes for
  SDRBench)
- `_canonical_labels_image` / `reference_image_ccl` (an INDEPENDENT
  8-connectivity reference: builds an explicit foreground-pixel adjacency
  matrix and reuses the EXISTING independent `_union_find_components`
  helper already used for the graph-WCC reference — a different algorithm
  family entirely from BUF's block-based approach, satisfying
  DOMAIN_GUIDE's reference-independence rule)
- `reference_cc`/`_cost_connected_components`/`ScipyCC.prepare()` each
  gained an `isinstance(w, Image2D)` branch (dispatch to the image path;
  `ScipyCC` raises `NotImplementedError` for images, since it genuinely
  only implements sparse-graph WCC/SCC)
- `smoke_workloads()` gained OPTIONAL `kernel`/`variant` parameters
  (additive per the runner's own already-established
  `smoke_workloads(kernel=...)`/`smoke_workloads(variant=...)` opt-in
  convention, used unchanged by dense.py/annsearch.py) — returns
  `Image2D`s ONLY when `variant == "cc-image-ccl-2d3d"`; every other
  kernel/variant's smoke set is byte-for-byte unchanged (verified: `bfs`
  and `connected-components --variant cc-bcc-kernel-inmemory` both still
  run against the original 2-graph smoke set, see Gate verification below)

**Disclosed simplification**: `cc-image-ccl-2d3d`'s own spec metric is
`labeling_time_ms` (a latency, not a throughput rate), but this harness's
`workload.register_cost` binds ONE throughput UNIT string per KERNEL
("GTEPS", shared with the other 3 connected-components variants) — adding
a genuinely per-variant unit would be a larger, non-additive harness change,
out of scope here. `_cost_connected_components`'s image branch is
disclosed in-code as reading "Gpixels/s", not literal edge throughput; the
real per-run `times_ms` (the actual latency data) is captured exactly
regardless of this label.

## Build

```
./build.sh
```
`nvcc -O3 -arch=sm_80 -std=c++14 -Xcompiler -fPIC --shared -I shim -I
source/cuda/src bridge_yacclab_buf.cu -o libyacclab_buf.so`. Built clean
after the shim reached the state described above. Idempotent.

## Gate verification (mandated command, run as specified)

```
$PY -m kernelbench.runner --kernel connected-components \
    --variant cc-image-ccl-2d3d --impl yacclab-buf-image-ccl \
    --smoke --warmup 1 --reps 3
```

```
  running yacclab-buf-image-ccl smoke-image-blobs-sparse ... 0.359 ms  0.01 GTEPS  (err 0.00e+00 <= None)
  running yacclab-buf-image-ccl smoke-image-blobs-medium ... 0.324 ms  0.01 GTEPS  (err 0.00e+00 <= None)
  running yacclab-buf-image-ccl smoke-image-blobs-dense  ... 0.331 ms  0.01 GTEPS  (err 0.00e+00 <= None)

3/3 runs valid
conforming: False (smoke: synthetic images, reduced protocol, overridden warmup/reps, login-node shared GPU -- expected/flagged, not a defect)
```

Exact match (`exact` mode, canonicalized-partition comparison via
`_canonical_labels_image`) against the independent union-find reference on
all 3. Full JSON at
`bench/results/connected-components_cc-image-ccl-2d3d_1788658378.json`.

Backward-compatibility check (existing behavior unchanged):
```
$PY -m kernelbench.runner --kernel connected-components \
    --variant cc-bcc-kernel-inmemory --impl scipy-cc --smoke --warmup 1 --reps 3
```
still runs the original 2-graph smoke set (`graph-smoke-uniform`,
`graph-smoke-powerlaw`), 2/2 valid, unaffected by the `Image2D` addition.

## Not done

- BKE (this paper's other contribution, Block-based Komura-Equivalence) —
  same shim would apply near-verbatim; not wrapped in this pass, scope was
  one algorithm.
- 3D/26-connectivity extension (`labeling_allegretti_2019_BUF_3D.cu` also
  exists in the repo) — the spec's `cc-image-ccl-2d3d` variant covers both
  2D and 3D; only 2D wrapped here.
- Real YACCLAB image datasets (3dpes/fingerprints/hamlet/medical/
  mirflickr/tobacco800/xdocs) not downloaded — synthetic surrogates only,
  always `synthetic: True` (login-node build+gate scope, ARTIFACT_GUIDE
  rule 5; consistent with this benchmark's existing posture for other
  large real corpora, e.g. SDRBench in compression.py).
- `memory_access_count` (the spec's secondary metric, YACCLAB's own
  dedicated `memory` test category) not wired up.

## Verdict

`yacclab-buf-image-ccl: BUILT+GATED exact match on 3/3 smoke workloads (BUF algorithm, 2D 8-connectivity; OpenCV shim, not real OpenCV; one real device-alignment bug found and fixed in the shim during integration)`

## Reproduction on zaratan (2026-09-09)

- Machine: UMD zaratan, NVIDIA A100-SXM4-40GB (sm_80, gres `a100_1g.5gb`
  request landed a full card this run), login-node build.
- Toolchain: nvcc 12.8.93 (conda-forge CUDA 12.8), g++ 13.4.0 (conda-forge),
  torch 2.8.0+cu128, Python 3.12.14, no cmake involved (single `nvcc
  -shared` call, `-std=c++14`); `-arch=sm_80`.
- Build: OK. Build-system changes: none.
- Gate: `cc-image-ccl-2d3d`/`yacclab-buf-image-ccl` (`--warmup 1 --reps 3` as
  recorded): PASS, 3/3 runs valid, err 0.00e+00 (exact) on all 3 synthetic
  smoke images (`smoke-image-blobs-sparse/medium/dense`); non-conforming
  (synthetic images, reduced protocol, high run-to-run variance on the
  shared GPU), as expected. Backward-compat re-check of
  `cc-bcc-kernel-inmemory`/`scipy-cc` not re-run this pass (unchanged code
  path, no touched files).
- Deviation from the recorded ruling: none.
- Verdict here: BUILT+GATED — equals the recorded ruling.
