# Survey — morphology-image-kernel (singleton track: 1 paper)

## conf/cgo/KoehlerS21 — Towards a Domain-Extensible Compiler: Optimizing an Image Processing
Pipeline on Mobile CPUs (CGO 2021)

- **Scope note**: this paper's benchmark subject is the **Harris corner detector**
  (grayscale conversion -> Sobel gradients -> Gaussian-weighted structure-tensor sums ->
  corner response), a stencil/convolution image-processing *pipeline*, not a mathematical-
  morphology operator (erosion/dilation/opening/closing with an explicit structuring
  element). It was classified under this project's `morphology-image-kernel` category by
  the Phase-1 pipeline as the corpus's representative "image processing kernel" paper; this
  survey does not second-guess that classification, but flags the mismatch with the parent
  task's suggested "structuring-element size" axis explicitly (see Divergences) rather than
  inventing a structuring-element parameter that does not exist in this paper.
- **What it does**: a domain-extensible compiler (Rise/Shine, and comparisons against Lift)
  that expresses image-processing pipelines via generic rewrite-rule-composed patterns
  rather than a hard-coded domain-specific compiler; demonstrates the approach by generating
  GPU/CPU OpenCL code for the Harris operator on ARM mobile CPUs and comparing against
  Halide (both a default and an auto-scheduled variant) and OpenCV.
- **workloads/inputs** (confirmed from `driver/src/benchmark.rs` + `driver/cpp/harris.cpp`):
  exactly **two images**, both bundled via the artifact's Halide/PolyMage submodule
  dependencies — a "small" image (`lib/halide/apps/images/rgb.png`, Halide's own bundled
  demo image) and a "big" image (`lib/polymage/images/venice_wikimedia.jpg`); exact pixel
  dimensions were not independently re-verified in this survey (not confirmable without
  fetching the binary image files from the submodules). Four ARM CPU targets are
  benchmarked (Cortex-A7, A15, A53, A73, via Odroid XU4/N2 boards) plus an example x86
  Intel-i7 config; each combination of {image, target} is a separate benchmark run.
- **timing protocol** (confirmed from `driver/cpp/harris.cpp` + `driver/cpp/time.hpp` +
  `driver/cpp/stats.hpp`): **30 timing iterations** per (image, implementation) pair
  (`timing_iterations` CLI arg, called with value `30` from `benchmark.rs`'s
  `target_run(bin)...arg("30")`), median/min/max reported via `time_stats()` (std::sort +
  index into the sorted sample array). Timer is `std::chrono::high_resolution_clock` (CPU
  wall time, wrapping `output.device_sync()` for Halide's async-queued execution) for
  Halide and OpenCV variants; OpenCL device-side profiling events
  (`clGetEventProfilingInfo`, `CL_PROFILING_COMMAND_START`/`_END`) for the Rise/Shine and
  Lift OpenCL-kernel variants — i.e. **two different timer mechanisms are used for
  different implementations in the same run**, a genuine methodological asymmetry (CPU
  wall-clock vs. GPU/OpenCL device-side event) confirmed directly in the source, not
  papered over here. **No explicit discarded-warmup iteration** is visible — all 30
  iterations are recorded and the median is taken over all 30 (median is somewhat
  warmup-robust by construction, but this is not the same as an explicit warmup count).
- **precision & correctness** (confirmed from `driver/cpp/stats.hpp`'s `error_stats()`
  function, called after every non-reference implementation's timed loop in `harris.cpp`):
  a **mandatory per-pixel + PSNR gate**, not merely reported — `error_stats(gold, other, n,
  tolerated_per_pixel=0.01, required_psnr=100)` computes the per-pixel absolute-difference
  range normalized by the gold image's own dynamic range, and the PSNR (10*log10(range^2 /
  MSE)); the program calls `exit(EXIT_FAILURE)` if `max_normalized_error > 0.01` OR
  `PSNR < 100`. The reference ("gold") is the Halide **default-schedule** output
  (`output1`); Rise/Shine, Lift, and OpenCV outputs are each checked against it. (Notably,
  in the exact artifact snapshot read, the Halide **auto-scheduled** variant's own timing
  loop is commented out — only its correctness check against the default-schedule gold
  remains active — so this spec does not include an "auto-scheduled Halide" timed
  implementation as confirmed-active in this artifact revision.)
- **metric**: median/min/max wall time in ms per (image resolution, implementation)
  pair, printed as `WxH <generator> <variant> <median> <min> <max>` — e.g. the paper's own
  headline claim (from the abstract) is up to 16x over OpenCV and up to 1.4x over Halide on
  the generated Harris kernel, on four ARM mobile CPUs.
- **baselines**: OpenCV (`cv::cornerHarris`, industrial reference CPU implementation),
  Halide (default schedule; auto-scheduled variant present in code but its own timing is
  disabled in this snapshot), Lift (an OpenCL-code-generating predecessor to Rise/Shine).
- **source**: `github.com/rise-lang/2021-CGO-artifact` — `README.md` (build/run
  instructions, target hardware list, OpenCV/dependency versions), `driver/src/benchmark.rs`
  (the `harris()` benchmark-orchestration function: uploads inputs to remote ARM targets over
  ssh, invokes the compiled `harris` binary with `timing_iterations=30`, parses/records
  `median min max` output), `driver/cpp/harris.cpp` (the actual C++/OpenCL benchmark driver:
  per-implementation timing loops, timer selection, `error_stats` gate calls),
  `driver/cpp/stats.hpp` (`TimeStats`/`error_stats`/`time_stats`/`ocl_time_stats`
  implementations), `driver/cpp/time.hpp` (clock-type selection).

## Divergences

- Single-paper track: no cross-paper divergence. The substantive divergence is between this
  track's ACTUAL available paper (a general image-processing-pipeline compiler benchmarked
  on the Harris corner detector, a convolution/stencil pipeline) and the parent task's
  ASSUMED axes (structuring-element size, iteration count for morphological
  erosion/dilation-style operators). **There is no structuring-element-size axis in this
  paper** — Harris uses fixed-size Sobel/Gaussian windows (window size 3, per the OpenCV
  baseline call `cv::cornerHarris(cv_gray, cv_out, 3, 3, 0.04, ...)`), not a
  variable-size structuring element. This spec's variant axes therefore substitute "image
  size" (small vs. big, the artifact's own two images) and "target CPU architecture" (the
  artifact's own four ARM cores) for the parent task's morphology-specific axes, flagged
  here rather than inventing a structuring-element parameter that this paper does not have.
- The artifact uses **two different timer mechanisms** for different implementations in the
  same benchmark run (CPU wall-clock `std::chrono` for Halide/OpenCV, OpenCL device-side
  profiling events for Rise/Shine/Lift) — a genuine fairness question (device-side event
  timers typically exclude host-side dispatch/queueing overhead that wall-clock timers
  include) this spec flags in `notes_on_fairness` rather than silently normalizing away.
- The Halide auto-scheduled variant's timing loop is commented out in the exact artifact
  snapshot read (only its correctness check remains active); this spec does not claim
  auto-scheduled-Halide timing numbers exist in this artifact revision.
