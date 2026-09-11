# quantized-gemm — artifacts

Fused (or explicit) dequantize-then-multiply GEMM for LLM linear layers
(`benchspecs/quantized-gemm/spec.yaml`). Baselines below are chosen per the
revised **baseline selection rule** (user decision, 2026-09-05; see
`bench/ARTIFACT_GUIDE.md`'s "Baseline selection rule" section and
`kernel-papers/output/baseline_selection.md`'s `quantized-gemm` section,
produced from `output/kernel_centrality.json`'s per-(paper, track) ratings):

1. **kernel centrality** `core` (the kernel IS the paper's headline
   contribution, evaluated at kernel level) > `component` (a part of a
   bigger system/pipeline/codegen output) — `tangential` is never a
   baseline;
2. **regime match** with the track spec's inputs: `matches` > `partial` >
   `mismatch`;
3. **single-NVIDIA-GPU path** required (current scope);
4. **recency** only as the tiebreak; up to 5 per track.

Applied blindly, an activity/recency-only rule would have picked a
tile-level DSL (`tilus`) and a compiler framework (`qfactory`) as this
track's "SOTA" baselines while skipping the canonical hand-written kernel
lines (MARLIN, FP6-LLM, MXBLAS) the spec's own shape suite is built from.
Under the revised rule, `tilus` and `qfactory` are **demoted to
competitors** (rated `component`: a DSL/compiler that *generates* a
quantized-GEMM kernel among other things, not a paper whose headline,
kernel-level contribution IS this kernel) — they stay in the registry and
run under the identical gate as every other implementation, but are no
longer treated as the human-SOTA reference for Phase 3.

## Artifacts

| dir | paper | centrality / regime | outcome |
|---|---|---|---|
| `marlin` | MARLIN, PPoPP'25 (`conf/ppopp/FrantarCCHA25`) | `core` / `matches` | **BUILT+GATED** (reused gemv/marlin build) — err 5.18e-05..3.65e-04 at decode/M<=64; real precision-degradation bug found in marlin's own `par>1` batched-M dispatch (M in {128,512,2048}) |
| `fp6llm` | Quant-LLM / FP6-LLM, USENIX ATC'24 (`conf/usenix/XiaZWCYYBWZZRHS24`) | `core` / `matches` | **BUILT+GATED** — err 2.68e-05..1.92e-04; `partial` regime match *within* this integration (FP6 minifloat, not the spec's integer bit-widths) required one new optional `ml.py` hook, `dequantized_W_override` (no regression on any existing impl, re-verified) |
| `mxblas` | MXBLAS, SC'25 (`conf/sc/WangXY0C25`) | `core` / `matches` | **DEFERRED-HARDWARE (needs sm_90a / Hopper, FP8; see `mxblas/REQUIRES_GPU`)** — hardcoded Hopper-only (`sm_90a`) codegen, no INT8 fallback path exists in the artifact (MX-GEMM here is FP8-only), plus an independent Python>=3.12 requirement this toolchain's 3.11.7 does not meet |
| `zipserv` (fused ZipGEMM path) | ZipServ, ASPLOS'26 (`conf/asplos/FanYPLLW0026`) | `core` / `partial` | **NOT ATTEMPTED** (optional per task brief) — see below |
| `tilus` | Tilus, ASPLOS'26 (`conf/asplos/DingHZL0Y0P26`) | `component` / `matches` | BUILT+GATED, **competitor** (demoted) — err 1.74e-04..2.68e-04 |
| `qfactory` | QFactory, USENIX ATC'25 (`conf/usenix/ZhangZSZ25`) | `component` / `matches` | BUILT+GATED, **competitor** (demoted) — err 6.55e-05..2.02e-04, 2 real artifact limitations documented |
| `quantix` | Quantix, PPoPP'26 (`conf/ppopp/ChenZY26`) | `core` / `partial` | **SKIPPED** — broken CMake build (3 missing files) + no real fp16->FP3 quantizer bound anywhere in the artifact |

Full detail, provenance, patches, and exact gate commands/numbers for each
outcome are in that artifact's own `STATUS.md`.

## ZipServ's fused decompress+GEMM path — why not attempted

ZipServ (`bench/artifacts/lossless-compression/zipserv/`) is already
BUILT for the lossless-compression track, wrapping its STANDALONE
decompress kernel (`BF16TripleBitmap_Decompress_API`) — that gate already
**fails**: a genuine, root-caused bit-exactness bug in ZipServ's own
compress-side code (`InitBF16MatrixTripleBitmap_Host` computes
`start_exp = top_exponent_values[0] - 1` into a `uint8_t`, which
silently underflows to 255 whenever the single most-frequent bf16
exponent value in the input happens to be exactly 0 — a data-dependent
trigger, documented in full in that artifact's own `STATUS.md`).

For quantized-gemm, only the artifact's SEPARATE fused decompress+GEMM
kernel (`BF16TripleBitmap_MM_API` / `BF16TripleBitmap_MM_Kernel`,
`csrc/L_API.cu`) would count — the standalone decompress kernel already
wrapped does not touch a GEMM operand at all. `BF16TripleBitmap_MM_API`
takes the SAME triple-bitmap-encoded arguments (`SignMantissa`,
`CompressedFull`, `Bitmap1/2/3`, tile offsets, `start_exp`) as the already-
wrapped decompress API, plus a dense weight-multiplicand `B`, output `C`,
and `M`/`N`/`K` — i.e. it consumes output from the exact same
`InitBF16MatrixTripleBitmap_Host` compress path that already has the
confirmed `start_exp` bug. This is genuinely separable (same "wrap the
finest boundary" reasoning the lossless-compression integration already
applied), but was **not attempted** here:

- it needs a new bridge (`bridge_zipserv_gemm.cu` or an extension of the
  existing `bridge_zipserv.cu`) and build step exposing this second API,
  distinct engineering from the already-built decompress bridge;
- ZipServ's own compression is **lossless bf16 round-trip**, not lossy
  weight quantization at all — `kernel_centrality.json`'s own
  `quantized-gemm|conf/asplos/FanYPLLW0026` entry already flags this
  ("its correctness gate should really be exact bit-reconstruction, not
  the 1e-3 relative-L1 tolerance"), meaning a fair gate here would need
  the SAME kind of new optional `ml.py` hook `fp6llm` needed
  (`dequantized_W_override`, already available and reusable) but held to
  an EXACT-match tolerance, not the parsed 1e-3;
  - whether the compress-side `start_exp` bug fires for THIS track's own
  bf16 weight distribution (drawn N(0,1), a very different exponent
  histogram from the lossless-compression track's byte-reinterpreted
  float32 smoke content that triggered it there) is an open, un-answered
  question — a new investigation, not a quick re-wrap of already-known
  behavior.

Per the task brief's own framing ("optional, only if time remains"), this
was left for a future pass rather than rushed within this integration's
budget. `zipserv`'s existing (lossless-compression-track) `STATUS.md` and
`bridge_zipserv.cu`/`csrc/L_API.cu` are read-only inputs a future
integration pass can build on directly.

## Fairness (per-track summary)

Every impl in this track is gated against the SAME shared quantized
values (`kernelbench/domains/ml.py`'s `QuantGemmWorkload.quantize()`, or —
for `fp6llm`, whose FP6 minifloat format `QuantGemmWorkload` genuinely
cannot express — the new `dequantized_W_override` params hook, which
mirrors `dense.py::reference_gemv`'s existing `dequantized_A_override`
pattern exactly). Quantization error is never part of the gate; only each
implementation's own post-quantization kernel arithmetic is. The `ml.py`
change was re-verified to be a no-op for every pre-existing caller
(`tilus`, `qfactory`, `marlin`'s own INT4 path, `numpy-dequant-gemm`) —
identical `max_scaled_err` numbers before and after, see `fp6llm/STATUS.md`
for the exact re-run commands and outputs.
