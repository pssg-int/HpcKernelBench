# ZipServ — STATUS

**Outcome: BUILT+GATED — the gate is correctly wired and runs cleanly to
completion, and it FAILS: a genuine bit-exactness bug in ZipServ's own
decompress kernel, root-caused below (research finding), not an adapter
defect. Per ARTIFACT_GUIDE rule 4 this is recorded as the result, the gate
is not loosened.**

Paper: "ZipServ: Fast and Memory-Efficient LLM Inference with
Hardware-Aware Lossless Compression", ASPLOS 2026. PAPER_KEY =
`conf/asplos/FanYPLLW0026` (matches `output/included.json`'s entry for
`https://github.com/xxyux/ZipServ` and
`benchspecs/lossless-compression/spec.yaml`'s own evidence key). Repo:
`https://github.com/xxyux/ZipServ`
(commit `f69cea959005222bed64fc847bc3fcdcca32a904`, 2026-03-07;
`git clone --depth 1` into `./source/`). `git status`/`git diff` inside
`source/` is empty — zero patches to the artifact's own code.

## Separability: why this is NOT the LLM-serving pipeline

The task's instruction was to wrap ZipServ's lossless (de)compression
kernel "at the finest boundary, NOT the LLM-serving pipeline," skipping
with evidence if the codec turns out inseparable from KV-cache serving
internals. It is separable, and more cleanly than expected:

- **Compress** (`kernel_benchmark/utils.h::InitBF16MatrixTripleBitmap_Host`)
  is host-only CPU code — no `<<<...>>>` kernel launch anywhere in the
  function. It is also the EXACT function ZipServ's own real production
  Python path calls: `LInfer_py/linfer_lib.cu`'s `compress_tensor_triple_bitmap`
  pybind11 binding (used by `LInfer_py/compress_model.py` at model-load
  time) calls `InitBF16MatrixTripleBitmap_Host` directly, NOT the
  differently-shaped `InitBF16MatrixTripleBitmap` declared in
  `source/build/L_API.cuh`/defined in `csrc/L_API.cu` (that one takes
  `top_exponents` as caller-supplied input rather than computing it — dead
  code as far as the current pybind wiring goes). This matches the spec's
  own framing exactly (`benchspecs/lossless-compression/spec.yaml`,
  `lossless-comp-structured-operand-fused` variant: "format: TCA-TBE
  fixed-length triple-bitmap encoding (given; one-shot compression not
  timed in this variant)").
- **Decompress** (`csrc/L_API.cu::BF16TripleBitmap_Decompress_API` ->
  `BF16TripleBitmap_Decompress_Kernel`) is a genuinely STANDALONE GPU
  kernel — it reconstructs the full `M_Global x K_Global` bf16 matrix on
  its own and takes NO `B`/`C` GEMM operands at all. It is architecturally
  distinct from the fused "ZipGEMM" kernel
  (`BF16TripleBitmap_MM_API`/`_Kernel`, ZipServ's headline
  decompress+GEMM contribution), which this adapter never calls. Nothing
  under `third_party/vllm/` or `LInfer_py/backend/` (the actual serving
  pipeline) is touched either.

Direction-fixed per the compression domain's contract
(`kernelbench/domains/compression.py`), but `"-decompress"` this time
(unlike this benchmark's `gpulz-compress`/`fzgpu-compress`/`cuszp-compress`
adapters): ZipServ is the one artifact in this track whose separable,
GPU-kernel-backed half is decompress, not compress.

## Bridge / build

`bridge_zipserv.cu` (this directory) is a thin allocation/H2D-copy wrapper
around the two functions above — no kernel code touched. `build.sh`
compiles `bridge_zipserv.cu` + `csrc/L_API.cu` directly with `nvcc`,
bypassing ZipServ's own `setup.py` (a full torch `CUDAExtension` build
needing a working PyTorch/vLLM conda env this integration doesn't set up)
in favor of the same minimal recipe `source/build/Makefile` already uses
for `libL_API.so` (retargeted from that Makefile's hardcoded
`-gencode arch=compute_89,code=sm_89`, an RTX 4090 target, to this
machine's `sm_80` A100). One build-system note, not a patch: the initial
build attempt linked `-lcublas -lcusparse` (matching
`kernel_benchmark/Makefile`'s own `LIBRARIES`) and failed with "cannot
find -lcublas" (no `-L` search path configured on this machine for those
libs); both flags were simply dropped rather than resolved, since
`InitBF16MatrixTripleBitmap_Host` — the only function this bridge calls
from `utils.h` — never calls into either library (their headers are
pulled in transitively by `utils.h` but unused by the code path this
integration exercises). Builds clean otherwise, 3 pre-existing unused-
variable warnings inside ZipServ's own `csrc/L_Kernel.cuh` (`
mask_before_pos2` x2, `laneId` x1) — not touched.

```
./build.sh
```

## Adapter

`IMPL_NAME = "zipserv-decompress"`, `PRECISIONS = ["fp32"]`.
`prepare()` calls ZipServ's own CPU compress function once (untimed
preprocessing) + H2D-copies every compressed piece + allocates the device
output buffer. `run()` times exactly one
`BF16TripleBitmap_Decompress_API` call. `to_host()` D2H-copies `run()`'s
own output directly (no extra decode needed for a `"-decompress"`-
direction adapter, unlike this benchmark's `"-compress"`-direction GPU
adapters). Dims: since the workload is a flat byte buffer, not an
already-shaped weight matrix, `bridge_zipserv.cu::zipserv_pick_dims()`
picks the smallest `(M, K)` with `M % 64 == 0`, `K % 64 == 0` (required by
`BF16TripleBitmap_Decompress_API` itself — asserted with a printed error
and `cudaErrorInvalidValue` return if violated) and `M*K*2 >= real_bytes`,
zero-padding the tail (same convention as this benchmark's other
GPU-compression bridges) — `K` fixed at the minimum tile size 64, `M`
rounded up. For the 55296-byte smoke `Field`s this picks `M=448, K=64`
(28672 bf16 elements, 2048 bytes of zero-padding). Tile hierarchy
`(tile_M, tile_M_median, tile_M_global, tile_K, tile_K_median,
tile_K_global) = (8, 16, 64, 8, 64, 64)` — ZipServ's own values, copied
verbatim from `kernel_benchmark/test_decompress.cu`'s and
`LInfer_py/linfer_lib.cu`'s own calls (not invented).

## Gate verification (mandated command, run as specified) — FAILS

```
$PY -m kernelbench.runner --kernel lossless-compression \
    --variant lossless-comp-gpu-multibyte-dual-scope --impl zipserv-decompress \
    --smoke --warmup 1 --reps 3
```
Result: **0/3 runs valid** — `correctness.metric == "exact"`,
`value == 1.0`, `passed == False` on all 3 smoke workloads (smooth/
turbulent/multiscale). No crash, no hang, no timeout — the gate runs
cleanly to completion and reports a genuine bit-exactness FAILURE (unlike
`mans-compress`, whose kernel crashes/hangs before a correctness
comparison is even possible). `achieved_compressed_bytes`/
`compression_ratio` were still recorded (ratio ~0.824 on all 3 — expansion,
same expected characteristic as this benchmark's `gpulz-compress` finding
on this high-entropy smoke content: TCA-TBE's fixed-length high-frequency
slots and 3 auxiliary bitmaps/tile-offset arrays add overhead that a
7-value-clustered-exponent scheme can't amortize against near-uniform
byte-reinterpreted float data).

## Root cause (research finding — a real bug in ZipServ's own codec, isolated by direct byte-level diffing)

Isolated by compressing/decompressing the exact smoke workload directly
against `bridge.so` (bypassing the runner) and diffing input vs. output
byte-for-byte: **184 of 55296 bytes differ** (0.33%) — a small, structured
minority, not wholesale corruption. The first mismatch: byte offset 901
(the HIGH byte of bf16 element 450, little-endian) is `0x00` in the
original, `0x80` in the decompressed output — i.e. **the SIGN BIT of that
element flipped from 0 to 1, with every other bit correct**. This is a
mechanical corruption pattern, not random noise.

Tracing the cause through ZipServ's own unmodified source (both files
confirmed via `git diff` on `source/` — empty):

1. `kernel_benchmark/utils.h::analyzeExponentDistribution_BF16` picks
   this smoke workload's 7 most-frequent bf16 exponent values as
   **`0 1 2 3 4 5 6`** (confirmed via the compress function's own stdout
   debug dump — this benchmark's high-entropy byte-reinterpreted `float32`
   smoke data genuinely has exponent-0 as one of its most common byte
   values, an entirely plausible outcome for real neural-network weight
   tensors too, since bf16 exponent 0 covers exact zero AND subnormals —
   common after pruning/quantization/padding, not just an artifact of this
   synthetic input).
2. `InitBF16MatrixTripleBitmap_Host` (same file) computes
   `start_exp = top_exponent_values[0] - 1 = 0 - 1 = -1`, assigned into a
   `uint8_t` variable — **silently underflows/wraps to 255**. No bounds
   check anywhere in the function guards against the smallest chosen
   top-exponent being 0.
3. `csrc/L_Kernel.cuh` (`BF16TripleBitmap_Decompress_Kernel`'s inlined
   helper, line 173 and its twin at ~805/826):
   `uint16_t exponent_bits_pos1 = (start_exp + code_pos1) << 7;` where
   `start_exp` is `uint8_t` (255) and `code_pos1` is `uint8_t` (1-7, the
   high-frequency slot index decoded from the 3 bitmap planes). C++
   integer promotion converts BOTH operands to `int` before the `+`, so
   `start_exp + code_pos1` is computed as a full 32-bit `255 + 1..7 =
   256..262` — **it does NOT wrap back into 0-255 the way `uint8_t`
   arithmetic would**. The subsequent `<< 7` then produces `32768..33536`
   (`0x8000..0x8300`), which fits inside the target `uint16_t` without
   truncation, so the error survives intact: bit 15 (the bf16 SIGN bit)
   ends up spuriously set on every element decoded from this exponent
   range, instead of the intended wrapped value `(0..6) << 7 =
   0x0000..0x0300` (no bit 15).

This exactly predicts the observed corruption (`0x00 -> 0x80`, a lone
sign-bit flip, nothing else) and needs no further hypothesis. It is a
genuine bug in ZipServ's own decompress kernel — its own bounds-unchecked
`uint8_t` underflow in the compress-side `start_exp` computation, combined
with its own decompress-side failure to re-truncate after C++'s standard
integer promotion — not an artifact of this integration's bridge or of
feeding it out-of-domain data. It also directly validates
`benchspecs/lossless-compression/spec.yaml`'s own stated rationale for
imposing a bit-exact tier-1 gate on this variant STRICTLY STRONGER than
ZipServ's own disclosed verification: ZipServ's own
`LInfer_py/verify_compression.py` checks ONLY downstream GEMM-output
cosine similarity (`> 0.999` vs. a bf16 cuBLAS reference) — a handful of
sign-flipped near-zero-magnitude weight elements, exactly what this bug
produces, would very plausibly still pass a whole-matrix cosine-similarity
threshold on typical LLM-scale weight tensors, meaning **ZipServ's own
public correctness verification would very likely miss this exact bug**,
just as the spec survey's `notes_on_fairness` flagged as a real gap before
any of this integration work began.

## Verdict

`zipserv-decompress: BUILT+GATED (clean run, no crash) — CORRECTNESS FAILS bit-exact roundtrip on all 3 smoke workloads (184/55296 bytes wrong, ratio~0.824). Root cause isolated to a uint8_t underflow (start_exp = top_exponent[0]-1 wrapping to 255 when the smallest high-frequency exponent is 0) combined with C++ integer promotion in the decompress kernel's (start_exp + code)<<7 exponent reconstruction, spuriously setting the sign bit -- a genuine artifact bug ZipServ's own GEMM-cosine-similarity-only verification would likely never catch.`

## Reproduction on zaratan (2026-09-09)

- Machine: UMD zaratan, A100 MIG 1g.5gb slice (sm_80, driver 595.71.05), login-node build.
- Toolchain: nvcc 12.8.93 (conda-forge CUDA 12.8), g++ 13.4.0 (conda-forge), torch 2.8.0+cu128,
  Python 3.12.14; `-arch=sm_80` unchanged from the recorded build (retargeted from the artifact's own
  Makefile default `sm_89`).
- Build: OK. Build-system changes: none (idempotent rebuild, no diff to `source/`; `-lcublas
  -lcusparse` already dropped in the committed `build.sh` as recorded above).
- Gate: `lossless-comp-gpu-multibyte-dual-scope` zipserv-decompress: FAIL on all 3 smoke workloads
  (smooth/turbulent/multiscale), `exact=1.000e+00 vs tol None`, 0/3 runs valid — same sign-bit
  corruption pattern as recorded (debug dump shows the same exponent-underflow path triggering).
  No crash, clean run to completion.
- Deviation from the recorded ruling: none.
- Verdict here: BUILT+GATED (gate fails: bit-exactness) — equals the recorded ruling.
