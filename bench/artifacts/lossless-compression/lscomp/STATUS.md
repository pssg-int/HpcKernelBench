# lsCOMP — STATUS (lossless-compression track)

**Outcome: BUILT+GATED — bit-exact round-trip PASSES on all 3 smoke workloads**

Paper: "lsCOMP: Efficient Light Source Compression", SC 2025.
PAPER_KEY = `conf/sc/HuangDUMCLSC25`. Repo:
`https://github.com/szcompressor/lsCOMP`
(commit `d920dd9dc8ddbd2684c777d1e2ba27397212e59c`, 2025-12-03;
`git clone --depth 1` into `./source/`).

This is a SEPARATE integration from `../../lossy-compression/lscomp/`
(same paper, SKIPPED there — uint32/uint16-only API with quantBins+pooling
lossy knobs has no closed-form error-bound equivalent to that track's
abs/rel fp32 contract). The lossless-compression track has no such
mismatch: lsCOMP's own README documents a mode with BOTH lossy stages
disabled ("`-b 1 1 1 1`"/"`-p 1`" ... "makes lsCOMP operate in a lossless
mode"), and its own example demonstrates "Max abs diff: 0" under exactly
this configuration.

## Byte-layout conversion (disclosed per ARTIFACT_GUIDE rule 8)

lsCOMP's API takes `uint16_t*`/`uint32_t*` detector-count arrays, not
arbitrary byte streams. `kernelbench/domains/compression.py`'s `Field`
workload is a synthetic fp32 scientific array. This adapter reinterprets
the fp32 bit pattern as **uint16** (`float32.view(uint16)`, 2 halves per
fp32 word, a bit-preserving VIEW, never a value-rounding cast) — see
"Real bug found" below for why uint16 and not the more natural 1:1 uint32
mapping.

## Real bug found in lsCOMP's uint32 lossless kernel (root-caused, not just observed)

The first version of this adapter used the uint32 entry points with a 1:1
`float32.view(uint32)` mapping. **2 of the 3 smoke workloads
(`smoke-turbulent-3d`, `smoke-multiscale-3d` — any field containing a
negative value) failed the bit-exact gate; only the strictly-nonnegative
`smoke-smooth-3d` passed.** Isolated with a standalone reproduction
(`debug_roundtrip.py`, this directory) down to a genuine 5-bit metadata
field overflow in `source/src/lsCOMP_kernel.cu`:

- `temp_rate = 32 - __clz(max_quantized_val)` (lines 120/129) ranges over
  **0..32 inclusive** (`__clz` of a value with the top bit set returns 0,
  giving `temp_rate = 32`).
- It is packed as `fixed_rate[j] = (bin_choice << 5) | temp_rate`
  (lines 122/131) and later re-extracted via `fixed_rate[j] & 0x1f`
  (line 261 in the compressor's own second pass; the decompressor does the
  same, `lsCOMP_kernel.cu` ~line 622) — a **5-bit** field, range 0..31.
- `temp_rate == 32` (`0b100000`) survives `& 0x1f` (`0b011111`) as exactly
  **0**, aliasing to the same encoding as a legitimate "uniform block,
  nothing to store." Any 64-element block containing a value with bit 31
  set (`>= 2**31`) hits this. For a reinterpreted IEEE-754 bit pattern,
  "bit 31 set" means simply "any negative float" — so the compressor
  silently writes ZERO payload bytes for that block (its own second pass
  re-derives the same masked, aliased rate) and the decompressor reads
  back exactly nothing for it, leaving whatever the caller's output buffer
  already contained (a genuinely all-zero block reconstructs correctly by
  coincidence; anything else does not).
- Confirmed empirically (`debug_roundtrip.py`): random uint32 data bounded
  to `< 2**31 - 1` round-trips perfectly every time; introducing values
  `>= 2**31` anywhere corrupts every block that contains one (for
  full-range random data, virtually every 64-element block contains at
  least one such value, so the visible effect is near-total corruption of
  the whole array — 13824/13824 elements wrong in the repro). A
  missing-pre-zero red herring (an all-zero input also "fails" on an
  uninitialized destination buffer, since the decompressor leaves
  zero-rate blocks untouched and expects a pre-zeroed `d_decData` — the
  same convention `../../lossy-compression/cuszp` and `cuszp-v1`'s own
  official examples use) was isolated and ruled out separately by
  re-running with a zeroed buffer.

This is a genuine, reproducible defect in the released kernel's **uint32**
lossless path, for any input distribution that can produce a value
`>= 2**31` in some 64-element block. It is never triggered by lsCOMP's own
target workload (light-source photon counts: small, non-negative) — only
by this domain's synthetic fp32 fields once bit-reinterpreted as uint32.

**Workaround used (adapter-side, no source/ patch)**: the uint16 entry
points (`lsCOMP_compression/decompression_uint16_bsize64`) share the exact
same kernel/metadata structure, but a uint16 value can never need more
than 16 bits, so `temp_rate` never approaches the 32-vs-31 collision
boundary. Confirmed empirically immune (`debug_uint16.py`): full-range
random uint16, constant `0xFFFF`, and this domain's actual sign-mixed fp32
smoke fields reinterpreted as uint16 pairs all round-trip bit-exactly.
This adapter therefore calls the uint16 API on a `float32.view(uint16)`
reinterpretation (doubling the element count, halving nothing about
byte-for-byte content) — still lsCOMP's own unmodified kernel, still the
documented lossless configuration, just the dtype entry point whose bit
budget this artifact's own metadata format can actually support for
arbitrary (not just small-magnitude) data. `git -C source diff` is empty —
zero kernel-code changes, per ARTIFACT_GUIDE rule 3.

## Second, minor finding: reported compressed size is unreliable for small (single-warp) inputs

Independently of the above: even on the working uint16 path, `cmpSize`
comes back as EXACTLY `blockNum` (216 for the (24,24,24) smoke shape's
432-uint16-block-equivalent grid) for all 3 smoke workloads regardless of
their different random content — i.e. it always reports "zero payload
bytes beyond metadata," which is implausible for genuinely-random
floating-point data. Root cause (read, not exhaustively proven by
additional repro): the global running total (`d_cmpOffset[cmpOffSize-1]`,
what `glob_sync`/the returned `cmpSize` is read from) is only ever written
inside an `if(warp>0){ ... if(warp==gridDim.x-1) cmpOffset[warp+1]=...;
...}` block (`lsCOMP_kernel.cu` ~lines 201-213) — when the whole workload
fits in a single warp of a single thread-block (`blockNum <=
bsize*block_per_thread = 1024`, true for every smoke-sized shape here),
`warp` is always 0, so this block never executes and the final total stays
at its `cudaMemset`-initialized 0. **This does not affect correctness** —
decompression recomputes each block's own offset from the persisted
per-block metadata bytes via the same prefix-sum arithmetic compression
used to place them, independent of the (buggy) reported total — confirmed
by the harness's own independent bit-exact gate passing on every run. It
DOES mean `achieved_compressed_bytes`/`compression_ratio` reported by this
adapter for these smoke-sized workloads (`ratio=128.0` on all 3, an
implausibly round, content-independent number) is an artifact of this
under-reporting bug, not a real measurement — disclosed here rather than
presented as a real compression ratio. A real (multi-thousand-block)
workload from `lossless-comp-gpu-multibyte-dual-scope`'s
`recommended_subset` would exceed the single-warp threshold and should
report a real total (not verified here — login-node build+gate scope only,
per ARTIFACT_GUIDE rule 5).

## `uint3 dims`

lsCOMP treats a 3D array as `dims.x` 2D slices of `(dims.y, dims.z)`,
`dims.z` fastest (README). The Field workload is row-major C-order with
its last axis already fastest-varying; after the uint16 view, the last
axis has 2x as many elements, so `dims = (shape[0], shape[1],
2*shape[2])`.

## Build

```
./build.sh
```
`nvcc -O3 -arch=sm_80 -std=c++17 -Xcompiler -fPIC --shared -I source/include
-I source/src source/src/{lsCOMP_kernel,lsCOMP_utility,lsCOMP_timer,
lsCOMP_entry}.cu -o liblscomp.so`. `include/lsCOMP_entry.h` already wraps
its declarations in `extern "C"`, so no bridge/shim file was needed (unlike
`../../lossy-compression/cuszp-v1/`). nvcc 12.9, via `../../toolchain.sh`.
Built clean, no warnings requiring investigation. Idempotent.

## Adapter

`IMPL_NAME = "lscomp-lossless-compress"`, `PRECISIONS = ["fp32"]`,
`direction = "compress"`. Device buffers via torch, same ctypes pattern as
this benchmark's other native-CUDA kernels. `d_dec` is explicitly
zero-initialized before every decode (see the "missing-pre-zero" note
above) — this is a real calling-convention requirement of the artifact's
own API, not a workaround for the bugs above.

## Gate verification (mandated command, run as specified)

```
$PY -m kernelbench.runner --kernel lossless-compression \
    --variant lossless-comp-gpu-multibyte-dual-scope --impl lscomp-lossless-compress \
    --smoke --warmup 1 --reps 3
```

```
  running lscomp-lossless-compress smoke-smooth-3d     ... 1.003 ms  0.06 GB/s  (err 0.00e+00 <= None)
  running lscomp-lossless-compress smoke-turbulent-3d  ... 0.407 ms  0.14 GB/s  (err 0.00e+00 <= None)
  running lscomp-lossless-compress smoke-multiscale-3d ... 0.401 ms  0.14 GB/s  (err 0.00e+00 <= None)

3/3 runs valid
conforming: False (smoke: synthetic matrices, reduced protocol, overridden warmup/reps, login-node shared GPU -- expected/flagged, not a defect)
```

Bit-exact (`exact` mode, `np.array_equal`) on all 3, confirmed twice
(harness's own gate via `to_host()`, and this adapter's own `free()`
post-check after fixing a shape-mismatch bug in that check itself — see
below). Full JSON at
`bench/results/lossless-compression_lossless-comp-gpu-multibyte-dual-scope_1788656964.json`.

## Patches / adapter-internal bugs found and fixed during this integration

- (adapter-internal, fixed, not lsCOMP's fault) `free()`'s own manual
  post-hoc exactness re-check initially compared `d_ori` (shape
  `(H,W,2*D)`, numpy's natural `.view()` reshape) against `d_dec`
  (allocated flat, shape `(N,)`) — same bytes, different `ndarray` shape,
  so `np.array_equal` reported a spurious mismatch on shape alone even
  though the harness's own independent gate (which reshapes before
  comparing) already proved the round trip bit-exact. Fixed by
  `.reshape(-1)`-ing both sides before comparison; `source/` untouched,
  this was purely an adapter bug.
- No patches to `source/` (`git -C source diff` empty).

## Not done

- Real (non-smoke) `recommended_subset` workloads not run (login-node
  build+gate scope only, per ARTIFACT_GUIDE rule 5) — the single-warp
  size-reporting bug above should not apply at that scale, but this was
  not verified.
- uint32 API path abandoned (see bug above); fp64 not applicable (lsCOMP
  has no fp64/double entry points at all, only uint16/uint32).

## Verdict

`lscomp-lossless-compress: BUILT+GATED bit-exact on 3/3 smoke workloads (uint16 path; uint32 path has a real 5-bit metadata overflow bug for values >= 2^31, documented above, worked around by using the uint16 entry points instead)`

## Reproduction on zaratan (2026-09-09)

- Machine: UMD zaratan, A100 MIG 1g.5gb slice (sm_80, driver 595.71.05), login-node build.
- Toolchain: nvcc 12.8.93 (conda-forge CUDA 12.8), g++ 13.4.0 (conda-forge), torch 2.8.0+cu128,
  Python 3.12.14; `-arch=sm_80` unchanged from the recorded build.
- Build: OK. Build-system changes: none (idempotent rebuild, no diff to `source/`).
- Gate: `lossless-comp-gpu-multibyte-dual-scope` lscomp-lossless-compress: PASS on all 3 smoke
  workloads (smooth/turbulent/multiscale), `err 0.00e+00 <= None` (exact metric), 3/3 runs valid.
  0.11-0.12 GB/s.
- Deviation from the recorded ruling: none.
- Verdict here: BUILT+GATED — equals the recorded ruling.
