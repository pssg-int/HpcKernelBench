# OptHuffDec — STATUS (lossless-compression track)

**Outcome: BUILT+GATED — bit-exact round-trip PASSES on all 3 smoke workloads**

Paper: "Optimizing Huffman Decoding for Error-Bounded Lossy Compression on
GPUs", IPDPS 2022. PAPER_KEY = `conf/ipps/RiveraDTYTC22`. Repo:
`https://github.com/codyjrivera/ipdps22-opthuffdec`
(commit `17405cb0c909e2f31f09d2220988ab8b9ec7e944`, 2022-03-01;
`git clone --depth 1` into `./source/`).

## Which of the 4 decoders

The repo ships 4 sibling decoders (its own README): `orig-self-sync`
(Weissenberger & Schmidt, ICPP'18, unmodified), `orig-gap-array`
(Yamamoto et al., ICPP'20, unmodified), `opt-self-sync` (ICPP'18 +
this paper's optimizations), `opt-gap-array` (ICPP'20 + this paper's
optimizations, built on the ICPP'18 codebase). **This integration wraps
`opt-gap-array`** — one of the paper's two headline optimized decoders.
Confirmed a genuinely distinct kernel from `opt-self-sync` by direct
inspection (different `cuhd_gpu_decoder.cu`: 926 vs. 1042 lines; `grep gap`
hits only in `opt-gap-array/{include,src}`).

## Why an encoder was easy: the artifact ships its own

Per the task description's fallback path ("if it ships no encoder usable on
our inputs... use a documented, independent Huffman encoder written in the
adapter") — **not needed here**. `opt-gap-array/encoder/` contains a
complete, unmodified, LGPL-3.0 reference encoder
(`llhuff::LLHuffmanEncoder`, length-limited Huffman, `MAX_CODEWORD_LENGTH =
14`) that produces exactly the codeword-stream + gap-array format the
paired GPU decoder consumes — the artifact's own README even calls this
"a basic, sequential encoder" included specifically for this purpose. Used
via `get_symbol_lengths` → `get_encoder_table` → `get_decoder_table` →
`encode_memory`, the identical 4-call sequence `source/opt-gap-array/src/
demo.cc`'s own `main()` uses.

## Byte-layout conversion (disclosed per ARTIFACT_GUIDE rule 8)

The artifact's own scope is a Huffman decoder for cuSZ-style
QUANTIZATION-CODE streams: `SYMBOL_TYPE = uint16_t`
(`source/opt-gap-array/include/cuhd_constants.h`), exactly the dtype cuSZ's
own quant codes use — this benchmark's `lossless-comp-gpu-multibyte-
dual-scope` variant's own `recommended_subset` names cuSZ quant-code
workloads explicitly ("HACC/vx quant-code ... uint16 -- used verbatim by
both GPULZ and the Huffman-decoder paper"), so uint16 symbols are this
artifact's own native regime, not a repurposing. The lossless-compression
domain's `Field` workload is a synthetic fp32 array, not a pre-existing
quant-code stream, so this adapter reinterprets the fp32 bit pattern as
uint16 (`float32.view(uint16)`, a bit-preserving VIEW, never a value-
rounding cast) before encoding. This is a different (denser-entropy, not
biased around one value the way real quant codes are) symbol distribution
than the paper's own target data, so the measured `compression_ratio`
(~1.15x on the smoke fields — plausible for high-entropy bit-reinterpreted
floats) is not comparable to the paper's own reported ratios; the
round-trip correctness gate and the measured DECODE kernel throughput
(the paper's actual optimization target) both remain legitimate.

## Bridge (why one was needed, no kernel code touched)

The artifact ships no library entry point or shared-object build target —
only `demo.cc`, an end-to-end CLI driver that loads a file, encodes,
allocates GPU buffers, times a `NROUNDS`-loop decode, and prints results
(exactly the "paper's own benchmark script" ARTIFACT_GUIDE rule 1 says to
wrap AROUND, not measure whole). `bridge_opthuffdec.cc` (this directory)
replaces it with a 5-function C API (`opthuffdec_prepare` / `_run` /
`_fetch_output` / `_compressed_bytes` / `_free`) that makes the EXACT SAME
sequence of calls into the artifact's own unmodified classes
(`llhuff::LLHuffmanEncoder::*`, `cuhd::CUHD{Input,Output,GPUInput,
GPUOutput,GPUCodetable,GPUDecoderMemory}Buffer`, `cuhd::CUHDGPUDecoder::
decode`) that `demo.cc`'s `main()` does — verifiable line-for-line against
that file — just split so PREPARE does everything `demo.cc` itself times as
separate ("encoding" / "GPU buffer allocation" / "GPU memcpy HtD") phases
BEFORE its own "decoding" phase, and RUN calls only `cuhd::CUHDGPUDecoder::
decode` (`opt-gap-array/src/cuhd_gpu_decoder.cu`, unmodified — the one real
kernel file, the artifact's actual contribution). `git -C source diff` is
empty.

Two of the artifact's own global variables (declared `extern` inside
`cuhd_gpu_decoder.cu` but only ever DEFINED in `demo.cc`, which this
integration does not compile) had to be provided somewhere for the link to
resolve: `p1Time`/`p2Time`/`p3Time`/`p4Time`/`tuneTime` (internal per-phase
profiling scratch, 1024-slot fixed arrays with no wraparound -- a
pre-existing limitation: more than 1024 total `decode()` calls in one
process will write past them, inherited unchanged from the original
`demo.cc`) and `shared_size` (demo.cc's optional performance-tuning CLI
flag; kept at demo.cc's own default, `-1` = auto). Defined in
`bridge_opthuffdec.cc`, not `source/`.

## Build

```
./build.sh
```
Compiles the artifact's own host `.cc` files (`g++`, matching its Makefile)
plus its one real kernel file (`cuhd_gpu_decoder.cu`, `nvcc -arch=sm_80`,
narrowed from upstream's default `sm_70`/V100 target — a build-flag change,
not a kernel-code change) plus `bridge_opthuffdec.cc`, linked into
`libopthuffdec.so`.

**Two build-flag-only fixes** (ARTIFACT_GUIDE rule 3: build-system fixes
fine, kernel-code changes not — neither touches `source/`):
- `--pre-include cstdint` (nvcc) / `-include cstdint` (g++): the artifact's
  own `cuhd_constants.h` uses `std::uint16_t`/`std::uint32_t` without
  including `<cstdint>` itself — worked on whatever GCC/libstdc++ this was
  developed against (some other transitively-included header must have
  pulled it in there), fails outright on this machine's GCC 14
  (`'uint16_t' in namespace 'std' does not name a type`).
- `-arch=sm_80` in place of upstream's `sm_70` default (this machine's A100,
  not the paper's V100).

nvcc 12.9, g++ 14 (`/opt/cray/pe/gcc-native/14/bin/g++`), via
`../../toolchain.sh`. Built clean after the above two fixes. Idempotent.

## Gate verification (mandated command, run as specified)

```
$PY -m kernelbench.runner --kernel lossless-compression \
    --variant lossless-comp-gpu-multibyte-dual-scope --impl opthuffdec-gap-decompress \
    --smoke --warmup 1 --reps 3
```

```
  running opthuffdec-gap-decompress smoke-smooth-3d     ... 2.776 ms  0.02 GB/s  (err 0.00e+00 <= None)
  running opthuffdec-gap-decompress smoke-turbulent-3d  ... 2.616 ms  0.02 GB/s  (err 0.00e+00 <= None)
  running opthuffdec-gap-decompress smoke-multiscale-3d ... 2.817 ms  0.02 GB/s  (err 0.00e+00 <= None)

3/3 runs valid
conforming: False (smoke: synthetic matrices, reduced protocol, overridden warmup/reps, login-node shared GPU -- expected/flagged, not a defect)
```

Bit-exact (`exact` mode, `np.array_equal`) on all 3. Compression ratio
~1.15x (`smoke-smooth-3d`: 1.1744; see the byte-layout note above for why
this is not comparable to the paper's own reported ratios). Full JSON at
`bench/results/lossless-compression_lossless-comp-gpu-multibyte-dual-scope_1788657557.json`.

## Adapter

`IMPL_NAME = "opthuffdec-gap-decompress"`, `PRECISIONS = ["fp32"]`,
`direction = "decompress"` (the artifact's headline contribution IS the
decoder; the reference encode happens in `prepare()`, untimed). ctypes
against `libopthuffdec.so`, same pattern as this benchmark's other
native-CUDA kernels.

## Not done

- `opt-self-sync` (the paper's other optimized decoder) not wrapped as a
  second competing implementation — `opt-gap-array` alone satisfies this
  integration's scope; could be added the same way if wanted later.
- Real (non-smoke) `recommended_subset` cuSZ quant-code workloads not run
  (login-node build+gate scope only, per ARTIFACT_GUIDE rule 5) — this
  artifact's OWN target regime, so a real run there would likely show a
  much higher, paper-comparable compression ratio than the smoke fields'
  ~1.15x.
- fp64 not applicable (the artifact's `SYMBOL_TYPE` is a fixed compile-time
  `uint16_t`; no fp64 path exists to wrap).

## Verdict

`opthuffdec-gap-decompress: BUILT+GATED bit-exact on 3/3 smoke workloads (opt-gap-array decoder; reference encoder is the artifact's own llhuff::LLHuffmanEncoder)`

## Reproduction on zaratan (2026-09-09)

- Machine: UMD zaratan, A100 MIG 1g.5gb slice (sm_80, driver 595.71.05), login-node build.
- Toolchain: nvcc 12.8.93 (conda-forge CUDA 12.8), g++ 13.4.0 (conda-forge), torch 2.8.0+cu128,
  Python 3.12.14; `-arch=sm_80` unchanged from the recorded build.
- Build: OK. Build-system changes: none (idempotent rebuild, no diff to `source/`; the two build-flag
  fixes recorded above, `-include cstdint` and `-arch=sm_80`, are already baked into `build.sh`).
- Gate: `lossless-comp-gpu-multibyte-dual-scope` opthuffdec-gap-decompress: PASS on all 3 smoke
  workloads (smooth/turbulent/multiscale), `err 0.00e+00 <= None` (exact metric), 3/3 runs valid.
  0.22-0.24 GB/s.
- Deviation from the recorded ruling: none.
- Verdict here: BUILT+GATED — equals the recorded ruling.
