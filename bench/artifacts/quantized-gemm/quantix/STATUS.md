# quantix (Quantix) — quantized-gemm

**Status: SKIPPED (evidence below; no kernel-code patch would fix this)**

- Paper: "High-Throughput Non-uniformly Quantized 3-bit LLM Inference",
  PPoPP'26. `PAPER_KEY = conf/ppopp/ChenZY26`.
- Artifact: https://github.com/yuang-chen/Quantix-PPoPP26
- Commit cloned: `742268af54eb02ef72d9ebee182276645048bb2f`, `git clone --depth 1`.
- Toolchain checked against: `nvcc` 12.9, GPU `sm_80` (A100). No build was
  ultimately produced (see below) — nothing GPU-specific caused this.

## Why SKIPPED (two independent, sufficient reasons)

### 1. The shipped CMake build is broken out of the box (3 missing files)

`cmake -DBUILD_TESTS=ON -DBUILD_PYTHON_BINDINGS=ON <source>` fails at
configure time with three separate errors, none introduced by this
integration:

```
CMake Error at CMakeLists.txt:21 (include):
  include could not find requested file: cmake/TorchAndPython.cmake
CMake Error at quantix/CMakeLists.txt:6 (include):
  include could not find requested file:
    <repo>/cmake/cuda_flags.cmake
CMake Error at CMakeLists.txt:34 (add_subdirectory):
  add_subdirectory given source "tests/cpp" which is not an existing
  directory.
```

The repo's own `cmake/` directory does not exist at all (`ls cmake/` →
`No such file or directory`), and `tests/cpp/` (the only place a
Python-bindings example or usage test might have lived, given the CMake
target name and the survey's earlier note that this repo's README shows
essentially no content) also does not exist. This is not a flag/path
mismatch fixable by a one-line `arch flags`-class patch (ARTIFACT_GUIDE rule
3's allowed category) — it is three referenced files/directories that were
simply never committed to this repository. `CMAKE_CUDA_ARCHITECTURES` (also
hardcoded to `89`, not this machine's `80`) was patched-and-reverted during
investigation but never mattered: configure fails before that flag is ever
used to compile anything.

An alternative bypass — write a standalone `wrapper.cu` that `#include`s
`csrc/fpx_linear.cuh` directly and calls `quantix_linear_kernel` without
CMake at all (`quantix/Makefile` already does something similar for a
Python-binding-free `.so`, and this project's own `turbofno` adapter
establishes the exact precedent for this bypass style) — was evaluated and
would have solved the BUILD problem. It does not solve reason 2 below, which
is why this bypass was not pursued further.

### 2. No Python-callable (or even complete) weight-quantization path exists

`quantix/csrc/pybind.cpp` binds exactly three functions:
`linear_forward_cuda` (the GEMM), `pack_weights_cpu` (bit-**packs** an
ALREADY fpx-coded int32 tensor into the kernel's tile layout — no rounding
or scale computation happens inside it, per its own docstring: input and
output are both `[OC, IC // 32 * x]` int tensors), and `dequant_cpu`
(dequantizes a packed tensor back to fp16 given a supplied scale). **There
is no bound function that quantizes a continuous fp16/fp32 weight into FPx
codes in the first place** — `quantix_quantize_fp16_to_fp3` (declared in
`fpx_linear.cuh`) is a C++-only symbol, never passed to `PYBIND11_MODULE`,
and its only definition is
`csrc/include/quantization/cpu/quant.h::cast_fp16_to_fp3` /
`cast_fp16_to_fp3_in_fp6`, which is unfinished/placeholder code, not a real
quantizer:

```c
fp3_vals[i] = static_cast<uint8_t>(__half2float(*reinterpret_cast<half*>(&fp16_val))) & 0x7;
```

i.e. it truncates the float toward zero to an **integer**, then masks its
low 3 bits — not any recognizable floating-point rounding/minifloat-encoding
scheme (real FPx quantization needs a per-group/per-channel SCALE step
first, which is entirely absent here). The file's own header comment reads
"To be used in the future as a tool to generating the FP6 matrix from the
FP16 matrix" (i.e. explicitly marked as not-yet-built tooling), and the
function body is littered with leftover debug `printf`s and a
module-level `global_counter` — clear signs of scratch/debug code, not a
shipped, tested quantization path. It is never called anywhere else in the
CUDA GEMM pipeline (`fpx_linear.cu`) either.

Consequently: producing a valid `fpx_tensor` input for `pack_weights_cpu`
would require **reimplementing** the paper's actual quantization algorithm
from scratch (whatever it really is — the paper itself is paywalled; the
survey already flagged this exact gap: "Quantix's... artifact repo has
essentially no README content and its paper is paywalled... its exact GEMM
shapes, timing protocol, and correctness tolerance could not be recovered"),
not wrapping the artifact's own code. That crosses ARTIFACT_GUIDE rule 1
("wrap the kernel, not the paper's benchmark script" — extended here: not
reimplement a missing stage of it either) and rule 3 ("touching kernel code
is not [an allowed patch]... if the kernel itself must change to run, mark
SKIPPED").

## Secondary observation: paper claim vs. shipped kernel

The paper's title/abstract describe **non-uniform, clustering-based** 3-bit
quantization with hardware-aligned bit-shuffling. The kernel that actually
ships (`qx_gemm_3bit_kernel_reg`, dispatched via an `Exponent`/`Mantissa`
template pair summing to `BIT_WIDTH-1`) is a **uniform FPx (exponent+
mantissa minifloat) scheme**, structurally the same family as FP6-LLM/
TC-FPx (a different paper, also in this track's survey) — not a
clustering/codebook scheme at all. This may reflect an early/incomplete
snapshot of the artifact rather than the paper's final method, but it is
what this commit ships, and it further supports treating this repo as not
yet representing the paper's real contribution in runnable form.

## What was NOT done

No kernel code was patched or reimplemented (per ARTIFACT_GUIDE rule 3, and
per this task's own instruction not to invent an artifact's missing
algorithm). No adapter.py or build.sh was written, since there is nothing
for either to legitimately wrap yet.
