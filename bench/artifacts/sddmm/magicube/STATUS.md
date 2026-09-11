# magicube (Magicube) — sddmm

**Status: BUILT** (16-bit quantized WMMA SDDMM kernel compiles and links
cleanly; correctness gate NOT attempted — see "Not done" below for why.
Not a GPU-architecture issue: `-arch=sm_80` builds and runs fine on this
A100.)

- Paper: "Magicube: Efficient Quantized Sparse Matrix Operations on Tensor
  Cores", SC'22 (Best Paper Finalist). `PAPER_KEY = conf/sc/LiOH22`.
- Artifact: https://github.com/ParCIS/Magicube, commit
  `8f92b69e9c1d7a0406eacb773ef5e79a71eda4f0` (2022-11-23), `git clone
  --depth 1` into `source/`. `source/` is byte-identical to upstream (no
  patch); everything WE wrote (`wrapper.cu`, `build.sh`) lives in this
  directory only.

## Scope per this task's instructions

"Integrate only its 16-bit SDDMM path against the track's fp16 gate if one
exists; document the quantized paths as out of the track's precision
classes." The 4-bit/8-bit paths (`wmmaSddmm_4b`/`wmmaSddmm_8b`,
`source/SDDMM/SDDMM/include/wmma_sddmm.cuh`) were NOT built or wrapped —
out of scope per this instruction, consistent with `benchspecs/sddmm/
spec.yaml`'s `sddmm-quantized-dlmc` variant treating 4b/8b/16b as three
separate, non-comparable rows.

## GPU-architecture check (ARTIFACT_GUIDE.md rule 9)

`source/SDDMM/SDDMM/Makefile:2`: `NVCC_FLAGS = -std=c++11 -arch=sm_80
-lineinfo -lcublas -lcusparse` — hardcoded `sm_80` (Ampere), matching the
paper's own stated eval GPU ("NVIDIA A100-SXM4-40GB", `source/README.md`).
Grepped `src/`+`include/` for any Hopper-only PTX (`wgmma`, `cp.async.bulk`,
`mbarrier`, `sm_90`) — zero hits. This artifact is NOT a
DEFERRED-HARDWARE case; it targets exactly this machine's A100 (sm_80) and
built successfully at `-gencode arch=compute_80,code=sm_80` (see build.sh).

## What was wrapped, and what it actually is (rule 1)

`sddmm::wmmaSddmm_16b` (`source/SDDMM/SDDMM/src/wmma_sddmm.cu` +
`include/wmma_sddmm.cuh`), compiled and linked completely unmodified into
`libmagicube_sddmm16b.so` via this directory's `wrapper.cu` (ctypes glue,
raw device pointers, no PyTorch dependency — same style as
`../../spmm/rode/wrapper.cu`).

Signature: `wmmaSddmm_16b(m_vec, k, n, row_indices, row_offsets,
col_indices, lhs_matrix, rhs_matrix, output_values, vec_length)` — ALL of
`row_indices/row_offsets/col_indices/lhs_matrix/rhs_matrix/output_values`
are **raw `int32` arrays**: `lhs_matrix`/`rhs_matrix` hold DENSE-operand
values PACKED two 16-bit sub-words per `int32` word (`32/preA = 2` for
`preA=16`, confirmed from `source/SDDMM/SDDMM/sddmm_benchmark.cpp`'s own
sizing: `lhs_matrix = new int[m*k/(32/preA)]`), and `output_values` is a
**raw 32-bit integer accumulator** (`accumulator += a_val*b_val` inside the
artifact's own host reference, `Host_sddmm_integers`, same file) — this is
a genuine fixed-point INTEGER dot-product kernel, not IEEE fp16 arithmetic
in any form, despite the paper's "16-bit" precision label referring to bit
WIDTH, not the IEEE half-precision float format.

## Build-system fix (rule 3, not a kernel patch)

The artifact's own `Makefile` pins `-std=c++11`. Under this machine's GCC 14
host compiler (`/opt/cray/pe/gcc-native/14/bin/g++`), nvcc's own
compatibility headers (`<type_traits>`, `<bits/stl_pair.h>`) use C++17
`inline constexpr` variable templates unconditionally, which GCC 14 rejects
under `-std=c++11` ("`__is_pair` is not a template", "`constexpr` is not
valid here" — reproduced verbatim compiling `wmma_sddmm.cu` unmodified).
Compiling the SAME file with `-std=c++17` instead succeeds with zero
errors, everything else (arch flags, include paths) identical — a
host-compiler/CUDA-toolchain-version guard per rule 3's own examples, no
source file touched.

## Not done: correctness gate (honest limitation, not a loosened gate)

Wiring `wmmaSddmm_16b` into an `adapter.py` needs a real, defensible
mapping from this track's `U(-1,1)` fp64 dense operands to Magicube's
packed-`int32`/16-bit-subword representation, AND a dequantization formula
turning the raw `int32` accumulator back into a comparable real value for
the correctness gate (`sddmm-quantized-dlmc`'s 16-bit tier, tolerance
`1e-3` — the semantically correct variant for this artifact per that
variant's own `evidence` entry, which cites this exact paper and this
exact `eval_matrices/s*.txt` file convention; NOT `sddmm-tensorcore-
blocked-fp16`, which is calibrated for genuine IEEE fp16 MMA rounding, a
different numeric regime from fixed-point quantization).

Investigating this exposed a real ambiguity this integration could not
responsibly resolve within budget: the artifact's OWN host-side reference
(`Host_sddmm_integers`, `sddmm_benchmark.cpp:24-56`) extracts each packed
16-bit sub-word via `a_val = ((maskA<<shift)&a_tile)>>shift` with
`maskA=0xFFFF` — i.e. it treats sub-words as **unsigned** 16-bit integers
(range 0..65535), consistent with how the SAME file's own random-data
generator fills test buffers (`bm_test_utils.h::MakeDenseMatrix<int>`:
`matrix[i] = 0x11111111 + int(117.0*rand())`, a near-constant positive bit
pattern with no notion of a real, sign-preserving quantized value at all —
this generator exists purely to produce THROUGHPUT-benchmark filler data,
not numerically meaningful quantized operands). A direct ctypes probe of
the compiled `wmmaSddmm_16b` kernel (feeding two 16-bit sub-words of
`0xFFFF` per lane, `k=32`, `vec_length=8`, one nonzero) returned
`-4194272` — neither the signed-two's-complement expectation
(`(-1)*(-1)*32 = 32`) nor the unsigned-bit-pattern expectation
(`65535*65535*32`), and only one of the `vec_length=8` output rows was
written at all, indicating the minimal test harness built for this probe
(a from-scratch aligned-CSR + packed-operand layout, not copied from any
existing adapter) still has an unresolved setup detail (likely the
`aligned_row_offsets`/`vec_length` addressing convention, which
`sddmm_benchmark.cpp` builds via a nontrivial padding/alignment scheme
not fully replicated in this quick probe). Continuing to iterate on this
diverges from a straightforward "wrap + gate" integration into
independently reverse-engineering an undocumented quantization/packing
convention with no reliable specification to check against (the artifact's
own reference is itself only a throughput-test scaffold, not a
value-preserving one) — flagged here for follow-up rather than shipped as
a guessed, possibly-wrong `adapter.py`/correctness result.

No `adapter.py` is shipped for this reason (cf. `../insum/`, similarly
undocumented as a paper artifact without one) — an adapter that cannot be
responsibly gated should not silently register as "ok" in
`--kernel sddmm --list`.

## Provenance

- Artifact commit: `8f92b69e9c1d7a0406eacb773ef5e79a71eda4f0`
- nvcc: 12.9 (`/opt/nvidia/hpc_sdk/Linux_x86_64/25.5/cuda/12.9`)
- Host compiler: g++ 14.3.0 (`/opt/cray/pe/gcc-native/14/bin/g++`)
- GPU arch: `sm_80` (A100)

## Not done

- Correctness gate (see above) — the quantization/packing convention needs
  further, dedicated reverse-engineering (or an upstream clarification)
  before a numerically meaningful `adapter.py` can be written.
- 4-bit/8-bit paths — explicitly out of scope per this task's instructions.
- `SpMM/` (Magicube also ships a quantized SpMM kernel) — out of scope for
  this sddmm-track task.
