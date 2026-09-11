# lsCOMP — STATUS

**Outcome: SKIPPED — API is integer-only (uint32/uint16), no abs/rel error-bound axis**

Paper: "lsCOMP: Efficient Light Source Compression", SC 2025.
PAPER_KEY = `conf/sc/HuangDUMCLSC25`. Repo:
`https://github.com/szcompressor/lsCOMP`
(commit `d920dd9dc8ddbd2684c777d1e2ba27397212e59c`, 2025-12-03;
`git clone --depth 50` into `./source/`).

## Why SKIPPED (evidence)

Inspected `include/lsCOMP_entry.h`, lsCOMP's own generic device-pointer C
API (the same style of clean library entry point cuSZp exposes):

```c
void lsCOMP_compression_uint32_bsize64(uint32_t* d_oriData, unsigned char* d_cmpBytes,
    size_t* cmpSize, uint3 dims, uint4 quantBins, float poolingSH, cudaStream_t stream=0);
void lsCOMP_decompression_uint32_bsize64(uint32_t* d_decData, unsigned char* d_cmpBytes,
    size_t cmpSize, uint3 dims, uint4 quantBins, float poolingSH, cudaStream_t stream=0);
// + matching uint16_t overloads
```

Every entry point takes `uint32_t*`/`uint16_t*` (unsigned-integer light-
source detector data), never `float*`/`double*` — there is no fp32/fp64
overload anywhere in `include/` or `src/`. Its lossy control surface is
`quantBins` (`uint4`: 4 quantization levels for a hierarchical scheme) and
`poolingSH` (a block-pooling threshold), not an error-bound value — README
confirms: "*Adaptive Scalar Quantization* and *Selective Pooling*... To
disable Adaptive Scalar Quantization, set `-b 1 1 1 1`; to disable
Selective Pooling, set `-p 1`" (lossless mode is the *only* mode with a
documented, exact guarantee; every lossy configuration's achieved error is
an emergent property of the bin/pooling choice, not a bound you hand in).

This benchmark's `lossy-compression` domain module
(`kernelbench/domains/compression.py`) generates `fp32` scientific
`Field` workloads and derives a single absolute error bound `eb` from the
spec's rel-sweep convention (`Field.correctness_tolerance`) — there is no
type-correct way to hand this workload to lsCOMP's API (`float*` doesn't
match `uint32_t*`/`uint16_t*`), and even bridging via a lossy int
requantization step (scale+round fp32 into uint32, invent a
quantBins/poolingSH combination, invert on decode) would not honor `eb`:
there is no closed-form mapping from an absolute fp32 error bound to
`(quantBins, poolingSH)` — the actual reconstruction error depends on the
input's local value distribution through both knobs jointly, not through
a single scalar the way SZ-family/cuSZp/PFPL's `errorBound` argument does.

This exact conclusion was already reached independently during the Phase-2
spec survey, before any artifact code was read for this integration —
`benchspecs/lossy-compression/spec.yaml`'s `notes_on_fairness` says lsCOMP
is "genuinely out of scope for the 3 variants above -- forcing them onto
the abs/rel SDRBench axis would misrepresent what they actually measure,"
and its `open_questions` says "lsCOMP's integer quantization-bin + pooling
lossy mode has no closed-form error-bound equivalent to compare against
the rest of the track's abs/rel convention; excluded from all 3 variants
rather than force-fit." Reading the actual API in this integration pass
confirms that survey-level judgment at the code level: it isn't just a
different convention for expressing the same kernel, it's a structurally
different lossy-compression kernel (integer, quantization+pooling-
controlled) than the one `lossy-comp-kernel-cpu-ebound` specifies
(float, `compress(D, eb) -> C` with `max|D-D'| <= eb`).

Matches ARTIFACT_GUIDE.md rule 7 ("If the artifact genuinely does not
implement the track's kernel ... mark SKIPPED with evidence and move to
the next candidate"). No build attempted (cheap skip — decision made from
header inspection + the pre-existing spec survey, no `cmake`/`nvcc`
invocation needed).

## Verdict

`lscomp: SKIPPED (uint32/uint16-only API, quantBins+poolingSH lossy knobs -- no abs/rel error-bound axis to match this track's fp32 compress(D,eb)->C contract; already flagged out-of-scope in the spec survey)`
