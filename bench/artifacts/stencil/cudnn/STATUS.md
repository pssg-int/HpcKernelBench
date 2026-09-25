# cuDNN stencil baseline — STATUS

**Outcome (2026-09-25): WRITTEN, NOT YET BUILT OR GATED ON A GPU.**
Two implementations share one `bridge.so`:

| impl | whose baseline | algorithm |
|---|---|---|
| `cudnn-stencil` | ConvStencil (PPoPP'24) `src/cudnn/conv_*.cu`; LoRAStencil (SC'24) and SPIDER (PPoPP'26) run the same programs | `IMPLICIT_PRECOMP_GEMM`, hard-coded as in those programs |
| `cudnn-stencil-fastest` (`../cudnn_fastest/`) | FlashFFTStencil (PPoPP'25) `benchmarks/cudnn/cudnn-test.cpp` | fastest of every forward algorithm cuDNN accepts, timed once in prepare() |

This is a library BASELINE, not a paper contribution. It exists because the
four Tensor-Core stencil papers all report speedup over "cuDNN", and the
earlier stand-in, `torch-conv-stencil`, is not what they ran: it pads
circularly with a new `F.pad` tensor every step and lets torch pick the
algorithm, which on H100 ran 20-40x slower than on A100
(`bench/DIAG_H100_TORCH_CONV.md`).

## What the bridge ports, and what differs

Kept from ConvStencil's programs: one `cudnnConvolutionForward` per step on
two ping-pong device buffers, no host transfer or sync inside the loop;
batch 1, one channel, `CUDNN_CROSS_CORRELATION`, stride 1, zero padding of
`radius` (same-size output); `CUDNN_TENSOR_OP_MATH_ALLOW_CONVERSION`; fp64
data and compute.

Differs: grid, T and weights come from the harness workload instead of the
programs' hard-coded 10000^2 / all-0.1111 filter; packed Nd descriptors for
every rank (the 1D/2D programs use 4-D NHWC with C=1, the same memory layout);
run() resets buffer 0 from a device copy of the initial field once per call
(as every other stencil adapter); an fp16 mode (half data, float compute,
float alpha/beta) that the papers did not run, added so SPIDER's fp16 result
has a same-precision comparator.

Boundary: cuDNN's zero padding is a zero-valued halo outside the grid with
every cell recomputed each step, i.e. the domain's `"zero-halo"` convention;
`prepare()` sets `params["boundary"] = "zero-halo"`.

## Verified so far (macOS, no GPU)

- `bridge.cu` passes `clang++ -std=c++17 -fsyntax-only -Wall -Wextra` against
  stub cuDNN/CUDA headers (declarations only; a real nvcc build has not run).
- The descriptor layout (1D as H=1 x W=N with a 1 x k filter and padding
  (0, r); 2D NCHW; 3D NCDHW; cross-correlation with `dense_kernel(flip=False)`)
  was emulated with torch CPU `conv2d`/`conv3d` over T ping-pong steps and
  matches `reference_stencil(boundary="zero-halo")` to <= 6.3e-16 (scaled)
  on star2d1r, box2d3r, box3d1r, star3d1r, 1d2r and box2d7r.

## To do on zaratan

    bench/gpu_run.sh -t 20 -- 'bash artifacts/stencil/cudnn/build.sh'
    bench/gpu_run.sh -- '$PY -m kernelbench.runner --kernel stencil \
        --variant stencil-cpu-gpu-kernel-fp64 --impl cudnn-stencil,cudnn-stencil-fastest --smoke'

The smoke gate must pass on all 5 synthetic smoke shapes (the nonlinear-free
`smoke-j2d5pt` too: its effective weights are c/divisor). Record the outcome,
the cuDNN version and the algorithm `cudnn-stencil-fastest` picked
(`params.cudnn_algo` in the result JSON) here.
