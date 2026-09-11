// bridge.cu (HPC-KernelBench, NOT part of the LoRAStencil artifact).
//
// #includes the artifact's own source/src/2d/gpu.cu VERBATIM (unmodified --
// see build.sh/STATUS.md) to get access to its extern "C++" host entry
// points `gpu_box_2d3r` / `gpu_star_2d3r` (both declared in 2d_utils.h,
// defined in gpu.cu) plus the `kernel2d_*` __global__ kernels and
// `matrix_*_d` __constant__ symbols compiled into the SAME translation
// unit. No kernel arithmetic is touched; this file only adds NEW host-side
// entry points that call the artifact's own unmodified functions.
//
// Kernel scope wrapped here, and why (see adapter.py's module docstring for
// the full reasoning; summarized):
//   - box2d3r  -> gpu_box_2d3r()  (native radius-3 box; wrapped)
//   - star2d3r -> gpu_star_2d3r() (native radius-3 star; wrapped)
//   - box2d1r  NOT wrapped: gpu_box_2d3r's own host-side "Factorize
//     parameter matrix" step (gpu.cu:290, `prop = params[(i+3)*7] /
//     params[0]`) divides by the WEIGHT MATRIX'S OWN CORNER ENTRY
//     (offset (-3,-3)). A true (AN5D) box2d1r stencil zero-pads its 3x3
//     support into this artifact's fixed 7x7 layout, so that corner entry
//     is exactly 0.0 -- `0.0/0.0 = NaN` in IEEE double, poisoning the whole
//     factorization. LoRAStencil's OWN CLI never actually exercises a true
//     zero-padded box2d1r: its hardcoded `param_box_2d1r` benchmark array
//     (source/src/2d/main.cu, lines ~148-165) is a full nonzero 7x7 "onion
//     ring" pattern (corner value 1, not 0) that happens to be labeled
//     "box2d1r" for CLI purposes only -- it is not radius-1 at the weight
//     level. Confirmed by direct arithmetic (see adapter.py), not asserted.
//   - star2d1r NOT wrapped: `gpu_star_2d1r()` (gpu.cu:484-487) ignores its
//     own `params` argument completely -- `param_u`/`param_v` are HARDCODED
//     local arrays (`{0, 1, 2, 4, 2, 1, 0}`), never read from `params`.
//     There is no way to inject the domain's own star2d1r weights into this
//     kernel at all; it always computes the same fixed synthetic pattern
//     regardless of caller input.
//   - 1D and 3D: not wired up in this integration pass (time budget), same
//     posture as the convstencil sibling artifact.
//
// PRECISION: DATA_TYPE is `double` throughout 2d_utils.h and every
// wmma::fragment in gpu.cu -- genuine fp64 Tensor-Core (DMMA) matmul, same
// situation as the convstencil/flashfftstencil siblings (both also fp64
// despite being named in spec.yaml's fp16-labeled variant-2 `suite` field).
//
// TIMESTEPS: gpu_box_2d3r/gpu_star_2d3r's own internal `times`-argument loop
// ping-pongs device buffers WITHOUT ever refreshing the periodic-wrap halo
// between sweeps (same finding as convstencil's gpu_box_2d1r) -- confirmed
// by reading the loop directly: `kernel2d_box2d3r<<<...>>>(array_d[i%2],
// array_d[(i+1)%2], cols)`, no host round-trip between launches. So this
// bridge always calls with times=1 (adapter.py's own outer T-sweep loop
// re-pads the CURRENT field with periodic wrap before every single-sweep
// call), matching the flashfftstencil/spider siblings' resolution of the
// identical class of issue.
//
// DEVICE-MEMORY LEAK (real artifact bug, found while wrapping, not
// introduced by this integration): none of gpu_box_2d3r/gpu_star_2d3r/
// gpu_star_2d1r ever cudaFree() their `array_d[0]`/`array_d[1]` buffers --
// grep confirms zero cudaFree calls anywhere in gpu.cu. Calling these
// functions repeatedly (once per adapter run() call, as this integration's
// times=1 discipline requires) leaks two device buffers per call. Not
// patched (kernel/host driver code, not a build-system fix) -- documented
// here and in STATUS.md; the login-node gate below uses a small, bounded
// number of calls, so the leak stays small and does not exhaust the GPU.
//
// OUTPUT VALID-REGION OFFSET (derived from reading kernel2d_box2d3r's/
// kernel2d_star2d3r's store_matrix_sync index arithmetic, confirmed
// empirically below): both kernels write valid output starting at buffer
// offset (row=4, col=4) within the (m+8)x(n+8) padded array -- a
// SYMMETRIC HALO=4 margin on all four sides (simpler than convstencil's
// asymmetric (3,4) case). `HALO` is gpu.cu's own #define (4), not
// reinvented here.
//
// SHAPE ALIGNMENT CONSTRAINT (inherited, not introduced): the kernel grid
// is `ceil(m/32) x ceil(n/64)` blocks (BLOCK_SIZE_ROW=32, BLOCK_SIZE_COL=64,
// gpu.cu), each block unconditionally loading/storing its full 32x64 tile
// with no boundary/tail guard -- so m must be a multiple of 32 and n a
// multiple of 64, or a block's tile will read/write past the padded
// array's allocation. adapter.py raises NotImplementedError otherwise.

#include "source/src/2d/gpu.cu"
#include <cstring>
#include <cstdlib>

extern "C" {

// field_padded: (m+8) x (n+8) row-major host buffer, periodic-wrap-padded
// by the caller (HALO=4 margin on every side). kernel49: 7x7 weight array,
// kernel49[i*7+j] weights offset (i-3, j-3) from the stencil's center
// (cross-correlation layout, verified in adapter.py -- see its docstring).
// out_valid: caller-allocated m*n buffer receiving the CROPPED (unpadded)
// single-sweep result. Exactly one kernel launch (times=1, see file
// docstring's TIMESTEPS note).
void lorastencil2d_box2d3r_sweep(const double *field_padded, double *out_valid,
                                  const double *kernel49, int m, int n) {
    int rows = m + 8, cols = n + 8;
    double *out_padded = (double *)malloc((size_t)rows * cols * sizeof(double));
    gpu_box_2d3r(field_padded, out_padded, kernel49, /*times=*/1, m, n);
    for (int i = 0; i < m; i++)
        memcpy(out_valid + (size_t)i * n, out_padded + (size_t)(i + 4) * cols + 4,
               n * sizeof(double));
    free(out_padded);
}

// Same contract as above, wrapping gpu_star_2d3r (native radius-3 star).
void lorastencil2d_star2d3r_sweep(const double *field_padded, double *out_valid,
                                   const double *kernel49, int m, int n) {
    int rows = m + 8, cols = n + 8;
    double *out_padded = (double *)malloc((size_t)rows * cols * sizeof(double));
    gpu_star_2d3r(field_padded, out_padded, kernel49, /*times=*/1, m, n);
    for (int i = 0; i < m; i++)
        memcpy(out_valid + (size_t)i * n, out_padded + (size_t)(i + 4) * cols + 4,
               n * sizeof(double));
    free(out_padded);
}

}  // extern "C"
