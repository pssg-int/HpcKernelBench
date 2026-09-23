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
//   - star2d3r -> kernel2d_star2d3r, set up as in gpu_star_2d3r() (wrapped)
//   - box2d3r  NOT wrapped any more: gpu_box_2d3r's factorization returns
//     NaN for this domain's weights (see adapter.py); its host-side sweep
//     entry point was dropped from this bridge on 2026-09-23.
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
//
// TIMESTEPS / GPU RESIDENCY (rewritten 2026-09-23): gpu_star_2d3r's own
// `times` loop ping-pongs device buffers WITHOUT refreshing the periodic-wrap
// halo between sweeps (same finding as convstencil's gpu_box_2d1r), so it
// cannot run a periodic T-step sweep. The first version of this bridge
// worked around that by calling gpu_star_2d3r(times=1) once per step from
// Python with a host re-pad in between, so every timed step paid a host
// re-pad plus cudaMalloc + H2D + D2H (and leaked the two buffers
// gpu_star_2d3r never frees). This version launches the artifact's own
// kernel2d_star2d3r directly, with gpu_star_2d3r's launch configuration and
// its parameter-matrix setup copied verbatim, and refreshes the periodic
// halo on the device after each step (../periodic_halo.cuh, O(perimeter)):
//
//   prepare  once, untimed: U/V parameter matrices -> __constant__, device
//            buffers, pristine padded initial field (H2D once, halo filled
//            on the device).
//   run(T)   timed: reset buf[0] from the pristine copy (D2D), then T x
//            { kernel2d_star2d3r; halo refresh }. No host transfer.
//   copy_out untimed: D2H of the interior.
//
// gpu_star_2d3r is no longer called, so its missing cudaFree (the artifact
// never frees array_d[0]/[1]) no longer leaks here.
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
#include "../periodic_halo.cuh"

struct LoraHandle {
    double *buf[2];
    double *init;
    int m, n, rows, cols, cur;
};

extern "C" {

// field: m x n row-major host array (unpadded). kernel49: 7x7 weight array,
// kernel49[i*7+j] weights offset (i-3, j-3) (cross-correlation layout, see
// adapter.py). Only the centre row and column are read (a star).
void *lorastencil2d_star2d3r_prepare(const double *field, const double *params,
                                     int m, int n) {
    // --- gpu_star_2d3r's parameter matrices, verbatim (gpu.cu) --------------
    double param_matrix_U[16*8] = {0.0};
    double param_matrix_V[16*8] = {0.0};
    for (int row = 0; row < 8; row++) {
        for (int col = 0; col < 7; col++) {
            param_matrix_U[row * 16 + col + 1 + row] = params[col * 7 + 3];
        }
    }
    for (int col = 0; col < 8; col++) {
        for (int row = 0; row < 7; row++) {
            if (row != 3) {
                param_matrix_V[(row + col + 1) * 8 + col] = params[3 * 7 + row];
            }
        }
    }
    CUDA_CHECK(cudaMemcpyToSymbol(matrix_star2d3r_U_d, param_matrix_U, 8 * 16 * sizeof(double)));
    CUDA_CHECK(cudaMemcpyToSymbol(matrix_star2d3r_V_d, param_matrix_V, 8 * 16 * sizeof(double)));

    LoraHandle *h = new LoraHandle();
    h->m = m; h->n = n; h->cur = 0;
    h->rows = m + 2 * HALO;
    h->cols = n + 2 * HALO;
    const size_t bytes = (size_t)h->rows * h->cols * sizeof(double);
    CUDA_CHECK(cudaMalloc(&h->buf[0], bytes));
    CUDA_CHECK(cudaMalloc(&h->buf[1], bytes));
    CUDA_CHECK(cudaMalloc(&h->init, bytes));
    CUDA_CHECK(cudaMemset(h->buf[0], 0, bytes));
    CUDA_CHECK(cudaMemset(h->buf[1], 0, bytes));
    CUDA_CHECK(cudaMemset(h->init, 0, bytes));
    CUDA_CHECK(cudaMemcpy2D(h->init + (size_t)HALO * h->cols + HALO,
                            h->cols * sizeof(double), field, n * sizeof(double),
                            n * sizeof(double), m, cudaMemcpyHostToDevice));
    kb_periodic_halo(h->init, h->cols, m, n, HALO, HALO, HALO, HALO);
    CUDA_CHECK(cudaDeviceSynchronize());
    return h;
}

// T periodic-wrap steps, all on the device, asynchronous (the caller's CUDA
// event timer syncs on its stop event).
void lorastencil2d_star2d3r_run(void *handle, int timesteps) {
    LoraHandle *h = (LoraHandle *)handle;
    const size_t bytes = (size_t)h->rows * h->cols * sizeof(double);
    CUDA_CHECK(cudaMemcpyAsync(h->buf[0], h->init, bytes, cudaMemcpyDeviceToDevice, 0));
    // gpu_star_2d3r's launch configuration
    dim3 grid_config((h->m + BLOCK_SIZE_ROW - 1) / BLOCK_SIZE_ROW,
                     (h->n + BLOCK_SIZE_COL - 1) / BLOCK_SIZE_COL);
    dim3 block_config(32 * WARP_PER_BLOCK);
    int cur = 0;
    for (int t = 0; t < timesteps; t++) {
        kernel2d_star2d3r<<<grid_config, block_config>>>(h->buf[cur], h->buf[1 - cur], h->cols);
        kb_periodic_halo(h->buf[1 - cur], h->cols, h->m, h->n, HALO, HALO, HALO, HALO);
        cur = 1 - cur;
    }
    CUDA_CHECK(cudaGetLastError());
    h->cur = cur;
}

void lorastencil2d_star2d3r_copy_out(void *handle, double *out) {
    LoraHandle *h = (LoraHandle *)handle;
    CUDA_CHECK(cudaMemcpy2D(out, h->n * sizeof(double),
                            h->buf[h->cur] + (size_t)HALO * h->cols + HALO,
                            h->cols * sizeof(double), h->n * sizeof(double), h->m,
                            cudaMemcpyDeviceToHost));
}

void lorastencil2d_star2d3r_free(void *handle) {
    LoraHandle *h = (LoraHandle *)handle;
    cudaFree(h->buf[0]); cudaFree(h->buf[1]); cudaFree(h->init);
    delete h;
}

}  // extern "C"
