// Thin ctypes-callable bridge around ConvStencil's 2D Tensor-Core stencil
// kernel (source/src/2d/gpu.cu: `kernel2d`, set up as in `gpu_box_2d1r`).
//
// NOT part of the artifact. gpu.cu is #included VERBATIM (unmodified) so
// this translation unit can see its `kernel2d` __global__ and the
// `param_matrix_d` __constant__ it reads.
//
// Why a bridge instead of the artifact's entry points:
//   (a) main.cu's CLI never populates weights for star2d1r/star2d3r
//       (`param_star_2d1r` is declared zero and never written), so we supply
//       our own 7x7 weight array, built from StencilWorkload.weights.
//   (b) gpu_box_2d1r() bundles allocation, H2D copy, a `times` loop and D2H
//       copy in one call taking host pointers, and its `times` loop
//       ping-pongs the device buffers WITHOUT refreshing the halo between
//       steps, so for T>1 it does not compute a periodic-wrap T-step sweep.
//
// History: the first version of this bridge (2026-09) worked around (b) by
// calling gpu_box_2d1r(times=1) once per step from Python and re-padding on
// the host, so every timed step paid a host re-pad plus cudaMalloc + H2D +
// D2H -- the timed region mostly measured PCIe traffic, not the kernel
// (6-11 ms for a 256^2, T=5 smoke run). This version (2026-09-23) keeps the
// grid on the device for the whole run:
//
//   prepare  once, untimed: weights -> param_matrix_d (gpu_box_2d1r's own
//            packing loops, copied verbatim), lookup tables (same), device
//            buffers, pristine padded initial field (H2D once, halo filled
//            on the device).
//   run(T)   timed: reset buf[0] from the pristine copy (D2D, O(grid) once
//            per call, same reset discipline as NumpyStencil.run()), then T x
//            { kernel2d (the artifact's kernel, same launch config as
//            gpu_box_2d1r); periodic halo refresh (periodic_halo.cuh,
//            O(perimeter)) }. No host transfer, no sync inside the loop.
//   copy_out untimed: D2H of the interior of the current buffer.
//
// The halo refresh is the one thing the artifact's own loop lacks and a
// correct periodic T-step sweep needs; its cost is O(m + n) per step against
// the kernel's O(m * n).
//
// Padded-buffer layout (unchanged from the first bridge, derived from
// kernel2d's load/store index arithmetic and verified empirically, see
// STATUS.md): rows = m + 2*HALO, cols (= ldm) = n + 2*HALO + 2, HALO = 3.
// kernel2d reads rows [0, m+6) and columns [1, n+7); it writes the interior
// at rows [3, 3+m), columns [4, 4+n). Columns 0 and n+7 are alignment padding
// and never read. So the halo to refresh is 3 rows above/below and 3 columns
// left/right of an interior anchored at (3, 4).
//
// Alignment: gpu_box_2d1r launches ceil(m/32) x ceil(n/64) blocks with no
// tail guard, so m must be a multiple of 32 and n of 64 (the adapter checks).
//
// params49: 7x7 row-major, params49[i*7+j] weights in[row+(i-3)][col+(j-3)]
// (main.cu's naive_box2d1r convention; radius-1 shapes are zero-padded into
// the 7x7 by the adapter).
//
// param_matrix_d is a module-level __constant__, so only one prepared handle
// is valid at a time (the harness prepares, runs and frees one at a time).

#include "gpu.cu"
#include "../periodic_halo.cuh"

#define CS_ROW0 HALO          // interior row origin
#define CS_COL0 (HALO + 1)    // interior column origin

struct CsHandle {
    double *buf[2];
    double *init;
    int *lt1, *lt2;
    int m, n, rows, cols, cur;
};

extern "C" {

void *convstencil2d_prepare(const double *field, const double *params, int m, int n) {
    // --- gpu_box_2d1r's parameter-matrix packing, verbatim (gpu.cu) --------
    double param_matrix_h[2][52 * 8] = {0.0};
    for (int col = 0; col < TENSOR_CORE_M; col++) {
        for(int i = 0; i < UNIT_LENGTH; i++) {
            for(int j = 0; j < UNIT_LENGTH; j++) {
                if (j >= col) {
                    param_matrix_h[0][(i * UNIT_LENGTH + j) * 8 + col] = params[i * UNIT_LENGTH + j - col];
                }
            }
        }
    }
    for (int col = 0; col < TENSOR_CORE_M; col++) {
        for(int i = 0; i < UNIT_LENGTH; i++) {
            for(int j = 0; j < UNIT_LENGTH; j++) {
                if (j < col) {
                    param_matrix_h[1][(i * UNIT_LENGTH + j) * 8 + col] = params[i * UNIT_LENGTH + j - col + 7];
                }
            }
        }
    }
    CUDA_CHECK(cudaMemcpyToSymbol(param_matrix_d, param_matrix_h, 2 * 8 * 52 * sizeof(double)));

    CsHandle *h = new CsHandle();
    h->m = m; h->n = n; h->cur = 0;
    h->rows = m + 2 * HALO;
    h->cols = n + 2 * HALO + 2;
    const size_t bytes = (size_t)h->rows * h->cols * sizeof(double);
    CUDA_CHECK(cudaMalloc(&h->buf[0], bytes));
    CUDA_CHECK(cudaMalloc(&h->buf[1], bytes));
    CUDA_CHECK(cudaMalloc(&h->init, bytes));
    CUDA_CHECK(cudaMemset(h->buf[0], 0, bytes));
    CUDA_CHECK(cudaMemset(h->buf[1], 0, bytes));
    CUDA_CHECK(cudaMemset(h->init, 0, bytes));

    // pristine initial field: interior H2D, halo filled on the device
    CUDA_CHECK(cudaMemcpy2D(h->init + (size_t)CS_ROW0 * h->cols + CS_COL0,
                            h->cols * sizeof(double), field, n * sizeof(double),
                            n * sizeof(double), m, cudaMemcpyHostToDevice));
    kb_periodic_halo(h->init, h->cols, m, n, CS_ROW0, CS_COL0, HALO, HALO);

    // --- gpu_box_2d1r's lookup tables, verbatim (gpu.cu) --------------------
    int lookup_table1_h[D_BLOCK_SIZE_ROW][D_BLOCK_SIZE_COL];
    int lookup_table2_h[D_BLOCK_SIZE_ROW][D_BLOCK_SIZE_COL];
    for (int i = 0; i < D_BLOCK_SIZE_ROW; i++) {
        for (int j = 0; j < D_BLOCK_SIZE_COL; j++) {
            if ((j + 1) % 8 != 0 && j < D_BLOCK_SIZE_COL - 2 * HALO - 1) {
                lookup_table1_h[i][j] = IDX(j / (UNIT_LENGTH + 1), UNIT_LENGTH * i + j % (UNIT_LENGTH + 1), SM_SIZE_COL);
            } else {
                lookup_table1_h[i][j] = SM_SIZE_ROW * SM_SIZE_COL - 1;
            }
            if ((j + 2) % 8 != 0 && j > 2 * HALO) {
                lookup_table2_h[i][j] = IDX((j - UNIT_LENGTH) / (UNIT_LENGTH + 1), UNIT_LENGTH * i + (j - UNIT_LENGTH) % (UNIT_LENGTH + 1), SM_SIZE_COL);
            } else {
                lookup_table2_h[i][j] = SM_SIZE_ROW * SM_SIZE_COL - 1;
            }
        }
    }
    const size_t lt_bytes = D_BLOCK_SIZE_ROW * D_BLOCK_SIZE_COL * sizeof(int);
    CUDA_CHECK(cudaMalloc(&h->lt1, lt_bytes));
    CUDA_CHECK(cudaMalloc(&h->lt2, lt_bytes));
    CUDA_CHECK(cudaMemcpy(h->lt1, lookup_table1_h, lt_bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(h->lt2, lookup_table2_h, lt_bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaDeviceSynchronize());
    return h;
}

// T periodic-wrap steps, all on the device. Asynchronous: the caller's CUDA
// event timer brackets it and synchronizes on its stop event.
void convstencil2d_run(void *handle, int timesteps) {
    CsHandle *h = (CsHandle *)handle;
    const size_t bytes = (size_t)h->rows * h->cols * sizeof(double);
    CUDA_CHECK(cudaMemcpyAsync(h->buf[0], h->init, bytes, cudaMemcpyDeviceToDevice, 0));
    // gpu_box_2d1r's launch configuration
    dim3 grid_config((h->m + BLOCK_SIZE_ROW - 1) / BLOCK_SIZE_ROW,
                     (h->n + BLOCK_SIZE_COL - 1) / BLOCK_SIZE_COL);
    dim3 block_config(32 * WARP_PER_BLOCK);
    int cur = 0;
    for (int t = 0; t < timesteps; t++) {
        kernel2d<<<grid_config, block_config>>>(h->buf[cur], h->buf[1 - cur], h->cols,
                                                h->lt1, h->lt2);
        kb_periodic_halo(h->buf[1 - cur], h->cols, h->m, h->n, CS_ROW0, CS_COL0, HALO, HALO);
        cur = 1 - cur;
    }
    CUDA_CHECK(cudaGetLastError());
    h->cur = cur;
}

// Interior of the current buffer -> out (m x n, row-major, host).
void convstencil2d_copy_out(void *handle, double *out) {
    CsHandle *h = (CsHandle *)handle;
    CUDA_CHECK(cudaMemcpy2D(out, h->n * sizeof(double),
                            h->buf[h->cur] + (size_t)CS_ROW0 * h->cols + CS_COL0,
                            h->cols * sizeof(double), h->n * sizeof(double), h->m,
                            cudaMemcpyDeviceToHost));
}

void convstencil2d_free(void *handle) {
    CsHandle *h = (CsHandle *)handle;
    cudaFree(h->buf[0]); cudaFree(h->buf[1]); cudaFree(h->init);
    cudaFree(h->lt1); cudaFree(h->lt2);
    delete h;
}

}  // extern "C"
