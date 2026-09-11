// Thin ctypes-callable bridge around CB-SpMV's own kernel + format-
// construction code (source/cb-spmv/src/{coo2block.h,cb-spmv.cuh}).
//
// NOT part of the artifact. It exists because the artifact only ships a
// monolithic driver, `cb_spmv()` (cb-spmv.cuh), that:
//   (a) builds ALL device buffers from a GLOBAL_BLOCK (the cache-block
//       format `coo2block_gather` -- coo2block.h -- constructs on the
//       CPU),
//   (b) launches the kernel ONCE (untimed) to get a value for its own
//       correctness check,
//   (c) then wraps ITER=10 (macros.h) further launches in a SINGLE
//       cudaEvent pair and reports the mean -- exactly the
//       "one timestamp pair around a batched loop" pattern our harness's
//       CudaEventTimer (kernelbench/impls/gpu_cuda.py) is built to avoid,
//       per this integration's brief ("its own code uses ITER=10; its
//       cache-block format construction belongs in prepare()").
//
// This file splits (a) into `cbspmv_prepare` (called once, timed as
// preprocessing) and re-exposes the untouched kernels
// (cb_spmv_detail::spmv_cuda_kernel_comp / _gather, defined in
// cb-spmv.cuh, included below verbatim) through `cbspmv_run`, called once
// per harness iteration. No kernel code is modified -- every CUDA_CHECK,
// device malloc, memcpy, and kernel launch here is lifted directly from
// cb_spmv() in cb-spmv.cuh, just reorganized across prepare/run instead of
// being bundled with the artifact's own ITER-loop timing.
//
// One correctness-relevant addition: the kernels accumulate into d_y via
// atomicAdd, and the artifact's own driver only zeros d_y ONCE (before its
// whole 10-call timing loop) and never re-validates output after the first
// call. Our harness calls run() independently many times (correctness
// check, warmup, measured reps) and requires EACH call to produce a
// correct y on its own, so cbspmv_run() re-zeros d_y (cudaMemsetAsync,
// O(y_elements), cheap) immediately before every launch.

#include "format.h"
#include "coo2block.h"
#include "cb-spmv.cuh"

#include <cstring>
#include <vector>

struct CbSpmvHandle {
    std::vector<char *> block_addresses;
    int *d_gb_row_idx = nullptr;
    int *d_gb_col_idx = nullptr;
    int *d_nnz_per_blk = nullptr;
    BlockType *d_type_per_blk = nullptr;
    int *d_block_cols = nullptr;
    int *d_block_cols_offset = nullptr;
    char **d_block_addresses = nullptr;
    ValType *d_x = nullptr;
    ValType *d_y = nullptr;
    int gb_nnzb = 0;
    bool gather_flag = false;
    int num_blocks = 0;
    int num_threads = 0;
    size_t y_elements = 0;
    int mtx_m = 0;
};

extern "C" {

// Builds the artifact's cache-block format (coo2block_gather, CPU/OpenMP)
// from a raw COO matrix, then allocates + H2D-copies every device buffer
// cb_spmv() would (block payloads via cb_spmv_detail::get_addr_with_*,
// metadata arrays, x). Returns an opaque handle. This whole function IS
// the artifact's preprocessing -- called once, timed once, by the adapter.
void *cbspmv_prepare(int mtx_m, int mtx_n, int mtx_nnz,
                      const int *row_idx, const int *col_idx,
                      const double *val, const double *x_in) {
    GLOBAL_BLOCK gb_block(mtx_m, mtx_n, mtx_nnz);
    coo2block_gather(row_idx, col_idx, val, gb_block);

    CbSpmvHandle *h = new CbSpmvHandle();
    h->gather_flag = gb_block.gather_flag;
    h->gb_nnzb = gb_block.gb_nnzb;
    h->mtx_m = mtx_m;

    const size_t x_elements = static_cast<size_t>(gb_block.gb_n) * BLOCK_SIZE;
    h->y_elements = static_cast<size_t>(gb_block.gb_m) * BLOCK_SIZE;

    std::vector<ValType> h_x(x_elements, 0.0);
    memcpy(h_x.data(), x_in, static_cast<size_t>(mtx_n) * sizeof(ValType));

    h->block_addresses.reserve(gb_block.gb_nnzb);
    for (int i = 0; i < gb_block.gb_nnzb; i++) {
        char *block_address = nullptr;
        if (gb_block.nnz_per_blk[i] > 0) {
            int old_idx = gb_block.block_idx_array[i].origin_idx;
            BlockType block_type = gb_block.type_per_blk[i];
            if (block_type == COO) {
                block_address = cb_spmv_detail::get_addr_with_coo_comp(
                    gb_block.data_per_coo_blk_comp.at(old_idx));
            } else if (block_type == CSR) {
                block_address = cb_spmv_detail::get_addr_with_csr(
                    gb_block.data_per_csr_blk.at(old_idx));
            } else if (block_type == DENSE) {
                block_address = cb_spmv_detail::get_addr_with_dense(
                    gb_block.data_per_dense_blk.at(old_idx));
            }
        }
        h->block_addresses.push_back(block_address);
    }

    CHECK_CUDA(cudaMalloc((void **)&h->d_gb_row_idx, gb_block.gb_nnzb * sizeof(int)));
    CHECK_CUDA(cudaMalloc((void **)&h->d_gb_col_idx, gb_block.gb_nnzb * sizeof(int)));
    CHECK_CUDA(cudaMalloc((void **)&h->d_nnz_per_blk, gb_block.gb_nnzb * sizeof(int)));
    CHECK_CUDA(cudaMalloc((void **)&h->d_type_per_blk, gb_block.gb_nnzb * sizeof(BlockType)));
    CHECK_CUDA(cudaMalloc((void **)&h->d_block_addresses, gb_block.gb_nnzb * sizeof(char *)));
    CHECK_CUDA(cudaMalloc((void **)&h->d_x, x_elements * sizeof(ValType)));
    CHECK_CUDA(cudaMalloc((void **)&h->d_y, h->y_elements * sizeof(ValType)));

    if (h->gather_flag) {
        CHECK_CUDA(cudaMalloc((void **)&h->d_block_cols, gb_block.block_cols.size() * sizeof(int)));
        CHECK_CUDA(cudaMalloc((void **)&h->d_block_cols_offset, gb_block.block_cols_offset.size() * sizeof(int)));
    }

    CHECK_CUDA(cudaMemcpy(h->d_gb_row_idx, gb_block.gb_row_idx.data(), gb_block.gb_nnzb * sizeof(int), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(h->d_gb_col_idx, gb_block.gb_col_idx.data(), gb_block.gb_nnzb * sizeof(int), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(h->d_nnz_per_blk, gb_block.nnz_per_blk.data(), gb_block.gb_nnzb * sizeof(int), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(h->d_type_per_blk, gb_block.type_per_blk.data(), gb_block.gb_nnzb * sizeof(BlockType), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(h->d_block_addresses, h->block_addresses.data(), gb_block.gb_nnzb * sizeof(char *), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(h->d_x, h_x.data(), x_elements * sizeof(ValType), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemset(h->d_y, 0, h->y_elements * sizeof(ValType)));

    if (h->gather_flag) {
        CHECK_CUDA(cudaMemcpy(h->d_block_cols, gb_block.block_cols.data(), gb_block.block_cols.size() * sizeof(int), cudaMemcpyHostToDevice));
        CHECK_CUDA(cudaMemcpy(h->d_block_cols_offset, gb_block.block_cols_offset.data(), gb_block.block_cols_offset.size() * sizeof(int), cudaMemcpyHostToDevice));
    }

    h->num_threads = WARP_PER_BLOCK * WARP_SIZE;
    h->num_blocks = (gb_block.gb_nnzb + WARP_PER_BLOCK - 1) / WARP_PER_BLOCK;

    return h;
}

// ONE y=Ax call: re-zero the atomicAdd accumulator, launch the artifact's
// own kernel exactly once, on `stream` (pass 0 / null for the default
// stream -- matches kernelbench/csrc/kernels.cu's own ctypes convention).
void cbspmv_run(void *handle, cudaStream_t stream) {
    CbSpmvHandle *h = reinterpret_cast<CbSpmvHandle *>(handle);
    CHECK_CUDA(cudaMemsetAsync(h->d_y, 0, h->y_elements * sizeof(ValType), stream));
    if (h->gather_flag) {
        cb_spmv_detail::spmv_cuda_kernel_gather<<<h->num_blocks, h->num_threads, 0, stream>>>(
            h->gb_nnzb, h->d_nnz_per_blk, h->d_type_per_blk, h->d_gb_row_idx, h->d_gb_col_idx,
            h->d_block_addresses, h->d_block_cols, h->d_block_cols_offset, h->d_x, h->d_y);
    } else {
        cb_spmv_detail::spmv_cuda_kernel_comp<<<h->num_blocks, h->num_threads, 0, stream>>>(
            h->gb_nnzb, h->d_nnz_per_blk, h->d_type_per_blk, h->d_gb_row_idx, h->d_gb_col_idx,
            h->d_block_addresses, h->d_x, h->d_y);
    }
}

// D2H copy of y (mtx_m elements); a plain cudaMemcpy on the default stream
// blocks until any prior kernel on that stream completes, so this also
// serves as the sync point between run() and the correctness/gate check.
void cbspmv_copy_y(void *handle, double *host_y_out) {
    CbSpmvHandle *h = reinterpret_cast<CbSpmvHandle *>(handle);
    CHECK_CUDA(cudaMemcpy(host_y_out, h->d_y, h->mtx_m * sizeof(ValType), cudaMemcpyDeviceToHost));
}

void cbspmv_free(void *handle) {
    CbSpmvHandle *h = reinterpret_cast<CbSpmvHandle *>(handle);
    if (!h) return;
    cudaFree(h->d_gb_row_idx);
    cudaFree(h->d_gb_col_idx);
    cudaFree(h->d_nnz_per_blk);
    cudaFree(h->d_type_per_blk);
    for (char *a : h->block_addresses) {
        if (a) cudaFree(a);
    }
    cudaFree(h->d_block_addresses);
    if (h->d_block_cols) cudaFree(h->d_block_cols);
    if (h->d_block_cols_offset) cudaFree(h->d_block_cols_offset);
    cudaFree(h->d_x);
    cudaFree(h->d_y);
    delete h;
}

}  // extern "C"
