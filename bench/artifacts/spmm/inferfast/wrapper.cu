/*
 * Thin extern-"C" glue between kernelbench's ctypes-based adapter contract
 * and InferFast's C++ (non-extern-"C") host/device API. This file lives
 * OUTSIDE source/ and does not modify a single line of the artifact -- it
 * only re-declares the two host-side functions from
 * source/csrc/SpMM_API.cu that source/build/SpMM_API.cuh does not already
 * declare (InferFast_InitSparseMatrixA IS declared there and is reused
 * as-is; reorder_matrices is not, so it is re-declared here with an
 * IDENTICAL signature so the linker resolves it to the same symbol already
 * compiled into libSpMM_API.so -- no reimplementation, no behavior change).
 *
 * Split into two extern "C" entry points, matching the kernelbench
 * Implementation.prepare()/run() split:
 *
 *   inferfast_compress()   -- runs the artifact's OWN host-side compression
 *                              (reorder_matrices + InferFast_InitSparseMatrixA,
 *                              both unmodified) on a dense fp16 M x K matrix.
 *                              Returns an opaque handle plus the buffer sizes
 *                              needed to size numpy arrays on the Python side.
 *   inferfast_copy_out()   -- memcpy's the compressed host buffers into
 *                              caller-provided (numpy-backed) destinations.
 *   inferfast_free_compress() -- frees the host-side compression result.
 *   inferfast_run()        -- calls InferFast_SpMM_SplitK_API (unmodified)
 *                              given already-H2D-copied device buffers.
 *
 * Alignment requirement (NOT hidden, done explicitly by the Python caller):
 * InferFast_InitSparseMatrixA divides work into (tile_M_global=128,
 * tile_K_global=64) tiles via integer division (num_global_tiles_M = M /
 * tile_M_global) and never processes a tail that doesn't fill a full tile --
 * it silently DROPS rows/cols beyond the last full tile rather than erroring.
 * So this wrapper requires M % 128 == 0 and K % 64 == 0 on entry; the
 * adapter is responsible for zero-padding the dense matrix to those bounds
 * before calling inferfast_compress(), and for cropping the output back to
 * the true M afterwards. This is exactly the kind of "artifact's own format
 * conversion, done in prepare()" ARTIFACT_GUIDE.md rule 2 describes.
 */

#include "source/build/SpMM_API.cuh"
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

// Re-declaration only (see file header) -- resolves to the symbol already
// compiled into libSpMM_API.so from source/csrc/SpMM_API.cu.
void reorder_matrices(half* A_h, uint16_t* reorder, int M_GLOBAL, int K_GLOBAL);

struct InferFastCompressed {
    half*     compressed_val;
    uint16_t* tile_offsets_median;
    int*      tile_offsets_global;
    uint32_t* bitmap;
    uint16_t* reorder;
    int M, K;
    int num_gtiles;
    int num_ltiles;
    int val_count;
    int max_nnz_intile;
};

extern "C" {

// A_h_bits: host pointer to M*K raw fp16 bit patterns (row-major), already
// zero-padded so M % 128 == 0 and K % 64 == 0.
InferFastCompressed* inferfast_compress(const void* A_h_bits, int M, int K) {
    half* A_h = (half*)malloc(sizeof(half) * (size_t)M * (size_t)K);
    memcpy(A_h, A_h_bits, sizeof(half) * (size_t)M * (size_t)K);

    uint16_t* reorder = (uint16_t*)malloc(sizeof(uint16_t) * (size_t)K * (size_t)(M / 128));
    for (int i = 0; i < M / 128; i++)
        for (int j = 0; j < K; j++)
            reorder[(size_t)i * K + j] = (uint16_t)j;
    reorder_matrices(A_h, reorder, M, K);

    auto* r = new InferFastCompressed();
    r->compressed_val = nullptr;
    r->tile_offsets_median = nullptr;
    r->tile_offsets_global = nullptr;
    r->bitmap = nullptr;
    r->M = M;
    r->K = K;

    int max_nnz = 0;
    int num_gtiles = InferFast_InitSparseMatrixA(
        reorder, A_h, M, K, /*tile_M_median=*/32, /*tile_M_global=*/128,
        /*tile_K_median=*/64, /*tile_K_global=*/64,
        &r->compressed_val, &r->tile_offsets_median, &r->tile_offsets_global,
        &r->bitmap, max_nnz);

    int num_ltiles = num_gtiles * 128;
    int val_count = r->tile_offsets_global[num_gtiles];
    if (max_nnz % 64 != 0)
        max_nnz = ((max_nnz / 64) + 1) * 64;
    if (val_count == 0)
        val_count = 1;  // matches the artifact's own 100%-sparsity guard

    r->reorder = reorder;
    r->num_gtiles = num_gtiles;
    r->num_ltiles = num_ltiles;
    r->val_count = val_count;
    r->max_nnz_intile = max_nnz;

    free(A_h);
    return r;
}

void inferfast_compress_sizes(InferFastCompressed* r, int* num_gtiles, int* num_ltiles,
                               int* val_count, int* max_nnz_intile, int* reorder_len) {
    *num_gtiles = r->num_gtiles;
    *num_ltiles = r->num_ltiles;
    *val_count = r->val_count;
    *max_nnz_intile = r->max_nnz_intile;
    *reorder_len = r->K * (r->M / 128);
}

// Destinations are caller-owned (numpy) buffers, sized per
// inferfast_compress_sizes() above.
void inferfast_copy_out(InferFastCompressed* r, void* out_compressed_val,
                         void* out_tile_offsets_median, void* out_tile_offsets_global,
                         void* out_bitmap, void* out_reorder) {
    memcpy(out_compressed_val, r->compressed_val, sizeof(uint16_t) * (size_t)r->val_count);
    memcpy(out_tile_offsets_median, r->tile_offsets_median,
           sizeof(uint16_t) * (size_t)r->num_ltiles);
    memcpy(out_tile_offsets_global, r->tile_offsets_global,
           sizeof(int) * (size_t)(r->num_gtiles + 1));
    memcpy(out_bitmap, r->bitmap, sizeof(uint32_t) * (size_t)r->num_ltiles);
    memcpy(out_reorder, r->reorder, sizeof(uint16_t) * (size_t)r->K * (size_t)(r->M / 128));
}

void inferfast_free_compress(InferFastCompressed* r) {
    free(r->compressed_val);
    free(r->tile_offsets_median);
    free(r->tile_offsets_global);
    free(r->bitmap);
    free(r->reorder);
    delete r;
}

// All *_dev pointers are already-uploaded device buffers (torch tensors'
// .data_ptr(), same convention as kernelbench/impls/gpu_cuda.py's
// CustomSpMM). Output C_dev is COLUMN-MAJOR (M, N) -- this is InferFast's
// own storage convention (source/csrc/SpMM_Kernel.cuh writes
// `BlockGlobalPTR[j + i * M_Global]`), so the adapter must transpose on
// read, not something introduced by this wrapper.
cudaError_t inferfast_run(const void* compressed_val_dev, const void* tile_offsets_global_dev,
                           const void* tile_offsets_median_dev, const void* reorder_dev,
                           const void* bitmap_dev, const void* max_nnz_intile_dev,
                           int max_nnz_intile_cpu, const void* B_dev, void* C_dev,
                           int M, int N, int K, int split_k) {
    return InferFast_SpMM_SplitK_API(
        /*stream=*/0, /*A=*/nullptr,  // A is a dead parameter: never read by
                                      // the kernel (only Compressed_A is),
                                      // confirmed by tracing SpMM_API.cu.
        (const half*)compressed_val_dev, (const int*)tile_offsets_global_dev,
        (const uint16_t*)tile_offsets_median_dev, (const uint16_t*)reorder_dev,
        (const uint32_t*)bitmap_dev, (const int*)max_nnz_intile_dev, max_nnz_intile_cpu,
        (const half*)B_dev, (half*)C_dev, M, N, K,
        /*Reduction_Workspace=*/nullptr,  // unused when split_k == 1 (checked
                                          // by the adapter before calling in)
        split_k);
}

}  // extern "C"
