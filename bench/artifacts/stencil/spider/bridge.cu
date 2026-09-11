// Thin ctypes-callable bridge around SPIDER's own 2D 7-radius half-precision
// sparse-Tensor-Core stencil kernel
// (source/src/2d_half_sparse/{gpu_2d_7r_half.h,gpu_2d_7r_half.cu}).
//
// NOT part of the artifact. `#include`s gpu_2d_7r_half.cu VERBATIM below
// (same trick the cb-spmv bridge uses on a .cuh -- the kernel is defined in
// a .cu, not a header, but textual #include works identically and keeps
// everything in one translation unit, no -rdc=true needed) so that
// `kernel_2d_7r` (the __global__ that does the actual mma.sp::ordered_
// metadata 2:4-structured-sparse-Tensor-Core work) and the artifact's own
// host-side param-compression helpers (`param_swap_to_structured_sparsity`,
// `compress_params`) stay completely untouched. Nothing about kernel math
// is changed anywhere in this file.
//
// Why a bridge was needed at all: the artifact's only host entry point,
// `gpu_2d_7r()` (gpu_2d_7r_half.cu), is monolithic -- one call does
// parameter compression (host), device malloc, H2D copy of the padded
// input, EITHER a single untimed correctness launch (check=true) OR an
// `times`-launch loop wrapped in its own cudaEvent pair (check=false), and
// finally a D2H copy -- all bundled together, exactly the "own T-sweep
// loop baked into one call, plus the artifact's own format construction"
// pattern the integration brief expects to need splitting. This file
// splits it into `spider2d7r_prepare()` (host param-compression + device
// alloc + H2D, called once, timed as preprocessing) and `spider2d7r_run()`
// (the kernel-launch loop only, called once per harness iteration).
//
// Two things this bridge does NOT inherit unmodified from gpu_2d_7r(),
// both purely host-side orchestration bugs/decisions, not kernel changes:
//
// 1. BUG FIX -- final-buffer off-by-one. gpu_2d_7r()'s own last line is
//        CUDA_CHECK(cudaMemcpy(out, array_d[(times + 1) % 2], ...));
//    but its ping-pong loop is
//        for (i = 0; i < times; i++)
//            kernel_2d_7r<<<>>>(array_d[i % 2], ..., array_d[(i + 1) % 2], ...);
//    so the LAST launch (i = times - 1) writes array_d[((times - 1) + 1) % 2]
//    == array_d[times % 2] -- but the final copy reads array_d[(times + 1) % 2],
//    which is the OTHER buffer (since (times+1) and (times-1) have the same
//    parity mod 2, i.e. opposite parity to `times`). Concretely, for
//    times=1 this reads back array_d[0], the UNTOUCHED ORIGINAL INPUT, not
//    the swept result in array_d[1]. This path is never exercised by the
//    artifact's own tests: check_result() (2d_stencil_half.cu) always calls
//    gpu_2d_7r(..., 1, /*check=*/true), which takes the OTHER branch
//    (times is reassigned to 0 first, single untimed launch, correct copy
//    from array_d[1]); the buggy multi-launch branch is only ever reached
//    from main()'s --profile path, whose output is never compared to
//    anything. spider2d7r_run() below tracks the correct last-written index
//    itself (`times % 2`) instead of reproducing this indexing.
//
// 2. Bounds requirement made explicit. kernel_2d_7r's grid is
//    dim3(CEIL(input_m, BLOCK_ROW), CEIL(input_n, BLOCK_COL)) and every
//    launched block writes its full BLOCK_ROW x BLOCK_COL output tile
//    unconditionally (gpu_2d_7r_half.cu, the final `out[begin + ... +
//    out_offset]` loop has no per-element bounds check against input_m/n).
//    If input_m/input_n are not exact multiples of BLOCK_ROW=64/BLOCK_COL=
//    128, the last row/col of blocks writes past the allocated (input_m +
//    2*HALO) x (input_n + 2*HALO) buffer. Rather than silently growing the
//    allocation (which would change what "the interior" means), prepare()
//    below rejects non-multiple shapes explicitly -- this is a real,
//    load-bearing limitation of the shipped kernel, not a bridge choice.
//
// BOUNDARY: SPIDER pads the domain with a halo of HALO=8 on every side and
// the kernel only ever writes the interior [HALO, HALO+input_m) x
// [HALO, HALO+input_n) region of its output buffer -- the halo of BOTH
// ping-pong buffers is written exactly once, at prepare() time (memset 0 /
// H2D copy of a host buffer we zero-fill ourselves), and is never touched
// by any kernel launch afterwards. This is a FIXED, ZERO-valued halo, not
// a periodic wrap -- adapter.py documents how the harness's periodic-wrap
// fp64 reference is reconciled with this (crop the boundary band before
// comparing; see adapter.py's module docstring).

#include "gpu_2d_7r_half.cu"

#include <cstdint>
#include <cstring>
#include <vector>

struct Spider2d7rHandle {
    TYPE *array_d[2] = {nullptr, nullptr};
    TYPE *compressed_params_d = nullptr;
    uint *metadata_d = nullptr;
    int input_m = 0;
    int input_n = 0;
    int rows = 0;   // input_m + 2*HALO
    int cols = 0;   // input_n + 2*HALO
    dim3 grid_config;
    dim3 block_config;
    int last_out_idx = 1;  // which array_d[] holds the most recent result
};

extern "C" {

// h_field: input_m*input_n row-major, IEEE754 binary16 bit patterns (the
// INTERIOR grid values only, no padding -- prepare() builds the padded,
// zero-halo host buffer itself, mirroring gpu_2d_7r()'s own H2D-copy step
// but with a caller-supplied, known interior instead of whatever raw bytes
// main()'s own malloc'd (uninitialized) halo happened to contain).
//
// h_params: 15*16*32 half bit patterns in the artifact's own dense-params
// layout (params[IDX(param_iter,0,16*32) + IDX(row,col,32)], see
// 2d_stencil_half.cu main()'s own generation loop, reproduced in
// adapter.py's _native_params()).
void *spider2d7r_prepare(const uint16_t *h_field, int input_m, int input_n,
                          const uint16_t *h_params) {
    if (input_m % BLOCK_ROW != 0 || input_n % BLOCK_COL != 0) {
        fprintf(stderr,
                "spider2d7r_prepare: grid %dx%d is not a multiple of "
                "BLOCK_ROW=%d x BLOCK_COL=%d (kernel_2d_7r has no per-block "
                "bounds check; refusing to risk an out-of-bounds write)\n",
                input_m, input_n, BLOCK_ROW, BLOCK_COL);
        return nullptr;
    }

    Spider2d7rHandle *h = new Spider2d7rHandle();
    h->input_m = input_m;
    h->input_n = input_n;
    h->rows = input_m + 2 * HALO;
    h->cols = input_n + 2 * HALO;
    const size_t array_elems = (size_t)h->rows * (size_t)h->cols;

    // Host-side padded buffer, zero-filled halo (fixed/non-periodic
    // boundary -- see this file's docstring), interior copied from caller.
    std::vector<TYPE> h_in(array_elems, TYPE(0));
    const TYPE *field = reinterpret_cast<const TYPE *>(h_field);
    for (int i = 0; i < input_m; i++) {
        memcpy(&h_in[(size_t)(i + HALO) * h->cols + HALO],
               field + (size_t)i * input_n, input_n * sizeof(TYPE));
    }

    // Artifact's own param compression -- verbatim functions from
    // gpu_2d_7r_half.cu, called exactly as gpu_2d_7r() itself calls them.
    TYPE *sparse_params = (TYPE *)malloc(15 * 16 * 32 * sizeof(TYPE));
    memcpy(sparse_params, h_params, 15 * 16 * 32 * sizeof(TYPE));
    param_swap_to_structured_sparsity(
        reinterpret_cast<const TYPE *>(h_params), sparse_params);

    TYPE *compressed_params = (TYPE *)malloc(15 * 16 * 16 * sizeof(TYPE));
    uint *metadata = (uint *)malloc(16 * sizeof(uint));
    memset(compressed_params, 0, 15 * 16 * 16 * sizeof(TYPE));
    memset(metadata, 0, 16 * sizeof(uint));
    compress_params(sparse_params, compressed_params, metadata);
    free(sparse_params);

    CUDA_CHECK(cudaMalloc(&h->compressed_params_d, 15 * 16 * 16 * sizeof(TYPE)));
    CUDA_CHECK(cudaMalloc(&h->metadata_d, 16 * sizeof(uint)));
    CUDA_CHECK(cudaMemcpy(h->compressed_params_d, compressed_params,
                           15 * 16 * 16 * sizeof(TYPE), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(h->metadata_d, metadata, 16 * sizeof(uint),
                           cudaMemcpyHostToDevice));
    free(compressed_params);
    free(metadata);

    CUDA_CHECK(cudaMalloc(&h->array_d[0], array_elems * sizeof(TYPE)));
    CUDA_CHECK(cudaMalloc(&h->array_d[1], array_elems * sizeof(TYPE)));
    CUDA_CHECK(cudaMemcpy(h->array_d[0], h_in.data(), array_elems * sizeof(TYPE),
                           cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemset(h->array_d[1], 0, array_elems * sizeof(TYPE)));

    h->grid_config = dim3(CEIL(input_m, BLOCK_ROW), CEIL(input_n, BLOCK_COL));
    h->block_config = dim3(THREAD_PER_BLOCK);
    h->last_out_idx = 1;

    return h;
}

// `times` ping-pong launches of the artifact's own kernel_2d_7r, unmodified.
// array_d[0] is only ever read (never written) when times==1 -- the only
// value this adapter actually uses (see adapter.py on why T is forced to
// 1) -- so repeated calls on the same handle each recompute the identical
// one-sweep result from the identical pristine input, matching the
// harness's "every run() call must independently be correct" requirement
// without needing an explicit re-seed step.
void spider2d7r_run(void *handle, int times, cudaStream_t stream) {
    Spider2d7rHandle *h = reinterpret_cast<Spider2d7rHandle *>(handle);
    for (int i = 0; i < times; i++) {
        kernel_2d_7r<<<h->grid_config, h->block_config, 0, stream>>>(
            h->array_d[i % 2], h->compressed_params_d, h->metadata_d,
            h->array_d[(i + 1) % 2], h->cols);
    }
    h->last_out_idx = times % 2;  // see bug-fix note in this file's docstring
}

// D2H copy of the last-written buffer, un-padded back to input_m x input_n
// (host_out). A blocking cudaMemcpy on the default stream, so this also
// serves as the sync point after run()'s kernel launch(es).
void spider2d7r_copy_out(void *handle, uint16_t *host_out) {
    Spider2d7rHandle *h = reinterpret_cast<Spider2d7rHandle *>(handle);
    const size_t array_elems = (size_t)h->rows * (size_t)h->cols;
    std::vector<TYPE> h_full(array_elems);
    CUDA_CHECK(cudaMemcpy(h_full.data(), h->array_d[h->last_out_idx],
                           array_elems * sizeof(TYPE), cudaMemcpyDeviceToHost));
    TYPE *out = reinterpret_cast<TYPE *>(host_out);
    for (int i = 0; i < h->input_m; i++) {
        memcpy(out + (size_t)i * h->input_n,
               &h_full[(size_t)(i + HALO) * h->cols + HALO],
               h->input_n * sizeof(TYPE));
    }
}

void spider2d7r_free(void *handle) {
    Spider2d7rHandle *h = reinterpret_cast<Spider2d7rHandle *>(handle);
    if (!h) return;
    if (h->array_d[0]) cudaFree(h->array_d[0]);
    if (h->array_d[1]) cudaFree(h->array_d[1]);
    if (h->compressed_params_d) cudaFree(h->compressed_params_d);
    if (h->metadata_d) cudaFree(h->metadata_d);
    delete h;
}

}  // extern "C"
