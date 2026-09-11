// Thin ctypes-callable bridge around this artifact's dense-vector SpMV
// kernel, HolaSpmv (source/hice-spmspv/hice/la/include/spmspv/csc-spmspv/
// detail/device/holaspmv.h: hola_pre<T> / hola_spmv<T>), per the
// integration brief: wrap the dense-vector SpMV path only -- the paper's
// headline ML-based adaptive SpMV/SpMSpV *selector* (hice/ml/, the
// decision-tree/SVM/GBDT/random-forest training pipeline in hice-spmspv's
// README) picks among 8 candidate kernels, of which HolaSpmv (from the
// same research group's own prior GPU-SpMV work) is the dense-input-vector
// member of that set; SpMSpV (sparse x) is a different track and is not
// wrapped here (hola_spmspv, same header, is left untouched).
//
// NOT part of the artifact. It exists because hola_spmv/hola_pre are
// templates with no extern "C" boundary and no standalone driver at all in
// this repo (the artifact's own scripts, hice/la/script/, only exercise
// the full adaptive framework via its CMake/CLI harness, which the
// integration brief's "wrap the kernel, not the paper's benchmark
// script"/finest-boundary rule says to bypass). This file explicitly
// instantiates both functions at double precision (the repo's own bottom
// of holaspmv.h already explicitly instantiates float AND double, so this
// is exactly what the artifact itself ships as its public surface, not an
// extra instantiation we invented) and exposes them through a tiny
// extern "C" pair matching this repo's prepare()/run() split:
//
//   holaspmv_prepare() -- H2D-copies the CSR matrix and x once, calls the
//                         artifact's own hola_pre<double>() to size the
//                         (small, O(nnz/blockSize) -- NOT O(nnz)) scratch
//                         buffer HolaSpmv needs, and allocates it. This is
//                         the artifact's own prepare/size step -- timed
//                         once, as preprocessing.
//   holaspmv_run()      -- ONE call to hola_spmv<double>(). Note this
//                         function's FIRST internal action is its own
//                         DetermineBlockStarts kernel, which both computes
//                         per-block row-start bookkeeping into the scratch
//                         buffer AND zeroes the output vector `res`
//                         (holaspmv.h: `outvec[id] = 0;`) -- i.e. the
//                         artifact's own design recomputes this
//                         bookkeeping and zeroes y on EVERY call, not just
//                         once, so no separate cudaMemset of y is added
//                         here; doing so would only duplicate work the
//                         kernel already does as part of what gets timed.
//
// No kernel code is modified: hola_pre/hola_spmv are included verbatim.

// holaspmv.h is written to be included transitively, after the artifact's
// own spmspv/config.h and spmspv/csc-spmspv/detail/util.h have already run
// (via spmspv/csc-spmspv/spmspv.h -- the SpMSpV path this integration does
// NOT wrap, per the brief). Included standalone, two names those headers
// would otherwise have supplied are genuinely undefined at compile time
// (nvcc: "identifier 'LM_WARP_SIZE'/'divup' is undefined") -- not missing
// from holaspmv.h itself, just from skipping the unrelated SpMSpV include
// chain around it. Both are trivial, content-free definitions (a warp-size
// constant and a ceiling-division template identical in effect to this
// same file's own `hola_divup`, hola_common.h) -- no kernel/algorithm code
// is being supplied here, just the two declarations bypassing the SpMSpV
// chain would have provided for free.
#define LM_WARP_SIZE 32  // matches spmspv/config.h's own definition
template <typename T>
__host__ __device__ __forceinline__ T divup(T a, T b) {
    return (a + b - 1) / b;
}

#include "holaspmv.h"

#include <cstdlib>

struct HolaSpmvHandle {
    int rows = 0, cols = 0, nnz = 0;
    int *d_csr_row = nullptr;
    int *d_csr_col = nullptr;
    double *d_csr_val = nullptr;
    double *d_x = nullptr;
    double *d_y = nullptr;
    void *d_tempmem = nullptr;
    size_t tempmemsize = 0;
};

extern "C" {

// H2D-copies the CSR matrix + x, sizes and allocates HolaSpmv's own
// scratch buffer via the artifact's own hola_pre<double>(). All one-time
// preprocessing under this variant's protocol.
void *holaspmv_prepare(int rows, int cols, int nnz,
                        const int *csr_row, const int *csr_col, const double *csr_val,
                        const double *x_in) {
    HolaSpmvHandle *h = new HolaSpmvHandle();
    h->rows = rows;
    h->cols = cols;
    h->nnz = nnz;

    cudaMalloc((void **)&h->d_csr_row, (size_t)(rows + 1) * sizeof(int));
    cudaMalloc((void **)&h->d_csr_col, (size_t)nnz * sizeof(int));
    cudaMalloc((void **)&h->d_csr_val, (size_t)nnz * sizeof(double));
    cudaMalloc((void **)&h->d_x, (size_t)cols * sizeof(double));
    cudaMalloc((void **)&h->d_y, (size_t)rows * sizeof(double));

    cudaMemcpy(h->d_csr_row, csr_row, (size_t)(rows + 1) * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(h->d_csr_col, csr_col, (size_t)nnz * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(h->d_csr_val, csr_val, (size_t)nnz * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(h->d_x, x_in, (size_t)cols * sizeof(double), cudaMemcpyHostToDevice);

    // hola_pre only reads m_rows/m_cols/m_nnz/v_size (structural sizing --
    // the artifact's own signature also takes v_data but never
    // dereferences it for this computation, see holaspmv.h); h->d_x is
    // passed through as-is per the artifact's own signature.
    hola_pre<double>(h->tempmemsize, rows, cols, nnz, cols, h->d_x);
    cudaMalloc((void **)&h->d_tempmem, h->tempmemsize);

    return h;
}

// ONE y=Ax call: the artifact's own hola_spmv<double>() -- its own first
// internal kernel (DetermineBlockStarts) zeroes d_y as part of computing
// this call's block-start bookkeeping (see file docstring), so no
// separate zero-fill is added here.
void holaspmv_run(void *handle) {
    HolaSpmvHandle *h = reinterpret_cast<HolaSpmvHandle *>(handle);
    hola_spmv<double>(h->d_tempmem, h->tempmemsize, h->rows, h->cols, h->nnz,
                      h->d_csr_row, h->d_csr_col, h->d_csr_val,
                      h->cols, h->d_x, h->d_y, /*padded=*/false);
}

// D2H copy of y (rows elements); a plain cudaMemcpy on the default stream
// blocks until any prior kernel on that stream completes, so this also
// serves as the sync point between run() and the correctness/gate check.
void holaspmv_copy_y(void *handle, double *host_y_out) {
    HolaSpmvHandle *h = reinterpret_cast<HolaSpmvHandle *>(handle);
    cudaMemcpy(host_y_out, h->d_y, (size_t)h->rows * sizeof(double), cudaMemcpyDeviceToHost);
}

void holaspmv_free(void *handle) {
    HolaSpmvHandle *h = reinterpret_cast<HolaSpmvHandle *>(handle);
    if (!h) return;
    cudaFree(h->d_csr_row);
    cudaFree(h->d_csr_col);
    cudaFree(h->d_csr_val);
    cudaFree(h->d_x);
    cudaFree(h->d_y);
    cudaFree(h->d_tempmem);
    delete h;
}

}  // extern "C"
