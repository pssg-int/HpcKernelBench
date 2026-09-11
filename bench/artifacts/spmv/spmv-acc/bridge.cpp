// Thin ctypes-callable bridge around spmv-acc's own kernel-library entry
// point, sparse_csr_spmv() (source/src/acc/api/spmv.h), compiled with
// KERNEL_STRATEGY_ADAPTIVE (compat/building_config.h) -- the paper's
// headline auto-selecting strategy. See STATUS.md's "What was going to be
// wrapped" section for the full rationale; this bridge is exactly that
// plan, now that a compatible HIP toolchain was found on this machine
// (module hip/5.6.1/gcc/11.3.0/nompi/cuda/12.3.0/zen2, HIP_PLATFORM=nvidia
// -- see build.sh and STATUS.md's "Reproduction on zaratan" entry).
//
// NOT part of the artifact. sparse_csr_spmv() takes BOTH a host-resident
// csr_desc (h_csr_desc -- the adaptive strategy's own quartile/nnz-balance
// analysis, adaptive.cpp, indexes h_csr_desc.row_ptr directly on the host,
// e.g. `h_csr_desc.row_ptr[m / 4]`) and a device-resident csr_desc
// (d_csr_desc -- what the actual kernels read). This bridge keeps both:
// the host CSR arrays stay resident (for h_csr_desc) alongside a device
// copy (for d_csr_desc), exactly mirroring what the artifact's own CLI
// driver (not wrapped here, see STATUS.md) does before calling
// sparse_csr_spmv().
//
//   spmvacc_prepare() -- H2D-copies the CSR matrix + x once; keeps the
//                        host CSR arrays alive too (needed by h_csr_desc
//                        on every run() call, not just once -- the
//                        adaptive strategy re-reads them each call to
//                        pick a kernel, exactly as the artifact's own
//                        strategy_picker.cpp does per call, not as a
//                        one-time setup this bridge invents).
//   spmvacc_run()      -- re-zeros d_y (cheap insurance; with beta=0 the
//                        kernels' own `fma(beta, y[row], alpha*sum)` write
//                        already reduces to a pure overwrite for finite
//                        y, matching cb-spmv/tilespmv's bridges' same
//                        precautionary convention here), then ONE
//                        sparse_csr_spmv() call.
//
// No kernel/strategy code under source/ is modified: adaptive_sparse_spmv
// and everything it dispatches to (hip-flat/hip-line/hip-line-enhance/
// hip-vector-row/hip-thread-row) are compiled and called unmodified.

#include <hip/hip_runtime.h>
#include <hip/hip_runtime_api.h>

#include <cstring>
#include <vector>

#include "api/spmv.h"
#include "api/types.h"

struct SpmvAccHandle {
    int rows = 0, cols = 0, nnz = 0;
    // host-resident CSR (kept alive for h_csr_desc, read by the adaptive
    // strategy's own per-call quartile/nnz-balance analysis -- see
    // adaptive.cpp, not something this bridge invents).
    std::vector<int> h_row_ptr;
    std::vector<int> h_col_index;
    std::vector<double> h_values;
    // device-resident CSR + vectors, read by the actual kernels.
    int *d_row_ptr = nullptr;
    int *d_col_index = nullptr;
    double *d_values = nullptr;
    double *d_x = nullptr;
    double *d_y = nullptr;
};

extern "C" {

void *spmvacc_prepare(int rows, int cols, int nnz,
                      const int *row_ptr, const int *col_index, const double *values,
                      const double *x_in) {
    SpmvAccHandle *h = new SpmvAccHandle();
    h->rows = rows;
    h->cols = cols;
    h->nnz = nnz;
    h->h_row_ptr.assign(row_ptr, row_ptr + rows + 1);
    h->h_col_index.assign(col_index, col_index + nnz);
    h->h_values.assign(values, values + nnz);

    hipMalloc((void **)&h->d_row_ptr, (size_t)(rows + 1) * sizeof(int));
    hipMalloc((void **)&h->d_col_index, (size_t)nnz * sizeof(int));
    hipMalloc((void **)&h->d_values, (size_t)nnz * sizeof(double));
    hipMalloc((void **)&h->d_x, (size_t)cols * sizeof(double));
    hipMalloc((void **)&h->d_y, (size_t)rows * sizeof(double));

    hipMemcpy(h->d_row_ptr, row_ptr, (size_t)(rows + 1) * sizeof(int), hipMemcpyHostToDevice);
    hipMemcpy(h->d_col_index, col_index, (size_t)nnz * sizeof(int), hipMemcpyHostToDevice);
    hipMemcpy(h->d_values, values, (size_t)nnz * sizeof(double), hipMemcpyHostToDevice);
    hipMemcpy(h->d_x, x_in, (size_t)cols * sizeof(double), hipMemcpyHostToDevice);

    return h;
}

// One y = alpha*A*x + beta*y call (alpha=1, beta=0), dispatched by the
// artifact's own strategy_picker.cpp (compiled with
// KERNEL_STRATEGY_ADAPTIVE) to adaptive_sparse_spmv, unmodified.
void spmvacc_run(void *handle) {
    SpmvAccHandle *h = reinterpret_cast<SpmvAccHandle *>(handle);
    hipMemsetAsync(h->d_y, 0, (size_t)h->rows * sizeof(double), 0);

    csr_desc<int, double> h_desc(h->rows, h->cols, h->nnz,
                                 h->h_row_ptr.data(), h->h_col_index.data(), h->h_values.data());
    csr_desc<int, double> d_desc(h->rows, h->cols, h->nnz,
                                 h->d_row_ptr, h->d_col_index, h->d_values);
    sparse_csr_spmv(operation_none, 1.0, 0.0, h_desc, d_desc, h->d_x, h->d_y);
    hipDeviceSynchronize();
}

// D2H copy of y (rows elements); also serves as the sync point between
// run() and the correctness/gate check (hipDeviceSynchronize already ran
// in spmvacc_run(), this is a plain blocking copy on top of that).
void spmvacc_copy_y(void *handle, double *host_y_out) {
    SpmvAccHandle *h = reinterpret_cast<SpmvAccHandle *>(handle);
    hipMemcpy(host_y_out, h->d_y, (size_t)h->rows * sizeof(double), hipMemcpyDeviceToHost);
}

void spmvacc_free(void *handle) {
    SpmvAccHandle *h = reinterpret_cast<SpmvAccHandle *>(handle);
    if (!h) return;
    hipFree(h->d_row_ptr);
    hipFree(h->d_col_index);
    hipFree(h->d_values);
    hipFree(h->d_x);
    hipFree(h->d_y);
    delete h;
}

}  // extern "C"
