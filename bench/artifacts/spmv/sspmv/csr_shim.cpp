// Thin ctypes-callable wrapper around SSpMV's (LeSpMV library) own CSR SpMV
// kernel: LeSpMV_csr<int, double>, defined in
// source/LeSpMV/src/spmv_csr.cpp (dispatches to __spmv_csr_omp_simple for
// kernel_flag=1, the library's default OpenMP-parallel CSR kernel).
//
// This file is NOT part of the artifact -- LeSpMV_csr is a C++ template
// function with no extern "C" / FFI entry point, so this shim exists purely
// to make it callable from ctypes, exactly the way pybind11 would (just
// without the pybind11 dependency). It calls the artifact's kernel
// unmodified; no kernel code is touched.
#include "LeSpMV.h"

extern "C" {

// y = A @ x, A given as CSR (int32 indices, fp64 values). Building the
// CSR_Matrix<int,double> aggregate IS the artifact's own "format
// construction" step (trivial for CSR since the harness's Matrix is
// already CSR -- this just wraps the three raw arrays plus metadata the
// artifact's struct expects) and belongs in the adapter's prepare(), which
// is where the Python side calls this at most once per benchmarked matrix.
void sspmv_csr_f64(int num_rows, int num_cols, long long num_nnzs,
                    const int* row_offset, const int* col_index,
                    const double* values, const double* x, double* y) {
    CSR_Matrix<int, double> csr;
    csr.num_rows = num_rows;
    csr.num_cols = num_cols;
    csr.num_nnzs = static_cast<int>(num_nnzs);
    csr.sparsity = 0.0;
    csr.partition = nullptr;
    csr.time = 0.0;
    csr.gflops = 0.0;
    csr.gbytes = 0.0;
    csr.tag = 0;
    csr.kernel_flag = 1;  // __spmv_csr_omp_simple: the library's default OpenMP CSR kernel
    csr.row_offset = const_cast<int*>(row_offset);
    csr.col_index = const_cast<int*>(col_index);
    csr.values = const_cast<double*>(values);

    LeSpMV_csr<int, double>(1.0, csr, x, 0.0, y);
}

}  // extern "C"
