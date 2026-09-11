// wrapper.cu -- ours (ARTIFACT_GUIDE.md rule 3: files we write live outside
// source/), NOT part of SMaT's own source/ clone.
//
// SMaT's own "hgemm" CLI binary (source/src/cuda_hgemm/src/main.cu) is a
// gflags-driven benchmark SCRIPT: it parses ~15 CLI flags, builds ONE
// `SparseMatrix` (source/.../common/matrix.h -- reads a .mtx file, builds
// CSR on the host, then converts CSR -> block-CSR (BCSR) of MMA_M x MMA_K
// = 16x16 tiles -- the paper's own real preprocessing, "sparse matrix
// permutation" bookkeeping included), then times a fixed kernel
// (`mmaCBTKernel`, source/.../src/mma/mmaCBT.cu -- the file's own comment
// says "Mma-CBT-Kernel", the only one main.cu actually calls; the others
// are commented out) over a warmup+profiling loop with `CudaTimer`
// (ARTIFACT_GUIDE.md rule 1: wrap the kernel, not this script).
//
// `SparseMatrix` (matrix.h) is header-only (every method is defined inline
// in the class body) and has ZERO dependency on gflags/OpenMP/cublas -- only
// `main.cu` (CLI flag parsing) and `cublas_tensor_op` (a dense cuBLAS
// baseline, also in main.cu's translation unit, unrelated to the sparse
// kernel) need those. That means this wrapper can link `matrix.h`'s
// SparseMatrix + `mmaCBT.cu`'s real, unmodified kernel WITHOUT SMaT's own
// CMake build (and therefore without gflags at all) -- confirmed by
// grepping every #include reachable from matrix.h (see STATUS.md): none of
// them pull in <gflags/gflags.h>. The "vendor gflags" build-system fix
// ARTIFACT_GUIDE.md anticipated turned out to be unnecessary once wrapping
// at this finer boundary (same principle as rode/mp-spmm: wrap the kernel's
// own dependency closure, not the whole CLI driver's).
//
// `mmaCBTKernelSparse` (the actual __global__, in mmaCBT.cu) is compiled
// completely unmodified and linked in as a separate translation unit;
// `mmaCBTKernel` (its host launcher, also unmodified) is declared here via
// `extern` and called directly -- exactly SMaT's own
// `tester.h::evaluateSparse2`'s call convention
// (bcsrValuesA, bcsrRowPtrA, bcsrColIdxA, B, C, M, N, K, nonzeroBlocks,
// blockInfo, relativeBlockIndexMapping).

#include "matrix.h"

extern void mmaCBTKernel(half *bcsrValuesA, int *bcsrRowPtrA, int *bcsrColIdxA, half *B, half *C,
                         size_t M, size_t N, size_t K, size_t nonzeroBlocks, int *blockInfo,
                         int *relativeBlockIndexMapping);

extern "C" {

// SparseMatrix's own constructor does ALL of SMaT's real preprocessing
// (CSR read from the .mtx file + CSR->BCSR block conversion + H2D upload) --
// called once from adapter.py::prepare(), timed as preprocessing per
// ARTIFACT_GUIDE.md rule 2.
void *smat_prepare(const char *mtx_path) {
    // SparseMatrix's constructor takes char* (non-const) -- copy, since we
    // are handed a Python-owned const string via ctypes.
    std::string path(mtx_path);
    std::vector<char> buf(path.begin(), path.end());
    buf.push_back('\0');
    return static_cast<void *>(new SparseMatrix("A", buf.data()));
}

unsigned long smat_get_row(void *h) { return static_cast<SparseMatrix *>(h)->getRow(); }
unsigned long smat_get_col(void *h) { return static_cast<SparseMatrix *>(h)->getCol(); }
unsigned long smat_get_nnz(void *h) { return static_cast<SparseMatrix *>(h)->getNnz(); }
unsigned long smat_get_nonzero_blocks(void *h) {
    return static_cast<SparseMatrix *>(h)->getNonzeroblocks();
}

// ONE call = ONE mmaCBTKernel launch (rule 1). B/C are harness-owned CUDA
// device pointers (torch tensors' .data_ptr(), same convention as
// rode/insum/mp-spmm) -- NOT SMaT's own Matrix-class-generated operands.
void smat_run(void *h, void *B, void *C, unsigned long N) {
    SparseMatrix *A = static_cast<SparseMatrix *>(h);
    mmaCBTKernel(A->getBcsrValues(), A->getBcsrRowPtr(), A->getBcsrColIdx(),
                static_cast<half *>(B), static_cast<half *>(C),
                A->getRow(), N, A->getCol(),
                A->getNonzeroblocks(), A->getBlockInfo_dev(),
                A->getRelativeBlockIndexMapping_dev());
}

void smat_free(void *h) { delete static_cast<SparseMatrix *>(h); }

}  // extern "C"
