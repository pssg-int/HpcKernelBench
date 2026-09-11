/*
 * Thin extern-"C" wrapper exposing AmgT's own standalone SpGEMM kernel entry
 * point directly:
 *
 *   void spgemm_amgT_fp64(hypre_CSRMatrix *A, hypre_CSRMatrix *B,
 *                          hypre_CSRMatrix **C_ptr);
 *
 * defined, unconditionally (not behind any #ifdef), at
 * source/AmgT_HYPRE/src/seq_mv/csr_spgemm_device.c:1527-1846. It takes two
 * arbitrary hypre_CSRMatrix* (a generic HYPRE sparse-matrix struct, not tied
 * to any AMG level/hierarchy) and computes *C_ptr = A @ B via AmgT's own
 * mBSR tensor-core-friendly block format -- genuinely separable from the AMG
 * solver: no AMG setup/solve runs here, only this one SpGEMM kernel call.
 * This file lives OUTSIDE source/ and touches zero lines of the artifact; it
 * only re-declares two of the artifact's own C functions that have no public
 * header declaration (CSR2BSR_GPU, spgemm_amgT_fp64 -- confirmed absent from
 * every installed header via grep) with IDENTICAL signatures so the linker
 * resolves them to the symbols already compiled into libHYPRE.a, plus adds
 * extern-"C" entry points that only marshal buffers/pointers. No artifact
 * logic is reimplemented anywhere in this file.
 *
 * Split into functions matching the kernelbench Implementation contract:
 *
 *   amgt_init()               -- MPI_Init (once) + HYPRE_Init + cudaSetDevice(0),
 *                                 the same ordering AmgT_test/test_new.c's own
 *                                 main() uses (HYPRE_Initialize() itself is
 *                                 already idempotent -- see utilities/general.c's
 *                                 `if (hypre_Initialized()) return;` -- so this
 *                                 is safe to call once at adapter-import time).
 *   amgt_build_and_convert()  -- H2D-copies a host CSR (row_ptr/col_idx/values)
 *                                 into a device hypre_CSRMatrix, then calls
 *                                 AmgT's own CSR2BSR_GPU() ONCE (its own
 *                                 idempotency guard, `if (!hypre_BSRTAG(A))` at
 *                                 csr_matvec_device.c:1200, makes any LATER
 *                                 CSR2BSR_GPU call on the same pointer a
 *                                 no-op) -- this whole function IS prepare()'s
 *                                 format-conversion step (ARTIFACT_GUIDE.md
 *                                 rule 2), called once, timed as preprocessing
 *                                 by the harness's own wall-clock wrapper
 *                                 around impl.prepare(), NOT by this file.
 *   amgt_spgemm_run()         -- ONE spgemm_amgT_fp64(A, B, &C) call. Because
 *                                 CSR2BSR_GPU already ran in
 *                                 amgt_build_and_convert(), the function's own
 *                                 internal `CSR2BSR_GPU(A); CSR2BSR_GPU(B);`
 *                                 calls (csr_spgemm_device.c:1531-1532) become
 *                                 free no-ops here, so the timed window this
 *                                 function spans is exactly symbolic pass +
 *                                 numeric_spgemm_hybrid kernel + C's device
 *                                 allocation + the BSR2CSR-back-to-CSR pass
 *                                 that produces *C_ptr -- matching
 *                                 spgemm-square-kernel-f64's timing_scope
 *                                 (benchspecs/spgemm/spec.yaml).
 *   amgt_csr_sizes/copy_out() -- read C's shape/nnz and D2H-copy its CSR
 *                                 arrays into caller-provided (numpy) host
 *                                 buffers -- to_host(), gate-check only, never
 *                                 timed.
 *   amgt_csr_destroy()        -- calls the artifact's OWN
 *                                 hypre_CSRMatrixDestroy(), unmodified. Read
 *                                 directly (source/AmgT_HYPRE/src/seq_mv/
 *                                 csr_matrix.c): it already frees i/j/data AND
 *                                 -- via its own
 *                                 `if (hypre_BSRTAG(matrix) == 1) { cudaFree
 *                                 (bsr_mat->blcPtr/blcIdx/blcVal/blcMap);
 *                                 free(bsr_mat); }` branch -- the attached
 *                                 mBSR side-structure CSR2BSR_GPU/
 *                                 spgemm_amgT_fp64 build. No separate
 *                                 BSR-freeing call is needed in this wrapper.
 *                                 hypre_TFree(..., HYPRE_MEMORY_DEVICE) inside
 *                                 it resolves to plain cudaFree in this build
 *                                 (confirmed: HYPRE_config.h has neither
 *                                 HYPRE_USING_UMPIRE_DEVICE nor
 *                                 HYPRE_USING_DEVICE_POOL defined, so
 *                                 hypre_DeviceMalloc/Free in utilities/
 *                                 memory.c fall through to plain
 *                                 cudaMalloc/cudaFree) -- compatible with the
 *                                 raw cudaMalloc'd buffers amgt_build_and_
 *                                 convert() hands to A/B below.
 *
 * Aliasing A == B for the self-product C = A @ A (this track's `operation`
 * field, benchspecs/spgemm/spec.yaml): verified safe by reading
 * spgemm_amgT_fp64's body directly -- `dmatA = *hypre_BSR(A); dmatB =
 * *hypre_BSR(B);` are local by-VALUE copies of the bsrMAT struct (itself just
 * a handful of device pointers), taken once at function entry and never
 * written back into *A or *B anywhere in the function; only C's own
 * freshly-allocated buffers are written. So the adapter builds ONE device
 * hypre_CSRMatrix and passes it as both A and B -- CSR2BSR_GPU only needs to
 * run once anyway (it would no-op on the second call regardless, since
 * hypre_BSRTAG would already be 1 on the same object).
 *
 * Stream: every AmgT kernel launch in csr_spgemm_device.c/csr_matvec_device.c
 * omits an explicit stream argument (confirmed by grep -- no
 * cudaStreamCreate/cudaStream_t anywhere in either file's CUDA code path),
 * i.e. everything runs on the default stream (stream 0) -- the same stream
 * torch's CudaEventTimer records its start/stop events on, so device-event
 * timing across adapter.py's run() call is valid with no cross-stream gap.
 */

#include "HYPRE.h"
#include "seq_mv.h"

#include <cuda_runtime.h>
#include <mpi.h>
#include <cstdio>
#include <cstdlib>

// Re-declarations only (see file header) -- resolve to the symbols already
// compiled into libHYPRE.a from source/AmgT_HYPRE/src/seq_mv/
// csr_matvec_device.c and csr_spgemm_device.c. Neither is declared in any
// installed HYPRE header (grep -rn over hypre/include confirms this), since
// they are AmgT's own additions, not part of upstream HYPRE's public API.
extern "C" {
void CSR2BSR_GPU(hypre_CSRMatrix *A);
void spgemm_amgT_fp64(hypre_CSRMatrix *A, hypre_CSRMatrix *B, hypre_CSRMatrix **C_ptr);
}

extern "C" {

// MPI_Init + HYPRE_Init + device selection, matching AmgT_test/test_new.c's
// own main() ordering (cudaSetDevice(0); ... MPI_Init(&argc,&argv);
// HYPRE_Init();). Guards MPI_Init with MPI_Initialized() (MPI_Init itself is
// NOT safe to call twice); HYPRE_Initialize() already guards itself
// internally. Returns 0 on success.
int amgt_init(void) {
    cudaError_t cerr = cudaSetDevice(0);
    if (cerr != cudaSuccess) {
        return (int)cerr + 1000;  // disambiguate from MPI error codes
    }
    int mpi_inited = 0;
    MPI_Initialized(&mpi_inited);
    if (!mpi_inited) {
        int rc = MPI_Init(NULL, NULL);
        if (rc != MPI_SUCCESS) {
            return rc;
        }
    }
    HYPRE_Init();
    return 0;
}

// Host CSR (row_ptr: nrows+1 int32, col_idx: nnz int32, vals: nnz float64)
// -> device hypre_CSRMatrix, converted ONCE to AmgT's mBSR format. This
// whole call is prepare()'s format-conversion step (see file header).
// HYPRE_Int == int and HYPRE_Complex == double in this build (confirmed:
// HYPRE_config.h has neither HYPRE_BIGINT nor HYPRE_COMPLEX/HYPRE_SINGLE/
// HYPRE_LONG_DOUBLE defined, so utilities/HYPRE_utilities.h's `#else
// /* default */` branches apply: HYPRE_Int=int, HYPRE_Real=double,
// HYPRE_Complex=HYPRE_Real=double), so host int32/float64 numpy buffers can
// be H2D-copied directly with no width conversion.
hypre_CSRMatrix *amgt_build_and_convert(int nrows, int ncols, int nnz,
                                        const int *h_rowptr, const int *h_colidx,
                                        const double *h_vals) {
    hypre_CSRMatrix *A = hypre_CSRMatrixCreate(nrows, ncols, nnz);
    hypre_CSRMatrixMemoryLocation(A) = HYPRE_MEMORY_DEVICE;

    int *d_rowptr = NULL, *d_colidx = NULL;
    double *d_vals = NULL;
    cudaMalloc((void **)&d_rowptr, sizeof(int) * (size_t)(nrows + 1));
    cudaMalloc((void **)&d_colidx, sizeof(int) * (size_t)nnz);
    cudaMalloc((void **)&d_vals, sizeof(double) * (size_t)nnz);
    cudaMemcpy(d_rowptr, h_rowptr, sizeof(int) * (size_t)(nrows + 1), cudaMemcpyHostToDevice);
    cudaMemcpy(d_colidx, h_colidx, sizeof(int) * (size_t)nnz, cudaMemcpyHostToDevice);
    cudaMemcpy(d_vals, h_vals, sizeof(double) * (size_t)nnz, cudaMemcpyHostToDevice);

    hypre_CSRMatrixI(A) = d_rowptr;
    hypre_CSRMatrixJ(A) = d_colidx;
    hypre_CSRMatrixData(A) = d_vals;

    CSR2BSR_GPU(A);  // artifact's own CSR->mBSR conversion, unmodified
    cudaDeviceSynchronize();
    return A;
}

// ONE SpGEMM kernel call: C = A @ B (A and B may be the SAME pointer -- see
// file header's aliasing note). Returns the freshly-allocated C, or NULL if
// a CUDA error occurred (checked via cudaGetLastError(), not part of the
// artifact's own error handling -- spgemm_amgT_fp64 itself has no return
// value / error path).
hypre_CSRMatrix *amgt_spgemm_run(hypre_CSRMatrix *A, hypre_CSRMatrix *B) {
    hypre_CSRMatrix *C = NULL;
    spgemm_amgT_fp64(A, B, &C);
    cudaError_t err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        fprintf(stderr, "[amgt wrapper] spgemm_amgT_fp64 CUDA error: %s\n",
                cudaGetErrorString(err));
        return NULL;
    }
    return C;
}

void amgt_csr_sizes(hypre_CSRMatrix *M, int *nrows, int *ncols, int *nnz) {
    *nrows = hypre_CSRMatrixNumRows(M);
    *ncols = hypre_CSRMatrixNumCols(M);
    *nnz = hypre_CSRMatrixNumNonzeros(M);
}

// D2H copy of M's CSR arrays into caller-provided (numpy-backed) host
// buffers, sized per amgt_csr_sizes() above -- to_host(), gate-check only.
void amgt_csr_copy_out(hypre_CSRMatrix *M, int *h_rowptr, int *h_colidx, double *h_vals) {
    int nrows = hypre_CSRMatrixNumRows(M);
    int nnz = hypre_CSRMatrixNumNonzeros(M);
    cudaMemcpy(h_rowptr, hypre_CSRMatrixI(M), sizeof(int) * (size_t)(nrows + 1),
              cudaMemcpyDeviceToHost);
    cudaMemcpy(h_colidx, hypre_CSRMatrixJ(M), sizeof(int) * (size_t)nnz,
              cudaMemcpyDeviceToHost);
    cudaMemcpy(h_vals, hypre_CSRMatrixData(M), sizeof(double) * (size_t)nnz,
              cudaMemcpyDeviceToHost);
}

// Frees a hypre_CSRMatrix built by amgt_build_and_convert() (A/B) or
// returned by amgt_spgemm_run() (C) -- see file header for why
// hypre_CSRMatrixDestroy (the artifact's own, unmodified) is sufficient on
// its own, no extra BSR-side freeing needed here.
void amgt_csr_destroy(hypre_CSRMatrix *M) {
    if (M) {
        hypre_CSRMatrixDestroy(M);
    }
}

}  // extern "C"
