// TileSpGEMM bridge (this directory, NOT part of the artifact).
//
// Paper: "TileSpGEMM: A Tiled Algorithm for Parallel Sparse General
// Matrix-Matrix Multiplication on GPUs" (PPoPP'22, conf/ppopp/NiuLJS0022).
// Artifact: https://github.com/SuperScientificSoftwareLaboratory/TileSpGEMM
//
// The artifact ships only an end-to-end binary (source/src/main.cu) that
// reads a .mtx file itself, overwrites A's values with a synthetic i%10
// pattern (main.cu:100-101), builds the tiled format, and calls
// tilespgemm() -- the ACTUAL kernel, source/src/tilespgemm-cuda.h -- once,
// inside an `#ifdef DEBUG` block that main.cu's own common.h already
// enables unconditionally (common.h:67 `#define DEBUG 1`, no `-D` flag
// needed; likewise TIMING/SPACE/CHECK_RESULT). Per ARTIFACT_GUIDE.md rule 1
// ("wrap the kernel, not the paper's benchmark script"), this bridge calls
// tilespgemm() directly from our own CSR arrays -- no file I/O, no
// synthetic value overwrite (real workload values are used, required for
// a real correctness gate).
//
// Boundary split:
//   tsg_prepare() -- builds the artifact's own SMatrix A/B directly from
//                    our CSR (host malloc + memcpy, no mmio_allinone file
//                    parsing), then calls the artifact's OWN
//                    csr2tile_row_major(A) / csr2tile_col_major(B)
//                    (source/src/csr2tile.h, unmodified) -- this IS the
//                    artifact's CSR->tile format conversion, which
//                    spgemm-square-kernel-f64's spec explicitly excludes
//                    from the timed window ("A/B mtx->CSR parsing and any
//                    algorithm-internal one-time restructuring of A/B into
//                    a persistent auxiliary layout ... excluded and
//                    reported once"). Also reproduces main.cu's own
//                    (untimed, host-side) intersection-bitmask
//                    construction (main.cu:182-217) and nnzCub count
//                    (main.cu:143-151) verbatim, since tilespgemm()'s call
//                    signature requires both and neither appears in any
//                    of TileSpGEMM's own published timing figures.
//   tsg_run()     -- ONE call to tilespgemm() (source/src/
//                    tilespgemm-cuda.h, unmodified) -- the artifact's own
//                    kernel function, REPEAT_NUM==1 (common.h:69, the
//                    artifact's OWN shipped default -- see
//                    benchspecs/spgemm/spec.yaml's notes_on_fairness: "The
//                    artifact ships with REPEAT_NUM=1"), so one call
//                    performs exactly one symbolic (step1) + numeric
//                    column/value (step2/step3) pass and returns.
//   tsg_finalize_csr() / tsg_to_host() -- tile2csr() (source/src/
//                    tile2csr.h, unmodified) converts the tiled C back to
//                    plain CSR on the host; called once, from Python
//                    to_host(), never inside the timed run() path.
//
// Disclosed timing contamination (documented, not patched -- same posture
// as ../ocean/adapter.py's Workspace-construction note): tilespgemm()
// mallocs+H2D-copies matrixA/matrixB's ALREADY-TILED host arrays into
// fresh device buffers, and D2H-copies the tiled C result back, ALL INSIDE
// the same call, then frees every device buffer it allocated before
// returning (verified by reading tilespgemm-cuda.h's own tail: it calls
// cudaFree on every d_* buffer, including d_blkrowptrA/B, d_blkcsr_*_A/B,
// d_blk_intersec_bitmask_A/B, for A and B, not only C). There is no
// separate, already-idempotent entry point that does the H2D setup once
// and the compute repeatedly -- that split does not exist anywhere in
// tilespgemm-cuda.h's ~600-line monolithic function, so hoisting the H2D
// transfer of A/B into tsg_prepare() would require restructuring the
// artifact's own control flow (a kernel-code change, forbidden by
// ARTIFACT_GUIDE.md rule 3). This bridge's measured per-call time is
// therefore a disclosed UPPER BOUND on spgemm-square-kernel-f64's
// timing_scope (symbolic+numeric+C-allocation only; A/B's persistent
// tile-structure H2D transfer is supposed to be excluded, "operands
// resident on device before timing starts") -- not an exact match, exactly
// like Ocean's disclosed Workspace-construction overhead.
//
// Zero lines of source/ were modified.

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>

// --- CUDA-version compat: force the SYSTEM cuda_fp16.h first (NOT part of
// the artifact) ---------------------------------------------------------
// source/src/ vendors its own cuda_fp16.h/cuda_fp16.hpp (CUDA ~11-era copy,
// no __nv_bfloat16 interop), included via a plain quoted
// `#include "cuda_fp16.h"` inside common.h:15. Under nvcc 12.9, libcu++
// headers thrust/utils_cuda_scan.h pulls in transitively (cuda/std/
// __cmath/traits.h, nvfp16.h) expect the CURRENT cuda_fp16.h/cuda_bf16.h
// pair and fail to compile against the vendored one ("no instance of
// overloaded function __half::__half matches ... const __nv_bfloat16",
// "calling a __device__ function ... from a __host__ __device__ function
// is not allowed" for __hisnan/__hisinf/__habs/__hmax). Both the vendored
// and the system cuda_fp16.h use the identical include-guard macro
// (__CUDA_FP16_H__, unchanged since CUDA 11) -- so including the SYSTEM
// header here, via an angle-bracket include resolved through -I ahead of
// source/src/, makes source/src/'s later quoted include a guarded no-op.
// This is a build-system/include-path fix (ARTIFACT_GUIDE.md rule 3, "CUDA-
// version guards"): no kernel arithmetic changes, and this codebase's fp64
// SpGEMM path never actually uses half precision -- the vendored header
// was dead weight even for the artifact's own intended build.
#include <cuda_fp16.h>
#include <cuda_bf16.h>

// --- CUDA-version compat shim (NOT part of the artifact) --------------
// source/src/spgemm_nsparse_kernel.h:858-859 call the pre-Volta unmasked
// __shfl() intrinsic on `int` and on `real` (== double, nsparse_asm.h's own
// #define) to broadcast a lane's (column, value) pair to the rest of the
// warp. CUDA 12.9 no longer declares unmasked __shfl at all targeting
// sm_70+ ("identifier '__shfl' is undefined") -- only the _sync forms
// exist. Same fix already used by artifacts/spmv/tilespmv/compat/
// shfl_compat.h: a templated shim forwarding to __shfl_sync with a
// full-warp mask (0xffffffffu), which reproduces the old implicit-all-
// lanes semantics exactly (both call sites here execute inside a loop
// every active lane of the warp reaches identically -- no divergent early
// exit before the shuffle). Must be defined BEFORE including the
// artifact's header so its two call sites resolve against it.
template <typename T>
__device__ __forceinline__ T __shfl(T var, int srcLane, int width = 32) {
    return __shfl_sync(0xffffffffu, var, srcLane, width);
}

#include "source/src/common.h"
#include "source/src/utils.h"
#include "source/src/utils_cuda_scan.h"
#include "source/src/spgemm_nsparse_kernel.h"
#include "source/src/csr2tile.h"
#include "source/src/tilespgemm-cuda.h"
#include "source/src/tile2csr.h"

extern "C" {

struct TsgHandle {
    SMatrix A;
    SMatrix B;
    SMatrix C;
    unsigned int *bitmaskA;
    unsigned int *bitmaskB;
    int bitmask_len;
    double densityA, densityB;
    unsigned long long nnzCub;
    bool have_c;
};

// matrixC's rowpointer/columnindex/value are set only by tile2csr() (once,
// from to_host()), separately from its tile_* fields (set by every
// tilespgemm() call) -- matrix_destroy() (source/src/csr2tile.h, the
// artifact's own cleanup helper, same one main.cu itself calls on A/B)
// frees only the tile_* side, so the CSR-side fields are freed here
// explicitly. free(NULL) is a no-op, so this is safe to call on a
// zero-initialized or partially-populated C.
static void free_c(SMatrix *c) {
    matrix_destroy(c);
    free(c->rowpointer);
    free(c->columnindex);
    free(c->value);
    c->rowpointer = nullptr;
    c->columnindex = nullptr;
    c->value = nullptr;
}

void *tsg_prepare(int n, long long nnz, const int *rowptr, const int *colidx,
                   const double *vals) {
    TsgHandle *h = new TsgHandle();
    memset(&h->A, 0, sizeof(SMatrix));
    memset(&h->B, 0, sizeof(SMatrix));
    memset(&h->C, 0, sizeof(SMatrix));
    h->have_c = false;

    // Host CSR built once, ALIASED by both A and B -- exactly main.cu's own
    // "-aat 0" (C = A^2, self-product) path (main.cu:132-140:
    // `matrixB->rowpointer = matrixA->rowpointer;` etc, a literal pointer
    // alias, not a copy). Verified safe by reading csr2tile_row_major/
    // csr2tile_col_major: both only READ rowpointer/columnindex/value
    // (csr2tile_col_major's matrix_transposition() call copies them into
    // its OWN cscColPtrB/cscRowIdxB/cscValB buffers before doing anything
    // else) and write exclusively into each SMatrix's own tile_* fields --
    // A's row-major pass and B's column-major pass never race or corrupt
    // the shared input.
    h->A.m = h->A.n = n;
    h->A.nnz = (int)nnz;
    h->A.isSymmetric = 0;
    h->A.rowpointer = (int *)malloc((size_t)(n + 1) * sizeof(int));
    h->A.columnindex = (int *)malloc((size_t)nnz * sizeof(int));
    h->A.value = (double *)malloc((size_t)nnz * sizeof(double));
    memcpy(h->A.rowpointer, rowptr, (size_t)(n + 1) * sizeof(int));
    memcpy(h->A.columnindex, colidx, (size_t)nnz * sizeof(int));
    memcpy(h->A.value, vals, (size_t)nnz * sizeof(double));

    h->B.m = h->B.n = n;
    h->B.nnz = (int)nnz;
    h->B.isSymmetric = 0;
    h->B.rowpointer = h->A.rowpointer;
    h->B.columnindex = h->A.columnindex;
    h->B.value = h->A.value;

    // Artifact's own CSR->tile conversion (source/src/csr2tile.h,
    // unmodified) -- the preprocessing spgemm-square-kernel-f64 excludes
    // from the timed window, hoisted here per ARTIFACT_GUIDE.md rule 2.
    csr2tile_row_major(&h->A);
    csr2tile_col_major(&h->B);

    // Intersection bitmasks: reproduced verbatim from main.cu's own
    // (untimed, host-side) construction loop (main.cu:182-217). Required
    // by tilespgemm()'s call signature; appears in no published TileSpGEMM
    // timing figure, so belongs in prepare() alongside the tile
    // conversion, not in the timed run().
    h->bitmask_len = (int)ceil((double)h->A.tilen / 32.0);
    long long lenA = (long long)h->A.tilem * h->bitmask_len;
    long long lenB = (long long)h->B.tilen * h->bitmask_len;
    h->bitmaskA = (unsigned int *)calloc((size_t)lenA, sizeof(unsigned int));
    h->bitmaskB = (unsigned int *)calloc((size_t)lenB, sizeof(unsigned int));
    for (int i = 0; i < h->A.tilem; i++) {
        for (int j = h->A.tile_ptr[i]; j < h->A.tile_ptr[i + 1]; j++) {
            int idx = h->A.tile_columnidx[j];
            unsigned int bitmask = 1u << (31 - (idx % 32));
            h->bitmaskA[(long long)i * h->bitmask_len + idx / 32] |= bitmask;
        }
    }
    for (int i = 0; i < h->B.tilen; i++) {
        for (int j = h->B.csc_tile_ptr[i]; j < h->B.csc_tile_ptr[i + 1]; j++) {
            int idx = h->B.csc_tile_rowidx[j];
            unsigned int bitmask = 1u << (31 - (idx % 32));
            h->bitmaskB[(long long)i * h->bitmask_len + idx / 32] |= bitmask;
        }
    }

    h->densityA = (double)h->A.numtile / ((double)h->A.tilem * (double)h->A.tilen);
    h->densityB = (double)h->B.numtile / ((double)h->B.tilem * (double)h->B.tilen);

    // nnzCub: main.cu's own naive-intermediate-product count
    // (main.cu:143-151), reproduced verbatim -- used only INSIDE
    // tilespgemm() for its own reported compression_rate/gflops_tile
    // (informational; this harness computes flops independently via
    // kernelbench.domains.sparse._cost_spgemm's own formula), but required
    // to satisfy tilespgemm()'s call signature.
    unsigned long long nnzCub = 0;
    for (int i = 0; i < h->A.nnz; i++) {
        int rowidx = h->A.columnindex[i];
        nnzCub += (unsigned long long)(h->B.rowpointer[rowidx + 1] - h->B.rowpointer[rowidx]);
    }
    h->nnzCub = nnzCub;

    return h;
}

void tsg_run(void *handle) {
    TsgHandle *h = (TsgHandle *)handle;
    // Free the PREVIOUS call's C (both its tile_* fields, set by every
    // tilespgemm() call, and its rowpointer/columnindex/value, set only if
    // tsg_finalize_csr() ran on it) before tilespgemm() overwrites the
    // struct's pointers wholesale -- otherwise every call after the first
    // leaks whatever the previous call allocated (the harness's own
    // warmup+measured-reps loop calls run() ~30 times per gate check).
    if (h->have_c) free_c(&h->C);
    memset(&h->C, 0, sizeof(SMatrix));

    unsigned long long nnzC_computed = 0;
    double compression_rate = 0, time_tile = 0, gflops_tile = 0;
    double time_step1 = 0, time_step2 = 0, time_step3 = 0, time_malloc = 0;

    // tilespgemm(): the artifact's own SpGEMM kernel (source/src/
    // tilespgemm-cuda.h, unmodified). REPEAT_NUM==1 (common.h:69, the
    // artifact's OWN shipped default), so this one call performs exactly
    // one H2D-of-tiled-A/B -> symbolic (step1) -> numeric column+value
    // compute (step2/step3) -> D2H-of-tiled-C pass, then frees every
    // device buffer it allocated (verified by reading the function's own
    // tail) before returning -- safe to call repeatedly with no device
    // memory leak across warmup+measured reps. See this file's header
    // comment for why the H2D-of-A/B transfer bundled inside this same
    // call is disclosed, not hoisted out.
    char empty_filename[1] = {0};
    tilespgemm(&h->A, &h->B, &h->C,
               h->bitmaskA, h->bitmaskB, h->bitmask_len,
               h->densityA, h->densityB, h->nnzCub,
               &nnzC_computed, &compression_rate, &time_tile, &gflops_tile,
               empty_filename,
               &time_step1, &time_step2, &time_step3, &time_malloc);
    h->have_c = true;
}

// Converts the tiled C sitting in the handle back to plain CSR (host,
// artifact's own tile2csr(), source/src/tile2csr.h, unmodified). Called
// ONCE from Python to_host() -- never inside the timed run() path.
void tsg_finalize_csr(void *handle) {
    TsgHandle *h = (TsgHandle *)handle;
    free(h->C.rowpointer);
    free(h->C.columnindex);
    free(h->C.value);
    h->C.rowpointer = nullptr;
    h->C.columnindex = nullptr;
    h->C.value = nullptr;
    tile2csr(&h->C);
}

void tsg_dims(void *handle, int *rows, int *cols, long long *nnz) {
    TsgHandle *h = (TsgHandle *)handle;
    *rows = h->C.m;
    *cols = h->C.n;
    *nnz = h->C.nnz;
}

void tsg_to_host(void *handle, int *rowptr_out, int *colidx_out, double *val_out) {
    TsgHandle *h = (TsgHandle *)handle;
    memcpy(rowptr_out, h->C.rowpointer, (size_t)(h->C.m + 1) * sizeof(int));
    memcpy(colidx_out, h->C.columnindex, (size_t)h->C.nnz * sizeof(int));
    memcpy(val_out, h->C.value, (size_t)h->C.nnz * sizeof(double));
}

void tsg_free(void *handle) {
    TsgHandle *h = (TsgHandle *)handle;
    if (h->have_c) free_c(&h->C);
    // matrix_destroy() (source/src/csr2tile.h, the artifact's own cleanup
    // helper -- the exact one main.cu itself calls on A/B) frees each
    // struct's own tile_ptr/tile_columnidx/tile_nnz/tile_csr_Value/
    // tile_csr_Col/tile_csr_Ptr/mask. It does NOT free B's
    // csc_tile_ptr/csc_tile_rowidx (aliased directly from an internal `BT`
    // struct inside csr2tile_col_major -- source/src/csr2tile.h -- not a
    // separate allocation matrix_destroy knows about) nor either struct's
    // tile_rowidx; main.cu's own cleanup leaks these same fields too (a
    // pre-existing, one-time, host-only artifact leak, not introduced by
    // this bridge -- freed here anyway since we CAN, unlike main.cu which
    // simply exits after one run).
    free(h->B.csc_tile_ptr);
    free(h->B.csc_tile_rowidx);
    free(h->A.tile_rowidx);
    free(h->B.tile_rowidx);
    matrix_destroy(&h->A);
    matrix_destroy(&h->B);
    free(h->A.rowpointer);   // shared by A and B (aliased in tsg_prepare); free once
    free(h->A.columnindex);
    free(h->A.value);
    free(h->bitmaskA);
    free(h->bitmaskB);
    delete h;
}

}  // extern "C"
