// Minimal driver that calls RASSM's own paper SDDMM kernel (sddmm_kstream,
// defined UNMODIFIED in source/code/include/sddmm/tiled.h) over its own
// Adaptive Tile Matrix (ATM) format, built via the paper's own Residue-based
// adaptive tile generator (source/code/include/Residue.h).
//
// This IS the artifact's "--type RASSM --kernel sddmm" code path from
// source/code/src/main.cpp -- but in the shipped CMake build that path is
// dead code: the S_atm construction + sddmm_kstream call in
// experiments.h::data_movement_experiment_sddmm() sit behind
// #ifdef RUN_CSR_ATM_KSTREAM, and that macro is never #defined anywhere in
// config.h (verified by grep across the whole include tree). So the shipped
// `rassm` binary silently no-ops for `--kernel sddmm` regardless of --type.
//
// We do NOT flip that macro or edit experiments.h/config.h (that would be
// patching the artifact's own driver, which the integration rules ask us to
// avoid touching). Instead this file -- new code, outside source/ -- calls
// the exact same sequence of unmodified artifact functions
// (Residue::adaptive_2d_greedy_Ti_greedy_Tj_tile_generator, the ATM
// constructor, sddmm_kstream) that the #ifdef'd block would have called.
// See STATUS.md for the full writeup.
//
// ATM reorders nonzeros internally into its own row_ptr/cols/vals arrays --
// NOT necessarily the caller's original CSR nnz order (RASSM's tiling groups
// nonzeros by column-tile for cache locality). We export the ATM's own
// row_ptr and cols alongside the O array so the Python adapter can realign
// each O[j] back to the caller's original (row,col) position before the
// harness's correctness gate runs -- the gate compares against a reference
// computed in the caller's original nnz order.

#include <cstring>
#include <string>
#include <vector>

#include "matrices/CSR.h"
#include "matrices/CSC.h"
#include "matrices/ATM.h"
#include "matrices/CSF.h"   // tiled.h also declares sddmm_csf(CSF&,...); needed to parse, unused here
#include "matrices/DCSC.h"  // tiled.h also declares sddmm_jstream(DCSC&,...); needed to parse, unused here
#include "Residue.h"
#include "sddmm/tiled.h"

// CACHE_NUM_WAYS (config.h: `extern ITYPE CACHE_NUM_WAYS;`, defined but left
// uninitialized in source/code/src/global.cpp) is normally set from the CLI
// binary's --numways option (main.cpp) before any Residue/tile-generator
// code runs; Residue.h::adaptive_2d_greedy_Ti_greedy_Tj_tile_generator
// divides by it. We call that path directly (see file header), so we set it
// here to the CLI's own default (--numways default_value 8, main.cpp).

struct RassmSddmmHandle {
    CSR<TYPE, ITYPE> *csr;
    ATM<TYPE, ITYPE> *atm;
    ITYPE Ti;
    ITYPE Tj;
};

extern "C" {

// nrows/ncols/nnz + expanded (row_idx[i], col_idx[i], vals[i]) triplets
// describing the sparsity pattern S, exactly as read from an .mtx file by
// the artifact's own Reader.h. feature = K (dense operand width). Ri/Rj =
// residue-matrix tile size (artifact CLI defaults: 64/64). cache_bytes =
// target cache size fed to the adaptive tile generator (artifact CLI
// default: DEFAULT_LLC = 1 MiB, see config.h).
void *rassm_sddmm_prepare(ITYPE nrows, ITYPE ncols, ITYPE nnz,
                           const ITYPE *row_idx, const ITYPE *col_idx,
                           const double *vals, ITYPE feature,
                           ITYPE Ri, ITYPE Rj, ITYPE cache_bytes) {
    CACHE_NUM_WAYS = 8;  // main.cpp's --numways default

    std::pair<ITYPE, ITYPE> *locs = new std::pair<ITYPE, ITYPE>[nnz];
    TYPE *vcopy = new TYPE[nnz];
    for (ITYPE i = 0; i < nnz; i++) {
        locs[i] = {row_idx[i], col_idx[i]};
        vcopy[i] = (TYPE) vals[i];
    }

    auto *csr = new CSR<TYPE, ITYPE>(nrows, ncols, nnz, locs, vcopy);
    auto *csc = new CSC<TYPE, ITYPE>(nrows, ncols, nnz, locs, vcopy);
    auto *res = new Residue<TYPE, ITYPE>(csr, csc, Ri, Rj, std::string("wrapped"),
                                          /*resolution=*/1, /*range_augmentation=*/false);

    // the paper's own adaptive residue-based 2D tile generator (unmodified)
    auto panels = res->adaptive_2d_greedy_Ti_greedy_Tj_tile_generator(
        feature, cache_bytes, /*cache_split=*/4, /*temporal_input=*/false,
        /*temporal_output=*/false, /*oi_aware=*/true);

    // the paper's own Adaptive Tile Matrix construction (unmodified)
    auto *atm = new ATM<TYPE, ITYPE>(nrows, ncols, nnz, locs, vcopy, panels);

    delete csc;
    delete res;
    delete[] locs;
    delete[] vcopy;

    auto *h = new RassmSddmmHandle;
    h->csr = csr;
    h->atm = atm;
    h->Ti = 512;  // matches main.cpp's CLI --Ti default
    h->Tj = 512;  // matches main.cpp's CLI --Tj default
    return (void *) h;
}

ITYPE rassm_sddmm_nnz(void *handle) {
    return ((RassmSddmmHandle *) handle)->atm->nnzs;
}

// Copy out the ATM's own row_ptr (length nrows+1) and cols (length nnz) so
// the caller can realign the O array from rassm_sddmm_run() to its original
// nnz order (see file header).
void rassm_sddmm_export_layout(void *handle, ITYPE *out_row_ptr, ITYPE *out_cols) {
    auto *h = (RassmSddmmHandle *) handle;
    std::memcpy(out_row_ptr, h->atm->row_ptr, sizeof(ITYPE) * (h->atm->nrows + 1));
    std::memcpy(out_cols, h->atm->cols, sizeof(ITYPE) * h->atm->nnzs);
}

// D1: dense operand indexed by column (N x feature, row-major, matches
// S.cols[ptr]*feature+k addressing in sddmm_kstream -- i.e. "B" in our
// P[i,j]=S[i,j]*dot(A[i,:],B[j,:]) convention).
// D2: dense operand indexed by row (M x feature, row-major -- "A" in our
// convention).
// O: output buffer, length nnz, in ATM's internal j-order (see
// rassm_sddmm_export_layout above for how to interpret it). Caller must have
// zeroed O before the call: sddmm_kstream accumulates with += (matches how
// experiments.h's MEM_RESET macro zeroes O before every measured call).
void rassm_sddmm_run(void *handle, const double *D1, const double *D2,
                      double *O, ITYPE feature) {
    auto *h = (RassmSddmmHandle *) handle;
    // TYPE == double (config.h), so no precision conversion is needed --
    // just call the artifact's own kernel directly on the caller's buffers.
    sddmm_kstream<TYPE, ITYPE>(*(h->atm), (TYPE *) D1, (TYPE *) D2, (TYPE *) O,
                                feature, h->Ti, h->Tj, /*chunk_size=*/1);
}

void rassm_sddmm_free(void *handle) {
    auto *h = (RassmSddmmHandle *) handle;
    delete h->csr;
    delete h->atm;
    delete h;
}

}  // extern "C"
