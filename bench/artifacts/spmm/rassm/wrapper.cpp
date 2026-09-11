/*
 * Thin extern-"C" wrapper exposing RASSM's own preprocessing pipeline
 * (CSR -> CSC -> Residue -> adaptive_2d_greedy_Ti_greedy_Tj_tile_generator
 * -> ATM) and its k-stream SpMM kernel (spmm_atm_kstream_compiler_
 * vectorized), called DIRECTLY -- bypassing source/code/src/main.cpp's
 * CLI / .mtx-file-reading `rassm` executable entirely.
 *
 * Why: RASSM ships no library, only an end-to-end binary. It reads a matrix
 * from an .mtx file it opens itself, runs its own internal timing loop, and
 * only prints aggregate "Median Time"/"GFLOPS" text -- there is no flag or
 * code path that returns (or even computes, outside an unused #ifdef
 * RUN_CORRECTNESS_CHECK/RUN_CHECK macro that only prints a boolean) the
 * actual output matrix, and no way to hand it a caller-chosen dense operand
 * (its own B is `generate_dense()` from Reader.h: a fixed deterministic
 * pattern `(i % 100) / 100.0`, not something the correctness gate's
 * reference could be matched against without also reimplementing that
 * exact pattern). ARTIFACT_GUIDE.md rule 1: "If the artifact only ships an
 * end-to-end binary that loads matrices itself, wrap at the finest boundary
 * available and document the contamination" -- the finest boundary here is
 * RASSM's own header-only preprocessing + kernel functions, all templates,
 * all callable directly with our own in-memory COO matrix and our own B/C
 * buffers. Every function this file calls (CSR, CSC, Residue, ATM
 * constructors; adaptive_2d_greedy_Ti_greedy_Tj_tile_generator;
 * spmm_atm_kstream_compiler_vectorized) is copied from nowhere -- it is the
 * artifact's own unmodified code in source/code/include/, exactly the
 * sequence source/code/src/main.cpp's runtype::RASSM branch +
 * source/code/include/experiments.h's mirrored branch use, minus
 * main.cpp's CLI parsing / stdout logging / mtx-file I/O.
 *
 * rassm_prepare() IS RASSM's own tiling preprocessing (Residue matrix +
 * adaptive greedy panel generation) -- ARTIFACT_GUIDE.md rule 2: this
 * belongs in the adapter's prepare(), timed as preprocessing, not folded
 * into the timed kernel call.
 */

#include "config.h"
#include "Reader.h"
#include "Residue.h"
#include "matrices/CSR.h"
#include "matrices/CSC.h"
#include "matrices/ATM.h"
#include "spmm/kstream.h"

#include <iostream>
#include <string>
#include <utility>
#include <vector>

struct RassmHandle {
    ATM<TYPE, ITYPE>* atm;
};

extern "C" {

// rows/cols/vals: nnz-length COO arrays (0-based row/col indices, double
// values) -- our workload's CSR converted to COO in adapter.py (trivial,
// no RASSM-specific logic; the actual format conversion RASSM cares about
// -- CSR/CSC/Residue/panels/ATM -- all happens below).
RassmHandle* rassm_prepare(int nrows, int ncols, int nnz,
                            const int* rows, const int* cols, const double* vals,
                            int feature, int Ri, int Rj, int target_cache_size,
                            int cache_split, int temporal_input, int temporal_output,
                            int oi_aware, int resolution) {
    // config.h declares `extern ITYPE CACHE_NUM_WAYS;` (global.cpp defines
    // it, zero-initialized); Residue.h's adaptive_2d_greedy_Ti_greedy_Tj_
    // tile_generator() divides by it directly
    // (`max_output_cache_volume = (cache_size*cache_split)/CACHE_NUM_WAYS`)
    // with no zero-guard. main.cpp is the ONLY place that ever assigns it,
    // from CLI option "numways" (default_value(8)) -- since this wrapper
    // bypasses main.cpp entirely, the global is left at its zero default
    // and this division is an integer divide-by-zero (SIGFPE), caught
    // empirically under gdb while building this adapter (see
    // STATUS.md). Setting it here to the artifact's own documented CLI
    // default (8) is not a behavior change, just supplying the
    // initialization main.cpp would have done.
    CACHE_NUM_WAYS = 8;

    auto* locs = new std::pair<ITYPE, ITYPE>[nnz];
    auto* v = new TYPE[nnz];
    for (int i = 0; i < nnz; i++) {
        locs[i] = std::make_pair((ITYPE)rows[i], (ITYPE)cols[i]);
        v[i] = (TYPE)vals[i];
    }

    auto* spm = new CSR<TYPE, ITYPE>((ITYPE)nrows, (ITYPE)ncols, (ITYPE)nnz, locs, v);
    auto* csc = new CSC<TYPE, ITYPE>((ITYPE)nrows, (ITYPE)ncols, (ITYPE)nnz, locs, v);

    auto* res = new Residue<TYPE, ITYPE>(
        spm, csc, (int64_t)Ri, (int64_t)Rj, std::string("matrix"), (ITYPE)resolution,
        (bool)(temporal_input || temporal_output));

    std::vector<panel_t> panels = res->adaptive_2d_greedy_Ti_greedy_Tj_tile_generator(
        (ITYPE)feature, (ITYPE)target_cache_size, (ITYPE)cache_split,
        (bool)temporal_input, (bool)temporal_output, (bool)oi_aware);

    delete csc;
    delete spm;
    delete res;

    auto* h = new RassmHandle();
    h->atm = new ATM<TYPE, ITYPE>((ITYPE)nrows, (ITYPE)ncols, (ITYPE)nnz, locs, v, panels);

#ifdef RASSM_DEBUG_VERIFY
    {
        // DEBUG ONLY (RASSM_DEBUG_VERIFY): cross-check the ATM's sparsity
        // structure against a plain CSR built from the same COO, using the
        // artifact's OWN verify_matrix_structure() (Reader.h) -- not
        // compiled into the normal build.
        CSR<TYPE, ITYPE> checkcsr((ITYPE)nrows, (ITYPE)ncols, (ITYPE)nnz, locs, v);
        bool ok = verify_matrix_structure<TYPE, ITYPE>(checkcsr, *(h->atm));
        std::cerr << "[rassm wrapper debug] ATM structure correct? " << ok << std::endl;
    }
#endif

    delete[] locs;
    delete[] v;
    return h;
}

// ONE SpMM kernel call: spmm_atm_kstream_compiler_vectorized computes
// O[i,k] += sum_j S[i,j] * I[j,k], row-major, feature-contiguous
// (I_data[col*feature+k], O_data[row*feature+k] -- confirmed by reading
// source/code/include/spmm/kstream.h; PADDING_B/PADDING_C are both 0 in
// config.h so there is no extra stride) -- this matches this track's
// B: K x N row-major / C: M x N row-major convention exactly, no
// transpose needed on either side. O accumulates (+=), so the caller must
// zero it before each call for a clean C = A@B (same requirement as
// insum-spmm-coo). Ti/Tj are dead parameters inside
// spmm_atm_kstream_compiler_vectorized (confirmed at build time:
// "-Wunused-parameter" fires on both; the ATM object already carries its
// own per-panel tiling from rassm_prepare()), so 0 is passed for both.
void rassm_run(RassmHandle* h, const double* I, double* O, int feature, int chunk_size) {
    spmm_atm_kstream_compiler_vectorized<TYPE, ITYPE>(
        *(h->atm), (TYPE*)I, (TYPE*)O, (ITYPE)feature,
        /*Ti=*/0, /*Tj=*/0, (ITYPE)chunk_size);
}

void rassm_free(RassmHandle* h) {
    delete h->atm;
    delete h;
}

}  // extern "C"
