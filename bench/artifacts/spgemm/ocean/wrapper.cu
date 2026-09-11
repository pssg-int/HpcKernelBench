// Minimal C-callable wrapper around Ocean's own SpGEMM library entry point
// (somespgemm::SpGEMM::run + include/CSR.h's cuCSR/convert()). Touches ZERO
// lines of source/ -- it only calls already-defined, unmodified classes and
// functions with extern "C" linkage so ctypes can drive them from Python
// (mirrors bench/artifacts/spmm/inferfast/wrapper.cu's shape).
//
// Boundary wrapped (ARTIFACT_GUIDE.md rule 1: wrap the kernel, not a
// benchmark script): `somespgemm::SpGEMM::run(A, B)`, Ocean's own library
// entry point for one SpGEMM call -- the SAME class src/main.cu's CLI uses,
// called directly instead of going through the CLI's file-I/O/config-JSON
// plumbing.
//
// Preprocessing split (rule 2): `ocean_build_csr()` does the artifact's own
// host->device CSR conversion (via CSR::alloc() + the artifact's own
// convert(cuCSR&, const CSR&) H2D copy) -- the adapter's prepare() calls
// this once per operand, timed as preprocessing.
//
// Disclosed timing contamination (see STATUS.md for the full discussion):
// `SpGEMM::run()` constructs a fresh `Workspace` on every call (20
// cudaStreamCreate + 200 cudaEventCreate calls, per kernels/SpGEMM.cuh's
// `Workspace` constructor), which is NOT something this wrapper can hoist
// out without editing SpGEMM.cuh's control flow (touching the artifact's
// own algorithm code, which ARTIFACT_GUIDE.md rule 3 discourages) -- unlike
// AmgT's CSR2BSR_GPU (an already-idempotent, already-separate function this
// track's other adapter calls once in prepare()), Ocean's per-call setup is
// intrinsic to how SpGEMM::run() is written. `ocean_spgemm_new()` below
// sets `warmup_iters=0, bench_iters=1` so Ocean's OWN internal
// warmup-then-average loop collapses to exactly one prologue -> analysis ->
// symbolic/estimation -> numeric pass per call (the harness does its own
// warmup/measured-reps counting at the Python level, per the spec), but the
// Workspace-construction overhead is unavoidably included in every
// `ocean_spgemm_run()` call's wall time, in addition to symbolic+numeric+
// C-allocation. This makes this adapter's measured per-call time a
// (disclosed) upper bound on the spec's `timing_scope`, not an exact match.

#include <cstdint>
#include <memory>

// cub.cuh must be included before SpGEMM.cuh's own headers (AccumulatorESC.cuh
// uses cuda::std::numeric_limits, declared transitively by cub/libcu++) --
// matching source/src/main.cu's own include order exactly (it includes
// <cub/cub.cuh> before "SpGEMM.cuh" too), not a change to any artifact file.
#include <cub/cub.cuh>

#include "CSR.h"
#include "SpGEMM.cuh"

using namespace somespgemm;

extern "C" {

// --- operand construction: host CSR -> Ocean's device cuCSR -------------
// int32 indices in (uint32_t indices out, per Ocean's own index_t) --
// matches every recommended_subset matrix's index range (largest is
// road_usa at 24M rows, well under uint32 range); not valid for a
// hypothetical >4B-nnz matrix, which is out of this track's scope anyway.
void* ocean_build_csr(int rows, int cols, long long nnz,
                       const int32_t* h_row_offsets,
                       const int32_t* h_col_ids,
                       const double* h_data) {
    CSR host;
    host.alloc((size_t)rows, (size_t)cols, (size_t)nnz);
    for (long long i = 0; i <= (long long)rows; ++i)
        host.row_offsets[i] = (index_t)h_row_offsets[i];
    for (long long i = 0; i < nnz; ++i) {
        host.col_ids[i] = (index_t)h_col_ids[i];
        host.data[i] = h_data[i];
    }
    auto* dev = new std::shared_ptr<cuCSR>(std::make_shared<cuCSR>());
    convert(**dev, host, 0);
    return (void*)dev;
}

void ocean_free_csr(void* handle) {
    delete reinterpret_cast<std::shared_ptr<cuCSR>*>(handle);
}

// --- SpGEMM object: warmup_iters=0/bench_iters=1, see header comment ----
void* ocean_spgemm_new(int track_stage_times) {
    Config cfg;
    cfg.warmup_iters = 0;
    cfg.bench_iters = 1;
    cfg.track_stage_times = track_stage_times != 0;
    cfg.check_correctness = false;
    cfg.output_stats = false;
    return (void*)(new SpGEMM(cfg));
}

void ocean_spgemm_free(void* handle) {
    delete reinterpret_cast<SpGEMM*>(handle);
}

// --- ONE SpGEMM call: C = A @ B ------------------------------------------
void* ocean_spgemm_run(void* spgemm_handle, void* a_handle, void* b_handle) {
    auto* sp = reinterpret_cast<SpGEMM*>(spgemm_handle);
    auto& A = *reinterpret_cast<std::shared_ptr<cuCSR>*>(a_handle);
    auto& B = *reinterpret_cast<std::shared_ptr<cuCSR>*>(b_handle);
    auto result = sp->run(A, B);
    return (void*)(new std::shared_ptr<cuCSR>(result));
}

void ocean_csr_dims(void* handle, int* rows, int* cols, long long* nnz) {
    auto& C = *reinterpret_cast<std::shared_ptr<cuCSR>*>(handle);
    *rows = (int)C->rows;
    *cols = (int)C->cols;
    *nnz = (long long)C->nnz;
}

// D2H copy (via the artifact's own convert(CSR&, const cuCSR&)) into
// caller-allocated host buffers sized from ocean_csr_dims() first. NOT
// timed by the harness -- called once for the correctness gate, never
// inside the measured loop.
//
// BUG FOUND IN OCEAN (worth recording, see STATUS.md): the device
// `cuCSR::row_offsets` array Ocean's own SpGEMM produces does NOT populate
// the terminal sentinel `row_offsets[rows]` (the "total nnz" entry every
// CSR consumer, including scipy, requires -- `indptr[-1] == nnz`); it reads
// back as whatever the cudaMalloc'd device buffer happened to contain
// (observed: 0), while `cuCSR::nnz` (a separate host-side field) DOES carry
// the correct value the whole time. Every other row_offsets[0..rows-1]
// entry is verified correct against an independent scipy computation on a
// hand-built test matrix (see git history / STATUS.md for the repro). This
// is patched here at the wrapper boundary (not in source/, per ARTIFACT_GUIDE.md
// rule 3) by overwriting the one wrong slot with the authoritative nnz.
void ocean_csr_to_host(void* handle, int32_t* h_row_offsets,
                        int32_t* h_col_ids, double* h_data) {
    auto& C = *reinterpret_cast<std::shared_ptr<cuCSR>*>(handle);
    CSR host;
    convert(host, *C, 0);
    for (size_t i = 0; i <= host.rows; ++i)
        h_row_offsets[i] = (int32_t)host.row_offsets[i];
    h_row_offsets[host.rows] = (int32_t)host.nnz;  // see BUG note above
    for (size_t i = 0; i < host.nnz; ++i) {
        h_col_ids[i] = (int32_t)host.col_ids[i];
        h_data[i] = host.data[i];
    }
}

}  // extern "C"
