// Minimal C-callable wrapper around BLCO's own MTTKRP-benchmark entry points
// (source/include/blco.hpp's gen_blcotensor_host/gen_blcotensor_device and
// source/include/alto_dev.hpp's mttkrp_alto_dev_onemode<IType>). Touches ZERO
// lines of source/ -- every function called below is already-defined,
// unmodified artifact code, with extern "C" entry points added here so
// ctypes can drive it from Python (mirrors bench/artifacts/spgemm/ocean/
// wrapper.cu's shape).
//
// Boundary wrapped (ARTIFACT_GUIDE.md rule 1): source/src/alto_dev.cu's
// mttkrp_alto_dev_onemode<IType> -- the artifact's OWN "single MTTKRP
// execution" function (its header comment literally says "used by both
// benchmark MTTKRP and CPD driver"), called ONCE per blco_run() (kernel_id=1,
// "BLCO Level-1" -- see adapter.py's docstring for why Level-1 was picked
// over kernel_id=10/AUTO). This is the SAME function BLCO's own benchmark
// driver (mttkrp_alto_dev, called by main.cpp's `-p` / --bench CLI flag)
// calls inside its own per-iteration timing loop -- we call it directly,
// once, instead of going through that loop or the CLI's own file-I/O.
//
// Preprocessing split (rule 2): blco_build() does the artifact's own COO ->
// BLCO conversion (CreateSparseTensor -> gen_blcotensor_host<IType>, i.e.
// ALTO linearization+sort followed by BLCO blocking/relinearization) and
// blco_upload() does the host->device transfer (gen_blcotensor_device +
// send_blcotensor_over + send_masks_over + make_device_copy(KruskalModel)) --
// adapter.py's prepare() calls both, once, timed as preprocessing.
//
// In-memory (non-streaming) path only (per the task brief: "use the in-
// memory path if available, single-GPU scope"): max_block_size = nnz in
// blco_build() and stream_data=false throughout, matching main.cpp's own
// `if (!stream_data) max_block_size = X->nnz;` branch and mttkrp_alto_dev's
// own `if (!stream_data) { num_streams = at_host->block_count; max_block_size
// = 0; ... send_blcotensor_over(...); }` branch -- i.e. the WHOLE BLCO tensor
// is resident on the GPU before blco_run() is ever called, exactly as
// mttkrp_alto_dev's non-streaming mode does. BLCO's own out-of-memory
// streaming machinery (stream_data=true, send_block_over() called freshly
// inside every mttkrp_alto_dev_onemode() call, source/src/alto_dev.cu
// line ~1985) is NOT exercised -- see STATUS.md.

#include "common.hpp"
#include "sptensor.hpp"
#include "kruskal_model.hpp"
#include "blco.hpp"
#include "alto_dev.hpp"

#include <cstdint>
#include <cstring>
#include <vector>

struct BlcoHandle {
    SparseTensor* spt = nullptr;
    blcotensor* at_host = nullptr;
    blcotensor* at_dev = nullptr;
    KruskalModel* M = nullptr;
    KruskalModel* M_dev = nullptr;
    int nmodes = 0;
    int target_mode = 0;
    int kernel_id = 1;
    IType thread_cf = 1;
    int nprtn = 1;
    IType rank = 0;
    IType tmode_length = 0;
};

extern "C" {

// --- COO -> BLCO (host), the artifact's own format conversion -----------
// dims/coo_flat are IType-width (unsigned long long == uint64_t on this
// platform); coo_flat is nnz*nmodes, row-major (one row per nonzero, one
// column per mode) -- the SAME layout CreateSparseTensor's own `IType* cidx`
// parameter expects (source/src/sptensor.cpp:251), and the SAME layout
// kernelbench.domains.tensor.SparseTensorWorkload.indices already uses, so
// no transpose is needed on the Python side.
void* blco_build(int nmodes, const uint64_t* dims_in, int64_t nnz,
                  const uint64_t* coo_flat, const double* values,
                  int64_t rank, int target_mode, int kernel_id) {
    BlcoHandle* h = new BlcoHandle();
    h->nmodes = nmodes;
    h->target_mode = target_mode;
    h->kernel_id = kernel_id;
    h->rank = (IType)rank;
    h->thread_cf = 1;   // unused by kernel_id=1 (lvl2-only parameter)
    h->nprtn = 1;        // unused by kernel_id=1 (lvl3-only parameter)

    std::vector<IType> dims(nmodes);
    for (int i = 0; i < nmodes; i++) dims[i] = (IType)dims_in[i];

    // CreateSparseTensor copies dims/cidx/vals internally (memcpy) -- the
    // const_cast on coo_flat/values is safe, it never writes through them.
    CreateSparseTensor((IType)nmodes, dims.data(), (IType)nnz,
                        (IType*)coo_flat, (FType*)values, &h->spt);

    // gen_blcotensor_host<IType>: ALTO linearization + sort (alto.hpp's
    // gen_alto<IType>) followed by BLCO's own relinearize+block step
    // (blco.hpp). max_block_size = nnz: ONE block covering the whole
    // tensor, matching main.cpp's own in-memory-path default (see module
    // docstring) -- this is the artifact's own choice, not a value we
    // invented for convenience.
    h->at_host = gen_blcotensor_host<IType>(h->spt, (IType)nnz);
    h->tmode_length = h->at_host->modes[target_mode];

    CreateKruskalModel(nmodes, dims.data(), (IType)rank, &h->M);
    // M->U[target_mode] is intentionally left uninitialized here -- it is
    // the OUTPUT buffer (mttkrp_alto_dev_onemode writes the MTTKRP result
    // directly into the device copy of this slot, exactly as BLCO's own
    // mttkrp_alto_dev() does, source/src/alto_dev.cu line ~1917: `output =
    // mats_staging_ptr[target_mode]`), never read as an input.
    return (void*)h;
}

// One non-target-mode factor matrix, dims[mode] x rank, ROW-MAJOR (matches
// KruskalModel's own M->U[n][j*rank+i] convention, source/src/kruskal_model.
// cpp's ExportKruskalModel -- so no transpose is needed from a numpy (rows,
// R) array either).
void blco_set_factor(void* handle, int mode, const double* data) {
    BlcoHandle* h = (BlcoHandle*)handle;
    IType n = h->M->dims[mode] * h->M->rank;
    memcpy(h->M->U[mode], data, sizeof(FType) * n);
}

// --- host -> device transfer, still "preprocessing" per rule 2 ----------
// Replicates mttkrp_alto_dev()'s own non-streaming prologue exactly
// (source/src/alto_dev.cu lines ~1858-1900): gen_blcotensor_device with
// num_streams = at_host->block_count and MB=0 (each device block sized to
// match its host block 1:1, i.e. no streaming), one synchronous
// send_blcotensor_over() call, send_masks_over() (the __constant__
// ALTO_MASKS/ALTO_POS symbols mttkrp_lvl1_*_kernel reads), then
// make_device_copy(KruskalModel*) (uploads every factor matrix, including
// the not-yet-written target-mode buffer that becomes the output).
void blco_upload(void* handle) {
    BlcoHandle* h = (BlcoHandle*)handle;
    IType num_streams = h->at_host->block_count;
    h->at_dev = gen_blcotensor_device(h->at_host, num_streams, /*MB=*/0,
                                       /*do_batching=*/false);
    send_blcotensor_over(h->at_dev->streams[0], h->at_dev, h->at_host);
    send_masks_over(h->at_host);
    h->M_dev = make_device_copy(h->M);

    // REQUIRED zero-init, not optional (see STATUS.md "Real artifact bug
    // found #2"): every mttkrp_lvl1_*_kernel in source/src/alto_dev.cu
    // accumulates into `output` via atomicAdd (never a plain store), so the
    // output buffer must start at exactly zero for the result to be
    // mathematically correct. M_dev->U[target_mode] is the device COPY of
    // M->U[target_mode], which CreateKruskalModel() (source/src/kruskal_
    // model.cpp) allocates via AlignedMalloc/posix_memalign and never
    // zeroes -- it is uninitialized host heap memory, copied verbatim to
    // device by make_device_copy() above. Without this memset, any output
    // row is atomicAdd's onto garbage; BLCO's own CLI (source/src/main.cpp)
    // has the SAME latent issue (KruskalModelRandomInit fills the
    // target-mode slot with pseudorandom U(0,1) noise instead of zeros,
    // and neither the plain -p/--bench path nor the -c/do_check path ever
    // memsets it before calling mttkrp_alto_dev) -- their benchmark-only
    // runs never notice because they don't inspect the output values, and
    // their own do_check path silently adds the true MTTKRP result on top
    // of that leftover noise. This memset is purely additive (a missing
    // buffer initialization the caller was always responsible for), not a
    // change to source/ or to any kernel's arithmetic.
    check_cuda(cudaMemset(h->M_dev->U[h->target_mode], 0,
                          sizeof(FType) * h->tmode_length * h->rank),
              "blco_upload zero output buffer");
    check_cuda(cudaDeviceSynchronize(), "blco_upload sync");
}

// --- ONE MTTKRP invocation -----------------------------------------------
// mttkrp_alto_dev_onemode<IType>, called once (this IS "one iteration" of
// BLCO's own iters-loop in mttkrp_alto_dev, source/src/alto_dev.cu line
// 1917 -- we call the same function that loop calls, directly). kernel_id=1
// (BLCO Level-1) needs no partial-matrix machinery (that is only allocated/
// used for kernel_id==3 or kernel_id==10-with-small-target-mode-length,
// source/src/alto_dev.cu lines ~1881-1892 and ~1958-1966) so
// partial_matrices/partial_matrices_dev are correctly passed as nullptr.
// cudaDeviceSynchronize() before returning makes this call's boundary safe
// for kernelbench.impls.gpu_cuda.CudaEventTimer regardless of which of
// BLCO's own (blocking, cudaStreamCreate-created) streams the kernel
// actually launched on.
void blco_run(void* handle) {
    BlcoHandle* h = (BlcoHandle*)handle;
    FType** mats_staging_ptr = h->M_dev->U;
    FType** mats_dev = h->M_dev->U_dev;
    FType* output = mats_staging_ptr[h->target_mode];
    mttkrp_alto_dev_onemode<IType>(
        h->at_host, h->at_dev, h->M, output, h->kernel_id, h->target_mode,
        h->thread_cf, /*stream_data=*/false, h->nprtn,
        mats_staging_ptr, mats_dev,
        /*partial_matrices=*/nullptr, /*partial_matrices_dev=*/nullptr);
    check_cuda(cudaGetLastError(), "blco_run kernel launch");
    check_cuda(cudaDeviceSynchronize(), "blco_run sync");
}

// D2H copy of the target mode's (now-overwritten) factor buffer -- NOT
// timed by the harness (called once for the correctness gate, never inside
// the measured loop). out_host must be caller-allocated, tmode_length*rank
// doubles.
void blco_get_output(void* handle, double* out_host) {
    BlcoHandle* h = (BlcoHandle*)handle;
    FType* dev_out = h->M_dev->U[h->target_mode];
    check_cuda(cudaMemcpy(out_host, dev_out,
                          sizeof(FType) * h->tmode_length * h->rank,
                          cudaMemcpyDeviceToHost),
              "blco_get_output D2H");
}

void blco_free(void* handle) {
    BlcoHandle* h = (BlcoHandle*)handle;
    if (h->M_dev) destroy_kruskal_model_dev(h->M_dev);
    if (h->at_dev) delete_blcotensor_device(h->at_dev);
    if (h->at_host) delete_blcotensor_host(h->at_host);
    if (h->M) DestroyKruskalModel(h->M);
    if (h->spt) DestroySparseTensor(h->spt);
    delete h;
}

}  // extern "C"
