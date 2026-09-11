// Thin ctypes-callable bridge around PFPL's own single-precision
// absolute-error-bound GPU encoder (source/src/f32_abs_comp_gpu.cu).
//
// NOT part of the artifact. It exists because PFPL only ships a
// standalone CLI binary (that file's own `main()`) which:
//   (a) reads the input array from a file,
//   (b) allocates every device buffer (d_input/d_encoded/d_encsize/
//       d_fullcarry) and H2D-copies the input,
//   (c) wraps its own NUM_RUNS=9 loop, ONE cudaEvent pair PER ITERATION
//       (PFPL already avoids the "one event pair around a whole batched
//       loop" anti-pattern flagged elsewhere in this benchmark -- see
//       cb-spmv's STATUS.md -- so no correctness fix was needed there),
//   (d) writes the encoded stream back out to a file.
//
// This file's job is only to replace (a)/(d) (file I/O) with an in-memory
// ctypes API split across prepare()/run()/free(), so the harness's own
// warmup/measured-reps loop can call the SAME `d_encode` kernel directly,
// once per call, on data already resident on the device.
//
// `#define main pfpl_unused_main_encode` below renames PFPL's own main()
// so this translation unit can #include the artifact's .cu file VERBATIM
// (no edits to source/ at all -- `git diff` inside source/ is empty) while
// still compiling into a shared library that also needs bridge_decode.cu's
// own (differently-renamed) main() in the same .so. The renamed function is
// dead code, never called. Every kernel launch, cudaMalloc, and cudaMemcpy
// below is either copied unmodified from PFPL's main() or is a new,
// clearly-marked host-side allocation (the decode-side buffers, so the
// gate's decompress call in bridge_decode.cu has somewhere to write
// without a second round-trip).
#include "pfpl_handle.h"

#include <cstring>
#include <limits>

#define main pfpl_unused_main_encode
#include "source/src/f32_abs_comp_gpu.cu"
#undef main

extern "C" {

// Allocates all device buffers (encode-side, lifted directly from PFPL's
// own main()'s allocation block, plus decode-side buffers sized to
// `insize` -- decompressing a valid PFPL stream always reconstructs
// exactly the original byte count) and H2D-copies the input. Called once,
// timed once by adapter.py::prepare() as preprocessing.
void *pfplc_prepare(int insize, const unsigned char *host_data, float errorbound) {
    PfplHandle *h = new PfplHandle();
    h->insize = insize;
    h->errorbound = errorbound;
    h->threshold = std::numeric_limits<float>::infinity();  // no sentinel passthrough

    cudaSetDevice(0);
    cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp, 0);
    const int SMs = deviceProp.multiProcessorCount;
    const int mTpSM = deviceProp.maxThreadsPerMultiProcessor;
    h->blocks = SMs * (mTpSM / TPB);
    h->chunks = (insize + CS - 1) / CS;
    h->maxsize = 3 * (int)sizeof(int) + h->chunks * (int)sizeof(short) + h->chunks * CS;

    cudaMalloc((void **)&h->d_input, insize);
    cudaMemcpy(h->d_input, host_data, insize, cudaMemcpyHostToDevice);
    cudaMalloc((void **)&h->d_encoded, h->maxsize);
    cudaMalloc((void **)&h->d_encsize, sizeof(int));
    cudaMalloc((void **)&h->d_fullcarry, h->chunks * sizeof(int));

    // decode-side buffers (plain cudaMalloc; no kernel code involved)
    cudaMalloc((void **)&h->d_decoded, insize);
    cudaMalloc((void **)&h->d_decsize, sizeof(int));

    return h;
}

// ONE d_encode launch -- PFPL's own kernel, byte-for-byte unmodified.
// Re-zeroes g_chunk_counter (d_reset) and d_fullcarry immediately before
// every launch: PFPL's own main() does the same reset every NUM_RUNS
// iteration (lines 373-374 of the included file) because d_encode's
// dynamic work assignment (atomicAdd on g_chunk_counter) and carry
// propagation (fullcarry) are only valid starting from a zeroed state.
// Our harness calls run() independently many times (an isolated
// correctness check, then warmup, then measured reps), so this reset
// happens once per run() call instead of once per PFPL's internal loop
// iteration -- same requirement, same fix, just relocated to match this
// benchmark's call pattern (the same category of fix as cb-spmv's
// per-launch d_y re-zero; see that adapter's bridge.cu).
void pfplc_run(void *handle) {
    PfplHandle *h = reinterpret_cast<PfplHandle *>(handle);
    d_reset<<<1, 1>>>();
    cudaMemset(h->d_fullcarry, 0, h->chunks * sizeof(int));
    cudaMemset(h->d_encsize, 0, sizeof(int));
    d_encode<<<h->blocks, TPB>>>(h->d_input, h->insize, h->d_encoded,
                                  h->d_encsize, h->d_fullcarry,
                                  h->errorbound, h->threshold);
    cudaDeviceSynchronize();
}

// D2H of just the 4-byte achieved-size counter -- cheap, used for ratio
// bookkeeping in adapter.py::free().
int pfplc_encoded_size(void *handle) {
    PfplHandle *h = reinterpret_cast<PfplHandle *>(handle);
    int sz = 0;
    cudaMemcpy(&sz, h->d_encsize, sizeof(int), cudaMemcpyDeviceToHost);
    return sz;
}

void pfplc_free(void *handle) {
    PfplHandle *h = reinterpret_cast<PfplHandle *>(handle);
    if (!h) return;
    cudaFree(h->d_input);
    cudaFree(h->d_encoded);
    cudaFree(h->d_encsize);
    cudaFree(h->d_fullcarry);
    cudaFree(h->d_decoded);
    cudaFree(h->d_decsize);
    delete h;
}

}  // extern "C"
