// Thin ctypes-callable bridge around PFPL's own single-precision
// absolute-error-bound GPU decoder (source/src/f32_abs_decomp_gpu.cu).
//
// Used ONLY by the harness's correctness gate (adapter.py::to_host() and
// ::free()), which per this integration's contract runs decompress+D2H
// OUTSIDE the timed region -- run() in bridge_encode.cu is the only call
// the harness's Timer ever brackets.
//
// Same `#define main ...` rename trick as bridge_encode.cu, applied to
// this file's own (different) main() -- see that file's docstring for the
// full rationale. No edits to source/ itself.
#include "pfpl_handle.h"

#define main pfpl_unused_main_decode
#include "source/src/f32_abs_decomp_gpu.cu"
#undef main

extern "C" {

// Decodes directly from `h->d_encoded` -- already resident on device from
// the matching pfplc_run() call in bridge_encode.cu, no re-upload of the
// compressed bytes needed -- into `h->d_decoded`, then D2H-copies the
// result into `host_out` (caller-allocated, exactly `h->insize` bytes:
// decompressing a valid PFPL stream always reconstructs the original byte
// count exactly). Returns the decoded byte count so the caller can assert
// it matches `insize` as a sanity check.
int pfpld_run_and_copy(void *handle, unsigned char *host_out) {
    PfplHandle *h = reinterpret_cast<PfplHandle *>(handle);

    cudaSetDevice(0);
    cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp, 0);
    const int SMs = deviceProp.multiProcessorCount;
    const int mTpSM = deviceProp.maxThreadsPerMultiProcessor;
    const int blocks = SMs * (mTpSM / TPB);

    d_reset<<<1, 1>>>();
    cudaMemset(h->d_decsize, 0, sizeof(int));
    d_decode<<<blocks, TPB>>>(h->d_encoded, h->d_decoded, h->d_decsize);
    cudaDeviceSynchronize();

    int decsize = 0;
    cudaMemcpy(&decsize, h->d_decsize, sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(host_out, h->d_decoded, decsize, cudaMemcpyDeviceToHost);
    return decsize;
}

}  // extern "C"
