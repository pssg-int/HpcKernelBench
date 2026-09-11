// Thin extern "C" re-export of cuSZp v1's (SC'23) device-pointer f32 API.
//
// Why this file exists: unlike ../cuszp (SC'25)'s include/cuSZp.h, which
// wraps its declarations in `#ifdef __cplusplus / extern "C" { ... }`, v1's
// include/cuSZp_entry_f32.h (source/, unmodified) has NO extern "C" guard --
// nvcc therefore compiles SZp_compress_deviceptr_f32/SZp_decompress_
// deviceptr_f32 as ordinary (Itanium-ABI-mangled) C++ symbols, e.g.
// `_Z26SZp_compress_deviceptr_f32PfPhmPmfP11CUstream_st` (confirmed via
// `nm -D libcuszp_v1.so`), which ctypes cannot look up by the plain name.
//
// Per ARTIFACT_GUIDE.md rule 3 ("Patch minimally... source/ is
// git-ignored... every file WE write lives in <short>/, never only inside
// source/"), this is solved WITHOUT touching source/ at all: this file
// lives in cuszp-v1/ (this directory), #includes the unmodified header, and
// re-exports two stable extern "C" trampolines that do nothing but forward
// to the real (mangled) functions. Same category of fix as GPULZ's
// bridge_gpulz.cu (../../lossless-compression/gpulz/) -- a linkage-only
// shim, zero kernel-code changes, zero diff inside source/.
#include <cuda_runtime.h>
#include "cuSZp_entry_f32.h"

extern "C" {

void cuszpv1_compress_f32(float* d_oriData, unsigned char* d_cmpBytes,
                          size_t nbEle, size_t* cmpSize, float errorBound,
                          cudaStream_t stream) {
    SZp_compress_deviceptr_f32(d_oriData, d_cmpBytes, nbEle, cmpSize, errorBound, stream);
}

void cuszpv1_decompress_f32(float* d_decData, unsigned char* d_cmpBytes,
                            size_t nbEle, size_t cmpSize, float errorBound,
                            cudaStream_t stream) {
    SZp_decompress_deviceptr_f32(d_decData, d_cmpBytes, nbEle, cmpSize, errorBound, stream);
}

}  // extern "C"
