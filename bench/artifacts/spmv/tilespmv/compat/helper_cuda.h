// Minimal stand-in for the NVIDIA CUDA-Samples "helper_cuda.h" header
// (common/inc/helper_cuda.h in NVIDIA's cuda-samples repo, BSD-3-Clause).
// NOT part of the TileSpMV artifact -- it is included (as a system header,
// via -I) only because source/src/external/CSR5_cuda/detail/cuda/
// common_cuda.h expects it on the include path, per the artifact's own
// (unmodified) Makefile, which points -I at a NVIDIA_CUDA_Samples checkout
// this machine doesn't have. Grepping the entire vendored CSR5_cuda/ tree
// shows exactly one macro from this header is actually used:
// checkCudaErrors(call), wrapping a CUDA API call with an error check. This
// file supplies just that macro, with the same call-then-check-then-report
// semantics as the original, so no tracked file under source/ needs to be
// patched -- build.sh only adds this directory to the include path.
#ifndef TILESPMV_COMPAT_HELPER_CUDA_H
#define TILESPMV_COMPAT_HELPER_CUDA_H

#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>

#define checkCudaErrors(val) tilespmv_compat::check((val), #val, __FILE__, __LINE__)

namespace tilespmv_compat {
inline void check(cudaError_t result, const char *func, const char *file, int line) {
    if (result != cudaSuccess) {
        std::fprintf(stderr, "CUDA error at %s:%d code=%d(%s) \"%s\"\n",
                     file, line, static_cast<int>(result),
                     cudaGetErrorString(result), func);
        std::exit(EXIT_FAILURE);
    }
}
}  // namespace tilespmv_compat

#endif
