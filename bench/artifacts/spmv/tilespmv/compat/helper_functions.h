// Minimal stand-in for NVIDIA CUDA-Samples' "helper_functions.h", included
// by source/src/external/CSR5_cuda/detail/cuda/common_cuda.h. Grepping the
// entire vendored CSR5_cuda/ tree finds no symbol from this header actually
// used (only helper_cuda.h's checkCudaErrors is) -- it's included only
// because the artifact's own Makefile assumes a NVIDIA_CUDA_Samples
// checkout on the include path. Empty stub so the #include resolves; see
// helper_cuda.h in this same directory for the one macro that IS needed.
#ifndef TILESPMV_COMPAT_HELPER_FUNCTIONS_H
#define TILESPMV_COMPAT_HELPER_FUNCTIONS_H
#endif
