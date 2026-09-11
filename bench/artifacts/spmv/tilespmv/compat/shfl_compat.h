// CUDA-version compat shim: the vendored CSR5_cuda/detail/cuda/*.h headers
// (source/src/external/CSR5_cuda/, written for CUDA ~8/9) call the
// pre-Volta warp-shuffle intrinsics __shfl/__shfl_up/__shfl_down/__shfl_xor
// without a mask argument. CUDA 12.9's headers no longer declare these at
// all (removed, not just deprecated -- "identifier __shfl is undefined" at
// compile time targeting sm_80); only the *_sync forms exist. This is
// exactly the CUDA-version-guard class of build-system fix
// ARTIFACT_GUIDE.md permits (no kernel arithmetic changes) -- NOT part of
// the artifact, included (via -I, see build.sh) before any CSR5_cuda header
// so these names resolve for the template-instantiated types actually used
// (int for tile-row indices, double for CSR5's value-scan/reduction code).
// Full-warp mask (0xffffffffu) reproduces the pre-Volta implicit-mask
// semantics these files assume (every thread in the warp participates,
// which holds here: CSR5's spmv kernels launch full warps with no
// divergent early-exit before these shuffles).
#ifndef TILESPMV_COMPAT_SHFL_H
#define TILESPMV_COMPAT_SHFL_H

template <typename T>
__device__ __forceinline__ T __shfl(T var, int srcLane, int width = 32) {
    return __shfl_sync(0xffffffffu, var, srcLane, width);
}

template <typename T>
__device__ __forceinline__ T __shfl_up(T var, unsigned int delta, int width = 32) {
    return __shfl_up_sync(0xffffffffu, var, delta, width);
}

template <typename T>
__device__ __forceinline__ T __shfl_down(T var, unsigned int delta, int width = 32) {
    return __shfl_down_sync(0xffffffffu, var, delta, width);
}

template <typename T>
__device__ __forceinline__ T __shfl_xor(T var, int laneMask, int width = 32) {
    return __shfl_xor_sync(0xffffffffu, var, laneMask, width);
}

#endif
