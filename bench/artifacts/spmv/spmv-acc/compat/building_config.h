// Hand-written stand-in for spmv-acc's CMake-generated
// src/building_config.h (from src/building_config.h.in via configure_file).
// NOT part of the artifact -- this integration bypasses the project's full
// CMake+HIP-CMake-module build (which needs `find_package(HIP)` pointed at
// a complete ROCm CMake tree, plus the `clipp` third-party dependency for
// the unrelated CLI target) and compiles the SpMV kernel library's own
// sources directly with hipcc (HIP_PLATFORM=nvidia -> nvcc under the
// hood), matching what -DKERNEL_STRATEGY=ADAPTIVE -DWAVEFRONT_SIZE=32
// would have generated for the "Adaptive" strategy on an NVIDIA target
// (WF_SIZE=32 = the actual CUDA warp size; ADAPTIVE is the paper's
// headline auto-selecting strategy -- see STATUS.md).
#ifndef SPMV_BUILDING_CONFIG_H
#define SPMV_BUILDING_CONFIG_H

#define ACCELERATE_ENABLED

#define __WF_SIZE__ 32
constexpr int __WRAP_SIZE__ = __WF_SIZE__;

#define WF_REDUCE_DEFAULT

#define KERNEL_STRATEGY_ADAPTIVE

#define ARCH_HIP
#define AVAILABLE_CU 108

#endif // SPMV_BUILDING_CONFIG_H
