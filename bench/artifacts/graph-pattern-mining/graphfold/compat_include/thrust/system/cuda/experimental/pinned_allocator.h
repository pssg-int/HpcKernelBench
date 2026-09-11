// Build-compat shim -- NOT part of the GraphFold artifact.
//
// The CUDA-Toolkit-bundled Thrust shipped with nvcc 12.9 no longer provides
// thrust/system/cuda/experimental/pinned_allocator.h (removed/deprecated in
// modern Thrust; GraphFold's source/thirdparty/thrust vendors an OLD Thrust
// that still has it, but that old tree is itself incompatible with CTK
// 12.9's bundled CUB when device_vector/host_vector internals actually get
// instantiated -- mixing them produces hundreds of "THRUST_NS_QUALIFIER
// undefined" style errors out of cub/device/dispatch/dispatch_streaming_
// reduce.cuh). So this build uses the CTK's own (version-matched) Thrust
// throughout, and only stubs this one missing header.
//
// src/utils/buffer.h includes this solely to name the type used by an
// UNUSED Buffer constructor overload:
//   explicit Buffer(const thrust::host_vector<
//                   T, thrust::cuda::experimental::pinned_allocator<T>>&)
// This shim's adapter code (gf_shim.cu) never builds a pinned host_vector --
// Buffer is only ever constructed here from a thrust::device_vector or a
// raw (T*, size_t) pair -- so the alias below only needs to make the NAME
// resolvable at parse time (required even for an uninstantiated template
// member's parameter type); it is never actually instantiated.
#pragma once
#include <memory>

namespace thrust {
namespace cuda {
namespace experimental {
template <typename T>
using pinned_allocator = std::allocator<T>;
} // namespace experimental
} // namespace cuda
} // namespace thrust
