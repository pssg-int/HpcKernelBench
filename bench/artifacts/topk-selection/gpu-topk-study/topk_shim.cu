// Thin extern "C" shim around gpu_topK_benchmark's GridSelect kernel
// (declared in source/include/grid_select.h; its CUDA implementation ships
// as a PREBUILT shared library, source/third_party/libgridselect.so --
// already checked into the artifact repo, no third-party download/build
// needed for this kernel specifically, unlike the suite's other algorithms
// which need RAFT/Faiss/gpu_selection/DrTopKSC -- see STATUS.md).
//
// Same role as bench/artifacts/triangle-counting/{tot,tc-compare}'s shims:
// a minimal wrapper exposing a ctypes-callable boundary around the
// artifact's own, unmodified kernel entry point. Zero artifact code
// touched -- this file only #includes the artifact's own header and links
// against its own prebuilt .so.
//
// GridSelect follows the same two-call workspace-query protocol every
// algorithm in this suite's Factory<T,idxT> uses (source/benchmark/
// benchmark.cu's run_algo(), source/include/cub_topk.cuh's own comment):
// call once with buf=nullptr to learn buf_size, allocate once, then call
// repeatedly with that buffer. gridselect_prepare()/gridselect_run() split
// along exactly that boundary.
#include "source/include/grid_select.h"

#include <cuda_runtime.h>

extern "C" {

struct TopkHandle {
    void* d_buf;
    size_t buf_size;
};

// One-time workspace-size query + allocation (batch_size/len/k fixed for
// the lifetime of the handle, matching benchmark.cu's own run_algo()).
void* gridselect_prepare(int batch_size, int len, int k) {
    TopkHandle* h = new TopkHandle();
    h->buf_size = 0;
    h->d_buf = nullptr;
    nv::grid_select(nullptr, h->buf_size, nullptr, batch_size, len, k,
                    nullptr, nullptr, /*greater=*/true, 0);
    if (h->buf_size > 0) {
        cudaMalloc(&h->d_buf, h->buf_size);
    }
    return (void*)h;
}

// The kernel-only call. d_in/d_out/d_out_idx are caller-owned device
// buffers (allocated once by the Python adapter's prepare(), reused every
// call -- this project's harness contract, not the artifact's own
// fresh-data-per-iteration convention, see adapter.py docstring).
void gridselect_run(void* handle, const float* d_in, int batch_size, int len, int k,
                    float* d_out, int* d_out_idx) {
    TopkHandle* h = (TopkHandle*)handle;
    nv::grid_select(h->d_buf, h->buf_size, d_in, batch_size, len, k,
                    d_out, d_out_idx, /*greater=*/true, 0);
}

unsigned long long gridselect_buf_size(void* handle) {
    TopkHandle* h = (TopkHandle*)handle;
    return (unsigned long long)h->buf_size;
}

void gridselect_free(void* handle) {
    TopkHandle* h = (TopkHandle*)handle;
    if (h->d_buf) {
        cudaFree(h->d_buf);
    }
    delete h;
}

}  // extern "C"
