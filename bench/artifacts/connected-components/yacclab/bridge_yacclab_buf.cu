// Driver for YACCLAB's BUF (Block-based Union-Find) 2D 8-connectivity GPU
// labeling algorithm -- source/cuda/src/labeling_allegretti_2019_BUF.cu,
// #include'd VERBATIM below (byte-identical to the tracked file; `git -C
// source diff` is empty). The four kernels this file defines
// (InitLabeling, Merge, Compression, FinalLabeling) and the `BUF` class
// orchestrating them are 100% unmodified upstream code -- this bridge only
// supplies (a) the OpenCV-shim types they compile against (see
// ../shim/{opencv2/cudafeatures2d.hpp,labeling_algorithms.h,register.h} --
// -I ordering in build.sh puts the shim directory before YACCLAB's own
// include/, so these #include directives resolve to our stand-ins, not the
// (absent) real OpenCV headers) and (b) a small extern "C" API that drives
// the `BUF` class directly, replacing main.cc/yacclab_tests.cc (YACCLAB's
// own benchmark driver, out of scope per ARTIFACT_GUIDE rule 1) with a
// prepare/run/fetch/free split matching this benchmark's harness contract.
#include <cstdint>
#include <cstddef>
#include <cuda_runtime.h>

#include "labeling_allegretti_2019_BUF.cu"  // verbatim, see build.sh -I path

extern "C" {

// PREPARE: allocate device image + labels buffers and upload the input
// image. Untimed preprocessing (matches the harness's own prepare()/run()
// split -- the actual CCL computation happens only in yacclab_buf_run()).
void* yacclab_buf_prepare(const unsigned char* h_img, int rows, int cols) {
    auto* obj = new BUF();
    obj->d_img_.create(rows, cols, CV_8UC1);
    obj->d_img_.upload_raw(h_img, static_cast<size_t>(cols));
    // Labels buffer is created by BUF::PerformLabeling() itself
    // (`d_img_labels_.create(d_img_.size(), CV_32SC1)`, unmodified upstream
    // code) on first call -- nothing else to do here.
    return obj;
}

// RUN: exactly one call to BUF::PerformLabeling() -- upstream's own
// unmodified method, launching the 4 kernels above in sequence
// (InitLabeling -> Merge -> Compression -> FinalLabeling) plus a
// cudaDeviceSynchronize() it already includes. This is what the harness's
// CudaEventTimer brackets.
void yacclab_buf_run(void* handle) {
    static_cast<BUF*>(handle)->PerformLabeling();
}

// FETCH: D2H copy of the int32 label image, for the correctness gate only
// -- deliberately outside the timed region.
void yacclab_buf_fetch_labels(void* handle, int32_t* out) {
    auto* obj = static_cast<BUF*>(handle);
    obj->d_img_labels_.download_raw(out, static_cast<size_t>(obj->d_img_labels_.cols) * sizeof(int32_t));
}

void yacclab_buf_free(void* handle) {
    delete static_cast<BUF*>(handle);
}

}  // extern "C"
