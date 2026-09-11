// Minimal stand-in for <opencv2/cudafeatures2d.hpp>, used ONLY to satisfy
// source/cuda/src/labeling_allegretti_2019_BUF.cu's (unmodified) #include of
// this header when OpenCV (with its CUDA-enabled "cudafeatures2d" contrib
// module) is not installed on this machine and building OpenCV+contrib from
// source is out of scope (ARTIFACT_GUIDE: "YACCLAB needs OpenCV: only if
// importable/installable artifact-locally without building OpenCV from
// source; otherwise wrap the CUDA kernels directly with a thin shim").
//
// This defines EXACTLY the two POD types the kernel functions themselves
// use as parameters -- cv::cuda::PtrStepSz<T> (aliased PtrStepSzb/PtrStepSzi)
// -- replicating real OpenCV's own field layout/semantics (T* data; size_t
// step in BYTES; int rows, cols; elem_size = sizeof(T)) closely enough that
// the KERNEL CODE (verbatim, unmodified, included from source/) behaves
// identically to how it would against the real type. This file introduces
// NO algorithmic logic of its own -- see ../../STATUS.md's "OpenCV shim"
// section for the full accounting of what real OpenCV entry point every
// stub here stands in for.
#ifndef YACCLAB_SHIM_CUDAFEATURES2D_HPP_
#define YACCLAB_SHIM_CUDAFEATURES2D_HPP_

#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cuda_runtime.h>

namespace cv {

struct Size {
    int width = 0, height = 0;
    Size() = default;
    Size(int w, int h) : width(w), height(h) {}
};

namespace cuda {

// Real cv::cuda::PtrStepSz<T> layout (opencv2/core/cuda_types.hpp): a
// device pointer + row pitch in BYTES + shape. Kernels index it as
// `labels[idx]` (operator[]) and read `.step`/`.elem_size`/`.rows`/`.cols`/
// `.data` as plain fields -- all reproduced here identically.
template <typename T>
struct PtrStepSz {
    T* data = nullptr;
    size_t step = 0;       // row pitch, BYTES
    int rows = 0, cols = 0;
    int elem_size = static_cast<int>(sizeof(T));

    __host__ __device__ __forceinline__ T& operator[](size_t i) { return data[i]; }
    __host__ __device__ __forceinline__ const T& operator[](size_t i) const { return data[i]; }
};

using PtrStepSzb = PtrStepSz<unsigned char>;
using PtrStepSzi = PtrStepSz<int>;

// Minimal GpuMat stand-in: a flat (non-pitch-padded) device allocation.
// Real OpenCV uses cudaMallocPitch for alignment; a plain contiguous
// allocation (step == cols*elem_size exactly) is a legal pitch value too --
// every address computation in the kernels goes through `.step` generically,
// so this is behaviorally equivalent, just without the (perf-only, not
// correctness-affecting) row-alignment padding real cudaMallocPitch adds.
class GpuMat {
public:
    void* data = nullptr;
    size_t step = 0;
    int rows = 0, cols = 0;
    int type_ = 0;

    GpuMat() = default;
    ~GpuMat() { release(); }
    GpuMat(const GpuMat&) = delete;
    GpuMat& operator=(const GpuMat&) = delete;

    Size size() const { return Size(cols, rows); }

    void release() {
        if (data) { cudaFree(data); data = nullptr; }
        rows = cols = 0; step = 0;
    }

    static int elem_size_of(int type) { return type == 0 ? 1 : 4; }  // 0=uchar(8UC1), 4=int32(32SC1)

    void create(Size sz, int type) { create(sz.height, sz.width, type); }

    void create(int r, int c, int type) {
        int es = elem_size_of(type);
        if (rows == r && cols == c && type_ == type && data) return;  // idempotent, matches GpuMat::create's own no-op-if-same-size behavior
        release();
        rows = r; cols = c; type_ = type;
        // cudaMallocPitch, NOT a flat cudaMalloc: real OpenCV's GpuMat uses
        // cudaMallocPitch internally for exactly the reason this matters
        // here -- the BUF/BKE kernels do 16-bit-aligned reads of pairs of
        // pixels (`*(reinterpret_cast<int16_t*>(img.data + img_index))`,
        // labeling_allegretti_2019_BUF.cu line ~119); a flat, unpadded
        // `step = cols*elem_size` is ODD whenever `cols` is odd, producing
        // an unaligned row start on odd rows and a device-side "misaligned
        // address" fault the FIRST time it was tried here (confirmed via
        // debug_buf.py before this fix; the padding is a real correctness
        // requirement of this kernel's own I/O pattern, not just a
        // performance nicety).
        size_t pitch = 0;
        cudaError_t e = cudaMallocPitch(&data, &pitch, static_cast<size_t>(c) * es, r);
        step = pitch;
        if (e != cudaSuccess) fprintf(stderr, "[yacclab-shim] cudaMallocPitch failed: %s (r=%d c=%d type=%d)\n", cudaGetErrorString(e), r, c, type);
    }

    void upload_raw(const void* host, size_t host_row_bytes) {
        cudaError_t e = cudaMemcpy2D(data, step, host, host_row_bytes, host_row_bytes, rows, cudaMemcpyHostToDevice);
        if (e != cudaSuccess) fprintf(stderr, "[yacclab-shim] upload cudaMemcpy2D failed: %s\n", cudaGetErrorString(e));
    }

    void download_raw(void* host, size_t host_row_bytes) const {
        cudaError_t e = cudaMemcpy2D(host, host_row_bytes, data, step, host_row_bytes, rows, cudaMemcpyDeviceToHost);
        if (e != cudaSuccess) fprintf(stderr, "[yacclab-shim] download cudaMemcpy2D failed: %s (data=%p step=%zu rows=%d host_row_bytes=%zu)\n", cudaGetErrorString(e), data, step, rows, host_row_bytes);
    }

    // Templated so this header does not need cv::Mat1b/Mat1i's definition
    // (declared later, in labeling_algorithms.h's shim) -- only
    // instantiated where BUF/BKE's own code actually calls upload()/
    // download() (their un-called-by-us MemoryTransferHostToDevice/
    // DeviceToHost methods), by which point Mat1b/Mat1i are fully defined.
    template <typename Mat1T>
    void upload(const Mat1T& m) {
        upload_raw(m.data, static_cast<size_t>(m.cols) * elem_size_of(type_));
    }
    template <typename Mat1T>
    void download(Mat1T& m) const {
        download_raw(m.data, static_cast<size_t>(m.cols) * elem_size_of(type_));
    }

    // Implicit conversions to the PtrStepSz types the kernels take by
    // value -- the same role real OpenCV's GpuMat::operator PtrStepSz<T>()
    // plays when a GpuMat is passed where a PtrStepSz<T> is expected
    // (e.g. `Merge<<<...>>>(d_img_, d_img_labels_)`).
    operator PtrStepSzb() const {
        PtrStepSzb p; p.data = static_cast<unsigned char*>(data);
        p.step = step; p.rows = rows; p.cols = cols; p.elem_size = 1;
        return p;
    }
    operator PtrStepSzi() const {
        PtrStepSzi p; p.data = static_cast<int*>(data);
        p.step = step; p.rows = rows; p.cols = cols; p.elem_size = 4;
        return p;
    }
};

}  // namespace cuda
}  // namespace cv

#define CV_8UC1  0
#define CV_32SC1 4

#endif  // YACCLAB_SHIM_CUDAFEATURES2D_HPP_
