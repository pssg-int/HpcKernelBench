// Minimal stand-in for YACCLAB's own include/labeling_algorithms.h --
// used ONLY to let source/cuda/src/labeling_allegretti_2019_BUF.cu (and
// _BKE.cu, unmodified, verbatim #include'd from our bridge) compile without
// the full YACCLAB benchmark harness (YacclabTensorInput/Output, the
// LabelingCheckSingleton correctness-check registry, PerformanceEvaluator's
// real timing, the CLI-driven algorithm/kernel registry) -- none of which
// this integration uses; per ARTIFACT_GUIDE rule 1 ("wrap the kernel, not
// the paper's benchmark script") we only need `class BUF`'s (and `BKE`'s)
// own code, verbatim, to compile and run against REAL device buffers this
// integration sets up itself (see bridge_yacclab_buf.cu).
//
// Every identifier here exists ONLY to satisfy a name `labeling_allegretti_
// 2019_BUF.cu` references directly (d_img_, d_img_labels_, img_,
// img_labels_, perf_, IsLabelBackground(), the PerformLabelingBlocksize
// override signature, Connectivity2D::CONN_8) -- no algorithmic logic
// lives here; the actual GPU kernels (InitLabeling/Merge/Compression/
// FinalLabeling for BUF; the BKE equivalents) are 100% the unmodified
// upstream file.
#ifndef YACCLAB_SHIM_LABELING_ALGORITHMS_H_
#define YACCLAB_SHIM_LABELING_ALGORITHMS_H_

#include <string>

#include <opencv2/cudafeatures2d.hpp>

enum class Connectivity2D { CONN_4, CONN_8 };

// Trivial host-side buffer stand-ins for cv::Mat1b/cv::Mat1i -- only
// referenced by BUF/BKE's own MemoryTransferHostToDevice/DeviceToHost
// methods, which this integration never calls (it drives d_img_/
// d_img_labels_ directly via the bridge); they only need to exist as
// valid types for those methods to type-check.
namespace cv {
class Mat1b { public: unsigned char* data = nullptr; int rows = 0, cols = 0; };
class Mat1i { public: int* data = nullptr; int rows = 0, cols = 0; };
}  // namespace cv

class PerformanceEvaluator {
public:
    void start() {}
    double stop() { return 0.0; }
    double last() { return 0.0; }
    void store(const std::string&, double) {}
};

enum StepType { ALLOC_DEALLOC = 0, FIRST_SCAN = 1, SECOND_SCAN = 2, ALL_SCANS = 3 };
inline std::string Step(StepType) { return ""; }

template <Connectivity2D Conn>
class GpuLabeling2D {
public:
    // Declaration order matters: reference members below bind to these
    // storage members (both zero-initialized), so storages must be
    // declared first (C++ initializes members in declaration order).
    cv::cuda::GpuMat img_storage_, labels_storage_;
    cv::cuda::GpuMat& d_img_ = img_storage_;
    cv::cuda::GpuMat& d_img_labels_ = labels_storage_;

    cv::Mat1b img_host_storage_;
    cv::Mat1i labels_host_storage_;
    cv::Mat1b& img_ = img_host_storage_;
    cv::Mat1i& img_labels_ = labels_host_storage_;

    PerformanceEvaluator perf_;

    GpuLabeling2D() = default;
    virtual ~GpuLabeling2D() = default;

    virtual void PerformLabeling() { }
    virtual void PerformLabelingWithSteps() { }
    virtual void PerformLabelingBlocksize(int, int, int) { }
    virtual std::string CheckAlg() const { return ""; }
    virtual bool IsLabelBackground() const { return false; }
    virtual std::string GetTitle() const { return ""; }
};

#endif  // YACCLAB_SHIM_LABELING_ALGORITHMS_H_
