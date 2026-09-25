// ctypes-callable bridge for the cuDNN stencil baseline that the Tensor-Core
// stencil papers compare against (ConvStencil PPoPP'24, LoRAStencil SC'24,
// FlashFFTStencil PPoPP'25, SPIDER PPoPP'26 all name "cuDNN").
//
// NOT part of any artifact: a port of the call sequence in ConvStencil's own
// baseline programs, source/src/cudnn/conv_{1d3p,1d5p,box2d9p,box2d49p,
// box3d27p}.cu (github.com/microsoft/ConvStencil @ 89688a1; SPIDER's
// scripts/Figure10_run.sh runs the same programs from its ConvStencil
// submodule). What is kept verbatim from those files:
//
//   - one cudnnConvolutionForward per time step, ping-ponging two device
//     buffers (data[t % 2] -> data[(t + 1) % 2]), no host transfer and no
//     synchronization inside the loop;
//   - batch 1, one input and one output channel, CUDNN_CROSS_CORRELATION,
//     stride 1, dilation 1, zero padding of `radius` on every spatial axis
//     (so the output has the input's size);
//   - algorithm CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_PRECOMP_GEMM (hard-coded
//     there; cudnnFindConvolutionForwardAlgorithm is commented out), math
//     type CUDNN_TENSOR_OP_MATH_ALLOW_CONVERSION;
//   - fp64: CUDNN_DATA_DOUBLE data and compute type.
//
// What differs, and why:
//   - the programs hard-code grid, T and an all-equal filter (0.1111); here
//     they come from the harness workload (any 1D/2D/3D shape, radius and
//     weights), so the correctness gate can check the result;
//   - the 1D/2D programs describe the tensors as 4-D NHWC with C = 1, the
//     3-D one as 5-D Nd; with one channel NHWC and NCHW are the same memory
//     layout, so this bridge uses packed Nd descriptors for every rank;
//   - fp16 (not in the papers' fp64 baseline runs; added so SPIDER's fp16
//     result has a same-precision library comparator): CUDNN_DATA_HALF data
//     with CUDNN_DATA_FLOAT compute, and float alpha/beta as cuDNN requires
//     for half data (the half programs in SPIDER's ConvStencil fork pass a
//     `half` alpha/beta, which cuDNN reads as float);
//   - run() first resets buffer 0 from a pristine device copy of the initial
//     field (one D2D copy per call, the same reset every other stencil
//     adapter does), so every timed call computes the same T-step sweep.
//
// algo_mode 1 ("fastest") is FlashFFTStencil's cuDNN baseline instead
// (benchmarks/cudnn/cudnn-test.cpp @ 4579ea1): it times every forward
// algorithm cuDNN accepts for the descriptors (2 warm-up + RUNS calls each,
// CUDA events; unsupported or out-of-memory algorithms skipped) and keeps the
// fastest. Here that search runs once in prepare() (untimed preprocessing)
// with FASTEST_RUNS calls per algorithm, and the chosen algorithm is reported.
//
// Boundary: cuDNN's zero padding is a zero-valued halo outside the grid and
// every cell is recomputed each step -- the domain's "zero-halo" convention
// (kernelbench/domains/stencil.py), which the adapter selects.

#include <cuda_runtime.h>
#include <cudnn.h>

#include <cstdarg>
#include <cstdio>
#include <cstdlib>
#include <cstring>

namespace {

char g_err[1024] = "";
int g_err_code = 0;  // 0 ok, 1 configuration not supported by cuDNN, 2 other error

void set_err(int code, const char *fmt, ...) {
    g_err_code = code;
    va_list ap;
    va_start(ap, fmt);
    vsnprintf(g_err, sizeof g_err, fmt, ap);
    va_end(ap);
    fprintf(stderr, "cudnn_stencil: %s\n", g_err);
}

bool not_supported(cudnnStatus_t s) {
    if (s == CUDNN_STATUS_NOT_SUPPORTED) return true;
#if CUDNN_MAJOR >= 9
    // cuDNN 9 reports NOT_SUPPORTED sub-categories as 3000-3999
    if ((int)s >= 3000 && (int)s < 4000) return true;
#endif
    return false;
}

struct Handle {
    cudnnHandle_t cudnn = nullptr;
    cudnnTensorDescriptor_t xdesc = nullptr;
    cudnnFilterDescriptor_t wdesc = nullptr;
    cudnnConvolutionDescriptor_t cdesc = nullptr;
    cudnnConvolutionFwdAlgo_t algo = CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_PRECOMP_GEMM;
    cudnnDataType_t dtype = CUDNN_DATA_DOUBLE;
    void *init = nullptr, *buf[2] = {nullptr, nullptr}, *filter = nullptr, *ws = nullptr;
    size_t bytes = 0, ws_bytes = 0;
    int cur = 0;
};

void destroy(Handle *h) {
    if (!h) return;
    cudaFree(h->init);
    cudaFree(h->buf[0]);
    cudaFree(h->buf[1]);
    cudaFree(h->filter);
    cudaFree(h->ws);
    if (h->cdesc) cudnnDestroyConvolutionDescriptor(h->cdesc);
    if (h->wdesc) cudnnDestroyFilterDescriptor(h->wdesc);
    if (h->xdesc) cudnnDestroyTensorDescriptor(h->xdesc);
    if (h->cudnn) cudnnDestroy(h->cudnn);
    delete h;
}

const int FASTEST_RUNS = 10;

// FlashFFTStencil's cudnn-test.cpp search: time every algorithm cuDNN
// accepts (2 warm-up + FASTEST_RUNS calls, buf[0] -> buf[1]) and keep the
// fastest in h->algo. Returns false (with the error set) if none ran.
bool pick_fastest(Handle *h) {
    const double a64 = 1.0, b64 = 0.0;
    const float a32 = 1.0f, b32 = 0.0f;
    const void *alpha = h->dtype == CUDNN_DATA_DOUBLE ? (const void *)&a64 : (const void *)&a32;
    const void *beta = h->dtype == CUDNN_DATA_DOUBLE ? (const void *)&b64 : (const void *)&b32;
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    float best_ms = -1.0f;
    for (int a = 0; a < CUDNN_CONVOLUTION_FWD_ALGO_COUNT; ++a) {
        const cudnnConvolutionFwdAlgo_t algo = (cudnnConvolutionFwdAlgo_t)a;
        size_t ws_bytes = 0;
        if (cudnnGetConvolutionForwardWorkspaceSize(h->cudnn, h->xdesc, h->wdesc, h->cdesc,
                                                    h->xdesc, algo, &ws_bytes)
            != CUDNN_STATUS_SUCCESS)
            continue;
        void *ws = nullptr;
        if (ws_bytes && cudaMalloc(&ws, ws_bytes) != cudaSuccess) {
            cudaGetLastError();  // clear the out-of-memory error, skip this algorithm
            continue;
        }
        bool ok = true;
        for (int i = 0; ok && i < 2 + FASTEST_RUNS; ++i) {
            if (i == 2) cudaEventRecord(start, 0);
            ok = cudnnConvolutionForward(h->cudnn, alpha, h->xdesc, h->buf[0], h->wdesc,
                                         h->filter, h->cdesc, algo, ws, ws_bytes, beta,
                                         h->xdesc, h->buf[1]) == CUDNN_STATUS_SUCCESS;
        }
        float ms = -1.0f;
        if (ok) {
            cudaEventRecord(stop, 0);
            cudaEventSynchronize(stop);
            cudaEventElapsedTime(&ms, start, stop);
            fprintf(stderr, "cudnn_stencil: algo %d  %.4f ms/call\n", a, ms / FASTEST_RUNS);
        }
        cudaDeviceSynchronize();
        cudaFree(ws);
        if (ok && (best_ms < 0.0f || ms < best_ms)) {
            best_ms = ms;
            h->algo = algo;
        }
    }
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    if (best_ms < 0.0f) {
        set_err(1, "no cuDNN forward algorithm ran for this configuration");
        return false;
    }
    return true;
}

}  // namespace

#define CK_CUDNN(expr)                                                              \
    do {                                                                            \
        cudnnStatus_t s_ = (expr);                                                  \
        if (s_ != CUDNN_STATUS_SUCCESS) {                                           \
            set_err(not_supported(s_) ? 1 : 2, "%s failed: %s", #expr,              \
                    cudnnGetErrorString(s_));                                       \
            destroy(h);                                                             \
            return nullptr;                                                         \
        }                                                                           \
    } while (0)

#define CK_CUDA(expr)                                                               \
    do {                                                                            \
        cudaError_t e_ = (expr);                                                    \
        if (e_ != cudaSuccess) {                                                    \
            set_err(2, "%s failed: %s", #expr, cudaGetErrorString(e_));             \
            destroy(h);                                                             \
            return nullptr;                                                         \
        }                                                                           \
    } while (0)

extern "C" {

const char *cudnn_stencil_last_error(void) { return g_err; }
int cudnn_stencil_last_error_code(void) { return g_err_code; }
size_t cudnn_stencil_version(void) { return cudnnGetVersion(); }

int cudnn_stencil_algo(void *handle) { return (int)static_cast<Handle *>(handle)->algo; }

// field: dims-rank grid, row-major, `precision` 0 = double, 1 = half (raw
// 16-bit values); kernel: (2r+1)^dims cross-correlation weights, same type.
// algo_mode: 0 = IMPLICIT_PRECOMP_GEMM (ConvStencil's programs), 1 = fastest
// measured forward algorithm (FlashFFTStencil's cudnn-test.cpp).
void *cudnn_stencil_prepare(const void *field, const void *kernel, int dims,
                            const int *shape, int radius, int precision, int algo_mode) {
    g_err[0] = '\0';
    g_err_code = 0;
    Handle *h = new Handle();
    if (dims < 1 || dims > 3 || radius < 0 || (precision != 0 && precision != 1)) {
        set_err(1, "unsupported dims=%d radius=%d precision=%d", dims, radius, precision);
        destroy(h);
        return nullptr;
    }
    h->dtype = precision == 0 ? CUDNN_DATA_DOUBLE : CUDNN_DATA_HALF;
    const cudnnDataType_t compute = precision == 0 ? CUDNN_DATA_DOUBLE : CUDNN_DATA_FLOAT;
    const size_t elt = precision == 0 ? sizeof(double) : 2;

    // Tensor/filter ranks: 4-D (N, C, H, W) for 1D and 2D grids, as the
    // papers' 1D/2D programs; 5-D (N, C, D, H, W) for 3D, as conv_box3d27p.
    const int nb = dims == 3 ? 5 : 4;
    const int nsp = nb - 2;  // spatial axes
    int tdim[5], tstride[5], fdim[5], pad[3], one[3] = {1, 1, 1};
    const int k = 2 * radius + 1;
    tdim[0] = tdim[1] = fdim[0] = fdim[1] = 1;
    for (int i = 0; i < nsp; ++i) {
        // a 1D grid is H = 1, W = N with a 1 x k filter and padding (0, r),
        // as conv_1d3p.cu / conv_1d5p.cu
        const bool dummy = dims == 1 && i == 0;
        tdim[2 + i] = dummy ? 1 : shape[i - (dims == 1 ? 1 : 0)];
        fdim[2 + i] = dummy ? 1 : k;
        pad[i] = dummy ? 0 : radius;
    }
    size_t cells = 1;
    for (int i = nb - 1; i >= 0; --i) {
        tstride[i] = (int)cells;
        cells *= (size_t)tdim[i];
    }
    h->bytes = cells * elt;

    CK_CUDNN(cudnnCreate(&h->cudnn));
    CK_CUDNN(cudnnCreateTensorDescriptor(&h->xdesc));
    CK_CUDNN(cudnnSetTensorNdDescriptor(h->xdesc, h->dtype, nb, tdim, tstride));
    CK_CUDNN(cudnnCreateFilterDescriptor(&h->wdesc));
    CK_CUDNN(cudnnSetFilterNdDescriptor(h->wdesc, h->dtype, CUDNN_TENSOR_NCHW, nb, fdim));
    CK_CUDNN(cudnnCreateConvolutionDescriptor(&h->cdesc));
    CK_CUDNN(cudnnSetConvolutionNdDescriptor(h->cdesc, nsp, pad, one, one,
                                             CUDNN_CROSS_CORRELATION, compute));
    CK_CUDNN(cudnnSetConvolutionMathType(h->cdesc, CUDNN_TENSOR_OP_MATH_ALLOW_CONVERSION));

    int odim[5];
    CK_CUDNN(cudnnGetConvolutionNdForwardOutputDim(h->cdesc, h->xdesc, h->wdesc, nb, odim));
    for (int i = 0; i < nb; ++i) {
        if (odim[i] != tdim[i]) {
            set_err(2, "output dim %d is %d, expected %d (same-size convolution)",
                    i, odim[i], tdim[i]);
            destroy(h);
            return nullptr;
        }
    }
    size_t kbytes = elt;
    for (int i = 0; i < nsp; ++i) kbytes *= (size_t)fdim[2 + i];
    CK_CUDA(cudaMalloc(&h->init, h->bytes));
    CK_CUDA(cudaMalloc(&h->buf[0], h->bytes));
    CK_CUDA(cudaMalloc(&h->buf[1], h->bytes));
    CK_CUDA(cudaMalloc(&h->filter, kbytes));
    CK_CUDA(cudaMemcpy(h->init, field, h->bytes, cudaMemcpyHostToDevice));
    CK_CUDA(cudaMemcpy(h->buf[0], h->init, h->bytes, cudaMemcpyDeviceToDevice));
    CK_CUDA(cudaMemcpy(h->filter, kernel, kbytes, cudaMemcpyHostToDevice));

    if (algo_mode == 1 && !pick_fastest(h)) {
        destroy(h);
        return nullptr;
    }
    CK_CUDNN(cudnnGetConvolutionForwardWorkspaceSize(h->cudnn, h->xdesc, h->wdesc, h->cdesc,
                                                     h->xdesc, h->algo, &h->ws_bytes));
    if (h->ws_bytes) CK_CUDA(cudaMalloc(&h->ws, h->ws_bytes));
    CK_CUDA(cudaDeviceSynchronize());
    return h;
}

// Timed: reset buffer 0, then T forward convolutions on the default stream.
// Returns 0 on success (launch errors only; the timer's stop event syncs).
int cudnn_stencil_run(void *handle, int timesteps) {
    Handle *h = static_cast<Handle *>(handle);
    cudaError_t e = cudaMemcpyAsync(h->buf[0], h->init, h->bytes, cudaMemcpyDeviceToDevice, 0);
    if (e != cudaSuccess) {
        set_err(2, "reset copy failed: %s", cudaGetErrorString(e));
        return 2;
    }
    const double a64 = 1.0, b64 = 0.0;
    const float a32 = 1.0f, b32 = 0.0f;
    const void *alpha = h->dtype == CUDNN_DATA_DOUBLE ? (const void *)&a64 : (const void *)&a32;
    const void *beta = h->dtype == CUDNN_DATA_DOUBLE ? (const void *)&b64 : (const void *)&b32;
    for (int t = 0; t < timesteps; ++t) {
        cudnnStatus_t s = cudnnConvolutionForward(
            h->cudnn, alpha, h->xdesc, h->buf[t % 2], h->wdesc, h->filter, h->cdesc,
            h->algo, h->ws, h->ws_bytes, beta, h->xdesc, h->buf[(t + 1) % 2]);
        if (s != CUDNN_STATUS_SUCCESS) {
            set_err(2, "cudnnConvolutionForward (step %d) failed: %s", t,
                    cudnnGetErrorString(s));
            return 2;
        }
    }
    h->cur = timesteps % 2;
    return 0;
}

int cudnn_stencil_copy_out(void *handle, void *out) {
    Handle *h = static_cast<Handle *>(handle);
    cudaError_t e = cudaMemcpy(out, h->buf[h->cur], h->bytes, cudaMemcpyDeviceToHost);
    if (e != cudaSuccess) {
        set_err(2, "copy_out failed: %s", cudaGetErrorString(e));
        return 2;
    }
    return 0;
}

void cudnn_stencil_free(void *handle) { destroy(static_cast<Handle *>(handle)); }

}  // extern "C"
