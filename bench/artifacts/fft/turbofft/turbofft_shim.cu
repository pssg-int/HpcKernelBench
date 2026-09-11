// turbofft_shim.cu -- thin extern "C" launcher wrapping TurboFFT's OWN
// codegen'd, unmodified radix-2 forward-C2C kernels. NOT part of the
// TurboFFT artifact (same pattern as this project's other *_shim.cu files,
// e.g. artifacts/bfs/efg/efg_shim.cu) -- ARTIFACT_GUIDE.md rule 1: wrap the
// kernel, not the paper's benchmark script (main.cu).
//
// Scope: single-kernel-launch entries only, i.e. the rows of TurboFFT's own
// per-GPU parameter table where row[1] (num decomposition stages) == 1:
//   source/TurboFFT/include/param/A100/param_float2.csv, rows 1-13
//   (logN 1..13, N = 2..8192). Rows 14+ decompose ONE big 1D transform into
// 2-3 sequential kernel launches (Cooley-Tukey four-step, see main.cu's
// `kernel_launch_times` loop) -- out of scope for this gate-sized shim, not
// because they don't work, but because wrapping a multi-launch pipeline
// correctly needs its own scratch-buffer bookkeeping that isn't needed to
// answer "does the codegen'd kernel compute a correct FFT" for the smoke
// gate. logN 14+ therefore fails available-N validation on the Python side
// (adapter.py), not here.
//
// This shim only calls the PLAIN (if_thread_ft=0, if_ft=0,
// if_err_injection=0) TurboFFT_Kernel_Entry<float2,0,0,0,80> specialization
// -- fault-tolerant/error-injection variants are a different paper claim
// (checksum-fused FFT), not wrapped here.
//
// Grid/block/shared-mem formula and kernel argument order are copied
// VERBATIM from TurboFFT's own source/TurboFFT/main.cu
// (test_turbofft<>, the kernel_launch_times==1 case) -- this is the
// artifact's own launch convention, not something we invented.

#include <cuda_runtime.h>

#include "TurboFFT/include/turbofft/macro_ops.h"
#include "TurboFFT/include/TurboFFT_radix_2_template.h"

#include "TurboFFT/include/code_gen/generated/float2/fft_radix_2_logN_1_upload_0.cuh"
#include "TurboFFT/include/code_gen/generated/float2/fft_radix_2_logN_2_upload_0.cuh"
#include "TurboFFT/include/code_gen/generated/float2/fft_radix_2_logN_3_upload_0.cuh"
#include "TurboFFT/include/code_gen/generated/float2/fft_radix_2_logN_4_upload_0.cuh"
#include "TurboFFT/include/code_gen/generated/float2/fft_radix_2_logN_5_upload_0.cuh"
#include "TurboFFT/include/code_gen/generated/float2/fft_radix_2_logN_6_upload_0.cuh"
#include "TurboFFT/include/code_gen/generated/float2/fft_radix_2_logN_7_upload_0.cuh"
#include "TurboFFT/include/code_gen/generated/float2/fft_radix_2_logN_8_upload_0.cuh"
#include "TurboFFT/include/code_gen/generated/float2/fft_radix_2_logN_9_upload_0.cuh"
#include "TurboFFT/include/code_gen/generated/float2/fft_radix_2_logN_10_upload_0.cuh"
#include "TurboFFT/include/code_gen/generated/float2/fft_radix_2_logN_11_upload_0.cuh"
#include "TurboFFT/include/code_gen/generated/float2/fft_radix_2_logN_12_upload_0.cuh"
#include "TurboFFT/include/code_gen/generated/float2/fft_radix_2_logN_13_upload_0.cuh"

// TurboFFT's own per-(N) launch shape, param_float2.csv rows 1-13:
// column order there is (id, ndims, dim_log2, ..., threadblock_bs, ...,
// WorkerFFTSize, ...); ndims==1 rows only.
static const int TB_BS[14]   = {0, 32, 16, 8, 4, 4, 2, 4, 4, 1, 1, 1, 1, 1};
static const int WORKER[14]  = {0, 2,  4,  8, 4, 8, 8, 8, 16, 8, 16, 16, 16, 32};

typedef void (*fft_kernel_t)(float2*, float2*, float2*, float2*, int, int);

static fft_kernel_t kernel_for_logn(int logN) {
    switch (logN) {
        case 1:  return fft_radix_2<float2, 1,  0, 0, 0, 0>;
        case 2:  return fft_radix_2<float2, 2,  0, 0, 0, 0>;
        case 3:  return fft_radix_2<float2, 3,  0, 0, 0, 0>;
        case 4:  return fft_radix_2<float2, 4,  0, 0, 0, 0>;
        case 5:  return fft_radix_2<float2, 5,  0, 0, 0, 0>;
        case 6:  return fft_radix_2<float2, 6,  0, 0, 0, 0>;
        case 7:  return fft_radix_2<float2, 7,  0, 0, 0, 0>;
        case 8:  return fft_radix_2<float2, 8,  0, 0, 0, 0>;
        case 9:  return fft_radix_2<float2, 9,  0, 0, 0, 0>;
        case 10: return fft_radix_2<float2, 10, 0, 0, 0, 0>;
        case 11: return fft_radix_2<float2, 11, 0, 0, 0, 0>;
        case 12: return fft_radix_2<float2, 12, 0, 0, 0, 0>;
        case 13: return fft_radix_2<float2, 13, 0, 0, 0, 0>;
        default: return nullptr;
    }
}

extern "C" {

// Returns 0 on success, negative on rejection/failure:
//  -1  N is not a power of two in [2, 8192] (logN 1..13)
//  -2  bs is not a multiple of TurboFFT's own threadblock batch size for N
//  -3  kernel launch / cudaFuncSetAttribute failed (see stderr)
// d_input / d_output: device pointers to interleaved-float32 (float2)
// arrays of N*bs complex elements each (out-of-place; may NOT alias).
int turbofft_forward_c2c_f32(const float* d_input, float* d_output,
                              long long N, long long bs, int thread_bs) {
    int logN = 0;
    while ((1LL << logN) < N && logN < 32) logN++;
    if (logN < 1 || logN > 13 || (1LL << logN) != N) return -1;

    int tb_bs = TB_BS[logN];
    int worker = WORKER[logN];
    if (bs % tb_bs != 0) return -2;
    if (thread_bs < 1) thread_bs = 1;

    fft_kernel_t kernel = kernel_for_logn(logN);
    if (!kernel) return -1;

    long long shared_size = N * (long long)tb_bs * (long long)sizeof(float2);
    long long blockdim = (N * (long long)tb_bs) / worker;
    long long griddim = ((N * bs + N * (long long)tb_bs - 1) / (N * (long long)tb_bs)) / thread_bs;
    if (griddim < 1) griddim = 1;

    cudaError_t attr_err = cudaFuncSetAttribute(
        (const void*)kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)shared_size);
    if (attr_err != cudaSuccess) return -3;

    kernel<<<(unsigned int)griddim, (unsigned int)blockdim, (size_t)shared_size>>>(
        (float2*)d_input, (float2*)d_output, /*twiddle=*/nullptr, /*checksum=*/nullptr,
        (int)bs, thread_bs);

    cudaError_t launch_err = cudaGetLastError();
    if (launch_err != cudaSuccess) return -3;
    return 0;
}

// Cheap query so the adapter never re-derives TurboFFT's own table in two
// places: returns TurboFFT's threadblock batch size for N (0 if N is out of
// this shim's supported range).
int turbofft_tb_bs_for_N(long long N) {
    int logN = 0;
    while ((1LL << logN) < N && logN < 32) logN++;
    if (logN < 1 || logN > 13 || (1LL << logN) != N) return 0;
    return TB_BS[logN];
}

}  // extern "C"
