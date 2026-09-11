// Thin ctypes-callable bridge around ZipServ's own TCA-TBE lossless weight
// codec, wrapped at the FINEST available boundary: the standalone
// decompress KERNEL (csrc/L_API.cu::BF16TripleBitmap_Decompress_API ->
// BF16TripleBitmap_Decompress_Kernel), which is genuinely separate from
// both (a) the fused decompress+GEMM "ZipGEMM" kernel
// (BF16TripleBitmap_MM_API/_Kernel, csrc/L_API.cu -- ZipServ's headline
// contribution, NOT called anywhere in this file) and (b) vLLM/KV-cache
// serving internals (third_party/vllm/, LInfer_py/backend/ -- also not
// touched). Confirmed separable by reading csrc/L_API.cu directly: the
// decompress kernel writes the full M_Global x K_Global bf16 matrix and
// takes no B/C GEMM operands at all.
//
// ZipServ's own compress step (kernel_benchmark/utils.h::
// InitBF16MatrixTripleBitmap_Host, the SAME function ZipServ's real Python
// production path calls -- LInfer_py/linfer_lib.cu's
// compress_tensor_triple_bitmap pybind11 binding, used by
// LInfer_py/compress_model.py at model-load time) is host-only CPU code
// (no CUDA kernel at all -- confirmed by reading it: no `<<<...>>>` launch
// anywhere in utils.h), matching the spec's own framing
// (benchspecs/lossless-compression/spec.yaml, variant
// lossless-comp-structured-operand-fused: "format: TCA-TBE fixed-length
// triple-bitmap encoding (given; one-shot compression not timed in this
// variant)"). This adapter therefore registers "-decompress" (not
// "-compress" like this benchmark's other lossless-compression GPU
// adapters): adapter.py::prepare() calls InitBF16MatrixTripleBitmap_Host
// once, untimed, as preprocessing; run() calls ONLY
// BF16TripleBitmap_Decompress_API, the one GPU kernel launch this domain's
// contract times.
//
// Neither InitBF16MatrixTripleBitmap_Host nor BF16TripleBitmap_Decompress_API
// is modified -- this file only allocates/copies buffers around them and
// exposes an extern "C" ABI ctypes can call.
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <vector>

#include <cuda_bf16.h>
#include <cuda_runtime.h>

// build/L_API.cuh: ZipServ's own header declaring BF16TripleBitmap_MM_API /
// BF16TripleBitmap_Decompress_API / InitBF16MatrixTripleBitmap, matching
// csrc/L_API.cu's actual definitions verbatim (this is the header
// source/build/Makefile itself builds against -- reused here unmodified).
#include "source/build/L_API.cuh"

// kernel_benchmark/utils.h: ZipServ's own CPU-side compress convenience
// function (InitBF16MatrixTripleBitmap_Host) -- the SAME function
// LInfer_py/linfer_lib.cu's real production Python binding calls. The
// header transitively includes <cublas_v2.h>/<cusparse_v2.h> (declarations
// only), but InitBF16MatrixTripleBitmap_Host itself calls neither library,
// so no -lcublas/-lcusparse is needed at link time here.
#include "source/kernel_benchmark/utils.h"

namespace {

constexpr int kTileM = 8, kTileMMedian = 16, kTileMGlobal = 64;
constexpr int kTileK = 8, kTileKMedian = 64, kTileKGlobal = 64;

}  // namespace

extern "C" {

struct ZipservHandle {
    int M = 0, K = 0;
    int num_tiles = 0, num_median_tiles = 0, num_global_tiles = 0;
    int high_freq_count = 0, full_count = 0;
    int max_high_freq_count = 0, max_full_count = 0;
    uint8_t start_exp = 0;
    uint8_t *d_sign_mantissa = nullptr;
    __nv_bfloat16 *d_compressed_full = nullptr;
    uint64_t *d_bitmap1 = nullptr, *d_bitmap2 = nullptr, *d_bitmap3 = nullptr;
    int *d_tile_offsets_median = nullptr, *d_tile_offsets_global = nullptr;
    __nv_bfloat16 *d_output = nullptr;
    size_t total_bf16_elems = 0;
    size_t real_bytes = 0;
};

// Picks the smallest (M, K) with M%64==0, K%64==0, M*K*2 >= real_bytes --
// K fixed at the minimum tile (64), M padded up. Exposed so adapter.py
// stays in sync with the SAME dims used for compress and decompress
// without duplicating the arithmetic in Python.
void zipserv_pick_dims(size_t real_bytes, int *out_M, int *out_K) {
    const size_t elements_needed = (real_bytes + 1) / 2;  // ceil(bytes/2)
    const int K = 64;
    size_t M = (elements_needed + K - 1) / K;              // ceil(elems/K)
    M = ((M + 63) / 64) * 64;                               // round up to /64
    if (M == 0) M = 64;
    *out_M = (int)M;
    *out_K = K;
}

// ZipServ's own CPU compress step (kernel_benchmark/utils.h::
// InitBF16MatrixTripleBitmap_Host, unmodified) -- host-only, no CUDA
// kernel. Builds a zero-padded host bf16 buffer from `real_bytes` of real
// data (matching this benchmark's other GPU-compression bridges'
// zero-pad-tail convention), compresses it, then H2D-copies every piece
// the decompress kernel needs. Called once, timed once by
// adapter.py::prepare() as preprocessing (ZipServ's own spec framing:
// "one-shot compression not timed in this variant").
void *zipserv_prepare(const uint8_t *host_raw_bytes, size_t real_bytes) {
    int M, K;
    zipserv_pick_dims(real_bytes, &M, &K);
    const size_t total_elems = (size_t)M * K;

    std::vector<__nv_bfloat16> A_bf16(total_elems);
    std::memset(A_bf16.data(), 0, total_elems * sizeof(__nv_bfloat16));
    const size_t copy_bytes = real_bytes < total_elems * sizeof(__nv_bfloat16)
                                   ? real_bytes
                                   : total_elems * sizeof(__nv_bfloat16);
    std::memcpy(A_bf16.data(), host_raw_bytes, copy_bytes);

    __nv_bfloat16 *top_exponents_cpu = nullptr;
    __nv_bfloat16 *compressed_full_cpu = nullptr;
    uint8_t *sign_mantissa_cpu = nullptr;
    uint64_t *bitmap1_cpu = nullptr, *bitmap2_cpu = nullptr, *bitmap3_cpu = nullptr;
    int *tile_offsets_cpu = nullptr, *tile_offsets_median_cpu = nullptr,
        *tile_offsets_global_cpu = nullptr;
    int max_high_freq_count = 0, max_full_count = 0;
    uint8_t start_exp = 0;

    // ZipServ's own compress function, unmodified (kernel_benchmark/utils.h,
    // the exact call LInfer_py/linfer_lib.cu's production Python binding
    // and kernel_benchmark/test_decompress.cu's own benchmark both make).
    const int num_global_tiles = InitBF16MatrixTripleBitmap_Host(
        A_bf16.data(), M, K, kTileM, kTileMMedian, kTileMGlobal, kTileK,
        kTileKMedian, kTileKGlobal, &top_exponents_cpu, &compressed_full_cpu,
        &sign_mantissa_cpu, &bitmap1_cpu, &bitmap2_cpu, &bitmap3_cpu,
        &tile_offsets_cpu, &tile_offsets_median_cpu, &tile_offsets_global_cpu,
        max_high_freq_count, max_full_count, start_exp);
    if (num_global_tiles <= 0) {
        return nullptr;
    }

    ZipservHandle *h = new ZipservHandle();
    h->M = M;
    h->K = K;
    h->num_tiles = (M / kTileM) * (K / kTileK);
    h->num_median_tiles = (M / kTileMMedian) * (K / kTileKMedian);
    h->num_global_tiles = num_global_tiles;
    h->high_freq_count = tile_offsets_global_cpu[num_global_tiles * 2];
    h->full_count = tile_offsets_global_cpu[num_global_tiles * 2 + 1];
    h->max_high_freq_count = max_high_freq_count;
    h->max_full_count = max_full_count;
    h->start_exp = start_exp;
    h->total_bf16_elems = total_elems;
    h->real_bytes = real_bytes;

    cudaMalloc(&h->d_sign_mantissa, (size_t)h->high_freq_count * sizeof(uint8_t));
    cudaMalloc(&h->d_compressed_full, (size_t)h->full_count * sizeof(__nv_bfloat16));
    cudaMalloc(&h->d_bitmap1, (size_t)h->num_tiles * sizeof(uint64_t));
    cudaMalloc(&h->d_bitmap2, (size_t)h->num_tiles * sizeof(uint64_t));
    cudaMalloc(&h->d_bitmap3, (size_t)h->num_tiles * sizeof(uint64_t));
    cudaMalloc(&h->d_tile_offsets_median, (size_t)h->num_median_tiles * 2 * sizeof(int));
    cudaMalloc(&h->d_tile_offsets_global, (size_t)(num_global_tiles + 1) * 2 * sizeof(int));
    cudaMalloc(&h->d_output, total_elems * sizeof(__nv_bfloat16));

    cudaMemcpy(h->d_sign_mantissa, sign_mantissa_cpu, (size_t)h->high_freq_count,
               cudaMemcpyHostToDevice);
    cudaMemcpy(h->d_compressed_full, compressed_full_cpu,
               (size_t)h->full_count * sizeof(__nv_bfloat16), cudaMemcpyHostToDevice);
    cudaMemcpy(h->d_bitmap1, bitmap1_cpu, (size_t)h->num_tiles * sizeof(uint64_t),
               cudaMemcpyHostToDevice);
    cudaMemcpy(h->d_bitmap2, bitmap2_cpu, (size_t)h->num_tiles * sizeof(uint64_t),
               cudaMemcpyHostToDevice);
    cudaMemcpy(h->d_bitmap3, bitmap3_cpu, (size_t)h->num_tiles * sizeof(uint64_t),
               cudaMemcpyHostToDevice);
    cudaMemcpy(h->d_tile_offsets_median, tile_offsets_median_cpu,
               (size_t)h->num_median_tiles * 2 * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(h->d_tile_offsets_global, tile_offsets_global_cpu,
               (size_t)(num_global_tiles + 1) * 2 * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemset(h->d_output, 0, total_elems * sizeof(__nv_bfloat16));

    std::free(top_exponents_cpu);
    std::free(compressed_full_cpu);
    std::free(sign_mantissa_cpu);
    std::free(bitmap1_cpu);
    std::free(bitmap2_cpu);
    std::free(bitmap3_cpu);
    std::free(tile_offsets_cpu);
    std::free(tile_offsets_median_cpu);
    std::free(tile_offsets_global_cpu);

    return h;
}

// ONE decompress call: BF16TripleBitmap_Decompress_API ->
// BF16TripleBitmap_Decompress_Kernel (ZipServ's own kernel, unmodified,
// csrc/L_API.cu) -- genuinely standalone, no GEMM/B/C operands. This is
// the only call the harness's CudaEventTimer ever brackets (adapter.py's
// run()). No re-zeroing needed between calls: the kernel deterministically
// overwrites every position of `d_output` from the same (unchanged)
// compressed input every time (confirmed by the exact round-trip result in
// STATUS.md).
int zipserv_run(void *handle) {
    ZipservHandle *h = reinterpret_cast<ZipservHandle *>(handle);
    if (!h) return -1;
    cudaError_t err = BF16TripleBitmap_Decompress_API(
        /*stream=*/0, h->d_sign_mantissa, h->d_compressed_full, h->d_bitmap1,
        h->d_bitmap2, h->d_bitmap3, h->d_tile_offsets_median,
        h->d_tile_offsets_global, h->max_high_freq_count, h->max_full_count,
        h->start_exp, h->d_output, h->M, h->K);
    return err == cudaSuccess ? 0 : -1;
}

// D2H copy of just the real (unpadded) `real_bytes` bytes of the
// decompressed output -- called from adapter.py::to_host(), which runs
// AFTER run()'s CudaEventTimer block has already closed (this copy is not
// part of the timed region).
int zipserv_copy_output(void *handle, uint8_t *host_out, size_t host_out_bytes) {
    ZipservHandle *h = reinterpret_cast<ZipservHandle *>(handle);
    if (!h) return -1;
    const size_t n = host_out_bytes < h->real_bytes ? host_out_bytes : h->real_bytes;
    cudaMemcpy(host_out, h->d_output, n, cudaMemcpyDeviceToHost);
    return (int)n;
}

// Achieved compressed byte count -- ZipServ's own formula
// (kernel_benchmark/utils.h::CompressBF16MatrixTripleBitmap_Host's own
// inline compression-ratio accounting, lines ~973-980): sign_mantissa +
// compressed_full (2B/elem) + 3 bitmaps (8B/tile) + median offsets
// (2 ints/tile) + global offsets (2 ints/tile, +1).
size_t zipserv_compressed_bytes(void *handle) {
    ZipservHandle *h = reinterpret_cast<ZipservHandle *>(handle);
    if (!h) return 0;
    return (size_t)h->high_freq_count * sizeof(uint8_t) +
           (size_t)h->full_count * sizeof(__nv_bfloat16) +
           (size_t)h->num_tiles * 3 * sizeof(uint64_t) +
           (size_t)h->num_median_tiles * 2 * sizeof(int) +
           (size_t)(h->num_global_tiles + 1) * 2 * sizeof(int);
}

void zipserv_free(void *handle) {
    ZipservHandle *h = reinterpret_cast<ZipservHandle *>(handle);
    if (!h) return;
    cudaFree(h->d_sign_mantissa);
    cudaFree(h->d_compressed_full);
    cudaFree(h->d_bitmap1);
    cudaFree(h->d_bitmap2);
    cudaFree(h->d_bitmap3);
    cudaFree(h->d_tile_offsets_median);
    cudaFree(h->d_tile_offsets_global);
    cudaFree(h->d_output);
    delete h;
}

}  // extern "C"
