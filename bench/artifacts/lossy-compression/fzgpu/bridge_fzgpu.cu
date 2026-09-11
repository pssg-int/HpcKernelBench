// Thin ctypes-callable bridge around FZ-GPU's own compress/decompress
// kernels (source/src/fz.cu).
//
// FZ-GPU ships no library entry point at all -- only a CLI binary whose
// single driver function, `runFzgpu()` (source/src/fz.cu), reads a file,
// then runs compression AND decompression back-to-back inside one
// function body with no way to call either half alone (the README says so
// explicitly: "Currently, FZ-GPU performs compression and decompression
// together, but we plan to provide options for performing compression and
// decompression separately in the future"). This benchmark's contract
// needs compression timed alone (run() = ONE compress call) and
// decompression only for the untimed gate, so this file splits
// `runFzgpu()`'s body at its own internal boundary (its own
// `compressionStart`/`compressionEnd`/`decompressionStart`/
// `decompressionEnd` timestamps already mark exactly where compression
// ends and decompression begins, source/src/fz.cu lines 461-481) into two
// halves, reusing every kernel call and buffer-sizing computation
// byte-for-byte from that function -- no kernel code modified, only
// reorganized across prepare()/run()/free() instead of being bundled with
// file I/O and an internal compress-then-decompress sequence that always
// runs both together.
//
// One deliberate deviation from `runFzgpu()`'s own math, NOT a kernel
// change: `runFzgpu()` computes its error bound as `eb * range` where
// `range` is `max-min` over ITS OWN zero-padded host buffer (padding
// pollutes that computation for our chunk-aligned padding, since our real
// data doesn't come from a file of exactly the padded size). This bridge
// sidesteps that entirely by passing the ALREADY-ABSOLUTE error bound
// (`w.correctness_tolerance` from the harness -- see adapter.py) directly
// into `launch_construct_LorenzoI_var`/`launch_reconstruct_LorenzoI_var`'s
// `eb` parameter, which both expect an absolute value (that is exactly
// what `eb * range` computes in the original). This is a change to HOW
// the absolute bound is computed (in Python, from the real unpadded data,
// matching every other implementation in this benchmark), not to what the
// kernels DO with it once they have it.
#include "fzgpu_handle.h"

#include <cstring>

#define main fzgpu_unused_main
#include "source/src/fz.cu"
#undef main

extern "C" {

// Allocates every device buffer `runFzgpu()` allocates (source/src/fz.cu
// lines 434-444), H2D-copies the real `dataTypeLen` values into the front
// of the chunk-aligned padded buffer and zero-fills the pad tail
// (matching FZ-GPU's own `read_binary_to_new_array`'s value-initialized
// `new T[]()`, source/src/fz.cu lines 89-100) -- called once, timed once
// by adapter.py::prepare() as preprocessing.
void *fzgpuc_prepare(int dimx, int dimy, int dimz, const float *host_data, double abs_eb) {
    FzgpuHandle *h = new FzgpuHandle();
    h->dimx = dimx; h->dimy = dimy; h->dimz = dimz;
    h->dataTypeLen = dimx * dimy * dimz;
    h->abs_eb = abs_eb;

    const int blockSize = 16;
    int quantizationCodeByteLen = h->dataTypeLen * 2;
    quantizationCodeByteLen = (quantizationCodeByteLen % 4096 == 0)
        ? quantizationCodeByteLen
        : quantizationCodeByteLen - quantizationCodeByteLen % 4096 + 4096;
    h->quantizationCodeByteLen = quantizationCodeByteLen;
    h->paddingDataTypeLen = quantizationCodeByteLen / 2;
    h->dataChunkSize = (quantizationCodeByteLen % (blockSize * UINT32_BIT_LEN) == 0)
        ? quantizationCodeByteLen / (blockSize * UINT32_BIT_LEN)
        : quantizationCodeByteLen / (blockSize * UINT32_BIT_LEN) + 1;
    h->gridX = (int)(h->paddingDataTypeLen / 2048);

    const size_t padN = (size_t)h->paddingDataTypeLen;
    const size_t nchunk4k = (size_t)(h->quantizationCodeByteLen / 4096);

    CHECK_CUDA(cudaMalloc((void **)&h->deviceInput, sizeof(float) * padN));
    CHECK_CUDA(cudaMalloc((void **)&h->deviceQuantizationCode, sizeof(uint16_t) * padN));
    CHECK_CUDA(cudaMalloc((void **)&h->deviceSignNum, sizeof(bool) * padN));
    CHECK_CUDA(cudaMalloc((void **)&h->deviceCompressedOutput, sizeof(uint16_t) * padN));
    CHECK_CUDA(cudaMalloc((void **)&h->deviceBitFlagArr, sizeof(uint32_t) * h->dataChunkSize));
    CHECK_CUDA(cudaMalloc((void **)&h->deviceDecompressedQuantizationCode, sizeof(uint16_t) * padN));
    CHECK_CUDA(cudaMalloc((void **)&h->deviceDecompressedOutput, sizeof(float) * padN));
    CHECK_CUDA(cudaMalloc((void **)&h->deviceOffsetCounter, sizeof(uint32_t)));
    CHECK_CUDA(cudaMalloc((void **)&h->deviceStartPosition, sizeof(uint32_t) * nchunk4k));
    CHECK_CUDA(cudaMalloc((void **)&h->deviceCompressedSize, sizeof(uint32_t) * nchunk4k));

    CHECK_CUDA(cudaMemset(h->deviceInput, 0, sizeof(float) * padN));  // zero-pad tail
    CHECK_CUDA(cudaMemcpy(h->deviceInput, host_data, sizeof(float) * h->dataTypeLen,
                          cudaMemcpyHostToDevice));

    return h;
}

// ONE compress call: `launch_construct_LorenzoI_var` (dimension-aware
// Lorenzo predictor/quantizer) + `compressionFusedKernel` (bitshuffle +
// lossless encode) -- exactly `runFzgpu()`'s own compression half
// (source/src/fz.cu lines 461-469), both kernels unmodified.
//
// Re-zeroes every accumulator/output buffer the fused kernel writes
// through `atomicAdd` (`deviceOffsetCounter`) or per-chunk indexing
// (`deviceBitFlagArr`/`deviceStartPosition`/`deviceCompressedSize`/
// `deviceQuantizationCode`) immediately before every launch -- required
// for correctness on repeated independent calls, the same category of fix
// as this benchmark's other GPU artifacts (cb-spmv's `d_y` re-zero,
// pfpl-compress's `d_fullcarry`/`g_chunk_counter` re-zero).
// `runFzgpu()` itself only zeroes these ONCE (before its single
// compress-then-decompress pass); our harness calls run() independently
// many times (an isolated correctness check, then warmup, then measured
// reps), so the reset moves here.
void fzgpuc_run(void *handle) {
    FzgpuHandle *h = reinterpret_cast<FzgpuHandle *>(handle);
    const size_t padN = (size_t)h->paddingDataTypeLen;
    const size_t nchunk4k = (size_t)(h->quantizationCodeByteLen / 4096);

    CHECK_CUDA(cudaMemset(h->deviceQuantizationCode, 0, sizeof(uint16_t) * padN));
    CHECK_CUDA(cudaMemset(h->deviceBitFlagArr, 0, sizeof(uint32_t) * h->dataChunkSize));
    CHECK_CUDA(cudaMemset(h->deviceOffsetCounter, 0, sizeof(uint32_t)));
    CHECK_CUDA(cudaMemset(h->deviceStartPosition, 0, sizeof(uint32_t) * nchunk4k));
    CHECK_CUDA(cudaMemset(h->deviceCompressedSize, 0, sizeof(uint32_t) * nchunk4k));

    dim3 inputDimension(h->dimx, h->dimy, h->dimz);
    dim3 block(32, 32);
    dim3 grid(h->gridX);
    float timeElapsed;  // discarded -- the harness's own CudaEventTimer brackets this call

    cusz::experimental::launch_construct_LorenzoI_var<float, uint16_t, float>(
        h->deviceInput, h->deviceQuantizationCode, h->deviceSignNum,
        inputDimension, h->abs_eb, timeElapsed, /*stream=*/0);

    compressionFusedKernel<<<grid, block>>>(
        (uint32_t *)h->deviceQuantizationCode, (uint32_t *)h->deviceCompressedOutput,
        h->deviceOffsetCounter, h->deviceBitFlagArr, h->deviceStartPosition,
        h->deviceCompressedSize);

    cudaDeviceSynchronize();
}

// achieved compressed byte count -- exactly runFzgpu()'s own formula
// (source/src/fz.cu lines 559-562), D2H of the 4-byte atomic offset
// counter plus the (already-known) fixed metadata overhead.
size_t fzgpuc_compressed_bytes(void *handle) {
    FzgpuHandle *h = reinterpret_cast<FzgpuHandle *>(handle);
    uint32_t offsetSum = 0;
    CHECK_CUDA(cudaMemcpy(&offsetSum, h->deviceOffsetCounter, sizeof(uint32_t),
                          cudaMemcpyDeviceToHost));
    const size_t nchunk4k = (size_t)(h->quantizationCodeByteLen / 4096);
    return sizeof(uint32_t) * (size_t)h->dataChunkSize
         + (size_t)offsetSum * sizeof(uint32_t)
         + sizeof(uint32_t) * nchunk4k;
}

void fzgpuc_free(void *handle) {
    FzgpuHandle *h = reinterpret_cast<FzgpuHandle *>(handle);
    if (!h) return;
    cudaFree(h->deviceInput);
    cudaFree(h->deviceQuantizationCode);
    cudaFree(h->deviceSignNum);
    cudaFree(h->deviceCompressedOutput);
    cudaFree(h->deviceBitFlagArr);
    cudaFree(h->deviceDecompressedQuantizationCode);
    cudaFree(h->deviceDecompressedOutput);
    cudaFree(h->deviceOffsetCounter);
    cudaFree(h->deviceStartPosition);
    cudaFree(h->deviceCompressedSize);
    delete h;
}

// Correctness gate only, called from adapter.py::to_host()/free(),
// DELIBERATELY outside the timed region: `decompressionFusedKernel` +
// `launch_reconstruct_LorenzoI_var` -- runFzgpu()'s own decompression half
// (source/src/fz.cu lines 472-481), unmodified -- then D2H-copies just the
// first `dataTypeLen` (real, unpadded) reconstructed values.
int fzgpud_run_and_copy(void *handle, float *host_out) {
    FzgpuHandle *h = reinterpret_cast<FzgpuHandle *>(handle);
    const size_t padN = (size_t)h->paddingDataTypeLen;

    CHECK_CUDA(cudaMemset(h->deviceDecompressedQuantizationCode, 0, sizeof(uint16_t) * padN));
    CHECK_CUDA(cudaMemset(h->deviceDecompressedOutput, 0, sizeof(float) * padN));

    dim3 inputDimension(h->dimx, h->dimy, h->dimz);
    dim3 block(32, 32);
    dim3 grid(h->gridX);
    float timeElapsed;

    decompressionFusedKernel<<<grid, block>>>(
        (uint32_t *)h->deviceCompressedOutput, (uint32_t *)h->deviceDecompressedQuantizationCode,
        h->deviceBitFlagArr, h->deviceStartPosition);

    cusz::experimental::launch_reconstruct_LorenzoI_var<float, uint16_t, float>(
        h->deviceSignNum, h->deviceDecompressedQuantizationCode, h->deviceDecompressedOutput,
        inputDimension, h->abs_eb, timeElapsed, /*stream=*/0);

    cudaDeviceSynchronize();
    CHECK_CUDA(cudaMemcpy(host_out, h->deviceDecompressedOutput,
                          sizeof(float) * h->dataTypeLen, cudaMemcpyDeviceToHost));
    return h->dataTypeLen;
}

}  // extern "C"
