// Thin ctypes-callable bridge around GPULZ's own compress/decompress
// kernels (source/gpulz.cu).
//
// GPULZ ships no library entry point at all -- only a CLI binary whose
// single driver function, main() (source/gpulz.cu), reads a file, then runs
// compression AND decompression back-to-back inside one function body with
// no way to call either half alone (the README says so explicitly:
// "At present, GPULZ performs compression and decompression simultaneously;
// however, we plan to offer options for performing these tasks separately
// in future iterations"). This benchmark's contract needs compression timed
// alone (run() = ONE compress pass) and decompression only for the untimed
// gate, so this file splits main()'s body at its own internal boundary (its
// own compStart/compStop/decompStart/decompStop cudaEvent markers already
// mark exactly where compression ends and decompression begins,
// source/gpulz.cu lines 450-485) into prepare()/run()/free() (compress
// path) and a separate decompress-and-copy entry point (gate-only) -- no
// kernel code modified, every kernel launch and buffer-sizing computation
// lifted verbatim from main().
//
// Same "#define main gpulz_unused_main" + "#include source/gpulz.cu" trick
// used by this benchmark's other single-file CLI GPU artifacts (fzgpu,
// pfpl): compiles GPULZ's own file verbatim into this translation unit (no
// edits under source/ at all) while renaming away its main() so it can
// coexist in a shared library alongside this file's own extern "C" API.
//
// BLOCK_SIZE/WINDOW_SIZE/INPUT_TYPE are GPULZ's own compile-time #defines
// (source/gpulz.cu lines 12-15) -- left at their shipped defaults
// (2048 bytes / 32 / uint32_t) here; the spec's block/window/symbol-length
// sweep (benchspecs/lossless-compression/spec.yaml variant
// lossless-comp-gpu-multibyte-dual-scope) is NOT implemented by this first
// integration pass (would need one rebuilt bridge.so per grid point, since
// GPULZ's own headers only support compile-time configuration of these
// three constants) -- see STATUS.md.
#include "gpulz_handle.h"

#include <cstring>

#define main gpulz_unused_main
#include "source/gpulz.cu"
#undef main

extern "C" {

// Allocates every device buffer main() allocates (source/gpulz.cu lines
// 394-406), zero-pads to GPULZ's own BLOCK_SIZE-aligned length (matching
// main()'s own cudaMemset-then-partial-H2D-copy pattern, lines 407-420),
// and sizes+allocates the cub::DeviceScan::ExclusiveSum scratch ONCE (its
// size depends only on numOfBlocks+1, not on data content, so it is safe to
// reuse across every run() call) -- called once, timed once by
// adapter.py::prepare() as preprocessing.
void *gpulzc_prepare(const uint8_t *host_data, uint32_t fileSize) {
    GpulzHandle *h = new GpulzHandle();
    h->fileSize = fileSize;
    h->paddingSize = fileSize % BLOCK_SIZE == 0 ? 0 : BLOCK_SIZE - fileSize % BLOCK_SIZE;
    h->datatypeSize = (fileSize + h->paddingSize) / sizeof(INPUT_TYPE);
    h->numOfBlocks = h->datatypeSize * sizeof(INPUT_TYPE) / BLOCK_SIZE;

    const uint32_t datatypeSize = h->datatypeSize;
    const uint32_t numOfBlocks = h->numOfBlocks;

    cudaMalloc(&h->deviceArray, fileSize + h->paddingSize);
    cudaMalloc(&h->deviceOutput, fileSize + h->paddingSize);
    cudaMalloc((void **)&h->flagArrSizeGlobal, sizeof(uint32_t) * (numOfBlocks + 1));
    cudaMalloc((void **)&h->flagArrOffsetGlobal, sizeof(uint32_t) * (numOfBlocks + 1));
    cudaMalloc((void **)&h->compressedDataSizeGlobal, sizeof(uint32_t) * (numOfBlocks + 1));
    cudaMalloc((void **)&h->compressedDataOffsetGlobal, sizeof(uint32_t) * (numOfBlocks + 1));
    cudaMalloc((void **)&h->tmpFlagArrGlobal, sizeof(uint8_t) * datatypeSize / 8);
    cudaMalloc((void **)&h->tmpCompressedDataGlobal, sizeof(INPUT_TYPE) * datatypeSize);
    cudaMalloc((void **)&h->flagArrGlobal, sizeof(uint8_t) * datatypeSize / 8);
    cudaMalloc((void **)&h->compressedDataGlobal, sizeof(INPUT_TYPE) * datatypeSize);

    cudaMemset(h->deviceArray, 0, fileSize + h->paddingSize);   // zero-pad tail
    cudaMemcpy(h->deviceArray, host_data, fileSize, cudaMemcpyHostToDevice);
    cudaMemset(h->deviceOutput, 0, fileSize + h->paddingSize);
    cudaMemset(h->flagArrSizeGlobal, 0, sizeof(uint32_t) * (numOfBlocks + 1));
    cudaMemset(h->flagArrOffsetGlobal, 0, sizeof(uint32_t) * (numOfBlocks + 1));
    cudaMemset(h->compressedDataSizeGlobal, 0, sizeof(uint32_t) * (numOfBlocks + 1));
    cudaMemset(h->compressedDataOffsetGlobal, 0, sizeof(uint32_t) * (numOfBlocks + 1));
    cudaMemset(h->tmpFlagArrGlobal, 0, sizeof(uint8_t) * datatypeSize / 8);
    cudaMemset(h->tmpCompressedDataGlobal, 0, sizeof(INPUT_TYPE) * datatypeSize);

    // size + allocate cub scratch once (main()'s own two-call
    // determine-then-allocate pattern, source/gpulz.cu lines 457-477)
    cub::DeviceScan::ExclusiveSum(h->flag_d_temp_storage, h->flag_temp_storage_bytes,
                                   h->flagArrSizeGlobal, h->flagArrOffsetGlobal, numOfBlocks + 1);
    cudaMalloc(&h->flag_d_temp_storage, h->flag_temp_storage_bytes);
    cub::DeviceScan::ExclusiveSum(h->data_d_temp_storage, h->data_temp_storage_bytes,
                                   h->compressedDataSizeGlobal, h->compressedDataOffsetGlobal,
                                   numOfBlocks + 1);
    cudaMalloc(&h->data_d_temp_storage, h->data_temp_storage_bytes);

    return h;
}

// ONE compress pass: compressKernelI + the two cub exclusive-sum scans +
// compressKernelIII -- exactly main()'s own compression half
// (source/gpulz.cu lines 450-466), all four steps unmodified.
//
// No re-zeroing needed between calls (unlike this benchmark's fzgpu-compress
// bridge, which resets atomicAdd-accumulated counters): every GPULZ output
// array here is written via a direct per-block/per-thread INDEX
// (compressedDataSizeGlobal[blockIdx.x] = ..., not += ...), so each call
// with the same (unchanged) input deterministically overwrites the same
// values -- idempotent across repeated calls, matching this handle's
// single, unchanging deviceArray.
void gpulzc_run(void *handle) {
    GpulzHandle *h = reinterpret_cast<GpulzHandle *>(handle);
    const uint32_t numOfBlocks = h->numOfBlocks;
    const int minEncodeLength = sizeof(INPUT_TYPE) == 1 ? 2 : 1;

    dim3 gridDim(numOfBlocks);
    dim3 blockDim(THREAD_SIZE);

    compressKernelI<INPUT_TYPE><<<gridDim, blockDim>>>(
        (INPUT_TYPE *)h->deviceArray, numOfBlocks, h->flagArrSizeGlobal,
        h->compressedDataSizeGlobal, h->tmpFlagArrGlobal, h->tmpCompressedDataGlobal,
        minEncodeLength);

    cub::DeviceScan::ExclusiveSum(h->flag_d_temp_storage, h->flag_temp_storage_bytes,
                                   h->flagArrSizeGlobal, h->flagArrOffsetGlobal, numOfBlocks + 1);
    cub::DeviceScan::ExclusiveSum(h->data_d_temp_storage, h->data_temp_storage_bytes,
                                   h->compressedDataSizeGlobal, h->compressedDataOffsetGlobal,
                                   numOfBlocks + 1);

    compressKernelIII<INPUT_TYPE><<<gridDim, blockDim>>>(
        numOfBlocks, h->flagArrOffsetGlobal, h->compressedDataOffsetGlobal,
        h->tmpFlagArrGlobal, h->tmpCompressedDataGlobal, h->flagArrGlobal,
        h->compressedDataGlobal);

    cudaDeviceSynchronize();
}

// achieved compressed byte count -- exactly main()'s own formula
// (source/gpulz.cu line 546): 2 uint32 per block (flag+data size-array
// overhead) plus the flag array's and compressed-data array's own final
// cumulative offsets. D2H of just those 2 scalars.
size_t gpulzc_compressed_bytes(void *handle) {
    GpulzHandle *h = reinterpret_cast<GpulzHandle *>(handle);
    uint32_t flagLast = 0, dataLast = 0;
    cudaMemcpy(&flagLast, &h->flagArrOffsetGlobal[h->numOfBlocks], sizeof(uint32_t),
               cudaMemcpyDeviceToHost);
    cudaMemcpy(&dataLast, &h->compressedDataOffsetGlobal[h->numOfBlocks], sizeof(uint32_t),
               cudaMemcpyDeviceToHost);
    return (size_t)sizeof(uint32_t) * (h->numOfBlocks + 1) * 2 + flagLast + dataLast;
}

void gpulzc_free(void *handle) {
    GpulzHandle *h = reinterpret_cast<GpulzHandle *>(handle);
    if (!h) return;
    cudaFree(h->deviceArray);
    cudaFree(h->deviceOutput);
    cudaFree(h->flagArrSizeGlobal);
    cudaFree(h->flagArrOffsetGlobal);
    cudaFree(h->compressedDataSizeGlobal);
    cudaFree(h->compressedDataOffsetGlobal);
    cudaFree(h->tmpFlagArrGlobal);
    cudaFree(h->tmpCompressedDataGlobal);
    cudaFree(h->flagArrGlobal);
    cudaFree(h->compressedDataGlobal);
    cudaFree(h->flag_d_temp_storage);
    cudaFree(h->data_d_temp_storage);
    delete h;
}

// Correctness gate only, called from adapter.py::to_host()/free(),
// DELIBERATELY outside the timed region: decompressKernel -- main()'s own
// decompression half (source/gpulz.cu lines 468-469), unmodified -- then
// D2H-copies just the first `fileSize` (real, unpadded) bytes.
int gpulzd_run_and_copy(void *handle, uint8_t *host_out, uint32_t host_out_bytes) {
    GpulzHandle *h = reinterpret_cast<GpulzHandle *>(handle);
    dim3 deGridDim((unsigned int)ceil(float(h->numOfBlocks) / 32));
    dim3 deBlockDim(32);

    decompressKernel<INPUT_TYPE><<<deGridDim, deBlockDim>>>(
        (INPUT_TYPE *)h->deviceOutput, h->numOfBlocks, h->flagArrOffsetGlobal,
        h->compressedDataOffsetGlobal, h->flagArrGlobal, h->compressedDataGlobal);
    cudaDeviceSynchronize();

    uint32_t n = h->fileSize < host_out_bytes ? h->fileSize : host_out_bytes;
    cudaMemcpy(host_out, h->deviceOutput, n, cudaMemcpyDeviceToHost);
    return (int)n;
}

}  // extern "C"
