// Shared handle struct for bridge_fzgpu.cu's compress/decompress halves.
#pragma once
#include <cstdint>

struct FzgpuHandle {
    float* deviceInput = nullptr;
    uint16_t* deviceQuantizationCode = nullptr;
    bool* deviceSignNum = nullptr;
    uint16_t* deviceCompressedOutput = nullptr;
    uint32_t* deviceBitFlagArr = nullptr;
    uint16_t* deviceDecompressedQuantizationCode = nullptr;
    float* deviceDecompressedOutput = nullptr;
    uint32_t* deviceOffsetCounter = nullptr;
    uint32_t* deviceStartPosition = nullptr;
    uint32_t* deviceCompressedSize = nullptr;

    int dimx = 0, dimy = 0, dimz = 0;
    int dataTypeLen = 0;            // real (unpadded) element count = dimx*dimy*dimz
    int paddingDataTypeLen = 0;     // FZ-GPU's own chunk-aligned padded length
    int quantizationCodeByteLen = 0;
    int dataChunkSize = 0;
    int gridX = 0;
    double abs_eb = 0.0;            // ALREADY absolute -- see bridge_fzgpu.cu
};
