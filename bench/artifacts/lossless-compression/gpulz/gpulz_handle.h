// Shared handle struct for bridge_gpulz.cu's compress/decompress halves.
#pragma once
#include <cstdint>

struct GpulzHandle {
    // primary buffers, sized in units of INPUT_TYPE (see gpulz.cu's own
    // #define INPUT_TYPE uint32_t) unless noted otherwise
    void* deviceArray = nullptr;          // real+zero-padded input
    void* deviceOutput = nullptr;         // decompressed output (gate-only)

    uint32_t* flagArrSizeGlobal = nullptr;
    uint32_t* flagArrOffsetGlobal = nullptr;
    uint32_t* compressedDataSizeGlobal = nullptr;
    uint32_t* compressedDataOffsetGlobal = nullptr;
    uint8_t* tmpFlagArrGlobal = nullptr;
    uint8_t* tmpCompressedDataGlobal = nullptr;
    uint8_t* flagArrGlobal = nullptr;
    uint8_t* compressedDataGlobal = nullptr;

    // cub::DeviceScan::ExclusiveSum scratch, sized once (depends only on
    // numOfBlocks+1, not on data content) and reused across every run() call
    void* flag_d_temp_storage = nullptr;
    size_t flag_temp_storage_bytes = 0;
    void* data_d_temp_storage = nullptr;
    size_t data_temp_storage_bytes = 0;

    uint32_t fileSize = 0;          // real (unpadded) byte count
    uint32_t paddingSize = 0;
    uint32_t datatypeSize = 0;      // padded length, in units of INPUT_TYPE
    uint32_t numOfBlocks = 0;
};
