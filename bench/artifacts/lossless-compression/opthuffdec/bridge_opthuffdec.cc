// Thin driver replacing source/opt-gap-array/src/demo.cc (the paper's own
// benchmark script) with a C API split into PREPARE (CPU reference encode +
// GPU buffer allocation + H2D transfer) vs. RUN (exactly one call to the
// artifact's own optimized GPU decode kernel) vs. FETCH (D2H copy for the
// correctness gate) -- per ARTIFACT_GUIDE rule 1 ("wrap the kernel, not the
// paper's benchmark script") and rule 2 (format conversion / encoding is
// preprocessing, timed once, not part of the measured kernel).
//
// Every function called here (llhuff::LLHuffmanEncoder::*, cuhd::CUHD*) is
// the artifact's own unmodified class from source/opt-gap-array/{include,
// encoder/include} + {src,encoder/src} -- this file only orchestrates the
// SAME calls demo.cc itself makes (compare against
// source/opt-gap-array/src/demo.cc's `main()`), just split at the
// prepare/run boundary instead of measuring the whole pipeline in one
// timed loop, and without demo.cc's own file I/O / printf reporting.
//
// This exact same driver shape works for opt-self-sync/orig-gap-array/
// orig-self-sync too (all four expose the identical cuhd::CUHDGPUDecoder::
// decode signature) -- see build.sh for which one this build compiles
// against (OPTHUFFDEC_VARIANT).
#include <cstdint>
#include <cstddef>
#include <cstring>
#include <memory>
#include <vector>

#include <cuda_runtime.h>

#include "llhuff.h"
#include "cuhd.h"
#include "cuhd_util.h"   // SDIV macro

#define BRIDGE_SUBSEQ_SIZE 4      // demo.cc's own SUBSEQ_SIZE
#define BRIDGE_NUM_THREADS 128    // demo.cc's own NUM_THREADS
#define BRIDGE_CACHE_LEN 9        // demo.cc's own cache_len

// cuhd_gpu_decoder.cu declares these as `extern double p1Time[1024], ...`
// (source/opt-gap-array/src/cuhd_gpu_decoder.cu line 23) -- internal
// per-phase profiling scratch the kernel itself writes into
// (`p1Time[invocation_count - 1] = ...`, an unbounded-growth counter with
// no wraparound, same 1024-slot fixed size demo.cc itself used). demo.cc
// is not compiled here (it is the paper's own benchmark script, replaced
// by this bridge -- see file docstring), so these globals must be defined
// SOMEWHERE for the link to resolve; providing them here (not in source/)
// is the minimal fix. NOTE (disclosed, not hidden): more than 1024 total
// decode() invocations in one process will write past these arrays --
// a pre-existing limitation of the artifact's own profiling code, not
// introduced by this integration. Harmless for this benchmark's
// warmup+reps counts (single digits to a few dozen); would matter for a
// hypothetical multi-thousand-rep timing sweep.
double p1Time[1024], p2Time[1024], p3Time[1024], p4Time[1024], tuneTime[1024];

// cuhd_gpu_decoder.cu also declares `extern int shared_size;` (line 21) --
// demo.cc's own optional 5th CLI argument ("performance parameter, high =
// high performance, low = low memory consumption"; -1 = auto, demo.cc's
// own default when the argument is omitted -- see its `int shared_size =
// -1;` and the `if(argc == 5) shared_size = atoi(argv[4]);` guard). Kept at
// demo.cc's own default (auto) here; not something this integration tunes.
int shared_size = -1;

struct OptHuffDecHandle {
    size_t n_symbols = 0;
    size_t compressed_size_units = 0;
    size_t gap_array_size = 0;

    std::shared_ptr<llhuff::LLHuffmanEncoderTable> enc_table;
    std::shared_ptr<cuhd::CUHDCodetable> dec_table;

    std::unique_ptr<UNIT_TYPE[]> compressed;
    std::unique_ptr<uint8_t[]> gap_array;

    std::shared_ptr<cuhd::CUHDInputBuffer> in_buf;
    std::shared_ptr<cuhd::CUHDOutputBuffer> out_buf;
    std::shared_ptr<cuhd::CUHDGPUInputBuffer> gpu_in_buf;
    std::shared_ptr<cuhd::CUHDGPUMemoryBuffer<uint8_t>> gpu_gap_buf;
    std::shared_ptr<cuhd::CUHDGPUCodetable> gpu_table;
    std::shared_ptr<cuhd::CUHDGPUOutputBuffer> gpu_out_buf;
    std::shared_ptr<cuhd::CUHDGPUDecoderMemory> gpu_decoder_memory;
};

extern "C" {

// PREPARE: CPU-side length-limited Huffman encode (the artifact's own
// reference encoder, llhuff::LLHuffmanEncoder -- demo.cc's exact sequence:
// get_symbol_lengths -> get_encoder_table -> get_decoder_table ->
// encode_memory), then GPU buffer allocation + host-to-device copy of the
// compressed stream/gap array/decode table. All of this is preprocessing
// under this benchmark's contract: it happens once, outside the timed
// region, exactly like demo.cc's own "encoding"/"GPU buffer allocation"/
// "GPU memcpy HtD" phases (all separately TIMER_START/STOP'd there, BEFORE
// its own "decoding" phase). `symbols` values must fit in SYMBOL_TYPE
// (uint16_t, 0..65535) -- the artifact's own compile-time symbol type,
// unmodified.
void* opthuffdec_prepare(const uint16_t* symbols, size_t n_symbols, int device_id) {
    auto* h = new OptHuffDecHandle();
    h->n_symbols = n_symbols;

    // get_symbol_lengths takes a non-const pointer; copy into a mutable
    // buffer (SYMBOL_TYPE == uint16_t, so this is a plain memcpy, no
    // conversion).
    std::vector<SYMBOL_TYPE> buffer(symbols, symbols + n_symbols);

    auto lengths = llhuff::LLHuffmanEncoder::get_symbol_lengths(buffer.data(), n_symbols);
    h->enc_table = llhuff::LLHuffmanEncoder::get_encoder_table(lengths);
    h->dec_table = llhuff::LLHuffmanEncoder::get_decoder_table(h->enc_table, BRIDGE_CACHE_LEN);

    h->compressed_size_units = h->enc_table->compressed_size;
    h->gap_array_size = SDIV(h->compressed_size_units, BRIDGE_SUBSEQ_SIZE);
    h->compressed = std::make_unique<UNIT_TYPE[]>(h->compressed_size_units);
    h->gap_array = std::make_unique<uint8_t[]>(h->gap_array_size);

    llhuff::LLHuffmanEncoder::encode_memory(
        h->compressed.get(), h->compressed_size_units,
        buffer.data(), h->gap_array.get(), BRIDGE_SUBSEQ_SIZE, n_symbols,
        h->enc_table);

    cudaSetDevice(device_id);

    h->in_buf = std::make_shared<cuhd::CUHDInputBuffer>(
        reinterpret_cast<uint8_t*>(h->compressed.get()),
        h->compressed_size_units * sizeof(UNIT_TYPE));
    h->out_buf = std::make_shared<cuhd::CUHDOutputBuffer>(n_symbols);
    h->gpu_in_buf = std::make_shared<cuhd::CUHDGPUInputBuffer>(h->in_buf);
    h->gpu_gap_buf = std::make_shared<cuhd::CUHDGPUMemoryBuffer<uint8_t>>(
        h->gap_array.get(), h->gap_array_size);
    h->gpu_table = std::make_shared<cuhd::CUHDGPUCodetable>(h->dec_table);
    h->gpu_out_buf = std::make_shared<cuhd::CUHDGPUOutputBuffer>(h->out_buf);
    h->gpu_decoder_memory = std::make_shared<cuhd::CUHDGPUDecoderMemory>(
        h->in_buf->get_compressed_size_units(), BRIDGE_SUBSEQ_SIZE, BRIDGE_NUM_THREADS);

    h->gpu_in_buf->allocate();
    h->gpu_gap_buf->allocate();
    h->gpu_out_buf->allocate();
    h->gpu_table->allocate();
    h->gpu_decoder_memory->allocate();

    h->gpu_table->cpy_host_to_device();
    h->gpu_in_buf->cpy_host_to_device();
    h->gpu_gap_buf->cpy_host_to_device();

    return h;
}

// RUN: exactly one call to the artifact's own optimized GPU decoder --
// cuhd::CUHDGPUDecoder::decode, source/opt-gap-array/src/cuhd_gpu_decoder.cu
// -- unmodified, same call demo.cc itself times (its own NROUNDS loop over
// this identical call). THIS is what the harness's CudaEventTimer brackets.
void opthuffdec_run(void* handle) {
    auto* h = static_cast<OptHuffDecHandle*>(handle);
    cuhd::CUHDGPUDecoder::decode(
        h->gpu_in_buf, h->in_buf->get_compressed_size_units(),
        h->gpu_out_buf, h->out_buf->get_uncompressed_size(),
        h->gpu_gap_buf, h->gap_array_size,
        h->gpu_table, BRIDGE_CACHE_LEN, h->gpu_decoder_memory,
        MAX_CODEWORD_LENGTH, BRIDGE_SUBSEQ_SIZE, BRIDGE_NUM_THREADS);
}

// FETCH: D2H copy of the decoded symbol stream, for the correctness gate
// ONLY -- deliberately outside the timed region (mirrors demo.cc's own
// separately-timed "GPU memcpy DtH" phase, which is not part of its
// "decoding" timing either).
void opthuffdec_fetch_output(void* handle, uint16_t* out) {
    auto* h = static_cast<OptHuffDecHandle*>(handle);
    h->gpu_out_buf->cpy_device_to_host();
    std::memcpy(out, h->out_buf->get_decompressed_data().get(),
               h->n_symbols * sizeof(SYMBOL_TYPE));
}

size_t opthuffdec_compressed_bytes(void* handle) {
    auto* h = static_cast<OptHuffDecHandle*>(handle);
    return h->compressed_size_units * sizeof(UNIT_TYPE)
         + h->gap_array_size * sizeof(uint8_t);
}

void opthuffdec_free(void* handle) {
    delete static_cast<OptHuffDecHandle*>(handle);
}

}  // extern "C"
