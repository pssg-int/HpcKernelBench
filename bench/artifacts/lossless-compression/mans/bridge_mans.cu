// Thin ctypes-callable extern "C" wrapper around MANS's own NVIDIA
// device-pointer API (source/nv/mans_nv.h::compress_internal_device /
// decompress_internal_device).
//
// Unlike this benchmark's other single-file-CLI GPU compression artifacts
// (gpulz, fzgpu, pfpl), MANS genuinely ships a reusable device-pointer
// library entry point for its NVIDIA backend -- no split of an
// end-to-end main() is needed here. This file exists ONLY because
// mans::nv::compress_internal_device/decompress_internal_device are C++
// (non-"extern C") functions taking a `std::size_t&` out-parameter, which
// ctypes cannot call directly; it forwards to them verbatim (same argument
// values, same call, no kernel code touched) through an ABI ctypes CAN
// call (a `size_t*` in, `size_t*` out signature).
//
// mans::nv::compress_internal_device/decompress_internal_device (not the
// higher-level mans::compress_device/decompress_device in mans_api.cpp) are
// called directly -- mans_api.cpp is an extra dispatch layer that also
// pulls in the CPU backend (mans_cpu.h, needing OpenMP/-march=native
// -mavx512f) this integration does not build; the NV-only entry point is
// the finest available boundary and is used as-is.
#include <cstdint>
#include <cstdio>
#include <stdexcept>

#include "source/mans_defs.h"
#include "source/nv/mans_nv.h"

extern "C" {

// Upper bound on compressed bytes for a given (dtype, mode, num_elements)
// -- MANS's own sizing function (mans::nv::get_max_compress_bytes),
// unmodified. Callers (adapter.py::prepare()) use this to size the device
// output buffer before calling mans_compress_device.
size_t mans_max_compress_bytes(uint32_t dtype, uint32_t mode, size_t num_elements) {
    mans::MansParams p{};
    p.backend = mans::Backend::NVIDIA;
    p.dtype = dtype;
    p.mode = mode;
    p.dims = 1;
    p.nx = static_cast<uint32_t>(num_elements);
    p.ny = 0;
    p.nz = 0;
    return mans::nv::get_max_compress_bytes(num_elements, p);
}

// ONE compress call: mans::nv::compress_internal_device (ADM mapping +
// entropy-code stage, MANS's own kernels, unmodified). d_input/d_out are
// device pointers (torch-allocated by adapter.py, same ctypes pattern this
// benchmark's cuszp-compress/pfpl-compress adapters already use).
// Returns 0 on success, -1 on a thrown exception (message printed to
// stderr for diagnosis; MANS's own API throws std::runtime_error on
// misuse, e.g. unsupported dtype).
int mans_compress_device(const void *d_input, size_t num_elements,
                         uint32_t dtype, uint32_t mode,
                         uint8_t *d_out, size_t *out_size) {
    mans::MansParams p{};
    p.backend = mans::Backend::NVIDIA;
    p.dtype = dtype;
    p.mode = mode;
    p.dims = 1;
    p.nx = static_cast<uint32_t>(num_elements);
    p.ny = 0;
    p.nz = 0;
    try {
        std::size_t sz = 0;
        mans::nv::compress_internal_device(d_input, num_elements, p, d_out, sz);
        *out_size = sz;
        return 0;
    } catch (const std::exception &e) {
        fprintf(stderr, "mans_compress_device: %s\n", e.what());
        return -1;
    }
}

// Correctness-gate-only decompress call: mans::nv::decompress_internal_device
// (MANS's own kernels, unmodified). `compressed_len` is the byte count
// mans_compress_device wrote into *out_size; dtype/mode must match what was
// used to compress (MANS's own on-wire MansHeader records dims/nx/ny/nz/
// raw_bytes but NOT dtype -- decompress_internal_device relies on the
// caller-supplied `params.dtype`, exactly as MANS's own header comment in
// mans_defs.h documents).
int mans_decompress_device(const void *d_input, size_t compressed_len,
                           uint32_t dtype, uint32_t mode,
                           uint8_t *d_out, size_t *out_size) {
    mans::MansParams p{};
    p.backend = mans::Backend::NVIDIA;
    p.dtype = dtype;
    p.mode = mode;
    try {
        std::size_t sz = 0;
        mans::nv::decompress_internal_device(d_input, compressed_len, p, d_out, sz);
        *out_size = sz;
        return 0;
    } catch (const std::exception &e) {
        fprintf(stderr, "mans_decompress_device: %s\n", e.what());
        return -1;
    }
}

}  // extern "C"
