// Provides `get_unsigned_type_size`, declared in source/src/util.h and
// called (only that one function) from source/src/csr.cpp and
// source/src/ef_layout.h. NOT part of the artifact.
//
// The artifact's own definition lives in source/src/util.cpp alongside
// `get_sha_sum`, which needs <openssl/sha.h> (-lssl -lcrypto). This project
// has no sudo / no system-package-install path (ARTIFACT_GUIDE.md) and
// get_sha_sum is never called by anything this adapter uses (it only
// existed for main.cu's own CLI self-consistency-hash printout, which this
// adapter's prepare()/run() split never reaches) -- so source/src/util.cpp
// is not compiled at all, and this file supplies ONLY the one function
// actually needed, with the EXACT same body as the artifact's own
// definition (a deterministic, unambiguous byte-width selector -- pure
// utility code, not part of the Elias-Fano encode/decode algorithm itself).
#include <cstdint>
#include <limits>

uint8_t get_unsigned_type_size(uint64_t val) {
    if (val > std::numeric_limits<uint32_t>::max()) {
        return sizeof(uint64_t);
    } else if (val > std::numeric_limits<uint16_t>::max()) {
        return sizeof(uint32_t);
    } else if (val > std::numeric_limits<uint8_t>::max()) {
        return sizeof(uint16_t);
    } else {
        return sizeof(uint8_t);
    }
}
