// Shared handle struct for bridge_encode.cu / bridge_decode.cu.
//
// Deliberately uses plain `unsigned char*` (not PFPL's own `byte` alias)
// so this header has no ordering dependency on the artifact source it is
// included alongside -- see bridge_encode.cu / bridge_decode.cu for why
// each is its own translation unit.
#pragma once

struct PfplHandle {
    unsigned char *d_input = nullptr;    // original data, device-resident
    unsigned char *d_encoded = nullptr;  // PFPL's own encoded-stream buffer
    int *d_encsize = nullptr;            // achieved encoded size (device int)
    int *d_fullcarry = nullptr;          // PFPL's own carry-propagation buffer
    unsigned char *d_decoded = nullptr;  // decode-side output buffer
    int *d_decsize = nullptr;            // achieved decoded size (device int)
    int insize = 0;                      // original uncompressed byte count
    int chunks = 0;                      // PFPL's own chunk count (CS-sized)
    int blocks = 0;                      // grid size PFPL computes from SM count
    int maxsize = 0;                     // PFPL's own worst-case encoded size
    float errorbound = 0.0f;
    float threshold = 0.0f;              // set to +inf (no sentinel passthrough)
};
