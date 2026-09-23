// periodic_halo.cuh (HPC-KernelBench, NOT part of any artifact).
//
// Device-side periodic-wrap halo refresh, shared by the ConvStencil and
// LoRAStencil bridges. Those artifacts' kernels read a halo around the
// interior and write only the interior, and their own multi-step loops never
// refresh that halo between steps (see each bridge.cu). Refreshing it here,
// on the device, after every step lets a whole T-step run stay GPU-resident
// instead of round-tripping the grid through the host every step.
//
// Layout: row-major buffer with leading dimension `ldm`; the m x n interior
// starts at (r0, c0). The band [r0-hr, r0) / [r0+m, r0+m+hr) is filled over
// columns [c0-hc, c0+n+hc) (corners included), and the band [c0-hc, c0) /
// [c0+n, c0+n+hc) over interior rows. Every halo cell copies the interior
// cell it wraps to, so reads touch only the interior and writes only the
// halo -- no ordering hazards within the launch. Work is O(perimeter), not
// O(grid). Requires m >= hr and n >= hc.

#pragma once
#include <cuda_runtime.h>

__global__ void kb_periodic_halo_rows(double *buf, int ldm, int m, int n,
                                      int r0, int c0, int hr, int hc) {
    const int width = n + 2 * hc;
    const int total = 2 * hr * width;
    for (int t = blockIdx.x * blockDim.x + threadIdx.x; t < total;
         t += gridDim.x * blockDim.x) {
        const int band = t / width;           // 0 .. 2*hr-1
        const int c = c0 - hc + t % width;
        const int r = band < hr ? r0 - hr + band : r0 + m + (band - hr);
        const int ri = ((r - r0) % m + m) % m;
        const int ci = ((c - c0) % n + n) % n;
        buf[(size_t)r * ldm + c] = buf[(size_t)(r0 + ri) * ldm + c0 + ci];
    }
}

__global__ void kb_periodic_halo_cols(double *buf, int ldm, int m, int n,
                                      int r0, int c0, int hc) {
    const int total = m * 2 * hc;
    for (int t = blockIdx.x * blockDim.x + threadIdx.x; t < total;
         t += gridDim.x * blockDim.x) {
        const int r = r0 + t / (2 * hc);
        const int k = t % (2 * hc);
        const int c = k < hc ? c0 - hc + k : c0 + n + (k - hc);
        const int ci = ((c - c0) % n + n) % n;
        buf[(size_t)r * ldm + c] = buf[(size_t)r * ldm + c0 + ci];
    }
}

// Launch both on the current (default) stream, in order after the step that
// produced `buf`'s interior.
static inline void kb_periodic_halo(double *buf, int ldm, int m, int n,
                                    int r0, int c0, int hr, int hc) {
    const int threads = 256;
    const int rows_total = 2 * hr * (n + 2 * hc);
    const int cols_total = m * 2 * hc;
    kb_periodic_halo_rows<<<(rows_total + threads - 1) / threads, threads>>>(
        buf, ldm, m, n, r0, c0, hr, hc);
    kb_periodic_halo_cols<<<(cols_total + threads - 1) / threads, threads>>>(
        buf, ldm, m, n, r0, c0, hc);
}
