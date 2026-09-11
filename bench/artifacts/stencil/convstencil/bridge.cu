// Thin ctypes-callable bridge around ConvStencil's 2D box-stencil-via-Tensor-
// Core kernel (source/src/2d/gpu.cu, function gpu_box_2d1r).
//
// NOT part of the artifact. It exists because the artifact's own entry point
// is main.cu's CLI driver (source/src/2d/main.cu), which:
//   (a) hardcodes a shape catalog (box2d1r/star2d1r/star2d3r/box2d3r) and
//       picks a `param` (49-double, 7x7 row-major) array per shape via a
//       switch statement -- but for the star_2d1r / star_2d3r cases that
//       switch selects `param_star_2d1r`, a buffer declared
//       `double param_star_2d1r[49] = {0.0}` and NEVER POPULATED anywhere
//       in main.cu (only param_box_2d1r is filled, via a box^3-collapse
//       polynomial expansion driven by --custom input). This is a bug/dead
//       feature in the artifact itself (verified by reading main.cu in
//       full): the CLI's star2d1r path silently computes with an all-zero
//       weight array. We bypass main.cu entirely and call gpu_box_2d1r()
//       directly with our OWN 49-double param array built from the
//       harness's StencilWorkload.weights, sidestepping that bug -- no
//       kernel code touched, we just supply correct weights where main.cu's
//       own CLI plumbing fails to.
//   (b) bundles host buffer allocation, H2D copy, an internal ping-pong
//       loop over `times` (invoked via kernel2d<<<...>>> repeatedly on
//       device-resident buffers, no halo refresh between iterations -- see
//       below), and D2H copy, all inside gpu_box_2d1r() itself. There is no
//       separate "prepare vs run" split available in the artifact; the
//       whole thing is one C function taking host pointers in and out.
//
// T-sweep semantics (checked directly in gpu.cu): gpu_box_2d1r's internal
// `times` loop ping-pongs the SAME device buffer pair without ever
// refreshing the halo band between iterations -- iteration 0 reads the
// halo as initialized by the H2D copy of the caller's padded host buffer;
// iterations 1..times-1 reuse that SAME (now-stale) halo, never
// re-wrapped from the evolving interior. This does not match our domain's
// periodic-wrap T-sweep recursion (kernelbench/domains/stencil.py
// reference_stencil: every sweep re-wraps from the CURRENT field). So we
// do NOT use gpu_box_2d1r's internal `times` parameter for multi-step runs;
// instead this bridge exposes a SINGLE-SWEEP entry point
// (convstencil2d_sweep, times=1 always) and the Python adapter's run()
// calls it `workload.timesteps` times, re-building the periodic-wrap padded
// host buffer between each call (host-side np.pad(mode="wrap"), matching
// kernelbench/domains/stencil.py's np.roll-based periodic reference). This
// is materially slower per-call than the artifact's own batched internal
// loop (H2D/D2H every sweep instead of once) but is the only way to get a
// CORRECT multi-timestep periodic result out of this kernel without
// touching gpu.cu; documented as a known performance caveat (not a
// correctness one) in STATUS.md -- login-node gate work only needs
// correctness here, per the integration contract's rule 5.
//
// Padded-buffer layout (rows = m + 2*HALO, cols = n + 2*HALO + 2, HALO=3,
// exactly as gpu_box_2d1r allocates internally and as main.cu's own host
// buffer is sized): the offset at which OUTPUT is actually written was
// determined by reading gpu.cu's kernel2d store index
// (`out + begin + IDX(HALO + col/7, HALO, ldm)` with
// `begin = IDX(blockIdx.x*BLOCK_SIZE_ROW, blockIdx.y*BLOCK_SIZE_COL+1, ldm)`)
// which places row 0 of valid output at buffer row HALO(=3) and column 0 of
// valid output at buffer column HALO+1(=4). Since cols_total - n = 2*HALO+2
// = 8 = 4+4, this is consistent with a SYMMETRIC 4-wide column margin (the
// extra +1/+2 beyond the plain 2*HALO in main.cu's own size arithmetic is
// tensor-core/alignment padding, not an asymmetric boundary). Row margin is
// the plain HALO=3 on both sides (rows_total - m = 2*HALO exactly, no
// extra). This bridge pads with row margin 3/3 and column margin 4/4,
// periodic-wrap (matching kernelbench/domains/stencil.py's boundary), and
// extracts the result from the same offsets. Verified empirically (see
// STATUS.md) against a hand-computed reference before wiring into the
// harness adapter.
//
// params49: 7x7 row-major, param[i*7+j] weights in[row+(i-3)][col+(j-3)]
// (matches main.cu's own naive_box2d1r reference loop exactly, and
// StencilWorkload.dense_kernel(flip=False)'s convention with radius fixed
// at 3 -- ConvStencil's kernel always operates on a 7x7 (radius-3) support
// regardless of shape name; smaller-radius shapes (radius 1) are expressed
// by zero-padding the unused outer ring of the 7x7, which the adapter does).

#include "2d_utils.h"

extern "C" {

// One stencil sweep: out = box7x7(in), no internal iteration (times=1
// always -- see docstring above for why the artifact's own multi-`times`
// path is not used). `padded_in`/`padded_out` are host pointers of size
// (m+6) x (n+8) doubles (row-major), already periodic-wrap padded by the
// caller. `params49` is a 49-double row-major 7x7 array. This is
// gpu_box_2d1r verbatim (source/src/2d/gpu.cu), unmodified, called with
// times=1.
void convstencil2d_sweep(const double *padded_in, double *padded_out,
                          const double *params49, int m, int n) {
    gpu_box_2d1r(padded_in, padded_out, params49, /*times=*/1, m, n);
}

}  // extern "C"
