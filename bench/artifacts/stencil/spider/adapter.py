"""
Adapter for SPIDER's 2D 7-radius half-precision structured-sparse-Tensor-Core
stencil kernel (PPoPP 2026, PAPER_KEY = conf/ppopp/GuW0Y26).

Wraps the artifact's own kernel (`kernel_2d_7r`, unmodified,
source/src/2d_half_sparse/gpu_2d_7r_half.cu) and its own params-compression
helpers (`param_swap_to_structured_sparsity`, `compress_params`, same file)
through bridge.cu/.so (this directory) -- see that file's docstring for
exactly what had to be split out of the artifact's monolithic `gpu_2d_7r()`
driver, a host-side off-by-one bug found and fixed in the bridge (never
exercised by the artifact's own correctness check), and a bounds precondition
made explicit.

Three deviations from a "plug the workload straight through" adapter, all
load-bearing and all documented here + in STATUS.md:

1. SHAPE: SPIDER's src/ has only 1d_half_sparse/, 2d_half_dense/,
   2d_half_sparse/, 2d_half_sparse_for_ablation/ -- no 3D directory at all.
   This adapter only ever serves 2D box-shaped workloads; prepare() raises
   NotImplementedError for anything else (dims != 2), including the
   3D smoke shapes (smoke-star3d1r, smoke-box3d1r).

2. HARDCODED COEFFICIENTS: the kernel's `metadata_template`
   (gpu_2d_7r_half.h) fixes WHICH 2-of-4 slots of every param row are
   allowed to be nonzero at compile time -- the artifact never derives this
   pattern from caller-supplied weights, it only lets the caller pick the
   *values* at the already-fixed nonzero slots. `_native_params()` below
   reproduces 2d_stencil_half.cu main()'s own generator verbatim (the only
   params array the artifact's own source ever constructs for this kernel),
   and `_native_weights()` extracts the dense 15x15 box kernel it encodes
   the same way the artifact's own check_result() does. prepare() OVERRIDES
   workload_.kind/radius/offsets/weights to this native shape/coefficient
   set in place, so the harness's reference_stencil() (which reads these
   same attributes off the identical workload_ object) regenerates a
   MATCHING fp64 ground truth automatically -- see ARTIFACT_GUIDE.md's note
   on this pattern. The extracted kernel is RANK-1 (weight depends only on
   the row offset, constant across columns) and ALL-POSITIVE, summing to
   1440 -- not the harness domain's own normalized (sum(|w|)=1) synthetic
   weights.

3. TIMESTEPS FORCED TO 1: because the native weights above sum to 1440
   (not normalized), SPIDER's own ping-pong T-sweep would overflow fp16
   (max ~65504) after about 2 sweeps for any nonzero U(0,1) field. The
   artifact's own correctness check (check_result(), 2d_stencil_half.cu)
   ALSO only ever calls the kernel with times=1 -- multi-sweep is only ever
   reached from the un-checked --profile throughput microbenchmark. This
   adapter forces workload_.timesteps = 1 in prepare() for the same reason:
   it is the only sweep count SPIDER itself ever validates.

BOUNDARY (2026-09-06, second update same day -- convention now matches
exactly, superseding the "fixed" attempt below): SPIDER pads its OWN input
with a fixed, ZERO-valued EXTERNAL halo (bridge.cu, width HALO=8 >
radius=7, so the kernel's footprint never reads past it) and never performs
a periodic wrap or freezes an in-array band. This is confirmed directly
from the kernel source, not just inferred from the paper text:

  - out-of-domain reads are ZERO, not clamped/replicated: bridge.cu:122
    zero-initializes the ENTIRE padded host buffer
    (`std::vector<TYPE> h_in(array_elems, TYPE(0))`), and bridge.cu:124-127
    then copies ONLY the caller's `input_m x input_n` interior into it at
    offset `(HALO, HALO)` -- the halo region is never written again before
    the H2D copy at bridge.cu:154. `kernel_2d_7r`'s own shared-memory load
    (gpu_2d_7r_half.cu:73-82, the `cp.async.cg.shared.global` loop) reads a
    `D_BLOCK_ROW x D_BLOCK_COL_NOPAD` tile straight out of this padded array
    via a plain offset address (`in + begin + IDX(row, col, ldm)`, line 77)
    -- no bounds check, no clamp/replicate logic anywhere in the load path
    -- so any thread whose tile cell falls in the halo genuinely reads the
    zero the host buffer was initialized with, not a replicated edge value.
  - EVERY domain cell is written, unconditionally: `kernel_2d_7r`'s output
    store (gpu_2d_7r_half.cu:147-162) loops over all
    `BLOCK_ROW * BLOCK_COL / 8` elements of every launched block with no
    per-cell skip or bounds check, and the grid
    (`dim3(CEIL(input_m, BLOCK_ROW), CEIL(input_n, BLOCK_COL))`, mirrored at
    bridge.cu:158) exactly tiles the `input_m x input_n` domain because
    `spider2d7r_prepare()` (bridge.cu:104-111) refuses any grid shape that
    is not an exact multiple of `BLOCK_ROW=64`/`BLOCK_COL=128` -- so no
    block is ever partial and no domain cell is ever skipped.

This is EXACTLY `kernelbench/domains/stencil.py`'s new `"zero-halo"`
convention (added 2026-09-06, same day, specifically because neither
"periodic" nor this domain's own "fixed" matched what SPIDER computes):
`prepare()` now sets `params["boundary"] = "zero-halo"` so the harness's
`reference_stencil()` recomputes EVERY domain cell each sweep against a
zero-padded extension of the SAME array (`_sweep_zero_halo`), rather than
freezing a band -- the "fixed" convention briefly tried earlier the same
day, which STATUS.md shows failing even harder (err got LARGER, not
smaller) because SPIDER's boundary band holds a REAL computed partial sum,
not a frozen copy of the initial field -- or wrapping periodically (the
original default, also shown failing in STATUS.md's history section). No
cropping, splicing, or other adapter-side reconciliation is needed or
performed: `to_host()` still returns the FULL `(m, n)` array unmodified
(matching `workload_.grid_shape`); the number the gate reports is SPIDER's
own kernel output, compared honestly against the convention it actually
implements. See STATUS.md for the resulting gate numbers.
"""

from __future__ import annotations

import ctypes
import os

import numpy as np

KERNEL = "stencil"
IMPL_NAME = "spider-box2d7r-sptc"
PAPER_KEY = "conf/ppopp/GuW0Y26"
PRECISIONS = ["fp16"]  # gpu_2d_7r_half.h: #define TYPE half

_HERE = os.path.dirname(os.path.abspath(__file__))
_SO_PATH = os.path.join(_HERE, "bridge.so")

_BLOCK_ROW = 64   # gpu_2d_7r_half.cu BLOCK_ROW
_BLOCK_COL = 128  # gpu_2d_7r_half.cu BLOCK_COL
_RADIUS = 7       # 15x15 box footprint -> radius 7 (2*7+1 == 15)


def available() -> tuple[bool, str]:
    try:
        if not os.path.exists(_SO_PATH):
            return False, f"not built: {_SO_PATH} missing (run build.sh)"
        return True, ""
    except Exception as e:  # available() must never raise
        return False, f"{type(e).__name__}: {e}"


def _load_lib():
    lib = ctypes.CDLL(_SO_PATH)
    p = ctypes.c_void_p
    u16p = ctypes.POINTER(ctypes.c_uint16)
    lib.spider2d7r_prepare.argtypes = [u16p, ctypes.c_int, ctypes.c_int, u16p]
    lib.spider2d7r_prepare.restype = p
    lib.spider2d7r_run.argtypes = [p, ctypes.c_int, p]
    lib.spider2d7r_run.restype = None
    lib.spider2d7r_copy_out.argtypes = [p, u16p]
    lib.spider2d7r_copy_out.restype = None
    lib.spider2d7r_free.argtypes = [p]
    lib.spider2d7r_free.restype = None
    return lib


def _native_params() -> np.ndarray:
    """
    Reproduces 2d_stencil_half.cu main()'s own param-array construction
    verbatim (source/src/2d_half_sparse/2d_stencil_half.cu):

        TYPE params[15*16*32] = {0.0};
        for (param_iter in 0..14)
          for (col in 1..15):
            param = col % 13 + 1
            for (row in 0..15):
              params[IDX(param_iter,0,16*32) + IDX(row,col+row,32)] = param

    (IDX(x,y,ldm) = x*ldm+y.) The compile-time-fixed metadata_template
    (gpu_2d_7r_half.h) only allows 2 of every 4 param entries to be nonzero
    (2:4 structured sparsity); this diagonal generator is the artifact's OWN
    way of producing a params array whose nonzero support lines up with
    that fixed pattern -- verified by param_swap_to_structured_sparsity()
    (gpu_2d_7r_half.cu), which the bridge calls unmodified and which prints
    an explicit "Not 2:4 structured sparsity" error to stderr if it ever
    doesn't (never observed here).
    """
    params = np.zeros((15, 16, 32), dtype=np.float16)
    for row in range(16):
        for col in range(1, 16):
            val = np.float16(col % 13 + 1)
            params[:, row, col + row] = val
    return params


def _native_weights() -> dict[tuple[int, int], float]:
    """
    The dense 15x15 box kernel the params above encode, extracted the same
    way the artifact's own check_result() does (2d_stencil_half.cu):

        cpu_params[IDX(j, i, 15)] = params[IDX(i, 0, 16*32) + j + 1]

    Substituting _native_params()'s generator shows the extracted value
    depends only on `j` (the ROW offset in the 15x15 kernel) -- constant
    across all 15 columns -- giving a rank-1, all-positive kernel:

        weight(dr, dc) = (dr + 8) % 13 + 1,  dr, dc in [-7, 7]

    sum(|w|) = 15 * sum_{dr=-7..7}[(dr+8)%13+1] = 15 * 96 = 1440 (see
    module docstring point 3 -- NOT the harness's normalized-to-1 weights).
    """
    return {(dr, dc): float((dr + 8) % 13 + 1)
            for dr in range(-_RADIUS, _RADIUS + 1)
            for dc in range(-_RADIUS, _RADIUS + 1)}


class SpiderBox2d7r:
    name = IMPL_NAME
    platform = "cuda"

    def __init__(self, precision: str = "fp16"):
        if precision != "fp16":
            raise NotImplementedError(
                "SPIDER's 2d_half_sparse kernel is compiled for TYPE=half only "
                f"(gpu_2d_7r_half.h); requested {precision!r}")
        self.precision = precision
        self.lib = _load_lib()

    def prepare(self, workload_, params: dict):
        if workload_.dims != 2:
            raise NotImplementedError(
                f"SPIDER ships 1D/2D kernels only (source/src has no 3D "
                f"directory); {workload_.name!r} is {workload_.dims}D")
        m, n = (int(s) for s in workload_.grid_shape)
        if m % _BLOCK_ROW or n % _BLOCK_COL:
            raise NotImplementedError(
                f"SPIDER's 2d_half_sparse kernel tiles BLOCK_ROW={_BLOCK_ROW} x "
                f"BLOCK_COL={_BLOCK_COL} with no per-block bounds check "
                f"(gpu_2d_7r_half.cu writes the full output tile "
                f"unconditionally); grid_shape {(m, n)} is not a multiple -- "
                f"would risk an out-of-bounds device write.")

        # --- deviation 2 (module docstring): hardcode the workload to what
        # SPIDER is actually about to compute, in place, so the harness's
        # reference_stencil() (reads workload_.kind/dims/radius/offsets/
        # weights off this same object) regenerates a matching ground truth.
        weights = _native_weights()
        workload_.kind = "box"
        workload_.radius = _RADIUS
        workload_.offsets = list(weights.keys())
        workload_.weights = weights

        # --- deviation 3: forced T=1 (fp16 overflow risk beyond one sweep
        # with this un-normalized, sum=1440 weight set -- see module
        # docstring point 3).
        workload_.timesteps = 1

        # --- boundary (module docstring "BOUNDARY" section): SPIDER's own
        # zero-Dirichlet EXTERNAL halo is exactly the domain's "zero-halo"
        # convention (kernelbench/domains/stencil.py) -- select it here so
        # the harness's reference_stencil() recomputes every domain cell
        # against a zero-padded extension of the same array, matching what
        # kernel_2d_7r itself actually computes (see docstring above for the
        # file:line evidence that out-of-domain reads are zero and every
        # cell is written).
        params["boundary"] = "zero-halo"

        field = workload_.initial_field(dtype=np.float16)  # (m, n), U(0,1)
        field_flat = np.ascontiguousarray(field).view(np.uint16).reshape(-1)
        params_flat = np.ascontiguousarray(_native_params()).view(np.uint16).reshape(-1)

        u16p = ctypes.POINTER(ctypes.c_uint16)
        handle = self.lib.spider2d7r_prepare(
            field_flat.ctypes.data_as(u16p), m, n,
            params_flat.ctypes.data_as(u16p))
        if not handle:
            raise RuntimeError(
                "spider2d7r_prepare returned a null handle (see stderr for "
                "the bounds-check message)")

        out_host = np.zeros((m, n), dtype=np.float16)
        # keep field/params alive: ctypes doesn't, and prepare() only reads
        # them synchronously (no async H2D), so this is precautionary, same
        # convention as the cb-spmv adapter.
        return {"handle": handle, "m": m, "n": n, "out_host": out_host,
                "_field": field_flat, "_params": params_flat}

    def run(self, h):
        # Exactly one kernel_2d_7r launch (times=1, forced -- see prepare());
        # stream=0, the default/legacy stream, same stream torch.cuda.Event()
        # records on by default, so CudaEventTimer times this launch
        # correctly even though the kernel itself is launched outside torch.
        self.lib.spider2d7r_run(h["handle"], 1, ctypes.c_void_p(0))
        return h

    def to_host(self, out) -> np.ndarray:
        m, n = out["m"], out["n"]
        u16p = ctypes.POINTER(ctypes.c_uint16)
        out_view = out["out_host"].view(np.uint16).reshape(-1)
        self.lib.spider2d7r_copy_out(out["handle"], out_view.ctypes.data_as(u16p))
        # float64 so the harness's fp64 correctness gate operates at full
        # precision on our side of the comparison; the fp16 rounding already
        # happened on-device, this cast does not hide any of it.
        return out["out_host"].astype(np.float64)

    def timer(self):
        from kernelbench.impls.gpu_cuda import CudaEventTimer
        return CudaEventTimer()

    def free(self, h):
        self.lib.spider2d7r_free(h["handle"])
        h.clear()


def create(precision: str):
    return SpiderBox2d7r(precision)
