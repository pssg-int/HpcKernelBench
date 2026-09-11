"""
MoonPoly adapter for the gemm track.

Paper: "Optimizing Dynamic-Shape Neural Networks on Accelerators via
On-the-Fly Micro-Kernel Polymerization" (ASPLOS'24; conf/asplos/YuLZCFX24).
Artifact: https://github.com/LinkZyy/MoonPoly-dev -- a CUTLASS-based
micro-kernel-polymerization GEMM library, pip-installable as a `moonpoly`
torch extension built from moonpoly/core/*.cu (setup.py's CUDAExtension).

Entry point wrapped (ARTIFACT_GUIDE.md rule 1: the artifact's own kernel
call, not its benchmark script): `moonpoly.linear(input, weight)`, the
top-level pybind op declared/registered in moonpoly/core/moonpoly_gemm.cu
(`PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) { m.def("linear", ...); }`). It
computes `output = input @ weight.T` -- identical semantics to
torch.nn.functional.linear; verified directly against source/tests/python/
test_moonpoly_linear.py, which checks it against F.linear. Internally it
dispatches on input's torch dtype (torchDtypeToMoonpoly: fp16/fp32/int8)
through moonpoly::moonpolyLinear -> moonpoly::moonpoly_gemm ->
run_fp16_gemm/run_fp32_gemm, each of which selects among ~40 CUTLASS
row-column-column micro-kernel candidates via the paper's own fitted/
analytic cost model (moonpoly/core/{fp16,fp32}/*_rcc.cu, *_rcc_cost.cu,
moonpoly/generated/{fp16,fp32}/rcc/*.inc) -- that dispatch/selection step
IS the paper's contribution ("on-the-fly micro-kernel polymerization"), so
`moonpoly.linear` is the correct kernel-level entry point, not a lower-level
fixed-micro-kernel call (e.g. one specific `pNN_rcc_run`).

There is also `moonpoly.linear_cpp_selector`, a separate entry point; traced
in moonpoly_gemm.cu it only engages its own fitted-predictor lookup for
fp16 2D contiguous inputs, and falls back to plain `moonpoly.linear` (the
identical code path) for everything else -- including all of fp32. So
`moonpoly.linear` is used uniformly for both precisions this adapter wires;
it is the same dispatch either way, not a downgrade for fp32.

PRECISIONS = ["fp16", "fp32"]: torchDtypeToMoonpoly also accepts int8, but
int8 GEMM has no meaningful correctness gate against a real-valued reference
without a quantization scheme -- out of scope per this integration's brief,
so int8 is not wired here even though the artifact supports it.

GEMM shape mapping: our workload is C (M,N) = A (M,K) @ B (K,N).
`moonpoly.linear(input, weight)` computes input @ weight.T where weight has
shape (out_features, in_features) = (N, K). So prepare() builds
`weight = B.T.contiguous()` -- an explicit transpose-and-copy that IS the
artifact's own required operand layout (its `linear` op has no "B already
transposed" variant reachable from Python), timed as preprocessing per rule
2, exactly like insum's CSR->COO conversion or inferfast's densify+pad.

Batched GEMM (workload.batch > 1, exercised by the gemm smoke set's
batch=4x64x64x64 fp32 shape): `moonpoly.linear` has no native batched entry
point -- `torch_helpers::createOutputTensor` reads `A.sizes()[0]`/
`B.sizes()[0]` positionally, so a 3D input would silently misinterpret the
leading batch dim as M/N; there is no batched-GEMM kernel anywhere under
moonpoly/core to call instead. Per ARTIFACT_GUIDE.md rule 1 ("wrap at the
finest boundary available"), run() loops the real per-batch-element
`moonpoly.linear` call `batch` times -- still the artifact's own unmodified
kernel, launched once per batch element, not a reimplementation of batching.

RNG gotcha (see kernelbench/domains/dense.py's own module docstring and the
insum/inferfast spmm adapters' STATUS.md, which independently rediscovered
the same issue): operands are generated with numpy's `default_rng`,
byte-for-byte matching dense.py's own `_rng_operand` -- generate at the
target precision, THEN move to the GPU -- NOT gpu_cuda.py's `_dense()`
helper, which uses `torch.Generator` and draws a different bit sequence for
the same integer seed.

Known correctness-gate risk, disclosed up front (see dense.py's module
docstring point 3): `spec.py`'s tolerance parser returns the FIRST numeric
bound in the spec's prose, which for gemm's variants is always the fp64
entry (1e-6), even when running fp32/fp16. A correct fp32 GEMM's scaled
error is itself typically right around 1e-6, so this is a known-tight gate
for ANY fp32 GEMM implementation on this harness, not specific to MoonPoly.
See STATUS.md for the actual measured value.
"""

from __future__ import annotations

import os
import sys

import numpy as np

KERNEL = "gemm"
IMPL_NAME = "moonpoly-gemm"
PAPER_KEY = "conf/asplos/YuLZCFX24"
PRECISIONS = ["fp16", "fp32"]

_HERE = os.path.dirname(os.path.abspath(__file__))
_SOURCE = os.path.join(_HERE, "source")

# matches kernelbench/domains/dense.py's own _B_OFFSET exactly, so a batch
# element's B operand never collides with another element's A/B stream.
_B_OFFSET = 1_000_003

_NP_DTYPE = {"fp16": np.float16, "fp32": np.float32}


def _ensure_path():
    if _SOURCE not in sys.path:
        sys.path.insert(0, _SOURCE)


def available() -> tuple[bool, str]:
    if not os.path.exists(os.path.join(_SOURCE, "moonpoly", "core", "moonpoly_gemm.cu")):
        return False, "source/moonpoly/core/moonpoly_gemm.cu missing -- clone not present"
    try:
        import torch
    except Exception as e:
        return False, f"torch import failed: {type(e).__name__}: {e}"
    if not torch.cuda.is_available():
        return False, "CUDA not available on this host"
    _ensure_path()
    try:
        import moonpoly  # noqa: F401
    except Exception as e:
        return False, f"moonpoly extension import failed (run build.sh): {type(e).__name__}: {e}"
    if not hasattr(moonpoly, "linear"):
        return False, "moonpoly extension built but has no .linear() -- unexpected ABI"
    return True, ""


def create(precision: str):
    return MoonPolyGemm(precision)


def _rng_operand(rows: int, cols: int, seed: int, dtype) -> np.ndarray:
    # Byte-for-byte match to kernelbench/domains/dense.py's own _rng_operand
    # -- see module docstring's RNG-gotcha note.
    rng = np.random.default_rng(seed)
    return rng.uniform(-1.0, 1.0, size=(rows, cols)).astype(dtype)


class MoonPolyGemm:
    name = IMPL_NAME
    platform = "cuda"

    def __init__(self, precision: str = "fp32"):
        if precision not in PRECISIONS:
            raise NotImplementedError(
                f"{IMPL_NAME} only wired for {PRECISIONS} (moonpoly.linear's "
                f"own dtype dispatch also covers int8, but int8 GEMM has no "
                f"real-valued correctness reference -- out of scope here); "
                f"requested {precision!r}")
        self.precision = precision

    def prepare(self, workload, params: dict):
        _ensure_path()
        import torch
        import moonpoly  # noqa: F401  -- imported so the torch op is registered

        seed = params.get("seed", workload.seed)
        dt_np = _NP_DTYPE[self.precision]
        dt_torch = torch.float16 if self.precision == "fp16" else torch.float32
        w = workload
        batch = max(w.batch, 1)

        A_list, W_list = [], []
        for b in range(batch):
            A_np = _rng_operand(w.M, w.K, seed + b, dt_np)
            # B is generated exactly as the reference does (K, N), then
            # transposed into moonpoly's required (N, K) "weight" layout --
            # this transpose+copy IS the artifact's own operand-layout
            # requirement (rule 2: preprocessing, timed as such).
            B_np = _rng_operand(w.K, w.N, seed + _B_OFFSET + b, dt_np)
            A_t = torch.as_tensor(A_np, device="cuda", dtype=dt_torch).contiguous()
            W_t = torch.as_tensor(B_np, device="cuda", dtype=dt_torch).t().contiguous()
            A_list.append(A_t)
            W_list.append(W_t)

        return {"A": A_list, "W": W_list, "batch": batch}

    def run(self, h):
        import moonpoly
        outs = [moonpoly.linear(h["A"][b], h["W"][b]) for b in range(h["batch"])]
        if h["batch"] == 1:
            return outs[0]
        import torch
        return torch.stack(outs)

    def to_host(self, out):
        import torch
        return out.detach().to("cpu", dtype=torch.float64).numpy()

    def timer(self):
        from kernelbench.impls.gpu_cuda import CudaEventTimer
        return CudaEventTimer()

    def free(self, h):
        import torch
        h.clear()
        torch.cuda.empty_cache()
