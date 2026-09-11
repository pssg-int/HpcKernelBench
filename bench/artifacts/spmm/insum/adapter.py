"""
Insum / IndirectEinsum adapter for the spmm track.

Paper: "Insum: Sparse GPU Kernels Simplified and Optimized with Indirect
Einsums" (ASPLOS'26; conf/asplos/WonAAE26in output/included.json).
Artifact: https://github.com/nullplay/IndirectEinsum

`Insum(einsum_str, **tensors)` lowers an indirect-einsum expression (gather/
scatter with index indirection, accumulation via `+=`) through
`torch.compile` -> `torch._inductor` into a fused Triton kernel. The repo's
own source/example.py demonstrates exactly the SpMM kernel this track needs:

    C[Row[p],n] += Val[p] * B[Col[p],n]

i.e. COO-format sparse-times-dense, which is a genuine SpMM kernel (not the
paper's benchmark harness -- we call the same `Insum(...)` entry point
directly, per ARTIFACT_GUIDE.md rule 1).

Environment note (see build.sh / STATUS.md): the README's fastest path needs
`torch._inductor.config.triton.native_matmul`, a knob shipped in PyTorch
2.10; this machine runs torch 2.8.0+cu128, where the attribute does not
exist. source/example.py was patched with a one-line `hasattr` guard (no
kernel code touched) so the identical Insum call still compiles via
torch.compile's default Triton lowering.

prepare() does the artifact's own format conversion: our workload's CSR is
converted to COO (row/col/value arrays) -- the format Insum's SpMM expression
actually consumes -- then moved to the GPU. That conversion is timed as
preprocessing per the harness contract.

B is intentionally generated with numpy's `default_rng` (matching
kernelbench.impls.cpu_ref.reference_spmm's `_dense_operand`, which is what
the correctness gate compares against) rather than the torch-Generator based
`_dense` helper in kernelbench/impls/gpu_cuda.py. Those two RNGs draw
different bit sequences from the same integer seed, so reusing the torch
helper here would make the gate compare Insum's output against a numerically
different B than the one actually multiplied -- not an Insum problem, but a
mismatch that also affects the harness's OWN built-in cusparse-csr-spmm/
custom-warp-csr-spmm impls (reproduced independently while building this
adapter and reported as a finding in STATUS.md). Matching the reference's own
operand generation directly sidesteps it for this adapter without touching
harness code.
"""

from __future__ import annotations

import os
import sys

KERNEL = "spmm"
IMPL_NAME = "insum-spmm-coo"
PAPER_KEY = "conf/asplos/WonAAE26"
PRECISIONS = ["fp32"]

_HERE = os.path.dirname(os.path.abspath(__file__))
_SOURCE = os.path.join(_HERE, "source")


def available() -> tuple[bool, str]:
    if not os.path.exists(os.path.join(_SOURCE, "indirecteinsum.py")):
        return False, "source/indirecteinsum.py missing -- clone not present"
    try:
        import torch
    except Exception as e:
        return False, f"torch import failed: {type(e).__name__}: {e}"
    if not torch.cuda.is_available():
        return False, "CUDA not available on this host"
    try:
        import triton  # noqa: F401
    except Exception as e:
        return False, f"triton import failed: {type(e).__name__}: {e}"
    return True, ""


def create(precision: str):
    return InsumSpMM(precision)


class InsumSpMM:
    name = IMPL_NAME
    platform = "cuda"

    def __init__(self, precision: str = "fp32"):
        if precision != "fp32":
            raise NotImplementedError(
                f"{IMPL_NAME} only wired for fp32 (artifact example is fp32/fp16; "
                f"requested {precision})")
        self.precision = precision
        self._compiled = None

    def _ensure_path(self):
        if _SOURCE not in sys.path:
            sys.path.insert(0, _SOURCE)

    def prepare(self, matrix, params: dict):
        self._ensure_path()
        import numpy as np
        import torch
        from indirecteinsum import Insum

        # --- artifact's own format conversion: CSR -> COO, timed as prep ---
        A = matrix.csr.tocoo()
        A.sum_duplicates()

        rows = torch.as_tensor(A.row.astype("int64"), device="cuda")
        cols = torch.as_tensor(A.col.astype("int64"), device="cuda")
        values = torch.as_tensor(A.data, dtype=torch.float32, device="cuda")

        N = int(params["N"])
        M, K = matrix.csr.shape
        # numpy RNG to match kernelbench.impls.cpu_ref.reference_spmm's
        # _dense_operand exactly (see module docstring) -- not torch's RNG.
        rng = np.random.default_rng(params.get("seed", 42))
        B_np = rng.uniform(-1.0, 1.0, size=(K, N)).astype("float32")
        B = torch.as_tensor(B_np, device="cuda")
        C = torch.zeros((M, N), dtype=torch.float32, device="cuda")

        if self._compiled is None:
            @torch.compile
            def _spmm_coo(C, B, Row, Col, Val):
                return Insum(
                    "C[Row[p],n] += Val[p] * B[Col[p],n]",
                    C=C, B=B, Row=Row, Col=Col, Val=Val,
                )
            self._compiled = _spmm_coo

        return {"C": C, "B": B, "rows": rows, "cols": cols, "values": values}

    def run(self, h):
        # Insum's index_put_ accumulates (+=); zero C so repeated calls in
        # the harness's warmup/measure loop each compute a fresh C = A@B,
        # not a running sum across iterations.
        h["C"].zero_()
        return self._compiled(h["C"], h["B"], h["rows"], h["cols"], h["values"])

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
