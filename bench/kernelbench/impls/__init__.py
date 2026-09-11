"""Implementation registry: kernel -> {name: factory}."""

from __future__ import annotations

from . import cpu_ref

REFERENCES = {
    "spmv": cpu_ref.reference_spmv,
    "spmm": cpu_ref.reference_spmm,
    "sddmm": cpu_ref.reference_sddmm,
}

# correctness comparison mode per kernel, following each spec's wording
CORRECTNESS_MODE = {
    "spmv": "max_scaled_err",   # scale = |A|@|x| (cancellation-robust)
    "spmm": "max_scaled_err",   # scale = |A|@|B|
    "sddmm": "max_scaled_err",  # scale = |S|*(|A|.|B|)
}

_CPU = {
    "spmv": {
        "scipy-csr-spmv": lambda p: cpu_ref.ScipySpMV(p),
        "naive-csr-spmv": lambda p: cpu_ref.NaiveSpMV(p),
    },
    "spmm": {"scipy-csr-spmm": lambda p: cpu_ref.ScipySpMM(p)},
    "sddmm": {"scipy-csr-sddmm": lambda p: cpu_ref.ScipySDDMM(p)},
}


def _cuda_table():
    from . import gpu_cuda as g
    return {
        "spmv": {
            "cusparse-csr-spmv": lambda p: g.TorchSpMV(p),
            "custom-warp-csr-spmv": lambda p: g.CustomSpMV(p),
        },
        "spmm": {
            "cusparse-csr-spmm": lambda p: g.TorchSpMM(p),
            "custom-warp-csr-spmm": lambda p: g.CustomSpMM(p),
        },
        "sddmm": {
            "torch-sampled-addmm-sddmm": lambda p: g.TorchSDDMM(p),
            "custom-warp-csr-sddmm": lambda p: g.CustomSDDMM(p),
        },
    }


def available(kernel: str, platform: str = "cpu") -> list[str]:
    if platform == "cpu":
        return sorted(_CPU.get(kernel, {}))
    try:
        return sorted(_cuda_table().get(kernel, {}))
    except Exception:
        return []


def build(kernel: str, impl_name: str, precision: str):
    if impl_name in _CPU.get(kernel, {}):
        return _CPU[kernel][impl_name](precision)
    table = _cuda_table()
    if impl_name in table.get(kernel, {}):
        return table[kernel][impl_name](precision)
    raise KeyError(
        f"unknown implementation {impl_name!r} for {kernel}; "
        f"cpu={available(kernel,'cpu')} cuda={available(kernel,'cuda')}")
