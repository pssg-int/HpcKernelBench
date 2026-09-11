"""
FlashAttention-T adapter for the attention-kernel track.

Paper: "FlashAttention-T: Towards Fully Tensorized Attention by Exploiting
Tensor-Vector Parallelism" (PPoPP'26; conf/ppopp/Xu0BXX00000C26 in
output/included.json).
Artifact: source/ (this repo's own AE reproduction package), commit
4a204b777f99606635a7a92886ee727d9043d32f. See STATUS.md for the full
provenance/build story -- summary: the Ampere ("ampere/") kernel ships no
pip-installable extension; we JIT-compile a NEW pybind11 boundary
(wrapper.cu, this directory) around the artifact's OWN unmodified
`custom_mha_fwd_{causal,noncausal}` C++ API (cxx-tests/custom_api/
flash_api_custom.cuh), scoped to forward-only, head_dim in {64,128},
fp16 (the CtaM/CtaN/NWarps and macro configuration copied verbatim from the
artifact's own benchmark, cxx-tests/fwd_bench/benches/*.cu / core.cuh).

Shape coverage: only head_dim in {64, 128} was compiled (see STATUS.md for
why -- these are the two head dims the paper's own Figure 8 benchmark uses,
and the CtaM/CtaN tile choice is head_dim-specific). Any other d --
including d=16, which is what the CLI's --smoke synthetic shapes all use --
raises NotImplementedError cleanly from prepare(); see STATUS.md's "Gate
verification" section for the standalone gate script used instead of
--smoke.

Layout: the artifact's custom API expects Q/K/V as (B, S, H, d) (not this
harness's (B, H, S, d)); the permute+contiguous() into that layout is the
artifact's OWN preprocessing step and is done inside prepare(), timed as
such, exactly like insum's CSR->COO conversion (bench/artifacts/spmm/insum/
adapter.py).
"""

from __future__ import annotations

import os

KERNEL = "attention-kernel"
IMPL_NAME = "flashattention-t-fwd"
PAPER_KEY = "conf/ppopp/Xu0BXX00000C26"
PRECISIONS = ["fp16"]  # bf16 template path exists in wrapper.cu but was not
                        # exercised/gated in the budget available -- see
                        # STATUS.md. Trim here rather than claim untested
                        # coverage.

_HERE = os.path.dirname(os.path.abspath(__file__))
_SRC_AMPERE = os.path.join(
    _HERE, "source", "1-figure8-main-results", "flashattention-t", "ampere")
_BUILD_DIR = os.path.join(_HERE, "_build")

_SUPPORTED_HEAD_DIMS = (64, 128)

_ext_cache = None
_ext_error = None


_SO_PATH = os.path.join(_BUILD_DIR, "flashattention_t_custom_fwd.so")


def _load_ext():
    """Import the extension build.sh compiled into _build/ -- directly, as a
    prebuilt Python extension module, WITHOUT going through
    torch.utils.cpp_extension.load().

    History (2026-09-04): this used to call load() with build_directory=_build,
    relying on ninja's no-op check to make it cheap. That broke the moment the
    login node's default cudatoolkit module drifted from 12.9 to 13.2:
    load() picked up the new CUDA_HOME, decided the cache was stale, deleted
    wrapper.cuda.o and started a ~10-minute recompile with the WRONG toolkit
    (against a cu128 torch) from inside available() -- i.e. from inside every
    `runner --list` / registry scan. Compilation now lives ONLY in build.sh
    (which pins CUDA_HOME/PATH to 12.9 explicitly); this loader never compiles.
    `import torch` must come first so libc10/libtorch are resident when the
    extension's own DT_NEEDED entries are resolved (same note as
    bench/artifacts/gemv/packkv/build.sh)."""
    global _ext_cache, _ext_error
    if _ext_cache is not None:
        return _ext_cache
    if _ext_error is not None:
        raise _ext_error
    try:
        if not os.path.exists(_SO_PATH):
            raise FileNotFoundError(
                f"{_SO_PATH} missing -- run build.sh (it compiles wrapper.cu "
                f"into _build/ with CUDA_HOME pinned to 12.9)")
        import torch  # noqa: F401  -- must precede the extension import
        import importlib.machinery
        import importlib.util
        name = "flashattention_t_custom_fwd"
        loader = importlib.machinery.ExtensionFileLoader(name, _SO_PATH)
        spec = importlib.util.spec_from_file_location(name, _SO_PATH, loader=loader)
        mod = importlib.util.module_from_spec(spec)
        loader.exec_module(mod)
        _ext_cache = mod
        return mod
    except Exception as e:  # noqa: BLE001 -- reported via available(), never raised there
        _ext_error = e
        raise


def available() -> tuple[bool, str]:
    """Cheap: file existence + one dlopen of the prebuilt .so. Never compiles
    (see _load_ext's history note) and never raises."""
    if not os.path.exists(_SO_PATH):
        return False, "build.sh has not been run (_build/flashattention_t_custom_fwd.so missing)"
    try:
        import torch
    except Exception as e:
        return False, f"torch import failed: {type(e).__name__}: {e}"
    if not torch.cuda.is_available():
        return False, "CUDA not available on this host"
    try:
        _load_ext()
    except Exception as e:
        msg = f"extension load failed: {type(e).__name__}: {e}"
        if "CXXABI_1.3.15" in str(e):
            # See STATUS.md "Finding: a second, distinct libstdc++ ABI trap"
            # -- this venv's python transitively loads an old bundled
            # libstdc++ (missing CXXABI_1.3.15) as soon as `torch` is
            # imported anywhere in the process; our extension (compiled with
            # system g++-14) then can't resolve that symbol against the
            # already-resident old copy. Fix is a process-startup env var,
            # not fixable from inside this already-running interpreter (an
            # LD_PRELOAD set via os.environ here would be a documented no-op
            # -- the loader already resolved everything by import time).
            msg += (" -- re-run with LD_PRELOAD=/usr/lib64/libstdc++.so.6 "
                    "set BEFORE the python process starts; see STATUS.md")
        return False, msg
    return True, ""


def create(precision: str):
    return FlashAttentionT(precision)


class FlashAttentionT:
    name = IMPL_NAME
    platform = "cuda"

    def __init__(self, precision: str = "fp16"):
        if precision not in PRECISIONS:
            raise NotImplementedError(
                f"{IMPL_NAME}: only {PRECISIONS} compiled/gated, got {precision!r} "
                "(see STATUS.md)")
        self.precision = precision

    def prepare(self, workload, params: dict):
        import numpy as np
        import torch

        d = workload.d
        if d not in _SUPPORTED_HEAD_DIMS:
            raise NotImplementedError(
                f"{IMPL_NAME}: only compiled explicit template instantiations for "
                f"head_dim in {_SUPPORTED_HEAD_DIMS} (wrapper.cu); workload {workload.name!r} "
                f"has d={d}. This includes every kernelbench --smoke synthetic shape, which "
                f"is fixed at d=16 -- see STATUS.md's Gate verification section for the "
                f"standalone gate script used instead of --smoke.")

        ext = _load_ext()

        # Operand generation: EXACT match to kernelbench.domains.ml._qkv's
        # np.random.default_rng formula (per this adapter's module docstring
        # and the project's documented RNG-mismatch trap, see
        # bench/artifacts/spmm/insum/STATUS.md's "Finding, not an Insum bug").
        # Do NOT use the artifact's own RNG.
        seed = params.get("seed", workload.seed)
        rng = np.random.default_rng(seed)
        np_dtype = {"fp32": np.float32, "fp16": np.float16, "bf16": np.float32}[self.precision]
        Q = rng.standard_normal((workload.B, workload.H, workload.Sq, workload.d)).astype(np_dtype)
        K = rng.standard_normal((workload.B, workload.Hkv, workload.Sk, workload.d)).astype(np_dtype)
        V = rng.standard_normal((workload.B, workload.Hkv, workload.Sk, workload.d)).astype(np_dtype)

        torch_dtype = {"fp16": torch.float16, "bf16": torch.bfloat16}[self.precision]
        Qt = torch.from_numpy(Q).to(device="cuda", dtype=torch_dtype)
        Kt = torch.from_numpy(K).to(device="cuda", dtype=torch_dtype)
        Vt = torch.from_numpy(V).to(device="cuda", dtype=torch_dtype)

        # Artifact's own required layout is (B, S, H, d), not this harness's
        # (B, H, S, d) -- the permute+contiguous() IS the artifact's format
        # conversion, timed here as preprocessing (ARTIFACT_GUIDE.md rule 2).
        Qt = Qt.permute(0, 2, 1, 3).contiguous()
        Kt = Kt.permute(0, 2, 1, 3).contiguous()
        Vt = Vt.permute(0, 2, 1, 3).contiguous()

        is_causal = (workload.mask == "causal")
        softmax_scale = 1.0 / (workload.d ** 0.5)

        return {
            "ext": ext,
            "q": Qt, "k": Kt, "v": Vt,
            "is_causal": is_causal,
            "softmax_scale": softmax_scale,
            "d": d,
            "is_bf16": self.precision == "bf16",
        }

    def run(self, h):
        # ONE forward call to the artifact's own (unmodified) kernel entry
        # point via wrapper.cu's fat_fwd -- not the paper's benchmark driver
        # (ARTIFACT_GUIDE.md rule 1).
        return h["ext"].fat_fwd(
            h["q"], h["k"], h["v"], h["is_causal"], h["softmax_scale"], h["d"], h["is_bf16"])

    def to_host(self, out):
        import torch
        # kernel output is (B, Sq, H, d) per the artifact's own layout;
        # transpose back to (B, H, Sq, d) to match reference_attention's
        # layout before the harness compares.
        return out.detach().permute(0, 2, 1, 3).contiguous().to("cpu", dtype=torch.float64).numpy()

    def timer(self):
        from kernelbench.impls.gpu_cuda import CudaEventTimer
        return CudaEventTimer()

    def free(self, h):
        import torch
        h.clear()
        torch.cuda.empty_cache()
