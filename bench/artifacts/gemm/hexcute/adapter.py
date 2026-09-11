"""
Hexcute adapter for the gemm track.

Paper: "Hexcute: A Compiler Framework for Automating Layout Synthesis in GPU
Programs" (CGO'26; `conf/cgo/ZhangDSHSP26`).
Artifact: https://github.com/hexcute/hexcute-bench is JUST benchmark scripts
(its own README literally says so) -- the real kernel-generating compiler is
its pinned submodule `hexcute/hidet` (a fork of hidet-org/hidet), cloned here
directly as `source/`, checked out at the pinned commit
`d817e28f3efe4732a57eaead724ee3b45995edfc`.

Hexcute's own contribution is a layout-synthesis compiler pass wired into
hidet's matmul op-resolution: `python/hidet/graph/ops/matmul/resolve.py`'s
`resolve_f16` dispatches to `matmul_f16_cute_experimental` (Hexcute's codegen)
when `hidet.option.hexcute_matmul(strategy='enable')` is set, vs. the stable
non-Hexcute lowering when 'disable'. `examples/cute/benchmark_matmul_hidet_cublas.py`
(at the pinned commit) shows the exact call sequence this adapter reproduces:
trace -> `hidet.graph.optimize` -> `get_compiled_task(0)` ->
`pick_best_candidate` -> the resulting `kernel(a, b, out)` callable IS the
one Hexcute-layout-synthesized GEMM kernel launch (not cuBLAS/cuDNN/Triton --
those are the paper's OWN baselines in that same file, not what we wrap).
`prepare()` hoists everything through candidate selection (rule 2: legitimate
one-shot preprocessing / autotuning); `run()` is exactly the `kernel(...)`
call, nothing else (rule 1).

PRECISION: `resolve_f16` only fires when BOTH operands are 16-bit float
(`Tensor.dtype.is_any_float16()` covers fp16 AND bf16 -- see
`python/hidet/ir/type.py`) AND the contraction dim K and free dim N are both
even (`a.shape[-1] % 2 == b.shape[-1] % 2 == 0` in resolve.py) -- otherwise
`resolve()` falls through to a *different*, non-Hexcute resolver, silently no
longer exercising this paper's own kernel. There is no fp32/fp64 path at all.
Verified working for fp16 only (this module's own standalone smoke test,
see STATUS.md); bf16 was not exercised, so `PRECISIONS = ["fp16"]` only --
not claiming bf16 works without having run it (ARTIFACT_GUIDE: "what the
artifact actually supports"). All three of dense.py's `_smoke_gemm()` shapes
(256/256/256; 384/256/512; 64/64/64x4-batch) have even K and N, so the
Hexcute path genuinely fires for the gate check below; shapes with odd K/N
elsewhere in the gemm spec's full sweep (e.g. 2049, 4097 square) would NOT
trigger `resolve_f16` and are out of scope for this adapter as currently
wired -- flagged here, not hidden.

Build-system note (`bin/nvcc`, see that file's own header comment for the
full story): hidet's own JIT CUDA backend (`python/hidet/backend/build.py`'s
`NVCC` class) resolves "nvcc" via `shutil.which()` and invokes it with NO
host-compiler override knob anywhere in its Python build path (no `-ccbin`,
no `CXX`/`CUDAHOSTCXX` env var read). nvcc 12.9's default host compiler on
this machine (SUSE `g++-14`) fails compiling `<bits/alloc_traits.h>`
(`__has_construct is undefined`) -- the exact issue already diagnosed and
fixed in `spmm/inferfast/build.sh`. Since hidet exposes no override
mechanism, this adapter uses a small `bin/nvcc` PATH shim (pure passthrough
to the real nvcc plus `-ccbin g++-12`) instead of editing hidet's Python --
zero hidet source changes, a build-system/host-compiler-selection fix per
ARTIFACT_GUIDE rule 3.

Dependency isolation (2026-09-05): this adapter used to rely on
`pip install --no-build-isolation -e source` having put this fork into the
shared plexus_env venv (after `pip uninstall`ing whatever "hidet" was there
first). That is now forbidden (plexus_env is shared with other work; do not
uninstall from it or install into it) and was also the actual root cause of
a real bug: `bench/artifacts/convolution/hidet/` ALSO `pip install`s a
package named "hidet" into the same venv, and whichever install ran last won
process-wide for every subsequent `import hidet` in that venv -- the day
convolution/hidet's plain `pip install hidet` ran after this adapter's
editable install, `import hidet` here started resolving to upstream 0.6.1
(no `get_compiled_task` on `FlowGraph`, crashing `prepare()` with
`AttributeError: 'FlowGraph' object has no attribute 'get_compiled_task'`)
while `available()`'s old check (`hasattr(hidet.option, "hexcute_matmul")`)
still returned True -- upstream 0.6.1 ships that SAME option name (a stub;
verified by diffing `option.py` -- present near-verbatim in both), so the
check could not tell the two apart. See STATUS.md's "Dependency isolation
(2026-09-05)" section for the full incident.

Fix: build.sh now runs source/setup.py's `CustomBuildCommand` logic (a plain
CMake build of `libhidet.so`/`libhidet_runtime.so`, then copied into
`source/python/hidet/lib/`) directly, with NO `pip install` of any kind --
`source/python/hidet/libinfo.py` locates its own runtime libraries relative
to the package's own on-disk location, not via any installed-package
metadata, so this works with zero site-packages entry. `_ensure_env()` below
puts `source/python` FIRST on `sys.path` (ahead of anything plexus_env
holds) before `import hidet`, so this adapter's behavior no longer depends
on install order in the shared venv at all -- analogous to
`convolution/hidet`'s own `pylibs/` sys.path trick. `available()` now
verifies `hidet.__file__` actually resolves under THIS artifact's
`source/python/` (not just that some `hidet.option.hexcute_matmul` exists)
and reports the wrong path in its reason string if not, so a collision like
the one above fails loud instead of silently crashing inside `prepare()`.
"""

from __future__ import annotations

import os
import sys

KERNEL = "gemm"
IMPL_NAME = "hexcute-gemm"
PAPER_KEY = "conf/cgo/ZhangDSHSP26"
PRECISIONS = ["fp16"]

_HERE = os.path.dirname(os.path.abspath(__file__))
_SOURCE = os.path.join(_HERE, "source")
_SOURCE_PY = os.path.join(_SOURCE, "python")
_BIN = os.path.join(_HERE, "bin")
_PYLIBS = os.path.join(_HERE, "pylibs")

# matches kernelbench.domains.dense._B_OFFSET exactly -- the offset applied
# to a workload's seed to derive operand B's (independent) RNG stream.
_B_OFFSET = 1_000_003


def _ensure_env():
    """Put bin/ (the nvcc host-compiler shim) first on PATH, source/python
    first on sys.path, so every hidet JIT compile AND every `import hidet`
    in this process resolves to THIS artifact's own fork -- never whatever
    plexus_env's shared site-packages happens to hold at the time (see
    module docstring's "Dependency isolation" section) -- and pylibs/ (this
    artifact's own `pip install --no-deps --target=pylibs`, see build.sh)
    appended to sys.path so hidet's pure-Python runtime dependencies that
    are NOT already present in the shared venv (tomlkit, psutil, tabulate,
    tqdm, nvtx, click, lark, gitpython, importlib_metadata, cuda-python's
    cuda-bindings) resolve without installing anything into the shared venv
    (rule: no pip install into the shared env). Appended, not prepended --
    packages the shared venv already provides (numpy, scipy, torch,
    packaging, requests, filelock, networkx) should keep resolving from
    there, only the missing ones fall through to pylibs/. Idempotent."""
    path = os.environ.get("PATH", "")
    if _BIN not in path.split(os.pathsep):
        os.environ["PATH"] = _BIN + os.pathsep + path
    if _SOURCE_PY not in sys.path:
        sys.path.insert(0, _SOURCE_PY)
    if os.path.isdir(_PYLIBS) and _PYLIBS not in sys.path:
        sys.path.append(_PYLIBS)


def available() -> tuple[bool, str]:
    _ensure_env()
    if not os.path.exists(os.path.join(_SOURCE_PY, "hidet", "__init__.py")):
        return False, "source/python/hidet missing -- clone/checkout not present"
    if not os.path.exists(os.path.join(_SOURCE_PY, "hidet", "lib", "libhidet_runtime.so")):
        return False, "source/python/hidet/lib/libhidet_runtime.so missing -- run build.sh"
    if not os.path.exists(os.path.join(_BIN, "nvcc")):
        return False, "bin/nvcc shim missing -- run build.sh"
    try:
        import torch
    except Exception as e:
        return False, f"torch import failed: {type(e).__name__}: {e}"
    if not torch.cuda.is_available():
        return False, "CUDA not available on this host"
    try:
        import hidet
    except Exception as e:
        return False, f"hidet import failed: {type(e).__name__}: {e}"
    hidet_file = os.path.normpath(os.path.abspath(hidet.__file__))
    if not hidet_file.startswith(os.path.normpath(_SOURCE_PY) + os.sep):
        return False, (
            f"hidet resolved to {hidet.__file__!r}, not this artifact's own "
            f"fork under {_SOURCE_PY!r} -- likely a shared-venv/sys.modules "
            "collision with convolution/hidet's own 'hidet'-named package "
            "(or sys.modules['hidet'] was already cached from earlier in "
            "this same process); see STATUS.md")
    if not hasattr(hidet.option, "hexcute_matmul"):
        return False, f"hidet at {hidet.__file__} has no hexcute_matmul option -- wrong build/fork picked up"
    if not hasattr(hidet.graph.FlowGraph, "get_compiled_task"):
        return False, (
            f"hidet at {hidet.__file__} has hexcute_matmul but FlowGraph has "
            "no get_compiled_task -- wrong build/fork picked up")
    return True, ""


def create(precision: str):
    return HexcuteGemm(precision)


class HexcuteGemm:
    name = IMPL_NAME
    platform = "cuda"

    def __init__(self, precision: str = "fp16"):
        if precision not in PRECISIONS:
            raise NotImplementedError(
                f"{IMPL_NAME} only wired for {PRECISIONS} (hexcute_matmul's "
                f"resolve_f16 path requires a 16-bit-float dtype on both "
                f"operands; requested {precision!r})")
        self.precision = precision
        # Autotuning depth for hidet's own candidate search (see module
        # docstring). 0 = single schedule (seconds per shape) -- the only
        # level that keeps prepare() bounded on a SHARED login node, per this
        # task's own budget. A real, dedicated-allocation timed run should
        # raise this (1 = "tens of schedules, <1 min"; 2 = "arbitrary large",
        # what the paper's own benchmark script uses) via
        # HEXCUTE_SEARCH_SPACE=1|2 -- prepare() is still correctly timed as
        # preprocessing either way (ARTIFACT_GUIDE rule 2), this only trades
        # search depth for prepare() wall-clock.
        self.search_space = int(os.environ.get("HEXCUTE_SEARCH_SPACE", "0"))

    def prepare(self, workload, params: dict):
        _ensure_env()
        import numpy as np
        import torch
        import hidet
        from hidet import ops

        M, N, K = workload.M, workload.N, workload.K
        L = max(workload.batch, 1)
        seed = params.get("seed", workload.seed)
        torch_dtype = torch.float16  # PRECISIONS is fp16-only, see above

        # --- operand generation: EXACTLY dense.py's _rng_operand recipe
        # (np.random.default_rng, uniform(-1,1)) so the correctness gate
        # (reference_gemm, same recipe widened to fp64) compares this
        # kernel's actual input bits, not a numerically different draw --
        # the RNG gotcha flagged in insum/inferfast's adapters (torch's
        # Generator draws a different bit sequence from the same seed).
        def _rng_operand(rows, cols, s):
            rng = np.random.default_rng(s)
            return rng.uniform(-1.0, 1.0, size=(rows, cols)).astype(np.float32)

        if L > 1:
            A_np = np.stack([_rng_operand(M, K, seed + b) for b in range(L)])
            B_np = np.stack([_rng_operand(K, N, seed + _B_OFFSET + b) for b in range(L)])
        else:
            A_np = _rng_operand(M, K, seed)[None, :, :]
            B_np = _rng_operand(K, N, seed + _B_OFFSET)[None, :, :]

        A_t = torch.from_numpy(A_np).to(torch_dtype).cuda()
        B_t = torch.from_numpy(B_np).to(torch_dtype).cuda()

        # --- artifact's own compile/autotune path, hoisted here as one-shot
        # preprocessing (rule 2): trace -> graph.optimize -> candidate
        # selection. Exactly
        # examples/cute/benchmark_matmul_hidet_cublas.py's sequence (see
        # module docstring).
        with hidet.option.context():
            hidet.option.parallel_k(strategy='disabled')
            hidet.option.search_space(self.search_space)
            hidet.option.hexcute_matmul(strategy='enable')
            a_sym = hidet.symbol([L, M, K], dtype="float16", device='cuda')
            b_sym = hidet.symbol([L, K, N], dtype="float16", device='cuda')
            c_sym = ops.matmul(a_sym, b_sym)
            graph_hidet = hidet.trace_from(c_sym, [a_sym, b_sym])
            graph_hidet = hidet.graph.optimize(graph_hidet)

            hidet_args = [hidet.from_torch(A_t), hidet.from_torch(B_t)]
            matmul_task = graph_hidet.get_compiled_task(0)
            out_hidet = matmul_task.create_outputs(hidet_args)[0]
            best = matmul_task.pick_best_candidate(hidet_args, [out_hidet])
            kernel = matmul_task.candidates[best]

        return {
            "kernel": kernel, "a": hidet_args[0], "b": hidet_args[1], "out": out_hidet,
            # kept alive for the handle's lifetime -- not read again, but the
            # compiled `kernel` closure may hold references owned by these.
            "_graph": graph_hidet, "_task": matmul_task,
            "L": L, "M": M, "N": N,
        }

    def run(self, h):
        h["kernel"](h["a"], h["b"], h["out"])
        return h["out"]

    def to_host(self, out):
        import torch
        t = out.torch().to(torch.float64).cpu().numpy()
        # hidet's [L, M, N] batch-leading layout matches reference_gemm's
        # np.stack(...) batch-leading layout directly; squeeze the batch axis
        # back out for L==1 to match the harness's non-batched (M, N)
        # reference shape.
        return t[0] if t.shape[0] == 1 else t

    def timer(self):
        from kernelbench.impls.gpu_cuda import CudaEventTimer
        return CudaEventTimer()

    def free(self, h):
        import torch
        h.clear()
        torch.cuda.empty_cache()
