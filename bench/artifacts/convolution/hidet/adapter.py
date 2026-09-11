"""
Hidet adapter for the convolution track.

Paper: "Hidet: Task-Mapping Programming Paradigm for Deep Learning Tensor
Programs" (ASPLOS'23). PAPER_KEY = "conf/asplos/DingYZLWP23".
Repo: https://github.com/hidet-org/hidet (archived/read-only, but still
`pip install`-able). Installed into an ARTIFACT-LOCAL `pylibs/` directory
(`pip install --target=pylibs --no-deps hidet==0.6.1`, git-ignored, same
pattern as `attention-kernel/pat`'s own `pylibs/`) -- see "Dependency
isolation" below, and build.sh / STATUS.md for the pinned version.

Dependency isolation (2026-09-05): this used to be `pip install hidet` into
the shared plexus_env venv, which collided with
`bench/artifacts/gemm/hexcute/`'s own build -- its fork of hidet ALSO
declares `name = "hidet"`, and whichever install ran last in the shared
site-packages won process-wide for every `import hidet` (see STATUS.md for
the full incident). `_ensure_paths()` below puts this artifact's own
`pylibs/` first on `sys.path` before ANY `import hidet`, so this adapter's
behavior no longer depends on what plexus_env happens to hold at the time.

Hidet is a DL compiler; it exposes a direct OP-LEVEL conv2d API, found by
introspecting the installed package:
`[x for x in dir(hidet.ops) if 'conv' in x.lower()]` lists
conv2d/conv2d_gemm/conv2d_winograd/conv2d_channel_last/... This is the
finest boundary available (ARTIFACT_GUIDE.md rule 1) -- no need to route
through `torch.compile(backend='hidet')` wrapping an nn.Conv2d module. We
build a one-op hidet FlowGraph directly:

    x_sym = hidet.symbol(X.shape, dtype='float32', device='cuda')
    w_sym = hidet.symbol(W.shape, dtype='float32', device='cuda')
    y_sym = hidet.ops.conv2d(x_sym, w_sym, stride=..., padding=..., groups=...)
    graph = hidet.graph.optimize(hidet.trace_from(y_sym, inputs=[x_sym, w_sym]))

prepare() does this trace+optimize (hidet's own graph-level schedule
selection/fusion) AND one warm call through the compiled graph. Hidet
JIT-compiles its generated CUDA source lazily on first invocation of a given
task/shape (confirmed empirically: the FIRST `graph(x, w)` call after
`optimize()` triggers an nvcc build, ~10s wall for the smoke shapes here;
every subsequent call on the same graph/shape object is <1ms, served from an
in-process + on-disk compiled-kernel cache under
`hidet.option.get_cache_dir()`, default `~/.hidet_cache`). Calling it once
inside prepare() puts that one-time JIT-compile cost where the harness
contract says it belongs -- timed once as preprocessing, never folded into
run()'s per-iteration measurement.

Build-environment fix (ARTIFACT_GUIDE.md rule 3 -- NOT a kernel-code patch):
this host's default `gcc`/`g++` on PATH is gcc-native/14 (SUSE gcc 14.3.0).
nvcc 12.9 cannot use it as a host compiler -- libstdc++'s gcc-14-only
internals (`<bits/alloc_traits.h>`, `<bits/hashtable.h>`) throw dozens of
"identifier is undefined" parse errors under nvcc's frontend. hidet resolves
nvcc's host compiler implicitly via whatever `g++`/`gcc` `shutil.which`
finds on PATH (see the installed package's `hidet/backend/build.py`,
`NVCC._resolve_nvcc_path`/compile command construction) -- there is no
hidet config knob to override this except PATH itself. Prepending
`/opt/cray/pe/gcc-native/12/bin` (gcc/g++ 12.3.0, a version nvcc 12.9
explicitly supports) fixes this with zero changes to any hidet or CUDA
source: exactly the "arch flags, include paths, CUDA-version guards" class
of fix rule 3 blesses (a compatible-host-compiler guard, same idea).

Correctness/precision finding (see STATUS.md for the full numeric gate
result) -- not an adapter bug: hidet's default fp32 schedule for groups=1
convolutions lowers through im2col + a tensor-core batched-GEMM
(`conv2d_gemm_image_transform ... batch_matmul`, confirmed from hidet's own
compile-log task names). `hidet/graph/ops/matmul/cuda_batch_matmul.py`'s
`resolve_mma_type` maps ANY non-fp16/bf16 input dtype pair -- including real
float32 -- to `mma_tf32_f32`: Ampere TF32 tensor-core MMA (~10-bit mantissa),
not full-precision FP32 CUDA-core multiply-accumulate. There is no
dtype/schedule flag exposed at the `hidet.ops.conv2d` level to opt out of
this; it is baked into the matmul schedule's dtype-rank resolution.
Depthwise/grouped convs take a different, non-GEMM direct-conv2d code path
(hidet's compile log shows a plain `conv2d(...)` task, no batch_matmul) and
are essentially unaffected: measured max_scaled_err ~1e-7 there vs ~1e-3 on
the dense/groups=1 GEMM path (see STATUS.md's per-shape smoke numbers). This
is a real, documented characteristic of hidet's default fp32 schedule on
this hardware -- rule 3: no kernel-code patch was attempted to work around
it; rule 4: the gate is reported as-is, not loosened.
"""

from __future__ import annotations

import os
import sys

KERNEL = "convolution"
IMPL_NAME = "hidet-conv2d"
PAPER_KEY = "conf/asplos/DingYZLWP23"
PRECISIONS = ["fp32"]

HERE = os.path.dirname(os.path.abspath(__file__))
PYLIBS = os.path.join(HERE, "pylibs")

# --- build-environment fix: prepend an nvcc-12.9-compatible host compiler.
# See module docstring. Applied unconditionally at import time (before any
# code path here can trigger hidet's first nvcc invocation) and is
# idempotent -- re-running just no-ops if already prepended.
_COMPAT_GCC_BIN = os.environ.get("KB_HOST_COMPILER_BIN", "/opt/cray/pe/gcc-native/12/bin")


def _ensure_compatible_host_compiler() -> None:
    path = os.environ.get("PATH", "")
    entries = path.split(os.pathsep) if path else []
    if os.path.isdir(_COMPAT_GCC_BIN) and (not entries or entries[0] != _COMPAT_GCC_BIN):
        os.environ["PATH"] = _COMPAT_GCC_BIN + os.pathsep + path
    # zaratan-specific addendum (2026-09-09): a PATH prepend alone is not
    # enough on this machine. hidet's NVCC.compile() (installed
    # hidet/backend/build.py) never passes -ccbin, so nvcc does its own
    # internal host-compiler search -- and conda-forge's nvcc (its binary
    # lives under $CUDA_HOME/bin, a full conda env that ALSO ships its own
    # gcc/g++ 13.4.0) prepends a few install-relative dirs resolving back to
    # $CUDA_HOME/bin ahead of our PATH, so it finds that gcc 13 first
    # regardless of PATH order (verified empirically, see STATUS.md's
    # Reproduction section). NVCC_APPEND_FLAGS is a real nvcc feature (extra
    # flags appended to every invocation) -- an explicit -ccbin there
    # bypasses nvcc's own host-compiler auto-search entirely. Perlmutter's
    # NVIDIA HPC SDK nvcc install dir has no competing gcc, so this was not
    # needed there; harmless / no-op if $CUDA_HOME has no bundled gcc.
    if "-ccbin" not in os.environ.get("NVCC_APPEND_FLAGS", ""):
        _gxx = os.path.join(_COMPAT_GCC_BIN, "g++")
        if os.path.exists(_gxx):
            os.environ["NVCC_APPEND_FLAGS"] = (
                os.environ.get("NVCC_APPEND_FLAGS", "") + f" -ccbin {_gxx}"
            ).strip()


def _ensure_paths() -> None:
    """Put this artifact's own pylibs/ FIRST on sys.path, ahead of anything
    the shared plexus_env venv holds, so `import hidet` below always resolves
    to OUR pinned copy regardless of what any other artifact (e.g.
    gemm/hexcute, whose fork also declares `name = "hidet"`) has installed
    into the shared venv. See module docstring's "Dependency isolation"
    section. Idempotent."""
    if PYLIBS not in sys.path:
        sys.path.insert(0, PYLIBS)


_ensure_compatible_host_compiler()
_ensure_paths()


def available() -> tuple[bool, str]:
    """Cheap and non-raising, per the adapter contract: import checks only,
    no CUDA source compilation attempted here (that happens in prepare()).

    Also guards against the shared-venv name collision documented in the
    module docstring ("Dependency isolation") and STATUS.md: plexus_env is
    shared with bench/artifacts/gemm/hexcute/, whose own fork of hidet ALSO
    declares `name = "hidet"` in its package metadata. `_ensure_paths()`
    prepends this artifact's own pylibs/ to sys.path so that should no
    longer matter in practice, but this check stays as a cheap, concrete
    guard against the collision recurring (e.g. if `sys.modules['hidet']`
    was already populated by something else earlier in this same process)
    rather than silently running the wrong package."""
    _ensure_paths()
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
    if not hidet_file.startswith(os.path.normpath(PYLIBS) + os.sep):
        return False, (
            f"hidet resolved to {hidet.__file__!r}, not this artifact's own "
            f"pylibs/ ({PYLIBS!r}) -- likely a shared-venv/sys.modules "
            "collision with gemm/hexcute's own 'hidet'-named package (or "
            "pylibs/ not built yet, run build.sh); see STATUS.md")
    if not hasattr(hidet, "ops") or not hasattr(hidet.ops, "conv2d"):
        return False, f"hidet {getattr(hidet, '__version__', '?')} has no ops.conv2d"
    return True, ""


def create(precision: str):
    return HidetConv2d(precision)


class HidetConv2d:
    name = IMPL_NAME
    platform = "cuda"

    def __init__(self, precision: str = "fp32"):
        if precision != "fp32":
            raise NotImplementedError(
                f"{IMPL_NAME} only wired for fp32 (PRECISIONS={PRECISIONS}); "
                f"requested {precision!r}")
        self.precision = precision

    def prepare(self, workload, params: dict):
        _ensure_compatible_host_compiler()
        _ensure_paths()
        import numpy as np
        import hidet

        w = workload
        N = int(params.get("N", w.N))
        seed = params.get("seed", w.seed)

        # --- EXACT operand recipe, matching kernelbench/domains/ml.py's
        # _make_conv_operands bit-for-bit: numpy default_rng, seed, X drawn
        # THEN W, U(-1,1), run-precision dtype. Must match or the
        # correctness gate compares against operands nobody actually
        # multiplied (the exact trap documented in
        # artifacts/spmm/insum/STATUS.md, sidestepped here by not using
        # gpu_cuda.py's torch-Generator-seeded `_dense` helper).
        rng = np.random.default_rng(seed)
        X = rng.uniform(-1.0, 1.0, size=(N, w.Cin, w.Hin, w.Win)).astype(np.float32)
        W = rng.uniform(-1.0, 1.0, size=(w.Cout, w.cin_per_group, w.Kh, w.Kw)).astype(np.float32)

        # --- build + optimize a one-op hidet FlowGraph. This graph-level
        # schedule search/fusion pass IS the artifact's own preprocessing.
        x_sym = hidet.symbol(list(X.shape), dtype="float32", device="cuda")
        w_sym = hidet.symbol(list(W.shape), dtype="float32", device="cuda")
        y_sym = hidet.ops.conv2d(
            x_sym, w_sym, stride=(w.stride, w.stride),
            padding=(w.pad_h, w.pad_w), groups=w.groups)
        graph = hidet.trace_from(y_sym, inputs=[x_sym, w_sym])
        graph = hidet.graph.optimize(graph)

        x_h = hidet.asarray(X, device="cuda")
        w_h = hidet.asarray(W, device="cuda")

        # --- warm call: forces hidet's lazy nvcc JIT-compile of the
        # generated CUDA source for this exact shape/schedule NOW, inside
        # prepare()'s timed preprocessing window, not inside the first
        # measured run() call. See module docstring for the ~10s-then-<1ms
        # behavior this was empirically confirmed against.
        graph(x_h, w_h)

        return {"graph": graph, "x": x_h, "w": w_h}

    def run(self, h):
        return h["graph"](h["x"], h["w"])

    def to_host(self, out):
        import torch
        # hidet.Tensor.torch() shares memory with the hidet tensor (no
        # copy) -- same interop idiom TorchConv2d/InsumSpMM use for their
        # own device tensors, just crossing the hidet/torch boundary first.
        return out.torch().detach().to("cpu", dtype=torch.float64).numpy()

    def timer(self):
        # Reused, not reimplemented: one CUDA-event pair per iteration, per
        # the specs. hidet defaults to executing on torch's current CUDA
        # stream (confirmed: hidet.option.is_use_torch_stream() == True out
        # of the box), so torch.cuda.Event correctly brackets hidet's kernel
        # launches -- same CUDA context/stream, verified empirically (5
        # timed hidet graph calls via this exact timer settled at a stable
        # ~0.18ms after one slightly-elevated first sample, consistent with
        # ordinary device-side event timing, not a stream-mismatch artifact
        # that would read as ~0 or wildly inconsistent).
        from kernelbench.impls.gpu_cuda import CudaEventTimer
        return CudaEventTimer()

    def free(self, h):
        import torch
        h.clear()
        torch.cuda.empty_cache()
