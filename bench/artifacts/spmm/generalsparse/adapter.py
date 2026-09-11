"""
GeneralSparse adapter for the spmm track.

Paper: "GeneralSparse: Bridging the Gap in SpMM for Pruned Large Language
Model Inference on GPUs", ATC'25. `PAPER_KEY = conf/usenix/WangGXCT25`.
Artifact: https://github.com/Wangyaoyuu/GeneralSparse (also benchmarks
SuiteSparse in its own README, matching this track's regime).

## What the artifact actually is

GeneralSparse is NOT a fixed kernel: it is a host-side C++ CODE GENERATOR
(source/code_generator.cc, source/code_builder.cc, ~330KB of it) that, per
input matrix, composes a chain of "operators" (box-division / thread-blocking
/ reduction strategies -- the paper's actual contribution, "the process of
dividing box to adapt dynamically to diverse pruning patterns") into a
bespoke `__global__ void kernel_<id>(...)` CUDA kernel, emits it as
`kernel_file.cu`, and (in the artifact's own driver, `source/token_test.cc`)
compiles+runs it via nvcc + a subprocess, timing a batch of `KERNAL_REPEAT_TIME`
(1200-10000, config-dependent) iterations with `gettimeofday` wall-clock.
`token_test.cc::main()` hardcodes a BENCHMARK SCRIPT: it loops through ~30
different such operator compositions per matrix and keeps whichever compiles
+runs fastest (README's own workflow: token_test -> data_source/<id>/a.out
for every candidate -> obtain_result.py greps all of them for the max
GFLOP/s). ARTIFACT_GUIDE.md rule 1 says wrap the kernel, not the paper's
benchmark script -- so this adapter does NOT run that ~30-strategy search on
every gate call. Instead:

  - `gs_emit.cc` (this directory, NOT under source/) reproduces exactly ONE
    of those strategies -- `test_spmm_warp_bitmap`'s operator sequence,
    copied verbatim from token_test.cc:1253-1308 (fixed column-direction
    thread blocking + a warp-bitmap reduction, a real, representative
    GeneralSparse composition, not a simplification of it) -- and stops at
    `code_generator::generate_final_program(1)`, which only WRITES
    kernel_file.cu (repeat_num=1, so it contains exactly one un-repeated
    kernel launch) without compiling or running it. Building the full
    30-strategy autotuning search for every gate call is out of scope for a
    login-node integration (documented here, not silently dropped); the
    KERNEL this adapter measures is byte-for-byte what GeneralSparse's own
    code_builder/code_generator produce for this operator sequence, not a
    reimplementation.
  - `gs_transform.py` (this directory) turns that generated kernel_file.cu
    into a small shared library exposing `kb_init()` (host reads + H2D
    copies of the box-division metadata this strategy needs -- GeneralSparse's
    own real preprocessing output, rule 2) and `kb_run(c_ptr, b_ptr)` (the
    ONE kernel-launch statement, textually lifted unchanged, now taking
    harness-supplied C/B device pointers instead of GeneralSparse's own
    generated host arrays). See that module's docstring for the exact,
    empirically-verified text split.

## IMPLEMENTATION FINDING: GeneralSparse's own .mtx reader discards real
## nonzero VALUES (independently confirmed in source code, not inferred from
## gate numbers -- same evidentiary bar as this track's dtcspmm/flashsparse
## finding in ../README.md)

`source/struct.cc::get_matrix_index_and_val_from_file` (both branches, lines
~140 and ~178) parses `val = atof(sv[2].c_str())` from each input line but
NEVER uses it: every stored value is instead hardcoded via
`float_val_vec.push_back(1)` / `double_val_vec.push_back(1)`. This is
independent of `HALF`/precision config -- it happens before any GPU code
exists, at plain host-side matrix ingestion. Effect: the sparse operand this
artifact's own kernel entry point ever computes with is the BINARY PATTERN of
whatever matrix is fed in, never its real weights -- structurally identical
in kind (though a different root cause: this is dead code in the file
reader, not the discard-on-purpose GNN-adjacency preprocessing traced for
dtcspmm/flashsparse) to the finding already documented for this track's
TC-GNN-lineage baselines. A trivial one-line-per-branch fix exists
(`push_back(val)` instead of `push_back(1)`) but is deliberately NOT applied
here, for the same reason dtcspmm/flashsparse's discard was left alone:
consistent, disclosed treatment of the same class of artifact behavior
across this track's baselines, rather than silently fixing one of three.

The DENSE operand (`x_arr`/B) is a separate matter and is NOT similarly
stuck: GeneralSparse's own generated code hardcodes it to all-ones too
(`x_arr[i] = 1;`, in `generate_matrix_format_read_code`), but this adapter's
`kb_run(c_ptr, b_ptr)` never emits or uses that generated x_arr/y_arr code at
all (see gs_transform.py's `_CUT_MARKER`) -- B and C are real harness-owned
CUDA tensors passed in as pointers, so this adapter DOES exercise the real
kernel arithmetic against a genuine `U(-1,1)` dense operand (matching
`cpu_ref.reference_spmm`'s `_dense_operand` exactly, same convention as
rode/insum/mp-spmm). Net correctness-gate expectation, precisely because of
the sparse-side finding above: PASS on matrices whose real values already
ARE the all-ones pattern (e.g. unweighted graphs: cora, citeseer, ...),
FAIL/mismatch on any matrix with real nonzero weights (e.g. cant, consph,
pdb1HYS, ...) -- this is a REAL result, not a workaround, and the gate is not
loosened to accommodate it (ARTIFACT_GUIDE.md rule 4).

## IMPLEMENTATION FINDING: a genuine double-accumulation bug in the
## artifact's OWN generated correctness check

Every generated `kernel_file.cu` launches `kernel_<id>` TWICE: once as an
unused "warm-up" immediately after setup, then a stray
`cudaMemcpy(y_arr, d_y_arr, ..., D2H)` of THAT warm-up result (into the SAME
host buffer later compared against the reference), THEN `gettimeofday(start)`,
THEN the "timed" launch (again writing into the SAME never-re-zeroed
`d_y_arr` via `atomicAdd`), THEN `gettimeofday(end)` -- with NO further D2H
copy afterward. So GeneralSparse's own `check_result()` call verifies the
WARM-UP run's output, not the timed run's (whose device buffor, after two
un-zeroed atomicAdd accumulations, is silently never inspected). Confirmed
by direct inspection of a real generated file (this adapter's own
`gs_transform.py` module docstring) -- not inferred. This adapter's
`run()` (below) zeroes C before every call precisely to avoid inheriting
this bug: repeated calls were verified empirically (while building this
adapter) to return bit-identical output when C is zeroed each time.

## Precision / variant

GeneralSparse's kernel body operates entirely in `half` (this machine's
`global_config.json` ships with `"HALF": true`, and the artifact provides no
CLI switch for it) -- fp16 compute, fp16 ACCUMULATE (register variables
`THREAD_META_one_register_result`/`WARP_META_one_register_result` are
declared `half`, not `float`), a strictly lower-precision-accumulate variant
than `spmm-tensorcore-fp16`'s "fp16 compute, fp32 accumulate" claim. Per this
task's instruction ("gate under the variant/precision it computes in"), this
is still gated under `spmm-tensorcore-fp16` (the variant exists precisely to
give any fp16-CLASS kernel a disclosed, non-1e-4-fp32-gated home; it is not
restricted to literal `mma.sync` PTX), with the fp16-accumulate detail
recorded here rather than silently assumed away.
"""

from __future__ import annotations

import ctypes
import os
import re
import subprocess
import sys

import numpy as np

KERNEL = "spmm"
IMPL_NAME = "generalsparse-warpbitmap"
PAPER_KEY = "conf/usenix/WangGXCT25"
PRECISIONS = ["fp16"]

_HERE = os.path.dirname(os.path.abspath(__file__))
_SOURCE = os.path.join(_HERE, "source")
_GS_EMIT = os.path.join(_HERE, "gs_emit")
_KERNEL_LIB_DIR = os.path.join(_SOURCE, "cuda_code")
_WORK = os.path.join(_HERE, "work")
_NVCC = os.environ.get("NVCC") or os.path.join(
    os.environ.get("CUDA_HOME", "/usr/local/cuda"), "bin", "nvcc")
_HOST_COMPILER = (os.environ.get("HOST_COMPILER") or os.environ.get("CXX")
                  or "/opt/cray/pe/gcc-native/14/bin/g++")

sys.path.insert(0, _HERE)
import gs_transform  # noqa: E402


def available() -> tuple[bool, str]:
    if not os.path.exists(_GS_EMIT):
        return False, "gs_emit not built -- run build.sh"
    if not os.path.exists(os.path.join(_KERNEL_LIB_DIR, "kernel_lib.hpp")):
        return False, "source/cuda_code/kernel_lib.hpp missing -- clone not present"
    try:
        import torch
    except Exception as e:
        return False, f"torch import failed: {type(e).__name__}: {e}"
    if not torch.cuda.is_available():
        return False, "CUDA not available on this host"
    return True, ""


def create(precision: str):
    return GeneralSparseSpMM(precision)


_OUTPUT_ID_RE = re.compile(r"^\d+$")


class GeneralSparseSpMM:
    name = IMPL_NAME
    platform = "cuda"

    def __init__(self, precision: str = "fp16"):
        if precision != "fp16":
            raise NotImplementedError(
                f"{IMPL_NAME}: GeneralSparse's generated kernels are compiled "
                f"entirely in `half` (global_config.json HALF=true, no CLI "
                f"switch -- see module docstring); requested {precision}")
        self.precision = precision
        os.makedirs(_WORK, exist_ok=True)

    def prepare(self, matrix, params: dict):
        import torch
        from scipy.io import mmwrite

        N = int(params["N"])
        M, K = matrix.csr.shape

        # --- write our workload as GeneralSparse's own input format
        #     (real values written for provenance even though struct.cc
        #     discards them -- see module docstring) ---
        mtx_path = os.path.join(_WORK, f"{matrix.name}.mtx")
        mmwrite(mtx_path, matrix.csr.astype(np.float64), field="real", symmetry="general")

        # --- run ONE fixed GeneralSparse operator composition (NOT its own
        #     ~30-strategy search -- see module docstring), timed as part of
        #     this prepare() call (ARTIFACT_GUIDE.md rule 2) ---
        proc = subprocess.run([_GS_EMIT, mtx_path, str(N)], cwd=_SOURCE,
                              capture_output=True, text=True, timeout=300)
        out_lines = [ln.strip() for ln in proc.stdout.splitlines() if ln.strip()]
        output_id = out_lines[-1] if out_lines else ""
        if proc.returncode != 0 or not _OUTPUT_ID_RE.match(output_id):
            # gs_emit reproduces exactly ONE of GeneralSparse's own ~30
            # operator compositions (test_spmm_warp_bitmap -- see module
            # docstring); GeneralSparse's own token_test.cc main() tries all
            # ~30 and keeps whichever one's internal assert()s don't fire
            # for a given matrix's structure (that IS its adaptive-search
            # contribution). Restricted to one fixed composition, some
            # matrices trip that composition's own internal
            # `assert(is_valid_according_to_metadata())` / padding-rate
            # bound (observed directly: e.g. cora aborts with "current
            # padding rate is 4.00047, higher than 4" under this specific
            # composition's parameters) -- a real, matrix-dependent
            # limitation of using one fixed strategy rather than the full
            # search, not a crash to paper over (ARTIFACT_GUIDE.md rule 8).
            raise NotImplementedError(
                f"{IMPL_NAME}: GeneralSparse's warp_bitmap operator "
                f"composition (the one fixed strategy this adapter uses, "
                f"see module docstring) does not accept {matrix.name!r} "
                f"N={N} -- its own internal metadata-validity assertion "
                f"failed (rc={proc.returncode}). GeneralSparse's own "
                f"benchmark script would fall back to one of ~29 other "
                f"compositions here, which this adapter deliberately does "
                f"not run (rule 1: wrap the kernel, not the search).\n"
                f"stdout(tail): {proc.stdout[-500:]}\n"
                f"stderr(tail): {proc.stderr[-500:]}")

        kernel_file = os.path.join(_SOURCE, "data_source", output_id, "kernel_file.cu")
        with open(kernel_file) as f:
            kernel_text = f.read()

        # --- text surgery: generated kernel_file.cu -> callable shim.cu
        #     (see gs_transform.py's module docstring for the exact split) ---
        shim_text = gs_transform.build_shim(kernel_text, _KERNEL_LIB_DIR)
        shim_path = os.path.join(_WORK, f"shim_{output_id}.cu")
        so_path = os.path.join(_WORK, f"shim_{output_id}.so")
        with open(shim_path, "w") as f:
            f.write(shim_text)

        # --- compile the shim (nvcc), timed as part of this prepare() call ---
        cc = subprocess.run(
            [_NVCC, "-ccbin", _HOST_COMPILER, "-O3", "-std=c++17",
             "-Xcompiler", "-fPIC",
             # arch from KB_SM (env.sh): sm_80 on A100, sm_90 on H100. Hardcoding
             # sm_80 gave a SASS-only shim that failed on H100 (no-kernel-image).
             "-gencode", f"arch=compute_{os.environ.get('KB_SM','80')},code=sm_{os.environ.get('KB_SM','80')}",
             "-shared", shim_path, "-o", so_path],
            capture_output=True, text=True, timeout=300)
        if cc.returncode != 0 or not os.path.exists(so_path):
            raise RuntimeError(
                f"{IMPL_NAME}: nvcc failed compiling {shim_path}\n"
                f"stdout(tail): {cc.stdout[-2000:]}\nstderr(tail): {cc.stderr[-2000:]}")

        lib = ctypes.CDLL(so_path)
        lib.kb_init.argtypes = []
        lib.kb_init.restype = None
        lib.kb_run.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
        lib.kb_run.restype = None
        if hasattr(lib, "kb_free"):
            lib.kb_free.argtypes = []
            lib.kb_free.restype = None
        lib.kb_init()

        # numpy RNG matching cpu_ref.reference_spmm's _dense_operand exactly
        # (see insum/rode/mp-spmm adapters' docstrings for why this matters).
        rng = np.random.default_rng(params.get("seed", 42))
        B_np = rng.uniform(-1.0, 1.0, size=(K, N)).astype(np.float32)
        B = torch.as_tensor(B_np, device="cuda", dtype=torch.float16)
        C = torch.zeros((M, N), dtype=torch.float16, device="cuda")

        return {"lib": lib, "B": B, "C": C, "so_path": so_path}

    def run(self, h):
        # kernel_<id> atomicAdd()s into C (row-decomposition-style reduction,
        # same convention as rode/insum/mp-spmm) -- zero before every call so
        # repeated timed reps each compute a fresh C = A@B, not a running
        # sum (see module docstring's double-accumulation finding).
        h["C"].zero_()
        h["lib"].kb_run(ctypes.c_void_p(h["C"].data_ptr()),
                        ctypes.c_void_p(h["B"].data_ptr()))
        return h["C"]

    def to_host(self, out):
        import torch
        return out.detach().to("cpu", dtype=torch.float64).numpy()

    def timer(self):
        from kernelbench.impls.gpu_cuda import CudaEventTimer
        return CudaEventTimer()

    def free(self, h):
        import torch
        if hasattr(h["lib"], "kb_free"):
            h["lib"].kb_free()
        h.clear()
        torch.cuda.empty_cache()
