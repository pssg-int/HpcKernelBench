"""
MP-SpMM adapter for the spmm track.

Paper: "Bridging the Gap between Unstructured SpMM and Structured Sparse
Tensor Cores", SC'25. `PAPER_KEY = conf/sc/DongS0LXHZ025`.
Artifact: https://github.com/CGCL-codes/MP-SpMM_SC25 (README-only pointer;
real code is the paper's Zenodo AD/AE release -- see source.provenance).

MP-SpMM converts an UNSTRUCTURED sparse matrix into Ampere's 2:4 structured
sparsity format via graph MATCHING (pair up rows so every pair's union fits a
16x16 tile with <=50% density per 4-element group) + zero-PADDING, then runs
one of two hand-written `mma.sp.sync.aligned.m16n8k16` kernels (fp16 compute,
fp32 accumulate) depending on N.

Kernel entry point wrapped (ARTIFACT_GUIDE.md rule 1): the artifact's own
`sparse_mma_kernel_base_Bhalf2_Cfloat4` (N=32) / `sparse_mma_kernel_base_
Buint64_Cfloat4` (N=128) __global__ kernels, called via `libmpspmm_wrapper.so`
(this directory's wrapper.cu, NOT under source/ -- see its header comment for
why a wrapper was needed: the artifact ships no library, only a `main()`
that reads a preprocessed binary from disk; wrapper.cu #includes
source/mpspmm/SpMM/kernels.cu UNMODIFIED and reproduces main()'s exact
launch configuration, plus a byte-for-byte copy of main()'s inline metadata-
packing helper functions -- full reasoning in wrapper.cu).

prepare() does the artifact's own preprocessing (rule 2, timed as such):
  1. write our workload's CSR to a Matrix-Market file at the exact path
     convention the artifact's preprocessing tool expects (dataset_path/
     dataset/<name>/<name>.mtx), field=real/symmetry=general so every
     explicit entry in our (already-expanded, see kernelbench/matrices.py)
     CSR is written once -- no reliance on the tool's own symmetric-matrix
     expansion path.
  2. run the artifact's own compiled preprocessing binary
     (`mpspmm_preprocess`, built from source/mpspmm/preprocessing/
     impl-iterative-2-4.cpp UNMODIFIED by build.sh) as a subprocess -- this
     IS the paper's matching+padding contribution (a real greedy matching
     heuristic over the sparsity graph), not something this adapter
     reimplements. Its wall time is included in prepare()'s measured
     preprocessing_ms.
  3. parse its output binary (our own small reader; the binary format is
     fully self-describing -- see _read_preprocessed_bin -- so no artifact
     code needed for this step beyond byte layout documented in
     source/mpspmm/SpMM/data_reader.cpp, which this mirrors read-only).
  4. pack the raw metadata array into the 32-bit words `mma.sp.sync` expects,
     via wrapper.cu's `mpspmm_pack_metadata` (verbatim copy of the artifact's
     own packing code, see wrapper.cu).
  5. H2D upload.

IMPORTANT (ARTIFACT_GUIDE.md's MP-SpMM-specific instruction): the padding
this preprocessing introduces must be exact zeros (mathematically inert), so
the correctness gate below compares against the reference computed on the
ORIGINAL, unpadded matrix (harness.run_variant always calls
`reference(matrix, params)` on the same `matrix` object passed to
`prepare()` -- nothing special needed here for that). If the gate fails, that
IS the result (see STATUS.md: it does, at N=128).

N support: the artifact's own `main()` (source/mpspmm/SpMM/spmm_sp_new.cu)
only ever launches a kernel for `N == 32` or `N == 128`
(`if (N==32) ... else if (N==128) ...`, no `else`); any other N is a silent
no-op in the ORIGINAL code (output buffer stays whatever it started as).
This adapter turns that into an honest `NotImplementedError` from prepare()
(ARTIFACT_GUIDE.md rule 8) instead of ever handing the gate a not-really-
computed buffer. The spmm-tensorcore-fp16 variant sweeps N in
{128, 256, 512}: N=256 and N=512 are therefore UNSUPPORTED here (correctly
recorded by the runner), and only N=128 actually runs under that variant's
default sweep. N=32 (the artifact's other supported size, outside the
variant's dim list) can be exercised manually with `--dims 32` -- see
STATUS.md for why this matters (checking whether the known N=128 bug is
N=128-specific).
"""

from __future__ import annotations

import ctypes
import os
import re
import subprocess

import numpy as np

KERNEL = "spmm"
IMPL_NAME = "mpspmm-2-4-iterative"
PAPER_KEY = "conf/sc/DongS0LXHZ025"
PRECISIONS = ["fp16"]

_HERE = os.path.dirname(os.path.abspath(__file__))
_LIB_PATH = os.path.join(_HERE, "libmpspmm_wrapper.so")
_PREPROCESS_BIN = os.path.join(_HERE, "mpspmm_preprocess")
_SOURCE = os.path.join(_HERE, "source")

# our own scratch tree, NOT under source/ (ARTIFACT_GUIDE.md: files we write
# live in <short>/). Layout forced by impl-iterative-2-4.cpp's own hardcoded
# conventions -- see the module docstring above and wrapper.cu.
_WORK = os.path.join(_HERE, "work")
_RUNCWD = os.path.join(_WORK, "_runcwd_a", "_runcwd_b")   # 2 levels under
                                                            # _WORK, so the
                                                            # artifact's own
                                                            # "../../path.txt"
                                                            # resolves to
                                                            # _WORK/path.txt
_PATHTXT = os.path.join(_WORK, "path.txt")

_NAME_RE = re.compile(r"^[A-Za-z0-9_.\-]+$")

_lib = None


def _load_lib():
    global _lib
    if _lib is not None:
        return _lib
    if not os.path.exists(_LIB_PATH):
        raise FileNotFoundError(f"{_LIB_PATH} not built -- run build.sh")
    lib = ctypes.CDLL(_LIB_PATH)
    i32p = ctypes.POINTER(ctypes.c_int32)
    u32p = ctypes.POINTER(ctypes.c_uint32)
    lib.mpspmm_pack_metadata.argtypes = [i32p, ctypes.c_int, i32p, ctypes.c_int, u32p]
    lib.mpspmm_pack_metadata.restype = None
    lib.mpspmm_run.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.c_int,
                               ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
                               ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p]
    lib.mpspmm_run.restype = ctypes.c_int
    _lib = lib
    return _lib


def available() -> tuple[bool, str]:
    if not os.path.exists(_LIB_PATH):
        return False, "libmpspmm_wrapper.so not built -- run build.sh"
    if not os.path.exists(_PREPROCESS_BIN):
        return False, "mpspmm_preprocess not built -- run build.sh"
    try:
        import torch
    except Exception as e:
        return False, f"torch import failed: {type(e).__name__}: {e}"
    if not torch.cuda.is_available():
        return False, "CUDA not available on this host"
    try:
        _load_lib()
    except Exception as e:
        return False, f"ctypes load failed: {type(e).__name__}: {e}"
    return True, ""


def create(precision: str):
    return MPSpMM(precision)


def _ensure_pathtxt():
    os.makedirs(_WORK, exist_ok=True)
    os.makedirs(_RUNCWD, exist_ok=True)
    # get_config_value() parses "key = value" lines; both project_path and
    # dataset_path are read by the tool at various points (see module
    # docstring / impl-iterative-2-4.cpp), point both at our own work tree.
    with open(_PATHTXT, "w") as f:
        f.write(f"dataset_path = {_WORK}\n")
        f.write(f"project_path = {_WORK}\n")


def _read_preprocessed_bin(path: str) -> dict:
    """
    Parses source/mpspmm/SpMM/data_reader.cpp's binary layout (read-only
    mirror of the artifact's own reader -- see module docstring). Format is
    fully self-describing: a little-endian int32 length prefixes every
    array.
    """
    with open(path, "rb") as f:
        def i32(n=1):
            return np.frombuffer(f.read(4 * n), dtype="<i4")

        num_rows = int(i32()[0])
        num_cols = int(i32()[0])
        num_nonzeros = int(i32()[0])
        n_off = int(i32()[0])
        tcblocks_offset = i32(n_off).copy()
        n_opd = int(i32()[0])
        opdA = np.frombuffer(f.read(8 * n_opd), dtype="<f8").copy()
        n_meta = int(i32()[0])
        metadata = i32(n_meta).copy()
        n_oldcol = int(i32()[0])
        old_col_all = i32(n_oldcol).copy()
    return dict(num_rows=num_rows, num_cols=num_cols, num_nonzeros=num_nonzeros,
                tcblocks_offset=tcblocks_offset, opdA=opdA, metadata=metadata,
                old_col_all=old_col_all)


class MPSpMM:
    name = IMPL_NAME
    platform = "cuda"
    _last_shape = None   # (Mp, N, M_orig), set by prepare(), read by to_host()

    def __init__(self, precision: str = "fp16"):
        if precision != "fp16":
            raise NotImplementedError(
                f"{IMPL_NAME} only wired for fp16 (Tensor Core `mma.sp.sync` "
                f"fp16 compute / fp32 accumulate, no fp32 path in the "
                f"artifact); requested {precision}")
        self.precision = precision

    def prepare(self, matrix, params: dict):
        import torch
        lib = _load_lib()

        N = int(params["N"])
        if N not in (32, 128):
            raise NotImplementedError(
                f"{IMPL_NAME}: the artifact only compiles a kernel for N=32 "
                f"or N=128 (source/mpspmm/SpMM/spmm_sp_new.cu main()'s "
                f"`if (N==32) ... else if (N==128) ...` has no else branch, "
                f"i.e. no kernel exists for other N -- see STATUS.md); "
                f"requested N={N}")

        name = matrix.name
        if not _NAME_RE.match(name):
            raise NotImplementedError(
                f"{IMPL_NAME}: matrix name {name!r} contains characters "
                "unsafe for the artifact's file-path-from-name convention "
                "(dataset_path/dataset/<name>/<name>.mtx)")

        _ensure_pathtxt()
        # impl-iterative-2-4.cpp's read_matrix(): mtx_path = "%s/%s/%s.mtx" %
        # (dataset_path, "dataset", mtx_name) -- i.e. dataset_path/dataset/
        # <name>.mtx, a FLAT layout (data_dir="dataset" is a hardcoded
        # constant, not derived from the name -- confirmed empirically, the
        # first version of this adapter guessed a nested <name>/<name>.mtx
        # layout and the artifact's own error message corrected it).
        mtx_dir = os.path.join(_WORK, "dataset")
        os.makedirs(mtx_dir, exist_ok=True)
        mtx_path = os.path.join(mtx_dir, f"{name}.mtx")

        # --- 1. write CSR as Matrix Market (artifact's own input format) ---
        from scipy.io import mmwrite
        # symmetry='general': our CSR (kernelbench.matrices.load_matrix uses
        # scipy.io.mmread, which already expands any symmetric SOURCE file
        # into a full matrix) is written with every explicit entry once,
        # bypassing the tool's own is_symmetric-triggered row-duplication
        # entirely -- avoids any dependence on that path matching scipy's.
        mmwrite(mtx_path, matrix.csr.astype(np.float64), field="real", symmetry="general")

        # --- 2. the artifact's own preprocessing binary (subprocess, timed
        #        as part of this prepare() call -- rule 2) ---
        bin_out = os.path.join(_WORK, "dataset_mp_processed", "2-4",
                                "1.Iterative-matching", f"{name}_data.bin")
        if os.path.exists(bin_out):
            os.remove(bin_out)  # avoid reading a stale file from a prior matrix reused this name
        proc = subprocess.run([_PREPROCESS_BIN, name], cwd=_RUNCWD,
                              capture_output=True, text=True, timeout=600)
        if proc.returncode != 0 or not os.path.exists(bin_out):
            raise RuntimeError(
                f"{IMPL_NAME}: mpspmm_preprocess failed (rc={proc.returncode}) "
                f"for matrix {name!r}.\nstdout(tail): {proc.stdout[-2000:]}\n"
                f"stderr(tail): {proc.stderr[-2000:]}")

        d = _read_preprocessed_bin(bin_out)
        Mp = d["num_rows"]       # already padded to a multiple of 16 by the
                                 # artifact's own read_matrix() (see docstring)
        K = d["num_cols"]
        M_orig = matrix.csr.shape[0]
        assert K == matrix.csr.shape[1], (
            f"{IMPL_NAME}: mtx round-trip K mismatch: wrote {matrix.csr.shape[1]}, "
            f"preprocessing tool read back {K}")
        assert Mp >= M_orig and Mp % 16 == 0, (
            f"{IMPL_NAME}: unexpected padded M={Mp} for original M={M_orig}")

        # --- 3. host-side cleanup identical to spmm_sp_new.cu main() ------
        opdA_clean = np.nan_to_num(d["opdA"], nan=0.0).astype(np.float32).astype(np.float16)
        old_col_clean = np.where(d["old_col_all"] == -1, 0, d["old_col_all"]).astype(np.int32)

        # --- 4. metadata packing via the artifact's own code (wrapper.cu) --
        metadata_new = np.zeros(len(d["metadata"]) // 16, dtype=np.uint32)
        i32p = ctypes.POINTER(ctypes.c_int32)
        u32p = ctypes.POINTER(ctypes.c_uint32)
        lib.mpspmm_pack_metadata(
            d["metadata"].ctypes.data_as(i32p), len(d["metadata"]),
            d["tcblocks_offset"].ctypes.data_as(i32p), len(d["tcblocks_offset"]),
            metadata_new.ctypes.data_as(u32p))

        # --- 5. dense operand B: numpy RNG matching cpu_ref.reference_spmm's
        #        _dense_operand exactly (see insum/inferfast STATUS.md for
        #        why this matters -- gpu_cuda.py's torch-RNG `_dense` helper
        #        draws a DIFFERENT B than the gate's numpy-RNG reference). ---
        rng = np.random.default_rng(params.get("seed", 42))
        B_np = rng.uniform(-1.0, 1.0, size=(K, N)).astype(np.float32).astype(np.float16)

        # --- 6. H2D -------------------------------------------------------
        opdA_dev = torch.from_numpy(opdA_clean).cuda()
        # torch has no uint32; .view() bit-reinterprets (zero-copy, no value
        # cast), so the raw bit pattern the artifact's own packing wrote
        # reaches the device unchanged (same convention as inferfast/adapter.py).
        metadata_dev = torch.from_numpy(metadata_new.view(np.int32)).cuda()
        tcblocks_offset_dev = torch.from_numpy(d["tcblocks_offset"]).cuda()
        old_col_dev = torch.from_numpy(old_col_clean).cuda()
        B_dev = torch.from_numpy(B_np).cuda()
        C_dev = torch.zeros(Mp * N, dtype=torch.float32, device="cuda")

        # to_host() only receives `out` (the Implementation protocol), not the
        # handle -- stash the crop shape here (prepare() is the only place it's
        # known), read there. Same pattern as inferfast/adapter.py.
        self._last_shape = (Mp, N, M_orig)

        return {
            "opdA": opdA_dev, "metadata": metadata_dev,
            "tcblocks_offset": tcblocks_offset_dev, "old_col": old_col_dev,
            "B": B_dev, "C": C_dev,
            "M_orig": M_orig, "Mp": Mp, "N": N, "K": K,
        }

    def run(self, h):
        lib = _load_lib()
        h["C"].zero_()
        ret = lib.mpspmm_run(
            h["Mp"], h["N"], h["K"],
            ctypes.c_void_p(h["opdA"].data_ptr()),
            ctypes.c_void_p(h["metadata"].data_ptr()),
            ctypes.c_void_p(h["tcblocks_offset"].data_ptr()),
            ctypes.c_void_p(h["old_col"].data_ptr()),
            ctypes.c_void_p(h["B"].data_ptr()),
            ctypes.c_void_p(h["C"].data_ptr()))
        if ret != 0:
            raise RuntimeError(f"{IMPL_NAME}: mpspmm_run returned {ret} "
                               f"(cudaError_t, or -1 for unsupported N -- "
                               f"should be unreachable, prepare() filters N)")
        return h["C"]

    def to_host(self, out):
        import torch
        Mp, N, M_orig = self._last_shape
        mat = out.detach().to("cpu", dtype=torch.float64).view(Mp, N)
        return mat[:M_orig, :].numpy()

    def timer(self):
        from kernelbench.impls.gpu_cuda import CudaEventTimer
        return CudaEventTimer()

    def free(self, h):
        import torch
        h.clear()
        torch.cuda.empty_cache()
