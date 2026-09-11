"""
hicma-x (HiCMA-PaRSEC) adapter for the cholesky track.

Paper: "Toward Capturing Genetic Epistasis From Multivariate Genome-Wide
Association Studies Using Mixed-Precision Kernel Ridge Regression" (SC'24,
ACM Gordon Bell Prize Finalist). `PAPER_KEY = conf/sc/LtaiefACRSKDBAK24`
(title-matched in ../../../output/included.json). Also the current home of
two earlier papers on the same repo lineage that this integration does not
separately wrap (see ../README.md): CaoAPBLKD22 (IPDPS'22) and CaoPABLKD21
(IPDPS'21).
Artifact: https://github.com/ecrc/hicma-x, commit
366f7e07ca9afc4149a8980a3ae9b5e13181ab00 (2026-03-18), cloned with
`git clone --depth 1` (submodules dplasma/hcore/stars-h -- and dplasma's own
nested parsec submodule -- fetched shallow too, see build.sh; all four keep
their own .git for provenance, see STATUS.md for each commit).

What is wrapped
----------------
`tests/testing_potrf.c` -> CMake target `testing_potrf_tlr`: an end-to-end
CLI driver that (1) generates its OWN SPD operand internally -- a STARS-H
tile-structured covariance matrix, kind_of_problem selects which generator
-- (2) runs one of nine PaRSEC/DPLASMA tile-Cholesky variants
(--kind_of_cholesky; this adapter uses 2 = DENSE_TLR_DP, "full double
precision", the paper's own dense/Exact baseline path -- see
kind_of_cholesky's docstring below for why not the TLR/mixed-precision
variants), (3) with --check, independently computes and PRINTS (not just a
pass/fail flag) ||A||_F, ||L'L-A||_F and their ratio, then (4) exits. There
is no separate library entry point (hicma_parsec.h's own API is a PaRSEC
task-graph interface, not a callable "factor this matrix" function), so per
ARTIFACT_GUIDE.md rule 1 ("if the artifact only ships an end-to-end binary
that loads matrices itself, wrap at the finest boundary available and
document the contamination") this adapter wraps the COMPILED BINARY as a
subprocess, matching cg-krylov/spcg's precedent.

Why the operand can't be OUR operand (and the resulting gate hook)
--------------------------------------------------------------------
Unlike spcg (which reads a --matrix-file), testing_potrf_tlr has no
"load a dense/tile matrix from a file" mode for any --kind_of_problem: it
always builds its own SPD operand (STARS-H covariance / random-TLR
generator) in-process, sized only by --N/--NB/--kind_of_problem and a
handful of generator-specific knobs (radius, kernel type, ...). There is
therefore no way to feed dense.py's own seeded `_spd_matrix` into this
binary -- the "prepare() converts OUR operand into the artifact's format"
discipline (rule 2) does not apply; the operand itself is the artifact's.

The spec's own normalized-residual FORMULA is still exactly reproducible
without dense.py's specific A: ||L L^T - A||_F / (||A||_F * n * eps_fp64).
--check's stdout gives us the artifact's own ||A||_F and ||L'L-A||_F
DIRECTLY (see `_RESIDUAL_RE` below) -- never a trusted pass/fail string, the
two Frobenius norms themselves, from which this adapter recomputes the exact
spec formula independently in Python. dense.py's `reference_cholesky` was
given one small, optional, documented hook for exactly this situation
(params["external_A_fro_norm"], see its docstring in
kernelbench/domains/dense.py) so the SAME formula/threshold this track uses
for every other cholesky impl (scipy-cholesky, torch-cholesky) also governs
this one -- only the operand differs, and that substitution is disclosed
here and in every result record (`params["hicma_x_operand_note"]`), never
hidden. This is the "minimal, optional, documented params hook" the task
brief anticipated; ScipyCholesky/TorchCholesky never set this key, so their
gate path is completely unchanged (verified: `--smoke` on scipy-cholesky
still passes after this change, see STATUS.md).

kind_of_cholesky=2 (DENSE_TLR_DP) and kind_of_problem=2 (st-2d-sqexp)
----------------------------------------------------------------------
`--kind_of_cholesky` selects among 9 PaRSEC potrf variants (1,3,4,5,8,9 are
genuinely low-rank/mixed-precision; 6,7 are sparse-distribution variants).
2 (DENSE_TLR_DP, "full double precision") is this repo's own dense-fp64
baseline entry point -- the closest fit to this track's
cholesky-dense-single-node-fp64-kernel variant available from this binary
(there is no separate "plain LAPACK-style dense potrf, no TLR machinery at
all" mode; TESTS.md's own examples only exercise TLR/mixed-precision
variants). `--kind_of_problem 2` (st-2d-sqexp, TESTS.md's own documented
"Statistics-2d-sqexp" example) was chosen over 0 (randtlr) after the latter
produced a numerically suspicious factorization at small N (LAPACK info=51
warning, NaN in the artifact's own OTHER internal check "HiCMA L vs dense
L" -- see STATUS.md for the exact transcript); st-2d-sqexp's STARS-H
covariance generator gave a clean, well-conditioned SPD operand and a
||L'L-A||_F/||A||_F ratio at machine precision (1.07e-16) in every size
tested. This choice is disclosed via `params["kind_of_problem"]`, not
silently baked in.

Patches (rule 3: none touch hicma-x's own source; see build.sh/STATUS.md)
---------------------------------------------------------------------------
No file inside `source/` (or its submodules) is modified. build.sh's own
compiler-compat flags (GCC-14 diagnostic downgrades, a force-included
3-function LAPACKE compatibility header living OUTSIDE every submodule) are
build-system fixes per rule 3; full rationale in build.sh's comments and
STATUS.md.
"""

from __future__ import annotations

import os
import re
import subprocess

import numpy as np

KERNEL = "cholesky"
IMPL_NAME = "hicma-x-potrf-tlr"
PAPER_KEY = "conf/sc/LtaiefACRSKDBAK24"
PRECISIONS = ["fp64"]

_HERE = os.path.dirname(os.path.abspath(__file__))
_SOURCE = os.path.join(_HERE, "source")
_BUILD = os.path.join(_SOURCE, "build")
_BINARY = os.path.join(_BUILD, "tests", "testing_potrf_tlr")

_CUDA_HOME = os.environ.get("KB_CUDA_HOME", os.environ.get("CUDA_HOME", "/opt/nvidia/hpc_sdk/Linux_x86_64/25.5/cuda/12.9"))
_MATHLIBS = os.environ.get("KB_CUDA_MATHLIBS", "/opt/nvidia/hpc_sdk/Linux_x86_64/25.5/math_libs/12.9")
_LIBSCI = os.environ.get("KB_BLAS_LIBDIR", "/opt/cray/pe/libsci/26.03.0/GNU/12/x86_64/lib")

# Parses the SECOND "Check Cholesky" block's residual line, e.g.:
#   -- ||A||_F = 1.513152e+02, ||L'L-A||_F = 1.624336e-14
# (--verbose 2 is required for this line to print; see run()). This is the
# check comparing HiCMA's computed L against the ORIGINAL A via L L^T -- the
# same relationship this track's own residual formula tests -- as opposed to
# the artifact's OTHER check ("HiCMA L vs dense L", an internal
# cross-validation against a second, uncompressed factorization) which this
# adapter does not use for the gate.
_RESIDUAL_RE = re.compile(
    r"\|\|A\|\|_F\s*=\s*([0-9.eE+-]+),\s*\|\|L'L-A\|\|_F\s*=\s*([0-9.eE+-]+)")


def available() -> tuple[bool, str]:
    if not os.path.exists(_BINARY):
        return False, f"binary not built: {_BINARY} missing (run build.sh)"
    if not os.access(_BINARY, os.X_OK):
        return False, f"binary present but not executable: {_BINARY}"
    return True, ""


def create(precision: str):
    return HicmaXPotrfTlr(precision)


class HicmaXPotrfTlr:
    """
    One `run()` call = one fresh `testing_potrf_tlr` subprocess = one
    complete STARS-H matrix generation + PaRSEC tile-Cholesky factorization
    + --check residual computation. `prepare()` resolves N/NB from the
    workload (NB must divide N; testing_potrf_tlr requires N % NB == 0) and
    records every resolved knob into `params` (never left implicit, matching
    ScipyCholesky.prepare's own discipline). `run()` launches the
    subprocess and parses its own printed Frobenius norms (never a
    pass/fail string). `to_host()` returns the ONE-ELEMENT
    ||L'L-A||_F array `reference_cholesky`'s external_A_fro_norm hook (set
    here in `prepare()`) expects to be compared against.
    """

    name = IMPL_NAME
    platform = "cuda"

    def __init__(self, precision: str = "fp64"):
        self.requested_precision = precision
        self.precision = "fp64"  # PaRSEC's dense/TLR-DP path is fp64-only
        self._params = None
        self._env = None

    def prepare(self, workload, params: dict):
        if self.requested_precision != "fp64":
            params["hicma_x_precision_note"] = (
                f"requested precision {self.requested_precision!r} != fp64; "
                "testing_potrf_tlr's DENSE_TLR_DP path is fp64-only, clamped")
        params["precision"] = "fp64"

        n = workload.n
        # NB must divide N exactly (CMake/PaRSEC assertion, not this
        # adapter's choice); pick the largest divisor of n that is
        # reasonably tile-sized (<=256), falling back to n itself (NB=N,
        # i.e. one tile -- still a valid, if degenerate, configuration) if
        # no better divisor exists. Recorded explicitly, never silent.
        nb = 1
        for cand in range(min(256, n), 0, -1):
            if n % cand == 0:
                nb = cand
                break
        if nb == 1:
            nb = n

        # Disclosed operand substitution -- see module docstring. Chosen
        # after the default kind_of_problem=0 (randtlr) produced a
        # numerically suspicious factorization at small N (see STATUS.md).
        kind_of_problem = int(params.get("kind_of_problem", 2))  # st-2d-sqexp
        kind_of_cholesky = int(params.get("kind_of_cholesky", 2))  # DENSE_TLR_DP
        fixedacc = params.get("fixedacc", "1e-8")
        maxrank = int(params.get("maxrank", max(1, nb // 2)))

        params["hicma_x_operand_note"] = (
            "operand is testing_potrf_tlr's OWN STARS-H-generated SPD "
            "matrix (kind_of_problem, see params), NOT dense.py's seeded "
            "_spd_matrix -- reference_cholesky's external_A_fro_norm hook "
            "is used so the gate formula/threshold is unchanged, only the "
            "operand differs (see adapter.py module docstring).")
        params["N"] = n
        params["NB"] = nb
        params["kind_of_problem"] = kind_of_problem
        params["kind_of_cholesky"] = kind_of_cholesky
        params["fixedacc"] = fixedacc
        params["maxrank"] = maxrank

        env = dict(os.environ)
        env["MPICH_GPU_SUPPORT_ENABLED"] = "0"
        # testing_potrf_tlr does MPI_Init as a singleton. On a Slurm+PMIx+
        # OpenMPI site (e.g. zaratan) inheriting the surrounding srun job
        # step's SLURM_*/PMIX_*/OMPI_*/PMI_* env makes OpenMPI's singleton
        # bootstrap attach to that namespace and DEADLOCK; strip them from the
        # child env (no-op on Cray). LD_LIBRARY_PATH (set below) is untouched.
        # Same fix as ../../cg-krylov/bootcmatchgx/adapter.py.
        env = {k: v for k, v in env.items()
               if not (k.startswith("SLURM_") or k.startswith("PMIX_")
                       or k.startswith("OMPI_") or k.startswith("PMI_"))}
        build_src = os.path.join(_BUILD, "src")
        build_parsec = os.path.join(_SOURCE, "dplasma", "parsec", "parsec")
        build_starsh = os.path.join(_BUILD, "stars-h", "src")
        build_hcore = os.path.join(_BUILD, "hcore", "src")
        ld_path = os.pathsep.join([
            build_src, build_parsec, build_starsh, build_hcore,
            f"{_CUDA_HOME}/lib64", f"{_CUDA_HOME}/extras/CUPTI/lib64",
            f"{_MATHLIBS}/lib64", _LIBSCI,
            env.get("LD_LIBRARY_PATH", ""),
        ])
        env["LD_LIBRARY_PATH"] = ld_path
        self._env = env
        self._params = params
        return {"n": n, "nb": nb}

    def run(self, h):
        p = self._params
        cmd = [
            _BINARY,
            "--N", str(h["n"]), "--NB", str(h["nb"]),
            "--kind_of_problem", str(p["kind_of_problem"]),
            "--kind_of_cholesky", str(p["kind_of_cholesky"]),
            "--fixedacc", str(p["fixedacc"]),
            "--maxrank", str(p["maxrank"]),
            "--check", "--verbose", "2",
            "--cores", "4", "--gpus", "1",
        ]
        proc = subprocess.run(
            cmd, env=self._env, capture_output=True, text=True, timeout=600)
        p["hicma_x_subprocess_returncode"] = proc.returncode
        p["hicma_x_cmd"] = " ".join(cmd)

        matches = _RESIDUAL_RE.findall(proc.stdout)
        if not matches:
            raise RuntimeError(
                f"testing_potrf_tlr produced no parseable residual line "
                f"(returncode={proc.returncode}); stdout tail:\n"
                f"{proc.stdout[-3000:]}\nstderr tail:\n{proc.stderr[-2000:]}")
        # The binary prints TWO such blocks per --check run (a "Check
        # compress and run dense dpotrf" pre-check with all-zero norms, and
        # the real "Check dpotrf" post-factorization block) -- the LAST
        # match is the real one (matches spcg's own "last row wins" CSV
        # discipline for a similarly repeated-per-call artifact output).
        a_fro, resid_fro = (float(x) for x in matches[-1])
        p["hicma_x_A_fro_norm"] = a_fro
        p["hicma_x_LLT_minus_A_fro_norm"] = resid_fro
        return resid_fro

    def to_host(self, out):
        p = self._params
        p["external_A_fro_norm"] = p["hicma_x_A_fro_norm"]
        return np.array([float(out)])

    def timer(self):
        from kernelbench.harness import Timer
        return Timer()

    def free(self, h):
        self._params = None
