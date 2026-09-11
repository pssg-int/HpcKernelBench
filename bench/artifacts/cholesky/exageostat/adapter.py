"""
exageostat adapter for the cholesky track.

Paper: "Accelerating Geostatistical Modeling and Prediction With
Mixed-Precision Computations: A High-Productivity Approach With PaRSEC"
(TPDS'22). `PAPER_KEY = journals/tpds/AbdulahCPBDGKLS22` (title-matched in
../../../output/included.json). Also the current mapping target for 3
CPU/distributed-only papers on the same repo (see ../README.md).
Artifact: https://github.com/ecrc/exageostat, commit
ee4a82bd5b9bf99089b76dc11d288c24c6da58e0, `git clone --depth 1`. Chameleon
submodule (release-1.1.0, gitlab.inria.fr/solverstack/chameleon) at
4db899ca30d29927018d83964b9b6d517269abe1.

Provenance mismatch, disclosed (see ../README.md): AbdulahCPBDGKLS22's own
title/abstract describe a PaRSEC-based three-precision Cholesky. The
ecrc/exageostat repository this project's paper->artifact mapping points to
is StarPU+Chameleon-based (confirmed from its own CMakeLists.txt/INSTALL --
no PaRSEC dependency anywhere), not the code that TPDS'22 paper actually
describes; that PaRSEC-based lineage appears to have been folded into
hicma-x's current "HiCMA-PaRSEC" incarnation instead (hicma-x's own README
literature list cites this exact paper). This adapter wraps what the
mapped repository actually contains -- a StarPU/Chameleon dense Cholesky --
which is a legitimate, load-bearing part of the ExaGeoStat geostatistics
stack even if not literally the PaRSEC code the paper's own headline number
was measured on.

What is wrapped, and why NOT ExaGeoStat's own MLE driver
------------------------------------------------------------
ExaGeoStat's own paper-level entry point is `synthetic_dmle_test`
(examples/synthetic_dmle_test.c): generate a synthetic geospatial dataset,
Cholesky-factor its covariance matrix once (data-generation phase), then run
an MLE optimization loop that repeatedly re-generates the covariance matrix
at trial theta values and re-factors it. Extensive testing (see STATUS.md
for the full transcript) found this driver's SECOND (and every subsequent)
Cholesky factorization deterministically fails ("the matrix is not positive
definite", a hard `exit(EXIT_FAILURE)` in ExaGeoStat's own SUCCESS() macro)
-- REGARDLESS of theta, kernel type, single- vs multi-threaded, or even
forcing the optimizer's search box to a single point identical to the
FIRST (successful) factorization's theta via degenerate --olb/--oub bounds.
This rules out parameter conditioning as the cause; it is either a genuine
bug in this exact repository state or an interaction this integration could
not isolate further within the login-node time budget. Per ARTIFACT_GUIDE.md
rule 3, this is squarely in "the kernel itself must change to run" territory
-- not something to patch -- so `synthetic_dmle_test` was NOT wrapped for
the correctness gate.

Deeper investigation (chameleon_dtesting, Chameleon's OWN unit-test binary,
also built by build.sh) isolated the failure precisely: `-o dpotrf --check`
with `--gpus 0` (CPU-only) reliably reports `SUCCESS`; the IDENTICAL
invocation with `--gpus 1` reliably reports `FAILED`, at every N tried
(512, 1000, 2048) and even with a large diagonal bump forcing extreme
diagonal dominance. `-o dgemm --check --gpus 1` ALSO fails, with a printed
residual (`||R||=4.5`, vs `||R||=4e-15` on `--gpus 0`) proving this is a
genuine GPU-path NUMERICAL CORRECTNESS bug in this Chameleon build, not
specific to POTRF and not a false negative from a broken check (a broken
check would also fail on CPU). This is very likely downstream of the
CHAMELEON_USE_CUBLAS_V2=ON cuBLAS-v2 migration this integration's own
build.sh required (see its own comments) interacting with cuBLAS v2's
default HOST pointer mode for scalar (alpha/beta) arguments -- plausible
but NOT confirmed within budget; recorded as an open finding in STATUS.md,
not patched (would require touching StarPU/Chameleon's own CUDA codelets,
squarely "kernel code").

This adapter therefore wraps `chameleon_dtesting -o dpotrf --check`
(finest available boundary: no separate library entry point is exposed for
"factor exactly this matrix" either -- both drivers build the operand
in-process) as a subprocess, exactly reproducing the differential result
above: **the GPU-path gate genuinely FAILS**, which is recorded here as-is
per ARTIFACT_GUIDE.md rule 4 ("if it fails the gate, that IS a result --
record it, do not loosen the gate"). The CPU-path result (SUCCESS) is
recorded in every run's params for comparison, never substituted for the
GPU result.

Gate mechanism
---------------
`chameleon_dtesting`'s own `--check` for `dpotrf` prints only a structured
`RETURN` field (`SUCCESS`/`FAILED`), not a numeric residual (unlike its own
`dgemm` op, which does print `||R||` -- see module docstring above for how
that was used to independently confirm the GPU bug is real and not an
artifact of a broken check). Since dense.py's `reference_cholesky` has no
way to construct a reference for an operand THIS binary generates
internally (same operand-substitution situation as hicma-x, see
kernelbench/domains/dense.py's `external_A_fro_norm` hook docstring), this
adapter reuses that same hook: `external_A_fro_norm=1.0` (an arbitrary
placeholder scale, since there is no printed norm to use) paired with
`to_host()` returning `0.0` on a parsed `SUCCESS` or a large sentinel on
`FAILED` -- so the harness's existing max_scaled_err gate machinery
faithfully passes through Chameleon's OWN pass/fail verdict without this
adapter inventing any numeric precision that was never measured.
"""

from __future__ import annotations

import os
import subprocess

import numpy as np

KERNEL = "cholesky"
IMPL_NAME = "exageostat-chameleon-dpotrf"
PAPER_KEY = "journals/tpds/AbdulahCPBDGKLS22"
PRECISIONS = ["fp64"]

_HERE = os.path.dirname(os.path.abspath(__file__))
_PREFIX = os.path.join(_HERE, "prefix")
_BINARY = os.path.join(_PREFIX, "bin", "chameleon_dtesting")

_CUDA_HOME = os.environ.get("KB_CUDA_HOME", os.environ.get("CUDA_HOME", "/opt/nvidia/hpc_sdk/Linux_x86_64/25.5/cuda/12.9"))
_MATHLIBS = os.environ.get("KB_CUDA_MATHLIBS", "/opt/nvidia/hpc_sdk/Linux_x86_64/25.5/math_libs/12.9")
_LIBSCI = os.environ.get("KB_BLAS_LIBDIR", "/opt/cray/pe/libsci/26.03.0/GNU/12/x86_64/lib")

_FAIL_SENTINEL = 1.0e30


def available() -> tuple[bool, str]:
    if not os.path.exists(_BINARY):
        return False, f"binary not built: {_BINARY} missing (run build.sh)"
    if not os.access(_BINARY, os.X_OK):
        return False, f"binary present but not executable: {_BINARY}"
    return True, ""


def create(precision: str):
    return ExageostatChameleonPotrf(precision)


def _run_dpotrf(binary, n, nb, threads, gpus, env):
    """One chameleon_dtesting -o dpotrf --check invocation; returns the
    parsed CSV dict for the single result row (see the module's
    'Id;Function;...;RETURN' header)."""
    cmd = [binary, "-o", "dpotrf", "-n", str(n), "-b", str(nb),
           "-t", str(threads), "-g", str(gpus), "--check", "--uplo", "Lower"]
    proc = subprocess.run(cmd, env=env, capture_output=True, text=True, timeout=300)
    lines = [ln for ln in proc.stdout.splitlines() if ln.strip()]
    if len(lines) < 2:
        raise RuntimeError(
            f"chameleon_dtesting produced no CSV result row "
            f"(returncode={proc.returncode}); cmd={' '.join(cmd)}\n"
            f"stdout tail:\n{proc.stdout[-3000:]}\nstderr tail:\n{proc.stderr[-2000:]}")
    header = lines[0].split(";")
    row = lines[-1].split(";")
    return dict(zip(header, row)), proc.returncode


class ExageostatChameleonPotrf:
    """
    One `run()` call = one fresh `chameleon_dtesting -o dpotrf --check`
    subprocess = one complete Chameleon dense tile-Cholesky factorization
    (StarPU-scheduled, CUDA-capable) with Chameleon's own internal
    backward-error check. See module docstring for exactly why this,
    rather than ExaGeoStat's own MLE driver, is wrapped, and for the
    GPU-path correctness bug this adapter's gate faithfully surfaces.
    """

    name = IMPL_NAME
    platform = "cuda"

    def __init__(self, precision: str = "fp64"):
        self.requested_precision = precision
        self.precision = "fp64"
        self._params = None
        self._env = None
        self._n = None
        self._nb = None

    def prepare(self, workload, params: dict):
        if self.requested_precision != "fp64":
            params["exageostat_precision_note"] = (
                f"requested precision {self.requested_precision!r} != fp64; "
                "chameleon_dtesting's dpotrf op here is fp64-only, clamped")
        params["precision"] = "fp64"

        n = workload.n
        nb = 1
        for cand in range(min(256, n), 0, -1):
            if n % cand == 0:
                nb = cand
                break
        if nb == 1:
            nb = n

        params["exageostat_operand_note"] = (
            "chameleon_dtesting generates its OWN random SPD operand "
            "in-process (LAPACK dlatms/dplgsy-style generator, seeded), "
            "NOT dense.py's seeded _spd_matrix -- reference_cholesky's "
            "external_A_fro_norm hook is used with a placeholder scale "
            "(1.0) since dpotrf's own --check prints no residual number, "
            "only a SUCCESS/FAILED verdict (see adapter.py docstring for "
            "why this is nonetheless a faithful, non-loosened pass-through "
            "of Chameleon's own gate, cross-validated against its dgemm "
            "op's numeric ||R|| on the same GPU path).")
        params["N"] = n
        params["NB"] = nb
        params["gpus"] = int(params.get("gpus", 1))
        params["threads"] = int(params.get("threads", 4))

        env = dict(os.environ)
        env["MPICH_GPU_SUPPORT_ENABLED"] = "0"
        ld_path = os.pathsep.join([
            os.path.join(_PREFIX, "lib"), os.path.join(_PREFIX, "lib64"),
            f"{_MATHLIBS}/lib64", f"{_CUDA_HOME}/lib64",
            f"{_CUDA_HOME}/extras/CUPTI/lib64", _LIBSCI,
            env.get("LD_LIBRARY_PATH", ""),
        ])
        env["LD_LIBRARY_PATH"] = ld_path
        self._env = env
        self._n, self._nb = n, nb
        self._params = params
        return {"n": n, "nb": nb}

    def run(self, h):
        p = self._params
        row, rc = _run_dpotrf(_BINARY, h["n"], h["nb"], p["threads"], p["gpus"], self._env)
        p["exageostat_subprocess_returncode"] = rc
        p["exageostat_dpotrf_return"] = row.get("RETURN", "")
        p["exageostat_dpotrf_gflops"] = row.get("gflops", "")

        # Cross-check on the SAME GPU path, CPU-only (--gpus 0), recorded
        # for audit only -- never substituted into the actual gate. This is
        # exactly the differential comparison that isolated the GPU-path
        # bug in the first place (see module docstring).
        cpu_row, _ = _run_dpotrf(_BINARY, h["n"], h["nb"], p["threads"], 0, self._env)
        p["exageostat_dpotrf_return_cpu_control"] = cpu_row.get("RETURN", "")

        return row.get("RETURN", "")

    def to_host(self, out):
        p = self._params
        p["external_A_fro_norm"] = 1.0  # placeholder scale, see docstring
        passed = (out == "SUCCESS")
        p["gate_passed_chameleon_own_check"] = bool(passed)
        return np.array([0.0 if passed else _FAIL_SENTINEL])

    def timer(self):
        from kernelbench.harness import Timer
        return Timer()

    def free(self, h):
        self._params = None
