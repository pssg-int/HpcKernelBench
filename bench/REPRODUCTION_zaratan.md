# Reproduction on zaratan (UMD), 2026-09-08 to 2026-09-09

Reproduction of the functional state recorded in `bench/artifacts/*/*/STATUS.md`
(reference machine: NERSC Perlmutter, A100 80GB PCIe, nvcc 12.9, gcc 14.3 /
12.3, Python 3.11.7, torch 2.8.0+cu128) on the UMD zaratan cluster. Per
artifact, the `## Reproduction on zaratan (2026-09-08)` section appended to
its STATUS.md carries the full evidence; this file is the roll-up.

## 1. Machine and toolchain

| Item | zaratan | reference (Perlmutter) |
|---|---|---|
| OS | RHEL 8.10, kernel 4.18 | SLES 15 (Cray) |
| Login node | 128 cores, 500 GB, **no GPU** | shared A100 |
| GPU used for gates | **A100-SXM4-40GB MIG 1g.5gb slice** (sm_80, 14 SMs, ~5 GB), `--gres=gpu:a100_1g.5gb:1` | A100 80GB PCIe (sm_80, 108 SMs) |
| Full GPUs available | A100-SXM4-40GB x4/node (15 nodes), H100 x4/node (8 nodes) | A100 80GB PCIe |
| Driver | 595.71.05 (CUDA 13.2 capable) | 580.159 |
| CUDA toolkit | **conda-forge `cuda-toolkit` 12.8.2, nvcc 12.8.93** | NVIDIA HPC SDK 25.5 cuda/12.9 + math_libs/12.9 |
| Host compiler | **conda-forge gcc/g++ 13.4.0** (`KB_CC`/`KB_CXX`) | gcc-native 14.3.0 |
| gcc 12 pair | **conda-forge gcc/g++ 12.4.0** (`KB_HOST_COMPILER_BIN`, second env `kb-gcc12`) | gcc-native 12.3.0 |
| Python | 3.12.14 (conda env `kb-env`) | 3.11.7 |
| torch / numpy / scipy | 2.8.0+cu128 / 2.3.3 / 1.16.2 (pip, `requirements.txt`) | same |
| cmake / ninja / make | 4.2.3 / 1.13.2 / 4.4.1 (conda-forge) | 3.28.3 / 1.10 |
| MPI (link-only) | OpenMPI 5.0.10 (conda-forge), `libmpi.so` (`KB_MPI_LIBNAME=mpi`) | Cray MPICH 9.1.0, `libmpi_gnu_123.so` |
| BLAS/LAPACK | OpenBLAS 0.3.34 (conda-forge), `KB_BLAS_LIB=libopenblas.so` | cray-libsci 26.03 `libsci_gnu.so` |
| hipcc | none in the harness env (site has hip/5.6.1 vs CUDA 12.3 modules) | hip 5.5.1 (broken vs CUDA 12.9) |
| libstdc++ trap | `KB_LD_PRELOAD` = the gcc-13 `libstdc++.so.6` from `kb-env` (system libstdc++ is gcc 8.5) | `/usr/lib64/libstdc++.so.6` |

Why conda-forge: the site's module tree tops out at gcc 11.3.0 and CUDA
12.3.0 (plus nvhpc/23.11 = CUDA 12.3), i.e. no gcc 12–14 and no 12.8-class
toolkit to pair with torch cu128. Two conda-forge environments under the
user's Miniforge (`.../condaroot/Miniforge3-24.11.3-2/envs/{kb-env,kb-gcc12}`)
supply everything; `bench/env.sh` defaults every `KB_*` knob to them
(commit "Point env.sh at the zaratan toolchain"). Two one-off adjustments
were made inside the `kb-env` prefix so the standard-toolkit layout that
`toolchain.sh` and the build scripts assume holds: the CUDA headers, which
conda-forge places only under `targets/x86_64-linux/include/`, were
symlinked into `kb-env/include/`; and `kb-env/lib64 -> lib` was added.

Setup log (all on the login node):

```bash
conda create -n kb-env -c conda-forge --override-channels python=3.12 cuda-toolkit=12.8 \
    cuda-nvcc=12.8 cuda-driver-dev=12.8 gcc=13 gxx=13 gfortran=13 cmake ninja make openmpi openblas \
    "libstdcxx-ng>=13" pkg-config autoconf automake libtool
conda create -n kb-gcc12 -c conda-forge --override-channels gcc=12 gxx=12
$KB_PY -m pip install --index-url https://download.pytorch.org/whl/cu128 torch==2.8.0
$KB_PY -m pip install -r bench/requirements.txt
source bench/env.sh && $PY bench/check_env.py      # 0 MISSING, 2 WARN (no GPU on the login node; no hipcc)
cd bench && ./smoke_all.sh                         # kernels green: 35 ; failing: 0
bench/artifacts/fetch_sources.sh                   # 135 directories complete, 0 failed, 6 without provenance
```

`fetch_sources.sh` first aborted with `toolchain.sh: line 74: LD_LIBRARY_PATH:
unbound variable` (it runs under `set -u`, and nothing on this machine had
set `LD_LIBRARY_PATH`); `toolchain.sh` now reads the variable through a
defaulted temporary. That is the only harness-side change.

### GPU access

Every GPU execution goes through `srun` on the `gpu` partition; nothing runs
on a login node. Full A100s were backlogged (estimated start ~8 h away) when
the gates were due, so the correctness gates ran on **A100 MIG 1g.5gb
slices** (`--gres=gpu:a100_1g.5gb:1`, node gpu-b11-6): same sm_80 ISA as the
recorded A100, 14 SMs, ~5 GB. The build scripts' sm_80 pins therefore needed
no change and no PTX JIT is involved. A slice is enough for every `--smoke`
gate that does not allocate more than ~5 GB; the per-artifact sections say
where a full A100 (`--gres=gpu:a100:1`) was needed instead.

### Timing / conforming runs

`conforming: true` is unreachable on this machine: locking GPU clocks needs
root (`nvidia-smi -lgc`), and the runner never sets `clocks_locked`
(`kernelbench/env.py` initialises it to `None` and nothing updates it), so
every result document carries "GPU clocks not locked" among its
non-conformance reasons even inside an exclusive allocation. See §4.

## 2. Per-adapter outcome (recorded ruling vs observed here)

Every integrated adapter (the `bench/artifacts/*/*/` directories carrying an
`adapter.py`, plus the two DEFERRED-HARDWARE dirs) was taken through
build + correctness gate on zaratan. Each carries a `## Reproduction on
zaratan (2026-09-0x)` section with the exact toolchain, gate numbers, and any
build-system change. Roll-up (from `output/benchmark_status.html`, regenerated
from those sections):

| outcome (current first-bold ruling) | count |
|---|---|
| BUILT+GATED (gate passes) | 84 |
| BUILT, gate FAILS (reproduced documented artifact bug) | 11 |
| BUILT, gate blocked / not loadable | 3 |
| DEFERRED-HARDWARE (needs sm_90a) | 2 |
| BUILD-FAILED (no-adapter dirs, unchanged from reference) | 4 |
| SKIPPED (no-adapter, out of scope, unchanged) | 37 |

The **CPU smoke suite stayed 35/35 green** throughout, and every GPU gate ran
through `bench/gpu_run.sh` (A100 MIG 1g.5gb slice unless noted).

### Same as the recorded ruling

The large majority (~90 adapters) reproduced their recorded ruling exactly —
including every recorded *gate failure*, which is the stronger check: the
documented artifact bugs reproduce on a different A100 SKU, nvcc minor, gcc
major and Python minor, usually to the same printed error digits. Examples:
`attention-kernel/et` (h2exp2 softmax bug, err 4.89e-1/4.70e-1/1.73e0),
`spmm/smat` (B-operand indexing, err ~1.7e4 on cant), `spmm/inferfast`
(fp16-vs-fp32-tolerance, err 4.4e-4..7.2e-4), `sddmm/flashsparse` (residue-tile
overflow RuntimeError, same message), `spgemm/tilespgemm` (wrong values on
synthetic, err 2.9e4), `lossless-compression/{mans,zipserv}` (crash / sign-bit
corruption), `gnn-aggregation/{tc-gnn,tlpgnn}`, `graph-pattern-mining/stmatch`,
`sptrsv/split-sptrsv`, `sequence-alignment/logan`.

### Deviations from the recorded ruling

| artifact | recorded (Perlmutter) | observed (zaratan) | cause |
|---|---|---|---|
| `spmv/spmv-acc` | BUILD-FAILED (hip 5.5.1 vs its CUDA) | **BUILT+GATED**, fp64 err<=1e-9 | zaratan's site `hip/5.6.1` module (CUDA 12.3) compiles HIP-on-CUDA cleanly; new bridge.cpp+adapter wrap the artifact's own `sparse_csr_spmv` |
| `cholesky/exageostat` | BUILT, GATE-FAILED (GPU path) | **BUILT+GATED** | the Chameleon/StarPU GPU dpotrf path passes here; see its STATUS for the toolchain difference |
| `gemm/moonpoly` | BUILT+GATED fp32 only (fp16 blocked) | **BUILT+GATED both fp32 and fp16** | a stale fp64 tolerance had blocked fp16; the per-precision tolerance table now admits it (harness fix landed between passes) |
| `topk-selection/gpu-topk-study` | BUILT+GATED (k<=2048) | **BUILT, gate cannot load** | its only artifact is a prebuilt `libgridselect.so` linked against glibc>=2.34; zaratan (RHEL 8.10) has glibc 2.28, and there is no source to recompile |
| `sptrsv/yysptrsv` | BUILT+GATED (err ~1.9e-15) | **BUILT, gate HANGS** | the synchronization-free SpTRSV busy-wait does not make forward progress here (confirmed reproducible on the MIG slice; same hang signature as its sibling `split-sptrsv`, whose missing-memory-fence hazard is a confirmed finding). Not retested on a full A100 (queue). |
| `fft/turbofft` | BUILT+GATED | BUILT+GATED (unchanged), *plus* the recorded uncaught-`--smoke`-crash no longer reproduces | a harness fix (per-workload `NotImplementedError` handling) landed independently between passes |
| `spmm/voltrix` | DEFERRED-HARDWARE (no sm_90a anywhere) | **BUILT on H100 (sm_90), gate FAILS** | ran on a real H100 (2026-09-10): builds + launches its compute_90a kernel, but `spmm-binary-adjacency-kernel` is 0/12 at max_scaled_err 2.9e-4 vs the fp32 tol 1e-4 — TF32 tensor-core roundoff level (the kernel computes in TF32, the adapter declares fp32; the sibling tf32-gated dtcspmm/flashsparse show the same magnitude and pass). Hardware blocker gone; a genuine precision-declaration gate failure. Not adjusted (would be gate-gaming). |
| `quantized-gemm/mxblas` | DEFERRED-HARDWARE (no sm_90a anywhere) | **BUILT + own-test GATED on H100** | ran on a real H100 (2026-09-10): in a 3-hour job (its FP8 CUTLASS kernel compile alone is ~1.5 h) MXBLAS's own tests/test_jit.py and tests/test_mxgemm.py both PASS — FP8-GEMM difference rate 0.071%, 432.9 TFLOPS. No kernelbench adapter by design (FP8/MX not modelled by the domain); its gate is the artifact's own test suite. Hardware+compile blocker fully resolved. |

None of these deviations is a loosened gate: every gate ran at its spec
tolerance. The two "improvements" (spmv-acc, moonpoly fp16, exageostat) reflect
zaratan having a capability the reference machine lacked or a harness fix that
landed between passes; the two "regressions" (topk glibc, yysptrsv hang) are
genuine machine differences (older glibc; fewer SMs / forward-progress on the
MIG slice), recorded with the Perlmutter result kept as "Earlier outcome".

## 3. Build-system patches and arch-flag changes made here

**Arch flags: unchanged.** zaratan's gate GPUs are A100 (sm_80), identical to
the reference machine, so all 78 `sm_80`/`compute_80` pins were left as-is; no
PTX-JIT and no `-arch` edits were needed. (The two sm_90a artifacts build with
their own hardcoded `compute_90a` on H100.)

All other changes are build-system only (no kernel/`source/` numerics touched);
each is parameterised through a `KB_*` knob with the Perlmutter value kept as
the `${VAR:-default}` fallback, so every script still works on the reference
machine. By category:

- **env.sh / toolchain.sh (machine config):** point every `KB_*` at the
  conda-forge toolchain; `KB_LD_PRELOAD` = the gcc-13 libstdc++; cap
  OMP/OPENBLAS/MKL/NUMEXPR threads to 4 (shared login-node `RLIMIT_NPROC`);
  guard `LD_LIBRARY_PATH` under `set -u`. `bench/gpu_run.sh` added (srun wrapper;
  login nodes have no GPU).
- **MPI library name** (`-lmpi_gnu_123` -> `-l"${KB_MPI_LIBNAME:-...}"`, OpenMPI
  here): `multigrid/amgt`, `graph-pattern-mining/graphset` (plus `spgemm/amgt`,
  `cg-krylov/bootcmatchgx`, `multigrid/bootcmatchgx` from the batch pass).
- **PMIx/Slurm subprocess-env strip** (MPI-singleton deadlock on Slurm+PMIx+
  OpenMPI): `cg-krylov/bootcmatchgx`, `multigrid/bootcmatchgx`,
  `cholesky/hicma-x` adapters. Machine-neutral (no-op on Cray).
- **libstdc++ `LD_PRELOAD` read from env** instead of a hardcoded
  `/usr/lib64/...` (whose gcc-8 libstdc++ lacks the needed CXXABI/GLIBCXX):
  `spmm/{flashsparse,dtcspmm}`, `sddmm/{flashsparse,rode,sputnik,tc-gnn}`,
  `spmm/{rode,sputnik,ge-spmm,tc-gnn}`, `quantized-gemm/{fp6llm,qfactory,tilus}`,
  `spmm/generalsparse` (its `gs_emit`), etc.
- **`nm`/`ldd` verification captured to a variable** so a `grep -q` pipeline
  cannot spuriously fail under `RLIMIT_NPROC`/SIGPIPE: `spmm/{rode,sputnik}`,
  `sddmm/{rode,sputnik}`, `spgemm/ocean`.
- **TBB linking** (only `/usr/lib64/libtbb.so.2` exists, no dev symlink/CMake
  package): `spmv/cb-spmv` (`-l:libtbb.so.2 -L/usr/lib64`),
  `string-regex-matching/ngap` (vendored oneTBB + `-L$TBB_VENDOR/lib` on the
  link line, since ngAP links a plain `-ltbb`).
- **`-j` / thread caps** for the login node's process limit:
  `tensor-contraction/fastkron`, `spmm/rassm`, `spmv/sspmv`, `cg-krylov/perks`.
- **gcc-14-only warning flag probed** instead of assumed (`-Wno-error=
  declaration-missing-parameter-type`, gcc 13 rejects the spelling):
  `cholesky/{exageostat,hicma-x}`.
- **downloads run with the conda `LD_LIBRARY_PATH` cleared** (its shadowed
  libcurl.so.4 is ABI-mismatched): `cholesky/exageostat` (StarPU/NLopt/GSL);
  GSL also built from source where no system GSL exists.
- **CC/CXX/FC honoured** instead of Cray `cc/CC/ftn` wrappers: `cholesky/hicma-x`.
- **misc build fixes:** `gemm/moonpoly` CUTLASS-patch idempotency via
  `git apply --check` (old grep marker false-positived); `gemm/hexcute`
  pip-installs hidet's undeclared deps into `pylibs/`, generated `bin/nvcc`
  reads `KB_GXX12`; `sparse-attention-kernel/sparse-transformer` build-dir glob
  + rpath + vendored matplotlib; `tensor-contraction/fastkron` vendors the
  uninitialised pybind11 submodule; `ann-search/pathweaver` relaxes a hardcoded
  nvcc-12.9 version check to 12.x; `spmm/generalsparse` links only objects that
  have a source file (an orphaned authors' `.o` fails to link).
- **new integrations** (DEFERRED-HARDWARE / previously-uncompilable, all
  wrapping the artifact's OWN kernel, no numerics touched): `spmv/spmv-acc`
  (bridge.cpp + adapter.py + build.sh), `spmm/voltrix` (adapter.py + build.sh).

The full per-file list is the non-`STATUS.md` diff of the reproduction commits.

## 4. Timing runs

None. A conforming (publishable) measurement needs an exclusive GPU with
**locked clocks**, which requires root (`nvidia-smi -lgc`) that a user does not
have on zaratan; the runner leaves `clocks_locked` unset and therefore stamps
every result `conforming: false` with "GPU clocks not locked" even inside an
exclusive `srun` allocation. All gates in this pass were functional
(`--smoke`, reduced protocol, most on a shared MIG slice) and are marked
non-conforming by construction, exactly as on the reference machine. Wiring up
locked-clock timing runs is out of scope for a functional-state reproduction.
