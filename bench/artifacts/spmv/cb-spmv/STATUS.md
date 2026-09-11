# CB-SpMV — STATUS

**Outcome: BUILT+GATED — gate PASSES**

Paper: "CB-SpMV: A Data Aggregating and Balance Algorithm for Cache-Friendly
Block-Based SpMV on GPUs", ICS 2025. PAPER_KEY = `conf/ics/CongSC00Q25`.
Repo: `https://github.com/xing-cong/CB-Sparse`
(commit `ef06b656a84e1f189eb459ea795e447e811c12f9`, 2026-07-04;
`git clone --depth 50` into `./source/`; the relevant subtree is
`source/cb-spmv/src/`, this repo also ships `cb-spgemm/` and `cb-spmm/`
siblings for other kernels, untouched).

## Why a bridge file was needed (no kernel code touched)

The artifact ships no library entry point, only an end-to-end binary
(`main.cu`, built via `src/Makefile`) whose driver function, `cb_spmv()`
(`src/cb-spmv.cuh`):

1. builds every device buffer from a `GLOBAL_BLOCK` — the cache-block
   format `coo2block_gather` (`src/coo2block.h`) constructs on the CPU —
   via per-block `cudaMalloc`/`cudaMemcpy` helpers
   (`cb_spmv_detail::get_addr_with_{coo_comp,csr,dense}`);
2. launches the kernel once, untimed, to get a value for its own
   correctness check;
3. then wraps **`ITER=10`** (`macros.h`) further launches in a **single**
   `cudaEvent` pair and reports the mean — exactly the "one timestamp pair
   around a batched loop" pattern flagged in advance for this artifact
   ("its own code uses ITER=10") and exactly what
   `kernelbench/impls/gpu_cuda.py`'s `CudaEventTimer` (one event pair per
   iteration) exists to avoid.

Per the brief ("its cache-block format construction belongs in
prepare()"), `bridge.cu` (this directory, NOT part of the artifact) splits
(1) into `cbspmv_prepare()` (called once by `adapter.py::prepare()`) and
re-exposes the kernels (`cb_spmv_detail::spmv_cuda_kernel_comp` /
`_gather`, defined in `cb-spmv.cuh` and included **verbatim, unmodified**)
through `cbspmv_run()`, one launch per harness iteration. Every CUDA call
in `bridge.cu` is lifted directly from `cb_spmv()` — nothing about the
device-side kernel logic was changed, only how the host orchestrates it.

**One necessary addition**: the kernels accumulate into `y` via
`atomicAdd`, and the artifact's own driver zeros `d_y` only **once**
(before its whole 10-call timing loop) — after the first call, output is
never re-validated, so the artifact never notices that calls 2-10 keep
accumulating onto the previous result. Our harness calls `run()`
independently many times (an isolated correctness check, then `warmup`,
then `reps` more) and each call must be independently correct, so
`cbspmv_run()` re-zeros `d_y` (`cudaMemsetAsync`, O(rows) — cheap relative
to the kernel) immediately before every launch. Documented in `bridge.cu`'s
docstring.

## Build-system fixes

1. **Arch flag**: the artifact's `src/Makefile` targets `-arch=sm_89` (its
   own dev machine, an RTX 4090); overridden to `-arch=sm_80` for this
   machine's A100, per the integration brief.
2. **`-ltbb`**: `coo2block.h` uses `std::sort(std::execution::par, ...)`
   (C++17 parallel algorithms) twice; GCC/libstdc++'s implementation of the
   `par` execution policy dispatches to Intel TBB at *link* time (observed
   directly: `undefined symbol: _ZTIN3tbb4taskE` when `ctypes.CDLL` loaded
   the first build). `-ltbb` added in `build.sh`; TBB is available
   system-wide at `/usr/lib64/libtbb.so` on this machine, no install
   needed.

No patches to `source/` itself — `bridge.cu` and `build.sh` live in this
directory and only `#include` the artifact's headers.

## Build

```
./build.sh
```
`nvcc -O3 -arch=sm_80 -Xcompiler -fopenmp -Xcompiler -fPIC -shared
bridge.cu -ltbb -o bridge.so`. nvcc 12.9
(`/opt/nvidia/hpc_sdk/.../cuda/12.9`). Idempotent (single `nvcc` invocation,
no intermediate build dir).

## Adapter

`IMPL_NAME = "cb-spmv"`, `PRECISIONS = ["fp64"]` (artifact's `ValType`,
`macros.h`, is `double`). `prepare()` converts the harness's CSR matrix to
COO (`matrix.csr.tocoo()`) — CB-SpMV's own `coo2block_gather` takes COO
input, matching `main.cu`'s own usage — then calls `cbspmv_prepare()`
(format construction + all device allocation/H2D, timed once as
preprocessing). `run()` is exactly one `cbspmv_run()` call (stream=0, the
default stream — same stream `torch.cuda.Event()` records on by default,
so `CudaEventTimer` correctly brackets this launch even though the kernel
itself is launched via raw ctypes, not torch — the same cross-runtime
pattern already used by `kernelbench/impls/gpu_cuda.py`'s own
`CustomSpMV`). `to_host()` D2H-copies `y` (a plain blocking `cudaMemcpy` on
the default stream, which also serves as the sync point after the kernel).

## Gate verification (mandated command, run as specified)

```
$PY -m kernelbench.runner --kernel spmv --variant spmv-csr-kernel \
    --impl cb-spmv --matrices cant --warmup 1 --reps 3
```

```
  loading cant ...
  running cb-spmv cant  ... 0.073 ms  110.34 GFLOP/s  (err 5.82e-16 <= 1e-09)  [118.4s]
1/1 runs valid
```

`max_scaled_err = 5.82e-16`, tolerance `1e-9` — **PASS**. Also verified
independently on a small synthetic banded matrix (500x500) via a direct
`harness.run_variant` call: `max_scaled_err = 3.15e-16` — PASS, consistent.

Note on the 118.4s wall-clock for this one gate call (not a timing claim —
the runner's own `conforming: False` already flags login-node/protocol-
override non-conformance): `nvidia-smi` during this run showed the GPU at
**100% utilization with ~30GB already in use by another user's process**
(PID visible, `python3`). CB-SpMV's own `prepare()` path does one
`cudaMalloc`+`cudaMemcpy` **per non-empty 16x16 block** (thousands of tiny
device allocations for `cant`'s 4M nnz over a 62451x62451 matrix) — this is
the artifact's own strategy (`cb_spmv_detail::get_addr_with_*`, called
unmodified), not something this adapter introduced, and every small device
op queues behind a saturated GPU's driver-level scheduling. The reported
kernel time itself (0.073ms/call, from `cudaEvent`s bracketing only the
kernel launch) is unaffected by this queuing.

## Verdict

`cb-spmv: BUILT+GATED err=5.82e-16 (tol 1e-9)`

## Reproduction on zaratan (2026-09-09)

- Machine: UMD zaratan, A100 MIG 1g.5gb slice (sm_80), login-node build.
- Toolchain: nvcc 12.8.93 (conda-forge CUDA 12.8), g++ 13.4.0, torch 2.8.0+cu128,
  Python 3.12.14; `-arch=sm_80` (unchanged).
- Build: OK (after one build-system fix, below). Build-system changes:
  `build.sh` -- the login-node link against the system `/usr/lib64/libtbb.so.2`
  (`-l:libtbb.so.2 -L/usr/lib64`) succeeds at link time but the GPU compute
  node used for the gate has NO TBB installed at all (different node image:
  `ls /usr/lib64/libtbb*` -> "No such file or directory" on the GPU node,
  confirmed directly), so the built `bridge.so` failed to `dlopen` at gate
  time (`OSError: libtbb.so.2: cannot open shared object file`) even though
  the earlier login-node build/link had reported success. Fixed by vendoring
  `libtbb.so.2` (located via `${KB_TBB_LIB:-$(ldconfig -p | ...)}`, falling
  back to `/usr/lib64/libtbb.so.2`) into this directory's own
  (gitignored, `bench/artifacts/*/*/vendor/` already in `.gitignore`)
  `vendor/libtbb.so.2` at build time and linking with
  `-Xlinker -rpath -Xlinker "$HERE/vendor"`, so `bridge.so` carries its own
  copy and resolves it on any node regardless of what that node has
  installed system-wide. No `source/` file touched; `source.provenance`'s
  `patch=none` remains accurate.
- Gate: spmv-csr-kernel --matrices cant --warmup 1 --reps 3: PASS, err
  5.82e-16 <= 1e-9 (matches the recorded value exactly).
- Deviation from the recorded ruling: none in outcome (same clean PASS,
  identical error). The TBB runtime-linkage gap above is a machine
  difference between this cluster's login and GPU-compute node images, not
  present on the reference machine (Perlmutter, where build and gate both
  ran on the same node type) -- worth flagging for anyone reproducing this
  artifact on a split login/compute-node cluster.
- Verdict here: BUILT+GATED -- same as the recorded ruling.
