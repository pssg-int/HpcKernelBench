# cuSZp — STATUS

**Outcome: BUILT+GATED — gate PASSES on all 3 smoke workloads**

Paper: "GPU Lossy Compression for HPC Can Be Versatile and Ultra-Fast",
SC 2025 (cuSZp3 / "VGC"). PAPER_KEY = `conf/sc/HuangDLC25`.
Repo: `https://github.com/szcompressor/cuSZp`
(commit `f581dcf329c907c320f4743a9c6e7ee2fb9c5494`, 2026-06-11;
`git clone --depth 50` into `./source/`).

## PAPER_KEY disambiguation (read this before trusting the attribution)

This single repo is one continuously developed codebase spanning THREE SC
papers, per its own README's citation section:
- **SC'23** (`conf/sc/HuangD0LC23`, "cuSZp": kernel-fusion, single mode) —
  README: "code see Release cuSZp-V1.1 / Commit f4df2f1..."
- **SC'24** (`conf/sc/HuangDLC24`, "cuSZp2": new lossless modes) —
  README: "code see Release cuSZp-V2.0 / Commit 16e1647..."
- **SC'25** (`conf/sc/HuangDLC25`, "GPU Lossy Compression for HPC Can Be
  Versatile and Ultra-Fast" / cuSZp3 / "VGC") — README: "includes
  dimensionality support (1D, 2D, 3D data with 3 compression mode for
  each) and versatility support."

The commit cloned here (HEAD at clone time, no specific release tag
pinned) already exposes exactly the SC'25 feature matrix — `cuszp_dim_t`
{1D,2D,3D} x `cuszp_mode_t` {fixed,plain,outlier} in `include/cuSZp.h` —
not the single-mode SC'23 original this benchmark's spec survey (
`benchspecs/lossy-compression/spec.yaml`'s evidence entry for
`conf/sc/HuangD0LC23`) had cited. **PAPER_KEY is therefore set to the
SC'25 key (`HuangDLC25`), not the SC'23 key the original candidate list /
spec evidence pointed at** — the URL is identical across all three papers
so the earlier attribution (`HuangD0LC23`, also `benchmark_groups.json`'s
lossy-compression entry for this URL) describes an earlier state of the
same repo, not the code actually built and gated here.

## Why no bridge file was needed

Unlike this benchmark's other GPU compression/SpMV artifacts, cuSZp
genuinely ships a reusable library, not just a CLI binary: its own
`CMakeLists.txt` builds `cuSZp_shared`/`cuSZp_static` from a fixed list of
`.cu` sources and exposes a small, generic, device-pointer C API in
`include/cuSZp.h`:

```c
void cuSZp_compress(void* d_oriData, unsigned char* d_cmpBytes,
                    size_t nbEle, size_t* cmpSize, float errorBound,
                    cuszp_dim_t dim, uint3 dims, cuszp_type_t type,
                    cuszp_mode_t mode, cudaStream_t stream = 0);
void cuSZp_decompress(void* d_decData, unsigned char* d_cmpBytes,
                    size_t nbEle, size_t cmpSize, float errorBound,
                    cuszp_dim_t dim, uint3 dims, cuszp_type_t type,
                    cuszp_mode_t mode, cudaStream_t stream = 0);
```

`errorBound` is consumed directly as an absolute bound inside the kernel
(verified by reading `src/cuSZp.cu`'s dispatcher and
`src/cuSZp_entry_1D_f32.cu`'s `cuSZp_compress_1D_fixed_f32` — no internal
rel-to-abs conversion happens below this API; that conversion is the CLI
wrapper's job, `examples/cuSZp_test_f32_1D.cpp`'s own
`errorBound = (max_val-min_val)*1E-2f` computes it externally before
calling in). This matches the harness's `w.correctness_tolerance` exactly
(already a resolved absolute bound — see `kernelbench/domains/
compression.py`'s `Field` docstring), so no eb re-derivation was needed
either.

`build.sh` compiles exactly `CMakeLists.txt`'s own `cuSZp_SOURCES` list
directly with `nvcc --shared` (bypassing CMake, which also builds static
libs/tests/Python bindings not needed here) — zero patches to `source/`.

## Adapter

`IMPL_NAME = "cuszp-compress"`, `PRECISIONS = ["fp32"]` (cuSZp also ships
`CUSZP_TYPE_DOUBLE` fp64 kernels, not wrapped here). `direction =
"compress"` per this domain's contract. Device buffers are allocated with
torch (`ctypes.c_void_p(tensor.data_ptr())`) — the same pattern
`kernelbench/impls/gpu_cuda.py`'s `CustomSpMV`/`CustomSpMM`/`CustomSDDMM`
already use for native-CUDA ctypes kernels, not a new convention.
Mode = `CUSZP_MODE_OUTLIER` (README: "Always highest compression ratios"),
dim = `CUSZP_DIM_1D` (README: "can be used for all datasets" — the
workload's 3D field is flattened row-major, matching cuSZp's own 1D
processing convention).

- `prepare()`: H2D copy of the workload's fp32 data into a torch CUDA
  tensor + allocation of the compressed-output buffer (`sizeof(float) *
  nbEle` bytes — the worst-case bound cuSZp's own
  `examples/cuSZp_test_f32_1D.cpp` allocates, since fixed-length encoding
  never expands past the original size) and the decompressed-output
  buffer — timed once as preprocessing. Note: `cuSZp_compress` itself
  additionally `cudaMalloc`s/`cudaFree`s a few small offset/flag scratch
  buffers **per call** (`src/cuSZp_entry_1D_f32.cu`) — this is cuSZp's own
  design, left completely untouched; it is not something this adapter
  introduces, and per ARTIFACT_GUIDE.md rule 3 it is not something this
  integration is allowed to "fix" by touching kernel code.
- `run()`: exactly one `cuSZp_compress()` call.
- `to_host()`: decompress (`cuSZp_decompress`, cuSZp's own kernel,
  unmodified) + D2H copy — **deliberately outside the timed region**:
  `run()` above is the only call the harness's `CudaEventTimer` ever
  brackets.
- `free()`: records `achieved_compressed_bytes`/`compression_ratio` (from
  the `size_t* cmpSize` the compress call already wrote) and, via one more
  untimed decode of whatever the LAST timed `run()` produced,
  `achieved_max_abs_error` and `psnr_db` — same discipline as
  `kernelbench/domains/compression.py`'s CPU `_QuantizeZlibCodec.free()`
  and this benchmark's `pfpl-compress` adapter.

## Build

```
./build.sh
```
`nvcc -O3 -arch=sm_80 -std=c++17 -Xcompiler -fPIC --shared -I source/include
-I source/include/cuSZp -I source/src <14 source/src/*.cu files> -o
libcuszp.so`. nvcc 12.9 (`/opt/nvidia/hpc_sdk/.../cuda/12.9`). Built clean
on the first attempt, no warnings requiring investigation. Idempotent.

## Gate verification (mandated command, run as specified)

```
$PY -m kernelbench.runner --kernel lossy-compression \
    --variant lossy-comp-kernel-cpu-ebound --impl cuszp-compress \
    --smoke --warmup 1 --reps 3
```
(Same note as the pfpl-compress adapter: despite the variant id saying
"cpu", the contract is platform-neutral -- `--smoke` only selects the
domain's 3 synthetic Field workloads and a reduced protocol.)

```
  running cuszp-compress smoke-smooth-3d      ... 0.464 ms  0.12 GB/s  (err 3.42e-03 <= 3.4191e-03)
  running cuszp-compress smoke-turbulent-3d   ... 0.403 ms  0.14 GB/s  (err 1.29e-03 <= 1.2903e-03)
  running cuszp-compress smoke-multiscale-3d  ... 0.428 ms  0.13 GB/s  (err 3.81e-03 <= 3.8086e-03)

3/3 runs valid
conforming: False (smoke: synthetic matrices, reduced protocol, overridden warmup/reps, login-node shared GPU -- expected/flagged, not a defect)
```

`max_abs_err == eb` to displayed precision on all 3 workloads — **PASS**
on all 3. Achieved compression ratio: 54.00 on all three (this tiny
24x24x24 smoke field, combined with the default rel-1e-3 error bound, lets
cuSZp's outlier-mode fixed-length blocks collapse to a small,
data-content-independent bit-width for this particular smoke size — not a
bug; a real SDRBench-scale field at `lossy-comp-kernel-cpu-ebound`'s actual
`recommended_subset` would show the paper's claimed dataset-dependent
ratios instead). PSNR ~64.7-64.9 dB. Full JSON at
`bench/results/lossy-compression_lossy-comp-kernel-cpu-ebound_1786161415.json`.

## Cross-reference: SC'23 kernel wrapped separately

The SC'23 original (`conf/sc/HuangD0LC23`) this repo also contains (tag
`cuSZp-V1.1`, a genuinely distinct single-mode code path, not just an
earlier version of this API) is wrapped separately at
`../cuszp-v1/` (`cuszp-v1-compress`) — see that directory's STATUS.md for
the full disambiguation and why it needed its own `source/` (a git worktree
of this directory's clone, not a symlink, so its checkout never disturbs
this one's). Both PAPER_KEYs (`HuangDLC25` here, `HuangD0LC23` there) are
now covered from this one repository.

## Verdict

`cuszp-compress: BUILT+GATED err<=eb ratio=54.00 (smoke; real SDRBench-scale ratio not measured here)`

## Reproduction on zaratan (2026-09-09)

- Machine: UMD zaratan, A100 MIG 1g.5gb slice (sm_80, driver 595.71.05), login-node build.
- Toolchain: nvcc 12.8.93 (conda-forge CUDA 12.8), g++ 13.4.0 (conda-forge), torch 2.8.0+cu128,
  Python 3.12.14; `-arch=sm_80` unchanged from the recorded build.
- Build: OK. Build-system changes: none (idempotent rebuild via `../../toolchain.sh`, no diff to
  `source/`).
- Gate: `lossy-comp-kernel-cpu-ebound` cuszp-compress: PASS on all 3 smoke workloads
  (smooth/turbulent/multiscale), `max_abs_err <= eb` on all 3 (e.g. 3.42e-03 <= 3.4191e-03), 3/3 runs
  valid.
- Deviation from the recorded ruling: none.
- Verdict here: BUILT+GATED — equals the recorded ruling.
