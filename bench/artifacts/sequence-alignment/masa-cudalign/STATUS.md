# masa-cudalign (MASA-CUDAlign / MultiBP) — sequence-alignment

**Status: BUILD-FAILED**

- Paper: "Parallel Fine-Grained Comparison of Long DNA Sequences in
  Homogeneous and Heterogeneous GPU Platforms With Pruning" (TPDS'21) --
  MultiBP / MASA-CUDAlign. `PAPER_KEY = journals/tpds/FigueiredoNSTM21`.
- Artifact: https://github.com/edanssandes/MASA-CUDAlign
- Commit: `9412ba29a7150c3a284cb3ef48c4c62a472cbf27` (2021-10-22), cloned with
  `git clone --depth 1` (kept for provenance; nothing was patched).
- Toolchain: `nvcc` release 12.9, V12.9.41 (via
  `bench/artifacts/toolchain.sh`, CUDA_HOME=
  `/opt/nvidia/hpc_sdk/Linux_x86_64/25.5/cuda/12.9`); autoconf 2.71/automake
  1.16/libtool 2.4 (system `/usr/bin/*`); GPU target requested: sm_80
  (single NVIDIA A100-PCIE-40GB, shared login-node GPU).

## What was attempted

The repo ships two release snapshots checked in directly (not just the
`releases/*.tar.gz`): `masa-cudalign-3.9.1.1024/` and
`masa-cudalign-4.0.2.1028/` (the latter is the one the README's own
"Compiling" section documents, so it is the target). This is the MASA
architecture's CUDA extension: an autotools project with a nested
`libs/masa-core` subproject (`AC_CONFIG_SUBDIRS`).

```
cd masa-cudalign-4.0.2.1028
./configure --with-cuda-arch=sm_80 --with-cuda="$CUDA_HOME"
make -j8
```

`./configure` succeeds cleanly (finds `libcudart.so` once `--with-cuda` is
pointed at the toolchain-pinned CUDA_HOME; the default `/usr/local/cuda`
does not exist on this machine). `make` fails compiling the ONLY `.cu` file
in the project, `src/CUDAligner.cu`:

```
src/CUDAligner.cu(74): error: texture is not a template
  texture<unsigned char, 1, cudaReadModeElementType> t_seq0;
src/CUDAligner.cu(79): error: texture is not a template
  texture<unsigned char, 1, cudaReadModeElementType> t_seq1;
src/CUDAligner.cu(94): error: texture is not a template
  texture< int2, 1, cudaReadModeElementType> t_busH;
src/CUDAligner.cu(251,252,253,254,443,452): error: no instance of
  overloaded function "tex1Dfetch" matches the argument list
src/CUDAligner.cu(1187): error: identifier "cudaBindTexture" is undefined
src/CUDAligner.cu(1195): error: identifier "cudaUnbindTexture" is undefined
src/CUDAligner.cu(1265): error: identifier "cudaBindTexture" is undefined
src/CUDAligner.cu(1277): error: identifier "cudaUnbindTexture" is undefined
13 errors detected in the compilation of "src/CUDAligner.cu".
```

## Root cause

MASA-CUDAlign's DNA-sequence lookup uses the CUDA **legacy texture
reference API** (`texture<T,dim,mode>` file-scope objects bound via
`cudaBindTexture`/`cudaUnbindTexture`, read via `tex1Dfetch`) -- a real
performance technique from the CUDA 3.x-5.x era this codebase targets, used
throughout `CUDAligner.cu` for sequence and score-row caching (3 texture
declarations, `cudaBindTexture`/`cudaUnbindTexture` calls at
lines 1183/1187/1195/1265/1277, `tex1Dfetch` calls at 6 sites). Confirmed by
inspection that the CUDA 12.9 toolkit installed on this machine (NVIDIA HPC
SDK 25.5) no longer ships the legacy texture-reference header at all:

```
$ find $CUDA_HOME/include -iname '*texture*'
texture_fetch_functions.h texture_indirect_functions.h texture_types.h
# no cuda_texture_types.h -- the header that declares the `texture<>`
# template class does not exist in this toolkit installation.
```

`texture_types.h` in this toolkit only defines the modern **texture
object** API (`cudaTextureObject_t`, `cudaTextureDesc`) -- the legacy
reference-based API (`texture<>`, `cudaBindTexture`) has been fully removed
from the headers, not merely deprecated-with-a-warning. The identical error
was reproduced with the OLDER `masa-cudalign-3.9.1.1024` release too (same
`texture<...>` declarations, same `cudaBindTexture` calls in its `.cu`
sources) -- this is a structural property of the MASA-CUDAlign codebase
across both shipped versions, not a one-off typo in one release.

## Why this is BUILD-FAILED, not a build-system patch

Migrating from texture references to texture objects requires changing:
(a) the 3 file-scope `texture<...>` declarations into `cudaTextureObject_t`
fields threaded through the kernel's parameter list, (b) every
`tex1Dfetch(t_seq0, i)` call site into `tex1Dfetch<unsigned char>(texObj, i)`
with the object passed in, and (c) the bind/unbind lifecycle
(`cudaBindTexture`/`cudaUnbindTexture`) into
`cudaCreateTextureObject`/`cudaDestroyTextureObject` with an explicit
`cudaResourceDesc`/`cudaTextureDesc`. This changes the DEVICE-SIDE MEMORY
ACCESS MECHANISM the kernel uses to read the two aligned sequences and the
score buffer -- not an arch flag, include path, or CUDA-version guard.
ARTIFACT_GUIDE.md rule 3 is explicit that this class of change ("touching
kernel code") is out of scope for this integration pass ("if the kernel
itself must change to run, mark SKIPPED and say why"). No such patch was
attempted.

## Disposition

BUILD-FAILED with root-cause evidence above (rule 7's "genuinely does not
implement the track's kernel" is the adjacent case; this is instead
"implements it, but cannot be compiled on the pinned toolchain without a
kernel-code change"). No adapter.py is provided (nothing to wrap). This
was the primary target for `seqalign-exact-pairwise-kernel`
(long-DNA exact Smith-Waterman/Needleman-Wunsch on GPU); with it
unavailable, that variant has no GPU paper-artifact competitor in this
integration pass -- see `../README.md` for the track-level accounting.

## Not done

- No attempt to install an older CUDA toolkit that still ships the legacy
  texture-reference headers: the toolchain pin (`bench/artifacts/
  toolchain.sh`, nvcc 12.9) is mandatory per this task's environment
  contract, and using a different, unpinned toolkit for one artifact would
  itself be an undocumented deviation this integration avoids.
- No patch to `CUDAligner.cu`'s texture usage (forbidden by rule 3, see
  above).
- `libs/masa-core` (the nested MASA architecture library `./configure`
  successfully recurses into) was never separately investigated further,
  since the blocking failure is entirely within `CUDAligner.cu`, the only
  GPU-facing file in the `masa-cudalign` extension itself.
