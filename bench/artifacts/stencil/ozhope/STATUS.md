# ozHOPE — STATUS

**Outcome: SKIPPED — tensor-core reconstruction kernel is not separable
with a plain-array interface (entangled with undocumented global env-var
state and mesh-derived channel geometry); evidence below, not asserted**

Paper: "High-Order-Preserving Acceleration of a Shallow-Water Dynamical
Core Using Tensor Units", IPDPS'26. `PAPER_KEY = conf/ipps/YaoZLX26`.
Repo: `https://github.com/jnyao/ozHOPE` (commit
`8529c63b77b188c9be5df3de19b9d86d7268c8d0`, 2026-05-29; `git clone --depth 1`
into `./source/`, untouched — read-only, no build attempted).

## What the artifact actually is

`source/FVM/` is the shallow-water dynamical core itself (Python/PyTorch:
`dycore.py` time integration, `recon.py` high-order finite-volume
reconstruction, `mesh.py` cubed-sphere geometry). `source/ozIMMU/` is a
CUDA/C++ extension implementing the Ozaki accurate-low-precision-emulation
scheme as a torch custom op (`pyt_ozaki_kernel.cu` -> `TORCH_LIBRARY(my_ops,
...)` registering `custom_ozcudnn`/`custom_accumulate_ozcudnn`,
`pyt_ozaki_op.cpp`), plus a standalone accurate-GEMM library (`src/gemm.cu`
etc., unrelated to the stencil path — not used by the reconstruction kernel,
confirmed no `ozimmu::` namespace call appears anywhere in
`pyt_ozaki_kernel.cu` despite `#include <ozimmu/ozimmu.hpp>`).

## The candidate "stencil kernel": `recon.py`'s `tpp_recon_class`

`recon.py:385` defines `class tpp_recon_class(torch.nn.Module)`, whose
`forward(field)` (`recon.py:522-541`) is architecturally a genuine
stencil-as-convolution kernel, structurally identical in spirit to the
convstencil/flashfftstencil/lorastencil siblings already in this
directory: at `prec_mode=0` it runs a plain cuDNN convolution
(`self.convgraph64.execute(...)`, `recon.py:526-533`) of `field` against a
FIXED, precomputed reconstruction-convolution weight tensor
`self.Rmtx_conv`; at `prec_mode=1` (the paper's actual Ozaki/tensor-core
contribution) it calls `torch.ops.my_ops.custom_ozcudnn(...)`
(`recon.py:502`) to Ozaki-split the fp64/fp32 input into fp16 slices, runs
the SAME convolution via a HALF-precision `cudnn.pygraph`
(`recon.py:443-491`, genuine Tensor-Core dispatch on cuDNN's fp16 conv
path), then `torch.ops.my_ops.custom_accumulate_ozcudnn(...)`
(`recon.py:519`) to reassemble the high-precision result. This part of the
investigation initially looked promising: `forward()` itself contains no
time-integration/flux-computation code, so at the level of "does `forward`
call into the RK loop," it is separable.

## Why it is NOT wrappable with a plain-array interface (verified, not assumed)

Confirmed the required Python dependency (`import cudnn` at `recon.py:4`,
the separately-distributed `nvidia-cudnn-frontend` PyPI package — NOT
vendored in this repo, NOT pre-installed in this project's shared
`plexus_env`) installs cleanly and its `cudnn.pygraph`/`data_type`/
`heur_mode` API matches this machine's torch-bundled cuDNN backend
(`torch.backends.cudnn.version() == 91002`; probe done with `pip install
--target=<scratch dir>`, isolated from the shared venv, per
ARTIFACT_GUIDE.md's own "a pip `--target` tree ... fetched by build.sh"
allowance — not installed into the shared environment). The ATen/cuDNN C++
headers `pyt_ozaki_kernel.cu` needs (`ATen/cudnn/{Handle,Types,Utils}.h`)
are also present in this machine's torch 2.8 install, and `ozIMMU/src/cutf`
(the header-only math-primitives dependency) is vendored directly in the
clone (not a git submodule needing a fetch). All of this looked buildable.

Reading `custom_ozcudnn`'s/`custom_accumulate_ozcudnn`'s actual C++ bodies
(`pyt_ozaki_kernel.cu:479-583`) is where the blocker is:

1. **Hardcoded channel-count constants baked into the compiled kernel**
   (`pyt_ozaki_kernel.cu:39-40`): `#define OZCUDNN_CHNLS 16` and
   `#define OZCUDNN_OUTCHNLS 64`. `ConvolutionLayer_sg` is constructed as
   `(channels*OZCUDNN_CHNLS, out_channels, kernel_size, width, height)`
   (`pyt_ozaki_kernel.cu:492-493`) — the Ozaki bit-slicing scheme's number
   of fp16 splits (16) and the reconstruction operator's output-channel
   budget (64) are compile-time constants tuned to ozHOPE's OWN
   reconstruction operator (`Rmtx_conv.shape[0]`, the number of polynomial
   reconstruction coefficients this dynamical core's high-order scheme
   produces per cell), not a parameter this integration can independently
   choose to match a plain 1-channel stencil weight kernel.
2. **Correctness depends on undocumented global environment-variable
   state**, read with NO validation or documented default:
   `getenv("NUMSPLIT")` (`pyt_ozaki_kernel.cu:502-503`, unchecked —
   `atoi(NULL)` if unset), `getenv("BITS_PER_SLICE")` and
   `getenv("OZCUDNN_OUTCHNLS")` (`pyt_ozaki_kernel.cu:557-560`, likewise
   unchecked). Neither the README nor any file under `source/FVM`
   (`recon.py`, `dycore.py`) sets these — they are presumably exported by
   a run script not included in the artifact's own repository (no
   `run.sh`/`launch.sh`/`.env` file exists anywhere in this clone). Without
   the paper's own (unpublished, in this artifact) values for these three
   knobs, the Ozaki split/accumulate math has no defined behavior to
   reproduce — supplying arbitrary values would not be "wrapping the
   artifact's kernel," it would be inventing a different computation.
3. **Constructor requires the full dynamical-core mesh object.**
   `tpp_recon_class.__init__(self, Rmtx_conv, nvar, mesh, device,
   prec_mode)` (`recon.py:385-386`) takes `mesh` — ozHOPE's cubed-sphere
   mesh/geometry object (`FVM/mesh.py`, 861 lines, itself dependent on the
   MPI-parallel panel decomposition in `FVM/parallel.py`) — not a bare
   shape. Independently reconstructing a compatible `mesh` object without
   also standing up ozHOPE's own mesh-generation pipeline was judged
   outside this integration's bounded scope.

Taken together — a kernel whose channel geometry is compiled in for one
specific reconstruction operator, whose numerical behavior depends on
environment variables this repository never sets or documents, and whose
constructor requires the dynamical core's own mesh object — this is not a
"plain-array interface, wrap the kernel" situation despite `forward()`'s
surface-level separability from the time-stepping loop. This is the
functional equivalent of the "fused with the dynamical core, not
separable" case the task anticipated: the tensor-core operator cannot be
exercised independently of ozHOPE's own reconstruction-setup machinery
within a reasonable integration effort, even though the Python call is not
textually inlined into the RK loop.

## What was NOT attempted, and why (rule 5 / time budget, stated honestly)

- No attempt to compile `ozIMMU`'s `ozaki_op` CUDA/C++ library (recorded
  above as *believed* buildable, based on header/dependency presence —
  never actually invoked, since points 1-3 above make a correct, gate-able
  invocation impossible without inventing undocumented configuration).
- No attempt to instantiate a real `mesh` object to satisfy
  `tpp_recon_class.__init__`.
- The `prec_mode=0` (plain fp64 cuDNN convolution, no Ozaki/Tensor-Core
  involvement at all) path was NOT wrapped either, even though it alone
  would have been comparatively easy to build (it needs only the
  `cudnn`-frontend Python package, no custom op compilation) — it is
  explicitly the paper's OWN non-contribution baseline, not the tensor-unit
  kernel this track's spec variant 2/3 asks for, so wrapping it alone would
  not represent what ozHOPE actually contributes.

## Verdict

`ozhope: SKIPPED (tensor-core reconstruction kernel entangled with
hardcoded channel-count constants, undocumented NUMSPLIT/BITS_PER_SLICE/
OZCUDNN_OUTCHNLS environment state, and the dynamical core's own mesh
object -- pyt_ozaki_kernel.cu:39-40,502-503,557-560; recon.py:385-386 --
not a plain-array-interface kernel separable from the dynamical core
within this integration's bounded scope)`
