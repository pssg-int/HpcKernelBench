# cuFalcon — STATUS

**Outcome: SKIPPED (confirmed, second pass) — a standalone kernel launch
boundary DOES exist (first pass missed it), but the transform it computes is
Falcon's own negacyclic-ring evaluation (evaluation at ODD powers of a
primitive 2N-th root of unity), empirically confirmed to require a
non-trivial per-index INPUT TWIST, not a mere permutation/scale, to match a
standard DFT of the input array. Per the task brief's own bar ("if the
output is a standard DFT up to a documented permutation/scale... wrap it;
if not, keep SKIPPED"), this fails on the twist, independent of the output
ordering.**

Paper: "cuFalcon: An Adaptive Parallel GPU Implementation for
High-Performance Falcon Acceleration", TPDS 2026.
PAPER_KEY = `journals/tpds/LiWSYDZ26`.
Repo: `https://github.com/encryptorion-lab/cuFalcon`
(commit `659abf54a7113b882c3684462bfbe9f37192d25f`, 2026-06-15;
`git clone --depth 50` into `./source/`).

## First-pass claim was incomplete

The first pass (2026-08-08) grepped only `cuFalcon_1024/include/fft.cuh`'s
`__device__`/`__global__` lines filtered to `fft1024`/`ifft1024` and
concluded "there is no `__global__` kernel anywhere in the repo whose job is
'just transform this array'". That grep pattern also matched a kernel it
did not mention: `include/fft.cuh:283`

```cpp
__global__ void ifft_t(fpr *d_b4, fpr *d_b5, size_t d_mem_pool_pitch);
```

whose body (`src/fft.cu:196-227`) is EXACTLY two independent calls to
`ifft1024` (once on `b4`, once on `b5`) plus a `1/N` rescale
(`fpr_mul(s_f[i], fpr_p2_tab[10])`, i.e. `* 2^-10`) — no
signing/sampling/NTT logic fused in at all, unlike `sign_dyn` /
`ffSampling_fft_dyntree`. This genuinely IS a standalone "just inverse-
transform this array" kernel, and the analogous forward launcher is a
one-for-one mirror of `convert_B_fft`'s own f0-block (`src/fft.cu:229-254`,
called at `src/api.cu:107` as `convert_B_fft<<<BATCH, Falcon_N/4,
...>>>(...)`) — i.e. wrapping the forward direction needs only a THIN new
`__global__` launcher with the artifact's own existing per-thread indexing
convention, not new kernel-internal logic. So the premise of the first
pass's SKIPPED reasoning (no launch boundary exists) does not hold; this
second pass instead settles the question the task actually asked
("determine exactly what it computes") empirically.

## What `fft1024`/`ifft1024` actually compute (empirical proof)

Falcon's own spec defines its "FFT" as evaluating a real polynomial
`f mod (x^n+1)` at the **odd** powers of a primitive `2n`-th root of unity,
`zeta_j = f(omega^(2j+1))` for `j = 0..n/2-1`, `omega = exp(i*pi/n)` — this
is the classical negacyclic-ring transform used for polynomial
multiplication mod `x^n+1`, algebraically distinct from a standard N-point
DFT (which evaluates at the **N-th** roots of unity, i.e. only the EVEN
powers of the same `omega`). `include/fft.cuh:665-823` (`fft1024`) matches
this exactly: level-by-level butterflies multiplying by `fpr_gm_tab[...]`
twiddle entries indexed `(2 + tid>>8)<<1`, `(4+tid>>7)<<1`, ...,
`(512+tid)<<1` (fft.cuh:669,687,705,723,742,760,779,798,815) — the classical
Falcon reference `fft.c` bit-reversed-`omega`-power table layout, not a
plain Cooley-Tukey N-th-root table.

Rather than trust that literature-matching alone, this was verified directly
on the GPU with a probe kernel (`/tmp/.../scratchpad/falcon_probe.cu`, NOT
part of the artifact or the adapter — test-only, compiled standalone against
the artifact's own unmodified `fft1024`/`fpr.cuh`/`api.cuh`, mirroring
`convert_B_fft`'s f0-block launch convention verbatim:
`fft_probe<<<1,256>>>`, `blockDim.x=Falcon_N/4` per `src/api.cu:107`):

Feed a real unit-impulse input `x_k = delta(k - K)` for `K = 0..5` and read
raw output slot 0 (`f0[0]`, `f0[512]` = real/imag of whatever frequency bin
Falcon's internal butterfly network places at output index 0):

```
impulse@K=0 -> output[0] = (1.000000000000, 0.000000000000)
impulse@K=1 -> output[0] = (0.999995293810, 0.003067956763)
impulse@K=2 -> output[0] = (0.999981175283, 0.006135884649)
impulse@K=3 -> output[0] = (0.999957644552, 0.009203754782)
impulse@K=4 -> output[0] = (0.999924701839, 0.012271538286)
impulse@K=5 -> output[0] = (0.999882347454, 0.015339206285)
```

These match, to full double precision, `omega^K = exp(i*pi*K/1024)` for
every `K` tested (`cos(pi*K/1024), sin(pi*K/1024)` computed independently in
Python: e.g. `K=3 -> (0.999957644552, 0.009203754782)`, bit-for-bit). This
is decisive: for a genuine (possibly permuted/relabeled) N-point DFT of the
input array, output slot 0 is ALWAYS the DC bin of whatever frequency label
the permutation assigns it, and the DC bin of an impulse response is
`sum_k x_k = 1` **for every impulse position K** — a plain DFT's DC-bin
value cannot depend on where the impulse sits, under ANY permutation of
which physical slot holds it (permuting output positions relabels bins, it
cannot make the fixed value at a given slot become K-dependent when swept
over K). Falcon's output slot 0 instead tracks `omega^K` exactly as `K`
varies — proof that a per-index, non-constant complex multiplier
(`x_k -> x_k * omega^k`, the classical negacyclic "twist") is applied to the
input before whatever DFT-like butterfly network runs, not merely a
post-hoc reordering or a global scale of a plain DFT's output.

Algebraically: `FalconFFT(f)_j = f(omega^(2j+1)) = sum_k f_k * omega^k *
exp(-i*2*pi*j*k/n) = DFT_n(twist(f))_j` where `twist(f)_k = f_k * omega^k`,
`omega = exp(i*pi/n)`. `to_host()` can only post-process the KERNEL'S
OUTPUT; it cannot retroactively undo an input-side twist without the
adapter multiplying the input by `omega^{-k}` BEFORE the (otherwise
unmodified) kernel call and by `omega^{+k}` after — i.e. computing the twist
ourselves in Python/host code, external to the kernel, so that the net
input-to-output map becomes a real DFT by CONSTRUCTION rather than by the
kernel's own semantics. That would benchmark "an artificial twist we wrote,
composed with a kernel repurposed for something it was never designed to
compute" — precisely the pattern rule 7 and the `ffcz`/`cutensor-tubal`
SKIPPED rulings in this same track reject (measuring a foreign transform's
speed and calling it the paper's own FFT kernel). The task brief's own bar
is "a standard DFT up to a documented **permutation/scale**" — a per-index
phase twist is neither; it is a different transform (well-defined,
invertible, and exactly what Falcon needs for negacyclic convolution, but
not a DFT of the given array).

`cuFalcon_512/include/fft.cuh` uses the identical `fpr_gm_tab`-indexed
butterfly structure (verified by grep: same twiddle-index pattern at
`fft.cuh:622-753`, same standalone `ifft_t` declaration at `fft.cuh:283`) —
the same conclusion applies to N=512 by construction; not re-run
empirically since N=1024's proof already settles the general claim (both
sizes implement the same Falcon FFT specification).

## Verdict

`cufalcon: SKIPPED (confirmed) -- ifft_t (fft.cu:196, a genuine standalone
kernel the first pass missed) and fft1024/ifft1024 (fft.cuh:665-940) compute
Falcon's negacyclic-ring transform (evaluation at ODD powers of a 2N-th root
of unity), NOT a DFT of the input array up to permutation/scale -- empirically
proven via impulse-response probe: output slot 0 for an impulse at position K
equals omega^K = exp(i*pi*K/1024) exactly (K=0..5, double-precision match),
whereas a genuine (possibly permuted) DFT's DC bin is K-independent (always
1+0i) for any impulse position. Recovering a DFT requires a non-trivial
per-index input twist, not a permutation/scale, which exceeds what
ARTIFACT_GUIDE.md rule 1 / this task's own bar allows without repurposing the
kernel for a transform it was not designed to compute.)`
