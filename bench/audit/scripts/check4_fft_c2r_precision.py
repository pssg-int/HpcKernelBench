"""
Check #4 (PRECISION DISCIPLINE, spectral.py FFT C2R/inverse operand build):
for a real-input INVERSE (C2R) transform, `_fft_operand()` builds the
frequency-domain operand by running a FORWARD rfftn on the ALREADY-ROUNDED
(dtype_real, e.g. fp32) real signal, using the CALLER's OWN backend rfftn.
`reference_fft()` independently builds its own frequency-domain operand by
running rfftn on an fp64 real signal (no rounding).

The other domains' convention (dense.py's own docstring, point analogous)
is "generate at the RUN's own precision first, THEN widen to fp64" so the
reference's operand is *exactly* the same underlying values as the impl's,
merely stored more precisely. Question: for FFT's C2R case, does the
reference's frequency-domain operand equal "widen(impl's fp32 real signal)
run through an fp64 forward FFT", i.e. is it still the SAME underlying
data merely computed more precisely -- or does it diverge from that,
because the reference's REAL signal itself (xr) is generated directly at
fp64 rather than derived from the fp32-rounded one the implementation
used?
"""
import sys
sys.path.insert(0, "/pscratch/sd/c/cunyang/msu/hpc-bench/kernel-papers/bench")

import numpy as np
from kernelbench.domains import spectral as sp

w = sp.FFTWorkload(name="probe-c2r", dims=(64,), batch=1, real_input=True,
                    direction="inverse", layout="out-of-place", seed=42)
params = {"seed": 42}

# What NumpyFFT.prepare() actually builds for run() (fp32 case)
impl = sp.NumpyFFT(precision="fp32")
h = impl.prepare(w, params)
impl_freq = h["x"]  # complex64 frequency-domain operand fed to irfftn

# What reference_fft() independently builds (fp64)
ref, scale = sp.reference_fft(w, params)

# reconstruct the REAL signal each one started from, to compare seeds/values
rng_impl = np.random.default_rng(42)
xr_impl_fp32 = rng_impl.uniform(-1.0, 1.0, size=(1, 64)).astype(np.float32)
rng_ref = np.random.default_rng(42)
xr_ref_fp64 = rng_ref.uniform(-1.0, 1.0, size=(1, 64)).astype(np.float64)

print("same seed -> same real signal BEFORE rounding?",
      np.array_equal(xr_impl_fp32.astype(np.float64), xr_ref_fp64))
print("max|xr_impl(fp32->fp64) - xr_ref(fp64)| =",
      np.abs(xr_impl_fp32.astype(np.float64) - xr_ref_fp64).max(),
      "  (nonzero => impl and reference do NOT start from identical real data;"
      " impl's real signal is fp32-ROUNDED before its own forward FFT builds"
      " the operand, reference's is not)")

# Now: is impl_freq == widen(fp64 forward FFT of the FP32-ROUNDED signal)?
# i.e. does the reference at least reproduce what a "correct fp32 kernel,
# widened" should look like for THIS specific operand-construction step?
freq_from_rounded_fp64compute = np.fft.rfftn(xr_impl_fp32.astype(np.float64), axes=(-1,))
d1 = np.abs(impl_freq.astype(np.complex128) - freq_from_rounded_fp64compute).max()
print("max|impl_freq(fp32 compute) - fp64-compute-of-SAME-rounded-signal| =", d1,
      " (this is the normal, expected fp32-arithmetic rounding gap)")

d2 = np.abs(ref.astype(np.float64) if False else None) if False else None
# Compare the *actual* correctness-gate inputs: run the inverse transform on
# each operand and see how the two errors decompose.
out_impl = impl.to_host(impl.run(h))
diff_vs_ref = np.abs(out_impl - ref)
print("\nmax|out_impl - ref| (the ACTUAL correctness-gate quantity) =",
      diff_vs_ref.max())

# Decompose: how much of that gap is "different real input" vs "fp32 arithmetic"?
# Build a fp64 impl (same code path) whose real input is deliberately the
# REFERENCE's own real signal, and see how close THAT gets to ref (should be
# ~machine epsilon, since then operand AND precision both match reference).
impl64 = sp.NumpyFFT(precision="fp64")
w64 = sp.FFTWorkload(name="probe-c2r-fp64", dims=(64,), batch=1, real_input=True,
                      direction="inverse", layout="out-of-place", seed=42)
h64 = impl64.prepare(w64, params)
out64 = impl64.to_host(impl64.run(h64))
diff_fp64_vs_ref = np.abs(out64 - ref)
print("max|out_impl(fp64) - ref| (same precision, same seed) =",
      diff_fp64_vs_ref.max(), " (near machine eps => fp64 impl matches ref"
      " operand-construction exactly, confirming the fp32 gap above comes"
      " from operand DIVERGENCE + arithmetic, not a bug at fp64)")
