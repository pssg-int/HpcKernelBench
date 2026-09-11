"""
Check #8 (REFERENCE INDEPENDENCE across stencil.py and spectral.py):

(a) stencil.py: reference_stencil() and NumpyStencil.run() both call the
    SAME module-level `_sweep()` helper. At NumpyStencil's own default
    precision (fp64), do they produce bit-identical output (i.e. is the
    "independent" reference actually the same code, called twice)?
    Contrast with ScipyConvolveStencil, which uses a genuinely different
    routine (scipy.ndimage.convolve).

(b) spectral.py: reference_fft() uses np.fft.*; NumpyFFT (the "numpy-fft"
    impl) ALSO uses np.fft.* -- literally the same function, at fp64. And
    even ScipyFFT ("scipy-fft", nominally a "different implementation") --
    does it give a BIT-IDENTICAL result to np.fft at fp64, suggesting they
    share the same underlying pocketfft core rather than being a genuinely
    independent numerical method?
"""
import sys
sys.path.insert(0, "/pscratch/sd/c/cunyang/msu/hpc-bench/kernel-papers/bench")
import numpy as np
from kernelbench.domains import stencil, spectral

print("=== (a) stencil.py reference sharing ===")
w = stencil.StencilWorkload(name="probe", kind="star", dims=2, radius=1,
                             grid_shape=(128, 128), timesteps=5)
params = {"timesteps": 5}
ref, scale = stencil.reference_stencil(w, params)

impl = stencil.NumpyStencil(precision="fp64")
h = impl.prepare(w, params)
out = impl.to_host(impl.run(h))
diff = np.abs(out - ref)
print(f"numpy-stencil (fp64, uses _sweep()) vs reference_stencil (also _sweep()):")
print(f"  max|out-ref| = {diff.max():.3e}   bit-identical = {np.array_equal(out, ref)}")

impl2 = stencil.ScipyConvolveStencil(precision="fp64")
h2 = impl2.prepare(w, params)
out2 = impl2.to_host(impl2.run(h2))
diff2 = np.abs(out2 - ref)
print(f"scipy-convolve-stencil (fp64, genuinely different code path -- "
      f"scipy.ndimage.convolve) vs reference_stencil:")
print(f"  max|out-ref| = {diff2.max():.3e}   bit-identical = {np.array_equal(out2, ref)}")

print("\n=== (b) spectral.py numpy.fft vs scipy.fft vs reference_fft ===")
wf = spectral.FFTWorkload(name="probe-fft", dims=(4096,), batch=4,
                           real_input=False, direction="forward",
                           layout="out-of-place", seed=42)
paramsf = {"seed": 42}
reff, scalef = spectral.reference_fft(wf, paramsf)

nf = spectral.NumpyFFT(precision="fp64")
hn = nf.prepare(wf, paramsf)
outn = nf.to_host(nf.run(hn))
print("numpy-fft (fp64) vs reference_fft (also np.fft, fp64): "
      f"max|out-ref|={np.abs(outn-reff).max():.3e}  "
      f"bit-identical={np.array_equal(outn, reff)}")

sf = spectral.ScipyFFT(precision="fp64")
hs = sf.prepare(wf, paramsf)
outs = sf.to_host(sf.run(hs))
print("scipy-fft (fp64, nominally a 'different' library) vs reference_fft (np.fft): "
      f"max|out-ref|={np.abs(outs-reff).max():.3e}  "
      f"bit-identical={np.array_equal(outs, reff)}")
print("numpy-fft vs scipy-fft directly (both fp64): "
      f"max diff={np.abs(outn-outs).max():.3e}  bit-identical={np.array_equal(outn, outs)}")
