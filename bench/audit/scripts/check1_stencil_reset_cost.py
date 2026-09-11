"""
Check #1 (PREPROCESSING PLACEMENT): does NumpyStencil.run()'s
`np.copyto(bufs[0], h["u0"])` buffer reset -- which sits INSIDE the timed
region (harness.py wraps `with impl.timer() as t: impl.run(handle)`) --
contaminate the reported GCUP/s in a way that matters?

Measure reset-alone cost vs. full run() cost at a MODEST T (kept small so the
probe finishes quickly on a login node) and reason analytically: reset cost
is O(cells), independent of T, while a full run() costs O(cells * support *
T); so reset/run's fraction should be ~ 1/(support*T) up to a constant --
measuring at small T and extrapolating to spec T=1000 is valid and much
faster than actually running T=1000. CPU only.
"""
import sys, time
sys.path.insert(0, "/pscratch/sd/c/cunyang/msu/hpc-bench/kernel-papers/bench")

import numpy as np
from kernelbench.domains import stencil


def time_it(fn, reps=5):
    times = []
    for _ in range(reps):
        t0 = time.perf_counter()
        fn()
        times.append(time.perf_counter() - t0)
    return sorted(times)[len(times) // 2]


def probe(name, w, T, reps=5):
    impl = stencil.NumpyStencil(precision="fp64")
    h = impl.prepare(w, {"timesteps": T})
    u0, bufs = h["u0"], h["bufs"]
    reset_t = time_it(lambda: np.copyto(bufs[0], u0), reps=reps)
    full_t = time_it(lambda: impl.run(h), reps=reps)
    frac = reset_t / full_t * 100
    print(f"{name:26s} T={T:5d} cells={w.cells:>10,d} support={w.support_size:2d} "
          f"reset={reset_t*1e3:8.4f}ms full_run={full_t*1e3:9.4f}ms "
          f"reset/run={frac:6.2f}%  (implied @T=1000: ~{frac*T/1000:.3f}%)",
          flush=True)
    return frac


print("=== stencil buffer-reset timing-scope contamination ===", flush=True)

# the module's OWN smoke config, run exactly as smoke_workloads() defines it
w_smoke = stencil.StencilWorkload(name="probe-smoke-star2d1r", kind="star", dims=2,
                                   radius=1, grid_shape=(256, 256), timesteps=5)
probe("smoke-star2d1r (T=5, AS-IS)", w_smoke, 5, reps=10)

# same shape, but with more timesteps to see how the fraction shrinks
probe("star2d1r 256x256", w_smoke, 50, reps=8)
probe("star2d1r 256x256", w_smoke, 200, reps=5)

# a bigger, spec-shaped grid (2D star2d1r) at modest T, kept small enough to
# finish quickly; support_size=5 same as the 16384^2 spec grid
w_mid = stencil.StencilWorkload(name="probe-mid-star2d1r", kind="star", dims=2,
                                 radius=1, grid_shape=(1024, 1024), timesteps=1)
probe("star2d1r 1024x1024", w_mid, 20, reps=5)
probe("star2d1r 1024x1024", w_mid, 100, reps=3)

print("\ndone", flush=True)
