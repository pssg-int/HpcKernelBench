"""
Diagnose why torch-conv-stencil (fp64 cuDNN conv) is slow on H100.

Times one stencil step (circular pad + conv2d, the same ops as
TorchConvStencil.run) with cudnn.benchmark off and on, times the pad and the
conv separately, compares against a plain device copy (memory-bandwidth
reference), and prints the CUDA kernel names the profiler saw.

    bench/gpu_run.sh -g h100 -t 15 -- '$PY diag_torch_conv.py'
    bench/gpu_run.sh -g a100 -t 15 -- '$PY diag_torch_conv.py'
"""
import torch
import torch.nn.functional as F

N, R, REPS = 16384, 1, 20
dt = torch.float64


def timeit(fn):
    for _ in range(3):
        fn()
    torch.cuda.synchronize()
    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(REPS):
        fn()
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / REPS


print(f"gpu    : {torch.cuda.get_device_name()}")
print(f"torch  : {torch.__version__}  cuda {torch.version.cuda}  cudnn {torch.backends.cudnn.version()}")
print(f"grid   : {N}x{N} fp64, radius {R}\n")

u = torch.rand(1, 1, N, N, dtype=dt, device="cuda")
w = torch.rand(1, 1, 2 * R + 1, 2 * R + 1, dtype=dt, device="cuda")
padded = F.pad(u, (R, R, R, R), mode="circular")
gcells = N * N / 1e9

copy_ms = timeit(lambda: u.clone())
print(f"plain copy (bandwidth reference) : {copy_ms:8.3f} ms  "
      f"({2 * u.numel() * 8 / copy_ms / 1e6:.0f} GB/s)")
print(f"circular pad only                : {timeit(lambda: F.pad(u, (R, R, R, R), mode='circular')):8.3f} ms")

for bench in (False, True):
    torch.backends.cudnn.benchmark = bench
    conv_ms = timeit(lambda: F.conv2d(padded, w))
    step_ms = timeit(lambda: F.conv2d(F.pad(u, (R, R, R, R), mode="circular"), w))
    print(f"cudnn.benchmark={bench!s:5}  conv only: {conv_ms:8.3f} ms   "
          f"pad+conv step: {step_ms:8.3f} ms  ({gcells / step_ms * 1e3:.2f} GCUP/s)")

torch.backends.cudnn.benchmark = False
with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CUDA]) as prof:
    for _ in range(3):
        F.conv2d(padded, w)
    torch.cuda.synchronize()
print("\nCUDA kernels used by conv2d (cudnn.benchmark=False):")
print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=8))
