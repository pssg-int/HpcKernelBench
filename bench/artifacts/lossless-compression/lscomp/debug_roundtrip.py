import os
import ctypes
import numpy as np
import torch

SO = os.path.join(os.path.dirname(os.path.abspath(__file__)), "liblscomp.so")


class Uint3(ctypes.Structure):
    _fields_ = [("x", ctypes.c_uint), ("y", ctypes.c_uint), ("z", ctypes.c_uint)]


class Uint4(ctypes.Structure):
    _fields_ = [("x", ctypes.c_uint), ("y", ctypes.c_uint), ("z", ctypes.c_uint), ("w", ctypes.c_uint)]


lib = ctypes.CDLL(SO)
p = ctypes.c_void_p
lib.lsCOMP_compression_uint32_bsize64.argtypes = [p, p, ctypes.POINTER(ctypes.c_size_t), Uint3, Uint4, ctypes.c_float, p]
lib.lsCOMP_compression_uint32_bsize64.restype = None
lib.lsCOMP_decompression_uint32_bsize64.argtypes = [p, p, ctypes.c_size_t, Uint3, Uint4, ctypes.c_float, p]
lib.lsCOMP_decompression_uint32_bsize64.restype = None

QB = Uint4(1, 1, 1, 1)
PTH = 1.0


def roundtrip(u32: np.ndarray, dims, zero_init=False):
    nbEle = u32.size
    d_ori = torch.from_numpy(np.ascontiguousarray(u32)).to("cuda")
    d_cmp = torch.empty(nbEle * 4 + 4096, dtype=torch.uint8, device="cuda")
    if zero_init:
        d_dec = torch.zeros(nbEle, dtype=torch.uint32, device="cuda")
    else:
        d_dec = torch.empty(nbEle, dtype=torch.uint32, device="cuda")
    cmp_size = ctypes.c_size_t(0)
    dimsC = Uint3(*dims)
    lib.lsCOMP_compression_uint32_bsize64(
        ctypes.c_void_p(d_ori.data_ptr()), ctypes.c_void_p(d_cmp.data_ptr()),
        ctypes.byref(cmp_size), dimsC, QB, ctypes.c_float(PTH), ctypes.c_void_p(0))
    torch.cuda.synchronize()
    lib.lsCOMP_decompression_uint32_bsize64(
        ctypes.c_void_p(d_dec.data_ptr()), ctypes.c_void_p(d_cmp.data_ptr()),
        cmp_size, dimsC, QB, ctypes.c_float(PTH), ctypes.c_void_p(0))
    torch.cuda.synchronize()
    dec = d_dec.detach().to("cpu").numpy()
    ok = np.array_equal(dec, u32)
    if not ok:
        diff = np.where(dec != u32)[0]
        print(f"  MISMATCH: {len(diff)}/{nbEle} elements differ; first few idx={diff[:10]}")
        for i in diff[:5]:
            print(f"    idx={i} orig={u32[i]:#010x} dec={dec[i]:#010x}")
    return ok, int(cmp_size.value)


shape = (24, 24, 24)
n = shape[0] * shape[1] * shape[2]

print("=== test 1: small non-negative ints (detector-like, < 65536) ===")
rng = np.random.default_rng(0)
a = rng.integers(0, 65536, size=n, dtype=np.uint32)
ok, cs = roundtrip(a, shape)
print("ok=", ok, "cmp_size=", cs, "orig_bytes=", n * 4)

print("=== test 2: full-range random uint32 ===")
b = rng.integers(0, 2**32 - 1, size=n, dtype=np.uint32, endpoint=True)
ok, cs = roundtrip(b, shape)
print("ok=", ok, "cmp_size=", cs)

print("=== test 3: fp32 turbulent-like field bit pattern ===")
f = rng.standard_normal(shape).astype(np.float32)
u = f.ravel().view(np.uint32)
ok, cs = roundtrip(u, shape)
print("ok=", ok, "cmp_size=", cs)

print("=== test 4: all zeros ===")
z = np.zeros(n, dtype=np.uint32)
ok, cs = roundtrip(z, shape)
print("ok=", ok, "cmp_size=", cs)

print("=== test 5: constant large value (0xFFFFFFFF) ===")
c = np.full(n, 0xFFFFFFFF, dtype=np.uint32)
ok, cs = roundtrip(c, shape)
print("ok=", ok, "cmp_size=", cs)

print("=== test 6: values with high bit set but small variety (simulate negative floats near 0) ===")
f2 = (rng.standard_normal(shape).astype(np.float32) * 1e-3)
u2 = f2.ravel().view(np.uint32)
print("sample bit patterns:", [hex(x) for x in u2[:5]])
ok, cs = roundtrip(u2, shape)
print("ok=", ok, "cmp_size=", cs)

print("=== test 7: all zeros, ZERO-INIT dec buffer ===")
ok, cs = roundtrip(z, shape, zero_init=True)
print("ok=", ok, "cmp_size=", cs)

print("=== test 8: full-range random uint32, ZERO-INIT dec buffer (isolate the temp_rate bug) ===")
ok, cs = roundtrip(b, shape, zero_init=True)
print("ok=", ok, "cmp_size=", cs)

print("=== test 9: bound values to < 2^30 (avoid MSB-set collision), ZERO-INIT ===")
b2 = rng.integers(0, 2**24, size=n, dtype=np.uint32)
ok, cs = roundtrip(b2, shape, zero_init=True)
print("ok=", ok, "cmp_size=", cs)

print("=== test 10: bound to < 2^31 exactly (one bit under the danger zone) ===")
b3 = rng.integers(0, 2**31 - 1, size=n, dtype=np.uint32)
ok, cs = roundtrip(b3, shape, zero_init=True)
print("ok=", ok, "cmp_size=", cs)
