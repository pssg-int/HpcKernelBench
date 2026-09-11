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
lib.lsCOMP_compression_uint16_bsize64.argtypes = [p, p, ctypes.POINTER(ctypes.c_size_t), Uint3, Uint4, ctypes.c_float, p]
lib.lsCOMP_compression_uint16_bsize64.restype = None
lib.lsCOMP_decompression_uint16_bsize64.argtypes = [p, p, ctypes.c_size_t, Uint3, Uint4, ctypes.c_float, p]
lib.lsCOMP_decompression_uint16_bsize64.restype = None

QB = Uint4(1, 1, 1, 1)
PTH = 1.0


def roundtrip16(u16: np.ndarray, dims):
    nbEle = u16.size
    d_ori = torch.from_numpy(np.ascontiguousarray(u16)).to("cuda")
    d_cmp = torch.empty(nbEle * 2 + 4096, dtype=torch.uint8, device="cuda")
    d_dec = torch.zeros(nbEle, dtype=torch.uint16, device="cuda")
    cmp_size = ctypes.c_size_t(0)
    dimsC = Uint3(*dims)
    lib.lsCOMP_compression_uint16_bsize64(
        ctypes.c_void_p(d_ori.data_ptr()), ctypes.c_void_p(d_cmp.data_ptr()),
        ctypes.byref(cmp_size), dimsC, QB, ctypes.c_float(PTH), ctypes.c_void_p(0))
    torch.cuda.synchronize()
    lib.lsCOMP_decompression_uint16_bsize64(
        ctypes.c_void_p(d_dec.data_ptr()), ctypes.c_void_p(d_cmp.data_ptr()),
        cmp_size, dimsC, QB, ctypes.c_float(PTH), ctypes.c_void_p(0))
    torch.cuda.synchronize()
    dec = d_dec.detach().to("cpu").numpy()
    ok = np.array_equal(dec, u16)
    if not ok:
        diff = np.where(dec != u16)[0]
        print(f"  MISMATCH: {len(diff)}/{nbEle}; first idx={diff[:5]}")
    return ok, int(cmp_size.value)


shape2 = (24, 24, 48)  # doubled last axis (2 uint16 per fp32 element), z fastest
rng = np.random.default_rng(0)
n3 = 24 * 24 * 24

print("=== uint16 test A: real fp32 turbulent-like field (mix of signs) reinterpreted as uint16 pairs ===")
f = rng.standard_normal((24, 24, 24)).astype(np.float32)
u16 = f.ravel().view(np.uint16)  # 2x elements, native byte order
print("n elements uint16:", u16.size, "expected:", n3 * 2)
ok, cs = roundtrip16(u16, shape2)
print("ok=", ok, "cmp_size=", cs, "orig_bytes=", u16.nbytes)

print("=== uint16 test B: constant 0xFFFF ===")
c = np.full(n3 * 2, 0xFFFF, dtype=np.uint16)
ok, cs = roundtrip16(c, shape2)
print("ok=", ok, "cmp_size=", cs)

print("=== uint16 test C: full random uint16 ===")
r = rng.integers(0, 65536, size=n3 * 2, dtype=np.uint32).astype(np.uint16)
ok, cs = roundtrip16(r, shape2)
print("ok=", ok, "cmp_size=", cs)
