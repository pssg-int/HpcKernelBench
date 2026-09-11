import os
import ctypes
import numpy as np
from scipy import ndimage

SO = os.path.join(os.path.dirname(os.path.abspath(__file__)), "libyacclab_buf.so")
lib = ctypes.CDLL(SO)
p = ctypes.c_void_p
lib.yacclab_buf_prepare.argtypes = [ctypes.POINTER(ctypes.c_uint8), ctypes.c_int, ctypes.c_int]
lib.yacclab_buf_prepare.restype = p
lib.yacclab_buf_run.argtypes = [p]
lib.yacclab_buf_run.restype = None
lib.yacclab_buf_fetch_labels.argtypes = [p, ctypes.POINTER(ctypes.c_int32)]
lib.yacclab_buf_fetch_labels.restype = None
lib.yacclab_buf_free.argtypes = [p]
lib.yacclab_buf_free.restype = None


def run(img):
    rows, cols = img.shape
    img_c = np.ascontiguousarray(img, dtype=np.uint8)
    h = lib.yacclab_buf_prepare(img_c.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8)), rows, cols)
    lib.yacclab_buf_run(h)
    out = np.empty((rows, cols), dtype=np.int32)
    lib.yacclab_buf_fetch_labels(h, out.ctypes.data_as(ctypes.POINTER(ctypes.c_int32)))
    lib.yacclab_buf_free(h)
    return out


def canonical(labels):
    labels = labels.copy()
    flat = labels.ravel()
    _uniq, first_idx, inv = np.unique(flat, return_index=True, return_inverse=True)
    order = np.argsort(first_idx)
    rank = np.empty_like(order)
    rank[order] = np.arange(len(order))
    return rank[inv].reshape(labels.shape)


rng = np.random.default_rng(0)
for trial in range(5):
    h, w = rng.integers(20, 60), rng.integers(20, 60)
    img = (rng.random((h, w)) < 0.45).astype(np.uint8)
    got = run(img)
    # 8-connectivity reference via scipy
    struct8 = np.ones((3, 3), dtype=int)
    ref, n = ndimage.label(img, structure=struct8)
    # background must stay 0 in both; foreground canonicalized
    got_bg = (got == 0)
    ref_bg = (ref == 0)
    bg_match = np.array_equal(got_bg, ref_bg)
    # canonicalize only foreground groupings (ignore numeric label identity)
    ok = bg_match and np.array_equal(canonical(got), canonical(ref))
    print(f"trial {trial}: shape=({h},{w}) n_components_ref={n} bg_match={bg_match} full_match={ok}")
    if not ok:
        print("  MISMATCH sample got[:5,:5]=", got[:5, :5].tolist())
        print("  ref[:5,:5]=", ref[:5, :5].tolist())
