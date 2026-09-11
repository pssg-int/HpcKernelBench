#!/usr/bin/env python3
"""
Build-time code generator for the AN5D adapter (bench/artifacts/stencil/an5d).

AN5D (CGO'20) is a PPCG-based source-to-source GENERATOR: its own contribution
is the CUDA it emits for a given stencil + tuning parameters (temporal
blocking degree "bt", spatial tile sizes, streaming length "sl"). The
generator toolchain itself could not be built on this machine within the
integration's time budget (see STATUS.md: PPCG's "pet" C-frontend submodule
is pinned to a 2019 fork requiring clang/LLVM <= ~3.8's C++ AST API, and only
llvm/{18,20,21,22} are available via `module spider llvm` here -- porting
across a ~15-major-version Clang API gap is source-porting work, not a build
flag, and building an old LLVM/Clang from source would itself exceed the
budget). AN5D-Artifact (the artifact repo we DID clone, `source/`) ships its
own PRE-GENERATED CUDA output for its benchmark suite under
`source/compiled/{double,float}/<shape>-<tuning>_{host,kernel}.cu` -- these
ARE the paper's own generated kernels (produced by running the generator
once, upstream, and checked into the artifact repo for the "Tuned"
evaluation path described in source/README.md). Wrapping them is the
documented fallback for a generator this integration could not build.

This script does NOT re-derive or hand-transcribe AN5D's generated dispatch
code. It MECHANICALLY extracts it from the artifact's own `_host.cu` file by
brace-matching (every `_host.cu` file the artifact ships has the identical
generated skeleton: `cudaMalloc` device buffer -> H2D copy -> a `{ ... }`
compound statement containing ONLY the temporal-blocking kernel-launch
cascade -> `cudaCheckKernel()` -> D2H copy -> `cudaFree`), and splices that
extracted text UNMODIFIED into a small `extern "C"` shim exposing separate
prepare()/run()/copy_out()/free() entry points -- the same
split-the-monolithic-driver pattern already used by the spider and
convstencil sibling adapters (see their bridge.cu docstrings), needed here
because AN5D's own `kernel_stencil()` does malloc+H2D+dispatch+D2H+free as
ONE function call, but our harness times prepare() (H2D) and run() (kernel
launches only) separately.

Usage: $PY gen_bridge.py   (invoked by build.sh; writes bridge_<shape>.cu
into this directory for each entry in SHAPES below)
"""
from __future__ import annotations

import os
import re

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(HERE, "source", "compiled", "double")

# One pre-generated tuning config per shape, chosen for the SMALLEST temporal
# blocking degree "bt" available in source/compiled/double/ (fewest remainder
# branches in the generated dispatch cascade -- see STATUS.md for the full
# file listing this was picked from). bt is cross-checked against the actual
# `__side0Len` in the extracted code at generation time (assert below), not
# just trusted from the filename.
SHAPES = {
    "star2d1r": dict(dims=2, cu="star2d1r-256-10-128", bt=10),
    "box2d1r":  dict(dims=2, cu="box2d1r-512-8-128",    bt=8),
    "star2d3r": dict(dims=2, cu="star2d3r-512-4-128",   bt=4),
    "box2d3r":  dict(dims=2, cu="box2d3r-128-2-128",    bt=2),
    "star3d1r": dict(dims=3, cu="star3d1r-32x32-4-128", bt=4),
    "box3d1r":  dict(dims=3, cu="box3d1r-32x32-3-128",  bt=3),
}

_CUDA_CHECK_MACROS = r"""
#define cudaCheckReturn(ret) \
  do { \
    cudaError_t cudaCheckReturn_e = (ret); \
    if (cudaCheckReturn_e != cudaSuccess) { \
      fprintf(stderr, "CUDA error: %s\n", cudaGetErrorString(cudaCheckReturn_e)); \
      fflush(stderr); \
    } \
    assert(cudaCheckReturn_e == cudaSuccess); \
  } while(0)
#define cudaCheckKernel() \
  do { \
    cudaCheckReturn(cudaGetLastError()); \
  } while(0)
"""


def extract_dispatch(host_cu_text: str, expected_bt: int) -> str:
    """
    Mechanically extract AN5D's own generated kernel-launch cascade from a
    `_host.cu` file: find the `#ifndef AN5D_TYPE` marker every AN5D-generated
    file starts its dispatch block with, walk back to the enclosing `{` (the
    compound statement opened right after the H2D memcpy), then brace-match
    forward to find its closing `}`. The immediately following
    `cudaCheckKernel();` statement is included too. No other part of the
    generated file (its own cudaMalloc/memcpy/cudaFree, which our own
    prepare()/copy_out()/free() replace) is used.
    """
    lines = host_cu_text.splitlines(keepends=True)
    marker_idx = next(i for i, l in enumerate(lines) if l.strip() == "#ifndef AN5D_TYPE")
    start = next(i for i in range(marker_idx - 1, -1, -1) if lines[i].strip() == "{")

    depth = 0
    end = None
    for i in range(start, len(lines)):
        depth += lines[i].count("{") - lines[i].count("}")
        if depth == 0 and i > start:
            end = i
            break
    if end is None:
        raise RuntimeError("extract_dispatch: brace matching did not close")

    body = "".join(lines[start:end + 1])

    j = end + 1
    while lines[j].strip() == "":
        j += 1
    if lines[j].strip() != "cudaCheckKernel();":
        raise RuntimeError(
            f"extract_dispatch: expected 'cudaCheckKernel();' right after the "
            f"dispatch block, found {lines[j]!r}")
    body += lines[j]

    # Sanity check: the first side0Len this file's PRIMARY (largest) block
    # declares must match the bt this script was told to expect (catches a
    # wrong file being picked, or AN5D-Artifact changing its own naming).
    m = re.search(r"const AN5D_TYPE __side0Len = (\d+);", body)
    if not m or int(m.group(1)) != expected_bt:
        raise RuntimeError(
            f"extract_dispatch: expected primary __side0Len={expected_bt}, "
            f"found {m.group(1) if m else None!r}")
    return body


def malloc_expr(dims: int) -> str:
    """(size_t)(2) * (size_t)(dimsize)^dims -- same expression AN5D's own
    cudaMalloc/cudaMemcpy lines use (verified against the actual files, not
    assumed): 2 buffers (ping-pong) times dimsize^dims doubles."""
    return "(size_t)(2)" + " * (size_t)(dimsize)" * dims


BRIDGE_TEMPLATE = '''\
// bridge_{shape}.cu -- GENERATED by gen_bridge.py from AN5D-Artifact's own
// pre-generated CUDA (source/compiled/double/{cu}_{{kernel.cu,kernel.hu}}).
// The #include below pulls in the artifact's OWN __global__ kernel
// definitions (kernel0_1..kernel0_{bt}) unmodified. The DISPATCH_BODY block
// further down is copied byte-for-byte (via brace-matching, not
// hand-transcribed -- see gen_bridge.py) from
// source/compiled/double/{cu}_host.cu's own generated temporal-blocking
// launch cascade. Do not edit by hand; re-run gen_bridge.py (called by
// build.sh) instead.
//
// -fvisibility=hidden (build.sh) keeps kernel0_N and every other symbol here
// local to THIS .so: all six shapes' pre-generated files independently
// define kernel0_1, kernel0_2, ... at file scope, so without hidden
// visibility, loading more than one shape's .so in the same process could
// have the dynamic linker resolve a call in one shape's dispatch cascade to
// a DIFFERENT shape's identically-named kernel (silently wrong results, not
// a link error). Only the four an5d_{shape}_* entry points below are
// exported (explicit default-visibility attribute).

#include <cuda_runtime.h>
#include <cstdio>
#include <cassert>
#include <cstring>

#include "source/compiled/double/{cu}_kernel.cu"

{macros}

struct An5dHandle {{
    double *dev_A;
    int dimsize;
}};

extern "C" __attribute__((visibility("default")))
void *an5d_{shape}_prepare(const double *host_field, int dimsize) {{
    double *dev_A;
    size_t nbytes = {malloc_expr} * sizeof(double);
    cudaCheckReturn(cudaMalloc((void **)&dev_A, nbytes));
    cudaCheckReturn(cudaMemcpy(dev_A, host_field, nbytes, cudaMemcpyHostToDevice));
    An5dHandle *h = new An5dHandle();
    h->dev_A = dev_A;
    h->dimsize = dimsize;
    return h;
}}

extern "C" __attribute__((visibility("default")))
void an5d_{shape}_run(void *handle, int timestep) {{
    An5dHandle *h = (An5dHandle *)handle;
    double *dev_A = h->dev_A;
    int dimsize = h->dimsize;
    assert(dimsize >= 3 && timestep >= 1);
{dispatch_body}
}}

extern "C" __attribute__((visibility("default")))
void an5d_{shape}_copy_out(void *handle, double *host_out) {{
    An5dHandle *h = (An5dHandle *)handle;
    int dimsize = h->dimsize;
    size_t nbytes = {malloc_expr} * sizeof(double);
    cudaCheckReturn(cudaMemcpy(host_out, h->dev_A, nbytes, cudaMemcpyDeviceToHost));
}}

extern "C" __attribute__((visibility("default")))
void an5d_{shape}_free(void *handle) {{
    An5dHandle *h = (An5dHandle *)handle;
    cudaCheckReturn(cudaFree(h->dev_A));
    delete h;
}}
'''


def main():
    for shape, cfg in SHAPES.items():
        host_path = os.path.join(SRC, f"{cfg['cu']}_host.cu")
        with open(host_path) as f:
            text = f.read()
        body = extract_dispatch(text, cfg["bt"])
        # indent the extracted body one level to sit inside an5d_<shape>_run()
        indented = "".join(("    " + l if l.strip() else l) for l in body.splitlines(keepends=True))
        out = BRIDGE_TEMPLATE.format(
            shape=shape, cu=cfg["cu"], bt=cfg["bt"], macros=_CUDA_CHECK_MACROS,
            malloc_expr=malloc_expr(cfg["dims"]), dispatch_body=indented)
        out_path = os.path.join(HERE, f"bridge_{shape}.cu")
        with open(out_path, "w") as f:
            f.write(out)
        print(f"generated {out_path} (bt={cfg['bt']}, dims={cfg['dims']}, "
              f"{len(body.splitlines())} dispatch lines from {cfg['cu']}_host.cu)")


if __name__ == "__main__":
    main()
