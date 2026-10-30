"""
cuGraph BFS -- the GPU library BASELINE for the bfs track (ROLE = "baseline").

efg (IPDPS'23) compares against cuGraph's uncompressed-CSR BFS: on graphs
that fit in GPU memory cuGraph is faster (efg reaches 0.82x of it); on
graphs that don't fit, efg is 3.8x faster than out-of-core CSR. BLEST does
not use cuGraph, but it is the vendor reference for both papers here.

How it runs: pylibcugraph (RAPIDS 25.06, build.sh) in a separate worker
process (cugraph_worker.py, this directory's own venv), so RAPIDS' CUDA
libraries never load into the harness process next to torch.
  - prepare(): writes the CSR to a temp dir, starts the worker, which builds
    a cuGraph SGGraph once (renumber=False). Not timed.
  - run(): asks the worker for one BFS from the next of the SAME 64
    fixed-seed roots blest/efg/gunrock/gap use. The harness times run() with
    a wall clock, which includes the pipe round trip (~0.1 ms); the
    worker's own CUDA-event time around the cuGraph call is recorded per
    call in params["cugraph_internal_ms"] -- use that one for comparisons.
  - Output: distances = BFS levels (fetched only for the gate call),
    checked exactly against the reference.

Direction optimization: OFF by default -- the default of cugraph.bfs, i.e.
what efg compared against. KB_CUGRAPH_DO=1 turns it on for undirected
(symmetric) graphs, the only ones cuGraph allows it for. Disclosed in params.
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import tempfile

import numpy as np

from kernelbench.domains.graph import _bfs_roots
from kernelbench.harness import Timer

KERNEL = "bfs"
IMPL_NAME = "cugraph-bfs"
PAPER_KEY = "conf/ipps/GeraK23"   # efg: compared against cuGraph CSR BFS
ROLE = "baseline"
PRECISIONS = ["int64"]

_HERE = os.path.dirname(os.path.abspath(__file__))
_VENV_PY = os.path.join(_HERE, "deps", "venv", "bin", "python")
_WORKER = os.path.join(_HERE, "cugraph_worker.py")


def available() -> tuple[bool, str]:
    try:
        if not os.path.exists(_VENV_PY):
            return False, f"not built: {_VENV_PY} missing (run build.sh)"
        return True, ""
    except Exception as e:  # available() must never raise
        return False, f"{type(e).__name__}: {e}"


class CugraphBFS:
    name = IMPL_NAME
    platform = "cuda"

    def __init__(self, precision: str = "int64"):
        if precision != "int64":
            raise NotImplementedError(
                f"{IMPL_NAME} only produces an exact integer level array; requested {precision}")
        self.precision = precision
        self.want_do = os.environ.get("KB_CUGRAPH_DO", "0") == "1"

    @staticmethod
    def _ask(proc, msg: dict) -> dict:
        proc.stdin.write(json.dumps(msg) + "\n")
        proc.stdin.flush()
        line = proc.stdout.readline()
        if not line:
            raise RuntimeError(f"{IMPL_NAME}: worker exited (exit code {proc.poll()})")
        reply = json.loads(line)
        if not reply.get("ok"):
            raise RuntimeError(f"{IMPL_NAME}: worker error: {reply.get('error')}")
        return reply

    def prepare(self, graph, params: dict):
        A = graph.csr
        n = A.shape[0]
        nnz = int(A.nnz)
        small = nnz < 2**31 and n < 2**31
        idt = np.int32 if small else np.int64
        tmpdir = tempfile.mkdtemp(prefix="cugraph_csr_")
        np.save(os.path.join(tmpdir, "indptr.npy"), np.ascontiguousarray(A.indptr, dtype=idt))
        np.save(os.path.join(tmpdir, "indices.npy"), np.ascontiguousarray(A.indices, dtype=idt))

        seed = int(params.get("bfs_root_seed", 0))
        k = int(params.get("num_roots", 64))
        roots, comp_edges, comp_verts = _bfs_roots(A, graph.directed, seed, k)
        do = self.want_do and not graph.directed
        params["bfs_root_seed"] = seed
        params["num_roots"] = int(len(roots))
        params["giant_component_vertices"] = comp_verts
        params["giant_component_edges"] = comp_edges
        params["direction_optimized"] = do
        params["cugraph_internal_ms"] = []

        proc = subprocess.Popen([_VENV_PY, "-u", _WORKER], stdin=subprocess.PIPE,
                                stdout=subprocess.PIPE, text=True, cwd=tmpdir)
        h = {"proc": proc, "tmpdir": tmpdir, "n": n, "roots": roots, "cursor": 0,
             "params": params, "levels": None}
        try:
            reply = self._ask(proc, {"cmd": "load", "dir": tmpdir,
                                     "directed": bool(graph.directed),
                                     "direction_optimizing": do})
        except Exception:
            self.free(h)
            raise
        params["cugraph_version"] = reply.get("version")
        params["cugraph_id_dtype"] = np.dtype(idt).name
        return h

    def run(self, h):
        src = int(h["roots"][h["cursor"] % len(h["roots"])])
        first = h["cursor"] == 0          # the harness gates on the first call only
        h["cursor"] += 1
        out = os.path.join(h["tmpdir"], "levels.npy") if first else None
        reply = self._ask(h["proc"], {"cmd": "bfs", "source": src, "levels_out": out})
        h["params"]["cugraph_internal_ms"].append(reply["ms"])
        if first:
            h["levels"] = np.load(out)
        return h

    def to_host(self, h):
        if h.get("levels") is None:
            raise RuntimeError(f"{IMPL_NAME}: no levels captured (gate must be the first run)")
        return np.asarray(h["levels"], dtype=np.int64)

    def timer(self):
        return Timer()

    def free(self, h):
        proc = h.get("proc")
        if proc is not None and proc.poll() is None:
            try:
                proc.stdin.write(json.dumps({"cmd": "quit"}) + "\n")
                proc.stdin.flush()
                proc.wait(timeout=30)
            except Exception:
                proc.kill()
        if h.get("tmpdir"):
            shutil.rmtree(h["tmpdir"], ignore_errors=True)
        h.clear()


def create(precision: str):
    return CugraphBFS(precision)
