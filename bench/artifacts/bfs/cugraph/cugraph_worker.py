"""
cuGraph BFS worker -- runs inside this directory's own venv (deps/venv, made
by build.sh), never inside the harness process, so RAPIDS' CUDA libraries
(rmm, libcugraph, cupy) cannot clash with torch's. The adapter talks to it
over stdin/stdout, one JSON object per line.

  {"cmd": "load", "dir": D, "directed": bool, "direction_optimizing": bool}
      reads D/indptr.npy and D/indices.npy, builds a pylibcugraph SGGraph
      from the CSR once (renumber=False, so vertex ids are unchanged).
  {"cmd": "bfs", "source": s, "levels_out": path-or-null}
      one pylibcugraph.bfs from s. "ms" = CUDA-event time around exactly
      that call plus a device synchronize (cuGraph's own C API call; the Python
      wrapper's has_vertex check and result copy are inside it, as for any
      cuGraph user). Writes the distance array to levels_out when given
      (outside the timed span).
  {"cmd": "quit"}
"""
import json
import sys

import cupy as cp
import numpy as np
import pylibcugraph as plc

INT32_MAX = np.iinfo(np.int32).max
INT64_MAX = np.iinfo(np.int64).max


def main():
    handle = plc.ResourceHandle()
    graph = None
    vdtype = np.int32
    do = False
    n = 0
    out = sys.stdout
    for line in sys.stdin:
        msg = json.loads(line)
        try:
            if msg["cmd"] == "load":
                indptr = np.load(f"{msg['dir']}/indptr.npy")
                indices = np.load(f"{msg['dir']}/indices.npy")
                vdtype = np.int32 if indices.dtype == np.int32 else np.int64
                n = int(indptr.shape[0] - 1)
                offsets_d = cp.asarray(indptr)
                indices_d = cp.asarray(indices)
                props = plc.GraphProperties(is_symmetric=not msg["directed"],
                                            is_multigraph=False)
                graph = plc.SGGraph(handle, props, offsets_d, indices_d,
                                    weight_array=None, store_transposed=False,
                                    renumber=False, do_expensive_check=False,
                                    input_array_format="CSR")
                del offsets_d, indices_d
                do = bool(msg["direction_optimizing"])
                cp.cuda.runtime.deviceSynchronize()
                reply = {"ok": True, "version": plc.__version__}
            elif msg["cmd"] == "bfs":
                src = cp.asarray([msg["source"]], dtype=vdtype)
                start, stop = cp.cuda.Event(), cp.cuda.Event()
                start.record()
                dist, _pred, verts = plc.bfs(handle, graph, src, do, -1, False, False)
                cp.cuda.runtime.deviceSynchronize()   # whatever stream cuGraph used
                stop.record()
                stop.synchronize()
                ms = cp.cuda.get_elapsed_time(start, stop)
                if msg.get("levels_out"):
                    d = cp.asnumpy(dist).astype(np.int64)
                    v = cp.asnumpy(verts).astype(np.int64)
                    level = np.full(n, -1, dtype=np.int64)
                    reached = (d != INT32_MAX) & (d != INT64_MAX) & (d >= 0)
                    level[v[reached]] = d[reached]
                    np.save(msg["levels_out"], level)
                reply = {"ok": True, "ms": float(ms)}
            elif msg["cmd"] == "quit":
                break
            else:
                reply = {"ok": False, "error": f"unknown cmd {msg['cmd']!r}"}
        except Exception as e:  # report, keep serving
            reply = {"ok": False, "error": f"{type(e).__name__}: {e}"}
        out.write(json.dumps(reply) + "\n")
        out.flush()


if __name__ == "__main__":
    main()
