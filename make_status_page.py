"""Build output/benchmark_status.html: collect everything the status page needs
into one JSON blob and inject it into output/benchmark_status.src.html.

Sources (all inside the project tree, no filesystem search):
  kernel-papers/benchspecs/<k>/spec.yaml      problem, variants, inputs, gates
  bench/kernelbench/domains/*                 what is implemented, CPU/CUDA impls
  bench/artifacts/<k>/<short>/{adapter.py,STATUS.md}   baseline artifacts + outcome
  kernel-papers/output/benchmark_groups.json  papers per kernel group
  kernel-papers/output/included.json          paper metadata by dblp key
"""
import json
import os
import re
import sys

import yaml

KP = os.path.dirname(os.path.abspath(__file__))
BENCH = os.path.join(KP, "bench")
sys.path.insert(0, BENCH)
from kernelbench import spec as specmod, domains  # noqa: E402

included = {p["key"]: p for p in json.load(open(os.path.join(KP, "output/included.json")))}
groups_raw = json.load(open(os.path.join(KP, "output/benchmark_groups.json")))
groups = {}
for name, papers in groups_raw.items():
    groups[name.replace("other:", "")] = papers

status_all = domains.load_all()
owner = domains.OWNER

# optional per-(track, paper) kernel-centrality ratings (merge_centrality.py)
_cpath = os.path.join(KP, "output/kernel_centrality.json")
centrality = json.load(open(_cpath)) if os.path.exists(_cpath) else {}

# optional per-artifact H100 (sm_90) baseline outcomes, keyed "<kernel>/<short>"
# (make_h100_status.py, from the shared H100 campaign TSVs). See bench/H100_baselines.md.
_h100path = os.path.join(KP, "bench/h100_status.json")
h100_status = json.load(open(_h100path)) if os.path.exists(_h100path) else {}


def rating(k, key):
    r = centrality.get(f"{k}|{key}")
    return {"centrality": r["centrality"], "regime_match": r["regime_match"], "regime": r.get("regime", ""),
            "gpu_single_card": bool(r.get("gpu_single_card", False))} if r else {}


def kernel_status(k):
    mod = owner.get(k)
    if mod is None:
        return "no-domain", None
    if status_all.get(mod) != "ok":
        return "no-domain", mod
    d = domains.load(k)
    if k in getattr(d, "KERNELS", []):
        return "implemented", mod
    if k in getattr(d, "PLANNED", []):
        return "planned", mod
    return "no-domain", mod


def impl_names(k, mod):
    d = domains.load(k)
    cpu = sorted((getattr(d, "CPU_IMPLS", {}) or {}).get(k, {}).keys())
    try:
        cuda = sorted((d.cuda_impls() or {}).get(k, {}).keys())
    except Exception:
        cuda = []
    mode = (getattr(d, "CORRECTNESS_MODE", {}) or {}).get(k, "")
    ref = (getattr(d, "REFERENCE_NAME", {}) or {}).get(k, "")
    return cpu, cuda, mode, ref


FAIL_WORDS = ("fails", "fail ", "failed", "gate-fails", "gate blocked", "crash", "not gateable",
              "not independently", "does not pass", "rejected")


def classify(header):
    # only the CURRENT ruling counts; STATUS headers keep superseded rulings
    # after these markers ("Earlier outcome, kept for the record: ...")
    s = header.lower()
    for marker in ("earlier outcome", "original ruling", "kept for the record", "earlier header"):
        i = s.find(marker)
        if i > 0:
            s = s[:i]
    if "deferred-hardware" in s:
        return "deferred-hw"
    if "not attempted" in s:
        return "skipped"          # deferred/not attempted: ruled out for now, with the reason in the text
    if "build-failed" in s or "build failed" in s:
        return "build-failed"
    if "skip" in s:
        return "skipped"
    if "built" in s or "gated" in s:
        if any(w in s for w in FAIL_WORDS):
            return "gate-fails"
        if "gated" in s:
            return "gated"
        return "built"
    return "unknown"


# Outcomes settled by the batch reports where the STATUS.md wording is ambiguous
# to a keyword classifier ("gate FAILS on cant but 3/3 smoke valid", "gate
# blocked by a precision policy", ...). Key = "<kernel>/<short>".
OVERRIDES = {
    "cg-krylov/millefeuille": "gated",      # 3/3 smoke valid; honest non-convergence on cant
    "convolution/hidet": "gated",           # 2/3 pass; 1x1 conv rejected (TF32) -- documented
    "gemm/hexcute": "gated",                # passes at the spec's fp16 bound since 2026-09-05
    "gemm/moonpoly": "gated",               # fp32 gated; fp16 blocked
    "spmm/inferfast": "built",              # fp16 vs fp32 tolerance: blocked
    "spmv/diaq": "gated",                   # gated on the substitute matrix (API needs dense materialization)
    "stencil/spider": "gated",              # interior check passes; full-array boundary convention differs
    "lossless-compression/mans": "gate-fails",   # gate crashes inside the artifact kernel
    "spmm/generalsparse": "gate-fails",          # 0/9: reader discards values
    "spmm/smat": "gate-fails",                   # 0/9: B-operand stride bug
    "gemm/ftgemm": "gated",                      # passes at the spec's fp32 bound (2026-09-05)
    # pattern-only SpMM kernels: GATED on spmm-binary-adjacency-kernel (their real
    # regime, 2026-09-06); the general weighted-variant failure stays in the text
    "spmm/dtcspmm": "gated",
    "spmm/flashsparse": "gated",
    "spmm/generalsparse": "gated",
    # AN5D: generator toolchain unbuildable (clang<=3.8), but the paper's own
    # pre-generated CUDA is wrapped and passes the full-array fixed-halo gate
    # (1e-8, all 6 shapes) -- the header keeps "BUILD-FAILED" for the generator
    "stencil/an5d": "gated",
    # NM-SpMM: gated on the synthetic N:M workload its kernel is built for (2e-7);
    # the header's "unsupported/fails" words refer to general SuiteSparse inputs
    "spmm/nm-spmm": "gated",
    # Magicube: built, gate not attempted (16-bit packing convention unresolved) -> BUILT, not skipped
    "sddmm/magicube": "built",
}


def status_header(path):
    lines = open(path, encoding="utf-8", errors="replace").read().splitlines()
    # Style B: "# <name> — <kernel> — STATUS: BUILT+GATED (...)" on the H1 line
    for ln in lines[:3]:
        m = re.search(r"STATUS:\s*(.+)$", ln)
        if m and ln.lstrip().startswith("#"):
            return m.group(1).strip()
    buf = []
    started = False
    for ln in lines[:40]:
        if not started and re.search(r"\*\*(Status|Outcome|Result)", ln):
            started = True
        if started:
            if ln.strip() == "" and buf:
                break
            buf.append(ln.strip())
    text = " ".join(buf)
    text = re.sub(r"\*\*", "", text)
    text = re.sub(r"^(Status|Outcome|Result):\s*", "", text)
    return text.strip()


def adapter_fields(path):
    s = open(path, encoding="utf-8", errors="replace").read()
    out = {}
    for f in ("IMPL_NAME", "PAPER_KEY"):
        m = re.search(rf'^{f}\s*=\s*["\']([^"\']+)["\']', s, re.M)
        out[f] = m.group(1) if m else ""
    m = re.search(r"^PRECISIONS\s*=\s*\[([^\]]*)\]", s, re.M)
    out["PRECISIONS"] = re.findall(r'["\'](\w+)["\']', m.group(1)) if m else []
    m = re.search(r"^\s+platform\s*=\s*[\"']([a-z]+)[\"']", s, re.M)   # Implementation.platform ("cpu" | "cuda")
    out["platform"] = m.group(1) if m else ""
    return out


def paper_key_from_status(path):
    s = open(path, encoding="utf-8", errors="replace").read()
    m = re.search(r"\b((?:conf|journals)/[a-z]+/[A-Za-z0-9]+)", s)
    return m.group(1) if m else ""


def artifacts_for(k):
    kdir = os.path.join(BENCH, "artifacts", k)
    out = []
    if not os.path.isdir(kdir):
        return out
    for short in sorted(os.listdir(kdir)):
        d = os.path.join(kdir, short)
        if not os.path.isdir(d):
            continue
        ap = os.path.join(d, "adapter.py")
        st = os.path.join(d, "STATUS.md")
        row = {"short": short, "has_adapter": os.path.exists(ap), "has_status": os.path.exists(st)}
        if os.path.exists(ap):
            row.update(adapter_fields(ap))
        header = status_header(st) if os.path.exists(st) else ""
        row["status_text"] = header[:420]
        row["status"] = classify(header) if header else ("built" if os.path.exists(ap) else "unknown")
        row["status"] = OVERRIDES.get(f"{k}/{short}", row["status"])
        row["self_reported"] = "self-reported" in header.lower()
        h = h100_status.get(f"{k}/{short}")
        if h:
            row["h100"] = h
        req = os.path.join(d, "REQUIRES_GPU")
        if os.path.exists(req):
            row["requires_gpu"] = open(req).read().strip()[:200]
            row["status"] = "deferred-hw"
        key = row.get("PAPER_KEY") or (paper_key_from_status(st) if os.path.exists(st) else "")
        row["paper_key"] = key
        row.update(rating(k, key))
        p = included.get(key)
        if p:
            row["paper"] = {"title": p["title"], "venue": p["venue"], "year": p["year"],
                            "url": p.get("artifact_url", ""), "platform": p.get("platform", [])}
        prov = os.path.join(d, "source.provenance")
        if os.path.exists(prov):
            for ln in open(prov):
                if ln.startswith("commit="):
                    row["commit"] = ln.split("=", 1)[1].strip()[:12]
                if ln.startswith("remote="):
                    row["remote"] = ln.split("=", 1)[1].strip()
        out.append(row)
    return out


def variants_for(k):
    sp = specmod.load(k)
    raw = yaml.safe_load(open(os.path.join(KP, "benchspecs", k, "spec.yaml")))
    raw_by_id = {v.get("id"): v for v in raw.get("variants", []) if isinstance(v, dict)}
    vs = []
    for vid in sp.variant_ids:
        v = sp.variant(vid)
        rv = raw_by_id.get(vid, {})
        inputs = v.inputs if isinstance(v.inputs, dict) else {}
        corr = rv.get("correctness")
        if isinstance(corr, dict):
            corr = "; ".join(f"{a}: {b}" for a, b in corr.items())
        vs.append({
            "id": vid,
            "claim": v.claim,
            "inputs": {a: (b if isinstance(b, (list, str, int, float)) else str(b))
                       for a, b in inputs.items()},
            "recommended_subset": v.recommended_subset(),
            "protocol": {"warmup": v.protocol.warmup, "reps": v.protocol.reps,
                         "statistic": v.protocol.statistic, "timer": v.protocol.timer,
                         "timing_scope": v.protocol.timing_scope,
                         "preprocessing_reported": v.protocol.preprocessing_reported},
            "metric": v.metric if isinstance(v.metric, dict) else {"primary": str(v.metric)},
            "correctness": (corr or v.correctness_text or ""),
            "tolerance": v.tolerance,
            "tolerance_by_precision": getattr(v, "tolerance_by_precision", {}) or {},
            "tolerance_provenance": v.tolerance_provenance,
            "gate_kind": v.gate_kind,
        })
    return sp, raw, vs


tracks = []
for k in sorted(os.listdir(os.path.join(KP, "benchspecs"))):
    if not os.path.isdir(os.path.join(KP, "benchspecs", k)):
        continue
    st, mod = kernel_status(k)
    sp, raw, vs = variants_for(k)
    cpu, cuda, mode, ref = impl_names(k, mod) if st == "implemented" else ([], [], "", "")
    papers = groups.get(k, [])
    arts = artifacts_for(k)
    tracks.append({
        "kernel": k,
        "domain": mod or "",
        "status": st,
        "summary": raw.get("summary", "") or sp.summary,
        "operation": raw.get("operation", ""),
        "notes_on_fairness": raw.get("notes_on_fairness", "") or "",
        "variants": vs,
        "cpu_impls": cpu, "cuda_impls": cuda, "correctness_mode": mode, "reference_name": ref,
        "papers": [{"key": p["key"], "title": p["title"], "venue": p["venue"], "year": p["year"],
                    "platform": p.get("platform", []), "url": p.get("url", ""),
                    "one_liner": p.get("one_liner", ""), **rating(k, p["key"])} for p in
                   sorted(papers, key=lambda p: (-int(p["year"]), p["venue"]))],
        "artifacts": arts,
    })

stats = {
    "specs": len(tracks),
    "implemented": sum(t["status"] == "implemented" for t in tracks),
    "planned": sum(t["status"] == "planned" for t in tracks),
    "no_domain": sum(t["status"] == "no-domain" for t in tracks),
    "tracks_with_adapters": sum(any(a["has_adapter"] for a in t["artifacts"]) for t in tracks),
    "adapters": sum(a["has_adapter"] for t in tracks for a in t["artifacts"]),
    "artifact_dirs": sum(len(t["artifacts"]) for t in tracks),
    "gated": sum(a["status"] == "gated" for t in tracks for a in t["artifacts"]),
    "gate_fails": sum(a["status"] == "gate-fails" for t in tracks for a in t["artifacts"]),
    "built": sum(a["status"] == "built" for t in tracks for a in t["artifacts"]),
    "build_failed": sum(a["status"] == "build-failed" for t in tracks for a in t["artifacts"]),
    "skipped": sum(a["status"] == "skipped" for t in tracks for a in t["artifacts"]),
    "deferred_hw": sum(a["status"] == "deferred-hw" for t in tracks for a in t["artifacts"]),
    "papers_in_groups": sum(len(t["papers"]) for t in tracks),
    "variants": sum(len(t["variants"]) for t in tracks),
    # H100 (sm_90) baseline campaign roll-up
    "h100_total": len(h100_status),
    "h100_pass": sum(1 for v in h100_status.values() if v["cls"] in ("PASS", "PASS-r")),
    "h100_recompile": sum(1 for v in h100_status.values() if v["cls"] == "PASS-r"),
    "h100_regression": sum(1 for v in h100_status.values() if v["cls"] == "regression"),
    "h100_realdata": sum(1 for v in h100_status.values() if "real" in v),
    "corpus": 7029, "included": 917, "with_artifacts": 308, "groups": 91,
    "smoke": "35/35",
    "rated_pairs": len(centrality),
    "core_pairs": sum(1 for v in centrality.values() if v["centrality"] == "core"),
}
out = {"generated": "2026-09-05", "stats": stats, "tracks": tracks}
dst = sys.argv[1] if len(sys.argv) > 1 else os.path.join(KP, "output/benchmark_status.data.json")
json.dump(out, open(dst, "w"), ensure_ascii=False, indent=0)
# inject into the page template (the template has no doctype/html/body on purpose:
# the artifact viewer wraps it; browsers render the bare fragment fine too)
blob = json.dumps(out, ensure_ascii=False, separators=(",", ":")).replace("</", "<\\/")
tpl = open(os.path.join(KP, "output/benchmark_status.src.html"), encoding="utf-8").read()
assert tpl.count("/*__DATA__*/") == 1
html_path = os.path.join(KP, "output/benchmark_status.html")
open(html_path, "w", encoding="utf-8").write(tpl.replace("/*__DATA__*/", blob))
print("wrote", html_path, os.path.getsize(html_path) // 1024, "KB")
print(json.dumps(stats, ensure_ascii=False))
print("unknown-status artifacts:", [(t["kernel"], a["short"], a["status_text"][:60])
                                    for t in tracks for a in t["artifacts"] if a["status"] == "unknown"])
print("gate-fails:", [(t["kernel"], a["short"]) for t in tracks for a in t["artifacts"] if a["status"] == "gate-fails"])
print("built:", [(t["kernel"], a["short"]) for t in tracks for a in t["artifacts"] if a["status"] == "built"])
