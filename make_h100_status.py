#!/usr/bin/env python3
"""Aggregate the H100 baseline campaign into bench/h100_status.json (committed),
keyed by "<kernel>/<short>", for make_status_page.py to render an H100 column.

Source data lives in the group-shared campaign dir (not in the repo, since the
raw per-run TSVs are large / machine-specific):
  $SHARED_KB_DATA/../  ->  /scratch/zt1/project/bhatele-lab/shared/kernel-bench/
    h100_results.tsv          functional gates (tag in col 3 supersedes smoke)
    h100_phase2_timing.tsv    indicative throughput on smoke inputs (subset)
    h100_realdata_timing.tsv  throughput on REAL datasets (SuiteSparse/SNAP/SIFT1M/FROSTT)
Re-run after new H100 results land, then re-run make_status_page.py.
"""
import os, re, json

SH = os.environ.get("KB_SHARED", "/scratch/zt1/project/bhatele-lab/shared/kernel-bench")
KP = os.path.dirname(os.path.abspath(__file__))
ATTN = {"bytetransformer", "et", "flashattention-t", "pat", "metaattention"}

# log-confirmed corrections where the result line alone is ambiguous (mirrors
# gen_h100_doc.py OVERRIDE / FORCED).
FORCED = {
    "stencil/spider": ("PASS", "1/1 runs valid (2 unsupported)"),
    "stencil/convstencil": ("regression", "CUDA error 700: illegal memory access"),
    "sddmm/flashsparse": ("gate-fail", "ran sm_90; KNOWN residue-tile overflow (STATUS.md)"),
    "attention-kernel/flashattention-t": ("slowbuild", "sm_90 CUTLASS rebuild >2.5h (deferred)"),
    "stencil/lorastencil": ("unsupported", "needs star2d3r real workload (bespoke gate)"),
}

def tag_prio(t):
    if t == "FORCE-sm90": return 60
    if t.startswith("REALWL-sm90"): return 55
    return {"REAL-WORKLOAD-v3": 50, "REAL-WORKLOAD-v2": 45,
            "REAL-WORKLOAD": 40, "REALWL": 38}.get(t, 10)

def vscore(r):
    if re.search(r'[1-9]\d*/\d+ runs valid', r) or 'valid=True' in r: return 3
    if re.search(r'0/[1-9]', r): return 2
    if 'unsupported' in r: return 1
    return 0

def classify(short, tag, r):
    if short in FORCED: return FORCED[short][0]
    sm90 = tag == "FORCE-sm90" or tag.startswith("REALWL") or tag in ("REAL-WORKLOAD-v3", "REAL-WORKLOAD-v2")
    if re.search(r'[1-9]\d*/\d+ runs valid', r) or 'valid=True' in r:
        return "PASS-r" if sm90 else "PASS"
    if 'different layout' in r or 'InternalError' in r: return "regression"
    if 'no kernel image' in r.lower() or 'rc=255' in r: return "nki"
    if re.search(r'0/[1-9]', r): return "gate-fail"
    if re.search(r'0/0.*unsupported', r): return "unsupported"
    if 'rc=124' in r: return "timeout"
    if 'rc=' in r or 'NO_' in r: return "error"
    return "unknown"

def norm_short(s):
    return f"attention-kernel/{s}" if s in ATTN else s

# ---- functional class (best row per artifact by tag priority) ----
best = {}
fp = f"{SH}/h100_results.tsv"
if os.path.exists(fp):
    for l in open(fp):
        p = l.rstrip("\n").split("\t")
        if len(p) < 5: continue
        short, tag, r = norm_short(p[0]), p[2], re.sub(r' ->.*', '', p[4])
        key = (tag_prio(tag), vscore(r))
        if short not in best or key >= best[short][0]:
            best[short] = (key, tag, r)
for s, (r, e) in FORCED.items():
    best[s] = ((99, 99), "log", e)

def best_rate(text):
    m = re.findall(r'[0-9.]+ (?:TFLOP/s|GFLOP/s|GB/s|GCUP/s|QPS)', text)
    if not m: return ""
    # keep the max magnitude (rough: compare leading number)
    return sorted(m, key=lambda x: float(x.split()[0]), reverse=True)[0]

# ---- smoke throughput (phase 2) ----
smoke = {}
sp = f"{SH}/h100_phase2_timing.tsv"
if os.path.exists(sp):
    for l in open(sp):
        if l.startswith('#'): continue
        p = l.rstrip("\n").split("\t")
        if len(p) < 6: continue
        smoke[norm_short(p[0])] = p[5].strip()

# ---- real-dataset throughput ----
real = {}
rp = f"{SH}/h100_realdata_timing.tsv"
if os.path.exists(rp):
    for l in open(rp):
        if l.startswith('#'): continue
        p = l.rstrip("\n").split("\t")
        if len(p) < 6: continue
        short = norm_short(p[0]); mtx = p[3]; valid = re.sub(r' ->.*', '', p[4]); rate = p[5].strip()
        per = p[6] if len(p) > 6 else ""
        if not rate:
            mq = re.search(r'[0-9.]+ (?:QPS|Melem/s|Gedges/s)', per)
            if mq: rate = mq.group(0)
        gate = "pass" if re.search(r'[1-9]\d*/\d+ runs valid', valid) else "fail"
        real[short] = {"matrices": mtx, "rate": rate, "gate": gate, "raw": valid[:40]}

out = {}
for short, (k, tag, r) in best.items():
    cls = classify(short, tag, r)
    rec = {"cls": cls, "gate": r[:48]}
    if smoke.get(short): rec["smoke_tput"] = smoke[short]
    if short in real: rec["real"] = real[short]
    out[short] = rec

dst = os.path.join(KP, "bench", "h100_status.json")
json.dump(out, open(dst, "w"), ensure_ascii=False, indent=0, sort_keys=True)
n_pass = sum(1 for v in out.values() if v["cls"] in ("PASS", "PASS-r"))
print(f"wrote {dst}: {len(out)} artifacts, {n_pass} PASS, {len(real)} with real-data throughput")
from collections import Counter
print(dict(Counter(v["cls"] for v in out.values())))
