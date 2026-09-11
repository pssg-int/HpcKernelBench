#!/usr/bin/env python3
"""
Build LLM-judge input for artifact matches that lack textual cross-reference:
all papers whose verified status is 'likely' or 'ambiguous'. Fetch each
candidate repo's README head + metadata, chunk ~35 papers per file into
data/judge_chunks/jchunk_NN.json. Judges write data/judged/jchunk_NN.json.
"""

import json
import os
import re
import subprocess
import time

import requests

HERE = os.path.dirname(os.path.abspath(__file__))
session = requests.Session()
TOKEN = subprocess.run(["gh", "auth", "token"], capture_output=True,
                       text=True).stdout.strip()
HDR = {"Authorization": f"Bearer {TOKEN}"}


def gh(url, raw=False):
    h = dict(HDR)
    h["Accept"] = "application/vnd.github.raw+json" if raw else "application/vnd.github+json"
    for t in range(3):
        try:
            r = session.get(url, headers=h, timeout=45)
            if r.status_code == 403:
                time.sleep(15)
                continue
            if r.status_code == 404:
                return None
            r.raise_for_status()
            return r.text if raw else r.json()
        except Exception:
            time.sleep(5)
    return None


def main():
    with open(os.path.join(HERE, "data", "artifacts_verified.json")) as f:
        ver = json.load(f)
    with open(os.path.join(HERE, "data", "artifacts_found.json")) as f:
        found = json.load(f)
    with open(os.path.join(HERE, "output", "included.json")) as f:
        papers = {p["key"]: p for p in json.load(f)}

    import glob
    already = set()
    for jf in glob.glob(os.path.join(HERE, "data", "judged", "jchunk_*.json")):
        with open(jf) as f:
            for j in json.load(f):
                already.add(j["key"])

    items = []
    for key, v in ver.items():
        if v["status"] not in ("likely", "ambiguous") or key not in papers \
           or key in already:
            continue
        p = papers[key]
        cands = []
        urls = [v["url"]] if v.get("url") else \
               [c["url"] for c in found.get(key, {}).get("candidates", [])][:3]
        for u in urls:
            m = re.match(r"https://github\.com/([^/]+)/([^/]+)", u)
            if not m:
                continue
            full = f"{m.group(1)}/{m.group(2)}"
            readme = gh(f"https://api.github.com/repos/{full}/readme", raw=True) or ""
            meta = gh(f"https://api.github.com/repos/{full}") or {}
            cands.append({"url": u, "desc": meta.get("description") or "",
                          "stars": meta.get("stargazers_count", 0),
                          "readme_head": readme[:1500]})
            time.sleep(0.15)
        if cands:
            items.append({"key": key, "title": p["title"], "venue": p["venue"],
                          "year": p["year"], "authors": p["authors"][:5],
                          "abstract": p["abstract"][:700], "candidates": cands})

    outdir = os.path.join(HERE, "data", "judge_chunks")
    os.makedirs(outdir, exist_ok=True)
    os.makedirs(os.path.join(HERE, "data", "judged"), exist_ok=True)
    start = len(glob.glob(os.path.join(outdir, "jchunk_*.json")))
    CH = 35
    n = 0
    for i in range(0, len(items), CH):
        with open(os.path.join(outdir, f"jchunk_{start + i//CH:02d}.json"), "w") as f:
            json.dump(items[i:i + CH], f, ensure_ascii=False, indent=1)
        n += 1
    print(f"{len(items)} papers -> chunks jchunk_{start:02d}..jchunk_{start+n-1:02d}")


if __name__ == "__main__":
    main()
