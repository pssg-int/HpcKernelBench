#!/usr/bin/env python3
"""
Merge artifact findings into output/included.json (and papers.json), then the
caller re-runs build_table.py + make_html.py.

Precedence per paper:
  1. LLM-judge decision (data/judged/jchunk_*.json) — overrides for the
     likely/ambiguous set it re-checked ('no' clears the artifact)
  2. data/artifacts_verified.json (verified/likely from scripted checks)

Adds fields: artifact_status (verified|likely|none), artifact_url,
artifact_source, artifact_stars, artifact_license, artifact_last_push.
"""

import glob
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "output")


def main():
    with open(os.path.join(OUT, "papers.json")) as f:
        papers = json.load(f)
    with open(os.path.join(HERE, "data", "artifacts_verified.json")) as f:
        ver = json.load(f)

    judged = {}
    for jf in glob.glob(os.path.join(HERE, "data", "judged", "jchunk_*.json")):
        with open(jf) as f:
            for v in json.load(f):
                judged[v["key"]] = v

    n_v = n_l = 0
    for p in papers:
        if not p["is_kernel_opt"]:
            continue
        key = p["key"]
        v = ver.get(key, {"status": "none"})
        status, url, source = v.get("status", "none"), v.get("url"), v.get("source", "")
        j = judged.get(key)
        if j is not None:
            if j["decision"] == "no":
                status, url = "none", None
            else:
                status, url = j["decision"], j["url"] or url
                source = source or "github-search"
        if status == "ambiguous":
            status = "none"
        p["artifact_status"] = status
        p["artifact_url"] = url or ""
        p["artifact_source"] = source if status != "none" else ""
        p["artifact_stars"] = v.get("stars", 0) if status != "none" else 0
        p["artifact_license"] = v.get("license", "") if status != "none" else ""
        p["artifact_last_push"] = v.get("last_push", "") if status != "none" else ""
        if status == "verified":
            n_v += 1
        elif status == "likely":
            n_l += 1

    with open(os.path.join(OUT, "papers.json"), "w") as f:
        json.dump(papers, f, ensure_ascii=False, indent=1)
    included = [p for p in papers if p["is_kernel_opt"]]
    with open(os.path.join(OUT, "included.json"), "w") as f:
        json.dump(included, f, ensure_ascii=False, indent=1)

    print(f"artifacts: {n_v} verified, {n_l} likely, "
          f"{len(included) - n_v - n_l} none / {len(included)} papers")


if __name__ == "__main__":
    main()
