#!/usr/bin/env python3
"""
Apply verify-pass corrections (data/verified/vchunk_*.json) onto output/papers.json,
then rewrite output/papers.json and output/included.json.

Verified verdicts REPLACE the originals for the re-checked keys; confirmed ones
get their (possibly upgraded) confidence. Prints a flip report.
"""

import glob
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "output")

VALID_CATS = {"dense_la", "sparse_la", "stencil_pde", "fft_spectral", "tensor",
              "graph", "primitives", "nbody_md", "unstructured_amr",
              "solver_components", "compression", "ml_kernels"}


def main():
    with open(os.path.join(OUT, "papers.json")) as f:
        papers = json.load(f)
    by_key = {p["key"]: p for p in papers}

    n_seen = n_flip_in = n_flip_out = 0
    for vf in sorted(glob.glob(os.path.join(HERE, "data", "verified", "vchunk_*.json"))):
        with open(vf) as f:
            arr = json.load(f)
        for v in arr:
            p = by_key.get(v.get("key"))
            if p is None:
                print(f"  WARN unknown key {v.get('key')} in {os.path.basename(vf)}")
                continue
            n_seen += 1
            new_inc = bool(v.get("is_kernel_opt"))
            if new_inc and not p["is_kernel_opt"]:
                n_flip_in += 1
            elif not new_inc and p["is_kernel_opt"]:
                n_flip_out += 1
            p["is_kernel_opt"] = new_inc
            p["confidence"] = v.get("confidence", p["confidence"])
            p["categories"] = [c for c in v.get("categories", []) if c in VALID_CATS]
            p["kernels"] = v.get("kernels", [])
            p["platform"] = v.get("platform", [])
            p["approach"] = v.get("approach")
            p["excluded_reason"] = v.get("excluded_reason")
            p["one_liner"] = v.get("one_liner", p["one_liner"])
            p["verified"] = True

    with open(os.path.join(OUT, "papers.json"), "w") as f:
        json.dump(papers, f, ensure_ascii=False, indent=1)
    included = [p for p in papers if p["is_kernel_opt"]]
    with open(os.path.join(OUT, "included.json"), "w") as f:
        json.dump(included, f, ensure_ascii=False, indent=1)

    print(f"verified {n_seen} papers: {n_flip_out} include->exclude, "
          f"{n_flip_in} exclude->include")
    print(f"final included: {len(included)} / {len(papers)}")


if __name__ == "__main__":
    main()
