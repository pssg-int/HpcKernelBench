#!/usr/bin/env python3
"""
Verify artifact candidates in data/artifacts_found.json via the GitHub API.

For each candidate github repo: repo exists, is not empty, language stats,
stars, license, last push; fetch README and cross-check against the paper
(title fragment / system name / author last name / venue+year).

Decides per paper:
  verified  — README(or repo) and paper cross-reference each other, or the URL
              came from the paper itself (abstract/arxiv-fulltext/crossref/pwc)
              and the repo is alive with code
  likely    — repo alive + name/topic match but no textual cross-reference
  ambiguous — candidates exist but none pass; goes to the LLM judge
  none      — no candidates
Writes data/artifacts_verified.json {key: {status, url, stars, license,
last_push, language, evidence, source}} plus data/artifacts_ambiguous.json
for the agent pass.
"""

import json
import os
import re
import subprocess
import time

import requests

HERE = os.path.dirname(os.path.abspath(__file__))
FOUND = os.path.join(HERE, "data", "artifacts_found.json")
OUT = os.path.join(HERE, "data", "artifacts_verified.json")
AMB = os.path.join(HERE, "data", "artifacts_ambiguous.json")

session = requests.Session()
TOKEN = subprocess.run(["gh", "auth", "token"], capture_output=True,
                       text=True).stdout.strip()
HDR = {"Authorization": f"Bearer {TOKEN}", "Accept": "application/vnd.github+json"}
# sources trusted to point at the paper's own artifact. Fulltext-extracted URLs
# are NOT trusted: papers cite baseline repos (transformers, zstd...) in
# references, which the regex also captures.
PAPER_SOURCES = {"abstract", "crossref", "pwc"}
FULLTEXT_SOURCES = {"arxiv-fulltext", "oa-fulltext"}


def gh(url, params=None, raw=False):
    for t in range(3):
        try:
            r = session.get(url, params=params,
                            headers=dict(HDR, **({"Accept": "application/vnd.github.raw+json"}
                                                 if raw else {})), timeout=45)
            if r.status_code == 403 and r.headers.get("X-RateLimit-Remaining") == "0":
                wait = int(r.headers.get("X-RateLimit-Reset", 0)) - int(time.time())
                time.sleep(max(5, min(wait + 2, 300)))
                continue
            if r.status_code in (404, 451):
                return None
            r.raise_for_status()
            return r.text if raw else r.json()
        except Exception:
            time.sleep(5 * (t + 1))
    return None


NAME_RE = re.compile(r"^([A-Za-z0-9+\-]{2,25}):\s")


def title_fragment(title, n=6):
    words = re.sub(r"[^A-Za-z0-9 ]", " ", title).split()
    return " ".join(words[:n]).lower()


def check_repo(url):
    m = re.match(r"https://github\.com/([^/]+)/([^/]+)", url)
    if not m:
        return None                     # zenodo/gitlab: keep, verified-lite below
    full = f"{m.group(1)}/{m.group(2)}"
    j = gh(f"https://api.github.com/repos/{full}")
    if not j or j.get("size", 0) == 0:
        return None
    langs = gh(f"https://api.github.com/repos/{full}/languages") or {}
    readme = gh(f"https://api.github.com/repos/{full}/readme", raw=True) or ""
    return {
        "full": j.get("full_name", full),
        "url": j.get("html_url", url),
        "stars": j.get("stargazers_count", 0),
        "license": (j.get("license") or {}).get("spdx_id") or "",
        "last_push": (j.get("pushed_at") or "")[:10],
        "language": ", ".join(list(langs)[:3]),
        "archived": j.get("archived", False),
        "readme": readme[:20000],
        "desc": j.get("description") or "",
    }


def readme_matches(info, paper):
    text = (info["readme"] + " " + info["desc"]).lower()
    if title_fragment(paper["title"]) in text:
        return "title-in-readme"
    m = NAME_RE.match(paper["title"])
    if m and m.group(1).lower() in info["full"].split("/")[1].lower():
        # name match alone is 'likely'; combined with venue/year or author => verified
        vy = f"{paper['venue'].lower()}"
        yr = str(paper["year"])
        authors = [a.split()[-1].lower() for a in paper["authors"][:4] if a.split()]
        if (vy in text and yr in text) or any(a in text for a in authors if len(a) > 3):
            return "sysname+cite"
        return "sysname-only"
    doi = paper.get("doi", "")
    if doi and doi.lower() in text:
        return "doi-in-readme"
    return None


def main():
    with open(FOUND) as f:
        found = json.load(f)
    with open(os.path.join(HERE, "output", "included.json")) as f:
        papers = {p["key"]: p for p in json.load(f)}
    done = {}
    if os.path.exists(OUT):
        with open(OUT) as f:
            done = json.load(f)

    ambiguous = []
    n = 0
    for key, e in found.items():
        if key not in papers:
            continue
        # skip if already decided, unless it was 'none' and new candidates appeared
        if key in done and not (done[key]["status"] == "none" and e.get("candidates")):
            continue
        p = papers[key]
        cands = e.get("candidates", [])
        if not cands:
            done[key] = {"status": "none"}
            continue
        best = None
        amb_cands = []
        for c in cands:
            if "github.com" in c["url"]:
                # paper-list/survey repos quote titles without being the artifact
                if c["source"] == "github-readme" and re.search(
                        r"awesome|paper|survey|reading|bibliograph|collection",
                        c["url"].lower()):
                    continue
                info = check_repo(c["url"])
                if not info:
                    continue
                match = readme_matches(info, p)
                from_paper = c["source"] in PAPER_SOURCES
                if c["source"] == "github-readme":
                    # found BY readme text: title match is circular evidence.
                    # Only author/doi-grade signals verify; the rest go to the judge.
                    if match in ("sysname+cite", "doi-in-readme"):
                        status = "verified"
                    elif info["language"]:
                        amb_cands.append({**c, "desc": info["desc"],
                                          "readme_head": info["readme"][:1200],
                                          "stars": info["stars"]})
                        continue
                    else:
                        continue
                elif c["source"] in FULLTEXT_SOURCES:
                    # URL appeared in the paper's fulltext — could be its artifact
                    # OR a cited baseline. Famous repos with no README link to the
                    # paper are cited baselines; the rest need README match or judge.
                    if match in ("title-in-readme", "sysname+cite", "doi-in-readme"):
                        status = "verified"
                    elif info["stars"] > 2000:
                        continue          # cited baseline, not this paper's artifact
                    elif info["language"]:
                        amb_cands.append({**c, "desc": info["desc"],
                                          "readme_head": info["readme"][:1200],
                                          "stars": info["stars"]})
                        continue
                    else:
                        continue
                elif from_paper or match in ("title-in-readme", "sysname+cite", "doi-in-readme"):
                    status = "verified"
                elif match == "sysname-only":
                    status = "likely"
                else:
                    amb_cands.append({**c, "desc": info["desc"],
                                      "readme_head": info["readme"][:1200],
                                      "stars": info["stars"]})
                    continue
                cand = {"status": status, "url": info["url"], "stars": info["stars"],
                        "license": info["license"], "last_push": info["last_push"],
                        "language": info["language"], "source": c["source"],
                        "evidence": match or c["source"]}
                if best is None or (cand["status"] == "verified" and best["status"] != "verified") \
                   or (cand["status"] == best["status"] and cand["stars"] > best["stars"]):
                    best = cand
            else:
                # zenodo / gitlab etc. from the paper itself -> verified-lite
                if c["source"] in PAPER_SOURCES and best is None:
                    best = {"status": "verified", "url": c["url"], "stars": 0,
                            "license": "", "last_push": "", "language": "",
                            "source": c["source"], "evidence": c["source"]}
        if best:
            done[key] = best
        elif amb_cands:
            ambiguous.append({"key": key, "title": p["title"], "venue": p["venue"],
                              "year": p["year"], "abstract": p["abstract"][:800],
                              "candidates": amb_cands[:3]})
            done[key] = {"status": "ambiguous"}
        else:
            done[key] = {"status": "none"}
        n += 1
        if n % 50 == 0:
            with open(OUT, "w") as f:
                json.dump(done, f)
            print(f"  {n} verified, ambiguous so far {len(ambiguous)}")

    with open(OUT, "w") as f:
        json.dump(done, f, ensure_ascii=False)
    with open(AMB, "w") as f:
        json.dump(ambiguous, f, ensure_ascii=False, indent=1)

    from collections import Counter
    print("status counts:", Counter(v["status"] for v in done.values()).most_common())
    print(f"ambiguous for agent judge: {len(ambiguous)}")


if __name__ == "__main__":
    main()
