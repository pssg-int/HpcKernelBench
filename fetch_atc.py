#!/usr/bin/env python3
"""
Fill abstracts for papers still missing one (mostly USENIX ATC, which has no DOI).

Round 1: Semantic Scholar /paper/search/match by title.
Round 2: scrape the USENIX presentation page (dblp <ee> URL) for the abstract.

Updates data/abstracts.json in place.
"""

import json
import os
import re
import sys
import time

import requests

HERE = os.path.dirname(os.path.abspath(__file__))
CORPUS = os.path.join(HERE, "data", "corpus.json")
OUT = os.path.join(HERE, "data", "abstracts.json")

session = requests.Session()
session.headers["User-Agent"] = "hpc-kernelbench-survey (mailto:weicunyang@gmail.com)"


def save(results):
    tmp = OUT + ".tmp"
    with open(tmp, "w") as f:
        json.dump(results, f, ensure_ascii=False)
    os.replace(tmp, OUT)


def s2_match(papers, results):
    todo = [p for p in papers
            if not results.get(p["key"], {}).get("abstract")]
    print(f"S2 title-match: {len(todo)} papers")
    hits = 0
    for n, p in enumerate(todo, 1):
        for t in range(3):
            try:
                r = session.get(
                    "https://api.semanticscholar.org/graph/v1/paper/search/match",
                    params={"query": p["title"],
                            "fields": "title,abstract,citationCount,openAccessPdf"},
                    timeout=30,
                )
                if r.status_code == 429:
                    time.sleep(6 * (t + 1))
                    continue
                if r.status_code == 404:   # no match found
                    break
                r.raise_for_status()
                data = r.json().get("data", [])
                if data:
                    w = data[0]
                    wt = (w.get("title") or "").lower().strip().rstrip(".")
                    if wt == p["title"].lower().strip() and w.get("abstract"):
                        prev = results.get(p["key"], {})
                        results[p["key"]] = {
                            "abstract": w["abstract"],
                            "cited_by": prev.get("cited_by") or w.get("citationCount", 0) or 0,
                            "oa_url": prev.get("oa_url") or
                                      ((w.get("openAccessPdf") or {}).get("url") or ""),
                            "source": "s2_match",
                        }
                        hits += 1
                break
            except Exception:
                time.sleep(3 * (t + 1))
        if n % 50 == 0:
            save(results)
            print(f"  {n}/{len(todo)}, +{hits} abstracts")
        time.sleep(1.1)
    save(results)
    print(f"S2 title-match done: +{hits}")


ABS_RE = re.compile(
    r'field-name-field-paper-description[^>]*>.*?<div class="field-item[^>]*">(.*?)</div>',
    re.S)
TAG_RE = re.compile(r"<[^>]+>")


def usenix_scrape(papers, results, ee_map):
    todo = [p for p in papers if p["venue"] == "ATC"
            and not results.get(p["key"], {}).get("abstract")
            and ee_map.get(p["key"], "").startswith("https://www.usenix.org")]
    print(f"USENIX scrape: {len(todo)} papers")
    hits = 0
    for n, p in enumerate(todo, 1):
        try:
            r = session.get(ee_map[p["key"]], timeout=30)
            if r.status_code == 200:
                m = ABS_RE.search(r.text)
                if m:
                    text = TAG_RE.sub(" ", m.group(1))
                    text = re.sub(r"\s+", " ", text).strip()
                    if len(text) > 100:
                        prev = results.get(p["key"], {})
                        results[p["key"]] = {
                            "abstract": text,
                            "cited_by": prev.get("cited_by", 0),
                            "oa_url": prev.get("oa_url", ""),
                            "source": "usenix",
                        }
                        hits += 1
        except Exception as e:
            print(f"  fail {p['key']}: {e}", file=sys.stderr)
        if n % 50 == 0:
            save(results)
            print(f"  {n}/{len(todo)}, +{hits} abstracts")
        time.sleep(0.4)
    save(results)
    print(f"USENIX scrape done: +{hits}")


def main():
    with open(CORPUS) as f:
        papers = json.load(f)
    with open(OUT) as f:
        results = json.load(f)

    missing = [p for p in papers if not results.get(p["key"], {}).get("abstract")]
    from collections import Counter
    print("missing per venue:", Counter(p["venue"] for p in missing).most_common())

    s2_match(missing, results)

    # dblp ee URLs for ATC: pull from raw XML once (corpus.json only kept DOIs)
    ee_map = {}
    atc_missing = [p for p in papers if p["venue"] == "ATC"
                   and not results.get(p["key"], {}).get("abstract")]
    if atc_missing:
        from lxml import etree
        dblp = os.path.join(HERE, "..", "HPCs", "hpc-first-authors", "data", "dblp.xml")
        wanted = {p["key"] for p in atc_missing}
        context = etree.iterparse(dblp, events=("end",), tag="inproceedings",
                                  load_dtd=True, resolve_entities=True, huge_tree=True)
        for _, elem in context:
            if elem.get("key") in wanted:
                for ee in elem.findall("ee"):
                    t = "".join(ee.itertext()).strip()
                    if "usenix.org" in t:
                        ee_map[elem.get("key")] = t
                        break
            elem.clear()
            while elem.getprevious() is not None:
                del elem.getparent()[0]
        print(f"resolved {len(ee_map)} usenix URLs")
        usenix_scrape(papers, results, ee_map)

    n_abs = sum(1 for v in results.values() if v.get("abstract"))
    print(f"\nTotal with abstract: {n_abs}/{len(papers)}")


if __name__ == "__main__":
    main()
