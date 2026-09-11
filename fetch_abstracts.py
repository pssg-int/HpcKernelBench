#!/usr/bin/env python3
"""
Fetch abstracts (+ citation counts, OA links) for the corpus via OpenAlex.

Primary: OpenAlex batch filter (50 DOIs per request, abstract_inverted_index).
Fallback for misses: Semantic Scholar batch API (may be publisher-elided for ACM),
then Crossref (has abstracts for some IEEE/ACM records).

Output: data/abstracts.json  {dblp_key: {abstract, cited_by, oa_url, source}}
Resumable: reruns skip keys already present in the output file.
"""

import json
import os
import sys
import time

import requests

HERE = os.path.dirname(os.path.abspath(__file__))
CORPUS = os.path.join(HERE, "data", "corpus.json")
OUT = os.path.join(HERE, "data", "abstracts.json")
MAILTO = "weicunyang@gmail.com"
BATCH = 50

session = requests.Session()
session.headers["User-Agent"] = f"hpc-kernelbench-survey (mailto:{MAILTO})"


def deinvert(inv):
    """Rebuild plain text from an OpenAlex abstract_inverted_index."""
    if not inv:
        return ""
    pos = []
    for word, idxs in inv.items():
        for i in idxs:
            pos.append((i, word))
    pos.sort()
    return " ".join(w for _, w in pos)


def get_json(url, params=None, tries=4):
    for t in range(tries):
        try:
            r = session.get(url, params=params, timeout=60)
            if r.status_code == 429:
                time.sleep(5 * (t + 1))
                continue
            if r.status_code == 404:
                return None
            r.raise_for_status()
            return r.json()
        except Exception as e:
            if t == tries - 1:
                print(f"  FAIL {url}: {e}", file=sys.stderr)
                return None
            time.sleep(3 * (t + 1))
    return None


def fetch_openalex(papers, results):
    todo = [p for p in papers if p["doi"] and p["key"] not in results]
    print(f"OpenAlex: fetching {len(todo)} papers in batches of {BATCH}")
    doi2key = {p["doi"].lower(): p["key"] for p in todo}
    dois = list(doi2key.keys())
    for i in range(0, len(dois), BATCH):
        chunk = dois[i:i + BATCH]
        j = get_json(
            "https://api.openalex.org/works",
            params={
                "filter": "doi:" + "|".join(chunk),
                "select": "doi,abstract_inverted_index,cited_by_count,open_access",
                "per-page": str(BATCH),
                "mailto": MAILTO,
            },
        )
        if not j:
            continue
        for w in j.get("results", []):
            doi = (w.get("doi") or "").replace("https://doi.org/", "").lower()
            key = doi2key.get(doi)
            if not key:
                continue
            abstract = deinvert(w.get("abstract_inverted_index"))
            results[key] = {
                "abstract": abstract,
                "cited_by": w.get("cited_by_count", 0),
                "oa_url": (w.get("open_access") or {}).get("oa_url") or "",
                "source": "openalex" if abstract else "openalex_noabs",
            }
        done = min(i + BATCH, len(dois))
        if done % 500 < BATCH or done == len(dois):
            n_abs = sum(1 for v in results.values() if v["abstract"])
            print(f"  {done}/{len(dois)} queried, {n_abs} abstracts so far")
            save(results)
        time.sleep(0.15)


def fetch_s2(papers, results):
    """Semantic Scholar batch fallback for papers still missing an abstract."""
    todo = [p for p in papers if p["doi"] and
            (p["key"] not in results or not results[p["key"]]["abstract"])]
    if not todo:
        return
    print(f"S2 fallback: {len(todo)} papers")
    for i in range(0, len(todo), 100):
        chunk = todo[i:i + 100]
        try:
            r = session.post(
                "https://api.semanticscholar.org/graph/v1/paper/batch",
                params={"fields": "abstract,citationCount,openAccessPdf"},
                json={"ids": [f"DOI:{p['doi']}" for p in chunk]},
                timeout=90,
            )
            if r.status_code == 429:
                time.sleep(10)
                continue
            r.raise_for_status()
            data = r.json()
        except Exception as e:
            print(f"  S2 batch failed: {e}", file=sys.stderr)
            time.sleep(5)
            continue
        for p, w in zip(chunk, data):
            if not w or not w.get("abstract"):
                continue
            prev = results.get(p["key"], {})
            results[p["key"]] = {
                "abstract": w["abstract"],
                "cited_by": prev.get("cited_by", w.get("citationCount", 0) or 0),
                "oa_url": prev.get("oa_url") or
                          ((w.get("openAccessPdf") or {}).get("url") or ""),
                "source": "s2",
            }
        save(results)
        n_abs = sum(1 for v in results.values() if v["abstract"])
        print(f"  S2 {min(i+100, len(todo))}/{len(todo)}, total abstracts {n_abs}")
        time.sleep(1.2)


def fetch_openalex_by_title(papers, results):
    """Per-paper title search for papers without a DOI (USENIX ATC)."""
    todo = [p for p in papers if not p["doi"] and p["key"] not in results]
    if not todo:
        return
    print(f"OpenAlex title search: {len(todo)} papers without DOI")
    for n, p in enumerate(todo, 1):
        # strip characters that break the search filter syntax
        q = p["title"].replace(",", " ").replace(":", " ").replace("|", " ")
        j = get_json(
            "https://api.openalex.org/works",
            params={
                "filter": f"title.search:{q},publication_year:{p['year']-1}|{p['year']}|{p['year']+1}",
                "select": "title,abstract_inverted_index,cited_by_count,open_access",
                "per-page": "3",
                "mailto": MAILTO,
            }, tries=2,
        )
        best = None
        for w in (j or {}).get("results", []):
            t = (w.get("title") or "").lower().strip().rstrip(".")
            if t == p["title"].lower().strip():
                best = w
                break
        if best is None:
            rs = (j or {}).get("results", [])
            if rs:  # accept top hit only if title closely overlaps
                t = (rs[0].get("title") or "").lower()
                a, b = set(t.split()), set(p["title"].lower().split())
                if a and len(a & b) / max(len(a), len(b)) > 0.8:
                    best = rs[0]
        if best:
            abstract = deinvert(best.get("abstract_inverted_index"))
            results[p["key"]] = {
                "abstract": abstract,
                "cited_by": best.get("cited_by_count", 0),
                "oa_url": (best.get("open_access") or {}).get("oa_url") or "",
                "source": "openalex_title" if abstract else "openalex_title_noabs",
            }
        if n % 100 == 0:
            save(results)
            print(f"  title-search {n}/{len(todo)}")
        time.sleep(0.12)
    save(results)


def fetch_crossref(papers, results):
    """Crossref per-DOI fallback, last resort."""
    todo = [p for p in papers if p["doi"] and
            (p["key"] not in results or not results[p["key"]]["abstract"])]
    if not todo:
        return
    print(f"Crossref fallback: {len(todo)} papers (per-DOI)")
    for n, p in enumerate(todo, 1):
        j = get_json(f"https://api.crossref.org/works/{p['doi']}",
                     params={"mailto": MAILTO}, tries=2)
        msg = (j or {}).get("message", {})
        abstract = msg.get("abstract", "")
        if abstract:
            # strip JATS tags
            import re
            abstract = re.sub(r"<[^>]+>", " ", abstract).strip()
            prev = results.get(p["key"], {})
            results[p["key"]] = {
                "abstract": abstract,
                "cited_by": prev.get("cited_by", 0),
                "oa_url": prev.get("oa_url", ""),
                "source": "crossref",
            }
        if n % 100 == 0:
            save(results)
            n_abs = sum(1 for v in results.values() if v["abstract"])
            print(f"  crossref {n}/{len(todo)}, total abstracts {n_abs}")
        time.sleep(0.1)
    save(results)


def save(results):
    tmp = OUT + ".tmp"
    with open(tmp, "w") as f:
        json.dump(results, f, ensure_ascii=False)
    os.replace(tmp, OUT)


def main():
    with open(CORPUS) as f:
        papers = json.load(f)
    results = {}
    if os.path.exists(OUT):
        with open(OUT) as f:
            results = json.load(f)
        print(f"Resuming: {len(results)} already fetched")

    no_doi = [p for p in papers if not p["doi"]]
    print(f"Corpus: {len(papers)} papers, {len(no_doi)} without DOI")

    fetch_openalex(papers, results)
    fetch_openalex_by_title(papers, results)
    fetch_s2(papers, results)
    fetch_crossref(papers, results)
    save(results)

    n_abs = sum(1 for v in results.values() if v["abstract"])
    print(f"\nDone. {n_abs}/{len(papers)} papers have abstracts "
          f"({len(papers) - n_abs} missing).")
    by_src = {}
    for v in results.values():
        if v["abstract"]:
            by_src[v["source"]] = by_src.get(v["source"], 0) + 1
    print("By source:", by_src)


if __name__ == "__main__":
    main()
