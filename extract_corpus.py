#!/usr/bin/env python3
"""
Extract the 2020+ main-track corpus for the HPC kernel-optimization survey.

Venues: SC, IPDPS, ICS, HPDC, ASPLOS, PPoPP, CGO, USENIX ATC, DAC (conferences);
TPDS (journal). Filtering mirrors HPCs/hpc-first-authors/build.py
(booktitle AND crossref-series must both match; TOC whitelist repairs proceedings
where DBLP folded workshops into the main volume).

Output: data/corpus.json — one record per paper with dblp key, venue, year,
title, authors, pages, DOI (from <ee>), and dblp url.
"""

import json
import os
import re
import sys
from collections import Counter

from lxml import etree

HERE = os.path.dirname(os.path.abspath(__file__))
HPCS_DATA = os.path.join(HERE, "..", "HPCs", "hpc-first-authors", "data")
DBLP_XML = os.path.join(HPCS_DATA, "dblp.xml")
TOC_WHITELIST_FILE = os.path.join(HPCS_DATA, "toc_mainconf_keys.json")
OUT = os.path.join(HERE, "data", "corpus.json")

START_YEAR = 2020
PAGE_THRESHOLD = 6

MAINBT_MAP = {
    "SC": "SC",
    "ICS": "ICS",
    "HPDC": "HPDC",
    "IPDPS": "IPDPS",
    "ASPLOS": "ASPLOS",
    "ASPLOS (1)": "ASPLOS", "ASPLOS (2)": "ASPLOS", "ASPLOS (3)": "ASPLOS",
    "ASPLOS (4)": "ASPLOS", "ASPLOS (5)": "ASPLOS", "ASPLOS (6)": "ASPLOS",
    "PPoPP": "PPoPP", "PPOPP": "PPoPP",
    "CGO": "CGO",
    "USENIX ATC": "ATC",
    "DAC": "DAC",
}
EXPECTED_SERIES = {
    "SC": "conf/sc", "ICS": "conf/ics", "HPDC": "conf/hpdc",
    "IPDPS": "conf/ipps", "ASPLOS": "conf/asplos", "PPoPP": "conf/ppopp",
    "CGO": "conf/cgo", "ATC": "conf/usenix", "DAC": "conf/dac",
}
JOURNAL_MAP = {
    "IEEE Trans. Parallel Distributed Syst.": "TPDS",
    "IEEE Trans. Parallel Distrib. Syst.": "TPDS",
}

_page_normal = re.compile(r"([0-9]+)-([0-9]+)")
_page_colon = re.compile(r"[0-9]+:([1-9][0-9]*)-[0-9]+:([1-9][0-9]*)")


def page_count(page_str):
    if not page_str:
        return -1
    m = _page_normal.match(page_str) or _page_colon.match(page_str)
    if m:
        return int(m.group(2)) - int(m.group(1)) + 1
    return -1


def count_paper(venue, year, pages):
    if year < START_YEAR:
        return False
    if pages != -1 and pages < PAGE_THRESHOLD:
        # SC proceedings in these years list short page ranges for full papers
        if venue == "SC" and year in {2020, 2021, 2024}:
            return True
        return False
    return True


def text_of(elem):
    return "".join(elem.itertext()).strip()


def series_of(xref):
    if not xref:
        return None
    parts = xref.split("/")
    return "/".join(parts[:2]) if len(parts) >= 2 else None


def main():
    with open(TOC_WHITELIST_FILE) as f:
        toc_wl = {proc: set(keys) for proc, keys in json.load(f).items()}

    papers = []
    venue_counts = Counter()
    toc_dropped = Counter()
    short_dropped = Counter()
    n_seen = 0

    context = etree.iterparse(
        DBLP_XML,
        events=("end",),
        tag=("inproceedings", "article"),
        load_dtd=True,
        resolve_entities=True,
        huge_tree=True,
    )

    for _, elem in context:
        n_seen += 1
        if n_seen % 1000000 == 0:
            print(f"  scanned {n_seen} records, kept {len(papers)}", file=sys.stderr)

        venue = None
        if elem.tag == "inproceedings":
            bt = elem.find("booktitle")
            raw = bt.text if bt is not None else None
            venue = MAINBT_MAP.get(raw) if raw else None
            if venue is not None:
                xr = elem.find("crossref")
                xref = xr.text if xr is not None else None
                series = series_of(xref)
                if series is not None and series != EXPECTED_SERIES[venue]:
                    venue = None
                if venue is not None and xref in toc_wl:
                    if elem.get("key") not in toc_wl[xref]:
                        toc_dropped[xref] += 1
                        venue = None
        else:
            jr = elem.find("journal")
            raw = jr.text if jr is not None else None
            if raw in JOURNAL_MAP:
                venue = JOURNAL_MAP[raw]

        if venue is not None:
            year_el = elem.find("year")
            year = int(text_of(year_el)) if year_el is not None and text_of(year_el).isdigit() else -1
            if year >= START_YEAR:
                title_el = elem.find("title")
                pages_el = elem.find("pages")
                url_el = elem.find("url")
                title = text_of(title_el) if title_el is not None else ""
                pages = text_of(pages_el) if pages_el is not None else ""
                pcnt = page_count(pages)
                authors = [text_of(a) for a in elem.findall("author")]
                # ICS booktitle is shared with ITCS (Innovations in Theoretical CS)
                url = text_of(url_el) if url_el is not None else ""
                if venue == "ICS" and "innovations" in url:
                    venue = None
                elif not count_paper(venue, year, pcnt):
                    short_dropped[venue] += 1
                    venue = None
                if venue is not None and authors:
                    doi = ""
                    for ee in elem.findall("ee"):
                        t = text_of(ee)
                        if "doi.org/" in t:
                            doi = t.split("doi.org/", 1)[1]
                            break
                    papers.append({
                        "key": elem.get("key"),
                        "venue": venue,
                        "year": year,
                        "title": title.rstrip("."),
                        "authors": authors,
                        "pages": pages,
                        "doi": doi,
                        "dblp_url": "https://dblp.org/rec/" + (elem.get("key") or ""),
                    })
                    venue_counts[venue] += 1

        elem.clear()
        while elem.getprevious() is not None:
            del elem.getparent()[0]

    papers.sort(key=lambda p: (p["venue"], p["year"], p["title"]))
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w") as f:
        json.dump(papers, f, indent=1, ensure_ascii=False)

    print(f"\nScanned {n_seen} records. Kept {len(papers)} papers -> {OUT}")
    print("\n=== papers per venue (2020+) ===")
    for v, c in sorted(venue_counts.items(), key=lambda x: -x[1]):
        print(f"  {v:8s} {c}")
    n_doi = sum(1 for p in papers if p["doi"])
    print(f"\nWith DOI: {n_doi}/{len(papers)}")
    if short_dropped:
        print("\n=== dropped short (<6p) ===")
        for v, c in short_dropped.most_common():
            print(f"  {v:8s} {c}")
    if toc_dropped:
        print("\n=== TOC whitelist drops ===")
        for proc, c in toc_dropped.most_common():
            print(f"  {proc}: {c}")


if __name__ == "__main__":
    main()
