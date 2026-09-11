#!/usr/bin/env python3
"""
Find open-source artifacts for the 917 included papers. Multi-signal cascade:

  A1 abstract regex        — repo URLs stated in the abstract
  A2 openalex locations    — resolve arXiv ids (batched, 50 DOIs/request)
  A3 arxiv fulltext        — fetch arxiv.org/html (fallback ar5iv), regex repo URLs
  A4 crossref relations    — is-supplemented-by artifact DOIs (Zenodo etc.)
  A5 paperswithcode        — title lookup
  B  github search         — system-name / title search via authenticated API,
                             README cross-check against paper title/authors

Writes data/artifacts_found.json incrementally:
  {key: {"candidates": [{"url","source","evidence"}], "checked": [stages...]}}
Resumable: completed stages per paper are skipped on rerun.
"""

import json
import os
import re
import subprocess
import sys
import time

import requests

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "data", "artifacts_found.json")
MAILTO = "weicunyang@gmail.com"

session = requests.Session()
session.headers["User-Agent"] = f"hpc-kernelbench-artifact-finder (mailto:{MAILTO})"

REPO_RE = re.compile(
    r"(?:https?://)?(?:www\.)?"
    r"(github\.com|gitlab\.com|bitbucket\.org|gitee\.com|zenodo\.org|figshare\.com)"
    r"/([A-Za-z0-9_.\-]+(?:/[A-Za-z0-9_.\-]+)?)", re.I)

STOP_OWNERS = {"features", "topics", "orgs", "search", "sponsors", "settings",
               "marketplace", "apps", "about", "collections", "trending"}


def norm_repo(host, path):
    path = path.rstrip(").,;:'\"]").removesuffix(".git")
    parts = path.split("/")
    if host.lower() == "zenodo.org":
        return f"https://zenodo.org/{'/'.join(parts[:2])}" if len(parts) >= 2 else None
    if len(parts) < 2 or parts[0].lower() in STOP_OWNERS:
        return None
    return f"https://{host.lower()}/{parts[0]}/{parts[1]}"


def extract_repos(text):
    out = []
    for m in REPO_RE.finditer(text or ""):
        u = norm_repo(m.group(1), m.group(2))
        if u and u not in out:
            out.append(u)
    return out


def load():
    if os.path.exists(OUT):
        with open(OUT) as f:
            return json.load(f)
    return {}


def save(db):
    tmp = OUT + ".tmp"
    with open(tmp, "w") as f:
        json.dump(db, f, ensure_ascii=False)
    os.replace(tmp, OUT)


def ent(db, key):
    if key not in db:
        db[key] = {"candidates": [], "checked": []}
    return db[key]


def add_cand(e, url, source, evidence=""):
    if not any(c["url"].lower() == url.lower() for c in e["candidates"]):
        e["candidates"].append({"url": url, "source": source, "evidence": evidence[:200]})


def get_json(url, params=None, headers=None, tries=3, timeout=45):
    for t in range(tries):
        try:
            r = session.get(url, params=params, headers=headers, timeout=timeout)
            if r.status_code in (403, 429):
                wait = int(r.headers.get("Retry-After", 0)) or 20 * (t + 1)
                time.sleep(min(wait, 120))
                continue
            if r.status_code == 404:
                return None
            r.raise_for_status()
            return r.json()
        except Exception:
            if t == tries - 1:
                return None
            time.sleep(5 * (t + 1))
    return None


# ---------------------------------------------------------------- stage A1
def stage_abstract(papers, db):
    n = 0
    for p in papers:
        e = ent(db, p["key"])
        if "abstract" in e["checked"]:
            continue
        for u in extract_repos(p["abstract"]):
            add_cand(e, u, "abstract", "URL in abstract")
            n += 1
        e["checked"].append("abstract")
    save(db)
    print(f"A1 abstract: +{n} candidate urls")


# ---------------------------------------------------------------- stage A2+A3
def stage_arxiv(papers, db):
    todo = [p for p in papers if "arxiv" not in ent(db, p["key"])["checked"] and p["doi"]]
    print(f"A2 openalex locations: {len(todo)} papers")
    arxiv_ids = {}
    for i in range(0, len(todo), 50):
        chunk = todo[i:i + 50]
        j = get_json("https://api.openalex.org/works", params={
            "filter": "doi:" + "|".join(p["doi"] for p in chunk),
            "select": "doi,locations", "per-page": "50", "mailto": MAILTO})
        for w in (j or {}).get("results", []):
            doi = (w.get("doi") or "").replace("https://doi.org/", "").lower()
            for loc in w.get("locations", []):
                lp = (loc.get("landing_page_url") or "")
                m = re.search(r"arxiv\.org/(?:abs|pdf)/([0-9]{4}\.[0-9]{4,5})", lp)
                if m:
                    arxiv_ids[doi] = m.group(1)
        time.sleep(0.15)
    doi2key = {p["doi"].lower(): p["key"] for p in todo}
    print(f"   resolved {len(arxiv_ids)} arXiv ids")

    n = 0
    for doi, aid in arxiv_ids.items():
        key = doi2key.get(doi)
        if key is None:
            continue
        e = ent(db, key)
        html = None
        for base in (f"https://arxiv.org/html/{aid}", f"https://ar5iv.labs.arxiv.org/html/{aid}"):
            try:
                r = session.get(base, timeout=45)
                if r.status_code == 200 and len(r.text) > 5000:
                    html = r.text
                    break
            except Exception:
                pass
            time.sleep(0.4)
        if html:
            for u in extract_repos(html):
                add_cand(e, u, "arxiv-fulltext", f"arXiv:{aid}")
                n += 1
        e.setdefault("arxiv", aid)
        time.sleep(0.4)
    for p in todo:
        e = ent(db, p["key"])
        if "arxiv" not in e["checked"]:
            e["checked"].append("arxiv")
    save(db)
    print(f"A3 arxiv fulltext: +{n} candidate urls")


# ---------------------------------------------------------------- stage A4
def stage_crossref(papers, db):
    todo = [p for p in papers if p["doi"] and "crossref" not in ent(db, p["key"])["checked"]]
    print(f"A4 crossref relations: {len(todo)} papers")
    n = 0
    for i, p in enumerate(todo):
        j = get_json(f"https://api.crossref.org/works/{p['doi']}",
                     params={"mailto": MAILTO}, tries=2)
        msg = (j or {}).get("message", {})
        rel = msg.get("relation", {}) or {}
        e = ent(db, p["key"])
        for rtype in ("is-supplemented-by", "references"):
            for r in rel.get(rtype, []):
                rid = str(r.get("id", ""))
                if "zenodo" in rid.lower() or "10.5281" in rid:
                    add_cand(e, f"https://doi.org/{rid}" if rid.startswith("10.") else rid,
                             "crossref", rtype)
                    n += 1
        e["checked"].append("crossref")
        if i % 200 == 199:
            save(db)
            print(f"   {i+1}/{len(todo)}")
        time.sleep(0.08)
    save(db)
    print(f"A4 crossref: +{n} artifact dois")


# ---------------------------------------------------------------- stage A5
def stage_pwc(papers, db):
    todo = [p for p in papers if "pwc" not in ent(db, p["key"])["checked"]]
    print(f"A5 paperswithcode: {len(todo)} papers")
    n = 0
    for i, p in enumerate(todo):
        e = ent(db, p["key"])
        j = get_json("https://paperswithcode.com/api/v1/papers/",
                     params={"title": p["title"][:120]}, tries=2, timeout=30)
        for hit in (j or {}).get("results", [])[:3]:
            if (hit.get("title") or "").lower().strip() == p["title"].lower().strip():
                pid = hit.get("id")
                rj = get_json(f"https://paperswithcode.com/api/v1/papers/{pid}/repositories/",
                              tries=2, timeout=30)
                for repo in (rj or {}).get("results", [])[:3]:
                    u = repo.get("url")
                    if u:
                        add_cand(e, u, "pwc", "paperswithcode title match")
                        n += 1
        e["checked"].append("pwc")
        if i % 100 == 99:
            save(db)
            print(f"   {i+1}/{len(todo)}, +{n}")
        time.sleep(0.8)
    save(db)
    print(f"A5 pwc: +{n}")


# ---------------------------------------------------------------- stage A6
def stage_oapdf(papers, db):
    import io
    from pypdf import PdfReader
    todo = [p for p in papers
            if p["oa_url"] and "dl.acm.org" not in p["oa_url"]
            and not ent(db, p["key"])["candidates"]
            and "oapdf" not in ent(db, p["key"])["checked"]]
    print(f"A6 OA pdf scan: {len(todo)} papers")
    n = 0
    for i, p in enumerate(todo):
        e = ent(db, p["key"])
        e["checked"].append("oapdf")
        try:
            r = session.get(p["oa_url"], timeout=60, allow_redirects=True)
            if r.status_code != 200:
                continue
            if "pdf" in r.headers.get("Content-Type", "") or r.content[:5] == b"%PDF-":
                reader = PdfReader(io.BytesIO(r.content))
                text = " ".join((pg.extract_text() or "") for pg in reader.pages[:3])
                # PDF extraction breaks urls across lines; collapse whitespace
                text = re.sub(r"\s+", "", text)
            else:
                text = r.text
            for u in extract_repos(text):
                add_cand(e, u, "oa-fulltext", p["oa_url"][:100])
                n += 1
        except Exception:
            pass
        if i % 25 == 24:
            save(db)
            print(f"   {i+1}/{len(todo)}, +{n}")
        time.sleep(0.3)
    save(db)
    print(f"A6 oa pdf: +{n} candidate urls")


# ---------------------------------------------------------------- stage B
NAME_RE = re.compile(r"^([A-Za-z0-9+\-]{2,25}):\s")


def gh_token():
    return subprocess.run(["gh", "auth", "token"], capture_output=True,
                          text=True).stdout.strip()


def stage_github(papers, db):
    token = gh_token()
    if not token:
        print("B github: no token, skipping")
        return
    hdr = {"Authorization": f"Bearer {token}",
           "Accept": "application/vnd.github+json"}

    def gh(url, params=None):
        while True:
            r = session.get(url, params=params, headers=hdr, timeout=45)
            if r.status_code == 403 and r.headers.get("X-RateLimit-Remaining") == "0":
                wait = int(r.headers.get("X-RateLimit-Reset", 0)) - int(time.time())
                time.sleep(max(5, min(wait + 2, 120)))
                continue
            if r.status_code in (404, 422):
                return None
            if r.status_code == 403:
                time.sleep(30)
                continue
            r.raise_for_status()
            return r.json()

    todo = [p for p in papers if "github" not in ent(db, p["key"])["checked"]]
    print(f"B github search: {len(todo)} papers")
    n = 0
    for i, p in enumerate(todo):
        e = ent(db, p["key"])
        m = NAME_RE.match(p["title"])
        queries = []
        if m and len(m.group(1)) >= 3:
            queries.append(f"{m.group(1)} in:name")
        # title-phrase fallback (first 8 words, quoted, in readme+description)
        words = re.sub(r"[^A-Za-z0-9 ]", " ", p["title"]).split()
        if len(words) >= 4:
            queries.append('"' + " ".join(words[:8]) + '"')
        found = []
        for q in queries:
            j = gh("https://api.github.com/search/repositories",
                   {"q": q, "per_page": "5", "sort": "stars"})
            for repo in (j or {}).get("items", [])[:5]:
                found.append(repo)
            time.sleep(2.2)   # 30 searches/min budget
            if found:
                break
        title_words = set(w.lower() for w in words if len(w) > 3)
        for repo in found[:3]:
            desc = (repo.get("description") or "")
            dwords = set(re.sub(r"[^A-Za-z0-9 ]", " ", desc).lower().split())
            overlap = len(title_words & dwords)
            sysname_hit = bool(m) and m.group(1).lower() == repo["name"].lower()
            if sysname_hit or overlap >= 3:
                add_cand(e, repo["html_url"], "github-search",
                         f"name={repo['full_name']} stars={repo.get('stargazers_count',0)} "
                         f"desc-overlap={overlap} sysname={sysname_hit}")
                n += 1
        e["checked"].append("github")
        if i % 50 == 49:
            save(db)
            print(f"   {i+1}/{len(todo)}, +{n} candidates")
    save(db)
    print(f"B github: +{n} candidates")


# ---------------------------------------------------------------- stage B2
def stage_github_readme(papers, db):
    token = gh_token()
    if not token:
        print("B2: no token")
        return
    hdr = {"Authorization": f"Bearer {token}",
           "Accept": "application/vnd.github+json"}
    todo = [p for p in papers if not ent(db, p["key"])["candidates"]
            and "github-readme" not in ent(db, p["key"])["checked"]]
    print(f"B2 github in:readme search: {len(todo)} papers")
    n = 0
    for i, p in enumerate(todo):
        e = ent(db, p["key"])
        e["checked"].append("github-readme")
        words = re.sub(r"[^A-Za-z0-9 ]", " ", p["title"]).split()
        if len(words) < 5:
            continue
        q = '"' + " ".join(words[:8]) + '" in:readme'
        while True:
            r = session.get("https://api.github.com/search/repositories",
                            params={"q": q, "per_page": "3"}, headers=hdr, timeout=45)
            if r.status_code == 403 and r.headers.get("X-RateLimit-Remaining") == "0":
                wait = int(r.headers.get("X-RateLimit-Reset", 0)) - int(time.time())
                time.sleep(max(5, min(wait + 2, 120)))
                continue
            break
        if r.status_code == 200:
            for repo in r.json().get("items", [])[:3]:
                add_cand(e, repo["html_url"], "github-readme",
                         f"title-phrase in README of {repo['full_name']} "
                         f"stars={repo.get('stargazers_count',0)}")
                n += 1
        if i % 50 == 49:
            save(db)
            print(f"   {i+1}/{len(todo)}, +{n}")
        time.sleep(2.2)
    save(db)
    print(f"B2 github-readme: +{n} candidates")


def main():
    with open(os.path.join(HERE, "output", "included.json")) as f:
        papers = json.load(f)
    db = load()
    stages = sys.argv[1:] or ["abstract", "arxiv", "crossref", "pwc", "oapdf", "github"]
    if "abstract" in stages:
        stage_abstract(papers, db)
    if "arxiv" in stages:
        stage_arxiv(papers, db)
    if "crossref" in stages:
        stage_crossref(papers, db)
    if "pwc" in stages:
        stage_pwc(papers, db)
    if "oapdf" in stages:
        stage_oapdf(papers, db)
    if "github" in stages:
        stage_github(papers, db)
    if "github-readme" in stages:
        stage_github_readme(papers, db)

    with_c = sum(1 for k in db if db[k]["candidates"])
    print(f"\npapers with >=1 candidate: {with_c}/{len(papers)}")


if __name__ == "__main__":
    main()
