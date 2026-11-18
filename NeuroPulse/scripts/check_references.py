#!/usr/bin/env python
"""Check every DOI in paper/references.bib against Crossref.

For each entry with a DOI the script reports whether the DOI resolves, whether
the registered title matches the title in the bibliography, and whether
Crossref records a retraction, withdrawal, removal or expression of concern
for the work (Crossref includes the Retraction Watch database).

This needs internet access. Run it once before every submission:

    python scripts/check_references.py            # prints a table, exit status 1 on any problem
    python scripts/check_references.py --email you@example.org   # polite-pool identification
"""
import argparse
import json
import re
import sys
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BIB = ROOT / "paper" / "references.bib"
PROBLEM_TYPES = {"retraction", "withdrawal", "removal", "expression_of_concern"}


def parse_bib(text):
    """Return a list of dicts with key, title and doi (if any)."""
    entries = []
    for m in re.finditer(r"@\w+\{([^,]+),(.*?)\n\}", text, re.S):
        body = m.group(2)
        title = re.search(r"title\s*=\s*\{(.*)\},?\s*\n", body)
        doi = re.search(r"doi:\s*(10\.\S+?)\s*[},]", body + "}")
        entries.append({"key": m.group(1).strip(),
                        "title": re.sub(r"[{}]", "", title.group(1)) if title else "",
                        "doi": doi.group(1) if doi else None})
    return entries


def normalise(s):
    return re.sub(r"[^a-z0-9]+", " ", s.lower()).strip()


def title_matches(bib_title, registered):
    a, b = normalise(bib_title), normalise(registered)
    return bool(a) and (a in b or b in a or len(set(a.split()) & set(b.split())) / max(len(set(a.split())), 1) > 0.8)


def retraction_status(message):
    """Return a list of problem descriptions found in a Crossref ``message`` object."""
    problems = []
    for update in message.get("updated-by", []) or []:
        kind = str(update.get("type", "")).lower()
        if kind in PROBLEM_TYPES:
            problems.append(f"{kind} notice {update.get('DOI', '')}".strip())
    for update in message.get("update-to", []) or []:  # the record is itself a notice
        kind = str(update.get("type", "")).lower()
        if kind in PROBLEM_TYPES:
            problems.append(f"this record is a {kind} notice for {update.get('DOI', '')}")
    title = " ".join(message.get("title", []) or []).lower()
    if title.startswith(("retracted", "retraction", "withdrawn")):
        problems.append("title marks the work as retracted or withdrawn")
    return problems


def fetch(doi, email=None):
    url = "https://api.crossref.org/works/" + urllib.parse.quote(doi)
    if email:
        url += "?mailto=" + urllib.parse.quote(email)
    req = urllib.request.Request(url, headers={"User-Agent": "NeuroPulse-reference-check/1.0"})
    with urllib.request.urlopen(req, timeout=30) as resp:
        return json.load(resp)["message"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--email", default=None)
    args = ap.parse_args()
    entries = parse_bib(BIB.read_text(encoding="utf-8"))
    bad = 0
    for e in entries:
        if not e["doi"]:
            print(f"[skip] {e['key']:<18} no DOI in the entry")
            continue
        try:
            msg = fetch(e["doi"], args.email)
        except urllib.error.HTTPError as exc:
            print(f"[FAIL] {e['key']:<18} DOI {e['doi']} not found (HTTP {exc.code})")
            bad += 1
            continue
        except OSError as exc:
            sys.exit(f"network error: {exc}. This check needs internet access.")
        problems = retraction_status(msg)
        if not title_matches(e["title"], " ".join(msg.get("title", []) or [])):
            problems.append("registered title differs from the bibliography title")
        if problems:
            bad += 1
            print(f"[FAIL] {e['key']:<18} " + "; ".join(problems))
        else:
            print(f"[ ok ] {e['key']:<18} {e['doi']}")
    print(f"\n{len(entries)} entries, {bad} problem(s).")
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
