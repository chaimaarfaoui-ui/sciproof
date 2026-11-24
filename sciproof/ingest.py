"""Module 1b: getting REAL papers in. (The synthetic corpus is for testing only.)

  parse_pubmed_xml : PubMed efetch XML -> Paper (structured abstracts become sections)
  fetch_pubmed     : esearch + efetch over HTTPS (needs internet; not run in the sandbox)
  parse_plain_text : a pasted/extracted full text -> Paper, splitting on standard headings
PDF: extract text with PyMuPDF (`pip install pymupdf`) and pass it to parse_plain_text.
"""
from __future__ import annotations

import re
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET

from .models import Paper, Section

PUBTYPE_MAP = {
    "randomized controlled trial": "RCT", "meta-analysis": "meta-analysis",
    "systematic review": "review", "review": "review", "observational study": "cohort",
    "cohort studies": "cohort", "case-control studies": "case-control",
    "clinical trial": "RCT",
}
HEADINGS = ["abstract", "introduction", "background", "methods", "materials and methods",
            "results", "discussion", "conclusion", "conclusions", "references"]


def _study_type(types: list[str]) -> str:
    low = [t.lower() for t in types]
    for key in ("meta-analysis", "randomized controlled trial", "systematic review",
                "cohort studies", "case-control studies", "observational study", "clinical trial", "review"):
        if key in low:
            return PUBTYPE_MAP[key]
    return "other"


def parse_pubmed_xml(xml_text: str) -> list[Paper]:
    root = ET.fromstring(xml_text)
    papers = []
    for art in root.iter("PubmedArticle"):
        pmid = art.findtext(".//MedlineCitation/PMID") or "unknown"
        title = "".join(art.find(".//ArticleTitle").itertext()) if art.find(".//ArticleTitle") is not None else ""
        year_txt = art.findtext(".//JournalIssue/PubDate/Year") or (art.findtext(".//JournalIssue/PubDate/MedlineDate") or "")[:4]
        year = int(year_txt) if year_txt.isdigit() else 0
        types = [t.text or "" for t in art.iterfind(".//PublicationType")]
        sections = []
        for node in art.iterfind(".//Abstract/AbstractText"):
            label = (node.get("Label") or "Abstract").title()
            text = "".join(node.itertext()).strip()
            if text:
                sections.append(Section(label, text))
        if sections:
            papers.append(Paper(f"PMID{pmid}", title, year, _study_type(types), sections))
    return papers


def fetch_pubmed(query: str, n: int = 50, email: str = "you@example.com") -> list[Paper]:
    base = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/"
    q = urllib.parse.urlencode({"db": "pubmed", "term": query, "retmax": n, "retmode": "json", "email": email})
    import json
    ids = json.loads(urllib.request.urlopen(base + "esearch.fcgi?" + q, timeout=30).read())["esearchresult"]["idlist"]
    q = urllib.parse.urlencode({"db": "pubmed", "id": ",".join(ids), "retmode": "xml", "email": email})
    return parse_pubmed_xml(urllib.request.urlopen(base + "efetch.fcgi?" + q, timeout=60).read().decode())


def parse_plain_text(text: str, paper_id: str, title: str, year: int, study_type: str = "other") -> Paper:
    pat = re.compile(r"^\s*(?:\d+\.?\s*)?(" + "|".join(HEADINGS) + r")\s*:?\s*$", re.I | re.M)
    marks = [(m.start(), m.end(), m.group(1).title()) for m in pat.finditer(text)]
    if not marks:
        return Paper(paper_id, title, year, study_type, [Section("Full Text", text.strip())])
    secs = []
    for i, (_, end, name) in enumerate(marks):
        stop = marks[i + 1][0] if i + 1 < len(marks) else len(text)
        body = text[end:stop].strip()
        if body:
            secs.append(Section(name, " ".join(body.split())))
    return Paper(paper_id, title, year, study_type, secs)
