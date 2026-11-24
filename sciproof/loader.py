"""Module 1: load papers from JSON into our data model."""
from __future__ import annotations

import json
from pathlib import Path

from .models import Paper, Section


def load_papers(path: str | Path) -> list[Paper]:
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    return [
        Paper(
            paper_id=p["paper_id"],
            title=p["title"],
            year=p["year"],
            study_type=p["study_type"],
            sections=[Section(s["name"], s["text"]) for s in p["sections"]],
        )
        for p in raw
    ]
