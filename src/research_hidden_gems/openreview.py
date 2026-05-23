"""OpenReview source (best-effort).

Recent-submission feeds on OpenReview are keyed by per-venue *invitation* ids
(e.g. ``ICLR.cc/2026/Conference/-/Submission``), which rotate every cycle. So we
require an explicit venue list (via config) and fail gracefully — returning ``[]``
when none are configured or the API is unavailable, rather than guessing stale
ids. This keeps "all sources" honest without shipping broken defaults.
"""

from __future__ import annotations

import re
from datetime import datetime, timezone

import httpx

from research_hidden_gems.models import Paper

OPENREVIEW_API = "https://api2.openreview.net/notes"
_ARXIV_RE = re.compile(r"arxiv\.org/(?:abs|pdf)/(\d{4}\.\d{4,5})", re.I)


def fetch_recent(
    *,
    venues: list[str] | None = None,
    max_results: int = 100,
    timeout: float = 20.0,
) -> list[Paper]:
    """Fetch recent submissions for the given OpenReview venue invitation ids."""
    venues = venues or []
    if not venues:
        return []
    papers: dict[str, Paper] = {}
    with httpx.Client(timeout=timeout, follow_redirects=True) as client:
        for invitation in venues:
            try:
                response = client.get(
                    OPENREVIEW_API,
                    params={"invitation": invitation, "sort": "tcdate:desc", "limit": min(100, max_results)},
                )
                response.raise_for_status()
                notes = response.json().get("notes", [])
            except (httpx.HTTPError, ValueError):
                continue
            for note in notes:
                paper = _note_to_paper(note, invitation)
                if paper and paper.key not in papers:
                    papers[paper.key] = paper
            if len(papers) >= max_results:
                break
    return list(papers.values())[:max_results]


def _note_to_paper(note: dict, invitation: str) -> Paper | None:
    content = note.get("content") or {}
    title = _value(content.get("title"))
    abstract = _value(content.get("abstract"))
    if not title:
        return None
    authors = _value(content.get("authors")) or []
    if isinstance(authors, str):
        authors = [authors]
    note_id = note.get("id", "")
    arxiv_id = _find_arxiv_id(content)
    published = _parse_ms(note.get("pdate") or note.get("cdate") or note.get("tcdate"))
    return Paper(
        arxiv_id=arxiv_id or "",
        title=" ".join(str(title).split()),
        authors=[str(a) for a in authors],
        summary=" ".join(str(abstract).split()),
        published=published,
        categories=[invitation],
        abs_url=arxiv_id and f"https://arxiv.org/abs/{arxiv_id}" or f"https://openreview.net/forum?id={note_id}",
        source="openreview",
        external_ids={"openreview": note_id},
    )


def _value(field):
    """OpenReview v2 wraps content values as {"value": ...}."""
    if isinstance(field, dict) and "value" in field:
        return field["value"]
    return field


def _find_arxiv_id(content: dict) -> str:
    for raw in content.values():
        value = _value(raw)
        if isinstance(value, str):
            match = _ARXIV_RE.search(value)
            if match:
                return match.group(1)
    return ""


def _parse_ms(value) -> datetime:
    try:
        return datetime.fromtimestamp(int(value) / 1000, tz=timezone.utc)
    except (TypeError, ValueError):
        return datetime.now(timezone.utc)
