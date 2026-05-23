"""HuggingFace Daily Papers source — a curated daily feed of arXiv papers."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import httpx

from research_hidden_gems.models import Paper

HF_DAILY_URL = "https://huggingface.co/api/daily_papers"
_UA = {"User-Agent": "research-hidden-gems/0.1 (+https://arxiv.org)"}


def fetch_recent(*, days: int = 7, max_results: int = 100, timeout: float = 20.0) -> list[Paper]:
    """Fetch HuggingFace Daily Papers over the last ``days`` days."""
    today = datetime.now(timezone.utc).date()
    papers: dict[str, Paper] = {}
    with httpx.Client(timeout=timeout, follow_redirects=True, headers=_UA) as client:
        for delta in range(max(1, days)):
            day = today - timedelta(days=delta)
            try:
                response = client.get(HF_DAILY_URL, params={"date": day.isoformat()})
                response.raise_for_status()
                items = response.json()
            except (httpx.HTTPError, ValueError):
                continue
            for item in items or []:
                paper = _to_paper(item)
                if paper and paper.arxiv_id not in papers:
                    papers[paper.arxiv_id] = paper
            if len(papers) >= max_results:
                break
    return list(papers.values())[:max_results]


def _to_paper(item: dict) -> Paper | None:
    raw = item.get("paper") or {}
    arxiv_id = raw.get("id") or item.get("id")
    if not arxiv_id:
        return None
    published = _parse_dt(raw.get("publishedAt") or item.get("publishedAt"))
    authors = [a.get("name", "") for a in raw.get("authors", []) if a.get("name")]
    upvotes = raw.get("upvotes") or item.get("upvotes") or 0
    return Paper(
        arxiv_id=str(arxiv_id),
        title=" ".join(str(raw.get("title", "")).split()),
        authors=authors,
        summary=" ".join(str(raw.get("summary", "")).split()),
        published=published,
        categories=[],
        abs_url=f"https://arxiv.org/abs/{arxiv_id}",
        pdf_url=f"https://arxiv.org/pdf/{arxiv_id}",
        source="huggingface_daily",
        external_ids={"hf_upvotes": str(upvotes)},
    )


def _parse_dt(value: str | None) -> datetime:
    if not value:
        return datetime.now(timezone.utc)
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return datetime.now(timezone.utc)
