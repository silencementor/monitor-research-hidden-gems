from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any


@dataclass(slots=True)
class Paper:
    arxiv_id: str
    title: str
    authors: list[str]
    summary: str
    published: datetime
    updated: datetime | None = None
    categories: list[str] = field(default_factory=list)
    abs_url: str | None = None
    pdf_url: str | None = None
    comment: str | None = None
    doi: str | None = None
    citation_count: int | None = None
    influential_citation_count: int | None = None
    semantic_scholar_url: str | None = None
    venue: str | None = None
    external_ids: dict[str, str] = field(default_factory=dict)

    @property
    def text(self) -> str:
        return f"{self.title}\n{self.summary}"

    def age_days(self, now: datetime | None = None) -> int:
        now = now or datetime.now(timezone.utc)
        published = self.published
        if published.tzinfo is None:
            published = published.replace(tzinfo=timezone.utc)
        return max(0, (now - published).days)


@dataclass(slots=True)
class ScoredPaper:
    paper: Paper
    score: float
    components: dict[str, float]
    techniques: list[str]
    problems: list[str]
    rare_terms: list[str]
    rationale: list[str]

    def to_dict(self) -> dict[str, Any]:
        paper = self.paper
        return {
            "arxiv_id": paper.arxiv_id,
            "title": paper.title,
            "authors": paper.authors,
            "published": paper.published.date().isoformat(),
            "categories": paper.categories,
            "abstract_url": paper.abs_url,
            "pdf_url": paper.pdf_url,
            "comment": paper.comment,
            "venue": paper.venue,
            "citation_count": paper.citation_count,
            "influential_citation_count": paper.influential_citation_count,
            "score": round(self.score, 2),
            "components": {key: round(value, 3) for key, value in self.components.items()},
            "techniques": self.techniques,
            "problems": self.problems,
            "rare_terms": self.rare_terms,
            "rationale": self.rationale,
        }
