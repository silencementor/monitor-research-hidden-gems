from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any


def _as_unit(value: Any, default: float = 0.0) -> float:
    """Coerce to a float clamped to [0, 1] (verdict scores live on this scale)."""
    try:
        if value is None:
            return default
        return max(0.0, min(1.0, float(value)))
    except (TypeError, ValueError):
        return default


@dataclass(slots=True)
class LLMVerdict:
    """Structured judgement produced by the Claude deep-judge stage."""

    novelty: float = 0.0           # 0..1 genuine novelty of the core idea
    transferability: float = 0.0   # 0..1 applicability to the user's domains
    confidence: float = 0.0        # 0..1 the judge's own confidence
    is_hidden_gem: bool = False
    technique: str = ""            # the core novel technique/problem, named
    one_liner: str = ""            # the idea in a single sentence
    why_overlooked: str = ""       # why it is plausibly underappreciated
    application_to_user: str = ""  # concrete way to apply it to the user's work
    raw: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "novelty": round(self.novelty, 3),
            "transferability": round(self.transferability, 3),
            "confidence": round(self.confidence, 3),
            "is_hidden_gem": self.is_hidden_gem,
            "technique": self.technique,
            "one_liner": self.one_liner,
            "why_overlooked": self.why_overlooked,
            "application_to_user": self.application_to_user,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "LLMVerdict":
        return cls(
            novelty=_as_unit(d.get("novelty")),
            transferability=_as_unit(d.get("transferability")),
            confidence=_as_unit(d.get("confidence")),
            is_hidden_gem=bool(d.get("is_hidden_gem", False)),
            technique=str(d.get("technique") or ""),
            one_liner=str(d.get("one_liner") or ""),
            why_overlooked=str(d.get("why_overlooked") or ""),
            application_to_user=str(d.get("application_to_user") or ""),
            raw=str(d.get("raw") or ""),
        )


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
    source: str = "arxiv"

    @property
    def text(self) -> str:
        return f"{self.title}\n{self.summary}"

    @property
    def key(self) -> str:
        """Stable dedup/display key across sources (arXiv id, else DOI, else URL)."""
        return self.arxiv_id or self.doi or self.abs_url or self.title

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
    verdict: LLMVerdict | None = None

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
            "verdict": self.verdict.to_dict() if self.verdict else None,
        }
