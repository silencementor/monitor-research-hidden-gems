"""Hybrid ranking pipeline.

    fetch (multi-source) -> dedup/merge -> citation enrichment
      -> lexical prefilter (Codex) + embedding-outlier novelty + relevance
      -> blended prefilter score -> Claude deep-judge on the top shortlist
      -> final blended score

Every stage degrades gracefully: a dead source, missing embeddings backend, or
absent API key just drops that signal instead of failing the run.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Callable

from research_hidden_gems import huggingface_daily, openalex, openreview, semantic_scholar
from research_hidden_gems.arxiv_client import ArxivClient
from research_hidden_gems.config import Config
from research_hidden_gems.embedding_signals import get_embedder, outlier_scores, relevance_scores
from research_hidden_gems.llm_judge import judge_papers
from research_hidden_gems.models import Paper, ScoredPaper
from research_hidden_gems.scoring import score_papers

Progress = Callable[[str], None]


def run_pipeline(
    config: Config,
    *,
    query: str = "",
    days: int = 30,
    max_results: int = 120,
    enrich: bool = True,
    do_judge: bool = True,
    openreview_venues: list[str] | None = None,
    now: datetime | None = None,
    progress: Progress | None = None,
) -> list[ScoredPaper]:
    now = now or datetime.now(timezone.utc)
    if progress:
        label = query or "all configured topics"
        progress(f"Pipeline started: query={label!r}, days={days}, max_results={max_results}")
    papers = _fetch_all(
        config,
        query=query,
        days=days,
        max_results=max_results,
        openreview_venues=openreview_venues,
        progress=progress,
    )
    if progress:
        progress(f"Deduplicating {len(papers)} fetched papers")
    papers = _dedup(papers)
    if progress:
        progress(f"Deduplicated to {len(papers)} papers")
    if enrich:
        _enrich(config, papers, progress=progress)
    elif progress:
        progress("Citation enrichment skipped")
    return rank_papers(config, papers, do_judge=do_judge, now=now, progress=progress)


def rank_papers(
    config: Config,
    papers: list[Paper],
    *,
    do_judge: bool = True,
    now: datetime | None = None,
    progress: Progress | None = None,
) -> list[ScoredPaper]:
    """Score an already-fetched list: lexical + semantic prefilter, judge, blend."""
    now = now or datetime.now(timezone.utc)
    if progress:
        progress(f"Scoring {len(papers)} papers with lexical signals")
    scored = score_papers(papers, now=now)
    _apply_semantic_signals(scored, config, progress=progress)
    if progress:
        progress("Blending prefilter scores")
    for item in scored:
        item.components["prefilter"] = round(_blend_prefilter(item, config), 4)
    scored.sort(key=lambda s: s.components["prefilter"], reverse=True)

    if do_judge:
        judge_papers(scored, config, progress=progress)
    elif progress:
        progress("LLM judge skipped")

    if progress:
        progress("Blending final scores")
    for item in scored:
        item.score = round(100.0 * _blend_final(item, config), 4)
        item.rationale = _augment_rationale(item)
    scored.sort(key=lambda s: s.score, reverse=True)
    if progress:
        progress(f"Ranking finished: {len(scored)} scored papers")
    return scored


# ---------------------------------------------------------------- fetch -----
def _fetch_all(
    config: Config,
    *,
    query: str,
    days: int,
    max_results: int,
    openreview_venues: list[str] | None,
    progress: Progress | None = None,
) -> list[Paper]:
    papers: list[Paper] = []
    sources = set(config.sources)

    if "arxiv" in sources:
        try:
            if progress:
                progress("Fetching arXiv")
            fetched = ArxivClient().search(
                query=query, categories=config.categories, days=days, max_results=max_results
            )
            papers += fetched
            if progress:
                progress(f"Fetched arXiv: {len(fetched)} papers")
        except Exception as exc:
            if progress:
                progress(f"arXiv fetch failed: {exc}")
    if "huggingface_daily" in sources:
        try:
            if progress:
                progress("Fetching Hugging Face Daily Papers")
            fetched = huggingface_daily.fetch_recent(days=days, max_results=max_results)
            papers += fetched
            if progress:
                progress(f"Fetched Hugging Face Daily Papers: {len(fetched)} papers")
        except Exception as exc:
            if progress:
                progress(f"Hugging Face Daily Papers fetch failed: {exc}")
    # OpenAlex discovery is topic-focused; only run it with a query (else too broad).
    if "openalex" in sources and query.strip():
        try:
            if progress:
                progress("Fetching OpenAlex")
            fetched = openalex.fetch_recent(
                days=max(days, 120), max_results=max_results, search=query, mailto=config.openalex_mailto
            )
            papers += fetched
            if progress:
                progress(f"Fetched OpenAlex: {len(fetched)} papers")
        except Exception as exc:
            if progress:
                progress(f"OpenAlex fetch failed: {exc}")
    elif "openalex" in sources and progress:
        progress("Skipping OpenAlex discovery because no query was provided")
    if "premium_venues" in sources:
        try:
            if progress:
                progress(f"Fetching premium venues via OpenAlex ({', '.join(config.premium_venues)})")
            fetched = openalex.fetch_premium_venues(
                days=days,
                max_results=max_results,
                search=query or None,
                venues=config.premium_venues,
                mailto=config.openalex_mailto,
            )
            papers += fetched
            if progress:
                progress(f"Fetched premium venues: {len(fetched)} papers")
        except Exception as exc:
            if progress:
                progress(f"Premium venue fetch failed: {exc}")
    if "openreview" in sources:
        try:
            if progress:
                progress("Fetching OpenReview")
            fetched = openreview.fetch_recent(venues=openreview_venues, max_results=max_results)
            papers += fetched
            if progress:
                progress(f"Fetched OpenReview: {len(fetched)} papers")
        except Exception as exc:
            if progress:
                progress(f"OpenReview fetch failed: {exc}")
    return papers


def _dedup(papers: list[Paper]) -> list[Paper]:
    merged: dict[str, Paper] = {}
    for paper in papers:
        key = paper.key
        if key in merged:
            _merge(merged[key], paper)
        else:
            merged[key] = paper
    return list(merged.values())


def _merge(into: Paper, other: Paper) -> None:
    if not into.summary and other.summary:
        into.summary = other.summary
    if not into.doi and other.doi:
        into.doi = other.doi
    if not into.abs_url and other.abs_url:
        into.abs_url = other.abs_url
    if not into.pdf_url and other.pdf_url:
        into.pdf_url = other.pdf_url
    if not into.venue and other.venue:
        into.venue = other.venue
    if not into.categories and other.categories:
        into.categories = other.categories
    if other.citation_count is not None:
        into.citation_count = max(into.citation_count or 0, other.citation_count)
    into.external_ids.update({k: v for k, v in other.external_ids.items() if v})
    if into.source != other.source:
        if other.source == "premium_venues":
            original_source = into.source
            _record_additional_source(into, original_source)
            into.external_ids.setdefault("source_before_premium", original_source)
            into.source = other.source
        else:
            _record_additional_source(into, other.source)


def _record_additional_source(paper: Paper, source: str) -> None:
    existing = {item for item in paper.external_ids.get("also_in", "").split(",") if item}
    existing.add(source)
    paper.external_ids["also_in"] = ",".join(sorted(existing))


def _enrich(config: Config, papers: list[Paper], progress: Progress | None = None) -> None:
    if "openalex" in set(config.sources):
        try:
            if progress:
                progress("Enriching citations from OpenAlex")
            openalex.enrich_with_openalex(papers, mailto=config.openalex_mailto)
            if progress:
                progress("OpenAlex citation enrichment finished")
        except Exception as exc:
            if progress:
                progress(f"OpenAlex citation enrichment failed: {exc}")
    try:
        if progress:
            progress("Enriching citations from Semantic Scholar")
        semantic_scholar.enrich_with_semantic_scholar([p for p in papers if p.arxiv_id])
        if progress:
            progress("Semantic Scholar citation enrichment finished")
    except Exception as exc:
        if progress:
            progress(f"Semantic Scholar citation enrichment failed: {exc}")


# --------------------------------------------------------- semantic signals -
def _apply_semantic_signals(scored: list[ScoredPaper], config: Config, progress: Progress | None = None) -> None:
    if not scored:
        return
    texts = [item.paper.text for item in scored]
    if progress:
        progress(f"Encoding {len(texts)} papers with {config.embed_backend} embeddings")
    embedder = get_embedder(config.embed_backend, config.embed_model, config.openai_embed_model)
    matrix = embedder.encode(texts)
    profile_vec = embedder.encode([config.profile])[0]
    if progress:
        progress(f"Embedding backend resolved to {embedder.backend}; computing relevance and outlier scores")
    outliers = outlier_scores(matrix)
    relevances = relevance_scores(matrix, profile_vec, texts, config.keywords)
    for item, outlier, relevance in zip(scored, outliers, relevances, strict=True):
        item.components["outlier"] = round(float(outlier), 4)
        item.components["relevance"] = round(float(relevance), 4)


# -------------------------------------------------------------- blending ----
def _blend_prefilter(item: ScoredPaper, config: Config) -> float:
    c = item.components
    lexical = _mean(c.get("novelty", 0.0), c.get("technique", 0.0), c.get("applicability", 0.0), c.get("rarity", 0.0))
    outlier = c.get("outlier", 0.0)
    hiddenness = c.get("hiddenness", 0.0)
    relevance = c.get("relevance", 0.0)
    return _clamp(
        config.w_lexical * lexical
        + config.w_outlier * outlier
        + config.w_hiddenness * hiddenness
        + config.w_relevance * relevance
    )


def _blend_final(item: ScoredPaper, config: Config) -> float:
    prefilter = item.components.get("prefilter", 0.0)
    if item.verdict is not None:
        v = item.verdict
        llm = _clamp(0.5 * v.novelty + 0.3 * v.transferability + 0.2 * (1.0 if v.is_hidden_gem else 0.0))
        base = config.w_final_prefilter * prefilter + config.w_final_llm * llm
    else:
        base = prefilter
    relevance = item.components.get("relevance", 0.0)
    gate = config.relevance_floor + (1.0 - config.relevance_floor) * relevance
    return _clamp(base * gate)


def _augment_rationale(item: ScoredPaper) -> list[str]:
    reasons = list(item.rationale)
    c = item.components
    reasons.append(
        f"Semantic signals: outlier-novelty={c.get('outlier', 0.0):.2f}, relevance={c.get('relevance', 0.0):.2f}."
    )
    if item.verdict is not None:
        v = item.verdict
        reasons.append(
            f"Claude judge: novelty={v.novelty:.2f}, transferability={v.transferability:.2f}, "
            f"hidden_gem={v.is_hidden_gem} (conf {v.confidence:.2f})."
        )
        if v.why_overlooked:
            reasons.append(f"Why overlooked: {v.why_overlooked}")
        if v.application_to_user:
            reasons.append(f"Apply to your work: {v.application_to_user}")
    return reasons


def _mean(*values: float) -> float:
    return sum(values) / len(values) if values else 0.0


def _clamp(value: float) -> float:
    return max(0.0, min(1.0, value))
