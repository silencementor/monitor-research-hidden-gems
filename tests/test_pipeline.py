from datetime import datetime, timezone

from research_hidden_gems.config import Config
from research_hidden_gems.models import Paper
from research_hidden_gems.pipeline import _dedup, rank_papers


def _paper(arxiv_id: str, title: str, summary: str, citations: int = 0, **kwargs) -> Paper:
    base = dict(
        arxiv_id=arxiv_id,
        title=title,
        authors=[],
        summary=summary,
        published=datetime(2026, 1, 1, tzinfo=timezone.utc),
        citation_count=citations,
        abs_url=f"https://arxiv.org/abs/{arxiv_id}",
    )
    base.update(kwargs)
    return Paper(**base)


def test_rank_papers_offline_blends_signals() -> None:
    cfg = Config(embed_backend="hashing", judge_enabled=False)
    papers = [
        _paper(
            "2601.00001",
            "Latent Debate Steering for Retrieval-Augmented Recommenders",
            "We introduce a novel training pipeline that combines retrieval with activation "
            "steering to address inference cost for language model recommender systems.",
            citations=1,
        ),
        _paper(
            "2601.00002",
            "A Survey of Recent Reasoning Benchmarks",
            "This paper reviews benchmark trends and summarizes evaluation protocols.",
            citations=400,
        ),
        _paper(
            "2601.00003",
            "Medieval Beekeeping Practices",
            "A historical study of honey production techniques in monasteries.",
            citations=0,
        ),
    ]

    scored = rank_papers(cfg, papers, do_judge=False, now=datetime(2026, 3, 1, tzinfo=timezone.utc))

    assert len(scored) == 3
    # On-topic + novel + low-citation paper should win over the popular survey and
    # the off-topic (high-outlier but irrelevant) beekeeping paper.
    assert "Latent Debate Steering" in scored[0].paper.title
    for item in scored:
        assert {"prefilter", "relevance", "outlier", "hiddenness"} <= set(item.components)
        assert 0.0 <= item.score <= 100.0
        assert item.verdict is None  # judge disabled


def test_relevance_gate_downranks_offtopic_outlier() -> None:
    cfg = Config(embed_backend="hashing", judge_enabled=False)
    papers = [
        _paper(
            "2601.00010",
            "Scalable Retrieval Indexing for Large Language Model Agents",
            "We propose a new vector indexing mechanism for agentic retrieval over big data.",
            citations=2,
        ),
        _paper(
            "2601.00011",
            "Glacial Sediment Cores from the Pleistocene",
            "An unusual geological survey of ancient sediment layers and ice formations.",
            citations=0,
        ),
    ]
    scored = rank_papers(cfg, papers, do_judge=False, now=datetime(2026, 3, 1, tzinfo=timezone.utc))
    by_title = {item.paper.title: item for item in scored}
    on_topic = by_title["Scalable Retrieval Indexing for Large Language Model Agents"]
    off_topic = by_title["Glacial Sediment Cores from the Pleistocene"]
    assert on_topic.components["relevance"] > off_topic.components["relevance"]
    assert on_topic.score > off_topic.score


def test_dedup_preserves_premium_venue_metadata() -> None:
    arxiv = _paper("2601.00020", "Indexed Agents", "A paper.", source="arxiv")
    premium = _paper(
        "2601.00020",
        "Indexed Agents",
        "A longer paper.",
        source="premium_venues",
        venue="International Conference on Learning Representations",
    )

    merged = _dedup([arxiv, premium])

    assert len(merged) == 1
    assert merged[0].source == "premium_venues"
    assert merged[0].venue == "International Conference on Learning Representations"
    assert merged[0].external_ids["source_before_premium"] == "arxiv"
