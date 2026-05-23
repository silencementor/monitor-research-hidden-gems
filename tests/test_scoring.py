from datetime import datetime, timezone

from research_hidden_gems.models import Paper
from research_hidden_gems.scoring import score_papers


def _paper(title: str, summary: str, citations: int | None = 0) -> Paper:
    return Paper(
        arxiv_id="2601.00001",
        title=title,
        authors=["A. Researcher"],
        summary=summary,
        published=datetime(2026, 1, 1, tzinfo=timezone.utc),
        citation_count=citations,
        abs_url="https://arxiv.org/abs/2601.00001",
    )


def test_scoring_prefers_transferable_novel_technique() -> None:
    technique_paper = _paper(
        "Latent Debate Steering for Compact Reasoning Models",
        (
            "We introduce a two-stage training pipeline that distills debate into a single model. "
            "To address inference cost, the method combines reward scheduling with activation steering "
            "and improves accuracy with fewer generated tokens."
        ),
    )
    generic_paper = _paper(
        "A Survey of Recent Reasoning Benchmarks",
        "This paper reviews benchmark trends and summarizes common evaluation protocols.",
    )

    scored = score_papers([technique_paper, generic_paper], now=datetime(2026, 2, 1, tzinfo=timezone.utc))

    assert scored[0].paper.title == technique_paper.title
    assert any("reward scheduling" in phrase for phrase in scored[0].techniques)
    assert scored[0].score > scored[1].score


def test_hiddenness_penalizes_high_citations() -> None:
    quiet = _paper("Novel Reward Routing Procedure", "We propose a new routing procedure.", citations=1)
    popular = _paper("Novel Reward Routing Procedure", "We propose a new routing procedure.", citations=500)

    quiet_score, popular_score = score_papers([quiet, popular], now=datetime(2026, 5, 1, tzinfo=timezone.utc))

    assert quiet_score.components["hiddenness"] > popular_score.components["hiddenness"]
    assert quiet_score.score > popular_score.score
