from datetime import datetime, timezone

from research_hidden_gems.config import Config
from research_hidden_gems.models import LLMVerdict, Paper


def _paper(**kwargs) -> Paper:
    base = dict(
        arxiv_id="2601.00001",
        title="T",
        authors=[],
        summary="s",
        published=datetime(2026, 1, 1, tzinfo=timezone.utc),
    )
    base.update(kwargs)
    return Paper(**base)


def test_llmverdict_from_dict_clamps_and_coerces() -> None:
    verdict = LLMVerdict.from_dict(
        {"novelty": 5, "transferability": -1, "is_hidden_gem": True, "technique": "X", "confidence": "0.4"}
    )
    assert verdict.novelty == 1.0
    assert verdict.transferability == 0.0
    assert verdict.confidence == 0.4
    assert verdict.is_hidden_gem is True
    assert verdict.technique == "X"


def test_paper_key_prefers_arxiv_then_doi_then_url() -> None:
    assert _paper().key == "2601.00001"
    assert _paper(arxiv_id="", doi="10.1/x").key == "10.1/x"
    assert _paper(arxiv_id="", doi=None, abs_url="https://openreview.net/forum?id=abc").key == (
        "https://openreview.net/forum?id=abc"
    )


def test_scored_paper_dict_includes_source() -> None:
    from research_hidden_gems.scoring import score_papers

    scored = score_papers([_paper(source="openalex")])

    assert scored[0].to_dict()["source"] == "openalex"


def test_config_env_override(monkeypatch) -> None:
    monkeypatch.setenv("RHG_JUDGE_MODEL", "claude-test")
    monkeypatch.setenv("RHG_JUDGE", "off")
    monkeypatch.setenv("RHG_EMBED_BACKEND", "hashing")
    cfg = Config.load(None)
    assert cfg.judge_model == "claude-test"
    assert cfg.judge_enabled is False
    assert cfg.embed_backend == "hashing"


def test_config_defaults_cover_user_domains() -> None:
    cfg = Config()
    for category in ("cs.IR", "cs.DB", "cs.MA"):
        assert category in cfg.categories
    assert set(cfg.sources) >= {"arxiv", "huggingface_daily", "openalex", "openreview"}
