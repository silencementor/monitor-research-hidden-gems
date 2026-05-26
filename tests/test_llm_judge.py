from research_hidden_gems.config import Config
from research_hidden_gems.llm_judge import (
    _extract_json,
    _infer_provider,
    _resolve_provider,
    is_available,
)


def test_extract_json_plain() -> None:
    assert _extract_json('{"novelty": 0.5}') == {"novelty": 0.5}


def test_extract_json_fenced() -> None:
    text = "Here is the verdict:\n```json\n{\"a\": 1, \"b\": 2}\n```\nthanks"
    assert _extract_json(text) == {"a": 1, "b": 2}


def test_extract_json_embedded() -> None:
    assert _extract_json('preface {"a": 2} trailing') == {"a": 2}


def test_extract_json_none() -> None:
    assert _extract_json("there is no json object here") is None


def test_is_available_false_without_key(monkeypatch) -> None:
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("ANTHROPIC_AUTH_TOKEN", raising=False)
    assert is_available(Config(judge_enabled=True)) is False


def test_is_available_false_when_disabled() -> None:
    assert is_available(Config(judge_enabled=False)) is False


def test_infer_provider_from_model_name() -> None:
    assert _infer_provider("gpt-4.1") == "openai"
    assert _infer_provider("o3-mini") == "openai"
    assert _infer_provider("claude-sonnet-4-6") == "anthropic"
    assert _infer_provider("") is None
    assert _infer_provider("mystery-model") is None


def test_resolve_forced_provider_uses_its_default_model() -> None:
    p_oai, m_oai = _resolve_provider(Config(judge_provider="openai"))
    assert (p_oai, m_oai) == ("openai", "gpt-4.1")
    p_ant, m_ant = _resolve_provider(Config(judge_provider="anthropic"))
    assert (p_ant, m_ant) == ("anthropic", "claude-sonnet-4-6")


def test_resolve_auto_infers_provider_from_model() -> None:
    assert _resolve_provider(Config(judge_provider="auto", judge_model="gpt-4.1")) == ("openai", "gpt-4.1")
    assert _resolve_provider(Config(judge_provider="auto", judge_model="claude-x")) == ("anthropic", "claude-x")


def test_is_available_false_without_openai_key(monkeypatch) -> None:
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    assert is_available(Config(judge_enabled=True, judge_provider="openai")) is False
