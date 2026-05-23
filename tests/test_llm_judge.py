from research_hidden_gems.config import Config
from research_hidden_gems.llm_judge import _extract_json, is_available


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
