import os
from pathlib import Path

from research_hidden_gems.paths import (
    configure_project_cache,
    default_cache_dir,
    default_report_dir,
    default_state_path,
    project_root,
)


def test_project_paths_default_to_working_tree(tmp_path, monkeypatch) -> None:
    (tmp_path / "pyproject.toml").write_text("[project]\nname = 'demo'\n", encoding="utf-8")
    nested = tmp_path / "src" / "pkg"
    nested.mkdir(parents=True)
    monkeypatch.chdir(nested)
    monkeypatch.delenv("RHG_PROJECT_DIR", raising=False)
    monkeypatch.delenv("RHG_RUNTIME_DIR", raising=False)
    monkeypatch.delenv("RHG_CACHE_DIR", raising=False)
    monkeypatch.delenv("RHG_STATE_PATH", raising=False)
    monkeypatch.delenv("RHG_REPORT_DIR", raising=False)

    assert project_root() == tmp_path
    assert default_state_path() == tmp_path / ".hidden-gems" / "state" / "seen.sqlite3"
    assert default_cache_dir() == tmp_path / ".hidden-gems" / "cache"
    assert default_report_dir() == tmp_path / "reports" / "hidden-gems"


def test_project_paths_allow_env_overrides(tmp_path, monkeypatch) -> None:
    runtime = tmp_path / "runtime"
    cache = tmp_path / "cache"
    state = tmp_path / "db" / "seen.sqlite3"
    reports = tmp_path / "reports"
    monkeypatch.setenv("RHG_RUNTIME_DIR", str(runtime))
    monkeypatch.setenv("RHG_CACHE_DIR", str(cache))
    monkeypatch.setenv("RHG_STATE_PATH", str(state))
    monkeypatch.setenv("RHG_REPORT_DIR", str(reports))

    assert default_state_path() == state
    assert default_cache_dir() == cache
    assert default_report_dir() == reports


def test_configure_project_cache_sets_local_cache_env(tmp_path, monkeypatch) -> None:
    cache = tmp_path / "cache"
    monkeypatch.setenv("RHG_CACHE_DIR", str(cache))
    for key in (
        "XDG_CACHE_HOME",
        "HF_HOME",
        "HUGGINGFACE_HUB_CACHE",
        "SENTENCE_TRANSFORMERS_HOME",
        "TRANSFORMERS_CACHE",
        "TORCH_HOME",
    ):
        monkeypatch.delenv(key, raising=False)

    configure_project_cache()

    assert Path(os.environ["XDG_CACHE_HOME"]) == cache / "xdg"
    assert Path(os.environ["HF_HOME"]) == cache / "huggingface"
    assert Path(os.environ["SENTENCE_TRANSFORMERS_HOME"]) == cache / "sentence-transformers"
