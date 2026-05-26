from __future__ import annotations

import os
from pathlib import Path


def project_root(start: Path | str | None = None) -> Path:
    if configured := os.getenv("RHG_PROJECT_DIR"):
        return Path(configured).expanduser().resolve()

    current = Path(start) if start is not None else Path.cwd()
    current = current.expanduser().resolve()
    for candidate in (current, *current.parents):
        if (candidate / "pyproject.toml").is_file() or (candidate / ".git").exists():
            return candidate
    return current


def runtime_dir() -> Path:
    if configured := os.getenv("RHG_RUNTIME_DIR"):
        return Path(configured).expanduser().resolve()
    return project_root() / ".hidden-gems"


def default_cache_dir() -> Path:
    if configured := os.getenv("RHG_CACHE_DIR"):
        return Path(configured).expanduser().resolve()
    return runtime_dir() / "cache"


def default_state_path() -> Path:
    if configured := os.getenv("RHG_STATE_PATH"):
        return Path(configured).expanduser().resolve()
    return runtime_dir() / "state" / "seen.sqlite3"


def default_report_dir() -> Path:
    if configured := os.getenv("RHG_REPORT_DIR"):
        return Path(configured).expanduser().resolve()
    return project_root() / "reports" / "hidden-gems"


def configure_project_cache() -> None:
    cache_dir = default_cache_dir()
    cache_dir.mkdir(parents=True, exist_ok=True)

    defaults = {
        "XDG_CACHE_HOME": cache_dir / "xdg",
        "HF_HOME": cache_dir / "huggingface",
        "HUGGINGFACE_HUB_CACHE": cache_dir / "huggingface" / "hub",
        "SENTENCE_TRANSFORMERS_HOME": cache_dir / "sentence-transformers",
        "TRANSFORMERS_CACHE": cache_dir / "huggingface" / "transformers",
        "TORCH_HOME": cache_dir / "torch",
    }
    for key, path in defaults.items():
        if key not in os.environ:
            path.mkdir(parents=True, exist_ok=True)
            os.environ[key] = str(path)
