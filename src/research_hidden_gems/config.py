"""Configuration: interest profile, sources, categories, and scoring weights.

Defaults are baked in so the tool runs with zero setup. Override via a TOML file
(``./hidden_gems.toml`` or ``~/.config/research-hidden-gems/config.toml``) or a
handful of environment variables (``RHG_*``).
"""

from __future__ import annotations

import os
import tomllib
from dataclasses import dataclass, field, replace
from pathlib import Path

# The user's stated research interests, written as a profile the embedding model
# and the LLM judge both consume to score "would this technique transfer to me?".
DEFAULT_PROFILE = (
    "I work on large language models (LLMs) and agentic AI: tool-using and "
    "multi-agent systems, planning, memory, and reasoning. I also work on "
    "databases, data mining, knowledge discovery, information retrieval, "
    "recommender systems, and Big Data systems. I am hunting for novel, "
    "underappreciated *techniques* (and novel problem formulations) that I can "
    "transfer into these areas to write new research papers: new training or "
    "inference recipes, retrieval/indexing mechanisms, ranking and recommendation "
    "methods, agent architectures, evaluation protocols, and scalable data "
    "algorithms. I care about transferable method ingredients, not narrow "
    "benchmark wins."
)

# Keyword boosts layered on top of embedding relevance.
DEFAULT_KEYWORDS = [
    "language model", "llm", "agent", "agentic", "multi-agent", "tool use",
    "retrieval", "rag", "information retrieval", "ranking", "recommender",
    "recommendation", "database", "query", "indexing", "vector search",
    "data mining", "knowledge discovery", "knowledge graph", "big data",
    "streaming", "distributed", "scalable", "embedding", "reasoning",
    "planning", "memory", "in-context", "fine-tuning", "evaluation",
]

# arXiv categories spanning the user's domains.
#   cs.LG/cs.AI/cs.CL  -> ML / LLMs / NLP
#   cs.IR              -> information retrieval & recommenders
#   cs.DB              -> databases
#   cs.DC / cs.DS      -> distributed / Big Data, algorithms & data structures
#   cs.MA              -> multi-agent / agentic
#   cs.SI              -> social & information networks (data mining)
DEFAULT_CATEGORIES = [
    "cs.LG", "cs.AI", "cs.CL", "cs.IR", "cs.DB", "cs.DC", "cs.MA", "cs.SI", "stat.ML",
]

# Sources to pull from. arXiv is the spine; the rest broaden discovery and the
# popularity signal. See sources/ for each fetcher.
DEFAULT_SOURCES = ["arxiv", "huggingface_daily", "openalex", "openreview"]

_CONFIG_SEARCH_PATHS = [
    Path("hidden_gems.toml"),
    Path.home() / ".config" / "research-hidden-gems" / "config.toml",
]


@dataclass(slots=True)
class Config:
    profile: str = DEFAULT_PROFILE
    keywords: list[str] = field(default_factory=lambda: list(DEFAULT_KEYWORDS))
    categories: list[str] = field(default_factory=lambda: list(DEFAULT_CATEGORIES))
    sources: list[str] = field(default_factory=lambda: list(DEFAULT_SOURCES))

    # LLM judge
    judge_enabled: bool = True
    judge_model: str = "claude-sonnet-4-6"
    judge_top: int = 15            # how many top-prefiltered papers to deep-judge
    judge_max_tokens: int = 900

    # embeddings
    embed_backend: str = "auto"    # auto | sentence-transformers | hashing
    embed_model: str = "sentence-transformers/all-MiniLM-L6-v2"

    # prefilter blend weights (lexical from Codex + new embedding/relevance signals)
    w_lexical: float = 0.30
    w_outlier: float = 0.25
    w_hiddenness: float = 0.20
    w_relevance: float = 0.25

    # final blend when an LLM verdict exists
    w_final_prefilter: float = 0.45
    w_final_llm: float = 0.55

    # relevance acts as a soft gate: final *= (relevance_floor + (1-floor)*relevance)
    relevance_floor: float = 0.35

    # optional deep mathematical-novelty assessment (uses math-skills as reference)
    math_depth: bool = False
    math_skills_path: str | None = None

    # OpenAlex polite-pool contact (recommended, not required)
    openalex_mailto: str | None = None

    @classmethod
    def load(cls, path: str | Path | None = None) -> "Config":
        cfg = cls()
        data = _read_toml(path)
        if data:
            cfg = _apply_toml(cfg, data)
        return _apply_env(cfg)


def _read_toml(path: str | Path | None) -> dict:
    candidates = [Path(path)] if path else _CONFIG_SEARCH_PATHS
    for candidate in candidates:
        if candidate.is_file():
            with candidate.open("rb") as handle:
                return tomllib.load(handle)
    return {}


def _apply_toml(cfg: Config, data: dict) -> Config:
    fields = {f for f in cfg.__slots__}  # type: ignore[attr-defined]
    updates = {key: value for key, value in data.items() if key in fields}
    return replace(cfg, **updates)


def _apply_env(cfg: Config) -> Config:
    updates: dict = {}
    if (val := os.getenv("RHG_JUDGE_MODEL")):
        updates["judge_model"] = val
    if (val := os.getenv("RHG_EMBED_BACKEND")):
        updates["embed_backend"] = val
    if (val := os.getenv("RHG_MATH_SKILLS_PATH")):
        updates["math_skills_path"] = val
        updates["math_depth"] = True
    if (val := os.getenv("RHG_OPENALEX_MAILTO")):
        updates["openalex_mailto"] = val
    if (val := os.getenv("RHG_JUDGE")) is not None:
        updates["judge_enabled"] = val.strip().lower() not in {"0", "false", "no", "off"}
    return replace(cfg, **updates) if updates else cfg
