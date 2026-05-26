from __future__ import annotations

import json
import re
import time
from datetime import datetime
from pathlib import Path
from typing import Annotated, Literal, Optional

import typer
from rich.console import Console
from rich.table import Table

from research_hidden_gems.arxiv_client import ArxivClient, arxiv_id_from_text
from research_hidden_gems.config import Config, load_env_files
from research_hidden_gems.llm_judge import is_available
from research_hidden_gems.models import ScoredPaper
from research_hidden_gems.openalex import enrich_with_openalex
from research_hidden_gems.paths import configure_project_cache, default_report_dir
from research_hidden_gems.pipeline import rank_papers, run_pipeline
from research_hidden_gems.semantic_scholar import enrich_with_semantic_scholar
from research_hidden_gems.storage import SeenStore, default_state_path

OutputFormat = Literal["table", "json", "markdown"]

app = typer.Typer(no_args_is_help=True, help="Find under-the-radar papers with transferable novel techniques.")
console = Console()
load_env_files()
configure_project_cache()

# --- shared option types -----------------------------------------------------
QueryOpt = Annotated[str, typer.Option("--query", "-q", help="arXiv query or plain phrase. Also enables OpenAlex discovery.")]
CategoriesOpt = Annotated[Optional[str], typer.Option("--categories", "-c", help="Comma-separated arXiv categories (overrides config).")]
SourcesOpt = Annotated[Optional[str], typer.Option("--sources", help="Comma-separated: arxiv,huggingface_daily,openalex,openreview.")]
ProfileOpt = Annotated[Optional[str], typer.Option("--profile", help="Interest profile text, or @path/to/file.txt (overrides config).")]
NoLlmOpt = Annotated[bool, typer.Option("--no-llm", help="Skip the LLM deep-judge; rank on heuristics + embeddings only.")]
ProviderOpt = Annotated[Optional[str], typer.Option("--provider", help="Judge provider: auto | anthropic | openai.")]
ModelOpt = Annotated[Optional[str], typer.Option("--model", help="Judge model, e.g. claude-... or gpt-... (default per provider).")]
JudgeTopOpt = Annotated[Optional[int], typer.Option("--judge-top", help="How many top-prefiltered papers to deep-judge.")]
MathDepthOpt = Annotated[bool, typer.Option("--math-depth", help="Ask the judge to also weigh mathematical novelty/rigor.")]
MathPathOpt = Annotated[Optional[str], typer.Option("--math-skills-path", help="Path to a math-skills repo to ground deep math assessment.")]
ConfigOpt = Annotated[Optional[Path], typer.Option("--config", help="Path to a TOML config file.")]


@app.command()
def search(
    query: QueryOpt = "",
    categories: CategoriesOpt = None,
    days: Annotated[int, typer.Option("--days", help="Only consider papers submitted in this many recent days.")] = 30,
    max_results: Annotated[int, typer.Option("--max-results", help="Papers to fetch per source before ranking.")] = 120,
    top_k: Annotated[int, typer.Option("--top-k", help="Number of ranked papers to display.")] = 20,
    threshold: Annotated[float, typer.Option("--threshold", help="Minimum hidden-gem score (0-100) to keep.")] = 0.0,
    output: Annotated[OutputFormat, typer.Option("--format", help="Output format.")] = "table",
    enrich: Annotated[bool, typer.Option("--enrich/--no-enrich", help="Fetch citation counts (OpenAlex + Semantic Scholar).")] = True,
    no_llm: NoLlmOpt = False,
    provider: ProviderOpt = None,
    model: ModelOpt = None,
    judge_top: JudgeTopOpt = None,
    math_depth: MathDepthOpt = False,
    math_skills_path: MathPathOpt = None,
    profile: ProfileOpt = None,
    sources: SourcesOpt = None,
    config_path: ConfigOpt = None,
) -> None:
    """Search multiple sources and rank papers by hidden-gem potential."""
    cfg = _build_config(config_path, categories, profile, sources, no_llm, model, provider, judge_top, math_depth, math_skills_path)
    scored = run_pipeline(
        cfg, query=query, days=days, max_results=max_results, enrich=enrich, do_judge=not no_llm
    )
    kept = [item for item in scored if item.score >= threshold][:top_k]
    _render(kept, output=output, cfg=cfg)


@app.command()
def monitor(
    query: QueryOpt = "",
    categories: CategoriesOpt = None,
    days: Annotated[int, typer.Option("--days", help="Lookback window per monitor run.")] = 7,
    max_results: Annotated[int, typer.Option("--max-results", help="Papers to fetch per source before ranking.")] = 100,
    top_k: Annotated[int, typer.Option("--top-k", help="Maximum new papers to display per run.")] = 20,
    threshold: Annotated[float, typer.Option("--threshold", help="Minimum hidden-gem score (0-100) to notify on.")] = 0.0,
    state: Annotated[Path, typer.Option("--state", help="SQLite file for seen-paper state.")] = default_state_path(),
    out: Annotated[Optional[Path], typer.Option("--out", help="Also write a markdown digest of new papers to this file.")] = None,
    report_dir: Annotated[Path, typer.Option("--report-dir", help="Directory for timestamped markdown reports.")] = default_report_dir(),
    reports: Annotated[bool, typer.Option("--reports/--no-reports", help="Write one timestamped markdown report per monitor run.")] = True,
    output: Annotated[OutputFormat, typer.Option("--format", help="Output format.")] = "table",
    enrich: Annotated[bool, typer.Option("--enrich/--no-enrich", help="Fetch citation counts (OpenAlex + Semantic Scholar).")] = True,
    no_llm: NoLlmOpt = False,
    provider: ProviderOpt = None,
    model: ModelOpt = None,
    judge_top: JudgeTopOpt = None,
    math_depth: MathDepthOpt = False,
    math_skills_path: MathPathOpt = None,
    profile: ProfileOpt = None,
    sources: SourcesOpt = None,
    config_path: ConfigOpt = None,
    interval_minutes: Annotated[Optional[float], typer.Option("--interval-minutes", help="Repeat forever every N minutes.")] = None,
) -> None:
    """Show only papers not already seen by this monitor (cron- or loop-friendly)."""
    cfg = _build_config(config_path, categories, profile, sources, no_llm, model, provider, judge_top, math_depth, math_skills_path)
    store = SeenStore(state)

    def run_once() -> None:
        scored = run_pipeline(
            cfg, query=query, days=days, max_results=max_results, enrich=enrich, do_judge=not no_llm
        )
        kept = [item for item in scored if item.score >= threshold]
        unseen = store.unseen(kept)
        store.upsert(kept)
        new_top = unseen[:top_k]
        _render(new_top, output=output, cfg=cfg)
        if out is not None:
            _write_markdown(out, new_top)
        if reports:
            report_path = _write_timestamped_report(report_dir, query, new_top)
            console.print(f"[dim]Wrote markdown report to {report_path}[/dim]")

    if interval_minutes is None:
        run_once()
        return
    while True:
        run_once()
        time.sleep(interval_minutes * 60)


@app.command()
def inspect(
    paper: Annotated[str, typer.Argument(help="arXiv id or URL, e.g. https://arxiv.org/abs/2604.24881")],
    output: Annotated[OutputFormat, typer.Option("--format", help="Output format.")] = "markdown",
    enrich: Annotated[bool, typer.Option("--enrich/--no-enrich", help="Fetch citation counts (OpenAlex + Semantic Scholar).")] = True,
    no_llm: NoLlmOpt = False,
    provider: ProviderOpt = None,
    model: ModelOpt = None,
    math_depth: MathDepthOpt = False,
    math_skills_path: MathPathOpt = None,
    config_path: ConfigOpt = None,
) -> None:
    """Explain why a single paper does or does not look like a hidden gem."""
    cfg = _build_config(config_path, None, None, None, no_llm, model, provider, None, math_depth, math_skills_path)
    arxiv_id = arxiv_id_from_text(paper)
    fetched = ArxivClient().get(arxiv_id)
    if enrich:
        enrich_with_openalex([fetched], mailto=cfg.openalex_mailto)
        enrich_with_semantic_scholar([fetched])
    scored = rank_papers(cfg, [fetched], do_judge=not no_llm)
    _render(scored, output=output, cfg=cfg)


# ----------------------------------------------------------------- config ---
def _build_config(
    config_path: Optional[Path],
    categories: Optional[str],
    profile: Optional[str],
    sources: Optional[str],
    no_llm: bool,
    model: Optional[str],
    provider: Optional[str],
    judge_top: Optional[int],
    math_depth: bool,
    math_skills_path: Optional[str],
) -> Config:
    cfg = Config.load(config_path)
    if categories:
        cfg.categories = _split_csv(categories)
    if sources:
        cfg.sources = _split_csv(sources)
    if profile:
        cfg.profile = _load_profile(profile)
    if no_llm:
        cfg.judge_enabled = False
    if provider:
        cfg.judge_provider = provider
    if model:
        cfg.judge_model = model
    if judge_top is not None:
        cfg.judge_top = judge_top
    if math_depth:
        cfg.math_depth = True
    if math_skills_path:
        cfg.math_skills_path = math_skills_path
        cfg.math_depth = True
    return cfg


def _load_profile(value: str) -> str:
    if value.startswith("@"):
        return Path(value[1:]).read_text(encoding="utf-8").strip()
    return value


# ----------------------------------------------------------------- render ---
def _render(papers: list[ScoredPaper], *, output: OutputFormat, cfg: Config) -> None:
    if output == "json":
        console.print_json(json.dumps([paper.to_dict() for paper in papers]))
        return
    if output == "markdown":
        console.print(_markdown(papers))
        return
    _table(papers, cfg=cfg)


def _judge_note(cfg: Config) -> str | None:
    if cfg.judge_enabled and not is_available(cfg):
        return (
            "[dim]LLM judge enabled but unavailable. Install the 'llm' extra and set "
            "ANTHROPIC_API_KEY or OPENAI_API_KEY (or pass --no-llm). "
            "Ranking on heuristics + embeddings only.[/dim]"
        )
    return None


def _table(papers: list[ScoredPaper], *, cfg: Config) -> None:
    note = _judge_note(cfg)
    if note:
        console.print(note)
    if not papers:
        console.print("[yellow]No papers matched the current filters.[/yellow]")
        return

    table = Table(title="Research Hidden Gems", show_lines=True)
    table.add_column("Score", justify="right", no_wrap=True)
    table.add_column("Rel", justify="right", no_wrap=True)
    table.add_column("Gem", justify="center", no_wrap=True)
    table.add_column("Title", ratio=3)
    table.add_column("Technique", ratio=2)
    table.add_column("Cites", justify="right", no_wrap=True)
    table.add_column("arXiv", no_wrap=True)

    for scored in papers:
        paper = scored.paper
        c = scored.components
        technique = scored.verdict.technique if scored.verdict and scored.verdict.technique else ", ".join(scored.techniques[:2])
        table.add_row(
            f"{scored.score:.1f}",
            f"{c.get('relevance', 0.0):.2f}",
            _gem_mark(scored),
            paper.title,
            technique or "-",
            "?" if paper.citation_count is None else str(paper.citation_count),
            paper.arxiv_id or paper.key,
        )
    console.print(table)


def _gem_mark(scored: ScoredPaper) -> str:
    if scored.verdict is None:
        return "[dim]?[/dim]"
    return "[green]✔[/green]" if scored.verdict.is_hidden_gem else "[dim]·[/dim]"


def _markdown(papers: list[ScoredPaper]) -> str:
    if not papers:
        return "No papers matched the current filters."
    blocks: list[str] = []
    for rank, scored in enumerate(papers, start=1):
        paper = scored.paper
        c = scored.components
        url = paper.abs_url or f"https://arxiv.org/abs/{paper.arxiv_id}"
        lines = [
            f"## {rank}. {paper.title}",
            "",
            f"- **Score:** {scored.score:.1f}  (relevance {c.get('relevance', 0.0):.2f}, "
            f"outlier-novelty {c.get('outlier', 0.0):.2f}, hiddenness {c.get('hiddenness', 0.0):.2f})",
            f"- **Link:** [{paper.arxiv_id or paper.key}]({url})  · source: {paper.source}",
            f"- **Published:** {paper.published.date().isoformat()}  · "
            f"**Citations:** {'unknown' if paper.citation_count is None else paper.citation_count}",
        ]
        if scored.verdict is not None:
            v = scored.verdict
            lines += [
                f"- **LLM verdict:** {'HIDDEN GEM' if v.is_hidden_gem else 'not flagged'} "
                f"(novelty {v.novelty:.2f}, transferability {v.transferability:.2f}, conf {v.confidence:.2f})",
                f"- **Technique:** {v.technique or 'n/a'}",
                f"- **In one line:** {v.one_liner or 'n/a'}",
                f"- **Why overlooked:** {v.why_overlooked or 'n/a'}",
                f"- **Apply to your work:** {v.application_to_user or 'n/a'}",
            ]
        else:
            lines += [
                f"- **Techniques:** {', '.join(scored.techniques[:8]) or 'none extracted'}",
                f"- **Problems:** {' | '.join(scored.problems[:2]) or 'none extracted'}",
                f"- **Rare terms:** {', '.join(scored.rare_terms[:8]) or 'none'}",
            ]
        blocks.append("\n".join(lines))
    return "\n\n".join(blocks)


def _write_markdown(path: Path, papers: list[ScoredPaper]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(_markdown(papers) + "\n", encoding="utf-8")
    return path


def _write_timestamped_report(report_dir: Path, query: str, papers: list[ScoredPaper]) -> Path:
    label = _slug(query) or "all"
    stamp = datetime.now().astimezone().strftime("%Y%m%d-%H%M%S")
    return _write_markdown(report_dir / f"{stamp}-{label}.md", papers)


def _slug(value: str, max_length: int = 80) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", value.lower()).strip("-")
    return slug[:max_length].strip("-")


def _split_csv(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def main() -> None:
    load_env_files()
    configure_project_cache()
    app()


if __name__ == "__main__":
    app()
