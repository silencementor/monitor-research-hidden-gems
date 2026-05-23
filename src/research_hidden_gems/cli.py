from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Annotated, Literal

import typer
from rich.console import Console
from rich.table import Table

from research_hidden_gems.arxiv_client import ArxivClient, arxiv_id_from_text
from research_hidden_gems.models import ScoredPaper
from research_hidden_gems.scoring import score_papers
from research_hidden_gems.semantic_scholar import enrich_with_semantic_scholar
from research_hidden_gems.storage import SeenStore, default_state_path

DEFAULT_CATEGORIES = "cs.AI,cs.CL,cs.LG,stat.ML"
OutputFormat = Literal["table", "json", "markdown"]

app = typer.Typer(no_args_is_help=True, help="Find under-the-radar papers with transferable novel techniques.")
console = Console()


@app.command()
def search(
    query: Annotated[str, typer.Option("--query", "-q", help="arXiv query or plain phrase.")] = "",
    categories: Annotated[str, typer.Option("--categories", "-c", help="Comma-separated arXiv categories.")] = DEFAULT_CATEGORIES,
    days: Annotated[int, typer.Option("--days", help="Only search papers submitted in this many recent days.")] = 30,
    max_results: Annotated[int, typer.Option("--max-results", help="Number of arXiv results to fetch before ranking.")] = 100,
    top_k: Annotated[int, typer.Option("--top-k", help="Number of ranked papers to display.")] = 20,
    threshold: Annotated[float, typer.Option("--threshold", help="Minimum hidden-gem score to keep.")] = 45.0,
    output: Annotated[OutputFormat, typer.Option("--format", help="Output format.")] = "table",
    enrich: Annotated[bool, typer.Option("--enrich/--no-enrich", help="Fetch citation counts from Semantic Scholar.")] = True,
) -> None:
    """Search arXiv and rank papers by hidden-gem potential."""
    scored = _rank(
        query=query,
        categories=_split_csv(categories),
        days=days,
        max_results=max_results,
        enrich=enrich,
        threshold=threshold,
    )
    _render(scored[:top_k], output=output)


@app.command()
def monitor(
    query: Annotated[str, typer.Option("--query", "-q", help="arXiv query or plain phrase.")] = "",
    categories: Annotated[str, typer.Option("--categories", "-c", help="Comma-separated arXiv categories.")] = DEFAULT_CATEGORIES,
    days: Annotated[int, typer.Option("--days", help="Lookback window per monitor run.")] = 7,
    max_results: Annotated[int, typer.Option("--max-results", help="Number of arXiv results to fetch before ranking.")] = 80,
    top_k: Annotated[int, typer.Option("--top-k", help="Maximum new papers to display per run.")] = 20,
    threshold: Annotated[float, typer.Option("--threshold", help="Minimum hidden-gem score to notify on.")] = 50.0,
    state: Annotated[Path, typer.Option("--state", help="SQLite file for seen-paper state.")] = default_state_path(),
    output: Annotated[OutputFormat, typer.Option("--format", help="Output format.")] = "table",
    enrich: Annotated[bool, typer.Option("--enrich/--no-enrich", help="Fetch citation counts from Semantic Scholar.")] = True,
    interval_minutes: Annotated[float | None, typer.Option("--interval-minutes", help="Repeat forever every N minutes.")] = None,
) -> None:
    """Show only papers that have not already been seen by this monitor."""
    store = SeenStore(state)

    def run_once() -> None:
        scored = _rank(
            query=query,
            categories=_split_csv(categories),
            days=days,
            max_results=max_results,
            enrich=enrich,
            threshold=threshold,
        )
        unseen = store.unseen(scored)
        store.upsert(scored)
        _render(unseen[:top_k], output=output)

    if interval_minutes is None:
        run_once()
        return

    while True:
        run_once()
        time.sleep(interval_minutes * 60)


@app.command()
def inspect(
    paper: Annotated[str, typer.Argument(help="arXiv id or URL, for example https://arxiv.org/abs/2604.24881")],
    output: Annotated[OutputFormat, typer.Option("--format", help="Output format.")] = "markdown",
    enrich: Annotated[bool, typer.Option("--enrich/--no-enrich", help="Fetch citation counts from Semantic Scholar.")] = True,
) -> None:
    """Explain why a single paper does or does not look like a hidden gem."""
    arxiv_id = arxiv_id_from_text(paper)
    fetched = ArxivClient().get(arxiv_id)
    if enrich:
        enrich_with_semantic_scholar([fetched])
    scored = score_papers([fetched])
    _render(scored, output=output)


def _rank(
    *,
    query: str,
    categories: list[str],
    days: int,
    max_results: int,
    enrich: bool,
    threshold: float,
) -> list[ScoredPaper]:
    papers = ArxivClient().search(query=query, categories=categories, days=days, max_results=max_results)
    if enrich:
        enrich_with_semantic_scholar(papers)
    scored = score_papers(papers)
    return [paper for paper in scored if paper.score >= threshold]


def _render(papers: list[ScoredPaper], *, output: OutputFormat) -> None:
    if output == "json":
        console.print_json(json.dumps([paper.to_dict() for paper in papers]))
        return
    if output == "markdown":
        console.print(_markdown(papers))
        return
    _table(papers)


def _table(papers: list[ScoredPaper]) -> None:
    if not papers:
        console.print("[yellow]No papers matched the current filters.[/yellow]")
        return

    table = Table(title="Research Hidden Gems", show_lines=True)
    table.add_column("Score", justify="right", no_wrap=True)
    table.add_column("Hidden", justify="right", no_wrap=True)
    table.add_column("Title")
    table.add_column("Techniques")
    table.add_column("Cites", justify="right", no_wrap=True)
    table.add_column("arXiv", no_wrap=True)

    for scored in papers:
        paper = scored.paper
        table.add_row(
            f"{scored.score:.1f}",
            f"{scored.components['hiddenness']:.2f}",
            paper.title,
            ", ".join(scored.techniques[:3]) or "-",
            "?" if paper.citation_count is None else str(paper.citation_count),
            paper.abs_url or f"https://arxiv.org/abs/{paper.arxiv_id}",
        )
    console.print(table)


def _markdown(papers: list[ScoredPaper]) -> str:
    if not papers:
        return "No papers matched the current filters."
    blocks: list[str] = []
    for rank, scored in enumerate(papers, start=1):
        paper = scored.paper
        url = paper.abs_url or f"https://arxiv.org/abs/{paper.arxiv_id}"
        blocks.append(
            "\n".join(
                [
                    f"## {rank}. {paper.title}",
                    "",
                    f"- Score: {scored.score:.1f}",
                    f"- arXiv: [{paper.arxiv_id}]({url})",
                    f"- Published: {paper.published.date().isoformat()}",
                    f"- Citations: {'unknown' if paper.citation_count is None else paper.citation_count}",
                    f"- Techniques: {', '.join(scored.techniques[:10]) or 'none extracted'}",
                    f"- Problems: {' | '.join(scored.problems[:2]) or 'none extracted'}",
                    f"- Rare terms: {', '.join(scored.rare_terms[:8]) or 'none'}",
                    "- Rationale:",
                    *[f"  - {reason}" for reason in scored.rationale],
                ]
            )
        )
    return "\n\n".join(blocks)


def _split_csv(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


if __name__ == "__main__":
    app()
