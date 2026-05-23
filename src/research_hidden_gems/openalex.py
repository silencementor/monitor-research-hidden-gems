"""OpenAlex source: free citation enrichment, plus discovery of older-but-uncited
arXiv works (the truest hidden gems — old enough to be ignored, still low-cited).

arXiv papers carry a registered DOI ``10.48550/arXiv.<id>`` which OpenAlex
indexes, so enrichment is reliable without any API key.
"""

from __future__ import annotations

import re
from collections.abc import Iterable
from datetime import datetime, timedelta, timezone

import httpx

from research_hidden_gems.models import Paper

OPENALEX_URL = "https://api.openalex.org/works"
ARXIV_SOURCE_ID = "S4306400194"  # OpenAlex source id for arXiv
_ARXIV_DOI_RE = re.compile(r"arxiv\.(\d{4}\.\d{4,5})", re.I)
_ARXIV_ABS_RE = re.compile(r"arxiv\.org/abs/(\d{4}\.\d{4,5})", re.I)


def enrich_with_openalex(papers: Iterable[Paper], *, mailto: str | None = None, timeout: float = 20.0) -> None:
    """Fill citation counts from OpenAlex for any paper that has an arXiv id."""
    targets = [p for p in papers if p.arxiv_id]
    if not targets:
        return
    with httpx.Client(timeout=timeout, follow_redirects=True) as client:
        for batch in _chunks(targets, 50):
            dois = "|".join(f"10.48550/arxiv.{p.arxiv_id.lower()}" for p in batch)
            params = {
                "filter": f"doi:{dois}",
                "per-page": 50,
                "select": "id,doi,cited_by_count,referenced_works_count",
            }
            if mailto:
                params["mailto"] = mailto
            try:
                response = client.get(OPENALEX_URL, params=params)
                response.raise_for_status()
                results = response.json().get("results", [])
            except (httpx.HTTPError, ValueError):
                continue
            by_id = {}
            for work in results:
                aid = _arxiv_id_from_work(work)
                if aid:
                    by_id[aid] = work
            for paper in batch:
                work = by_id.get(paper.arxiv_id)
                if not work:
                    continue
                cites = work.get("cited_by_count")
                if isinstance(cites, int):
                    paper.citation_count = max(paper.citation_count or 0, cites)
                if work.get("id"):
                    paper.external_ids.setdefault("openalex", str(work["id"]))


def fetch_recent(
    *,
    days: int = 180,
    max_results: int = 100,
    search: str | None = None,
    mailto: str | None = None,
    timeout: float = 25.0,
) -> list[Paper]:
    """Discover arXiv works indexed by OpenAlex in a (typically wider) window.

    Using a window of months rather than days surfaces papers old enough to have
    been ignored — citations are attached, so the pipeline can flag low-cited ones.
    """
    from_date = (datetime.now(timezone.utc).date() - timedelta(days=max(1, days))).isoformat()
    filters = [f"primary_location.source.id:{ARXIV_SOURCE_ID}", f"from_publication_date:{from_date}"]
    if search:
        filters.append(f"title_and_abstract.search:{search}")
    params = {
        "filter": ",".join(filters),
        "sort": "publication_date:desc",
        "per-page": min(200, max_results),
        "select": "id,doi,title,abstract_inverted_index,publication_date,cited_by_count,authorships,primary_location",
    }
    if mailto:
        params["mailto"] = mailto

    papers: dict[str, Paper] = {}
    cursor = "*"
    with httpx.Client(timeout=timeout, follow_redirects=True) as client:
        while len(papers) < max_results and cursor:
            page_params = dict(params, cursor=cursor)
            try:
                response = client.get(OPENALEX_URL, params=page_params)
                response.raise_for_status()
                payload = response.json()
            except (httpx.HTTPError, ValueError):
                break
            for work in payload.get("results", []):
                paper = _work_to_paper(work)
                if paper and paper.arxiv_id not in papers:
                    papers[paper.arxiv_id] = paper
            cursor = (payload.get("meta") or {}).get("next_cursor")
    return list(papers.values())[:max_results]


def _work_to_paper(work: dict) -> Paper | None:
    arxiv_id = _arxiv_id_from_work(work)
    if not arxiv_id:
        return None
    authors = [
        (auth.get("author") or {}).get("display_name", "")
        for auth in work.get("authorships", [])
    ]
    return Paper(
        arxiv_id=arxiv_id,
        title=" ".join(str(work.get("title") or "").split()),
        authors=[a for a in authors if a],
        summary=_abstract_from_index(work.get("abstract_inverted_index")),
        published=_parse_date(work.get("publication_date")),
        categories=[],
        abs_url=f"https://arxiv.org/abs/{arxiv_id}",
        pdf_url=f"https://arxiv.org/pdf/{arxiv_id}",
        citation_count=work.get("cited_by_count") if isinstance(work.get("cited_by_count"), int) else None,
        source="openalex",
        external_ids={"openalex": str(work.get("id", ""))},
    )


def _arxiv_id_from_work(work: dict) -> str:
    match = _ARXIV_DOI_RE.search(work.get("doi") or "")
    if match:
        return match.group(1)
    landing = (work.get("primary_location") or {}).get("landing_page_url") or ""
    match = _ARXIV_ABS_RE.search(landing) or _ARXIV_DOI_RE.search(landing)
    return match.group(1) if match else ""


def _abstract_from_index(index: dict | None) -> str:
    if not index:
        return ""
    positions: list[tuple[int, str]] = []
    for word, locs in index.items():
        for loc in locs:
            positions.append((loc, word))
    positions.sort()
    return " ".join(word for _, word in positions)


def _parse_date(value: str | None) -> datetime:
    if not value:
        return datetime.now(timezone.utc)
    try:
        return datetime.fromisoformat(value).replace(tzinfo=timezone.utc)
    except ValueError:
        return datetime.now(timezone.utc)


def _chunks(items: list, size: int):
    for start in range(0, len(items), size):
        yield items[start : start + size]
