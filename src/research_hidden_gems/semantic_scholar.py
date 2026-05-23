from __future__ import annotations

import os
from collections.abc import Iterable

import httpx

from research_hidden_gems.models import Paper

S2_BATCH_URL = "https://api.semanticscholar.org/graph/v1/paper/batch"
S2_FIELDS = "externalIds,url,venue,citationCount,influentialCitationCount,publicationDate"


def enrich_with_semantic_scholar(papers: Iterable[Paper], *, timeout: float = 20.0) -> None:
    paper_list = list(papers)
    if not paper_list:
        return

    headers = {}
    api_key = os.getenv("S2_API_KEY")
    if api_key:
        headers["x-api-key"] = api_key

    ids = [f"ARXIV:{paper.arxiv_id}" for paper in paper_list]
    try:
        with httpx.Client(timeout=timeout, headers=headers) as client:
            response = client.post(S2_BATCH_URL, params={"fields": S2_FIELDS}, json={"ids": ids})
            response.raise_for_status()
            payload = response.json()
    except (httpx.HTTPError, ValueError):
        return

    for paper, item in zip(paper_list, payload, strict=False):
        if not item:
            continue
        paper.citation_count = _as_int(item.get("citationCount"))
        paper.influential_citation_count = _as_int(item.get("influentialCitationCount"))
        paper.semantic_scholar_url = item.get("url")
        paper.venue = item.get("venue") or paper.venue
        external_ids = item.get("externalIds") or {}
        paper.external_ids.update({str(key): str(value) for key, value in external_ids.items() if value})


def _as_int(value: object) -> int | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    return None
