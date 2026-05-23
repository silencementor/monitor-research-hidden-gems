from __future__ import annotations

import re
import time
import xml.etree.ElementTree as ET
from datetime import datetime, timedelta, timezone
from urllib.parse import urlparse

import httpx

from research_hidden_gems.models import Paper

ARXIV_API_URL = "https://export.arxiv.org/api/query"
ATOM = "{http://www.w3.org/2005/Atom}"
ARXIV = "{http://arxiv.org/schemas/atom}"
ARXIV_ID_RE = re.compile(r"(?P<id>\d{4}\.\d{4,5})(?:v\d+)?")


class ArxivClient:
    def __init__(self, *, timeout: float = 30.0, polite_delay_seconds: float = 3.0) -> None:
        self.timeout = timeout
        self.polite_delay_seconds = polite_delay_seconds
        self._last_request_at = 0.0

    def search(
        self,
        *,
        query: str = "",
        categories: list[str] | None = None,
        days: int | None = 30,
        max_results: int = 100,
    ) -> list[Paper]:
        search_query = build_search_query(query=query, categories=categories or [], days=days)
        return self._request(
            {
                "search_query": search_query,
                "start": "0",
                "max_results": str(max_results),
                "sortBy": "submittedDate",
                "sortOrder": "descending",
            }
        )

    def get(self, arxiv_id: str) -> Paper:
        papers = self._request({"id_list": arxiv_id})
        if not papers:
            raise LookupError(f"No arXiv paper found for {arxiv_id}")
        return papers[0]

    def _request(self, params: dict[str, str]) -> list[Paper]:
        elapsed = time.monotonic() - self._last_request_at
        if self._last_request_at and elapsed < self.polite_delay_seconds:
            time.sleep(self.polite_delay_seconds - elapsed)

        with httpx.Client(timeout=self.timeout, follow_redirects=True) as client:
            response = client.get(ARXIV_API_URL, params=params)
            response.raise_for_status()
        self._last_request_at = time.monotonic()
        return parse_atom(response.text)


def build_search_query(*, query: str, categories: list[str], days: int | None) -> str:
    parts: list[str] = []
    cleaned_query = query.strip()
    if cleaned_query:
        if _looks_like_arxiv_query(cleaned_query):
            parts.append(f"({cleaned_query})")
        else:
            parts.append(f'all:"{cleaned_query}"')

    if categories:
        category_query = " OR ".join(f"cat:{category.strip()}" for category in categories if category.strip())
        if category_query:
            parts.append(f"({category_query})")

    if days is not None and days > 0:
        start = datetime.now(timezone.utc) - timedelta(days=days)
        start_token = start.strftime("%Y%m%d%H%M")
        parts.append(f"submittedDate:[{start_token} TO 999912312359]")

    return " AND ".join(parts) if parts else "all:*"


def parse_atom(xml_text: str) -> list[Paper]:
    root = ET.fromstring(xml_text)
    papers: list[Paper] = []
    for entry in root.findall(f"{ATOM}entry"):
        arxiv_id = _extract_arxiv_id(_text(entry, "id"))
        title = _normalize(_text(entry, "title"))
        summary = _normalize(_text(entry, "summary"))
        authors = [_normalize(author.findtext(f"{ATOM}name", default="")) for author in entry.findall(f"{ATOM}author")]
        published = _parse_datetime(_text(entry, "published"))
        updated_text = _text(entry, "updated")
        updated = _parse_datetime(updated_text) if updated_text else None
        categories = [node.attrib.get("term", "") for node in entry.findall(f"{ATOM}category")]
        abs_url = _text(entry, "id")
        pdf_url = None
        for link in entry.findall(f"{ATOM}link"):
            if link.attrib.get("title") == "pdf" or link.attrib.get("type") == "application/pdf":
                pdf_url = link.attrib.get("href")
                break

        comment = _optional_text(entry, f"{ARXIV}comment")
        doi = _optional_text(entry, f"{ARXIV}doi")
        papers.append(
            Paper(
                arxiv_id=arxiv_id,
                title=title,
                authors=[author for author in authors if author],
                summary=summary,
                published=published,
                updated=updated,
                categories=[category for category in categories if category],
                abs_url=abs_url,
                pdf_url=pdf_url,
                comment=comment,
                doi=doi,
            )
        )
    return papers


def arxiv_id_from_text(value: str) -> str:
    match = ARXIV_ID_RE.search(value)
    if match:
        return match.group("id")
    parsed = urlparse(value)
    if parsed.netloc.endswith("arxiv.org"):
        candidate = parsed.path.rsplit("/", maxsplit=1)[-1]
        match = ARXIV_ID_RE.search(candidate)
        if match:
            return match.group("id")
    raise ValueError(f"Could not find an arXiv id in {value!r}")


def _looks_like_arxiv_query(query: str) -> bool:
    return any(token in query for token in ("cat:", "ti:", "abs:", "au:", "all:", "AND", "OR", "NOT"))


def _extract_arxiv_id(url_or_id: str) -> str:
    return arxiv_id_from_text(url_or_id)


def _parse_datetime(value: str) -> datetime:
    normalized = value.replace("Z", "+00:00")
    return datetime.fromisoformat(normalized)


def _text(entry: ET.Element, tag: str) -> str:
    return entry.findtext(f"{ATOM}{tag}", default="")


def _optional_text(entry: ET.Element, tag: str) -> str | None:
    value = entry.findtext(tag)
    return _normalize(value) if value else None


def _normalize(value: str) -> str:
    return " ".join(value.split())
