"""OpenAlex source: free citation enrichment, venue discovery, and discovery of
older-but-uncited arXiv works (the truest hidden gems — old enough to be ignored,
still low-cited).

arXiv papers carry a registered DOI ``10.48550/arXiv.<id>`` which OpenAlex
indexes, so enrichment is reliable without any API key.
"""

from __future__ import annotations

import re
from collections.abc import Iterable
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone

import httpx

from research_hidden_gems.models import Paper

OPENALEX_URL = "https://api.openalex.org/works"
OPENALEX_SOURCES_URL = "https://api.openalex.org/sources"
ARXIV_SOURCE_ID = "S4306400194"  # OpenAlex source id for arXiv
_ARXIV_DOI_RE = re.compile(r"arxiv\.(\d{4}\.\d{4,5})", re.I)
_ARXIV_ABS_RE = re.compile(r"arxiv\.org/abs/(\d{4}\.\d{4,5})", re.I)
_ARXIV_PDF_RE = re.compile(r"arxiv\.org/pdf/(\d{4}\.\d{4,5})", re.I)
_OPENALEX_SOURCE_RE = re.compile(r"/(S\d+)$")
_DOI_URL_RE = re.compile(r"^https?://(?:dx\.)?doi\.org/", re.I)

_PREMIUM_VENUE_ALIASES = {
    "neurips": ["NeurIPS", "Neural Information Processing Systems"],
    "icml": ["ICML", "International Conference on Machine Learning"],
    "iclr": ["ICLR", "International Conference on Learning Representations"],
    "kdd": ["KDD", "SIGKDD", "Knowledge Discovery and Data Mining"],
    "sigmod": [
        "SIGMOD",
        "ACM SIGMOD",
        "Management of Data",
        "Proceedings of the ACM on Management of Data",
        "International Conference on Management of Data",
    ],
    "vldb": ["VLDB", "Proceedings of the VLDB Endowment"],
    "icde": ["ICDE", "International Conference on Data Engineering"],
    "icdm": ["ICDM", "International Conference on Data Mining"],
    "sigir": ["SIGIR", "ACM SIGIR"],
    "iccv": ["ICCV", "International Conference on Computer Vision"],
    "cvpr": ["CVPR", "Computer Vision and Pattern Recognition"],
    "acl": ["ACL", "Association for Computational Linguistics"],
    "emnlp": ["EMNLP", "Empirical Methods in Natural Language Processing"],
    "www": ["WWW", "The Web Conference"],
    "aaai": ["AAAI", "Association for the Advancement of Artificial Intelligence"],
    "ijcai": ["IJCAI", "International Joint Conference on Artificial Intelligence"],
}

_PREMIUM_VENUE_EXCLUSIONS = {
    "icml": ["applications", "icmla"],
}


@dataclass(frozen=True, slots=True)
class VenueSource:
    query: str
    source_id: str
    display_name: str


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


def fetch_premium_venues(
    *,
    days: int = 180,
    max_results: int = 100,
    search: str | None = None,
    venues: list[str] | None = None,
    mailto: str | None = None,
    timeout: float = 25.0,
) -> list[Paper]:
    """Discover recent works from configured top-tier venues via OpenAlex.

    Unlike ``fetch_recent`` this intentionally keeps DOI/web-only records; many
    conference proceedings papers have no arXiv id, but are still useful hidden
    gem candidates when the venue and citation count are known.
    """
    venue_queries = [venue.strip() for venue in (venues or []) if venue.strip()]
    if not venue_queries:
        return []

    from_date = (datetime.now(timezone.utc).date() - timedelta(days=max(1, days))).isoformat()
    papers: dict[str, Paper] = {}

    with httpx.Client(timeout=timeout, follow_redirects=True) as client:
        sources = _resolve_venue_sources(client, venue_queries, mailto=mailto)
        if not sources:
            return []

        source_by_id = {source.source_id: source for source in sources}
        filters = [
            "primary_location.source.id:" + "|".join(source_by_id),
            f"from_publication_date:{from_date}",
        ]
        if search:
            filters.append(f"title_and_abstract.search:{_filter_value(search)}")
        params = {
            "filter": ",".join(filters),
            "sort": "publication_date:desc",
            "per-page": min(200, max_results),
            "select": (
                "id,doi,title,abstract_inverted_index,publication_date,cited_by_count,"
                "authorships,primary_location,best_oa_location,locations"
            ),
        }
        if mailto:
            params["mailto"] = mailto

        cursor = "*"
        while len(papers) < max_results and cursor:
            try:
                response = client.get(OPENALEX_URL, params={**params, "cursor": cursor})
                response.raise_for_status()
                payload = response.json()
            except (httpx.HTTPError, ValueError):
                break
            for work in payload.get("results", []):
                paper = _premium_work_to_paper(work, source_by_id)
                if paper and paper.key not in papers:
                    papers[paper.key] = paper
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


def _premium_work_to_paper(work: dict, source_by_id: dict[str, VenueSource]) -> Paper | None:
    title = " ".join(str(work.get("title") or "").split())
    if not title:
        return None

    source = _primary_source(work)
    source_id = _source_id(source)
    venue_source = source_by_id.get(source_id)
    venue_name = source.get("display_name") or (venue_source.display_name if venue_source else "")
    venue = str(venue_name).strip() or None
    arxiv_id = _arxiv_id_from_work(work)
    doi = _doi_from_work(work)
    abstract_url = f"https://arxiv.org/abs/{arxiv_id}" if arxiv_id else _landing_page_url(work)
    pdf_url = f"https://arxiv.org/pdf/{arxiv_id}" if arxiv_id else _pdf_url(work)
    authors = [
        (auth.get("author") or {}).get("display_name", "")
        for auth in work.get("authorships", [])
    ]
    external_ids = {"openalex": str(work.get("id", ""))}
    if source_id:
        external_ids["openalex_source"] = source_id
    if venue_source:
        external_ids["premium_venue_query"] = venue_source.query

    return Paper(
        arxiv_id=arxiv_id,
        title=title,
        authors=[a for a in authors if a],
        summary=_abstract_from_index(work.get("abstract_inverted_index")),
        published=_parse_date(work.get("publication_date")),
        categories=[],
        abs_url=abstract_url or str(work.get("id") or ""),
        pdf_url=pdf_url or None,
        doi=doi,
        citation_count=work.get("cited_by_count") if isinstance(work.get("cited_by_count"), int) else None,
        venue=venue,
        source="premium_venues",
        external_ids=external_ids,
    )


def _arxiv_id_from_work(work: dict) -> str:
    match = _ARXIV_DOI_RE.search(work.get("doi") or "")
    if match:
        return match.group(1)
    for location in _locations(work):
        landing = (location or {}).get("landing_page_url") or ""
        pdf = (location or {}).get("pdf_url") or ""
        match = (
            _ARXIV_ABS_RE.search(landing)
            or _ARXIV_PDF_RE.search(pdf)
            or _ARXIV_ABS_RE.search(pdf)
            or _ARXIV_DOI_RE.search(landing)
        )
        if match:
            return match.group(1)
    return ""


def _resolve_venue_sources(
    client: httpx.Client,
    venues: list[str],
    *,
    mailto: str | None = None,
    max_sources_per_venue: int = 1,
) -> list[VenueSource]:
    resolved: dict[str, VenueSource] = {}
    for venue in venues:
        candidates: list[dict] = []
        for alias in _venue_aliases(venue):
            candidates.extend(_search_sources(client, alias, mailto=mailto))
            if candidates and max(_source_rank(source, venue) for source in candidates) >= 60:
                break
        candidates.sort(key=lambda source: _source_rank(source, venue), reverse=True)
        chosen = 0
        for source in candidates:
            source_id = _source_id(source)
            display_name = str(source.get("display_name") or source_id or venue).strip()
            if not source_id or source_id in resolved:
                continue
            resolved[source_id] = VenueSource(query=venue, source_id=source_id, display_name=display_name)
            chosen += 1
            if chosen >= max_sources_per_venue:
                break
    return list(resolved.values())


def _search_sources(client: httpx.Client, query: str, *, mailto: str | None = None) -> list[dict]:
    params = {
        "search": query,
        "per-page": 8,
        "select": "id,display_name,abbreviated_title,type,works_count",
    }
    if mailto:
        params["mailto"] = mailto
    try:
        response = client.get(OPENALEX_SOURCES_URL, params=params)
        response.raise_for_status()
        results = response.json().get("results", [])
    except (httpx.HTTPError, ValueError):
        return []
    return [source for source in results if isinstance(source, dict)]


def _venue_aliases(venue: str) -> list[str]:
    aliases = [venue]
    aliases.extend(_PREMIUM_VENUE_ALIASES.get(_norm(venue), []))
    out: list[str] = []
    seen: set[str] = set()
    for alias in aliases:
        key = alias.lower()
        if key not in seen:
            out.append(alias)
            seen.add(key)
    return out


def _source_rank(source: dict, venue: str) -> float:
    query = _norm(venue)
    names = [_norm(source.get("display_name")), _norm(source.get("abbreviated_title"))]
    aliases = [_norm(alias) for alias in _venue_aliases(venue)]
    text = " ".join(name for name in names if name)
    score = 0.0
    if query in names:
        score += 100
    if _has_token(source.get("display_name"), venue) or _has_token(source.get("abbreviated_title"), venue):
        score += 50
    if any(alias and alias in text for alias in aliases):
        score += 35
    if source.get("type") in {"conference", "journal"}:
        score += 10
    if "proceedings" in text or "conference" in text:
        score += 20
    if "record" in text or "digitalreview" in text or "workshop" in text:
        score -= 45
    if any(term in text for term in _PREMIUM_VENUE_EXCLUSIONS.get(query, [])):
        score -= 80
    try:
        works_count = float(source.get("works_count") or 0)
        if works_count <= 0:
            score -= 80
        score += min(10.0, works_count / 10000)
    except (TypeError, ValueError):
        pass
    return score


def _source_id(source: dict) -> str:
    raw = str(source.get("id") or "").strip()
    if raw.startswith("S") and raw[1:].isdigit():
        return raw
    match = _OPENALEX_SOURCE_RE.search(raw)
    return match.group(1) if match else ""


def _primary_source(work: dict) -> dict:
    primary_source = ((work.get("primary_location") or {}).get("source") or {})
    if isinstance(primary_source, dict) and primary_source:
        return primary_source
    for location in work.get("locations") or []:
        source = (location or {}).get("source") or {}
        if isinstance(source, dict) and source:
            return source
    return {}


def _landing_page_url(work: dict) -> str:
    for location in _locations(work):
        url = str((location or {}).get("landing_page_url") or "").strip()
        if url:
            return url
    return str(work.get("id") or "").strip()


def _pdf_url(work: dict) -> str:
    for location in _locations(work):
        url = str((location or {}).get("pdf_url") or "").strip()
        if url:
            return url
    return ""


def _locations(work: dict) -> list[dict]:
    locations: list[dict] = []
    for key in ("best_oa_location", "primary_location"):
        location = work.get(key)
        if isinstance(location, dict):
            locations.append(location)
    locations.extend(location for location in work.get("locations") or [] if isinstance(location, dict))
    return locations


def _doi_from_work(work: dict) -> str | None:
    doi = str(work.get("doi") or "").strip()
    if not doi:
        return None
    return _DOI_URL_RE.sub("", doi)


def _filter_value(value: str) -> str:
    return " ".join(value.replace(",", " ").split())


def _norm(value: object) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(value or "").lower())


def _has_token(value: object, token: str) -> bool:
    token = re.escape(str(token or "").strip())
    return bool(token and re.search(rf"(?<![A-Za-z0-9]){token}(?![A-Za-z0-9])", str(value or ""), re.I))


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
