"""News source — trendy or under-discussed industry signals that seed research ideas.

Papers tell you what the field *has done*. News tells you what just became true
about the world the field assumes: a price that fell 100x, weights that got
released, a capability that shipped, a regulation that landed. Those are
*freshness drivers* — the "what changed in the last two years" a research problem
needs in order not to read as five years old.

The fetchers here return ordinary :class:`~research_hidden_gems.models.Paper`
objects with ``source="news"``, so the rest of the pipeline (dedup, embeddings,
relevance gating, seen-state, dashboard) works on news items unchanged. What is
news-specific lives in two places: the lexical signals in ``scoring.py`` and the
deep-judge prompt in ``llm_judge.py``, both keyed off ``paper.source``.

Every feed is best-effort: a dead host, a malformed document, or a rate limit
drops that feed rather than failing the run.
"""

from __future__ import annotations

import html
import re
from datetime import datetime, timedelta, timezone
from email.utils import parsedate_to_datetime
from urllib.parse import urlsplit, urlunsplit
from xml.etree import ElementTree

import httpx

from research_hidden_gems.models import Paper

_UA = {"User-Agent": "research-hidden-gems/0.1 (news scout; +https://arxiv.org)"}

# Curated high-signal feeds. Mixed on purpose: lab/vendor blogs announce the
# capability, practitioner newsletters price it, and trade press reports the
# market move. Any of them can 404 without affecting the others.
DEFAULT_NEWS_FEEDS = [
    "https://huggingface.co/blog/feed.xml",
    "https://qwenlm.github.io/blog/index.xml",
    "https://blog.vllm.ai/feed.xml",
    "https://lmsys.org/rss.xml",
    "https://simonwillison.net/atom/everything/",
    "https://www.interconnects.ai/feed",
    "https://importai.substack.com/feed",
    "https://magazine.sebastianraschka.com/feed",
    "https://bair.berkeley.edu/blog/feed.xml",
    "https://research.google/blog/rss/",
    "https://deepmind.google/blog/rss.xml",
    "https://openai.com/news/rss.xml",
    "https://www.marktechpost.com/feed/",
    "https://venturebeat.com/category/ai/feed/",
    "https://www.theregister.com/software/ai_ml/headlines.atom",
]

# A news item earns its place only if it plausibly *changes an assumption* a
# research problem could rest on. These are the phrase families that mark such
# items; they double as the on-topic filter for the (very noisy) news firehose.
NEWS_SIGNAL_TERMS = {
    # economics — the 2026 driver: price collapse and open weights
    "price", "pricing", "per million tokens", "per token", "cheaper", "cost",
    "free tier", "discount", "price cut", "open weight", "open-weight",
    "open source", "open-sourced", "weights", "license", "self-host",
    # capability and supply
    "release", "released", "launch", "ships", "available", "preview",
    "context window", "benchmark", "state of the art", "outperform",
    "parameters", "mixture of experts", "quantization", "distilled",
    # operations — where the durable white space is (skill Lesson #27)
    "deprecat", "end of life", "sunset", "rate limit", "outage", "latency",
    "throughput", "capacity", "serving", "inference cost", "gpu",
    # governance / constraints that create new problems
    "regulation", "compliance", "audit", "ban", "export control", "policy",
    "data retention", "privacy",
}

# Hacker News is the "trendy" half of the mandate; the feeds are the "hidden"
# half. Queries are deliberately narrow — HN's firehose is mostly off-topic.
DEFAULT_HN_QUERIES = [
    "open weights model",
    "LLM pricing",
    "inference cost",
    "model release",
    "LLM benchmark",
    "agent framework",
]
HN_SEARCH_URL = "https://hn.algolia.com/api/v1/search"

_TAG_RE = re.compile(r"<[^>]+>")
_WS_RE = re.compile(r"\s+")
_ATOM = "{http://www.w3.org/2005/Atom}"


def fetch_recent(
    *,
    days: int = 7,
    max_results: int = 80,
    feeds: list[str] | None = None,
    keywords: list[str] | None = None,
    hn_queries: list[str] | None = None,
    hn_min_points: int = 20,
    timeout: float = 15.0,
) -> list[Paper]:
    """Fetch recent news items from RSS/Atom feeds plus Hacker News.

    ``keywords`` (usually the config's research keywords) widen the on-topic
    filter beyond :data:`NEWS_SIGNAL_TERMS`; an item is kept when it matches
    either set. Returns newest-first, capped at ``max_results``.
    """
    cutoff = datetime.now(timezone.utc) - timedelta(days=max(1, days))
    wanted = {term.lower() for term in NEWS_SIGNAL_TERMS} | {
        term.lower() for term in (keywords or [])
    }
    items: dict[str, Paper] = {}

    with httpx.Client(timeout=timeout, follow_redirects=True, headers=_UA) as client:
        for feed_url in feeds if feeds is not None else DEFAULT_NEWS_FEEDS:
            for paper in _fetch_feed(client, feed_url):
                _keep(items, paper, cutoff=cutoff, wanted=wanted)
        for query in hn_queries if hn_queries is not None else DEFAULT_HN_QUERIES:
            for paper in _fetch_hn(client, query, cutoff=cutoff, min_points=hn_min_points):
                _keep(items, paper, cutoff=cutoff, wanted=wanted)

    ordered = sorted(items.values(), key=lambda paper: paper.published, reverse=True)
    return ordered[:max_results]


def is_on_topic(paper: Paper, wanted: set[str]) -> bool:
    """True when the item mentions something that could change an assumption."""
    text = paper.text.lower()
    return any(term in text for term in wanted)


# ------------------------------------------------------------------ feeds ---
def _fetch_feed(client: httpx.Client, url: str) -> list[Paper]:
    try:
        response = client.get(url)
        response.raise_for_status()
        root = ElementTree.fromstring(response.content)
    except (httpx.HTTPError, ElementTree.ParseError, ValueError):
        return []
    outlet = _outlet(url)
    entries = root.findall(".//item") or root.findall(f".//{_ATOM}entry")
    papers = [_entry_to_paper(entry, outlet=outlet, feed=url) for entry in entries]
    return [paper for paper in papers if paper is not None]


def _entry_to_paper(entry: ElementTree.Element, *, outlet: str, feed: str) -> Paper | None:
    title = _clean(_first_text(entry, ["title", f"{_ATOM}title"]))
    link = _entry_link(entry)
    if not title or not link:
        return None
    summary = _clean(
        _first_text(
            entry,
            [
                "description",
                f"{_ATOM}summary",
                "{http://purl.org/rss/1.0/modules/content/}encoded",
                f"{_ATOM}content",
            ],
        )
    )
    published = _parse_date(
        _first_text(entry, ["pubDate", f"{_ATOM}published", f"{_ATOM}updated", "{http://purl.org/dc/elements/1.1/}date"])
    )
    author = _clean(_first_text(entry, ["{http://purl.org/dc/elements/1.1/}creator", f"{_ATOM}author/{_ATOM}name"]))
    return Paper(
        arxiv_id="",
        title=title,
        authors=[author] if author else [],
        summary=summary[:2000],
        published=published,
        categories=["news"],
        abs_url=_canonical(link),
        venue=outlet,
        source="news",
        external_ids={"outlet": outlet, "feed": feed, "news_kind": "feed"},
    )


def _entry_link(entry: ElementTree.Element) -> str:
    text = (entry.findtext("link") or "").strip()
    if text:
        return text
    for element in entry.findall(f"{_ATOM}link"):
        rel = element.get("rel") or "alternate"
        if rel == "alternate" and element.get("href"):
            return element.get("href", "").strip()
    guid = (entry.findtext("guid") or "").strip()
    return guid if guid.startswith("http") else ""


# --------------------------------------------------------------------- HN ---
def _fetch_hn(client: httpx.Client, query: str, *, cutoff: datetime, min_points: int) -> list[Paper]:
    params = {
        "query": query,
        "tags": "story",
        "hitsPerPage": "30",
        "numericFilters": f"created_at_i>{int(cutoff.timestamp())},points>={max(0, min_points)}",
    }
    try:
        response = client.get(HN_SEARCH_URL, params=params)
        response.raise_for_status()
        hits = response.json().get("hits") or []
    except (httpx.HTTPError, ValueError, AttributeError):
        return []

    papers: list[Paper] = []
    for hit in hits:
        title = _clean(hit.get("title") or "")
        url = (hit.get("url") or "").strip()
        object_id = str(hit.get("objectID") or "")
        if not title:
            continue
        link = url or (f"https://news.ycombinator.com/item?id={object_id}" if object_id else "")
        if not link:
            continue
        points = int(hit.get("points") or 0)
        comments = int(hit.get("num_comments") or 0)
        papers.append(
            Paper(
                arxiv_id="",
                title=title,
                authors=[],
                summary=_clean(hit.get("story_text") or "")[:2000],
                published=_parse_date(hit.get("created_at")),
                categories=["news"],
                abs_url=_canonical(link),
                venue="Hacker News",
                source="news",
                external_ids={
                    "outlet": "Hacker News",
                    "news_kind": "hn",
                    "hn_points": str(points),
                    "hn_comments": str(comments),
                    "hn_discussion": f"https://news.ycombinator.com/item?id={object_id}",
                    "hn_query": query,
                },
            )
        )
    return papers


# ------------------------------------------------------------------ utils ---
def _keep(items: dict[str, Paper], paper: Paper, *, cutoff: datetime, wanted: set[str]) -> None:
    if paper.published < cutoff:
        return
    if not is_on_topic(paper, wanted):
        return
    key = paper.abs_url or paper.title
    existing = items.get(key)
    if existing is None:
        items[key] = paper
        return
    # Same story from two places: keep the richer summary, merge the signals.
    if len(paper.summary) > len(existing.summary):
        existing.summary = paper.summary
    existing.external_ids.update({k: v for k, v in paper.external_ids.items() if v})


def _canonical(url: str) -> str:
    """Drop tracking query strings and fragments so the same story dedups."""
    try:
        parts = urlsplit(url.strip())
    except ValueError:
        return url.strip()
    if not parts.scheme:
        return url.strip()
    return urlunsplit((parts.scheme, parts.netloc, parts.path.rstrip("/") or "/", "", ""))


def _outlet(feed_url: str) -> str:
    host = urlsplit(feed_url).netloc.lower()
    return host[4:] if host.startswith("www.") else host


def _first_text(entry: ElementTree.Element, paths: list[str]) -> str:
    for path in paths:
        found = entry.find(path)
        if found is not None and (found.text or "").strip():
            return found.text or ""
    return ""


def _clean(value: str) -> str:
    return _WS_RE.sub(" ", html.unescape(_TAG_RE.sub(" ", value or ""))).strip()


def _parse_date(value: str | None) -> datetime:
    raw = (value or "").strip()
    if raw:
        for parse in (_parse_iso, parsedate_to_datetime):
            try:
                parsed = parse(raw)
            except (TypeError, ValueError):
                continue
            if parsed is not None:
                return parsed if parsed.tzinfo else parsed.replace(tzinfo=timezone.utc)
    return datetime.now(timezone.utc)


def _parse_iso(value: str) -> datetime:
    return datetime.fromisoformat(value.replace("Z", "+00:00"))
