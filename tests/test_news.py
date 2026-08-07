from datetime import datetime, timedelta, timezone

import httpx
import pytest

from research_hidden_gems import news
from research_hidden_gems.config import Config
from research_hidden_gems.llm_judge import build_systems
from research_hidden_gems.models import LLMVerdict, Paper
from research_hidden_gems.pipeline import rank_papers
from research_hidden_gems.scoring import score_paper

RSS = """<?xml version="1.0"?>
<rss version="2.0"><channel>
  <item>
    <title>DeepSeek cuts V4-Flash to $0.14 per million input tokens</title>
    <link>https://example.com/deepseek-price?utm_source=rss</link>
    <description>&lt;p&gt;The new price is 14x cheaper than the frontier tier.&lt;/p&gt;</description>
    <pubDate>Mon, 03 Aug 2026 09:00:00 +0000</pubDate>
  </item>
  <item>
    <title>A profile of a startup founder</title>
    <link>https://example.com/profile</link>
    <description>Nothing measurable here.</description>
    <pubDate>Mon, 03 Aug 2026 09:00:00 +0000</pubDate>
  </item>
</channel></rss>
"""

ATOM = """<?xml version="1.0"?>
<feed xmlns="http://www.w3.org/2005/Atom">
  <entry>
    <title>Kimi K3 open weights released</title>
    <link rel="alternate" href="https://blog.example.org/kimi-k3/"/>
    <summary>2.8 trillion parameters, weights available under a permissive license.</summary>
    <published>2026-08-02T12:00:00Z</published>
  </entry>
</feed>
"""

HN = {
    "hits": [
        {
            "title": "Qwen3.8-Max ships with open weights next week",
            "url": "https://news.example.net/qwen",
            "objectID": "42",
            "points": 310,
            "num_comments": 88,
            "created_at": "2026-08-04T08:00:00Z",
        }
    ]
}


def _client(handler) -> httpx.Client:
    return httpx.Client(transport=httpx.MockTransport(handler), base_url="https://example.com")


def _now_days_ago(days: float) -> datetime:
    return datetime.now(timezone.utc) - timedelta(days=days)


def _news_paper(**overrides) -> Paper:
    fields = dict(
        arxiv_id="",
        title="DeepSeek cuts V4-Flash pricing",
        authors=[],
        summary="The price drops to $0.14 per million tokens, 14x cheaper than before.",
        published=_now_days_ago(1),
        categories=["news"],
        abs_url="https://example.com/a",
        venue="example.com",
        source="news",
        external_ids={"outlet": "example.com", "news_kind": "feed"},
    )
    fields.update(overrides)
    return Paper(**fields)


# ----------------------------------------------------------------- parsing --
def test_rss_parsing_keeps_signal_items_and_drops_off_topic_ones():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, text=RSS)

    with _client(handler) as client:
        papers = news._fetch_feed(client, "https://www.example.com/feed.xml")

    assert [paper.title for paper in papers] == [
        "DeepSeek cuts V4-Flash to $0.14 per million input tokens",
        "A profile of a startup founder",
    ]
    priced, profile = papers
    # HTML is stripped, entities decoded, tracking params dropped.
    assert priced.summary == "The new price is 14x cheaper than the frontier tier."
    assert priced.abs_url == "https://example.com/deepseek-price"
    assert priced.source == "news" and priced.venue == "example.com"
    assert priced.published.year == 2026

    wanted = {term.lower() for term in news.NEWS_SIGNAL_TERMS}
    assert news.is_on_topic(priced, wanted)
    assert not news.is_on_topic(profile, wanted)


def test_atom_parsing_uses_alternate_link():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, text=ATOM)

    with _client(handler) as client:
        (paper,) = news._fetch_feed(client, "https://blog.example.org/atom.xml")

    assert paper.title == "Kimi K3 open weights released"
    assert paper.abs_url == "https://blog.example.org/kimi-k3"
    assert paper.key == "https://blog.example.org/kimi-k3"  # no arXiv id => URL is the key


def test_hacker_news_carries_the_popularity_signal():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=HN)

    with _client(handler) as client:
        (paper,) = news._fetch_hn(client, "open weights", cutoff=_now_days_ago(7), min_points=20)

    assert paper.external_ids["hn_points"] == "310"
    assert paper.external_ids["hn_discussion"].endswith("id=42")
    assert paper.venue == "Hacker News"


@pytest.mark.parametrize(
    "raw, expected_scheme",
    [("https://a.example/x/?utm=1#frag", "https"), ("http://a.example/x", "http"), ("not a url", "")],
)
def test_canonical_url_strips_tracking(raw: str, expected_scheme: str):
    canonical = news._canonical(raw)
    assert "utm" not in canonical and "#" not in canonical
    if expected_scheme:
        assert canonical.startswith(expected_scheme + "://")


def test_dead_feed_degrades_to_empty_rather_than_raising():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(503, text="down")

    with _client(handler) as client:
        assert news._fetch_feed(client, "https://example.com/feed.xml") == []


def test_malformed_xml_degrades_to_empty():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, text="<rss><channel><item>")

    with _client(handler) as client:
        assert news._fetch_feed(client, "https://example.com/feed.xml") == []


def test_keep_drops_stale_items_and_merges_duplicates():
    cutoff = _now_days_ago(7)
    wanted = {"price"}
    items: dict[str, Paper] = {}

    news._keep(items, _news_paper(published=_now_days_ago(30)), cutoff=cutoff, wanted=wanted)
    assert items == {}

    news._keep(items, _news_paper(summary="price"), cutoff=cutoff, wanted=wanted)
    news._keep(
        items,
        _news_paper(summary="price cut, a much longer excerpt", external_ids={"hn_points": "120"}),
        cutoff=cutoff,
        wanted=wanted,
    )
    assert len(items) == 1
    merged = items["https://example.com/a"]
    assert merged.summary == "price cut, a much longer excerpt"
    assert merged.external_ids["hn_points"] == "120"


# ----------------------------------------------------------------- scoring --
def test_news_items_score_on_the_driver_signal_not_technique_phrases():
    driver = score_paper(_news_paper())
    filler = score_paper(
        _news_paper(
            title="Our thoughts on the road ahead",
            summary="A reflection on where things may go.",
        )
    )
    assert "news_driver" in driver.components
    assert driver.components["news_driver"] > filler.components["news_driver"]
    assert driver.score > filler.score


def test_news_hiddenness_falls_as_hacker_news_points_rise():
    quiet = score_paper(_news_paper()).components["hiddenness"]
    loud = score_paper(_news_paper(external_ids={"hn_points": "900"})).components["hiddenness"]
    assert quiet > loud


def test_paper_scoring_is_unchanged_by_the_news_branch():
    paper = Paper(
        arxiv_id="2601.00001",
        title="A new routing mechanism for retrieval",
        authors=["A"],
        summary="We propose a novel routing mechanism that improves retrieval by 12%.",
        published=_now_days_ago(3),
        categories=["cs.IR"],
        source="arxiv",
    )
    scored = score_paper(paper)
    assert "news_driver" not in scored.components
    assert scored.components["technique"] > 0


def test_rank_papers_blends_news_and_papers_without_the_judge():
    items = [_news_paper(), _news_paper(title="Nothing happened", summary="No numbers.", abs_url="https://e/b")]
    ranked = rank_papers(Config(judge_enabled=False), items, do_judge=False)
    assert [item.paper.title for item in ranked][0] == "DeepSeek cuts V4-Flash pricing"
    assert all("prefilter" in item.components for item in ranked)


# ------------------------------------------------------------------- judge --
def test_news_and_paper_judges_are_separate_prompts():
    systems = build_systems(Config(profile="I work on data mining."))
    assert set(systems) == {"paper", "news"}
    assert "FRESHNESS DRIVER" in systems["news"]
    assert "broken_assumption" in systems["news"] and "research_hook" in systems["news"]
    assert "broken_assumption" not in systems["paper"]
    assert "I work on data mining." in systems["news"]


def test_verdict_round_trips_the_news_only_fields():
    verdict = LLMVerdict.from_dict(
        {
            "novelty": 0.8,
            "is_hidden_gem": True,
            "broken_assumption": "Routers are evaluated at a fixed price vector.",
            "research_hook": "Report the region of price space over which a router wins.",
        }
    )
    assert verdict.broken_assumption.startswith("Routers are evaluated")
    assert verdict.to_dict()["research_hook"].startswith("Report the region")
    # Absent keys stay empty rather than None, so existing consumers are unaffected.
    assert LLMVerdict.from_dict({}).research_hook == ""
