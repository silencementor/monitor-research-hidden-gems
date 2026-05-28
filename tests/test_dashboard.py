from datetime import datetime, timezone
import json

from research_hidden_gems.dashboard import build_dashboard_data, load_report_sources, load_seen_records, write_dashboard
from research_hidden_gems.models import Paper
from research_hidden_gems.scoring import score_papers
from research_hidden_gems.storage import SeenStore


def _store_with_paper(tmp_path):
    paper = Paper(
        arxiv_id="2601.00001",
        title="Vector Index Routing for Agent Memory",
        authors=["A. Researcher"],
        summary="We introduce a novel vector indexing method for agent memory and retrieval.",
        published=datetime(2026, 1, 1, tzinfo=timezone.utc),
        citation_count=0,
        abs_url="https://arxiv.org/abs/2601.00001",
        source="openalex",
    )
    scored = score_papers([paper])
    scored[0].components["relevance"] = 0.9
    scored[0].components["outlier"] = 0.4
    store = SeenStore(tmp_path / "seen.sqlite3")
    store.upsert(scored)
    return store


def test_load_seen_records_includes_monitor_metadata(tmp_path) -> None:
    store = _store_with_paper(tmp_path)

    records = load_seen_records(store.path)

    assert len(records) == 1
    assert records[0]["paper_key"] == "2601.00001"
    assert records[0]["source"] == "openalex"
    assert records[0]["first_seen"]


def test_build_dashboard_data_summarizes_records(tmp_path) -> None:
    store = _store_with_paper(tmp_path)
    records = load_seen_records(store.path)

    data = build_dashboard_data(records, store.path)

    assert data["summary"]["total"] == 1
    assert data["summary"]["zero_citation"] == 1
    assert data["source_counts"] == [{"label": "openalex", "count": 1}]
    assert data["papers"][0]["title"] == "Vector Index Routing for Agent Memory"


def test_write_dashboard_outputs_html_and_json(tmp_path) -> None:
    store = _store_with_paper(tmp_path)

    html_path = write_dashboard(store.path, tmp_path / "dashboard")

    data_path = html_path.parent / "data.json"
    assert html_path.is_file()
    assert data_path.is_file()
    assert "Research Hidden Gems Dashboard" in html_path.read_text(encoding="utf-8")
    data = json.loads(data_path.read_text(encoding="utf-8"))
    assert data["summary"]["total"] == 1


def test_load_report_sources_parses_markdown_reports(tmp_path) -> None:
    report_dir = tmp_path / "reports"
    report_dir.mkdir()
    (report_dir / "digest.md").write_text(
        "- **Link:** [2601.00001](https://arxiv.org/abs/2601.00001)  · source: huggingface_daily\n",
        encoding="utf-8",
    )

    assert load_report_sources(report_dir) == {"2601.00001": "huggingface_daily"}
