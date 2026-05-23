from datetime import datetime, timezone

from research_hidden_gems.models import Paper
from research_hidden_gems.scoring import score_papers
from research_hidden_gems.storage import SeenStore


def test_seen_store_filters_unseen(tmp_path) -> None:
    paper = Paper(
        arxiv_id="2601.00001",
        title="Novel Internalization Procedure",
        authors=[],
        summary="We introduce a new training procedure for compact reasoning.",
        published=datetime(2026, 1, 1, tzinfo=timezone.utc),
    )
    scored = score_papers([paper])
    store = SeenStore(tmp_path / "seen.sqlite3")

    assert store.unseen(scored) == scored
    store.upsert(scored)
    assert store.unseen(scored) == []
