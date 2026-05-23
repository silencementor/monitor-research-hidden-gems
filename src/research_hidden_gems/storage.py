from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

from research_hidden_gems.models import ScoredPaper


def default_state_path() -> Path:
    return Path.home() / ".cache" / "research-hidden-gems" / "seen.sqlite3"


class SeenStore:
    def __init__(self, path: Path | str | None = None) -> None:
        self.path = Path(path) if path else default_state_path()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def seen_ids(self) -> set[str]:
        with self._connect() as connection:
            rows = connection.execute("SELECT arxiv_id FROM seen_papers").fetchall()
        return {row[0] for row in rows}

    def unseen(self, papers: list[ScoredPaper]) -> list[ScoredPaper]:
        seen = self.seen_ids()
        return [paper for paper in papers if paper.paper.arxiv_id not in seen]

    def upsert(self, papers: list[ScoredPaper]) -> None:
        now = datetime.now(timezone.utc).isoformat()
        with self._connect() as connection:
            for scored in papers:
                connection.execute(
                    """
                    INSERT INTO seen_papers (
                        arxiv_id, first_seen, last_seen, title, score, payload_json
                    )
                    VALUES (?, ?, ?, ?, ?, ?)
                    ON CONFLICT(arxiv_id) DO UPDATE SET
                        last_seen = excluded.last_seen,
                        title = excluded.title,
                        score = excluded.score,
                        payload_json = excluded.payload_json
                    """,
                    (
                        scored.paper.arxiv_id,
                        now,
                        now,
                        scored.paper.title,
                        scored.score,
                        json.dumps(scored.to_dict(), sort_keys=True),
                    ),
                )

    def _connect(self) -> sqlite3.Connection:
        return sqlite3.connect(self.path)

    def _init_db(self) -> None:
        with self._connect() as connection:
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS seen_papers (
                    arxiv_id TEXT PRIMARY KEY,
                    first_seen TEXT NOT NULL,
                    last_seen TEXT NOT NULL,
                    title TEXT NOT NULL,
                    score REAL NOT NULL,
                    payload_json TEXT NOT NULL
                )
                """
            )
