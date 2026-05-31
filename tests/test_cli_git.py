from pathlib import Path

from research_hidden_gems.cli import _repo_relative_paths


def test_repo_relative_paths_keeps_only_repo_files(tmp_path) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    inside = repo / "index.html"
    nested = repo / "reports" / "dashboard" / "data.json"
    outside = tmp_path / "outside.md"

    paths = _repo_relative_paths(repo, [inside, nested, outside, inside])

    assert paths == ["index.html", "reports/dashboard/data.json"]
