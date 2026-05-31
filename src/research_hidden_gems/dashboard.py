from __future__ import annotations

import json
import re
import sqlite3
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any

from research_hidden_gems.paths import default_dashboard_dir, default_report_dir
from research_hidden_gems.storage import default_state_path

COMPONENT_KEYS = [
    "relevance",
    "outlier",
    "hiddenness",
    "prefilter",
    "novelty",
    "technique",
    "applicability",
    "rarity",
]
_SOURCE_RE = re.compile(r"\*\*Link:\*\* \[([^\]]+)\]\([^)]+\)\s+.*source:\s+([A-Za-z0-9_.-]+)")


def write_dashboard(
    state_path: Path | str | None = None,
    dashboard_dir: Path | str | None = None,
    report_dir: Path | str | None = None,
    publish_dir: Path | str | None = None,
) -> Path:
    state = Path(state_path) if state_path is not None else default_state_path()
    out_dir = Path(dashboard_dir) if dashboard_dir is not None else default_dashboard_dir()
    out_dir.mkdir(parents=True, exist_ok=True)

    source_map = load_report_sources(Path(report_dir) if report_dir is not None else default_report_dir())
    records = load_seen_records(state, source_map=source_map)
    data = build_dashboard_data(records, state)
    data_text = json.dumps(data, indent=2, sort_keys=True)
    html_text = render_dashboard_html(data)
    data_path = out_dir / "data.json"
    data_path.write_text(data_text, encoding="utf-8")
    html_path = out_dir / "index.html"
    html_path.write_text(html_text, encoding="utf-8")
    if publish_dir is not None:
        publish = Path(publish_dir)
        publish.mkdir(parents=True, exist_ok=True)
        (publish / "index.html").write_text(html_text, encoding="utf-8")
        (publish / "data.json").write_text(data_text, encoding="utf-8")
    return html_path


def load_seen_records(state_path: Path | str, source_map: dict[str, str] | None = None) -> list[dict[str, Any]]:
    path = Path(state_path)
    if not path.is_file():
        return []

    with sqlite3.connect(path) as connection:
        connection.row_factory = sqlite3.Row
        rows = connection.execute(
            """
            SELECT paper_key, first_seen, last_seen, title, score, payload_json
            FROM seen_papers
            ORDER BY score DESC, first_seen DESC
            """
        ).fetchall()

    records: list[dict[str, Any]] = []
    for row in rows:
        try:
            payload = json.loads(row["payload_json"])
        except json.JSONDecodeError:
            payload = {}
        if not isinstance(payload, dict):
            payload = {}
        payload.setdefault("title", row["title"])
        payload.setdefault("score", row["score"])
        payload["paper_key"] = row["paper_key"]
        payload["first_seen"] = row["first_seen"]
        payload["last_seen"] = row["last_seen"]
        payload["source"] = payload.get("source") or (source_map or {}).get(row["paper_key"]) or "unknown"
        payload["score"] = _num(payload.get("score"))
        payload["citation_count"] = _optional_int(payload.get("citation_count"))
        payload["components"] = payload.get("components") if isinstance(payload.get("components"), dict) else {}
        payload["verdict"] = payload.get("verdict") if isinstance(payload.get("verdict"), dict) else None
        for key in ("authors", "categories", "techniques", "problems", "rare_terms", "rationale"):
            payload[key] = payload.get(key) if isinstance(payload.get(key), list) else []
        records.append(payload)
    return records


def load_report_sources(report_dir: Path | str) -> dict[str, str]:
    path = Path(report_dir)
    if not path.is_dir():
        return {}
    sources: dict[str, str] = {}
    for report in sorted(path.glob("*.md")):
        try:
            text = report.read_text(encoding="utf-8")
        except OSError:
            continue
        for key, source in _SOURCE_RE.findall(text):
            sources[key] = source
    return sources


def build_dashboard_data(records: list[dict[str, Any]], state_path: Path | str) -> dict[str, Any]:
    ordered = sorted(records, key=lambda item: _num(item.get("score")), reverse=True)
    hidden = [item for item in ordered if _is_hidden_gem(item)]
    judged = [item for item in ordered if item.get("verdict")]
    zero_cites = [item for item in ordered if item.get("citation_count") == 0]
    scores = [_num(item.get("score")) for item in ordered]

    return {
        "generated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "state_path": str(Path(state_path)),
        "summary": {
            "total": len(ordered),
            "hidden_gems": len(hidden),
            "judged": len(judged),
            "zero_citation": len(zero_cites),
            "average_score": round(sum(scores) / len(scores), 2) if scores else 0.0,
            "top_score": round(max(scores), 2) if scores else 0.0,
        },
        "trend": _trend(ordered),
        "score_buckets": _score_buckets(ordered),
        "source_counts": _counter_items(item.get("source") or "unknown" for item in ordered),
        "venue_counts": _counter_items(item.get("venue") for item in ordered if item.get("venue")),
        "category_counts": _category_counts(ordered),
        "technique_counts": _list_counts(ordered, "techniques", limit=18),
        "rare_term_counts": _list_counts(ordered, "rare_terms", limit=18),
        "component_averages": _component_averages(ordered),
        "papers": [_paper_row(item) for item in ordered],
    }


def render_dashboard_html(data: dict[str, Any]) -> str:
    data_json = json.dumps(data, ensure_ascii=True, sort_keys=True).replace("</", "<\\/")
    return _HTML_TEMPLATE.replace("__RHG_DATA__", data_json)


def _paper_row(item: dict[str, Any]) -> dict[str, Any]:
    verdict = item.get("verdict") if isinstance(item.get("verdict"), dict) else None
    components = item.get("components") if isinstance(item.get("components"), dict) else {}
    return {
        "key": item.get("paper_key") or item.get("arxiv_id") or item.get("title") or "",
        "title": item.get("title") or "",
        "arxiv_id": item.get("arxiv_id") or "",
        "doi": item.get("doi") or "",
        "abstract_url": item.get("abstract_url") or _arxiv_url(item.get("arxiv_id")),
        "pdf_url": item.get("pdf_url") or "",
        "authors": item.get("authors") or [],
        "published": item.get("published") or "",
        "first_seen": item.get("first_seen") or "",
        "last_seen": item.get("last_seen") or "",
        "source": item.get("source") or "unknown",
        "categories": item.get("categories") or [],
        "venue": item.get("venue") or "",
        "score": _num(item.get("score")),
        "citation_count": item.get("citation_count"),
        "influential_citation_count": item.get("influential_citation_count"),
        "components": {key: _num(components.get(key)) for key in COMPONENT_KEYS if key in components},
        "techniques": item.get("techniques") or [],
        "problems": item.get("problems") or [],
        "rare_terms": item.get("rare_terms") or [],
        "rationale": item.get("rationale") or [],
        "verdict": verdict,
        "hidden_gem": bool(verdict and verdict.get("is_hidden_gem")),
    }


def _trend(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    buckets: dict[str, dict[str, Any]] = defaultdict(lambda: {"date": "", "count": 0, "hidden": 0, "score_sum": 0.0})
    for item in records:
        date = _date_part(item.get("first_seen")) or "unknown"
        bucket = buckets[date]
        bucket["date"] = date
        bucket["count"] += 1
        bucket["hidden"] += 1 if _is_hidden_gem(item) else 0
        bucket["score_sum"] += _num(item.get("score"))
    out: list[dict[str, Any]] = []
    for date in sorted(buckets):
        bucket = buckets[date]
        count = bucket["count"]
        out.append(
            {
                "date": date,
                "count": count,
                "hidden": bucket["hidden"],
                "average_score": round(bucket["score_sum"] / count, 2) if count else 0.0,
            }
        )
    return out


def _score_buckets(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    ranges = [(0, 20), (20, 40), (40, 60), (60, 70), (70, 80), (80, 101)]
    buckets = [{"label": f"{low}-{high if high < 101 else 100}", "low": low, "high": high, "count": 0} for low, high in ranges]
    for item in records:
        score = _num(item.get("score"))
        for bucket in buckets:
            if bucket["low"] <= score < bucket["high"]:
                bucket["count"] += 1
                break
    return buckets


def _category_counts(records: list[dict[str, Any]], limit: int = 16) -> list[dict[str, Any]]:
    counter: Counter[str] = Counter()
    for item in records:
        categories = item.get("categories") or ["uncategorized"]
        counter.update(str(category) for category in categories if category)
    return _counter_to_items(counter, limit)


def _list_counts(records: list[dict[str, Any]], key: str, limit: int) -> list[dict[str, Any]]:
    counter: Counter[str] = Counter()
    for item in records:
        values = item.get(key) or []
        counter.update(str(value).strip() for value in values[:8] if str(value).strip())
    return _counter_to_items(counter, limit)


def _component_averages(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for key in COMPONENT_KEYS:
        values = []
        for item in records:
            components = item.get("components") if isinstance(item.get("components"), dict) else {}
            if key in components:
                values.append(_num(components.get(key)))
        out.append({"label": key, "value": round(sum(values) / len(values), 3) if values else 0.0})
    return out


def _counter_items(values) -> list[dict[str, Any]]:
    return _counter_to_items(Counter(str(value) for value in values if str(value)), 16)


def _counter_to_items(counter: Counter[str], limit: int) -> list[dict[str, Any]]:
    return [{"label": label, "count": count} for label, count in counter.most_common(limit)]


def _is_hidden_gem(item: dict[str, Any]) -> bool:
    verdict = item.get("verdict")
    return bool(isinstance(verdict, dict) and verdict.get("is_hidden_gem"))


def _date_part(value: Any) -> str:
    if not value:
        return ""
    return str(value).split("T", 1)[0]


def _arxiv_url(arxiv_id: Any) -> str:
    return f"https://arxiv.org/abs/{arxiv_id}" if arxiv_id else ""


def _num(value: Any, default: float = 0.0) -> float:
    try:
        if value is None:
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def _optional_int(value: Any) -> int | None:
    try:
        if value is None:
            return None
        return int(value)
    except (TypeError, ValueError):
        return None


_HTML_TEMPLATE = r"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Research Hidden Gems Dashboard</title>
  <style>
    :root {
      --bg: #f6f7f4;
      --surface: #ffffff;
      --ink: #202124;
      --muted: #66706a;
      --line: #dfe4df;
      --teal: #087f8c;
      --green: #2f855a;
      --amber: #b7791f;
      --red: #c2410c;
      --blue: #2563eb;
      --violet: #7c3aed;
      --shadow: 0 18px 50px rgba(32, 33, 36, 0.08);
      font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
    }
    * { box-sizing: border-box; }
    body { margin: 0; background: var(--bg); color: var(--ink); }
    a { color: var(--teal); text-decoration: none; }
    a:hover { text-decoration: underline; }
    button, input, select { font: inherit; }
    .app { min-height: 100vh; }
    header {
      display: grid;
      grid-template-columns: 1fr auto;
      gap: 16px;
      align-items: end;
      padding: 24px 28px 18px;
      border-bottom: 1px solid var(--line);
      background: #fbfcf9;
      position: sticky;
      top: 0;
      z-index: 5;
    }
    h1 { margin: 0; font-size: 24px; line-height: 1.15; letter-spacing: 0; }
    .meta { color: var(--muted); font-size: 13px; margin-top: 6px; }
    .header-actions { display: flex; gap: 8px; flex-wrap: wrap; justify-content: flex-end; }
    .btn {
      min-height: 36px;
      border: 1px solid var(--line);
      background: var(--surface);
      border-radius: 8px;
      padding: 7px 11px;
      color: var(--ink);
      cursor: pointer;
    }
    .btn:hover { border-color: #aab5ad; }
    main { padding: 18px 28px 28px; }
    .metrics {
      display: grid;
      grid-template-columns: repeat(6, minmax(120px, 1fr));
      gap: 10px;
      margin-bottom: 14px;
    }
    .metric, .panel {
      background: var(--surface);
      border: 1px solid var(--line);
      border-radius: 8px;
      box-shadow: var(--shadow);
    }
    .metric { padding: 13px 14px; }
    .metric-label { color: var(--muted); font-size: 12px; }
    .metric-value { font-size: 26px; font-weight: 760; line-height: 1.1; margin-top: 4px; }
    .controls {
      display: grid;
      grid-template-columns: minmax(220px, 1fr) 170px 160px 140px 170px auto;
      gap: 10px;
      margin: 14px 0;
      align-items: end;
    }
    label { display: grid; gap: 5px; color: var(--muted); font-size: 12px; }
    input, select {
      width: 100%;
      min-height: 38px;
      border: 1px solid var(--line);
      border-radius: 8px;
      background: var(--surface);
      color: var(--ink);
      padding: 8px 10px;
    }
    .grid {
      display: grid;
      grid-template-columns: minmax(0, 1.4fr) minmax(360px, 0.9fr);
      gap: 14px;
      align-items: start;
    }
    .panel { padding: 14px; min-width: 0; }
    .panel h2 {
      margin: 0 0 10px;
      font-size: 15px;
      line-height: 1.25;
      letter-spacing: 0;
    }
    .chart-grid {
      display: grid;
      grid-template-columns: repeat(2, minmax(0, 1fr));
      gap: 14px;
    }
    .chart { min-height: 242px; }
    .wide { grid-column: 1 / -1; }
    svg { width: 100%; height: 210px; display: block; overflow: visible; }
    .axis { stroke: #cbd5cd; stroke-width: 1; }
    .chart-label { fill: var(--muted); font-size: 11px; }
    .bar { cursor: pointer; }
    .bar:hover, .dot:hover { opacity: 0.75; }
    .legend { display: flex; gap: 12px; align-items: center; color: var(--muted); font-size: 12px; margin-top: 6px; flex-wrap: wrap; }
    .swatch { display: inline-block; width: 10px; height: 10px; border-radius: 3px; margin-right: 5px; vertical-align: -1px; }
    .table-wrap { overflow: auto; max-height: 640px; border: 1px solid var(--line); border-radius: 8px; }
    table { width: 100%; border-collapse: collapse; min-width: 860px; }
    th, td { padding: 9px 10px; border-bottom: 1px solid var(--line); text-align: left; vertical-align: top; }
    th { position: sticky; top: 0; background: #f8faf7; color: var(--muted); font-size: 12px; z-index: 1; }
    td { font-size: 13px; }
    tr { cursor: pointer; }
    tr:hover td { background: #f6fbf8; }
    .title-cell { font-weight: 650; min-width: 260px; }
    .num { text-align: right; white-space: nowrap; }
    .tag-list { display: flex; flex-wrap: wrap; gap: 6px; }
    .tag {
      border: 1px solid var(--line);
      background: #fbfcf9;
      color: #39433d;
      padding: 3px 7px;
      border-radius: 999px;
      font-size: 12px;
    }
    .detail {
      position: sticky;
      top: 102px;
      display: grid;
      gap: 12px;
    }
    .detail-title { font-size: 18px; line-height: 1.25; margin: 0; }
    .detail-meta { color: var(--muted); font-size: 13px; display: flex; flex-wrap: wrap; gap: 8px; }
    .score-row { display: grid; grid-template-columns: repeat(3, 1fr); gap: 8px; }
    .mini { border: 1px solid var(--line); border-radius: 8px; padding: 9px; background: #fbfcf9; }
    .mini strong { display: block; font-size: 18px; }
    .mini span { color: var(--muted); font-size: 12px; }
    .component { display: grid; grid-template-columns: 92px 1fr 46px; gap: 8px; align-items: center; font-size: 12px; margin: 7px 0; }
    .track { height: 8px; background: #edf1ed; border-radius: 999px; overflow: hidden; }
    .fill { height: 100%; background: var(--teal); border-radius: 999px; }
    .section-title { color: var(--muted); font-size: 12px; margin: 8px 0 5px; text-transform: uppercase; }
    .empty { color: var(--muted); padding: 28px; text-align: center; }
    .pill-hidden { color: #14532d; background: #dcfce7; border-color: #bbf7d0; }
    .pill-unknown { color: #713f12; background: #fef3c7; border-color: #fde68a; }
    @media (max-width: 1100px) {
      header { grid-template-columns: 1fr; }
      .header-actions { justify-content: flex-start; }
      .metrics { grid-template-columns: repeat(3, minmax(120px, 1fr)); }
      .controls { grid-template-columns: repeat(2, minmax(0, 1fr)); }
      .grid { grid-template-columns: 1fr; }
      .detail { position: static; }
    }
    @media (max-width: 700px) {
      main, header { padding-left: 14px; padding-right: 14px; }
      .metrics { grid-template-columns: repeat(2, minmax(0, 1fr)); }
      .controls, .chart-grid, .score-row { grid-template-columns: 1fr; }
      h1 { font-size: 21px; }
    }
  </style>
</head>
<body>
  <div class="app">
    <header>
      <div>
        <h1>Research Hidden Gems Dashboard</h1>
        <div class="meta" id="generatedMeta"></div>
      </div>
      <div class="header-actions">
        <a class="btn" href="./data.json" target="_blank" rel="noreferrer">Data JSON</a>
        <button class="btn" id="resetBtn" type="button">Reset Filters</button>
      </div>
    </header>
    <main>
      <section class="metrics" id="metrics"></section>
      <section class="controls">
        <label>Search<input id="searchInput" type="search" placeholder="title, technique, term, author"></label>
        <label>Source<select id="sourceFilter"></select></label>
        <label>Verdict<select id="verdictFilter"><option value="all">All</option><option value="hidden">Hidden gems</option><option value="judged">Judged</option><option value="unjudged">Unjudged</option></select></label>
        <label>Min Score<input id="scoreFilter" type="number" min="0" max="100" step="1" value="0"></label>
        <label>Sort<select id="sortBy"><option value="score">Score</option><option value="first_seen">First seen</option><option value="published">Published</option><option value="citations">Citations</option><option value="relevance">Relevance</option><option value="outlier">Outlier</option></select></label>
        <button class="btn" id="downloadCsvBtn" type="button">CSV</button>
      </section>
      <section class="grid">
        <div class="chart-grid">
          <article class="panel chart wide"><h2>Discovery Trend</h2><div id="trendChart"></div><div class="legend"><span><span class="swatch" style="background:#087f8c"></span>New papers</span><span><span class="swatch" style="background:#c2410c"></span>Hidden gems</span></div></article>
          <article class="panel chart"><h2>Score Distribution</h2><div id="scoreChart"></div></article>
          <article class="panel chart"><h2>Source Mix</h2><div id="sourceChart"></div></article>
          <article class="panel chart"><h2>Venue Mix</h2><div id="venueChart"></div></article>
          <article class="panel chart"><h2>Signal Averages</h2><div id="componentChart"></div></article>
          <article class="panel chart"><h2>Relevance vs Outlier</h2><div id="scatterChart"></div><div class="legend"><span><span class="swatch" style="background:#087f8c"></span>Candidate</span><span><span class="swatch" style="background:#c2410c"></span>Hidden gem</span></div></article>
          <article class="panel wide"><h2>Top Techniques</h2><div class="tag-list" id="techniqueTags"></div></article>
          <article class="panel wide"><h2>Rare Terms</h2><div class="tag-list" id="rareTags"></div></article>
          <article class="panel wide"><h2>Papers</h2><div class="table-wrap"><table><thead><tr><th class="num">Score</th><th>Title</th><th>Source</th><th>Venue</th><th>Published</th><th class="num">Cites</th><th class="num">Rel</th><th class="num">Outlier</th><th>Verdict</th></tr></thead><tbody id="paperRows"></tbody></table></div></article>
        </div>
        <aside class="panel detail" id="detailPanel"></aside>
      </section>
    </main>
  </div>
  <script id="dashboard-data" type="application/json">__RHG_DATA__</script>
  <script>
    const data = JSON.parse(document.getElementById("dashboard-data").textContent);
    let state = { search: "", source: "all", verdict: "all", minScore: 0, sortBy: "score", selected: null };
    const $ = (id) => document.getElementById(id);
    const fmt = (n, digits = 0) => Number(n || 0).toLocaleString(undefined, { maximumFractionDigits: digits });
    const pct = (n) => `${Math.round(Number(n || 0) * 100)}%`;
    const dateOnly = (value) => String(value || "").split("T")[0] || "";

    function init() {
      $("generatedMeta").textContent = `Updated ${new Date(data.generated_at).toLocaleString()} from ${data.state_path}`;
      renderMetrics();
      populateFilters();
      wireControls();
      render();
    }

    function renderMetrics() {
      const s = data.summary;
      const metrics = [
        ["Papers", s.total],
        ["Hidden Gems", s.hidden_gems],
        ["Judged", s.judged],
        ["Zero Citation", s.zero_citation],
        ["Avg Score", fmt(s.average_score, 1)],
        ["Top Score", fmt(s.top_score, 1)],
      ];
      $("metrics").replaceChildren(...metrics.map(([label, value]) => {
        const node = document.createElement("article");
        node.className = "metric";
        node.append(el("div", "metric-label", label), el("div", "metric-value", String(value)));
        return node;
      }));
    }

    function populateFilters() {
      const select = $("sourceFilter");
      const sources = ["all", ...new Set(data.papers.map((p) => p.source || "unknown"))].sort();
      select.replaceChildren(...sources.map((source) => {
        const option = document.createElement("option");
        option.value = source;
        option.textContent = source === "all" ? "All" : source;
        return option;
      }));
    }

    function wireControls() {
      $("searchInput").addEventListener("input", (e) => { state.search = e.target.value; render(); });
      $("sourceFilter").addEventListener("change", (e) => { state.source = e.target.value; render(); });
      $("verdictFilter").addEventListener("change", (e) => { state.verdict = e.target.value; render(); });
      $("scoreFilter").addEventListener("input", (e) => { state.minScore = Number(e.target.value || 0); render(); });
      $("sortBy").addEventListener("change", (e) => { state.sortBy = e.target.value; render(); });
      $("resetBtn").addEventListener("click", resetFilters);
      $("downloadCsvBtn").addEventListener("click", downloadCsv);
    }

    function resetFilters() {
      state = { search: "", source: "all", verdict: "all", minScore: 0, sortBy: "score", selected: null };
      $("searchInput").value = "";
      $("sourceFilter").value = "all";
      $("verdictFilter").value = "all";
      $("scoreFilter").value = 0;
      $("sortBy").value = "score";
      render();
    }

    function filteredPapers() {
      const q = state.search.trim().toLowerCase();
      const rows = data.papers.filter((paper) => {
        if (state.source !== "all" && paper.source !== state.source) return false;
        if (paper.score < state.minScore) return false;
        if (state.verdict === "hidden" && !paper.hidden_gem) return false;
        if (state.verdict === "judged" && !paper.verdict) return false;
        if (state.verdict === "unjudged" && paper.verdict) return false;
        if (!q) return true;
        const haystack = [
          paper.title, paper.source, paper.arxiv_id, paper.venue,
          ...(paper.authors || []), ...(paper.techniques || []), ...(paper.rare_terms || []),
          paper.verdict?.technique, paper.verdict?.one_liner, paper.verdict?.application_to_user,
        ].join(" ").toLowerCase();
        return haystack.includes(q);
      });
      rows.sort(sorter(state.sortBy));
      return rows;
    }

    function sorter(key) {
      return (a, b) => {
        if (key === "first_seen") return String(b.first_seen).localeCompare(String(a.first_seen));
        if (key === "published") return String(b.published).localeCompare(String(a.published));
        if (key === "citations") return Number(b.citation_count ?? -1) - Number(a.citation_count ?? -1);
        if (key === "relevance") return Number(b.components.relevance || 0) - Number(a.components.relevance || 0);
        if (key === "outlier") return Number(b.components.outlier || 0) - Number(a.components.outlier || 0);
        return Number(b.score || 0) - Number(a.score || 0);
      };
    }

    function render() {
      const papers = filteredPapers();
      renderTrend();
      renderScoreChart();
      renderSourceChart();
      renderVenueChart();
      renderComponentChart();
      renderScatter(papers);
      renderTags("techniqueTags", data.technique_counts, "technique");
      renderTags("rareTags", data.rare_term_counts, "rare");
      renderTable(papers);
      const selected = papers.find((paper) => paper.key === state.selected) || papers[0] || null;
      state.selected = selected?.key || null;
      renderDetail(selected);
    }

    function renderTrend() {
      const items = data.trend.map((item) => ({ label: item.date.slice(5), count: item.count, hidden: item.hidden }));
      stackedBar("trendChart", items, { primary: "count", secondary: "hidden", onClick: null });
    }

    function renderScoreChart() {
      barChart("scoreChart", data.score_buckets, {
        label: "label", value: "count", color: "#2563eb",
        onClick: (item) => { $("scoreFilter").value = item.low; state.minScore = item.low; render(); },
      });
    }

    function renderSourceChart() {
      barChart("sourceChart", data.source_counts, {
        label: "label", value: "count", color: "#087f8c",
        onClick: (item) => { $("sourceFilter").value = item.label; state.source = item.label; render(); },
      });
    }

    function renderVenueChart() {
      barChart("venueChart", data.venue_counts, {
        label: "label", value: "count", color: "#b7791f",
        onClick: (item) => { $("searchInput").value = item.label; state.search = item.label; render(); },
      });
    }

    function renderComponentChart() {
      barChart("componentChart", data.component_averages, { label: "label", value: "value", color: "#7c3aed", valueFormat: pct });
    }

    function renderScatter(papers) {
      const box = $("scatterChart");
      box.replaceChildren();
      const width = 640, height = 210, pad = 28;
      const svg = svgEl("svg", { viewBox: `0 0 ${width} ${height}`, role: "img" });
      svg.append(svgEl("line", { x1: pad, y1: height - pad, x2: width - pad, y2: height - pad, class: "axis" }));
      svg.append(svgEl("line", { x1: pad, y1: pad, x2: pad, y2: height - pad, class: "axis" }));
      svg.append(svgEl("text", { x: width - 110, y: height - 7, class: "chart-label" }, "relevance"));
      svg.append(svgEl("text", { x: 4, y: 16, class: "chart-label" }, "outlier"));
      papers.slice(0, 180).forEach((paper) => {
        const x = pad + clamp(paper.components.relevance || 0) * (width - pad * 2);
        const y = height - pad - clamp(paper.components.outlier || 0) * (height - pad * 2);
        const r = 3 + Math.max(0, Math.min(7, paper.score / 16));
        const dot = svgEl("circle", {
          cx: x, cy: y, r, fill: paper.hidden_gem ? "#c2410c" : "#087f8c",
          opacity: paper.hidden_gem ? "0.9" : "0.62", class: "dot",
        });
        dot.addEventListener("click", () => { state.selected = paper.key; renderDetail(paper); });
        dot.append(svgEl("title", {}, `${paper.title} (${fmt(paper.score, 1)})`));
        svg.append(dot);
      });
      box.append(svg);
    }

    function barChart(id, items, opts) {
      const box = $(id);
      box.replaceChildren();
      const width = 640, height = 210, pad = 28;
      const max = Math.max(1, ...items.map((item) => Number(item[opts.value] || 0)));
      const barGap = 5;
      const barWidth = Math.max(8, (width - pad * 2 - barGap * Math.max(0, items.length - 1)) / Math.max(1, items.length));
      const svg = svgEl("svg", { viewBox: `0 0 ${width} ${height}`, role: "img" });
      svg.append(svgEl("line", { x1: pad, y1: height - pad, x2: width - pad, y2: height - pad, class: "axis" }));
      items.forEach((item, index) => {
        const value = Number(item[opts.value] || 0);
        const x = pad + index * (barWidth + barGap);
        const h = (value / max) * (height - pad * 2);
        const y = height - pad - h;
        const bar = svgEl("rect", { x, y, width: barWidth, height: Math.max(1, h), rx: 4, fill: opts.color, class: "bar" });
        if (opts.onClick) bar.addEventListener("click", () => opts.onClick(item));
        bar.append(svgEl("title", {}, `${item[opts.label]}: ${opts.valueFormat ? opts.valueFormat(value) : fmt(value, 2)}`));
        svg.append(bar);
        const text = svgEl("text", { x: x + barWidth / 2, y: height - 8, class: "chart-label", "text-anchor": "middle" }, truncate(String(item[opts.label]), Math.max(4, Math.floor(barWidth / 6))));
        svg.append(text);
      });
      box.append(svg);
    }

    function stackedBar(id, items, opts) {
      const box = $(id);
      box.replaceChildren();
      const width = 900, height = 210, pad = 30;
      const max = Math.max(1, ...items.map((item) => Number(item[opts.primary] || 0)));
      const barGap = 5;
      const barWidth = Math.max(8, (width - pad * 2 - barGap * Math.max(0, items.length - 1)) / Math.max(1, items.length));
      const svg = svgEl("svg", { viewBox: `0 0 ${width} ${height}`, role: "img" });
      svg.append(svgEl("line", { x1: pad, y1: height - pad, x2: width - pad, y2: height - pad, class: "axis" }));
      items.forEach((item, index) => {
        const total = Number(item[opts.primary] || 0);
        const hidden = Number(item[opts.secondary] || 0);
        const x = pad + index * (barWidth + barGap);
        const totalHeight = (total / max) * (height - pad * 2);
        const hiddenHeight = total ? (hidden / total) * totalHeight : 0;
        svg.append(svgEl("rect", { x, y: height - pad - totalHeight, width: barWidth, height: Math.max(1, totalHeight), rx: 4, fill: "#087f8c", class: "bar" }));
        if (hiddenHeight > 0) svg.append(svgEl("rect", { x, y: height - pad - hiddenHeight, width: barWidth, height: Math.max(1, hiddenHeight), rx: 4, fill: "#c2410c", class: "bar" }));
        svg.append(svgEl("text", { x: x + barWidth / 2, y: height - 8, class: "chart-label", "text-anchor": "middle" }, item.label));
      });
      box.append(svg);
    }

    function renderTags(id, items, filterMode) {
      const box = $(id);
      const nodes = items.map((item) => {
        const tag = el("button", `tag ${filterMode === "rare" ? "pill-unknown" : ""}`, `${item.label} ${item.count}`);
        tag.type = "button";
        tag.addEventListener("click", () => { $("searchInput").value = item.label; state.search = item.label; render(); });
        return tag;
      });
      box.replaceChildren(...nodes);
    }

    function renderTable(papers) {
      const body = $("paperRows");
      if (!papers.length) {
        const row = document.createElement("tr");
        const cell = document.createElement("td");
        cell.colSpan = 9;
        cell.className = "empty";
        cell.textContent = "No papers match the current filters.";
        row.append(cell);
        body.replaceChildren(row);
        return;
      }
      const rows = papers.map((paper) => {
        const row = document.createElement("tr");
        row.addEventListener("click", () => { state.selected = paper.key; renderDetail(paper); });
        row.append(
          td(fmt(paper.score, 1), "num"),
          td(paper.title, "title-cell"),
          td(paper.source || "unknown"),
          td(paper.venue || ""),
          td(paper.published || ""),
          td(paper.citation_count == null ? "?" : fmt(paper.citation_count), "num"),
          td(fmt(paper.components.relevance || 0, 2), "num"),
          td(fmt(paper.components.outlier || 0, 2), "num"),
          td(paper.hidden_gem ? "Hidden gem" : paper.verdict ? "Judged" : "Unjudged"),
        );
        return row;
      });
      body.replaceChildren(...rows);
    }

    function renderDetail(paper) {
      const panel = $("detailPanel");
      panel.replaceChildren();
      if (!paper) {
        panel.append(el("div", "empty", "No paper selected."));
        return;
      }
      const title = el("h2", "detail-title", paper.title);
      const meta = el("div", "detail-meta");
      [paper.source || "unknown", paper.venue || "", paper.published || "no date", `${paper.citation_count ?? "?"} citations`, paper.arxiv_id || paper.doi || paper.key].forEach((value) => {
        if (value) meta.append(el("span", "", value));
      });
      const scoreRow = el("div", "score-row");
      scoreRow.append(mini("Score", fmt(paper.score, 1)), mini("Relevance", fmt(paper.components.relevance || 0, 2)), mini("Outlier", fmt(paper.components.outlier || 0, 2)));
      const links = el("div", "tag-list");
      if (paper.abstract_url) links.append(link("Abstract", paper.abstract_url));
      if (paper.pdf_url) links.append(link("PDF", paper.pdf_url));
      if (paper.hidden_gem) links.append(el("span", "tag pill-hidden", "Hidden gem"));
      panel.append(title, meta, scoreRow, links);
      panel.append(section("Verdict", verdictText(paper)));
      panel.append(componentBlock(paper.components || {}));
      panel.append(section("Techniques", tagBlock(paper.techniques)));
      panel.append(section("Rare Terms", tagBlock(paper.rare_terms)));
      panel.append(section("Problems", listBlock(paper.problems)));
      panel.append(section("Rationale", listBlock(paper.rationale)));
    }

    function verdictText(paper) {
      const v = paper.verdict;
      if (!v) return el("div", "empty", "No LLM verdict stored.");
      const box = document.createElement("div");
      [
        ["Technique", v.technique],
        ["One line", v.one_liner],
        ["Why overlooked", v.why_overlooked],
        ["Apply", v.application_to_user],
        ["Scores", `novelty ${fmt(v.novelty, 2)}, transferability ${fmt(v.transferability, 2)}, confidence ${fmt(v.confidence, 2)}`],
      ].forEach(([label, value]) => {
        if (!value) return;
        const item = el("div", "", "");
        item.append(el("div", "section-title", label), el("div", "", value));
        box.append(item);
      });
      return box;
    }

    function componentBlock(components) {
      const box = document.createElement("div");
      box.append(el("div", "section-title", "Signals"));
      Object.entries(components).forEach(([key, value]) => {
        const row = el("div", "component");
        row.append(el("span", "", key));
        const track = el("div", "track");
        const fill = el("div", "fill");
        fill.style.width = `${Math.round(clamp(value) * 100)}%`;
        track.append(fill);
        row.append(track, el("span", "num", fmt(value, 2)));
        box.append(row);
      });
      return box;
    }

    function section(title, child) {
      const box = document.createElement("div");
      box.append(el("div", "section-title", title), child);
      return box;
    }

    function tagBlock(values) {
      const box = el("div", "tag-list");
      (values || []).slice(0, 10).forEach((value) => box.append(el("span", "tag", value)));
      if (!box.children.length) box.append(el("span", "empty", "None"));
      return box;
    }

    function listBlock(values) {
      const list = document.createElement("ul");
      (values || []).slice(0, 8).forEach((value) => {
        const item = document.createElement("li");
        item.textContent = value;
        list.append(item);
      });
      if (!list.children.length) return el("div", "empty", "None");
      return list;
    }

    function downloadCsv() {
      const rows = filteredPapers();
      const header = ["score", "title", "source", "venue", "published", "citations", "relevance", "outlier", "hidden_gem", "url"];
      const csv = [header, ...rows.map((p) => [
        p.score, p.title, p.source, p.venue || "", p.published, p.citation_count ?? "", p.components.relevance ?? "", p.components.outlier ?? "", p.hidden_gem, p.abstract_url || "",
      ])].map((row) => row.map(csvCell).join(",")).join("\n");
      const blob = new Blob([csv], { type: "text/csv" });
      const url = URL.createObjectURL(blob);
      const a = document.createElement("a");
      a.href = url;
      a.download = "hidden-gems-dashboard.csv";
      a.click();
      URL.revokeObjectURL(url);
    }

    function csvCell(value) {
      const text = String(value ?? "");
      return /[",\n]/.test(text) ? `"${text.replaceAll('"', '""')}"` : text;
    }

    function mini(label, value) {
      const node = el("div", "mini");
      node.append(el("strong", "", value), el("span", "", label));
      return node;
    }

    function link(text, href) {
      const a = document.createElement("a");
      a.className = "tag";
      a.href = href;
      a.target = "_blank";
      a.rel = "noreferrer";
      a.textContent = text;
      return a;
    }

    function td(text, className = "") {
      const cell = document.createElement("td");
      cell.className = className;
      cell.textContent = text;
      return cell;
    }

    function el(tag, className, text) {
      const node = document.createElement(tag);
      if (className) node.className = className;
      if (text !== undefined) node.textContent = text;
      return node;
    }

    function svgEl(tag, attrs = {}, text) {
      const node = document.createElementNS("http://www.w3.org/2000/svg", tag);
      Object.entries(attrs).forEach(([key, value]) => node.setAttribute(key, value));
      if (text !== undefined) node.textContent = text;
      return node;
    }

    function clamp(value) { return Math.max(0, Math.min(1, Number(value || 0))); }
    function truncate(text, length) { return text.length > length ? `${text.slice(0, Math.max(1, length - 1))}...` : text; }
    init();
  </script>
</body>
</html>
"""
