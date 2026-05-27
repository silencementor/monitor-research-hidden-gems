# monitor-research-hidden-gems

Surface **hidden gems**: research papers whose core *technique* or *problem
formulation* is genuinely novel and transferable, yet which are still
underappreciated (few citations, niche framing, buried in a crowded area). The
goal is to find ideas you can carry into your own work and turn into new papers.

It is a **hybrid** ranker — a cheap, explainable prefilter narrows the daily
firehose, then Claude deep-judges the shortlist for genuine novelty and *how it
could apply to you*:

```
fetch (arXiv · HF Daily · OpenAlex · OpenReview)
  → dedup + citation enrichment
  → prefilter:  lexical signals (novelty/technique language, rare terms)
              + embedding-outlier novelty (semantic isolation)
              + relevance to YOUR interest profile
              + hiddenness (low citations for its age)
  → Claude deep-judge on the top shortlist  (novelty · transferability · why-overlooked · how-to-apply)
  → blended hidden-gem score
```

Every stage degrades gracefully: a dead source, a missing embeddings backend, or
an absent API key simply drops that signal instead of failing the run.

## Setup

This project uses `uv`:

```bash
uv sync                       # core install (runs in heuristic + hashing mode)
uv sync --extra embeddings    # add semantic embeddings (sentence-transformers)
uv sync --extra llm           # add the Claude deep-judge (anthropic SDK)
uv sync --extra all           # everything
```

To unlock the deep-judge, set an API key for **either** provider (auto-selected,
Anthropic preferred when both are present):

```bash
export ANTHROPIC_API_KEY=sk-ant-...   # Claude judge
export OPENAI_API_KEY=sk-...          # or OpenAI judge
```

Or keep them in a `.env` file — the nearest one (walking up from your working
directory) is auto-loaded, and a real shell variable still takes precedence:

```bash
cp .env.example .env    # then fill in your keys
```

Force one with `--provider openai` / `--provider anthropic`, or just name a model
(`--model gpt-4.1` infers OpenAI, `--model claude-sonnet-4-6` infers Anthropic).

For embeddings, install the `embeddings` extra (sentence-transformers) **or** set
`embed_backend = "openai"` to use your OpenAI key; otherwise a dependency-free
hashing fallback is used automatically.

If Hugging Face downloads are slow or rate-limited, put `HF_TOKEN=hf_...` in
your `.env`; the sentence-transformers backend will use it when downloading
models.

## Search

```bash
uv run hidden-gems search --days 30 --top-k 20
```

Focus on a technique area (a `--query` also enables OpenAlex discovery of
older-but-uncited work — often the truest hidden gems):

```bash
uv run hidden-gems search -q "approximate nearest neighbor" --days 120 --format markdown
```

Useful flags: `--no-llm` (heuristics only), `--provider openai`, `--model gpt-4.1`,
`--judge-top N` (how many to deep-judge — controls cost), `--threshold 50`,
`--sources arxiv,openalex`, `--categories cs.IR,cs.DB`, `--profile @my_interests.txt`.

## Personalize

Ranking is tuned to a research profile. The default targets LLMs, agentic AI,
databases, data mining, knowledge discovery, IR, recommender systems, and Big
Data, across `cs.LG, cs.AI, cs.CL, cs.IR, cs.DB, cs.DC, cs.MA, cs.SI, stat.ML`.

Override anything in `hidden_gems.toml` (or `~/.config/research-hidden-gems/config.toml`):

```toml
profile = "I work on retrieval-augmented LLM agents and recommender systems; I want transferable indexing, ranking, and training techniques."
categories = ["cs.IR", "cs.DB", "cs.LG", "cs.MA"]
sources = ["arxiv", "huggingface_daily", "openalex"]
judge_provider = "auto"         # auto | anthropic | openai
judge_model = ""                # empty => provider default below
anthropic_model = "claude-sonnet-4-6"
openai_model = "gpt-4.1"
judge_top = 15
embed_backend = "auto"          # auto | sentence-transformers | openai | hashing
openai_embed_model = "text-embedding-3-small"
# scoring weights (prefilter blend)
w_lexical = 0.30
w_outlier = 0.25
w_hiddenness = 0.20
w_relevance = 0.25
relevance_floor = 0.35          # how hard relevance gates off-topic papers
openalex_mailto = "you@example.com"   # OpenAlex polite pool
```

Quick env overrides: `RHG_JUDGE_PROVIDER`, `RHG_JUDGE_MODEL`, `RHG_EMBED_BACKEND`,
`RHG_JUDGE=off`, `RHG_OPENALEX_MAILTO`, `RHG_MATH_SKILLS_PATH`.

## Monitor & schedule

A one-shot run that prints only papers not seen before (cron-friendly), and can
also writes a timestamped markdown report under `reports/hidden-gems/`:

```bash
uv run hidden-gems monitor --days 7 --threshold 50
```

Schedule it with cron (daily 8am):

```cron
0 8 * * *  cd /path/to/repo && uv run hidden-gems monitor --days 2
```

Or run a long-lived local watcher:

```bash
uv run hidden-gems monitor -q "vector indexing" --provider openai --format markdown --interval-minutes 360
```

Monitor runs print timestamped stage updates by default: source fetches,
deduplication, citation enrichment, embedding/ranking, LLM judging, report
writing, and the next scheduled run time. Add `--no-progress` for quiet output.

By default, monitor state and cache stay in the project:

- seen-paper DB: `.hidden-gems/state/seen.sqlite3`
- model/app cache: `.hidden-gems/cache/`
- markdown reports: `reports/hidden-gems/YYYYMMDD-HHMMSS-query.md`

Override with `--state`, `--report-dir`, `RHG_RUNTIME_DIR`, `RHG_CACHE_DIR`,
`RHG_STATE_PATH`, or `RHG_REPORT_DIR`. Use `--no-reports` if you only want
console output. To keep uv's own package cache inside the project too, prefix the
command with `UV_CACHE_DIR="$PWD/.hidden-gems/cache/uv"`.

## Inspect one paper

```bash
uv run hidden-gems inspect https://arxiv.org/abs/2604.24881
```

Explains the score and, with the judge enabled, names the core technique, why it
may be overlooked, and a concrete way to apply it to your work.

## Sources

- **arXiv** — the discovery spine (recent submissions in your categories).
- **HuggingFace Daily Papers** — curated daily arXiv feed.
- **OpenAlex** — free citation enrichment (via the arXiv DOI `10.48550/arXiv.<id>`)
  and, when a `--query` is given, discovery of older arXiv works whose low
  citation counts make them candidate gems.
- **OpenReview** — best-effort; recent submissions are keyed by per-venue
  invitation ids that rotate each cycle, so it returns nothing until you supply
  venues in config (it never ships stale guesses).

## Mathematical depth (optional)

For math-heavy papers you can ask the judge to also weigh formal novelty/rigor,
grounded in a local math-skills library:

```bash
uv run hidden-gems search -q "category theory" --math-depth \
  --math-skills-path /path/to/math-skills
# or: export RHG_MATH_SKILLS_PATH=/path/to/math-skills
```

## Output formats

`--format table` (scan), `--format markdown` (full, actionable — the verdict and
"apply to your work" notes), `--format json` (automation / feed into another agent).

## Test

```bash
uv run pytest
```

## Notes

This is a discovery aid, not a literature review. The heuristic prefilter and
embedding outliers find *candidates*; the Claude judge adds calibrated judgement
but only sees title + abstract. Treat high scores as prompts for manual reading.
