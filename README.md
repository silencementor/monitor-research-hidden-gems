# monitor-research-hidden-gems

Find under-the-radar research papers that contain transferable novel ideas:
new training recipes, mechanisms, problem formulations, ablations, or control
techniques that may be useful in your own research.

The tool ranks arXiv papers by an explainable hidden-gem score:

- `novelty`: abstract/title language that suggests a new method, mechanism, or problem.
- `technique`: extracted transferable method phrases such as training pipelines, steering methods, reward schedules, or latent mechanisms.
- `rarity`: uncommon terms compared with the current search batch.
- `hiddenness`: low citation/attention signal relative to age, when Semantic Scholar data is available.
- `applicability`: signs that the idea is a reusable procedure rather than only a narrow empirical report.

## Setup

This project is configured for `uv`:

```bash
uv sync
```

If you do not have `uv` installed yet, install it first from the official Astral
instructions, then run the command above.

## Search

Search recent AI/ML/NLP arXiv papers and show the top ranked hidden gems:

```bash
uv run hidden-gems search \
  --categories cs.AI,cs.CL,cs.LG,stat.ML \
  --days 30 \
  --max-results 150 \
  --top-k 20
```

Search around a specific technique area:

```bash
uv run hidden-gems search \
  --query "activation steering" \
  --categories cs.AI,cs.CL \
  --days 180 \
  --format markdown
```

The query accepts either a plain phrase or native arXiv query syntax, for example:

```bash
uv run hidden-gems search \
  --query 'abs:"multi-agent debate" OR abs:"post-training"' \
  --categories cs.AI,cs.CL
```

## Inspect One Paper

Inspect why a paper looks interesting:

```bash
uv run hidden-gems inspect https://arxiv.org/abs/2604.24881
```

This produces technique phrases, problem hooks, rare terms, citation context,
and a short rationale.

## Monitor

Run a one-shot monitor that only prints papers you have not seen before:

```bash
uv run hidden-gems monitor \
  --categories cs.AI,cs.CL,cs.LG \
  --days 7 \
  --threshold 50
```

The monitor stores seen papers in:

```text
~/.cache/research-hidden-gems/seen.sqlite3
```

Use a custom state file if you want separate monitors for different topics:

```bash
uv run hidden-gems monitor \
  --query "mechanistic interpretability" \
  --state .cache/mech-interp.sqlite3
```

For a long-running local watcher:

```bash
uv run hidden-gems monitor --interval-minutes 360
```

## Output Formats

All commands support:

```bash
--format table
--format markdown
--format json
```

JSON output is useful for downstream automation or feeding selected papers into
your own literature-review agent.

## Citation Enrichment

By default the tool asks Semantic Scholar for citation counts. If the API is
rate-limited or unavailable, scoring still works with a neutral hiddenness
fallback.

Set an API key for higher Semantic Scholar limits:

```bash
export S2_API_KEY=...
```

Disable enrichment when you want a pure arXiv-only run:

```bash
uv run hidden-gems search --no-enrich
```

## Test

```bash
uv run pytest
```

## Notes

This is a discovery aid, not a replacement for a literature review. Treat high
scores as prompts for manual reading: the goal is to surface papers whose method
ingredients are unusual, reusable, and not already saturated by citations.
