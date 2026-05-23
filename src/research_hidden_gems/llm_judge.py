"""Claude deep-judge stage: structured novelty / transferability assessment.

Runs only on the top-prefiltered shortlist (cost control). The big static
instructions + interest profile live in a cached system block, so every call
after the first is a prompt-cache hit. Degrades to a no-op when the ``anthropic``
package or an API key is missing — the pipeline then ranks on heuristics alone.
"""

from __future__ import annotations

import json
import os
import re
from datetime import datetime, timezone
from pathlib import Path

from research_hidden_gems.config import Config
from research_hidden_gems.models import LLMVerdict, ScoredPaper

_SYSTEM = """\
You are a research-novelty scout for an experienced ML/CS researcher. Your job is \
to spot HIDDEN GEMS: papers whose core *technique* or *problem formulation* is \
genuinely novel and transferable, yet which are likely underappreciated (few \
citations, niche venue, awkward framing, or buried in a crowded area).

You are NOT rewarding: incremental benchmark wins, well-known methods reapplied, \
survey/position papers, or hype language ("novel", "first") unaccompanied by a \
real mechanism. Be skeptical and calibrated — most papers are NOT hidden gems.

Judge transferability against THIS researcher's interests:
<profile>
{profile}
</profile>
{math_block}
Return ONLY a JSON object (no prose, no code fence) with exactly these keys:
- "novelty": float 0..1  — how genuinely new the core idea/mechanism is
- "transferability": float 0..1 — how usefully it could transfer to the profile above
- "confidence": float 0..1 — your confidence given only title+abstract
- "is_hidden_gem": boolean — novel AND transferable AND plausibly underappreciated
- "technique": string — name the core transferable technique or problem, concretely
- "one_liner": string — the idea in one sentence
- "why_overlooked": string — why it is plausibly underappreciated (or "" if it isn't)
- "application_to_user": string — one concrete way THIS researcher could apply or \
extend the technique in their own work
Keep strings under ~240 characters. Output the JSON object only."""

_MATH_BLOCK = """\
This researcher also values mathematical depth. For math-heavy papers, also weigh \
the rigor and novelty of the formal contribution (proofs, new objects, structural \
results), not just empirical claims. Reference background:
<math_reference>
{reference}
</math_reference>
"""


def is_available(config: Config) -> bool:
    if not config.judge_enabled:
        return False
    try:
        import anthropic  # noqa: F401
    except Exception:
        return False
    return bool(os.getenv("ANTHROPIC_API_KEY") or os.getenv("ANTHROPIC_AUTH_TOKEN"))


def judge_papers(scored: list[ScoredPaper], config: Config) -> int:
    """Attach an LLMVerdict to the top ``config.judge_top`` papers, in place.

    Returns the number of papers actually judged. No-op (returns 0) when the
    judge is unavailable.
    """
    if not scored or not is_available(config):
        return 0

    import anthropic

    client = anthropic.Anthropic()
    system = _build_system(config)
    judged = 0
    for item in scored[: config.judge_top]:
        verdict = _judge_one(client, system, item, config)
        if verdict is not None:
            item.verdict = verdict
            judged += 1
    return judged


def _build_system(config: Config) -> list[dict]:
    math_block = ""
    if config.math_depth:
        reference = _math_reference(config.math_skills_path)
        math_block = "\n" + _MATH_BLOCK.format(reference=reference) + "\n"
    text = _SYSTEM.format(profile=config.profile.strip(), math_block=math_block)
    # single cached block — reused across every paper in the shortlist
    return [{"type": "text", "text": text, "cache_control": {"type": "ephemeral"}}]


def _judge_one(client, system: list[dict], item: ScoredPaper, config: Config) -> LLMVerdict | None:
    try:
        response = client.messages.create(
            model=config.judge_model,
            max_tokens=config.judge_max_tokens,
            system=system,
            messages=[{"role": "user", "content": _paper_prompt(item)}],
        )
        text = "".join(block.text for block in response.content if getattr(block, "type", "") == "text")
    except Exception:
        return None

    data = _extract_json(text)
    if data is None:
        return None
    data["raw"] = text
    return LLMVerdict.from_dict(data)


def _paper_prompt(item: ScoredPaper) -> str:
    paper = item.paper
    now = datetime.now(timezone.utc)
    cites = "unknown" if paper.citation_count is None else str(paper.citation_count)
    lines = [
        f"Title: {paper.title}",
        f"Categories: {', '.join(paper.categories) or 'n/a'}",
        f"Source: {paper.source}; Published: {paper.published.date().isoformat()}; "
        f"Age(days): {paper.age_days(now)}; Citations: {cites}",
    ]
    if item.techniques:
        lines.append(f"Heuristic technique phrases: {', '.join(item.techniques[:8])}")
    lines.append("")
    lines.append("Abstract:")
    lines.append(paper.summary)
    return "\n".join(lines)


def _math_reference(path: str | None, limit: int = 1500) -> str:
    if not path:
        return "(no math-skills reference configured)"
    readme = Path(path) / "README.md"
    try:
        if readme.is_file():
            return readme.read_text(encoding="utf-8")[:limit]
    except OSError:
        pass
    return "(math-skills reference unavailable)"


_FENCE_RE = re.compile(r"```(?:json)?\s*(\{.*?\})\s*```", re.S)


def _extract_json(text: str) -> dict | None:
    text = text.strip()
    try:
        obj = json.loads(text)
        return obj if isinstance(obj, dict) else None
    except json.JSONDecodeError:
        pass
    match = _FENCE_RE.search(text)
    if match:
        try:
            return json.loads(match.group(1))
        except json.JSONDecodeError:
            pass
    start, end = text.find("{"), text.rfind("}")
    if 0 <= start < end:
        try:
            obj = json.loads(text[start : end + 1])
            return obj if isinstance(obj, dict) else None
        except json.JSONDecodeError:
            return None
    return None
