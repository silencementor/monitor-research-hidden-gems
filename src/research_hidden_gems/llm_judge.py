"""LLM deep-judge stage: structured novelty / transferability assessment.

Supports two providers — Anthropic (Claude) and OpenAI — selected via
``config.judge_provider`` ("auto" | "anthropic" | "openai"). In "auto" mode the
provider is inferred from an explicit model name, else from whichever SDK +
API key is usable (Anthropic preferred). Runs only on the top-prefiltered
shortlist (cost control), and degrades to a no-op when no provider is usable —
the pipeline then ranks on heuristics + embeddings alone.
"""

from __future__ import annotations

import importlib.util
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


# --------------------------------------------------------- provider routing -
def _import_ok(module: str) -> bool:
    try:
        return importlib.util.find_spec(module) is not None
    except (ImportError, ValueError):
        return False


def _usable(provider: str) -> bool:
    if provider == "openai":
        return _import_ok("openai") and bool(os.getenv("OPENAI_API_KEY"))
    return _import_ok("anthropic") and bool(
        os.getenv("ANTHROPIC_API_KEY") or os.getenv("ANTHROPIC_AUTH_TOKEN")
    )


def _infer_provider(model: str) -> str | None:
    low = model.lower()
    if low.startswith(("gpt", "o1", "o3", "o4", "chatgpt", "text-", "davinci")):
        return "openai"
    if low.startswith("claude"):
        return "anthropic"
    return None


def _resolve_provider(config: Config) -> tuple[str, str]:
    """Return (provider, model)."""
    provider = (config.judge_provider or "auto").lower()
    model = (config.judge_model or "").strip()
    if provider not in ("anthropic", "openai"):
        inferred = _infer_provider(model) if model else None
        if inferred:
            provider = inferred
        else:
            provider = next((cand for cand in ("anthropic", "openai") if _usable(cand)), "anthropic")
    if not model:
        model = config.anthropic_model if provider == "anthropic" else config.openai_model
    return provider, model


def is_available(config: Config) -> bool:
    if not config.judge_enabled:
        return False
    provider, _ = _resolve_provider(config)
    return _usable(provider)


# ------------------------------------------------------------------- judge --
def judge_papers(scored: list[ScoredPaper], config: Config) -> int:
    """Attach an LLMVerdict to the top ``config.judge_top`` papers, in place.

    Returns the number of papers judged. No-op (0) when no provider is usable.
    """
    if not scored or not is_available(config):
        return 0

    provider, model = _resolve_provider(config)
    client = _make_client(provider)
    system_text = _build_system(config)
    judged = 0
    for item in scored[: config.judge_top]:
        verdict = _judge_one(provider, client, system_text, item, config, model)
        if verdict is not None:
            item.verdict = verdict
            judged += 1
    return judged


def _make_client(provider: str):
    if provider == "openai":
        from openai import OpenAI

        return OpenAI()
    import anthropic

    return anthropic.Anthropic()


def _build_system(config: Config) -> str:
    math_block = ""
    if config.math_depth:
        reference = _math_reference(config.math_skills_path)
        math_block = "\n" + _MATH_BLOCK.format(reference=reference) + "\n"
    return _SYSTEM.format(profile=config.profile.strip(), math_block=math_block)


def _judge_one(provider, client, system_text, item, config, model) -> LLMVerdict | None:
    user = _paper_prompt(item)
    try:
        if provider == "openai":
            text = _call_openai(client, system_text, user, model, config.judge_max_tokens)
        else:
            text = _call_anthropic(client, system_text, user, model, config.judge_max_tokens)
    except Exception:
        return None

    data = _extract_json(text)
    if data is None:
        return None
    data["raw"] = text
    return LLMVerdict.from_dict(data)


def _call_anthropic(client, system_text: str, user: str, model: str, max_tokens: int) -> str:
    # The big static system block is cached (cache hit on every call after the first).
    response = client.messages.create(
        model=model,
        max_tokens=max_tokens,
        system=[{"type": "text", "text": system_text, "cache_control": {"type": "ephemeral"}}],
        messages=[{"role": "user", "content": user}],
    )
    return "".join(b.text for b in response.content if getattr(b, "type", "") == "text")


def _call_openai(client, system_text: str, user: str, model: str, max_tokens: int) -> str:
    # OpenAI auto-caches long shared prefixes; json_object mode guarantees parseable output.
    kwargs = dict(
        model=model,
        response_format={"type": "json_object"},
        messages=[
            {"role": "system", "content": system_text},
            {"role": "user", "content": user},
        ],
    )
    try:
        response = client.chat.completions.create(max_tokens=max_tokens, **kwargs)
    except TypeError:
        response = client.chat.completions.create(max_completion_tokens=max_tokens, **kwargs)
    except Exception as exc:  # newer models reject 'max_tokens' at the API layer
        if "max_tokens" in str(exc) or "max_completion_tokens" in str(exc):
            response = client.chat.completions.create(max_completion_tokens=max_tokens, **kwargs)
        else:
            raise
    return response.choices[0].message.content or ""


# -------------------------------------------------------------- prompt I/O --
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
