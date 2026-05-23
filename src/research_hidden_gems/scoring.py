from __future__ import annotations

import math
import re
from collections import Counter
from datetime import datetime, timezone

from research_hidden_gems.models import Paper, ScoredPaper

STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "across",
    "after",
    "also",
    "be",
    "before",
    "between",
    "by",
    "can",
    "for",
    "from",
    "has",
    "have",
    "in",
    "into",
    "is",
    "it",
    "its",
    "may",
    "model",
    "models",
    "more",
    "of",
    "often",
    "on",
    "or",
    "our",
    "that",
    "the",
    "them",
    "their",
    "these",
    "then",
    "they",
    "this",
    "through",
    "to",
    "using",
    "via",
    "we",
    "while",
    "within",
    "with",
}

BAD_PHRASE_BOUNDARY_TOKENS = {
    "one",
    "two",
    "three",
    "four",
    "five",
    "six",
    "seven",
    "eight",
    "nine",
    "ten",
    "many",
    "several",
    "various",
}

BAD_PHRASE_TOKENS = {
    "combining",
    "compared",
    "develop",
    "demonstrate",
    "finding",
    "findings",
    "improve",
    "show",
}

NOVELTY_PATTERNS = [
    re.compile(pattern, re.I)
    for pattern in [
        r"\bwe (introduce|propose|present|develop|design|formulate|derive|show|demonstrate)\b",
        r"\bnovel\b",
        r"\bnew\b",
        r"\bfirst\b",
        r"\bpreviously unexplored\b",
        r"\bto address\b",
        r"\bthis work\b",
    ]
]

TECHNIQUE_TERMS = {
    "agent",
    "agent-specific",
    "agents",
    "alignment",
    "activation",
    "architecture",
    "architectures",
    "benchmark",
    "clipping",
    "communication",
    "compression",
    "control",
    "debate",
    "decoding",
    "distill",
    "distillation",
    "dynamic",
    "embedding",
    "evaluation",
    "fine-tuning",
    "framework",
    "frameworks",
    "internalization",
    "intervention",
    "latent",
    "learning",
    "length",
    "masking",
    "mechanism",
    "memory",
    "method",
    "methods",
    "mode",
    "modes",
    "optimization",
    "pipeline",
    "pipelines",
    "policy",
    "post-training",
    "preference",
    "procedure",
    "procedures",
    "reasoning",
    "representation",
    "reward",
    "routing",
    "scheduling",
    "search",
    "simulation",
    "steering",
    "subspace",
    "subspaces",
    "topology",
    "training",
}

PROBLEM_TERMS = {
    "bottleneck",
    "challenge",
    "cost",
    "failure",
    "inefficiency",
    "limitation",
    "problem",
    "requires",
    "risk",
    "scarce",
    "underexplored",
}

TRANSFER_TERMS = {
    "algorithm",
    "architecture",
    "distill",
    "framework",
    "general",
    "mechanism",
    "method",
    "modular",
    "pipeline",
    "procedure",
    "recipe",
    "scalable",
    "steering",
    "training",
    "transfer",
}

RESULT_TERMS = {
    "achieve",
    "improve",
    "outperform",
    "reduce",
    "fewer",
    "less",
    "efficient",
    "performance",
    "state-of-the-art",
}

TOKEN_RE = re.compile(r"[a-z][a-z0-9-]{2,}", re.I)
SENTENCE_RE = re.compile(r"(?<=[.!?])\s+")


def score_papers(papers: list[Paper], *, now: datetime | None = None) -> list[ScoredPaper]:
    now = now or datetime.now(timezone.utc)
    idf = _corpus_idf(papers)
    scored = [score_paper(paper, idf=idf, now=now) for paper in papers]
    return sorted(scored, key=lambda item: item.score, reverse=True)


def score_paper(paper: Paper, *, idf: dict[str, float] | None = None, now: datetime | None = None) -> ScoredPaper:
    now = now or datetime.now(timezone.utc)
    idf = idf or _corpus_idf([paper])
    text = paper.text.lower()
    title = paper.title.lower()

    novelty_hits = sum(len(pattern.findall(text)) for pattern in NOVELTY_PATTERNS)
    title_technique_hits = sum(1 for token in tokenize(title) if token in TECHNIQUE_TERMS)
    technique_phrases = extract_techniques(paper)
    problems = extract_problem_phrases(paper)
    rare_terms = extract_rare_terms(paper, idf)

    novelty = _clamp(0.12 * novelty_hits + 0.08 * title_technique_hits + 0.08 * len(problems))
    technique = _clamp(0.16 * len(technique_phrases) + 0.04 * title_technique_hits)
    rarity = _rarity_score(paper, idf)
    hiddenness = _hiddenness_score(paper, now=now)
    applicability = _applicability_score(paper)

    components = {
        "novelty": novelty,
        "technique": technique,
        "rarity": rarity,
        "hiddenness": hiddenness,
        "applicability": applicability,
    }
    score = 100 * (
        0.28 * novelty
        + 0.26 * technique
        + 0.18 * rarity
        + 0.18 * hiddenness
        + 0.10 * applicability
    )

    return ScoredPaper(
        paper=paper,
        score=round(score, 4),
        components=components,
        techniques=technique_phrases[:12],
        problems=problems[:4],
        rare_terms=rare_terms[:8],
        rationale=_rationale(paper, components, technique_phrases, problems, rare_terms, now=now),
    )


def extract_techniques(paper: Paper) -> list[str]:
    phrases: Counter[str] = Counter()
    for source, multiplier in ((paper.title, 2), (paper.summary, 1)):
        for tokens in _content_spans(tokenize(source)):
            for size in range(2, 5):
                for index in range(0, max(0, len(tokens) - size + 1)):
                    window = tokens[index : index + size]
                    if _is_interesting_phrase(window, TECHNIQUE_TERMS):
                        phrase = _phrase(window)
                        anchor_hits = sum(1 for token in window if token in TECHNIQUE_TERMS)
                        concise_anchor_bonus = 8 if size == 2 and anchor_hits == 2 else 0
                        phrases[phrase] += multiplier * 3 + anchor_hits * 2 + (5 - size) + concise_anchor_bonus
    selected: list[str] = []
    for phrase, _ in phrases.most_common():
        if any(_contains_same_ordered_tokens(phrase, existing) for existing in selected):
            continue
        selected.append(phrase)
    return selected


def extract_problem_phrases(paper: Paper) -> list[str]:
    sentences = _sentences(paper.summary)
    matches: list[str] = []
    for sentence in sentences:
        lowered = sentence.lower()
        if any(term in lowered for term in PROBLEM_TERMS) or "however" in lowered or "to address" in lowered:
            matches.append(_shorten(sentence, 180))
    return matches


def extract_rare_terms(paper: Paper, idf: dict[str, float]) -> list[str]:
    tokens = [token for token in tokenize(paper.text) if token not in STOPWORDS and len(token) > 3]
    counts = Counter(tokens)
    ranked = sorted(counts, key=lambda token: (idf.get(token, 1.0), counts[token], len(token)), reverse=True)
    return ranked[:8]


def tokenize(text: str) -> list[str]:
    return [match.group(0).lower() for match in TOKEN_RE.finditer(text)]


def _corpus_idf(papers: list[Paper]) -> dict[str, float]:
    doc_count = max(1, len(papers))
    document_frequency: Counter[str] = Counter()
    for paper in papers:
        document_frequency.update(set(tokenize(paper.text)) - STOPWORDS)
    return {
        token: math.log((doc_count + 1) / (frequency + 1)) + 1.0
        for token, frequency in document_frequency.items()
    }


def _rarity_score(paper: Paper, idf: dict[str, float]) -> float:
    tokens = [token for token in tokenize(paper.text) if token not in STOPWORDS and len(token) > 3]
    if not tokens:
        return 0.0
    ranked = sorted((idf.get(token, 1.0) for token in tokens), reverse=True)
    top = ranked[: min(12, len(ranked))]
    return _clamp((sum(top) / len(top) - 1.0) / 1.8)


def _hiddenness_score(paper: Paper, *, now: datetime) -> float:
    age_days = max(1, paper.age_days(now))
    citations = paper.citation_count
    if citations is None:
        return 0.66 if age_days <= 180 else 0.52

    citation_velocity = citations / max(age_days, 14)
    popularity_pressure = min(1.0, math.log1p(citations) / math.log1p(80) * 0.65 + citation_velocity * 0.7)
    hidden = 1.0 - popularity_pressure
    if age_days <= 45 and citations <= 5:
        hidden = max(hidden, 0.82)
    if age_days <= 14 and citations <= 12:
        hidden = max(hidden, 0.74)
    if paper.influential_citation_count is not None:
        hidden -= min(0.15, 0.03 * paper.influential_citation_count)
    return _clamp(hidden)


def _applicability_score(paper: Paper) -> float:
    text = paper.text.lower()
    transfer_hits = sum(text.count(term) for term in TRANSFER_TERMS)
    result_hits = sum(text.count(term) for term in RESULT_TERMS)
    percent_or_number = 1 if re.search(r"\b\d+(\.\d+)?\s*%|\b\d+x\b", text) else 0
    return _clamp(0.08 * transfer_hits + 0.07 * result_hits + 0.14 * percent_or_number)


def _is_interesting_phrase(tokens: list[str], anchors: set[str]) -> bool:
    if any(token in STOPWORDS for token in (tokens[0], tokens[-1])):
        return False
    if tokens[0] in BAD_PHRASE_BOUNDARY_TOKENS or tokens[-1] in BAD_PHRASE_BOUNDARY_TOKENS:
        return False
    if any(token in BAD_PHRASE_TOKENS for token in tokens):
        return False
    if not any(token in anchors for token in tokens):
        return False
    if tokens[-1] not in anchors:
        return False
    if len(set(tokens)) < len(tokens):
        return False
    return sum(1 for token in tokens if token not in STOPWORDS) >= 2


def _phrase(tokens: list[str]) -> str:
    return " ".join(tokens)


def _content_spans(tokens: list[str]) -> list[list[str]]:
    spans: list[list[str]] = []
    current: list[str] = []
    for token in tokens:
        if token in STOPWORDS:
            if current:
                spans.append(current)
                current = []
            continue
        current.append(token)
    if current:
        spans.append(current)
    return spans


def _contains_same_ordered_tokens(candidate: str, existing: str) -> bool:
    candidate_tokens = candidate.split()
    existing_tokens = existing.split()
    if len(candidate_tokens) <= len(existing_tokens):
        return False
    for index in range(0, len(candidate_tokens) - len(existing_tokens) + 1):
        if candidate_tokens[index : index + len(existing_tokens)] == existing_tokens:
            return True
    return False


def _sentences(text: str) -> list[str]:
    return [sentence.strip() for sentence in SENTENCE_RE.split(text) if sentence.strip()]


def _rationale(
    paper: Paper,
    components: dict[str, float],
    techniques: list[str],
    problems: list[str],
    rare_terms: list[str],
    *,
    now: datetime,
) -> list[str]:
    reasons: list[str] = []
    if techniques:
        reasons.append(f"Technique signals: {', '.join(techniques[:3])}.")
    if problems:
        reasons.append(f"Problem hook: {_shorten(problems[0], 140)}")
    if rare_terms:
        reasons.append(f"Uncommon terms in this batch: {', '.join(rare_terms[:5])}.")
    if paper.citation_count is None:
        reasons.append("Popularity signal: citation data unavailable, so hiddenness is treated as neutral-high.")
    else:
        reasons.append(
            f"Popularity signal: {paper.citation_count} citations over {paper.age_days(now)} days."
        )
    strongest = max(components, key=components.__getitem__)
    reasons.append(f"Strongest score component: {strongest}={components[strongest]:.2f}.")
    return reasons


def _shorten(value: str, limit: int) -> str:
    value = " ".join(value.split())
    if len(value) <= limit:
        return value
    return value[: limit - 1].rstrip() + "..."


def _clamp(value: float) -> float:
    return max(0.0, min(1.0, value))
