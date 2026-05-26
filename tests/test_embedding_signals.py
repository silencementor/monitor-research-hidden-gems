import sys
import types

import numpy as np

from research_hidden_gems.embedding_signals import (
    Embedder,
    _hf_token_kwargs,
    keyword_score,
    outlier_scores,
    relevance_scores,
)


def test_hash_embed_is_deterministic_and_normalized() -> None:
    embedder = Embedder(backend="hashing")
    vecs = embedder.encode(["retrieval augmented generation", "retrieval augmented generation"])
    assert vecs.shape[0] == 2
    np.testing.assert_allclose(vecs[0], vecs[1])
    np.testing.assert_allclose(np.linalg.norm(vecs, axis=1), 1.0, atol=1e-5)


def test_outlier_scores_flag_isolated_point() -> None:
    embedder = Embedder(backend="hashing")
    texts = [
        "graph neural networks for recommendation",
        "graph neural networks for recommendation systems",
        "graph neural network recommender model",
        "a historical treatise on medieval beekeeping and honey",
    ]
    scores = outlier_scores(embedder.encode(texts), k=2)
    assert int(np.argmax(scores)) == 3  # the off-topic text is the semantic outlier


def test_relevance_prefers_profile_topic() -> None:
    embedder = Embedder(backend="hashing")
    profile = "large language models, retrieval, recommender systems"
    texts = [
        "a new retrieval method for language models",
        "protein folding via molecular dynamics simulation",
    ]
    matrix = embedder.encode(texts)
    profile_vec = embedder.encode([profile])[0]
    rel = relevance_scores(matrix, profile_vec, texts, ["retrieval", "language model", "recommender"])
    assert rel[0] > rel[1]


def test_keyword_score_bounds() -> None:
    assert keyword_score("we study retrieval and ranking", ["retrieval", "ranking", "agent"]) > 0
    assert keyword_score("an unrelated sentence", ["retrieval"]) == 0.0


def test_outlier_single_paper_is_neutral() -> None:
    embedder = Embedder(backend="hashing")
    scores = outlier_scores(embedder.encode(["only one paper"]))
    assert list(scores) == [0.5]


def test_hf_token_kwargs_uses_hf_token(monkeypatch) -> None:
    class SentenceTransformer:
        def __init__(self, model_name_or_path: str, *, token: str | None = None) -> None:
            pass

    monkeypatch.setenv("HF_TOKEN", "hf_test")

    assert _hf_token_kwargs(SentenceTransformer) == {"token": "hf_test"}


def test_hf_token_kwargs_supports_legacy_use_auth_token(monkeypatch) -> None:
    class SentenceTransformer:
        def __init__(self, model_name_or_path: str, *, use_auth_token: str | None = None) -> None:
            pass

    monkeypatch.setenv("HF_TOKEN", "hf_test")

    assert _hf_token_kwargs(SentenceTransformer) == {"use_auth_token": "hf_test"}


def test_sentence_transformer_loads_hf_token_from_dotenv(tmp_path, monkeypatch) -> None:
    recorded: dict[str, str | None] = {}

    class SentenceTransformer:
        def __init__(self, model_name_or_path: str, *, token: str | None = None) -> None:
            recorded["model"] = model_name_or_path
            recorded["token"] = token

        def encode(self, texts, normalize_embeddings: bool, show_progress_bar: bool):
            return np.ones((len(texts), 2), dtype=np.float32)

    fake_module = types.ModuleType("sentence_transformers")
    fake_module.SentenceTransformer = SentenceTransformer
    (tmp_path / ".env").write_text("HF_TOKEN=hf_from_dotenv\n")
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("HF_TOKEN", raising=False)
    monkeypatch.setitem(sys.modules, "sentence_transformers", fake_module)

    embedder = Embedder(backend="sentence-transformers", model="sentence-transformers/test-model")

    assert embedder.backend == "sentence-transformers"
    assert recorded == {"model": "sentence-transformers/test-model", "token": "hf_from_dotenv"}
