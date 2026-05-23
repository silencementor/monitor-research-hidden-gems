import numpy as np

from research_hidden_gems.embedding_signals import (
    Embedder,
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
