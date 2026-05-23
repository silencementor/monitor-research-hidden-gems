from research_hidden_gems.arxiv_client import arxiv_id_from_text, build_search_query


def test_arxiv_id_from_url() -> None:
    assert arxiv_id_from_text("https://arxiv.org/abs/2604.24881") == "2604.24881"
    assert arxiv_id_from_text("2604.24881v1") == "2604.24881"


def test_build_search_query_combines_categories_and_dates() -> None:
    query = build_search_query(query="activation steering", categories=["cs.AI", "cs.CL"], days=14)
    assert 'all:"activation steering"' in query
    assert "(cat:cs.AI OR cat:cs.CL)" in query
    assert "submittedDate:[" in query
