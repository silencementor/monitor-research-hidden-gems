from research_hidden_gems.openalex import (
    VenueSource,
    _arxiv_id_from_work,
    _doi_from_work,
    _premium_work_to_paper,
    _source_id,
    _source_rank,
    _venue_aliases,
)


def test_source_id_normalizes_openalex_urls() -> None:
    assert _source_id({"id": "https://openalex.org/S123456"}) == "S123456"
    assert _source_id({"id": "S987"}) == "S987"
    assert _source_id({"id": "https://openalex.org/W123"}) == ""


def test_venue_aliases_include_known_full_names() -> None:
    assert "Neural Information Processing Systems" in _venue_aliases("NeurIPS")
    assert "Proceedings of the VLDB Endowment" in _venue_aliases("VLDB")


def test_source_rank_prioritizes_matching_venue_names() -> None:
    matching = {
        "display_name": "Proceedings of the VLDB Endowment",
        "abbreviated_title": "PVLDB",
        "type": "journal",
        "works_count": 50000,
    }
    generic = {
        "display_name": "Database Systems Journal",
        "abbreviated_title": "DBSJ",
        "type": "journal",
        "works_count": 50000,
    }

    assert _source_rank(matching, "VLDB") > _source_rank(generic, "VLDB")


def test_source_rank_avoids_sigmod_newsletter_over_proceedings() -> None:
    proceedings = {
        "display_name": "Proceedings of the ACM on Management of Data",
        "abbreviated_title": "",
        "type": "journal",
        "works_count": 1160,
    }
    record = {
        "display_name": "ACM SIGMOD Record",
        "abbreviated_title": "",
        "type": "journal",
        "works_count": 3232,
    }

    assert _source_rank(proceedings, "SIGMOD") > _source_rank(record, "SIGMOD")


def test_source_rank_avoids_empty_stale_sigmod_source() -> None:
    active = {
        "display_name": "Proceedings of the ACM on Management of Data",
        "abbreviated_title": "",
        "type": "journal",
        "works_count": 1160,
    }
    stale = {
        "display_name": "Proceedings - ACM-SIGMOD International Conference on Management of Data",
        "abbreviated_title": "",
        "type": "conference",
        "works_count": 0,
    }

    assert _source_rank(active, "SIGMOD") > _source_rank(stale, "SIGMOD")


def test_source_rank_avoids_icmla_when_resolving_icml() -> None:
    icml = {
        "display_name": "International Conference on Machine Learning",
        "abbreviated_title": "",
        "type": "conference",
        "works_count": 1907,
    }
    icmla = {
        "display_name": "IEEE International Conference on Machine Learning and Applications (ICMLA)",
        "abbreviated_title": "",
        "type": "conference",
        "works_count": 285,
    }

    assert _source_rank(icml, "ICML") > _source_rank(icmla, "ICML")


def test_premium_work_to_paper_keeps_doi_venue_and_urls() -> None:
    work = {
        "id": "https://openalex.org/W123",
        "doi": "https://doi.org/10.1145/1234567",
        "title": "Graph Vector Indexing for Agent Retrieval",
        "abstract_inverted_index": {"Graph": [0], "retrieval": [2], "works": [1]},
        "publication_date": "2026-05-20",
        "cited_by_count": 2,
        "authorships": [{"author": {"display_name": "A. Researcher"}}],
        "primary_location": {
            "landing_page_url": "https://dl.acm.org/doi/10.1145/1234567",
            "source": {
                "id": "https://openalex.org/S123",
                "display_name": "ACM SIGMOD Conference",
            },
        },
        "best_oa_location": {"pdf_url": "https://example.org/paper.pdf"},
    }

    paper = _premium_work_to_paper(
        work,
        {"S123": VenueSource(query="SIGMOD", source_id="S123", display_name="ACM SIGMOD Conference")},
    )

    assert paper is not None
    assert paper.source == "premium_venues"
    assert paper.key == "10.1145/1234567"
    assert paper.venue == "ACM SIGMOD Conference"
    assert paper.abs_url == "https://dl.acm.org/doi/10.1145/1234567"
    assert paper.pdf_url == "https://example.org/paper.pdf"
    assert paper.citation_count == 2
    assert paper.external_ids["premium_venue_query"] == "SIGMOD"
    assert paper.summary == "Graph works retrieval"


def test_doi_from_work_strips_doi_url_prefix() -> None:
    assert _doi_from_work({"doi": "https://doi.org/10.5555/ABC"}) == "10.5555/ABC"
    assert _doi_from_work({"doi": None}) is None


def test_arxiv_id_from_work_checks_oa_locations() -> None:
    work = {
        "doi": "https://doi.org/10.14778/3796195.3796212",
        "primary_location": {"landing_page_url": "https://doi.org/10.14778/3796195.3796212"},
        "best_oa_location": {
            "landing_page_url": "http://arxiv.org/abs/2505.02922",
            "pdf_url": "https://arxiv.org/pdf/2505.02922",
        },
    }

    assert _arxiv_id_from_work(work) == "2505.02922"
