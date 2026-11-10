from pathlib import Path

import pytest

from foodops_ai.evaluation.retrieval import (
    EvaluationCase,
    evaluate_retrieval,
    load_evaluation_cases,
)
from foodops_ai.knowledge.search_models import SearchResult


class FakeSearcher:
    def __init__(self, results_by_question: dict[str, list[SearchResult]]) -> None:
        self.results_by_question = results_by_question

    def search(self, query: str, limit: int = 5) -> list[SearchResult]:
        return self.results_by_question[query][:limit]


def make_result(source: str) -> SearchResult:
    return SearchResult(
        content="Example context",
        source=source,
        chunk_index=0,
        start_char=0,
        end_char=14,
        distance=0.1,
    )


def test_evaluation_calculates_hit_rate_and_reciprocal_rank() -> None:
    cases = [
        EvaluationCase("question one", "a.md"),
        EvaluationCase("question two", "b.md"),
    ]
    searcher = FakeSearcher(
        {
            "question one": [make_result("a.md"), make_result("b.md")],
            "question two": [make_result("a.md"), make_result("b.md")],
        }
    )

    summary = evaluate_retrieval(cases, searcher, top_k=2)

    assert summary.query_count == 2
    assert summary.hit_rate_at_k == 1.0
    assert summary.mean_reciprocal_rank == 0.75
    assert summary.hits_by_source == {"a.md": 1, "b.md": 1}


def test_evaluation_reports_zero_when_expected_source_is_not_retrieved() -> None:
    cases = [EvaluationCase("question", "expected.md")]
    searcher = FakeSearcher({"question": [make_result("other.md")]})

    summary = evaluate_retrieval(cases, searcher, top_k=1)

    assert summary.hit_rate_at_k == 0.0
    assert summary.mean_reciprocal_rank == 0.0


def test_evaluation_rejects_invalid_top_k() -> None:
    with pytest.raises(ValueError, match="greater than zero"):
        evaluate_retrieval(
            [EvaluationCase("question", "expected.md")],
            FakeSearcher({"question": []}),
            top_k=0,
        )


def test_dataset_contains_forty_labeled_questions() -> None:
    cases = load_evaluation_cases(Path("data/evaluation/retrieval_questions.json"))

    assert len(cases) == 40
    assert {case.expected_source for case in cases} == {
        "late_delivery.md",
        "refund_policy.md",
        "unavailable_items.md",
    }
