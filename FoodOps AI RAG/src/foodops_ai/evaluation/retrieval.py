"""Measure semantic retrieval quality against labeled questions."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol

from foodops_ai.knowledge.search_models import SearchResult


@dataclass(frozen=True)
class EvaluationCase:
    """A question and the document expected to contain its answer."""

    question: str
    expected_source: str


@dataclass
class RetrievalEvaluationSummary:
    """Aggregate retrieval metrics for a set of evaluation questions."""

    query_count: int
    hit_count: int = 0
    reciprocal_rank_sum: float = 0.0
    hits_by_source: dict[str, int] = field(default_factory=dict)
    totals_by_source: dict[str, int] = field(default_factory=dict)

    @property
    def hit_rate_at_k(self) -> float:
        """Return the fraction of questions with a relevant result in top-k."""

        return self.hit_count / self.query_count if self.query_count else 0.0

    @property
    def mean_reciprocal_rank(self) -> float:
        """Return the mean reciprocal rank of the first relevant result."""

        return self.reciprocal_rank_sum / self.query_count if self.query_count else 0.0


class Searcher(Protocol):
    """Minimal interface required by the retrieval evaluator."""

    def search(self, query: str, limit: int = 5) -> list[SearchResult]:
        """Return the most relevant chunks for a query."""


def load_evaluation_cases(path: Path) -> list[EvaluationCase]:
    """Load and validate a JSON list of labeled retrieval questions."""

    try:
        raw_cases = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        message = f"Could not read evaluation dataset path: {error}"
        raise ValueError(message) from error

    if not isinstance(raw_cases, list) or not raw_cases:
        raise ValueError("The evaluation dataset must be a non-empty JSON list.")

    cases: list[EvaluationCase] = []
    for index, raw_case in enumerate(raw_cases, start=1):
        if not isinstance(raw_case, dict):
            raise ValueError(f"Evaluation case {index} must be a JSON object.")

        question = raw_case.get("question")
        expected_source = raw_case.get("expected_source")
        if not isinstance(question, str) or not question.strip():
            raise ValueError(f"Evaluation case {index} needs a non-empty question.")
        if not isinstance(expected_source, str) or not expected_source.strip():
            raise ValueError(f"Evaluation case {index} needs an expected_source.")

        cases.append(
            EvaluationCase(
                question=question.strip(),
                expected_source=expected_source.strip(),
            )
        )

    return cases


def evaluate_retrieval(
    cases: list[EvaluationCase], searcher: Searcher, top_k: int = 3
) -> RetrievalEvaluationSummary:
    """Measure hit rate and reciprocal rank over the first top_k results."""

    if top_k <= 0:
        raise ValueError("top_k must be greater than zero.")
    if not cases:
        raise ValueError("At least one evaluation case is required.")

    summary = RetrievalEvaluationSummary(query_count=len(cases))
    for case in cases:
        summary.totals_by_source[case.expected_source] = (
            summary.totals_by_source.get(case.expected_source, 0) + 1
        )
        results = searcher.search(case.question, limit=top_k)
        first_relevant_rank = next(
            (
                rank
                for rank, result in enumerate(results, start=1)
                if result.source == case.expected_source
            ),
            None,
        )
        if first_relevant_rank is not None:
            summary.hit_count += 1
            summary.reciprocal_rank_sum += 1 / first_relevant_rank
            summary.hits_by_source[case.expected_source] = (
                summary.hits_by_source.get(case.expected_source, 0) + 1
            )

    return summary
