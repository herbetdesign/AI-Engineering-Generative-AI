"""Command-line interface for evaluating semantic document retrieval."""

import argparse
from pathlib import Path

from foodops_ai.config import settings
from foodops_ai.evaluation.retrieval import evaluate_retrieval, load_evaluation_cases
from foodops_ai.knowledge import VectorStore


def main() -> int:
    """Evaluate the configured vector store against labeled questions."""

    parser = argparse.ArgumentParser(
        description="Measure semantic retrieval quality without calling an LLM."
    )
    parser.add_argument(
        "--dataset",
        type=Path,
        default=Path("data/evaluation/retrieval_questions.json"),
        help="JSON file containing questions and expected source filenames.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=3,
        help="Number of retrieved chunks to inspect for each question.",
    )
    args = parser.parse_args()

    try:
        cases = load_evaluation_cases(args.dataset)
        store = VectorStore(persist_directory=settings.vector_store_path)
        summary = evaluate_retrieval(cases, store, top_k=args.top_k)
    except (ValueError, OSError) as error:
        parser.error(str(error))

    print(f"Perguntas avaliadas: {summary.query_count}")
    print(f"Top-k: {args.top_k}")
    print(f"Hit Rate@{args.top_k}: {summary.hit_rate_at_k:.1%}")
    print(f"MRR: {summary.mean_reciprocal_rank:.3f}")
    print("Acertos por documento esperado:")
    for source, total in sorted(summary.totals_by_source.items()):
        hits = summary.hits_by_source.get(source, 0)
        print(f"  {source}: {hits}/{total} ({hits / total:.1%})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
