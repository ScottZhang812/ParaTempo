"""Evaluate or summarize ParaTempo result directories."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from paratempo.utils import check_answer


DATASET_TYPES = {
    "aime26": "math",
    "gpqa": "gpqa",
    "hmmt25": "math",
    "hmmt26": "math",
}


def evaluate_single_dir(
    results_dir: str,
    dataset: str,
    force_reeval: bool = False,
) -> dict[str, Any] | None:
    results_path = Path(results_dir)
    summary_path = results_path / "summary.json"
    if not summary_path.exists():
        return None

    with summary_path.open("r", encoding="utf-8") as handle:
        summary = json.load(handle)
    if not force_reeval:
        return summary

    dataset_type = DATASET_TYPES.get(dataset, "math")
    per_question = summary.get("per_question", [])
    correct = 0
    for question in per_question:
        predicted = question.get("predicted_answer")
        gold = question.get("gold_answer")
        if predicted and gold:
            question["is_correct"] = check_answer(predicted, gold, dataset_type)
        if question.get("is_correct"):
            correct += 1

    summary["results"]["accuracy"] = correct / len(per_question) if per_question else 0
    summary["results"]["correct"] = correct
    with summary_path.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, default=str)
    return summary


def print_summary_table(results_base: str) -> None:
    base_path = Path(results_base)
    rows = []
    for result_dir in sorted(base_path.iterdir()):
        if not result_dir.is_dir():
            continue
        summary_path = result_dir / "summary.json"
        if not summary_path.exists():
            continue
        with summary_path.open("r", encoding="utf-8") as handle:
            summary = json.load(handle)
        exp = summary.get("experiment", {})
        res = summary.get("results", {})
        rows.append(
            {
                "dir": result_dir.name,
                "model": exp.get("model", "?").split("/")[-1],
                "dataset": exp.get("dataset", "?"),
                "method": exp.get("method", "?"),
                "N": exp.get("num_branches", "?"),
                "accuracy": res.get("accuracy", 0),
                "avg_tokens": res.get("avg_tokens_per_question", 0),
                "avg_gen": res.get("avg_gen_tokens", 0),
                "avg_probe": res.get("avg_probe_tokens", 0),
                "avg_seq": res.get("avg_sequential_tokens", 0),
                "avg_raw_time": res.get("avg_raw_end_to_end_latency", 0),
                "avg_filt_time": res.get("avg_filtered_stable_backend_latency", 0),
                "forks": res.get("avg_forks", 0),
                "retired": res.get("avg_retired", 0),
            }
        )

    if not rows:
        print("No results found.")
        return

    header = (
        f"{'Model':<20} {'Dataset':<8} {'Method':<15} {'N':>3} "
        f"{'Acc':>7} {'AvgTok':>8} {'Gen':>8} {'Probe':>6} {'Seq':>8} "
        f"{'Raw':>6} {'Filt':>6} {'Fork':>4} {'Ret':>4}"
    )
    print("=" * len(header))
    print(header)
    print("-" * len(header))
    for row in rows:
        print(
            f"{row['model']:<20} {row['dataset']:<8} {row['method']:<15} "
            f"{row['N']:>3} {row['accuracy']:>6.1%} "
            f"{row['avg_tokens']:>8,.0f} {row['avg_gen']:>8,.0f} "
            f"{row['avg_probe']:>6,.0f} {row['avg_seq']:>8,.0f} "
            f"{row['avg_raw_time']:>5.1f}s {row['avg_filt_time']:>5.1f}s "
            f"{row['forks']:>4.1f} {row['retired']:>4.1f}"
        )
    print("=" * len(header))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results_dir", required=True)
    parser.add_argument("--dataset", default=None, choices=list(DATASET_TYPES))
    parser.add_argument("--force_reeval", action="store_true")
    parser.add_argument("--summary", action="store_true")
    args = parser.parse_args()

    if args.summary:
        print_summary_table(args.results_dir)
        return

    if not args.dataset:
        summary_path = Path(args.results_dir) / "summary.json"
        if summary_path.exists():
            with summary_path.open("r", encoding="utf-8") as handle:
                args.dataset = json.load(handle).get("experiment", {}).get("dataset")
    if not args.dataset:
        print("Error: --dataset required")
        return

    result = evaluate_single_dir(args.results_dir, args.dataset, args.force_reeval)
    if result:
        metrics = result["results"]
        print(f"Accuracy: {metrics['accuracy']:.1%} ({metrics['correct']}/{metrics['total']})")
        print(f"Tokens:   {metrics.get('avg_tokens_per_question', 0):,.0f} avg/question")


if __name__ == "__main__":
    main()

