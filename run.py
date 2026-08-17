"""Main experiment runner for ParaTempo.

Example:
    python run.py \
        --model Qwen/Qwen3.5-35B-A3B \
        --model_type qwen \
        --dataset aime26 \
        --port PORT_XXX \
        --output_dir results/paratempo_aime26
"""
import argparse
import asyncio
import json
import logging
import math
import pickle
import time
from datetime import datetime
from pathlib import Path

from paratempo.utils import (
    create_client,
    format_prompt,
    extract_boxed_answer,
    check_answer,
    warmup_backend,
)
from paratempo.core import paratempo, ParaTempoConfig


# ============= CONSTANTS =============

DATASET_TYPES = {
    "aime26": "math",
    "gpqa": "gpqa",
    "hmmt25": "math",
    "hmmt26": "math",
}

SUPPORTED_MODELS = ("Qwen/Qwen3.5-35B-A3B", "openai/gpt-oss-20b")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

DEFAULT_BASE_SEED = 20260423
REPEAT_SEED_STRIDE = 10_000_000
QUESTION_SEED_STRIDE = 10_000


# ============= DATASET LOADING =============

def load_benchmark(dataset_name: str):
    """Load benchmark dataset. Returns list of dicts with 'problem' and 'answer'."""
    from datasets import load_dataset

    if dataset_name == "aime26":
        ds = load_dataset("math-ai/aime26")["test"]
        return [{"problem": row["problem"], "answer": str(row["answer"])} for row in ds]
    elif dataset_name == "gpqa":
        data_path = Path(__file__).parent / "data" / "gpqa" / "gpqa_diamond_test.jsonl"
        if not data_path.exists():
            raise FileNotFoundError(f"GPQA dataset not found at {data_path}")
        with open(data_path, "r", encoding="utf-8") as f:
            data = [json.loads(line.strip()) for line in f]
        for item in data:
            ans = item["answer"].strip()
            extracted = extract_boxed_answer(ans, "gpqa")
            if extracted and len(extracted) == 1 and extracted.upper() in "ABCD":
                item["answer"] = extracted.upper()
        return data
    elif dataset_name == "hmmt25":
        ds = load_dataset("MathArena/hmmt_nov_2025")["train"]
        return [{"problem": row["problem"], "answer": str(row["answer"])} for row in ds]
    elif dataset_name == "hmmt26":
        ds = load_dataset("MathArena/hmmt_feb_2026")["train"]
        return [{"problem": row["problem"], "answer": str(row["answer"])} for row in ds]
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")


def derive_question_seed(base_seed: int, repeat_index: int, qid: int) -> int:
    return base_seed + repeat_index * REPEAT_SEED_STRIDE + qid * QUESTION_SEED_STRIDE


# ============= EXPERIMENT =============

def _mean(values):
    return sum(values) / len(values) if values else 0.0


def _std(values):
    if len(values) <= 1:
        return 0.0
    mu = _mean(values)
    return math.sqrt(sum((v - mu) ** 2 for v in values) / len(values))


async def run_experiment(args):
    from tqdm import tqdm

    dataset_type = DATASET_TYPES[args.dataset]
    effective_fork_mode = "no-fork" if args.disable_fork else args.fork_mode
    effective_retire_stability_windows = 0 if args.disable_retire else args.retire_stability_windows

    logger.info(f"Loading dataset: {args.dataset}")
    data = load_benchmark(args.dataset)
    logger.info(f"  Loaded {len(data)} questions")
    if args.qid is not None:
        if args.qid < 0 or args.qid >= len(data):
            raise ValueError(f"--qid must be in [0, {len(data) - 1}], got {args.qid}")
        selected_qids = [args.qid]
        logger.info(f"  Single-question mode: qid={args.qid}")
    else:
        selected_qids = list(range(len(data)))

    client = create_client(args.model, port=args.port, api_key=args.api_key)

    config = ParaTempoConfig(
        num_branches=args.num_branches,
        probe_interval=args.probe_interval,
        max_tokens_per_branch=args.max_tokens,
        window=args.window,
        num_warmup=args.num_warmup,
        enable_pruning=not args.disable_prune,
        dynamic_prune_percentile=args.dynamic_prune_percentile,
        enable_retirement=not args.disable_retire,
        retire_stability_windows=effective_retire_stability_windows,
        theta_retire=args.theta_retire,
        temperature=args.temperature,
        top_p=args.top_p,
        fork_temp_increment=args.fork_temp_increment,
        fork_mode=effective_fork_mode,
        global_early_stop_enabled=not args.disable_global_early_stop,
        global_early_stop_confidence_fraction=args.global_early_stop_confidence_fraction,
    )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Resume
    results = []
    correct_count = 0
    total_tokens_all = 0
    total_raw_latency = 0.0
    total_filtered_latency = 0.0

    selected_qid_set = set(selected_qids)
    existing_by_qid = {}
    existing_pkls = sorted(output_dir.glob("qid*.pkl"), key=lambda p: int(p.stem[3:]))
    for pkl_path in existing_pkls:
        qid_from_name = int(pkl_path.stem[3:])
        if qid_from_name not in selected_qid_set:
            continue
        with open(pkl_path, "rb") as f:
            saved = pickle.load(f)
        question_result = {k: v for k, v in saved.items() if k != "full_result"}
        existing_by_qid[qid_from_name] = question_result
        results.append(question_result)
        if question_result["is_correct"]:
            correct_count += 1
        total_tokens_all += question_result["total_tokens"]
        total_raw_latency += question_result.get("raw_end_to_end_latency", question_result.get("elapsed_time", 0.0))
        total_filtered_latency += question_result.get("filtered_stable_backend_latency", question_result.get("elapsed_time", 0.0))

    if results:
        logger.info(f"Resumed {len(results)}/{len(selected_qids)} selected questions from {output_dir}")

    for _, qid in tqdm(list(enumerate(selected_qids)), total=len(selected_qids), desc="ParaTempo"):
        if qid in existing_by_qid:
            continue

        item = data[qid]
        question_prompt = format_prompt(item, dataset_type)
        gold_answer = item["answer"]
        question_seed = derive_question_seed(args.base_seed, args.repeat_index, qid)

        warmup_timings = await warmup_backend(
            client, args.model, question_prompt, args.model_type,
            seed=question_seed - 2,
            include_probe=True,
            probe_top_logprobs=config.probe_top_logprobs,
        )

        start = time.time()
        result = await paratempo(
            client, args.model, question_prompt, args.model_type, config, dataset_type,
            base_seed=question_seed,
        )
        raw_latency = time.time() - start
        initialization_time = result.get("initialization_time", 0.0)
        filtered_latency = max(0.0, raw_latency - initialization_time)

        predicted = result["final_answer"]
        is_correct = check_answer(predicted, gold_answer, dataset_type)

        question_result = {
            "qid": qid,
            "question": item["problem"][:200],
            "gold_answer": gold_answer,
            "predicted_answer": f"\\boxed{{{predicted}}}" if predicted else predicted,
            "is_correct": is_correct,
            "total_tokens": result["total_tokens"],
            "total_gen_tokens": result["total_gen_tokens"],
            "total_probe_tokens": result["total_probe_tokens"],
            "sequential_tokens": result["sequential_tokens"],
            "dynamic_prune_threshold": result.get("dynamic_prune_threshold"),
            "prune_enabled": result.get("prune_enabled", not args.disable_prune),
            "num_forks": result["num_forks"],
            "fork_enabled": result.get(
                "fork_enabled",
                (not args.disable_prune) and effective_fork_mode != "no-fork",
            ),
            "retire_enabled": result.get("retire_enabled", effective_retire_stability_windows > 0),
            "num_retired": result["num_retired"],
            "num_pruned": result.get("num_pruned", 0),
            "active_branches_final": result.get("active_branches_final", args.num_branches),
            "early_stopped": result["early_stopped"],
            "early_stop_type": result.get("early_stop_type"),
            "global_early_stop_answer": (
                result.get("global_early_stop") or {}
            ).get("winning_answer"),
            "global_early_stop_confidence_sum": (
                result.get("global_early_stop") or {}
            ).get("winning_answer_confidence_sum"),
            "global_early_stop_total_confidence_sum": (
                result.get("global_early_stop") or {}
            ).get("total_confidence_sum"),
            "global_early_stop_threshold": (
                result.get("global_early_stop") or {}
            ).get("threshold"),
            "raw_end_to_end_latency": raw_latency,
            "filtered_stable_backend_latency": filtered_latency,
            "elapsed_time": raw_latency,
            "initialization_time": initialization_time,
            "backend_warmup_timings": warmup_timings,
        }
        results.append(question_result)

        if is_correct:
            correct_count += 1
        total_tokens_all += result["total_tokens"]
        total_raw_latency += raw_latency
        total_filtered_latency += filtered_latency

        pkl_path = output_dir / f"qid{qid}.pkl"
        with open(pkl_path, "wb") as f:
            pickle.dump({**question_result, "full_result": result}, f)

        completed = len(results)
        if completed % 5 == 0 or completed == len(selected_qids):
            acc = correct_count / completed * 100
            logger.info(
                f"  [{completed}/{len(selected_qids)}] qid={qid} Acc={acc:.1f}% | "
                f"Tokens={result['total_tokens']:,} (gen={result['total_gen_tokens']:,} probe={result['total_probe_tokens']:,}) | "
                f"Forks={result['num_forks']} Retired={result['num_retired']} Pruned={result.get('num_pruned', 0)} | "
                f"Latency raw={raw_latency:.1f}s filt={filtered_latency:.1f}s | "
                f"{'correct' if is_correct else 'wrong'}"
            )

    # ============= SUMMARY =============
    n = len(results) if results else 1
    correctness_values = [1.0 if r["is_correct"] else 0.0 for r in results]
    raw_latencies = [
        r.get("raw_end_to_end_latency", r.get("elapsed_time", 0.0))
        for r in results
    ]
    filtered_latencies = [
        r.get("filtered_stable_backend_latency", r.get("elapsed_time", 0.0))
        for r in results
    ]
    wrong_qids = sorted(r["qid"] for r in results if not r["is_correct"])
    early_stop_rate = sum(1 for r in results if r.get("early_stopped")) / len(results) if results else 0.0
    accuracy = correct_count / n
    summary = {
        "experiment": {
            "method": "paratempo",
            "model": args.model,
            "model_type": args.model_type,
            "dataset": args.dataset,
            "num_branches": args.num_branches,
            "max_tokens": args.max_tokens,
            "probe_interval": args.probe_interval,
            "temperature": args.temperature,
            "top_p": args.top_p,
            "window": args.window,
            "num_warmup": args.num_warmup,
            "prune_enabled": not args.disable_prune,
            "fork_fresh_prune_probe_requirement": args.window,
            "fork_fresh_donor_probe_requirement": args.window,
            "fork_fresh_retire_probe_requirement": 0 if effective_retire_stability_windows > 0 else None,
            "dynamic_prune_percentile": args.dynamic_prune_percentile,
            "retire_enabled": effective_retire_stability_windows > 0,
            "retire_stability_windows": effective_retire_stability_windows,
            "theta_retire": args.theta_retire,
            "fork_temp_increment": args.fork_temp_increment,
            "fork_mode": effective_fork_mode,
            "fork_enabled": (not args.disable_prune) and effective_fork_mode != "no-fork",
            "disable_prune": args.disable_prune,
            "disable_retire": args.disable_retire,
            "disable_fork": args.disable_fork,
            "global_early_stop_enabled": not args.disable_global_early_stop,
            "global_early_stop_confidence_fraction": args.global_early_stop_confidence_fraction,
            "global_early_stop_threshold": args.num_branches * args.global_early_stop_confidence_fraction,
            "global_early_stop_threshold_scope": "active_branches",
            "base_seed": args.base_seed,
            "repeat_index": args.repeat_index,
            "qid": args.qid,
            "timestamp": datetime.now().isoformat(),
        },
        "results": {
            "accuracy": accuracy,
            "mean_accuracy": accuracy,
            "std_accuracy": _std(correctness_values),
            "correct": correct_count,
            "total": len(results),
            "total_tokens": total_tokens_all,
            "avg_tokens_per_question": total_tokens_all / n,
            "avg_gen_tokens": sum(r["total_gen_tokens"] for r in results) / n,
            "avg_probe_tokens": sum(r["total_probe_tokens"] for r in results) / n,
            "avg_sequential_tokens": sum(r["sequential_tokens"] for r in results) / n,
            "avg_raw_end_to_end_latency": total_raw_latency / n,
            "std_raw_end_to_end_latency": _std(raw_latencies),
            "avg_filtered_stable_backend_latency": total_filtered_latency / n,
            "std_filtered_stable_backend_latency": _std(filtered_latencies),
            "avg_forks": sum(r["num_forks"] for r in results) / n,
            "avg_retired": sum(r["num_retired"] for r in results) / n,
            "avg_pruned": sum(r.get("num_pruned", 0) for r in results) / n,
            "avg_active_branches_final": sum(r.get("active_branches_final", args.num_branches) for r in results) / n,
            "early_stop_rate": early_stop_rate,
            "wrong_qids": wrong_qids,
        },
        "per_question": results,
    }

    summary_path = output_dir / "summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2, default=str)

    summary_log_lines = [
        "=" * 70,
        f"EXPERIMENT COMPLETE: ParaTempo @{args.num_branches}",
        f"  Model:          {args.model}",
        f"  Dataset:        {args.dataset} ({len(results)} questions)",
        f"  Prune Enabled:  {not args.disable_prune}",
        f"  Retire Enabled: {effective_retire_stability_windows > 0}",
        f"  Fork Mode:      {effective_fork_mode}",
        f"  Mean Acc:       {accuracy:.1%} ({correct_count}/{len(results)})",
        f"  Std Acc:        {_std(correctness_values):.3f}",
        f"  Mean Latency:   raw={total_raw_latency/n:.1f}s filt={total_filtered_latency/n:.1f}s",
        f"  Std Latency:    raw={_std(raw_latencies):.1f}s filt={_std(filtered_latencies):.1f}s",
        f"  Avg Tokens:     {total_tokens_all/n:,.0f}",
        f"  Avg SeqTokens:  {sum(r['sequential_tokens'] for r in results)/n:,.0f}",
        f"  Avg Forks:      {sum(r['num_forks'] for r in results)/n:.1f}",
        f"  Avg Retired:    {sum(r['num_retired'] for r in results)/n:.1f}",
        f"  Avg Pruned:     {sum(r.get('num_pruned', 0) for r in results)/n:.1f}",
        f"  Early Stop:     {early_stop_rate:.0%}",
        f"  Wrong QIDs:     {wrong_qids}",
        f"  Output:         {output_dir}",
        "=" * 70,
    ]
    with open(output_dir / "summary.log", "w") as f:
        f.write("\n".join(summary_log_lines) + "\n")

    for line in summary_log_lines:
        logger.info(line)
    return summary


# ============= CLI =============

def parse_args():
    p = argparse.ArgumentParser(description="ParaTempo Benchmark Runner")
    p.add_argument("--model", default="Qwen/Qwen3.5-35B-A3B", choices=SUPPORTED_MODELS)
    p.add_argument("--model_type", default="qwen", choices=["qwen", "gpt"])
    p.add_argument("--port", type=int, default=None)
    p.add_argument("--api_key", default=None)
    p.add_argument("--dataset", required=True, choices=["aime26", "gpqa", "hmmt25", "hmmt26"])
    p.add_argument("--num_branches", type=int, default=16)
    p.add_argument("--max_tokens", type=int, default=16384)
    p.add_argument("--temperature", type=float, default=0.6)
    p.add_argument("--top_p", type=float, default=0.95)
    p.add_argument("--probe_interval", type=int, default=500)
    p.add_argument("--window", type=int, default=7, help="W: SWGD sliding window size")
    p.add_argument("--num_warmup", type=int, default=15, help="N_warmup: no dynamic control during the first N_warmup probes")
    p.add_argument("--disable_prune", action="store_true", help="disable pruning; fork is also effectively disabled because it is prune-triggered")
    p.add_argument("--dynamic_prune_percentile", type=float, default=0.50, help="warmup percentile used to set dynamic prune threshold")
    p.add_argument("--retire_stability_windows", type=int, default=9, help="X: consecutive probe-time SWGD top-1 probabilities above theta_retire; <=0 disables")
    p.add_argument("--disable_retire", action="store_true", help="shortcut for --retire_stability_windows 0")
    p.add_argument("--theta_retire", type=float, default=0.90, help="theta_retire: retire threshold for consecutive SWGD top-1 probabilities")
    p.add_argument("--fork_temp_increment", type=float, default=0.05)
    p.add_argument("--fork_mode", choices=["fork-low", "fork-high", "no-fork"], default="fork-low", help="donor selection mode after prune; no-fork disables fork for ablation")
    p.add_argument("--disable_fork", action="store_true", help="shortcut for --fork_mode no-fork")
    p.add_argument("--disable_global_early_stop", action="store_true", help="disable global early-stop by answer-bucketed latest SWGD top-1 probabilities")
    p.add_argument("--global_early_stop_confidence_fraction", type=float, default=0.5, help="stop when one answer bucket's sum(latest top1_prob) > num_branches * this fraction")
    p.add_argument("--output_dir", default="./results")
    p.add_argument("--base_seed", type=int, default=DEFAULT_BASE_SEED)
    p.add_argument("--repeat_index", type=int, default=0)
    p.add_argument("--qid", type=int, default=None, help="run only one question id")
    return p.parse_args()


def main():
    args = parse_args()
    asyncio.run(run_experiment(args))


if __name__ == "__main__":
    main()
