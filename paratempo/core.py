"""ParaTempo: temporal-confidence control for parallel reasoning.

Each branch runs as an independent async coroutine. ParaTempo aggregates the
last W probes' first-token logprobs into a Sliding-Window Group Distribution
(SWGD), deriving a unified Group Entropy metric that drives all decisions:

  - Prune:  H_group > θ_prune → branch is chaotic → replace with fork.
  - Retire: latest X probe-time SWGD windows all have top-1 probability
            above θ_retire → branch converged → stop generating, keep for
            final vote.
  - θ_prune is estimated from the explicit no-intervention phase N_warmup.
  - Termination: all branches finished/retired or global early-stop triggers
    → weighted majority vote.
"""
import asyncio
import logging
import math
import time
from dataclasses import dataclass
from typing import Optional

from .utils import (
    _extract_boxed_raw,
    build_chunk_messages,
    build_probe_messages,
    compute_swgd,
    create_model_response,
    extract_final_text_from_response,
    extract_text_from_response,
    first_token_top_logprobs,
    is_branch_finished,
    reasoning_stop_tokens,
    weighted_majority_vote,
)

logger = logging.getLogger(__name__)


@dataclass
class ParaTempoConfig:
    num_branches: int = 16
    probe_interval: int = 500        # Δ: tokens per chunk
    max_tokens_per_branch: int = 16384
    window: int = 7                  # W: sliding window size for SWGD
    num_warmup: int = 15             # N_warmup: no dynamic control during the first N_warmup probes
    enable_pruning: bool = True
    dynamic_prune_percentile: float = 0.50  # Warmup percentile used for θ_prune
    enable_retirement: bool = True
    retire_stability_windows: int = 9  # X: consecutive probe-time SWGD top-1 probs above θ_retire; <=0 disables
    theta_retire: float = 0.90       # θ_retire: confidence threshold for Retire
    temperature: float = 0.6
    top_p: float = 0.95
    fork_temp_increment: float = 0.05
    fork_mode: str = "fork-low"      # fork-low/high: donor selection; no-fork: disable fork ablation
    global_early_stop_enabled: bool = True
    global_early_stop_confidence_fraction: float = 0.5  # Trigger when one answer bucket's sum(top1_prob) > N * this value
    probe_max_tokens: int = 20
    probe_top_logprobs: int = 20     # L: top logprobs per probe


# ============= LOW-LEVEL API CALLS =============

async def _generate_chunk(client, model, question_prompt, prefix, model_type, config, seed, temperature, max_chunk_tokens):
    """Generate Δ tokens of reasoning for one branch."""
    messages, extra_body = build_chunk_messages(question_prompt, prefix, model_type)
    try:
        response = await create_model_response(
            client, model, messages, extra_body, model_type,
            temperature=temperature, top_p=config.top_p,
            max_tokens=max_chunk_tokens,
            stop=reasoning_stop_tokens(model_type),
            seed=seed,
        )
        new_text = extract_text_from_response(response, model_type)
        tokens = response.usage.completion_tokens if response.usage else 0
        finish_reason = response.choices[0].finish_reason or ""
        finished = is_branch_finished(new_text, model_type)
        if finish_reason in ("stop", "eos"):
            finished = True
        elif not new_text and tokens == 0:
            # No text and no token progress: stop this branch instead of spinning.
            finished = True
        return {"new_text": new_text, "tokens": tokens, "finished": finished, "error": None}
    except Exception as e:
        logger.warning(f"Chunk generation error: {e}")
        return {"new_text": "", "tokens": 0, "finished": True, "error": str(e)}


async def _probe_branch_with_logprobs(client, model, question_prompt, prefix, model_type, dataset_type, config, seed):
    """Probe a branch: extract answer, FTE, and raw first-token logprobs for SWGD."""
    messages, extra_body = build_probe_messages(question_prompt, prefix, model_type)
    try:
        response = await create_model_response(
            client, model, messages, extra_body, model_type,
            temperature=0.0, top_p=1.0,
            max_tokens=config.probe_max_tokens,
            stop=["<|end|>", "<|return|>"] if model_type == "gpt" else None,
            seed=seed,
            logprobs=True,
            top_logprobs=config.probe_top_logprobs,
        )
        text = extract_final_text_from_response(response, model_type)

        # --- Extract answer ---
        raw_probe_text = "The final answer is \\boxed{" + text
        boxed_content = _extract_boxed_raw(raw_probe_text)
        answer = None
        if boxed_content is not None:
            try:
                from math_verify import parse
                parsed = parse(boxed_content)
                if parsed:
                    answer = str(parsed[0])
            except Exception:
                pass
            if answer is None:
                answer = boxed_content

        # --- Extract first-token logprobs + compute FTE ---
        fte = 0.0
        first_token_logprobs = {}
        first_token_logprobs = first_token_top_logprobs(response)
        if first_token_logprobs:
            # FTE is retained as a scalar probe-confidence diagnostic.
            probs = {tok: math.exp(lp) for tok, lp in first_token_logprobs.items()}
            total = sum(probs.values())
            if total > 0:
                probs = {tok: p / total for tok, p in probs.items()}
                fte = -sum(p * math.log(p) for p in probs.values() if p > 0)

        tokens = response.usage.completion_tokens if response.usage else 0
        return {
            "answer": answer,
            "fte": fte,
            "first_token_logprobs": first_token_logprobs,
            "raw_probe_text": raw_probe_text,
            "tokens": tokens,
            "error": None,
        }
    except Exception as e:
        logger.warning(f"Probe error: {e}")
        return {"answer": None, "fte": 0.0, "first_token_logprobs": {}, "tokens": 0, "error": str(e)}


# ============= MAIN ALGORITHM =============

async def paratempo(
    client,
    model: str,
    question_prompt: str,
    model_type: str,
    config: ParaTempoConfig,
    dataset_type: str = "math",
    base_seed: int = 0,
) -> dict:
    """Run ParaTempo — SWGD-based prune/retire/fork plus global early-stop."""

    if config.fork_mode not in ("fork-low", "fork-high", "no-fork"):
        raise ValueError(f"Unknown fork_mode: {config.fork_mode}")

    K = config.num_branches
    start_time = time.time()
    fork_counter = 0
    phase_timings = {
        "initialization_time": 0.0,
        "cumulative_generate_request_time": 0.0,
        "cumulative_probe_request_time": 0.0,
    }
    global_early_stop_event = asyncio.Event()
    global_early_stop_initial_threshold = K * config.global_early_stop_confidence_fraction
    global_early_stop_info = None
    init_ready_branches = set()
    warmup_group_entropies = []
    initial_warmup_complete = False
    dynamic_prune_threshold = None

    # ========== SHARED STATE ==========
    lock = asyncio.Lock()

    branches = []
    for i in range(K):
        branches.append({
            "id": i,
            "prefix": "",
            "seed": base_seed + i,
            "temperature": config.temperature,
            "gen_tokens": 0,
            "spent_gen_tokens": 0,
            "probe_tokens": 0,
            "spent_probe_tokens": 0,
            "naturally_finished": False,
            "retired": False,
            "pruned": False,
            "histories": [],
            "probe_count": 0,
            "born_at_step": 0,
            "fresh_probe_count_since_fork": 0,
            "fork_count": 0,
            "last_fork_parent": None,
            "last_fork_inherited_gen_tokens": None,
            "last_fork_prefix_chars": 0,
            "stopped_by_global_early_stop": False,
            "swgd_cache": None,
        })

    lifecycle_events = []
    probe_log = []

    # ========== HELPERS ==========
    def _is_swgd_ready(br) -> bool:
        """A branch can participate in SWGD decisions only after a full window."""
        return br["probe_count"] >= config.window

    def _control_start_probe() -> int:
        """Earliest probe count at which prune/retire/fork may intervene."""
        return max(0, config.num_warmup)

    def _is_warmup_done(br) -> bool:
        """An original branch exits initial warmup once N_warmup is reached or it is terminal."""
        return (
            br["probe_count"] >= _control_start_probe()
            or br["naturally_finished"]
            or br["retired"]
            or br["pruned"]
        )

    def _get_swgd(br):
        """Get SWGD for a branch, using cache if valid."""
        if not _is_swgd_ready(br):
            return None
        if br["swgd_cache"] is None and br["histories"]:
            br["swgd_cache"] = compute_swgd(br["histories"], config.window)
        return br["swgd_cache"]

    def _text_tail(text: str, max_chars: int = 160) -> str:
        """Keep a short suffix for debug-friendly event payloads."""
        if not text:
            return ""
        return text if len(text) <= max_chars else text[-max_chars:]

    def _record_warmup_swgd_sample(br):
        """Collect one original-branch SWGD sample during the no-intervention phase."""
        if dynamic_prune_threshold is not None or br["last_fork_parent"] is not None:
            return
        if config.window <= br["probe_count"] <= _control_start_probe():
            swgd = compute_swgd(br["histories"], config.window)
            warmup_group_entropies.append(swgd["group_entropy"])

    def _fork_fresh_prune_ready(br) -> bool:
        """A forked branch may be pruned only after one fresh W-sized window."""
        if br["last_fork_parent"] is None:
            return True
        return br["fresh_probe_count_since_fork"] >= config.window

    def _retire_enabled() -> bool:
        return config.enable_retirement and config.retire_stability_windows > 0

    def _fork_enabled() -> bool:
        return config.enable_pruning and config.fork_mode != "no-fork"

    def _fork_fresh_donor_ready(br) -> bool:
        """A forked branch may serve as donor only after one fresh W-sized window."""
        if br["last_fork_parent"] is None:
            return True
        return br["fresh_probe_count_since_fork"] >= config.window

    def _fork_fresh_retire_requirement() -> Optional[int]:
        """Fresh probes required before a forked branch can retire."""
        if not _retire_enabled():
            return None
        return 0

    def _fork_fresh_retire_ready(br) -> bool:
        """Retire has no fork-specific fresh-probe guard."""
        return _retire_enabled()

    def _recent_retire_top1_probs(br):
        """Return latest X probe-time top-1 probs if all exceed θ_retire."""
        X = config.retire_stability_windows
        W = config.window
        histories = br["histories"]
        if X <= 0 or len(histories) < X:
            return None

        top1_probs = []
        for end_idx in range(len(histories) - X, len(histories)):
            if end_idx + 1 < W:
                return None
            window_histories = histories[end_idx - W + 1:end_idx + 1]
            swgd = compute_swgd(window_histories, W)
            top1_prob = float(swgd.get("top1_prob", 0.0) or 0.0)
            if top1_prob <= config.theta_retire:
                return None
            top1_probs.append(top1_prob)

        return top1_probs

    def _maybe_finalize_dynamic_prune_threshold():
        nonlocal dynamic_prune_threshold, initial_warmup_complete
        if not initial_warmup_complete:
            if not all(_is_warmup_done(br) for br in branches):
                return
            initial_warmup_complete = True

        if dynamic_prune_threshold is not None or not warmup_group_entropies:
            return

        sorted_samples = sorted(warmup_group_entropies)
        rank = math.ceil(len(sorted_samples) * config.dynamic_prune_percentile) - 1
        rank = min(max(rank, 0), len(sorted_samples) - 1)
        dynamic_prune_threshold = sorted_samples[rank]
        lifecycle_events.append({
            "event": "dynamic_prune_threshold_ready",
            "window": config.window,
            "num_warmup": config.num_warmup,
            "retire_stability_windows": config.retire_stability_windows,
            "theta_retire": config.theta_retire,
            "control_start_probe": _control_start_probe(),
            "sample_count": len(sorted_samples),
            "percentile": config.dynamic_prune_percentile,
            "rank_index": rank,
            "dynamic_prune_threshold": dynamic_prune_threshold,
        })
        logger.info(
            "Dynamic prune threshold ready: theta=%.4f from %d warmup SWGD samples at percentile %.2f",
            dynamic_prune_threshold,
            len(sorted_samples),
            config.dynamic_prune_percentile,
        )

    def _get_latest_group_confidence(br) -> float:
        """Return the latest available group confidence for final weighted voting."""
        if not br["histories"]:
            return 0.0
        swgd = _get_swgd(br)
        if swgd is None:
            swgd = compute_swgd(br["histories"], config.window)
        return float(swgd.get("top1_prob", 0.0) or 0.0)

    def _latest_vote_snapshot(require_complete_swgd: bool = False):
        """Collect latest answers and SWGD top-1 probabilities for voting."""
        answers = []
        confidences = []
        ready_count = 0
        active_count = 0
        for br in branches:
            if br["pruned"]:
                answers.append(None)
                confidences.append(0.0)
                continue
            active_count += 1
            answers.append(br["histories"][-1]["answer"] if br["histories"] else None)
            if require_complete_swgd and not _is_swgd_ready(br):
                confidences.append(0.0)
                continue
            if _is_swgd_ready(br):
                ready_count += 1
            confidences.append(_get_latest_group_confidence(br))
        return answers, confidences, ready_count, active_count

    def _maybe_trigger_global_early_stop(trigger_branch: int) -> bool:
        """Stop once one answer bucket has decisive aggregate top-1 confidence."""
        nonlocal global_early_stop_info

        if global_early_stop_event.is_set():
            return True
        if not config.global_early_stop_enabled:
            return False
        if not initial_warmup_complete:
            return False

        answers, confidences, ready_count, active_count = _latest_vote_snapshot(require_complete_swgd=True)
        if active_count <= 0 or ready_count < active_count:
            return False

        answer_confidence_sums = {}
        answer_vote_counts = {}
        answer_branch_ids = {}
        for branch_id, (answer, confidence) in enumerate(zip(answers, confidences)):
            if answer is None:
                continue
            answer_confidence_sums[answer] = answer_confidence_sums.get(answer, 0.0) + confidence
            answer_vote_counts[answer] = answer_vote_counts.get(answer, 0) + 1
            answer_branch_ids.setdefault(answer, []).append(branch_id)

        if not answer_confidence_sums:
            return False

        winning_answer = max(
            answer_confidence_sums,
            key=lambda answer: (answer_confidence_sums[answer], answer_vote_counts[answer]),
        )
        winning_confidence_sum = answer_confidence_sums[winning_answer]
        active_threshold = active_count * config.global_early_stop_confidence_fraction
        if winning_confidence_sum <= active_threshold:
            return False

        global_early_stop_info = {
            "event": "global_early_stop",
            "branch": trigger_branch,
            "elapsed_time_at_stop": time.time() - start_time,
            "confidence_sum": winning_confidence_sum,
            "winning_answer_confidence_sum": winning_confidence_sum,
            "threshold": active_threshold,
            "initial_threshold": global_early_stop_initial_threshold,
            "confidence_fraction": config.global_early_stop_confidence_fraction,
            "ready_branches": ready_count,
            "active_branches": active_count,
            "num_branches": K,
            "pruned_branches": [br["id"] for br in branches if br["pruned"]],
            "winning_answer": winning_answer,
            "winning_answer_branch_ids": list(answer_branch_ids[winning_answer]),
            "answer_confidence_sums": dict(answer_confidence_sums),
            "answer_vote_counts": dict(answer_vote_counts),
            "total_confidence_sum": sum(confidences),
            "final_answer_snapshot": winning_answer,
            "all_final_answers_snapshot": list(answers),
            "all_final_group_confidences_snapshot": list(confidences),
        }
        lifecycle_events.append(dict(global_early_stop_info))
        global_early_stop_event.set()
        logger.info(
            "Global early-stop: answer=%s bucket_sum_top1_prob=%.3f > %.3f over %d active branches",
            winning_answer,
            winning_confidence_sum,
            active_threshold,
            active_count,
        )
        return True

    def _select_donor(exclude_id: int, prune_threshold: float) -> Optional[int]:
        """Pick a non-pruned donor according to the configured fork mode."""
        candidates = []
        for i, br in enumerate(branches):
            if br["pruned"] or i == exclude_id or not _is_swgd_ready(br) or not _fork_fresh_donor_ready(br):
                continue
            swgd = _get_swgd(br)
            entropy = (swgd or {}).get("group_entropy")
            if entropy is not None and entropy < prune_threshold:
                candidates.append(i)
        if not candidates:
            return None
        key = lambda j: (_get_swgd(branches[j]) or {}).get("group_entropy", float("inf"))
        if config.fork_mode == "fork-high":
            return max(candidates, key=key)
        return min(candidates, key=key)

    def _build_probe_debug_series(histories):
        """Build a compact rolling SWGD trace for output debugging."""
        probe_group_entropies = []
        probe_top1_tokens = []
        probe_top1_probs = []
        for idx in range(len(histories)):
            swgd = compute_swgd(histories[: idx + 1], config.window)
            probe_group_entropies.append(swgd.get("group_entropy"))
            probe_top1_tokens.append(swgd.get("top1_token"))
            probe_top1_probs.append(swgd.get("top1_prob"))
        return probe_group_entropies, probe_top1_tokens, probe_top1_probs

    def _record_probe_result(br, bid: int, probe_result: dict, probe_elapsed: float, *, terminal: bool = False):
        """Record one probe result in shared bookkeeping structures."""
        phase_timings["cumulative_probe_request_time"] += probe_elapsed
        br["histories"].append({
            "answer": probe_result["answer"],
            "fte": probe_result["fte"],
            "first_token_logprobs": probe_result["first_token_logprobs"],
            "raw_probe_text": probe_result.get("raw_probe_text", ""),
            "gen_tokens_at_probe": br["gen_tokens"],
        })
        br["probe_tokens"] += probe_result["tokens"]
        br["spent_probe_tokens"] += probe_result["tokens"]
        br["probe_count"] += 1
        if br["last_fork_parent"] is not None:
            br["fresh_probe_count_since_fork"] += 1
        br["swgd_cache"] = None

        if bid not in init_ready_branches:
            init_ready_branches.add(bid)
            if len(init_ready_branches) == K and phase_timings["initialization_time"] == 0.0:
                phase_timings["initialization_time"] = time.time() - start_time

        probe_log.append({
            "branch": bid,
            "probe_count": br["probe_count"],
            "gen_tokens": br["gen_tokens"],
            "answer": probe_result["answer"],
            "fte": probe_result["fte"],
            "terminal": terminal,
        })

    # ========== BRANCH COROUTINE ==========
    async def branch_loop(bid: int):
        nonlocal fork_counter

        br = branches[bid]

        while True:
            if global_early_stop_event.is_set():
                br["stopped_by_global_early_stop"] = True
                break
            if br["pruned"]:
                break
            if br["retired"] or br["naturally_finished"] or br["gen_tokens"] >= config.max_tokens_per_branch:
                if not br["retired"]:
                    br["naturally_finished"] = True
                break

            # --- 1. GENERATE chunk ---
            remaining_budget = config.max_tokens_per_branch - br["gen_tokens"]
            gen_start = time.time()
            gen_result = await _generate_chunk(
                client, model, question_prompt, br["prefix"],
                model_type, config, br["seed"], br["temperature"],
                min(config.probe_interval, remaining_budget),
            )
            gen_elapsed = time.time() - gen_start

            phase_timings["cumulative_generate_request_time"] += gen_elapsed
            br["prefix"] += gen_result["new_text"]
            br["gen_tokens"] += gen_result["tokens"]
            br["spent_gen_tokens"] += gen_result["tokens"]

            if global_early_stop_event.is_set():
                br["stopped_by_global_early_stop"] = True
                break

            if gen_result["finished"] or br["gen_tokens"] >= config.max_tokens_per_branch:
                terminal_probe_result = None
                terminal_probe_elapsed = 0.0
                if br["prefix"]:
                    probe_start = time.time()
                    terminal_probe_result = await _probe_branch_with_logprobs(
                        client, model, question_prompt, br["prefix"],
                        model_type, dataset_type, config,
                        seed=br["seed"] + br["probe_count"],
                    )
                    terminal_probe_elapsed = time.time() - probe_start
                br["naturally_finished"] = True
                async with lock:
                    # NOTE: Refresh the branch's final vote on the terminal prefix so
                    # majority voting does not reuse a stale pre-terminal probe answer.
                    if terminal_probe_result is not None:
                        _record_probe_result(
                            br, bid, terminal_probe_result, terminal_probe_elapsed, terminal=True
                        )
                        _record_warmup_swgd_sample(br)
                        _maybe_finalize_dynamic_prune_threshold()
                        if not (
                            br["last_fork_parent"] is None
                            and br["probe_count"] <= _control_start_probe()
                        ):
                            _maybe_trigger_global_early_stop(bid)
                    elif bid not in init_ready_branches:
                        init_ready_branches.add(bid)
                        if len(init_ready_branches) == K and phase_timings["initialization_time"] == 0.0:
                            phase_timings["initialization_time"] = time.time() - start_time
                    lifecycle_events.append({
                        "event": "naturally_finished",
                        "branch": bid,
                        "gen_tokens": br["gen_tokens"],
                        "reason": "eos" if gen_result["finished"] else "budget",
                    })
                break

            # --- 2. PROBE with logprobs ---
            probe_start = time.time()
            probe_result = await _probe_branch_with_logprobs(
                client, model, question_prompt, br["prefix"],
                model_type, dataset_type, config,
                seed=br["seed"] + br["probe_count"],
            )
            probe_elapsed = time.time() - probe_start

            # --- 3. UPDATE shared state (under lock) ---
            async with lock:
                _record_probe_result(br, bid, probe_result, probe_elapsed)
                _record_warmup_swgd_sample(br)
                _maybe_finalize_dynamic_prune_threshold()

                # --- Natural no-intervention phase: collect SWGD samples only. ---
                if not initial_warmup_complete:
                    continue
                if br["last_fork_parent"] is None and br["probe_count"] <= _control_start_probe():
                    continue

                # --- Cold-start guard: no intervention before a full W-sized window. ---
                if not _is_swgd_ready(br):
                    continue

                swgd = compute_swgd(br["histories"], config.window)
                br["swgd_cache"] = swgd

                if _maybe_trigger_global_early_stop(bid):
                    br["stopped_by_global_early_stop"] = True
                    break

                # ── Prune check ──
                active_prune_threshold = dynamic_prune_threshold
                if (
                    config.enable_pruning
                    and active_prune_threshold is not None
                    and _fork_fresh_prune_ready(br)
                    and swgd["group_entropy"] > active_prune_threshold
                ):
                    if config.fork_mode == "no-fork":
                        branch_last_hist = br["histories"][-1] if br["histories"] else {}
                        br["pruned"] = True
                        br["retired"] = False
                        br["naturally_finished"] = False
                        lifecycle_events.append({
                            "event": "pruned_no_fork",
                            "branch": bid,
                            "branch_group_entropy": swgd["group_entropy"],
                            "prune_threshold": active_prune_threshold,
                            "branch_gen_tokens": br["gen_tokens"],
                            "branch_probe_count": br["probe_count"],
                            "branch_last_answer": branch_last_hist.get("answer"),
                            "branch_prefix_tail": _text_tail(br["prefix"]),
                            "fork_mode": config.fork_mode,
                        })
                        logger.debug(
                            "[B%d] PRUNED without fork (H_group=%.3f > %.4f)",
                            bid, swgd["group_entropy"], active_prune_threshold,
                        )
                        break

                    donor_id = _select_donor(bid, active_prune_threshold)
                    if donor_id is not None:
                        donor = branches[donor_id]
                        donor_swgd = _get_swgd(donor) or {}
                        donor_entropy = donor_swgd.get("group_entropy")
                        if donor_entropy is not None and donor_entropy < active_prune_threshold:
                            fork_counter += 1
                            new_seed = base_seed + K + fork_counter
                            new_temperature = donor["temperature"] + config.fork_temp_increment
                            inherited_prefix_chars = len(donor["prefix"])
                            branch_last_hist = br["histories"][-1] if br["histories"] else {}
                            donor_last_hist = donor["histories"][-1] if donor["histories"] else {}

                            lifecycle_events.append({
                                "event": "prune_fork",
                                "branch": bid,
                                "branch_group_entropy": swgd["group_entropy"],
                                "branch_gen_tokens": br["gen_tokens"],
                                "branch_probe_count": br["probe_count"],
                                "branch_last_answer": branch_last_hist.get("answer"),
                                "branch_prefix_tail": _text_tail(br["prefix"]),
                                "donor": donor_id,
                                "donor_group_entropy": donor_entropy,
                                "donor_gen_tokens": donor["gen_tokens"],
                                "donor_probe_count": donor["probe_count"],
                                "donor_last_answer": donor_last_hist.get("answer"),
                                "donor_prefix_tail": _text_tail(donor["prefix"]),
                                "fork_mode": config.fork_mode,
                                "new_seed": new_seed,
                                "new_temperature": new_temperature,
                            })
                            logger.debug(
                                "[B%d] PRUNED (H_group=%.3f > %.4f) → fork from B%d (H=%.3f)",
                                bid, swgd["group_entropy"], active_prune_threshold,
                                donor_id, donor_entropy,
                            )

                            donor["fork_count"] += 1
                            br["prefix"] = donor["prefix"]
                            br["seed"] = new_seed
                            br["temperature"] = new_temperature
                            br["gen_tokens"] = donor["gen_tokens"]
                            br["probe_tokens"] = 0
                            br["naturally_finished"] = False
                            br["retired"] = False
                            br["pruned"] = False
                            inherited_count = max(0, config.window - 1)
                            inherited_histories = donor["histories"][-inherited_count:] if inherited_count > 0 else []
                            br["histories"] = [hist.copy() for hist in inherited_histories]
                            br["probe_count"] = len(br["histories"])
                            br["born_at_step"] = br["probe_count"]
                            br["fresh_probe_count_since_fork"] = 0
                            br["last_fork_parent"] = donor_id
                            br["last_fork_inherited_gen_tokens"] = donor["gen_tokens"]
                            br["last_fork_prefix_chars"] = inherited_prefix_chars
                            br["swgd_cache"] = None
                        else:
                            logger.debug(
                                "[B%d] prune skipped: donor=%s donor_H=%s",
                                bid,
                                donor_id,
                                f"{donor_entropy:.3f}" if donor_entropy is not None else "None",
                            )
                    else:
                        logger.debug(
                            "[B%d] prune skipped: no eligible donor",
                            bid,
                        )
                    continue

                # ── Retire check ──
                retire_top1_probs = _recent_retire_top1_probs(br) if _fork_fresh_retire_ready(br) else None
                if retire_top1_probs is not None:
                    br["retired"] = True
                    lifecycle_events.append({
                        "event": "retired",
                        "branch": bid,
                        "gen_tokens": br["gen_tokens"],
                        "retire_stability_windows": config.retire_stability_windows,
                        "theta_retire": config.theta_retire,
                        "retire_top1_probs": retire_top1_probs,
                        "top1_token": swgd["top1_token"],
                        "top1_prob": swgd["top1_prob"],
                        "group_entropy": swgd["group_entropy"],
                    })
                    logger.debug(
                        "[B%d] RETIRED (latest X=%d top1_probs all > %.3f, prob=%.3f H=%.3f)",
                        bid, config.retire_stability_windows, config.theta_retire,
                        swgd["top1_prob"], swgd["group_entropy"],
                    )
                    break

    # ========== LAUNCH ==========
    tasks = [asyncio.create_task(branch_loop(i)) for i in range(K)]
    control_start_probe = _control_start_probe()
    logger.info(
        "Launching ParaTempo: branches=%d window=%d num_warmup=%d prune_enabled=%s retire_enabled=%s retire_stability_windows=%d theta_retire=%.3f control_start_probe=%d fork_fresh_prune=%d fork_fresh_donor=%d fork_fresh_retire=%s fork_mode=%s dynamic_prune_percentile=%.2f global_es=%s global_es_threshold=%.3f",
        K, config.window, config.num_warmup,
        config.enable_pruning, _retire_enabled(), config.retire_stability_windows,
        config.theta_retire, control_start_probe,
        config.window, config.window,
        _fork_fresh_retire_requirement() if _fork_fresh_retire_requirement() is not None else "disabled",
        config.fork_mode,
        config.dynamic_prune_percentile,
        config.global_early_stop_enabled,
        global_early_stop_initial_threshold,
    )
    await asyncio.gather(*tasks, return_exceptions=True)

    # ========== FINAL ANSWER ==========
    elapsed = time.time() - start_time
    logger.info(
        "ParaTempo timings: total=%.2fs init=%.2fs cum_gen=%.2fs cum_probe=%.2fs",
        elapsed, phase_timings["initialization_time"],
        phase_timings["cumulative_generate_request_time"],
        phase_timings["cumulative_probe_request_time"],
    )

    # Weighted majority vote over all branches' last extracted answers.  If
    # global early-stop fired, use the exact vote snapshot from the stop moment.
    if global_early_stop_info is not None:
        all_answers = list(global_early_stop_info["all_final_answers_snapshot"])
        all_group_confidences = list(global_early_stop_info["all_final_group_confidences_snapshot"])
        final_answer = global_early_stop_info["final_answer_snapshot"]
    else:
        all_answers, all_group_confidences, _, active_branches_final = _latest_vote_snapshot()
        final_answer = weighted_majority_vote(all_answers, all_group_confidences)

    # ========== TOKEN STATISTICS ==========
    total_gen_tokens = sum(br["spent_gen_tokens"] for br in branches)
    total_probe_tokens = sum(br["spent_probe_tokens"] for br in branches)
    total_tokens = total_gen_tokens  # align with PP: exclude probe tokens
    sequential_tokens = max((br["gen_tokens"] for br in branches), default=0)
    num_forks = fork_counter
    num_retired = sum(1 for br in branches if br["retired"])
    num_pruned = sum(1 for br in branches if br["pruned"])
    if global_early_stop_info is not None:
        active_branches_final = int(global_early_stop_info.get("active_branches", K - num_pruned))
    else:
        active_branches_final = K - num_pruned

    per_branch = []
    for br in branches:
        last_hist = br["histories"][-1] if br["histories"] else {"answer": None, "fte": 0.0}
        swgd = _get_swgd(br) or {}
        probe_group_entropies, probe_top1_tokens, probe_top1_probs = _build_probe_debug_series(br["histories"])
        if br["last_fork_parent"] is not None:
            post_fork_text = br["prefix"][br["last_fork_prefix_chars"]:]
        else:
            post_fork_text = ""
        per_branch.append({
            "id": br["id"],
            "gen_tokens": br["gen_tokens"],
            "probe_tokens": br["probe_tokens"],
            "spent_gen_tokens": br["spent_gen_tokens"],
            "spent_probe_tokens": br["spent_probe_tokens"],
            "naturally_finished": br["naturally_finished"],
            "retired": br["retired"],
            "pruned": br["pruned"],
            "final_answer": last_hist["answer"],
            "voting_answer": None if br["pruned"] else last_hist["answer"],
            "final_fte": last_hist.get("fte", 0.0),
            "final_group_entropy": swgd.get("group_entropy"),
            "final_top1_prob": swgd.get("top1_prob"),
            "num_probes": br["probe_count"],
            "fresh_probe_count_since_fork": br["fresh_probe_count_since_fork"],
            "fork_fresh_prune_ready": _fork_fresh_prune_ready(br),
            "fork_fresh_donor_ready": _fork_fresh_donor_ready(br),
            "fork_fresh_retire_ready": _fork_fresh_retire_ready(br),
            "fork_count": br["fork_count"],
            "born_at_step": br["born_at_step"],
            "last_fork_parent": br["last_fork_parent"],
            "last_fork_inherited_gen_tokens": br["last_fork_inherited_gen_tokens"],
            "post_fork_text_chars": len(post_fork_text),
            "post_fork_text_tail": _text_tail(post_fork_text),
            "stopped_by_global_early_stop": br["stopped_by_global_early_stop"],
            "full_text": br["prefix"],
            "probe_answers": [h["answer"] for h in br["histories"]],
            "probe_ftes": [h["fte"] for h in br["histories"]],
            "probe_gen_tokens": [h.get("gen_tokens_at_probe", 0) for h in br["histories"]],
            # Persist compact probe traces for post-run inspection.
            "probe_raw": [h.get("raw_probe_text", "") for h in br["histories"]],
            "probe_first_token_logprobs": [
                dict(h.get("first_token_logprobs", {})) for h in br["histories"]
            ],
            "probe_group_entropies": probe_group_entropies,
            "probe_top1_tokens": probe_top1_tokens,
            "probe_top1_probs": probe_top1_probs,
        })

    return {
        "final_answer": final_answer,
        "method": "paratempo",
        "num_branches": K,
        "early_stopped": global_early_stop_info is not None,
        "early_stop_type": "global_answer_bucket_top1_prob_sum" if global_early_stop_info is not None else None,
        "global_early_stop": global_early_stop_info,
        # Token stats
        "total_tokens": total_tokens,
        "total_gen_tokens": total_gen_tokens,
        "total_probe_tokens": total_probe_tokens,
        "sequential_tokens": sequential_tokens,
        # Lifecycle stats
        "num_forks": num_forks,
        "num_retired": num_retired,
        "num_pruned": num_pruned,
        "active_branches_final": active_branches_final,
        "prune_enabled": config.enable_pruning,
        "retire_enabled": _retire_enabled(),
        "fork_enabled": _fork_enabled(),
        "lifecycle_events": lifecycle_events,
        "probe_log": probe_log,
        # Timing
        "elapsed_time": elapsed,
        "phase_timings": phase_timings,
        "initialization_time": phase_timings["initialization_time"],
        # Config snapshot
        "config": {
            "num_branches": config.num_branches,
            "probe_interval": config.probe_interval,
            "max_tokens_per_branch": config.max_tokens_per_branch,
            "window": config.window,
            "num_warmup": config.num_warmup,
            "control_start_probe": control_start_probe,
            "prune_enabled": config.enable_pruning,
            "fork_fresh_prune_probe_requirement": config.window,
            "fork_fresh_donor_probe_requirement": config.window,
            "fork_fresh_retire_probe_requirement": _fork_fresh_retire_requirement(),
            "dynamic_prune_percentile": config.dynamic_prune_percentile,
            "retire_enabled": _retire_enabled(),
            "retire_stability_windows": config.retire_stability_windows,
            "theta_retire": config.theta_retire,
            "temperature": config.temperature,
            "top_p": config.top_p,
            "fork_temp_increment": config.fork_temp_increment,
            "fork_mode": config.fork_mode,
            "fork_enabled": _fork_enabled(),
            "global_early_stop_enabled": config.global_early_stop_enabled,
            "global_early_stop_confidence_fraction": config.global_early_stop_confidence_fraction,
            "global_early_stop_initial_threshold": global_early_stop_initial_threshold,
            "global_early_stop_threshold_scope": "active_branches",
            "probe_max_tokens": config.probe_max_tokens,
            "probe_top_logprobs": config.probe_top_logprobs,
        },
        # Details
        "all_final_answers": all_answers,
        "all_final_group_confidences": all_group_confidences,
        "final_vote_method": "weighted_majority_by_top1_prob",
        "dynamic_prune_threshold": dynamic_prune_threshold,
        "warmup_group_entropy_samples": warmup_group_entropies,
        "per_branch_details": per_branch,
    }
