"""Shared utilities for ParaTempo."""
import logging
import math
import os
import re
import time
from collections import Counter, defaultdict
from typing import Any, Optional, List, Dict

from .gpt_oss_completion import (
    create_model_response,
    extract_final_response_text,
    extract_reasoning_response_text,
    first_token_top_logprobs,
)


logger = logging.getLogger(__name__)

_BACKEND_WARMUP_PROMPT = (
    "Warmup request only. Think very briefly and end with \\boxed{0}."
)


# ============= MODEL REGISTRY =============

DEFAULT_PORTS = {
    "Qwen/Qwen3.5-35B-A3B": "PORT_XXX",
    "openai/gpt-oss-20b": "PORT_XXX",
}

SUPPORTED_MODELS = set(DEFAULT_PORTS)
GPT_OSS_MODELS = {"openai/gpt-oss-20b"}


def is_gpt_oss(model: str) -> bool:
    return model in GPT_OSS_MODELS


def reasoning_stop_tokens(model_type: str) -> list[str]:
    if model_type == "gpt":
        return ["<|end|>"]
    return ["</think>"]


def get_port(model: str, override_port: Optional[int] = None) -> str:
    if override_port is not None:
        return str(override_port)
    env_key = f"MODEL_PORT_{model.split('/')[-1].upper().replace('.', '_').replace('-', '_')}"
    port = os.environ.get(env_key, DEFAULT_PORTS.get(model, "PORT_XXX"))
    if port == "PORT_XXX":
        raise ValueError(
            f"Set --port or environment variable {env_key}; PORT_XXX is a placeholder."
        )
    return port


def create_client(
    model: str,
    port: Optional[int] = None,
    api_key: Optional[str] = None,
) -> Any:
    import httpx
    from openai import AsyncOpenAI

    if model not in SUPPORTED_MODELS:
        raise ValueError(f"Unsupported model: {model}")
    actual_port = get_port(model, port)
    actual_api_key = api_key or os.environ.get("VLLM_API_KEY", "API_KEY_XXX")
    return AsyncOpenAI(
        api_key=actual_api_key,
        base_url=f"http://localhost:{actual_port}/v1",
        http_client=httpx.AsyncClient(trust_env=False),
    )


async def warmup_backend(
    client,
    model: str,
    question_prompt: str,
    model_type: str,
    *,
    seed: int = 0,
    include_probe: bool = False,
    probe_top_logprobs: Optional[int] = None,
) -> dict:
    """Issue tiny untimed requests to warm a cold backend before the real run."""
    timings = {
        "generation_warmup_time": 0.0,
        "probe_warmup_time": 0.0,
    }

    warmup_prompt = _BACKEND_WARMUP_PROMPT
    messages, extra_body = build_chunk_messages(warmup_prompt, "", model_type)
    gen_start = time.time()
    await create_model_response(
        client,
        model,
        messages,
        extra_body,
        model_type,
        temperature=0.0,
        top_p=1.0,
        max_tokens=1,
        stop=reasoning_stop_tokens(model_type),
        seed=seed,
    )
    timings["generation_warmup_time"] = time.time() - gen_start

    if include_probe:
        messages, extra_body = build_probe_messages(warmup_prompt, "", model_type)
        probe_kwargs = dict(
            model=model,
            messages=messages,
            temperature=0.0,
            top_p=1.0,
            max_tokens=1,
            seed=seed + 1,
            extra_body=extra_body,
        )
        if probe_top_logprobs is not None:
            probe_kwargs["logprobs"] = True
            probe_kwargs["top_logprobs"] = probe_top_logprobs
        probe_start = time.time()
        await create_model_response(
            client,
            model,
            messages,
            extra_body,
            model_type,
            temperature=probe_kwargs["temperature"],
            top_p=probe_kwargs["top_p"],
            max_tokens=probe_kwargs["max_tokens"],
            seed=probe_kwargs["seed"],
            logprobs=probe_kwargs.get("logprobs"),
            top_logprobs=probe_kwargs.get("top_logprobs"),
        )
        timings["probe_warmup_time"] = time.time() - probe_start

    timings["total_warmup_time"] = (
        timings["generation_warmup_time"] + timings["probe_warmup_time"]
    )
    logger.info(
        "Backend warmup complete: gen=%.2fs probe=%.2fs total=%.2fs",
        timings["generation_warmup_time"],
        timings["probe_warmup_time"],
        timings["total_warmup_time"],
    )
    return timings


# ============= PROMPT FORMATTING =============

def format_prompt(problem: dict, dataset_type: str) -> str:
    """Format prompt — aligned with DeepConf baseline for fair comparison."""
    if dataset_type == "math":
        return (
            "Please reason step by step, and put your final answer within \\boxed{}.\n\n"
            f"{problem['problem']}"
        )
    elif dataset_type == "gpqa":
        return (
            "What is the correct answer to this question? "
            "Please reason step by step, and put your final answer "
            "strictly in the format \\boxed{X}, where X is a single letter (A, B, C, or D).\n\n"
            f"{problem['problem']}"
        )
    else:
        raise ValueError(f"Unknown dataset_type: {dataset_type}")


# ============= MESSAGE BUILDING =============

def _clean_gpt_oss_prefix(prefix: str) -> str:
    if prefix is None:
        return ""
    text = prefix
    for marker in (
        "<|start|>assistant<|channel|>final<|message|>",
        "<|channel|>final<|message|>",
    ):
        if marker in text:
            text = text.split(marker, 1)[0]
    for marker in (
        "<think>\n",
        "<think>",
        "</think>",
        "<|start|>assistant<|channel|>analysis<|message|>",
        "<|channel|>analysis<|message|>",
        "<|end|>",
        "<|return|>",
    ):
        text = text.replace(marker, "")
    return text


def build_chunk_messages(question_prompt: str, prefix: str, model_type: str):
    """Build messages for chunk generation. Returns (messages, extra_body)."""
    user_msg = {"role": "user", "content": question_prompt}
    if model_type == "gpt":
        prefix = _clean_gpt_oss_prefix(prefix)
        if not prefix:
            return [user_msg], {"add_generation_prompt": True}
        return [
            user_msg,
            {"role": "assistant", "content": "", "reasoning_content": prefix},
        ], {"add_generation_prompt": False}

    if "</think>" in prefix:
        prefix = prefix.split("</think>")[0]
    think_content = f"<think>\n{prefix}" if prefix else "<think>\n"
    return [
        user_msg,
        {"role": "assistant", "content": think_content},
    ], {"add_generation_prompt": False, "continue_final_message": True}


def build_probe_messages(question_prompt: str, prefix: str, model_type: str):
    """Build messages for probing (force intermediate answer). Returns (messages, extra_body)."""
    user_msg = {"role": "user", "content": question_prompt}
    if model_type == "gpt":
        return [
            user_msg,
            {
                "role": "assistant",
                "content": "The final answer is \\boxed{",
                "reasoning_content": _clean_gpt_oss_prefix(prefix),
            },
        ], {"add_generation_prompt": False, "continue_final_message": True}

    if "</think>" in prefix:
        prefix = prefix.split("</think>")[0]
    probe_content = f"<think>\n{prefix}\n</think>\n\nThe final answer is: \\boxed{{"
    return [
        user_msg,
        {"role": "assistant", "content": probe_content},
    ], {"add_generation_prompt": False, "continue_final_message": True}


# ============= RESPONSE / ANSWER EXTRACTION =============

def extract_text_from_response(response, model_type: str) -> str:
    if model_type == "gpt":
        choice = response.choices[0]
        if getattr(choice, "text", None):
            return choice.text or ""
        if getattr(choice, "message", None) is None:
            return ""
        return extract_reasoning_response_text(response)
    return response.choices[0].message.content or ""


def extract_final_text_from_response(response, model_type: str) -> str:
    if model_type == "gpt":
        choice = response.choices[0]
        if getattr(choice, "text", None):
            return choice.text or ""
        if getattr(choice, "message", None) is None:
            return ""
        return extract_final_response_text(response)
    return response.choices[0].message.content or ""


def _extract_boxed_raw(text: str) -> Optional[str]:
    """Extract raw content from last \\boxed{...} using brace matching."""
    if "boxed" not in text:
        return None
    ans = text.split("boxed")[-1]
    if not ans:
        return None
    if ans[0] == "{":
        stack = 1
        a = ""
        for c in ans[1:]:
            if c == "{":
                stack += 1
                a += c
            elif c == "}":
                stack -= 1
                if stack == 0:
                    break
                a += c
            else:
                a += c
        return a.strip() if a.strip() else None
    else:
        result = ans.split("$")[0].strip()
        return result if result else None


def extract_boxed_answer(text: str, dataset_type: str = "math") -> Optional[str]:
    if text is None:
        return None
    if dataset_type == "gpqa":
        raw = _extract_boxed_raw(text)
        if raw and len(raw.strip()) == 1 and raw.strip().upper() in "ABCD":
            return raw.strip().upper()
        for pattern in [r'(?:answer|choice)\s*(?:is|:)\s*\(?([A-D])\)?', r'\b([A-D])\b\s*$']:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                return match.group(1).upper()
        return None
    raw = _extract_boxed_raw(text)
    if raw is not None:
        try:
            from math_verify import parse
            parsed = parse(raw)
            if parsed:
                return str(parsed[0])
        except Exception:
            pass
        return raw
    return None


def extract_answer_from_full_response(text: str, model_type: str, dataset_type: str = "math") -> Optional[str]:
    if model_type != "gpt" and "</think>" in text:
        answer_text = text.split("</think>", 1)[1].strip()
    else:
        answer_text = text
    answer = extract_boxed_answer(answer_text, dataset_type)
    if answer is None:
        answer = extract_boxed_answer(text, dataset_type)
    return answer


# ============= MAJORITY VOTING =============

def majority_vote(answers: List[Optional[str]]) -> Optional[str]:
    valid = [a for a in answers if a is not None]
    if not valid:
        return None
    return Counter(valid).most_common(1)[0][0]


def weighted_majority_vote(
    answers: List[Optional[str]],
    weights: List[float],
) -> Optional[str]:
    valid_answers = []
    for idx, answer in enumerate(answers):
        if answer is None:
            continue
        weight = weights[idx] if idx < len(weights) else 0.0
        try:
            weight = float(weight)
        except (TypeError, ValueError):
            weight = 0.0
        valid_answers.append((answer, max(0.0, weight)))

    if not valid_answers:
        return None

    weight_sums: Dict[str, float] = defaultdict(float)
    vote_counts: Counter = Counter()
    for answer, weight in valid_answers:
        weight_sums[answer] += weight
        vote_counts[answer] += 1

    if any(weight > 0 for weight in weight_sums.values()):
        return max(weight_sums, key=lambda ans: (weight_sums[ans], vote_counts[ans]))
    return majority_vote([answer for answer, _ in valid_answers])


# ============= SWGD (Sliding-Window Group Distribution) =============

def compute_swgd(
    histories: list,
    window: int,
) -> dict:
    """Compute Sliding-Window Group Distribution from recent probe logprobs.

    Args:
        histories: list of probe dicts, each containing "first_token_logprobs"
        window: number of recent probes to aggregate

    Returns:
        dict with keys: group_entropy, top1_token, top1_prob,
                        group_probs (full distribution)
    """
    recent = histories[-window:]
    if not recent:
        return {
            "group_entropy": float("inf"),
            "top1_token": None,
            "top1_prob": 0.0,
            "group_probs": {},
        }

    # Step 1+2: per-probe softmax → aggregate by token
    raw_weights: Dict[str, float] = defaultdict(float)

    for probe in recent:
        logprobs = probe.get("first_token_logprobs")
        if not logprobs:
            continue
        # Per-probe softmax normalization
        probs = {tok: math.exp(lp) for tok, lp in logprobs.items()}
        total = sum(probs.values())
        if total <= 0:
            continue
        probs = {tok: p / total for tok, p in probs.items()}
        for tok, p in probs.items():
            raw_weights[tok] += p

    # Normalize across probes
    grand_total = sum(raw_weights.values())
    if grand_total <= 0:
        return {
            "group_entropy": float("inf"),
            "top1_token": None,
            "top1_prob": 0.0,
            "group_probs": {},
        }
    group_probs = {tok: w / grand_total for tok, w in raw_weights.items()}

    # Group entropy
    group_entropy = -sum(p * math.log(p) for p in group_probs.values() if p > 0)

    # Top-1
    top1_token = max(group_probs, key=group_probs.get)
    top1_prob = group_probs[top1_token]

    return {
        "group_entropy": group_entropy,
        "top1_token": top1_token,
        "top1_prob": top1_prob,
        "group_probs": group_probs,
    }


# ============= ANSWER CHECKING =============

def check_answer(predicted: str, gold: str, dataset_type: str) -> bool:
    if predicted is None:
        return False
    predicted = predicted.strip()
    gold = gold.strip()
    while '\\text{' in predicted:
        start = predicted.find('\\text{')
        end = predicted.find('}', start)
        if end == -1:
            break
        predicted = predicted[:start] + predicted[start + 6:end] + predicted[end + 1:]
    if dataset_type == "gpqa":
        return predicted.upper() == gold.upper()
    try:
        from math_verify import parse, verify
        parsed_gold = parse(f"\\boxed{{{gold}}}") if "boxed" not in gold else parse(gold)
        parsed_answer = parse(predicted)
        if parsed_answer and verify(gold=parsed_gold, target=parsed_answer):
            return True
        boxed_answer = parse(f"\\boxed{{{predicted}}}") if "boxed" not in predicted else parsed_answer
        return bool(boxed_answer) and verify(gold=parsed_gold, target=boxed_answer)
    except Exception:
        return str(predicted).strip() == str(gold).strip()


# ============= BRANCH TERMINATION =============

def is_branch_finished(new_text: str, model_type: str) -> bool:
    if model_type == "gpt":
        if new_text is None:
            return False
        return any(marker in new_text for marker in ("<|end|>", "<|return|>", "\boxed", "Answer:", "ANSWER:"))
    return new_text is not None and "</think>" in new_text
