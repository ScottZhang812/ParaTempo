"""GPT-OSS prompt helpers for local vLLM-compatible backends.

Machine-specific tokenizer and template paths are intentionally represented by
environment variables:

- `GPT_OSS_TOKENIZER_PATH`
- `GPT_OSS_TEMPLATE_PATH`

If these are not set, a compact fallback renderer is used for the message
shapes required by the benchmark entrypoint.
"""

from __future__ import annotations

import os
from datetime import datetime
from functools import lru_cache
from pathlib import Path
from typing import Any


GPT_OSS_PROTOCOL_TOKENS = {
    "<|start|>",
    "<|end|>",
    "<|channel|>",
    "analysis",
    "commentary",
    "final",
    "<|message|>",
    "<|return|>",
    "<|call|>",
    "<|constrain|>",
}


@lru_cache(maxsize=1)
def _gpt_oss_tokenizer() -> Any:
    from transformers import AutoTokenizer

    tokenizer_path = os.environ.get("GPT_OSS_TOKENIZER_PATH", "TOKENIZER_PATH_XXX")
    template_path = os.environ.get("GPT_OSS_TEMPLATE_PATH", "TEMPLATE_PATH_XXX")
    if tokenizer_path == "TOKENIZER_PATH_XXX" or template_path == "TEMPLATE_PATH_XXX":
        raise ValueError(
            "Set GPT_OSS_TOKENIZER_PATH and GPT_OSS_TEMPLATE_PATH, or rely on "
            "render_gpt_oss_prompt_fallback without tokenizer rendering."
        )

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=True)
    tokenizer.chat_template = Path(template_path).read_text(encoding="utf-8")
    return tokenizer


def render_gpt_oss_prompt(
    messages: list[dict[str, Any]],
    extra_body: dict[str, Any] | None = None,
) -> str:
    kwargs = dict(extra_body or {})
    try:
        return _gpt_oss_tokenizer().apply_chat_template(
            messages,
            tokenize=False,
            **kwargs,
        )
    except Exception:
        return render_gpt_oss_prompt_fallback(messages, kwargs)


def _message_attr(message: Any, field_name: str) -> Any:
    if isinstance(message, dict):
        return message.get(field_name)
    return getattr(message, field_name, None)


def normalize_message_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, list):
        parts = []
        for item in value:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict):
                text = item.get("text")
                if text is None and item.get("type") == "text":
                    text = item.get("content")
                if text is not None:
                    parts.append(str(text))
        return "".join(parts)
    return str(value)


def render_gpt_oss_prompt_fallback(
    messages: list[dict[str, Any]],
    extra_body: dict[str, Any] | None = None,
) -> str:
    """Render the GPT-OSS prompt shape used by this benchmark."""

    kwargs = dict(extra_body or {})
    model_identity = kwargs.get(
        "model_identity",
        "You are a large language model.",
    )
    reasoning_effort = kwargs.get("reasoning_effort", "medium")
    rendered = [
        "<|start|>system<|message|>",
        model_identity + "\n",
        "Knowledge cutoff: 2024-06\n",
        "Current date: " + datetime.now().strftime("%Y-%m-%d") + "\n\n",
        "Reasoning: " + str(reasoning_effort) + "\n\n",
        "# Valid channels: analysis, commentary, final. "
        "Channel must be included for every message.",
        "<|end|>",
    ]

    for message in messages:
        role = _message_attr(message, "role")
        content = normalize_message_text(_message_attr(message, "content"))
        reasoning = (
            normalize_message_text(_message_attr(message, "reasoning_content"))
            or normalize_message_text(_message_attr(message, "reasoning"))
            or normalize_message_text(_message_attr(message, "thinking"))
        )
        if role in {"developer", "system"}:
            rendered.append(
                "<|start|>developer<|message|># Instructions\n\n"
                + content
                + "<|end|>"
            )
        elif role == "user":
            rendered.append("<|start|>user<|message|>" + content + "<|end|>")
        elif role == "assistant":
            if reasoning:
                rendered.append(
                    "<|start|>assistant<|channel|>analysis<|message|>"
                    + reasoning
                )
                if content:
                    rendered.append(
                        "<|end|><|start|>assistant<|channel|>final<|message|>"
                        + content
                    )
            else:
                rendered.append(
                    "<|start|>assistant<|channel|>final<|message|>" + content
                )

    if kwargs.get("add_generation_prompt"):
        rendered.append("<|start|>assistant<|channel|>analysis<|message|>")
    return "".join(rendered)


def get_message_field(message: Any, field_name: str) -> str:
    if isinstance(message, dict):
        return normalize_message_text(message.get(field_name))
    return normalize_message_text(getattr(message, field_name, None))


def _choice_text(choice: Any) -> str:
    return normalize_message_text(getattr(choice, "text", None))


def extract_reasoning_response_text(response: Any) -> str:
    choice = response.choices[0]
    completion_text = _choice_text(choice)
    if completion_text:
        return completion_text

    msg = getattr(choice, "message", None)
    if msg is None:
        return ""
    reasoning_text = (
        get_message_field(msg, "reasoning_content")
        or get_message_field(msg, "reasoning")
        or get_message_field(msg, "thinking")
    )
    if reasoning_text:
        return reasoning_text
    return get_message_field(msg, "content")


def extract_final_response_text(response: Any) -> str:
    choice = response.choices[0]
    completion_text = _choice_text(choice)
    if completion_text:
        return completion_text

    msg = getattr(choice, "message", None)
    if msg is None:
        return ""
    content_text = get_message_field(msg, "content")
    if content_text:
        return content_text
    return (
        get_message_field(msg, "reasoning_content")
        or get_message_field(msg, "reasoning")
        or get_message_field(msg, "thinking")
    )


def _completion_kwargs(
    model: str,
    messages: list[dict[str, Any]],
    extra_body: dict[str, Any] | None,
    kwargs: dict[str, Any],
) -> dict[str, Any]:
    completion_kwargs = {
        "model": model,
        "prompt": render_gpt_oss_prompt(messages, extra_body),
    }
    for key in ("temperature", "top_p", "max_tokens", "stop", "seed"):
        value = kwargs.get(key)
        if value is not None:
            completion_kwargs[key] = value

    top_logprobs = kwargs.get("top_logprobs")
    logprobs = kwargs.get("logprobs")
    if top_logprobs is not None:
        completion_kwargs["logprobs"] = int(top_logprobs)
    elif isinstance(logprobs, int):
        completion_kwargs["logprobs"] = logprobs
    elif logprobs:
        completion_kwargs["logprobs"] = 20
    return completion_kwargs


async def create_model_response(
    client: Any,
    model: str,
    messages: list[dict[str, Any]],
    extra_body: dict[str, Any] | None,
    model_type: str,
    **kwargs: Any,
) -> Any:
    if model_type == "gpt":
        return await client.completions.create(
            **_completion_kwargs(model, messages, extra_body, kwargs)
        )
    return await client.chat.completions.create(
        model=model,
        messages=messages,
        extra_body=extra_body,
        **kwargs,
    )


def create_model_response_sync(
    client: Any,
    model: str,
    messages: list[dict[str, Any]],
    extra_body: dict[str, Any] | None,
    model_type: str,
    **kwargs: Any,
) -> Any:
    if model_type == "gpt":
        return client.completions.create(
            **_completion_kwargs(model, messages, extra_body, kwargs)
        )
    return client.chat.completions.create(
        model=model,
        messages=messages,
        extra_body=extra_body,
        **kwargs,
    )


def completion_tokens(response: Any) -> int:
    return response.usage.completion_tokens if getattr(response, "usage", None) else 0


def finish_reason(response: Any) -> str:
    return response.choices[0].finish_reason or ""


def _is_protocol_token(token: Any) -> bool:
    return str(token) in GPT_OSS_PROTOCOL_TOKENS


def _top_logprob_mapping(raw_top_logprobs: Any) -> dict[str, float]:
    if not raw_top_logprobs:
        return {}
    if isinstance(raw_top_logprobs, dict):
        return {
            str(token): float(logprob)
            for token, logprob in raw_top_logprobs.items()
            if logprob is not None
        }
    result = {}
    for entry in raw_top_logprobs:
        token = getattr(entry, "token", None)
        logprob = getattr(entry, "logprob", None)
        if token is not None and logprob is not None:
            result[str(token)] = float(logprob)
    return result


def first_token_top_logprobs(response: Any) -> dict[str, float]:
    logprobs = getattr(response.choices[0], "logprobs", None)
    if not logprobs:
        return {}

    top_logprobs = getattr(logprobs, "top_logprobs", None)
    if top_logprobs:
        tokens = list(getattr(logprobs, "tokens", None) or [])
        fallback = _top_logprob_mapping(top_logprobs[0])
        for index, raw_top_logprobs in enumerate(top_logprobs):
            token = tokens[index] if index < len(tokens) else None
            if _is_protocol_token(token):
                continue
            selected = _top_logprob_mapping(raw_top_logprobs)
            if selected:
                return selected
        return fallback

    content = getattr(logprobs, "content", None)
    if content:
        fallback = _top_logprob_mapping(getattr(content[0], "top_logprobs", None))
        for entry in content:
            if _is_protocol_token(getattr(entry, "token", None)):
                continue
            selected = _top_logprob_mapping(getattr(entry, "top_logprobs", None))
            if selected:
                return selected
        return fallback

    return {}

