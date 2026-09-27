"""
OpenCode Go Responses API client for Muse Spark Contributor models.

Muse Spark Contributor (muse-spark-1.3-contributor, muse-spark-1.2-contributor)
is served at https://opencode.ai/zen/go/v1/responses with the OpenAI Responses
format (api = "openai-responses"), NOT /chat/completions.

BridgeClip planning prompts are built as chat messages; this module converts
them to Responses `input` items, requests structured JSON output, and returns
a chat-completions-shaped body so the existing planner parser keeps working.
"""

from __future__ import annotations

import base64
import json
import logging
import uuid
from typing import Any, Optional

import httpx

logger = logging.getLogger(__name__)

MAX_RESPONSES_BYTES = 2 * 1024 * 1024
RETRYABLE_STATUS_CODES = {408, 429, 500, 502, 503, 504}


class OpenCodeError(Exception):
    """An OpenCode Go request failed. `retryable` marks transient failures."""

    def __init__(self, message: str, retryable: bool = False):
        super().__init__(message)
        self.retryable = retryable


def _session_headers() -> dict[str, str]:
    # OpenCode Go monitors for coding-agent traffic: send a stable session id
    # and a real client UA instead of a generic SDK name.
    return {
        "Accept-Encoding": "identity",
        "User-Agent": "BridgeClip/0.1.18 (opencode-go; muse-spark)",
        "X-Title": "BridgeClip AI Clipping Agent",
        "HTTP-Referer": "https://github.com/notromka/bridgeclip",
        "x-opencode-session": f"bridgeclip-{uuid.uuid4()}",
    }


def chat_messages_to_responses_input(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Convert OpenAI chat messages to Responses API input items."""
    items: list[dict[str, Any]] = []
    for msg in messages:
        role = msg.get("role", "user")
        if role == "system":
            content = msg.get("content", "")
            if isinstance(content, list):
                content = " ".join(
                    p.get("text", "") for p in content if isinstance(p, dict)
                )
            items.append({
                "role": "system",
                "content": [{"type": "input_text", "text": str(content)}],
            })
            continue
        content = msg.get("content")
        parts: list[dict[str, Any]] = []
        if isinstance(content, str):
            parts.append({"type": "input_text", "text": content})
        elif isinstance(content, list):
            for part in content:
                if not isinstance(part, dict):
                    continue
                ptype = part.get("type")
                if ptype in ("text", "input_text"):
                    parts.append({"type": "input_text", "text": str(part.get("text", ""))})
                elif ptype == "image_url":
                    url = (part.get("image_url") or {}).get("url", "")
                    if url.startswith("data:"):
                        # data:image/jpeg;base64,XXXX -> Responses image input
                        try:
                            b64 = url.split(",", 1)[1]
                            base64.b64decode(b64, validate=False)
                            parts.append({
                                "type": "input_image",
                                "image_url": url,
                            })
                        except Exception:
                            continue
                    elif url:
                        parts.append({"type": "input_image", "image_url": url})
                elif ptype == "input_image":
                    parts.append(part)
        else:
            parts.append({"type": "input_text", "text": str(content or "")})
        items.append({"role": role if role in ("user", "assistant", "system") else "user", "content": parts})
    return items


def responses_output_text(body: dict[str, Any]) -> tuple[Optional[str], Optional[str]]:
    """Extract (text, finish_reason) from a Responses API body."""
    texts: list[str] = []
    for item in body.get("output") or []:
        if not isinstance(item, dict):
            continue
        for part in item.get("content") or []:
            if isinstance(part, dict) and part.get("type") in ("output_text", "text"):
                t = part.get("text")
                if isinstance(t, str):
                    texts.append(t)
                elif isinstance(t, dict) and isinstance(t.get("value"), str):
                    texts.append(t["value"])
    content = "".join(texts) or None
    # Map to chat-like finish reason for the planner parser.
    status = body.get("status")
    finish = "stop" if status in ("completed", None) else status
    if body.get("incomplete_details"):
        finish = "length"
    return content, finish


def to_chat_completions_body(model: str, body: dict[str, Any]) -> dict[str, Any]:
    """Shape a Responses body like chat/completions for existing parsers."""
    content, finish = responses_output_text(body)
    usage = body.get("usage") or {}
    return {
        "id": body.get("id", ""),
        "object": "chat.completion",
        "model": body.get("model", model),
        "choices": [{"message": {"role": "assistant", "content": content or ""}, "finish_reason": finish or "stop"}],
        "usage": {
            "prompt_tokens": usage.get("input_tokens", 0),
            "completion_tokens": usage.get("output_tokens", 0),
            "total_tokens": usage.get("total_tokens", 0),
            "cost": None,
        },
    }


async def responses_completion(
    client: httpx.AsyncClient,
    *,
    model: str,
    messages: list[dict[str, Any]],
    max_output_tokens: int = 32000,
    reasoning_effort: str = "medium",
    json_schema: Optional[dict[str, Any]] = None,
    schema_name: str = "clip_plan",
) -> tuple[dict, dict]:
    """POST /responses for Muse Spark Contributor models.

    Returns (chat-like body, usage) to reuse the planner's JSON parsing.
    """
    payload: dict[str, Any] = {
        "model": model,
        "input": chat_messages_to_responses_input(messages),
        "max_output_tokens": max_output_tokens,
        "store": False,
    }
    if reasoning_effort and reasoning_effort != "none":
        payload["reasoning"] = {"effort": reasoning_effort, "summary": "auto"}
    if json_schema is not None:
        payload["text"] = {
            "format": {
                "type": "json_schema",
                "name": schema_name,
                "strict": True,
                "schema": json_schema,
            }
        }
    try:
        async with client.stream(
            "POST", "/responses", json=payload,
            headers=_session_headers(), follow_redirects=False,
        ) as response:
            if response.headers.get("content-encoding", "identity").lower() != "identity":
                raise OpenCodeError("OpenCode Go returned an unsupported response encoding")
            content = bytearray()
            async for chunk in response.aiter_raw():
                if len(chunk) > MAX_RESPONSES_BYTES - len(content):
                    raise OpenCodeError("OpenCode Go response exceeds the size limit")
                content.extend(chunk)
            status = response.status_code
    except (httpx.TimeoutException, httpx.TransportError):
        raise OpenCodeError("OpenCode Go request failed", retryable=True) from None

    if status == 402:
        raise OpenCodeError("OpenCode Go balance is out of credits. Check opencode.ai/auth billing.")
    if status in (401, 403):
        raise OpenCodeError("OpenCode Go rejected the API key. Check it in Settings.")
    if status != 200:
        raise OpenCodeError(
            f"OpenCode Go API error ({status})",
            retryable=status in RETRYABLE_STATUS_CODES,
        )
    try:
        body = json.loads(content)
    except (ValueError, UnicodeError, RecursionError):
        raise OpenCodeError("OpenCode Go returned invalid JSON") from None
    if not isinstance(body, dict):
        raise OpenCodeError("OpenCode Go returned an invalid response")
    if body.get("error"):
        raise OpenCodeError("OpenCode Go provider error", retryable=True)

    chat_body = to_chat_completions_body(model, body)
    usage = (chat_body.get("usage") or {})
    usage_data = {
        "prompt_tokens": usage.get("prompt_tokens", 0),
        "completion_tokens": usage.get("completion_tokens", 0),
        "total_tokens": usage.get("total_tokens", 0),
        "cost": None,
    }
    logger.info(
        "OpenCode Go usage (%s): %s prompt, %s completion",
        body.get("model", model), usage_data["prompt_tokens"], usage_data["completion_tokens"],
    )
    return chat_body, usage_data
