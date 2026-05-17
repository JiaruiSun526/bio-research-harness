"""LLM client — thin wrapper over litellm, isolating the third-party dependency.

All LLM calls go through this module. No other module imports litellm directly.
Responsible for converting litellm's raw response into our LLMResponse model.
"""

from __future__ import annotations

import json
import logging
import os
import time
from contextlib import contextmanager
from typing import Any, Iterator

import litellm

from .schemas import LLMResponse
from .schemas import ToolCallRequest

log = logging.getLogger(__name__)

# Transient errors worth retrying — connection drops, rate limits, server errors.
_RETRYABLE_EXCEPTIONS = (
    litellm.exceptions.APIConnectionError,
    litellm.exceptions.RateLimitError,
    litellm.exceptions.ServiceUnavailableError,
    litellm.exceptions.Timeout,
    litellm.exceptions.APIError,
)

_MAX_RETRIES = 10
_BASE_DELAY = 2.0  # seconds; doubles each retry, capped at 60s
_PROXY_ENV_VARS = ("HTTP_PROXY", "HTTPS_PROXY", "http_proxy", "https_proxy")


class LLMClient:
    """Thin wrapper over litellm for testability and dependency isolation.

    Uses litellm.completion() which supports 30+ LLM providers
    through the OpenAI-compatible API format.

    Per-client proxy isolation: each client applies its own proxy setting
    around each chat() call via a context manager. This prevents one
    provider's proxy (e.g. OpenRouter requires proxy) from leaking to
    another (e.g. MiMo/xiaomi must be called without proxy).
    """

    def __init__(
        self,
        default_model: str,
        proxy: str | None = None,
        api_base: str | None = None,
        api_key: str | None = None,
        cache_enabled: bool = False,
    ) -> None:
        """
        Args:
            default_model: Default model identifier (e.g. "gpt-4o", "mimo-v2-pro").
                Can be overridden per call.
            proxy: Optional HTTP proxy URL for outbound requests. Applied only
                around chat() invocations; does not mutate process-wide env at
                construction time.
            api_base: Optional API base URL override (e.g. "https://api.xiaomimimo.com/v1"
                for OpenAI-compatible providers).
            api_key: Optional API key. If provided, passed directly to litellm
                instead of relying on environment variables.
            cache_enabled: If True, attach Anthropic-style ``cache_control``
                markers to the system message and (when present) the last tool
                definition for each call. Safe to enable for OpenAI-compatible
                providers as well — they ignore the unknown fields and rely on
                automatic prefix caching. Cache token usage from the response
                is reported regardless of this flag.
        """
        self.default_model = default_model
        self.proxy = proxy or None  # normalize "" to None
        self.api_base = api_base
        self.api_key = api_key
        self.cache_enabled = cache_enabled

    @contextmanager
    def _proxy_env(self) -> Iterator[None]:
        """Scope HTTP_PROXY/HTTPS_PROXY to the wrapped block.

        - If self.proxy is set: temporarily set HTTP_PROXY/HTTPS_PROXY (and
          their lowercase variants) to self.proxy.
        - If self.proxy is None/empty: temporarily remove those env vars so
          providers without proxy requirement (e.g. MiMo) don't inherit a
          foreign proxy from the shell or another client.

        Always restores prior values in finally. Agent and SimulatedUser calls
        are sequential, so no race condition.
        """
        saved: dict[str, str | None] = {var: os.environ.get(var) for var in _PROXY_ENV_VARS}
        try:
            if self.proxy:
                for var in _PROXY_ENV_VARS:
                    os.environ[var] = self.proxy
            else:
                for var in _PROXY_ENV_VARS:
                    os.environ.pop(var, None)
            yield
        finally:
            for var, prior in saved.items():
                if prior is None:
                    os.environ.pop(var, None)
                else:
                    os.environ[var] = prior

    def chat(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        model: str | None = None,
    ) -> LLMResponse:
        """Send messages to the LLM and return a normalized response.

        Args:
            messages: Message list in OpenAI format
                ([{"role": "system", "content": "..."}, ...]).
            tools: Tool definitions in OpenAI function-calling format.
                None means no tools available for this call.
            model: Override the default model for this call.

        Returns:
            LLMResponse with content and/or tool_calls.
        """
        if self.cache_enabled:
            messages = _mark_system_cache(messages)
            if tools:
                tools = _mark_last_tool_cache(tools)

        completion_kwargs: dict[str, Any] = {
            "model": model or self.default_model,
            "messages": messages,
        }
        if tools:
            completion_kwargs["tools"] = tools
        if self.api_base:
            completion_kwargs["api_base"] = self.api_base
        if self.api_key:
            completion_kwargs["api_key"] = self.api_key

        with self._proxy_env():
            last_exc: Exception | None = None
            for attempt in range(_MAX_RETRIES):
                try:
                    response = litellm.completion(**completion_kwargs)
                    break
                except _RETRYABLE_EXCEPTIONS as exc:
                    last_exc = exc
                    if attempt == _MAX_RETRIES - 1:
                        log.error("LLM call failed after %d retries: %s", _MAX_RETRIES, exc)
                        raise
                    delay = min(_BASE_DELAY * (2 ** attempt), 60.0)
                    log.warning(
                        "LLM call attempt %d/%d failed (%s: %s), retrying in %.0fs",
                        attempt + 1, _MAX_RETRIES, type(exc).__name__, exc, delay,
                    )
                    time.sleep(delay)

        message = response.choices[0].message
        raw_tool_calls = message.tool_calls or []
        usage = getattr(response, "usage", None)

        return LLMResponse(
            content=message.content,
            tool_calls=[
                ToolCallRequest(
                    id=tool_call.id,
                    name=tool_call.function.name,
                    arguments=json.loads(tool_call.function.arguments),
                )
                for tool_call in raw_tool_calls
            ],
            prompt_tokens=getattr(usage, "prompt_tokens", 0) or 0,
            completion_tokens=getattr(usage, "completion_tokens", 0) or 0,
            cache_creation_tokens=_extract_cache_creation_tokens(usage),
            cache_read_tokens=_extract_cache_read_tokens(usage),
        )


def _mark_system_cache(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Return a shallow copy of messages with cache_control on the system message.

    Anthropic prompt caching requires the cached content to be expressed as a
    list of content blocks with ``cache_control: {"type": "ephemeral"}`` on the
    final block of the cached prefix. The system message is the largest stable
    block in our requests, so caching it alone covers most of the prefix.

    Mutating the caller's list/dicts would leak cache markers back to the loop
    after the call returns; we only modify a shallow copy.
    """

    if not messages or messages[0].get("role") != "system":
        return messages

    out = list(messages)
    system_msg = dict(out[0])
    content = system_msg.get("content")
    if isinstance(content, str):
        system_msg["content"] = [
            {
                "type": "text",
                "text": content,
                "cache_control": {"type": "ephemeral"},
            }
        ]
    elif isinstance(content, list) and content:
        # Already in list-of-blocks form — mark the last block.
        new_blocks = list(content)
        last = dict(new_blocks[-1])
        last["cache_control"] = {"type": "ephemeral"}
        new_blocks[-1] = last
        system_msg["content"] = new_blocks
    out[0] = system_msg
    return out


def _mark_last_tool_cache(tools: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Return a shallow copy of tools with cache_control on the final entry."""

    if not tools:
        return tools
    out = list(tools)
    last = dict(out[-1])
    last["cache_control"] = {"type": "ephemeral"}
    out[-1] = last
    return out


def _extract_cache_creation_tokens(usage: Any) -> int:
    """Pull cache-write token count from a litellm usage object, if reported."""

    if usage is None:
        return 0
    return int(getattr(usage, "cache_creation_input_tokens", 0) or 0)


def _extract_cache_read_tokens(usage: Any) -> int:
    """Pull cache-hit token count, supporting both Anthropic and OpenAI shapes.

    Anthropic exposes ``cache_read_input_tokens`` directly on usage; OpenAI
    exposes ``prompt_tokens_details.cached_tokens``. We fall back through both.
    """

    if usage is None:
        return 0
    direct = getattr(usage, "cache_read_input_tokens", None)
    if direct:
        return int(direct)
    details = getattr(usage, "prompt_tokens_details", None)
    if details is not None:
        return int(getattr(details, "cached_tokens", 0) or 0)
    return 0
