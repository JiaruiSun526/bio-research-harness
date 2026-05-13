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
        """
        self.default_model = default_model
        self.proxy = proxy or None  # normalize "" to None
        self.api_base = api_base
        self.api_key = api_key

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
        )
