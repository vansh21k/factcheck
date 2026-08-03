"""Model access, behind one narrow port.

``LLMClient`` is deliberately the smallest surface that both model passes need. That
narrowness is what lets the whole test suite run with a scripted model, no network,
and no API key.
"""

from __future__ import annotations

import os
from typing import Any

from ..errors import MissingAPIKeyError, ModelCallError
from ..ports import LLMClient


class CallCounter:
    """Shared tally, so per-claim model cost is visible during ordinary use."""

    def __init__(self) -> None:
        self.calls = 0


class CountingLLM:
    """Counts calls and delegates. Wrapping rather than instrumenting each stage keeps
    the count correct when a stage is swapped for a different implementation."""

    def __init__(self, inner: LLMClient, counter: CallCounter) -> None:
        self._inner = inner
        self._counter = counter

    def complete(
        self,
        *,
        system: str,
        user: str,
        model: str,
        temperature: float,
        tool: dict[str, Any] | None = None,
        max_tokens: int = 1024,
    ) -> dict[str, Any]:
        self._counter.calls += 1
        return self._inner.complete(
            system=system,
            user=user,
            model=model,
            temperature=temperature,
            tool=tool,
            max_tokens=max_tokens,
        )


class AnthropicClient:
    """The real client. Tool use is forced, so structured output is not optional."""

    def __init__(self, api_key: str | None = None) -> None:
        key = api_key or os.environ.get("ANTHROPIC_API_KEY")
        if not key:
            raise MissingAPIKeyError(
                "ANTHROPIC_API_KEY is not set. Export it (or add it to your shell profile) "
                "before running verification."
            )
        from anthropic import Anthropic  # imported here so offline commands never need it

        self._client = Anthropic(api_key=key)

    def complete(
        self,
        *,
        system: str,
        user: str,
        model: str,
        temperature: float,
        tool: dict[str, Any] | None = None,
        max_tokens: int = 1024,
    ) -> dict[str, Any]:
        kwargs: dict[str, Any] = {
            "model": model,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "system": system,
            "messages": [{"role": "user", "content": user}],
        }
        if tool is not None:
            kwargs["tools"] = [tool]
            kwargs["tool_choice"] = {"type": "tool", "name": tool["name"]}

        try:
            response = self._client.messages.create(**kwargs)
        except Exception as exc:  # SDK error types stay behind this boundary
            raise ModelCallError(f"{model} call failed: {_reason(exc)}") from exc

        for block in response.content:
            if getattr(block, "type", None) == "tool_use":
                result: dict[str, Any] = dict(block.input)
                return result
        text = "".join(getattr(b, "text", "") for b in response.content)
        if tool is not None:
            # Forced tool choice makes this unreachable in practice; if the API ever
            # returns prose anyway, an empty object degrades to `unknown` rather than
            # letting unparsed text through as a verdict.
            return {}
        return {"text": text}


def _reason(exc: Exception) -> str:
    """The API's own message when there is one; the exception otherwise.

    Billing and rate-limit failures are the two an operator can actually act on, and
    the SDK buries both inside a nested error object.
    """
    body = getattr(exc, "body", None)
    if isinstance(body, dict):
        error = body.get("error")
        if isinstance(error, dict) and error.get("message"):
            return str(error["message"])
    return str(exc)
