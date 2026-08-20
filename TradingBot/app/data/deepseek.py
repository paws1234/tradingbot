"""Async DeepSeek client — the Stage 3 JSON veto gate (Plan.md; strategy.md §6.3).

Contract (Plan.md pipeline stage 3; task 12):

- `evaluate(signal)` — one veto verdict for a candidate :class:`Signal`,
  returned as a :class:`~app.models.schemas.TradeDecision`.
- JSON-only: the request uses ``response_format={"type": "json_object"}`` and
  the reply must parse into a `TradeDecision`. Any failure to produce a valid
  verdict — malformed JSON, missing keys, bad types, an empty body, or an
  unrecoverable API error — fails safe to ``execute=false``, so a broken
  model reply can never place an order.
- Retry/backoff: transient API errors (connection, timeout, rate limit, 5xx)
  are retried with exponential backoff up to ``max_retries``. Non-transient
  API errors (bad key, malformed request) fail safe immediately without
  retry. The SDK's own retry loop is disabled (``max_retries=0``) so backoff
  and fail-safe live in exactly one place.
- This layer reports the verdict only. Applying the confidence threshold
  (``execute and confidence >= min_confidence``) is the engine's job
  (Task 15).

`client` is an injection seam for tests (respx mocks the transport; or pass a
fake `AsyncOpenAI`). Production passes nothing and gets a real client.
"""

import asyncio
import json
import logging
from typing import Any

import httpx
from openai import (
    APIConnectionError,
    APITimeoutError,
    APIError,
    AsyncOpenAI,
    InternalServerError,
    RateLimitError,
)
from pydantic import ValidationError

from app.config import Settings
from app.models.schemas import Signal, TradeDecision

logger = logging.getLogger(__name__)

# Transient failures worth a retry — the next attempt may well survive.
_RETRYABLE = (APIConnectionError, APITimeoutError, RateLimitError, InternalServerError)


class DeepSeekClient:
    """Async DeepSeek JSON veto gate (Plan.md Stage 3)."""

    def __init__(
        self,
        settings: Settings,
        client: AsyncOpenAI | None = None,
        max_retries: int = 2,
        initial_backoff: float = 1.0,
        max_backoff: float = 30.0,
    ) -> None:
        self._settings = settings
        if client is None:
            client = AsyncOpenAI(
                api_key=settings.deepseek_api_key,
                base_url=settings.deepseek_base_url,
                # Retrying is this layer's job; the SDK's built-in loop would
                # stack on top of ours and double the waits. The default read
                # timeout is 600 s — far too long for a live veto call, so
                # bound it like the other data-layer clients.
                max_retries=0,
                timeout=httpx.Timeout(
                    connect=10.0, read=30.0, write=10.0, pool=10.0
                ),
            )
        self._client = client
        self._max_retries = max_retries
        self._initial_backoff = initial_backoff
        self._max_backoff = max_backoff

    async def evaluate(self, signal: Signal) -> TradeDecision:
        """Ask DeepSeek to veto one signal; never raises on API or parse failures.

        Returns the parsed verdict on success, or a fail-safe
        ``execute=false`` decision after retry exhaustion, on a non-transient
        API error, or when the model's reply cannot be turned into a valid
        `TradeDecision`.
        """
        attempts = 0
        backoff = self._initial_backoff
        while True:
            attempts += 1
            try:
                response = await self._client.chat.completions.create(
                    model=self._settings.deepseek_model,
                    messages=self._build_messages(signal),
                    response_format={"type": "json_object"},
                )
                decision = self._parse_decision(self._extract_content(response))
                if decision is not None:
                    return decision
                # A malformed reply won't fix itself on retry — fail safe now.
                return self._fail_safe("unparseable model reply")
            except _RETRYABLE as exc:
                logger.warning("DeepSeek veto failed (%s), retrying", exc)
            except APIError as exc:
                # Permanent API failure (bad key, malformed request) — the veto
                # gate is mandatory, so fail safe instead of raising.
                return self._fail_safe(f"veto API error: {exc}")
            if attempts > self._max_retries:
                return self._fail_safe(
                    f"veto unavailable after {self._max_retries} retries"
                )
            await asyncio.sleep(backoff)
            backoff = min(backoff * 2, self._max_backoff)

    @staticmethod
    def _build_messages(signal: Signal) -> list[dict[str, str]]:
        """System + user messages presenting one candidate trade to veto.

        JSON mode requires the word "json" to appear in the prompt — the
        system message satisfies that.
        """
        summary = (
            f"Strategy {signal.strategy} flags a {signal.side} entry on "
            f"{signal.instrument}. Entry {signal.entry}, stop-loss "
            f"{signal.stop_loss}, take-profit {signal.take_profit}, ATR "
            f"{signal.atr}. Reason: {signal.reason}."
        )
        return [
            {
                "role": "system",
                "content": (
                    "You are the risk manager of an FX/CFD trading bot. You "
                    "never place orders. Review one candidate trade and reply "
                    "with a json object: {\"execute\": true|false, "
                    "\"confidence\": 0-10, \"reason\": \"short justification\"}. "
                    "Set execute=true only if the trade is sound and matches "
                    "the stated strategy; when in doubt, execute=false."
                ),
            },
            {"role": "user", "content": summary},
        ]

    @staticmethod
    def _extract_content(response: Any) -> str | None:
        """Pull the assistant's message text; None on an empty/unusable body."""
        try:
            return response.choices[0].message.content
        except (IndexError, AttributeError, TypeError):
            logger.warning("DeepSeek veto: response carried no message content")
            return None

    @staticmethod
    def _parse_decision(content: str | None) -> TradeDecision | None:
        """Parse the model's JSON reply into a TradeDecision; None on failure."""
        if content is None:
            return None
        try:
            payload = json.loads(content)
        except (ValueError, TypeError) as exc:
            logger.warning("DeepSeek veto: unparseable reply (%s)", exc)
            return None
        if not isinstance(payload, dict):
            logger.warning("DeepSeek veto: reply is not a JSON object")
            return None
        try:
            return TradeDecision.model_validate(payload)
        except ValidationError as exc:
            logger.warning("DeepSeek veto: reply failed validation: %s", exc)
            return None

    @staticmethod
    def _fail_safe(reason: str) -> TradeDecision:
        """The safe default: never trade when no verdict can be produced."""
        logger.warning("DeepSeek veto fail-safe: %s", reason)
        return TradeDecision(execute=False, confidence=0, reason=f"fail_safe: {reason}")

    async def close(self) -> None:
        """Shut the underlying client down (engine lifespan teardown)."""
        await self._client.close()
