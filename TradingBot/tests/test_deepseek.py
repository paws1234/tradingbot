"""Tests for app.data.deepseek — the Stage 3 JSON veto gate (Task 21).

Drives the real ``DeepSeekClient`` with an injected fake ``AsyncOpenAI``
client — no network. Verification points:

- A valid JSON reply parses into the returned ``TradeDecision``, and the
  request carries ``response_format={"type": "json_object"}`` with the
  configured model.
- Every failure to produce a verdict fails safe to ``execute=false``:
  malformed JSON, a non-object reply, a reply that fails ``TradeDecision``
  validation, an empty body, and a non-transient API error (no retry).
- Transient API errors are retried with backoff, then succeed or fail safe
  after retry exhaustion.
- ``close()`` shuts the underlying client down.
"""

import json
from collections.abc import Callable
from datetime import datetime, timezone
from types import SimpleNamespace

import httpx
import pytest
from openai import (
    APIConnectionError,
    APITimeoutError,
    APIError,
    InternalServerError,
    RateLimitError,
)

from app.config import Settings
from app.data.deepseek import DeepSeekClient
from app.models.schemas import Signal, TradeDecision

T0 = datetime(2026, 8, 21, 12, 0, tzinfo=timezone.utc)

_REQUEST = httpx.Request("POST", "https://api.deepseek.com/chat/completions")
_RESPONSE = httpx.Response(500, request=_REQUEST)


def make_signal() -> Signal:
    """A valid BUY signal with a 10.0 stop distance."""
    return Signal(
        strategy="ema_fvg",
        side="BUY",
        instrument="XAU_USD",
        entry=100.0,
        stop_loss=90.0,
        take_profit=110.0,
        atr=1.0,
        reason="test signal",
        timestamp=T0,
    )


def json_reply(payload: object) -> SimpleNamespace:
    """A response whose message content is `payload` serialized as JSON."""
    return SimpleNamespace(
        choices=[
            SimpleNamespace(message=SimpleNamespace(content=json.dumps(payload)))
        ]
    )


def raw_reply(content: str) -> SimpleNamespace:
    """A response whose message content is the raw string `content`."""
    return SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content=content))]
    )


def empty_reply() -> SimpleNamespace:
    """A response with no choices — an unusable body."""
    return SimpleNamespace(choices=[])


def rate_limited() -> RateLimitError:
    return RateLimitError("rate limited", response=_RESPONSE, body=b"")


def internal_server() -> InternalServerError:
    return InternalServerError("internal error", response=_RESPONSE, body=b"")


def timeout() -> APITimeoutError:
    return APITimeoutError(request=_REQUEST)


def connection_error() -> APIConnectionError:
    return APIConnectionError(request=_REQUEST)


def api_error() -> APIError:
    return APIError("bad request", request=_REQUEST, body=None)


class _Completions:
    """Plays a script of outcomes; each `create` pops the next one."""

    def __init__(self, script: list[object]) -> None:
        self._script = list(script)
        self.calls: list[dict] = []

    async def create(self, **kwargs: object) -> object:
        self.calls.append(kwargs)
        outcome = self._script.pop(0)
        if isinstance(outcome, Exception):
            raise outcome
        return outcome


class FakeChat:
    def __init__(self, script: list[object]) -> None:
        self.completions = _Completions(script)


class FakeOpenAI:
    def __init__(self, script: list[object]) -> None:
        self.chat = FakeChat(script)
        self.closed = False

    async def close(self) -> None:
        self.closed = True


def make_client(
    settings: Settings, script: list[object], max_retries: int = 2
) -> tuple[DeepSeekClient, FakeOpenAI]:
    """A DeepSeekClient over a fake client, with zero retry sleep."""
    fake = FakeOpenAI(script)
    client = DeepSeekClient(
        settings, client=fake, max_retries=max_retries, initial_backoff=0.0
    )
    return client, fake


VALID = {"execute": True, "confidence": 8, "reason": "trend intact"}


# --- happy path -------------------------------------------------------------


@pytest.mark.asyncio
async def test_evaluate_returns_parsed_decision(
    make_settings: Callable[..., Settings],
) -> None:
    client, _ = make_client(make_settings(), [json_reply(VALID)])

    decision = await client.evaluate(make_signal())

    assert decision == TradeDecision(execute=True, confidence=8, reason="trend intact")


@pytest.mark.asyncio
async def test_evaluate_passes_through_legitimate_veto(
    make_settings: Callable[..., Settings],
) -> None:
    # A well-formed model veto is returned as-is — it is NOT rewritten into a
    # fail-safe, which would hide the model's reasoning from the audit trail.
    veto = {"execute": False, "confidence": 3, "reason": "counter-trend risk"}
    client, _ = make_client(make_settings(), [json_reply(veto)])

    decision = await client.evaluate(make_signal())

    assert decision == TradeDecision(
        execute=False, confidence=3, reason="counter-trend risk"
    )
    assert not decision.reason.startswith("fail_safe:")


@pytest.mark.asyncio
async def test_evaluate_requests_json_object_format_and_model(
    make_settings: Callable[..., Settings],
) -> None:
    settings = make_settings()
    client, fake = make_client(settings, [json_reply(VALID)])

    await client.evaluate(make_signal())

    call = fake.chat.completions.calls[0]
    assert call["model"] == settings.deepseek_model
    assert call["response_format"] == {"type": "json_object"}
    system_content = call["messages"][0]["content"]
    assert "json" in system_content  # JSON mode requires the word "json" in-prompt
    user_content = call["messages"][1]["content"]
    assert "ema_fvg" in user_content
    assert "BUY" in user_content
    assert "XAU_USD" in user_content


# --- parse/validation failures fail safe ------------------------------------


@pytest.mark.asyncio
async def test_evaluate_malformed_json_fails_safe_execute_false(
    make_settings: Callable[..., Settings],
) -> None:
    # The acceptance criterion: a parse failure yields execute=false.
    client, fake = make_client(make_settings(), [raw_reply("{not valid json")])

    decision = await client.evaluate(make_signal())

    assert decision.execute is False
    assert decision.reason.startswith("fail_safe:")
    assert len(fake.chat.completions.calls) == 1  # malformed replies are not retried


@pytest.mark.asyncio
async def test_evaluate_non_object_reply_fails_safe(
    make_settings: Callable[..., Settings],
) -> None:
    client, _ = make_client(make_settings(), [json_reply([1, 2, 3])])

    decision = await client.evaluate(make_signal())

    assert decision.execute is False


@pytest.mark.asyncio
async def test_evaluate_validation_failure_fails_safe(
    make_settings: Callable[..., Settings],
) -> None:
    # Missing the required `reason` field — pydantic rejects the payload.
    client, _ = make_client(
        make_settings(), [json_reply({"execute": True, "confidence": 8})]
    )

    decision = await client.evaluate(make_signal())

    assert decision.execute is False


@pytest.mark.asyncio
async def test_evaluate_out_of_range_confidence_fails_safe(
    make_settings: Callable[..., Settings],
) -> None:
    client, _ = make_client(
        make_settings(),
        [json_reply({"execute": True, "confidence": 11, "reason": "x"})],
    )

    decision = await client.evaluate(make_signal())

    assert decision.execute is False


@pytest.mark.asyncio
async def test_evaluate_empty_body_fails_safe(
    make_settings: Callable[..., Settings],
) -> None:
    client, _ = make_client(make_settings(), [empty_reply()])

    decision = await client.evaluate(make_signal())

    assert decision.execute is False


# --- API errors --------------------------------------------------------------


@pytest.mark.asyncio
async def test_evaluate_nontransient_api_error_fails_safe_no_retry(
    make_settings: Callable[..., Settings],
) -> None:
    client, fake = make_client(make_settings(), [api_error()])

    decision = await client.evaluate(make_signal())

    assert decision.execute is False
    assert "API error" in decision.reason
    assert len(fake.chat.completions.calls) == 1  # permanent error is not retried


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "error",
    [
        pytest.param(rate_limited(), id="rate_limit"),
        pytest.param(internal_server(), id="internal_server"),
        pytest.param(timeout(), id="timeout"),
        pytest.param(connection_error(), id="connection"),
    ],
)
async def test_evaluate_retries_transient_then_returns(
    make_settings: Callable[..., Settings], error: Exception
) -> None:
    client, fake = make_client(make_settings(), [error, json_reply(VALID)])

    decision = await client.evaluate(make_signal())

    assert decision == TradeDecision(execute=True, confidence=8, reason="trend intact")
    assert len(fake.chat.completions.calls) == 2


@pytest.mark.asyncio
async def test_evaluate_retry_exhaustion_fails_safe(
    make_settings: Callable[..., Settings],
) -> None:
    client, fake = make_client(
        make_settings(),
        [rate_limited(), rate_limited(), rate_limited()],
        max_retries=2,
    )

    decision = await client.evaluate(make_signal())

    assert decision.execute is False
    assert "retries" in decision.reason
    assert len(fake.chat.completions.calls) == 3  # 1 + max_retries attempts


# --- lifecycle ----------------------------------------------------------------


@pytest.mark.asyncio
async def test_close_closes_underlying_client(
    make_settings: Callable[..., Settings],
) -> None:
    client, fake = make_client(make_settings(), [json_reply(VALID)])

    await client.close()

    assert fake.closed is True
