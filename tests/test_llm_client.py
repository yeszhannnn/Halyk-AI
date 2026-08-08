from __future__ import annotations

import asyncio
from decimal import Decimal
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from anthropic import RateLimitError as AnthropicRateLimitError
from openai import RateLimitError

from agent.llm.client import (
    LLMTransportExhaustedError,
    _cache_key,
    _estimate_request_tokens,
    _is_rate_limited,
    _is_transport_error,
    _is_validation_error,
    _retry_after_seconds,
    _usage_fields,
    vision_image_block,
)
from agent.llm.token_bucket import TokenBucket
from agent.stages.s4a_covenants import (
    _extract_threshold_candidates,
    _normalize_threshold_number,
    _threshold_matches_candidates,
)


def _rate_limit_error(message: str) -> RateLimitError:
    response = MagicMock()
    response.headers = {}
    response.status_code = 429
    return RateLimitError(message, response=response, body=message)


def _anthropic_rate_limit_error(message: str) -> AnthropicRateLimitError:
    response = MagicMock()
    response.headers = {}
    response.status_code = 429
    return AnthropicRateLimitError(message, response=response, body=message)


def test_is_rate_limited_classifies_anthropic_429_as_transport() -> None:
    exc = _anthropic_rate_limit_error("Rate limit reached. Please try again in 618ms.")
    assert _is_rate_limited(exc)
    assert _is_transport_error(exc)
    assert not _is_validation_error(exc)
    assert _retry_after_seconds(exc) == pytest.approx(0.618)


def test_usage_fields_reads_anthropic_token_names() -> None:
    usage = SimpleNamespace(input_tokens=120, output_tokens=45)
    assert _usage_fields(usage) == {
        "prompt_tokens": 120,
        "completion_tokens": 45,
        "total_tokens": 165,
    }


def test_cache_key_differs_by_pass_index() -> None:
    base_kwargs = {
        "provider": "openai",
        "model": "gpt-4o-mini",
        "messages": [{"role": "user", "content": "hello"}],
        "params": {"temperature": 0},
    }
    voted_key = _cache_key(**base_kwargs)
    pass_keys = [_cache_key(**base_kwargs, pass_index=index) for index in range(3)]
    assert len(set(pass_keys)) == 3
    assert voted_key not in pass_keys


def test_cache_key_includes_provider() -> None:
    openai_key = _cache_key(
        provider="openai",
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": "hello"}],
        params={"temperature": 0},
    )
    anthropic_key = _cache_key(
        provider="anthropic",
        model="claude-haiku-4-5-20251001",
        messages=[{"role": "user", "content": "hello"}],
        params={"temperature": 0},
    )
    assert openai_key != anthropic_key


def test_vision_image_block_formats_by_provider(tmp_path: Path) -> None:
    image_path = tmp_path / "page.png"
    image_path.write_bytes(b"fake-png")

    with patch("agent.llm.client.LLM_PROVIDER", "openai"):
        openai_block = vision_image_block(image_path)
    assert openai_block["type"] == "image_url"
    assert openai_block["image_url"]["url"].startswith("data:image/png;base64,")

    with patch("agent.llm.client.LLM_PROVIDER", "anthropic"):
        anthropic_block = vision_image_block(image_path)
    assert anthropic_block == {
        "type": "image",
        "source": {
            "type": "base64",
            "media_type": "image/png",
            "data": openai_block["image_url"]["url"].split(",", 1)[1],
        },
    }


def test_retry_after_seconds_parses_milliseconds() -> None:
    exc = _rate_limit_error("Rate limit reached. Please try again in 618ms.")
    assert _retry_after_seconds(exc) == pytest.approx(0.618)


def test_retry_after_seconds_parses_seconds() -> None:
    exc = _rate_limit_error("Rate limit reached. Please try again in 2.5s.")
    assert _retry_after_seconds(exc) == pytest.approx(2.5)


def test_is_rate_limited_unwraps_instructor_wrapper() -> None:
    from instructor.core import InstructorRetryException

    inner = _rate_limit_error("429 Too Many Requests")
    outer = InstructorRetryException(
        "Max retries exceeded. Total attempts: 1",
        last_completion=None,
        n_attempts=1,
        total_usage=None,
        create_kwargs=None,
        messages=[],
    )
    outer.__cause__ = inner
    assert _is_rate_limited(outer)
    assert _is_transport_error(outer)
    assert not _is_validation_error(outer)


@pytest.mark.asyncio
async def test_token_bucket_blocks_over_budget() -> None:
    bucket = TokenBucket(100, window_seconds=0.2)
    await bucket.acquire(80)
    await bucket.acquire(20)

    started = asyncio.get_event_loop().time()
    await bucket.acquire(1)
    elapsed = asyncio.get_event_loop().time() - started
    assert elapsed >= 0.05


def test_estimate_request_tokens_is_positive() -> None:
    tokens = _estimate_request_tokens([{"role": "user", "content": "hello"}])
    assert tokens > 0


def test_threshold_candidates_from_clause_text() -> None:
    text = (
        "не допускать превышения 0.42x и при условии превышения $4,000,000.00 "
        "применяется тест 1.70x"
    )
    candidates = _extract_threshold_candidates(text)
    values = [value for value, _token in candidates]
    assert Decimal("0.42") in values
    assert Decimal("4000000.00") in values
    assert Decimal("1.70") in values


def test_threshold_matches_candidates() -> None:
    candidates = [(Decimal("0.42"), "0.42x"), (Decimal("4000000"), "$4,000,000.00")]
    assert _threshold_matches_candidates(Decimal("0.42"), candidates)
    assert not _threshold_matches_candidates(Decimal("0.43"), candidates)


def test_normalize_threshold_number() -> None:
    assert _normalize_threshold_number("4,000,000.00") == Decimal("4000000.00")
    assert _normalize_threshold_number("1,70") == Decimal("1.70")


@pytest.mark.asyncio
async def test_transport_retries_at_least_five_times_on_429() -> None:
    from agent.llm.client import LLMClient, MAX_REQUEST_ATTEMPTS

    attempts = {"count": 0}

    async def flaky_create(**_kwargs: object) -> tuple[object, object]:
        attempts["count"] += 1
        raise _rate_limit_error("Please try again in 1ms")

    client = LLMClient(semaphore=asyncio.Semaphore(1))
    client._client.create_with_completion = AsyncMock(side_effect=flaky_create)  # type: ignore[method-assign]

    with patch("agent.llm.client.get_token_bucket") as bucket_mock:
        bucket_mock.return_value.acquire = AsyncMock()
        with patch("agent.llm.client._sleep_with_backoff", new=AsyncMock()):
            with pytest.raises(LLMTransportExhaustedError) as exc_info:
                from pydantic import BaseModel

                class Probe(BaseModel):
                    answer: str

                await client._request_with_retries(
                    response_model=Probe,
                    messages=[{"role": "user", "content": "probe"}],
                    request_params={},
                )

    assert attempts["count"] == MAX_REQUEST_ATTEMPTS
    assert exc_info.value.attempts == MAX_REQUEST_ATTEMPTS


@pytest.mark.asyncio
async def test_unknown_exception_retries_before_exhausting() -> None:
    from agent.llm.client import LLMClient, MAX_REQUEST_ATTEMPTS

    attempts = {"count": 0}

    async def flaky_create(**_kwargs: object) -> tuple[object, object]:
        attempts["count"] += 1
        raise RuntimeError("truncated tool call")

    client = LLMClient(semaphore=asyncio.Semaphore(1))
    client._client.create_with_completion = AsyncMock(side_effect=flaky_create)  # type: ignore[method-assign]

    with patch("agent.llm.client.get_token_bucket") as bucket_mock:
        bucket_mock.return_value.acquire = AsyncMock()
        with patch("agent.llm.client._sleep_with_backoff", new=AsyncMock()):
            with pytest.raises(LLMTransportExhaustedError) as exc_info:
                from pydantic import BaseModel

                class Probe(BaseModel):
                    answer: str

                await client._request_with_retries(
                    response_model=Probe,
                    messages=[{"role": "user", "content": "probe"}],
                    request_params={},
                )

    assert attempts["count"] == MAX_REQUEST_ATTEMPTS
    assert exc_info.value.attempts == MAX_REQUEST_ATTEMPTS
