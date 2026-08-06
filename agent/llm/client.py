from __future__ import annotations

import asyncio
import hashlib
import json
import random
from dataclasses import dataclass, field
from decimal import Decimal
from pathlib import Path
from typing import Any, Callable, TypeVar

import instructor
from openai import APIConnectionError, APITimeoutError, AsyncOpenAI, RateLimitError
from pydantic import BaseModel

from agent.config import BUDGET_USD, MAX_CONCURRENT, MODEL_ID, OPENAI_SEED, TEMPERATURE

T = TypeVar("T", bound=BaseModel)

CACHE_DIR = Path(".cache/llm")

MODEL_PRICING_PER_MILLION: dict[str, tuple[Decimal, Decimal]] = {
    "gpt-4o": (Decimal("2.50"), Decimal("10.00")),
    "gpt-4o-mini": (Decimal("0.15"), Decimal("0.60")),
}


class BudgetExceededError(RuntimeError):
    pass


@dataclass
class RunCounter:
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    cost_usd: Decimal = field(default_factory=lambda: Decimal("0"))
    cache_hits: int = 0
    cache_misses: int = 0
    api_calls: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "total_tokens": self.total_tokens,
            "cost_usd": str(self.cost_usd.quantize(Decimal("0.000001"))),
            "cache_hits": self.cache_hits,
            "cache_misses": self.cache_misses,
            "api_calls": self.api_calls,
        }


RUN_COUNTER = RunCounter()


def _pricing_for_model(model: str) -> tuple[Decimal, Decimal]:
    if model in MODEL_PRICING_PER_MILLION:
        return MODEL_PRICING_PER_MILLION[model]
    for key, value in MODEL_PRICING_PER_MILLION.items():
        if model.startswith(key):
            return value
    return Decimal("1.00"), Decimal("3.00")


def _estimate_cost(model: str, prompt_tokens: int, completion_tokens: int) -> Decimal:
    input_rate, output_rate = _pricing_for_model(model)
    prompt_cost = Decimal(prompt_tokens) * input_rate / Decimal("1000000")
    completion_cost = Decimal(completion_tokens) * output_rate / Decimal("1000000")
    return prompt_cost + completion_cost


def _cache_key(*, model: str, messages: list[dict[str, Any]], params: dict[str, Any]) -> str:
    payload = {
        "model": model,
        "messages": messages,
        "params": params,
    }
    digest = hashlib.sha256(
        json.dumps(payload, sort_keys=True, ensure_ascii=False).encode("utf-8"),
    )
    return digest.hexdigest()


def _cache_path(cache_key: str) -> Path:
    return CACHE_DIR / f"{cache_key}.json"


def _read_cache(cache_key: str, response_model: type[T], counter: RunCounter) -> T | None:
    path = _cache_path(cache_key)
    if not path.is_file():
        return None
    cached = json.loads(path.read_text(encoding="utf-8"))
    counter.cache_hits += 1
    counter.prompt_tokens += int(cached.get("usage", {}).get("prompt_tokens", 0))
    counter.completion_tokens += int(cached.get("usage", {}).get("completion_tokens", 0))
    counter.total_tokens += int(cached.get("usage", {}).get("total_tokens", 0))
    counter.cost_usd += Decimal(str(cached.get("cost_usd", "0")))
    return response_model.model_validate(cached["response"])


def _write_cache(
    cache_key: str,
    *,
    response: BaseModel,
    usage: dict[str, int],
    cost_usd: Decimal,
) -> None:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    payload = {
        "response": response.model_dump(mode="json"),
        "usage": usage,
        "cost_usd": str(cost_usd),
    }
    _cache_path(cache_key).write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def _record_usage(model: str, usage: Any, counter: RunCounter) -> Decimal:
    prompt_tokens = int(getattr(usage, "prompt_tokens", 0) or 0)
    completion_tokens = int(getattr(usage, "completion_tokens", 0) or 0)
    total_tokens = int(getattr(usage, "total_tokens", prompt_tokens + completion_tokens) or 0)
    cost = _estimate_cost(model, prompt_tokens, completion_tokens)

    counter.prompt_tokens += prompt_tokens
    counter.completion_tokens += completion_tokens
    counter.total_tokens += total_tokens
    counter.cost_usd += cost
    counter.api_calls += 1
    return cost


def _ensure_budget() -> None:
    if RUN_COUNTER.cost_usd >= BUDGET_USD:
        raise BudgetExceededError(
            f"LLM budget exceeded: spent {RUN_COUNTER.cost_usd} >= cap {BUDGET_USD}",
        )


def _retry_after_seconds(exc: RateLimitError) -> float | None:
    headers = getattr(exc, "response", None)
    if headers is None:
        return None
    header_value = headers.headers.get("retry-after")
    if header_value is None:
        return None
    try:
        return float(header_value)
    except ValueError:
        return None


async def _sleep_with_backoff(attempt: int, exc: Exception | None = None) -> None:
    if isinstance(exc, RateLimitError):
        retry_after = _retry_after_seconds(exc)
        if retry_after is not None:
            await asyncio.sleep(retry_after)
            return

    base = min(60.0, 2**attempt)
    jitter = random.uniform(0, base * 0.25)
    await asyncio.sleep(base + jitter)


class LLMClient:
    def __init__(
        self,
        *,
        counter: RunCounter | None = None,
        semaphore: asyncio.Semaphore | None = None,
        model: str | None = None,
    ) -> None:
        self.counter = counter or RUN_COUNTER
        self._semaphore = semaphore or asyncio.Semaphore(MAX_CONCURRENT)
        self.model = model or MODEL_ID
        self._client = instructor.from_openai(AsyncOpenAI())

    async def complete(
        self,
        *,
        response_model: type[T],
        messages: list[dict[str, Any]],
        use_cache: bool = True,
        **params: Any,
    ) -> T:
        request_params = {
            "temperature": TEMPERATURE,
            "seed": OPENAI_SEED,
            **params,
        }
        serializable_messages = json.loads(json.dumps(messages, ensure_ascii=False))
        cache_key = _cache_key(
            model=self.model,
            messages=serializable_messages,
            params=request_params,
        )

        if use_cache:
            cached = _read_cache(cache_key, response_model, self.counter)
            if cached is not None:
                return cached

        self.counter.cache_misses += 1
        _ensure_budget()

        last_exc: Exception | None = None
        for attempt in range(6):
            try:
                async with self._semaphore:
                    _ensure_budget()
                    parsed, completion = await self._client.chat.completions.create_with_completion(
                        model=self.model,
                        messages=messages,
                        response_model=response_model,
                        **request_params,
                    )
                usage = getattr(completion, "usage", None)
                usage_dict = {
                    "prompt_tokens": int(getattr(usage, "prompt_tokens", 0) or 0),
                    "completion_tokens": int(getattr(usage, "completion_tokens", 0) or 0),
                    "total_tokens": int(getattr(usage, "total_tokens", 0) or 0),
                }
                cost = _record_usage(self.model, usage, self.counter)
                if use_cache:
                    _write_cache(
                        cache_key,
                        response=parsed,
                        usage=usage_dict,
                        cost_usd=cost,
                    )
                return parsed
            except BudgetExceededError:
                raise
            except (RateLimitError, APIConnectionError, APITimeoutError) as exc:
                last_exc = exc
                await _sleep_with_backoff(attempt, exc)
            except Exception as exc:
                last_exc = exc
                if attempt >= 5:
                    break
                await _sleep_with_backoff(attempt, exc)

        raise RuntimeError("LLM request failed after retries") from last_exc

    async def complete_verified(
        self,
        *,
        response_model: type[T],
        messages: list[dict[str, Any]],
        quote_checks: Callable[[T], list[tuple[str, str, str]]],
        **params: Any,
    ) -> T:
        from agent.evidence.quotes import apply_quote_verification_with_retry

        result = await self.complete(
            response_model=response_model,
            messages=messages,
            **params,
        )
        payload = result.model_dump(mode="python")
        failed, should_retry = apply_quote_verification_with_retry(
            payload,
            fields=quote_checks(result),
            retried=False,
        )
        if should_retry:
            retry_messages = [
                *messages,
                {
                    "role": "user",
                    "content": (
                        "Some quoted fields were not verbatim substrings of the source page. "
                        f"Fix these fields with exact quotes: {', '.join(failed)}"
                    ),
                },
            ]
            result = await self.complete(
                response_model=response_model,
                messages=retry_messages,
                use_cache=False,
                **params,
            )
            payload = result.model_dump(mode="python")
            apply_quote_verification_with_retry(
                payload,
                fields=quote_checks(result),
                retried=True,
            )
        return response_model.model_validate(payload)
