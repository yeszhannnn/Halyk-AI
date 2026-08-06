from __future__ import annotations

import asyncio
import base64
import hashlib
import json
import random
import re
from dataclasses import dataclass, field
from decimal import Decimal
from pathlib import Path
from typing import Any, Callable, TypeVar

import instructor
from instructor.core import InstructorRetryException, ResponseParsingError
from openai import APIConnectionError, APITimeoutError, AsyncOpenAI, RateLimitError
from pydantic import BaseModel, ValidationError as PydanticValidationError

from agent.config import BUDGET_USD, MAX_CONCURRENT, MODEL_ID, OPENAI_SEED, TEMPERATURE

T = TypeVar("T", bound=BaseModel)

CACHE_DIR = Path(".cache/llm")
MAX_TRANSPORT_RETRIES = 5
MAX_VALIDATION_RETRIES = 1
_RETRY_BODY_HINT = re.compile(r"try again in (\d+(?:\.\d+)?)\s*s", re.IGNORECASE)

MODEL_PRICING_PER_MILLION: dict[str, tuple[Decimal, Decimal]] = {
    "gpt-4o": (Decimal("2.50"), Decimal("10.00")),
    "gpt-4o-mini": (Decimal("0.15"), Decimal("0.60")),
}


class BudgetExceededError(RuntimeError):
    pass


class LLMValidationError(RuntimeError):
    """Raised when the model output cannot be parsed into the response schema."""

    def __init__(self, message: str, *, raw_output: Any = None) -> None:
        super().__init__(message)
        self.raw_output = raw_output


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


def _cache_key(
    *,
    model: str,
    messages: list[dict[str, Any]],
    params: dict[str, Any],
    image_digests: list[str] | None = None,
) -> str:
    payload = {
        "model": model,
        "messages": messages,
        "params": params,
        "image_digests": image_digests or [],
    }
    digest = hashlib.sha256(
        json.dumps(payload, sort_keys=True, ensure_ascii=False).encode("utf-8"),
    )
    return digest.hexdigest()


def _image_digest(image_path: Path) -> str:
    return hashlib.sha256(image_path.read_bytes()).hexdigest()


def _encode_image_data_url(image_path: Path) -> str:
    suffix = image_path.suffix.lower()
    media_type = {
        ".png": "image/png",
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".webp": "image/webp",
    }.get(suffix, "image/png")
    encoded = base64.b64encode(image_path.read_bytes()).decode("ascii")
    return f"data:{media_type};base64,{encoded}"


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
    response = getattr(exc, "response", None)
    if response is not None:
        header_value = response.headers.get("retry-after")
        if header_value is not None:
            try:
                return float(header_value)
            except ValueError:
                pass

    for chunk in (str(exc), str(getattr(exc, "body", ""))):
        match = _RETRY_BODY_HINT.search(chunk)
        if match:
            return float(match.group(1))
    return None


def _is_transport_error(exc: Exception) -> bool:
    return isinstance(exc, (RateLimitError, APIConnectionError, APITimeoutError))


def _is_validation_error(exc: Exception) -> bool:
    return isinstance(
        exc,
        (
            InstructorRetryException,
            PydanticValidationError,
            ResponseParsingError,
            LLMValidationError,
        ),
    )


def _format_validation_failure(exc: Exception) -> tuple[str, Any]:
    if isinstance(exc, InstructorRetryException):
        raw = exc.last_completion
        lines = [f"validation failed after {exc.n_attempts} attempt(s)"]
        for attempt in exc.failed_attempts:
            lines.append(f"  attempt {attempt.attempt_number}: {attempt.exception}")
        if raw is not None:
            lines.append(f"raw model output: {raw}")
        return "\n".join(lines), raw
    return str(exc), None


async def _sleep_with_backoff(attempt: int, exc: Exception | None = None) -> None:
    if isinstance(exc, RateLimitError):
        retry_after = _retry_after_seconds(exc)
        if retry_after is not None:
            await asyncio.sleep(retry_after + random.uniform(0, 0.25))
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
        self._client = instructor.from_openai(AsyncOpenAI(max_retries=0))

    async def _request_with_retries(
        self,
        *,
        response_model: type[T],
        messages: list[dict[str, Any]],
        request_params: dict[str, Any],
    ) -> tuple[T, Any]:
        last_exc: Exception | None = None
        max_transport_attempts = MAX_TRANSPORT_RETRIES + 1
        for transport_attempt in range(max_transport_attempts):
            try:
                async with self._semaphore:
                    _ensure_budget()
                    return await self._client.chat.completions.create_with_completion(
                        model=self.model,
                        messages=messages,
                        response_model=response_model,
                        max_retries=MAX_VALIDATION_RETRIES,
                        **request_params,
                    )
            except BudgetExceededError:
                raise
            except Exception as exc:
                last_exc = exc
                if _is_validation_error(exc):
                    message, raw = _format_validation_failure(exc)
                    raise LLMValidationError(message, raw_output=raw) from exc
                if transport_attempt >= MAX_TRANSPORT_RETRIES or not _is_transport_error(exc):
                    break
                await _sleep_with_backoff(transport_attempt, exc)

        raise RuntimeError(
            f"LLM transport request failed after {max_transport_attempts} attempts",
        ) from last_exc

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

        parsed, completion = await self._request_with_retries(
            response_model=response_model,
            messages=messages,
            request_params=request_params,
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

    async def complete_vision(
        self,
        *,
        response_model: type[T],
        prompt: str,
        image_paths: list[Path],
        system_prompt: str | None = None,
        use_cache: bool = True,
        **params: Any,
    ) -> T:
        content: list[dict[str, Any]] = [{"type": "text", "text": prompt}]
        for image_path in image_paths:
            content.append(
                {
                    "type": "image_url",
                    "image_url": {"url": _encode_image_data_url(image_path)},
                },
            )

        messages: list[dict[str, Any]] = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": content})

        request_params = {
            "temperature": TEMPERATURE,
            "seed": OPENAI_SEED,
            **params,
        }
        serializable_messages = json.loads(json.dumps(messages, ensure_ascii=False))
        image_digests = [_image_digest(path) for path in image_paths]
        cache_key = _cache_key(
            model=self.model,
            messages=serializable_messages,
            params=request_params,
            image_digests=image_digests,
        )

        if use_cache:
            cached = _read_cache(cache_key, response_model, self.counter)
            if cached is not None:
                return cached

        self.counter.cache_misses += 1
        _ensure_budget()

        parsed, completion = await self._request_with_retries(
            response_model=response_model,
            messages=messages,
            request_params=request_params,
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
