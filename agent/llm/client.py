from __future__ import annotations

import asyncio
import base64
import hashlib
import json
import logging
import random
import re
from dataclasses import dataclass, field
from decimal import Decimal
from pathlib import Path
from typing import Any, Callable, TypeVar

import instructor
from instructor import Mode
from instructor.core import InstructorRetryException, ResponseParsingError
from instructor.v2.providers.anthropic import from_anthropic
from anthropic import APIConnectionError as AnthropicAPIConnectionError
from anthropic import APITimeoutError as AnthropicAPITimeoutError
from anthropic import AsyncAnthropic
from anthropic import RateLimitError as AnthropicRateLimitError
from openai import APIConnectionError, APITimeoutError, AsyncOpenAI, RateLimitError
from pydantic import BaseModel, ValidationError as PydanticValidationError

from agent.config import (
    ANTHROPIC_MAX_TOKENS,
    BUDGET_USD,
    LLM_PROVIDER,
    MAX_CONCURRENT,
    MODEL_ID,
    OPENAI_MODEL_ID,
    OPENAI_SEED,
    TEMPERATURE,
)
from agent.llm.token_bucket import get_token_bucket
from agent.parsing.numbers import exception_mentions_absent_sentinel

T = TypeVar("T", bound=BaseModel)

logger = logging.getLogger(__name__)

CACHE_DIR = Path(".cache/llm")
MAX_TRANSPORT_RETRIES = 5
MAX_VALIDATION_RETRIES = 2
MAX_REQUEST_ATTEMPTS = MAX_TRANSPORT_RETRIES + MAX_VALIDATION_RETRIES + 1
_COMPLETION_TOKEN_ESTIMATE = 500
_RETRY_BODY_HINT_SECONDS = re.compile(
    r"try again in (\d+(?:\.\d+)?)\s*s",
    re.IGNORECASE,
)
_RETRY_BODY_HINT_MILLISECONDS = re.compile(
    r"try again in (\d+(?:\.\d+)?)\s*ms",
    re.IGNORECASE,
)

MODEL_PRICING_PER_MILLION: dict[str, tuple[Decimal, Decimal]] = {
    "gpt-4o": (Decimal("2.50"), Decimal("10.00")),
    "gpt-4o-mini": (Decimal("0.15"), Decimal("0.60")),
    OPENAI_MODEL_ID: (Decimal("0.20"), Decimal("1.20")),
    "claude-haiku-4-5-20251001": (Decimal("1.00"), Decimal("5.00")),
}

_TRANSPORT_ERRORS = (
    RateLimitError,
    APIConnectionError,
    APITimeoutError,
    AnthropicRateLimitError,
    AnthropicAPIConnectionError,
    AnthropicAPITimeoutError,
)
_RATE_LIMIT_ERRORS = (RateLimitError, AnthropicRateLimitError)


class BudgetExceededError(RuntimeError):
    pass


class LLMValidationError(RuntimeError):
    """Raised when the model output cannot be parsed into the response schema."""

    def __init__(self, message: str, *, raw_output: Any = None) -> None:
        super().__init__(message)
        self.raw_output = raw_output


class LLMTransportExhaustedError(RuntimeError):
    """Raised when transport retries are exhausted for a single LLM request."""

    def __init__(self, message: str, *, attempts: int, cause: Exception | None = None) -> None:
        super().__init__(message)
        self.attempts = attempts
        self.cause = cause


@dataclass
class RunCounter:
    prompt_tokens: int = 0
    completion_tokens: int = 0
    reasoning_tokens: int = 0
    total_tokens: int = 0
    cost_usd: Decimal = field(default_factory=lambda: Decimal("0"))
    cache_hits: int = 0
    cache_misses: int = 0
    api_calls: int = 0

    def to_dict(self) -> dict[str, Any]:
        stats: dict[str, Any] = {
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "total_tokens": self.total_tokens,
            "cost_usd": str(self.cost_usd.quantize(Decimal("0.000001"))),
            "cache_hits": self.cache_hits,
            "cache_misses": self.cache_misses,
            "api_calls": self.api_calls,
        }
        if self.reasoning_tokens:
            stats["reasoning_tokens"] = self.reasoning_tokens
        return stats


RUN_COUNTER = RunCounter()

REPLAY_DIR: Path | None = None
RECORD_DIR: Path | None = None


class ReplayMissError(RuntimeError):
    """Raised in replay mode when no stored response exists for a prompt hash."""


def set_replay_dir(path: Path | None) -> None:
    global REPLAY_DIR
    REPLAY_DIR = path


def set_record_dir(path: Path | None) -> None:
    global RECORD_DIR
    RECORD_DIR = path


def _fixture_path(cache_key: str, root: Path) -> Path:
    return root / f"{cache_key}.json"


def _read_fixture(
    cache_key: str,
    response_model: type[T],
    counter: RunCounter,
    *,
    root: Path,
) -> T:
    path = _fixture_path(cache_key, root)
    if not path.is_file():
        raise ReplayMissError(f"no replay fixture for prompt hash {cache_key}")
    cached = json.loads(path.read_text(encoding="utf-8"))
    counter.cache_hits += 1
    usage = cached.get("usage", {})
    counter.prompt_tokens += int(usage.get("prompt_tokens", 0))
    counter.completion_tokens += int(usage.get("completion_tokens", 0))
    counter.reasoning_tokens += int(usage.get("reasoning_tokens", 0))
    counter.total_tokens += int(usage.get("total_tokens", 0))
    counter.cost_usd += Decimal(str(cached.get("cost_usd", "0")))
    return response_model.model_validate(cached["response"])


def _write_fixture(
    cache_key: str,
    *,
    response: BaseModel,
    usage: dict[str, int],
    cost_usd: Decimal,
    root: Path,
) -> None:
    root.mkdir(parents=True, exist_ok=True)
    payload = {
        "response": response.model_dump(mode="json"),
        "usage": usage,
        "cost_usd": str(cost_usd),
    }
    _fixture_path(cache_key, root).write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


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
    provider: str,
    model: str,
    messages: list[dict[str, Any]],
    params: dict[str, Any],
    image_digests: list[str] | None = None,
) -> str:
    payload = {
        "provider": provider,
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


def _image_media_type(image_path: Path) -> str:
    suffix = image_path.suffix.lower()
    return {
        ".png": "image/png",
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".webp": "image/webp",
    }.get(suffix, "image/png")


def vision_image_block(image_path: Path) -> dict[str, Any]:
    """Return the provider-specific vision content block for one image."""
    media_type = _image_media_type(image_path)
    encoded = base64.b64encode(image_path.read_bytes()).decode("ascii")
    if LLM_PROVIDER == "anthropic":
        return {
            "type": "image",
            "source": {
                "type": "base64",
                "media_type": media_type,
                "data": encoded,
            },
        }
    return {
        "type": "image_url",
        "image_url": {"url": f"data:{media_type};base64,{encoded}"},
    }


def _build_request_params(**params: Any) -> dict[str, Any]:
    request_params = {
        "temperature": TEMPERATURE,
        **params,
    }
    if LLM_PROVIDER == "openai":
        request_params["seed"] = OPENAI_SEED
    else:
        request_params.setdefault("max_tokens", ANTHROPIC_MAX_TOKENS)
    return request_params


def _read_token_field(usage: Any, *field_names: str) -> int | None:
    for field_name in field_names:
        if not hasattr(usage, field_name):
            continue
        value = getattr(usage, field_name)
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            return int(value)
    return None


def _read_reasoning_tokens(usage: Any) -> int:
    direct = _read_token_field(usage, "reasoning_tokens")
    if direct is not None:
        return direct
    details = getattr(usage, "completion_tokens_details", None)
    if details is not None:
        return _read_token_field(details, "reasoning_tokens") or 0
    return 0


def _usage_fields(usage: Any) -> dict[str, int]:
    prompt_tokens = _read_token_field(usage, "prompt_tokens", "input_tokens") or 0
    completion_tokens = _read_token_field(usage, "completion_tokens", "output_tokens") or 0
    reasoning_tokens = _read_reasoning_tokens(usage)
    total_tokens = _read_token_field(usage, "total_tokens") or (prompt_tokens + completion_tokens)
    fields = {
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": total_tokens,
    }
    if reasoning_tokens:
        fields["reasoning_tokens"] = reasoning_tokens
    return fields


def _cache_path(cache_key: str) -> Path:
    return CACHE_DIR / f"{cache_key}.json"


def _read_cache(cache_key: str, response_model: type[T], counter: RunCounter) -> T | None:
    path = _cache_path(cache_key)
    if not path.is_file():
        return None
    cached = json.loads(path.read_text(encoding="utf-8"))
    counter.cache_hits += 1
    usage = cached.get("usage", {})
    counter.prompt_tokens += int(usage.get("prompt_tokens", 0))
    counter.completion_tokens += int(usage.get("completion_tokens", 0))
    counter.reasoning_tokens += int(usage.get("reasoning_tokens", 0))
    counter.total_tokens += int(usage.get("total_tokens", 0))
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
    usage_fields = _usage_fields(usage)
    prompt_tokens = usage_fields["prompt_tokens"]
    completion_tokens = usage_fields["completion_tokens"]
    reasoning_tokens = usage_fields.get("reasoning_tokens", 0)
    total_tokens = usage_fields["total_tokens"]
    cost = _estimate_cost(model, prompt_tokens, completion_tokens)

    counter.prompt_tokens += prompt_tokens
    counter.completion_tokens += completion_tokens
    counter.reasoning_tokens += reasoning_tokens
    counter.total_tokens += total_tokens
    counter.cost_usd += cost
    counter.api_calls += 1
    return cost


def _ensure_budget() -> None:
    if RUN_COUNTER.cost_usd >= BUDGET_USD:
        raise BudgetExceededError(
            f"LLM budget exceeded: spent {RUN_COUNTER.cost_usd} >= cap {BUDGET_USD}",
        )


def _iter_exception_chain(exc: BaseException) -> list[BaseException]:
    chain: list[BaseException] = []
    seen: set[int] = set()
    current: BaseException | None = exc
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        chain.append(current)
        current = current.__cause__ or current.__context__
    return chain


def _root_cause(exc: BaseException) -> BaseException:
    chain = _iter_exception_chain(exc)
    return chain[-1] if chain else exc


def _find_in_chain(exc: BaseException, predicate: Callable[[BaseException], bool]) -> BaseException | None:
    for item in _iter_exception_chain(exc):
        if predicate(item):
            return item
    return None


def _estimate_request_tokens(messages: list[dict[str, Any]]) -> int:
    serialized = json.dumps(messages, ensure_ascii=False)
    return max(1, len(serialized) // 4) + _COMPLETION_TOKEN_ESTIMATE


def _retry_after_seconds(exc: Exception) -> float | None:
    rate_limit = _find_in_chain(exc, lambda item: isinstance(item, _RATE_LIMIT_ERRORS))
    if rate_limit is None:
        return None

    response = getattr(rate_limit, "response", None)
    if response is not None:
        header_value = response.headers.get("retry-after")
        if header_value is not None:
            try:
                return float(header_value)
            except ValueError:
                pass

    for chunk in (str(rate_limit), str(getattr(rate_limit, "body", ""))):
        match = _RETRY_BODY_HINT_MILLISECONDS.search(chunk)
        if match:
            return float(match.group(1)) / 1000.0
        match = _RETRY_BODY_HINT_SECONDS.search(chunk)
        if match:
            return float(match.group(1))
    return None


def _is_transport_error(exc: Exception) -> bool:
    return _find_in_chain(
        exc,
        lambda item: isinstance(item, _TRANSPORT_ERRORS),
    ) is not None


def _is_rate_limited(exc: Exception) -> bool:
    if _find_in_chain(exc, lambda item: isinstance(item, _RATE_LIMIT_ERRORS)) is not None:
        return True
    message = " ".join(str(item) for item in _iter_exception_chain(exc)).casefold()
    return "rate limit" in message or "429" in message


def _is_validation_error(exc: Exception) -> bool:
    if _is_transport_error(exc) or _is_rate_limited(exc):
        return False
    return _find_in_chain(
        exc,
        lambda item: isinstance(
            item,
            (
                InstructorRetryException,
                PydanticValidationError,
                ResponseParsingError,
                LLMValidationError,
            ),
        ),
    ) is not None


def _format_validation_failure(exc: Exception) -> tuple[str, Any]:
    root = _root_cause(exc)
    if isinstance(root, PydanticValidationError):
        return str(root), getattr(exc, "last_completion", None) if isinstance(exc, InstructorRetryException) else None
    if isinstance(exc, InstructorRetryException):
        raw = exc.last_completion
        for attempt in exc.failed_attempts:
            cause = attempt.exception
            if isinstance(cause, PydanticValidationError):
                return str(cause), raw
            nested = _find_in_chain(cause, lambda item: isinstance(item, PydanticValidationError))
            if nested is not None:
                return str(nested), raw
        if raw is not None:
            return f"schema validation failed: {root}", raw
        return f"schema validation failed: {root}", None
    return str(root), None


async def _sleep_with_backoff(attempt: int, exc: Exception | None = None) -> None:
    if exc is not None and _is_rate_limited(exc):
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
        self.provider = LLM_PROVIDER
        self.model = model or MODEL_ID
        if self.provider == "anthropic":
            self._sdk_client = AsyncAnthropic(max_retries=0)
            self._client = from_anthropic(self._sdk_client, mode=Mode.TOOLS)
        else:
            self._sdk_client = AsyncOpenAI(max_retries=0)
            self._client = instructor.from_openai(self._sdk_client)
        logging.getLogger("instructor").setLevel(logging.CRITICAL)

    async def aclose(self) -> None:
        await self._sdk_client.close()

    async def __aenter__(self) -> LLMClient:
        return self

    async def __aexit__(self, *exc: object) -> None:
        await self.aclose()

    async def _request_with_retries(
        self,
        *,
        response_model: type[T],
        messages: list[dict[str, Any]],
        request_params: dict[str, Any],
    ) -> tuple[T, Any]:
        last_exc: Exception | None = None
        estimated_tokens = _estimate_request_tokens(messages)

        for attempt in range(1, MAX_REQUEST_ATTEMPTS + 1):
            try:
                await get_token_bucket().acquire(estimated_tokens)
                async with self._semaphore:
                    _ensure_budget()
                    return await self._client.create_with_completion(
                        model=self.model,
                        messages=messages,
                        response_model=response_model,
                        max_retries=0,
                        **request_params,
                    )
            except BudgetExceededError:
                raise
            except Exception as exc:
                last_exc = exc
                validation_error = _is_validation_error(exc)
                transport_error = _is_transport_error(exc) or _is_rate_limited(exc)
                retryable = validation_error or transport_error
                # Retrying an absence sentinel never changes the answer: allow one retry.
                sentinel_error = validation_error and exception_mentions_absent_sentinel(exc)
                max_attempts = 2 if sentinel_error else MAX_REQUEST_ATTEMPTS
                if not retryable or attempt >= max_attempts:
                    if validation_error:
                        message, raw = _format_validation_failure(exc)
                        raise LLMValidationError(message, raw_output=raw) from exc
                    raise LLMTransportExhaustedError(
                        f"LLM transport request failed after {attempt} attempt(s)",
                        attempts=attempt,
                        cause=exc,
                    ) from exc

                error_kind = "validation" if validation_error else "transport"
                logger.warning(
                    "LLM attempt %d/%d failed (%s): %s",
                    attempt,
                    max_attempts,
                    error_kind,
                    _root_cause(exc),
                )
                if transport_error:
                    await _sleep_with_backoff(attempt - 1, exc)

        raise LLMTransportExhaustedError(
            f"LLM request failed after {MAX_REQUEST_ATTEMPTS} attempts",
            attempts=MAX_REQUEST_ATTEMPTS,
            cause=last_exc,
        ) from last_exc

    async def complete(
        self,
        *,
        response_model: type[T],
        messages: list[dict[str, Any]],
        use_cache: bool = True,
        validation_context: dict[str, Any] | None = None,
        **params: Any,
    ) -> T:
        request_params = _build_request_params(**params)
        serializable_messages = json.loads(json.dumps(messages, ensure_ascii=False))
        cache_key = _cache_key(
            provider=self.provider,
            model=self.model,
            messages=serializable_messages,
            params=request_params,
        )

        if REPLAY_DIR is not None:
            return _read_fixture(cache_key, response_model, self.counter, root=REPLAY_DIR)

        if use_cache and RECORD_DIR is None:
            cached = _read_cache(cache_key, response_model, self.counter)
            if cached is not None:
                return cached

        self.counter.cache_misses += 1
        if REPLAY_DIR is None:
            _ensure_budget()

        parsed, completion = await self._request_with_retries(
            response_model=response_model,
            messages=messages,
            request_params=request_params,
        )
        usage = getattr(completion, "usage", None)
        usage_dict = _usage_fields(usage)
        cost = _record_usage(self.model, usage, self.counter)
        if RECORD_DIR is not None:
            _write_fixture(
                cache_key,
                response=parsed,
                usage=usage_dict,
                cost_usd=cost,
                root=RECORD_DIR,
            )
        if use_cache:
            _write_cache(
                cache_key,
                response=parsed,
                usage=usage_dict,
                cost_usd=cost,
            )
        if validation_context is not None:
            parsed = response_model.model_validate(
                parsed.model_dump(mode="python"),
                context=validation_context,
            )
        return parsed

    async def complete_verified(
        self,
        *,
        response_model: type[T],
        messages: list[dict[str, Any]],
        quote_checks: Callable[[T], list[tuple[str, str, str]]],
        validation_context: dict[str, Any] | None = None,
        **params: Any,
    ) -> T:
        from agent.evidence.quotes import apply_quote_verification_with_retry

        result = await self.complete(
            response_model=response_model,
            messages=messages,
            validation_context=validation_context,
            **params,
        )
        if REPLAY_DIR is not None:
            return result

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
                validation_context=validation_context,
                **params,
            )
            payload = result.model_dump(mode="python")
            apply_quote_verification_with_retry(
                payload,
                fields=quote_checks(result),
                retried=True,
            )
        return response_model.model_validate(
            payload,
            context=validation_context,
        )

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
            content.append(vision_image_block(image_path))

        messages: list[dict[str, Any]] = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": content})

        request_params = _build_request_params(**params)
        serializable_messages = json.loads(json.dumps(messages, ensure_ascii=False))
        image_digests = [_image_digest(path) for path in image_paths]
        cache_key = _cache_key(
            provider=self.provider,
            model=self.model,
            messages=serializable_messages,
            params=request_params,
            image_digests=image_digests,
        )

        if REPLAY_DIR is not None:
            return _read_fixture(cache_key, response_model, self.counter, root=REPLAY_DIR)

        if use_cache and RECORD_DIR is None:
            cached = _read_cache(cache_key, response_model, self.counter)
            if cached is not None:
                return cached

        self.counter.cache_misses += 1
        if REPLAY_DIR is None:
            _ensure_budget()

        parsed, completion = await self._request_with_retries(
            response_model=response_model,
            messages=messages,
            request_params=request_params,
        )
        usage = getattr(completion, "usage", None)
        usage_dict = _usage_fields(usage)
        cost = _record_usage(self.model, usage, self.counter)
        if RECORD_DIR is not None:
            _write_fixture(
                cache_key,
                response=parsed,
                usage=usage_dict,
                cost_usd=cost,
                root=RECORD_DIR,
            )
        if use_cache:
            _write_cache(
                cache_key,
                response=parsed,
                usage=usage_dict,
                cost_usd=cost,
            )
        return parsed
