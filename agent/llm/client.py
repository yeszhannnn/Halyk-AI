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
from instructor.core import InstructorRetryException, ResponseParsingError
from openai import APIConnectionError, APITimeoutError, AsyncOpenAI, RateLimitError
from pydantic import BaseModel, ValidationError as PydanticValidationError

from agent.config import BUDGET_USD, MAX_CONCURRENT, MODEL_ID, OPENAI_SEED, TEMPERATURE
from agent.llm.token_bucket import get_token_bucket

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
}


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
    counter.prompt_tokens += int(cached.get("usage", {}).get("prompt_tokens", 0))
    counter.completion_tokens += int(cached.get("usage", {}).get("completion_tokens", 0))
    counter.total_tokens += int(cached.get("usage", {}).get("total_tokens", 0))
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
    rate_limit = _find_in_chain(exc, lambda item: isinstance(item, RateLimitError))
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
        lambda item: isinstance(item, (RateLimitError, APIConnectionError, APITimeoutError)),
    ) is not None


def _is_rate_limited(exc: Exception) -> bool:
    if _find_in_chain(exc, lambda item: isinstance(item, RateLimitError)) is not None:
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
        self.model = model or MODEL_ID
        self._openai = AsyncOpenAI(max_retries=0)
        self._client = instructor.from_openai(self._openai)
        logging.getLogger("instructor").setLevel(logging.CRITICAL)

    async def aclose(self) -> None:
        await self._openai.close()

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
                    return await self._client.chat.completions.create_with_completion(
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
                if not retryable or attempt >= MAX_REQUEST_ATTEMPTS:
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
                    MAX_REQUEST_ATTEMPTS,
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
        usage_dict = {
            "prompt_tokens": int(getattr(usage, "prompt_tokens", 0) or 0),
            "completion_tokens": int(getattr(usage, "completion_tokens", 0) or 0),
            "total_tokens": int(getattr(usage, "total_tokens", 0) or 0),
        }
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
        usage_dict = {
            "prompt_tokens": int(getattr(usage, "prompt_tokens", 0) or 0),
            "completion_tokens": int(getattr(usage, "completion_tokens", 0) or 0),
            "total_tokens": int(getattr(usage, "total_tokens", 0) or 0),
        }
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
