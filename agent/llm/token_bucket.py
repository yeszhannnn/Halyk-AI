from __future__ import annotations

import asyncio
import time
from collections import deque

from agent.config import TPM_LIMIT


class TokenBucket:
    """Sliding-window token budget to stay under a tokens-per-minute ceiling."""

    def __init__(self, tpm_limit: int, *, window_seconds: float = 60.0) -> None:
        self.tpm_limit = tpm_limit
        self.window_seconds = window_seconds
        self._lock = asyncio.Lock()
        self._entries: deque[tuple[float, int]] = deque()

    def _prune(self, now: float) -> None:
        cutoff = now - self.window_seconds
        while self._entries and self._entries[0][0] <= cutoff:
            self._entries.popleft()

    def _used_tokens(self) -> int:
        return sum(tokens for _, tokens in self._entries)

    async def acquire(self, tokens: int) -> None:
        if tokens <= 0:
            return
        while True:
            async with self._lock:
                now = time.monotonic()
                self._prune(now)
                used = self._used_tokens()
                if used + tokens <= self.tpm_limit:
                    self._entries.append((now, tokens))
                    return
                if not self._entries:
                    self._entries.append((now, tokens))
                    return
                wait_seconds = self.window_seconds - (now - self._entries[0][0]) + 0.05
            await asyncio.sleep(max(wait_seconds, 0.05))


_BUCKET: TokenBucket | None = None


def get_token_bucket() -> TokenBucket:
    global _BUCKET
    if _BUCKET is None:
        _BUCKET = TokenBucket(TPM_LIMIT)
    return _BUCKET
