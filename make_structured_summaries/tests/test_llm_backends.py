from __future__ import annotations

import asyncio
from time import monotonic

from structured_summaries.llm_backends import (
    AsyncRequestLimiter,
    default_model_for_backend,
)


def test_default_model_for_backend_uses_current_gemini_pro() -> None:
    assert default_model_for_backend("gemini") == "gemini-3.1-pro-preview"
    assert default_model_for_backend("claude") == "sonnet"


async def _measure_peak_concurrency(
    limiter: AsyncRequestLimiter,
    *,
    task_count: int,
    sleep_seconds: float,
) -> int:
    active = 0
    peak = 0
    active_lock = asyncio.Lock()

    async def worker() -> None:
        nonlocal active, peak
        async with limiter.limit():
            async with active_lock:
                active += 1
                peak = max(peak, active)
            await asyncio.sleep(sleep_seconds)
            async with active_lock:
                active -= 1

    await asyncio.gather(*(worker() for _ in range(task_count)))
    return peak


def test_async_request_limiter_caps_peak_concurrency() -> None:
    limiter = AsyncRequestLimiter(max_concurrency=2)
    peak = asyncio.run(
        _measure_peak_concurrency(
            limiter,
            task_count=5,
            sleep_seconds=0.02,
        )
    )

    assert peak == 2


def test_async_request_limiter_respects_request_window() -> None:
    limiter = AsyncRequestLimiter(
        max_concurrency=3,
        requests_per_window=2,
        window_seconds=0.05,
    )
    starts: list[float] = []

    async def worker() -> None:
        async with limiter.limit():
            starts.append(monotonic())
            await asyncio.sleep(0.01)

    async def main() -> None:
        await asyncio.gather(*(worker() for _ in range(3)))

    asyncio.run(main())

    ordered_starts = sorted(starts)
    assert len(ordered_starts) == 3
    assert ordered_starts[2] - ordered_starts[0] >= 0.045
