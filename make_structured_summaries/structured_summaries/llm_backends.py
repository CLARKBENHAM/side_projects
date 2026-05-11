"""Thin wrappers around local LLM CLIs."""

from __future__ import annotations

import asyncio
import subprocess
from collections import deque
from contextlib import asynccontextmanager
from dataclasses import dataclass
from datetime import datetime
from time import monotonic


class LLMExecutionError(RuntimeError):
    """Raised when a backend CLI fails."""


DEFAULT_MODEL_BY_BACKEND: dict[str, str | None] = {
    "gemini": "gemini-3.1-pro-preview",
    "claude": "sonnet",
    "stub": None,
}


@dataclass(frozen=True)
class PromptExecution:
    output: str
    command: tuple[str, ...]
    started_at: str
    finished_at: str
    duration_seconds: float


class AsyncRequestLimiter:
    """Limits concurrent prompt starts and request start rate."""

    def __init__(
        self,
        *,
        max_concurrency: int = 2,
        requests_per_window: int | None = None,
        window_seconds: float = 60.0,
    ) -> None:
        if max_concurrency < 1:
            raise ValueError("max_concurrency must be at least 1")
        if requests_per_window is not None and requests_per_window < 1:
            raise ValueError("requests_per_window must be at least 1 when provided")
        if window_seconds <= 0:
            raise ValueError("window_seconds must be positive")
        self._semaphore = asyncio.Semaphore(max_concurrency)
        self._start_lock = asyncio.Lock()
        self._started_at: deque[float] = deque()
        self.requests_per_window = requests_per_window
        self.window_seconds = window_seconds

    async def _wait_for_start_slot(self) -> None:
        if self.requests_per_window is None:
            return
        while True:
            async with self._start_lock:
                now = monotonic()
                while (
                    self._started_at
                    and now - self._started_at[0] >= self.window_seconds
                ):
                    self._started_at.popleft()
                if len(self._started_at) < self.requests_per_window:
                    self._started_at.append(now)
                    return
                wait_seconds = self.window_seconds - (now - self._started_at[0])
            await asyncio.sleep(max(wait_seconds, 0.01))

    @asynccontextmanager
    async def limit(self) -> object:
        await self._semaphore.acquire()
        try:
            await self._wait_for_start_slot()
            yield
        finally:
            self._semaphore.release()


def default_model_for_backend(backend: str) -> str | None:
    return DEFAULT_MODEL_BY_BACKEND.get(backend.lower())


def _build_command(
    backend: str,
    prompt: str,
    *,
    model: str | None = None,
    system_prompt: str | None = None,
) -> list[str]:
    backend = backend.lower()
    if backend == "stub":
        return []
    if backend == "gemini":
        command = ["gemini"]
        if model:
            command.extend(["-m", model])
        command.extend(["-p", prompt, "--raw-output", "--accept-raw-output-risk"])
    elif backend == "claude":
        command = ["claude", "-p", "--output-format", "text"]
        if model:
            command.extend(["--model", model])
        if system_prompt:
            command.extend(["--system-prompt", system_prompt])
        command.append(prompt)
    else:
        raise ValueError(f"Unsupported backend: {backend}")
    return command


def run_prompt_with_metadata(
    backend: str,
    prompt: str,
    *,
    model: str | None = None,
    system_prompt: str | None = None,
    timeout: int = 900,
) -> PromptExecution:
    if backend.lower() == "stub":
        now = datetime.now().isoformat()
        return PromptExecution(
            output=prompt,
            command=(),
            started_at=now,
            finished_at=now,
            duration_seconds=0.0,
        )

    command = _build_command(
        backend,
        prompt,
        model=model,
        system_prompt=system_prompt,
    )
    started_wall = datetime.now().isoformat()
    started_monotonic = monotonic()
    try:
        result = subprocess.run(
            command,
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
        )
    except subprocess.TimeoutExpired as exc:
        raise LLMExecutionError(f"{backend} timed out after {timeout} seconds") from exc
    duration_seconds = monotonic() - started_monotonic
    finished_wall = datetime.now().isoformat()
    if result.returncode != 0:
        stderr = result.stderr.strip()
        raise LLMExecutionError(stderr or f"{backend} exited with {result.returncode}")
    return PromptExecution(
        output=result.stdout.strip(),
        command=tuple(command),
        started_at=started_wall,
        finished_at=finished_wall,
        duration_seconds=duration_seconds,
    )


def run_prompt(
    backend: str,
    prompt: str,
    *,
    model: str | None = None,
    system_prompt: str | None = None,
    timeout: int = 900,
) -> str:
    return run_prompt_with_metadata(
        backend,
        prompt,
        model=model,
        system_prompt=system_prompt,
        timeout=timeout,
    ).output


async def run_prompt_async(
    backend: str,
    prompt: str,
    *,
    model: str | None = None,
    system_prompt: str | None = None,
    timeout: int = 900,
    limiter: AsyncRequestLimiter | None = None,
) -> PromptExecution:
    if backend.lower() == "stub":
        now = datetime.now().isoformat()
        return PromptExecution(
            output=prompt,
            command=(),
            started_at=now,
            finished_at=now,
            duration_seconds=0.0,
        )

    command = _build_command(
        backend,
        prompt,
        model=model,
        system_prompt=system_prompt,
    )

    @asynccontextmanager
    async def _no_limit() -> object:
        yield

    async with limiter.limit() if limiter else _no_limit():
        started_wall = datetime.now().isoformat()
        started_monotonic = monotonic()
        process = await asyncio.create_subprocess_exec(
            *command,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        try:
            stdout, stderr = await asyncio.wait_for(
                process.communicate(), timeout=timeout
            )
        except TimeoutError as exc:
            process.kill()
            await process.wait()
            raise LLMExecutionError(
                f"{backend} timed out after {timeout} seconds"
            ) from exc
        duration_seconds = monotonic() - started_monotonic
        finished_wall = datetime.now().isoformat()
        if process.returncode != 0:
            decoded_stderr = stderr.decode("utf-8", errors="ignore").strip()
            raise LLMExecutionError(
                decoded_stderr or f"{backend} exited with {process.returncode}"
            )
        return PromptExecution(
            output=stdout.decode("utf-8", errors="ignore").strip(),
            command=tuple(command),
            started_at=started_wall,
            finished_at=finished_wall,
            duration_seconds=duration_seconds,
        )
