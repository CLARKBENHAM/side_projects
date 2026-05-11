"""Summary orchestration."""

from __future__ import annotations

import asyncio
import json
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path

from .chunk_companion import render_expanded_chunk_notes
from .chunking import split_text_into_chunks
from .llm_backends import (
    AsyncRequestLimiter,
    PromptExecution,
    run_prompt_async,
    run_prompt_with_metadata,
)
from .models import BookRecord
from .prompts import (
    build_challenge_prompt,
    build_chunk_analysis_prompt,
    build_synthesis_prompt,
)
from .text_extraction import extract_book_text

PROJECT_ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class SummaryConfig:
    chunk_backend: str = "gemini"
    chunk_model: str | None = "gemini-3.1-pro-preview"
    synthesis_backend: str = "claude"
    synthesis_model: str | None = "sonnet"
    critique_backend: str | None = "claude"
    critique_model: str | None = "sonnet"
    chunk_chars: int = 28_000
    overlap_chars: int = 1_500
    max_chunks: int | None = None
    chunk_concurrency: int = 4
    llm_timeout_seconds: int = 900
    force: bool = False
    dry_run: bool = False


@dataclass(frozen=True)
class SummaryArtifacts:
    extracted_text_path: Path
    chunk_dir: Path
    expanded_notes_path: Path | None
    summary_path: Path | None
    critique_path: Path | None
    manifest_path: Path


def _now_iso() -> str:
    return datetime.now().isoformat()


def _runtime_dirs(project_root: Path) -> dict[str, Path]:
    data_dir = project_root / "data"
    extracted_dir = data_dir / "extracted_text"
    chunk_dir = data_dir / "chunk_notes"
    summary_dir = data_dir / "summaries"
    for directory in [data_dir, extracted_dir, chunk_dir, summary_dir]:
        directory.mkdir(parents=True, exist_ok=True)
    return {
        "data_dir": data_dir,
        "extracted_dir": extracted_dir,
        "chunk_dir": chunk_dir,
        "summary_dir": summary_dir,
    }


def _write_text(path: Path, text: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path


def _write_manifest(path: Path, manifest: dict[str, object]) -> None:
    _write_text(path, json.dumps(manifest, indent=2, ensure_ascii=False))


def _build_manifest(
    *,
    book: BookRecord,
    config: SummaryConfig,
    extracted_text_path: Path,
    chunk_dir: Path,
    expanded_notes_path: Path,
    summary_path: Path,
    critique_path: Path,
) -> dict[str, object]:
    return {
        "book": book.to_row(),
        "config": asdict(config),
        "timestamp_started": _now_iso(),
        "timestamp_finished": None,
        "status": "running",
        "current_stage": "setup",
        "chunk_count": 0,
        "stage_records": [],
        "paths": {
            "extracted_text_path": str(extracted_text_path),
            "chunk_dir": str(chunk_dir),
            "expanded_notes_path": str(expanded_notes_path),
            "summary_path": str(summary_path),
            "critique_path": str(critique_path),
        },
    }


def _start_stage(
    manifest: dict[str, object],
    manifest_path: Path,
    stage_name: str,
    *,
    metadata: dict[str, object] | None = None,
) -> dict[str, object]:
    stage = {
        "name": stage_name,
        "status": "running",
        "started_at": _now_iso(),
    }
    if metadata:
        stage["metadata"] = metadata
    manifest["current_stage"] = stage_name
    stage_records = manifest.setdefault("stage_records", [])
    if not isinstance(stage_records, list):
        raise TypeError("stage_records must be a list")
    stage_records.append(stage)
    _write_manifest(manifest_path, manifest)
    return stage


def _finish_stage(
    manifest: dict[str, object],
    manifest_path: Path,
    stage: dict[str, object],
    *,
    metadata: dict[str, object] | None = None,
) -> None:
    stage["status"] = "completed"
    stage["finished_at"] = _now_iso()
    if metadata:
        existing = stage.setdefault("metadata", {})
        if not isinstance(existing, dict):
            raise TypeError("stage metadata must be a dict")
        existing.update(metadata)
    _write_manifest(manifest_path, manifest)


def _fail_stage(
    manifest: dict[str, object],
    manifest_path: Path,
    stage: dict[str, object],
    *,
    error: str,
) -> None:
    stage["status"] = "failed"
    stage["finished_at"] = _now_iso()
    metadata = stage.setdefault("metadata", {})
    if not isinstance(metadata, dict):
        raise TypeError("stage metadata must be a dict")
    metadata["error"] = error
    manifest["status"] = "failed"
    manifest["timestamp_finished"] = _now_iso()
    _write_manifest(manifest_path, manifest)


def _complete_manifest(
    manifest: dict[str, object],
    manifest_path: Path,
    *,
    expanded_notes_path: Path | None,
    summary_path: Path | None,
    critique_path: Path | None,
) -> None:
    manifest["status"] = "completed"
    manifest["current_stage"] = "completed"
    manifest["timestamp_finished"] = _now_iso()
    paths = manifest.get("paths", {})
    if not isinstance(paths, dict):
        raise TypeError("paths must be a dict")
    paths["expanded_notes_path"] = str(expanded_notes_path or "")
    paths["summary_path"] = str(summary_path or "")
    paths["critique_path"] = str(critique_path or "")
    _write_manifest(manifest_path, manifest)


def _execution_metadata(
    prompt_path: Path,
    response_path: Path,
    execution: PromptExecution,
) -> dict[str, object]:
    return {
        "command": list(execution.command),
        "started_at": execution.started_at,
        "finished_at": execution.finished_at,
        "duration_seconds": round(execution.duration_seconds, 3),
        "prompt_path": str(prompt_path),
        "prompt_bytes": prompt_path.stat().st_size,
        "response_path": str(response_path),
        "response_bytes": response_path.stat().st_size,
    }


def _fail_running_stages(
    manifest: dict[str, object],
    manifest_path: Path,
    *,
    error: str,
) -> None:
    stage_records = manifest.get("stage_records", [])
    if not isinstance(stage_records, list):
        return
    mutated = False
    for stage in stage_records:
        if not isinstance(stage, dict) or stage.get("status") != "running":
            continue
        stage["status"] = "failed"
        stage["finished_at"] = _now_iso()
        metadata = stage.setdefault("metadata", {})
        if isinstance(metadata, dict):
            metadata.setdefault("error", error)
        mutated = True
    if not mutated:
        return
    manifest["status"] = "failed"
    manifest["timestamp_finished"] = _now_iso()
    _write_manifest(manifest_path, manifest)


async def _run_chunk_stage_async(
    *,
    book: BookRecord,
    chunk_text: str,
    chunk_index: int,
    total_chunks: int,
    chunk_dir: Path,
    config: SummaryConfig,
    manifest: dict[str, object],
    manifest_path: Path,
    manifest_lock: asyncio.Lock,
    chunk_semaphore: asyncio.Semaphore,
    limiter: AsyncRequestLimiter | None,
) -> str:
    prompt = build_chunk_analysis_prompt(
        book,
        chunk_text,
        chunk_index=chunk_index,
        total_chunks=total_chunks,
    )
    prompt_path = chunk_dir / f"chunk_{chunk_index:03d}_prompt.txt"
    _write_text(prompt_path, prompt)
    if config.dry_run:
        return ""

    async with chunk_semaphore:
        async with manifest_lock:
            stage = _start_stage(
                manifest,
                manifest_path,
                f"chunk_{chunk_index:03d}_llm",
                metadata={
                    "backend": config.chunk_backend,
                    "model": config.chunk_model or "",
                },
            )
        try:
            execution = await run_prompt_async(
                config.chunk_backend,
                prompt,
                model=config.chunk_model,
                timeout=config.llm_timeout_seconds,
                limiter=limiter,
            )
            response_path = chunk_dir / f"chunk_{chunk_index:03d}_response.txt"
            _write_text(response_path, execution.output)
            async with manifest_lock:
                _finish_stage(
                    manifest,
                    manifest_path,
                    stage,
                    metadata=_execution_metadata(prompt_path, response_path, execution),
                )
            return execution.output
        except asyncio.CancelledError:
            async with manifest_lock:
                _fail_stage(
                    manifest,
                    manifest_path,
                    stage,
                    error="cancelled due to sibling chunk failure",
                )
            raise
        except Exception as exc:
            async with manifest_lock:
                _fail_stage(
                    manifest,
                    manifest_path,
                    stage,
                    error=str(exc),
                )
            raise


def _prepare_run(
    book: BookRecord,
    *,
    project_root: Path,
    config: SummaryConfig,
) -> tuple[
    dict[str, Path],
    Path,
    Path,
    Path,
    Path,
    Path,
    Path,
    dict[str, object],
    list[str],
]:
    runtime = _runtime_dirs(project_root)
    extracted_text_path = runtime["extracted_dir"] / f"{book.book_id}.txt"
    chunk_dir = runtime["chunk_dir"] / book.book_id
    expanded_notes_path = runtime["summary_dir"] / f"{book.book_id}.expanded.md"
    summary_path = runtime["summary_dir"] / f"{book.book_id}.md"
    critique_path = runtime["summary_dir"] / f"{book.book_id}.critique.md"
    manifest_path = runtime["summary_dir"] / f"{book.book_id}.manifest.json"
    manifest = _build_manifest(
        book=book,
        config=config,
        extracted_text_path=extracted_text_path,
        chunk_dir=chunk_dir,
        expanded_notes_path=expanded_notes_path,
        summary_path=summary_path,
        critique_path=critique_path,
    )
    _write_manifest(manifest_path, manifest)

    extract_stage = _start_stage(
        manifest,
        manifest_path,
        "extract_text",
        metadata={"source_path": str(book.primary_path)},
    )
    try:
        if config.force or not extracted_text_path.exists():
            _write_text(extracted_text_path, extract_book_text(book))
        text = extracted_text_path.read_text(encoding="utf-8")
        chunks = split_text_into_chunks(
            text,
            target_chars=config.chunk_chars,
            overlap_chars=config.overlap_chars,
            max_chunks=config.max_chunks,
        )
        manifest["chunk_count"] = len(chunks)
        _finish_stage(
            manifest,
            manifest_path,
            extract_stage,
            metadata={
                "extracted_text_path": str(extracted_text_path),
                "char_count": len(text),
                "chunk_count": len(chunks),
            },
        )
    except Exception as exc:
        _fail_stage(
            manifest,
            manifest_path,
            extract_stage,
            error=str(exc),
        )
        raise
    return (
        runtime,
        extracted_text_path,
        chunk_dir,
        expanded_notes_path,
        summary_path,
        critique_path,
        manifest_path,
        manifest,
        chunks,
    )


def _final_artifacts(
    *,
    extracted_text_path: Path,
    chunk_dir: Path,
    expanded_notes_path: Path | None,
    summary_path: Path | None,
    critique_path: Path | None,
    manifest_path: Path,
) -> SummaryArtifacts:
    return SummaryArtifacts(
        extracted_text_path=extracted_text_path,
        chunk_dir=chunk_dir,
        expanded_notes_path=expanded_notes_path,
        summary_path=summary_path,
        critique_path=critique_path,
        manifest_path=manifest_path,
    )


def summarize_book(
    book: BookRecord,
    *,
    project_root: Path = PROJECT_ROOT,
    config: SummaryConfig | None = None,
) -> SummaryArtifacts:
    config = config or SummaryConfig()
    (
        _runtime,
        extracted_text_path,
        chunk_dir,
        expanded_notes_path,
        summary_path,
        critique_path,
        manifest_path,
        manifest,
        chunks,
    ) = _prepare_run(book, project_root=project_root, config=config)

    chunk_outputs: list[str] = []
    chunk_dir.mkdir(parents=True, exist_ok=True)
    try:
        for index, chunk in enumerate(chunks, start=1):
            prompt = build_chunk_analysis_prompt(
                book,
                chunk,
                chunk_index=index,
                total_chunks=len(chunks),
            )
            prompt_path = chunk_dir / f"chunk_{index:03d}_prompt.txt"
            _write_text(prompt_path, prompt)
            if config.dry_run:
                continue
            stage = _start_stage(
                manifest,
                manifest_path,
                f"chunk_{index:03d}_llm",
                metadata={
                    "backend": config.chunk_backend,
                    "model": config.chunk_model or "",
                },
            )
            execution = run_prompt_with_metadata(
                config.chunk_backend,
                prompt,
                model=config.chunk_model,
                timeout=config.llm_timeout_seconds,
            )
            response_path = chunk_dir / f"chunk_{index:03d}_response.txt"
            _write_text(response_path, execution.output)
            _finish_stage(
                manifest,
                manifest_path,
                stage,
                metadata=_execution_metadata(prompt_path, response_path, execution),
            )
            chunk_outputs.append(execution.output)

        expanded_notes_result_path: Path | None = None
        summary_result_path: Path | None = None
        critique_result_path: Path | None = None

        if config.dry_run:
            _write_text(
                chunk_dir / "NEXT_STEPS.txt",
                "Dry run completed. Chunk prompts were written. Re-run without --dry-run to call the configured backends.\n",
            )
        else:
            expanded_notes_stage = _start_stage(
                manifest,
                manifest_path,
                "expanded_notes_render",
                metadata={"chunk_count": len(chunk_outputs)},
            )
            expanded_notes_text = render_expanded_chunk_notes(book, chunk_outputs)
            expanded_notes_result_path = _write_text(
                expanded_notes_path,
                expanded_notes_text,
            )
            _finish_stage(
                manifest,
                manifest_path,
                expanded_notes_stage,
                metadata={
                    "expanded_notes_path": str(expanded_notes_result_path),
                    "char_count": len(expanded_notes_text),
                },
            )
            synthesis_prompt = build_synthesis_prompt(book, chunk_outputs)
            synthesis_prompt_path = _write_text(
                chunk_dir / "synthesis_prompt.txt",
                synthesis_prompt,
            )
            synthesis_stage = _start_stage(
                manifest,
                manifest_path,
                "synthesis_llm",
                metadata={
                    "backend": config.synthesis_backend,
                    "model": config.synthesis_model or "",
                },
            )
            synthesis_execution = run_prompt_with_metadata(
                config.synthesis_backend,
                synthesis_prompt,
                model=config.synthesis_model,
                timeout=config.llm_timeout_seconds,
            )
            summary_result_path = _write_text(summary_path, synthesis_execution.output)
            _finish_stage(
                manifest,
                manifest_path,
                synthesis_stage,
                metadata=_execution_metadata(
                    synthesis_prompt_path,
                    summary_result_path,
                    synthesis_execution,
                ),
            )

            if config.critique_backend:
                critique_prompt = build_challenge_prompt(
                    book,
                    summary_result_path.read_text(encoding="utf-8"),
                )
                critique_prompt_path = _write_text(
                    chunk_dir / "critique_prompt.txt",
                    critique_prompt,
                )
                critique_stage = _start_stage(
                    manifest,
                    manifest_path,
                    "critique_llm",
                    metadata={
                        "backend": config.critique_backend,
                        "model": config.critique_model or "",
                    },
                )
                critique_execution = run_prompt_with_metadata(
                    config.critique_backend,
                    critique_prompt,
                    model=config.critique_model,
                    timeout=config.llm_timeout_seconds,
                )
                critique_result_path = _write_text(
                    critique_path,
                    critique_execution.output,
                )
                _finish_stage(
                    manifest,
                    manifest_path,
                    critique_stage,
                    metadata=_execution_metadata(
                        critique_prompt_path,
                        critique_result_path,
                        critique_execution,
                    ),
                )

        _complete_manifest(
            manifest,
            manifest_path,
            expanded_notes_path=expanded_notes_result_path,
            summary_path=summary_result_path,
            critique_path=critique_result_path,
        )
        return _final_artifacts(
            extracted_text_path=extracted_text_path,
            chunk_dir=chunk_dir,
            expanded_notes_path=expanded_notes_result_path,
            summary_path=summary_result_path,
            critique_path=critique_result_path,
            manifest_path=manifest_path,
        )
    except Exception as exc:
        _fail_running_stages(manifest, manifest_path, error=str(exc))
        raise


async def summarize_book_async(
    book: BookRecord,
    *,
    project_root: Path = PROJECT_ROOT,
    config: SummaryConfig | None = None,
    limiter: AsyncRequestLimiter | None = None,
) -> SummaryArtifacts:
    config = config or SummaryConfig()
    (
        _runtime,
        extracted_text_path,
        chunk_dir,
        expanded_notes_path,
        summary_path,
        critique_path,
        manifest_path,
        manifest,
        chunks,
    ) = _prepare_run(book, project_root=project_root, config=config)

    chunk_outputs: list[str] = []
    chunk_dir.mkdir(parents=True, exist_ok=True)
    try:
        if config.dry_run:
            for index, chunk in enumerate(chunks, start=1):
                prompt = build_chunk_analysis_prompt(
                    book,
                    chunk,
                    chunk_index=index,
                    total_chunks=len(chunks),
                )
                prompt_path = chunk_dir / f"chunk_{index:03d}_prompt.txt"
                _write_text(prompt_path, prompt)
        else:
            manifest_lock = asyncio.Lock()
            chunk_semaphore = asyncio.Semaphore(max(config.chunk_concurrency, 1))
            chunk_tasks = [
                asyncio.create_task(
                    _run_chunk_stage_async(
                        book=book,
                        chunk_text=chunk,
                        chunk_index=index,
                        total_chunks=len(chunks),
                        chunk_dir=chunk_dir,
                        config=config,
                        manifest=manifest,
                        manifest_path=manifest_path,
                        manifest_lock=manifest_lock,
                        chunk_semaphore=chunk_semaphore,
                        limiter=limiter,
                    )
                )
                for index, chunk in enumerate(chunks, start=1)
            ]
            try:
                chunk_outputs = list(await asyncio.gather(*chunk_tasks))
            except Exception:
                for task in chunk_tasks:
                    if not task.done():
                        task.cancel()
                await asyncio.gather(*chunk_tasks, return_exceptions=True)
                raise

        expanded_notes_result_path: Path | None = None
        summary_result_path: Path | None = None
        critique_result_path: Path | None = None

        if config.dry_run:
            _write_text(
                chunk_dir / "NEXT_STEPS.txt",
                "Dry run completed. Chunk prompts were written. Re-run without --dry-run to call the configured backends.\n",
            )
        else:
            expanded_notes_stage = _start_stage(
                manifest,
                manifest_path,
                "expanded_notes_render",
                metadata={"chunk_count": len(chunk_outputs)},
            )
            expanded_notes_text = render_expanded_chunk_notes(book, chunk_outputs)
            expanded_notes_result_path = _write_text(
                expanded_notes_path,
                expanded_notes_text,
            )
            _finish_stage(
                manifest,
                manifest_path,
                expanded_notes_stage,
                metadata={
                    "expanded_notes_path": str(expanded_notes_result_path),
                    "char_count": len(expanded_notes_text),
                },
            )
            synthesis_prompt = build_synthesis_prompt(book, chunk_outputs)
            synthesis_prompt_path = _write_text(
                chunk_dir / "synthesis_prompt.txt",
                synthesis_prompt,
            )
            synthesis_stage = _start_stage(
                manifest,
                manifest_path,
                "synthesis_llm",
                metadata={
                    "backend": config.synthesis_backend,
                    "model": config.synthesis_model or "",
                },
            )
            synthesis_execution = await run_prompt_async(
                config.synthesis_backend,
                synthesis_prompt,
                model=config.synthesis_model,
                timeout=config.llm_timeout_seconds,
                limiter=limiter,
            )
            summary_result_path = _write_text(summary_path, synthesis_execution.output)
            _finish_stage(
                manifest,
                manifest_path,
                synthesis_stage,
                metadata=_execution_metadata(
                    synthesis_prompt_path,
                    summary_result_path,
                    synthesis_execution,
                ),
            )

            if config.critique_backend:
                critique_prompt = build_challenge_prompt(
                    book,
                    summary_result_path.read_text(encoding="utf-8"),
                )
                critique_prompt_path = _write_text(
                    chunk_dir / "critique_prompt.txt",
                    critique_prompt,
                )
                critique_stage = _start_stage(
                    manifest,
                    manifest_path,
                    "critique_llm",
                    metadata={
                        "backend": config.critique_backend,
                        "model": config.critique_model or "",
                    },
                )
                critique_execution = await run_prompt_async(
                    config.critique_backend,
                    critique_prompt,
                    model=config.critique_model,
                    timeout=config.llm_timeout_seconds,
                    limiter=limiter,
                )
                critique_result_path = _write_text(
                    critique_path,
                    critique_execution.output,
                )
                _finish_stage(
                    manifest,
                    manifest_path,
                    critique_stage,
                    metadata=_execution_metadata(
                        critique_prompt_path,
                        critique_result_path,
                        critique_execution,
                    ),
                )

        _complete_manifest(
            manifest,
            manifest_path,
            expanded_notes_path=expanded_notes_result_path,
            summary_path=summary_result_path,
            critique_path=critique_result_path,
        )
        return _final_artifacts(
            extracted_text_path=extracted_text_path,
            chunk_dir=chunk_dir,
            expanded_notes_path=expanded_notes_result_path,
            summary_path=summary_result_path,
            critique_path=critique_result_path,
            manifest_path=manifest_path,
        )
    except Exception as exc:
        _fail_running_stages(manifest, manifest_path, error=str(exc))
        raise


async def summarize_books_async(
    books: list[BookRecord],
    *,
    project_root: Path = PROJECT_ROOT,
    config: SummaryConfig | None = None,
    max_concurrency: int = 2,
    requests_per_window: int | None = None,
    window_seconds: float = 60.0,
    book_concurrency: int = 2,
) -> list[tuple[BookRecord, SummaryArtifacts]]:
    config = config or SummaryConfig()
    limiter = AsyncRequestLimiter(
        max_concurrency=max_concurrency,
        requests_per_window=requests_per_window,
        window_seconds=window_seconds,
    )
    book_semaphore = asyncio.Semaphore(book_concurrency)

    async def _run(book: BookRecord) -> tuple[BookRecord, SummaryArtifacts]:
        async with book_semaphore:
            artifacts = await summarize_book_async(
                book,
                project_root=project_root,
                config=config,
                limiter=limiter,
            )
            return book, artifacts

    return await asyncio.gather(*(_run(book) for book in books))
