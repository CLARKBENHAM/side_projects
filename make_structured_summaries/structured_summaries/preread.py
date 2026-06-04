"""Pre-reading brief orchestration."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path

from .chunking import split_text_into_chunks
from .llm_backends import PromptExecution, run_prompt_with_metadata
from .models import BookRecord
from .prompts import build_preread_chunk_prompt, build_preread_synthesis_prompt
from .text_extraction import extract_book_text

PROJECT_ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class PrereadConfig:
    chunk_backend: str = "claude"
    chunk_model: str | None = "sonnet"
    synthesis_backend: str = "claude"
    synthesis_model: str | None = "sonnet"
    chunk_chars: int = 60_000
    overlap_chars: int = 2_000
    max_chunks: int | None = None
    llm_timeout_seconds: int = 900
    force: bool = False
    dry_run: bool = False


@dataclass(frozen=True)
class PrereadArtifacts:
    extracted_text_path: Path
    chunk_dir: Path
    summary_path: Path | None
    manifest_path: Path


def _now_iso() -> str:
    return datetime.now().isoformat()


def _write_text(path: Path, text: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path


def _write_manifest(path: Path, manifest: dict[str, object]) -> None:
    _write_text(path, json.dumps(manifest, indent=2, ensure_ascii=False))


def _runtime_dirs(project_root: Path) -> dict[str, Path]:
    data_dir = project_root / "data"
    extracted_dir = data_dir / "extracted_text"
    chunk_dir = data_dir / "preread_chunk_notes"
    summary_dir = data_dir / "preread_summaries"
    for directory in [data_dir, extracted_dir, chunk_dir, summary_dir]:
        directory.mkdir(parents=True, exist_ok=True)
    return {
        "data_dir": data_dir,
        "extracted_dir": extracted_dir,
        "chunk_dir": chunk_dir,
        "summary_dir": summary_dir,
    }


def _clear_generated_outputs(chunk_dir: Path, summary_path: Path) -> None:
    if chunk_dir.exists():
        for pattern in (
            "chunk_*_prompt.txt",
            "chunk_*_response.txt",
            "synthesis_prompt.txt",
            "NEXT_STEPS.txt",
        ):
            for path in chunk_dir.glob(pattern):
                path.unlink()
    if summary_path.exists():
        summary_path.unlink()


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


def _complete_manifest(
    manifest: dict[str, object],
    manifest_path: Path,
    *,
    summary_path: Path | None,
) -> None:
    manifest["status"] = "completed"
    manifest["current_stage"] = "completed"
    manifest["timestamp_finished"] = _now_iso()
    paths = manifest.get("paths", {})
    if not isinstance(paths, dict):
        raise TypeError("paths must be a dict")
    paths["summary_path"] = str(summary_path or "")
    _write_manifest(manifest_path, manifest)


def _fail_manifest(
    manifest: dict[str, object],
    manifest_path: Path,
    *,
    error: str,
) -> None:
    manifest["status"] = "failed"
    manifest["current_stage"] = "failed"
    manifest["timestamp_finished"] = _now_iso()
    manifest["error"] = error
    stage_records = manifest.get("stage_records", [])
    if isinstance(stage_records, list):
        for stage in stage_records:
            if isinstance(stage, dict) and stage.get("status") == "running":
                stage["status"] = "failed"
                stage["finished_at"] = _now_iso()
                metadata = stage.setdefault("metadata", {})
                if isinstance(metadata, dict):
                    metadata["error"] = error
    _write_manifest(manifest_path, manifest)


def _build_manifest(
    *,
    book: BookRecord,
    config: PrereadConfig,
    extracted_text_path: Path,
    chunk_dir: Path,
    summary_path: Path,
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
            "summary_path": str(summary_path),
        },
    }


def _prepare_run(
    book: BookRecord,
    *,
    project_root: Path,
    config: PrereadConfig,
) -> tuple[Path, Path, Path, Path, dict[str, object], list[str]]:
    runtime = _runtime_dirs(project_root)
    extracted_text_path = runtime["extracted_dir"] / f"{book.book_id}.txt"
    chunk_dir = runtime["chunk_dir"] / book.book_id
    summary_path = runtime["summary_dir"] / f"{book.book_id}.md"
    manifest_path = runtime["summary_dir"] / f"{book.book_id}.manifest.json"
    if config.force:
        _clear_generated_outputs(chunk_dir, summary_path)
    manifest = _build_manifest(
        book=book,
        config=config,
        extracted_text_path=extracted_text_path,
        chunk_dir=chunk_dir,
        summary_path=summary_path,
    )
    _write_manifest(manifest_path, manifest)

    extract_stage = _start_stage(
        manifest,
        manifest_path,
        "extract_text",
        metadata={"source_path": str(book.primary_path)},
    )
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
    return extracted_text_path, chunk_dir, summary_path, manifest_path, manifest, chunks


def build_preread_brief(
    book: BookRecord,
    *,
    project_root: Path = PROJECT_ROOT,
    config: PrereadConfig | None = None,
) -> PrereadArtifacts:
    config = config or PrereadConfig()
    (
        extracted_text_path,
        chunk_dir,
        summary_path,
        manifest_path,
        manifest,
        chunks,
    ) = _prepare_run(book, project_root=project_root, config=config)

    chunk_outputs: list[str] = []
    chunk_dir.mkdir(parents=True, exist_ok=True)
    try:
        for index, chunk in enumerate(chunks, start=1):
            prompt = build_preread_chunk_prompt(
                book,
                chunk,
                chunk_index=index,
                total_chunks=len(chunks),
            )
            prompt_path = _write_text(chunk_dir / f"chunk_{index:03d}_prompt.txt", prompt)
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
            response_path = _write_text(
                chunk_dir / f"chunk_{index:03d}_response.txt",
                execution.output,
            )
            _finish_stage(
                manifest,
                manifest_path,
                stage,
                metadata=_execution_metadata(prompt_path, response_path, execution),
            )
            chunk_outputs.append(execution.output)

        if config.dry_run:
            _write_text(
                chunk_dir / "NEXT_STEPS.txt",
                "Dry run completed. Chunk prompts were written. Re-run without --dry-run to call the configured backends.\n",
            )
            _complete_manifest(manifest, manifest_path, summary_path=None)
            return PrereadArtifacts(
                extracted_text_path=extracted_text_path,
                chunk_dir=chunk_dir,
                summary_path=None,
                manifest_path=manifest_path,
            )

        synthesis_prompt = build_preread_synthesis_prompt(book, chunk_outputs)
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
        _complete_manifest(manifest, manifest_path, summary_path=summary_result_path)
        return PrereadArtifacts(
            extracted_text_path=extracted_text_path,
            chunk_dir=chunk_dir,
            summary_path=summary_result_path,
            manifest_path=manifest_path,
        )
    except Exception as exc:
        _fail_manifest(manifest, manifest_path, error=str(exc))
        raise
