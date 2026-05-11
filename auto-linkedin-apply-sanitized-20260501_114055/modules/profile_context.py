import os
import re
import shutil
import subprocess
from functools import lru_cache
from pathlib import Path


DEFAULT_PROFILE_CONTEXT_PATH = Path("all resumes/default_resume.pdf")
DEFAULT_MAX_PROFILE_CONTEXT_CHARS = 12000


def _normalize_text(text: str) -> str:
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def _pdftotext_binary() -> str:
    configured_binary = os.environ.get("AUTO_APPLIER_PDFTOTEXT_BIN", "").strip()
    if configured_binary:
        return configured_binary
    return shutil.which("pdftotext") or "pdftotext"


def _extract_pdf_text(path: Path) -> str:
    completed = subprocess.run(
        [_pdftotext_binary(), "-layout", str(path), "-"],
        text=True,
        capture_output=True,
        timeout=15,
        check=False,
    )
    if completed.returncode != 0:
        return ""
    return completed.stdout


def _read_profile_context_file(path: Path) -> str:
    if not path.exists():
        return ""
    if path.suffix.lower() == ".pdf":
        return _extract_pdf_text(path)
    return path.read_text(encoding="utf-8", errors="ignore")


def _max_context_chars() -> int:
    raw_value = os.environ.get("AUTO_APPLIER_PROFILE_CONTEXT_MAX_CHARS", "").strip()
    if not raw_value:
        return DEFAULT_MAX_PROFILE_CONTEXT_CHARS
    value = int(raw_value)
    if value < 1000:
        raise ValueError("AUTO_APPLIER_PROFILE_CONTEXT_MAX_CHARS must be >= 1000")
    return value


@lru_cache(maxsize=1)
def load_profile_context() -> str:
    raw_path = os.environ.get("AUTO_APPLIER_PROFILE_CONTEXT_PATH", "").strip()
    path = Path(raw_path).expanduser() if raw_path else DEFAULT_PROFILE_CONTEXT_PATH
    text = _normalize_text(_read_profile_context_file(path))
    if not text:
        return ""

    max_chars = _max_context_chars()
    if len(text) > max_chars:
        text = text[:max_chars].rstrip()
    return f"Profile context source: {path}\n{text}"
