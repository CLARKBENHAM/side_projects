#!/usr/bin/env python3
import json
import os
import shutil
import sys
from datetime import datetime
from pathlib import Path


def main() -> int:
    try:
        payload = json.load(sys.stdin)
    except json.JSONDecodeError:
        return 0

    transcript_path = payload.get("transcript_path")
    session_id = payload.get("session_id", "unknown-session")
    trigger = payload.get("trigger", "unknown")
    cwd = payload.get("cwd") or os.getcwd()

    if not transcript_path:
        return 0

    repo_root = Path(cwd)
    backup_dir = repo_root / ".claude" / "backups"
    backup_dir.mkdir(parents=True, exist_ok=True)

    src = Path(transcript_path)
    if not src.exists():
        return 0

    ts = datetime.now().strftime("%Y%m%d-%H%M%S")
    dst = backup_dir / f"{ts}--{trigger}--{session_id}.jsonl"

    shutil.copy2(src, dst)
    print(f"Backed up transcript to {dst}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
