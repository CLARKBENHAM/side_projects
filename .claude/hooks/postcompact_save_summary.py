#!/usr/bin/env python3
import json
import os
from datetime import datetime
from pathlib import Path


def main() -> int:
    try:
        payload = json.load(__import__("sys").stdin)
    except json.JSONDecodeError:
        return 0

    cwd = payload.get("cwd") or os.getcwd()
    session_id = payload.get("session_id", "unknown-session")
    trigger = payload.get("trigger", "unknown")
    compact_summary = payload.get("compact_summary", "")

    out_dir = Path(cwd) / ".claude" / "backups"
    out_dir.mkdir(parents=True, exist_ok=True)

    ts = datetime.now().strftime("%Y%m%d-%H%M%S")
    out = out_dir / f"{ts}--{trigger}--{session_id}--compact-summary.md"
    out.write_text(compact_summary, encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
