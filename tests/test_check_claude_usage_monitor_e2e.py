"""Tests for Claude usage-monitor end-to-end verifier helpers."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys


MODULE_PATH = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "temp"
    / "check_claude_usage_monitor_e2e.py"
)
SPEC = importlib.util.spec_from_file_location(
    "check_claude_usage_monitor_e2e", MODULE_PATH
)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def test_verify_no_tool_stream_accepts_expected_sequence() -> None:
    events = [
        {"type": "system", "subtype": "hook_started", "hook_event": "SessionStart"},
        {"type": "system", "subtype": "hook_started", "hook_event": "UserPromptSubmit"},
        {
            "type": "assistant",
            "message": {"content": [{"type": "text", "text": "hello world"}]},
        },
        {"type": "system", "subtype": "hook_started", "hook_event": "Stop"},
        {"type": "result", "result": "hello world"},
    ]
    assert MODULE.verify_no_tool_stream(events) == []


def test_verify_tool_stream_requires_tool_use_and_tool_result() -> None:
    events = [
        {"type": "system", "subtype": "hook_started", "hook_event": "SessionStart"},
        {"type": "system", "subtype": "hook_started", "hook_event": "UserPromptSubmit"},
        {"type": "system", "subtype": "hook_started", "hook_event": "PreToolUse"},
        {"type": "system", "subtype": "hook_started", "hook_event": "PostToolUse"},
        {
            "type": "assistant",
            "message": {"content": [{"type": "tool_use", "name": "Bash"}]},
        },
        {
            "type": "user",
            "message": {"content": [{"type": "tool_result", "content": "hello world"}]},
        },
        {
            "type": "assistant",
            "message": {"content": [{"type": "text", "text": "hello world"}]},
        },
        {"type": "system", "subtype": "hook_started", "hook_event": "Stop"},
        {"type": "result", "result": "hello world"},
    ]
    assert MODULE.verify_tool_stream(events) == []


def test_verify_hook_log_matches_suffix(tmp_path: Path) -> None:
    events_path = tmp_path / "events.jsonl"
    rows = [
        {"session_id": "abc", "hook_event": "SessionStart"},
        {"session_id": "abc", "hook_event": "UserPromptSubmit"},
        {"session_id": "abc", "hook_event": "Stop"},
        {"session_id": "abc", "hook_event": "SessionEnd"},
    ]
    events_path.write_text(
        "\n".join(json.dumps(row) for row in rows) + "\n",
        encoding="utf-8",
    )
    assert (
        MODULE.verify_hook_log(
            events_path,
            "abc",
            ["SessionStart", "UserPromptSubmit", "Stop", "SessionEnd"],
            timeout_seconds=0.1,
        )
        == []
    )
