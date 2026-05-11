"""Tests for Claude usage-monitor hook and analyzer."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
HOOK_PATH = ROOT / "scripts" / "temp" / "claude_usage_hook.py"
ANALYZE_PATH = ROOT / "scripts" / "temp" / "analyze_claude_usage_monitor.py"
INSTALL_PATH = ROOT / "scripts" / "temp" / "install_claude_usage_monitor.py"


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


HOOK = load_module("claude_usage_hook", HOOK_PATH)
ANALYZE = load_module("analyze_claude_usage_monitor", ANALYZE_PATH)
INSTALL = load_module("install_claude_usage_monitor", INSTALL_PATH)


def test_hook_state_updates_incrementally(tmp_path: Path) -> None:
    transcript = tmp_path / "session.jsonl"
    first_rows = [
        {
            "timestamp": "2026-03-31T18:00:00Z",
            "type": "user",
            "sessionId": "abc123",
            "message": {"role": "user", "content": "go"},
        },
        {
            "timestamp": "2026-03-31T18:01:00Z",
            "type": "assistant",
            "sessionId": "abc123",
            "message": {
                "role": "assistant",
                "model": "claude-opus-4-6",
                "usage": {
                    "input_tokens": 10,
                    "cache_creation_input_tokens": 20,
                    "cache_read_input_tokens": 30,
                    "output_tokens": 5,
                },
                "content": [{"type": "tool_use", "name": "Bash"}],
            },
        },
    ]
    transcript.write_text(
        "\n".join(json.dumps(row) for row in first_rows) + "\n",
        encoding="utf-8",
    )

    state = HOOK.empty_state(str(transcript))
    state = HOOK.update_state_from_transcript(transcript, state)
    assert state["assistant_messages"] == 1
    assert state["user_messages"] == 1
    assert state["tool_uses"] == 1
    assert state["usage_totals"]["input_tokens"] == 10
    assert state["usage_totals"]["cache_creation_input_tokens"] == 20
    assert state["usage_totals"]["cache_read_input_tokens"] == 30
    assert state["usage_totals"]["output_tokens"] == 5
    first_file_size = state["file_size"]

    second_row = {
        "timestamp": "2026-03-31T18:02:00Z",
        "type": "assistant",
        "sessionId": "abc123",
        "error": "rate_limit",
        "message": {
            "role": "assistant",
            "model": "claude-opus-4-6",
            "usage": {
                "input_tokens": 1,
                "cache_creation_input_tokens": 2,
                "cache_read_input_tokens": 3,
                "output_tokens": 4,
            },
            "content": [
                {
                    "type": "text",
                    "text": "You're out of extra usage · resets 9pm (America/Los_Angeles)",
                }
            ],
        },
    }
    with transcript.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(second_row) + "\n")

    state = HOOK.update_state_from_transcript(transcript, state)
    assert state["file_size"] > first_file_size
    assert state["assistant_messages"] == 2
    assert state["usage_totals"]["input_tokens"] == 11
    assert state["usage_totals"]["cache_creation_input_tokens"] == 22
    assert state["usage_totals"]["cache_read_input_tokens"] == 33
    assert state["usage_totals"]["output_tokens"] == 9
    assert state["rate_limit_error_count"] == 1
    assert "out of extra usage" in state["last_rate_limit_error_text"].lower()


def test_analyzer_builds_positive_intervals_and_markdown() -> None:
    records = [
        {
            "_logged_at_dt": ANALYZE.parse_iso("2026-03-31T18:00:00+00:00"),
            "hook_event": "SessionStart",
            "session_id": "abc123",
            "cwd": "/tmp/work",
            "payload_known_fields": {
                "prompt": {"preview": "Build a summary", "chars": 15}
            },
            "transcript_snapshot": {
                "usage_totals": {
                    "input_tokens": 10,
                    "cache_creation_input_tokens": 20,
                    "cache_read_input_tokens": 30,
                    "output_tokens": 5,
                },
                "usage_by_model": {
                    "claude-opus-4-6": {
                        "input_tokens": 10,
                        "cache_creation_input_tokens": 20,
                        "cache_read_input_tokens": 30,
                        "output_tokens": 5,
                    }
                },
                "api_cost_exact_usd_total": 1.23,
                "api_cost_opus_upper_usd_total": 1.23,
                "rate_limit_error_count": 0,
            },
        },
        {
            "_logged_at_dt": ANALYZE.parse_iso("2026-03-31T18:05:00+00:00"),
            "hook_event": "PostToolUse",
            "session_id": "abc123",
            "cwd": "/tmp/work",
            "payload_known_fields": {"tool_name": "Bash"},
            "transcript_snapshot": {
                "usage_totals": {
                    "input_tokens": 13,
                    "cache_creation_input_tokens": 25,
                    "cache_read_input_tokens": 45,
                    "output_tokens": 9,
                },
                "usage_by_model": {
                    "claude-opus-4-6": {
                        "input_tokens": 13,
                        "cache_creation_input_tokens": 25,
                        "cache_read_input_tokens": 45,
                        "output_tokens": 9,
                    }
                },
                "api_cost_exact_usd_total": 2.0,
                "api_cost_opus_upper_usd_total": 2.0,
                "rate_limit_error_count": 1,
            },
        },
    ]

    intervals = ANALYZE.build_intervals(records)
    assert len(intervals) == 2
    delta_interval = intervals[1]
    assert delta_interval["usage_delta"]["input_tokens"] == 3
    assert delta_interval["usage_delta"]["cache_creation_input_tokens"] == 5
    assert delta_interval["usage_delta"]["cache_read_input_tokens"] == 15
    assert delta_interval["usage_delta"]["output_tokens"] == 4
    assert delta_interval["exact_cost_delta"] == 0.77
    assert delta_interval["rate_limit_delta"] == 1

    markdown = ANALYZE.build_markdown(records, intervals)
    assert "Exact API-equivalent delta" in markdown
    assert "PostToolUse" in markdown
    assert "Bash" in markdown


def test_install_hooks_preserves_existing_commands() -> None:
    settings = {
        "model": "opus",
        "hooks": {
            "PreToolUse": [
                {
                    "matcher": "Bash",
                    "hooks": [{"type": "command", "command": "python3 /tmp/other.py"}],
                }
            ]
        },
    }

    merged = INSTALL.install_hooks(settings, "/tmp/claude_usage_hook.py")
    pre_tool_hooks = merged["hooks"]["PreToolUse"]
    commands = [hook["command"] for entry in pre_tool_hooks for hook in entry["hooks"]]
    assert "python3 /tmp/other.py" in commands
    assert "python3 /tmp/claude_usage_hook.py PreToolUse" in commands
    assert "SessionStart" in merged["hooks"]
