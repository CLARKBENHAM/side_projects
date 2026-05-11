"""Tests for fixed-session Claude usage analysis."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys
from zoneinfo import ZoneInfo


MODULE_PATH = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "temp"
    / "check_claude_usage_limits.py"
)
SPEC = importlib.util.spec_from_file_location("check_claude_usage_limits", MODULE_PATH)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def test_parse_reset_at_same_day() -> None:
    tz = ZoneInfo("America/Los_Angeles")
    hit_time = MODULE.datetime(2026, 3, 31, 19, 2, tzinfo=tz)
    reset_at = MODULE.parse_reset_at(
        hit_time,
        "You're out of extra usage · resets 9pm (America/Los_Angeles)",
    )
    assert reset_at == MODULE.datetime(2026, 3, 31, 21, 0, tzinfo=tz)


def test_parse_reset_at_next_day_when_past_reset_hour() -> None:
    tz = ZoneInfo("America/Los_Angeles")
    hit_time = MODULE.datetime(2026, 3, 31, 21, 6, tzinfo=tz)
    reset_at = MODULE.parse_reset_at(
        hit_time,
        "Current session resets 9pm",
    )
    assert reset_at == MODULE.datetime(2026, 4, 1, 21, 0, tzinfo=tz)


def test_bucket_summary_ignores_same_session_usage_before_bucket(
    tmp_path: Path,
) -> None:
    session_file = tmp_path / "abc123.jsonl"
    rows = [
        {
            "timestamp": "2026-03-31T15:30:00Z",
            "type": "assistant",
            "sessionId": "abc123",
            "message": {
                "role": "assistant",
                "model": "claude-opus-4-6",
                "usage": {
                    "input_tokens": 100,
                    "cache_creation_input_tokens": 200,
                    "cache_read_input_tokens": 300,
                    "output_tokens": 25,
                },
                "content": [{"type": "tool_use", "name": "Read"}],
            },
        },
        {
            "timestamp": "2026-03-31T18:30:00Z",
            "type": "user",
            "sessionId": "abc123",
            "message": {"role": "user", "content": "keep going"},
        },
        {
            "timestamp": "2026-03-31T18:31:00Z",
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
        {
            "timestamp": "2026-03-31T20:29:04Z",
            "type": "assistant",
            "sessionId": "abc123",
            "error": "rate_limit",
            "message": {
                "role": "assistant",
                "model": "<synthetic>",
                "content": [
                    {
                        "type": "text",
                        "text": "You're out of extra usage · resets 4pm (America/Los_Angeles)",
                    }
                ],
            },
        },
    ]
    session_file.write_text(
        "\n".join(json.dumps(row) for row in rows), encoding="utf-8"
    )

    tz = ZoneInfo("America/Los_Angeles")
    events = MODULE.load_events(tmp_path, tz)
    hits = MODULE.extract_rate_limit_hits(events)
    assert len(hits) == 1

    summary = MODULE.summarize_bucket(events, hits[0])
    assert summary.total_uncached_input == 30
    assert summary.total_cache_read_input == 30
    assert summary.total_output_tokens == 5
    assert summary.user_messages == 1
    assert summary.total_tool_uses == 1
    assert summary.triggering_session_prior_uncached_input == 300
    assert summary.triggering_session_prior_cache_read_input == 300
    assert summary.triggering_session_prior_output_tokens == 25


def test_candidate_buckets_include_hit_and_non_hit_windows(tmp_path: Path) -> None:
    session_file = tmp_path / "abc123.jsonl"
    rows = [
        {
            "timestamp": "2026-03-31T16:30:00Z",
            "type": "assistant",
            "sessionId": "abc123",
            "message": {
                "role": "assistant",
                "model": "claude-opus-4-6",
                "usage": {
                    "input_tokens": 1,
                    "cache_creation_input_tokens": 1,
                    "cache_read_input_tokens": 1,
                    "output_tokens": 1,
                },
                "content": [],
            },
        },
        {
            "timestamp": "2026-03-31T18:05:00Z",
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
        {
            "timestamp": "2026-03-31T19:00:00Z",
            "type": "assistant",
            "sessionId": "abc123",
            "message": {
                "role": "assistant",
                "model": "claude-haiku-4-5-20251001",
                "usage": {
                    "input_tokens": 2,
                    "cache_creation_input_tokens": 3,
                    "cache_read_input_tokens": 4,
                    "output_tokens": 1,
                },
                "content": [],
            },
        },
        {
            "timestamp": "2026-03-31T20:29:04Z",
            "type": "assistant",
            "sessionId": "abc123",
            "error": "rate_limit",
            "message": {
                "role": "assistant",
                "model": "<synthetic>",
                "content": [
                    {
                        "type": "text",
                        "text": "You're out of extra usage · resets 4pm (America/Los_Angeles)",
                    }
                ],
            },
        },
    ]
    session_file.write_text(
        "\n".join(json.dumps(row) for row in rows), encoding="utf-8"
    )

    tz = ZoneInfo("America/Los_Angeles")
    events = MODULE.load_events(tmp_path, tz)
    hits = MODULE.extract_rate_limit_hits(events)
    buckets = MODULE.build_candidate_buckets(events, hits)

    hit_buckets = [bucket for bucket in buckets if bucket.has_hit]
    non_hit_buckets = [bucket for bucket in buckets if not bucket.has_hit]
    assert len(hit_buckets) == 1
    assert non_hit_buckets

    hit_bucket = hit_buckets[0]
    assert hit_bucket.bucket_start == MODULE.datetime(2026, 3, 31, 11, 0, tzinfo=tz)
    assert hit_bucket.bucket_end == MODULE.datetime(2026, 3, 31, 16, 0, tzinfo=tz)
    assert hit_bucket.total_uncached_input == 35
    assert hit_bucket.total_output_tokens == 6
    assert hit_bucket.opus_messages == 1
    assert hit_bucket.haiku_messages == 1


def test_candidate_bucket_features_include_rates_and_priors() -> None:
    tz = ZoneInfo("America/Los_Angeles")
    bucket = MODULE.CandidateBucket(
        bucket_start=MODULE.datetime(2026, 3, 31, 11, 0, tzinfo=tz),
        bucket_end=MODULE.datetime(2026, 3, 31, 16, 0, tzinfo=tz),
        bucket_anchor_reset=MODULE.datetime(2026, 3, 31, 16, 0, tzinfo=tz),
        has_hit=True,
        hit_count=1,
        first_hit_time=MODULE.datetime(2026, 3, 31, 13, 30, tzinfo=tz),
        observed_end=MODULE.datetime(2026, 3, 31, 13, 30, tzinfo=tz),
        is_partial=False,
        total_uncached_input=120,
        total_cache_read_input=300,
        total_output_tokens=30,
        assistant_messages=6,
        user_messages=4,
        total_tool_uses=3,
        active_sessions=2,
        distinct_models=2,
        opus_messages=5,
        haiku_messages=1,
        sonnet_messages=0,
        opus_uncached_plus_output=120,
        haiku_uncached_plus_output=30,
        sonnet_uncached_plus_output=0,
        prior_uncached_input_active_sessions=500,
        prior_cache_read_input_active_sessions=800,
        prior_output_tokens_active_sessions=50,
        prior_messages_active_sessions=40,
        max_prior_messages_single_session=25,
        triggering_session_id="abc123",
    )

    features = MODULE.candidate_bucket_features(bucket)
    assert features["observed_hours"] == 2.5
    assert features["observed_fraction"] == 0.5
    assert features["total_uncached_plus_output"] == 150
    assert features["total_effective_input"] == 420
    assert features["total_work_tokens"] == 450
    assert features["uncached_plus_output_per_hour"] == 60
    assert features["opus_share_uncached_plus_output"] == 0.8
    assert features["prior_uncached_plus_output_active_sessions"] == 550
    assert features["prior_effective_input_active_sessions"] == 1300
    assert features["prior_work_tokens_active_sessions"] == 1350


def test_build_horizon_buckets_excludes_hits_before_horizon(tmp_path: Path) -> None:
    session_file = tmp_path / "abc123.jsonl"
    rows = [
        {
            "timestamp": "2026-03-31T18:05:00Z",
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
        {
            "timestamp": "2026-03-31T18:40:00Z",
            "type": "assistant",
            "sessionId": "abc123",
            "error": "rate_limit",
            "message": {
                "role": "assistant",
                "model": "<synthetic>",
                "content": [
                    {
                        "type": "text",
                        "text": "You're out of extra usage · resets 4pm (America/Los_Angeles)",
                    }
                ],
            },
        },
    ]
    session_file.write_text(
        "\n".join(json.dumps(row) for row in rows), encoding="utf-8"
    )

    tz = ZoneInfo("America/Los_Angeles")
    events = MODULE.load_events(tmp_path, tz)
    hits = MODULE.extract_rate_limit_hits(events)
    buckets = MODULE.build_candidate_buckets(events, hits)

    half_hour = MODULE.build_horizon_buckets(events, buckets, 0.5)
    one_hour = MODULE.build_horizon_buckets(events, buckets, 1.0)
    assert len(half_hour) == 1
    assert len(one_hour) == 0
