import pytest
from datetime import date
from datetime import datetime

from modules.daily_limits import (
    count_recent_linkedin_application_actions,
    count_todays_linkedin_application_actions,
    count_todays_linkedin_applications,
    daily_linkedin_application_cap_reached,
    record_linkedin_application_action,
    resolve_max_daily_linkedin_applications,
    resolve_max_result_pages_per_search,
    resolve_max_run_linkedin_applications,
)


def test_resolve_max_daily_linkedin_applications_defaults_when_unset(monkeypatch):
    monkeypatch.delenv("AUTO_APPLIER_MAX_DAILY_LINKEDIN_APPLICATIONS", raising=False)
    monkeypatch.delenv("AUTO_APPLIER_MAX_DAILY_APPLICATIONS", raising=False)
    monkeypatch.delenv("AUTO_APPLIER_MAX_DAILY_EASY_APPLIES", raising=False)

    assert resolve_max_daily_linkedin_applications() == 0


def test_resolve_max_daily_linkedin_applications_reads_positive_env(monkeypatch):
    monkeypatch.setenv("AUTO_APPLIER_MAX_DAILY_LINKEDIN_APPLICATIONS", "15")

    assert resolve_max_daily_linkedin_applications() == 15


def test_resolve_max_run_linkedin_applications_reads_positive_env(monkeypatch):
    monkeypatch.setenv("AUTO_APPLIER_MAX_RUN_LINKEDIN_APPLICATIONS", "15")

    assert resolve_max_run_linkedin_applications() == 15


@pytest.mark.parametrize("raw_value", ["0", "-1"])
def test_resolve_max_daily_linkedin_applications_rejects_non_positive_values(
    monkeypatch, raw_value
):
    monkeypatch.setenv("AUTO_APPLIER_MAX_DAILY_LINKEDIN_APPLICATIONS", raw_value)

    with pytest.raises(ValueError):
        resolve_max_daily_linkedin_applications()


def test_daily_linkedin_application_cap_reached_only_after_threshold():
    assert not daily_linkedin_application_cap_reached(14, 15)
    assert daily_linkedin_application_cap_reached(15, 15)
    assert daily_linkedin_application_cap_reached(16, 15)
    assert not daily_linkedin_application_cap_reached(100, 0)


def test_count_todays_linkedin_applications_excludes_external_sites(tmp_path):
    history = tmp_path / "history.csv"
    history.write_text(
        "\n".join(
            [
                "Job ID,Date Applied,External Job link",
                "1,2026-04-28 09:00:00.000000,Easy Applied",
                "2,2026-04-28 10:00:00.000000,https://ats.example/apply",
                "3,2026-04-27 09:00:00.000000,Easy Applied",
                "4,Pending,Easy Applied",
                "5,2026-04-28 11:00:00.000000,Easy Applied",
                "6,2026-04-28 12:00:00.000000,LinkedIn Applied",
            ]
        ),
        encoding="utf-8",
    )

    assert (
        count_todays_linkedin_applications(str(history), today=date(2026, 4, 28)) == 3
    )


def test_count_todays_linkedin_application_actions_deduplicates_history_and_actions(
    tmp_path,
):
    history = tmp_path / "history.csv"
    history.write_text(
        "\n".join(
            [
                "Job ID,Date Applied,External Job link",
                "1,2026-04-28 09:00:00.000000,Easy Applied",
                "2,2026-04-28 10:00:00.000000,https://ats.example/apply",
            ]
        ),
        encoding="utf-8",
    )
    actions = tmp_path / "actions.csv"
    record_linkedin_application_action(
        str(actions),
        "1",
        "Already counted",
        "Example Co",
        recorded_at=datetime(2026, 4, 28, 9, 1),
    )
    record_linkedin_application_action(
        str(actions),
        "3",
        "Interrupted submit",
        "Example Co",
        recorded_at=datetime(2026, 4, 28, 9, 2),
    )
    record_linkedin_application_action(
        str(actions),
        "4",
        "Yesterday submit",
        "Example Co",
        recorded_at=datetime(2026, 4, 27, 9, 2),
    )

    assert (
        count_todays_linkedin_application_actions(
            str(history), str(actions), today=date(2026, 4, 28)
        )
        == 2
    )


def test_count_recent_linkedin_application_actions_uses_rolling_window(tmp_path):
    history = tmp_path / "history.csv"
    history.write_text(
        "\n".join(
            [
                "Job ID,Date Applied,External Job link",
                "1,2026-04-29 23:10:00.000000,Easy Applied",
                "2,2026-04-29 08:00:00.000000,Easy Applied",
                "3,2026-04-30 07:00:00.000000,https://ats.example/apply",
            ]
        ),
        encoding="utf-8",
    )
    actions = tmp_path / "actions.csv"
    record_linkedin_application_action(
        str(actions),
        "1",
        "Already counted",
        "Example Co",
        recorded_at=datetime(2026, 4, 29, 23, 11),
    )
    record_linkedin_application_action(
        str(actions),
        "4",
        "Recent interrupted submit",
        "Example Co",
        recorded_at=datetime(2026, 4, 30, 8, 0),
    )

    assert (
        count_recent_linkedin_application_actions(
            str(history),
            str(actions),
            now=datetime(2026, 4, 30, 9, 0),
            window_hours=24,
        )
        == 2
    )


def test_resolve_max_result_pages_per_search_reads_positive_env(monkeypatch):
    monkeypatch.setenv("AUTO_APPLIER_MAX_RESULT_PAGES_PER_SEARCH", "2")

    assert resolve_max_result_pages_per_search() == 2
