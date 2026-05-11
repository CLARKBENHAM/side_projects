"""Join journal labels to productivity outcomes and write a deep-dive summary."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import statsmodels.api as sm
from scipy import stats

from productivity_analysis import DAILY_SUMMARY_CSV, load_daily_summary_full

BASE_DIR = Path(__file__).resolve().parent
INPUT_DIR = BASE_DIR / "journal_outputs"
OUTPUT_SUMMARY = BASE_DIR / "JOURNAL_DEEP_DIVE_SUMMARY.md"

SCALE_FIELDS = [
    "mood_valence",
    "energy_activation",
    "goal_clarity",
    "execution_focus",
    "overanalysis",
    "avoidance",
    "self_criticism",
    "relationship_conflict",
    "relationship_closeness",
    "alcohol_issue",
    "porn_issue",
    "phone_internet_issue",
    "travel_disruption",
    "illness_or_pain",
    "external_structure",
    "accountability_support",
    "compelling_problem",
]
TEXT_FIELDS = ["dominant_obstacle", "dominant_driver", "dominant_mode"]


def load_labels(path: Path) -> pd.DataFrame:
    records = []
    if not path.exists():
        return pd.DataFrame(columns=["entry_id"])
    with path.open() as handle:
        for line in handle:
            records.append(json.loads(line))
    if not records:
        return pd.DataFrame(columns=["entry_id"])
    df = pd.DataFrame(records)
    # Keep the latest record per entry_id to absorb resumable reruns.
    return df.drop_duplicates(subset="entry_id", keep="last").copy()


def build_weekly_outcomes() -> pd.DataFrame:
    daily = (
        load_daily_summary_full(DAILY_SUMMARY_CSV)
        .sort_values("date")
        .reset_index(drop=True)
    )
    daily["day_of_week"] = daily["date"].dt.dayofweek
    daily["is_weekend"] = (daily["day_of_week"] >= 5).astype(int)
    daily["week_start"] = daily["date"] - pd.to_timedelta(
        daily["day_of_week"], unit="D"
    )

    def dominant_regime(series: pd.Series) -> str:
        non_null = series.dropna()
        if non_null.empty:
            return "Unknown"
        return str(non_null.mode().iloc[0])

    weekly = (
        daily.groupby("week_start")
        .agg(
            total_hours=("Hours Working", "sum"),
            weekday_hours=(
                "Hours Working",
                lambda x: x[daily.loc[x.index, "is_weekend"] == 0].sum(),
            ),
            weekend_hours=(
                "Hours Working",
                lambda x: x[daily.loc[x.index, "is_weekend"] == 1].sum(),
            ),
            workdays=("Hours Working", lambda x: (x >= 2).sum()),
            mean_energy=("Energy", "mean"),
            mean_focus=("Focus", "mean"),
            mean_value=("Value", "mean"),
            dominant_regime=("regime", dominant_regime),
        )
        .reset_index()
    )
    weekly = weekly[weekly["workdays"] >= 2].copy()
    weekly = weekly.sort_values("week_start").reset_index(drop=True)
    weekly["week_ordinal"] = range(len(weekly))
    weekly["prev_total_hours"] = weekly["total_hours"].shift(1)
    weekly["prev_weekend_hours"] = weekly["weekend_hours"].shift(1)
    weekly["next_total_hours"] = weekly["total_hours"].shift(-1)
    weekly["next_weekend_hours"] = weekly["weekend_hours"].shift(-1)

    for regime in ["Hive", "Mats", "diesl"]:
        weekly[f"regime_{regime.lower()}"] = (
            weekly["dominant_regime"].fillna("Unknown") == regime
        ).astype(int)

    p85 = weekly["total_hours"].quantile(0.85)
    p95 = weekly["total_hours"].quantile(0.95)
    weekend_p75 = weekly["weekend_hours"].quantile(0.75)
    weekly["is_p85"] = (weekly["total_hours"] >= p85).astype(int)
    weekly["is_p95"] = (weekly["total_hours"] >= p95).astype(int)
    weekly["high_weekend"] = (weekly["weekend_hours"] >= weekend_p75).astype(int)
    weekly["weekend_ratio"] = weekly["weekend_hours"] / weekly["total_hours"].clip(
        lower=1
    )
    return weekly


def summarize_ttests(
    df: pd.DataFrame, group_col: str, min_group_rows: int = 8
) -> list[str]:
    lines = []
    for field in SCALE_FIELDS:
        if field not in df.columns:
            continue
        hi = df[df[group_col] == 1][field].dropna()
        lo = df[df[group_col] == 0][field].dropna()
        if len(hi) < min_group_rows or len(lo) < min_group_rows:
            continue
        _, p_value = stats.ttest_ind(hi, lo, equal_var=False)
        diff = hi.mean() - lo.mean()
        lines.append(
            (
                abs(diff),
                f"- `{field}`: {hi.mean():.2f} vs {lo.mean():.2f} "
                f"(diff {diff:+.2f}, p={p_value:.4f})",
            )
        )
    lines.sort(reverse=True)
    return [line for _, line in lines[:8]]


def text_top_categories(df: pd.DataFrame, value_col: str, top_n: int = 8) -> list[str]:
    counts = df[value_col].dropna().value_counts().head(top_n)
    return [f"- `{idx}`: {val}" for idx, val in counts.items()]


def controlled_effects(
    df: pd.DataFrame, outcome: str, controls: list[str], min_rows: int = 30
) -> list[str]:
    lines = []
    control_cols = [col for col in controls if col in df.columns]
    for field in SCALE_FIELDS:
        needed = [outcome, field] + control_cols
        sample = df[needed].dropna().copy()
        if len(sample) < min_rows or sample[field].std() < 0.05:
            continue
        X = sm.add_constant(sample[control_cols + [field]])
        model = sm.OLS(sample[outcome], X).fit()
        coef = model.params[field]
        p_value = model.pvalues[field]
        lines.append(
            (
                abs(coef),
                f"- `{field}`: coef {coef:+.2f}, p={p_value:.4f}, n={len(sample)}",
            )
        )
    lines.sort(reverse=True)
    return [line for _, line in lines[:8]]


def categorical_hours(df: pd.DataFrame, field: str) -> list[str]:
    if field not in df.columns:
        return []
    grouped = (
        df.groupby(field)["total_hours"].agg(["mean", "count"]).sort_values("mean")
    )
    grouped = grouped[grouped["count"] >= 3].head(6)
    return [
        f"- `{label}`: mean {row['mean']:.2f} hours across {int(row['count'])} weeks"
        for label, row in grouped.iterrows()
    ]


def representative_examples(df: pd.DataFrame, top: bool) -> list[str]:
    if df.empty:
        return []
    ordered = df.sort_values("total_hours", ascending=not top).head(4)
    lines = []
    for _, row in ordered.iterrows():
        lines.append(
            f"- {row['week_start'].date()}: {row['total_hours']:.1f}h, "
            f"`{row.get('dominant_mode', 'unknown')}` / "
            f"`{row.get('dominant_obstacle', 'unknown')}`. "
            f"{row['title'][:180].strip()}"
        )
    return lines


def weekly_label_summary(weekly_df: pd.DataFrame) -> str:
    lines = ["## Weekly Summary Labels"]
    lines.append(
        f"Weekly summaries joined to outcomes: {len(weekly_df)}. "
        f"P85 weeks: {int(weekly_df['is_p85'].sum())}; "
        f"P95 weeks: {int(weekly_df['is_p95'].sum())}."
    )
    lines.append("")
    lines.append("### Most common dominant obstacles")
    lines.extend(text_top_categories(weekly_df, "dominant_obstacle"))
    lines.append("")
    lines.append("### Most common dominant drivers")
    lines.extend(text_top_categories(weekly_df, "dominant_driver"))
    lines.append("")
    lines.append("### Most common dominant modes")
    lines.extend(text_top_categories(weekly_df, "dominant_mode"))
    lines.append("")
    lines.append("### What differs in P85 weeks")
    lines.extend(summarize_ttests(weekly_df, "is_p85"))
    lines.append("")
    lines.append("### What differs in P95 weeks")
    lines.extend(summarize_ttests(weekly_df, "is_p95"))
    lines.append("")
    lines.append("### Lowest-hour obstacle buckets")
    lines.extend(categorical_hours(weekly_df, "dominant_obstacle"))
    lines.append("")
    lines.append("### Highest-hour examples")
    lines.extend(representative_examples(weekly_df, top=True))
    lines.append("")
    lines.append("### Lowest-hour examples")
    lines.extend(representative_examples(weekly_df, top=False))
    return "\n".join(lines)


def controlled_summary(weekly_df: pd.DataFrame) -> str:
    controls = [
        "prev_total_hours",
        "week_ordinal",
        "regime_hive",
        "regime_mats",
        "regime_diesl",
    ]
    next_controls = [
        "total_hours",
        "week_ordinal",
        "regime_hive",
        "regime_mats",
        "regime_diesl",
    ]
    weekend_controls = [
        "weekday_hours",
        "prev_weekend_hours",
        "week_ordinal",
        "regime_hive",
        "regime_mats",
        "regime_diesl",
    ]

    lines = ["## Controlled Journal Effects"]
    lines.append(
        "These results are the more rigorous part of the journal analysis. "
        "Each label is tested after controlling for prior week level and regime "
        "instead of only comparing raw means."
    )
    lines.append("")
    lines.append("### Same-week total hours beyond prior week and regime")
    lines.extend(controlled_effects(weekly_df, "total_hours", controls))
    lines.append("")
    lines.append("### Next-week total hours beyond current week and regime")
    lines.extend(controlled_effects(weekly_df, "next_total_hours", next_controls))
    lines.append("")
    lines.append("### Weekend hours beyond weekday hours and regime")
    lines.extend(controlled_effects(weekly_df, "weekend_hours", weekend_controls))
    return "\n".join(lines)


def daily_label_summary(daily_df: pd.DataFrame, weekly_outcomes: pd.DataFrame) -> str:
    if daily_df.empty:
        return "## Daily Entry Labels\n\nNo daily labels were available."

    agg = daily_df.groupby("week_start").agg(
        {field: "mean" for field in SCALE_FIELDS if field in daily_df.columns}
    )
    joined = weekly_outcomes.merge(agg.reset_index(), on="week_start", how="inner")
    if joined.empty:
        return "## Daily Entry Labels\n\nNo joined daily entry labels were available."

    lines = ["## Daily Entry Labels"]
    lines.append(f"Joined weekly sample from daily labels: {len(joined)} weeks.")
    lines.append("")
    lines.append("### Labels associated with P85 weeks")
    lines.extend(summarize_ttests(joined, "is_p85", min_group_rows=4))
    lines.append("")
    lines.append("### Labels associated with high-work weekends")
    lines.extend(summarize_ttests(joined, "high_weekend", min_group_rows=4))
    lines.append("")
    lines.append("### Controlled effect on weekend hours")
    lines.extend(
        controlled_effects(
            joined,
            "weekend_hours",
            ["weekday_hours", "prev_weekend_hours", "week_ordinal"],
            min_rows=20,
        )
    )
    return "\n".join(lines)


def hypothesis_section(weekly_df: pd.DataFrame) -> str:
    lines = ["## Working Hypotheses"]
    lines.append(
        "These are interpretations of the joined journal + calendar/work data, "
        "not final truths."
    )
    lines.append("")

    obstacle_means = (
        weekly_df.groupby("dominant_obstacle")["total_hours"].mean().sort_values()
    )
    driver_means = (
        weekly_df.groupby("dominant_driver")["total_hours"]
        .mean()
        .sort_values(ascending=False)
    )

    lines.append("### Low-output patterns")
    for obstacle, mean_hours in obstacle_means.head(6).items():
        lines.append(f"- `{obstacle}` weeks average {mean_hours:.2f} focused hours.")
    lines.append("")
    lines.append("### High-output patterns")
    for driver, mean_hours in driver_means.head(6).items():
        lines.append(f"- `{driver}` weeks average {mean_hours:.2f} focused hours.")
    lines.append("")
    lines.append("### Synthesis")
    lines.append(
        "- The journals can test whether your worst weeks are mostly relationship/"
        "socially captured, collapse into drift/avoidance, or are genuinely low-energy."
    )
    lines.append(
        "- If external structure, accountability, and compelling problems remain "
        "positive even after controlling for prior week and regime, that supports "
        "environment design as the main lever."
    )
    lines.append(
        "- If overanalysis, avoidance, and phone/internet drift remain negative "
        "after controls, the journals are capturing something upstream of the "
        "calendar categories."
    )
    lines.append(
        "- Weekend-specific effects matter separately because your largest recoverable "
        "hour pool may still be on Saturdays and Sundays rather than ordinary weekdays."
    )
    return "\n".join(lines)


def main() -> None:
    entries = pd.read_csv(
        INPUT_DIR / "journal_entries.csv", parse_dates=["date", "week_start"]
    )
    weekly_labels = load_labels(INPUT_DIR / "weekly_summary_labels.jsonl")
    daily_labels = load_labels(INPUT_DIR / "daily_entry_labels.jsonl")
    weekly_outcomes = build_weekly_outcomes()

    weekly_entries = entries[entries["entry_type"] == "weekly_summary"].copy()
    daily_entries = entries[entries["entry_type"] == "daily_entry"].copy()

    weekly_joined = weekly_entries.merge(weekly_labels, on="entry_id", how="inner")
    weekly_joined = (
        weekly_joined.sort_values(
            ["week_start", "text_len", "date"], ascending=[True, False, False]
        )
        .drop_duplicates(subset="week_start", keep="first")
        .copy()
    )
    weekly_joined = weekly_joined.merge(weekly_outcomes, on="week_start", how="inner")
    daily_joined = daily_entries.merge(daily_labels, on="entry_id", how="inner")

    parts = [
        "# Journal Deep Dive",
        "",
        "This file joins LLM-labeled journal text to the work logs and asks what the journals add beyond the numeric trackers.",
        "",
        f"Weekly labels loaded: {len(weekly_labels)} unique entries.",
        f"Daily labels loaded: {len(daily_labels)} unique entries.",
        "",
        weekly_label_summary(weekly_joined),
        "",
        controlled_summary(weekly_joined),
        "",
        daily_label_summary(daily_joined, weekly_outcomes),
        "",
        hypothesis_section(weekly_joined),
    ]

    OUTPUT_SUMMARY.write_text("\n".join(parts) + "\n")
    print(f"Wrote {OUTPUT_SUMMARY}")
    print(f"Weekly labels joined: {len(weekly_joined)}")
    print(f"Daily labels joined: {len(daily_joined)}")


if __name__ == "__main__":
    main()
