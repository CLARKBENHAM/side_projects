"""Weekend-only journal/work analysis."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import statsmodels.api as sm
from scipy import stats

from productivity_analysis import DAILY_SUMMARY_CSV, load_daily_summary_full

BASE_DIR = Path(__file__).resolve().parent
INPUT_DIR = BASE_DIR / "journal_outputs"
OUTPUT_SUMMARY = BASE_DIR / "WEEKEND_MODEL_SUMMARY.md"

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
    if not path.exists():
        return pd.DataFrame(columns=["entry_id"])
    rows = [json.loads(line) for line in path.open()]
    if not rows:
        return pd.DataFrame(columns=["entry_id"])
    return pd.DataFrame(rows).drop_duplicates(subset="entry_id", keep="last")


def build_daily_base() -> pd.DataFrame:
    daily = (
        load_daily_summary_full(DAILY_SUMMARY_CSV)
        .sort_values("date")
        .reset_index(drop=True)
    )
    full_dates = pd.DataFrame(
        {"date": pd.date_range(daily["date"].min(), daily["date"].max(), freq="D")}
    )
    daily = full_dates.merge(daily, on="date", how="left")
    daily["Hours Working"] = daily["Hours Working"].fillna(0.0)
    daily["regime"] = daily["regime"].ffill().bfill().fillna("Unknown")
    daily["day_of_week"] = daily["date"].dt.dayofweek
    daily["is_weekend"] = (daily["day_of_week"] >= 5).astype(int)
    daily["is_sunday"] = (daily["day_of_week"] == 6).astype(int)
    daily["week_start"] = daily["date"] - pd.to_timedelta(
        daily["day_of_week"], unit="D"
    )
    daily["prev_day_hours"] = daily["Hours Working"].shift(1)
    daily["real_workday"] = (daily["Hours Working"] >= 2).astype(int)

    weekday_hours = (
        daily[daily["day_of_week"] < 5]
        .groupby("week_start")["Hours Working"]
        .sum()
        .rename("weekday_hours")
    )
    weekend_hours = (
        daily[daily["day_of_week"] >= 5]
        .groupby("week_start")["Hours Working"]
        .sum()
        .rename("weekend_hours")
    )
    weekly = pd.concat([weekday_hours, weekend_hours], axis=1).reset_index()
    weekly = weekly.fillna({"weekday_hours": 0.0, "weekend_hours": 0.0})
    weekly = weekly.sort_values("week_start").reset_index(drop=True)
    weekly["prev_weekend_hours"] = weekly["weekend_hours"].shift(1)
    weekly["week_ordinal"] = range(len(weekly))

    regime = (
        daily.groupby("week_start")["regime"]
        .agg(lambda x: x.dropna().mode().iloc[0] if not x.dropna().empty else "Unknown")
        .rename("dominant_regime")
        .reset_index()
    )
    weekly = weekly.merge(regime, on="week_start", how="left")
    for label, value in [
        ("regime_hive", "Hive"),
        ("regime_mats", "Mats"),
        ("regime_diesl", "diesl"),
    ]:
        weekly[label] = (weekly["dominant_regime"] == value).astype(int)

    daily = daily.merge(
        weekly[
            [
                "week_start",
                "weekday_hours",
                "weekend_hours",
                "prev_weekend_hours",
                "week_ordinal",
                "dominant_regime",
                "regime_hive",
                "regime_mats",
                "regime_diesl",
            ]
        ],
        on="week_start",
        how="left",
    )
    return daily


def build_labeled_weekend_days() -> pd.DataFrame:
    entries = pd.read_csv(
        INPUT_DIR / "journal_entries.csv", parse_dates=["date", "week_start"]
    )
    labels = load_labels(INPUT_DIR / "daily_entry_labels.jsonl")
    daily = build_daily_base()

    labeled = entries[entries["entry_type"] == "daily_entry"].merge(
        labels, on="entry_id", how="inner"
    )
    labeled = labeled.merge(
        daily[
            [
                "date",
                "Hours Working",
                "real_workday",
                "day_of_week",
                "is_sunday",
                "week_start",
                "weekday_hours",
                "weekend_hours",
                "prev_weekend_hours",
                "prev_day_hours",
                "week_ordinal",
                "dominant_regime",
                "regime_hive",
                "regime_mats",
                "regime_diesl",
            ]
        ],
        on=["date", "week_start"],
        how="inner",
    )
    return labeled[labeled["day_of_week"] >= 5].copy()


def text_top_categories(df: pd.DataFrame, field: str, top_n: int = 6) -> list[str]:
    counts = df[field].dropna().value_counts().head(top_n)
    return [f"- `{idx}`: {val}" for idx, val in counts.items()]


def summarize_ttests(
    df: pd.DataFrame, outcome_flag: str, min_group_rows: int = 10
) -> list[str]:
    lines = []
    for field in SCALE_FIELDS:
        hi = df[df[outcome_flag] == 1][field].dropna()
        lo = df[df[outcome_flag] == 0][field].dropna()
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


def controlled_effects(
    df: pd.DataFrame, outcome: str, controls: list[str], min_rows: int = 40
) -> list[str]:
    lines = []
    control_cols = [col for col in controls if col in df.columns]
    for field in SCALE_FIELDS:
        sample = df[[outcome, field] + control_cols].dropna().copy()
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


def weekend_day_summary(df: pd.DataFrame) -> str:
    df = df.copy()
    high_day_threshold = df["Hours Working"].quantile(0.75)
    df["high_weekend_day"] = (df["Hours Working"] >= high_day_threshold).astype(int)

    lines = ["## Weekend-Day Model"]
    lines.append(
        f"Labeled weekend days: {len(df)} across {df['week_start'].nunique()} weeks. "
        f"Median weekend-day hours: {df['Hours Working'].median():.2f}. "
        f"P75 threshold: {high_day_threshold:.2f} hours."
    )
    lines.append("")
    lines.append("### Most common weekend dominant obstacles")
    lines.extend(text_top_categories(df, "dominant_obstacle"))
    lines.append("")
    lines.append("### Most common weekend dominant modes")
    lines.extend(text_top_categories(df, "dominant_mode"))
    lines.append("")
    lines.append("### Raw differences for high-work weekend days")
    lines.extend(summarize_ttests(df, "high_weekend_day"))
    lines.append("")
    lines.append("### Controlled effects on weekend-day hours")
    lines.extend(
        controlled_effects(
            df,
            "Hours Working",
            [
                "is_sunday",
                "weekday_hours",
                "prev_day_hours",
                "prev_weekend_hours",
                "week_ordinal",
                "regime_hive",
                "regime_mats",
                "regime_diesl",
            ],
        )
    )
    lines.append("")
    lines.append("### Controlled effects on probability of a real weekend workday")
    lines.extend(
        controlled_effects(
            df,
            "real_workday",
            [
                "is_sunday",
                "weekday_hours",
                "prev_day_hours",
                "prev_weekend_hours",
                "week_ordinal",
                "regime_hive",
                "regime_mats",
                "regime_diesl",
            ],
        )
    )
    return "\n".join(lines)


def weekend_week_summary(df: pd.DataFrame) -> str:
    weekend_week = (
        df.groupby("week_start")
        .agg(
            labeled_weekend_days=("date", "nunique"),
            weekend_hours=("weekend_hours", "first"),
            weekday_hours=("weekday_hours", "first"),
            prev_weekend_hours=("prev_weekend_hours", "first"),
            week_ordinal=("week_ordinal", "first"),
            regime_hive=("regime_hive", "first"),
            regime_mats=("regime_mats", "first"),
            regime_diesl=("regime_diesl", "first"),
            **{field: (field, "mean") for field in SCALE_FIELDS},
        )
        .reset_index()
    )
    high_weekend_threshold = weekend_week["weekend_hours"].quantile(0.75)
    weekend_week["high_weekend"] = (
        weekend_week["weekend_hours"] >= high_weekend_threshold
    ).astype(int)

    lines = ["## Whole-Weekend Model"]
    lines.append(
        f"Labeled weekends: {len(weekend_week)}. "
        f"Median weekend hours: {weekend_week['weekend_hours'].median():.2f}. "
        f"P75 threshold: {high_weekend_threshold:.2f} hours."
    )
    lines.append("")
    lines.append("### Raw differences for high-work weekends")
    lines.extend(summarize_ttests(weekend_week, "high_weekend", min_group_rows=6))
    lines.append("")
    lines.append("### Controlled effects on total weekend hours")
    lines.extend(
        controlled_effects(
            weekend_week,
            "weekend_hours",
            [
                "weekday_hours",
                "prev_weekend_hours",
                "week_ordinal",
                "regime_hive",
                "regime_mats",
                "regime_diesl",
            ],
            min_rows=25,
        )
    )
    return "\n".join(lines)


def write_takeaways(df: pd.DataFrame) -> str:
    weekend_week = (
        df.groupby("week_start")
        .agg(
            weekend_hours=("weekend_hours", "first"),
            **{field: (field, "mean") for field in SCALE_FIELDS},
        )
        .reset_index()
    )
    high_threshold = weekend_week["weekend_hours"].quantile(0.75)
    weekend_week["high_weekend"] = (
        weekend_week["weekend_hours"] >= high_threshold
    ).astype(int)
    hi = weekend_week[weekend_week["high_weekend"] == 1]
    lo = weekend_week[weekend_week["high_weekend"] == 0]
    zero_weekends = int((weekend_week["weekend_hours"] == 0).sum())

    lines = ["## Weekend Takeaways"]
    lines.append(
        "This is still a partial-label sample, so it should be read as directional "
        "rather than final. But it is already useful for isolating weekends as "
        "their own process."
    )
    lines.append("")
    lines.append(
        f"- Weekend collapse is real. In the current labeled sample, {zero_weekends} "
        f"of {len(weekend_week)} weekends have exactly 0 tracked work hours."
    )
    lines.append(
        f"- High-work weekends are not the calm ones. Raw mood valence is "
        f"{hi['mood_valence'].mean():.2f} on high-work weekends vs "
        f"{lo['mood_valence'].mean():.2f} otherwise."
    )
    lines.append(
        f"- High-work weekends show more `compelling_problem` "
        f"({hi['compelling_problem'].mean():.2f} vs {lo['compelling_problem'].mean():.2f}) "
        f"and more `relationship_closeness` "
        f"({hi['relationship_closeness'].mean():.2f} vs {lo['relationship_closeness'].mean():.2f})."
    )
    if not weekend_week.empty:
        top_field = max(
            SCALE_FIELDS,
            key=lambda field: abs(
                weekend_week[field].corr(weekend_week["weekend_hours"])
            ),
        )
        corr = weekend_week[top_field].corr(weekend_week["weekend_hours"])
        lines.append(
            f"- In the current labeled weekend sample, `{top_field}` has the largest "
            f"simple correlation with weekend hours ({corr:+.2f})."
        )
    lines.append(
        "- The practical implication is that weekends likely need their own design: "
        "a non-negotiable first block, explicit social windows, and stronger rules "
        "for late-night internet/alcohol drift. Weekday discipline alone will not fix this."
    )
    return "\n".join(lines)


def main() -> None:
    weekend_days = build_labeled_weekend_days()
    parts = [
        "# Weekend Model Summary",
        "",
        "This file isolates weekends as their own system rather than treating them as noisier weekdays.",
        "",
        weekend_day_summary(weekend_days),
        "",
        weekend_week_summary(weekend_days),
        "",
        write_takeaways(weekend_days),
    ]
    OUTPUT_SUMMARY.write_text("\n".join(parts) + "\n")
    print(f"Wrote {OUTPUT_SUMMARY}")
    print(f"Labeled weekend days: {len(weekend_days)}")
    print(f"Labeled weekend weeks: {weekend_days['week_start'].nunique()}")


if __name__ == "__main__":
    main()
