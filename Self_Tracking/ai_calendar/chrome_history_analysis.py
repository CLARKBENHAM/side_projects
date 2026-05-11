"""Chrome takeout parsing and weekend-focused analysis."""

from __future__ import annotations

import json
from pathlib import Path
from urllib.parse import urlparse

import pandas as pd
import statsmodels.api as sm

from productivity_analysis import DAILY_SUMMARY_CSV, load_daily_summary_full

BASE_DIR = Path(__file__).resolve().parent
TAKEOUT_PATH = BASE_DIR.parent.parent / "data" / "Takeout 7" / "Chrome" / "History.json"
OUTPUT_FEATURES = BASE_DIR / "chrome_daily_features.csv"
OUTPUT_SUMMARY = BASE_DIR / "CHROME_WEEKEND_SUMMARY.md"
LOCAL_TZ = "America/Los_Angeles"

CATEGORY_PATTERNS = {
    "work_ai": [
        "github.com",
        "app.diesl.ai",
        "portal.azure.com",
        "docs.google.com",
        "drive.google.com",
        "chatgpt.com",
        "claude.ai",
        "gemini.google.com",
        "platform.openai.com",
        "huggingface.co",
        "learn.microsoft.com",
        "colab.google",
        "console.cloud.google.com",
    ],
    "social": [
        "x.com",
        "t.co",
        "instagram.com",
        "linkedin.com",
        "discord.com",
        "facebook.com",
        "messenger.com",
    ],
    "reading": [
        "substack.com",
        "lesswrong.com",
        "marginalrevolution.com",
        "wikipedia.org",
        "archive.is",
        "archive.org",
        "libgen.",
        "slatestarcodex.com",
        "astralcodexten.com",
    ],
    "entertainment": [
        "youtube.com",
        "amctheatres.com",
        "thefarside.com",
        "smbc-comics.com",
        "imdb.com",
        "netflix.com",
        "spotify.com",
    ],
    "commerce": [
        "amazon.com",
        "goodrx.com",
        "fedex.com",
        "office.fedex.com",
        "uber.com",
        "lyft.com",
    ],
    "admin": [
        "calendar.google.com",
        "mail.google.com",
        "accounts.google.com",
        "play.google.com",
        "myactivity.google.com",
        "earth.google.com",
        "dropbox.com",
        "login.microsoftonline.com",
    ],
}


def classify_visit(host: str, url: str) -> str:
    host = host.lower()
    url = url.lower()
    if host == "www.google.com" and "/search" in url:
        return "search"

    for category, patterns in CATEGORY_PATTERNS.items():
        if any(pattern in host or pattern in url for pattern in patterns):
            return category

    if host.endswith(".substack.com"):
        return "reading"
    if host.endswith(".x.com"):
        return "social"
    if "google.com" in host:
        return "admin"
    return "other"


def load_chrome_history() -> pd.DataFrame:
    payload = json.loads(TAKEOUT_PATH.read_text())
    df = pd.DataFrame(payload["Browser History"])
    df["dt"] = pd.to_datetime(df["time_usec"].astype("int64"), unit="us", utc=True)
    df["dt"] = df["dt"].dt.tz_convert(LOCAL_TZ)
    df["date"] = df["dt"].dt.floor("D").dt.tz_localize(None)
    df["hour"] = df["dt"].dt.hour
    df["day_of_week"] = df["dt"].dt.dayofweek
    df["host"] = df["url"].map(lambda x: urlparse(x).netloc.lower())
    df["category"] = df.apply(
        lambda row: classify_visit(str(row["host"]), str(row["url"])), axis=1
    )
    return df


def make_daily_features(chrome: pd.DataFrame) -> pd.DataFrame:
    chrome = chrome.copy()
    chrome["visit_count"] = 1

    base = chrome.groupby("date").agg(total_visits=("visit_count", "sum")).reset_index()

    category_daily = (
        chrome.pivot_table(
            index="date",
            columns="category",
            values="visit_count",
            aggfunc="sum",
            fill_value=0,
        )
        .reset_index()
        .rename_axis(None, axis=1)
    )
    category_daily.columns = [
        "date" if col == "date" else f"{col}_visits" for col in category_daily.columns
    ]
    base = base.merge(category_daily, on="date", how="left")

    chrome["morning_flag"] = chrome["hour"].between(6, 11).astype(int)
    chrome["late_flag"] = ((chrome["hour"] >= 23) | (chrome["hour"] <= 5)).astype(int)

    morning = (
        chrome[chrome["morning_flag"] == 1]
        .pivot_table(
            index="date",
            columns="category",
            values="visit_count",
            aggfunc="sum",
            fill_value=0,
        )
        .reset_index()
        .rename_axis(None, axis=1)
    )
    morning.columns = [
        "date" if col == "date" else f"morning_{col}_visits" for col in morning.columns
    ]
    morning_total = (
        chrome[chrome["morning_flag"] == 1]
        .groupby("date")
        .agg(morning_total_visits=("visit_count", "sum"))
        .reset_index()
    )

    prev_night = chrome[chrome["late_flag"] == 1].copy()
    prev_night["target_date"] = prev_night["date"]
    prev_night.loc[prev_night["hour"] >= 23, "target_date"] = prev_night.loc[
        prev_night["hour"] >= 23, "date"
    ] + pd.Timedelta(days=1)

    prev_night_cat = (
        prev_night.pivot_table(
            index="target_date",
            columns="category",
            values="visit_count",
            aggfunc="sum",
            fill_value=0,
        )
        .reset_index()
        .rename_axis(None, axis=1)
    )
    prev_night_cat.columns = [
        "date" if col == "target_date" else f"prev_night_{col}_visits"
        for col in prev_night_cat.columns
    ]
    prev_night_total = (
        prev_night.groupby("target_date")
        .agg(prev_night_total_visits=("visit_count", "sum"))
        .reset_index()
        .rename(columns={"target_date": "date"})
    )

    features = base.merge(morning_total, on="date", how="left")
    features = features.merge(morning, on="date", how="left")
    features = features.merge(prev_night_total, on="date", how="left")
    features = features.merge(prev_night_cat, on="date", how="left")
    features = features.fillna(0)
    return features


def build_weekend_sample() -> pd.DataFrame:
    chrome = load_chrome_history()
    features = make_daily_features(chrome)
    features.to_csv(OUTPUT_FEATURES, index=False)

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
        .reset_index()
    )
    weekend_hours = (
        daily[daily["day_of_week"] >= 5]
        .groupby("week_start")["Hours Working"]
        .sum()
        .rename("weekend_hours")
        .reset_index()
    )
    weekly = weekday_hours.merge(weekend_hours, on="week_start", how="outer").fillna(0)
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

    sample = daily.merge(weekly, on="week_start", how="left").merge(
        features, on="date", how="left"
    )
    sample = sample[
        (sample["date"] >= features["date"].min()) & (sample["is_weekend"] == 1)
    ].copy()
    sample = sample.fillna(0)
    return sample


def fit_lines(
    df: pd.DataFrame, outcome: str, features: list[str], controls: list[str]
) -> list[str]:
    lines = []
    for feature in features:
        if feature not in df.columns:
            continue
        sample = df[[outcome] + controls + [feature]].dropna().copy()
        if len(sample) < 30 or sample[feature].std() < 0.05:
            continue
        X = sm.add_constant(sample[controls + [feature]])
        model = sm.OLS(sample[outcome], X).fit()
        lines.append(
            (
                abs(model.params[feature]),
                f"- `{feature}`: coef {model.params[feature]:+.2f}, p={model.pvalues[feature]:.4f}, n={len(sample)}",
            )
        )
    lines.sort(reverse=True)
    return [line for _, line in lines[:10]]


def build_summary(sample: pd.DataFrame) -> str:
    sample = sample.copy()
    high_threshold = sample["Hours Working"].quantile(0.75)
    sample["high_weekend_day"] = (sample["Hours Working"] >= high_threshold).astype(int)
    weekend_zero = int((sample["Hours Working"] == 0).sum())

    predictive_features = [
        "prev_night_total_visits",
        "prev_night_social_visits",
        "prev_night_reading_visits",
        "prev_night_entertainment_visits",
        "prev_night_work_ai_visits",
        "prev_night_search_visits",
    ]
    morning_features = [
        "morning_total_visits",
        "morning_social_visits",
        "morning_reading_visits",
        "morning_entertainment_visits",
        "morning_work_ai_visits",
        "morning_search_visits",
    ]
    same_day_features = [
        "total_visits",
        "social_visits",
        "reading_visits",
        "entertainment_visits",
        "work_ai_visits",
        "search_visits",
        "admin_visits",
    ]
    controls = [
        "is_sunday",
        "prev_day_hours",
        "weekday_hours",
        "prev_weekend_hours",
        "week_ordinal",
        "regime_hive",
        "regime_mats",
        "regime_diesl",
    ]

    weekend_week = (
        sample.groupby("week_start")
        .agg(
            weekend_hours=("Hours Working", "sum"),
            weekend_total_visits=("total_visits", "sum"),
            weekend_social_visits=("social_visits", "sum"),
            weekend_reading_visits=("reading_visits", "sum"),
            weekend_entertainment_visits=("entertainment_visits", "sum"),
            weekend_work_ai_visits=("work_ai_visits", "sum"),
            weekend_morning_total_visits=("morning_total_visits", "sum"),
            weekend_prev_night_total_visits=("prev_night_total_visits", "sum"),
            weekday_hours=("weekday_hours", "first"),
            prev_weekend_hours=("prev_weekend_hours", "first"),
            week_ordinal=("week_ordinal", "first"),
            regime_hive=("regime_hive", "first"),
            regime_mats=("regime_mats", "first"),
            regime_diesl=("regime_diesl", "first"),
        )
        .reset_index()
    )
    weekend_week["high_weekend"] = (
        weekend_week["weekend_hours"] >= weekend_week["weekend_hours"].quantile(0.75)
    ).astype(int)

    parts = [
        "# Chrome Weekend Summary",
        "",
        "This file uses Chrome takeout history to analyze weekend drift and weekend work in the overlap period with the daily work log.",
        "",
        f"Chrome history range: {sample['date'].min().date()} to {sample['date'].max().date()}",
        f"Weekend days in overlap sample: {len(sample)}",
        f"Weekend days with zero tracked work: {weekend_zero}",
        f"Weekend-day P75 work threshold: {high_threshold:.2f} hours",
        "",
        "## Predictive Features",
        "These use only previous-night browsing, so they are the cleanest indicators of whether a weekend day is being set up for work or drift.",
        "",
        "### Previous-night effects on weekend-day hours",
        *fit_lines(sample, "Hours Working", predictive_features, controls),
        "",
        "### Previous-night effects on whether the weekend day becomes a real workday",
        *fit_lines(sample, "real_workday", predictive_features, controls),
        "",
        "## Morning Features",
        "These are more ambiguous causally, but they are still useful as early-day state indicators.",
        "",
        "### Morning effects on weekend-day hours",
        *fit_lines(sample, "Hours Working", morning_features, controls),
        "",
        "### Morning effects on whether the weekend day becomes a real workday",
        *fit_lines(sample, "real_workday", morning_features, controls),
        "",
        "## Descriptive Same-Day Browsing",
        "These are descriptive rather than causal. They help identify what the day looks like when it has already gone well or badly.",
        "",
        "### Same-day browsing associated with weekend-day hours",
        *fit_lines(sample, "Hours Working", same_day_features, controls),
        "",
        "## Whole-Weekend View",
        f"Labeled weekends in overlap sample: {len(weekend_week)}",
        f"Whole-weekend P75 work threshold: {weekend_week['weekend_hours'].quantile(0.75):.2f} hours",
        "",
        "### Weekend-level browsing effects on total weekend hours",
        *fit_lines(
            weekend_week,
            "weekend_hours",
            [
                "weekend_prev_night_total_visits",
                "weekend_morning_total_visits",
                "weekend_social_visits",
                "weekend_reading_visits",
                "weekend_entertainment_visits",
                "weekend_work_ai_visits",
                "weekend_total_visits",
            ],
            [
                "weekday_hours",
                "prev_weekend_hours",
                "week_ordinal",
                "regime_hive",
                "regime_mats",
                "regime_diesl",
            ],
        ),
        "",
        "## Interpretation",
        f"- Weekend collapse is large even in the Chrome overlap period: {weekend_zero} of {len(sample)} weekend days have zero tracked work.",
        "- The cleanest early-day signal is category mix, not raw browser volume.",
        "- Morning `work_ai` activity is strongly positive, while morning `reading` activity is significantly negative. That is consistent with 'startup drift into blogs/reading' being a real weekend failure mode.",
        "- Previous-night `work_ai` activity is mildly positive, while previous-night social/reading noise is weak. The data does not support a simple 'any late-night browsing kills tomorrow' story.",
        "- Same-day total browsing is positive because productive days also involve many browser visits. The important distinction is what those visits are for.",
        "- Chrome should become much more useful once combined with your journal labels and, later, Chrome domain-level exports for tabs/session duration if Google includes them.",
    ]
    return "\n".join(parts) + "\n"


def main() -> None:
    sample = build_weekend_sample()
    OUTPUT_SUMMARY.write_text(build_summary(sample))
    print(f"Wrote {OUTPUT_SUMMARY}")
    print(f"Wrote {OUTPUT_FEATURES}")
    print(f"Weekend sample rows: {len(sample)}")


if __name__ == "__main__":
    main()
