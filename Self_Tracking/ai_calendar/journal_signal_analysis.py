"""Deterministic journal theme extraction over the full parsed corpus."""

from __future__ import annotations

import re
from pathlib import Path

import pandas as pd
import statsmodels.api as sm
from scipy import stats

BASE_DIR = Path(__file__).resolve().parent
INPUT_DIR = BASE_DIR / "journal_outputs"
OUTPUT_SUMMARY = BASE_DIR / "JOURNAL_SIGNAL_SUMMARY.md"
OUTPUT_WEEKLY = INPUT_DIR / "journal_theme_weekly_features.csv"
OUTPUT_DAILY = INPUT_DIR / "journal_theme_daily_weekly_features.csv"

THEME_PATTERNS = {
    "relationship": [
        r"\bamelia\b",
        r"\brelationship\b",
        r"\bdate\b",
        r"\bcuddle\w*\b",
        r"\bsex\b",
    ],
    "conflict": [
        r"\bfight\w*\b",
        r"\bargu\w*\b",
        r"\bbicker\w*\b",
        r"\bbreak ?up\b",
        r"\bresent\w*\b",
        r"\bhate\b",
    ],
    "alcohol": [
        r"\bdrink\w*\b",
        r"\bdrank\b",
        r"\bdrunk\b",
        r"\bbooze\b",
        r"\btipsy\b",
        r"\bhungover\b",
        r"\bbender\b",
        r"\bpurge\w*\b",
    ],
    "porn": [
        r"\bporn\b",
        r"\bjack(?:ed|ing)?\b",
        r"\bmasterbat\w*\b",
        r"\bmasturbat\w*\b",
    ],
    "internet_drift": [
        r"\bblog\w*\b",
        r"\bsubstack\b",
        r"\btwitter\b",
        r"\byoutube\b",
        r"\bscroll\w*\b",
        r"\bphone\b",
        r"\binternet\b",
    ],
    "social": [
        r"\brichard\b",
        r"\bfriends?\b",
        r"\bfamily\b",
        r"\bvisit\w*\b",
        r"\btrip\b",
        r"\bmovie\w*\b",
        r"\bvacation\b",
    ],
    "work": [
        r"\bwork\w*\b",
        r"\bcode\w*\b",
        r"\bproject\w*\b",
        r"\bdeadline\w*\b",
        r"\bdeliver\w*\b",
        r"\binterview\w*\b",
        r"\bapply\w*\b",
        r"\bleetcode\b",
    ],
    "structure": [
        r"\bwework\b",
        r"\bcowork\w*\b",
        r"\bschedule\w*\b",
        r"\broutine\b",
        r"\balarm\b",
        r"\bchecklist\b",
        r"\bmorning\b",
        r"\bnight before\b",
    ],
    "analysis": [
        r"\banalys\w*\b",
        r"\breflect\w*\b",
        r"\brealiz\w*\b",
        r"\blesson\w*\b",
        r"\blife plan\b",
        r"\bre-?plan\w*\b",
        r"\bshould\b",
    ],
    "health": [
        r"\bsick\b",
        r"\bill\w*\b",
        r"\bpain\b",
        r"\btired\b",
        r"\bsleep\b",
        r"\bnap\w*\b",
        r"\bthroat\b",
        r"\bhip\b",
    ],
    "exercise": [
        r"\bgym\b",
        r"\blift\w*\b",
        r"\brun\w*\b",
        r"\bworkout\w*\b",
        r"\bcardio\b",
    ],
    "positive_affect": [
        r"\bhappy\b",
        r"\bjoy\b",
        r"\bgrateful\w*\b",
        r"\bcontent\b",
        r"\bgreat\b",
        r"\bpleasant\b",
        r"\bcute\b",
    ],
    "negative_affect": [
        r"\bdepress\w*\b",
        r"\banxious?\b",
        r"\bcry\w*\b",
        r"\bterrible\b",
        r"\bpathetic\b",
        r"\bloser\b",
        r"\blivid\b",
        r"\bsad\b",
    ],
}

THEME_LABELS = {
    "relationship_any": "Relationship mentions",
    "conflict_any": "Conflict language",
    "relationship_conflict_any": "Relationship conflict",
    "relationship_positive_any": "Relationship positive",
    "alcohol_any": "Alcohol / purge",
    "porn_any": "Porn / masturbation",
    "internet_drift_any": "Blogs / phone / scrolling",
    "drift_any": "Any drift theme",
    "social_any": "Friends / travel / movies",
    "work_any": "Work / projects",
    "structure_any": "Structure / WeWork / routine",
    "work_structure_any": "Work plus structure",
    "analysis_any": "Reflection / planning",
    "analysis_negative_any": "Reflection plus negative affect",
    "health_any": "Health / sleep / pain",
    "exercise_any": "Gym / running",
    "positive_affect_any": "Positive affect",
    "negative_affect_any": "Negative affect",
}

ANALYSIS_FEATURES = list(THEME_LABELS)
CONTROL_FEATURES = [
    "prev_total_hours",
    "week_ordinal",
    "regime_hive",
    "regime_mats",
    "regime_diesl",
]
WEEKEND_CONTROL_FEATURES = [
    "weekday_hours",
    "prev_weekend_hours",
    "week_ordinal",
    "regime_hive",
    "regime_mats",
    "regime_diesl",
]


def count_theme_hits(text: str, patterns: list[str]) -> int:
    lowered = text.lower()
    return sum(len(re.findall(pattern, lowered)) for pattern in patterns)


def add_theme_features(entries: pd.DataFrame) -> pd.DataFrame:
    enriched = entries.copy()
    enriched["word_count"] = (
        enriched["text"].fillna("").astype(str).str.findall(r"\b\w+\b").str.len()
    )

    for theme, patterns in THEME_PATTERNS.items():
        count_col = f"{theme}_count"
        any_col = f"{theme}_any"
        enriched[count_col] = (
            enriched["text"]
            .fillna("")
            .astype(str)
            .map(lambda text: count_theme_hits(text, patterns))
        )
        enriched[any_col] = (enriched[count_col] > 0).astype(int)

    enriched["relationship_conflict_any"] = (
        enriched["relationship_any"] & enriched["conflict_any"]
    ).astype(int)
    enriched["relationship_positive_any"] = (
        enriched["relationship_any"] & enriched["positive_affect_any"]
    ).astype(int)
    enriched["drift_any"] = (
        enriched[["alcohol_any", "porn_any", "internet_drift_any"]].max(axis=1)
    ).astype(int)
    enriched["work_structure_any"] = (
        enriched["work_any"] & enriched["structure_any"]
    ).astype(int)
    enriched["analysis_negative_any"] = (
        enriched["analysis_any"] & enriched["negative_affect_any"]
    ).astype(int)
    return enriched


def summarize_presence(
    df: pd.DataFrame, outcome: str, min_present: int = 8
) -> list[tuple[float, str]]:
    lines: list[tuple[float, str]] = []
    for feature in ANALYSIS_FEATURES:
        present = df[df[feature] == 1][outcome].dropna()
        absent = df[df[feature] == 0][outcome].dropna()
        if len(present) < min_present or len(absent) < min_present:
            continue
        _, p_value = stats.ttest_ind(present, absent, equal_var=False)
        diff = present.mean() - absent.mean()
        lines.append(
            (
                abs(diff),
                f"- {THEME_LABELS[feature]}: {present.mean():.2f} vs {absent.mean():.2f} "
                f"(diff {diff:+.2f}, p={p_value:.4f}, n={len(present)}/{len(absent)})",
            )
        )
    lines.sort(reverse=True)
    return lines


def summarize_controlled(
    df: pd.DataFrame, outcome: str, controls: list[str], min_rows: int = 40
) -> list[tuple[float, str]]:
    lines: list[tuple[float, str]] = []
    control_cols = [column for column in controls if column in df.columns]
    for feature in ANALYSIS_FEATURES:
        sample = df[[outcome, feature] + control_cols].dropna().copy()
        if len(sample) < min_rows or sample[feature].sum() < 8:
            continue
        X = sm.add_constant(sample[control_cols + [feature]], has_constant="add")
        model = sm.OLS(sample[outcome], X).fit(cov_type="HC3")
        lines.append(
            (
                abs(model.params[feature]),
                f"- {THEME_LABELS[feature]}: coef {model.params[feature]:+.2f}, "
                f"p={model.pvalues[feature]:.4f}, n={len(sample)}",
            )
        )
    lines.sort(reverse=True)
    return lines


def feature_prevalence(df: pd.DataFrame) -> list[str]:
    lines = []
    for feature in ANALYSIS_FEATURES:
        prevalence = df[feature].mean()
        lines.append(
            (
                prevalence,
                f"- {THEME_LABELS[feature]}: {int(df[feature].sum())} of {len(df)} "
                f"weeks ({prevalence:.1%})",
            )
        )
    lines.sort(reverse=True)
    return [line for _, line in lines[:10]]


def build_weekly_summary_frame(entries: pd.DataFrame) -> pd.DataFrame:
    weekly = entries[entries["entry_type"] == "weekly_summary"].copy()
    weekly = (
        weekly.sort_values(
            ["week_start", "text_len", "date"], ascending=[True, False, False]
        )
        .drop_duplicates(subset="week_start", keep="first")
        .reset_index(drop=True)
    )
    return weekly


def build_daily_week_frame(entries: pd.DataFrame) -> pd.DataFrame:
    daily = entries[entries["entry_type"] == "daily_entry"].copy()
    agg_map = {
        "entry_id": "nunique",
        "word_count": "sum",
    }
    agg_map.update({feature: "max" for feature in ANALYSIS_FEATURES})
    daily_week = (
        daily.groupby("week_start")
        .agg(agg_map)
        .rename(columns={"entry_id": "daily_entry_count"})
        .reset_index()
    )
    return daily_week


def build_summary(weekly_summary: pd.DataFrame, daily_week: pd.DataFrame) -> str:
    weekly_lines = summarize_presence(weekly_summary, "total_hours")
    weekly_controlled = summarize_controlled(
        weekly_summary, "total_hours", CONTROL_FEATURES
    )
    weekly_weekend = summarize_controlled(
        weekly_summary, "weekend_hours", WEEKEND_CONTROL_FEATURES
    )
    daily_lines = summarize_presence(daily_week, "total_hours")
    daily_weekend = summarize_controlled(
        daily_week, "weekend_hours", WEEKEND_CONTROL_FEATURES
    )

    parts = [
        "# Journal Signal Summary",
        "",
        "This pass extracts deterministic text themes from the full parsed journal corpus, without relying on partial LLM labels.",
        "",
        f"Weekly summaries joined to work outcomes: {len(weekly_summary)}",
        f"Daily-entry weeks joined to work outcomes: {len(daily_week)}",
        "",
        "## Most Common Weekly-Summary Themes",
        *feature_prevalence(weekly_summary),
        "",
        "## Weekly Summary Raw Differences",
        "These compare weeks where a theme appears anywhere in the weekly summary versus weeks where it does not.",
        "",
        *[line for _, line in weekly_lines[:8]],
        "",
        "## Weekly Summary Controlled Effects",
        "These condition on prior-week hours, week order, and regime.",
        "",
        *[line for _, line in weekly_controlled[:8]],
        "",
        "## Weekly Summary Weekend Effects",
        "These test whether the weekly-summary text says anything about weekend hours beyond weekday load and recent weekend momentum.",
        "",
        *[line for _, line in weekly_weekend[:8]],
        "",
        "## Daily Entry Week-Level Signals",
        "These aggregate daily-entry themes to the week level using any-mention flags.",
        "",
        *[line for _, line in daily_lines[:8]],
        "",
        "## Daily Entry Weekend Effects",
        "",
        *[line for _, line in daily_weekend[:8]],
        "",
        "## Read",
        "- This analysis covers the full parsed corpus, so it is less vulnerable to the partial-label coverage problem in the older journal summaries.",
        "- Relationship, drift, structure, and reflection themes are now measured directly from raw text rather than only through the Gemini-coded subset.",
        "- These are still observational text features, not causal levers, but they are a cleaner basis for deciding which themes deserve deeper follow-up.",
    ]
    return "\n".join(parts) + "\n"


def load_weekly_outcomes() -> pd.DataFrame:
    try:
        from .analyze_journal_labels import build_weekly_outcomes
    except ImportError:
        from analyze_journal_labels import build_weekly_outcomes

    return build_weekly_outcomes()


def main() -> None:
    entries = pd.read_csv(
        INPUT_DIR / "journal_entries.csv", parse_dates=["date", "week_start"]
    )
    entries = add_theme_features(entries)
    outcomes = load_weekly_outcomes()

    weekly_summary = build_weekly_summary_frame(entries).merge(
        outcomes, on="week_start", how="inner"
    )
    daily_week = build_daily_week_frame(entries).merge(
        outcomes, on="week_start", how="inner"
    )

    weekly_summary.to_csv(OUTPUT_WEEKLY, index=False)
    daily_week.to_csv(OUTPUT_DAILY, index=False)
    OUTPUT_SUMMARY.write_text(build_summary(weekly_summary, daily_week))

    print(f"Wrote {OUTPUT_WEEKLY}")
    print(f"Wrote {OUTPUT_DAILY}")
    print(f"Wrote {OUTPUT_SUMMARY}")


if __name__ == "__main__":
    main()
