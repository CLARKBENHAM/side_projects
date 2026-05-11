"""Combined weekend collapse detector using journals, Chrome, and work logs."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import statsmodels.api as sm

from chrome_history_analysis import build_weekend_sample

BASE_DIR = Path(__file__).resolve().parent
INPUT_DIR = BASE_DIR / "journal_outputs"
OUTPUT_SUMMARY = BASE_DIR / "COMBINED_WEEKEND_DETECTOR.md"

JOURNAL_FEATURES = [
    "execution_focus",
    "goal_clarity",
    "overanalysis",
    "avoidance",
    "relationship_conflict",
    "relationship_closeness",
    "alcohol_issue",
    "phone_internet_issue",
    "compelling_problem",
]
CHROME_FEATURES = [
    "prev_night_total_visits",
    "prev_night_social_visits",
    "prev_night_work_ai_visits",
    "morning_total_visits",
    "morning_reading_visits",
    "morning_social_visits",
    "morning_work_ai_visits",
]
CONTROL_FEATURES = [
    "is_sunday",
    "prev_day_hours",
    "weekday_hours",
    "prev_weekend_hours",
    "week_ordinal",
    "regime_hive",
    "regime_mats",
    "regime_diesl",
]


def load_labels(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame(columns=["entry_id"])
    rows = [json.loads(line) for line in path.open()]
    if not rows:
        return pd.DataFrame(columns=["entry_id"])
    return pd.DataFrame(rows).drop_duplicates(subset="entry_id", keep="last")


def build_sample() -> pd.DataFrame:
    chrome_weekend = build_weekend_sample()
    entries = pd.read_csv(
        INPUT_DIR / "journal_entries.csv", parse_dates=["date", "week_start"]
    )
    labels = load_labels(INPUT_DIR / "daily_entry_labels.jsonl")

    journal = entries[entries["entry_type"] == "daily_entry"].merge(
        labels, on="entry_id", how="inner"
    )
    journal = journal[
        ["date", "entry_id"]
        + [col for col in JOURNAL_FEATURES if col in journal.columns]
    ].drop_duplicates(subset="date", keep="last")

    sample = chrome_weekend.merge(journal, on="date", how="inner")
    sample["collapsed_day"] = (sample["Hours Working"] == 0).astype(int)
    sample["real_workday"] = (sample["Hours Working"] >= 2).astype(int)
    return sample[sample["date"].dt.dayofweek >= 5].copy()


def fit_binary_model(sample: pd.DataFrame, features: list[str]):
    X = sm.add_constant(sample[features], has_constant="add")
    return sm.GLM(
        sample["collapsed_day"],
        X,
        family=sm.families.Binomial(),
    ).fit()


def controlled_univariate(
    df: pd.DataFrame, outcome: str, features: list[str], controls: list[str]
) -> list[tuple[float, str]]:
    lines: list[tuple[float, str]] = []
    for feature in features:
        if feature not in df.columns:
            continue
        sample = df[[outcome] + controls + [feature]].dropna().copy()
        if len(sample) < 35 or sample[feature].std() < 0.05:
            continue
        model = fit_binary_model(
            sample.rename(columns={outcome: "collapsed_day"}), controls + [feature]
        )
        lines.append(
            (
                abs(model.params[feature]),
                f"- `{feature}`: coef {model.params[feature]:+.3f}, p={model.pvalues[feature]:.4f}, n={len(sample)}",
            )
        )
    lines.sort(reverse=True)
    return lines


def backward_select(
    df: pd.DataFrame, outcome: str, candidates: list[str], controls: list[str]
) -> tuple[list[str], object]:
    kept = [feature for feature in candidates if feature in df.columns]
    protected = list(controls)
    sample = df[[outcome] + protected + kept].dropna().copy()
    while kept:
        model = fit_binary_model(
            sample.rename(columns={outcome: "collapsed_day"}), protected + kept
        )
        pvalues = model.pvalues.drop(
            labels=["const", *protected], errors="ignore"
        ).dropna()
        if pvalues.empty or pvalues.max() <= 0.10:
            return kept, model
        worst = pvalues.idxmax()
        if worst not in kept:
            return kept, model
        kept.remove(worst)
    model = fit_binary_model(
        sample.rename(columns={outcome: "collapsed_day"}), protected
    )
    return [], model


def detector_examples(df: pd.DataFrame, features: list[str], model) -> list[str]:
    sample = df.dropna(subset=features + CONTROL_FEATURES).copy()
    X = sm.add_constant(sample[CONTROL_FEATURES + features], has_constant="add")
    sample["predicted_collapse"] = model.predict(X)
    sample = sample.sort_values("predicted_collapse", ascending=False)
    lines = []
    for _, row in sample.head(5).iterrows():
        lines.append(
            f"- {row['date'].date()}: predicted collapse {row['predicted_collapse']:.2f}, "
            f"actual hours {row['Hours Working']:.2f}, morning_reading {row.get('morning_reading_visits', 0):.0f}, "
            f"morning_work_ai {row.get('morning_work_ai_visits', 0):.0f}, "
            f"avoidance {row.get('avoidance', 0):.1f}, compelling_problem {row.get('compelling_problem', 0):.1f}"
        )
    return lines


def build_summary(sample: pd.DataFrame) -> str:
    collapsed_rate = sample["collapsed_day"].mean()
    journal_lines = controlled_univariate(
        sample, "collapsed_day", JOURNAL_FEATURES, CONTROL_FEATURES
    )
    chrome_lines = controlled_univariate(
        sample, "collapsed_day", CHROME_FEATURES, CONTROL_FEATURES
    )

    selected, model = backward_select(
        sample,
        "collapsed_day",
        JOURNAL_FEATURES + CHROME_FEATURES,
        CONTROL_FEATURES,
    )

    model_lines = []
    for feature in selected:
        model_lines.append(
            f"- `{feature}`: coef {model.params[feature]:+.3f}, p={model.pvalues[feature]:.4f}"
        )

    detector_logic = []
    if "morning_reading_visits" in selected:
        detector_logic.append("more morning reading/blog browsing raises collapse risk")
    if "morning_work_ai_visits" in selected:
        detector_logic.append("more morning work-tool browsing lowers collapse risk")
    if "avoidance" in selected:
        detector_logic.append("higher journal-coded avoidance raises collapse risk")
    if "compelling_problem" in selected:
        detector_logic.append(
            "feeling pulled by a concrete problem lowers collapse risk"
        )
    if "phone_internet_issue" in selected:
        detector_logic.append("journal-coded phone/internet issue raises collapse risk")

    parts = [
        "# Combined Weekend Detector",
        "",
        "This file combines weekend-day journal labels, Chrome browsing features, and the work log to model weekend collapse directly.",
        "",
        f"Weekend days with both Chrome and journal coverage: {len(sample)}",
        f"Collapsed weekend days (`0` work hours): {int(sample['collapsed_day'].sum())} "
        f"of {len(sample)} ({collapsed_rate:.1%})",
        "",
        "## Controlled Journal Signals",
        "Each line conditions on recent work level, weekday hours, weekend carryover, and regime.",
        "",
        *[line for _, line in journal_lines[:8]],
        "",
        "## Controlled Chrome Signals",
        "These use the same controls. Morning features are especially useful because they are close to an actionable startup rule.",
        "",
        *[line for _, line in chrome_lines[:8]],
        "",
        "## Compact Detector Model",
        f"Selected features: {', '.join(selected) if selected else 'controls only'}",
        f"McFadden pseudo-R^2: {model.pseudo_rsquared(kind='mcf'):.3f}",
        "",
        *model_lines,
        "",
        "## Detector Read",
        (
            "- The simplest reading of the combined detector is: "
            + "; ".join(detector_logic)
            if detector_logic
            else "- The combined model did not keep any non-control features at the current threshold."
        ),
        "- This is not a causal proof. It is a compact risk model for whether a weekend day is about to die.",
        "- The most actionable class of signal is still startup behavior: what the first browser session is for, and whether the journal state is coded as avoidance vs compelling problem.",
        "",
        "## Highest-Risk Examples",
        *detector_examples(sample, selected, model),
    ]
    return "\n".join(parts) + "\n"


def main() -> None:
    sample = build_sample()
    OUTPUT_SUMMARY.write_text(build_summary(sample))
    print(f"Wrote {OUTPUT_SUMMARY}")
    print(f"Combined weekend sample rows: {len(sample)}")


if __name__ == "__main__":
    main()
