"""Microdata analysis using the detailed distracted log and Chrome timing."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import statsmodels.api as sm

from chrome_history_analysis import load_chrome_history
from productivity_analysis import (
    CALENDAR_DIR,
    DAILY_SUMMARY_CSV,
    DISTRACTED_CSV,
    extract_daily_supplements,
    load_daily_summary_full,
    load_calendar_sleep,
    load_distracted_stacked,
)

BASE_DIR = Path(__file__).resolve().parent
OUTPUT_FEATURES = BASE_DIR / "microdata_daily_features.csv"
OUTPUT_SUMMARY = BASE_DIR / "MICRODATA_SUMMARY.md"

DISTRACTION_CODES = ["w", "c", "b", "d", "t", "s"]
CONTROL_FEATURES = [
    "prev_hours",
    "is_weekend",
    "day_of_week",
    "regime_hive",
    "regime_mats",
    "regime_diesl",
]
NON_SUPPLEMENT_FEATURES = {
    "sleep_hours",
    "nap_hours",
    "meals_kcal",
    "snacks",
    "meditation_min",
    "is_plug",
}


@dataclass
class RankedLine:
    score: float
    text: str


@dataclass
class ValidatedFeature:
    feature: str
    coef: float
    p_value: float
    delta_holdout_r2: float
    holdout_r2: float
    n_rows: int


@dataclass
class ModelSelectionResult:
    lines: list[RankedLine]
    selected_features: list[str]
    n_rows: int
    adj_r2: float
    val_r2: float
    random_week_r2: float
    random_week_r2_std: float
    random_month_r2: float
    random_month_r2_std: float
    dropped_correlated: list[str]
    coefficients: dict[str, float]


def load_events() -> pd.DataFrame:
    events = load_distracted_stacked(DISTRACTED_CSV).copy()
    events["time_text"] = events["time"].astype(str).str.strip()
    events["clock_time"] = pd.to_datetime(
        events["time_text"], format="mixed", errors="coerce"
    )
    events["event_dt"] = (
        events["date"].dt.normalize()
        + pd.to_timedelta(events["clock_time"].dt.hour.fillna(0), unit="h")
        + pd.to_timedelta(events["clock_time"].dt.minute.fillna(0), unit="m")
        + pd.to_timedelta(events["clock_time"].dt.second.fillna(0), unit="s")
    )
    events["comment"] = events["comment"].fillna("").astype(str).str.strip()
    events["type"] = events["type"].fillna("").astype(str).str.strip()
    events = events.dropna(subset=["event_dt"]).copy()
    events = events.sort_values(["date", "event_dt", "work_increment"]).reset_index(
        drop=True
    )
    return events


def count_nonempty(values: Iterable[str]) -> int:
    return sum(1 for value in values if str(value).strip())


def task_switches(task_series: pd.Series) -> int:
    tasks = [task for task in task_series.astype(str).str.strip() if task]
    if not tasks:
        return 0
    switches = 0
    prev = tasks[0]
    for task in tasks[1:]:
        if task != prev:
            switches += 1
        prev = task
    return switches


def entropy_from_counts(counts: pd.Series) -> float:
    if counts.empty:
        return 0.0
    probs = counts / counts.sum()
    return float(-(probs * np.log2(probs)).sum())


def first_timestamp(day: pd.DataFrame, event_types: set[str]) -> pd.Timestamp | pd.NaT:
    subset = day[day["type"].isin(event_types)]
    if subset.empty:
        return pd.NaT
    return subset["event_dt"].min()


def safe_minutes(delta: pd.Timedelta | pd.NaT) -> float:
    if pd.isna(delta):
        return np.nan
    return float(delta.total_seconds() / 60)


def build_daily_microfeatures(events: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, float | int | str | pd.Timestamp]] = []

    for date, day in events.groupby("date"):
        day = day.sort_values("event_dt").copy()
        task_events = day[day["type"] == "t"].copy()
        d_events = day[day["type"] == "d"].copy()
        u_events = day[day["type"] == "u"].copy()
        c_events = day[day["type"] == "c"].copy()
        i_events = day[day["type"] == "i"].copy()

        first_start = first_timestamp(day, {"s"})
        first_task = first_timestamp(day, {"t"})
        anchor_dt = min(
            [ts for ts in [first_start, first_task] if pd.notna(ts)],
            default=pd.NaT,
        )
        first_distraction = day[
            day["type"].isin(["d", "u", "c"]) & (day["event_dt"] >= anchor_dt)
        ]["event_dt"].min()

        if pd.notna(anchor_dt):
            start_window_end = anchor_dt + pd.Timedelta(minutes=90)
            start_window = day[
                (day["event_dt"] >= anchor_dt) & (day["event_dt"] < start_window_end)
            ].copy()
        else:
            start_window = day.iloc[0:0].copy()

        session_spans = (
            day.groupby("work_increment")["event_dt"].agg(["min", "max"]).reset_index()
        )
        session_spans["session_span_min"] = (
            session_spans["max"] - session_spans["min"]
        ).dt.total_seconds() / 60
        session_span_total_min = float(session_spans["session_span_min"].sum())
        span_hours = (
            session_span_total_min / 60 if session_span_total_min > 0 else np.nan
        )

        task_counts = task_events["comment"].value_counts()
        row: dict[str, float | int | str | pd.Timestamp] = {
            "date": date,
            "micro_rows": len(day),
            "session_count": int(day["work_increment"].nunique()),
            "start_count": int((day["type"] == "s").sum()),
            "end_count": int((day["type"] == "e").sum()),
            "task_event_count": len(task_events),
            "unique_tasks": int(task_events["comment"].nunique()),
            "task_switches": task_switches(task_events["comment"]),
            "task_entropy": entropy_from_counts(task_counts),
            "dominant_task_share": (
                float(task_counts.max() / task_counts.sum())
                if not task_counts.empty
                else 0.0
            ),
            "d_event_count": len(d_events),
            "u_event_count": len(u_events),
            "c_event_count": len(c_events),
            "i_event_count": len(i_events),
            "d_minutes_total": float(d_events["length_minutes"].fillna(0).sum()),
            "d_minutes_mean": (
                float(d_events["length_minutes"].fillna(0).mean())
                if not d_events.empty
                else 0.0
            ),
            "d_minutes_p90": (
                float(d_events["length_minutes"].fillna(0).quantile(0.9))
                if not d_events.empty
                else 0.0
            ),
            "u_minutes_total": float(u_events["length_minutes"].fillna(0).sum()),
            "c_minutes_total": float(c_events["length_minutes"].fillna(0).sum()),
            "session_span_total_min": session_span_total_min,
            "longest_session_min": (
                float(session_spans["session_span_min"].max())
                if not session_spans.empty
                else 0.0
            ),
            "median_session_min": (
                float(session_spans["session_span_min"].median())
                if not session_spans.empty
                else 0.0
            ),
            "multi_session_day": int(day["work_increment"].nunique() >= 2),
            "anchor_hour": (
                anchor_dt.hour + anchor_dt.minute / 60
                if pd.notna(anchor_dt)
                else np.nan
            ),
            "last_event_hour": (
                day["event_dt"].max().hour + day["event_dt"].max().minute / 60
            ),
            "first_distraction_min": (
                safe_minutes(first_distraction - anchor_dt)
                if pd.notna(anchor_dt) and pd.notna(first_distraction)
                else np.nan
            ),
            "clean_start_min": (
                safe_minutes(first_distraction - anchor_dt)
                if pd.notna(anchor_dt) and pd.notna(first_distraction)
                else np.nan
            ),
            "start_window_task_events": int((start_window["type"] == "t").sum()),
            "start_window_d_events": int((start_window["type"] == "d").sum()),
            "start_window_u_events": int((start_window["type"] == "u").sum()),
            "start_window_c_events": int((start_window["type"] == "c").sum()),
            "start_window_d_minutes": float(
                start_window.loc[start_window["type"] == "d", "length_minutes"]
                .fillna(0)
                .sum()
            ),
            "start_window_task_switches": task_switches(
                start_window.loc[start_window["type"] == "t", "comment"]
            ),
            "morning_task_events": int(((task_events["event_dt"].dt.hour < 12)).sum()),
            "evening_task_events": int(((task_events["event_dt"].dt.hour >= 20)).sum()),
            "morning_d_minutes": float(
                d_events.loc[d_events["event_dt"].dt.hour < 12, "length_minutes"]
                .fillna(0)
                .sum()
            ),
            "evening_d_minutes": float(
                d_events.loc[d_events["event_dt"].dt.hour >= 20, "length_minutes"]
                .fillna(0)
                .sum()
            ),
        }

        for code in DISTRACTION_CODES:
            subset = d_events[d_events["comment"] == code]
            row[f"d_{code}_count"] = int(len(subset))
            row[f"d_{code}_minutes"] = float(subset["length_minutes"].fillna(0).sum())

        blank_subset = d_events[d_events["comment"] == ""]
        row["d_blank_count"] = int(len(blank_subset))
        row["d_blank_minutes"] = float(blank_subset["length_minutes"].fillna(0).sum())
        row["u_blank_count"] = int((u_events["comment"] == "").sum())
        row["d_events_per_span_hour"] = (
            row["d_event_count"] / span_hours if span_hours else np.nan
        )
        row["u_events_per_span_hour"] = (
            row["u_event_count"] / span_hours if span_hours else np.nan
        )
        row["d_minutes_per_span_hour"] = (
            row["d_minutes_total"] / span_hours if span_hours else np.nan
        )
        row["task_events_per_span_hour"] = (
            row["task_event_count"] / span_hours if span_hours else np.nan
        )
        row["task_switches_per_span_hour"] = (
            row["task_switches"] / span_hours if span_hours else np.nan
        )
        row["task_switches_per_task"] = (
            row["task_switches"] / row["task_event_count"]
            if row["task_event_count"]
            else np.nan
        )
        row["unique_tasks_per_task"] = (
            row["unique_tasks"] / row["task_event_count"]
            if row["task_event_count"]
            else np.nan
        )
        row["d_blank_share"] = (
            row["d_blank_count"] / row["d_event_count"]
            if row["d_event_count"]
            else np.nan
        )
        row["start_window_d_rate"] = row["start_window_d_events"] / 1.5
        row["start_window_task_switch_rate"] = row["start_window_task_switches"] / 1.5
        row["start_window_task_rate"] = row["start_window_task_events"] / 1.5
        for code in DISTRACTION_CODES:
            row[f"d_{code}_share"] = (
                row[f"d_{code}_count"] / row["d_event_count"]
                if row["d_event_count"]
                else np.nan
            )

        rows.append(row)

    return pd.DataFrame(rows).sort_values("date").reset_index(drop=True)


def build_chrome_timing_features(
    events: pd.DataFrame, daily_features: pd.DataFrame
) -> pd.DataFrame:
    chrome = load_chrome_history().copy()
    chrome_start = chrome["date"].min()
    chrome_end = chrome["date"].max()
    anchors = daily_features[["date", "anchor_hour", "clean_start_min"]].copy()
    anchors = anchors.dropna(subset=["anchor_hour"])
    if anchors.empty:
        return pd.DataFrame(columns=["date"])

    anchors["anchor_dt"] = anchors["date"] + pd.to_timedelta(
        anchors["anchor_hour"], unit="h"
    )
    anchors["first_block_end_dt"] = anchors["anchor_dt"] + pd.to_timedelta(
        anchors["clean_start_min"].clip(lower=0, upper=180).fillna(0), unit="m"
    )
    anchors["first90_end_dt"] = anchors["anchor_dt"] + pd.Timedelta(minutes=90)
    anchors["pre60_start_dt"] = anchors["anchor_dt"] - pd.Timedelta(minutes=60)

    def classify(series: pd.Series) -> pd.Series:
        mapping = {}
        for category in [
            "work_ai",
            "reading",
            "social",
            "entertainment",
            "search",
            "admin",
            "other",
        ]:
            mapping[category] = (series == category).astype(int)
        return pd.DataFrame(mapping)

    chrome = chrome[["dt", "date", "category"]].copy()
    rows: list[dict[str, float | pd.Timestamp]] = []
    for anchor in anchors.itertuples(index=False):
        day_visits = chrome[chrome["date"] == anchor.date].copy()
        prev_visits = chrome[
            (chrome["dt"].dt.tz_localize(None) >= anchor.pre60_start_dt)
            & (chrome["dt"].dt.tz_localize(None) < anchor.anchor_dt)
        ].copy()
        first90 = chrome[
            (chrome["dt"].dt.tz_localize(None) >= anchor.anchor_dt)
            & (chrome["dt"].dt.tz_localize(None) < anchor.first90_end_dt)
        ].copy()
        first_block = chrome[
            (chrome["dt"].dt.tz_localize(None) >= anchor.anchor_dt)
            & (chrome["dt"].dt.tz_localize(None) < anchor.first_block_end_dt)
        ].copy()

        row: dict[str, float | pd.Timestamp] = {
            "date": anchor.date,
            "chrome_range_overlap": float(chrome_start <= anchor.date <= chrome_end),
            "pre_start_60m_total_visits": float(len(prev_visits)),
            "first90_total_visits": float(len(first90)),
            "pre_first_distraction_total_visits": float(len(first_block)),
        }
        for prefix, frame in [
            ("pre_start_60m", prev_visits),
            ("first90", first90),
            ("pre_first_distraction", first_block),
            ("same_day", day_visits),
        ]:
            counts = frame["category"].value_counts()
            for category in [
                "work_ai",
                "reading",
                "social",
                "entertainment",
                "search",
                "admin",
                "other",
            ]:
                row[f"{prefix}_{category}_visits"] = float(counts.get(category, 0))
        rows.append(row)

    return pd.DataFrame(rows)


def build_model_frame() -> pd.DataFrame:
    events = load_events()
    micro = build_daily_microfeatures(events)
    chrome = build_chrome_timing_features(events, micro)
    supplements = extract_daily_supplements(events)
    calendar = load_calendar_sleep(CALENDAR_DIR)
    daily = (
        load_daily_summary_full(DAILY_SUMMARY_CSV)
        .sort_values("date")
        .reset_index(drop=True)
    )
    daily["day_of_week"] = daily["date"].dt.dayofweek
    daily["is_weekend"] = (daily["day_of_week"] >= 5).astype(int)
    daily["prev_hours"] = daily["Hours Working"].shift(1)
    daily["regime_hive"] = (daily["regime"] == "Hive").astype(int)
    daily["regime_mats"] = (daily["regime"] == "Mats").astype(int)
    daily["regime_diesl"] = (daily["regime"] == "diesl").astype(int)

    model = (
        daily.merge(micro, on="date", how="left")
        .merge(chrome, on="date", how="left")
        .merge(supplements, on="date", how="left")
        .merge(calendar, on="date", how="left")
    )
    supplement_cols = [
        col
        for col in supplements.columns
        if col != "date" and col not in NON_SUPPLEMENT_FEATURES
    ]
    zero_fill_cols = supplement_cols + [
        "nap_hours",
        "meals_kcal",
        "snacks",
        "meditation_min",
        "is_plug",
    ]
    for column in zero_fill_cols:
        if column in model.columns:
            model[column] = model[column].fillna(0)
    for column in supplement_cols:
        model[f"{column}_any"] = (model[column].fillna(0) > 0).astype(int)
    fill_zero = [
        col
        for col in model.columns
        if col.endswith("_visits")
        or col.endswith("_count")
        or col.endswith("_minutes")
        or col.endswith("_hours")
        or col.endswith("_duration")
        or col.endswith("_kcal")
        or col.endswith("_min")
        or col.endswith("_any")
    ]
    model[fill_zero] = model[fill_zero].fillna(0)
    return model.sort_values("date").reset_index(drop=True)


def rank_features(
    df: pd.DataFrame,
    outcome: str,
    features: list[str],
    controls: list[str] | None = None,
    min_rows: int = 80,
) -> list[RankedLine]:
    ranked: list[RankedLine] = []
    controls = controls or []
    for feature in features:
        if feature not in df.columns:
            continue
        sample = df[[outcome, feature] + controls].dropna().copy()
        if len(sample) < min_rows or sample[feature].std() < 0.05:
            continue
        X = sm.add_constant(sample[controls + [feature]])
        model = sm.OLS(sample[outcome], X).fit()
        coef = model.params[feature]
        p_value = model.pvalues[feature]
        ranked.append(
            RankedLine(
                score=abs(coef),
                text=(
                    f"- `{feature}`: coef {coef:+.3f}, p={p_value:.4f}, "
                    f"n={len(sample)}, adjR2={model.rsquared_adj:.3f}"
                ),
            )
        )
    return sorted(ranked, key=lambda line: line.score, reverse=True)


def fit_ols(
    df: pd.DataFrame, outcome: str, features: list[str], min_rows: int
) -> tuple[sm.regression.linear_model.RegressionResultsWrapper | None, pd.DataFrame]:
    sample = df[[outcome] + features].dropna().copy()
    if len(sample) < min_rows:
        return None, sample
    design = sample[features]
    if design.empty:
        return None, sample
    X = sm.add_constant(design, has_constant="add")
    model = sm.OLS(sample[outcome], X).fit(cov_type="HC3")
    return model, sample


def prune_correlated_features(
    df: pd.DataFrame, features: list[str], threshold: float = 0.9
) -> tuple[list[str], list[str]]:
    kept = [feature for feature in features if feature in df.columns]
    dropped: list[str] = []
    changed = True
    while changed:
        changed = False
        sample = df[kept].dropna()
        if sample.empty or len(kept) < 2:
            break
        corr = sample.corr().abs()
        upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
        hit = None
        max_corr = threshold
        for left in upper.columns:
            for right, value in upper[left].dropna().items():
                if value > max_corr:
                    hit = (left, right, value)
                    max_corr = value
        if hit is None:
            break
        left, right, _ = hit
        left_score = (
            df[left].notna().sum(),
            df[left].std(skipna=True),
        )
        right_score = (
            df[right].notna().sum(),
            df[right].std(skipna=True),
        )
        drop = right if left_score >= right_score else left
        kept.remove(drop)
        dropped.append(drop)
        changed = True
    return kept, dropped


def time_split_r2(df: pd.DataFrame, outcome: str, features: list[str]) -> float:
    if "date" not in df.columns:
        return float("nan")
    sample = df[["date", outcome] + features].dropna().sort_values("date").copy()
    if len(sample) < 120:
        return float("nan")
    split = int(len(sample) * 0.7)
    if split < len(features) + 10 or len(sample) - split < max(30, len(features) + 5):
        return float("nan")
    train = sample.iloc[:split].copy()
    val = sample.iloc[split:].copy()
    X_train = sm.add_constant(train[features], has_constant="add")
    X_val = sm.add_constant(val[features], has_constant="add")
    model = sm.OLS(train[outcome], X_train).fit()
    preds = model.predict(X_val)
    baseline = val[outcome].mean()
    sse = float(((val[outcome] - preds) ** 2).sum())
    tss = float(((val[outcome] - baseline) ** 2).sum())
    if tss <= 0:
        return float("nan")
    return 1 - sse / tss


def holdout_r2_from_sample(
    sample: pd.DataFrame, outcome: str, features: list[str]
) -> float:
    if "date" not in sample.columns:
        return float("nan")
    ordered = sample.sort_values("date").copy()
    if len(ordered) < 120:
        return float("nan")
    split = int(len(ordered) * 0.7)
    if split < len(features) + 10 or len(ordered) - split < max(30, len(features) + 5):
        return float("nan")
    train = ordered.iloc[:split].copy()
    val = ordered.iloc[split:].copy()
    X_train = sm.add_constant(train[features], has_constant="add")
    X_val = sm.add_constant(val[features], has_constant="add")
    model = sm.OLS(train[outcome], X_train).fit()
    preds = model.predict(X_val)
    baseline = val[outcome].mean()
    sse = float(((val[outcome] - preds) ** 2).sum())
    tss = float(((val[outcome] - baseline) ** 2).sum())
    if tss <= 0:
        return float("nan")
    return 1 - sse / tss


def grouped_holdout_r2_from_sample(
    sample: pd.DataFrame,
    outcome: str,
    features: list[str],
    group_unit: str,
    test_frac: float = 0.3,
    n_repeats: int = 200,
    seed: int = 42,
) -> tuple[float, float]:
    if "date" not in sample.columns:
        return float("nan"), float("nan")

    ordered = sample.sort_values("date").copy()
    if len(ordered) < 120:
        return float("nan"), float("nan")

    if group_unit == "week":
        groups = ordered["date"].dt.to_period("W-SUN")
    elif group_unit == "month":
        groups = ordered["date"].dt.to_period("M")
    else:
        raise ValueError(f"Unsupported group unit: {group_unit}")

    ordered = ordered.assign(_holdout_group=groups.astype(str))
    unique_groups = ordered["_holdout_group"].unique()
    if len(unique_groups) < 8:
        return float("nan"), float("nan")

    n_test_groups = max(1, int(len(unique_groups) * test_frac))
    if n_test_groups >= len(unique_groups) - 1:
        return float("nan"), float("nan")

    rng = np.random.default_rng(seed)
    r2s: list[float] = []

    for _ in range(n_repeats):
        test_groups = set(rng.choice(unique_groups, size=n_test_groups, replace=False))
        test_mask = ordered["_holdout_group"].isin(test_groups)
        train = ordered.loc[~test_mask]
        val = ordered.loc[test_mask]
        if len(train) < len(features) + 10 or len(val) < max(30, len(features) + 5):
            continue

        X_train = sm.add_constant(train[features], has_constant="add")
        X_val = sm.add_constant(val[features], has_constant="add")
        model = sm.OLS(train[outcome], X_train).fit()
        preds = model.predict(X_val)
        baseline = val[outcome].mean()
        sse = float(((val[outcome] - preds) ** 2).sum())
        tss = float(((val[outcome] - baseline) ** 2).sum())
        if tss <= 0:
            continue
        r2s.append(1 - sse / tss)

    if not r2s:
        return float("nan"), float("nan")
    return float(np.mean(r2s)), float(np.std(r2s))


def grouped_holdout_r2(
    df: pd.DataFrame,
    outcome: str,
    features: list[str],
    group_unit: str,
    test_frac: float = 0.3,
    n_repeats: int = 200,
    seed: int = 42,
) -> tuple[float, float]:
    if "date" not in df.columns:
        return float("nan"), float("nan")
    sample = df[["date", outcome] + features].dropna().copy()
    return grouped_holdout_r2_from_sample(
        sample,
        outcome,
        features,
        group_unit,
        test_frac=test_frac,
        n_repeats=n_repeats,
        seed=seed,
    )


def validated_single_feature_ranks(
    df: pd.DataFrame,
    outcome: str,
    features: list[str],
    controls: list[str] | None = None,
    min_rows: int = 120,
) -> list[ValidatedFeature]:
    controls = [feature for feature in (controls or []) if feature in df.columns]
    ranked: list[ValidatedFeature] = []
    for feature in features:
        if feature not in df.columns:
            continue
        cols = ["date", outcome, feature] + controls
        sample = df[cols].dropna().copy()
        if len(sample) < min_rows or sample[feature].std() < 0.05:
            continue
        full_features = controls + [feature]
        full_model, full_sample = fit_ols(
            sample, outcome, full_features, min_rows=min_rows
        )
        if full_model is None or feature not in full_model.params.index:
            continue
        base_model, base_sample = fit_ols(sample, outcome, controls, min_rows=min_rows)
        if base_model is None:
            continue
        full_holdout = holdout_r2_from_sample(
            full_sample.assign(date=sample["date"].values), outcome, full_features
        )
        base_holdout = holdout_r2_from_sample(
            base_sample.assign(date=sample["date"].values), outcome, controls
        )
        ranked.append(
            ValidatedFeature(
                feature=feature,
                coef=float(full_model.params[feature]),
                p_value=float(full_model.pvalues[feature]),
                delta_holdout_r2=float(full_holdout - base_holdout),
                holdout_r2=float(full_holdout),
                n_rows=len(full_sample),
            )
        )
    return sorted(ranked, key=lambda item: item.delta_holdout_r2, reverse=True)


def standardized_coef(
    sample: pd.DataFrame,
    model: sm.regression.linear_model.RegressionResultsWrapper,
    feature: str,
    outcome: str,
) -> float:
    x_std = sample[feature].std()
    y_std = sample[outcome].std()
    if x_std == 0 or y_std == 0:
        return 0.0
    return float(model.params[feature] * x_std / y_std)


def select_multivariate_features(
    df: pd.DataFrame,
    outcome: str,
    features: list[str],
    controls: list[str] | None = None,
    min_rows: int = 80,
    max_features: int = 8,
    p_enter: float = 0.05,
    min_delta_adj_r2: float = 0.002,
    corr_threshold: float = 0.9,
) -> ModelSelectionResult:
    controls = [feature for feature in (controls or []) if feature in df.columns]
    candidates = [feature for feature in features if feature in df.columns]
    candidates = [feature for feature in candidates if feature not in controls]
    candidates, dropped_correlated = prune_correlated_features(
        df, candidates, threshold=corr_threshold
    )
    selected: list[str] = []

    while len(selected) < max_features:
        best_choice: str | None = None
        best_delta = min_delta_adj_r2

        for candidate in candidates:
            if candidate in selected:
                continue
            model, sample = fit_ols(
                df, outcome, controls + selected + [candidate], min_rows=min_rows
            )
            if model is None or candidate not in model.params.index:
                continue
            base_model, _ = fit_ols(df, outcome, controls + selected, min_rows=min_rows)
            base_adj_r2 = base_model.rsquared_adj if base_model is not None else 0.0
            delta = model.rsquared_adj - base_adj_r2
            if model.pvalues[candidate] > p_enter or delta < best_delta:
                continue
            best_choice = candidate
            best_delta = delta

        if best_choice is None:
            break
        selected.append(best_choice)

    changed = True
    while changed and selected:
        changed = False
        model, _ = fit_ols(df, outcome, controls + selected, min_rows=min_rows)
        if model is None:
            break
        removable = [
            feature
            for feature in selected
            if feature in model.pvalues.index and model.pvalues[feature] > p_enter
        ]
        if removable:
            worst = max(removable, key=lambda feature: model.pvalues[feature])
            selected.remove(worst)
            changed = True

    final_model, final_sample = fit_ols(
        df, outcome, controls + selected, min_rows=min_rows
    )
    if final_model is None:
        return ModelSelectionResult(
            lines=[],
            selected_features=[],
            n_rows=0,
            adj_r2=float("nan"),
            val_r2=float("nan"),
            random_week_r2=float("nan"),
            random_week_r2_std=float("nan"),
            random_month_r2=float("nan"),
            random_month_r2_std=float("nan"),
            dropped_correlated=dropped_correlated,
            coefficients={},
        )

    lines: list[RankedLine] = []
    coefficients: dict[str, float] = {}
    for feature in selected:
        std_beta = standardized_coef(final_sample, final_model, feature, outcome)
        coefficients[feature] = float(final_model.params[feature])
        lines.append(
            RankedLine(
                score=abs(std_beta),
                text=(
                    f"- `{feature}`: coef {final_model.params[feature]:+.3f}, "
                    f"std_beta {std_beta:+.3f}, p={final_model.pvalues[feature]:.4f}, "
                    f"n={len(final_sample)}, adjR2={final_model.rsquared_adj:.3f}"
                ),
            )
        )
    lines.sort(key=lambda line: line.score, reverse=True)
    week_r2, week_r2_std = grouped_holdout_r2(df, outcome, controls + selected, "week")
    month_r2, month_r2_std = grouped_holdout_r2(
        df, outcome, controls + selected, "month"
    )

    return ModelSelectionResult(
        lines=lines,
        selected_features=selected,
        n_rows=len(final_sample),
        adj_r2=float(final_model.rsquared_adj),
        val_r2=time_split_r2(df, outcome, controls + selected),
        random_week_r2=week_r2,
        random_week_r2_std=week_r2_std,
        random_month_r2=month_r2,
        random_month_r2_std=month_r2_std,
        dropped_correlated=dropped_correlated,
        coefficients=coefficients,
    )


def make_section(title: str, lines: list[RankedLine], top_n: int = 10) -> list[str]:
    section = [title]
    if not lines:
        section.append("- No stable features met the coverage threshold.")
        return section
    section.extend(line.text for line in lines[:top_n])
    return section


def make_model_section(title: str, result: ModelSelectionResult) -> list[str]:
    section = [title]
    if not result.lines:
        section.append("- No multivariate model met the coverage threshold.")
        return section
    section.append(
        f"- Selected {len(result.selected_features)} features, n={result.n_rows}, "
        f"adjR2={result.adj_r2:.3f}, timeSplitR2={result.val_r2:.3f}, "
        f"randomWeekHoldoutR2={result.random_week_r2:.3f}±{result.random_week_r2_std:.3f}, "
        f"randomMonthHoldoutR2={result.random_month_r2:.3f}±{result.random_month_r2_std:.3f}."
    )
    if result.dropped_correlated:
        dropped = ", ".join(f"`{feature}`" for feature in result.dropped_correlated[:8])
        suffix = " ..." if len(result.dropped_correlated) > 8 else ""
        section.append(f"- Dropped as redundant before selection: {dropped}{suffix}")
    section.extend(line.text for line in result.lines)
    return section


def make_validated_feature_section(
    title: str, features: list[ValidatedFeature], top_n: int = 6
) -> list[str]:
    section = [title]
    if not features:
        section.append("- No single actionable feature met the coverage threshold.")
        return section
    if all(item.holdout_r2 <= 0 for item in features):
        section.append(
            "- None of the single-feature actionable models achieved positive holdout R2. The lines below are only the least-bad relative improvements."
        )
    kept = 0
    for item in features:
        if item.delta_holdout_r2 <= 0:
            continue
        section.append(
            f"- `{item.feature}`: coef {item.coef:+.3f}, p={item.p_value:.4f}, "
            f"delta_holdoutR2={item.delta_holdout_r2:+.3f}, holdoutR2={item.holdout_r2:.3f}, n={item.n_rows}"
        )
        kept += 1
        if kept >= top_n:
            break
    if kept == 0:
        section.append(
            "- No single actionable feature improved holdout R2 beyond the base controls."
        )
    return section


def brainstorm_lines(model: pd.DataFrame) -> list[str]:
    lines = [
        "## Hypotheses To Pressure-Test",
        "- The actionable sections are the main read. The descriptive sections can explain variance without identifying a lever.",
        "- Earlier anchors and cleaner first blocks are more believable levers than full-day variables like last event hour or calendar blue time.",
        "- Any feature based on unnamed distraction codes should be treated as provisional until the code meanings are mapped.",
        "- Some positive coefficients on switching and evening activity may reflect successful long days rather than causes of them.",
        "- Browser use is not automatically bad. Early work-directed browsing can be productive, while reading/blog drift is the more plausible leak.",
        "- Calendar and supplement variables belong in the same frame because some apparent task effects may actually be sleep, gym, stimulant, or waste-time effects.",
        "",
        "## Candidate Interventions",
        "- Protect the first block: no reading/blog/internet drift before the first task block has real traction.",
        "- Start earlier and measure the first 90 minutes separately from the rest of the day.",
        "- Reduce task churn: fewer distinct tasks early, especially before a clear main task has traction.",
        "- Budget distraction type, not just total distraction minutes. Some codes may need hard constraints while others do not.",
        "- Use browser startup defaults that open work tabs or tools rather than feeds or reading queues.",
        "- Treat supplements and gym timing as secondary modifiers, not primary explanations, unless they survive the actionable model repeatedly.",
        "- Consider a 'clean start score' as a daily lead indicator: anchor hour, first-distraction latency, first-block browser mix, and early task switches.",
    ]
    overlap = int(model.get("chrome_range_overlap", pd.Series(dtype=float)).sum())
    if overlap:
        lines.append("")
        lines.append(
            f"## Chrome Tie-In\n- Minute-level Chrome overlap exists for {overlap} days, so browser mix around the first block is now a measurable lever rather than a vague intuition."
        )
    return lines


def supported_conclusion_lines(
    actionable_results: dict[str, ModelSelectionResult],
    validated_hits: dict[str, list[ValidatedFeature]],
) -> list[str]:
    feature_hits: dict[str, list[tuple[str, float]]] = {}
    for outcome, hits in validated_hits.items():
        for hit in hits:
            if hit.delta_holdout_r2 <= 0.01 or hit.p_value > 0.05:
                continue
            feature_hits.setdefault(hit.feature, []).append((outcome, hit.coef))

    label_map = {
        "Hours Working": "hours",
        "Focus": "focus",
        "Value": "value",
        "work_productivity": "value-hours",
    }
    conclusions = ["## Validation Takeaways"]
    good_models = sum(
        1
        for result in actionable_results.values()
        if not np.isnan(result.val_r2) and result.val_r2 > 0.05
    )
    if good_models == 0:
        conclusions.append(
            "- The multivariate early-day models did not hold up well out of sample. Treat them as hypothesis generators, not hard findings."
        )
    positive_holdout_hits = sum(
        1
        for hits in validated_hits.values()
        for hit in hits
        if hit.delta_holdout_r2 > 0.01 and hit.holdout_r2 > 0 and hit.p_value <= 0.05
    )
    if positive_holdout_hits == 0:
        conclusions.append(
            "- No single early-day feature produced a genuinely positive holdout model beyond the base controls. The best actionable signals are only tentative."
        )
    added = 0
    for feature, hits in sorted(
        feature_hits.items(), key=lambda item: (-len(item[1]), item[0])
    ):
        if len(hits) < 2:
            continue
        signs = {np.sign(coef) for _, coef in hits if coef != 0}
        if len(signs) != 1:
            continue
        direction = "higher" if list(signs)[0] > 0 else "lower"
        outcomes = ", ".join(label_map[outcome] for outcome, _ in hits)
        conclusions.append(
            f"- Tentative only: `{feature}` shows a consistent {direction}-is-better pattern across {len(hits)} outcomes: {outcomes}."
        )
        added += 1
        if added >= 5:
            break
    if added == 0:
        conclusions.append(
            "- No early-day feature had a clean, repeated validation win across multiple outcomes."
        )
    return conclusions


def main() -> None:
    model = build_model_frame()
    model.to_csv(OUTPUT_FEATURES, index=False)

    micro_features = [
        "session_count",
        "task_event_count",
        "unique_tasks",
        "task_switches",
        "task_entropy",
        "dominant_task_share",
        "d_event_count",
        "u_event_count",
        "c_event_count",
        "i_event_count",
        "d_minutes_total",
        "d_minutes_mean",
        "d_minutes_p90",
        "session_span_total_min",
        "longest_session_min",
        "median_session_min",
        "multi_session_day",
        "anchor_hour",
        "last_event_hour",
        "clean_start_min",
        "start_window_task_events",
        "start_window_d_events",
        "start_window_u_events",
        "start_window_d_minutes",
        "start_window_task_switches",
        "morning_task_events",
        "evening_task_events",
        "morning_d_minutes",
        "evening_d_minutes",
        "d_events_per_span_hour",
        "u_events_per_span_hour",
        "d_minutes_per_span_hour",
        "task_events_per_span_hour",
        "task_switches_per_span_hour",
        "task_switches_per_task",
        "unique_tasks_per_task",
        "d_blank_share",
        "start_window_d_rate",
        "start_window_task_switch_rate",
        "start_window_task_rate",
        "d_blank_count",
        "d_blank_minutes",
        *[f"d_{code}_count" for code in DISTRACTION_CODES],
        *[f"d_{code}_minutes" for code in DISTRACTION_CODES],
        *[f"d_{code}_share" for code in DISTRACTION_CODES],
    ]
    startup_micro_features = [
        "anchor_hour",
        "clean_start_min",
        "first_distraction_min",
        "start_window_task_events",
        "start_window_d_events",
        "start_window_u_events",
        "start_window_c_events",
        "start_window_d_minutes",
        "start_window_task_switches",
        "start_window_d_rate",
        "start_window_task_switch_rate",
        "start_window_task_rate",
    ]
    early_chrome_features = [
        "pre_start_60m_total_visits",
        "pre_start_60m_work_ai_visits",
        "pre_start_60m_reading_visits",
        "pre_start_60m_social_visits",
        "first90_total_visits",
        "first90_work_ai_visits",
        "first90_reading_visits",
        "first90_social_visits",
        "pre_first_distraction_total_visits",
        "pre_first_distraction_work_ai_visits",
        "pre_first_distraction_reading_visits",
    ]
    same_day_browser_features = [
        "same_day_work_ai_visits",
        "same_day_reading_visits",
        "same_day_social_visits",
    ]
    actionable_context_features = [
        "sleep_hours",
        "nap_hours",
        "meditation_min",
        "calendar_sleep_hours",
        "wake_hour",
        "gym_hour",
        "gym_duration",
        "caffeine",
        "adderall",
        "modafinil",
        "piracetam",
        "choline",
        "nicotine",
        "bronkaid",
        "caffeine_any",
        "adderall_any",
        "modafinil_any",
        "piracetam_any",
        "choline_any",
        "nicotine_any",
        "bronkaid_any",
    ]
    descriptive_context_features = [
        "calendar_blue_hours",
        "calendar_waste_hours",
        "meals_kcal",
        "snacks",
        "c_event_count",
    ]
    actionable_features = (
        startup_micro_features + early_chrome_features + actionable_context_features
    )
    descriptive_proxy_features = (
        same_day_browser_features + descriptive_context_features
    )

    chrome_model = model[model.get("chrome_range_overlap", 0) == 1].copy()

    parts: list[str] = [
        "# Microdata Summary",
        "",
        "This file analyzes the minute-level distracted log as an event stream and joins it with the daily summary, parsed supplements, calendar-derived sleep/productive/waste time, and Chrome timing.",
        "",
        f"Model frame rows: {len(model)} days.",
        f"Microdata coverage: {int(model['task_event_count'].notna().sum())} days.",
        f"Chrome timing overlap: {len(chrome_model)} days.",
        "Main read: the actionable model uses only variables available by the start of the day or within the first block. Descriptive models are reported separately.",
        "",
    ]
    actionable_results: dict[str, ModelSelectionResult] = {}
    validated_actionable_hits: dict[str, list[ValidatedFeature]] = {}

    for outcome, title in [
        ("Hours Working", "## Same-Day Hours"),
        ("Focus", "## Focus"),
        ("Value", "## Value"),
        ("work_productivity", "## Value x Hours"),
    ]:
        actionable_model = select_multivariate_features(
            model,
            outcome,
            actionable_features,
            controls=CONTROL_FEATURES,
            max_features=6,
            corr_threshold=0.85,
        )
        actionable_results[outcome] = actionable_model
        validated_actionable = validated_single_feature_ranks(
            model,
            outcome,
            actionable_features,
            controls=CONTROL_FEATURES,
            min_rows=150,
        )
        validated_actionable_hits[outcome] = validated_actionable
        controlled_micro = select_multivariate_features(
            model,
            outcome,
            micro_features,
            controls=CONTROL_FEATURES,
            corr_threshold=0.85,
        )
        descriptive_proxy_model = select_multivariate_features(
            model,
            outcome,
            descriptive_proxy_features,
            controls=CONTROL_FEATURES,
            min_rows=80,
            max_features=4,
            corr_threshold=0.85,
        )
        parts.append(title)
        parts.append("")
        parts.extend(
            make_model_section(
                "### Actionable early-day model",
                actionable_model,
            )
        )
        parts.append("")
        parts.extend(
            make_validated_feature_section(
                "### Relative best single-feature actionable checks",
                validated_actionable,
            )
        )
        parts.append("")
        parts.extend(
            make_model_section(
                "### Descriptive full-day microdata model",
                controlled_micro,
            )
        )
        parts.append("")
        parts.extend(
            make_model_section(
                "### Descriptive same-day proxy model",
                descriptive_proxy_model,
            )
        )
        parts.append("")

    parts.extend(
        supported_conclusion_lines(actionable_results, validated_actionable_hits)
    )
    parts.append("")
    parts.extend(brainstorm_lines(model))
    OUTPUT_SUMMARY.write_text("\n".join(parts) + "\n")

    print(f"Wrote {OUTPUT_FEATURES}")
    print(f"Wrote {OUTPUT_SUMMARY}")
    print(f"Days in model frame: {len(model)}")


if __name__ == "__main__":
    main()
