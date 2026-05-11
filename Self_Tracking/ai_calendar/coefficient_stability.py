"""Coefficient stability analysis for the full productivity regression model."""

# ruff: noqa: E402

from __future__ import annotations

import math
import os
from pathlib import Path

OUT_DIR = Path(__file__).resolve().parent
os.environ.setdefault("MPLCONFIGDIR", str(OUT_DIR / ".mpl-cache"))

import matplotlib

matplotlib.use("Agg")

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm

from productivity_analysis import (
    CALENDAR_DIR,
    DAILY_SUMMARY_CSV,
    DISTRACTED_CSV,
    extract_daily_supplements,
    extract_work_start_time,
    load_calendar_full,
    load_calendar_sleep,
    load_daily_summary_full,
    load_distracted_stacked,
)

TARGET = "Hours Working"
ROLLING_PLOT_PATH = OUT_DIR / "coefficient_stability_rolling.png"
BOOTSTRAP_N = 2000
ROLLING_WINDOW_DAYS = 365
ROLLING_STEP_DAYS = 30
MIN_EXTRA_OBS = 8
MIN_BINARY_POSITIVE_CASES = 10
EPS = 1e-9

SUPPLEMENT_ORDER = [
    "caffeine",
    "adderall",
    "modafinil",
    "bronkaid",
    "nicotine",
    "piracetam",
    "choline",
    "ashwagandha",
    "creatine",
    "l_theanine",
    "sulbutiamine",
    "lavender",
    "zembrin",
    "potassium_gluconate",
    "zinc",
]
BASE_FEATURES = [
    "is_hive",
    "is_mats",
    "is_diesl",
    "is_weekend",
    "day_of_week",
    "cal_waste",
    "cal_things",
    "cal_nap",
    "cal_amelia",
    "work_start_hour",
    "streak",
]
KEY_EVENTS = ["amelia", "gym", "walk", "nap", "meditation"]
REGIME_FEATURES = ["is_hive", "is_mats", "is_diesl", "is_early_hive"]
TIME_CONTROL_FEATURES = ["is_weekend", "day_of_week"]
SUPPLEMENT_PREFIX = "has_"


def build_full_model_sample() -> (
    tuple[pd.DataFrame, list[str], sm.regression.linear_model.RegressionResultsWrapper]
):
    """Recreate the exact full multivariate model used in prior reporting."""
    print("Loading data and rebuilding the full multivariate model sample...")
    daily = load_daily_summary_full(DAILY_SUMMARY_CSV)
    distracted = load_distracted_stacked(DISTRACTED_CSV)
    supplements = extract_daily_supplements(distracted)
    work_starts = extract_work_start_time(distracted)
    cal_sleep = load_calendar_sleep(CALENDAR_DIR)
    cal_full = load_calendar_full(CALENDAR_DIR)

    df = daily.merge(supplements, on="date", how="left")
    df = df.merge(cal_sleep, on="date", how="left")
    df = df.merge(work_starts, on="date", how="left")
    df["day_of_week"] = df["date"].dt.dayofweek
    df["is_weekend"] = (df["day_of_week"] >= 5).astype(int)

    cat_daily = (
        cal_full.groupby(["date", "category"])["duration"].sum().unstack(fill_value=0)
    )
    for cat in ["green", "things", "waste", "blue"]:
        if cat not in cat_daily.columns:
            cat_daily[cat] = 0.0
    cat_daily = cat_daily.rename(
        columns={
            "green": "cal_green",
            "things": "cal_things",
            "waste": "cal_waste",
            "blue": "cal_blue",
        }
    )
    df = df.merge(cat_daily, left_on="date", right_index=True, how="left")
    for column in ["cal_green", "cal_things", "cal_waste", "cal_blue"]:
        df[column] = df[column].fillna(0)

    event_daily = (
        cal_full.groupby(["date", "event_lower"])["duration"]
        .sum()
        .unstack(fill_value=0)
    )
    for event_name in KEY_EVENTS:
        column_name = f"cal_{event_name.replace(' ', '_')}"
        if event_name in event_daily.columns:
            df = df.merge(
                event_daily[[event_name]].rename(columns={event_name: column_name}),
                left_on="date",
                right_index=True,
                how="left",
            )
            df[column_name] = df[column_name].fillna(0)

    df["is_hive"] = (df["regime"] == "Hive").astype(int)
    df["is_mats"] = (df["regime"] == "Mats").astype(int)
    df["is_diesl"] = (df["regime"] == "diesl").astype(int)
    df["is_early_hive"] = (
        (df["regime"] == "Hive") & (df["date"] < pd.Timestamp("2022-08-01"))
    ).astype(int)

    df = df.sort_values("date").reset_index(drop=True)
    df["worked"] = df[TARGET] > 2
    streak = 0
    streaks: list[int] = []
    for worked in df["worked"]:
        streak = streak + 1 if worked else 0
        streaks.append(streak)
    df["streak"] = streaks

    valid = df[df[TARGET] > 0].copy()

    for supplement in SUPPLEMENT_ORDER:
        if supplement in valid.columns:
            valid[f"{SUPPLEMENT_PREFIX}{supplement}"] = (valid[supplement] > 0).astype(
                int
            )

    if "cal_meditation" in valid.columns:
        valid["has_meditation"] = (valid["cal_meditation"] > 0).astype(int)

    features = list(BASE_FEATURES)
    for supplement in SUPPLEMENT_ORDER:
        if supplement in valid.columns and (valid[supplement] > 0).sum() > 30:
            features.append(f"{SUPPLEMENT_PREFIX}{supplement}")

    for column_name in ["cal_gym", "cal_walk"]:
        if column_name in valid.columns:
            features.append(column_name)

    if "has_meditation" in valid.columns:
        features.append("has_meditation")

    features.append("is_early_hive")
    features = [
        feature
        for feature in features
        if feature in valid.columns and valid[feature].notna().sum() > 100
    ]

    sample = valid[["date", TARGET] + features].dropna().copy()
    model = fit_ols(sample, features)

    print(
        f"Full model rebuilt: R^2={model.rsquared:.3f}, adj R^2={model.rsquared_adj:.3f}, n={int(model.nobs)}"
    )
    print(f"Features ({len(features)}): {features}")
    return sample, features, model


def fit_ols(
    data: pd.DataFrame, features: list[str]
) -> sm.regression.linear_model.RegressionResultsWrapper:
    X = sm.add_constant(data[features], has_constant="add")
    return sm.OLS(data[TARGET], X).fit()


def fit_ols_optional(
    data: pd.DataFrame, features: list[str]
) -> tuple[
    sm.regression.linear_model.RegressionResultsWrapper | None, list[str], pd.DataFrame
]:
    use_features = [
        feature
        for feature in features
        if feature in data.columns and data[feature].notna().sum() > 0
    ]
    if not use_features:
        return None, [], data.iloc[0:0].copy()

    sample = data[["date", TARGET] + use_features].dropna().copy()
    use_features = [
        feature for feature in use_features if sample[feature].std(ddof=0) > EPS
    ]
    if len(use_features) == 0:
        return None, [], sample

    sample = data[["date", TARGET] + use_features].dropna().copy()
    if len(sample) <= len(use_features) + MIN_EXTRA_OBS:
        return None, use_features, sample

    try:
        model = fit_ols(sample, use_features)
    except np.linalg.LinAlgError:
        return None, use_features, sample
    return model, use_features, sample


def is_binary_feature(sample: pd.DataFrame, feature: str) -> bool:
    values = sample[feature].dropna().unique()
    if len(values) == 0:
        return False
    return set(np.round(values, 8)).issubset({0.0, 1.0})


def has_feature_support(sample: pd.DataFrame, feature: str) -> bool:
    if feature not in sample.columns:
        return False
    values = sample[feature].dropna()
    if values.empty:
        return False
    if is_binary_feature(sample, feature):
        positive_count = int((values > 0.5).sum())
        negative_count = int((values <= 0.5).sum())
        return (
            positive_count >= MIN_BINARY_POSITIVE_CASES
            and negative_count >= MIN_BINARY_POSITIVE_CASES
        )
    return values.std(ddof=0) > EPS


def bootstrap_analysis(
    sample: pd.DataFrame,
    features: list[str],
    full_model: sm.regression.linear_model.RegressionResultsWrapper,
) -> pd.DataFrame:
    print(f"\n1. Bootstrap confidence intervals ({BOOTSTRAP_N} resamples)")
    design = sm.add_constant(sample[features], has_constant="add")
    X = design.to_numpy()
    y = sample[TARGET].to_numpy()
    rng = np.random.default_rng(42)

    coef_names = design.columns.tolist()
    coef_storage = np.empty((BOOTSTRAP_N, len(coef_names)))

    for i in range(BOOTSTRAP_N):
        idx = rng.integers(0, len(sample), len(sample))
        coef_storage[i] = np.linalg.lstsq(X[idx], y[idx], rcond=None)[0]
        if (i + 1) % 250 == 0:
            print(f"  completed {i + 1}/{BOOTSTRAP_N}")

    boot_df = pd.DataFrame(coef_storage, columns=coef_names)
    records: list[dict[str, object]] = []
    for feature in features:
        boot_coef = boot_df[feature]
        ci_low = float(boot_coef.quantile(0.025))
        ci_high = float(boot_coef.quantile(0.975))
        boot_se = float(boot_coef.std(ddof=1))
        ols_coef = float(full_model.params[feature])
        ols_se = float(full_model.bse[feature])
        records.append(
            {
                "feature": feature,
                "ols_estimate": ols_coef,
                "ols_se": ols_se,
                "boot_se": boot_se,
                "boot_ci_low": ci_low,
                "boot_ci_high": ci_high,
                "ci_crosses_zero": ci_low <= 0 <= ci_high,
                "boot_se_gt_1p5x_ols": boot_se > 1.5 * ols_se,
            }
        )

    result = pd.DataFrame(records).sort_values(
        "ols_estimate", key=np.abs, ascending=False
    )
    display_cols = [
        "feature",
        "ols_estimate",
        "ols_se",
        "boot_se",
        "boot_ci_low",
        "boot_ci_high",
        "ci_crosses_zero",
        "boot_se_gt_1p5x_ols",
    ]
    print(
        result[display_cols].to_string(index=False, float_format=lambda x: f"{x: .3f}")
    )
    return result


def leave_one_year_out_analysis(
    sample: pd.DataFrame,
    features: list[str],
    full_model: sm.regression.linear_model.RegressionResultsWrapper,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    print("\n2. Leave-one-year-out sensitivity")
    year_records: list[dict[str, object]] = []
    years = sorted(sample["date"].dt.year.unique())
    sparse_fold_counts = {feature: 0 for feature in features}

    for year in years:
        subset = sample[sample["date"].dt.year != year].copy()
        model, use_features, used_sample = fit_ols_optional(subset, features)
        print(f"  drop {year}: n={len(used_sample)}")
        if model is None:
            continue
        for feature in features:
            if not has_feature_support(used_sample, feature):
                sparse_fold_counts[feature] += 1
                coef = np.nan
            else:
                coef = model.params.get(feature, np.nan)
            year_records.append(
                {
                    "feature": feature,
                    "dropped_year": year,
                    "coef": coef,
                }
            )

    year_df = pd.DataFrame(year_records)
    summaries: list[dict[str, object]] = []
    for feature in features:
        values = year_df.loc[year_df["feature"] == feature, "coef"].dropna()
        full_coef = float(full_model.params[feature])
        coef_min = float(values.min()) if not values.empty else np.nan
        coef_max = float(values.max()) if not values.empty else np.nan
        sign_flip = values.lt(0).any() and values.gt(0).any()
        relative_change = (
            float(np.max(np.abs(values - full_coef)) / max(abs(full_coef), 0.05))
            if not values.empty
            else np.nan
        )
        summaries.append(
            {
                "feature": feature,
                "full_coef": full_coef,
                "lyo_min": coef_min,
                "lyo_max": coef_max,
                "lyo_range": coef_max - coef_min if not values.empty else np.nan,
                "lyo_sign_flip": sign_flip,
                "lyo_change_gt_100pct": (
                    relative_change > 1.0 if not np.isnan(relative_change) else False
                ),
                "lyo_max_relative_change": relative_change,
                "lyo_valid_folds": int(values.notna().sum()),
                "lyo_sparse_folds": sparse_fold_counts[feature],
            }
        )

    summary_df = pd.DataFrame(summaries).sort_values(
        "full_coef", key=np.abs, ascending=False
    )
    print(
        summary_df[
            [
                "feature",
                "full_coef",
                "lyo_min",
                "lyo_max",
                "lyo_range",
                "lyo_valid_folds",
                "lyo_sparse_folds",
                "lyo_sign_flip",
                "lyo_change_gt_100pct",
            ]
        ].to_string(index=False, float_format=lambda x: f"{x: .3f}")
    )
    return summary_df, year_df


def rolling_window_analysis(
    sample: pd.DataFrame,
    features: list[str],
    full_model: sm.regression.linear_model.RegressionResultsWrapper,
) -> pd.DataFrame:
    print("\n3. Rolling 365-day window estimation")
    rolling_rows: list[dict[str, object]] = []

    start_date = sample["date"].min().normalize()
    end_date = sample["date"].max().normalize()
    window_start = start_date
    while window_start + pd.Timedelta(days=ROLLING_WINDOW_DAYS - 1) <= end_date:
        window_end = window_start + pd.Timedelta(days=ROLLING_WINDOW_DAYS - 1)
        subset = sample[
            (sample["date"] >= window_start) & (sample["date"] <= window_end)
        ].copy()
        midpoint = window_start + pd.Timedelta(days=ROLLING_WINDOW_DAYS // 2)
        model, _, used_sample = fit_ols_optional(subset, features)
        print(
            f"  window {window_start.date()} to {window_end.date()}: n={len(used_sample)}"
        )
        if model is not None:
            for feature in features:
                rolling_rows.append(
                    {
                        "window_mid": midpoint,
                        "feature": feature,
                        "coef": model.params.get(feature, np.nan),
                    }
                )
        window_start += pd.Timedelta(days=ROLLING_STEP_DAYS)

    rolling_df = pd.DataFrame(rolling_rows)
    summary_rows: list[dict[str, object]] = []
    for feature in features:
        values = rolling_df.loc[rolling_df["feature"] == feature, "coef"].dropna()
        full_coef = float(full_model.params[feature])
        summary_rows.append(
            {
                "feature": feature,
                "rolling_min": float(values.min()) if not values.empty else np.nan,
                "rolling_max": float(values.max()) if not values.empty else np.nan,
                "rolling_range": (
                    float(values.max() - values.min()) if not values.empty else np.nan
                ),
                "rolling_sign_flip": values.lt(0).any() and values.gt(0).any(),
                "rolling_max_relative_change": (
                    float(
                        np.max(np.abs(values - full_coef)) / max(abs(full_coef), 0.05)
                    )
                    if not values.empty
                    else np.nan
                ),
                "rolling_windows": int(values.notna().sum()),
            }
        )

    save_rolling_plot(rolling_df, features, full_model)
    summary_df = pd.DataFrame(summary_rows).sort_values("feature")
    print(f"Saved rolling coefficient plot to {ROLLING_PLOT_PATH}")
    return summary_df


def save_rolling_plot(
    rolling_df: pd.DataFrame,
    features: list[str],
    full_model: sm.regression.linear_model.RegressionResultsWrapper,
) -> None:
    n_cols = 4
    n_rows = math.ceil(len(features) / n_cols)
    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(18, 3.3 * n_rows),
        sharex=True,
        constrained_layout=True,
    )
    axes_array = np.atleast_1d(axes).reshape(n_rows, n_cols)

    for idx, feature in enumerate(features):
        ax = axes_array[idx // n_cols, idx % n_cols]
        subset = rolling_df[rolling_df["feature"] == feature].sort_values("window_mid")
        if subset.empty:
            ax.set_title(feature)
            ax.text(
                0.5,
                0.5,
                "no estimate",
                ha="center",
                va="center",
                transform=ax.transAxes,
            )
            continue
        ax.plot(
            subset["window_mid"],
            subset["coef"],
            marker="o",
            linewidth=1.6,
            markersize=3,
        )
        ax.axhline(
            float(full_model.params[feature]),
            color="black",
            linestyle="--",
            linewidth=1,
        )
        ax.axhline(0, color="gray", linestyle=":", linewidth=0.8)
        ax.set_title(feature)
        ax.xaxis.set_major_locator(mdates.YearLocator())
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
        ax.tick_params(axis="x", rotation=45)

    for idx in range(len(features), n_rows * n_cols):
        axes_array[idx // n_cols, idx % n_cols].axis("off")

    fig.suptitle("Rolling 365-day coefficient estimates (30-day step)", fontsize=14)
    fig.savefig(ROLLING_PLOT_PATH, dpi=180, bbox_inches="tight")
    plt.close(fig)


def feature_blocks(features: list[str]) -> dict[str, str]:
    block_map: dict[str, str] = {}
    for feature in features:
        if feature in REGIME_FEATURES:
            block_map[feature] = "regime"
        elif feature in TIME_CONTROL_FEATURES:
            block_map[feature] = "time_controls"
        elif feature.startswith(SUPPLEMENT_PREFIX):
            block_map[feature] = "supplements"
        elif feature in {"cal_waste", "cal_things"}:
            block_map[feature] = "calendar_categories"
        elif feature in {
            "cal_nap",
            "cal_amelia",
            "cal_gym",
            "cal_walk",
            "has_meditation",
        }:
            block_map[feature] = "activities"
        elif feature in {"work_start_hour"}:
            block_map[feature] = "timing"
        elif feature in {"streak"}:
            block_map[feature] = "momentum"
        else:
            block_map[feature] = "other"
    return block_map


def spec_features_for_feature(
    feature: str, features: list[str]
) -> dict[str, list[str]]:
    block_map = feature_blocks(features)
    same_block = [
        candidate
        for candidate in features
        if block_map[candidate] == block_map[feature]
    ]
    minus_same_block = [
        candidate
        for candidate in features
        if candidate == feature or candidate not in same_block
    ]

    if block_map[feature] == "regime":
        minimal = [feature] + TIME_CONTROL_FEATURES
    else:
        minimal = [feature] + REGIME_FEATURES + TIME_CONTROL_FEATURES

    minimal = [
        candidate
        for candidate in minimal
        if candidate in features or candidate == feature
    ]
    minimal = list(dict.fromkeys(minimal))
    return {
        "full": list(features),
        "minus_same_block": minus_same_block,
        "minimal": minimal,
    }


def specification_sensitivity_analysis(
    sample: pd.DataFrame,
    features: list[str],
    full_model: sm.regression.linear_model.RegressionResultsWrapper,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    print("\n4. Specification sensitivity")
    rows: list[dict[str, object]] = []

    for feature in features:
        spec_map = spec_features_for_feature(feature, features)
        full_coef = float(full_model.params[feature])
        for spec_name, spec_features in spec_map.items():
            model, _, used_sample = fit_ols_optional(sample, spec_features)
            coef = np.nan if model is None else model.params.get(feature, np.nan)
            rows.append(
                {
                    "feature": feature,
                    "spec": spec_name,
                    "coef": coef,
                    "n": len(used_sample),
                }
            )
        print(f"  finished specs for {feature}")

    spec_df = pd.DataFrame(rows)
    summary_rows: list[dict[str, object]] = []
    for feature in features:
        values = spec_df.loc[spec_df["feature"] == feature, "coef"].dropna()
        full_coef = float(full_model.params[feature])
        summary_rows.append(
            {
                "feature": feature,
                "full_coef": full_coef,
                "minus_same_block_coef": spec_df.loc[
                    (spec_df["feature"] == feature)
                    & (spec_df["spec"] == "minus_same_block"),
                    "coef",
                ].iloc[0],
                "minimal_coef": spec_df.loc[
                    (spec_df["feature"] == feature) & (spec_df["spec"] == "minimal"),
                    "coef",
                ].iloc[0],
                "spec_max_abs_change": (
                    float(np.max(np.abs(values - full_coef)))
                    if not values.empty
                    else np.nan
                ),
                "spec_max_relative_change": (
                    float(
                        np.max(np.abs(values - full_coef)) / max(abs(full_coef), 0.05)
                    )
                    if not values.empty
                    else np.nan
                ),
            }
        )

    summary_df = pd.DataFrame(summary_rows).sort_values(
        "full_coef", key=np.abs, ascending=False
    )
    print(
        summary_df[
            [
                "feature",
                "full_coef",
                "minus_same_block_coef",
                "minimal_coef",
                "spec_max_abs_change",
            ]
        ].to_string(index=False, float_format=lambda x: f"{x: .3f}")
    )
    return summary_df, spec_df


def cooks_distance_analysis(
    sample: pd.DataFrame,
    features: list[str],
    full_model: sm.regression.linear_model.RegressionResultsWrapper,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    print("\n5. Influential observation diagnostics")
    influence = full_model.get_influence()
    cooks = pd.Series(
        influence.cooks_distance[0], index=sample.index, name="cooks_distance"
    )
    threshold = 4 / len(sample)
    n_flagged = int((cooks > threshold).sum())
    top_n = max(1, math.ceil(len(sample) * 0.01))
    top_idx = cooks.nlargest(top_n).index
    trimmed = sample.drop(index=top_idx).copy()
    trimmed_model = fit_ols(trimmed, features)

    print(f"  Cook's distance threshold 4/n = {threshold:.5f}")
    print(f"  Observations above threshold: {n_flagged} / {len(sample)}")
    print(f"  Refit excluding top 1% most influential points: removed {top_n} rows")

    rows: list[dict[str, object]] = []
    for feature in features:
        full_coef = float(full_model.params[feature])
        trimmed_coef = float(trimmed_model.params[feature])
        rows.append(
            {
                "feature": feature,
                "full_coef": full_coef,
                "trimmed_coef": trimmed_coef,
                "cooks_abs_change": abs(trimmed_coef - full_coef),
                "cooks_relative_change": abs(trimmed_coef - full_coef)
                / max(abs(full_coef), 0.05),
                "cooks_sign_flip": np.sign(trimmed_coef) != np.sign(full_coef)
                and abs(trimmed_coef) > EPS
                and abs(full_coef) > EPS,
            }
        )

    coef_df = pd.DataFrame(rows).sort_values("full_coef", key=np.abs, ascending=False)
    print(
        coef_df[
            [
                "feature",
                "full_coef",
                "trimmed_coef",
                "cooks_abs_change",
                "cooks_sign_flip",
            ]
        ].to_string(index=False, float_format=lambda x: f"{x: .3f}")
    )

    observation_df = sample[["date", TARGET]].copy()
    observation_df["cooks_distance"] = cooks.values
    observation_df["flagged_4_over_n"] = observation_df["cooks_distance"] > threshold
    observation_df["excluded_top_1pct"] = observation_df.index.isin(top_idx)
    return coef_df, observation_df


def assign_stability_grade(row: pd.Series) -> str:
    major_flags = 0
    minor_flags = 0

    if row["ci_crosses_zero"]:
        major_flags += 1
    if row["boot_se_gt_1p5x_ols"]:
        minor_flags += 1
    if row["lyo_sign_flip"]:
        major_flags += 1
    elif row["lyo_change_gt_100pct"]:
        minor_flags += 1
    if row["lyo_sparse_folds"] > 0:
        minor_flags += 1
    if row["rolling_sign_flip"]:
        major_flags += 1
    elif row["rolling_max_relative_change"] > 1.0:
        minor_flags += 1
    if row["spec_max_relative_change"] > 1.0:
        major_flags += 1
    elif row["spec_max_relative_change"] > 0.5:
        minor_flags += 1
    if row["cooks_sign_flip"]:
        major_flags += 1
    elif row["cooks_relative_change"] > 0.5:
        minor_flags += 1

    if major_flags == 0 and minor_flags == 0:
        return "A"
    if major_flags == 0 and minor_flags <= 1:
        return "B"
    if major_flags <= 1 and minor_flags <= 2:
        return "C"
    return "F"


def build_summary_table(
    full_model: sm.regression.linear_model.RegressionResultsWrapper,
    bootstrap_df: pd.DataFrame,
    lyo_df: pd.DataFrame,
    rolling_df: pd.DataFrame,
    spec_df: pd.DataFrame,
    cooks_df: pd.DataFrame,
) -> pd.DataFrame:
    summary = (
        bootstrap_df.merge(lyo_df, on="feature", how="left")
        .merge(rolling_df, on="feature", how="left")
        .merge(spec_df, on="feature", how="left", suffixes=("", "_spec"))
        .merge(cooks_df, on="feature", how="left", suffixes=("", "_cooks"))
    )
    summary["stability_grade"] = summary.apply(assign_stability_grade, axis=1)
    summary["bootstrap_ci"] = summary.apply(
        lambda row: f"[{row['boot_ci_low']:.3f}, {row['boot_ci_high']:.3f}]",
        axis=1,
    )
    summary["leave_year_out_range"] = summary.apply(
        lambda row: f"[{row['lyo_min']:.3f}, {row['lyo_max']:.3f}]",
        axis=1,
    )
    summary["cooks_sensitivity"] = summary["cooks_abs_change"]
    summary = summary.sort_values("ols_estimate", key=np.abs, ascending=False)

    print("\n6. Summary table")
    print(
        summary[
            [
                "feature",
                "ols_estimate",
                "bootstrap_ci",
                "leave_year_out_range",
                "spec_max_abs_change",
                "cooks_sensitivity",
                "stability_grade",
            ]
        ].to_string(index=False)
    )

    print("\nStability grades")
    grade_table = (
        summary.groupby("stability_grade")["feature"].apply(list).sort_index().to_dict()
    )
    for grade, feature_list in grade_table.items():
        print(f"  {grade}: {feature_list}")

    print("\nLikely trustworthy tradeoffs (A/B)")
    trustworthy = summary[summary["stability_grade"].isin(["A", "B"])]
    print(
        trustworthy[
            ["feature", "ols_estimate", "bootstrap_ci", "stability_grade"]
        ].to_string(index=False)
    )

    print("\nLikely fragile tradeoffs (C/F)")
    fragile = summary[summary["stability_grade"].isin(["C", "F"])]
    print(
        fragile[
            ["feature", "ols_estimate", "bootstrap_ci", "stability_grade"]
        ].to_string(index=False)
    )
    return summary


def main() -> None:
    sample, features, full_model = build_full_model_sample()
    bootstrap_df = bootstrap_analysis(sample, features, full_model)
    lyo_summary, _ = leave_one_year_out_analysis(sample, features, full_model)
    rolling_summary = rolling_window_analysis(sample, features, full_model)
    spec_summary, _ = specification_sensitivity_analysis(sample, features, full_model)
    cooks_summary, _ = cooks_distance_analysis(sample, features, full_model)
    build_summary_table(
        full_model,
        bootstrap_df,
        lyo_summary,
        rolling_summary,
        spec_summary,
        cooks_summary,
    )


if __name__ == "__main__":
    main()
