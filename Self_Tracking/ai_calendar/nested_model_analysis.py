"""Nested model decomposition for productivity drivers.

Separates background regime effects from momentum/state effects and same-day
behavioral proxies, while pruning highly collinear and non-predictive features.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor

from productivity_analysis import (
    CALENDAR_DIR,
    DAILY_SUMMARY_CSV,
    DISTRACTED_CSV,
    SUPPLEMENT_ALIASES,
    extract_daily_supplements,
    extract_work_start_time,
    load_calendar_sleep,
    load_daily_summary_full,
    load_distracted_stacked,
)

TARGET = "Hours Working"
CONTROLS = ["day_of_week"]
REGIME_BLOCK = ["is_hive", "is_mats", "is_diesl", "is_early_hive"]
MOMENTUM_BLOCK = [
    "prev_hours",
    "hours_rolling_7d_mean",
    "hours_rolling_7d_std",
    "work_streak",
    "days_since_rest",
]
BEHAVIOR_BLOCK = [
    "sleep_hours",
    "nap_hours",
    "meals_kcal",
    "meditation_min",
    "caffeine_any",
    "adderall_any",
    "modafinil_any",
    "piracetam_any",
    "choline_any",
    "nicotine_any",
]
TIMING_BLOCK = ["work_start_hour"]
STATE_CONTROL_BLOCK = [
    "day_of_week",
    "prev_hours",
    "hours_rolling_7d_mean",
    "hours_rolling_7d_std",
]
ACTIONABLE_LEVER_BLOCK = [
    "sleep_hours",
    "nap_hours",
    "days_since_rest",
    "meditation_min",
    "caffeine_any",
    "adderall_any",
    "modafinil_any",
    "nicotine_any",
    "work_start_hour",
]
HIGH_CORR_THRESHOLD = 0.9
VIF_THRESHOLD = 5.0
P_THRESHOLD = 0.05


@dataclass
class ModelSummary:
    name: str
    features: list[str]
    model: sm.regression.linear_model.RegressionResultsWrapper


def build_base_df() -> pd.DataFrame:
    daily = load_daily_summary_full(DAILY_SUMMARY_CSV)
    distracted = load_distracted_stacked(DISTRACTED_CSV)
    supplements = extract_daily_supplements(distracted)
    work_starts = extract_work_start_time(distracted)
    cal_sleep = load_calendar_sleep(CALENDAR_DIR)

    df = daily.merge(supplements, on="date", how="left")
    df = df.merge(cal_sleep, on="date", how="left")
    df = df.merge(work_starts, on="date", how="left")
    df["day_of_week"] = df["date"].dt.dayofweek
    df["is_weekend"] = (df["day_of_week"] >= 5).astype(int)
    df["is_hive"] = (df["regime"] == "Hive").astype(int)
    df["is_mats"] = (df["regime"] == "Mats").astype(int)
    df["is_diesl"] = (df["regime"] == "diesl").astype(int)
    df["is_early_hive"] = (
        (df["regime"] == "Hive") & (df["date"] < pd.Timestamp("2022-08-01"))
    ).astype(int)

    for supplement in set(SUPPLEMENT_ALIASES.values()):
        if supplement in df.columns:
            df[f"{supplement}_any"] = (df[supplement] > 0).astype(int)

    df = df.sort_values("date").reset_index(drop=True)
    df["prev_hours"] = df[TARGET].shift(1)
    df["hours_rolling_7d_mean"] = df[TARGET].rolling(7, min_periods=3).mean().shift(1)
    df["hours_rolling_7d_std"] = df[TARGET].rolling(7, min_periods=3).std().shift(1)

    worked = (df[TARGET] >= 2).astype(int)
    streak = worked.copy()
    for i in range(1, len(streak)):
        streak.iloc[i] = streak.iloc[i - 1] + 1 if streak.iloc[i] == 1 else 0
    df["work_streak"] = streak.shift(1)

    rest_day = (df[TARGET] < 2).astype(int)
    days_since_rest = rest_day.copy()
    for i in range(1, len(days_since_rest)):
        days_since_rest.iloc[i] = (
            days_since_rest.iloc[i - 1] + 1 if rest_day.iloc[i] == 0 else 0
        )
    df["days_since_rest"] = days_since_rest

    return df[(df[TARGET] > 0) & (~df["is_weekend"].astype(bool))].copy()


def candidate_features(df: pd.DataFrame) -> list[str]:
    features = CONTROLS + REGIME_BLOCK + MOMENTUM_BLOCK + BEHAVIOR_BLOCK + TIMING_BLOCK
    return [
        feature
        for feature in features
        if feature in df.columns and df[feature].std() > 0.01
    ]


def choose_from_pair(
    sample: pd.DataFrame, feature_a: str, feature_b: str, controls: list[str]
) -> tuple[str, str, dict[str, float]]:
    scores: dict[str, float] = {}
    for feature in (feature_a, feature_b):
        X = sm.add_constant(sample[controls + [feature]])
        model = sm.OLS(sample[TARGET], X).fit()
        scores[feature] = model.rsquared_adj
    keep = max(scores, key=scores.get)
    drop = feature_b if keep == feature_a else feature_a
    return keep, drop, scores


def prune_high_corr(
    sample: pd.DataFrame, features: list[str]
) -> tuple[list[str], list[str]]:
    kept = list(features)
    decisions: list[str] = []
    corr = sample[features].corr().abs()

    for feature_a in features:
        for feature_b in features:
            if feature_a >= feature_b or feature_a not in kept or feature_b not in kept:
                continue
            if corr.loc[feature_a, feature_b] < HIGH_CORR_THRESHOLD:
                continue
            keep, drop, scores = choose_from_pair(
                sample, feature_a, feature_b, CONTROLS
            )
            kept.remove(drop)
            decisions.append(
                f"drop {drop} for collinearity with {keep} "
                f"(corr={corr.loc[feature_a, feature_b]:.3f}, "
                f"adjR2 keep={scores[keep]:.3f}, drop={scores[drop]:.3f})"
            )

    return kept, decisions


def highest_vif(sample: pd.DataFrame, features: list[str]) -> tuple[str, float]:
    X = sm.add_constant(sample[features])
    pairs = []
    for i, column in enumerate(X.columns):
        if column == "const":
            continue
        pairs.append((column, variance_inflation_factor(X.values, i)))
    return max(pairs, key=lambda item: item[1])


def prune_vif(sample: pd.DataFrame, features: list[str]) -> tuple[list[str], list[str]]:
    kept = list(features)
    decisions: list[str] = []
    protected = set(CONTROLS + REGIME_BLOCK)

    while len(kept) > len(protected) + 1:
        feature, vif = highest_vif(sample, kept)
        if vif <= VIF_THRESHOLD:
            break
        if feature in protected:
            break
        kept.remove(feature)
        decisions.append(f"drop {feature} for VIF={vif:.2f}")

    return kept, decisions


def fit_model(sample: pd.DataFrame, name: str, features: list[str]) -> ModelSummary:
    X = sm.add_constant(sample[features])
    model = sm.OLS(sample[TARGET], X).fit()
    return ModelSummary(name=name, features=features, model=model)


def print_model_summary(
    summary: ModelSummary, previous: ModelSummary | None = None
) -> None:
    model = summary.model
    print(f"\n{summary.name}")
    print(f"  features: {summary.features}")
    print(
        f"  R²={model.rsquared:.3f}, adjR²={model.rsquared_adj:.3f}, "
        f"AIC={model.aic:.1f}, BIC={model.bic:.1f}, n={int(model.nobs)}"
    )
    if previous is not None:
        f_stat, p_value, _ = model.compare_f_test(previous.model)
        print(
            f"  ΔR² vs {previous.name}: {model.rsquared - previous.model.rsquared:+.3f}, "
            f"nested-model p={p_value:.4g}"
        )
    for feature in summary.features:
        print(
            f"  {feature:22s} coef={model.params[feature]:+7.3f} "
            f"p={model.pvalues[feature]:.4f}"
        )


def backward_select(
    sample: pd.DataFrame, initial_features: list[str]
) -> tuple[list[str], list[str]]:
    kept = list(initial_features)
    decisions: list[str] = []
    protected = set(CONTROLS)

    while True:
        model = fit_model(sample, "candidate", kept).model
        removable = [feature for feature in kept if feature not in protected]
        pvalues = {feature: model.pvalues[feature] for feature in removable}
        if not pvalues:
            break
        worst_feature = max(pvalues, key=pvalues.get)
        worst_p = pvalues[worst_feature]
        if worst_p <= P_THRESHOLD:
            break
        kept.remove(worst_feature)
        decisions.append(f"drop {worst_feature} for p={worst_p:.4f}")

    return kept, decisions


def print_std_coefficients(
    sample: pd.DataFrame, features: list[str], label: str
) -> None:
    std_X = (sample[features] - sample[features].mean()) / sample[features].std()
    std_model = sm.OLS(sample[TARGET], sm.add_constant(std_X)).fit()
    print(f"\nStandardized coefficients: {label}")
    ordered = std_model.params.drop("const").sort_values(key=np.abs, ascending=False)
    for feature, coef in ordered.items():
        print(
            f"  {feature:22s} std_coef={coef:+.3f} p={std_model.pvalues[feature]:.4f}"
        )


def run_actionable_model(sample: pd.DataFrame, pruned_features: list[str]) -> None:
    state_features = [
        feature for feature in STATE_CONTROL_BLOCK if feature in pruned_features
    ]
    lever_features = [
        feature for feature in ACTIONABLE_LEVER_BLOCK if feature in pruned_features
    ]
    actionable_seed = state_features + lever_features
    selected_features, decisions = backward_select(sample, actionable_seed)

    print("\nActionable pruning")
    for decision in decisions:
        print(f"  {decision}")

    actionable_summary = fit_model(
        sample,
        "Actionable model: state controls + plausible levers",
        selected_features,
    )
    print_model_summary(actionable_summary)
    print_std_coefficients(sample, selected_features, "actionable model")


def main() -> None:
    df = build_base_df()
    features = candidate_features(df)
    sample = df.dropna(subset=features + [TARGET]).copy()

    print("NESTED PRODUCTIVITY MODEL")
    print(f"Observations on common sample: {len(sample)}")
    print(f"Date range: {sample['date'].min().date()} to {sample['date'].max().date()}")

    pruned_corr, corr_decisions = prune_high_corr(sample, features)
    pruned_features, vif_decisions = prune_vif(sample, pruned_corr)

    print("\nCollinearity pruning")
    for decision in corr_decisions + vif_decisions:
        print(f"  {decision}")

    regime_features = [
        feature for feature in REGIME_BLOCK + CONTROLS if feature in pruned_features
    ]
    momentum_features = [
        feature for feature in MOMENTUM_BLOCK if feature in pruned_features
    ]
    behavior_features = [
        feature for feature in BEHAVIOR_BLOCK if feature in pruned_features
    ]
    timing_features = [
        feature for feature in TIMING_BLOCK if feature in pruned_features
    ]

    nested_models = [
        fit_model(sample, "Model 1: Regime + DOW", regime_features),
        fit_model(sample, "Model 2: Momentum + DOW", CONTROLS + momentum_features),
        fit_model(
            sample,
            "Model 3: Regime + Momentum + DOW",
            regime_features + momentum_features,
        ),
        fit_model(
            sample,
            "Model 4: Add Behavior",
            regime_features + momentum_features + behavior_features,
        ),
        fit_model(
            sample,
            "Model 5: Add Timing",
            regime_features + momentum_features + behavior_features + timing_features,
        ),
    ]

    previous: ModelSummary | None = None
    for summary in nested_models:
        print_model_summary(summary, previous)
        previous = summary

    final_seed = nested_models[-1].features
    selected_features, selection_decisions = backward_select(sample, final_seed)

    print("\nPredictive pruning")
    for decision in selection_decisions:
        print(f"  {decision}")

    final_summary = fit_model(sample, "Final pruned model", selected_features)
    print_model_summary(final_summary)
    print_std_coefficients(sample, selected_features, "final predictive model")
    run_actionable_model(sample, pruned_features)


if __name__ == "__main__":
    main()
