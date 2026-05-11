"""Regime-level analysis: how much do 3-tier pressure regimes explain,
and does momentum add anything within regimes?

Outputs:
  1. R² from simple 3-tier regime dummies (high/medium/no pressure)
  2. R² from finer-grained ~9-period regime coding
  3. Momentum coefficients within each regime tier
  4. Table of mean behavior/supplement/sleep/etc by regime tier
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import statsmodels.api as sm

from productivity_analysis import (
    CALENDAR_DIR,
    DAILY_SUMMARY_CSV,
    DISTRACTED_CSV,
    SUPPLEMENT_ALIASES,
    extract_daily_supplements,
    extract_work_start_time,
    load_calendar_full,
    load_calendar_sleep,
    load_daily_summary_full,
    load_distracted_stacked,
)

TARGET = "Hours Working"

# ── Fine-grained period definitions (Layer 1+2) ──────────────────────────────
# Each tuple: (label, start_date_inclusive, end_date_exclusive, tier)
# Tier: "high", "medium", "none"
PERIODS = [
    ("Early Hive (assigned tasks)", "2021-06-07", "2022-08-22", "high"),
    ("Hive research + own", "2022-08-22", "2023-07-23", "medium"),
    ("Summer 23 nothing", "2023-07-23", "2023-09-18", "none"),
    ("AI safety self-directed", "2023-09-18", "2024-05-20", "medium"),
    ("MATS", "2024-06-17", "2024-08-25", "high"),
    ("Post-MATS moping", "2024-08-26", "2025-02-03", "none"),
    ("Hadrian / vLLM", "2025-02-04", "2025-07-27", "medium"),
    ("Diesl real work", "2025-08-11", "2025-12-07", "high"),
    ("Post-Diesl misc", "2025-12-17", "2026-12-31", "none"),
]

KEY_EVENTS = ["gym", "nap", "walk", "amelia", "meditation"]

SUPPLEMENT_NAMES = sorted(set(SUPPLEMENT_ALIASES.values()))

BEHAVIOR_COLS = [
    "sleep_hours",
    "nap_hours",
    "meals_kcal",
    "meditation_min",
    "work_start_hour",
    "cal_gym",
    "cal_walk",
    "cal_nap",
    "cal_amelia",
]

MOMENTUM_FEATURES = [
    "prev_hours",
    "hours_rolling_7d_mean",
    "hours_rolling_7d_std",
    "work_streak",
    "days_since_rest",
]


def build_df() -> pd.DataFrame:
    daily = load_daily_summary_full(DAILY_SUMMARY_CSV)
    distracted = load_distracted_stacked(DISTRACTED_CSV)
    supplements = extract_daily_supplements(distracted)
    work_starts = extract_work_start_time(distracted)
    cal_sleep = load_calendar_sleep(CALENDAR_DIR)
    cal_full = load_calendar_full(CALENDAR_DIR)

    df = daily.merge(supplements, on="date", how="left")
    df = df.merge(cal_sleep, on="date", how="left")
    df = df.merge(work_starts, on="date", how="left")

    # Calendar category totals
    cat_daily = (
        cal_full.groupby(["date", "category"])["duration"].sum().unstack(fill_value=0)
    )
    for col in ["green", "things", "waste", "blue"]:
        if col not in cat_daily.columns:
            cat_daily[col] = 0.0
    cat_daily = cat_daily.rename(
        columns={
            "green": "cal_green",
            "things": "cal_things",
            "waste": "cal_waste",
            "blue": "cal_blue",
        }
    )
    df = df.merge(cat_daily, left_on="date", right_index=True, how="left")

    # Calendar event totals
    event_daily = (
        cal_full.groupby(["date", "event_lower"])["duration"]
        .sum()
        .unstack(fill_value=0)
    )
    for ev in KEY_EVENTS:
        col_name = f"cal_{ev}"
        if ev in event_daily.columns:
            df = df.merge(
                event_daily[[ev]].rename(columns={ev: col_name}),
                left_on="date",
                right_index=True,
                how="left",
            )
        else:
            df[col_name] = 0.0

    df["day_of_week"] = df["date"].dt.dayofweek
    df["is_weekend"] = (df["day_of_week"] >= 5).astype(int)

    # Binary supplement flags
    for supplement in set(SUPPLEMENT_ALIASES.values()):
        if supplement in df.columns:
            df[f"has_{supplement}"] = (df[supplement] > 0).astype(int)

    # Momentum features
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

    # Assign fine-grained period and 3-tier pressure
    df["period"] = "unassigned"
    df["pressure"] = "unassigned"
    for label, start, end, tier in PERIODS:
        mask = (df["date"] >= pd.Timestamp(start)) & (
            df["date"] < pd.Timestamp(end)
        )
        df.loc[mask, "period"] = label
        df.loc[mask, "pressure"] = tier

    df = df.fillna(0)
    return df


def fit_ols(y: pd.Series, X: pd.DataFrame, label: str) -> sm.regression.linear_model.RegressionResultsWrapper:
    X_c = sm.add_constant(X.astype(float))
    model = sm.OLS(y.astype(float), X_c).fit()
    return model


def print_section(title: str) -> None:
    print(f"\n{'='*70}")
    print(f"  {title}")
    print(f"{'='*70}\n")


def main() -> None:
    df_all = build_df()
    # Working days only (Hours Working > 0), drop weekends
    df = df_all[(df_all[TARGET] > 0) & (~df_all["is_weekend"].astype(bool))].copy()
    df = df.dropna(subset=[TARGET])
    print(f"Total working weekdays: {len(df)}")
    print(f"Date range: {df['date'].min().date()} to {df['date'].max().date()}")

    assigned = df[df["pressure"] != "unassigned"]
    unassigned = df[df["pressure"] == "unassigned"]
    print(f"Assigned to a regime tier: {len(assigned)}  ({len(unassigned)} unassigned)")
    if len(unassigned) > 0:
        print(f"  Unassigned date range: {unassigned['date'].min().date()} to {unassigned['date'].max().date()}")

    # ── 1. Simple 3-tier model ────────────────────────────────────────────
    print_section("1. THREE-TIER PRESSURE MODEL (high / medium / none)")

    work = assigned.copy()
    work["is_high"] = (work["pressure"] == "high").astype(int)
    work["is_medium"] = (work["pressure"] == "medium").astype(int)
    # reference = "none"

    regime_feats = ["is_high", "is_medium"]
    model_3tier = fit_ols(work[TARGET], work[regime_feats], "3-tier")
    print(model_3tier.summary2().tables[1].to_string())
    print(f"\n  R² = {model_3tier.rsquared:.4f},  Adj-R² = {model_3tier.rsquared_adj:.4f},  n = {model_3tier.nobs:.0f}")

    # Add day_of_week control
    model_3tier_dow = fit_ols(work[TARGET], work[regime_feats + ["day_of_week"]], "3-tier + DOW")
    print(f"\n  With day_of_week:  R² = {model_3tier_dow.rsquared:.4f},  Adj-R² = {model_3tier_dow.rsquared_adj:.4f}")

    # Means by tier
    tier_means = work.groupby("pressure")[TARGET].agg(["mean", "std", "count"])
    print(f"\n  Means by tier:")
    for tier in ["high", "medium", "none"]:
        if tier in tier_means.index:
            row = tier_means.loc[tier]
            print(f"    {tier:>8s}: {row['mean']:.2f}h  (SD={row['std']:.2f}, n={row['count']:.0f})")

    # ── 2. Fine-grained period model ─────────────────────────────────────
    print_section("2. FINE-GRAINED PERIOD MODEL (~9 periods)")

    period_dummies = pd.get_dummies(work["period"], drop_first=True, dtype=float)
    model_periods = fit_ols(work[TARGET], period_dummies, "fine-grained periods")
    print(model_periods.summary2().tables[1].to_string())
    print(f"\n  R² = {model_periods.rsquared:.4f},  Adj-R² = {model_periods.rsquared_adj:.4f},  n = {model_periods.nobs:.0f}")

    period_means = work.groupby("period")[TARGET].agg(["mean", "std", "count"]).sort_values("mean", ascending=False)
    print(f"\n  Means by period:")
    for period, row in period_means.iterrows():
        print(f"    {period:<35s}: {row['mean']:.2f}h  (SD={row['std']:.2f}, n={row['count']:.0f})")

    # ── 3. Existing regime dummies comparison ─────────────────────────────
    print_section("3. COMPARISON: existing regime dummies (is_hive, is_early_hive, is_mats, is_diesl)")

    work["is_hive"] = (work["regime"] == "Hive").astype(int)
    work["is_mats"] = (work["regime"] == "Mats").astype(int)
    work["is_diesl"] = (work["regime"] == "diesl").astype(int)
    work["is_early_hive"] = (
        (work["regime"] == "Hive") & (work["date"] < pd.Timestamp("2022-08-01"))
    ).astype(int)
    old_regime_feats = ["is_hive", "is_mats", "is_diesl", "is_early_hive"]
    model_old = fit_ols(work[TARGET], work[old_regime_feats], "old regime dummies")
    print(f"  R² = {model_old.rsquared:.4f},  Adj-R² = {model_old.rsquared_adj:.4f},  n = {model_old.nobs:.0f}")

    # ── 4. Does momentum add within regimes? ─────────────────────────────
    print_section("4. MOMENTUM WITHIN REGIME TIERS")

    mom_feats = [f for f in MOMENTUM_FEATURES if f in work.columns and work[f].std() > 0.01]
    valid_mom = work.dropna(subset=mom_feats)

    # 4a. Regime-only → regime + momentum (full sample)
    model_regime_only = fit_ols(valid_mom[TARGET], valid_mom[regime_feats], "regime only (mom sample)")
    model_regime_mom = fit_ols(valid_mom[TARGET], valid_mom[regime_feats + mom_feats], "regime + momentum")
    print(f"  Full sample (n={len(valid_mom)}):")
    print(f"    Regime only:       R² = {model_regime_only.rsquared:.4f}")
    print(f"    Regime + momentum: R² = {model_regime_mom.rsquared:.4f}  (ΔR² = {model_regime_mom.rsquared - model_regime_only.rsquared:+.4f})")

    # 4b. Momentum-only within each tier
    print(f"\n  Momentum within each tier:")
    for tier in ["high", "medium", "none"]:
        tier_df = valid_mom[valid_mom["pressure"] == tier].copy()
        if len(tier_df) < 30:
            print(f"    {tier:>8s}: too few obs (n={len(tier_df)})")
            continue
        tier_mom_feats = [f for f in mom_feats if tier_df[f].std() > 0.01]
        if not tier_mom_feats:
            print(f"    {tier:>8s}: no varying momentum features")
            continue

        model_tier_null = fit_ols(tier_df[TARGET], tier_df[["day_of_week"]], f"{tier}: DOW only")
        model_tier_mom = fit_ols(tier_df[TARGET], tier_df[["day_of_week"] + tier_mom_feats], f"{tier}: DOW + momentum")
        print(f"    {tier:>8s} (n={len(tier_df):>4d}):  DOW R²={model_tier_null.rsquared:.4f}  →  DOW+mom R²={model_tier_mom.rsquared:.4f}  (ΔR²={model_tier_mom.rsquared - model_tier_null.rsquared:+.4f})")

        # Show which momentum features matter within this tier
        coefs = model_tier_mom.summary2().tables[1]
        for feat in tier_mom_feats:
            if feat in coefs.index:
                c = coefs.loc[feat]
                sig = "*" if c["P>|t|"] < 0.05 else " "
                print(f"      {feat:<25s}: coef={c['Coef.']:+.4f}  p={c['P>|t|']:.3f} {sig}")

    # 4c. Full model: regime + momentum + DOW
    model_full_base = fit_ols(valid_mom[TARGET], valid_mom[regime_feats + ["day_of_week"] + mom_feats], "regime + DOW + momentum")
    print(f"\n  Combined (regime + DOW + momentum): R² = {model_full_base.rsquared:.4f}")

    # ── 5. Behavior/supplement means by regime tier ──────────────────────
    print_section("5. BEHAVIOR & SUPPLEMENT MEANS BY REGIME TIER")

    has_supplement_cols = [f"has_{s}" for s in SUPPLEMENT_NAMES if f"has_{s}" in work.columns]
    dose_cols = [s for s in SUPPLEMENT_NAMES if s in work.columns and work[s].std() > 0]
    summary_cols = BEHAVIOR_COLS + has_supplement_cols + [TARGET, "Energy", "Focus"]
    summary_cols = [c for c in summary_cols if c in work.columns]

    tier_summary = work.groupby("pressure")[summary_cols].mean()
    tier_counts = work.groupby("pressure")[TARGET].count()

    # Reorder tiers
    tier_order = ["high", "medium", "none"]
    tier_summary = tier_summary.reindex(tier_order)
    tier_counts = tier_counts.reindex(tier_order)

    print(f"{'Variable':<30s}", end="")
    for tier in tier_order:
        n = tier_counts.get(tier, 0)
        print(f"  {tier:>10s} (n={n:>3.0f})", end="")
    print()
    print("-" * 85)

    for col in summary_cols:
        label = col.replace("has_", "% ").replace("cal_", "cal:")
        print(f"{label:<30s}", end="")
        for tier in tier_order:
            val = tier_summary.loc[tier, col] if tier in tier_summary.index else np.nan
            if col.startswith("has_"):
                print(f"  {val*100:>14.1f}%", end="")
            else:
                print(f"  {val:>15.2f}", end="")
        print()

    # Also show dose means (only on days taken)
    print(f"\n  Mean dose (on days taken only):")
    for sup in dose_cols:
        print(f"  {sup:<25s}", end="")
        for tier in tier_order:
            tier_df = work[(work["pressure"] == tier) & (work[sup] > 0)]
            if len(tier_df) > 0:
                print(f"  {tier_df[sup].mean():>12.1f} (n={len(tier_df):>3d})", end="")
            else:
                print(f"  {'—':>18s}", end="")
        print()

    # ── 6. How much does the fine-grained regime eat into the actionable model? ──
    print_section("6. ACTIONABLE MODEL: with vs without regime tiers")

    actionable_feats = ["day_of_week", "work_start_hour"] + mom_feats + has_supplement_cols
    actionable_feats = [f for f in actionable_feats if f in valid_mom.columns and valid_mom[f].std() > 0.01]

    model_act_no_regime = fit_ols(valid_mom[TARGET], valid_mom[actionable_feats], "actionable, no regime")
    model_act_with_regime = fit_ols(valid_mom[TARGET], valid_mom[regime_feats + actionable_feats], "actionable + regime")
    print(f"  Actionable model (no regime):   R² = {model_act_no_regime.rsquared:.4f}  (n={model_act_no_regime.nobs:.0f})")
    print(f"  Actionable model + regime:      R² = {model_act_with_regime.rsquared:.4f}  (n={model_act_with_regime.nobs:.0f})")
    print(f"  ΔR² from adding regime:         {model_act_with_regime.rsquared - model_act_no_regime.rsquared:+.4f}")

    # Show how momentum coefficients change when regime is added
    print(f"\n  Momentum coefficient comparison (actionable model):")
    print(f"  {'Feature':<25s}  {'No regime':>12s}  {'With regime':>12s}  {'Change':>10s}")
    print(f"  {'-'*65}")
    for feat in mom_feats:
        coef_no = model_act_no_regime.params.get(feat, np.nan)
        coef_with = model_act_with_regime.params.get(feat, np.nan)
        if not np.isnan(coef_no) and not np.isnan(coef_with):
            pct = (coef_with - coef_no) / abs(coef_no) * 100 if coef_no != 0 else 0
            print(f"  {feat:<25s}  {coef_no:>+12.4f}  {coef_with:>+12.4f}  {pct:>+9.1f}%")


if __name__ == "__main__":
    main()
