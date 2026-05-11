"""Compare the full multivariate model under two regime codings:
  A) Old regime dummies: is_hive, is_early_hive, is_mats, is_diesl (4 dummies)
  B) 3-tier pressure: is_high, is_medium (2 dummies, ref=none)

Outputs:
  1. Side-by-side coefficient table for all shared features
  2. R² comparison
  3. Bootstrap CIs for both models (are shared-feature CIs overlapping?)
  4. Which coefficients move the most? Is the movement within prior bootstrap CIs?
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

from regime_analysis import PERIODS

TARGET = "Hours Working"
BOOTSTRAP_N = 2000
KEY_EVENTS = ["amelia", "gym", "walk", "nap", "meditation"]

SUPPLEMENT_ORDER = [
    "caffeine",
    "adderall",
    "modafinil",
    "bronkaid",
    "nicotine",
    "piracetam",
    "choline",
    "l_theanine",
]

OLD_REGIME_FEATURES = ["is_hive", "is_mats", "is_diesl", "is_early_hive"]
TIER_FEATURES = ["is_high", "is_medium"]

SHARED_NON_REGIME = [
    "is_weekend",
    "day_of_week",
    "cal_waste",
    "cal_things",
    "cal_nap",
    "cal_amelia",
    "cal_gym",
    "cal_walk",
    "work_start_hour",
    "streak",
    "has_caffeine",
    "has_adderall",
    "has_modafinil",
    "has_nicotine",
    "has_piracetam",
    "has_choline",
    "has_l_theanine",
    "has_meditation",
]


def build_sample() -> pd.DataFrame:
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
    for col in ["cal_green", "cal_things", "cal_waste", "cal_blue"]:
        df[col] = df[col].fillna(0)

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
            df[col_name] = df[col_name].fillna(0)

    # Old regime dummies
    df["is_hive"] = (df["regime"] == "Hive").astype(int)
    df["is_mats"] = (df["regime"] == "Mats").astype(int)
    df["is_diesl"] = (df["regime"] == "diesl").astype(int)
    df["is_early_hive"] = (
        (df["regime"] == "Hive") & (df["date"] < pd.Timestamp("2022-08-01"))
    ).astype(int)

    # 3-tier pressure dummies
    df["pressure"] = "unassigned"
    for label, start, end, tier in PERIODS:
        mask = (df["date"] >= pd.Timestamp(start)) & (df["date"] < pd.Timestamp(end))
        df.loc[mask, "pressure"] = tier
    df["is_high"] = (df["pressure"] == "high").astype(int)
    df["is_medium"] = (df["pressure"] == "medium").astype(int)

    # Streak (matching coefficient_stability.py exactly)
    df = df.sort_values("date").reset_index(drop=True)
    df["worked"] = df[TARGET] > 2
    streak = 0
    streaks: list[int] = []
    for w in df["worked"]:
        streak = streak + 1 if w else 0
        streaks.append(streak)
    df["streak"] = streaks

    valid = df[df[TARGET] > 0].copy()

    for sup in SUPPLEMENT_ORDER:
        if sup in valid.columns:
            valid[f"has_{sup}"] = (valid[sup] > 0).astype(int)
    if "cal_meditation" in valid.columns:
        valid["has_meditation"] = (valid["cal_meditation"] > 0).astype(int)

    return valid


def fit_model(
    data: pd.DataFrame, features: list[str]
) -> sm.regression.linear_model.RegressionResultsWrapper:
    available = [f for f in features if f in data.columns and data[f].std() > 1e-9]
    sample = data[["date", TARGET] + available].dropna()
    X = sm.add_constant(sample[available], has_constant="add")
    return sm.OLS(sample[TARGET], X).fit(), available, sample


def bootstrap_cis(
    sample: pd.DataFrame, features: list[str], n_boot: int = BOOTSTRAP_N
) -> dict[str, tuple[float, float, float]]:
    X = sm.add_constant(sample[features], has_constant="add").to_numpy()
    y = sample[TARGET].to_numpy()
    rng = np.random.default_rng(42)
    coefs = np.empty((n_boot, X.shape[1]))
    for i in range(n_boot):
        idx = rng.integers(0, len(sample), len(sample))
        coefs[i] = np.linalg.lstsq(X[idx], y[idx], rcond=None)[0]
    names = ["const"] + features
    result = {}
    for j, name in enumerate(names):
        if name == "const":
            continue
        result[name] = (
            float(np.percentile(coefs[:, j], 2.5)),
            float(np.percentile(coefs[:, j], 50)),
            float(np.percentile(coefs[:, j], 97.5)),
        )
    return result


def print_section(title: str) -> None:
    print(f"\n{'='*80}")
    print(f"  {title}")
    print(f"{'='*80}\n")


def main() -> None:
    valid = build_sample()
    print(f"Sample: {len(valid)} days with Hours Working > 0")

    # Build both feature lists
    shared = [f for f in SHARED_NON_REGIME if f in valid.columns and valid[f].std() > 1e-9]
    feats_old = OLD_REGIME_FEATURES + shared
    feats_tier = TIER_FEATURES + shared

    # Fit both models on the SAME sample (intersection of non-missing)
    all_feats = list(set(feats_old + feats_tier))
    common_sample = valid[["date", TARGET] + all_feats].dropna()
    print(f"Common sample (no NaN in any feature): {len(common_sample)} days")

    model_old, used_old, _ = fit_model(common_sample, feats_old)
    model_tier, used_tier, _ = fit_model(common_sample, feats_tier)

    # ── 1. R² comparison ──────────────────────────────────────────────────
    print_section("1. MODEL FIT COMPARISON")
    print(f"  Model A (old dummies):  R² = {model_old.rsquared:.4f}  Adj-R² = {model_old.rsquared_adj:.4f}  params = {int(model_old.df_model)+1}")
    print(f"  Model B (3-tier):       R² = {model_tier.rsquared:.4f}  Adj-R² = {model_tier.rsquared_adj:.4f}  params = {int(model_tier.df_model)+1}")
    print(f"  ΔR²:                    {model_tier.rsquared - model_old.rsquared:+.4f}")
    print(f"  ΔAdj-R²:                {model_tier.rsquared_adj - model_old.rsquared_adj:+.4f}")

    # ── 2. Regime-specific coefficients ───────────────────────────────────
    print_section("2. REGIME COEFFICIENTS")
    print("  Model A (old dummies):")
    for f in OLD_REGIME_FEATURES:
        if f in model_old.params:
            print(f"    {f:<20s}  coef={model_old.params[f]:+.4f}  p={model_old.pvalues[f]:.4f}")

    print("\n  Model B (3-tier):")
    for f in TIER_FEATURES:
        if f in model_tier.params:
            print(f"    {f:<20s}  coef={model_tier.params[f]:+.4f}  p={model_tier.pvalues[f]:.4f}")

    # ── 3. Side-by-side coefficient comparison for shared features ────────
    print_section("3. SHARED FEATURE COEFFICIENTS: OLD vs 3-TIER")

    print(f"  {'Feature':<22s}  {'Old coef':>10s}  {'Old p':>8s}  {'Tier coef':>10s}  {'Tier p':>8s}  {'Δ coef':>10s}  {'% change':>10s}")
    print(f"  {'-'*85}")

    movements = []
    for f in shared:
        if f not in model_old.params or f not in model_tier.params:
            continue
        c_old = model_old.params[f]
        c_tier = model_tier.params[f]
        p_old = model_old.pvalues[f]
        p_tier = model_tier.pvalues[f]
        delta = c_tier - c_old
        pct = (delta / abs(c_old) * 100) if abs(c_old) > 1e-6 else 0
        movements.append((f, c_old, p_old, c_tier, p_tier, delta, pct))
        print(f"  {f:<22s}  {c_old:>+10.4f}  {p_old:>8.4f}  {c_tier:>+10.4f}  {p_tier:>8.4f}  {delta:>+10.4f}  {pct:>+9.1f}%")

    # ── 4. Bootstrap analysis ─────────────────────────────────────────────
    print_section("4. BOOTSTRAP CIs (2000 resamples)")

    sample_old = common_sample[["date", TARGET] + used_old].dropna()
    sample_tier = common_sample[["date", TARGET] + used_tier].dropna()

    print("  Computing bootstrap CIs for old-dummy model...")
    boot_old = bootstrap_cis(sample_old, used_old)
    print("  Computing bootstrap CIs for 3-tier model...")
    boot_tier = bootstrap_cis(sample_tier, used_tier)

    print(f"\n  {'Feature':<22s}  {'Old [2.5%, 97.5%]':>25s}  {'Tier [2.5%, 97.5%]':>25s}  {'Old CI covers Tier?':>20s}  {'Tier CI covers Old?':>20s}")
    print(f"  {'-'*120}")

    for f in shared:
        if f not in boot_old or f not in boot_tier:
            continue
        lo_old, _, hi_old = boot_old[f]
        lo_tier, _, hi_tier = boot_tier[f]
        c_old = model_old.params[f]
        c_tier = model_tier.params[f]
        old_covers_tier = lo_old <= c_tier <= hi_old
        tier_covers_old = lo_tier <= c_old <= hi_tier
        print(f"  {f:<22s}  [{lo_old:>+8.3f}, {hi_old:>+8.3f}]       [{lo_tier:>+8.3f}, {hi_tier:>+8.3f}]       {'YES':>8s}" if old_covers_tier else f"  {f:<22s}  [{lo_old:>+8.3f}, {hi_old:>+8.3f}]       [{lo_tier:>+8.3f}, {hi_tier:>+8.3f}]       {'NO':>8s}", end="")
        print(f"              {'YES':>8s}" if tier_covers_old else f"              {'NO':>8s}")

    # ── 5. Biggest movers ─────────────────────────────────────────────────
    print_section("5. FEATURES WITH LARGEST COEFFICIENT CHANGE")
    movements.sort(key=lambda x: abs(x[5]), reverse=True)
    print(f"  {'Feature':<22s}  {'Old coef':>10s}  {'Tier coef':>10s}  {'Δ coef':>10s}  {'% change':>10s}  {'Within old boot CI?':>20s}")
    print(f"  {'-'*90}")
    for f, c_old, p_old, c_tier, p_tier, delta, pct in movements[:15]:
        if f in boot_old:
            lo, _, hi = boot_old[f]
            within = lo <= c_tier <= hi
            within_str = "YES" if within else "NO ←"
        else:
            within_str = "n/a"
        print(f"  {f:<22s}  {c_old:>+10.4f}  {c_tier:>+10.4f}  {delta:>+10.4f}  {pct:>+9.1f}%  {within_str:>20s}")

    # ── 6. Summary statistics ─────────────────────────────────────────────
    print_section("6. SUMMARY")
    n_shared = len([m for m in movements if m[0] in boot_old])
    n_within = sum(1 for f, c_old, _, c_tier, _, _, _ in movements if f in boot_old and boot_old[f][0] <= c_tier <= boot_old[f][2])
    n_sign_change = sum(1 for _, c_old, _, c_tier, _, _, _ in movements if c_old * c_tier < 0)
    median_abs_pct = np.median([abs(m[6]) for m in movements])
    max_abs_pct = max(abs(m[6]) for m in movements)

    print(f"  Shared features: {n_shared}")
    print(f"  Tier point est within old bootstrap CI: {n_within}/{n_shared}")
    print(f"  Sign changes: {n_sign_change}")
    print(f"  Median |% change|: {median_abs_pct:.1f}%")
    print(f"  Max |% change|: {max_abs_pct:.1f}%")
    print(f"  R² old: {model_old.rsquared:.4f}  →  R² tier: {model_tier.rsquared:.4f}  (Δ={model_tier.rsquared - model_old.rsquared:+.4f})")

    # Interpretation
    print(f"\n  Interpretation:")
    if n_within == n_shared:
        print(f"  All {n_shared} shared coefficients stay within prior bootstrap CIs.")
        print(f"  → Swapping regime coding does NOT meaningfully change the other coefficients.")
    elif n_within >= n_shared * 0.8:
        print(f"  {n_within}/{n_shared} coefficients stay within CIs — mostly stable.")
        outliers = [m[0] for m in movements if m[0] in boot_old and not (boot_old[m[0]][0] <= m[3] <= boot_old[m[0]][2])]
        if outliers:
            print(f"  Moved outside CI: {', '.join(outliers)}")
    else:
        print(f"  Only {n_within}/{n_shared} stay within CIs — substantial coefficient instability.")


if __name__ == "__main__":
    main()
