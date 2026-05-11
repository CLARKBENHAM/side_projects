# %% use conda env: side_projects
"""Controlled analysis: redo all key findings with regime + confounders controlled."""
import matplotlib

matplotlib.use("Agg")

import numpy as np
import pandas as pd
import statsmodels.api as sm

from productivity_analysis import (
    load_calendar_full,
    load_daily_summary_full,
    load_distracted_stacked,
    extract_daily_supplements,
    extract_work_start_time,
    load_calendar_sleep,
    CALENDAR_DIR,
    DAILY_SUMMARY_CSV,
    DISTRACTED_CSV,
)

print("Loading data...")
daily = load_daily_summary_full(DAILY_SUMMARY_CSV)
distracted = load_distracted_stacked(DISTRACTED_CSV)
supplements = extract_daily_supplements(distracted)
work_starts = extract_work_start_time(distracted)
cal_sleep = load_calendar_sleep(CALENDAR_DIR)
cal_full = load_calendar_full(CALENDAR_DIR)

# Build merged df
df = daily.merge(supplements, on="date", how="left")
df = df.merge(cal_sleep, on="date", how="left")
df = df.merge(work_starts, on="date", how="left")
df["day_of_week"] = df["date"].dt.dayofweek
df["is_weekend"] = (df["day_of_week"] >= 5).astype(int)

# Calendar category daily totals
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
df[["cal_green", "cal_things", "cal_waste", "cal_blue"]] = df[
    ["cal_green", "cal_things", "cal_waste", "cal_blue"]
].fillna(0)

# Key event daily totals
event_daily = (
    cal_full.groupby(["date", "event_lower"])["duration"].sum().unstack(fill_value=0)
)
key_events = [
    "amelia",
    "twitter",
    "porn",
    "blogs",
    "sleep",
    "nap",
    "gym",
    "walk",
    "family",
    "friends",
    "youtube",
    "insta",
    "jack",
    "anki",
    "meditation",
    "cook",
    "drink",
    "amelia argue",
]
for ev in key_events:
    col = f"cal_{ev.replace(' ', '_')}"
    if ev in event_daily.columns:
        df = df.merge(
            event_daily[[ev]].rename(columns={ev: col}),
            left_on="date",
            right_index=True,
            how="left",
        )
        df[col] = df[col].fillna(0)

# Regime dummies
df["is_hive"] = (df["regime"] == "Hive").astype(int)
df["is_mats"] = (df["regime"] == "Mats").astype(int)
df["is_diesl"] = (df["regime"] == "diesl").astype(int)
df["is_early_hive"] = (
    (df["regime"] == "Hive") & (df["date"] < pd.Timestamp("2022-08-01"))
).astype(int)
# Personal is the reference category

# Streaks
df = df.sort_values("date").reset_index(drop=True)
df["worked"] = df["Hours Working"] > 2
streak = 0
streaks = []
for w in df["worked"]:
    streak = streak + 1 if w else 0
    streaks.append(streak)
df["streak"] = streaks

# Previous day hours
df["prev_hours"] = df["Hours Working"].shift(1)

# Work sessions per day
work_starts_daily = distracted[distracted["type"] == "s"].groupby("date").size()
work_starts_daily.name = "n_sessions"
df = df.merge(work_starts_daily, left_on="date", right_index=True, how="left")
df["n_sessions"] = df["n_sessions"].fillna(1)

valid = df[df["Hours Working"] > 0].copy()

# Standard controls for all regressions
BASE_CONTROLS = [
    "is_hive",
    "is_mats",
    "is_diesl",
    "is_early_hive",
    "is_weekend",
    "day_of_week",
]


def controlled_effect(
    data: pd.DataFrame,
    treatment_col: str,
    outcome: str = "Hours Working",
    extra_controls: list[str] | None = None,
    binary: bool = False,
) -> dict:
    """Estimate effect of treatment on outcome, controlling for regime + weekend + DOW.
    Returns dict with coefficient, p-value, n, and full model for inspection."""
    controls = BASE_CONTROLS.copy()
    if extra_controls:
        controls.extend(extra_controls)

    cols = [outcome, treatment_col] + controls
    sub = data[cols].dropna()
    if len(sub) < 30:
        return {"coef": np.nan, "p": np.nan, "n": len(sub)}

    X = sm.add_constant(sub[[treatment_col] + controls])
    y = sub[outcome]
    model = sm.OLS(y, X).fit()
    return {
        "coef": model.params[treatment_col],
        "p": model.pvalues[treatment_col],
        "n": len(sub),
        "se": model.bse[treatment_col],
        "r2": model.rsquared,
        "model": model,
    }


def print_effect(label: str, result: dict, unit: str = "h") -> None:
    sig = (
        "***"
        if result["p"] < 0.001
        else "**" if result["p"] < 0.01 else "*" if result["p"] < 0.05 else ""
    )
    print(
        f"  {label:40s}: {result['coef']:+.3f}{unit}  (p={result['p']:.3f}, n={result['n']}) {sig}"
    )


# ============================================================================
print("=" * 70)
print("ALL EFFECTS CONTROLLED FOR: regime (Hive/Mats/diesl), weekend, DOW")
print("=" * 70)

# ============================================================================
# 1. NAPS
# ============================================================================
print("\n--- 1. NAPS ---")
valid["has_nap"] = (valid["cal_nap"] > 0).astype(int)
r = controlled_effect(valid, "has_nap")
print_effect("Nap (any) → Hours Working", r)
r = controlled_effect(valid, "cal_nap")
print_effect("Nap duration (hours) → Hours Working", r)

# Nap buckets via controlled means
for lo, hi, label in [
    (0.01, 0.5, "Nap <30min"),
    (0.5, 1.0, "Nap 30-60min"),
    (1.0, 2.0, "Nap 1-2h"),
    (2.0, 10, "Nap >2h"),
]:
    valid[f"nap_{label}"] = ((valid["cal_nap"] >= lo) & (valid["cal_nap"] < hi)).astype(
        int
    )
    r = controlled_effect(valid, f"nap_{label}")
    print_effect(f"  {label} → Hours Working", r)

# Nap → next day
valid["nap_yesterday"] = valid["has_nap"].shift(1)
r = controlled_effect(valid.dropna(subset=["nap_yesterday"]), "nap_yesterday")
print_effect("Nap yesterday → today Hours Working", r)

# ============================================================================
# 2. AMELIA
# ============================================================================
print("\n--- 2. AMELIA TIME ---")
r = controlled_effect(valid, "cal_amelia")
print_effect("Amelia hours → Hours Working", r)

# Amelia buckets
for lo, hi, label in [
    (0, 0.5, "<30min"),
    (0.5, 2, "30min-2h"),
    (2, 4, "2-4h"),
    (4, 24, ">4h"),
]:
    valid[f"amelia_{label}"] = (
        (valid["cal_amelia"] >= lo) & (valid["cal_amelia"] < hi)
    ).astype(int)
    r = controlled_effect(valid, f"amelia_{label}")
    print_effect(f"  Amelia {label} (vs other) → Hours", r)

# Arguments
if "cal_amelia_argue" in valid.columns:
    valid["had_argue"] = (valid["cal_amelia_argue"] > 0).astype(int)
    r = controlled_effect(valid, "had_argue")
    print_effect("Argue day → Hours Working", r)
    valid["argued_yesterday"] = valid["had_argue"].shift(1)
    r = controlled_effect(valid.dropna(subset=["argued_yesterday"]), "argued_yesterday")
    print_effect("Argued yesterday → Hours Working", r)

# ============================================================================
# 3. EXERCISE
# ============================================================================
print("\n--- 3. EXERCISE ---")
valid["has_gym"] = (valid["cal_gym"] > 0).astype(int)
r = controlled_effect(valid, "has_gym")
print_effect("Gym (any) → Hours Working", r)
r = controlled_effect(valid, "cal_gym")
print_effect("Gym duration → Hours Working", r)

for lo, hi, label in [
    (0.01, 0.5, "<30min"),
    (0.5, 1.0, "30-60min"),
    (1.0, 1.5, "1-1.5h"),
    (1.5, 4, ">1.5h"),
]:
    valid[f"gym_{label}"] = ((valid["cal_gym"] >= lo) & (valid["cal_gym"] < hi)).astype(
        int
    )
    r = controlled_effect(valid, f"gym_{label}")
    print_effect(f"  Gym {label} → Hours Working", r)

# Gym → Energy
r = controlled_effect(valid, "has_gym", outcome="Energy")
print_effect("Gym (any) → Energy", r, unit="pts")

valid["has_walk"] = (valid["cal_walk"] > 0).astype(int)
r = controlled_effect(valid, "has_walk")
print_effect("Walk (any) → Hours Working", r)
r = controlled_effect(valid, "has_walk", outcome="Energy")
print_effect("Walk (any) → Energy", r, unit="pts")

# ============================================================================
# 4. WASTE CATEGORIES
# ============================================================================
print("\n--- 4. WASTE CATEGORIES ---")
r = controlled_effect(valid, "cal_waste")
print_effect("Total waste hours → Hours Working", r)

for ev, col in [
    ("twitter", "cal_twitter"),
    ("blogs", "cal_blogs"),
    ("porn", "cal_porn"),
    ("youtube", "cal_youtube"),
    ("jack", "cal_jack"),
    ("insta", "cal_insta"),
]:
    if col in valid.columns:
        r = controlled_effect(valid, col)
        print_effect(f"  {ev} hours → Hours Working", r)

# ============================================================================
# 5. DRINKING
# ============================================================================
print("\n--- 5. DRINKING ---")
if "cal_drink" in valid.columns:
    valid["drank"] = (valid["cal_drink"] > 0).astype(int)
    r = controlled_effect(valid, "drank")
    print_effect("Drank today → Hours Working", r)
    valid["drank_yesterday"] = valid["drank"].shift(1)
    r = controlled_effect(valid.dropna(subset=["drank_yesterday"]), "drank_yesterday")
    print_effect("Drank yesterday → Hours Working", r)
    r = controlled_effect(
        valid.dropna(subset=["drank_yesterday"]), "drank_yesterday", outcome="Energy"
    )
    print_effect("Drank yesterday → Energy", r, unit="pts")

# ============================================================================
# 6. MEDITATION
# ============================================================================
print("\n--- 6. MEDITATION ---")
if "cal_meditation" in valid.columns:
    valid["has_meditation"] = (valid["cal_meditation"] > 0).astype(int)
    r = controlled_effect(valid, "has_meditation")
    print_effect("Meditation → Hours Working", r)
    r = controlled_effect(valid, "has_meditation", outcome="Energy")
    print_effect("Meditation → Energy", r, unit="pts")
    r = controlled_effect(valid, "has_meditation", outcome="Focus")
    print_effect("Meditation → Focus", r, unit="pts")

# ============================================================================
# 7. STREAKS
# ============================================================================
print("\n--- 7. WORK STREAKS ---")
r = controlled_effect(valid, "streak")
print_effect("Streak length → Hours Working", r)

# Streak buckets
for lo, hi, label in [
    (1, 1, "Day 1"),
    (2, 5, "Days 2-5"),
    (6, 10, "Days 6-10"),
    (11, 20, "Days 11-20"),
    (21, 200, "Days 21+"),
]:
    valid[f"streak_{label}"] = (
        (valid["streak"] >= lo) & (valid["streak"] <= hi)
    ).astype(int)
    r = controlled_effect(valid, f"streak_{label}")
    print_effect(f"  {label} → Hours Working", r)

# ============================================================================
# 8. WORK START TIME
# ============================================================================
print("\n--- 8. WORK START TIME ---")
r = controlled_effect(valid, "work_start_hour")
print_effect("Work start hour → Hours Working", r)

# ============================================================================
# 9. SESSIONS PER DAY
# ============================================================================
print("\n--- 9. WORK SESSIONS ---")
valid["multi_session"] = (valid["n_sessions"] >= 2).astype(int)
r = controlled_effect(valid, "multi_session")
print_effect("2+ sessions → Hours Working", r)

# ============================================================================
# 10. PORN/JACK NEXT-DAY CONTROLLED
# ============================================================================
print("\n--- 10. PORN/JACK NEXT-DAY ---")
for ev, col in [("porn", "cal_porn"), ("jack", "cal_jack")]:
    if col in valid.columns:
        valid[f"had_{ev}"] = (valid[col] > 0).astype(int)
        r = controlled_effect(valid, f"had_{ev}")
        print_effect(f"  {ev} same-day → Hours Working", r)
        valid[f"{ev}_yesterday"] = valid[f"had_{ev}"].shift(1)
        r = controlled_effect(
            valid.dropna(subset=[f"{ev}_yesterday"]), f"{ev}_yesterday"
        )
        print_effect(f"  {ev} yesterday → Hours Working", r)
        r = controlled_effect(
            valid.dropna(subset=[f"{ev}_yesterday"]),
            f"{ev}_yesterday",
            outcome="Energy",
        )
        print_effect(f"  {ev} yesterday → Energy", r, unit="pts")

# ============================================================================
# 11. SUPPLEMENT DOSE-RESPONSE CURVES
# ============================================================================
print("\n" + "=" * 70)
print("11. SUPPLEMENT DOSE-RESPONSE (controlled for regime + weekend + DOW)")
print("=" * 70)

# List all supplement columns
supp_cols = [
    c
    for c in valid.columns
    if c
    in [
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
]

for supp in supp_cols:
    col_data = valid[supp]
    n_nonzero = (col_data > 0).sum()
    if n_nonzero < 15:
        continue

    print(f"\n  === {supp.upper()} (n={n_nonzero} days with usage) ===")

    # Presence/absence
    valid[f"has_{supp}"] = (col_data > 0).astype(int)
    r = controlled_effect(valid, f"has_{supp}")
    print_effect(f"    Any {supp} → Hours Working", r)
    r = controlled_effect(valid, f"has_{supp}", outcome="Energy")
    print_effect(f"    Any {supp} → Energy", r, unit="pts")
    r = controlled_effect(valid, f"has_{supp}", outcome="Focus")
    print_effect(f"    Any {supp} → Focus", r, unit="pts")

    # Continuous dose (only if there's variance in dose)
    doses = col_data[col_data > 0]
    if doses.std() > 0.01 and len(doses.unique()) > 3:
        r = controlled_effect(valid[valid[supp] > 0], supp)
        print_effect(f"    {supp} dose (among users) → Hours", r)

        # Dose buckets
        percentiles = doses.quantile([0.33, 0.67])
        low_thresh = percentiles.iloc[0]
        high_thresh = percentiles.iloc[1]

        low_dose = valid[(col_data > 0) & (col_data <= low_thresh)]
        mid_dose = valid[(col_data > low_thresh) & (col_data <= high_thresh)]
        high_dose = valid[col_data > high_thresh]
        none_dose = valid[col_data == 0]

        if len(low_dose) > 5 and len(high_dose) > 5:
            print(
                f"    Dose buckets (low ≤{low_thresh:.0f}, mid ≤{high_thresh:.0f}, high >{high_thresh:.0f}):"
            )
            for label, subset in [
                ("none", none_dose),
                ("low", low_dose),
                ("mid", mid_dose),
                ("high", high_dose),
            ]:
                if len(subset) > 5:
                    print(
                        f"      {label:6s}: {subset['Hours Working'].mean():.2f}h, "
                        f"Energy={subset['Energy'].mean():.1f}, "
                        f"Focus={subset['Focus'].mean():.1f} (n={len(subset)})"
                    )

    # Interaction with regime: does the effect differ by regime?
    for regime_name, regime_col in [("Hive", "is_hive"), ("diesl", "is_diesl")]:
        regime_subset = valid[valid[regime_col] == 1]
        n_with = (regime_subset[supp] > 0).sum()
        if n_with > 10 and (regime_subset[supp] == 0).sum() > 10:
            r = controlled_effect(
                regime_subset,
                f"has_{supp}",
                extra_controls=[c for c in BASE_CONTROLS if c != regime_col],
            )
            if not np.isnan(r["coef"]):
                print_effect(f"    {supp} → Hours ({regime_name} only)", r)


# ============================================================================
# 12. FULL MULTIVARIATE MODEL
# ============================================================================
print("\n" + "=" * 70)
print("12. FULL MULTIVARIATE MODEL: What matters most?")
print("=" * 70)

features = [
    "is_hive",
    "is_mats",
    "is_diesl",
    "is_early_hive",
    "is_weekend",
    "day_of_week",
    "cal_waste",
    "cal_things",
    "cal_nap",
    "cal_amelia",
    "work_start_hour",
    "streak",
]

# Add supplements that have enough data
for supp in supp_cols:
    if (valid[supp] > 0).sum() > 30:
        valid[f"has_{supp}"] = (valid[supp] > 0).astype(int)
        features.append(f"has_{supp}")

if "cal_gym" in valid.columns:
    features.append("cal_gym")
if "cal_walk" in valid.columns:
    features.append("cal_walk")
if "has_meditation" in valid.columns:
    features.append("has_meditation")

# Only keep features with enough non-null data
keep_features = []
for f in features:
    if f in valid.columns and valid[f].notna().sum() > 100:
        keep_features.append(f)

sub = valid[["Hours Working"] + keep_features].dropna()
X = sm.add_constant(sub[keep_features])
y = sub["Hours Working"]
model = sm.OLS(y, X).fit()
print(
    f"\n  R² = {model.rsquared:.3f}, Adj R² = {model.rsquared_adj:.3f}, n = {len(sub)}"
)

# Sort by absolute t-stat
coefs = model.params.drop("const")
pvals = model.pvalues.drop("const")
tstats = model.tvalues.drop("const")
sorted_idx = tstats.abs().sort_values(ascending=False).index

print(f"\n  {'Feature':35s} {'Coef':>8s} {'t-stat':>8s} {'p':>8s}")
for feat in sorted_idx:
    sig = (
        "***"
        if pvals[feat] < 0.001
        else "**" if pvals[feat] < 0.01 else "*" if pvals[feat] < 0.05 else ""
    )
    print(
        f"  {feat:35s} {coefs[feat]:+8.3f} {tstats[feat]:8.2f} {pvals[feat]:8.3f} {sig}"
    )

# Standardized coefficients
print("\n  --- Standardized (effect size comparison) ---")
X_std = (sub[keep_features] - sub[keep_features].mean()) / sub[keep_features].std()
X_std = sm.add_constant(X_std)
model_std = sm.OLS(y, X_std).fit()
std_coefs = model_std.params.drop("const")
std_pvals = model_std.pvalues.drop("const")
sorted_std = std_coefs.abs().sort_values(ascending=False).index
print(f"  {'Feature':35s} {'Std Coef':>10s} {'p':>8s}")
for feat in sorted_std:
    sig = (
        "***"
        if std_pvals[feat] < 0.001
        else "**" if std_pvals[feat] < 0.01 else "*" if std_pvals[feat] < 0.05 else ""
    )
    print(f"  {feat:35s} {std_coefs[feat]:+10.3f} {std_pvals[feat]:8.3f} {sig}")

# ============================================================================
# 13. SAME ANALYSES FOR ENERGY AND FOCUS
# ============================================================================
for outcome in ["Energy", "Focus"]:
    print(f"\n{'='*70}")
    print(f"13. KEY EFFECTS ON {outcome.upper()} (controlled)")
    print(f"{'='*70}")

    for label, col in [
        ("Nap (any)", "has_nap"),
        ("Gym (any)", "has_gym"),
        ("Walk (any)", "has_walk"),
        ("Amelia hours", "cal_amelia"),
        ("Waste hours", "cal_waste"),
        ("Streak", "streak"),
        ("Meditation", "has_meditation"),
        ("Work start hour", "work_start_hour"),
    ]:
        if col in valid.columns:
            r = controlled_effect(valid, col, outcome=outcome)
            if not np.isnan(r["coef"]):
                print_effect(f"  {label} → {outcome}", r, unit="pts")

    # Key supplements
    for supp in ["caffeine", "adderall", "modafinil"]:
        col = f"has_{supp}"
        if col in valid.columns:
            r = controlled_effect(valid, col, outcome=outcome)
            if not np.isnan(r["coef"]):
                print_effect(f"  {supp} → {outcome}", r, unit="pts")
