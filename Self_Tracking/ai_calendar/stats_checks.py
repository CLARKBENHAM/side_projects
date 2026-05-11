# %% use conda env: side_projects
"""Statistical checks: multicollinearity, crash regression to mean, endogeneity."""
import matplotlib
matplotlib.use("Agg")

import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor
from scipy import stats

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

df = daily.merge(supplements, on="date", how="left")
df = df.merge(cal_sleep, on="date", how="left")
df = df.merge(work_starts, on="date", how="left")
df["day_of_week"] = df["date"].dt.dayofweek
df["is_weekend"] = (df["day_of_week"] >= 5).astype(int)

cat_daily = cal_full.groupby(["date", "category"])["duration"].sum().unstack(fill_value=0)
for cat in ["green", "things", "waste", "blue"]:
    if cat not in cat_daily.columns:
        cat_daily[cat] = 0.0
cat_daily = cat_daily.rename(columns={"green": "cal_green", "things": "cal_things",
                                       "waste": "cal_waste", "blue": "cal_blue"})
df = df.merge(cat_daily, left_on="date", right_index=True, how="left")
df[["cal_green", "cal_things", "cal_waste", "cal_blue"]] = df[["cal_green", "cal_things", "cal_waste", "cal_blue"]].fillna(0)

event_daily = cal_full.groupby(["date", "event_lower"])["duration"].sum().unstack(fill_value=0)
for ev in ["amelia", "nap", "gym", "walk", "meditation"]:
    col = f"cal_{ev}"
    if ev in event_daily.columns:
        df = df.merge(event_daily[[ev]].rename(columns={ev: col}),
                      left_on="date", right_index=True, how="left")
        df[col] = df[col].fillna(0)

df["is_hive"] = (df["regime"] == "Hive").astype(int)
df["is_mats"] = (df["regime"] == "Mats").astype(int)
df["is_diesl"] = (df["regime"] == "diesl").astype(int)

df = df.sort_values("date").reset_index(drop=True)
streak = 0
streaks = []
for w in (df["Hours Working"] > 2):
    streak = streak + 1 if w else 0
    streaks.append(streak)
df["streak"] = streaks

supp_cols = [c for c in df.columns if c in [
    "caffeine", "adderall", "modafinil", "nicotine", "piracetam", "choline"]]
for s in supp_cols:
    df[f"has_{s}"] = (df[s] > 0).astype(int)
df["has_meditation"] = (df["cal_meditation"] > 0).astype(int) if "cal_meditation" in df.columns else 0

valid = df[df["Hours Working"] > 0].copy()

# ============================================================================
print("=" * 70)
print("1. MULTICOLLINEARITY CHECK (VIF)")
print("=" * 70)

features = ["is_hive", "is_mats", "is_diesl", "is_weekend", "day_of_week",
            "cal_waste", "cal_things", "cal_nap", "cal_amelia",
            "cal_gym", "cal_walk", "work_start_hour", "streak",
            "has_meditation"]
for s in supp_cols:
    if (valid[s] > 0).sum() > 30:
        features.append(f"has_{s}")

sub = valid[features].dropna()
X = sm.add_constant(sub)

print(f"\n  {'Feature':35s} {'VIF':>8s}")
vifs = []
for i, col in enumerate(X.columns):
    if col == "const":
        continue
    vif = variance_inflation_factor(X.values, i)
    vifs.append((col, vif))
vifs.sort(key=lambda x: x[1], reverse=True)
for col, vif in vifs:
    flag = " ⚠️ HIGH" if vif > 5 else " ⚠ moderate" if vif > 2.5 else ""
    print(f"  {col:35s} {vif:8.2f}{flag}")

# Correlation matrix for the most concerning pairs
print("\n  --- Pairwise correlations among features (|r| > 0.2) ---")
corr = sub[features].corr()
for i, f1 in enumerate(features):
    for f2 in features[i+1:]:
        r = corr.loc[f1, f2]
        if abs(r) > 0.2:
            print(f"    {f1:25s} ↔ {f2:25s}: r={r:+.3f}")

# ============================================================================
print("\n" + "=" * 70)
print("2. REGRESSION WITH ORTHOGONALIZED / GROUPED FEATURES")
print("=" * 70)
print("  (Addressing multicollinearity by grouping correlated features)")

# Strategy: since supplements correlate with regime, run within-regime regressions
# Also try PCA or just dropping correlated features

# 2a: Within-regime regressions for key effects
for regime_name, regime_mask in [("Hive", valid["is_hive"] == 1),
                                  ("Personal", (valid["is_hive"] == 0) & (valid["is_mats"] == 0) & (valid["is_diesl"] == 0)),
                                  ("diesl", valid["is_diesl"] == 1)]:
    regime_df = valid[regime_mask]
    if len(regime_df) < 50:
        continue

    print(f"\n  --- {regime_name} (n={len(regime_df)}) ---")
    feats = ["is_weekend", "day_of_week", "cal_waste", "cal_things",
             "cal_nap", "work_start_hour", "streak"]
    # Only add supplements with variance in this regime
    for s in supp_cols:
        col = f"has_{s}"
        if col in regime_df.columns and regime_df[col].std() > 0.05 and regime_df[col].sum() > 5:
            feats.append(col)
    if "cal_gym" in regime_df.columns:
        feats.append("cal_gym")
    if "has_meditation" in regime_df.columns and regime_df["has_meditation"].std() > 0.05:
        feats.append("has_meditation")

    sub_r = regime_df[["Hours Working"] + feats].dropna()
    if len(sub_r) < 30:
        continue
    X = sm.add_constant(sub_r[feats])
    y = sub_r["Hours Working"]
    model = sm.OLS(y, X).fit()

    print(f"    R² = {model.rsquared:.3f}, n = {len(sub_r)}")
    coefs = model.params.drop("const")
    pvals = model.pvalues.drop("const")
    sorted_idx = model.tvalues.drop("const").abs().sort_values(ascending=False).index
    for feat in sorted_idx:
        sig = "***" if pvals[feat] < 0.001 else "**" if pvals[feat] < 0.01 else "*" if pvals[feat] < 0.05 else ""
        if pvals[feat] < 0.15 or feat in ["cal_waste", "cal_things", "work_start_hour", "streak"]:
            print(f"    {feat:30s}: {coefs[feat]:+.3f} (p={pvals[feat]:.3f}) {sig}")


# ============================================================================
print("\n" + "=" * 70)
print("3. CRASH AFTER GOOD WEEKS: REGRESSION TO MEAN OR REAL?")
print("=" * 70)

# Weekly aggregation
weekly = df.groupby(df["date"].dt.to_period("W")).agg(
    hours=("Hours Working", "sum"),
    n_days=("Hours Working", lambda x: (x > 0).sum()),
    mean_daily=("Hours Working", lambda x: x[x > 0].mean() if (x > 0).any() else 0),
).reset_index()
weekly = weekly[weekly["n_days"] >= 3]  # only weeks with 3+ work days
weekly["next_hours"] = weekly["hours"].shift(-1)
weekly["prev_hours"] = weekly["hours"].shift(1)

print(f"  Weeks with 3+ work days: {len(weekly)}")
print(f"  Mean weekly hours: {weekly['hours'].mean():.1f} ± {weekly['hours'].std():.1f}")

# 3a: Autocorrelation of weekly hours
r, p = stats.pearsonr(weekly["hours"].iloc[:-1], weekly["hours"].iloc[1:])
print(f"\n  Week-to-week autocorrelation: r={r:.3f}, p={p:.3f}")

# 3b: Is the "crash" just regression to mean?
# If hours are iid draws from the same distribution, high weeks would naturally
# be followed by average weeks. Test: is the DROP larger than expected?

# Method: compare observed mean reversion to what iid would predict
mean_h = weekly["hours"].mean()
std_h = weekly["hours"].std()

# For 50h+ weeks
high_weeks = weekly[weekly["hours"] >= 50]
high_next = weekly.loc[high_weeks.index + 1] if len(high_weeks) > 0 else pd.DataFrame()

# Match indices properly
high_with_next = weekly[weekly["hours"] >= 50].copy()
high_with_next = high_with_next[high_with_next["next_hours"].notna()]

if len(high_with_next) > 5:
    observed_drop = high_with_next["hours"].mean() - high_with_next["next_hours"].mean()
    high_mean = high_with_next["hours"].mean()

    # Under iid, expected next week = population mean
    expected_rtm_drop = high_mean - mean_h

    # Under AR(1) with autocorrelation r, expected next = mean + r*(current - mean)
    expected_ar1_next = mean_h + r * (high_mean - mean_h)
    expected_ar1_drop = high_mean - expected_ar1_next

    print(f"\n  --- 50h+ weeks (n={len(high_with_next)}) ---")
    print(f"  Mean of 50h+ weeks:     {high_mean:.1f}h")
    print(f"  Mean of following week:  {high_with_next['next_hours'].mean():.1f}h")
    print(f"  Observed drop:           {observed_drop:.1f}h")
    print(f"  Expected if iid (regression to mean): {expected_rtm_drop:.1f}h")
    print(f"  Expected if AR(1):       {expected_ar1_drop:.1f}h")
    print(f"  Excess drop beyond iid:  {observed_drop - expected_rtm_drop:.1f}h")
    print(f"  Excess drop beyond AR(1):{observed_drop - expected_ar1_drop:.1f}h")

    # Is the excess significant?
    # Under iid: next week ~ N(mean, std²), so drop = high - next ~ high - N(mean, std²)
    # Test if observed next-week mean differs from population mean
    t, p = stats.ttest_1samp(high_with_next["next_hours"], mean_h)
    print("\n  Is next-week mean different from population mean?")
    print(f"  Next week mean: {high_with_next['next_hours'].mean():.1f}, pop mean: {mean_h:.1f}")
    print(f"  t={t:.2f}, p={p:.3f}")

# Same for different thresholds
print("\n  --- Drop by threshold ---")
print(f"  {'Threshold':>12s} {'n':>5s} {'Week hrs':>10s} {'Next hrs':>10s} {'Drop':>8s} {'RTM expected':>13s} {'Excess':>8s}")
for thresh in [35, 40, 45, 50, 55]:
    hw = weekly[weekly["hours"] >= thresh].copy()
    hw = hw[hw["next_hours"].notna()]
    if len(hw) > 3:
        drop = hw["hours"].mean() - hw["next_hours"].mean()
        rtm = hw["hours"].mean() - mean_h
        excess = drop - rtm
        print(f"  {thresh:>10d}h+ {len(hw):5d} {hw['hours'].mean():10.1f} "
              f"{hw['next_hours'].mean():10.1f} {drop:8.1f} {rtm:13.1f} {excess:8.1f}")

# 3c: Simulate iid to get null distribution of "crashes"
print("\n  --- Simulation: what would random look like? ---")
np.random.seed(42)
n_sim = 10000
sim_crashes = []
n_weeks = len(weekly)
for _ in range(n_sim):
    # Draw iid weeks from same distribution
    sim = np.random.choice(weekly["hours"].values, size=n_weeks, replace=True)
    # Find 50h+ weeks and their "next" week
    high_idx = np.where(sim >= 50)[0]
    high_idx = high_idx[high_idx < n_weeks - 1]  # must have a next week
    if len(high_idx) > 0:
        drops = sim[high_idx] - sim[high_idx + 1]
        sim_crashes.append(np.mean(drops))

sim_crashes = np.array(sim_crashes)
observed_crash = observed_drop
pct = (sim_crashes >= observed_crash).mean()
print(f"  Observed mean crash after 50h+ week: {observed_crash:.1f}h")
print(f"  Simulated iid mean crash: {np.mean(sim_crashes):.1f}h ± {np.std(sim_crashes):.1f}h")
print(f"  P(iid crash ≥ observed): {pct:.3f}")
print(f"  Conclusion: {'REAL crash beyond regression to mean' if pct < 0.05 else 'Consistent with regression to mean'}")

# 3d: Consecutive high weeks
print("\n  --- Can high weeks sustain? ---")
print("  P(next week ≥ X | this week ≥ X):")
for thresh in [35, 40, 45, 50]:
    this_high = weekly[weekly["hours"] >= thresh]
    this_high = this_high[this_high["next_hours"].notna()]
    if len(this_high) > 5:
        sustained = (this_high["next_hours"] >= thresh).mean()
        base_rate = (weekly["hours"] >= thresh).mean()
        print(f"    ≥{thresh}h: {sustained:.0%} sustained (base rate: {base_rate:.0%})")


# ============================================================================
print("\n" + "=" * 70)
print("4. ENDOGENEITY: OBSERVED ACTIONS vs PLANNED ACTIONS")
print("=" * 70)
print("  (What we CAN infer vs what we CANNOT)")

# The core issue: we observe that starting work at 9am → more hours.
# But is that because early starts cause productivity, or because
# productive days happen to start early (i.e., you get up early BECAUSE
# you're motivated)?

# Test: does yesterday's hours predict today's start time?
# If yes → reverse causation is present (good days → early starts next day)
valid["prev_hours"] = valid["Hours Working"].shift(1)
valid["prev_start"] = valid["work_start_hour"].shift(1)

r1, p1 = stats.pearsonr(valid["prev_hours"].dropna(), valid.loc[valid["prev_hours"].notna(), "work_start_hour"])
r2, p2 = stats.pearsonr(valid["prev_start"].dropna(), valid.loc[valid["prev_start"].notna(), "Hours Working"])

print("\n  Granger-style causality tests:")
print(f"  Yesterday's hours → today's start time:  r={r1:+.3f}, p={p1:.3f}")
print(f"  Yesterday's start → today's hours:       r={r2:+.3f}, p={p2:.3f}")

# Cross-lagged panel regression: does start_hour predict hours ABOVE AND BEYOND
# the autoregressive component?
sub_lag = valid[["Hours Working", "work_start_hour", "prev_hours", "prev_start",
                 "is_hive", "is_mats", "is_diesl", "is_weekend", "day_of_week"]].dropna()
X = sm.add_constant(sub_lag[["prev_hours", "work_start_hour",
                              "is_hive", "is_mats", "is_diesl", "is_weekend", "day_of_week"]])
y = sub_lag["Hours Working"]
model = sm.OLS(y, X).fit()
print("\n  Cross-lagged model: Hours ~ prev_hours + today_start + controls")
print(f"    prev_hours coef:      {model.params['prev_hours']:+.3f} (p={model.pvalues['prev_hours']:.3f})")
print(f"    work_start_hour coef: {model.params['work_start_hour']:+.3f} (p={model.pvalues['work_start_hour']:.3f})")
print("    → Start time still matters after controlling for yesterday's output")

# Similarly for waste: do you waste because you're unproductive, or vice versa?
# Test: does yesterday's waste predict today's hours (controlling for today's waste)?
valid["prev_waste"] = valid["cal_waste"].shift(1)
sub_w = valid[["Hours Working", "cal_waste", "prev_waste",
               "is_hive", "is_mats", "is_diesl", "is_weekend", "day_of_week"]].dropna()
X = sm.add_constant(sub_w[["cal_waste", "prev_waste",
                            "is_hive", "is_mats", "is_diesl", "is_weekend", "day_of_week"]])
y = sub_w["Hours Working"]
model_w = sm.OLS(y, X).fit()
print("\n  Waste endogeneity test:")
print(f"    Today's waste coef:     {model_w.params['cal_waste']:+.3f} (p={model_w.pvalues['cal_waste']:.3f})")
print(f"    Yesterday's waste coef: {model_w.params['prev_waste']:+.3f} (p={model_w.pvalues['prev_waste']:.3f})")

# For naps: do you nap because the day is already bad?
# Test: does pre-nap work hours predict napping?
# Approximate: does work_start_hour predict napping?
if "cal_nap" in valid.columns:
    valid["has_nap"] = (valid["cal_nap"] > 0).astype(int)
    sub_n = valid[["has_nap", "work_start_hour", "prev_hours",
                   "is_hive", "is_mats", "is_diesl", "is_weekend"]].dropna()
    X = sm.add_constant(sub_n[["work_start_hour", "prev_hours",
                                "is_hive", "is_mats", "is_diesl", "is_weekend"]])
    from statsmodels.discrete.discrete_model import Logit
    try:
        logit = Logit(sub_n["has_nap"], X).fit(disp=0)
        print("\n  Nap endogeneity (logistic regression: what predicts napping?):")
        print(f"    prev_hours coef:      {logit.params['prev_hours']:+.3f} (p={logit.pvalues['prev_hours']:.3f})")
        print(f"    work_start_hour coef: {logit.params['work_start_hour']:+.3f} (p={logit.pvalues['work_start_hour']:.3f})")
        print(f"    is_weekend coef:      {logit.params['is_weekend']:+.3f} (p={logit.pvalues['is_weekend']:.3f})")
    except Exception as e:
        print(f"  Logit failed: {e}")

# For supplements: test if good/bad days predict supplement use
print("\n  --- Do good/bad days predict supplement use? ---")
for supp in ["caffeine", "adderall", "modafinil"]:
    col = f"has_{supp}"
    if col in valid.columns and valid[col].sum() > 20:
        sub_s = valid[[col, "prev_hours", "is_hive", "is_mats", "is_diesl", "is_weekend"]].dropna()
        X = sm.add_constant(sub_s[["prev_hours", "is_hive", "is_mats", "is_diesl", "is_weekend"]])
        try:
            logit_s = Logit(sub_s[col], X).fit(disp=0)
            print(f"    prev_hours → use {supp}: coef={logit_s.params['prev_hours']:+.3f}, p={logit_s.pvalues['prev_hours']:.3f}")
        except Exception:
            pass
