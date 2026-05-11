# %% use conda env: side_projects
"""Next round of analysis: addresses TODOs from ANALYSIS_SUMMARY.txt."""
import matplotlib

matplotlib.use("Agg")

import re
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.linear_model import LassoCV, RidgeCV
from sklearn.preprocessing import StandardScaler

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

# ============================================================================
# DATA LOADING (same pattern as controlled_analysis.py)
# ============================================================================
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

# Streaks
df = df.sort_values("date").reset_index(drop=True)
df["worked"] = df["Hours Working"] > 2
streak = 0
streaks: list[int] = []
for w in df["worked"]:
    streak = streak + 1 if w else 0
    streaks.append(streak)
df["streak"] = streaks

df["prev_hours"] = df["Hours Working"].shift(1)

# Work sessions per day
work_starts_daily = distracted[distracted["type"] == "s"].groupby("date").size()
work_starts_daily.name = "n_sessions"
df = df.merge(work_starts_daily, left_on="date", right_index=True, how="left")
df["n_sessions"] = df["n_sessions"].fillna(1)

valid = df[df["Hours Working"] > 0].copy()

BASE_CONTROLS = ["is_hive", "is_mats", "is_diesl", "is_weekend", "day_of_week"]


def controlled_effect(
    data: pd.DataFrame,
    treatment_col: str,
    outcome: str = "Hours Working",
    extra_controls: list[str] | None = None,
    binary: bool = False,
) -> dict:
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
        f"  {label:45s}: {result['coef']:+.3f}{unit}  "
        f"(p={result['p']:.3f}, n={result['n']}) {sig}"
    )


print(
    f"Loaded: {len(df)} daily rows, {len(cal_full)} calendar events, "
    f"{len(distracted)} distraction entries"
)
print()


# ============================================================================
# TODO 1: FIX SLEEP DATA
# ============================================================================
def fix_sleep_data(cal_full: pd.DataFrame, df: pd.DataFrame) -> None:
    print("=" * 70)
    print("TODO 1: FIX SLEEP DATA — use graph_sleep_nap logic")
    print("=" * 70)

    # Replicate graph_sleep_nap: sum sleep/nap per day from cal_full
    cf = cal_full.copy()
    df_sleep = cf[cf["event_lower"] == "sleep"].copy()
    df_nap = cf[cf["event_lower"] == "nap"].copy()

    sleep_daily = df_sleep.groupby("date")["duration"].sum()
    nap_daily = df_nap.groupby("date")["duration"].sum()

    # Full daily index
    start_date = min(
        sleep_daily.index.min() if not sleep_daily.empty else cf["date"].min(),
        nap_daily.index.min() if not nap_daily.empty else cf["date"].min(),
    )
    end_date = max(
        sleep_daily.index.max() if not sleep_daily.empty else cf["date"].max(),
        nap_daily.index.max() if not nap_daily.empty else cf["date"].max(),
    )
    full_index = pd.date_range(start=start_date, end=end_date, freq="D")
    sleep_daily = sleep_daily.reindex(full_index, fill_value=0)
    nap_daily = nap_daily.reindex(full_index, fill_value=0)

    sleep_df = pd.DataFrame(
        {"sleep_corrected": sleep_daily, "nap_raw": nap_daily}, index=full_index
    )

    print(
        f"\n  Raw calendar sleep: mean={sleep_df['sleep_corrected'].mean():.2f}h, "
        f"median={sleep_df['sleep_corrected'].median():.2f}h"
    )
    print(
        f"  Raw calendar nap:   mean={sleep_df['nap_raw'].mean():.2f}h, "
        f"median={sleep_df['nap_raw'][sleep_df['nap_raw'] > 0].median():.2f}h"
    )

    # Reclassify early-morning naps as interrupted sleep:
    # If a nap starts within 2h of the sleep event ending on the same date,
    # count it as sleep instead.
    reclassified_count = 0
    sleep_events = df_sleep[["date", "start_time", "end_time", "duration"]].copy()
    nap_events = df_nap[["date", "start_time", "end_time", "duration"]].copy()

    reclassified_nap_hours: dict[pd.Timestamp, float] = defaultdict(float)

    for _, nap_row in nap_events.iterrows():
        nap_date = nap_row["date"]
        nap_start = nap_row["start_time"]
        # Find sleep events ending on same date
        day_sleeps = sleep_events[sleep_events["date"] == nap_date]
        for _, sleep_row in day_sleeps.iterrows():
            sleep_end = sleep_row["end_time"]
            gap_hours = (nap_start - sleep_end).total_seconds() / 3600
            if 0 <= gap_hours <= 2:
                reclassified_nap_hours[nap_date] += nap_row["duration"]
                reclassified_count += 1
                break

    # Apply reclassification
    for date, hours in reclassified_nap_hours.items():
        if date in sleep_df.index:
            sleep_df.loc[date, "sleep_corrected"] += hours
            sleep_df.loc[date, "nap_raw"] -= hours

    sleep_df["nap_corrected"] = sleep_df["nap_raw"].clip(lower=0)
    sleep_df["total_rest"] = sleep_df["sleep_corrected"] + sleep_df["nap_corrected"]

    print(f"\n  Reclassified {reclassified_count} naps as interrupted sleep")
    print(
        f"  Corrected sleep: mean={sleep_df['sleep_corrected'].mean():.2f}h, "
        f"median={sleep_df['sleep_corrected'].median():.2f}h"
    )
    print(f"  Corrected nap:   mean={sleep_df['nap_corrected'].mean():.2f}h")

    # Merge onto df and rerun regressions
    sleep_merge = sleep_df[["sleep_corrected", "nap_corrected"]].reset_index()
    sleep_merge = sleep_merge.rename(columns={"index": "date"})
    df_s = df.merge(sleep_merge, on="date", how="left")
    df_s = df_s[df_s["Hours Working"] > 0].copy()

    print("\n  --- OLS with corrected sleep ---")
    r = controlled_effect(df_s, "sleep_corrected")
    print_effect("Sleep (corrected) → Hours Working", r)
    r = controlled_effect(df_s, "nap_corrected")
    print_effect("Nap (corrected) → Hours Working", r)

    # Compare with uncorrected
    if "cal_sleep" in df_s.columns:
        r_old = controlled_effect(df_s, "cal_sleep")
        print_effect("Sleep (uncorrected cal_sleep) → Hours Working", r_old)
    if "calendar_sleep_hours" in df_s.columns:
        r_old = controlled_effect(df_s, "calendar_sleep_hours")
        print_effect("Sleep (calendar_sleep_hours) → Hours Working", r_old)
    if "cal_nap" in df_s.columns:
        r_old = controlled_effect(df_s, "cal_nap")
        print_effect("Nap (uncorrected cal_nap) → Hours Working", r_old)

    print()


# ============================================================================
# TODO 2: AMELIA SUBCATEGORIZATION
# ============================================================================
def amelia_subcategorization(cal_full: pd.DataFrame, df: pd.DataFrame) -> None:
    print("=" * 70)
    print("TODO 2: AMELIA SUBCATEGORIZATION")
    print("=" * 70)

    amelia = cal_full[
        cal_full["event_lower"].str.contains("amelia", case=False, na=False)
    ].copy()

    if amelia.empty:
        print("  No Amelia events found.")
        return

    # Regex from calendar_analysis.py:2673
    negative_keywords = [
        "argue",
        "fight",
        "rant",
        "tiff",
        "yell",
        "scream",
        "flip",
        "stern",
    ]
    relationship_keywords = [
        "talk",
        "talked",
        "talking",
        "talks",
        "notes",
        "journal",
        "blogs",
        "made up",
        "silence",
        "silience",
        "discussion",
        "write",
        "think",
        "seethe",
        "ruminate",
        "type",
        "writing",
    ]
    negative_pattern = (
        r"(?<!\w)(" + "|".join(re.escape(k) for k in negative_keywords) + r")(?!\w)"
    )
    relationship_pattern = (
        r"(?<!\w)(" + "|".join(re.escape(k) for k in relationship_keywords) + r")(?!\w)"
    )

    neg_mask = amelia["event_name"].str.contains(
        negative_pattern, case=False, regex=True, na=False
    )
    rel_mask = (
        amelia["event_name"].str.contains(
            relationship_pattern, case=False, regex=True, na=False
        )
        & ~neg_mask
    )

    amelia["subtype"] = "quality_time"
    amelia.loc[neg_mask, "subtype"] = "negative"
    amelia.loc[rel_mask, "subtype"] = "relationship_work"

    print(f"\n  Total Amelia events: {len(amelia)}")
    for st in ["negative", "relationship_work", "quality_time"]:
        sub = amelia[amelia["subtype"] == st]
        print(f"    {st:25s}: {len(sub):4d} events, {sub['duration'].sum():.1f}h total")

    # Daily hours per subtype
    sub_daily = (
        amelia.groupby(["date", "subtype"])["duration"].sum().unstack(fill_value=0)
    )
    for st in ["negative", "relationship_work", "quality_time"]:
        if st not in sub_daily.columns:
            sub_daily[st] = 0.0
    sub_daily = sub_daily.rename(
        columns={
            "negative": "amelia_negative",
            "relationship_work": "amelia_relwork",
            "quality_time": "amelia_quality",
        }
    )

    df_a = df.merge(sub_daily, left_on="date", right_index=True, how="left")
    df_a[["amelia_negative", "amelia_relwork", "amelia_quality"]] = df_a[
        ["amelia_negative", "amelia_relwork", "amelia_quality"]
    ].fillna(0)
    df_a = df_a[df_a["Hours Working"] > 0].copy()

    print("\n  --- Controlled effects by Amelia subtype ---")
    print(f"  {'Subtype → Outcome':50s} {'Coef':>8s} {'p':>8s} {'n':>6s}")
    for subtype_col in ["amelia_negative", "amelia_relwork", "amelia_quality"]:
        for outcome in ["Hours Working", "Energy", "Focus"]:
            r = controlled_effect(df_a, subtype_col, outcome=outcome)
            if not np.isnan(r["coef"]):
                unit = "h" if outcome == "Hours Working" else "pts"
                sig = (
                    "***"
                    if r["p"] < 0.001
                    else "**" if r["p"] < 0.01 else "*" if r["p"] < 0.05 else ""
                )
                print(
                    f"  {subtype_col + ' → ' + outcome:50s} "
                    f"{r['coef']:+8.3f}{unit} {r['p']:8.3f} n={r['n']:4d} {sig}"
                )
    print()


# ============================================================================
# TODO 3: TOP WEEKS ANALYSIS
# ============================================================================
def top_weeks_analysis(df: pd.DataFrame) -> None:
    print("=" * 70)
    print("TODO 3: WHAT MAKES 80-95th PERCENTILE WEEKS?")
    print("=" * 70)

    # Load weekly summary CSV
    weekly_csv = pd.read_csv(
        "data/Work Summary  - Weekly Summary_Projects.csv",
        header=None,
        skiprows=1,
    )
    # Columns: For(0), Start(1), End(2), Days Worked(3), Value(4), Hours(5), ...
    weekly = pd.DataFrame(
        {
            "regime": weekly_csv.iloc[:, 0],
            "start": pd.to_datetime(
                weekly_csv.iloc[:, 1], format="mixed", errors="coerce"
            ),
            "end": pd.to_datetime(
                weekly_csv.iloc[:, 2], format="mixed", errors="coerce"
            ),
            "days_worked": pd.to_numeric(weekly_csv.iloc[:, 3], errors="coerce"),
            "hours": pd.to_numeric(weekly_csv.iloc[:, 5], errors="coerce"),
            "summary": weekly_csv.iloc[:, 11],
            "lessons": weekly_csv.iloc[:, 12],
        }
    )
    weekly = weekly.dropna(subset=["start", "hours"])
    weekly = weekly[weekly["hours"] > 0].reset_index(drop=True)

    p80 = weekly["hours"].quantile(0.80)
    p95 = weekly["hours"].quantile(0.95)
    print(
        f"\n  Weekly hours: mean={weekly['hours'].mean():.1f}, "
        f"median={weekly['hours'].median():.1f}"
    )
    print(f"  80th percentile: {p80:.1f}h, 95th percentile: {p95:.1f}h")
    print(
        f"  Weeks >= 80th: {(weekly['hours'] >= p80).sum()}, "
        f"Weeks >= 95th: {(weekly['hours'] >= p95).sum()}"
    )

    # Aggregate daily df into weeks aligned to Monday starts from weekly CSV
    df_w = df.copy()
    # Assign each day to its week start (Monday)
    df_w["week_start"] = df_w["date"] - pd.to_timedelta(
        df_w["date"].dt.dayofweek, unit="D"
    )

    week_agg = (
        df_w.groupby("week_start")
        .agg(
            hours_working=("Hours Working", "sum"),
            mean_waste=("cal_waste", "mean"),
            mean_things=("cal_things", "mean"),
            mean_nap=(
                ("cal_nap", "mean")
                if "cal_nap" in df_w.columns
                else ("Hours Working", "count")
            ),
            mean_work_start=("work_start_hour", "mean"),
            max_streak=("streak", "max"),
            mean_energy=("Energy", "mean"),
            mean_focus=("Focus", "mean"),
            n_days=("Hours Working", "count"),
        )
        .reset_index()
    )

    # Add supplement usage rates per week
    supp_cols = [
        c
        for c in df_w.columns
        if c in ["caffeine", "adderall", "modafinil", "nicotine", "piracetam"]
    ]
    for sc in supp_cols:
        week_supp = (
            df_w.groupby("week_start")[sc]
            .apply(lambda x: (x > 0).mean())
            .reset_index(name=f"pct_{sc}")
        )
        week_agg = week_agg.merge(week_supp, on="week_start", how="left")

    # Filter to weeks with >= 4 days of data
    week_agg = week_agg[week_agg["n_days"] >= 4].copy()

    # Mark top weeks
    week_agg["is_top80"] = (week_agg["hours_working"] >= p80).astype(int)
    week_agg["is_top95"] = (week_agg["hours_working"] >= p95).astype(int)

    print(f"\n  Weeks with >= 4 days data: {len(week_agg)}")
    top = week_agg[week_agg["is_top80"] == 1]
    bottom = week_agg[week_agg["is_top80"] == 0]

    print("\n  --- Profile: Top 80th pctl weeks vs rest ---")
    profile_cols = [
        "mean_waste",
        "mean_things",
        "mean_work_start",
        "max_streak",
        "mean_energy",
        "mean_focus",
    ]
    if "cal_nap" in df_w.columns:
        profile_cols.insert(2, "mean_nap")

    for col in profile_cols:
        if col in week_agg.columns:
            t_mean = top[col].mean()
            b_mean = bottom[col].mean()
            print(
                f"    {col:25s}: top={t_mean:.2f}, rest={b_mean:.2f}, "
                f"diff={t_mean - b_mean:+.2f}"
            )

    # Logistic regression: P(top80) ~ features
    feature_cols = [c for c in profile_cols if c in week_agg.columns]
    for sc in supp_cols:
        if f"pct_{sc}" in week_agg.columns:
            feature_cols.append(f"pct_{sc}")

    sub = week_agg[["is_top80"] + feature_cols].dropna()
    if len(sub) > 30:
        X = sm.add_constant(sub[feature_cols])
        y = sub["is_top80"]
        try:
            logit = sm.Logit(y, X).fit(disp=0)
            print("\n  --- Logistic regression: P(week >= 80th pctl) ---")
            print(f"  Pseudo-R² = {logit.prsquared:.3f}, n = {len(sub)}")
            for feat in feature_cols:
                sig = (
                    "***"
                    if logit.pvalues[feat] < 0.001
                    else (
                        "**"
                        if logit.pvalues[feat] < 0.01
                        else "*" if logit.pvalues[feat] < 0.05 else ""
                    )
                )
                print(
                    f"    {feat:25s}: coef={logit.params[feat]:+.3f}, "
                    f"p={logit.pvalues[feat]:.3f} {sig}"
                )
        except Exception as e:
            print(f"  Logistic regression failed: {e}")

    # Prior-week analysis
    week_agg = week_agg.sort_values("week_start").reset_index(drop=True)
    for col in feature_cols:
        week_agg[f"prev_{col}"] = week_agg[col].shift(1)
    week_agg["prev_hours"] = week_agg["hours_working"].shift(1)

    prev_features = [f"prev_{c}" for c in feature_cols] + ["prev_hours"]
    sub2 = week_agg[["is_top80"] + prev_features].dropna()
    if len(sub2) > 30:
        X = sm.add_constant(sub2[prev_features])
        y = sub2["is_top80"]
        try:
            logit2 = sm.Logit(y, X).fit(disp=0)
            print("\n  --- Prior-week predictors of top week ---")
            print(f"  Pseudo-R² = {logit2.prsquared:.3f}, n = {len(sub2)}")
            for feat in prev_features:
                sig = (
                    "***"
                    if logit2.pvalues[feat] < 0.001
                    else (
                        "**"
                        if logit2.pvalues[feat] < 0.01
                        else "*" if logit2.pvalues[feat] < 0.05 else ""
                    )
                )
                print(
                    f"    {feat:30s}: coef={logit2.params[feat]:+.3f}, "
                    f"p={logit2.pvalues[feat]:.3f} {sig}"
                )
        except Exception as e:
            print(f"  Prior-week logistic regression failed: {e}")

    print()


# ============================================================================
# TODO 4: REGULARIZED MODELS (Ridge/Lasso)
# ============================================================================
def regularized_models(df: pd.DataFrame) -> None:
    print("=" * 70)
    print("TODO 4: L1/L2 REGULARIZATION — Ridge & Lasso")
    print("=" * 70)

    v = df[df["Hours Working"] > 0].copy()

    # Same feature set as controlled_analysis.py section 12
    features = [
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

    supp_cols = [
        c
        for c in v.columns
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
        ]
    ]
    for supp in supp_cols:
        if (v[supp] > 0).sum() > 30:
            v[f"has_{supp}"] = (v[supp] > 0).astype(int)
            features.append(f"has_{supp}")

    for col in ["cal_gym", "cal_walk"]:
        if col in v.columns:
            features.append(col)
    if "has_meditation" not in v.columns and "cal_meditation" in v.columns:
        v["has_meditation"] = (v["cal_meditation"] > 0).astype(int)
    if "has_meditation" in v.columns:
        features.append("has_meditation")

    keep_features = [f for f in features if f in v.columns and v[f].notna().sum() > 100]
    sub = v[["Hours Working"] + keep_features].dropna()
    X_raw = sub[keep_features].values
    y = sub["Hours Working"].values

    # OLS baseline
    X_ols = sm.add_constant(sub[keep_features])
    ols_model = sm.OLS(y, X_ols).fit()
    ols_coefs = ols_model.params.drop("const")

    # StandardScaler for regularized models
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_raw)

    alphas = np.logspace(-3, 3, 50)

    ridge = RidgeCV(alphas=alphas, cv=5)
    ridge.fit(X_scaled, y)
    ridge_coefs_scaled = pd.Series(ridge.coef_, index=keep_features)
    # Unscale coefficients for comparison
    ridge_coefs = ridge_coefs_scaled / scaler.scale_

    lasso = LassoCV(alphas=alphas, cv=5, max_iter=10000)
    lasso.fit(X_scaled, y)
    lasso_coefs_scaled = pd.Series(lasso.coef_, index=keep_features)
    lasso_coefs = lasso_coefs_scaled / scaler.scale_

    print(f"\n  n = {len(sub)}")
    print(f"  OLS R² = {ols_model.rsquared:.3f}")
    print(f"  Ridge alpha = {ridge.alpha_:.3f}, CV R² = {ridge.best_score_:.3f}")
    print(f"  Lasso alpha = {lasso.alpha_:.3f}, CV R² = {lasso.score(X_scaled, y):.3f}")

    print(
        f"\n  {'Feature':30s} {'OLS':>8s} {'Ridge':>8s} {'Lasso':>8s} {'Survives?':>10s}"
    )
    print("  " + "-" * 70)
    for feat in keep_features:
        survives = "YES" if abs(lasso_coefs[feat]) > 0.001 else ""
        print(
            f"  {feat:30s} {ols_coefs[feat]:+8.3f} {ridge_coefs[feat]:+8.3f} "
            f"{lasso_coefs[feat]:+8.3f} {survives:>10s}"
        )

    n_survived = (lasso_coefs.abs() > 0.001).sum()
    print(f"\n  Lasso retains {n_survived}/{len(keep_features)} features")
    print()


# ============================================================================
# TODO 5: CALENDAR vs CSV DISCREPANCY
# ============================================================================
def calendar_csv_discrepancy(df: pd.DataFrame) -> None:
    print("=" * 70)
    print("TODO 5: CALENDAR vs CSV DISCREPANCY")
    print("=" * 70)

    sub = df[["date", "cal_blue", "Hours Working"]].dropna().copy()
    sub["discrepancy"] = sub["cal_blue"] - sub["Hours Working"]

    print("\n  Discrepancy (cal_blue - Hours Working):")
    print(f"    mean = {sub['discrepancy'].mean():.2f}h")
    print(f"    median = {sub['discrepancy'].median():.2f}h")
    print(f"    std = {sub['discrepancy'].std():.2f}h")
    print(
        f"    min = {sub['discrepancy'].min():.2f}h, max = {sub['discrepancy'].max():.2f}h"
    )

    # Histogram
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.hist(sub["discrepancy"], bins=50, edgecolor="black", alpha=0.7)
    ax.axvline(0, color="red", linestyle="--", label="Zero")
    ax.axvline(sub["discrepancy"].mean(), color="blue", linestyle="--", label="Mean")
    ax.set_xlabel("Calendar Blue - CSV Hours Working (hours)")
    ax.set_ylabel("Count")
    ax.set_title("Calendar vs CSV Work Hours Discrepancy")
    ax.legend()
    plt.tight_layout()
    plt.savefig("Self_Tracking/calendar_csv_discrepancy.png", dpi=150)
    plt.close()
    print("  Saved: Self_Tracking/calendar_csv_discrepancy.png")

    # Flag days with large discrepancy
    large = sub[sub["discrepancy"].abs() > 3].sort_values(
        "discrepancy", key=abs, ascending=False
    )
    print(f"\n  Days with |discrepancy| > 3h: {len(large)}")
    print("  Top 10:")
    for _, row in large.head(10).iterrows():
        print(
            f"    {row['date'].date()}: cal_blue={row['cal_blue']:.1f}h, "
            f"CSV={row['Hours Working']:.1f}h, diff={row['discrepancy']:+.1f}h"
        )

    # What predicts discrepancy?
    disc_df = df.copy()
    disc_df["discrepancy"] = disc_df["cal_blue"] - disc_df["Hours Working"]
    disc_df = disc_df[disc_df["Hours Working"] > 0].copy()

    predictors = ["is_weekend", "day_of_week", "is_hive", "is_mats", "is_diesl"]
    if "n_sessions" in disc_df.columns:
        predictors.append("n_sessions")
    if "work_start_hour" in disc_df.columns:
        predictors.append("work_start_hour")

    sub2 = disc_df[["discrepancy"] + predictors].dropna()
    if len(sub2) > 30:
        X = sm.add_constant(sub2[predictors])
        model = sm.OLS(sub2["discrepancy"], X).fit()
        print(f"\n  --- What predicts discrepancy? (R²={model.rsquared:.3f}) ---")
        for feat in predictors:
            sig = (
                "***"
                if model.pvalues[feat] < 0.001
                else (
                    "**"
                    if model.pvalues[feat] < 0.01
                    else "*" if model.pvalues[feat] < 0.05 else ""
                )
            )
            print(
                f"    {feat:25s}: {model.params[feat]:+.3f} (p={model.pvalues[feat]:.3f}) {sig}"
            )
    print()


# ============================================================================
# TODO 6: MISSING CSV DAYS
# ============================================================================
def missing_csv_days(df: pd.DataFrame, distracted: pd.DataFrame) -> None:
    print("=" * 70)
    print("TODO 6: MISSING CSV DAYS & STREAK BUG")
    print("=" * 70)

    full_range = pd.date_range(start=df["date"].min(), end=df["date"].max(), freq="D")
    existing_dates = set(df["date"].dt.normalize())
    missing_dates = sorted(set(full_range) - existing_dates)

    print(f"\n  Date range: {df['date'].min().date()} to {df['date'].max().date()}")
    print(f"  Total days in range: {len(full_range)}")
    print(f"  Days with data: {len(existing_dates)}")
    print(f"  Missing days: {len(missing_dates)}")

    if not missing_dates:
        print("  No missing days found.")
        return

    # Gap analysis
    missing_s = pd.Series(missing_dates)
    gaps: list[tuple[pd.Timestamp, int]] = []
    gap_start = missing_s.iloc[0]
    gap_len = 1
    for i in range(1, len(missing_s)):
        if (missing_s.iloc[i] - missing_s.iloc[i - 1]).days == 1:
            gap_len += 1
        else:
            gaps.append((gap_start, gap_len))
            gap_start = missing_s.iloc[i]
            gap_len = 1
    gaps.append((gap_start, gap_len))

    gap_lengths = [g[1] for g in gaps]
    print(f"\n  Number of gaps: {len(gaps)}")
    print(
        f"  Gap lengths: min={min(gap_lengths)}, max={max(gap_lengths)}, "
        f"mean={np.mean(gap_lengths):.1f}, median={np.median(gap_lengths):.1f}"
    )

    # Weekday vs weekend breakdown
    missing_dow = pd.Series(missing_dates).dt.dayofweek
    n_weekend_missing = (missing_dow >= 5).sum()
    n_weekday_missing = (missing_dow < 5).sum()
    print(
        f"  Missing weekdays: {n_weekday_missing}, missing weekends: {n_weekend_missing}"
    )

    # Show longest gaps
    gaps_sorted = sorted(gaps, key=lambda x: x[1], reverse=True)[:10]
    print("\n  Longest gaps:")
    for start, length in gaps_sorted:
        end = start + pd.Timedelta(days=length - 1)
        print(f"    {start.date()} to {end.date()} ({length} days)")

    # Streak bug analysis
    # Current code: iterates sorted df rows, streak = streak+1 if worked else 0
    # Missing days are skipped, so streak continues across gaps
    print("\n  --- Streak Bug Analysis ---")
    print("  Current behavior: missing days are skipped, streak continues across gaps.")

    # Count how many streaks span a gap
    df_sorted = df.sort_values("date").copy()
    streaks_spanning_gap = 0
    for i in range(1, len(df_sorted)):
        prev_date = df_sorted.iloc[i - 1]["date"]
        curr_date = df_sorted.iloc[i]["date"]
        day_gap = (curr_date - prev_date).days
        if day_gap > 1 and df_sorted.iloc[i]["streak"] > 1:
            streaks_spanning_gap += 1

    print(f"  Streaks that span a gap (bug instances): {streaks_spanning_gap}")
    print("  Proposed fix: reset streak to 0 when gap > 1 day between rows")

    # Show impact: what would max streaks look like with fix?
    df_fixed = df_sorted.copy()
    streak = 0
    fixed_streaks: list[int] = []
    prev_date = None
    for _, row in df_fixed.iterrows():
        if prev_date is not None and (row["date"] - prev_date).days > 1:
            streak = 0
        streak = streak + 1 if row["Hours Working"] > 2 else 0
        fixed_streaks.append(streak)
        prev_date = row["date"]
    df_fixed["streak_fixed"] = fixed_streaks

    print(f"\n  Current max streak: {df_sorted['streak'].max()}")
    print(f"  Fixed max streak:   {df_fixed['streak_fixed'].max()}")
    print(
        f"  Current mean streak (when >0): "
        f"{df_sorted[df_sorted['streak'] > 0]['streak'].mean():.1f}"
    )
    print(
        f"  Fixed mean streak (when >0):   "
        f"{df_fixed[df_fixed['streak_fixed'] > 0]['streak_fixed'].mean():.1f}"
    )
    print()


# ============================================================================
# TODO 7: DISTRACTION TIMING
# ============================================================================
def distraction_timing(distracted: pd.DataFrame) -> None:
    print("=" * 70)
    print("TODO 7: DISTRACTION TIMING — when in a session do distractions occur?")
    print("=" * 70)

    # Parse sessions: 's' starts, 'e' ends
    dist = distracted.copy()
    dist["time_parsed"] = pd.to_datetime(dist["time"], format="mixed", errors="coerce")

    # Build sessions per day
    sessions: list[dict] = []
    for date, day_data in dist.groupby("date"):
        day_sorted = day_data.sort_values("time_parsed")
        starts = day_sorted[day_sorted["type"] == "s"]
        ends = day_sorted[day_sorted["type"] == "e"]

        for i, (_, start_row) in enumerate(starts.iterrows()):
            session_start = start_row["time_parsed"]
            # Find matching end (next 'e' after this 's')
            matching_ends = ends[ends["time_parsed"] > session_start]
            if matching_ends.empty:
                continue
            session_end = matching_ends.iloc[0]["time_parsed"]

            # Get distractions in this session
            session_distractions = day_sorted[
                (day_sorted["type"].isin(["d", "u"]))
                & (day_sorted["time_parsed"] > session_start)
                & (day_sorted["time_parsed"] < session_end)
            ]

            session_duration_h = (session_end - session_start).total_seconds() / 3600

            if session_duration_h <= 0 or session_duration_h > 16:
                continue

            for _, d_row in session_distractions.iterrows():
                hours_into = (
                    d_row["time_parsed"] - session_start
                ).total_seconds() / 3600
                sessions.append(
                    {
                        "date": date,
                        "session_start": session_start,
                        "session_duration_h": session_duration_h,
                        "hours_into_session": hours_into,
                        "distraction_type": d_row["type"],
                    }
                )

    if not sessions:
        print("  No sessions with distractions found.")
        return

    sess_df = pd.DataFrame(sessions)
    print(f"\n  Total distraction-in-session records: {len(sess_df)}")
    print(f"  Mean hours into session: {sess_df['hours_into_session'].mean():.2f}h")

    # Bin by time into session
    max_hours = min(sess_df["hours_into_session"].max(), 8)

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    for ax, bin_size, label in [
        (axes[0], 10 / 60, "10-min bins"),
        (axes[1], 20 / 60, "20-min bins"),
        (axes[2], 1, "1-hour bins"),
    ]:
        bins = np.arange(0, max_hours + bin_size, bin_size)
        counts, _ = np.histogram(sess_df["hours_into_session"], bins=bins)

        bin_centers = (bins[:-1] + bins[1:]) / 2
        ax.bar(bin_centers, counts, width=bin_size * 0.9, alpha=0.6, color="steelblue")

        # Smoothed line (rolling average of bin counts)
        if len(counts) > 5:
            window = min(5, len(counts) // 2)
            smoothed = pd.Series(counts).rolling(window, center=True).mean()
            ax.plot(bin_centers, smoothed, color="red", linewidth=2, label="Smoothed")

        ax.set_xlabel("Hours into session")
        ax.set_ylabel("Number of distractions")
        ax.set_title(label)
        ax.legend()

    plt.suptitle("Distraction Timing Within Work Sessions", fontsize=14)
    plt.tight_layout()
    plt.savefig("Self_Tracking/distraction_timing.png", dpi=150)
    plt.close()
    print("  Saved: Self_Tracking/distraction_timing.png")

    # Also print rate by hour
    print("\n  Distraction count by hour into session:")
    hourly = sess_df["hours_into_session"].apply(lambda x: int(x))
    hourly_counts = hourly.value_counts().sort_index()
    for hour, count in hourly_counts.items():
        print(f"    Hour {hour}-{hour+1}: {count} distractions")
    print()


# ============================================================================
# TODO 8: DISTRACTION x CALENDAR EVENTS
# ============================================================================
def distraction_x_calendar(distracted: pd.DataFrame, cal_full: pd.DataFrame) -> None:
    print("=" * 70)
    print("TODO 8: DISTRACTION × CALENDAR EVENTS")
    print("=" * 70)

    # Build full datetime for each distraction
    dist = distracted[distracted["type"].isin(["d", "u"])].copy()
    dist["time_parsed"] = pd.to_datetime(dist["time"], format="mixed", errors="coerce")
    dist = dist.dropna(subset=["time_parsed"])

    # Construct full datetime from date + time
    dist["datetime"] = dist.apply(
        lambda row: pd.Timestamp(
            year=row["date"].year,
            month=row["date"].month,
            day=row["date"].day,
            hour=row["time_parsed"].hour,
            minute=row["time_parsed"].minute,
            second=row["time_parsed"].second,
        ),
        axis=1,
    )
    # Localize to Central then convert to UTC for matching with calendar
    import pytz

    ct = pytz.timezone("US/Central")
    dist["datetime_utc"] = dist["datetime"].apply(
        lambda x: ct.localize(x).astimezone(pytz.UTC)
    )

    # For each distraction, find concurrent calendar events
    cal = cal_full[
        ["event_lower", "start_time", "end_time", "category", "duration"]
    ].copy()

    matches: list[dict] = []
    # Group by date for efficiency
    cal["date_local"] = cal["start_time"].dt.tz_convert(ct).dt.date
    dist["date_key"] = dist["date"].dt.date

    for date_key, day_dist in dist.groupby("date_key"):
        day_cal = cal[cal["date_local"] == date_key]
        if day_cal.empty:
            continue
        for _, d_row in day_dist.iterrows():
            dt = d_row["datetime_utc"]
            concurrent = day_cal[
                (day_cal["start_time"] <= dt) & (day_cal["end_time"] > dt)
            ]
            for _, c_row in concurrent.iterrows():
                matches.append(
                    {
                        "event": c_row["event_lower"],
                        "category": c_row["category"],
                        "event_duration": c_row["duration"],
                        "distraction_type": d_row["type"],
                    }
                )

    if not matches:
        print("  No distraction-calendar matches found.")
        return

    match_df = pd.DataFrame(matches)
    print(f"\n  Total distraction-event matches: {len(match_df)}")

    # Group by category
    print("\n  --- Distractions by calendar category ---")
    cat_stats = (
        match_df.groupby("category")
        .agg(
            n_distractions=("event", "count"),
        )
        .reset_index()
    )

    # Total hours per category (from cal_full)
    cat_hours = cal_full.groupby("category")["duration"].sum()
    cat_stats = cat_stats.merge(
        cat_hours.reset_index().rename(columns={"duration": "total_hours"}),
        on="category",
        how="left",
    )
    cat_stats["distractions_per_hour"] = (
        cat_stats["n_distractions"] / cat_stats["total_hours"]
    )
    cat_stats = cat_stats.sort_values("distractions_per_hour", ascending=False)

    print(
        f"  {'Category':15s} {'Distractions':>13s} {'Total Hours':>12s} {'Rate/hr':>8s}"
    )
    for _, row in cat_stats.iterrows():
        print(
            f"  {row['category']:15s} {row['n_distractions']:13d} "
            f"{row['total_hours']:12.1f} {row['distractions_per_hour']:8.2f}"
        )

    # Group by event name (top 20 by count)
    print("\n  --- Top 20 events by distraction count ---")
    ev_stats = (
        match_df.groupby("event")
        .agg(
            n_distractions=("category", "count"),
        )
        .reset_index()
    )

    ev_hours = cal_full.groupby("event_lower")["duration"].sum()
    ev_stats = ev_stats.merge(
        ev_hours.reset_index().rename(
            columns={"event_lower": "event", "duration": "total_hours"}
        ),
        on="event",
        how="left",
    )
    ev_stats["distractions_per_hour"] = (
        ev_stats["n_distractions"] / ev_stats["total_hours"]
    )
    ev_stats = ev_stats.sort_values("n_distractions", ascending=False)

    print(f"  {'Event':30s} {'Count':>6s} {'Hours':>8s} {'Rate/hr':>8s}")
    for _, row in ev_stats.head(20).iterrows():
        print(
            f"  {str(row['event'])[:30]:30s} {row['n_distractions']:6d} "
            f"{row['total_hours']:8.1f} {row['distractions_per_hour']:8.2f}"
        )

    # Top 20 by rate (min 5 distractions, min 2 hours)
    ev_filtered = ev_stats[
        (ev_stats["n_distractions"] >= 5) & (ev_stats["total_hours"] >= 2)
    ].sort_values("distractions_per_hour", ascending=False)

    if not ev_filtered.empty:
        print(
            "\n  --- Top 20 events by distraction RATE (min 5 distractions, min 2h) ---"
        )
        print(f"  {'Event':30s} {'Count':>6s} {'Hours':>8s} {'Rate/hr':>8s}")
        for _, row in ev_filtered.head(20).iterrows():
            print(
                f"  {str(row['event'])[:30]:30s} {row['n_distractions']:6d} "
                f"{row['total_hours']:8.1f} {row['distractions_per_hour']:8.2f}"
            )
    print()


# ============================================================================
# RUN ALL
# ============================================================================
if __name__ == "__main__":
    fix_sleep_data(cal_full, df)
    amelia_subcategorization(cal_full, df)
    top_weeks_analysis(df)
    regularized_models(df)
    calendar_csv_discrepancy(df)
    missing_csv_days(df, distracted)
    distraction_timing(distracted)
    distraction_x_calendar(distracted, cal_full)

    print("=" * 70)
    print("ALL DONE")
    print("=" * 70)
