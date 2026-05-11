# %% use conda env: side_projects
"""Deep exploratory analysis: what non-obvious patterns exist in the data?"""
import matplotlib
matplotlib.use("Agg")

import pandas as pd
from scipy import stats
from datetime import timedelta

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

# Build merged daily df
df = daily.merge(supplements, on="date", how="left")
df = df.merge(cal_sleep, on="date", how="left")
df = df.merge(work_starts, on="date", how="left")
df["day_of_week"] = df["date"].dt.dayofweek
df["is_weekend"] = (df["day_of_week"] >= 5).astype(int)
df["month"] = df["date"].dt.month

# Add calendar category daily totals
cat_daily = cal_full.groupby(["date", "category"])["duration"].sum().unstack(fill_value=0)
for cat in ["green", "things", "waste", "blue"]:
    if cat not in cat_daily.columns:
        cat_daily[cat] = 0.0
cat_daily = cat_daily.rename(columns={"green": "cal_green", "things": "cal_things",
                                       "waste": "cal_waste", "blue": "cal_blue"})
df = df.merge(cat_daily, left_on="date", right_index=True, how="left")
df[["cal_green", "cal_things", "cal_waste", "cal_blue"]] = df[["cal_green", "cal_things", "cal_waste", "cal_blue"]].fillna(0)

# Add specific event daily totals
event_daily = cal_full.groupby(["date", "event_lower"])["duration"].sum().unstack(fill_value=0)

# Key events to track
key_events = ["amelia", "twitter", "porn", "blogs", "sleep", "nap", "gym",
              "walk", "family", "friends", "youtube", "insta", "jack",
              "anki", "meditation", "cook", "drink"]
for ev in key_events:
    col = f"cal_{ev}"
    if ev in event_daily.columns:
        df = df.merge(event_daily[[ev]].rename(columns={ev: col}),
                      left_on="date", right_index=True, how="left")
        df[col] = df[col].fillna(0)

# Add lag features
df = df.sort_values("date").reset_index(drop=True)
df["prev_hours"] = df["Hours Working"].shift(1)
df["prev_waste"] = df["cal_waste"].shift(1)
df["prev_things"] = df["cal_things"].shift(1)
df["next_hours"] = df["Hours Working"].shift(-1)

# Rolling features
df["rolling_7d_hours"] = df["Hours Working"].rolling(7, min_periods=3).mean()
df["rolling_7d_waste"] = df["cal_waste"].rolling(7, min_periods=3).mean()

print(f"Analysis df: {len(df)} rows, {df['date'].min().date()} to {df['date'].max().date()}")
print()

# ============================================================================
# 1. ACTIVITY CORRELATIONS WITH WORK OUTPUT
# ============================================================================
print("=" * 70)
print("1. WHAT ACTIVITIES PREDICT TOMORROW'S WORK OUTPUT?")
print("=" * 70)

# For each calendar event, does doing it today predict more/less work tomorrow?
valid = df[df["Hours Working"] > 0].copy()

print("\n--- Same-day correlations with Hours Working ---")
corr_cols = [c for c in df.columns if c.startswith("cal_") and df[c].sum() > 0]
results = []
for col in corr_cols:
    mask = valid[col].notna() & valid["Hours Working"].notna()
    if mask.sum() > 30:
        r, p = stats.pearsonr(valid.loc[mask, col], valid.loc[mask, "Hours Working"])
        results.append((col, r, p, mask.sum()))
results.sort(key=lambda x: abs(x[1]), reverse=True)
for col, r, p, n in results[:25]:
    sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else ""
    print(f"  {col:30s}: r={r:+.3f} p={p:.3f} n={n:4d} {sig}")

print("\n--- Does yesterday's activity predict today's work? (lag correlations) ---")
lag_results = []
for col in corr_cols:
    lagged = df[col].shift(1)
    mask = lagged.notna() & df["Hours Working"].notna() & (df["Hours Working"] > 0)
    if mask.sum() > 30:
        r, p = stats.pearsonr(lagged[mask], df.loc[mask, "Hours Working"])
        lag_results.append((col, r, p, mask.sum()))
lag_results.sort(key=lambda x: abs(x[1]), reverse=True)
for col, r, p, n in lag_results[:20]:
    sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else ""
    print(f"  {col:30s}: r={r:+.3f} p={p:.3f} n={n:4d} {sig}")

# ============================================================================
# 2. NAP ANALYSIS
# ============================================================================
print("\n" + "=" * 70)
print("2. NAP ANALYSIS: DO NAPS HELP OR HURT?")
print("=" * 70)

nap_col = "cal_nap" if "cal_nap" in df.columns else None
if nap_col:
    nap_days = valid[valid[nap_col] > 0]
    no_nap_days = valid[valid[nap_col] == 0]
    print(f"  Days with nap: {len(nap_days)}, without: {len(no_nap_days)}")
    print(f"  Hours Working with nap:    {nap_days['Hours Working'].mean():.2f} ± {nap_days['Hours Working'].std():.2f}")
    print(f"  Hours Working without nap: {no_nap_days['Hours Working'].mean():.2f} ± {no_nap_days['Hours Working'].std():.2f}")
    t, p = stats.ttest_ind(nap_days["Hours Working"], no_nap_days["Hours Working"])
    print(f"  t={t:.2f}, p={p:.3f}")

    # Nap length buckets
    print("\n  By nap duration:")
    for lo, hi, label in [(0.01, 0.5, "< 30min"), (0.5, 1.0, "30-60min"), (1.0, 2.0, "1-2h"), (2.0, 10, "> 2h")]:
        subset = valid[(valid[nap_col] >= lo) & (valid[nap_col] < hi)]
        if len(subset) > 5:
            print(f"    {label:12s}: {subset['Hours Working'].mean():.2f}h work (n={len(subset)})")

    # Does napping predict NEXT day's work?
    nap_lag = df.copy()
    nap_lag["nap_yesterday"] = nap_lag[nap_col].shift(1) > 0
    nap_lag = nap_lag[nap_lag["Hours Working"] > 0]
    nap_yes = nap_lag[nap_lag["nap_yesterday"] == True]
    nap_no = nap_lag[nap_lag["nap_yesterday"] == False]
    if len(nap_yes) > 10:
        print("\n  Next-day effect:")
        print(f"    After nap day:    {nap_yes['Hours Working'].mean():.2f}h (n={len(nap_yes)})")
        print(f"    After no-nap day: {nap_no['Hours Working'].mean():.2f}h (n={len(nap_no)})")

# ============================================================================
# 3. AMELIA TIME AND WORK: IS THERE A SWEET SPOT?
# ============================================================================
print("\n" + "=" * 70)
print("3. AMELIA TIME: SWEET SPOT ANALYSIS")
print("=" * 70)

if "cal_amelia" in df.columns:
    am = valid[valid["cal_amelia"].notna()].copy()
    print("  Overall correlation with Hours Working: ", end="")
    r, p = stats.pearsonr(am["cal_amelia"], am["Hours Working"])
    print(f"r={r:+.3f}, p={p:.3f}")

    # Buckets
    print("\n  By Amelia time bucket:")
    for lo, hi, label in [(0, 0.5, "< 30min"), (0.5, 2, "30min-2h"), (2, 4, "2-4h"),
                           (4, 6, "4-6h"), (6, 24, "> 6h")]:
        subset = am[(am["cal_amelia"] >= lo) & (am["cal_amelia"] < hi)]
        if len(subset) > 10:
            print(f"    {label:12s}: {subset['Hours Working'].mean():.2f}h work, "
                  f"Energy={subset['Energy'].mean():.1f}, "
                  f"Value={subset['Value'].mean():.1f} (n={len(subset)})")

    # Next-day effect
    am_lag = df.copy()
    am_lag["amelia_yesterday"] = am_lag["cal_amelia"].shift(1)
    am_lag = am_lag[am_lag["Hours Working"] > 0]
    r, p = stats.pearsonr(am_lag["amelia_yesterday"].fillna(0), am_lag["Hours Working"])
    print(f"\n  Yesterday's Amelia time → today's work: r={r:+.3f}, p={p:.3f}")

    # Amelia argue days
    if "amelia argue" in event_daily.columns:
        argue_daily = event_daily[["amelia argue"]].rename(columns={"amelia argue": "argue_hours"})
        am2 = valid.merge(argue_daily, left_on="date", right_index=True, how="left")
        am2["argue_hours"] = am2["argue_hours"].fillna(0)
        argue_days = am2[am2["argue_hours"] > 0]
        no_argue = am2[am2["argue_hours"] == 0]
        if len(argue_days) > 5:
            print(f"\n  Argue days: {argue_days['Hours Working'].mean():.2f}h work (n={len(argue_days)})")
            print(f"  No-argue:   {no_argue['Hours Working'].mean():.2f}h work (n={len(no_argue)})")

            # Next day after argument
            am2["argued_yesterday"] = am2["argue_hours"].shift(1) > 0
            argue_next = am2[am2["argued_yesterday"] == True]
            no_argue_next = am2[am2["argued_yesterday"] == False]
            if len(argue_next) > 5:
                print(f"  Day AFTER argue: {argue_next['Hours Working'].mean():.2f}h (n={len(argue_next)})")
                print(f"  Day after no-argue: {no_argue_next['Hours Working'].mean():.2f}h (n={len(no_argue_next)})")

# ============================================================================
# 4. EXERCISE TIMING AND TYPE
# ============================================================================
print("\n" + "=" * 70)
print("4. EXERCISE: WHAT TYPE AND HOW MUCH?")
print("=" * 70)

if "cal_gym" in df.columns:
    gym_days = valid[valid["cal_gym"] > 0]
    no_gym = valid[valid["cal_gym"] == 0]
    print(f"  Gym days: {len(gym_days)}, No gym: {len(no_gym)}")
    print(f"  With gym: {gym_days['Hours Working'].mean():.2f}h, Energy={gym_days['Energy'].mean():.1f}")
    print(f"  No gym:   {no_gym['Hours Working'].mean():.2f}h, Energy={no_gym['Energy'].mean():.1f}")

    # Gym duration sweet spot
    print("\n  By gym duration:")
    for lo, hi, label in [(0.01, 0.5, "< 30min"), (0.5, 1.0, "30-60min"), (1.0, 1.5, "1-1.5h"), (1.5, 4, "> 1.5h")]:
        subset = valid[(valid["cal_gym"] >= lo) & (valid["cal_gym"] < hi)]
        if len(subset) > 5:
            print(f"    {label:12s}: {subset['Hours Working'].mean():.2f}h work, Energy={subset['Energy'].mean():.1f} (n={len(subset)})")

if "cal_walk" in df.columns:
    walk_days = valid[valid["cal_walk"] > 0]
    no_walk = valid[valid["cal_walk"] == 0]
    if len(walk_days) > 10:
        print(f"\n  Walk days: {walk_days['Hours Working'].mean():.2f}h work, Energy={walk_days['Energy'].mean():.1f} (n={len(walk_days)})")
        print(f"  No walk:   {no_walk['Hours Working'].mean():.2f}h work, Energy={no_walk['Energy'].mean():.1f} (n={len(no_walk)})")

# ============================================================================
# 5. WASTE TIME PATTERNS: WHAT PREDICTS WASTE SPIRALS?
# ============================================================================
print("\n" + "=" * 70)
print("5. WASTE SPIRALS: WHAT TRIGGERS HIGH-WASTE DAYS?")
print("=" * 70)

high_waste = valid[valid["cal_waste"] > valid["cal_waste"].quantile(0.75)]
low_waste = valid[valid["cal_waste"] < valid["cal_waste"].quantile(0.25)]
print(f"  High waste days (>75th pct, >{valid['cal_waste'].quantile(0.75):.1f}h): n={len(high_waste)}")
print(f"  Low waste days (<25th pct, <{valid['cal_waste'].quantile(0.25):.1f}h): n={len(low_waste)}")

compare_cols = ["Hours Working", "Energy", "Focus", "Value", "cal_green", "cal_things"]
for col in compare_cols:
    if col in high_waste.columns:
        print(f"  {col:25s}: high_waste={high_waste[col].mean():.2f}, low_waste={low_waste[col].mean():.2f}")

# Does waste beget waste? (autocorrelation)
waste_ac = df[df["cal_waste"] > 0]["cal_waste"]
if len(waste_ac) > 30:
    r, p = stats.pearsonr(waste_ac.iloc[:-1].values, waste_ac.iloc[1:].values)
    print(f"\n  Waste autocorrelation (day-to-day): r={r:.3f}, p={p:.3f}")

# What predicts a high-waste day?
print("\n  What predicts waste? (correlation with cal_waste):")
pred_cols = ["Hours Working", "Energy", "Focus", "prev_hours", "cal_green",
             "cal_things", "is_weekend", "day_of_week"]
if "sleep" in supplements.columns:
    pred_cols.append("sleep")
for col in pred_cols:
    if col in valid.columns:
        mask = valid[col].notna() & valid["cal_waste"].notna()
        if mask.sum() > 30:
            r, p = stats.pearsonr(valid.loc[mask, col], valid.loc[mask, "cal_waste"])
            sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else ""
            print(f"    {col:25s}: r={r:+.3f} p={p:.3f} {sig}")

# Specific waste categories: which co-occur?
waste_events = cal_full[cal_full["category"] == "waste"].copy()
waste_pivot = waste_events.groupby(["date", "event_lower"])["duration"].sum().unstack(fill_value=0)
top_waste = ["twitter", "blogs", "porn", "youtube", "insta", "jack"]
available_waste = [w for w in top_waste if w in waste_pivot.columns]
if len(available_waste) > 1:
    print("\n  Waste category co-occurrence (correlations):")
    for i, w1 in enumerate(available_waste):
        for w2 in available_waste[i+1:]:
            r, p = stats.pearsonr(waste_pivot[w1], waste_pivot[w2])
            if abs(r) > 0.05:
                sig = "*" if p < 0.05 else ""
                print(f"    {w1:12s} ↔ {w2:12s}: r={r:+.3f} {sig}")

# ============================================================================
# 6. TIME OF DAY PATTERNS FROM DISTRACTED CSV
# ============================================================================
print("\n" + "=" * 70)
print("6. WHEN DURING THE DAY DO DISTRACTIONS HAPPEN?")
print("=" * 70)

dist = distracted.copy()
dist["hour"] = pd.to_datetime(dist["time"], format="%H:%M:%S", errors="coerce").dt.hour
dist_typed = dist[dist["type"].isin(["d", "u", "c", "b"])].copy()

if not dist_typed.empty and dist_typed["hour"].notna().any():
    hourly = dist_typed.groupby("hour").size()
    total_per_hour = dist.groupby("hour").size()
    dist_rate = (hourly / total_per_hour).dropna()

    print("  Distraction entries by hour of day:")
    for h in range(6, 24):
        if h in hourly.index:
            rate = dist_rate.get(h, 0)
            bar = "█" * int(rate * 50) if rate > 0 else ""
            print(f"    {h:2d}:00  {hourly.get(h, 0):5d} entries  ({rate:.0%} distraction rate)  {bar}")

# ============================================================================
# 7. ENERGY AND FOCUS PATTERNS
# ============================================================================
print("\n" + "=" * 70)
print("7. ENERGY/FOCUS: WHAT DRIVES THEM?")
print("=" * 70)

for outcome in ["Energy", "Focus"]:
    print(f"\n  --- Predictors of {outcome} ---")
    pred_cols2 = ["calendar_sleep_hours", "cal_green", "cal_gym", "cal_waste",
                  "cal_things", "is_weekend", "prev_hours"]
    if "sleep" in df.columns:
        pred_cols2.append("sleep")
    if "cal_walk" in df.columns:
        pred_cols2.append("cal_walk")
    if "cal_nap" in df.columns:
        pred_cols2.append("cal_nap")
    for col in pred_cols2:
        if col in valid.columns:
            mask = valid[col].notna() & valid[outcome].notna()
            if mask.sum() > 30:
                r, p = stats.pearsonr(valid.loc[mask, col], valid.loc[mask, outcome])
                sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else ""
                if abs(r) > 0.03:
                    print(f"    {col:30s}: r={r:+.3f} p={p:.3f} {sig}")

# ============================================================================
# 8. STREAK AND MOMENTUM ANALYSIS
# ============================================================================
print("\n" + "=" * 70)
print("8. MOMENTUM: HOW DOES CONSECUTIVE WORK AFFECT OUTPUT?")
print("=" * 70)

# Calculate streaks of consecutive work days
df["worked"] = df["Hours Working"] > 2  # meaningful work threshold
df["streak"] = 0
streak = 0
for i in range(len(df)):
    if df.iloc[i]["worked"]:
        streak += 1
    else:
        streak = 0
    df.iloc[i, df.columns.get_loc("streak")] = streak

streak_df = df[df["worked"]].copy()
print("  Work output by consecutive work day streak:")
for s_lo, s_hi, label in [(1, 1, "Day 1 (restart)"), (2, 3, "Days 2-3"),
                            (4, 5, "Days 4-5"), (6, 10, "Days 6-10"),
                            (11, 20, "Days 11-20"), (21, 100, "Days 21+")]:
    subset = streak_df[(streak_df["streak"] >= s_lo) & (streak_df["streak"] <= s_hi)]
    if len(subset) > 5:
        print(f"    {label:18s}: {subset['Hours Working'].mean():.2f}h, "
              f"Energy={subset['Energy'].mean():.1f}, Focus={subset['Focus'].mean():.1f} (n={len(subset)})")

# ============================================================================
# 9. DAY-OF-WEEK DEEP DIVE
# ============================================================================
print("\n" + "=" * 70)
print("9. DAY-OF-WEEK PATTERNS")
print("=" * 70)

dow_names = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
print(f"  {'Day':5s} {'Hours':>7s} {'Energy':>7s} {'Focus':>7s} {'Waste':>7s} {'Things':>7s} {'Value':>7s}")
for dow in range(7):
    subset = valid[valid["day_of_week"] == dow]
    if len(subset) > 10:
        print(f"  {dow_names[dow]:5s} {subset['Hours Working'].mean():7.2f} "
              f"{subset['Energy'].mean():7.1f} {subset['Focus'].mean():7.1f} "
              f"{subset['cal_waste'].mean():7.2f} {subset['cal_things'].mean():7.2f} "
              f"{subset['Value'].mean():7.1f}")

# ============================================================================
# 10. SEASONAL PATTERNS
# ============================================================================
print("\n" + "=" * 70)
print("10. SEASONAL PATTERNS")
print("=" * 70)

print(f"  {'Month':7s} {'Hours':>7s} {'Energy':>7s} {'Focus':>7s} {'Waste':>7s} {'n':>5s}")
for m in range(1, 13):
    subset = valid[valid["month"] == m]
    if len(subset) > 10:
        month_name = ["", "Jan", "Feb", "Mar", "Apr", "May", "Jun",
                      "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"][m]
        print(f"  {month_name:7s} {subset['Hours Working'].mean():7.2f} "
              f"{subset['Energy'].mean():7.1f} {subset['Focus'].mean():7.1f} "
              f"{subset['cal_waste'].mean():7.2f} {subset['n' if 'n' in subset.columns else 'Hours Working'].count():5d}")

# ============================================================================
# 11. MEDITATION EFFECT
# ============================================================================
print("\n" + "=" * 70)
print("11. MEDITATION EFFECT")
print("=" * 70)

if "cal_meditation" in df.columns:
    med_days = valid[valid["cal_meditation"] > 0]
    no_med = valid[valid["cal_meditation"] == 0]
    if len(med_days) > 10:
        print(f"  With meditation: {med_days['Hours Working'].mean():.2f}h, "
              f"Energy={med_days['Energy'].mean():.1f}, Focus={med_days['Focus'].mean():.1f} (n={len(med_days)})")
        print(f"  Without:         {no_med['Hours Working'].mean():.2f}h, "
              f"Energy={no_med['Energy'].mean():.1f}, Focus={no_med['Focus'].mean():.1f} (n={len(no_med)})")

# ============================================================================
# 12. COOKING AND SELF-CARE
# ============================================================================
print("\n" + "=" * 70)
print("12. COOKING / MEAL PREP EFFECT")
print("=" * 70)

if "cal_cook" in df.columns:
    cook_days = valid[valid["cal_cook"] > 0]
    no_cook = valid[valid["cal_cook"] == 0]
    if len(cook_days) > 10:
        print(f"  Cook days:    {cook_days['Hours Working'].mean():.2f}h, Energy={cook_days['Energy'].mean():.1f} (n={len(cook_days)})")
        print(f"  No-cook days: {no_cook['Hours Working'].mean():.2f}h, Energy={no_cook['Energy'].mean():.1f} (n={len(no_cook)})")

# ============================================================================
# 13. ANKI / LEARNING HABITS
# ============================================================================
print("\n" + "=" * 70)
print("13. ANKI / STRUCTURED LEARNING")
print("=" * 70)

if "cal_anki" in df.columns:
    anki_days = valid[valid["cal_anki"] > 0]
    no_anki = valid[valid["cal_anki"] == 0]
    if len(anki_days) > 10:
        print(f"  Anki days:    {anki_days['Hours Working'].mean():.2f}h, "
              f"Focus={anki_days['Focus'].mean():.1f} (n={len(anki_days)})")
        print(f"  No-Anki days: {no_anki['Hours Working'].mean():.2f}h, "
              f"Focus={no_anki['Focus'].mean():.1f} (n={len(no_anki)})")

# ============================================================================
# 14. PORN / MASTURBATION EFFECT (NEXT DAY)
# ============================================================================
print("\n" + "=" * 70)
print("14. PORN/JACK EFFECT ON NEXT DAY")
print("=" * 70)

for ev_name, col_name in [("porn", "cal_porn"), ("jack", "cal_jack")]:
    if col_name in df.columns:
        lag_df = df.copy()
        lag_df[f"{ev_name}_yesterday"] = lag_df[col_name].shift(1)
        lag_df = lag_df[lag_df["Hours Working"] > 0]

        high = lag_df[lag_df[f"{ev_name}_yesterday"] > 0]
        low = lag_df[lag_df[f"{ev_name}_yesterday"] == 0]
        if len(high) > 10:
            print(f"  Day after {ev_name}:    {high['Hours Working'].mean():.2f}h, Energy={high['Energy'].mean():.1f} (n={len(high)})")
            print(f"  Day after no {ev_name}: {low['Hours Working'].mean():.2f}h, Energy={low['Energy'].mean():.1f} (n={len(low)})")

# Also same-day
for ev_name, col_name in [("porn", "cal_porn"), ("jack", "cal_jack")]:
    if col_name in df.columns:
        high = valid[valid[col_name] > 0]
        low = valid[valid[col_name] == 0]
        if len(high) > 10:
            print(f"  Same-day {ev_name}:    {high['Hours Working'].mean():.2f}h (n={len(high)})")
            print(f"  Same-day no {ev_name}: {low['Hours Working'].mean():.2f}h (n={len(low)})")

# ============================================================================
# 15. DRINKING EFFECT
# ============================================================================
print("\n" + "=" * 70)
print("15. DRINKING EFFECT")
print("=" * 70)

if "cal_drink" in df.columns:
    lag_df = df.copy()
    lag_df["drank_yesterday"] = lag_df["cal_drink"].shift(1) > 0
    lag_df = lag_df[lag_df["Hours Working"] > 0]

    drank = lag_df[lag_df["drank_yesterday"] == True]
    sober = lag_df[lag_df["drank_yesterday"] == False]
    if len(drank) > 5:
        print(f"  Day after drinking: {drank['Hours Working'].mean():.2f}h, Energy={drank['Energy'].mean():.1f} (n={len(drank)})")
        print(f"  Day after sober:    {sober['Hours Working'].mean():.2f}h, Energy={sober['Energy'].mean():.1f} (n={len(sober)})")

# ============================================================================
# 16. WORK SESSION FRAGMENTATION
# ============================================================================
print("\n" + "=" * 70)
print("16. WORK SESSION FRAGMENTATION")
print("=" * 70)

# From distracted CSV: how many separate work sessions per day?
work_entries = distracted[distracted["type"].isin(["s", "e"])].copy()
sessions_per_day = work_entries[work_entries["type"] == "s"].groupby("date").size()
sessions_per_day.name = "n_sessions"
frag = valid.merge(sessions_per_day, left_on="date", right_index=True, how="left")
frag["n_sessions"] = frag["n_sessions"].fillna(1)

print("  Work output by number of work sessions per day:")
for lo, hi, label in [(1, 1, "1 session"), (2, 2, "2 sessions"), (3, 3, "3 sessions"),
                       (4, 5, "4-5 sessions"), (6, 20, "6+ sessions")]:
    subset = frag[(frag["n_sessions"] >= lo) & (frag["n_sessions"] <= hi)]
    if len(subset) > 5:
        print(f"    {label:15s}: {subset['Hours Working'].mean():.2f}h (n={len(subset)})")

# ============================================================================
# 17. DISTRACTION TYPES AND THEIR COSTS
# ============================================================================
print("\n" + "=" * 70)
print("17. DISTRACTION TYPES: WHICH COST THE MOST TIME?")
print("=" * 70)

dist_with_len = distracted[distracted["type"].isin(["d", "u", "c"]) & (distracted["length_minutes"] > 0)].copy()
if not dist_with_len.empty:
    # By comment/description - what are the actual distractions?
    dist_with_len["comment_lower"] = dist_with_len["comment"].str.lower().str.strip()
    top_distractions = dist_with_len.groupby("comment_lower").agg(
        count=("length_minutes", "size"),
        total_min=("length_minutes", "sum"),
        avg_min=("length_minutes", "mean"),
    ).sort_values("total_min", ascending=False)

    print("  Top distractions by total time lost:")
    for name, row in top_distractions.head(25).iterrows():
        if row["total_min"] > 60:
            print(f"    {str(name)[:35]:35s}: {row['total_min']/60:7.1f}h total, "
                  f"{row['avg_min']:.1f}min avg, {int(row['count']):5d}x")

# ============================================================================
# 18. VALUE vs HOURS WORKING RELATIONSHIP
# ============================================================================
print("\n" + "=" * 70)
print("18. VALUE PERCEPTION: WHEN DO YOU FEEL PRODUCTIVE?")
print("=" * 70)

v = valid[valid["Value"].notna() & (valid["Value"] > 0)].copy()
if len(v) > 30:
    # High value but low hours = efficient days
    v["value_per_hour"] = v["Value"] / v["Hours Working"].clip(lower=0.5)
    high_vph = v[v["value_per_hour"] > v["value_per_hour"].quantile(0.75)]
    low_vph = v[v["value_per_hour"] < v["value_per_hour"].quantile(0.25)]
    print(f"  High value/hour days (top 25%): {high_vph['Hours Working'].mean():.1f}h, "
          f"Energy={high_vph['Energy'].mean():.1f}, Focus={high_vph['Focus'].mean():.1f}")
    print(f"  Low value/hour days (bot 25%):  {low_vph['Hours Working'].mean():.1f}h, "
          f"Energy={low_vph['Energy'].mean():.1f}, Focus={low_vph['Focus'].mean():.1f}")

    # What predicts high perceived value?
    print("\n  Correlations with Value:")
    for col in ["Hours Working", "Energy", "Focus", "cal_waste", "cal_things",
                "cal_gym", "is_weekend", "streak"]:
        if col in v.columns:
            mask = v[col].notna()
            if mask.sum() > 30:
                r, p = stats.pearsonr(v.loc[mask, col], v.loc[mask, "Value"])
                sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else ""
                print(f"    {col:25s}: r={r:+.3f} {sig}")

# ============================================================================
# 19. OPTIMAL DAY STRUCTURE
# ============================================================================
print("\n" + "=" * 70)
print("19. OPTIMAL DAY: PROFILE OF YOUR BEST WORK DAYS")
print("=" * 70)

top_10pct = valid.nlargest(int(len(valid) * 0.1), "Hours Working")
bot_10pct = valid.nsmallest(int(len(valid) * 0.1), "Hours Working")

profile_cols = ["Hours Working", "Energy", "Focus", "Value", "cal_green", "cal_things",
                "cal_waste", "cal_blue", "is_weekend", "day_of_week"]
if "sleep" in valid.columns:
    profile_cols.append("sleep")
if "wake_hour" in valid.columns:
    profile_cols.append("wake_hour")
if "cal_gym" in valid.columns:
    profile_cols.append("cal_gym")
if "cal_amelia" in valid.columns:
    profile_cols.append("cal_amelia")
if "cal_nap" in valid.columns:
    profile_cols.append("cal_nap")
if "work_start_hour" in valid.columns:
    profile_cols.append("work_start_hour")

print(f"  {'Metric':30s} {'Top 10%':>10s} {'Bottom 10%':>12s} {'Avg':>10s}")
for col in profile_cols:
    if col in top_10pct.columns:
        print(f"  {col:30s} {top_10pct[col].mean():10.2f} {bot_10pct[col].mean():12.2f} {valid[col].mean():10.2f}")

print("\n  Top 10% day-of-week distribution:")
for dow in range(7):
    pct = (top_10pct["day_of_week"] == dow).mean() * 100
    print(f"    {dow_names[dow]}: {pct:.0f}%")

# ============================================================================
# 20. REGIME TRANSITION ANALYSIS
# ============================================================================
print("\n" + "=" * 70)
print("20. REGIME TRANSITIONS: WHAT HAPPENED AT EACH SWITCH?")
print("=" * 70)

df_sorted = df.sort_values("date")
df_sorted["regime_change"] = df_sorted["regime"] != df_sorted["regime"].shift(1)
transitions = df_sorted[df_sorted["regime_change"] & df_sorted["regime"].notna()].copy()

for _, row in transitions.iterrows():
    date = row["date"]
    new_regime = row["regime"]
    # Get 14 days before and after
    before = df_sorted[(df_sorted["date"] >= date - timedelta(days=14)) &
                        (df_sorted["date"] < date) &
                        (df_sorted["Hours Working"] > 0)]
    after = df_sorted[(df_sorted["date"] >= date) &
                       (df_sorted["date"] < date + timedelta(days=14)) &
                       (df_sorted["Hours Working"] > 0)]
    if len(before) > 3 and len(after) > 3:
        print(f"  {date.date()}: → {new_regime}")
        print(f"    Before: {before['Hours Working'].mean():.1f}h, Energy={before['Energy'].mean():.1f}")
        print(f"    After:  {after['Hours Working'].mean():.1f}h, Energy={after['Energy'].mean():.1f}")
