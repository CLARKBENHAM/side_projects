# %% use conda env: side_projects
"""Micro-level analysis of minute-by-minute distraction data.

Goes beyond daily aggregates to look at within-session patterns,
distraction sequences, recovery dynamics, and temporal microstructure.
"""
import matplotlib

matplotlib.use("Agg")

import re
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats

from productivity_analysis import (
    CALENDAR_DIR,
    DAILY_SUMMARY_CSV,
    DISTRACTED_CSV,
    NON_SUPPLEMENT_KEYS,
    SUPPLEMENT_ALIASES,
    extract_daily_supplements,
    extract_work_start_time,
    load_calendar_full,
    load_calendar_sleep,
    load_daily_summary_full,
    load_distracted_stacked,
    parse_supplement_dict,
)

# ============================================================================
# DATA LOADING
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

# Regime dummies
df["is_hive"] = (df["regime"] == "Hive").astype(int)
df["is_mats"] = (df["regime"] == "Mats").astype(int)
df["is_diesl"] = (df["regime"] == "diesl").astype(int)

BASE_CONTROLS = ["is_hive", "is_mats", "is_diesl", "is_weekend", "day_of_week"]


def controlled_effect(
    data: pd.DataFrame,
    treatment_col: str,
    outcome: str = "Hours Working",
    extra_controls: list[str] | None = None,
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
    if np.isnan(result["coef"]):
        return
    sig = (
        "***"
        if result["p"] < 0.001
        else "**" if result["p"] < 0.01 else "*" if result["p"] < 0.05 else ""
    )
    print(
        f"  {label:50s}: {result['coef']:+.3f}{unit}  "
        f"(p={result['p']:.3f}, n={result['n']}) {sig}"
    )


# ============================================================================
# PARSE ALL ENTRIES INTO SESSIONS WITH TIMING
# ============================================================================
print("Building session-level data...")

dist = distracted.copy()
dist["clock_time"] = pd.to_datetime(dist["time"], format="mixed", errors="coerce")
dist = dist.dropna(subset=["clock_time"])
# Build real datetime: date + clock time (fixes Bug #2: timestamps had no real date)
dist["event_dt"] = (
    dist["date"].dt.normalize()
    + pd.to_timedelta(dist["clock_time"].dt.hour, unit="h")
    + pd.to_timedelta(dist["clock_time"].dt.minute, unit="m")
    + pd.to_timedelta(dist["clock_time"].dt.second, unit="s")
)
dist["hour_decimal"] = dist["clock_time"].dt.hour + dist["clock_time"].dt.minute / 60

# Parse sessions by work_increment instead of grouping by date.
# (Fixes Bug #1: cross-midnight sessions were dropped when grouped by date.)
sessions: list[dict] = []
session_events: list[dict] = []  # every event tagged with its session

for wi, wi_data in dist.groupby("work_increment"):
    wi_sorted = wi_data.sort_values("event_dt").reset_index(drop=True)
    starts = wi_sorted[wi_sorted["type"] == "s"]
    ends = wi_sorted[wi_sorted["type"] == "e"]

    if starts.empty:
        continue

    start_row = starts.iloc[0]
    session_start = start_row["event_dt"]
    session_date = start_row["date"]  # assign session to date of its start

    if ends.empty:
        continue
    end_row = ends.iloc[-1]
    session_end = end_row["event_dt"]

    # Handle cross-midnight: if end clock time < start clock time, add a day
    if session_end < session_start:
        session_end += pd.Timedelta(days=1)

    session_duration_h = (session_end - session_start).total_seconds() / 3600
    if session_duration_h <= 0 or session_duration_h > 16:
        continue

    in_session = wi_sorted.copy()
    # Fix event_dt for cross-midnight events: any event with clock < start clock
    # that belongs to this session actually happened after midnight
    for idx in in_session.index:
        if in_session.loc[idx, "event_dt"] < session_start:
            in_session.loc[idx, "event_dt"] += pd.Timedelta(days=1)

    n_distractions = len(in_session[in_session["type"] == "d"])
    n_unfocused = len(in_session[in_session["type"] == "u"])
    n_interrupts = len(in_session[in_session["type"] == "i"])
    n_tasks = len(in_session[in_session["type"] == "t"])

    distraction_minutes = in_session.loc[
        in_session["type"].isin(["d", "u", "i"]), "length_minutes"
    ].sum()
    distraction_minutes = (
        distraction_minutes if not np.isnan(distraction_minutes) else 0
    )

    start_comment = start_row["comment"] if pd.notna(start_row["comment"]) else ""

    sessions.append(
        {
            "date": session_date,
            "regime": start_row.get("for", ""),
            "session_start": session_start,
            "session_end": session_end,
            "start_hour": session_start.hour + session_start.minute / 60,
            "duration_h": session_duration_h,
            "n_distractions": n_distractions,
            "n_unfocused": n_unfocused,
            "n_interrupts": n_interrupts,
            "n_tasks": n_tasks,
            "total_disruptions": n_distractions + n_unfocused + n_interrupts,
            "distraction_minutes": distraction_minutes,
            "work_increment": wi,
            "start_comment": start_comment,
            "n_events": len(in_session),
        }
    )

    for _, ev in in_session.iterrows():
        hours_into = (ev["event_dt"] - session_start).total_seconds() / 3600
        pct_into = hours_into / session_duration_h if session_duration_h > 0 else 0
        session_events.append(
            {
                "date": session_date,
                "event_dt": ev["event_dt"],
                "type": ev["type"],
                "comment": ev["comment"],
                "length_minutes": ev["length_minutes"],
                "hours_into_session": hours_into,
                "pct_into_session": pct_into,
                "session_duration_h": session_duration_h,
                "session_start_hour": session_start.hour + session_start.minute / 60,
                "work_increment": wi,
            }
        )

sess_df = pd.DataFrame(sessions)
ev_df = pd.DataFrame(session_events)

# Merge daily data onto sessions
sess_df = sess_df.merge(
    df[
        [
            "date",
            "Hours Working",
            "Energy",
            "Focus",
            "Value",
            "is_hive",
            "is_mats",
            "is_diesl",
            "is_weekend",
            "day_of_week",
        ]
    ],
    on="date",
    how="left",
)

print(f"Built {len(sess_df)} sessions, {len(ev_df)} session events")
print(f"Date range: {sess_df['date'].min().date()} to {sess_df['date'].max().date()}")
print()


# ============================================================================
# 1. DISTRACTION RATE OVER SESSION LIFETIME
# ============================================================================
print("=" * 70)
print("1. DISTRACTION RATE OVER SESSION LIFETIME")
print("   (are you more vulnerable early or late in a session?)")
print("=" * 70)

disruptions = ev_df[ev_df["type"].isin(["d", "u"])].copy()

# Bin by percentage into session (0-100% in 10% buckets)
disruptions["pct_bin"] = (disruptions["pct_into_session"] * 10).astype(int).clip(0, 9)
# Count all events in each bin for normalization
ev_df["pct_bin"] = (ev_df["pct_into_session"] * 10).astype(int).clip(0, 9)

pct_dist_counts = disruptions.groupby("pct_bin").size()
pct_total_counts = ev_df.groupby("pct_bin").size()
pct_rate = (pct_dist_counts / pct_total_counts).fillna(0)

print("\n  Distraction rate by % through session:")
print(f"  {'Session %':>12s} {'Distractions':>13s} {'Total events':>13s} {'Rate':>8s}")
for b in range(10):
    lo = b * 10
    hi = (b + 1) * 10
    d_ct = pct_dist_counts.get(b, 0)
    t_ct = pct_total_counts.get(b, 0)
    rate = pct_rate.get(b, 0)
    bar = "█" * int(rate * 50)
    print(f"  {lo:3d}-{hi:3d}%      {d_ct:13d} {t_ct:13d} {rate:8.1%}  {bar}")

# Also by absolute hour into session
disruptions["hour_bin"] = disruptions["hours_into_session"].astype(int).clip(0, 10)
hour_dist = disruptions.groupby("hour_bin").size()
# Count sessions that lasted at least this long for proper rate
session_reach = []
for h in range(11):
    n_sessions_reaching = (sess_df["duration_h"] > h).sum()
    session_reach.append({"hour_bin": h, "sessions_reaching": n_sessions_reaching})
reach_df = pd.DataFrame(session_reach)

print("\n  Distraction count & rate by hour into session:")
print(
    f"  {'Hour':>6s} {'Distractions':>13s} {'Sessions active':>16s} {'Rate/session':>13s}"
)
for _, row in reach_df.iterrows():
    h = int(row["hour_bin"])
    d_ct = hour_dist.get(h, 0)
    n_active = int(row["sessions_reaching"])
    rate = d_ct / n_active if n_active > 0 else 0
    print(f"  {h:3d}-{h+1:3d}  {d_ct:13d} {n_active:16d} {rate:13.2f}")

print()


# ============================================================================
# 2. FIRST DISTRACTION TIMING — HOW LONG UNTIL FOCUS BREAKS?
# ============================================================================
print("=" * 70)
print("2. FIRST DISTRACTION — how long until focus first breaks?")
print("=" * 70)

# Bug #3 fix: include sessions with NO distractions (survivorship bias).
# Sessions that never broke focus are right-censored at session duration.
first_dist = disruptions.groupby("work_increment")["hours_into_session"].min()
first_dist_df = first_dist.reset_index(name="first_distraction_h")
first_dist_df = first_dist_df.merge(
    sess_df[
        [
            "work_increment",
            "date",
            "duration_h",
            "start_hour",
            "Hours Working",
            "Energy",
            "Focus",
            "start_comment",
        ]
    ].drop_duplicates("work_increment"),
    on="work_increment",
    how="left",
)

# Add sessions with zero distractions — censored at session duration
no_dist_sessions = sess_df[
    ~sess_df["work_increment"].isin(first_dist_df["work_increment"])
][
    [
        "work_increment",
        "date",
        "duration_h",
        "start_hour",
        "Hours Working",
        "Energy",
        "Focus",
        "start_comment",
    ]
].copy()
no_dist_sessions["first_distraction_h"] = no_dist_sessions["duration_h"]
no_dist_sessions["censored"] = True
first_dist_df["censored"] = False
first_dist_all = pd.concat([first_dist_df, no_dist_sessions], ignore_index=True)

n_censored = first_dist_all["censored"].sum()
n_uncensored = (~first_dist_all["censored"]).sum()
print(
    f"\n  Sessions analyzed: {len(first_dist_all)} "
    f"({n_uncensored} had distractions, {n_censored} never broke focus)"
)
print(
    f"  Mean time to first distraction (all): {first_dist_all['first_distraction_h'].mean():.2f}h "
    f"({first_dist_all['first_distraction_h'].mean() * 60:.0f} min)"
)
print(
    f"  Mean (only sessions that broke): {first_dist_df['first_distraction_h'].mean():.2f}h "
    f"({first_dist_df['first_distraction_h'].mean() * 60:.0f} min)"
)
print(
    f"  Median time to first distraction: {first_dist_all['first_distraction_h'].median():.2f}h "
    f"({first_dist_all['first_distraction_h'].median() * 60:.0f} min)"
)

# Distribution (including censored sessions in 'never' bucket)
print("\n  Distribution of first-distraction timing:")
for lo, hi, label in [
    (0, 5 / 60, "<5 min"),
    (5 / 60, 15 / 60, "5-15 min"),
    (15 / 60, 30 / 60, "15-30 min"),
    (30 / 60, 1, "30-60 min"),
    (1, 2, "1-2 hours"),
    (2, 4, "2-4 hours"),
    (4, 20, "4+ hours"),
]:
    n = (
        (first_dist_all["first_distraction_h"] >= lo)
        & (first_dist_all["first_distraction_h"] < hi)
    ).sum()
    pct = n / len(first_dist_all) * 100
    bar = "█" * int(pct / 2)
    print(f"    {label:15s}: {n:5d} ({pct:5.1f}%) {bar}")
print(
    f"    {'Never (cens.)':15s}: {n_censored:5d} ({n_censored/len(first_dist_all)*100:5.1f}%)"
)

# Does early first distraction predict a bad day?
first_dist_daily = (
    first_dist_all.groupby("date")
    .agg(
        min_first_dist=("first_distraction_h", "min"),
        mean_first_dist=("first_distraction_h", "mean"),
    )
    .reset_index()
)
first_dist_daily = first_dist_daily.merge(
    df[["date", "Hours Working", "Energy", "Focus"] + BASE_CONTROLS],
    on="date",
    how="left",
)

print("\n  --- Does time-to-first-distraction predict daily output? ---")
r = controlled_effect(first_dist_daily, "min_first_dist")
print_effect("Min first-dist (earliest session) → Hours", r)
r = controlled_effect(first_dist_daily, "mean_first_dist")
print_effect("Mean first-dist (avg across sessions) → Hours", r)
r = controlled_effect(first_dist_daily, "mean_first_dist", outcome="Focus")
print_effect("Mean first-dist → Focus", r, unit="pts")

# Does time of day affect first distraction timing?
morning = first_dist_all[first_dist_all["start_hour"] < 12]
afternoon = first_dist_all[
    (first_dist_all["start_hour"] >= 12) & (first_dist_all["start_hour"] < 17)
]
evening = first_dist_all[first_dist_all["start_hour"] >= 17]
print("\n  First distraction by session start time:")
print(
    f"    Morning (<12):   {morning['first_distraction_h'].mean():.2f}h "
    f"({morning['first_distraction_h'].mean()*60:.0f} min), n={len(morning)}"
)
print(
    f"    Afternoon (12-5): {afternoon['first_distraction_h'].mean():.2f}h "
    f"({afternoon['first_distraction_h'].mean()*60:.0f} min), n={len(afternoon)}"
)
print(
    f"    Evening (5+):    {evening['first_distraction_h'].mean():.2f}h "
    f"({evening['first_distraction_h'].mean()*60:.0f} min), n={len(evening)}"
)
print()


# ============================================================================
# 3. DISTRACTION CASCADES — does one distraction trigger more?
# ============================================================================
print("=" * 70)
print("3. DISTRACTION CASCADES — does one distraction trigger more?")
print("=" * 70)

# For each distraction, measure time until next distraction in same session
disruptions_sorted = disruptions.sort_values(
    ["work_increment", "hours_into_session"]
).reset_index(drop=True)

cascade_gaps: list[dict] = []
for wi, grp in disruptions_sorted.groupby("work_increment"):
    times = grp["hours_into_session"].values
    for i in range(len(times) - 1):
        gap_min = (times[i + 1] - times[i]) * 60
        cascade_gaps.append(
            {
                "work_increment": wi,
                "distraction_number": i + 1,  # 1-indexed: gap after 1st, 2nd, etc
                "gap_minutes": gap_min,
                "hours_into": times[i],
            }
        )

if cascade_gaps:
    gap_df = pd.DataFrame(cascade_gaps)

    print(f"\n  Total inter-distraction gaps: {len(gap_df)}")
    print(f"  Mean gap between distractions: {gap_df['gap_minutes'].mean():.1f} min")
    print(f"  Median gap: {gap_df['gap_minutes'].median():.1f} min")

    # Does the gap shrink after more distractions? (cascade effect)
    print("\n  Gap between distractions by distraction number:")
    print(
        f"  {'After distraction #':>22s} {'Mean gap (min)':>15s} {'Median (min)':>13s} {'n':>6s}"
    )
    for d_num in range(1, 8):
        sub = gap_df[gap_df["distraction_number"] == d_num]
        if len(sub) > 20:
            print(
                f"  {'#' + str(d_num):>22s} {sub['gap_minutes'].mean():15.1f} "
                f"{sub['gap_minutes'].median():13.1f} {len(sub):6d}"
            )

    # After the 3rd+ distraction, does the gap get shorter?
    early = gap_df[gap_df["distraction_number"] <= 2]["gap_minutes"]
    late = gap_df[gap_df["distraction_number"] >= 4]["gap_minutes"]
    if len(early) > 20 and len(late) > 20:
        t_stat, p_val = stats.mannwhitneyu(early, late, alternative="greater")
        print(
            f"\n  Gap after distractions 1-2: {early.mean():.1f} min (n={len(early)})"
        )
        print(f"  Gap after distractions 4+:  {late.mean():.1f} min (n={len(late)})")
        print(f"  Mann-Whitney test (early > late): p={p_val:.4f}")

    # Short gap cascade: if gap < 10min, what fraction of time does
    # the NEXT gap also < 10min?
    gap_df["short_gap"] = gap_df["gap_minutes"] < 10
    consecutive = []
    for wi, grp in gap_df.groupby("work_increment"):
        vals = grp.sort_values("distraction_number")["short_gap"].values
        for i in range(len(vals) - 1):
            consecutive.append(
                {
                    "this_short": vals[i],
                    "next_short": vals[i + 1],
                }
            )
    if consecutive:
        consec_df = pd.DataFrame(consecutive)
        after_short = consec_df[consec_df["this_short"]]["next_short"].mean()
        after_long = consec_df[~consec_df["this_short"]]["next_short"].mean()
        print(f"\n  P(next gap <10min | this gap <10min): {after_short:.1%}")
        print(f"  P(next gap <10min | this gap >=10min): {after_long:.1%}")
        print(
            f"  → Short gaps are {'self-reinforcing' if after_short > after_long else 'NOT self-reinforcing'}"
        )

print()


# ============================================================================
# 4. DISTRACTION TYPE TAXONOMY — what specifically pulls you away?
# ============================================================================
print("=" * 70)
print("4. DISTRACTION TYPE TAXONOMY — what specifically pulls you away?")
print("=" * 70)

dist_typed = dist[dist["type"].isin(["d", "u"])].copy()
dist_typed["comment_clean"] = dist_typed["comment"].fillna("").str.lower().str.strip()

# Map single-letter codes to readable names
code_map = {
    "b": "blog",
    "c": "chatting",
    "d": "drink",
    "s": "spotify",
    "z": "zone_out",
    "t": "toil",
    "i": "interrupt",
    "w": "bathroom",
    "": "unspecified",
}
dist_typed["category"] = (
    dist_typed["comment_clean"].map(code_map).fillna(dist_typed["comment_clean"])
)
# Truncate long comments
dist_typed["category"] = dist_typed["category"].str[:40]

# Count and duration by type
type_stats = (
    dist_typed.groupby("category")
    .agg(
        count=("length_minutes", "size"),
        total_min=("length_minutes", "sum"),
        avg_min=("length_minutes", "mean"),
        median_min=("length_minutes", "median"),
    )
    .sort_values("total_min", ascending=False)
)

print(f"\n  Total distraction entries: {len(dist_typed)}")
print(
    f"\n  {'Category':40s} {'Count':>6s} {'Total hrs':>10s} {'Avg min':>8s} {'Med min':>8s}"
)
print("  " + "-" * 76)
for name, row in type_stats.head(30).iterrows():
    if row["count"] >= 5:
        total_h = row["total_min"] / 60
        print(
            f"  {str(name)[:40]:40s} {int(row['count']):6d} {total_h:10.1f} "
            f"{row['avg_min']:8.1f} {row['median_min']:8.1f}"
        )

# d vs u: are "got up" distractions worse than "stayed at desk" unfocused?
d_entries = dist_typed[dist_typed["type"] == "d"]
u_entries = dist_typed[dist_typed["type"] == "u"]
print(
    f"\n  Distraction (d, got up):  mean={d_entries['length_minutes'].mean():.1f}min, "
    f"median={d_entries['length_minutes'].median():.1f}min, n={len(d_entries)}"
)
print(
    f"  Unfocused (u, at desk):   mean={u_entries['length_minutes'].mean():.1f}min, "
    f"median={u_entries['length_minutes'].median():.1f}min, n={len(u_entries)}"
)

# Which distraction types lead to the longest recovery?
# Recovery = gap between distraction end and next productive event
print()


# ============================================================================
# 5. SESSION SHAPE — what does a productive vs unproductive session look like?
# ============================================================================
print("=" * 70)
print("5. SESSION SHAPE — productive vs unproductive sessions")
print("=" * 70)

# Compute distraction rate and efficiency per session
sess_df["disruption_rate"] = sess_df["total_disruptions"] / sess_df["duration_h"]
sess_df["pct_distracted"] = (
    sess_df["distraction_minutes"] / (sess_df["duration_h"] * 60)
).clip(0, 1)

# Split into quartiles by daily output
daily_hours = (
    sess_df.groupby("date")["duration_h"].sum().reset_index(name="total_session_h")
)
sess_df = sess_df.merge(daily_hours, on="date", how="left")

print("\n  Session statistics:")
print(f"    Mean duration: {sess_df['duration_h'].mean():.2f}h")
print(f"    Mean disruptions/session: {sess_df['total_disruptions'].mean():.1f}")
print(f"    Mean disruption rate: {sess_df['disruption_rate'].mean():.2f}/h")
print(f"    Mean % time distracted: {sess_df['pct_distracted'].mean():.1%}")

# Profile by session duration bucket
print("\n  --- Session profile by duration ---")
print(
    f"  {'Duration':15s} {'n':>5s} {'Dist/hr':>8s} {'%Distracted':>12s} {'Disruptions':>12s}"
)
for lo, hi, label in [
    (0, 1, "<1h"),
    (1, 2, "1-2h"),
    (2, 4, "2-4h"),
    (4, 6, "4-6h"),
    (6, 8, "6-8h"),
    (8, 16, "8+h"),
]:
    sub = sess_df[(sess_df["duration_h"] >= lo) & (sess_df["duration_h"] < hi)]
    if len(sub) > 10:
        print(
            f"  {label:15s} {len(sub):5d} {sub['disruption_rate'].mean():8.2f} "
            f"{sub['pct_distracted'].mean():12.1%} {sub['total_disruptions'].mean():12.1f}"
        )


# Does session-level distraction rate predict daily outcomes?
# Aggregate sessions to daily — Bug #5 fix: duration-weight the averages
# instead of unweighted mean across sessions (short 30min sessions shouldn't
# count the same as 4h sessions)
def _wavg(group: pd.DataFrame, val_col: str, wt_col: str = "duration_h") -> float:
    w = group[wt_col]
    v = group[val_col]
    valid = w.notna() & v.notna() & (w > 0)
    if valid.sum() == 0:
        return np.nan
    return float(np.average(v[valid], weights=w[valid]))


daily_micro = (
    sess_df.groupby("date")
    .apply(
        lambda g: pd.Series(
            {
                "n_sessions": len(g),
                "total_session_h": g["duration_h"].sum(),
                "mean_disruption_rate": _wavg(g, "disruption_rate"),
                "total_disruptions": g["total_disruptions"].sum(),
                "mean_pct_distracted": _wavg(g, "pct_distracted"),
                "total_distraction_min": g["distraction_minutes"].sum(),
                "first_session_start": g["start_hour"].min(),
                "longest_session": g["duration_h"].max(),
            }
        ),
        include_groups=False,
    )
    .reset_index()
)

daily_micro = daily_micro.merge(
    df[["date", "Hours Working", "Energy", "Focus", "Value"] + BASE_CONTROLS],
    on="date",
    how="left",
)

print("\n  --- Session metrics → daily outcomes (controlled) ---")
for metric, label in [
    ("mean_disruption_rate", "Mean disruption rate/hr"),
    ("mean_pct_distracted", "Mean % time distracted"),
    ("total_distraction_min", "Total distraction minutes"),
    ("longest_session", "Longest session (hours)"),
    ("n_sessions", "Number of sessions"),
    ("first_session_start", "First session start hour"),
]:
    r = controlled_effect(daily_micro, metric)
    print_effect(f"{label} → Hours Working", r)

for metric, label in [
    ("mean_disruption_rate", "Mean disruption rate/hr"),
    ("mean_pct_distracted", "Mean % time distracted"),
]:
    r = controlled_effect(daily_micro, metric, outcome="Focus")
    print_effect(f"{label} → Focus", r, unit="pts")

print()


# ============================================================================
# 6. DISTRACTION DURATION MATTERS — short vs long disruptions
# ============================================================================
print("=" * 70)
print("6. DISTRACTION DURATION — do short or long breaks differ in impact?")
print("=" * 70)

dist_with_len = dist[
    dist["type"].isin(["d", "u", "i"]) & (dist["length_minutes"] > 0)
].copy()

# Distribution of distraction durations
print(f"\n  Total disruptions with duration: {len(dist_with_len)}")
print(f"  Mean duration: {dist_with_len['length_minutes'].mean():.1f} min")
print(f"  Median duration: {dist_with_len['length_minutes'].median():.1f} min")

print("\n  Duration distribution:")
for lo, hi, label in [
    (0, 3, "0-3 min"),
    (3, 5, "3-5 min"),
    (5, 10, "5-10 min"),
    (10, 20, "10-20 min"),
    (20, 30, "20-30 min"),
    (30, 60, "30-60 min"),
    (60, 500, "60+ min"),
]:
    n = (
        (dist_with_len["length_minutes"] >= lo) & (dist_with_len["length_minutes"] < hi)
    ).sum()
    pct = n / len(dist_with_len) * 100
    bar = "█" * int(pct / 2)
    print(f"    {label:12s}: {n:5d} ({pct:5.1f}%) {bar}")

# Aggregate daily: average distraction duration, and count of long vs short
daily_dist_profile = (
    dist_with_len.groupby("date")
    .agg(
        mean_dist_duration=("length_minutes", "mean"),
        median_dist_duration=("length_minutes", "median"),
        n_short=("length_minutes", lambda x: (x < 10).sum()),
        n_long=("length_minutes", lambda x: (x >= 20).sum()),
        max_dist_duration=("length_minutes", "max"),
        total_dist_minutes=("length_minutes", "sum"),
    )
    .reset_index()
)

daily_dist_profile = daily_dist_profile.merge(
    df[["date", "Hours Working", "Energy", "Focus"] + BASE_CONTROLS],
    on="date",
    how="left",
)

# Bug #6 fix: control for total work hours so short-distraction count
# isn't just a proxy for "worked a long time = more distractions".
daily_dist_profile = daily_dist_profile.merge(
    df[["date", "Hours Working"]].rename(columns={"Hours Working": "_total_work_h"}),
    on="date",
    how="left",
)

print("\n  --- Distraction duration metrics → outcomes (controlled) ---")
print("  (also controlling for total work hours to avoid confounding)")
for metric, label in [
    ("mean_dist_duration", "Mean distraction duration (min)"),
    ("n_short", "Count of short (<10min) distractions"),
    ("n_long", "Count of long (>=20min) distractions"),
    ("max_dist_duration", "Longest single distraction (min)"),
]:
    r = controlled_effect(daily_dist_profile, metric, extra_controls=["_total_work_h"])
    print_effect(f"{label} → Hours", r)

# Are long distractions just as predictive of bad days as many short ones?
daily_dist_profile["short_ratio"] = daily_dist_profile["n_short"] / (
    daily_dist_profile["n_short"] + daily_dist_profile["n_long"]
).replace(0, np.nan)
r = controlled_effect(
    daily_dist_profile, "short_ratio", extra_controls=["_total_work_h"]
)
print_effect("Short-distraction ratio → Hours Working", r)

print()


# ============================================================================
# 7. TIME OF DAY VULNERABILITY — when are you most distractible?
# ============================================================================
print("=" * 70)
print("7. TIME OF DAY VULNERABILITY — when are you most distractible?")
print("=" * 70)

# Bug #4 fix: normalize by active session-minutes per hour, not entry counts.
# Compute how many session-minutes fall in each clock hour by checking which
# sessions span each hour.
disruptions_timed = dist[dist["type"].isin(["d", "u"])].copy()
disruptions_timed["hour"] = disruptions_timed["clock_time"].dt.hour
hourly_dist = disruptions_timed.groupby("hour").size()

# Build active-minutes per hour from session spans
active_minutes_by_hour: dict[int, float] = defaultdict(float)
for _, sess in sess_df.iterrows():
    s_start: pd.Timestamp = sess["session_start"]
    s_end: pd.Timestamp = sess["session_end"]
    # Walk through each hour the session covers
    cursor = s_start.replace(minute=0, second=0, microsecond=0)
    while cursor < s_end:
        h = cursor.hour
        hour_end = cursor + pd.Timedelta(hours=1)
        overlap_start = max(cursor, s_start)
        overlap_end = min(hour_end, s_end)
        mins = (overlap_end - overlap_start).total_seconds() / 60
        if mins > 0:
            active_minutes_by_hour[h] += mins
        cursor = hour_end

print(f"\n  {'Hour':>6s} {'Distractions':>13s} {'Active min':>11s} {'Rate/hr':>8s}")
for h in range(6, 24):
    d_ct = hourly_dist.get(h, 0)
    a_min = active_minutes_by_hour.get(h, 0)
    rate = d_ct / (a_min / 60) if a_min > 0 else 0  # distractions per active hour
    bar = "█" * int(rate * 5)
    print(f"  {h:4d}:00 {d_ct:13d} {a_min:11.0f} {rate:8.2f}/h {bar}")

# Build hourly_rate series for the plot
hourly_rate = pd.Series(
    {
        h: hourly_dist.get(h, 0) / (active_minutes_by_hour.get(h, 1) / 60)
        for h in range(24)
    }
)

# Which hours have highest distraction duration?
dist_timed_len = dist_with_len.copy()
dist_timed_len["hour"] = dist_timed_len["clock_time"].dt.hour
hourly_dur = dist_timed_len.groupby("hour")["length_minutes"].mean()

print("\n  Average distraction duration by hour:")
print(f"  {'Hour':>6s} {'Mean dur (min)':>15s}")
for h in range(6, 24):
    if h in hourly_dur.index:
        print(f"  {h:4d}:00 {hourly_dur[h]:15.1f}")

print()


# ============================================================================
# 8. TASK SWITCHING — what tasks precede distractions?
# ============================================================================
print("=" * 70)
print("8. TASK CONTEXT — what were you doing when distracted?")
print("=" * 70)

# For each distraction, find the most recent 't' (task) entry
ev_sorted = ev_df.sort_values(["work_increment", "hours_into_session"]).copy()

task_before_dist: list[dict] = []
for wi, grp in ev_sorted.groupby("work_increment"):
    grp = grp.reset_index(drop=True)
    current_task = None
    for _, row in grp.iterrows():
        if row["type"] == "t":
            current_task = (
                str(row["comment"]).lower().strip()
                if pd.notna(row["comment"])
                else "unknown"
            )
        elif row["type"] in ("d", "u") and current_task:
            task_before_dist.append(
                {
                    "task": current_task[:40],
                    "type": row["type"],
                    "length_minutes": row["length_minutes"],
                    "hours_into_session": row["hours_into_session"],
                }
            )

if task_before_dist:
    task_df = pd.DataFrame(task_before_dist)

    # Bug #7 fix: use time-on-task (hours) instead of occurrence count for rate.
    # Estimate time on each task stint: time from task entry to next event.
    task_times: list[dict] = []
    for wi, grp in ev_sorted.groupby("work_increment"):
        grp = grp.sort_values("hours_into_session").reset_index(drop=True)
        for i, row in grp.iterrows():
            if row["type"] == "t":
                # Time until next event in same session
                remaining = grp.loc[grp.index > i]
                if not remaining.empty:
                    dur_h = (
                        remaining.iloc[0]["hours_into_session"]
                        - row["hours_into_session"]
                    )
                else:
                    dur_h = row["session_duration_h"] - row["hours_into_session"]
                task_name = (
                    str(row["comment"]).lower().strip()[:40]
                    if pd.notna(row["comment"])
                    else "unknown"
                )
                task_times.append({"task": task_name, "time_h": max(dur_h, 0)})

    task_time_df = pd.DataFrame(task_times)
    task_hours = task_time_df.groupby("task")["time_h"].sum().rename("total_hours")

    task_stats = (
        task_df.groupby("task")
        .agg(
            n_distractions=("type", "size"),
            mean_dist_len=("length_minutes", "mean"),
            total_dist_min=("length_minutes", "sum"),
        )
        .sort_values("n_distractions", ascending=False)
    )

    all_tasks = ev_sorted[ev_sorted["type"] == "t"].copy()
    all_tasks["task"] = (
        all_tasks["comment"].fillna("unknown").str.lower().str.strip().str[:40]
    )
    task_occurrences = all_tasks.groupby("task").size().rename("n_occurrences")
    task_stats = task_stats.merge(
        task_occurrences, left_index=True, right_index=True, how="left"
    )
    task_stats = task_stats.merge(
        task_hours, left_index=True, right_index=True, how="left"
    )
    task_stats["dist_per_hour"] = task_stats["n_distractions"] / task_stats[
        "total_hours"
    ].replace(0, np.nan)
    task_stats["dist_per_occurrence"] = (
        task_stats["n_distractions"] / task_stats["n_occurrences"]
    )

    print("\n  Tasks with most distractions (min 5):")
    print(f"  {'Task':40s} {'Dists':>6s} {'Hours':>7s} {'Rate/h':>7s} {'Avg min':>8s}")
    shown = task_stats[task_stats["n_distractions"] >= 5].sort_values(
        "n_distractions", ascending=False
    )
    for name, row in shown.head(25).iterrows():
        print(
            f"  {str(name):40s} {int(row['n_distractions']):6d} "
            f"{row['total_hours']:7.1f} {row['dist_per_hour']:7.2f} "
            f"{row['mean_dist_len']:8.1f}"
        )

    # Tasks with highest rate per hour (min 2h total time)
    high_rate = task_stats[task_stats["total_hours"] >= 2].sort_values(
        "dist_per_hour", ascending=False
    )
    if not high_rate.empty:
        print("\n  Tasks with highest distraction RATE per hour (min 2h on task):")
        print(f"  {'Task':40s} {'Rate/h':>7s} {'Dists':>6s} {'Hours':>7s}")
        for name, row in high_rate.head(15).iterrows():
            print(
                f"  {str(name):40s} {row['dist_per_hour']:7.2f} "
                f"{int(row['n_distractions']):6d} {row['total_hours']:7.1f}"
            )

print()


# ============================================================================
# 9. SUPPLEMENTS × SESSION-LEVEL FOCUS
# ============================================================================
print("=" * 70)
print("9. SUPPLEMENTS × SESSION-LEVEL FOCUS")
print("   (do supplements change within-session distraction patterns?)")
print("=" * 70)

# Parse supplement info from session start comments
supp_data: list[dict] = []
for _, sess in sess_df.iterrows():
    parsed = parse_supplement_dict(str(sess["start_comment"]))
    row_data: dict[str, float] = {
        "work_increment": sess["work_increment"],
        "disruption_rate": sess["disruption_rate"],
        "pct_distracted": sess["pct_distracted"],
        "duration_h": sess["duration_h"],
        "total_disruptions": sess["total_disruptions"],
    }
    for key, val in parsed.items():
        canonical = SUPPLEMENT_ALIASES.get(key, key)
        if key not in NON_SUPPLEMENT_KEYS:
            row_data[f"has_{canonical}"] = 1.0
        elif key == "s":
            row_data["session_sleep"] = val
        elif key in ("med",):
            row_data["session_meditation"] = val
        elif key == "caf":
            row_data["has_caffeine"] = 1.0
            row_data["caffeine_dose"] = val
    supp_data.append(row_data)

supp_sess = pd.DataFrame(supp_data)
# Fill missing supplement flags with 0
supp_cols = [c for c in supp_sess.columns if c.startswith("has_")]
supp_sess[supp_cols] = supp_sess[supp_cols].fillna(0)

# Bug #9 fix: raw Mann-Whitney is confounded by regime, day-of-week, session
# duration, etc. Use OLS controlling for session duration + daily controls.
# Merge daily controls onto the session-level supplement data.
supp_sess = supp_sess.merge(
    sess_df[["work_increment", "date", "duration_h"]].rename(
        columns={"duration_h": "_sess_dur"}
    ),
    on="work_increment",
    how="left",
)
supp_sess = supp_sess.merge(
    df[["date"] + BASE_CONTROLS],
    on="date",
    how="left",
)
SUPP_CONTROLS = BASE_CONTROLS + ["_sess_dur"]

print(f"\n  Sessions with parsed supplement data: {len(supp_sess)}")
print(
    f"\n  {'Supplement':25s} {'Sessions':>9s} {'Raw diff':>9s} "
    f"{'Ctrl coef':>10s} {'p (ctrl)':>9s}"
)
print("  " + "-" * 70)

for col in sorted(supp_cols):
    n_with = int(supp_sess[col].sum())
    if n_with < 15:
        continue
    with_supp = supp_sess[supp_sess[col] == 1]["disruption_rate"]
    without_supp = supp_sess[supp_sess[col] == 0]["disruption_rate"]
    if len(with_supp) > 10 and len(without_supp) > 10:
        raw_diff = with_supp.mean() - without_supp.mean()
        # Controlled regression
        ctrl_result = controlled_effect(
            supp_sess,
            col,
            outcome="disruption_rate",
            extra_controls=["_sess_dur"],
        )
        sig = (
            "***"
            if ctrl_result["p"] < 0.001
            else (
                "**"
                if ctrl_result["p"] < 0.01
                else "*" if ctrl_result["p"] < 0.05 else ""
            )
        )
        name = col.replace("has_", "")
        print(
            f"  {name:25s} {n_with:9d} {raw_diff:+9.2f} "
            f"{ctrl_result['coef']:+10.3f} {ctrl_result['p']:9.3f} {sig}"
        )

# Caffeine dose-response on session focus
if "caffeine_dose" in supp_sess.columns:
    caf = supp_sess[supp_sess["caffeine_dose"] > 0].copy()
    if len(caf) > 20:
        print(f"\n  Caffeine dose-response on disruption rate (n={len(caf)}):")
        for lo, hi, label in [
            (1, 50, "Low (<50mg)"),
            (50, 100, "Med (50-100mg)"),
            (100, 200, "High (100-200mg)"),
            (200, 1000, "Very high (200+mg)"),
        ]:
            sub = caf[(caf["caffeine_dose"] >= lo) & (caf["caffeine_dose"] < hi)]
            if len(sub) > 5:
                print(
                    f"    {label:25s}: rate={sub['disruption_rate'].mean():.2f}/h, "
                    f"n={len(sub)}"
                )

print()


# ============================================================================
# 10. INTER-SESSION RECOVERY — does break length matter?
# ============================================================================
print("=" * 70)
print("10. INTER-SESSION RECOVERY — does break length between sessions matter?")
print("=" * 70)

# For multi-session days, compute gap between sessions
multi_sess = sess_df.sort_values(["date", "session_start"]).copy()
gaps: list[dict] = []

for date, day_sessions in multi_sess.groupby("date"):
    if len(day_sessions) < 2:
        continue
    day_sorted = day_sessions.sort_values("session_start").reset_index(drop=True)
    for i in range(1, len(day_sorted)):
        prev_end = day_sorted.iloc[i - 1]["session_end"]
        curr_start = day_sorted.iloc[i]["session_start"]
        gap_h = (curr_start - prev_end).total_seconds() / 3600
        if gap_h < 0 or gap_h > 8:
            continue
        gaps.append(
            {
                "date": date,
                "gap_hours": gap_h,
                "prev_session_disruption_rate": day_sorted.iloc[i - 1][
                    "disruption_rate"
                ],
                "next_session_disruption_rate": day_sorted.iloc[i]["disruption_rate"],
                "prev_session_duration": day_sorted.iloc[i - 1]["duration_h"],
                "next_session_duration": day_sorted.iloc[i]["duration_h"],
                "prev_pct_distracted": day_sorted.iloc[i - 1]["pct_distracted"],
                "next_pct_distracted": day_sorted.iloc[i]["pct_distracted"],
            }
        )

if gaps:
    gap_df = pd.DataFrame(gaps)
    print(f"\n  Inter-session gaps analyzed: {len(gap_df)}")
    print(
        f"  Mean gap: {gap_df['gap_hours'].mean():.2f}h, "
        f"median: {gap_df['gap_hours'].median():.2f}h"
    )

    # Does longer break improve next session?
    print("\n  Next session disruption rate by break length:")
    for lo, hi, label in [
        (0, 0.25, "<15 min"),
        (0.25, 0.5, "15-30 min"),
        (0.5, 1, "30-60 min"),
        (1, 2, "1-2 hours"),
        (2, 4, "2-4 hours"),
        (4, 8, "4-8 hours"),
    ]:
        sub = gap_df[(gap_df["gap_hours"] >= lo) & (gap_df["gap_hours"] < hi)]
        if len(sub) > 10:
            print(
                f"    {label:15s}: rate={sub['next_session_disruption_rate'].mean():.2f}/h, "
                f"n={len(sub)}"
            )

    # Correlation: gap length vs next session quality
    r, p = stats.pearsonr(gap_df["gap_hours"], gap_df["next_session_disruption_rate"])
    print(
        f"\n  Correlation: gap length ↔ next session disruption rate: r={r:+.3f}, p={p:.3f}"
    )

    # Does a bad session predict a bad next session?
    r2, p2 = stats.pearsonr(
        gap_df["prev_session_disruption_rate"],
        gap_df["next_session_disruption_rate"],
    )
    print(
        f"  Correlation: prev session rate ↔ next session rate: r={r2:+.3f}, p={p2:.3f}"
    )

print()


# ============================================================================
# 11. WORK INCREMENT TRENDS — do you improve over months?
# ============================================================================
print("=" * 70)
print("11. LONG-TERM TRENDS — are you getting better at focusing?")
print("=" * 70)

sess_df["year_month"] = sess_df["date"].dt.to_period("M")
monthly = (
    sess_df.groupby("year_month")
    .agg(
        mean_disruption_rate=("disruption_rate", "mean"),
        mean_pct_distracted=("pct_distracted", "mean"),
        mean_duration=("duration_h", "mean"),
        n_sessions=("duration_h", "count"),
    )
    .reset_index()
)

monthly = monthly[monthly["n_sessions"] >= 10]  # only months with enough data

print(
    f"\n  {'Month':>10s} {'Dist rate':>10s} {'% Distracted':>13s} {'Sess dur':>9s} {'n':>5s}"
)
for _, row in monthly.iterrows():
    print(
        f"  {str(row['year_month']):>10s} {row['mean_disruption_rate']:10.2f} "
        f"{row['mean_pct_distracted']:13.1%} {row['mean_duration']:9.2f}h "
        f"{int(row['n_sessions']):5d}"
    )

# Is there a significant trend?
monthly["month_num"] = range(len(monthly))
if len(monthly) > 5:
    r, p = stats.pearsonr(monthly["month_num"], monthly["mean_disruption_rate"])
    direction = "improving" if r < 0 else "worsening"
    print(f"\n  Trend in disruption rate: r={r:+.3f}, p={p:.3f} ({direction})")

    r2, p2 = stats.pearsonr(monthly["month_num"], monthly["mean_pct_distracted"])
    direction2 = "improving" if r2 < 0 else "worsening"
    print(f"  Trend in % time distracted: r={r2:+.3f}, p={p2:.3f} ({direction2})")

print()


# ============================================================================
# 12. PRODUCTIVE DEEP WORK BLOCKS — what predicts sustained focus?
# ============================================================================
print("=" * 70)
print("12. DEEP WORK — what predicts sustained, uninterrupted focus?")
print("=" * 70)

# Define "deep work blocks": stretches within a session with no d/u events
# For each session, find longest gap between distractions
deep_blocks: list[dict] = []

for wi, grp in ev_sorted.groupby("work_increment"):
    session_info = sess_df[sess_df["work_increment"] == wi]
    if session_info.empty:
        continue
    session_info = session_info.iloc[0]

    dist_times = grp[grp["type"].isin(["d", "u"])]["hours_into_session"].values
    duration = session_info["duration_h"]

    if len(dist_times) == 0:
        # Entire session was uninterrupted
        deep_blocks.append(
            {
                "work_increment": wi,
                "date": session_info["date"],
                "longest_focus_h": duration,
                "session_duration_h": duration,
                "n_distractions": 0,
                "start_hour": session_info["start_hour"],
            }
        )
    else:
        # Compute gaps: start→first_dist, between dists, last_dist→end
        boundaries = np.concatenate([[0], dist_times, [duration]])
        max_gap = np.max(np.diff(boundaries))
        deep_blocks.append(
            {
                "work_increment": wi,
                "date": session_info["date"],
                "longest_focus_h": max_gap,
                "session_duration_h": duration,
                "n_distractions": len(dist_times),
                "start_hour": session_info["start_hour"],
            }
        )

deep_df = pd.DataFrame(deep_blocks)

print(f"\n  Sessions analyzed: {len(deep_df)}")
print(
    f"  Mean longest focus block: {deep_df['longest_focus_h'].mean():.2f}h "
    f"({deep_df['longest_focus_h'].mean() * 60:.0f} min)"
)
print(
    f"  Median longest focus block: {deep_df['longest_focus_h'].median():.2f}h "
    f"({deep_df['longest_focus_h'].median() * 60:.0f} min)"
)

print("\n  Longest focus block distribution:")
for lo, hi, label in [
    (0, 15 / 60, "<15 min"),
    (15 / 60, 30 / 60, "15-30 min"),
    (30 / 60, 1, "30-60 min"),
    (1, 2, "1-2 hours"),
    (2, 3, "2-3 hours"),
    (3, 20, "3+ hours"),
]:
    n = ((deep_df["longest_focus_h"] >= lo) & (deep_df["longest_focus_h"] < hi)).sum()
    pct = n / len(deep_df) * 100
    bar = "█" * int(pct / 2)
    print(f"    {label:15s}: {n:5d} ({pct:5.1f}%) {bar}")

# Daily longest focus → daily output
daily_deep = (
    deep_df.groupby("date")
    .agg(
        max_focus_block=("longest_focus_h", "max"),
        mean_focus_block=("longest_focus_h", "mean"),
    )
    .reset_index()
)

daily_deep = daily_deep.merge(
    df[["date", "Hours Working", "Energy", "Focus", "Value"] + BASE_CONTROLS],
    on="date",
    how="left",
)

print("\n  --- Deep work → daily outcomes (controlled) ---")
r = controlled_effect(daily_deep, "max_focus_block")
print_effect("Longest focus block (max across sessions) → Hours", r)
r = controlled_effect(daily_deep, "mean_focus_block")
print_effect("Mean focus block length → Hours Working", r)
r = controlled_effect(daily_deep, "max_focus_block", outcome="Focus")
print_effect("Longest focus block → Focus rating", r, unit="pts")
r = controlled_effect(daily_deep, "max_focus_block", outcome="Value")
print_effect("Longest focus block → perceived Value", r, unit="pts")

# Morning sessions have longer deep work?
morning_deep = deep_df[deep_df["start_hour"] < 12]["longest_focus_h"]
afternoon_deep = deep_df[(deep_df["start_hour"] >= 12) & (deep_df["start_hour"] < 17)][
    "longest_focus_h"
]
evening_deep = deep_df[deep_df["start_hour"] >= 17]["longest_focus_h"]
print("\n  Deep work by time of day:")
print(
    f"    Morning (<12):    {morning_deep.mean():.2f}h ({morning_deep.mean()*60:.0f} min), n={len(morning_deep)}"
)
print(
    f"    Afternoon (12-5): {afternoon_deep.mean():.2f}h ({afternoon_deep.mean()*60:.0f} min), n={len(afternoon_deep)}"
)
print(
    f"    Evening (5+):     {evening_deep.mean():.2f}h ({evening_deep.mean()*60:.0f} min), n={len(evening_deep)}"
)

print()


# ============================================================================
# 13. "GETTING STARTED" PROBLEM — session 1 vs session 2
# ============================================================================
print("=" * 70)
print("13. GETTING STARTED — is the first session different?")
print("=" * 70)

# Tag sessions as 1st, 2nd, 3rd etc within each day
sess_ordered = sess_df.sort_values(["date", "session_start"]).copy()
sess_ordered["session_num"] = sess_ordered.groupby("date").cumcount() + 1

print(
    f"\n  {'Session #':>12s} {'n':>6s} {'Duration':>9s} {'Dist rate':>10s} "
    f"{'% Distracted':>13s} {'Start hour':>11s}"
)
for sn in range(1, 5):
    sub = sess_ordered[sess_ordered["session_num"] == sn]
    if len(sub) > 20:
        print(
            f"  {'#' + str(sn):>12s} {len(sub):6d} {sub['duration_h'].mean():9.2f}h "
            f"{sub['disruption_rate'].mean():10.2f}/h "
            f"{sub['pct_distracted'].mean():13.1%} "
            f"{sub['start_hour'].mean():11.1f}"
        )

# Does first session quality predict the rest of the day?
first_sessions = sess_ordered[sess_ordered["session_num"] == 1][
    ["date", "disruption_rate", "pct_distracted", "duration_h"]
].rename(
    columns={
        "disruption_rate": "first_sess_dist_rate",
        "pct_distracted": "first_sess_pct_dist",
        "duration_h": "first_sess_duration",
    }
)

later_sessions = (
    sess_ordered[sess_ordered["session_num"] > 1]
    .groupby("date")
    .agg(
        later_dist_rate=("disruption_rate", "mean"),
        later_pct_dist=("pct_distracted", "mean"),
        later_total_hours=("duration_h", "sum"),
    )
    .reset_index()
)

first_later = first_sessions.merge(later_sessions, on="date", how="inner")
if len(first_later) > 30:
    r, p = stats.pearsonr(
        first_later["first_sess_dist_rate"],
        first_later["later_dist_rate"],
    )
    print(
        f"\n  First session dist rate → later sessions dist rate: r={r:+.3f}, p={p:.3f}"
    )
    r2, p2 = stats.pearsonr(
        first_later["first_sess_duration"],
        first_later["later_total_hours"],
    )
    print(f"  First session duration → later total hours: r={r2:+.3f}, p={p2:.3f}")

print()


# ============================================================================
# 14. DAY-OF-WEEK × TIME-OF-DAY INTERACTION
# ============================================================================
print("=" * 70)
print("14. DAY-OF-WEEK × TIME-OF-DAY INTERACTION")
print("=" * 70)

sess_ordered["dow"] = sess_ordered["date"].dt.dayofweek
dow_names = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]

print(f"\n  {'':5s}", end="")
for period_label in ["Morning", "Afternoon", "Evening"]:
    print(f"  {period_label:>12s}", end="")
print()

for dow in range(7):
    dow_data = sess_ordered[sess_ordered["dow"] == dow]
    print(f"  {dow_names[dow]:5s}", end="")
    for lo, hi in [(0, 12), (12, 17), (17, 24)]:
        sub = dow_data[(dow_data["start_hour"] >= lo) & (dow_data["start_hour"] < hi)]
        if len(sub) > 5:
            print(f"  {sub['disruption_rate'].mean():8.2f}/h  ", end="")
        else:
            print(f"  {'n/a':>12s}", end="")
    print()

print()


# ============================================================================
# 15. DISTRACTION COMMENT PATTERNS — NLP-LITE
# ============================================================================
print("=" * 70)
print("15. DISTRACTION PATTERNS — what words appear in distraction comments?")
print("=" * 70)

dist_comments = dist[dist["type"].isin(["d", "u"])].copy()
dist_comments["comment_clean"] = (
    dist_comments["comment"].fillna("").str.lower().str.strip()
)
# Filter out single-letter codes
long_comments = dist_comments[dist_comments["comment_clean"].str.len() > 2]

if not long_comments.empty:
    # Word frequency across all distraction comments
    all_words: dict[str, int] = defaultdict(int)
    for comment in long_comments["comment_clean"]:
        for word in re.split(r"\W+", comment):
            if len(word) > 2:
                all_words[word] += 1

    sorted_words = sorted(all_words.items(), key=lambda x: x[1], reverse=True)
    print("\n  Top 30 words in distraction comments:")
    for word, count in sorted_words[:30]:
        print(f"    {word:20s}: {count:5d}")

    # Categorize distractions by theme
    themes = {
        "social_media": ["twitter", "insta", "instagram", "reddit", "hacker", "hn"],
        "food_drink": [
            "lunch",
            "food",
            "eat",
            "snack",
            "drink",
            "coffee",
            "water",
            "breakfast",
            "dinner",
        ],
        "communication": ["chat", "text", "call", "phone", "slack", "message", "email"],
        "reading": ["blog", "read", "article", "news", "book"],
        "body": ["bathroom", "stretch", "walk", "gym"],
        "entertainment": ["youtube", "video", "music", "spotify", "game", "porn"],
        "planning": ["plan", "think", "organize", "calendar", "schedule", "todo"],
    }

    print("\n  Distraction themes (from comments with >2 chars):")
    print(f"  {'Theme':20s} {'Count':>6s} {'Mean dur (min)':>15s}")
    for theme_name, keywords in themes.items():
        pattern = "|".join(keywords)
        matches = long_comments[
            long_comments["comment_clean"].str.contains(pattern, na=False)
        ]
        if len(matches) > 5:
            mean_dur = matches["length_minutes"].mean()
            print(f"  {theme_name:20s} {len(matches):6d} {mean_dur:15.1f}")

print()


# ============================================================================
# 16. OPTIMAL SESSION LENGTH — diminishing returns?
# ============================================================================
print("=" * 70)
print("16. OPTIMAL SESSION LENGTH — diminishing returns?")
print("=" * 70)

# For each session, compute "effective work hours" = duration - distraction time
sess_df["effective_hours"] = (
    sess_df["duration_h"] - sess_df["distraction_minutes"] / 60
).clip(lower=0)
sess_df["efficiency"] = (sess_df["effective_hours"] / sess_df["duration_h"]).clip(0, 1)

print("\n  Session efficiency by planned duration:")
print(
    f"  {'Duration':15s} {'n':>5s} {'Effective h':>12s} {'Efficiency':>11s} {'Dist rate':>10s}"
)
for lo, hi, label in [
    (0, 1, "<1h"),
    (1, 2, "1-2h"),
    (2, 3, "2-3h"),
    (3, 4, "3-4h"),
    (4, 5, "4-5h"),
    (5, 6, "5-6h"),
    (6, 8, "6-8h"),
    (8, 16, "8+h"),
]:
    sub = sess_df[(sess_df["duration_h"] >= lo) & (sess_df["duration_h"] < hi)]
    if len(sub) > 10:
        print(
            f"  {label:15s} {len(sub):5d} {sub['effective_hours'].mean():12.2f} "
            f"{sub['efficiency'].mean():11.1%} {sub['disruption_rate'].mean():10.2f}"
        )

# Bug #8 fix: marginal returns.
# The old approach compared different-length sessions (selection bias: people who
# work 8h sessions are different from those who work 2h sessions).
# Better: within each session, compute the efficiency of each hour-block, so we
# see whether the 4th hour of a long session is as productive as the 1st hour
# of that same session.
print("\n  Within-session efficiency by hour-block (uses only sessions long enough):")
print(f"  {'Hour block':>12s} {'Eff work min':>13s} {'% of 60':>8s} {'Sessions':>9s}")
for h in range(0, 10):
    # Only sessions that lasted at least to the end of this hour block
    long_enough = sess_df[sess_df["duration_h"] > h + 1]
    if len(long_enough) < 20:
        continue
    # Get events in this hour-block of each session
    block_events = ev_df[
        (ev_df["work_increment"].isin(long_enough["work_increment"]))
        & (ev_df["hours_into_session"] >= h)
        & (ev_df["hours_into_session"] < h + 1)
    ]
    dist_in_block = block_events[block_events["type"].isin(["d", "u", "i"])]
    dist_min = dist_in_block.groupby("work_increment")["length_minutes"].sum()
    # Fill sessions with 0 distraction minutes in this block
    all_wi = long_enough["work_increment"]
    dist_min = dist_min.reindex(all_wi, fill_value=0)
    eff_min = 60 - dist_min.clip(upper=60)
    print(
        f"  {f'Hour {h+1}':>12s} {eff_min.mean():13.1f} {eff_min.mean()/60:8.1%} "
        f"{len(long_enough):9d}"
    )

print()


# ============================================================================
# 17. INTERRUPT vs DISTRACTION — planned breaks vs unplanned
# ============================================================================
print("=" * 70)
print("17. INTERRUPTS vs DISTRACTIONS — planned vs unplanned breaks")
print("=" * 70)

for t_type, label in [
    ("d", "Distraction (got up)"),
    ("u", "Unfocused (at desk)"),
    ("i", "Interrupt (planned)"),
    ("c", "Consume (food/supp)"),
]:
    entries = dist[dist["type"] == t_type]
    if len(entries) > 10:
        print(
            f"  {label:30s}: n={len(entries):5d}, "
            f"mean={entries['length_minutes'].mean():.1f}min, "
            f"median={entries['length_minutes'].median():.1f}min"
        )

# Does having more planned interrupts (i) reduce unplanned distractions (d/u)?
daily_types = (
    dist[dist["type"].isin(["d", "u", "i", "c"])]
    .groupby(["date", "type"])
    .size()
    .unstack(fill_value=0)
)

for col in ["d", "u", "i", "c"]:
    if col not in daily_types.columns:
        daily_types[col] = 0

daily_types = daily_types.reset_index()
daily_types["unplanned"] = daily_types["d"] + daily_types["u"]
daily_types["planned"] = daily_types["i"] + daily_types["c"]

daily_types = daily_types.merge(
    df[["date", "Hours Working", "Energy", "Focus"] + BASE_CONTROLS],
    on="date",
    how="left",
)

if len(daily_types) > 30:
    r, p = stats.pearsonr(daily_types["planned"], daily_types["unplanned"])
    print(f"\n  Planned breaks ↔ unplanned distractions: r={r:+.3f}, p={p:.3f}")

    r = controlled_effect(daily_types, "planned")
    print_effect("Planned breaks → Hours Working", r)
    r = controlled_effect(daily_types, "unplanned")
    print_effect("Unplanned distractions → Hours Working", r)

    # Ratio of planned to unplanned
    daily_types["planned_ratio"] = daily_types["planned"] / (
        daily_types["planned"] + daily_types["unplanned"]
    ).replace(0, np.nan)
    r = controlled_effect(daily_types, "planned_ratio")
    print_effect("Planned break ratio → Hours Working", r)

print()


# ============================================================================
# 18. GENERATE SUMMARY PLOTS
# ============================================================================
print("=" * 70)
print("18. GENERATING SUMMARY PLOTS")
print("=" * 70)

fig, axes = plt.subplots(2, 3, figsize=(18, 10))

# 1. Distraction rate over session lifetime
ax = axes[0, 0]
pct_bins = range(10)
rates = [pct_rate.get(b, 0) for b in pct_bins]
ax.bar([b * 10 + 5 for b in pct_bins], rates, width=8, color="steelblue", alpha=0.7)
ax.set_xlabel("% through session")
ax.set_ylabel("Distraction rate")
ax.set_title("Distraction Rate Over Session Lifetime")

# 2. First distraction timing histogram
ax = axes[0, 1]
ax.hist(
    first_dist_all["first_distraction_h"] * 60, bins=30, edgecolor="black", alpha=0.7
)
ax.axvline(
    first_dist_all["first_distraction_h"].median() * 60,
    color="red",
    linestyle="--",
    label=f"Median: {first_dist_all['first_distraction_h'].median()*60:.0f}min",
)
ax.set_xlabel("Minutes to first distraction")
ax.set_ylabel("Count")
ax.set_title("Time to First Distraction")
ax.legend()

# 3. Hour of day distraction rate
ax = axes[0, 2]
hours = range(6, 24)
rates_by_hour = [hourly_rate.get(h, 0) for h in hours]
ax.bar(hours, rates_by_hour, color="coral", alpha=0.7)
ax.set_xlabel("Hour of day")
ax.set_ylabel("Distraction rate")
ax.set_title("Distractibility by Hour")

# 4. Session efficiency by duration
ax = axes[1, 0]
dur_bins = [(0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (5, 6), (6, 8), (8, 16)]
effs = []
labels = []
for lo, hi in dur_bins:
    sub = sess_df[(sess_df["duration_h"] >= lo) & (sess_df["duration_h"] < hi)]
    if len(sub) > 10:
        effs.append(sub["efficiency"].mean())
        labels.append(f"{lo}-{hi}h")
ax.bar(range(len(effs)), effs, tick_label=labels, color="seagreen", alpha=0.7)
ax.set_ylabel("Session efficiency")
ax.set_title("Efficiency by Session Length")
ax.set_ylim(0, 1)

# 5. Cascade effect
ax = axes[1, 1]
if cascade_gaps:
    cascade_df = pd.DataFrame(cascade_gaps)
    cascade_by_num = cascade_df.groupby("distraction_number")["gap_minutes"].mean()
    valid_nums = cascade_by_num[cascade_by_num.index <= 8]
    ax.bar(valid_nums.index, valid_nums.values, color="orchid", alpha=0.7)
    ax.set_xlabel("Distraction #")
    ax.set_ylabel("Mean gap to next (min)")
    ax.set_title("Cascade Effect: Gaps Between Distractions")

# 6. Monthly trend in disruption rate
ax = axes[1, 2]
if len(monthly) > 3:
    ax.plot(
        range(len(monthly)),
        monthly["mean_disruption_rate"].values,
        marker="o",
        color="navy",
    )
    ax.set_xticks(range(0, len(monthly), max(1, len(monthly) // 8)))
    ax.set_xticklabels(
        [
            str(monthly.iloc[i]["year_month"])
            for i in range(0, len(monthly), max(1, len(monthly) // 8))
        ],
        rotation=45,
    )
    ax.set_ylabel("Mean disruption rate/h")
    ax.set_title("Focus Trend Over Time")

plt.suptitle("Micro-Level Distraction Analysis", fontsize=16, y=1.02)
plt.tight_layout()
plt.savefig(
    "Self_Tracking/ai_calendar/micro_analysis_summary.png", dpi=150, bbox_inches="tight"
)
plt.close()
print("  Saved: Self_Tracking/ai_calendar/micro_analysis_summary.png")

print()
print("=" * 70)
print("ALL DONE — MICRO ANALYSIS COMPLETE")
print("=" * 70)
