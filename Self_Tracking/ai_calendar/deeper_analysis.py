# %% use conda env: side_projects
"""
Deeper analysis: addresses the unexplored leads from CONTINUATION_SPEC.md.

Covers:
1. Supplement effects on distraction rate (within-session)
2. No-diminishing-returns check beyond 10h
3. Supplement interaction effects (formal interaction terms)
4. Calendar event-level prediction (specific events, not just categories)
5. Task-level text mining (Tasks Summary, Reflections, Productivity Notes)
6. Comprehensive regularized model with holdout validation
7. Actionable morning model vs descriptive end-of-day model (clean separation)

Generates: DEEPER_ANALYSIS_SUMMARY.md
"""
import re
import warnings
from collections import Counter, defaultdict
from io import StringIO
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import ElasticNetCV, LassoCV, RidgeCV
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler

from productivity_analysis import (
    CALENDAR_DIR,
    DAILY_SUMMARY_CSV,
    DISTRACTED_CSV,
    REPO_DIR,
    SUPPLEMENT_ALIASES,
    build_analysis_df,
    extract_daily_supplements,
    extract_work_start_time,
    load_calendar_full,
    load_calendar_sleep,
    load_daily_summary_full,
    load_distracted_stacked,
    parse_supplement_dict,
)

warnings.filterwarnings("ignore", category=FutureWarning)

# Capture all output for summary file
output_buffer = StringIO()


def tee_print(*args: object, **kwargs: object) -> None:
    print(*args, **kwargs)
    print(*args, **kwargs, file=output_buffer)


# ============================================================================
# DATA LOADING
# ============================================================================
tee_print("Loading data...")
daily = load_daily_summary_full(DAILY_SUMMARY_CSV)
distracted = load_distracted_stacked(DISTRACTED_CSV)
supplements = extract_daily_supplements(distracted)
work_starts = extract_work_start_time(distracted)
cal_sleep = load_calendar_sleep(CALENDAR_DIR)
cal_full = load_calendar_full(CALENDAR_DIR)

df = build_analysis_df(daily, supplements, cal_sleep, distracted)

# Add calendar category totals
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
    "youtube",
    "insta",
    "anki",
    "meditation",
    "drink",
]
for ev in key_events:
    col = f"cal_{ev}"
    if ev in event_daily.columns:
        df = df.merge(
            event_daily[[ev]].rename(columns={ev: col}),
            left_on="date",
            right_index=True,
            how="left",
        )
        df[col] = df[col].fillna(0)

# Work sessions per day
n_sessions = distracted[distracted["type"] == "s"].groupby("date").size()
n_sessions.name = "n_sessions"
df = df.merge(n_sessions, left_on="date", right_index=True, how="left")
df["n_sessions"] = df["n_sessions"].fillna(1)

valid = df[df["Hours Working"] > 0].copy()

BASE_CONTROLS = ["is_hive", "is_mats", "is_diesl", "is_weekend", "day_of_week"]

tee_print(
    f"Loaded: {len(df)} daily rows, {len(cal_full)} calendar events, "
    f"{len(distracted)} distraction entries"
)
tee_print()


def controlled_effect(
    data: pd.DataFrame,
    treatment_col: str,
    outcome: str = "Hours Working",
    extra_controls: list[str] | None = None,
) -> dict[str, float | int | object]:
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
    }


# ============================================================================
# SHARED: Parse all sessions and individual distraction events with timestamps
# ============================================================================
def _build_session_and_event_data() -> (
    tuple[pd.DataFrame, pd.DataFrame, dict[pd.Timestamp, list[dict[str, object]]]]
):
    """Parse distracted CSV into sessions, individual timed events, and
    per-day supplement consumption times.

    Returns (sessions_df, distraction_events_df, supp_times_by_day).
    - sessions_df: one row per session (date, start, end, duration, supplement flags)
    - distraction_events_df: one row per distraction/unfocused event with
      minutes_into_session and session-level supplement flags
    - supp_times_by_day: {date: [{supplement, time_parsed, dose}, ...]}
    """
    dist = distracted.copy()
    dist["time_parsed"] = pd.to_datetime(dist["time"], format="mixed", errors="coerce")

    # --- Collect supplement consumption times (not just daily totals) ---
    supp_times_by_day: dict[pd.Timestamp, list[dict[str, object]]] = defaultdict(list)
    daily_supps: dict[pd.Timestamp, dict[str, float]] = {}
    supp_entries = dist[dist["type"].isin(["s", "c"])].copy()
    for _, row in supp_entries.iterrows():
        parsed = parse_supplement_dict(row["comment"])
        if not parsed:
            continue
        d = row["date"]
        if d not in daily_supps:
            daily_supps[d] = {}
        for k, v in parsed.items():
            if k in SUPPLEMENT_ALIASES:
                canon = SUPPLEMENT_ALIASES[k]
                daily_supps[d][canon] = daily_supps[d].get(canon, 0) + v
                supp_times_by_day[d].append(
                    {"supplement": canon, "time_parsed": row["time_parsed"], "dose": v}
                )

    # --- Build sessions + individual distraction events ---
    session_rows: list[dict[str, object]] = []
    event_rows: list[dict[str, object]] = []

    for date, day_data in dist.groupby("date"):
        day_sorted = day_data.sort_values("time_parsed")
        starts = day_sorted[day_sorted["type"] == "s"]
        ends = day_sorted[day_sorted["type"] == "e"]
        supp_data = daily_supps.get(date, {})

        cum_work_min = 0.0
        for _, start_row in starts.iterrows():
            session_start = start_row["time_parsed"]
            matching_ends = ends[ends["time_parsed"] > session_start]
            if matching_ends.empty:
                continue
            session_end = matching_ends.iloc[0]["time_parsed"]
            duration_min = (session_end - session_start).total_seconds() / 60
            if duration_min <= 0 or duration_min > 16 * 60:
                continue

            session_events = day_sorted[
                (day_sorted["time_parsed"] > session_start)
                & (day_sorted["time_parsed"] < session_end)
            ]
            distractions = session_events[session_events["type"].isin(["d", "u"])]
            total_dist_min = distractions["length_minutes"].sum()

            supp_names = ["caffeine", "adderall", "modafinil", "nicotine"]
            supp_flags = {f"had_{s}": int(supp_data.get(s, 0) > 0) for s in supp_names}
            supp_doses = {f"dose_{s}": supp_data.get(s, 0.0) for s in supp_names}

            regime = start_row.get("for", "")
            session_rows.append(
                {
                    "date": date,
                    "regime": regime,
                    "session_start": session_start,
                    "session_end": session_end,
                    "duration_min": duration_min,
                    "cum_work_min_before": cum_work_min,
                    "n_distractions": len(distractions),
                    "total_dist_min": total_dist_min,
                    **supp_flags,
                    **supp_doses,
                }
            )

            for _, d_row in distractions.iterrows():
                min_into = (d_row["time_parsed"] - session_start).total_seconds() / 60
                event_rows.append(
                    {
                        "date": date,
                        "regime": regime,
                        "session_start": session_start,
                        "session_duration_min": duration_min,
                        "cum_work_min_before": cum_work_min,
                        "min_into_session": min_into,
                        "min_into_day": cum_work_min + min_into,
                        "dist_type": d_row["type"],
                        "length_minutes": d_row["length_minutes"],
                        **supp_flags,
                    }
                )

            cum_work_min += duration_min

    sess_df = pd.DataFrame(session_rows)
    ev_df = pd.DataFrame(event_rows)
    return sess_df, ev_df, supp_times_by_day


# ============================================================================
# ANALYSIS 1: SUPPLEMENT EFFECTS ON DISTRACTION RATE (TIME-COURSE)
# ============================================================================
def supplement_distraction_analysis() -> None:
    tee_print("=" * 70)
    tee_print("ANALYSIS 1: DO SUPPLEMENTS REDUCE DISTRACTION RATE WITHIN SESSIONS?")
    tee_print("=" * 70)

    sess_df, ev_df, supp_times = _build_session_and_event_data()

    if sess_df.empty:
        tee_print("  No sessions found.")
        return

    tee_print(f"\n  Sessions: {len(sess_df)}, Distraction events: {len(ev_df)}")

    # --- 1a: Daily session-level comparison (with vs without) ---
    # Focus % = (1 - distraction_minutes / session_minutes) * 100
    sess_df["focus_pct"] = (
        (1 - sess_df["total_dist_min"] / sess_df["duration_min"]) * 100
    ).clip(0, 100)

    # "Any stimulant" flag — the right comparison when you almost always take *something*
    stim_cols = [
        c
        for c in ["had_caffeine", "had_adderall", "had_modafinil", "had_nicotine"]
        if c in sess_df.columns
    ]
    sess_df["had_any_stimulant"] = (sess_df[stim_cols].sum(axis=1) > 0).astype(int)
    ev_df["had_any_stimulant"] = (ev_df[stim_cols].sum(axis=1) > 0).astype(int)

    tee_print("\n  --- Session-level % time focused WITH vs WITHOUT ---")
    tee_print(
        f"  {'Supplement':15s} {'With %':>8s} {'W/o %':>8s} "
        f"{'Diff':>7s} {'p':>8s} {'n_w':>5s} {'n_wo':>5s}"
    )
    all_supps = ["caffeine", "adderall", "modafinil", "nicotine", "any_stimulant"]
    for supp in all_supps:
        col = f"had_{supp}"
        if col not in sess_df.columns:
            continue
        with_s = sess_df[sess_df[col] == 1]["focus_pct"]
        without_s = sess_df[sess_df[col] == 0]["focus_pct"]
        if len(with_s) < 10 or len(without_s) < 10:
            continue
        diff = with_s.mean() - without_s.mean()
        _, p = stats.mannwhitneyu(with_s, without_s, alternative="two-sided")
        sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else ""
        tee_print(
            f"  {supp:15s} {with_s.mean():8.1f} {without_s.mean():8.1f} "
            f"{diff:+7.1f} {p:8.4f} {len(with_s):5d} {len(without_s):5d} {sig}"
        )

    # Weekday-only "any stimulant" comparison (fairer — removes weekend noise)
    wkday = sess_df[sess_df["date"].dt.dayofweek < 5]
    with_wk = wkday[wkday["had_any_stimulant"] == 1]["focus_pct"]
    without_wk = wkday[wkday["had_any_stimulant"] == 0]["focus_pct"]
    if len(with_wk) >= 10 and len(without_wk) >= 10:
        _, p_wk = stats.mannwhitneyu(with_wk, without_wk, alternative="two-sided")
        tee_print(
            f"  {'any_stim(wkday)':15s} {with_wk.mean():8.1f} {without_wk.mean():8.1f} "
            f"{with_wk.mean() - without_wk.mean():+7.1f} {p_wk:8.4f} "
            f"{len(with_wk):5d} {len(without_wk):5d}"
        )
    else:
        tee_print(
            f"  any_stim(wkday): insufficient no-stimulant weekday sessions "
            f"(n={len(without_wk)})"
        )

    # Cross-tab: how often is "no caffeine" really "has adderall instead"?
    tee_print(
        "\n  --- Stimulant overlap (are 'no caffeine' days really 'adderall' days?) ---"
    )
    for a, b in [
        ("caffeine", "adderall"),
        ("caffeine", "modafinil"),
        ("adderall", "modafinil"),
    ]:
        ca, cb = f"had_{a}", f"had_{b}"
        if ca not in sess_df.columns or cb not in sess_df.columns:
            continue
        neither = ((sess_df[ca] == 0) & (sess_df[cb] == 0)).sum()
        a_only = ((sess_df[ca] == 1) & (sess_df[cb] == 0)).sum()
        b_only = ((sess_df[ca] == 0) & (sess_df[cb] == 1)).sum()
        both = ((sess_df[ca] == 1) & (sess_df[cb] == 1)).sum()
        tee_print(
            f"  {a} x {b}: neither={neither}, {a}-only={a_only}, "
            f"{b}-only={b_only}, both={both}"
        )

    # --- 1b: Time-course — % time focused in 30-min windows since session start,
    #     split by supplement status ---
    tee_print("\n  --- % time focused by 30-min window into session ---")

    bin_width = 30  # minutes
    max_min = 360  # 6 hours
    bin_edges = np.arange(0, max_min + bin_width, bin_width)
    bin_labels = [f"{int(e)}-{int(e + bin_width)}" for e in bin_edges[:-1]]

    def compute_focus_pct(sessions: pd.DataFrame, events: pd.DataFrame) -> pd.DataFrame:
        """For each time bin, compute % of time focused (not distracted)."""
        rows = []
        for left, right, label in zip(bin_edges[:-1], bin_edges[1:], bin_labels):
            clipped_end = np.minimum(sessions["duration_min"].values, right)
            exposure_min = np.maximum(clipped_end - left, 0).sum()

            # Sum distraction *minutes* for events starting in [left, right)
            bin_mask = (events["min_into_session"] >= left) & (
                events["min_into_session"] < right
            )
            dist_min = events.loc[bin_mask, "length_minutes"].sum()

            focus_pct = (
                (1 - dist_min / exposure_min) * 100 if exposure_min > 0 else np.nan
            )
            rows.append(
                {
                    "bin": label,
                    "bin_center_min": (left + right) / 2,
                    "exposure_min": exposure_min,
                    "dist_min": dist_min,
                    "focus_pct": focus_pct,
                }
            )
        return pd.DataFrame(rows)

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    for ax, supp in zip(axes.flat, ["caffeine", "adderall", "modafinil", "nicotine"]):
        col = f"had_{supp}"
        if col not in sess_df.columns or col not in ev_df.columns:
            ax.set_title(f"{supp}: no data")
            continue

        with_sess = sess_df[sess_df[col] == 1]
        without_sess = sess_df[sess_df[col] == 0]
        with_ev = ev_df[ev_df[col] == 1]
        without_ev = ev_df[ev_df[col] == 0]

        if len(with_sess) < 10 or len(without_sess) < 10:
            ax.set_title(f"{supp}: insufficient data")
            continue

        stats_with = compute_focus_pct(with_sess, with_ev)
        stats_without = compute_focus_pct(without_sess, without_ev)

        # Only plot bins with >= 600 exposure-minutes (10 exposure-hours)
        mask_w = stats_with["exposure_min"] >= 600
        mask_wo = stats_without["exposure_min"] >= 600

        ax.plot(
            stats_with.loc[mask_w, "bin_center_min"],
            stats_with.loc[mask_w, "focus_pct"],
            "o-",
            color="tab:red",
            label=f"with {supp} (n={len(with_sess)} sess)",
        )
        ax.plot(
            stats_without.loc[mask_wo, "bin_center_min"],
            stats_without.loc[mask_wo, "focus_pct"],
            "s-",
            color="tab:blue",
            label=f"without (n={len(without_sess)} sess)",
        )
        ax.set_xlabel("Minutes into session")
        ax.set_ylabel("% of time focused")
        ax.set_title(f"Focus %: {supp}")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
        ax.set_ylim(50, 100)

    plt.suptitle(
        "% Time Focused by Time into Session — Supplement vs No Supplement",
        fontsize=13,
    )
    plt.tight_layout()
    plt.savefig(
        str(Path(__file__).parent / "supplement_focus_pct_timecourse.png"), dpi=150
    )
    plt.close()
    tee_print("  Saved: supplement_focus_pct_timecourse.png")

    # --- 1b-regime: Same time-course split by regime (weekdays only) ---
    regime_map = {"Hive": "Hive", "Mats": "MATS", "diesl": "Diesl"}
    weekday_sess = sess_df[sess_df["date"].dt.dayofweek < 5]
    weekday_ev = ev_df[ev_df["date"].dt.dayofweek < 5]

    for supp in ["caffeine", "adderall", "modafinil", "nicotine"]:
        col = f"had_{supp}"
        if col not in sess_df.columns:
            continue
        # Check we have enough data for at least 2 regimes
        regime_counts = weekday_sess.groupby("regime").size()
        usable_regimes = [
            r for r in regime_map if r in regime_counts.index and regime_counts[r] >= 20
        ]
        if len(usable_regimes) < 2:
            continue

        fig_r, axes_r = plt.subplots(
            1, len(usable_regimes), figsize=(7 * len(usable_regimes), 6)
        )
        if len(usable_regimes) == 1:
            axes_r = [axes_r]

        for ax_r, regime in zip(axes_r, usable_regimes):
            r_sess = weekday_sess[weekday_sess["regime"] == regime]
            r_ev = weekday_ev[weekday_ev["regime"] == regime]

            with_sess_r = r_sess[r_sess[col] == 1]
            without_sess_r = r_sess[r_sess[col] == 0]
            with_ev_r = r_ev[r_ev[col] == 1]
            without_ev_r = r_ev[r_ev[col] == 0]

            if len(with_sess_r) < 5 or len(without_sess_r) < 5:
                ax_r.set_title(f"{regime_map[regime]}: insufficient data for {supp}")
                continue

            stats_w = compute_focus_pct(with_sess_r, with_ev_r)
            stats_wo = compute_focus_pct(without_sess_r, without_ev_r)
            mask_w = stats_w["exposure_min"] >= 300
            mask_wo = stats_wo["exposure_min"] >= 300

            ax_r.plot(
                stats_w.loc[mask_w, "bin_center_min"],
                stats_w.loc[mask_w, "focus_pct"],
                "o-",
                color="tab:red",
                label=f"with {supp} (n={len(with_sess_r)})",
            )
            ax_r.plot(
                stats_wo.loc[mask_wo, "bin_center_min"],
                stats_wo.loc[mask_wo, "focus_pct"],
                "s-",
                color="tab:blue",
                label=f"without (n={len(without_sess_r)})",
            )
            ax_r.set_xlabel("Minutes into session")
            ax_r.set_ylabel("% of time focused")
            ax_r.set_title(f"{regime_map[regime]} weekdays: {supp}")
            ax_r.legend(fontsize=8)
            ax_r.grid(True, alpha=0.3)
            ax_r.set_ylim(50, 100)

        plt.suptitle(
            f"Focus % Time-Course — {supp} by Regime (Weekdays Only)", fontsize=13
        )
        plt.tight_layout()
        plt.savefig(
            str(
                Path(__file__).parent
                / f"supplement_focus_pct_timecourse_{supp}_by_regime.png"
            ),
            dpi=150,
        )
        plt.close()
        tee_print(f"  Saved: supplement_focus_pct_timecourse_{supp}_by_regime.png")

    # --- 1c: Distraction rate in 30-min windows relative to supplement consumption time ---
    tee_print("\n  --- Distraction rate by minutes SINCE supplement consumption ---")

    dist_full = distracted.copy()
    dist_full["time_parsed"] = pd.to_datetime(
        dist_full["time"], format="mixed", errors="coerce"
    )

    fig2, axes2 = plt.subplots(2, 2, figsize=(16, 12))

    for ax, supp in zip(axes2.flat, ["caffeine", "adderall", "modafinil", "nicotine"]):
        # Gather distraction events on days when supp was taken, with
        # minutes-since-consumption
        records: list[dict[str, float]] = []
        baseline_records: list[dict[str, float]] = []

        for date, day_data in dist_full.groupby("date"):
            day_sorted = day_data.sort_values("time_parsed")
            distractions = day_sorted[day_sorted["type"].isin(["d", "u"])]
            if distractions.empty:
                continue

            day_supps = supp_times.get(date, [])
            supp_consumptions = [s for s in day_supps if s["supplement"] == supp]

            if supp_consumptions:
                earliest_supp_time = min(s["time_parsed"] for s in supp_consumptions)
                for _, d_row in distractions.iterrows():
                    min_since = (
                        d_row["time_parsed"] - earliest_supp_time
                    ).total_seconds() / 60
                    if -60 <= min_since <= 480:
                        records.append(
                            {
                                "min_since_supp": min_since,
                                "length_minutes": d_row["length_minutes"],
                            }
                        )
            else:
                # Baseline day: use distraction timing relative to first
                # work start for shape comparison
                first_start = day_sorted[day_sorted["type"] == "s"]
                if first_start.empty:
                    continue
                ref_time = first_start.iloc[0]["time_parsed"]
                for _, d_row in distractions.iterrows():
                    min_since = (d_row["time_parsed"] - ref_time).total_seconds() / 60
                    if -60 <= min_since <= 480:
                        baseline_records.append(
                            {
                                "min_since_supp": min_since,
                                "length_minutes": d_row["length_minutes"],
                            }
                        )

        if len(records) < 20:
            ax.set_title(f"{supp}: insufficient data ({len(records)} events)")
            continue

        rec_df = pd.DataFrame(records)
        base_df = pd.DataFrame(baseline_records) if baseline_records else None

        # Bin and compute counts (not exposure-adjusted — just count histograms)
        bins_rel = np.arange(-60, 481, 30)
        counts, _ = np.histogram(rec_df["min_since_supp"], bins=bins_rel)
        bin_centers = (bins_rel[:-1] + bins_rel[1:]) / 2

        n_supp_days = sum(
            1 for d in supp_times if any(s["supplement"] == supp for s in supp_times[d])
        )
        n_base_days = len(
            set(dist_full["date"].unique())
            - set(
                d
                for d in supp_times
                if any(s["supplement"] == supp for s in supp_times[d])
            )
        )

        # Normalize to distractions per day per 30-min window
        norm_counts = counts / max(n_supp_days, 1)

        ax.bar(
            bin_centers,
            norm_counts,
            width=25,
            alpha=0.6,
            color="tab:red",
            label=f"{supp} days (n={n_supp_days}d)",
        )

        if base_df is not None and len(base_df) > 20:
            base_counts, _ = np.histogram(base_df["min_since_supp"], bins=bins_rel)
            base_norm = base_counts / max(n_base_days, 1)
            ax.step(
                bin_centers,
                base_norm,
                where="mid",
                color="tab:blue",
                linewidth=2,
                label=f"no-{supp} days (n={n_base_days}d)",
            )

        ax.axvline(0, color="black", linestyle="--", alpha=0.5, label="consumption")
        ax.set_xlabel("Minutes since supplement (or session start)")
        ax.set_ylabel("Distractions per day per 30-min bin")
        ax.set_title(f"{supp}: distractions relative to consumption")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    plt.suptitle(
        "Distraction Count Relative to Supplement Consumption Time", fontsize=13
    )
    plt.tight_layout()
    plt.savefig(
        str(Path(__file__).parent / "supplement_distraction_since_consumption.png"),
        dpi=150,
    )
    plt.close()
    tee_print("  Saved: supplement_distraction_since_consumption.png")

    # --- 1d: Residual approach — predict session focus % from rich day-level
    #     features (matching Analysis 6's actionable set), then check if
    #     supplement days have different residuals ---
    tee_print(
        "\n  --- Residual analysis: supplement effect after controlling for day ---"
    )

    # Merge many day-level features into sessions
    merge_cols = [
        "date",
        "is_weekend",
        "day_of_week",
        "is_hive",
        "is_mats",
        "is_diesl",
        "sleep_hours",
        "work_start_hour",
        "prev_hours",
        "hours_rolling_7d_mean",
        "hours_rolling_7d_std",
        "hours_rolling_30d_mean",
        "work_streak",
        "days_since_rest",
    ]
    merge_cols = [c for c in merge_cols if c in df.columns]
    sess_merged = sess_df.merge(df[merge_cols], on="date", how="left")
    sess_merged["focus_pct"] = (
        (1 - sess_merged["total_dist_min"] / sess_merged["duration_min"]) * 100
    ).clip(0, 100)

    # Day-of-week dummies (Mon=0..Sun=6 → dow_tue..dow_sun, Mon is reference)
    dow_names = ["dow_tue", "dow_wed", "dow_thu", "dow_fri", "dow_sat", "dow_sun"]
    for i, name in enumerate(dow_names, start=1):
        sess_merged[name] = (sess_merged["day_of_week"] == (i % 7)).astype(int)

    # Build control feature list (excluding supplements — those are what we test)
    # No is_weekend since it's redundant with dow_sat + dow_sun
    rich_controls = [
        "is_hive",
        "is_mats",
        "is_diesl",
    ] + dow_names
    for col in [
        "sleep_hours",
        "work_start_hour",
        "prev_hours",
        "hours_rolling_7d_mean",
        "hours_rolling_7d_std",
        "hours_rolling_30d_mean",
        "work_streak",
        "days_since_rest",
        "duration_min",
    ]:
        if col in sess_merged.columns and sess_merged[col].notna().sum() > 50:
            rich_controls.append(col)

    sess_clean = sess_merged.dropna(subset=rich_controls + ["focus_pct"])

    if len(sess_clean) > 50:
        X_base = sm.add_constant(sess_clean[rich_controls])
        model_base = sm.OLS(sess_clean["focus_pct"], X_base).fit()
        sess_clean = sess_clean.copy()
        sess_clean["residual"] = model_base.resid

        tee_print(
            f"  Rich baseline model R²={model_base.rsquared:.3f} "
            f"(n={len(sess_clean)}, {len(rich_controls)} features)"
        )

        # Print all coefficients
        tee_print(f"\n  {'Feature':30s} {'Coef':>8s} {'SE':>8s} {'p':>8s}")
        for feat in rich_controls:
            coef = model_base.params[feat]
            se = model_base.bse[feat]
            p = model_base.pvalues[feat]
            sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else ""
            tee_print(f"  {feat:30s} {coef:+8.3f} {se:8.3f} {p:8.4f} {sig}")

        # Propagate any_stimulant flag to merged data
        sess_clean["had_any_stimulant"] = (
            sess_clean[[c for c in stim_cols if c in sess_clean.columns]]
            .sum(axis=1)
            .clip(upper=1)
            .astype(int)
        )

        tee_print("\n  Supplement effect on residuals:")
        supps_to_test = [
            "caffeine",
            "adderall",
            "modafinil",
            "nicotine",
            "any_stimulant",
        ]
        fig3, axes3 = plt.subplots(2, 3, figsize=(18, 10))
        for ax, supp in zip(axes3.flat, supps_to_test):
            col = f"had_{supp}"
            if col not in sess_clean.columns:
                ax.set_title(f"{supp}: no data")
                continue
            with_r = sess_clean[sess_clean[col] == 1]["residual"]
            without_r = sess_clean[sess_clean[col] == 0]["residual"]
            if len(with_r) < 10 or len(without_r) < 10:
                ax.set_title(f"{supp}: insufficient data")
                continue

            _, p = stats.mannwhitneyu(with_r, without_r, alternative="two-sided")
            sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else ""
            tee_print(
                f"  {supp:15s}: with_resid={with_r.mean():+.1f}%, "
                f"without_resid={without_r.mean():+.1f}%, "
                f"diff={with_r.mean() - without_r.mean():+.1f}%, "
                f"p={p:.4f} {sig}"
            )

            ax.hist(
                without_r,
                bins=30,
                alpha=0.5,
                density=True,
                color="tab:blue",
                label=f"no {supp}",
            )
            ax.hist(
                with_r,
                bins=30,
                alpha=0.5,
                density=True,
                color="tab:red",
                label=f"with {supp}",
            )
            ax.axvline(without_r.mean(), color="tab:blue", linestyle="--")
            ax.axvline(with_r.mean(), color="tab:red", linestyle="--")
            ax.set_xlabel("Residual focus % (after rich controls)")
            ax.set_ylabel("Density")
            ax.set_title(f"{supp}: residual focus % (p={p:.3f}{sig})")
            ax.legend(fontsize=8)

        # Hide unused subplot(s) in the 2x3 grid
        for i in range(len(supps_to_test), len(axes3.flat)):
            axes3.flat[i].set_visible(False)

        plt.suptitle(
            f"Focus % Residuals (R²={model_base.rsquared:.3f}, "
            f"{len(rich_controls)} controls incl. DOW dummies)",
            fontsize=13,
        )
        plt.tight_layout()
        plt.savefig(
            str(Path(__file__).parent / "supplement_focus_pct_residuals.png"),
            dpi=150,
        )
        plt.close()
        tee_print("  Saved: supplement_focus_pct_residuals.png")

    tee_print()


# ============================================================================
# ANALYSIS 2: WITHIN-SESSION DISTRACTION RATE BY MINUTES INTO SESSION
# ============================================================================
def diminishing_returns_analysis() -> None:
    tee_print("=" * 70)
    tee_print(
        "ANALYSIS 2: WITHIN-SESSION DISTRACTION RATE OVER TIME "
        "(DIMINISHING RETURNS?)"
    )
    tee_print("=" * 70)

    sess_df, ev_df, _ = _build_session_and_event_data()

    if ev_df.empty:
        tee_print("  No distraction events found.")
        return

    # --- 2a: Focus % by 30-min bin into session ---
    bin_width = 30  # minutes
    max_min = 600  # 10 hours
    bin_edges = np.arange(0, max_min + bin_width, bin_width)

    exposure_min = np.zeros(len(bin_edges) - 1)
    dist_min_arr = np.zeros(len(bin_edges) - 1)

    for i, (left, right) in enumerate(zip(bin_edges[:-1], bin_edges[1:])):
        clipped_end = np.minimum(sess_df["duration_min"].values, right)
        contrib = np.maximum(clipped_end - left, 0)
        exposure_min[i] = contrib.sum()

        bin_mask = (ev_df["min_into_session"] >= left) & (
            ev_df["min_into_session"] < right
        )
        dist_min_arr[i] = ev_df.loc[bin_mask, "length_minutes"].sum()

    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    focus_pct = np.where(
        exposure_min > 0, (1 - dist_min_arr / exposure_min) * 100, np.nan
    )

    # Only show bins with >= 5 exposure-hours (300 min)
    valid_mask = exposure_min >= 300

    tee_print("\n  % time focused by 30-min window into session:")
    tee_print(f"  {'Window':>12s} {'Focus%':>8s} {'DistMin':>8s} {'ExposMin':>9s}")
    for i in range(len(bin_edges) - 1):
        if not valid_mask[i]:
            continue
        label = f"{int(bin_edges[i])}-{int(bin_edges[i+1])}m"
        tee_print(
            f"  {label:>12s} {focus_pct[i]:7.1f}% {dist_min_arr[i]:8.0f} "
            f"{exposure_min[i]:9.0f}"
        )

    # --- 2b: Also by cumulative day-hours (minutes_into_day) ---
    day_bin_edges = np.arange(0, max_min + bin_width, bin_width)
    day_exposure_min = np.zeros(len(day_bin_edges) - 1)
    day_dist_min = np.zeros(len(day_bin_edges) - 1)

    for _, s in sess_df.iterrows():
        s_start = s["cum_work_min_before"]
        s_end = s_start + s["duration_min"]
        for i, (left, right) in enumerate(zip(day_bin_edges[:-1], day_bin_edges[1:])):
            overlap = max(0, min(s_end, right) - max(s_start, left))
            day_exposure_min[i] += overlap

    for i, (left, right) in enumerate(zip(day_bin_edges[:-1], day_bin_edges[1:])):
        bin_mask = (ev_df["min_into_day"] >= left) & (ev_df["min_into_day"] < right)
        day_dist_min[i] = ev_df.loc[bin_mask, "length_minutes"].sum()

    day_bin_centers = (day_bin_edges[:-1] + day_bin_edges[1:]) / 2
    day_focus_pct = np.where(
        day_exposure_min > 0, (1 - day_dist_min / day_exposure_min) * 100, np.nan
    )
    day_valid = day_exposure_min >= 300

    tee_print("\n  % time focused by cumulative work-minutes in the day:")
    tee_print(f"  {'Window':>12s} {'Focus%':>8s} {'DistMin':>8s} {'ExposMin':>9s}")
    for i in range(len(day_bin_edges) - 1):
        if not day_valid[i]:
            continue
        label = f"{int(day_bin_edges[i])}-{int(day_bin_edges[i+1])}m"
        tee_print(
            f"  {label:>12s} {day_focus_pct[i]:7.1f}% {day_dist_min[i]:8.0f} "
            f"{day_exposure_min[i]:9.0f}"
        )

    # --- 2c: Graphs ---
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    ax = axes[0]
    ax.bar(
        bin_centers[valid_mask] / 60,
        focus_pct[valid_mask],
        width=bin_width / 60 * 0.85,
        alpha=0.6,
        color="steelblue",
    )
    if valid_mask.sum() > 3:
        x_valid = bin_centers[valid_mask] / 60
        y_valid = focus_pct[valid_mask]
        z = np.polyfit(x_valid, y_valid, 1)
        ax.plot(
            x_valid,
            np.polyval(z, x_valid),
            "r--",
            linewidth=2,
            label=f"trend: {z[0]:+.1f}%/hour",
        )
        ax.legend()
    ax.set_xlabel("Hours into session")
    ax.set_ylabel("% of time focused")
    ax.set_title("Focus % by Time into Session")
    ax.set_ylim(0, 105)
    ax.grid(True, alpha=0.3)

    ax = axes[1]
    ax.bar(
        day_bin_centers[day_valid] / 60,
        day_focus_pct[day_valid],
        width=bin_width / 60 * 0.85,
        alpha=0.6,
        color="darkorange",
    )
    if day_valid.sum() > 3:
        x_v = day_bin_centers[day_valid] / 60
        y_v = day_focus_pct[day_valid]
        z2 = np.polyfit(x_v, y_v, 1)
        ax.plot(
            x_v,
            np.polyval(z2, x_v),
            "r--",
            linewidth=2,
            label=f"trend: {z2[0]:+.1f}%/cum. hour",
        )
        ax.legend()
    ax.set_xlabel("Cumulative work-hours in the day")
    ax.set_ylabel("% of time focused")
    ax.set_title("Focus % by Cumulative Day Hours")
    ax.set_ylim(0, 105)
    ax.grid(True, alpha=0.3)

    plt.suptitle(
        "Does Focus Decline Over Time? (% of Time Spent Focused)",
        fontsize=13,
    )
    plt.tight_layout()
    plt.savefig(
        str(Path(__file__).parent / "diminishing_returns_focus_pct.png"),
        dpi=150,
    )
    plt.close()
    tee_print("  Saved: diminishing_returns_focus_pct.png")

    # --- 2d: Regime-split diminishing returns (weekdays only) ---
    regime_map = {"Hive": "Hive", "Mats": "MATS", "diesl": "Diesl"}
    regime_colors = {"Hive": "steelblue", "Mats": "tab:purple", "diesl": "tab:orange"}
    weekday_sess = sess_df[sess_df["date"].dt.dayofweek < 5]
    weekday_ev = ev_df[ev_df["date"].dt.dayofweek < 5]

    regime_counts = weekday_sess.groupby("regime").size()
    usable_regimes = [
        r for r in regime_map if r in regime_counts.index and regime_counts[r] >= 20
    ]

    if len(usable_regimes) >= 2:
        fig_r, axes_r = plt.subplots(
            1, len(usable_regimes), figsize=(7 * len(usable_regimes), 6), sharey=True
        )
        if len(usable_regimes) == 1:
            axes_r = [axes_r]

        for ax_r, regime in zip(axes_r, usable_regimes):
            r_sess = weekday_sess[weekday_sess["regime"] == regime]
            r_ev = weekday_ev[weekday_ev["regime"] == regime]

            r_exposure = np.zeros(len(bin_edges) - 1)
            r_dist = np.zeros(len(bin_edges) - 1)
            for i, (left, right) in enumerate(zip(bin_edges[:-1], bin_edges[1:])):
                clipped = np.minimum(r_sess["duration_min"].values, right)
                r_exposure[i] = np.maximum(clipped - left, 0).sum()
                bm = (r_ev["min_into_session"] >= left) & (
                    r_ev["min_into_session"] < right
                )
                r_dist[i] = r_ev.loc[bm, "length_minutes"].sum()

            r_focus = np.where(r_exposure > 0, (1 - r_dist / r_exposure) * 100, np.nan)
            r_valid = r_exposure >= 120  # lower threshold per regime

            ax_r.bar(
                bin_centers[r_valid] / 60,
                r_focus[r_valid],
                width=bin_width / 60 * 0.85,
                alpha=0.6,
                color=regime_colors[regime],
            )
            if r_valid.sum() > 3:
                xv = bin_centers[r_valid] / 60
                yv = r_focus[r_valid]
                z_r = np.polyfit(xv, yv, 1)
                ax_r.plot(
                    xv,
                    np.polyval(z_r, xv),
                    "r--",
                    linewidth=2,
                    label=f"trend: {z_r[0]:+.1f}%/h",
                )
                ax_r.legend()
            ax_r.set_xlabel("Hours into session")
            ax_r.set_ylabel("% of time focused")
            ax_r.set_title(f"{regime_map[regime]} weekdays (n={len(r_sess)} sess)")
            ax_r.set_ylim(0, 105)
            ax_r.grid(True, alpha=0.3)

        plt.suptitle(
            "Focus % by Time into Session — By Regime (Weekdays Only)", fontsize=13
        )
        plt.tight_layout()
        plt.savefig(
            str(Path(__file__).parent / "diminishing_returns_focus_pct_by_regime.png"),
            dpi=150,
        )
        plt.close()
        tee_print("  Saved: diminishing_returns_focus_pct_by_regime.png")

    # --- Statistical tests ---
    # Compare focus % in first 2h vs after 4h
    exp_early = exposure_min[:4].sum()
    dist_early = dist_min_arr[:4].sum()
    exp_late = exposure_min[8:].sum()
    dist_late = dist_min_arr[8:].sum()
    if exp_early > 0 and exp_late > 0:
        focus_early = (1 - dist_early / exp_early) * 100
        focus_late = (1 - dist_late / exp_late) * 100
        tee_print(
            f"\n  Focus first 2h: {focus_early:.1f}% "
            f"({dist_early:.0f} dist-min in {exp_early:.0f} exposure-min)"
        )
        tee_print(
            f"  Focus after 4h: {focus_late:.1f}% "
            f"({dist_late:.0f} dist-min in {exp_late:.0f} exposure-min)"
        )

    tee_print()


# ============================================================================
# ANALYSIS 3: SUPPLEMENT INTERACTION EFFECTS
# ============================================================================
def supplement_interactions() -> None:
    tee_print("=" * 70)
    tee_print("ANALYSIS 3: SUPPLEMENT INTERACTION EFFECTS")
    tee_print("=" * 70)

    v = valid.copy()

    # Build interaction terms for supplements with sufficient data
    supp_pairs = [
        ("caffeine", "adderall"),
        ("caffeine", "modafinil"),
        ("caffeine", "nicotine"),
        ("adderall", "nicotine"),
    ]

    for s1, s2 in supp_pairs:
        c1 = f"{s1}_any"
        c2 = f"{s2}_any"
        if c1 not in v.columns or c2 not in v.columns:
            continue
        if (v[c1] > 0).sum() < 15 or (v[c2] > 0).sum() < 15:
            continue

        # 2x2 table
        neither = v[(v[c1] == 0) & (v[c2] == 0)]["Hours Working"]
        only1 = v[(v[c1] == 1) & (v[c2] == 0)]["Hours Working"]
        only2 = v[(v[c1] == 0) & (v[c2] == 1)]["Hours Working"]
        both = v[(v[c1] == 1) & (v[c2] == 1)]["Hours Working"]

        tee_print(f"\n  --- {s1} x {s2} ---")
        tee_print(f"  Neither:    {neither.mean():.2f}h (n={len(neither)})")
        tee_print(f"  {s1} only: {only1.mean():.2f}h (n={len(only1)})")
        tee_print(f"  {s2} only: {only2.mean():.2f}h (n={len(only2)})")
        tee_print(f"  Both:       {both.mean():.2f}h (n={len(both)})")

        # Additive prediction vs actual
        additive_pred = (
            neither.mean()
            + (only1.mean() - neither.mean())
            + (only2.mean() - neither.mean())
        )
        if len(both) >= 5:
            synergy = both.mean() - additive_pred
            tee_print(
                f"  Additive prediction: {additive_pred:.2f}h, "
                f"Actual both: {both.mean():.2f}h, "
                f"Synergy: {synergy:+.2f}h"
            )

        # Formal interaction term in regression
        interaction_col = f"{s1}_x_{s2}"
        v_int = v.copy()
        v_int[interaction_col] = v_int[c1] * v_int[c2]
        features = BASE_CONTROLS + [c1, c2, interaction_col]
        sub = v_int[["Hours Working"] + features].dropna()
        if len(sub) > 30:
            X = sm.add_constant(sub[features])
            model = sm.OLS(sub["Hours Working"], X).fit()
            int_coef = model.params[interaction_col]
            int_p = model.pvalues[interaction_col]
            sig = (
                "***"
                if int_p < 0.001
                else "**" if int_p < 0.01 else "*" if int_p < 0.05 else ""
            )
            tee_print(f"  Interaction term: {int_coef:+.3f}h (p={int_p:.3f}) {sig}")

    tee_print()


# ============================================================================
# ANALYSIS 4: CALENDAR EVENT-LEVEL PREDICTION
# ============================================================================
def event_level_prediction() -> None:
    tee_print("=" * 70)
    tee_print("ANALYSIS 4: WHICH SPECIFIC CALENDAR EVENTS PREDICT WORK OUTPUT?")
    tee_print("=" * 70)

    # Get hours per event per day
    ev_daily = cal_full.groupby(["date", "event_lower"])["duration"].sum().reset_index()

    # Find events that appear on at least 20 different days
    event_day_counts = ev_daily.groupby("event_lower")["date"].nunique()
    frequent_events = event_day_counts[event_day_counts >= 20].index.tolist()

    # Remove sleep (always present, not actionable)
    frequent_events = [e for e in frequent_events if e not in ["sleep"]]

    tee_print(f"\n  Events appearing on 20+ days: {len(frequent_events)}")

    # Pivot to get event hours per day
    ev_pivot = ev_daily[ev_daily["event_lower"].isin(frequent_events)].pivot_table(
        index="date", columns="event_lower", values="duration", fill_value=0
    )

    # Merge with daily df
    df_ev = df.merge(ev_pivot, left_on="date", right_index=True, how="left")
    df_ev[frequent_events] = df_ev[frequent_events].fillna(0)
    df_ev = df_ev[df_ev["Hours Working"] > 0].copy()

    # Controlled effect of each event on Hours Working
    tee_print("\n  --- Controlled effect on Hours Working (regime + weekend + DOW) ---")
    tee_print(f"  {'Event':35s} {'Coef':>8s} {'p':>8s} {'n':>6s}")

    results: list[dict[str, object]] = []
    for ev in sorted(frequent_events):
        r = controlled_effect(df_ev, ev)
        if np.isnan(r["coef"]):
            continue
        results.append({"event": ev, **r})

    results.sort(key=lambda x: abs(x["coef"]), reverse=True)
    for r in results[:30]:
        sig = (
            "***"
            if r["p"] < 0.001
            else "**" if r["p"] < 0.01 else "*" if r["p"] < 0.05 else ""
        )
        tee_print(
            f"  {str(r['event']):35s} {r['coef']:+8.3f}h {r['p']:8.3f} "
            f"n={r['n']:4d} {sig}"
        )

    # Same-day vs next-day: which events predict TOMORROW's output?
    tee_print("\n  --- Next-day prediction (event today → Hours Working tomorrow) ---")
    df_ev_sorted = df_ev.sort_values("date").reset_index(drop=True)
    df_ev_sorted["next_hours"] = df_ev_sorted["Hours Working"].shift(-1)
    # Only use consecutive days
    df_ev_sorted["next_date"] = df_ev_sorted["date"].shift(-1)
    df_ev_sorted["is_consecutive"] = (
        df_ev_sorted["next_date"] - df_ev_sorted["date"]
    ).dt.days == 1
    df_consec = df_ev_sorted[df_ev_sorted["is_consecutive"]].copy()

    next_day_results: list[dict[str, object]] = []
    for ev in frequent_events:
        r = controlled_effect(df_consec, ev, outcome="next_hours")
        if np.isnan(r["coef"]):
            continue
        next_day_results.append({"event": ev, **r})

    next_day_results.sort(key=lambda x: abs(x["coef"]), reverse=True)
    tee_print(f"  {'Event':35s} {'Coef':>8s} {'p':>8s} {'n':>6s}")
    for r in next_day_results[:20]:
        sig = (
            "***"
            if r["p"] < 0.001
            else "**" if r["p"] < 0.01 else "*" if r["p"] < 0.05 else ""
        )
        tee_print(
            f"  {str(r['event']):35s} {r['coef']:+8.3f}h {r['p']:8.3f} "
            f"n={r['n']:4d} {sig}"
        )

    tee_print()


# ============================================================================
# ANALYSIS 5: TASK-LEVEL AND TEXT FIELD MINING
# ============================================================================
def text_field_mining() -> None:
    tee_print("=" * 70)
    tee_print("ANALYSIS 5: TEXT FIELD MINING (Tasks Summary, Notes, Reflections)")
    tee_print("=" * 70)

    # Load raw CSV with text fields
    raw_daily = pd.read_csv(DAILY_SUMMARY_CSV)
    raw_daily["date"] = pd.to_datetime(raw_daily["Start Date"], errors="coerce")
    for c in ["Hours Working", "Energy", "Focus", "Value"]:
        raw_daily[c] = pd.to_numeric(raw_daily[c], errors="coerce")

    # --- 5a: Tasks Summary analysis ---
    tasks_df = raw_daily.dropna(subset=["Tasks Summary", "Hours Working"]).copy()
    tasks_df = tasks_df[tasks_df["Hours Working"] > 0]

    if len(tasks_df) > 50:
        tee_print(f"\n  Tasks Summary entries: {len(tasks_df)}")

        # Extract common task keywords
        # Split on semicolons and commas for individual tasks
        task_items: list[str] = []
        for text in tasks_df["Tasks Summary"].fillna(""):
            items = re.split(r"[;,]", text.lower())
            task_items.extend(t.strip() for t in items if len(t.strip()) > 3)

        # Count task keywords
        word_counter: Counter[str] = Counter()
        for item in task_items:
            words = item.split()
            for w in words:
                if len(w) > 3 and w not in {
                    "with",
                    "from",
                    "that",
                    "this",
                    "been",
                    "have",
                    "some",
                    "more",
                    "than",
                    "also",
                    "them",
                    "were",
                    "into",
                    "work",
                    "worked",
                    "working",
                    "tried",
                    "done",
                    "make",
                    "made",
                }:
                    word_counter[w] += 1

        tee_print("\n  Top 20 task keywords:")
        for word, count in word_counter.most_common(20):
            tee_print(f"    {word:20s}: {count}")

        # TF-IDF on task descriptions → correlate with outcomes
        vectorizer = TfidfVectorizer(
            max_features=50, min_df=10, stop_words="english", ngram_range=(1, 2)
        )
        tfidf_matrix = vectorizer.fit_transform(tasks_df["Tasks Summary"].fillna(""))
        feature_names = vectorizer.get_feature_names_out()

        tee_print("\n  --- Task keywords correlated with Hours Working ---")
        tee_print(f"  {'Keyword':30s} {'Corr':>8s} {'p':>8s}")
        keyword_effects: list[tuple[str, float, float]] = []
        for i, feature in enumerate(feature_names):
            col = tfidf_matrix[:, i].toarray().ravel()
            nonzero = col > 0
            if nonzero.sum() < 15:
                continue
            r, p = stats.pointbiserialr(
                nonzero.astype(int), tasks_df["Hours Working"].values
            )
            keyword_effects.append((feature, r, p))

        keyword_effects.sort(key=lambda x: abs(x[1]), reverse=True)
        for kw, r, p in keyword_effects[:15]:
            sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else ""
            tee_print(f"  {kw:30s} {r:+8.3f} {p:8.3f} {sig}")

        # Same for Energy
        tee_print("\n  --- Task keywords correlated with Energy ---")
        tee_print(f"  {'Keyword':30s} {'Corr':>8s} {'p':>8s}")
        energy_tasks = tasks_df.dropna(subset=["Energy"])
        if len(energy_tasks) > 50:
            tfidf_e = vectorizer.transform(energy_tasks["Tasks Summary"].fillna(""))
            energy_effects: list[tuple[str, float, float]] = []
            for i, feature in enumerate(feature_names):
                col = tfidf_e[:, i].toarray().ravel()
                nonzero = col > 0
                if nonzero.sum() < 15:
                    continue
                r, p = stats.pointbiserialr(
                    nonzero.astype(int), energy_tasks["Energy"].values
                )
                energy_effects.append((feature, r, p))
            energy_effects.sort(key=lambda x: abs(x[1]), reverse=True)
            for kw, r, p in energy_effects[:15]:
                sig = (
                    "***"
                    if p < 0.001
                    else "**" if p < 0.01 else "*" if p < 0.05 else ""
                )
                tee_print(f"  {kw:30s} {r:+8.3f} {p:8.3f} {sig}")

    # --- 5b: Productivity Notes analysis ---
    notes_df = raw_daily.dropna(subset=["Productivity Notes", "Hours Working"]).copy()
    notes_df = notes_df[notes_df["Hours Working"] > 0]

    if len(notes_df) > 30:
        tee_print(f"\n  --- Productivity Notes ({len(notes_df)} entries) ---")
        note_counter: Counter[str] = Counter()
        for text in notes_df["Productivity Notes"].fillna(""):
            note_counter[text.strip().lower()] += 1

        tee_print("  Most common notes:")
        for note, count in note_counter.most_common(15):
            if count >= 3:
                subset = notes_df[
                    notes_df["Productivity Notes"].str.strip().str.lower() == note
                ]
                mean_h = subset["Hours Working"].mean()
                tee_print(f"    {note[:40]:40s}: n={count}, mean_hours={mean_h:.1f}")

    # --- 5c: Reflections analysis ---
    reflect_df = raw_daily.dropna(subset=["Reflections", "Hours Working"]).copy()
    reflect_df = reflect_df[reflect_df["Hours Working"] > 0]

    if len(reflect_df) > 30:
        tee_print(f"\n  --- Reflections text ({len(reflect_df)} entries) ---")
        # Look for themes in reflections
        themes = {
            "should_have": r"should'?v?e?\b|shouldn'?t",
            "distracted": r"distract|unfocus|zone",
            "tired": r"tired|exhaust|sleep|fatigue",
            "motivated": r"motivat|energiz|excit|flow",
            "stuck": r"stuck|block|struggle|confus",
        }

        for theme, pattern in themes.items():
            matches = reflect_df[
                reflect_df["Reflections"].str.contains(
                    pattern, case=False, na=False, regex=True
                )
            ]
            if len(matches) >= 5:
                non_matches = reflect_df[~reflect_df.index.isin(matches.index)]
                diff = (
                    matches["Hours Working"].mean()
                    - non_matches["Hours Working"].mean()
                )
                tee_print(
                    f"  Theme '{theme}': n={len(matches)}, "
                    f"mean_hours={matches['Hours Working'].mean():.1f} "
                    f"(vs {non_matches['Hours Working'].mean():.1f}, "
                    f"diff={diff:+.1f})"
                )

    # --- 5d: Weekly Summary & Lessons mining ---
    tee_print("\n  --- Weekly Summary & Lessons ---")
    weekly_csv = pd.read_csv(
        str(Path(REPO_DIR) / "data" / "Work Summary  - Weekly Summary_Projects.csv"),
        header=None,
        skiprows=1,
    )
    weekly = pd.DataFrame(
        {
            "regime": weekly_csv.iloc[:, 0],
            "start": pd.to_datetime(
                weekly_csv.iloc[:, 1], format="mixed", errors="coerce"
            ),
            "hours": pd.to_numeric(weekly_csv.iloc[:, 5], errors="coerce"),
            "summary": weekly_csv.iloc[:, 11],
            "lessons": weekly_csv.iloc[:, 12],
        }
    )
    weekly = weekly.dropna(subset=["start", "hours"])
    weekly = weekly[weekly["hours"] > 0]

    # Correlate lessons content with weekly hours
    lessons_df = weekly.dropna(subset=["lessons"])
    if len(lessons_df) > 20:
        tee_print(f"  Weeks with lessons: {len(lessons_df)}")
        # Weeks that mention specific themes
        lesson_themes = {
            "focus": r"focus|concentr|atten",
            "planning": r"plan|schedul|organiz|structure",
            "debugging": r"debug|bug|error|fix",
            "learning": r"learn|stud|read|book",
        }
        for theme, pattern in lesson_themes.items():
            matches = lessons_df[
                lessons_df["lessons"].str.contains(
                    pattern, case=False, na=False, regex=True
                )
            ]
            if len(matches) >= 3:
                tee_print(
                    f"  Lessons mentioning '{theme}': n={len(matches)}, "
                    f"mean_hours={matches['hours'].mean():.1f}h"
                )

    tee_print()


# ============================================================================
# ANALYSIS 6: COMPREHENSIVE REGULARIZED MODEL WITH HOLDOUT
# ============================================================================
def comprehensive_regularized_model() -> None:
    tee_print("=" * 70)
    tee_print("ANALYSIS 6: COMPREHENSIVE REGULARIZED MODEL WITH TEMPORAL HOLDOUT")
    tee_print("=" * 70)

    v = valid.sort_values("date").reset_index(drop=True).copy()

    # --- Build feature sets ---
    # ACTIONABLE features (known before the day starts):
    actionable_features = []
    # Timing/schedule
    for col in ["is_weekend", "day_of_week", "work_start_hour"]:
        if col in v.columns and v[col].notna().sum() > 100:
            actionable_features.append(col)

    # Regime
    for col in ["is_hive", "is_mats", "is_diesl"]:
        if col in v.columns:
            actionable_features.append(col)

    # Momentum/history
    for col in [
        "prev_hours",
        "hours_rolling_7d_mean",
        "hours_rolling_7d_std",
        "hours_rolling_30d_mean",
        "work_streak",
        "days_since_rest",
    ]:
        if col in v.columns and v[col].notna().sum() > 100:
            actionable_features.append(col)

    # Supplements (taken in the morning, known early)
    supp_any_cols = [
        c for c in v.columns if c.endswith("_any") and (v[c] > 0).sum() > 15
    ]
    actionable_features.extend(supp_any_cols)

    # Sleep (known at day start)
    if "sleep_hours" in v.columns and v["sleep_hours"].notna().sum() > 100:
        actionable_features.append("sleep_hours")

    # DESCRIPTIVE features (known only after the day):
    descriptive_features = list(actionable_features)
    for col in [
        "cal_things",
        "cal_waste",
        "cal_green",
        "n_sessions",
        "# Distractions",
        "Length Distractions",
        "# Unfocused",
    ]:
        if col in v.columns and v[col].notna().sum() > 100:
            descriptive_features.append(col)

    # Calendar events
    for ev in key_events:
        col = f"cal_{ev}"
        if col in v.columns and (v[col] > 0).sum() > 20:
            descriptive_features.append(col)

    tee_print(f"\n  Actionable features: {len(actionable_features)}")
    tee_print(f"  Descriptive features: {len(descriptive_features)}")

    # --- Temporal train/test split ---
    # Use last 20% of data as holdout
    split_idx = int(len(v) * 0.8)
    train = v.iloc[:split_idx]
    test = v.iloc[split_idx:]
    tee_print(
        f"  Train: {len(train)} days ({train['date'].min().date()} to "
        f"{train['date'].max().date()})"
    )
    tee_print(
        f"  Test:  {len(test)} days ({test['date'].min().date()} to "
        f"{test['date'].max().date()})"
    )

    def run_model(features: list[str], label: str) -> None:
        sub_train = train[["Hours Working"] + features].dropna()
        sub_test = test[["Hours Working"] + features].dropna()

        if len(sub_train) < 50 or len(sub_test) < 20:
            tee_print(
                f"\n  {label}: insufficient data (train={len(sub_train)}, test={len(sub_test)})"
            )
            return

        X_train = sub_train[features].values
        y_train = sub_train["Hours Working"].values
        X_test = sub_test[features].values
        y_test = sub_test["Hours Working"].values

        scaler = StandardScaler()
        X_train_s = scaler.fit_transform(X_train)
        X_test_s = scaler.transform(X_test)

        alphas = np.logspace(-3, 3, 50)
        tscv = TimeSeriesSplit(n_splits=5)

        # OLS baseline
        X_ols = sm.add_constant(sub_train[features])
        ols = sm.OLS(y_train, X_ols).fit()
        X_test_ols = sm.add_constant(sub_test[features])
        ols_pred = ols.predict(X_test_ols)
        ols_r2_test = 1 - np.sum((y_test - ols_pred) ** 2) / np.sum(
            (y_test - y_test.mean()) ** 2
        )

        # Ridge
        ridge = RidgeCV(alphas=alphas, cv=tscv)
        ridge.fit(X_train_s, y_train)
        ridge_pred = ridge.predict(X_test_s)
        ridge_r2_test = 1 - np.sum((y_test - ridge_pred) ** 2) / np.sum(
            (y_test - y_test.mean()) ** 2
        )

        # Lasso
        lasso = LassoCV(alphas=alphas, cv=tscv, max_iter=10000)
        lasso.fit(X_train_s, y_train)
        lasso_pred = lasso.predict(X_test_s)
        lasso_r2_test = 1 - np.sum((y_test - lasso_pred) ** 2) / np.sum(
            (y_test - y_test.mean()) ** 2
        )

        # ElasticNet
        enet = ElasticNetCV(
            l1_ratio=[0.1, 0.5, 0.7, 0.9], alphas=alphas, cv=tscv, max_iter=10000
        )
        enet.fit(X_train_s, y_train)
        enet_pred = enet.predict(X_test_s)
        enet_r2_test = 1 - np.sum((y_test - enet_pred) ** 2) / np.sum(
            (y_test - y_test.mean()) ** 2
        )

        tee_print(f"\n  --- {label} ---")
        tee_print(f"  Train n={len(sub_train)}, Test n={len(sub_test)}")
        tee_print(f"  {'Model':15s} {'Train R²':>10s} {'Test R²':>10s}")
        tee_print(f"  {'OLS':15s} {ols.rsquared:10.3f} {ols_r2_test:10.3f}")
        tee_print(
            f"  {'Ridge':15s} {ridge.score(X_train_s, y_train):10.3f} "
            f"{ridge_r2_test:10.3f}"
        )
        tee_print(
            f"  {'Lasso':15s} {lasso.score(X_train_s, y_train):10.3f} "
            f"{lasso_r2_test:10.3f}"
        )
        tee_print(
            f"  {'ElasticNet':15s} {enet.score(X_train_s, y_train):10.3f} "
            f"{enet_r2_test:10.3f}"
        )

        # Feature importance from best model
        best_name, best_coefs = "Lasso", lasso.coef_
        if ridge_r2_test > lasso_r2_test and ridge_r2_test > enet_r2_test:
            best_name, best_coefs = "Ridge", ridge.coef_
        elif enet_r2_test > lasso_r2_test:
            best_name, best_coefs = "ElasticNet", enet.coef_

        coef_series = pd.Series(best_coefs / scaler.scale_, index=features)
        coef_series = coef_series[coef_series.abs() > 0.001].sort_values(
            key=abs, ascending=False
        )

        tee_print(f"\n  Top features ({best_name}, unscaled coefficients):")
        tee_print(f"  {'Feature':35s} {'Coef':>10s}")
        for feat, coef in coef_series.head(15).items():
            tee_print(f"  {str(feat):35s} {coef:+10.3f}")

    run_model(actionable_features, "ACTIONABLE (morning) model")
    run_model(descriptive_features, "DESCRIPTIVE (full-day) model")

    tee_print()


# ============================================================================
# ANALYSIS 7: WEEKLY PROJECT-LEVEL ANALYSIS
# ============================================================================
def weekly_project_analysis() -> None:
    tee_print("=" * 70)
    tee_print("ANALYSIS 7: WEEKLY PROJECT-LEVEL VALUE PER HOUR")
    tee_print("=" * 70)

    weekly_csv = pd.read_csv(
        str(Path(REPO_DIR) / "data" / "Work Summary  - Weekly Summary_Projects.csv"),
        header=None,
        skiprows=1,
    )
    weekly = pd.DataFrame(
        {
            "regime": weekly_csv.iloc[:, 0],
            "start": pd.to_datetime(
                weekly_csv.iloc[:, 1], format="mixed", errors="coerce"
            ),
            "value": pd.to_numeric(weekly_csv.iloc[:, 4], errors="coerce"),
            "hours": pd.to_numeric(weekly_csv.iloc[:, 5], errors="coerce"),
            "projects": weekly_csv.iloc[:, 14] if weekly_csv.shape[1] > 14 else None,
        }
    )
    weekly = weekly.dropna(subset=["start", "hours"])
    weekly = weekly[weekly["hours"] > 0]

    if "projects" in weekly.columns:
        proj_df = weekly.dropna(subset=["projects"])
        if len(proj_df) > 10:
            tee_print(f"\n  Weeks with project data: {len(proj_df)}")
            # Extract project names
            proj_counter: Counter[str] = Counter()
            for text in proj_df["projects"]:
                items = re.split(r"[;,\n]", str(text).lower())
                for item in items:
                    item = item.strip()
                    if len(item) > 2:
                        proj_counter[item] += 1

            tee_print("\n  Most common projects mentioned:")
            for proj, count in proj_counter.most_common(15):
                tee_print(f"    {proj[:40]:40s}: {count} weeks")

    # Value per hour by regime
    tee_print("\n  --- Value per hour by regime ---")
    weekly["value_per_hour"] = weekly["value"] / weekly["hours"]
    regime_vph = weekly.groupby("regime").agg(
        mean_vph=("value_per_hour", "mean"),
        mean_hours=("hours", "mean"),
        n=("hours", "count"),
    )
    for regime, row in regime_vph.iterrows():
        tee_print(
            f"  {str(regime):15s}: value/h={row['mean_vph']:.2f}, "
            f"mean_hours={row['mean_hours']:.1f}, n={int(row['n'])}"
        )

    tee_print()


# ============================================================================
# ANALYSIS 8: DOSE-RESPONSE, HEAD-TO-HEAD, AND FOCUSED HOURS
# ============================================================================
def dose_response_and_focused_hours() -> None:
    tee_print("=" * 70)
    tee_print("ANALYSIS 8: DOSE-RESPONSE, CAFFEINE vs ADDERALL, FOCUSED HOURS")
    tee_print("=" * 70)

    sess_df, ev_df, _ = _build_session_and_event_data()
    if sess_df.empty:
        tee_print("  No sessions found.")
        return

    sess_df["focus_pct"] = (
        (1 - sess_df["total_dist_min"] / sess_df["duration_min"]) * 100
    ).clip(0, 100)
    sess_df["focused_hours"] = sess_df["duration_min"] * sess_df["focus_pct"] / 100 / 60

    # --- 8a: Caffeine dose-response (within caffeine-only days) ---
    tee_print("\n  --- 8a: Caffeine dose-response (caffeine-only sessions) ---")
    caf_only = sess_df[
        (sess_df["had_caffeine"] == 1) & (sess_df["had_adderall"] == 0)
    ].copy()
    tee_print(f"  Caffeine-only sessions: {len(caf_only)}")

    if len(caf_only) > 30 and "dose_caffeine" in caf_only.columns:
        # Show dose distribution
        doses = caf_only["dose_caffeine"]
        tee_print(
            f"  Dose range: {doses.min():.0f} - {doses.max():.0f} mg, "
            f"median={doses.median():.0f}, mean={doses.mean():.0f}"
        )

        # Bin doses into tertiles
        caf_only["dose_bin"] = pd.qcut(
            caf_only["dose_caffeine"],
            q=3,
            labels=["low", "med", "high"],
            duplicates="drop",
        )
        tee_print(
            f"\n  {'Dose bin':>10s} {'Range':>15s} {'Focus%':>8s} "
            f"{'Foc.hrs':>8s} {'Dur.min':>8s} {'n':>5s}"
        )
        for label in ["low", "med", "high"]:
            sub = caf_only[caf_only["dose_bin"] == label]
            if sub.empty:
                continue
            dose_range = (
                f"{sub['dose_caffeine'].min():.0f}-{sub['dose_caffeine'].max():.0f}mg"
            )
            tee_print(
                f"  {label:>10s} {dose_range:>15s} {sub['focus_pct'].mean():8.1f} "
                f"{sub['focused_hours'].mean():8.2f} {sub['duration_min'].mean():8.0f} "
                f"{len(sub):5d}"
            )

        # Continuous correlation
        r_focus, p_focus = stats.spearmanr(
            caf_only["dose_caffeine"], caf_only["focus_pct"]
        )
        r_fhrs, p_fhrs = stats.spearmanr(
            caf_only["dose_caffeine"], caf_only["focused_hours"]
        )
        r_dur, p_dur = stats.spearmanr(
            caf_only["dose_caffeine"], caf_only["duration_min"]
        )
        tee_print("\n  Spearman correlations with caffeine dose:")
        tee_print(f"    Focus %:       r={r_focus:+.3f}, p={p_focus:.4f}")
        tee_print(f"    Focused hours: r={r_fhrs:+.3f}, p={p_fhrs:.4f}")
        tee_print(f"    Session dur:   r={r_dur:+.3f}, p={p_dur:.4f}")

        # Scatter plot
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))

        for ax, (y_col, y_label) in zip(
            axes,
            [
                ("focus_pct", "Focus %"),
                ("focused_hours", "Focused Hours"),
                ("duration_min", "Session Duration (min)"),
            ],
        ):
            ax.scatter(
                caf_only["dose_caffeine"],
                caf_only[y_col],
                alpha=0.3,
                s=20,
                color="brown",
            )
            # Trend line
            z = np.polyfit(caf_only["dose_caffeine"], caf_only[y_col], 1)
            x_range = np.linspace(
                caf_only["dose_caffeine"].min(), caf_only["dose_caffeine"].max(), 100
            )
            ax.plot(x_range, np.polyval(z, x_range), "r--", linewidth=2)
            ax.set_xlabel("Caffeine dose (mg)")
            ax.set_ylabel(y_label)
            ax.set_title(f"Caffeine dose vs {y_label}")
            ax.grid(True, alpha=0.3)

        plt.suptitle("Caffeine Dose-Response (caffeine-only sessions)", fontsize=13)
        plt.tight_layout()
        plt.savefig(str(Path(__file__).parent / "caffeine_dose_response.png"), dpi=150)
        plt.close()
        tee_print("  Saved: caffeine_dose_response.png")

    # --- 8b: Adderall dose-response ---
    tee_print("\n  --- 8b: Adderall dose-response (adderall-only sessions) ---")
    add_only = sess_df[
        (sess_df["had_adderall"] == 1) & (sess_df["had_caffeine"] == 0)
    ].copy()
    tee_print(f"  Adderall-only sessions: {len(add_only)}")

    if len(add_only) > 20 and "dose_adderall" in add_only.columns:
        doses_a = add_only["dose_adderall"]
        tee_print(
            f"  Dose range: {doses_a.min():.0f} - {doses_a.max():.0f} mg, "
            f"median={doses_a.median():.0f}, mean={doses_a.mean():.0f}"
        )

        r_focus_a, p_focus_a = stats.spearmanr(
            add_only["dose_adderall"], add_only["focus_pct"]
        )
        r_fhrs_a, p_fhrs_a = stats.spearmanr(
            add_only["dose_adderall"], add_only["focused_hours"]
        )
        r_dur_a, p_dur_a = stats.spearmanr(
            add_only["dose_adderall"], add_only["duration_min"]
        )
        tee_print("\n  Spearman correlations with adderall dose:")
        tee_print(f"    Focus %:       r={r_focus_a:+.3f}, p={p_focus_a:.4f}")
        tee_print(f"    Focused hours: r={r_fhrs_a:+.3f}, p={p_fhrs_a:.4f}")
        tee_print(f"    Session dur:   r={r_dur_a:+.3f}, p={p_dur_a:.4f}")

    # --- 8c: Head-to-head caffeine-only vs adderall-only ---
    tee_print("\n  --- 8c: Caffeine-only vs Adderall-only (head-to-head) ---")
    if len(caf_only) >= 10 and len(add_only) >= 10:
        tee_print(
            f"  {'Metric':20s} {'Caffeine':>10s} {'Adderall':>10s} "
            f"{'Diff':>8s} {'p':>8s}"
        )
        for metric, label in [
            ("focus_pct", "Focus %"),
            ("focused_hours", "Focused hours"),
            ("duration_min", "Duration (min)"),
            ("n_distractions", "# Distractions"),
            ("total_dist_min", "Dist. minutes"),
        ]:
            c_vals = caf_only[metric]
            a_vals = add_only[metric]
            diff = c_vals.mean() - a_vals.mean()
            _, p = stats.mannwhitneyu(c_vals, a_vals, alternative="two-sided")
            sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else ""
            tee_print(
                f"  {label:20s} {c_vals.mean():10.1f} {a_vals.mean():10.1f} "
                f"{diff:+8.1f} {p:8.4f} {sig}"
            )
        tee_print(f"  n: caffeine-only={len(caf_only)}, adderall-only={len(add_only)}")

        # Also weekday-only head-to-head (fairer comparison)
        caf_wk = caf_only[caf_only["date"].dt.dayofweek < 5]
        add_wk = add_only[add_only["date"].dt.dayofweek < 5]
        if len(caf_wk) >= 10 and len(add_wk) >= 10:
            tee_print(
                f"\n  Weekday-only head-to-head "
                f"(n: caf={len(caf_wk)}, add={len(add_wk)}):"
            )
            tee_print(
                f"  {'Metric':20s} {'Caffeine':>10s} {'Adderall':>10s} "
                f"{'Diff':>8s} {'p':>8s}"
            )
            for metric, label in [
                ("focus_pct", "Focus %"),
                ("focused_hours", "Focused hours"),
                ("duration_min", "Duration (min)"),
            ]:
                c_vals = caf_wk[metric]
                a_vals = add_wk[metric]
                diff = c_vals.mean() - a_vals.mean()
                _, p = stats.mannwhitneyu(c_vals, a_vals, alternative="two-sided")
                sig = (
                    "***"
                    if p < 0.001
                    else "**" if p < 0.01 else "*" if p < 0.05 else ""
                )
                tee_print(
                    f"  {label:20s} {c_vals.mean():10.1f} {a_vals.mean():10.1f} "
                    f"{diff:+8.1f} {p:8.4f} {sig}"
                )

    # --- 8d: Day-level analysis with multiple outcome metrics ---
    # IMPORTANT: "Hours Working" from the daily summary is ALREADY focused time
    # (distractions are tracked separately in "Length Distractions").
    # So session-level focused_hours ≈ Hours Working. We verify this below.
    # The right outcomes to test are:
    #   - Hours Working (self-reported focused hours from daily summary)
    #   - work_productivity = Value * Hours Working (value-weighted focused hours)
    #   - total_duration_hours from sessions (total desk/butt-in-seat time)
    tee_print(
        "\n  --- 8d: Day-level outcomes (Hours Working, Value*Hours, desk time) ---"
    )

    # Aggregate sessions to day level
    day_sess = (
        sess_df.groupby("date")
        .agg(
            total_focused_hours=("focused_hours", "sum"),
            total_duration_hours=("duration_min", lambda x: x.sum() / 60),
            total_dist_min=("total_dist_min", "sum"),
            day_focus_pct=("focus_pct", "mean"),
            n_sessions=("date", "count"),
            had_caffeine=("had_caffeine", "max"),
            had_adderall=("had_adderall", "max"),
            had_modafinil=("had_modafinil", "max"),
            had_nicotine=("had_nicotine", "max"),
            dose_caffeine=("dose_caffeine", "max"),
            dose_adderall=("dose_adderall", "max"),
        )
        .reset_index()
    )

    stim_cols_day = ["had_caffeine", "had_adderall", "had_modafinil", "had_nicotine"]
    day_sess["had_any_stimulant"] = (day_sess[stim_cols_day].sum(axis=1) > 0).astype(
        int
    )

    # Merge with daily features (including Value and work_productivity)
    merge_cols = [
        c
        for c in [
            "date",
            "is_weekend",
            "day_of_week",
            "is_hive",
            "is_mats",
            "is_diesl",
            "sleep_hours",
            "work_start_hour",
            "Hours Working",
            "Value",
            "work_productivity",
            "Length Distractions",
            "prev_hours",
            "hours_rolling_7d_mean",
            "hours_rolling_30d_mean",
            "work_streak",
            "days_since_rest",
        ]
        if c in df.columns
    ]
    day_merged = day_sess.merge(df[merge_cols], on="date", how="left")

    # Diagnostic: verify Hours Working ≈ session focused_hours
    both_valid = day_merged.dropna(subset=["Hours Working", "total_focused_hours"])
    if len(both_valid) > 50:
        corr = both_valid["Hours Working"].corr(both_valid["total_focused_hours"])
        mean_hw = both_valid["Hours Working"].mean()
        mean_fh = both_valid["total_focused_hours"].mean()
        mean_desk = both_valid["total_duration_hours"].mean()
        tee_print("\n  Diagnostic — Hours Working vs session-derived metrics:")
        tee_print(
            f"    Hours Working (daily summary, self-reported focused): {mean_hw:.2f}h"
        )
        tee_print(
            f"    Session focused hours (duration - distractions):      {mean_fh:.2f}h"
        )
        tee_print(
            f"    Session total desk hours (start-to-end clock time):   {mean_desk:.2f}h"
        )
        tee_print(
            f"    Correlation (Hours Working vs session focused):      {corr:.3f}"
        )
        tee_print(
            "    → Hours Working is self-reported focused time; it's close to but "
            "not identical with session-derived focused hours."
        )

    # Compare outcomes across stimulant conditions
    tee_print(
        f"\n  {'Condition':20s} {'HrsWork':>8s} {'Value':>6s} {'V*H':>8s} "
        f"{'DeskHrs':>8s} {'Focus%':>7s} {'n':>5s}"
    )
    conditions = [
        ("Any stimulant", day_merged[day_merged["had_any_stimulant"] == 1]),
        ("No stimulant", day_merged[day_merged["had_any_stimulant"] == 0]),
        (
            "Caffeine only",
            day_merged[
                (day_merged["had_caffeine"] == 1) & (day_merged["had_adderall"] == 0)
            ],
        ),
        (
            "Adderall only",
            day_merged[
                (day_merged["had_adderall"] == 1) & (day_merged["had_caffeine"] == 0)
            ],
        ),
        (
            "Both caf+add",
            day_merged[
                (day_merged["had_caffeine"] == 1) & (day_merged["had_adderall"] == 1)
            ],
        ),
        (
            "Neither (wkday)",
            day_merged[
                (day_merged["had_any_stimulant"] == 0) & (day_merged["is_weekend"] == 0)
            ],
        ),
    ]
    for label, sub in conditions:
        if len(sub) < 3:
            continue
        hw = sub["Hours Working"].mean() if "Hours Working" in sub else float("nan")
        val = sub["Value"].mean() if "Value" in sub else float("nan")
        vh = (
            sub["work_productivity"].mean()
            if "work_productivity" in sub
            else float("nan")
        )
        desk = sub["total_duration_hours"].mean()
        fp = sub["day_focus_pct"].mean()
        tee_print(
            f"  {label:20s} {hw:8.2f} {val:6.1f} {vh:8.1f} "
            f"{desk:8.2f} {fp:7.1f} {len(sub):5d}"
        )

    # Controlled regressions across three outcome metrics
    controls = [
        c
        for c in [
            "is_weekend",
            "day_of_week",
            "is_hive",
            "is_mats",
            "is_diesl",
            "sleep_hours",
            "work_start_hour",
            "prev_hours",
            "hours_rolling_7d_mean",
            "hours_rolling_30d_mean",
            "work_streak",
            "days_since_rest",
        ]
        if c in day_merged.columns
    ]

    outcomes = [
        ("Hours Working", "Hours Working (self-reported focused)"),
        ("work_productivity", "Value * Hours (value-hours)"),
        ("total_duration_hours", "Total desk hours (session clock time)"),
    ]
    for outcome_col, outcome_label in outcomes:
        if outcome_col not in day_merged.columns:
            continue
        tee_print(f"\n  --- Controlled effect on {outcome_label} ---")
        for treatment, treat_label in [
            ("had_any_stimulant", "Any stimulant"),
            ("had_caffeine", "Caffeine"),
            ("had_adderall", "Adderall"),
        ]:
            cols = [outcome_col, treatment] + controls
            sub = day_merged[cols].dropna()
            if len(sub) < 30:
                continue
            X = sm.add_constant(sub[[treatment] + controls])
            y = sub[outcome_col]
            model = sm.OLS(y, X).fit()
            coef = model.params[treatment]
            p = model.pvalues[treatment]
            sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else ""
            tee_print(
                f"  {treat_label:20s}: {coef:+.2f} "
                f"(p={p:.4f}) R²={model.rsquared:.3f} {sig}"
            )

    # --- 8e: Visualization ---
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    # Bar chart: Hours Working and Value*Hours by condition
    ax = axes[0]
    bar_data = []
    for label, sub in conditions:
        if len(sub) >= 3 and "Hours Working" in sub and "work_productivity" in sub:
            bar_data.append(
                {
                    "condition": label,
                    "hours": sub["Hours Working"].mean(),
                    "se": sub["Hours Working"].std() / np.sqrt(len(sub)),
                    "value_hours": sub["work_productivity"].mean(),
                    "vh_se": (
                        sub["work_productivity"].dropna().std()
                        / np.sqrt(sub["work_productivity"].notna().sum())
                        if sub["work_productivity"].notna().sum() > 1
                        else 0
                    ),
                }
            )
    if bar_data:
        bd = pd.DataFrame(bar_data)
        y_pos = np.arange(len(bd))
        ax.barh(
            y_pos + 0.15,
            bd["hours"],
            height=0.3,
            xerr=bd["se"],
            alpha=0.7,
            color="steelblue",
            label="Hours Working",
        )
        ax.barh(
            y_pos - 0.15,
            bd["value_hours"],
            height=0.3,
            xerr=bd["vh_se"],
            alpha=0.7,
            color="tab:orange",
            label="Value * Hours",
        )
        ax.set_yticks(y_pos)
        ax.set_yticklabels(bd["condition"], fontsize=9)
        ax.set_xlabel("Mean / Day")
        ax.set_title("Hours Working & Value*Hours by Condition")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3, axis="x")

    # Caffeine dose vs value-hours (day-level)
    ax = axes[1]
    caf_days = day_merged[
        (day_merged["had_caffeine"] == 1) & (day_merged["dose_caffeine"] > 0)
    ].dropna(subset=["work_productivity"])
    if len(caf_days) > 20:
        ax.scatter(
            caf_days["dose_caffeine"],
            caf_days["work_productivity"],
            alpha=0.3,
            s=20,
            color="brown",
        )
        z = np.polyfit(caf_days["dose_caffeine"], caf_days["work_productivity"], 1)
        x_range = np.linspace(
            caf_days["dose_caffeine"].min(), caf_days["dose_caffeine"].max(), 100
        )
        ax.plot(x_range, np.polyval(z, x_range), "r--", linewidth=2)
        ax.set_xlabel("Caffeine dose (mg)")
        ax.set_ylabel("Value * Hours")
        ax.set_title("Caffeine Dose vs Value-Hours (day-level)")
        ax.grid(True, alpha=0.3)

    # Caffeine-only vs adderall-only box plot (value-hours)
    ax = axes[2]
    caf_days_only = day_merged[
        (day_merged["had_caffeine"] == 1) & (day_merged["had_adderall"] == 0)
    ].dropna(subset=["work_productivity"])
    add_days_only = day_merged[
        (day_merged["had_adderall"] == 1) & (day_merged["had_caffeine"] == 0)
    ].dropna(subset=["work_productivity"])
    neither_days = day_merged[day_merged["had_any_stimulant"] == 0].dropna(
        subset=["work_productivity"]
    )
    box_data = []
    box_labels = []
    if len(caf_days_only) >= 5:
        box_data.append(caf_days_only["work_productivity"].values)
        box_labels.append(f"Caffeine\n(n={len(caf_days_only)})")
    if len(add_days_only) >= 5:
        box_data.append(add_days_only["work_productivity"].values)
        box_labels.append(f"Adderall\n(n={len(add_days_only)})")
    if len(neither_days) >= 5:
        box_data.append(neither_days["work_productivity"].values)
        box_labels.append(f"None\n(n={len(neither_days)})")
    if box_data:
        bp = ax.boxplot(box_data, tick_labels=box_labels, patch_artist=True)
        colors = ["#d4a574", "#7fb3d4", "#b0b0b0"]
        for patch, color in zip(bp["boxes"], colors[: len(box_data)]):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
        ax.set_ylabel("Value * Hours Working")
        ax.set_title("Value-Hours Distribution")
        ax.grid(True, alpha=0.3, axis="y")

    plt.suptitle("Dose-Response & Head-to-Head Stimulant Comparison", fontsize=13)
    plt.tight_layout()
    plt.savefig(
        str(Path(__file__).parent / "stimulant_dose_response_focused_hours.png"),
        dpi=150,
    )
    plt.close()
    tee_print("  Saved: stimulant_dose_response_focused_hours.png")

    tee_print()


# ============================================================================
# ANALYSIS 9: ALL SUPPLEMENTS — WEEKDAYS AT REGIME ONLY
# ============================================================================
def all_supplements_weekday_regime() -> None:
    tee_print("=" * 70)
    tee_print("ANALYSIS 9: EVERY SUPPLEMENT EFFECT — WEEKDAYS AT HIVE/MATS/DIESL")
    tee_print("=" * 70)

    # Work with the day-level data, filtered to weekday regime days
    v = valid.copy()
    wk_regime = v[
        (v["is_weekend"] == 0)
        & ((v["is_hive"] == 1) | (v["is_mats"] == 1) | (v["is_diesl"] == 1))
    ].copy()
    tee_print(f"\n  Weekday regime days: {len(wk_regime)}")

    # Find all supplements with enough usage
    all_supp_names = sorted(set(SUPPLEMENT_ALIASES.values()))
    supp_cols = []
    for s in all_supp_names:
        col = f"{s}_any"
        if col in wk_regime.columns and (wk_regime[col] > 0).sum() >= 10:
            supp_cols.append((s, col))

    tee_print(f"  Supplements with >=10 usage days: {len(supp_cols)}")

    # Controls (no regime dummies needed since we're within-regime, but include
    # them anyway since the mix of Hive/MATS/Diesl varies by supplement)
    controls = [
        c
        for c in [
            "day_of_week",
            "is_hive",
            "is_mats",
            "is_diesl",
            "sleep_hours",
            "work_start_hour",
            "prev_hours",
            "hours_rolling_7d_mean",
            "hours_rolling_30d_mean",
            "work_streak",
            "days_since_rest",
        ]
        if c in wk_regime.columns
    ]

    # --- 9a: Raw with/without comparison ---
    tee_print("\n  --- Raw with/without (weekday regime days) ---")
    tee_print(
        f"  {'Supplement':18s} {'n_with':>6s} {'n_wo':>6s} "
        f"{'H_with':>7s} {'H_wo':>7s} {'diff':>6s} {'p_H':>7s}  "
        f"{'VH_with':>8s} {'VH_wo':>8s} {'diff':>6s} {'p_VH':>7s}"
    )
    raw_results: list[dict[str, object]] = []
    for supp_name, col in supp_cols:
        with_mask = wk_regime[col] > 0
        wo_mask = wk_regime[col] == 0
        with_h = wk_regime.loc[with_mask, "Hours Working"]
        wo_h = wk_regime.loc[wo_mask, "Hours Working"]
        n_with = len(with_h)
        n_wo = len(wo_h)
        if n_with < 10 or n_wo < 10:
            continue

        _, p_h = stats.mannwhitneyu(with_h, wo_h, alternative="two-sided")
        sig_h = (
            "***" if p_h < 0.001 else "**" if p_h < 0.01 else "*" if p_h < 0.05 else ""
        )

        with_vh = wk_regime.loc[with_mask, "work_productivity"].dropna()
        wo_vh = wk_regime.loc[wo_mask, "work_productivity"].dropna()
        if len(with_vh) >= 10 and len(wo_vh) >= 10:
            _, p_vh = stats.mannwhitneyu(with_vh, wo_vh, alternative="two-sided")
            sig_vh = (
                "***"
                if p_vh < 0.001
                else "**" if p_vh < 0.01 else "*" if p_vh < 0.05 else ""
            )
        else:
            p_vh = float("nan")
            sig_vh = ""

        diff_h = with_h.mean() - wo_h.mean()
        diff_vh = with_vh.mean() - wo_vh.mean() if len(with_vh) >= 10 else float("nan")

        tee_print(
            f"  {supp_name:18s} {n_with:6d} {n_wo:6d} "
            f"{with_h.mean():7.2f} {wo_h.mean():7.2f} {diff_h:+6.2f} {p_h:7.4f}{sig_h:3s} "
            f"{with_vh.mean():8.1f} {wo_vh.mean():8.1f} {diff_vh:+6.1f} {p_vh:7.4f}{sig_vh:3s}"
        )
        raw_results.append(
            {
                "supplement": supp_name,
                "n_with": n_with,
                "n_wo": n_wo,
                "diff_h": diff_h,
                "p_h": p_h,
                "diff_vh": diff_vh,
                "p_vh": p_vh,
            }
        )

    # --- 9b: Controlled regressions (each supplement one at a time) ---
    tee_print(
        f"\n  --- Controlled regressions (each supplement individually, "
        f"{len(controls)} controls) ---"
    )
    tee_print(
        f"  {'Supplement':18s} {'HrsWork coef':>12s} {'p':>7s}   "
        f"{'V*H coef':>10s} {'p':>7s}   {'n':>5s}"
    )

    controlled_results: list[dict[str, object]] = []
    for supp_name, col in supp_cols:
        # Hours Working
        cols_h = ["Hours Working", col] + controls
        sub_h = wk_regime[cols_h].dropna()
        if len(sub_h) < 30 or (sub_h[col] > 0).sum() < 10:
            continue
        X_h = sm.add_constant(sub_h[[col] + controls])
        model_h = sm.OLS(sub_h["Hours Working"], X_h).fit()
        coef_h = model_h.params[col]
        p_h = model_h.pvalues[col]
        sig_h = (
            "***" if p_h < 0.001 else "**" if p_h < 0.01 else "*" if p_h < 0.05 else ""
        )

        # Value * Hours
        coef_vh = float("nan")
        p_vh = float("nan")
        sig_vh = ""
        if "work_productivity" in wk_regime.columns:
            cols_vh = ["work_productivity", col] + controls
            sub_vh = wk_regime[cols_vh].dropna()
            if len(sub_vh) >= 30 and (sub_vh[col] > 0).sum() >= 10:
                X_vh = sm.add_constant(sub_vh[[col] + controls])
                model_vh = sm.OLS(sub_vh["work_productivity"], X_vh).fit()
                coef_vh = model_vh.params[col]
                p_vh = model_vh.pvalues[col]
                sig_vh = (
                    "***"
                    if p_vh < 0.001
                    else "**" if p_vh < 0.01 else "*" if p_vh < 0.05 else ""
                )

        tee_print(
            f"  {supp_name:18s} {coef_h:+12.3f} {p_h:7.4f}{sig_h:3s} "
            f"{coef_vh:+10.2f} {p_vh:7.4f}{sig_vh:3s} {len(sub_h):5d}"
        )
        controlled_results.append(
            {
                "supplement": supp_name,
                "coef_h": coef_h,
                "p_h": p_h,
                "coef_vh": coef_vh,
                "p_vh": p_vh,
                "n": len(sub_h),
            }
        )

    # --- 9c: Dose-response for each supplement with enough range ---
    tee_print("\n  --- Dose-response (Spearman, within users of each supplement) ---")
    tee_print(
        f"  {'Supplement':18s} {'n':>5s} {'dose range':>15s} "
        f"{'r(H)':>7s} {'p':>7s}   {'r(V*H)':>7s} {'p':>7s}"
    )
    for supp_name, col in supp_cols:
        dose_col = supp_name  # raw dose column
        if dose_col not in wk_regime.columns:
            continue
        users = wk_regime[wk_regime[dose_col] > 0].dropna(subset=["Hours Working"])
        if len(users) < 15:
            continue
        doses = users[dose_col]
        if doses.nunique() < 3:
            continue

        r_h, p_h = stats.spearmanr(doses, users["Hours Working"])
        sig_h = (
            "***" if p_h < 0.001 else "**" if p_h < 0.01 else "*" if p_h < 0.05 else ""
        )

        vh_users = users.dropna(subset=["work_productivity"])
        if len(vh_users) >= 15:
            r_vh, p_vh = stats.spearmanr(
                vh_users[dose_col], vh_users["work_productivity"]
            )
            sig_vh = (
                "***"
                if p_vh < 0.001
                else "**" if p_vh < 0.01 else "*" if p_vh < 0.05 else ""
            )
        else:
            r_vh, p_vh, sig_vh = float("nan"), float("nan"), ""

        dose_range = f"{doses.min():.0f}-{doses.max():.0f}"
        tee_print(
            f"  {supp_name:18s} {len(users):5d} {dose_range:>15s} "
            f"{r_h:+7.3f} {p_h:7.4f}{sig_h:3s} {r_vh:+7.3f} {p_vh:7.4f}{sig_vh:3s}"
        )

    # --- 9d: Summary visualization ---
    if controlled_results:
        cr = pd.DataFrame(controlled_results).sort_values("coef_h")
        fig, axes = plt.subplots(1, 2, figsize=(14, max(6, len(cr) * 0.5)))

        ax = axes[0]
        colors_h = [
            (
                "tab:green"
                if p < 0.05 and c > 0
                else "tab:red" if p < 0.05 and c < 0 else "gray"
            )
            for c, p in zip(cr["coef_h"], cr["p_h"])
        ]
        ax.barh(range(len(cr)), cr["coef_h"], color=colors_h, alpha=0.7)
        ax.set_yticks(range(len(cr)))
        ax.set_yticklabels(cr["supplement"], fontsize=9)
        ax.set_xlabel("Controlled effect (hours)")
        ax.set_title("Effect on Hours Working\n(weekday regime, controlled)")
        ax.axvline(0, color="black", linewidth=0.5)
        ax.grid(True, alpha=0.3, axis="x")

        ax = axes[1]
        cr_vh = cr.dropna(subset=["coef_vh"])
        if not cr_vh.empty:
            colors_vh = [
                (
                    "tab:green"
                    if p < 0.05 and c > 0
                    else "tab:red" if p < 0.05 and c < 0 else "gray"
                )
                for c, p in zip(cr_vh["coef_vh"], cr_vh["p_vh"])
            ]
            ax.barh(range(len(cr_vh)), cr_vh["coef_vh"], color=colors_vh, alpha=0.7)
            ax.set_yticks(range(len(cr_vh)))
            ax.set_yticklabels(cr_vh["supplement"], fontsize=9)
            ax.set_xlabel("Controlled effect (value-hours)")
            ax.set_title("Effect on Value * Hours\n(weekday regime, controlled)")
            ax.axvline(0, color="black", linewidth=0.5)
            ax.grid(True, alpha=0.3, axis="x")

        plt.suptitle(
            "All Supplements — Controlled Effects (Weekday Regime Days Only)",
            fontsize=13,
        )
        plt.tight_layout()
        plt.savefig(
            str(Path(__file__).parent / "all_supplements_controlled_effects.png"),
            dpi=150,
        )
        plt.close()
        tee_print("  Saved: all_supplements_controlled_effects.png")

    tee_print()


# ============================================================================
# RUN ALL
# ============================================================================
if __name__ == "__main__":
    supplement_distraction_analysis()
    diminishing_returns_analysis()
    supplement_interactions()
    event_level_prediction()
    text_field_mining()
    comprehensive_regularized_model()
    weekly_project_analysis()
    dose_response_and_focused_hours()
    all_supplements_weekday_regime()

    tee_print("=" * 70)
    tee_print("ALL ANALYSES COMPLETE")
    tee_print("=" * 70)

    # Write summary
    summary_path = Path(__file__).parent / "DEEPER_ANALYSIS_SUMMARY.md"
    summary_path.write_text(
        "# Deeper Analysis Summary\n\n"
        "Generated by `deeper_analysis.py`\n\n"
        "```\n" + output_buffer.getvalue() + "\n```\n"
    )
    tee_print(f"\nSaved summary to {summary_path}")
