# %% use conda env: side_projects
"""1. Monthly average prediction error for the full multivariate and actionable models.
2. Random-week holdout R² table across regimes (Hive/Mats/diesl) and test fractions."""

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm

from productivity_analysis import (
    BASE_DIR,
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

OUT_DIR = BASE_DIR

# ---------------------------------------------------------------------------
# Build dataset (same pipeline as controlled_analysis.py + nested_model_analysis.py)
# ---------------------------------------------------------------------------
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
for col in ["cal_green", "cal_things", "cal_waste", "cal_blue"]:
    df[col] = df[col].fillna(0)

# Key event daily totals
event_daily = (
    cal_full.groupby(["date", "event_lower"])["duration"].sum().unstack(fill_value=0)
)
key_events = [
    "amelia",
    "gym",
    "walk",
    "nap",
    "meditation",
]
for ev in key_events:
    col_name = f"cal_{ev.replace(' ', '_')}"
    if ev in event_daily.columns:
        df = df.merge(
            event_daily[[ev]].rename(columns={ev: col_name}),
            left_on="date",
            right_index=True,
            how="left",
        )
        df[col_name] = df[col_name].fillna(0)

# Regime dummies
df["is_hive"] = (df["regime"] == "Hive").astype(int)
df["is_mats"] = (df["regime"] == "Mats").astype(int)
df["is_diesl"] = (df["regime"] == "diesl").astype(int)

# Supplements binary
for supplement in set(SUPPLEMENT_ALIASES.values()):
    if supplement in df.columns:
        df[f"has_{supplement}"] = (df[supplement] > 0).astype(int)

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

# Rolling features
df["hours_rolling_7d_mean"] = (
    df["Hours Working"].rolling(7, min_periods=3).mean().shift(1)
)
df["hours_rolling_7d_std"] = (
    df["Hours Working"].rolling(7, min_periods=3).std().shift(1)
)

# Days since rest
rest_day = (df["Hours Working"] < 2).astype(int)
dsr = rest_day.copy()
for i in range(1, len(dsr)):
    dsr.iloc[i] = dsr.iloc[i - 1] + 1 if rest_day.iloc[i] == 0 else 0
df["days_since_rest"] = dsr

# Early Hive: structurally higher output period (Jun 2021 – Jul 2022)
df["is_early_hive"] = (
    (df["regime"] == "Hive") & (df["date"] < pd.Timestamp("2022-08-01"))
).astype(int)

# Filter to days with work
valid = df[df["Hours Working"] > 0].copy()

# ---------------------------------------------------------------------------
# Define both models
# ---------------------------------------------------------------------------
# Full multivariate model (R²≈0.639)
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

full_base = [
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
for supp in supp_cols:
    if (valid[supp] > 0).sum() > 30:
        valid[f"has_{supp}"] = (valid[supp] > 0).astype(int)
        full_base.append(f"has_{supp}")
for col_name in ["cal_gym", "cal_walk"]:
    if col_name in valid.columns:
        full_base.append(col_name)
if "has_meditation" in valid.columns or "cal_meditation" in valid.columns:
    med_col = (
        "cal_meditation" if "cal_meditation" in valid.columns else "has_meditation"
    )
    valid["has_meditation"] = (valid[med_col] > 0).astype(int)
    full_base.append("has_meditation")

full_base = [
    f for f in full_base if f in valid.columns and valid[f].notna().sum() > 100
]

# Actionable model base — only pre-work features
actionable_base = [
    "day_of_week",
    "hours_rolling_7d_mean",
    "hours_rolling_7d_std",
    "prev_hours",
    "days_since_rest",
    "caffeine_any",
    "adderall_any",
    "modafinil_any",
    "work_start_hour",
]
# Ensure caffeine_any etc. exist
for supp in ["caffeine", "adderall", "modafinil", "nicotine"]:
    col_name = f"{supp}_any"
    if col_name not in valid.columns and supp in valid.columns:
        valid[col_name] = (valid[supp] > 0).astype(int)

actionable_base = [
    f for f in actionable_base if f in valid.columns and valid[f].notna().sum() > 100
]

# Build model variants: original, + early_hive, + years_since_2020, + both
full_features = list(full_base)
actionable_features = list(actionable_base)

full_plus_earlyhive = list(full_base) + ["is_early_hive"]
actionable_plus_earlyhive = list(actionable_base) + ["is_early_hive"]

TARGET = "Hours Working"

# ---------------------------------------------------------------------------
# Part 1: Monthly average error plot — compare original vs +trend vs +early_hive
# ---------------------------------------------------------------------------
print("\n" + "=" * 70)
print("PART 1: Monthly prediction error")
print("=" * 70)

# All model variants to compare
ALL_MODELS: list[tuple[str, list[str]]] = [
    ("Full", full_features),
    ("Full + early_hive", full_plus_earlyhive),
    ("Actionable", actionable_features),
    ("Actionable + early_hive", actionable_plus_earlyhive),
]

# First: print R² comparison table for all variants
print("\n--- Model R² Comparison ---")
print(f"  {'Model':<35s}  {'R²':>6s}  {'adjR²':>6s}  {'n':>5s}  {'new coefs':>30s}")
for model_name, features in ALL_MODELS:
    sample = valid[features + [TARGET]].dropna()
    X = sm.add_constant(sample[features])
    y = sample[TARGET]
    m = sm.OLS(y, X).fit()
    # Show coefficients for the new features
    new_feats = [f for f in ["is_early_hive", "years_since_2020"] if f in features]
    coef_str = "  ".join(
        f"{f}={m.params[f]:+.3f}(p={m.pvalues[f]:.3f})" for f in new_feats
    )
    print(
        f"  {model_name:<35s}  {m.rsquared:.4f}  {m.rsquared_adj:.4f}  "
        f"{int(m.nobs):5d}  {coef_str}"
    )

# Plot: 2 rows (Full, Actionable) x 2 cols (original, +both)
plot_pairs: list[tuple[str, list[str]]] = [
    ("Full Multivariate", full_features),
    ("Full + early_hive", full_plus_earlyhive),
    ("Actionable (morning)", actionable_features),
    ("Actionable + early_hive", actionable_plus_earlyhive),
]

fig, axes = plt.subplots(2, 2, figsize=(18, 10), sharex=True)

for idx, (model_name, features) in enumerate(plot_pairs):
    ax = axes[idx // 2, idx % 2]
    sample = valid[features + [TARGET, "date"]].dropna().copy()
    X = sm.add_constant(sample[features])
    y = sample[TARGET]
    model = sm.OLS(y, X).fit()

    sample["predicted"] = model.predict(X)
    sample["error"] = sample["predicted"] - y
    sample["abs_error"] = sample["error"].abs()
    sample["year_month"] = sample["date"].dt.to_period("M")

    monthly = sample.groupby("year_month").agg(
        mean_error=("error", "mean"),
        mae=("abs_error", "mean"),
        rmse=("error", lambda x: np.sqrt((x**2).mean())),
        mean_hours=(TARGET, "mean"),
        n=("error", "count"),
    )

    x_labels = [str(p) for p in monthly.index]
    x_pos = np.arange(len(x_labels))

    ax.bar(
        x_pos,
        monthly["mean_error"],
        alpha=0.6,
        color="steelblue",
        label="Mean Error (bias)",
    )
    ax.plot(x_pos, monthly["mae"], color="coral", lw=2, marker="o", ms=3, label="MAE")
    ax.plot(
        x_pos,
        monthly["rmse"],
        color="darkred",
        lw=2,
        marker="s",
        ms=3,
        label="RMSE",
    )
    ax.axhline(0, color="black", lw=0.8, ls="--")
    ax.set_ylabel("Hours")
    ax.set_title(f"{model_name} (R²={model.rsquared:.3f})", fontsize=10)
    ax.legend(fontsize=7)

    if idx >= 2:  # bottom row
        ax.set_xticks(x_pos[::6])
        ax.set_xticklabels(
            [x_labels[i] for i in range(0, len(x_labels), 6)],
            rotation=45,
            ha="right",
            fontsize=7,
        )
        ax.set_xlabel("Month")

    # Print monthly bias for the +both variants
    if "both" in model_name or model_name in (
        "Full Multivariate",
        "Actionable (morning)",
    ):
        print(f"\n{model_name}: R²={model.rsquared:.3f}, n={len(sample)}")
        print(
            f"  {'Month':>8s}  {'Bias':>6s}  {'MAE':>5s}  "
            f"{'RMSE':>5s}  {'Mean h':>6s}  {'n':>4s}"
        )
        for period, row in monthly.iterrows():
            print(
                f"  {str(period):>8s}  {row['mean_error']:+6.2f}  {row['mae']:5.2f}  "
                f"{row['rmse']:5.2f}  {row['mean_hours']:6.2f}  {int(row['n']):4d}"
            )

fig.suptitle(
    "Monthly Prediction Error: Original vs + early_hive + years_since_2020",
    fontsize=13,
    y=1.01,
)
fig.tight_layout()
out_path = OUT_DIR / "model_monthly_error.png"
fig.savefig(out_path, dpi=150, bbox_inches="tight")
print(f"\nSaved to {out_path}")
plt.close(fig)


# ---------------------------------------------------------------------------
# Part 2: Random-week holdout R² by regime
# ---------------------------------------------------------------------------
print("\n" + "=" * 70)
print("PART 2: Random-week holdout R² by regime")
print("=" * 70)

REGIMES = [
    ("Hive", "is_hive"),
    ("Mats", "is_mats"),
    ("diesl", "is_diesl"),
]
TEST_FRACTIONS = [0.10, 0.20, 0.30, 0.40, 0.50]
N_REPEATS = 200


def holdout_cv(
    data: pd.DataFrame,
    features: list[str],
    test_frac: float,
    n_repeats: int,
    rng: np.random.Generator,
) -> dict[str, float]:
    """Random-week holdout CV. Returns mean R² and MAE across repeats."""
    # Drop regime dummies and other constant columns within this subset
    use_features = [
        f
        for f in features
        if f in data.columns and data[f].notna().sum() > 5 and data[f].std() > 0.01
    ]
    sample = data[use_features + [TARGET, "date"]].dropna().copy()
    sample["week"] = (
        sample["date"].dt.isocalendar().week.astype(int) + sample["date"].dt.year * 100
    )
    weeks = sample["week"].unique()

    if len(weeks) < 4:
        return {"r2": np.nan, "mae": np.nan, "n_weeks": len(weeks)}

    n_test = max(1, int(len(weeks) * test_frac))
    if n_test >= len(weeks) - 2:
        return {"r2": np.nan, "mae": np.nan, "n_weeks": len(weeks)}

    r2s: list[float] = []
    maes: list[float] = []
    for _ in range(n_repeats):
        test_weeks = rng.choice(weeks, size=n_test, replace=False)
        test_mask = sample["week"].isin(test_weeks)
        train = sample[~test_mask]
        test = sample[test_mask]

        if len(train) < len(use_features) + 5 or len(test) < 3:
            continue

        # Drop any features that are constant in the training split
        split_features = [f for f in use_features if train[f].std() > 0.01]
        if len(split_features) < 2:
            continue

        train_X = train[split_features].copy()
        test_X = test[split_features].copy()
        train_X.insert(0, "const", 1.0)
        test_X.insert(0, "const", 1.0)
        y_train = train[TARGET]
        y_test = test[TARGET]

        try:
            model = sm.OLS(y_train, train_X).fit()
            y_pred = model.predict(test_X)
        except Exception:
            continue

        ss_res = ((y_test - y_pred) ** 2).sum()
        ss_tot = ((y_test - y_test.mean()) ** 2).sum()
        r2 = 1 - ss_res / ss_tot if ss_tot > 0 else np.nan
        mae = (y_test - y_pred).abs().mean()

        if not np.isnan(r2):
            r2s.append(r2)
            maes.append(mae)

    if not r2s:
        return {"r2": np.nan, "mae": np.nan, "n_weeks": len(weeks)}

    return {
        "r2": np.mean(r2s),
        "mae": np.mean(maes),
        "n_weeks": len(weeks),
        "r2_std": np.std(r2s),
        "mae_std": np.std(maes),
    }


rng = np.random.default_rng(42)


def trim_regime(data: pd.DataFrame, regime_name: str) -> pd.DataFrame:
    """Select 20-80% date range for a regime to avoid transition effects."""
    regime_data = data[data["regime"] == regime_name].copy()
    dates = regime_data["date"].sort_values()
    n_days = len(dates)
    lo_date = dates.iloc[int(n_days * 0.20)]
    hi_date = dates.iloc[int(n_days * 0.80)]
    return regime_data[
        (regime_data["date"] >= lo_date) & (regime_data["date"] <= hi_date)
    ].copy()


def print_holdout_table(
    model_name: str,
    features: list[str],
    regimes: list[tuple[str, str]],
    data: pd.DataFrame,
) -> None:
    """Print compact R² and MAE holdout table for one model variant."""
    frac_labels = [f"{int(f * 100)}%" for f in TEST_FRACTIONS]

    # R² table
    print(f"\n  {model_name}:")
    print(f"  {'Regime':>8s}  {'n_d':>4s} {'n_wk':>5s}", end="")
    for fl in frac_labels:
        print(f"  {'R²@' + fl:>12s}", end="")
    print()

    for regime_name, _ in regimes:
        trimmed = trim_regime(data, regime_name)
        print(f"  {regime_name:>8s}  {len(trimmed):4d}", end="")
        n_wk_str = ""
        parts = []
        for test_frac in TEST_FRACTIONS:
            result = holdout_cv(trimmed, features, test_frac, N_REPEATS, rng)
            if n_wk_str == "":
                n_wk_str = f" {result['n_weeks']:4d} "
            if not np.isnan(result["r2"]):
                parts.append(f"  {result['r2']:+.3f}±{result['r2_std']:.2f}")
            else:
                parts.append(f"  {'N/A':>12s}")
        print(n_wk_str, end="")
        for p in parts:
            print(p, end="")
        print()

    # MAE table
    print(f"  {'':>8s}  {'':>4s} {'':>5s}", end="")
    for fl in frac_labels:
        print(f"  {'MAE@' + fl:>12s}", end="")
    print()

    for regime_name, _ in regimes:
        trimmed = trim_regime(data, regime_name)
        print(f"  {regime_name:>8s}  {len(trimmed):4d}", end="")
        n_wk_str = ""
        parts = []
        for test_frac in TEST_FRACTIONS:
            result = holdout_cv(trimmed, features, test_frac, N_REPEATS, rng)
            if n_wk_str == "":
                n_wk_str = f" {result['n_weeks']:4d} "
            if not np.isnan(result["mae"]):
                parts.append(f"  {result['mae']:.2f}±{result['mae_std']:.2f}")
            else:
                parts.append(f"  {'N/A':>12s}")
        print(n_wk_str, end="")
        for p in parts:
            print(p, end="")
        print()


# Models to test in holdout
CV_MODELS: list[tuple[str, list[str]]] = [
    ("Full", full_features),
    ("Full + early_hive", full_plus_earlyhive),
    ("Actionable", actionable_features),
    ("Actionable + early_hive", actionable_plus_earlyhive),
]

for model_name, features in CV_MODELS:
    print_holdout_table(model_name, features, REGIMES, valid)
