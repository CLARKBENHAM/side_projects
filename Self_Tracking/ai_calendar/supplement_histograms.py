# %% use conda env: side_projects
"""Two-column plot grid: for each supplement, show (1) histogram of Hours Working
with/without, and (2) histogram of residuals from a baseline model with/without."""
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm
from pathlib import Path

# ---------------------------------------------------------------------------
# Load data
# ---------------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "microdata_daily_features.csv"
df = pd.read_csv(DATA_PATH, parse_dates=["date"])

SUPPLEMENTS: list[tuple[str, str]] = [
    ("caffeine_any", "Caffeine"),
    ("adderall_any", "Adderall"),
    ("nicotine_any", "Nicotine"),
    ("piracetam_any", "Piracetam"),
    ("choline_any", "Choline"),
    ("ashwagandha_any", "Ashwagandha"),
    ("creatine_any", "Creatine"),
    ("sulbutiamine_any", "Sulbutiamine"),
    ("modafinil_any", "Modafinil"),
    ("bronkaid_any", "Bronkaid"),
    ("lavender_any", "Lavender"),
    ("potassium_gluconate_any", "Potassium Gluconate"),
    ("zinc_any", "Zinc"),
    ("zembrin_any", "Zembrin"),
]

BASE_CONTROLS = [
    "regime_hive",
    "regime_mats",
    "regime_diesl",
    "is_weekend",
    "day_of_week",
]
OPTIONAL_FEATURES = [
    "calendar_waste_hours",
    "nap_hours",
    "wake_hour",
    "sleep_hours",
    "prev_hours",
]

OUT_DIR = BASE_DIR

df["week_start"] = df["date"] - pd.to_timedelta(df["date"].dt.weekday, unit="D")
weekly_hours = (
    df.groupby("week_start", as_index=False)["Hours Working"]
    .sum()
    .rename(columns={"Hours Working": "weekly_hours"})
)
df = df.merge(weekly_hours, on="week_start", how="left")


def plot_threshold(sub: pd.DataFrame, label_suffix: str, suffix: str) -> None:
    print(f"\n{'='*60}\nFilter: {label_suffix}  ({len(sub)} days)")

    # Supplements with enough data at this threshold
    supplements_to_plot: list[tuple[str, str]] = []
    for col, slabel in SUPPLEMENTS:
        if col in sub.columns and sub[col].sum() >= 10:
            supplements_to_plot.append((col, slabel))

    # Baseline model
    features = BASE_CONTROLS.copy()
    for feat in OPTIONAL_FEATURES:
        if feat in sub.columns and sub[feat].notna().sum() > 100:
            features.append(feat)

    model_df = sub[["Hours Working"] + features].dropna().copy()
    X = sm.add_constant(model_df[features])
    y = model_df["Hours Working"]
    baseline_model = sm.OLS(y, X).fit()
    print(f"  Baseline R²={baseline_model.rsquared:.3f}, n={len(model_df)}")

    sub.loc[model_df.index, "residual"] = baseline_model.resid

    # Plot
    n_rows = len(supplements_to_plot)
    if n_rows == 0:
        print("  No supplements with enough data, skipping.")
        return
    fig, axes = plt.subplots(n_rows, 2, figsize=(14, 3.2 * n_rows))
    if n_rows == 1:
        axes = axes.reshape(1, -1)

    for i, (col, slabel) in enumerate(supplements_to_plot):
        ax_hist = axes[i, 0]
        ax_resid = axes[i, 1]

        mask_on = sub[col] == 1
        mask_off = sub[col] == 0
        n_on = mask_on.sum()
        n_off = mask_off.sum()

        hours_on = sub.loc[mask_on, "Hours Working"].dropna()
        hours_off = sub.loc[mask_off, "Hours Working"].dropna()

        # -- Left: Hours Working histograms --
        bin_floor = max(0, sub["Hours Working"].min())
        bins = np.linspace(
            bin_floor, sub["Hours Working"].quantile(0.99), 30
        )
        ax_hist.hist(
            hours_off,
            bins=bins,
            alpha=0.5,
            density=True,
            label=f"Without (n={n_off})",
            color="steelblue",
            edgecolor="white",
        )
        ax_hist.hist(
            hours_on,
            bins=bins,
            alpha=0.5,
            density=True,
            label=f"With (n={n_on})",
            color="coral",
            edgecolor="white",
        )
        ax_hist.axvline(hours_off.mean(), color="steelblue", ls="--", lw=1.5)
        ax_hist.axvline(hours_on.mean(), color="coral", ls="--", lw=1.5)
        diff = hours_on.mean() - hours_off.mean()
        ax_hist.set_title(f"{slabel}: Hours Working  (Δ={diff:+.2f}h)")
        ax_hist.set_xlabel("Hours Working")
        ax_hist.set_ylabel("Density")
        ax_hist.legend(fontsize=8)

        # -- Right: Residual histograms --
        resid_on = sub.loc[mask_on, "residual"].dropna()
        resid_off = sub.loc[mask_off, "residual"].dropna()

        if len(resid_on) > 5 and len(resid_off) > 5:
            rbins = np.linspace(
                sub["residual"].dropna().quantile(0.01),
                sub["residual"].dropna().quantile(0.99),
                30,
            )
            ax_resid.hist(
                resid_off,
                bins=rbins,
                alpha=0.5,
                density=True,
                label=f"Without (n={len(resid_off)})",
                color="steelblue",
                edgecolor="white",
            )
            ax_resid.hist(
                resid_on,
                bins=rbins,
                alpha=0.5,
                density=True,
                label=f"With (n={len(resid_on)})",
                color="coral",
                edgecolor="white",
            )
            ax_resid.axvline(
                resid_off.mean(), color="steelblue", ls="--", lw=1.5
            )
            ax_resid.axvline(
                resid_on.mean(), color="coral", ls="--", lw=1.5
            )
            rdiff = resid_on.mean() - resid_off.mean()
            ax_resid.set_title(f"{slabel}: Residuals  (Δ={rdiff:+.2f}h)")
            ax_resid.set_xlabel("Residual (Hours Working)")
            ax_resid.set_ylabel("Density")
            ax_resid.legend(fontsize=8)
        else:
            ax_resid.text(
                0.5,
                0.5,
                "Insufficient data",
                ha="center",
                va="center",
                transform=ax_resid.transAxes,
            )
            ax_resid.set_title(f"{slabel}: Residuals")

    fig.suptitle(
        f"Supplement Effects on Hours Working (days {label_suffix} only)\n"
        f"Baseline controls: {', '.join(features)}  (R²={baseline_model.rsquared:.3f})",
        fontsize=13,
        y=1.01,
    )
    fig.tight_layout()

    out_path = OUT_DIR / f"supplement_hours_histograms{suffix}.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"  Saved to {out_path}")
    plt.close(fig)


for min_hours in [0, 2, 3, 4]:
    sub = df[df["Hours Working"] > min_hours].copy()
    label_suffix = f">{min_hours}h" if min_hours > 0 else ">0h"
    suffix = f"_{min_hours}h" if min_hours > 0 else ""
    plot_threshold(sub=sub, label_suffix=label_suffix, suffix=suffix)

for weekly_min_hours in [20, 30]:
    sub = df[df["weekly_hours"] > weekly_min_hours].copy()
    label_suffix = f"weeks >{weekly_min_hours}h"
    suffix = f"_week_gt{weekly_min_hours}h"
    plot_threshold(sub=sub, label_suffix=label_suffix, suffix=suffix)
