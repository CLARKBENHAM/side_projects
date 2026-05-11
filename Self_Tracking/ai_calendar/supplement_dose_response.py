# %% use conda env: side_projects
"""Dose-response scatter plots: x=dosage, y=hours worked, with LOESS trend line.
Filtered by weekly hours thresholds."""
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.stats import spearmanr

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "microdata_daily_features.csv"
OUT_DIR = BASE_DIR

df = pd.read_csv(DATA_PATH, parse_dates=["date"])
df = df[df["Hours Working"] > 0].copy()

df["week_start"] = df["date"] - pd.to_timedelta(df["date"].dt.weekday, unit="D")
weekly_hours = (
    df.groupby("week_start", as_index=False)["Hours Working"]
    .sum()
    .rename(columns={"Hours Working": "weekly_hours"})
)
df = df.merge(weekly_hours, on="week_start", how="left")

# Supplements with actual dose variance
DOSE_SUPPLEMENTS: list[tuple[str, str, str]] = [
    ("caffeine", "caffeine_any", "Caffeine (mg)"),
    ("adderall", "adderall_any", "Adderall (mg)"),
    ("modafinil", "modafinil_any", "Modafinil (mg)"),
    ("nicotine", "nicotine_any", "Nicotine (mg)"),
]

FILTERS: list[tuple[str, str]] = [
    ("All days (>0h)", ""),
    ("Weeks >20h", "_week_gt20h"),
    ("Weeks >30h", "_week_gt30h"),
]


def lowess_smooth(
    x: np.ndarray, y: np.ndarray, frac: float = 0.4, n_points: int = 50
) -> tuple[np.ndarray, np.ndarray]:
    """Simple binned mean as a robust smoother."""
    x_grid = np.linspace(x.min(), x.max(), n_points)
    y_smooth = np.full_like(x_grid, np.nan)
    bw = (x.max() - x.min()) * frac / 2
    for i, xg in enumerate(x_grid):
        mask = np.abs(x - xg) <= bw
        if mask.sum() >= 3:
            y_smooth[i] = np.mean(y[mask])
    valid = ~np.isnan(y_smooth)
    return x_grid[valid], y_smooth[valid]


for filter_label, suffix in FILTERS:
    if "20h" in filter_label:
        sub = df[df["weekly_hours"] > 20].copy()
    elif "30h" in filter_label:
        sub = df[df["weekly_hours"] > 30].copy()
    else:
        sub = df.copy()

    # Only plot supplements with >=10 usage days in this subset
    supps_to_plot = []
    for dose_col, any_col, label in DOSE_SUPPLEMENTS:
        if dose_col in sub.columns:
            usage = sub[sub[dose_col] > 0]
            if len(usage) >= 10 and usage[dose_col].nunique() >= 3:
                supps_to_plot.append((dose_col, any_col, label))

    if not supps_to_plot:
        print(f"  {filter_label}: no supplements with enough dose data, skipping.")
        continue

    n_rows = len(supps_to_plot)
    fig, axes = plt.subplots(n_rows, 2, figsize=(14, 3.5 * n_rows))
    if n_rows == 1:
        axes = axes.reshape(1, -1)

    for i, (dose_col, any_col, label) in enumerate(supps_to_plot):
        ax_raw = axes[i, 0]
        ax_with_zero = axes[i, 1]

        usage = sub[sub[dose_col] > 0].copy()
        dose = usage[dose_col].values
        hours = usage["Hours Working"].values

        # Left: scatter among users only (dose > 0)
        ax_raw.scatter(dose, hours, alpha=0.3, s=20, color="coral", edgecolors="none")

        # Spearman correlation
        r, p = spearmanr(dose, hours)

        # Smoothed trend
        if len(dose) >= 10:
            xs, ys = lowess_smooth(dose, hours)
            if len(xs) > 1:
                ax_raw.plot(xs, ys, color="darkred", lw=2.5, label="Smoothed mean")

        ax_raw.set_xlabel(label)
        ax_raw.set_ylabel("Hours Working")
        ax_raw.set_title(
            f"{label.split('(')[0].strip()}: Dose → Hours  (r={r:.2f}, p={p:.3f}, n={len(dose)})"
        )
        ax_raw.legend(fontsize=8)

        # Right: include zero-dose as reference, show means by bin
        no_use = sub[sub[dose_col] == 0]["Hours Working"]
        all_doses = np.concatenate([[0], np.sort(dose[dose > 0])])
        unique_doses = np.sort(np.unique(all_doses))

        if len(unique_doses) > 8:
            # Bin into quantiles for readability
            usage_only = sub[sub[dose_col] > 0].copy()
            n_bins = min(5, usage_only[dose_col].nunique())
            usage_only["dose_bin"] = pd.qcut(
                usage_only[dose_col], q=n_bins, duplicates="drop"
            )
            bin_stats = usage_only.groupby("dose_bin", observed=True)[
                "Hours Working"
            ].agg(["mean", "count", "std"])
            bin_mids = [iv.mid for iv in bin_stats.index]

            # Zero dose bar
            x_positions = [0] + bin_mids
            means = [no_use.mean()] + list(bin_stats["mean"])
            counts = [len(no_use)] + list(bin_stats["count"])
            errors = [no_use.std() / np.sqrt(len(no_use))] + list(
                bin_stats["std"] / np.sqrt(bin_stats["count"])
            )
            colors = ["steelblue"] + ["coral"] * len(bin_mids)
            labels_bar = [f"None\n(n={len(no_use)})"] + [
                f"{iv.left:.0f}-{iv.right:.0f}\n(n={c})"
                for iv, c in zip(bin_stats.index, bin_stats["count"])
            ]
        else:
            # Few unique doses — use each as its own bin
            groups = sub.groupby(sub[dose_col])["Hours Working"]
            x_positions = list(groups.groups.keys())
            means = [groups.get_group(k).mean() for k in x_positions]
            counts = [len(groups.get_group(k)) for k in x_positions]
            errors = [
                groups.get_group(k).std() / np.sqrt(len(groups.get_group(k)))
                for k in x_positions
            ]
            colors = ["steelblue" if x == 0 else "coral" for x in x_positions]
            labels_bar = [f"{x:.0f}\n(n={c})" for x, c in zip(x_positions, counts)]

        ax_with_zero.bar(
            range(len(x_positions)),
            means,
            yerr=errors,
            color=colors,
            alpha=0.7,
            edgecolor="white",
            capsize=3,
        )
        ax_with_zero.set_xticks(range(len(x_positions)))
        ax_with_zero.set_xticklabels(labels_bar, fontsize=7)
        ax_with_zero.set_xlabel(f"{label} (binned)")
        ax_with_zero.set_ylabel("Mean Hours Working")
        ax_with_zero.set_title(f"{label.split('(')[0].strip()}: Mean Hours by Dose Bin")

    fig.suptitle(
        f"Supplement Dose-Response ({filter_label})",
        fontsize=14,
        y=1.01,
    )
    fig.tight_layout()
    out_path = OUT_DIR / f"supplement_dose_response{suffix}.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"  Saved to {out_path}")
    plt.close(fig)
