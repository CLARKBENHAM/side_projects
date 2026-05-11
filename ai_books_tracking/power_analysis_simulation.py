#!/usr/bin/env python3
"""
Power analysis: How many algorithm-selected books must I read to detect
that the algorithm's picks are better than my historical average?

Two scenarios:
  A) Unread pool has the same mean as historical (algorithm helps by filtering)
  B) Unread pool mean is 0.2 lower (remaining books are slightly worse)

Focus: General Reading + Business categories (user's likely picks).
Ranking: 2.5 * z(usefulness) + z(enjoyment), matching the recommendation list.
"""

import numpy as np
from scipy import stats
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ── Empirical parameters ─────────────────────────────────────────────────────
# Historical means for RATED books, by category (from goodreads_residuals_by_category.csv)
#   Gen Reading: enjoy=3.625, useful=2.336, n=38
#   Business:    enjoy=3.413, useful=2.038, n=40
#   CS:          enjoy=2.768, useful=2.536, n=14
# Overall historical (all 268 books): enjoy=3.31, useful=1.82

CATEGORY_CONFIGS = {
    "Gen Reading + Business": {
        "hist_enjoy": 3.516,  # weighted (38*3.625+40*3.413)/78
        "hist_useful": 2.183,  # weighted (38*2.336+40*2.038)/78
        "pool_size": 140,
    },
    "Gen Reading + Business + CS": {
        "hist_enjoy": 3.402,  # weighted (38*3.625+40*3.413+14*2.768)/92
        "hist_useful": 2.237,  # weighted
        "pool_size": 200,
    },
}

OVERALL_HIST_ENJOY = 3.31  # all 268 rated books
OVERALL_HIST_USEFUL = 1.82

# Within-category standard deviations (pooled from empirical data)
ENJOY_STD = 0.80
USEFUL_STD = 0.80

# Correlation between enjoyment and usefulness (cross books, same person)
ENJOY_USEFUL_CORR = 0.40

# Test-retest measurement noise (your own rating variability)
ENJOY_NOISE_STD = 0.66
USEFUL_NOISE_STD = 0.59

# Algorithm signal strength (observed Spearman rho on 2026 holdout)
ALGO_RHO_ENJOY = 0.30
ALGO_RHO_USEFUL = 0.52

# Composite ranking weights (matching unread_book_recommendations_weighted_useful.txt)
USEFUL_WEIGHT = 2.5
ENJOY_WEIGHT = 1.0

# Simulation grid
N_SIM = 10_000
N_BOOKS_LIST = [3, 5, 7, 10, 15, 20, 25, 30]
POOL_SHIFTS = [0.0, -0.2]
ALPHA_LEVELS = [0.05, 0.10, 0.20]
HOURS_PER_BOOK = 5


def generate_pool(
    n: int,
    enjoy_mean: float,
    useful_mean: float,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    """Generate correlated true enjoyment and usefulness ratings for a pool."""
    cov = ENJOY_USEFUL_CORR * ENJOY_STD * USEFUL_STD
    mean = [enjoy_mean, useful_mean]
    cov_mat = [[ENJOY_STD**2, cov], [cov, USEFUL_STD**2]]
    samples = rng.multivariate_normal(mean, cov_mat, n)
    return samples[:, 0], samples[:, 1]


def predict(true_vals: np.ndarray, rho: float, rng: np.random.Generator) -> np.ndarray:
    """Algorithm predictions with known correlation to true values."""
    mu = true_vals.mean()
    sd = true_vals.std()
    if sd < 1e-10:
        return true_vals.copy()
    z = (true_vals - mu) / sd
    noise = rng.standard_normal(len(true_vals))
    z_pred = rho * z + np.sqrt(max(0, 1 - rho**2)) * noise
    return z_pred * sd + mu


def run_power_analysis(
    cat_label: str,
    hist_enjoy: float,
    hist_useful: float,
    pool_size: int,
    pool_shift: float,
    n_sim: int,
    rng: np.random.Generator,
) -> pd.DataFrame:
    """Run power simulation for one category config + pool shift."""
    pool_enjoy_mean = hist_enjoy + pool_shift
    pool_useful_mean = hist_useful + pool_shift

    rows: list[dict] = []

    for n_books in N_BOOKS_LIST:
        # Arrays to collect per-simulation results
        cat_enjoy_p = np.empty(n_sim)
        cat_useful_p = np.empty(n_sim)
        all_enjoy_p = np.empty(n_sim)
        all_useful_p = np.empty(n_sim)
        obs_enjoy_means = np.empty(n_sim)
        obs_useful_means = np.empty(n_sim)

        for i in range(n_sim):
            # 1. True ratings for unread pool
            true_e, true_u = generate_pool(pool_size, pool_enjoy_mean, pool_useful_mean, rng)

            # 2. Algorithm predictions
            pred_e = predict(true_e, ALGO_RHO_ENJOY, rng)
            pred_u = predict(true_u, ALGO_RHO_USEFUL, rng)

            # 3. Composite ranking (z-scored then weighted)
            pe_std = pred_e.std()
            pu_std = pred_u.std()
            z_pe = (pred_e - pred_e.mean()) / (pe_std if pe_std > 1e-10 else 1.0)
            z_pu = (pred_u - pred_u.mean()) / (pu_std if pu_std > 1e-10 else 1.0)
            composite = USEFUL_WEIGHT * z_pu + ENJOY_WEIGHT * z_pe

            # 4. Select top N
            top_idx = np.argsort(composite)[-n_books:]

            # 5. User reads books (true rating + measurement noise, clipped 1-5)
            obs_e = np.clip(
                true_e[top_idx] + rng.normal(0, ENJOY_NOISE_STD, n_books), 1, 5
            )
            obs_u = np.clip(
                true_u[top_idx] + rng.normal(0, USEFUL_NOISE_STD, n_books), 1, 5
            )

            obs_enjoy_means[i] = obs_e.mean()
            obs_useful_means[i] = obs_u.mean()

            # 6. One-sided t-tests (we want to detect improvement)
            def onesided_p(obs: np.ndarray, mu0: float) -> float:
                if obs.std() < 1e-10:
                    return 0.0 if obs.mean() > mu0 else 1.0
                t, p2 = stats.ttest_1samp(obs, mu0)
                return p2 / 2 if t > 0 else 1 - p2 / 2

            cat_enjoy_p[i] = onesided_p(obs_e, hist_enjoy)
            cat_useful_p[i] = onesided_p(obs_u, hist_useful)
            all_enjoy_p[i] = onesided_p(obs_e, OVERALL_HIST_ENJOY)
            all_useful_p[i] = onesided_p(obs_u, OVERALL_HIST_USEFUL)

        # Collect results for each (target, reference) combination
        for target, pvals_cat, pvals_all, obs_means, hist_ref, all_ref in [
            (
                "enjoyment",
                cat_enjoy_p,
                all_enjoy_p,
                obs_enjoy_means,
                hist_enjoy,
                OVERALL_HIST_ENJOY,
            ),
            (
                "usefulness",
                cat_useful_p,
                all_useful_p,
                obs_useful_means,
                hist_useful,
                OVERALL_HIST_USEFUL,
            ),
        ]:
            for ref_label, pvals, ref_mean in [
                ("vs_category_hist", pvals_cat, hist_ref),
                ("vs_overall_hist", pvals_all, all_ref),
            ]:
                effects = obs_means - ref_mean
                rec: dict = {
                    "categories": cat_label,
                    "pool_shift": pool_shift,
                    "target": target,
                    "reference": ref_label,
                    "ref_mean": ref_mean,
                    "n_books": n_books,
                    "hours": n_books * HOURS_PER_BOOK,
                    "expected_obs_mean": obs_means.mean(),
                    "expected_effect": effects.mean(),
                    "effect_p10": np.percentile(effects, 10),
                    "effect_p50": np.median(effects),
                    "effect_p90": np.percentile(effects, 90),
                    "pct_positive": (effects > 0).mean() * 100,
                    "median_p": np.median(pvals),
                }
                for alpha in ALPHA_LEVELS:
                    rec[f"power_{alpha}"] = (pvals < alpha).mean()
                rows.append(rec)

    return pd.DataFrame(rows)


def print_summary_table(df: pd.DataFrame, cat: str, shift: float, reference: str) -> None:
    """Print a readable summary table for one scenario."""
    sub = df[
        (df["categories"] == cat)
        & (df["pool_shift"] == shift)
        & (df["reference"] == reference)
    ].copy()
    if sub.empty:
        return

    shift_label = "same pool mean" if shift == 0 else f"pool {shift:+.1f}"
    ref_label = "category" if "category" in reference else "overall (3.31/1.82)"
    print(f"\n{'─'*95}")
    print(f"  {cat} | {shift_label} | compared to {ref_label} historical mean")
    print(f"{'─'*95}")

    for target in ["enjoyment", "usefulness"]:
        t = sub[sub["target"] == target]
        ref_val = t["ref_mean"].iloc[0]
        print(f"\n  {target.upper()} (historical ref = {ref_val:.2f})")
        print(
            f"  {'N':>4}  {'hrs':>4}  {'E[mean]':>7}  {'E[lift]':>7}  "
            f"{'lift CI80':>14}  {'%pos':>5}  "
            f"{'pow.05':>6}  {'pow.10':>6}  {'pow.20':>6}  {'med p':>6}"
        )
        for _, r in t.iterrows():
            print(
                f"  {r['n_books']:4.0f}  {r['hours']:4.0f}  "
                f"{r['expected_obs_mean']:7.3f}  {r['expected_effect']:+7.3f}  "
                f"[{r['effect_p10']:+.2f}, {r['effect_p90']:+.2f}]  "
                f"{r['pct_positive']:5.1f}  "
                f"{r['power_0.05']:6.1%}  {r['power_0.1']:6.1%}  {r['power_0.2']:6.1%}  "
                f"{r['median_p']:6.3f}"
            )


def find_n_for_power(
    df: pd.DataFrame, target: str, power_threshold: float = 0.80, alpha: float = 0.05
) -> str:
    """Find minimum N achieving given power, or report max power if not reached."""
    col = f"power_{alpha}"
    above = df[(df["target"] == target) & (df[col] >= power_threshold)]
    if len(above) > 0:
        n = int(above["n_books"].min())
        return f"{n} books ({n * HOURS_PER_BOOK} hrs)"
    best = df[df["target"] == target].iloc[-1]
    return f">30 books (power={best[col]:.0%} at n=30)"


def make_power_curves(df: pd.DataFrame, outpath: str) -> None:
    """Power vs N for the primary comparison (vs category historical)."""
    sub = df[df["reference"] == "vs_category_hist"]

    fig, axes = plt.subplots(2, 2, figsize=(14, 10), sharey=True)
    fig.suptitle(
        "Power to Detect Algorithm Improvement vs Category Historical Mean\n"
        "(one-sided t-test, 10K simulations, composite 2.5U+E ranking)",
        fontsize=13,
    )

    for col_idx, shift in enumerate(sorted(sub["pool_shift"].unique())):
        shift_label = "Same pool mean" if shift == 0 else f"Pool mean {shift:+.1f}"
        for row_idx, target in enumerate(["enjoyment", "usefulness"]):
            ax = axes[row_idx, col_idx]
            mask = (sub["pool_shift"] == shift) & (sub["target"] == target)
            for cat in sub["categories"].unique():
                c = sub[mask & (sub["categories"] == cat)]
                ax.plot(
                    c["n_books"], c["power_0.05"], "o-", label=f"{cat} (α=.05)", lw=2, ms=5
                )
                ax.plot(
                    c["n_books"],
                    c["power_0.1"],
                    "s--",
                    label=f"{cat} (α=.10)",
                    lw=1.2,
                    ms=4,
                    alpha=0.6,
                )
            ax.axhline(0.80, color="gray", ls="--", alpha=0.5, label="80% power")
            ax.axhline(0.50, color="gray", ls=":", alpha=0.3)
            ax.set_xlabel("Books read")
            ax.set_ylabel("Power")
            ax.set_title(f"{target.title()} — {shift_label}")
            ax.legend(fontsize=7, loc="lower right")
            ax.set_ylim(0, 1.02)
            ax.set_xticks(N_BOOKS_LIST)
            ax.grid(True, alpha=0.2)

            # Hours axis on top
            ax2 = ax.twiny()
            ax2.set_xlim(ax.get_xlim())
            ax2.set_xticks(N_BOOKS_LIST)
            ax2.set_xticklabels([f"{n * HOURS_PER_BOOK}h" for n in N_BOOKS_LIST], fontsize=7)
            if row_idx == 0:
                ax2.set_xlabel("Reading time")

    plt.tight_layout()
    fig.savefig(outpath, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {outpath}")


def make_effect_and_pvalue_plot(df: pd.DataFrame, outpath: str) -> None:
    """Effect size with 80% CI and median p-value for Gen+Bus (primary scenario)."""
    cat = "Gen Reading + Business"
    sub = df[(df["categories"] == cat) & (df["reference"] == "vs_category_hist")]

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(
        f"Expected Effect Size and P-values — {cat}\n"
        "(vs category historical mean, composite ranking)",
        fontsize=13,
    )

    for col_idx, target in enumerate(["enjoyment", "usefulness"]):
        # Top row: effect sizes
        ax_eff = axes[0, col_idx]
        for shift in sorted(sub["pool_shift"].unique()):
            s = sub[(sub["pool_shift"] == shift) & (sub["target"] == target)]
            label = "Same pool" if shift == 0 else f"Pool {shift:+.1f}"
            ax_eff.fill_between(s["n_books"], s["effect_p10"], s["effect_p90"], alpha=0.15)
            ax_eff.plot(s["n_books"], s["expected_effect"], "o-", label=label, lw=2)
        ax_eff.axhline(0, color="black", lw=0.5)
        ax_eff.set_xlabel("Books read")
        ax_eff.set_ylabel("Rating lift vs historical")
        ax_eff.set_title(f"{target.title()} — Expected lift (80% CI)")
        ax_eff.legend()
        ax_eff.set_xticks(N_BOOKS_LIST)
        ax_eff.grid(True, alpha=0.2)

        # Bottom row: median p-values
        ax_p = axes[1, col_idx]
        for shift in sorted(sub["pool_shift"].unique()):
            s = sub[(sub["pool_shift"] == shift) & (sub["target"] == target)]
            label = "Same pool" if shift == 0 else f"Pool {shift:+.1f}"
            ax_p.plot(s["n_books"], s["median_p"], "o-", label=label, lw=2)
        for alpha in [0.05, 0.10]:
            ax_p.axhline(alpha, color="red", ls="--", alpha=0.4, label=f"α={alpha}")
        ax_p.set_xlabel("Books read")
        ax_p.set_ylabel("Median p-value")
        ax_p.set_title(f"{target.title()} — Median p-value")
        ax_p.legend(fontsize=8)
        ax_p.set_xticks(N_BOOKS_LIST)
        ax_p.set_yscale("log")
        ax_p.set_ylim(0.001, 1)
        ax_p.grid(True, alpha=0.2)

    plt.tight_layout()
    fig.savefig(outpath, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {outpath}")


def make_alpha_power_tradeoff(df: pd.DataFrame, outpath: str) -> None:
    """For key N values, show the full alpha-power tradeoff curve."""
    cat = "Gen Reading + Business"
    sub = df[(df["categories"] == cat) & (df["reference"] == "vs_category_hist")]

    highlight_ns = [5, 10, 15, 20]
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(
        f"Alpha vs Power Tradeoff — {cat}, Same Pool Mean\n"
        "(higher alpha = more lenient test, more false positives)",
        fontsize=13,
    )

    s0 = sub[sub["pool_shift"] == 0.0]

    for idx, target in enumerate(["enjoyment", "usefulness"]):
        ax = axes[idx]
        t = s0[s0["target"] == target]
        for n_books in highlight_ns:
            row = t[t["n_books"] == n_books].iloc[0]
            alphas = ALPHA_LEVELS
            powers = [row[f"power_{a}"] for a in alphas]
            ax.plot(alphas, powers, "o-", label=f"n={n_books} ({n_books*5}h)", lw=2)
        ax.plot([0, 0.5], [0, 0.5], "k:", alpha=0.3, label="no signal (diagonal)")
        ax.set_xlabel("Significance level (α)")
        ax.set_ylabel("Power (1 - β)")
        ax.set_title(f"{target.title()}")
        ax.legend()
        ax.set_xlim(0, 0.25)
        ax.set_ylim(0, 1.02)
        ax.grid(True, alpha=0.2)

    plt.tight_layout()
    fig.savefig(outpath, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {outpath}")


def main() -> None:
    rng = np.random.default_rng(42)
    frames: list[pd.DataFrame] = []

    for cat_label, cfg in CATEGORY_CONFIGS.items():
        for shift in POOL_SHIFTS:
            shift_str = f"{shift:+.1f}" if shift != 0 else "0.0"
            print(f"Simulating: {cat_label}, pool shift={shift_str} ...")
            result = run_power_analysis(
                cat_label,
                cfg["hist_enjoy"],
                cfg["hist_useful"],
                cfg["pool_size"],
                shift,
                N_SIM,
                rng,
            )
            frames.append(result)

    results = pd.concat(frames, ignore_index=True)
    out_csv = "ai_actions/power_analysis_results.csv"
    results.to_csv(out_csv, index=False)
    print(f"\nSaved {out_csv}")

    # ── Print summary tables ─────────────────────────────────────────────────
    print("\n" + "=" * 95)
    print("  POWER ANALYSIS: CAN I TELL IF THE ALGORITHM WORKS?")
    print("=" * 95)
    print(
        f"\n  Setup: pick top N books by composite ranking (2.5*useful + enjoy)"
        f"\n  from ~140-200 unread General Reading / Business / CS books."
        f"\n  Read them (~{HOURS_PER_BOOK}h each), rate, then one-sided t-test vs historical mean."
        f"\n  Algorithm signal: rho={ALGO_RHO_ENJOY} (enjoy), rho={ALGO_RHO_USEFUL} (useful)"
        f"\n  Rating noise: SD={ENJOY_NOISE_STD} (enjoy), SD={USEFUL_NOISE_STD} (useful)"
    )

    # Primary: vs category historical mean
    print("\n" + "=" * 95)
    print("  PRIMARY: vs category historical mean (pure algorithm signal)")
    print("=" * 95)
    for cat in CATEGORY_CONFIGS:
        for shift in POOL_SHIFTS:
            print_summary_table(results, cat, shift, "vs_category_hist")

    # Secondary: vs overall historical mean (includes category bonus)
    print("\n" + "=" * 95)
    print("  SECONDARY: vs overall historical mean (enjoy=3.31, useful=1.82)")
    print("  (includes ~+0.2 'category bonus' from choosing Gen Reading/Business)")
    print("=" * 95)
    for cat in CATEGORY_CONFIGS:
        for shift in POOL_SHIFTS:
            print_summary_table(results, cat, shift, "vs_overall_hist")

    # ── Key takeaways ────────────────────────────────────────────────────────
    print("\n" + "=" * 95)
    print("  KEY TAKEAWAYS")
    print("=" * 95)

    cat = "Gen Reading + Business"
    for reference in ["vs_category_hist", "vs_overall_hist"]:
        ref_label = "category" if "category" in reference else "overall"
        print(f"\n  Compared to {ref_label} historical mean:")
        for shift in POOL_SHIFTS:
            shift_label = "same quality" if shift == 0 else f"pool {shift:+.1f}"
            sub = results[
                (results["categories"] == cat)
                & (results["pool_shift"] == shift)
                & (results["reference"] == reference)
            ]
            for target in ["enjoyment", "usefulness"]:
                need = find_n_for_power(sub, target, 0.80, 0.05)
                print(f"    {target:12s} ({shift_label}): {need} for 80% power at α=0.05")
            # Also report α=0.10
            for target in ["enjoyment", "usefulness"]:
                need = find_n_for_power(sub, target, 0.80, 0.10)
                print(f"    {target:12s} ({shift_label}): {need} for 80% power at α=0.10")

    # ── Practical recommendation ─────────────────────────────────────────────
    print("\n" + "=" * 95)
    print("  PRACTICAL RECOMMENDATION")
    print("=" * 95)

    # Get effect sizes at n=10 for the primary scenario
    primary = results[
        (results["categories"] == cat)
        & (results["pool_shift"] == 0.0)
        & (results["reference"] == "vs_category_hist")
    ]
    for target in ["enjoyment", "usefulness"]:
        row10 = primary[(primary["target"] == target) & (primary["n_books"] == 10)].iloc[0]
        row20 = primary[(primary["target"] == target) & (primary["n_books"] == 20)].iloc[0]
        print(
            f"\n  {target.title()} (same pool quality, vs category mean):"
            f"\n    After 10 books (50 hrs): expected lift {row10['expected_effect']:+.2f}, "
            f"power={row10['power_0.05']:.0%} (α=.05), {row10['power_0.1']:.0%} (α=.10)"
            f"\n    After 20 books (100 hrs): expected lift {row20['expected_effect']:+.2f}, "
            f"power={row20['power_0.05']:.0%} (α=.05), {row20['power_0.1']:.0%} (α=.10)"
        )

    print(
        "\n  Note: usefulness is much easier to detect because the algorithm's signal"
        "\n  is stronger (rho=0.52 vs 0.30) and the composite ranking weights it 2.5x."
        "\n  If you want the fastest signal that the algorithm 'works', track usefulness."
        "\n\n  The comparison vs overall historical mean is easier (larger effect)"
        "\n  because Gen Reading + Business are inherently above-average categories."
        "\n  The category-specific comparison isolates the pure algorithm contribution."
    )

    # ── Plots ────────────────────────────────────────────────────────────────
    print("\nGenerating plots...")
    make_power_curves(results, "ai_actions/power_analysis_curves.png")
    make_effect_and_pvalue_plot(results, "ai_actions/power_analysis_effects.png")
    make_alpha_power_tradeoff(results, "ai_actions/power_analysis_alpha_tradeoff.png")

    print("\nDone.")


if __name__ == "__main__":
    main()
