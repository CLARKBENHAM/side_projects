"""Focused simulation: R² → filtering gain with optimal/fixed cutoffs.

User's correlation definition: "noise accounts for X% of signal variance"
  noise_var = X * signal_var → R² = 1/(1+X) → r = 1/sqrt(1+X)
  15% noise → r=0.93, 50% noise → r=0.82, 100% → r=0.71, 200% → r=0.58, 500% → r=0.41

Sweep from r=0.10 (very weak, like actual GR→enjoyment) to r=0.97 (near-perfect).
Use empirical enjoyment distribution + 40% self-noise.
"""

import sys
from pathlib import Path

import numpy as np

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression

sys.path.insert(0, str(Path(__file__).resolve().parent))
from book_decision_analysis import load_and_join, impute_missing

AI = Path(__file__).resolve().parent
PLOTS = AI / "plots"
PLOTS.mkdir(exist_ok=True)

N_TRAIN = 200
N_TEST = 70
N_SIMS = 3000
SELF_NOISE = 0.40


def load_empirical() -> np.ndarray:
    df = load_and_join()
    df = impute_missing(df)
    return df["avg_enjoyment"].dropna().values


def sim_one(
    n_signals: int,
    signal_r: float,
    empirical_y: np.ndarray,
    rng: np.random.Generator,
    n_sims: int = N_SIMS,
) -> dict[str, np.ndarray]:
    """Run simulation, return per-sim arrays of test_r2 and gains at each cutoff."""
    n_total = N_TRAIN + N_TEST
    cutoffs = np.arange(0, 100, 5)  # 0%, 5%, 10%, ..., 95%
    n_cuts = len(cutoffs)

    test_r2 = np.zeros(n_sims)
    # gains[i, j] = avg rating gain for sim i at cutoff j
    gains_enjoy = np.zeros((n_sims, n_cuts))
    gains_util = np.zeros((n_sims, n_cuts))

    for sim in range(n_sims):
        # Sample true quality from empirical distribution
        idx = rng.choice(len(empirical_y), n_total, replace=True)
        true_y = empirical_y[idx] + rng.normal(0, 0.05, n_total)
        true_y = np.clip(true_y, 1.0, 5.0)

        # Add self-rating noise
        y_var = true_y.var()
        noise_var = y_var * SELF_NOISE / (1 - SELF_NOISE + 1e-10)
        observed_y = true_y + rng.normal(0, np.sqrt(noise_var), n_total)
        observed_y = np.clip(observed_y, 1.0, 5.0)

        # Generate signals correlated with TRUE quality (not observed)
        z_true = (true_y - true_y.mean()) / (true_y.std() + 1e-10)
        X = np.zeros((n_total, n_signals))
        for k in range(n_signals):
            noise = rng.standard_normal(n_total)
            X[:, k] = signal_r * z_true + np.sqrt(1 - signal_r**2) * noise

        # Split
        X_tr, X_te = X[:N_TRAIN], X[N_TRAIN:]
        y_tr = observed_y[:N_TRAIN]
        y_te = observed_y[N_TRAIN:]

        # Fit and predict
        reg = LinearRegression().fit(X_tr, y_tr)
        pred_te = reg.predict(X_te)

        # Test R²
        ss_res = ((y_te - pred_te) ** 2).sum()
        ss_tot = ((y_te - y_te.mean()) ** 2).sum()
        test_r2[sim] = max(0, 1 - ss_res / ss_tot) if ss_tot > 0 else 0

        # Filtering gains at each cutoff
        avg_all = y_te.mean()
        util_all = (1.3 ** (y_te - 1) - 1).mean()
        for j, pct in enumerate(cutoffs):
            if pct == 0:
                gains_enjoy[sim, j] = 0.0
                gains_util[sim, j] = 0.0
            else:
                thresh = np.percentile(pred_te, pct)
                kept = y_te[pred_te >= thresh]
                if len(kept) > 0:
                    gains_enjoy[sim, j] = kept.mean() - avg_all
                    u_kept = (1.3 ** (kept - 1) - 1).mean()
                    gains_util[sim, j] = (
                        (u_kept - util_all) / util_all * 100 if util_all > 0 else 0
                    )

    return {
        "test_r2": test_r2,
        "gains_enjoy": gains_enjoy,
        "gains_util": gains_util,
        "cutoffs": cutoffs,
    }


def main() -> None:
    print("Loading empirical data...")
    empirical_y = load_empirical()
    print(
        f"  {len(empirical_y)} ratings, mean={empirical_y.mean():.2f}, std={empirical_y.std():.2f}"
    )

    rng = np.random.default_rng(42)

    # Sweep signal correlations: noise_frac of signal variance
    # noise_frac: 0.03→r=0.985, 0.10→0.953, 0.15→0.930, 0.30→0.877,
    #             0.50→0.816, 1.0→0.707, 2.0→0.577, 5.0→0.408, 10.0→0.302
    noise_fracs = [
        0.03,
        0.07,
        0.10,
        0.15,
        0.20,
        0.30,
        0.50,
        0.75,
        1.0,
        1.5,
        2.0,
        3.0,
        5.0,
        8.0,
        12.0,
        20.0,
    ]
    signal_rs = [1.0 / np.sqrt(1 + nf) for nf in noise_fracs]

    # Also run with actual empirical correlations for reference points
    empirical_ref = {
        "GR (r=0.22)": 0.222,
        "AMZ (r=0.15)": 0.151,
        "OL (r=0.12)": 0.117,
    }

    # Use 3 signals (matching GR/OL/AMZ) for main sweep
    n_signals = 3

    # Run all simulations
    print(f"\nRunning {len(signal_rs)} configurations × {N_SIMS} sims each...")
    results: list[dict] = []
    for i, (nf, sr) in enumerate(zip(noise_fracs, signal_rs)):
        res = sim_one(n_signals, sr, empirical_y, rng)
        mean_r2 = res["test_r2"].mean()
        results.append({"noise_frac": nf, "signal_r": sr, "mean_r2": mean_r2, **res})
        print(f"  noise={nf:.0%} of signal var, r={sr:.3f} → test R²={mean_r2:.4f}")

    # Also run single-signal configs for empirical reference
    ref_results: dict[str, dict] = {}
    for label, r_val in empirical_ref.items():
        res = sim_one(1, r_val, empirical_y, rng)
        ref_results[label] = {"mean_r2": res["test_r2"].mean(), **res}
        print(f"  Reference {label}: test R²={res['test_r2'].mean():.4f}")

    # Combined 3-source reference
    res_combined = sim_combined(empirical_y, rng)
    ref_results["GR+OL+AMZ combined"] = {
        "mean_r2": res_combined["test_r2"].mean(),
        **res_combined,
    }
    print(f"  Reference GR+OL+AMZ: test R²={res_combined['test_r2'].mean():.4f}")

    cutoffs = results[0]["cutoffs"]

    # ═══════════════════════════════════════════════════════════════
    # PLOT 1: R² → gain at "optimal" cutoff with labels
    # ═══════════════════════════════════════════════════════════════
    # For each R², find cutoff that maximizes utility gain
    # (always the highest cutoff, so instead show the Pareto: for each R²,
    # plot gains at 20%, 40%, 50%, 60%, 80%, 90%, 95%)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))

    r2_vals = [r["mean_r2"] for r in results]
    select_pcts = [10, 20, 30, 40, 50, 60, 70, 80, 90, 95]
    pct_indices = [list(cutoffs).index(p) for p in select_pcts]
    colors = plt.cm.plasma(np.linspace(0.1, 0.9, len(select_pcts)))

    for ci, (pct_val, pct_idx) in enumerate(zip(select_pcts, pct_indices)):
        enjoy_gains = [r["gains_enjoy"].mean(axis=0)[pct_idx] for r in results]
        util_gains = [r["gains_util"].mean(axis=0)[pct_idx] for r in results]
        ax1.plot(
            r2_vals,
            enjoy_gains,
            "o-",
            color=colors[ci],
            label=f"drop {pct_val}%",
            markersize=3,
            linewidth=1.5,
        )
        ax2.plot(
            r2_vals,
            util_gains,
            "o-",
            color=colors[ci],
            label=f"drop {pct_val}%",
            markersize=3,
            linewidth=1.5,
        )

    # Add empirical reference points
    for label, ref in ref_results.items():
        r2 = ref["mean_r2"]
        # Show gain at 50% cutoff as representative
        idx50 = list(ref["cutoffs"]).index(50)
        g50 = ref["gains_enjoy"].mean(axis=0)[idx50]
        u50 = ref["gains_util"].mean(axis=0)[idx50]
        ax1.axvline(r2, color="red", linestyle=":", alpha=0.4)
        ax1.annotate(
            label,
            (r2, g50),
            textcoords="offset points",
            xytext=(5, 10),
            fontsize=6,
            color="red",
            rotation=45,
        )
        ax2.axvline(r2, color="red", linestyle=":", alpha=0.4)
        ax2.annotate(
            label,
            (r2, u50),
            textcoords="offset points",
            xytext=(5, 10),
            fontsize=6,
            color="red",
            rotation=45,
        )

    ax1.set_xlabel("Test R² (3 signals, 40% self-noise, empirical dist)")
    ax1.set_ylabel("Avg enjoyment gain of kept books")
    ax1.set_title("Enjoyment gain by R² and cutoff level")
    ax1.legend(fontsize=7, ncol=2, loc="upper left")
    ax1.axhline(0, color="gray", linestyle=":", alpha=0.3)

    ax2.set_xlabel("Test R²")
    ax2.set_ylabel("Utility gain % (1.3^(r-1)-1)")
    ax2.set_title("Utility gain by R² and cutoff level")
    ax2.legend(fontsize=7, ncol=2, loc="upper left")
    ax2.axhline(0, color="gray", linestyle=":", alpha=0.3)

    # Add noise-fraction as top x-axis
    ax1_top = ax1.twiny()
    ax1_top.set_xlim(ax1.get_xlim())
    # Map a few R² values to noise fracs
    label_nfs = [0.10, 0.30, 0.50, 1.0, 2.0, 5.0, 12.0]
    label_r2s = []
    for nf in label_nfs:
        sr = 1.0 / np.sqrt(1 + nf)
        # Approximate test R² from our data
        closest = min(results, key=lambda r: abs(r["signal_r"] - sr))
        label_r2s.append(closest["mean_r2"])
    ax1_top.set_xticks(label_r2s)
    ax1_top.set_xticklabels([f"{nf:.0%}" for nf in label_nfs], fontsize=7)
    ax1_top.set_xlabel("Noise as % of signal variance", fontsize=8)

    plt.tight_layout()
    plt.savefig(PLOTS / "sim_r2_vs_gain_all_cutoffs.png", dpi=150)
    plt.close()
    print(f"\n→ Plot 1 saved: {PLOTS / 'sim_r2_vs_gain_all_cutoffs.png'}")

    # ═══════════════════════════════════════════════════════════════
    # PLOT 2: R² → gain at fixed drop=50%, with confidence bands
    # ═══════════════════════════════════════════════════════════════
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    idx50 = list(cutoffs).index(50)

    enjoy_means = [r["gains_enjoy"][:, idx50].mean() for r in results]
    enjoy_p10 = [np.percentile(r["gains_enjoy"][:, idx50], 10) for r in results]
    enjoy_p90 = [np.percentile(r["gains_enjoy"][:, idx50], 90) for r in results]
    util_means = [r["gains_util"][:, idx50].mean() for r in results]
    util_p10 = [np.percentile(r["gains_util"][:, idx50], 10) for r in results]
    util_p90 = [np.percentile(r["gains_util"][:, idx50], 90) for r in results]

    ax1.plot(
        r2_vals, enjoy_means, "o-", color="#2196F3", markersize=5, label="Mean gain"
    )
    ax1.fill_between(
        r2_vals,
        enjoy_p10,
        enjoy_p90,
        alpha=0.2,
        color="#2196F3",
        label="10-90th pctile",
    )
    ax2.plot(
        r2_vals,
        util_means,
        "o-",
        color="#FF9800",
        markersize=5,
        label="Mean utility gain",
    )
    ax2.fill_between(
        r2_vals, util_p10, util_p90, alpha=0.2, color="#FF9800", label="10-90th pctile"
    )

    # Reference points
    for label, ref in ref_results.items():
        r2 = ref["mean_r2"]
        g = ref["gains_enjoy"][:, idx50].mean()
        u = ref["gains_util"][:, idx50].mean()
        ax1.plot(r2, g, "D", color="red", markersize=8, zorder=5)
        ax1.annotate(
            label,
            (r2, g),
            textcoords="offset points",
            xytext=(8, 5),
            fontsize=7,
            color="red",
        )
        ax2.plot(r2, u, "D", color="red", markersize=8, zorder=5)
        ax2.annotate(
            label,
            (r2, u),
            textcoords="offset points",
            xytext=(8, 5),
            fontsize=7,
            color="red",
        )

    # Annotate noise fractions on the curve
    for r in results:
        nf = r["noise_frac"]
        if nf in [0.10, 0.30, 0.50, 1.0, 2.0, 5.0]:
            r2 = r["mean_r2"]
            g = r["gains_enjoy"][:, idx50].mean()
            ax1.annotate(
                f"noise={nf:.0%}",
                (r2, g),
                textcoords="offset points",
                xytext=(-5, -15),
                fontsize=6,
                color="gray",
            )

    ax1.set_xlabel("Test R²")
    ax1.set_ylabel("Avg enjoyment gain (drop bottom 50%)")
    ax1.set_title("Drop bottom 50%: enjoyment gain vs model R²")
    ax1.legend(fontsize=8)
    ax1.axhline(0, color="gray", linestyle=":", alpha=0.3)

    ax2.set_xlabel("Test R²")
    ax2.set_ylabel("Utility gain % (drop bottom 50%)")
    ax2.set_title("Drop bottom 50%: utility gain vs model R²")
    ax2.legend(fontsize=8)
    ax2.axhline(0, color="gray", linestyle=":", alpha=0.3)

    plt.tight_layout()
    plt.savefig(PLOTS / "sim_drop50_vs_r2.png", dpi=150)
    plt.close()
    print(f"→ Plot 2 saved: {PLOTS / 'sim_drop50_vs_r2.png'}")

    # ═══════════════════════════════════════════════════════════════
    # PLOT 3: R² → gain at fixed drop=80%, with confidence bands
    # ═══════════════════════════════════════════════════════════════
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    idx80 = list(cutoffs).index(80)

    enjoy_means = [r["gains_enjoy"][:, idx80].mean() for r in results]
    enjoy_p10 = [np.percentile(r["gains_enjoy"][:, idx80], 10) for r in results]
    enjoy_p90 = [np.percentile(r["gains_enjoy"][:, idx80], 90) for r in results]
    util_means = [r["gains_util"][:, idx80].mean() for r in results]
    util_p10 = [np.percentile(r["gains_util"][:, idx80], 10) for r in results]
    util_p90 = [np.percentile(r["gains_util"][:, idx80], 90) for r in results]

    ax1.plot(
        r2_vals, enjoy_means, "o-", color="#2196F3", markersize=5, label="Mean gain"
    )
    ax1.fill_between(
        r2_vals,
        enjoy_p10,
        enjoy_p90,
        alpha=0.2,
        color="#2196F3",
        label="10-90th pctile",
    )
    ax2.plot(
        r2_vals,
        util_means,
        "o-",
        color="#FF9800",
        markersize=5,
        label="Mean utility gain",
    )
    ax2.fill_between(
        r2_vals, util_p10, util_p90, alpha=0.2, color="#FF9800", label="10-90th pctile"
    )

    # Reference points
    for label, ref in ref_results.items():
        r2 = ref["mean_r2"]
        g = ref["gains_enjoy"][:, idx80].mean()
        u = ref["gains_util"][:, idx80].mean()
        ax1.plot(r2, g, "D", color="red", markersize=8, zorder=5)
        ax1.annotate(
            label,
            (r2, g),
            textcoords="offset points",
            xytext=(8, 5),
            fontsize=7,
            color="red",
        )
        ax2.plot(r2, u, "D", color="red", markersize=8, zorder=5)
        ax2.annotate(
            label,
            (r2, u),
            textcoords="offset points",
            xytext=(8, 5),
            fontsize=7,
            color="red",
        )

    # Annotate noise fractions
    for r in results:
        nf = r["noise_frac"]
        if nf in [0.10, 0.30, 0.50, 1.0, 2.0, 5.0]:
            r2 = r["mean_r2"]
            g = r["gains_enjoy"][:, idx80].mean()
            ax1.annotate(
                f"noise={nf:.0%}",
                (r2, g),
                textcoords="offset points",
                xytext=(-5, -15),
                fontsize=6,
                color="gray",
            )

    ax1.set_xlabel("Test R²")
    ax1.set_ylabel("Avg enjoyment gain (drop bottom 80%)")
    ax1.set_title("Drop bottom 80%: enjoyment gain vs model R²")
    ax1.legend(fontsize=8)
    ax1.axhline(0, color="gray", linestyle=":", alpha=0.3)

    ax2.set_xlabel("Test R²")
    ax2.set_ylabel("Utility gain % (drop bottom 80%)")
    ax2.set_title("Drop bottom 80%: utility gain vs model R²")
    ax2.legend(fontsize=8)
    ax2.axhline(0, color="gray", linestyle=":", alpha=0.3)

    plt.tight_layout()
    plt.savefig(PLOTS / "sim_drop80_vs_r2.png", dpi=150)
    plt.close()
    print(f"→ Plot 3 saved: {PLOTS / 'sim_drop80_vs_r2.png'}")

    # ═══════════════════════════════════════════════════════════════
    # Summary table
    # ═══════════════════════════════════════════════════════════════
    print("\n" + "=" * 90)
    print(
        "SUMMARY TABLE: noise fraction → signal r → test R² → gains at 50% and 80% cutoff"
    )
    print("=" * 90)
    print(
        f"  {'Noise%':>7s} {'Sig r':>6s} {'Test R²':>8s} │ "
        f"{'E@50%':>7s} {'U@50%':>7s} │ "
        f"{'E@80%':>7s} {'U@80%':>7s} │ "
        f"{'E@95%':>7s} {'U@95%':>7s}"
    )
    print("  " + "─" * 86)
    idx95 = list(cutoffs).index(95)
    for r in results:
        nf = r["noise_frac"]
        sr = r["signal_r"]
        r2 = r["mean_r2"]
        e50 = r["gains_enjoy"][:, idx50].mean()
        u50 = r["gains_util"][:, idx50].mean()
        e80 = r["gains_enjoy"][:, idx80].mean()
        u80 = r["gains_util"][:, idx80].mean()
        e95 = r["gains_enjoy"][:, idx95].mean()
        u95 = r["gains_util"][:, idx95].mean()
        print(
            f"  {nf:7.0%} {sr:6.3f} {r2:8.4f} │ "
            f"{e50:+7.3f} {u50:+6.1f}% │ "
            f"{e80:+7.3f} {u80:+6.1f}% │ "
            f"{e95:+7.3f} {u95:+6.1f}%"
        )

    print("\n  Reference (empirical signals):")
    for label, ref in ref_results.items():
        r2 = ref["mean_r2"]
        e50 = ref["gains_enjoy"][:, idx50].mean()
        u50 = ref["gains_util"][:, idx50].mean()
        e80 = ref["gains_enjoy"][:, idx80].mean()
        u80 = ref["gains_util"][:, idx80].mean()
        e95 = ref["gains_enjoy"][:, idx95].mean()
        u95 = ref["gains_util"][:, idx95].mean()
        print(
            f"  {label:30s} R²={r2:.4f} │ "
            f"E@50%={e50:+.3f} U@50%={u50:+.1f}% │ "
            f"E@80%={e80:+.3f} U@80%={u80:+.1f}% │ "
            f"E@95%={e95:+.3f} U@95%={u95:+.1f}%"
        )


def sim_combined(
    empirical_y: np.ndarray,
    rng: np.random.Generator,
) -> dict[str, np.ndarray]:
    """Simulate with heterogeneous signals matching actual GR/OL/AMZ correlations."""
    n_total = N_TRAIN + N_TEST
    cutoffs = np.arange(0, 100, 5)
    n_cuts = len(cutoffs)

    test_r2 = np.zeros(N_SIMS)
    gains_enjoy = np.zeros((N_SIMS, n_cuts))
    gains_util = np.zeros((N_SIMS, n_cuts))

    actual_rs = [0.222, 0.151, 0.117]  # GR, AMZ, OL

    for sim in range(N_SIMS):
        idx = rng.choice(len(empirical_y), n_total, replace=True)
        true_y = empirical_y[idx] + rng.normal(0, 0.05, n_total)
        true_y = np.clip(true_y, 1.0, 5.0)

        y_var = true_y.var()
        noise_var = y_var * SELF_NOISE / (1 - SELF_NOISE + 1e-10)
        observed_y = true_y + rng.normal(0, np.sqrt(noise_var), n_total)
        observed_y = np.clip(observed_y, 1.0, 5.0)

        z_true = (true_y - true_y.mean()) / (true_y.std() + 1e-10)
        X = np.zeros((n_total, 3))
        for k, r in enumerate(actual_rs):
            noise = rng.standard_normal(n_total)
            X[:, k] = r * z_true + np.sqrt(1 - r**2) * noise

        X_tr, X_te = X[:N_TRAIN], X[N_TRAIN:]
        y_tr = observed_y[:N_TRAIN]
        y_te = observed_y[N_TRAIN:]

        reg = LinearRegression().fit(X_tr, y_tr)
        pred_te = reg.predict(X_te)

        ss_res = ((y_te - pred_te) ** 2).sum()
        ss_tot = ((y_te - y_te.mean()) ** 2).sum()
        test_r2[sim] = max(0, 1 - ss_res / ss_tot) if ss_tot > 0 else 0

        avg_all = y_te.mean()
        util_all = (1.3 ** (y_te - 1) - 1).mean()
        for j, pct in enumerate(cutoffs):
            if pct == 0:
                continue
            thresh = np.percentile(pred_te, pct)
            kept = y_te[pred_te >= thresh]
            if len(kept) > 0:
                gains_enjoy[sim, j] = kept.mean() - avg_all
                u_kept = (1.3 ** (kept - 1) - 1).mean()
                gains_util[sim, j] = (
                    (u_kept - util_all) / util_all * 100 if util_all > 0 else 0
                )

    return {
        "test_r2": test_r2,
        "gains_enjoy": gains_enjoy,
        "gains_util": gains_util,
        "cutoffs": cutoffs,
    }


if __name__ == "__main__":
    main()
