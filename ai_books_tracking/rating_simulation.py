"""Simulation: how signal correlation → prediction accuracy → filtering gains.

Part 1: Pure simulation varying n_signals and per-signal correlation
Part 2: Transform effects (training in utility space, forgetting inverse)
Part 3: Empirical distributions (actual rating shapes, bounded [1,5])
Part 4: Heterogeneous signal strengths (one source r=0.15, another r=0.30, etc.)
Part 5: Self-noise ceiling (40% measurement noise matching empirical)
"""

import sys
from pathlib import Path

import numpy as np

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# Load real data for empirical distributions
sys.path.insert(0, str(Path(__file__).resolve().parent))
from book_decision_analysis import load_and_join, impute_missing

AI = Path(__file__).resolve().parent
PLOTS = AI / "plots"
PLOTS.mkdir(exist_ok=True)

# Match real dataset sizes
N_TRAIN = 200
N_TEST = 70
N_SIMS = 2000

SIGNAL_CORRS = [0.15, 0.25, 0.35, 0.50]
N_SIGNALS_LIST = [1, 2, 3, 4, 5]
FILTER_PCTS = [0.0, 0.20, 0.40, 0.50, 0.60, 0.75, 0.80, 0.90, 0.95]


def generate_correlated_signals(
    true_y: np.ndarray,
    n_signals: int,
    correlations: list[float],
    rng: np.random.Generator,
) -> np.ndarray:
    """Generate signals with specified correlations to true_y.

    Each signal_i = r_i * z_y + sqrt(1 - r_i^2) * noise_i,
    where z_y = (true_y - mean) / std.
    correlations is cycled if shorter than n_signals.
    """
    n = len(true_y)
    z_y = (true_y - true_y.mean()) / (true_y.std() + 1e-10)
    signals = np.zeros((n, n_signals))
    for i in range(n_signals):
        r = correlations[i % len(correlations)]
        noise = rng.standard_normal(n)
        signals[:, i] = r * z_y + np.sqrt(1 - r**2) * noise
    return signals


def run_sim(
    n_train: int,
    n_test: int,
    n_signals: int,
    correlations: list[float],
    n_sims: int,
    rng: np.random.Generator,
    empirical_y: np.ndarray | None = None,
    self_noise_frac: float = 0.0,
    nonlinear_latent: bool = False,
    utility_transform: bool = False,
    forget_inverse: bool = False,
) -> dict[str, np.ndarray]:
    """Monte Carlo simulation. Returns arrays of metrics across sims."""
    n_total = n_train + n_test

    train_r2 = np.zeros(n_sims)
    test_r2 = np.zeros(n_sims)
    train_r = np.zeros(n_sims)
    test_r = np.zeros(n_sims)
    # Filtering gains: avg rating increase when keeping top (1-pct) books
    filter_gains = {pct: np.zeros(n_sims) for pct in FILTER_PCTS}
    util_gains = {pct: np.zeros(n_sims) for pct in FILTER_PCTS}

    for sim in range(n_sims):
        # True quality
        if empirical_y is not None:
            idx = rng.choice(len(empirical_y), n_total, replace=True)
            true_y = empirical_y[idx] + rng.normal(0, 0.05, n_total)
        else:
            true_y = rng.normal(3.3, 0.885, n_total)
        true_y = np.clip(true_y, 1.0, 5.0)

        # Unmodeled nonlinear component
        if nonlinear_latent:
            cats = rng.choice(5, n_total)
            cat_effects = np.array([-0.4, -0.15, 0.0, 0.15, 0.4])
            true_y = true_y + cat_effects[cats]
            true_y = np.clip(true_y, 1.0, 5.0)

        # Self-rating noise (measurement noise in the target)
        if self_noise_frac > 0:
            y_var = true_y.var()
            noise_var = y_var * self_noise_frac / (1 - self_noise_frac + 1e-10)
            observed_y = true_y + rng.normal(0, np.sqrt(noise_var), n_total)
            observed_y = np.clip(observed_y, 1.0, 5.0)
        else:
            observed_y = true_y.copy()

        # Signals
        X = generate_correlated_signals(true_y, n_signals, correlations, rng)

        # Split
        X_tr, X_te = X[:n_train], X[n_train:]
        y_tr_raw = observed_y[:n_train]
        y_te_raw = observed_y[n_train:]

        # Optional utility transform on training target
        if utility_transform:
            y_tr = 1.3 ** (y_tr_raw - 1) - 1
        else:
            y_tr = y_tr_raw

        # Fit
        reg = LinearRegression().fit(X_tr, y_tr)
        p_tr = reg.predict(X_tr)
        p_te = reg.predict(X_te)

        # Inverse transform if needed
        if utility_transform and not forget_inverse:
            p_tr = 1 + np.log(np.maximum(p_tr + 1, 1e-10)) / np.log(1.3)
            p_te = 1 + np.log(np.maximum(p_te + 1, 1e-10)) / np.log(1.3)
            y_tr_eval = y_tr_raw
            y_te_eval = y_te_raw
        elif utility_transform and forget_inverse:
            # Predictions stay in utility space, compare to raw
            y_tr_eval = y_tr_raw
            y_te_eval = y_te_raw
        else:
            y_tr_eval = y_tr_raw
            y_te_eval = y_te_raw

        # Metrics
        ss_res_tr = ((y_tr_eval - p_tr) ** 2).sum()
        ss_tot_tr = ((y_tr_eval - y_tr_eval.mean()) ** 2).sum()
        train_r2[sim] = max(0, 1 - ss_res_tr / ss_tot_tr) if ss_tot_tr > 0 else 0

        ss_res_te = ((y_te_eval - p_te) ** 2).sum()
        ss_tot_te = ((y_te_eval - y_te_eval.mean()) ** 2).sum()
        test_r2[sim] = max(0, 1 - ss_res_te / ss_tot_te) if ss_tot_te > 0 else 0

        r_tr = np.corrcoef(p_tr, y_tr_eval)[0, 1] if p_tr.std() > 0 else 0
        r_te = np.corrcoef(p_te, y_te_eval)[0, 1] if p_te.std() > 0 else 0
        train_r[sim] = r_tr if np.isfinite(r_tr) else 0
        test_r[sim] = r_te if np.isfinite(r_te) else 0

        # Filtering gains (always measured in raw rating space)
        for pct in FILTER_PCTS:
            if pct == 0.0:
                filter_gains[pct][sim] = 0.0
                util_gains[pct][sim] = 0.0
                continue
            thresh = np.percentile(p_te, pct * 100)
            kept = y_te_raw[p_te >= thresh]
            if len(kept) > 0:
                filter_gains[pct][sim] = kept.mean() - y_te_raw.mean()
                u_kept = (1.3 ** (kept - 1) - 1).mean()
                u_all = (1.3 ** (y_te_raw - 1) - 1).mean()
                util_gains[pct][sim] = (
                    (u_kept - u_all) / u_all * 100 if u_all > 0 else 0
                )

    return {
        "train_r2": train_r2,
        "test_r2": test_r2,
        "train_r": train_r,
        "test_r": test_r,
        "filter_gains": filter_gains,
        "util_gains": util_gains,
    }


def part1_accuracy_vs_signals() -> None:
    """How n_signals × correlation → train/test R²."""
    print("\n" + "=" * 70)
    print("PART 1: Signal count × correlation → train/test accuracy")
    print("=" * 70)

    rng = np.random.default_rng(42)
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Collect results into table
    rows: list[dict] = []
    for n_sig in N_SIGNALS_LIST:
        for r in SIGNAL_CORRS:
            res = run_sim(N_TRAIN, N_TEST, n_sig, [r], N_SIMS, rng)
            rows.append(
                {
                    "n_signals": n_sig,
                    "signal_r": r,
                    "train_r2_mean": res["train_r2"].mean(),
                    "train_r2_std": res["train_r2"].std(),
                    "test_r2_mean": res["test_r2"].mean(),
                    "test_r2_std": res["test_r2"].std(),
                    "train_r_mean": res["train_r"].mean(),
                    "test_r_mean": res["test_r"].mean(),
                    "overfit_ratio": (
                        res["train_r2"].mean() / res["test_r2"].mean()
                        if res["test_r2"].mean() > 0
                        else float("inf")
                    ),
                }
            )
            print(
                f"  {n_sig} signals, r={r:.2f}: train R²={res['train_r2'].mean():.4f}, "
                f"test R²={res['test_r2'].mean():.4f}, "
                f"ratio={res['train_r2'].mean() / (res['test_r2'].mean() + 1e-10):.2f}x"
            )

    # Plot: x = n_signals, separate lines per correlation
    colors = plt.cm.viridis(np.linspace(0.2, 0.9, len(SIGNAL_CORRS)))
    for i, r in enumerate(SIGNAL_CORRS):
        subset = [d for d in rows if d["signal_r"] == r]
        ns = [d["n_signals"] for d in subset]
        train_vals = [d["train_r2_mean"] for d in subset]
        test_vals = [d["test_r2_mean"] for d in subset]
        axes[0].plot(
            ns, train_vals, "o-", color=colors[i], label=f"r={r} (train)", alpha=0.8
        )
        axes[0].plot(
            ns,
            test_vals,
            "s--",
            color=colors[i],
            label=f"r={r} (test)",
            alpha=0.8,
        )

    axes[0].set_xlabel("Number of signals")
    axes[0].set_ylabel("R²")
    axes[0].set_title("Train vs Test R² by signal count and correlation")
    axes[0].legend(fontsize=7, ncol=2)
    axes[0].set_xticks(N_SIGNALS_LIST)

    # Plot: overfit ratio
    for i, r in enumerate(SIGNAL_CORRS):
        subset = [d for d in rows if d["signal_r"] == r]
        ns = [d["n_signals"] for d in subset]
        ratios = [d["overfit_ratio"] for d in subset]
        axes[1].plot(ns, ratios, "o-", color=colors[i], label=f"r={r}")

    axes[1].set_xlabel("Number of signals")
    axes[1].set_ylabel("Train R² / Test R² (overfit ratio)")
    axes[1].set_title("Overfitting: train/test R² ratio")
    axes[1].axhline(1.0, color="gray", linestyle=":", alpha=0.5)
    axes[1].legend(fontsize=8)
    axes[1].set_xticks(N_SIGNALS_LIST)

    plt.tight_layout()
    plt.savefig(PLOTS / "sim_accuracy_vs_signals.png", dpi=150)
    plt.close()
    print(f"  → Saved {PLOTS / 'sim_accuracy_vs_signals.png'}")


def part2_filtering_gains() -> None:
    """How test R² translates to filtering gains."""
    print("\n" + "=" * 70)
    print("PART 2: Test R² → filtering gains")
    print("=" * 70)

    rng = np.random.default_rng(123)

    # Run many configs to get a range of R² values
    all_test_r2: list[float] = []
    all_gains: dict[float, list[float]] = {pct: [] for pct in FILTER_PCTS}
    all_util_gains: dict[float, list[float]] = {pct: [] for pct in FILTER_PCTS}
    configs: list[tuple[int, float]] = []

    for n_sig in N_SIGNALS_LIST:
        for r in SIGNAL_CORRS:
            res = run_sim(N_TRAIN, N_TEST, n_sig, [r], N_SIMS, rng)
            all_test_r2.append(res["test_r2"].mean())
            for pct in FILTER_PCTS:
                all_gains[pct].append(res["filter_gains"][pct].mean())
                all_util_gains[pct].append(res["util_gains"][pct].mean())
            configs.append((n_sig, r))

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Plot 1: R² vs avg rating gain, lines for different drop percentiles
    pct_colors = plt.cm.plasma(np.linspace(0.1, 0.9, len(FILTER_PCTS) - 1))
    for i, pct in enumerate(FILTER_PCTS[1:]):  # skip 0%
        axes[0].plot(
            all_test_r2,
            all_gains[pct],
            "o-",
            color=pct_colors[i],
            label=f"drop {pct:.0%}",
            markersize=4,
        )
    axes[0].set_xlabel("Test R²")
    axes[0].set_ylabel("Avg rating gain (kept books − all)")
    axes[0].set_title("Filtering gain vs model accuracy")
    axes[0].legend(fontsize=7)
    axes[0].axhline(0, color="gray", linestyle=":", alpha=0.5)

    # Plot 2: Same but utility gain %
    for i, pct in enumerate(FILTER_PCTS[1:]):
        axes[1].plot(
            all_test_r2,
            all_util_gains[pct],
            "o-",
            color=pct_colors[i],
            label=f"drop {pct:.0%}",
            markersize=4,
        )
    axes[1].set_xlabel("Test R²")
    axes[1].set_ylabel("Utility gain % (1.3^(r-1)-1)")
    axes[1].set_title("Utility gain vs model accuracy")
    axes[1].legend(fontsize=7)
    axes[1].axhline(0, color="gray", linestyle=":", alpha=0.5)

    plt.tight_layout()
    plt.savefig(PLOTS / "sim_filtering_vs_r2.png", dpi=150)
    plt.close()
    print(f"  → Saved {PLOTS / 'sim_filtering_vs_r2.png'}")

    # Also: filtering curves (x = pct dropped, y = gain) for select configs
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    select_configs = [(1, 0.15), (1, 0.50), (3, 0.25), (3, 0.50), (5, 0.50)]
    sel_colors = plt.cm.tab10(np.linspace(0, 0.5, len(select_configs)))

    for j, (n_sig, r) in enumerate(select_configs):
        idx = configs.index((n_sig, r))
        gains = [all_gains[pct][idx] for pct in FILTER_PCTS]
        utils = [all_util_gains[pct][idx] for pct in FILTER_PCTS]
        pcts_100 = [p * 100 for p in FILTER_PCTS]
        label = f"{n_sig}sig r={r:.2f} (R²={all_test_r2[idx]:.3f})"
        axes[0].plot(
            pcts_100, gains, "o-", color=sel_colors[j], label=label, markersize=4
        )
        axes[1].plot(
            pcts_100, utils, "o-", color=sel_colors[j], label=label, markersize=4
        )

    axes[0].set_xlabel("% of books dropped (by prediction)")
    axes[0].set_ylabel("Avg rating gain of kept books")
    axes[0].set_title("Filtering curves by signal strength")
    axes[0].legend(fontsize=7)
    axes[0].axhline(0, color="gray", linestyle=":", alpha=0.5)

    axes[1].set_xlabel("% of books dropped")
    axes[1].set_ylabel("Utility gain %")
    axes[1].set_title("Utility filtering curves")
    axes[1].legend(fontsize=7)
    axes[1].axhline(0, color="gray", linestyle=":", alpha=0.5)

    plt.tight_layout()
    plt.savefig(PLOTS / "sim_filtering_curves.png", dpi=150)
    plt.close()
    print(f"  → Saved {PLOTS / 'sim_filtering_curves.png'}")

    # Print table
    print("\n  Filtering gains (avg rating increase over baseline):")
    header = f"  {'Config':30s}"
    for pct in [0.20, 0.50, 0.80, 0.95]:
        header += f"  drop{pct:.0%}"
    print(header)
    for j, (n_sig, r) in enumerate(configs):
        line = f"  {n_sig}sig r={r:.2f} (R²={all_test_r2[j]:.3f})"
        for pct in [0.20, 0.50, 0.80, 0.95]:
            line += f"  {all_gains[pct][j]:+.3f}"
        print(line)


def part3_transform_effects() -> None:
    """What happens if we train on utility(y) and forget to inverse-transform."""
    print("\n" + "=" * 70)
    print("PART 3: Transform effects")
    print("=" * 70)

    rng = np.random.default_rng(456)
    configs_transform = [
        ("Raw ratings", False, False),
        ("Utility → correct inverse", True, False),
        ("Utility → forgot inverse", True, True),
    ]

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    colors_t = ["#2196F3", "#4CAF50", "#F44336"]

    for i, (label, ut, fi) in enumerate(configs_transform):
        test_r2s: list[float] = []
        gains_50: list[float] = []
        for n_sig in N_SIGNALS_LIST:
            res = run_sim(
                N_TRAIN,
                N_TEST,
                n_sig,
                [0.35],
                N_SIMS,
                rng,
                utility_transform=ut,
                forget_inverse=fi,
            )
            test_r2s.append(res["test_r2"].mean())
            gains_50.append(res["filter_gains"][0.50].mean())

        axes[0].plot(N_SIGNALS_LIST, test_r2s, "o-", color=colors_t[i], label=label)
        axes[1].plot(N_SIGNALS_LIST, gains_50, "o-", color=colors_t[i], label=label)
        print(f"  {label}:")
        for j, n in enumerate(N_SIGNALS_LIST):
            print(
                f"    {n} signals: test R²={test_r2s[j]:.4f}, gain@50%={gains_50[j]:+.3f}"
            )

    axes[0].set_xlabel("Number of signals (each r=0.35)")
    axes[0].set_ylabel("Test R²")
    axes[0].set_title("Transform effect on R²")
    axes[0].legend(fontsize=8)

    axes[1].set_xlabel("Number of signals")
    axes[1].set_ylabel("Avg rating gain (drop bottom 50%)")
    axes[1].set_title("Transform effect on filtering gain")
    axes[1].legend(fontsize=8)

    plt.tight_layout()
    plt.savefig(PLOTS / "sim_transform_effects.png", dpi=150)
    plt.close()
    print(f"  → Saved {PLOTS / 'sim_transform_effects.png'}")


def part4_empirical_distributions() -> None:
    """Use actual rating distributions instead of normal."""
    print("\n" + "=" * 70)
    print("PART 4: Empirical vs normal distributions")
    print("=" * 70)

    df = load_and_join()
    df = impute_missing(df)
    empirical_enjoy = df["avg_enjoyment"].dropna().values

    rng = np.random.default_rng(789)

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    dist_configs = [
        ("Normal(3.3, 0.885)", None),
        ("Empirical enjoyment dist", empirical_enjoy),
    ]
    # Top row: R² comparison
    for di, (dlabel, emp) in enumerate(dist_configs):
        test_r2_by_nsig: dict[int, list[float]] = {n: [] for n in N_SIGNALS_LIST}
        for n_sig in N_SIGNALS_LIST:
            for r in SIGNAL_CORRS:
                res = run_sim(N_TRAIN, N_TEST, n_sig, [r], N_SIMS, rng, empirical_y=emp)
                test_r2_by_nsig[n_sig].append(res["test_r2"].mean())

        for n_sig in N_SIGNALS_LIST:
            axes[0, di].plot(
                SIGNAL_CORRS,
                test_r2_by_nsig[n_sig],
                "o-",
                label=f"{n_sig} signals",
                markersize=4,
            )
        axes[0, di].set_xlabel("Signal correlation")
        axes[0, di].set_ylabel("Test R²")
        axes[0, di].set_title(f"Test R² — {dlabel}")
        axes[0, di].legend(fontsize=7)

    # Bottom row: filtering gains
    for di, (dlabel, emp) in enumerate(dist_configs):
        for n_sig in [1, 3, 5]:
            gains_by_pct: list[float] = []
            for pct in FILTER_PCTS:
                res = run_sim(
                    N_TRAIN, N_TEST, n_sig, [0.35], 1000, rng, empirical_y=emp
                )
                gains_by_pct.append(res["filter_gains"][pct].mean())
            axes[1, di].plot(
                [p * 100 for p in FILTER_PCTS],
                gains_by_pct,
                "o-",
                label=f"{n_sig} sig, r=0.35",
                markersize=4,
            )
        axes[1, di].set_xlabel("% dropped")
        axes[1, di].set_ylabel("Avg rating gain")
        axes[1, di].set_title(f"Filtering gains — {dlabel}")
        axes[1, di].legend(fontsize=7)
        axes[1, di].axhline(0, color="gray", linestyle=":", alpha=0.5)

    plt.tight_layout()
    plt.savefig(PLOTS / "sim_empirical_vs_normal.png", dpi=150)
    plt.close()
    print(f"  → Saved {PLOTS / 'sim_empirical_vs_normal.png'}")


def part5_heterogeneous_and_ceiling() -> None:
    """Heterogeneous signal strengths + self-noise ceiling."""
    print("\n" + "=" * 70)
    print("PART 5: Heterogeneous signals + self-noise ceiling")
    print("=" * 70)

    rng = np.random.default_rng(999)
    df = load_and_join()
    df = impute_missing(df)
    empirical_enjoy = df["avg_enjoyment"].dropna().values

    # Heterogeneous configs (matching real: GR~0.22, OL~0.12, AMZ~0.15)
    hetero_configs: list[tuple[str, list[float]]] = [
        ("Empirical (0.22, 0.12, 0.15)", [0.22, 0.12, 0.15]),
        ("Modest (0.15, 0.30, 0.45)", [0.15, 0.30, 0.45]),
        ("Strong (0.30, 0.40, 0.50)", [0.30, 0.40, 0.50]),
        ("3× same r=0.35", [0.35, 0.35, 0.35]),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Top-left: test R² for heterogeneous configs, no ceiling
    print("\n  Heterogeneous signals (no self-noise):")
    for label, corrs in hetero_configs:
        res = run_sim(
            N_TRAIN, N_TEST, len(corrs), corrs, N_SIMS, rng, empirical_y=empirical_enjoy
        )
        print(
            f"    {label}: train R²={res['train_r2'].mean():.4f}, "
            f"test R²={res['test_r2'].mean():.4f}"
        )

    # Top-left: self-noise sweep
    noise_fracs = [0.0, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60]
    for r in [0.25, 0.35, 0.50]:
        test_r2s = []
        for nf in noise_fracs:
            res = run_sim(
                N_TRAIN,
                N_TEST,
                3,
                [r],
                N_SIMS,
                rng,
                empirical_y=empirical_enjoy,
                self_noise_frac=nf,
            )
            test_r2s.append(res["test_r2"].mean())
        axes[0, 0].plot(
            [nf * 100 for nf in noise_fracs],
            test_r2s,
            "o-",
            label=f"3sig r={r}",
            markersize=4,
        )
    axes[0, 0].axvline(
        40, color="red", linestyle="--", alpha=0.7, label="Your noise=40%"
    )
    axes[0, 0].set_xlabel("Self-rating noise (% of variance)")
    axes[0, 0].set_ylabel("Test R²")
    axes[0, 0].set_title("R² ceiling from self-noise")
    axes[0, 0].legend(fontsize=7)

    # Top-right: filtering gain with vs without self-noise
    for nf, ls in [(0.0, "-"), (0.40, "--")]:
        for r in [0.25, 0.50]:
            gains = []
            for pct in FILTER_PCTS:
                res = run_sim(
                    N_TRAIN,
                    N_TEST,
                    3,
                    [r],
                    N_SIMS,
                    rng,
                    empirical_y=empirical_enjoy,
                    self_noise_frac=nf,
                )
                gains.append(res["filter_gains"][pct].mean())
            nf_label = f"noise={nf:.0%}" if nf > 0 else "no noise"
            axes[0, 1].plot(
                [p * 100 for p in FILTER_PCTS],
                gains,
                f"o{ls}",
                label=f"r={r}, {nf_label}",
                markersize=4,
            )
    axes[0, 1].set_xlabel("% dropped")
    axes[0, 1].set_ylabel("Avg rating gain")
    axes[0, 1].set_title("Filtering with/without self-noise")
    axes[0, 1].legend(fontsize=7)
    axes[0, 1].axhline(0, color="gray", linestyle=":", alpha=0.5)

    # Bottom-left: nonlinear latent (unmodeled categories)
    print("\n  Effect of unmodeled nonlinear latent:")
    for nl, ls in [(False, "-"), (True, "--")]:
        for r in [0.25, 0.50]:
            test_r2s_nl = []
            for n_sig in N_SIGNALS_LIST:
                res = run_sim(
                    N_TRAIN,
                    N_TEST,
                    n_sig,
                    [r],
                    N_SIMS,
                    rng,
                    empirical_y=empirical_enjoy,
                    nonlinear_latent=nl,
                )
                test_r2s_nl.append(res["test_r2"].mean())
            nl_label = "nonlinear" if nl else "linear"
            axes[1, 0].plot(
                N_SIGNALS_LIST,
                test_r2s_nl,
                f"o{ls}",
                label=f"r={r}, {nl_label}",
                markersize=4,
            )
            if nl:
                print(
                    f"    r={r}, nonlinear: test R²s = "
                    + ", ".join(f"{v:.4f}" for v in test_r2s_nl)
                )
    axes[1, 0].set_xlabel("Number of signals")
    axes[1, 0].set_ylabel("Test R²")
    axes[1, 0].set_title("Linear DGP vs unmodeled category effects")
    axes[1, 0].legend(fontsize=7)

    # Bottom-right: combined realistic scenario
    # (empirical dist, 40% self-noise, 3 heterogeneous signals)
    print("\n  Realistic scenario (empirical dist, 40% noise, heterogeneous signals):")
    realistic_configs: list[tuple[str, list[float], float]] = [
        ("Actual empirical (r=0.22,0.12,0.15)", [0.22, 0.12, 0.15], 0.40),
        ("If we had better data (r=0.35,0.25,0.20)", [0.35, 0.25, 0.20], 0.40),
        ("Optimistic (r=0.50,0.35,0.25)", [0.50, 0.35, 0.25], 0.40),
        ("Best case no noise (r=0.50,0.35,0.25)", [0.50, 0.35, 0.25], 0.0),
    ]
    real_colors = plt.cm.RdYlGn(np.linspace(0.15, 0.85, len(realistic_configs)))

    for j, (label, corrs, nf) in enumerate(realistic_configs):
        gains = []
        utils = []
        for pct in FILTER_PCTS:
            res = run_sim(
                N_TRAIN,
                N_TEST,
                len(corrs),
                corrs,
                N_SIMS,
                rng,
                empirical_y=empirical_enjoy,
                self_noise_frac=nf,
            )
            gains.append(res["filter_gains"][pct].mean())
            utils.append(res["util_gains"][pct].mean())
        axes[1, 1].plot(
            [p * 100 for p in FILTER_PCTS],
            gains,
            "o-",
            color=real_colors[j],
            label=label,
            markersize=4,
        )
        print(
            f"    {label}: gain@50%={gains[FILTER_PCTS.index(0.50)]:+.3f}, "
            f"gain@80%={gains[FILTER_PCTS.index(0.80)]:+.3f}, "
            f"util@50%={utils[FILTER_PCTS.index(0.50)]:+.1f}%"
        )

    axes[1, 1].set_xlabel("% dropped")
    axes[1, 1].set_ylabel("Avg rating gain")
    axes[1, 1].set_title("Realistic scenarios: filtering gains")
    axes[1, 1].legend(fontsize=6)
    axes[1, 1].axhline(0, color="gray", linestyle=":", alpha=0.5)

    plt.tight_layout()
    plt.savefig(PLOTS / "sim_ceiling_and_realistic.png", dpi=150)
    plt.close()
    print(f"  → Saved {PLOTS / 'sim_ceiling_and_realistic.png'}")


def part6_comprehensive_summary() -> None:
    """Big summary table and key plot: R² → filtering gain with annotations."""
    print("\n" + "=" * 70)
    print("PART 6: Comprehensive summary — what R² buys you in practice")
    print("=" * 70)

    rng = np.random.default_rng(2024)
    df = load_and_join()
    df = impute_missing(df)
    empirical_enjoy = df["avg_enjoyment"].dropna().values

    # Sweep R² from near-zero to theoretical max
    # Use 3 signals, vary correlation, with 40% self-noise
    sweep_corrs = np.arange(0.05, 0.71, 0.05)
    test_r2_list: list[float] = []
    gain_50_list: list[float] = []
    gain_80_list: list[float] = []
    util_50_list: list[float] = []
    util_80_list: list[float] = []

    for r in sweep_corrs:
        res = run_sim(
            N_TRAIN,
            N_TEST,
            3,
            [float(r)],
            N_SIMS,
            rng,
            empirical_y=empirical_enjoy,
            self_noise_frac=0.40,
        )
        test_r2_list.append(res["test_r2"].mean())
        gain_50_list.append(res["filter_gains"][0.50].mean())
        gain_80_list.append(res["filter_gains"][0.80].mean())
        util_50_list.append(res["util_gains"][0.50].mean())
        util_80_list.append(res["util_gains"][0.80].mean())

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Left: R² → rating gain
    axes[0].plot(test_r2_list, gain_50_list, "o-", label="Drop 50%", markersize=4)
    axes[0].plot(test_r2_list, gain_80_list, "s-", label="Drop 80%", markersize=4)
    axes[0].set_xlabel("Test R² (with 40% self-noise)")
    axes[0].set_ylabel("Avg rating gain of kept books")
    axes[0].set_title("What model accuracy buys you")
    axes[0].legend()
    axes[0].axhline(0, color="gray", linestyle=":", alpha=0.5)
    # Mark empirical R²
    empirical_r2 = 0.049  # GR-only
    axes[0].axvline(
        empirical_r2,
        color="red",
        linestyle="--",
        alpha=0.7,
        label=f"Current GR-only R²={empirical_r2}",
    )
    axes[0].legend(fontsize=8)

    # Right: R² → utility gain %
    axes[1].plot(test_r2_list, util_50_list, "o-", label="Drop 50%", markersize=4)
    axes[1].plot(test_r2_list, util_80_list, "s-", label="Drop 80%", markersize=4)
    axes[1].set_xlabel("Test R² (with 40% self-noise)")
    axes[1].set_ylabel("Utility gain % (1.3^(r-1)-1)")
    axes[1].set_title("Utility gains by model accuracy")
    axes[1].legend()
    axes[1].axhline(0, color="gray", linestyle=":", alpha=0.5)
    axes[1].axvline(empirical_r2, color="red", linestyle="--", alpha=0.7)

    plt.tight_layout()
    plt.savefig(PLOTS / "sim_r2_to_gains_summary.png", dpi=150)
    plt.close()
    print(f"  → Saved {PLOTS / 'sim_r2_to_gains_summary.png'}")

    # Summary table
    print("\n  Summary: 3 signals, 40% self-noise, empirical distribution")
    print(
        f"  {'Signal r':>10s} {'Test R²':>8s} {'Gain@50%':>9s} {'Gain@80%':>9s} "
        f"{'Util@50%':>9s} {'Util@80%':>9s}"
    )
    for j, r in enumerate(sweep_corrs):
        print(
            f"  {r:10.2f} {test_r2_list[j]:8.4f} {gain_50_list[j]:+9.3f} "
            f"{gain_80_list[j]:+9.3f} {util_50_list[j]:+9.1f}% {util_80_list[j]:+9.1f}%"
        )


def main() -> None:
    part1_accuracy_vs_signals()
    part2_filtering_gains()
    part3_transform_effects()
    part4_empirical_distributions()
    part5_heterogeneous_and_ceiling()
    part6_comprehensive_summary()

    print("\n" + "=" * 70)
    print("ALL SIMULATION PLOTS SAVED")
    print("=" * 70)


if __name__ == "__main__":
    main()
