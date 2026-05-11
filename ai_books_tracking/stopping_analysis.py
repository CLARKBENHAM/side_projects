"""Deep analysis of the stopping rules from play_books_greedy_optimal.py.

Key questions:
1. How sensitive are the results to the error_sigma function shape?
   (power law vs constant vs fast-then-flat)
2. How sensitive to the VALUE_BASE parameter?
3. How sensitive to SEARCH_COST_HOURS?
4. What does the intuition "you can tell within 15 pages" imply for the error function?
5. Can we validate the error function shape empirically?
"""

import pandas as pd
import numpy as np
import numpy.random as rnd
from pathlib import Path
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

DATA_DIR = Path(__file__).parent.parent / "data"
OUTPUT_DIR = Path(__file__).parent
PLAY_EXPORT = DATA_DIR / "Books Read and their effects - Play Export.csv"
RATINGS_2 = DATA_DIR / "Books Read and their effects - Ratings 2.csv"

# Base parameters (from play_books_greedy_optimal.py)
READ_TIME_HOURS = 3.5
SEARCH_COST_HOURS = 0.25
PARTIAL_RATING_PENALTY = 0.1
VALUE_BASE = 2

F_GRID = np.concatenate(
    [
        np.arange(0.01, 0.4, 0.02),
        np.arange(0.4, 1.01, 0.1),
    ]
)


def utility_value(r: np.ndarray | float) -> np.ndarray | float:
    return VALUE_BASE ** (np.asarray(r) - 1) - 1


def inverse_utility_value(u: np.ndarray | float) -> np.ndarray | float:
    return np.log(np.asarray(u) + 1) / np.log(VALUE_BASE) + 1


def load_data() -> pd.DataFrame:
    df = pd.read_csv(PLAY_EXPORT)
    df.columns = df.columns.str.strip()
    return df


def analyze_error_function_shapes() -> None:
    """Compare different assumptions about how estimation error decreases with reading progress."""
    print("=" * 70)
    print("1. ERROR FUNCTION SHAPE ANALYSIS")
    print("=" * 70)

    # The original error function
    def error_original(f: float) -> float:
        return 2 * (1 - f) ** 1.8 + 0.25

    # Alternative: you can tell within 15 pages (~5% of a 300-page book)
    # Error drops fast initially, then plateaus
    def error_fast_start(f: float) -> float:
        return 2 * (1 - f) ** 0.5 + 0.25

    # Alternative: constant error until the end
    def error_flat(f: float) -> float:
        return max(2.25 * (1 - f), 0.25)

    # Alternative: step function - you basically know at 5%
    def error_step(f: float) -> float:
        if f < 0.05:
            return 2.25
        return 0.5

    # Alternative: slow learning (quadratic)
    def error_slow(f: float) -> float:
        return 2 * (1 - f) ** 3 + 0.25

    error_fns = {
        "Original (power 1.8)": error_original,
        "Fast start (power 0.5)": error_fast_start,
        "Linear": error_flat,
        "Step at 5%": error_step,
        "Slow (power 3)": error_slow,
    }

    # Plot the error functions
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    f_plot = np.linspace(0, 1, 200)
    for name, fn in error_fns.items():
        errors = [fn(f) for f in f_plot]
        ax1.plot(f_plot, errors, label=name, linewidth=2)
    ax1.set_xlabel("Fraction Read")
    ax1.set_ylabel("Error Half-Width (rating points)")
    ax1.set_title("How Estimation Error Decreases with Reading Progress")
    ax1.legend()
    ax1.grid(alpha=0.3)
    ax1.set_ylim(0, 2.5)

    # For each error function, compute the optimal drop schedule for fiction
    df = load_data()
    fiction = df[df["Bookshelf"] == "fiction"]["Enjoyment (/5)"].dropna().values

    drop_schedules = {}
    for name, fn in error_fns.items():
        drops = simulate_drop_schedule(fiction, fn)
        drop_schedules[name] = drops
        cum_drop = 1 - np.cumprod(1 - drops)
        ax2.plot(F_GRID, cum_drop, label=name, linewidth=2)

    ax2.set_xlabel("Fraction Read")
    ax2.set_ylabel("Cumulative Drop Fraction")
    ax2.set_title("Optimal Dropping for Fiction Under Different Error Models")
    ax2.legend()
    ax2.grid(alpha=0.3)
    ax2.set_ylim(0, 1)

    plt.tight_layout()
    fig.savefig(
        OUTPUT_DIR / "error_function_sensitivity.png", dpi=150, bbox_inches="tight"
    )
    print(f"Saved plot to {OUTPUT_DIR / 'error_function_sensitivity.png'}")
    plt.close()

    # Print summary
    print("\nFinal cumulative drop % for fiction under each error model:")
    for name, drops in drop_schedules.items():
        cum_drop_final = 1 - np.prod(1 - drops)
        early_drop = 1 - np.prod(1 - drops[:5])  # first ~10%
        print(f"  {name:<30}: final={cum_drop_final:.1%}, by 10%={early_drop:.1%}")


def simulate_drop_schedule(
    true_ratings: np.ndarray,
    error_fn: callable,
    n_sim: int = 100,
    value_base: float = VALUE_BASE,
    search_cost: float = SEARCH_COST_HOURS,
) -> np.ndarray:
    """Simplified greedy optimal stopping simulation.
    Returns average instant drop fraction at each F_GRID point."""

    def util(r: np.ndarray) -> np.ndarray:
        return value_base ** (np.asarray(r) - 1) - 1

    book_time = READ_TIME_HOURS + search_cost
    hourly_opp = np.percentile(util(true_ratings), 30) / book_time

    all_drops = np.zeros((n_sim, len(F_GRID)))

    for sim in range(n_sim):
        n = len(true_ratings)
        active = np.ones(n, dtype=bool)
        # Generate correlated estimates
        est = _simulate_estimates(true_ratings, error_fn)

        for idx_f, f in enumerate(F_GRID):
            if f >= 1.0 or active.sum() == 0:
                continue

            est_now = est[:, idx_f]
            a_est = est_now[active]
            remaining_t = (1 - f) * READ_TIME_HOURS

            # Estimated marginal utility rate of continuing
            est_finish_u = util(a_est)
            est_partial_u = util(np.maximum(a_est - PARTIAL_RATING_PENALTY, 1))
            current_u = f * est_partial_u
            marginal_rate = (est_finish_u - current_u) / max(remaining_t, 0.01)

            # Drop books where marginal rate < opportunity cost
            drop_mask = marginal_rate < hourly_opp
            n_drop = drop_mask.sum()

            if n_drop > 0:
                all_drops[sim, idx_f] = n_drop / active.sum()
                active_indices = np.where(active)[0]
                active[active_indices[drop_mask]] = False

    return all_drops.mean(axis=0)


def _simulate_estimates(
    true_ratings: np.ndarray,
    error_fn: callable,
    rho: float = 0.9,
) -> np.ndarray:
    """AR(1) noisy estimates."""
    n = len(true_ratings)
    est = np.zeros((n, len(F_GRID)))
    sigmas = np.array([error_fn(f) for f in F_GRID])

    e_prev = rnd.uniform(-sigmas[0], sigmas[0], size=n)
    est[:, 0] = np.clip(true_ratings + e_prev, 1, 5)

    for j in range(1, len(F_GRID)):
        scale = sigmas[j] / sigmas[j - 1] if sigmas[j - 1] > 0 else 0
        e_auto = rho * e_prev * scale
        e_innov = np.sqrt(1 - rho**2) * sigmas[j] * rnd.normal(0, 1, n)
        e_current = e_auto + e_innov
        est[:, j] = np.clip(true_ratings + e_current, 1, 5)
        e_prev = e_current

    return est


def parameter_sensitivity() -> None:
    """Test sensitivity to VALUE_BASE, SEARCH_COST, and autocorrelation."""
    print("\n" + "=" * 70)
    print("2. PARAMETER SENSITIVITY")
    print("=" * 70)

    df = load_data()
    fiction = df[df["Bookshelf"] == "fiction"]["Enjoyment (/5)"].dropna().values

    def error_original(f: float) -> float:
        return 2 * (1 - f) ** 1.8 + 0.25

    # Vary VALUE_BASE
    print("\nVarying VALUE_BASE (utility = BASE^(r-1) - 1):")
    print(f"  {'BASE':>6} {'Final Drop%':>12} {'Drop by 10%':>12}")
    for base in [1.5, 2.0, 2.5, 3.0, 4.0]:
        drops = simulate_drop_schedule(fiction, error_original, value_base=base)
        cum_final = 1 - np.prod(1 - drops)
        early = 1 - np.prod(1 - drops[:5])
        print(f"  {base:>6.1f} {cum_final:>12.1%} {early:>12.1%}")

    # Vary SEARCH_COST
    print("\nVarying SEARCH_COST_HOURS:")
    print(f"  {'Cost':>6} {'Final Drop%':>12} {'Drop by 10%':>12}")
    for cost in [0.0, 0.1, 0.25, 0.5, 1.0, 2.0]:
        drops = simulate_drop_schedule(fiction, error_original, search_cost=cost)
        cum_final = 1 - np.prod(1 - drops)
        early = 1 - np.prod(1 - drops[:5])
        print(f"  {cost:>6.2f} {cum_final:>12.1%} {early:>12.1%}")


def validate_error_function() -> None:
    """Use the re-test reliability data to empirically validate the error function.

    The re-test data gives us error at f=1.0 (after finishing).
    We know error_sigma(1.0) should match the observed re-test RMSE.
    """
    print("\n" + "=" * 70)
    print("3. EMPIRICAL VALIDATION OF ERROR FUNCTION")
    print("=" * 70)

    df1 = pd.read_csv(PLAY_EXPORT)
    df2 = pd.read_csv(RATINGS_2)
    df1.columns = df1.columns.str.strip()
    df2.columns = df2.columns.str.strip()

    merged = pd.merge(
        df1[["title", "Enjoyment (/5)", "Usefulness /5 to Me"]],
        df2[["title", "Enjoyment (/5)", "Usefulness /5 to Me"]],
        on="title",
        suffixes=("_1", "_2"),
    )

    enjoy_diff = merged["Enjoyment (/5)_1"].astype(float) - merged[
        "Enjoyment (/5)_2"
    ].astype(float)
    enjoy_diff = enjoy_diff.dropna()

    rmse_at_f1 = np.sqrt((enjoy_diff**2).mean())
    mad_at_f1 = np.abs(enjoy_diff).mean()
    sd_at_f1 = enjoy_diff.std()

    print("\nRe-test error at f=1.0 (after finishing the book):")
    print(f"  RMSE = {rmse_at_f1:.3f}")
    print(f"  MAD  = {mad_at_f1:.3f}")
    print(f"  SD   = {sd_at_f1:.3f}")

    # The error_sigma function gives half-width of uniform noise
    # For uniform[-s, s]: SD = s/sqrt(3), so s = SD * sqrt(3)
    # Or if we interpret sigma as SD directly: sigma(1) should = 0.66
    def error_original(f: float) -> float:
        return 2 * (1 - f) ** 1.8 + 0.25

    sigma_at_f1 = error_original(1.0)
    sigma_at_f0 = error_original(0.0)
    print("\nOriginal error_sigma function:")
    print(f"  sigma(0.0) = {sigma_at_f0:.3f} (initial uncertainty)")
    print(f"  sigma(1.0) = {sigma_at_f1:.3f} (after finishing)")
    print(f"  Empirical re-test SD = {sd_at_f1:.3f}")
    print(f"  Ratio: sigma(1)/empirical = {sigma_at_f1 / sd_at_f1:.2f}")
    print(
        f"  -> The residual error ({sigma_at_f1:.2f}) is {'close to' if abs(sigma_at_f1 - sd_at_f1) < 0.2 else 'different from'} "
        f"the empirical re-test SD ({sd_at_f1:.2f})"
    )

    # What the blog says about partway estimates
    print("\nThe blog mentions being able to rate books ~0.6 pts off after finishing.")
    print(f"The error function gives sigma(1) = {sigma_at_f1:.2f}, which is used as")
    print(
        f"half-width of uniform noise. Uniform[-{sigma_at_f1:.2f}, {sigma_at_f1:.2f}] has"
    )
    print(
        f"SD = {sigma_at_f1 / np.sqrt(3):.2f}, which is {'too low' if sigma_at_f1/np.sqrt(3) < sd_at_f1*0.8 else 'comparable'} vs empirical {sd_at_f1:.2f}"
    )

    # Calibrated alternative
    print("\nCalibrated error function (matching empirical re-test at f=1):")
    # If sigma(1) should equal empirical SD for normal noise:
    calibrated_floor = sd_at_f1
    print(f"  Floor should be ~{calibrated_floor:.2f} (not {sigma_at_f1:.2f})")

    # What about at the start? The blog says z=2.07 at 1/3 of the way through
    # with mean estimate of 2.5 (true range 1-5)
    # sigma(0) corresponds to "I generally have a bit of info after deciding to pick the book"
    # The code comment says: "at 0 2.25 is correct noise"
    print(
        f"  Start noise: sigma(0) = {sigma_at_f0:.2f} (code comment says 2.25 is correct)"
    )

    def error_calibrated(f: float) -> float:
        return 2 * (1 - f) ** 1.8 + calibrated_floor

    # Compare: does the shape matter more or the floor?
    print("\nDoes the floor or the shape matter more?")
    fiction = load_data()
    fiction = (
        fiction[fiction["Bookshelf"] == "fiction"]["Enjoyment (/5)"].dropna().values
    )

    test_fns = {
        f"Original (floor={sigma_at_f1:.2f})": error_original,
        f"Calibrated (floor={calibrated_floor:.2f})": error_calibrated,
    }
    for name, fn in test_fns.items():
        drops = simulate_drop_schedule(fiction, fn)
        cum_final = 1 - np.prod(1 - drops)
        print(f"  {name:<40}: final drop = {cum_final:.1%}")


def intuition_check() -> None:
    """Check: 'you should be able to tell within 15 pages' intuition.

    15 pages of a 300 page book = 5%. If you can tell by then, the error function
    should be nearly flat after f=0.05.
    """
    print("\n" + "=" * 70)
    print("4. '15 PAGES' INTUITION CHECK")
    print("=" * 70)

    def error_original(f: float) -> float:
        return 2 * (1 - f) ** 1.8 + 0.25

    f_values = [0.0, 0.02, 0.05, 0.10, 0.15, 0.20, 0.30, 0.50, 1.0]
    print("\nError at different reading fractions:")
    print(
        f"  {'Fraction':>10} {'~Pages (300pg)':>15} {'Error (original)':>18} {'% of initial':>15}"
    )
    initial = error_original(0.0)
    for f in f_values:
        e = error_original(f)
        pct = (e / initial) * 100
        pages = int(f * 300)
        print(f"  {f:>10.2f} {pages:>15} {e:>18.3f} {pct:>14.0f}%")

    print(
        f"\nAt 5% (15 pages), error has dropped to {error_original(0.05)/initial:.0%} of initial."
    )
    print(
        f"At 10% (30 pages), error has dropped to {error_original(0.10)/initial:.0%} of initial."
    )
    print(
        f"At 33% (100 pages), error has dropped to {error_original(0.33)/initial:.0%} of initial."
    )

    print("\nImplication: With the original power-1.8 model, you still have")
    print(
        f"  {error_original(0.05):.2f}/{initial:.2f} = {error_original(0.05)/initial:.0%} of your initial uncertainty at 15 pages."
    )
    print("  If the '15 pages' intuition is correct, the error function should drop")
    print("  much faster initially (power < 1, like 0.3-0.5).")

    # What power law matches "most info by 5%"?
    # error(0.05) should be ~40% of error(0) (you've learned 60% of what you'll learn)
    # 2*(1-0.05)^k + 0.25 = 0.4 * (2*(1-0)^k + 0.25) = 0.4*2.25 = 0.9
    # 2*0.95^k = 0.65
    # k = log(0.325) / log(0.95)
    target_ratio = 0.4
    target_error = target_ratio * initial
    # 2*(0.95)^k + 0.25 = target_error
    k_needed = np.log((target_error - 0.25) / 2) / np.log(0.95)
    print(f"\n  To have 60% info learned by 5%: need power = {k_needed:.1f}")
    print("  Original uses power = 1.8")
    print(
        f"  Ratio: the '15 pages' intuition needs ~{k_needed/1.8:.0f}x faster learning"
    )


def category_specific_stopping() -> None:
    """Do different categories have different optimal error functions?"""
    print("\n" + "=" * 70)
    print("5. CATEGORY-SPECIFIC STOPPING")
    print("=" * 70)

    df = load_data()

    def error_original(f: float) -> float:
        return 2 * (1 - f) ** 1.8 + 0.25

    major_cats = [
        "Business, management",
        "Computer Science",
        "fiction",
        "General Reading",
        "Literature",
    ]

    fig, axes = plt.subplots(1, len(major_cats), figsize=(20, 5), sharey=True)

    print(
        f"\n{'Category':<25} {'N':>4} {'Final Drop':>11} {'By 5%':>8} {'By 10%':>8} "
        f"{'By 33%':>8} {'Mean Enjoy':>11}"
    )
    print("-" * 80)

    for i, cat in enumerate(major_cats):
        ratings = df[df["Bookshelf"] == cat]["Enjoyment (/5)"].dropna().values
        if len(ratings) < 5:
            continue

        drops = simulate_drop_schedule(ratings, error_original)
        cum = 1 - np.cumprod(1 - drops)

        # Find indices
        idx_5 = np.abs(F_GRID - 0.05).argmin()
        idx_10 = np.abs(F_GRID - 0.10).argmin()
        idx_33 = np.abs(F_GRID - 0.33).argmin()

        final_drop = cum[-1]
        drop_5 = cum[idx_5]
        drop_10 = cum[idx_10]
        drop_33 = cum[idx_33]

        print(
            f"  {cat:<25} {len(ratings):>4} {final_drop:>11.1%} {drop_5:>8.1%} "
            f"{drop_10:>8.1%} {drop_33:>8.1%} {ratings.mean():>11.2f}"
        )

        axes[i].plot(F_GRID, cum, "b-", linewidth=2)
        axes[i].set_title(f"{cat}\n(n={len(ratings)})")
        axes[i].set_xlabel("Fraction Read")
        if i == 0:
            axes[i].set_ylabel("Cumulative Drop Fraction")
        axes[i].set_ylim(0, 1)
        axes[i].grid(alpha=0.3)
        axes[i].axvline(x=0.05, color="r", linestyle="--", alpha=0.5, label="5% (15pg)")
        axes[i].axvline(
            x=0.33, color="g", linestyle="--", alpha=0.5, label="33% (100pg)"
        )
        if i == len(major_cats) - 1:
            axes[i].legend(fontsize=8)

    plt.suptitle("Optimal Cumulative Drop Schedule by Category", fontsize=14)
    plt.tight_layout()
    fig.savefig(OUTPUT_DIR / "stopping_by_category.png", dpi=150, bbox_inches="tight")
    print(f"\nSaved plot to {OUTPUT_DIR / 'stopping_by_category.png'}")
    plt.close()


def marginal_info_analysis() -> None:
    """How much MORE info do you get from pages 15-100 vs pages 1-15?"""
    print("\n" + "=" * 70)
    print("6. MARGINAL INFORMATION VALUE")
    print("=" * 70)

    def error_original(f: float) -> float:
        return 2 * (1 - f) ** 1.8 + 0.25

    f_vals = np.linspace(0, 1, 1000)
    errors = np.array([error_original(f) for f in f_vals])

    # Information gained = reduction in error
    initial_error = errors[0]
    info_gained = initial_error - errors  # cumulative info

    # Find info gained in different intervals
    intervals = [(0, 0.05), (0.05, 0.10), (0.10, 0.33), (0.33, 0.50), (0.50, 1.0)]
    print("\nInformation gained by reading interval (original model):")
    print(
        f"  {'Interval':<15} {'Pages (300pg)':>15} {'Info Gained':>12} {'% of Total':>12}"
    )
    total_info = info_gained[-1]
    for start, end in intervals:
        idx_s = np.abs(f_vals - start).argmin()
        idx_e = np.abs(f_vals - end).argmin()
        delta = info_gained[idx_e] - info_gained[idx_s]
        pages = f"{int(start*300)}-{int(end*300)}"
        print(
            f"  {start:.0%}-{end:.0%}{'':<8} {pages:>15} {delta:>12.3f} {delta/total_info:>12.0%}"
        )

    print(f"\n  Total info: {total_info:.3f}")
    print(
        f"\n  The model says pages 1-15 give you {info_gained[np.abs(f_vals-0.05).argmin()]/total_info:.0%} "
        f"of the total info, while pages 15-100 give another "
        f"{(info_gained[np.abs(f_vals-0.33).argmin()] - info_gained[np.abs(f_vals-0.05).argmin()])/total_info:.0%}."
    )
    print(
        "  Your intuition ('tell within 15 pages') implies the first 5% should give >50% of info."
    )
    print(
        f"  The current model gives only {info_gained[np.abs(f_vals-0.05).argmin()]/total_info:.0%}."
    )


def main() -> None:
    analyze_error_function_shapes()
    parameter_sensitivity()
    validate_error_function()
    intuition_check()
    category_specific_stopping()
    marginal_info_analysis()


if __name__ == "__main__":
    main()
