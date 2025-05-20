# %%
# Explciitly compute the optimal stopping fraction for each category at each reading fraction
# GREEDY CUTS AT EACH STAGE
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List
import pandas as pd
import numpy as np
import os
import math
from pathlib import Path
import numpy.random as rnd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from typing import Dict, List

# ---------------- Parameters ----------------
READ_TIME_HOURS = 3.5  # full reading time
SEARCH_COST_HOURS = 0.25  # discovery cost of getting a new book
PARTIAL_RATING_PENALTY = 0.1  # rating loss when abandoning, assumed utility loss linear
VALUE_BASE = 1.75  # utility = VALUE_BASE ** rating

# Quit levels
QUIT_TABLE = {
    # might 28, wont 39 is slightly off from other time I did counts but I'm not sure which I'll finish
    "Business, management": {"finished": 44, "might finish": 5, "wont finish": 6},
    "Computer Science": {"finished": 14, "might finish": 6, "wont finish": 10},
    "fiction": {"finished": 51, "might finish": 0, "wont finish": 6},
    "General Reading": {"finished": 40, "might finish": 5, "wont finish": 7},
    "Literature": {"finished": 50, "might finish": 3, "wont finish": 6},
    "Machine Learning": {"finished": 5, "might finish": 3, "wont finish": 2},
    "Math": {"finished": 3, "might finish": 6, "wont finish": 2},
}
QUIT_USEFULNESS = 1.2
QUIT_ENJOYMENT = 1.4
# I quit some books since mid or bored, not because I hated them
QUIT_AT_FRACTION = 0.15  # but this would vary a lot?

# Static: O(D*F)
F_GRID = np.concatenate(
    [
        np.arange(0.01, 0.4, 0.02),  # more precise in first half
        np.arange(0.4, 1.01, 0.1),  # less precise in second half
    ]
)
D_GRID = np.concatenate(
    [
        np.arange(0.00, 0.10, 0.01),  # dropping up to 30% in 1 step. Depends on F_GRID size
        np.arange(0.10, 0.31, 0.07),
    ]
)

N_SIM = 200  # MC replications


# ---------------- Utility helpers ----------------
def utility_value(r):
    """Transform rating (0‑5) into cardinal utility."""
    return VALUE_BASE**r


def inverse_utility_value(u):
    """Transform utility into rating."""
    return np.log(u) / np.log(VALUE_BASE)


# Not sure which to go with
def utility_value2(r):
    return r**VALUE_BASE


def inverse_utility_value2(u):
    return u ** (1 / VALUE_BASE)


# had 4/207 books I'd say were "average" at 1/3 (or 1/2 way thru?) and 5 at end;
# z=2.07 and mean of 2.5 implies at half way sd=(5-2.5)/z =1.21 or 1.45 if use mean =2
# at 0.95 I'd say 0.25 is correct noise, my own ratings aren't even that good
# at 0 2.25 is correct noise, I generally have a bit of info after decising to pick book
#   and need to adjust for general range restrictions around means?
# assume has power law form
# 1.21  = 2 * ((1 - 1/3) ** k) + 0.25
# np.log(((1.21-0.25)/2))/np.log(0.667) = 1.8
# np.log(((1.21-0.25)/2))/np.log(0.5) = 1.06
def error_sigma(f):
    """Error half‑width at progress f of uniform noise"""
    return 2 * (1 - f) ** 1.8 + 0.25
    # return max(1 - 2 * f, 0.3)  # ±2.5 at start → ±0.3 by halfway


def error_sigma2(f):
    # return 1 - f**0.5
    # if had as much info at 1/3 as do at end
    # m =(0.6-2.5)/f
    return max(2.5 - 5.7 * f, 0.6)
    # but how does this interact with the AR(1) process?


def simulate_estimates(true_ratings: np.ndarray, error_fn=error_sigma, rho=0.9) -> np.ndarray:
    """
    Simulate noisy rating estimates for each book × each f in F_GRID using an AR(1) process.
    The noise variance decreases as reading progress increases.

    Args:
        true_ratings: True ratings for each book (shape: n_books)

    Returns:
        Array of estimated ratings (shape: n_books, len(F_GRID))
    """
    n_books = true_ratings.shape[0]
    est = np.zeros((n_books, len(F_GRID)))
    sigmas = np.array([error_fn(f) for f in F_GRID])
    # autocorrelation coefficient

    # Generate all random numbers upfront for each book
    # This ensures each book has its own independent sequence
    innovations = np.random.normal(0, 1, size=(n_books, len(F_GRID)))

    # Initial error using uniform noise
    e_prev = np.random.uniform(-sigmas[0], sigmas[0], size=n_books)
    est[:, 0] = np.clip(true_ratings + e_prev, 1, 5)

    # Subsequent errors with autocorrelatio9n
    for j in range(1, len(F_GRID)):
        # Scale previous error by rho
        scale_factor = sigmas[j] / sigmas[j - 1] if sigmas[j - 1] > 0 else 0
        e_autocorr = rho * e_prev * scale_factor

        # Add innovation term scaled by sqrt(1-rho^2) to maintain variance
        e_innovation = np.sqrt(1 - rho**2) * sigmas[j] * innovations[:, j]
        e_current = e_autocorr + e_innovation

        est[:, j] = np.clip(true_ratings + e_current, 1, 5)
        e_prev = e_current

    return est


def _check_error_graph():
    plt.plot(F_GRID, np.array([error_sigma(f) for f in F_GRID]), label="sigma")
    plt.plot(F_GRID, np.array([error_sigma2(f) for f in F_GRID]), label="sigma2")
    plt.legend()
    plt.show()

    # Check dispersion
    for i in [1, 3, 4, 5]:
        plt.plot(
            F_GRID,
            [p for ix, p in enumerate(simulate_estimates(np.array([i] * 30)).T)],
            alpha=0.2,
        )
        plt.title(f"True Value {i}")
        plt.show()


# _check_error_graph()


# ---------------- Core optimiser ----------------
def optimise_schedule_greedy(
    true_ratings: np.ndarray,
    hourly_opportunity=utility_value(2) / (READ_TIME_HOURS + SEARCH_COST_HOURS),
) -> Dict[str, np.ndarray]:
    """
    Using greedy approach at each time step find the drop fraction of estimated books that maximize
    the true final utility.
    returns what fraction of books remaining to drop at each reading fraction f
    Args:
        true_ratings: Array of true ratings for each book
        hourly_opportunity: utility per hour of the marginal book
    returns
        best_cum_drop: fraction of currently active books to drop at each reading fraction f
        best_cutoffs: what rating cutoff to use for each f
        true_avg_utils: average utility from following current strategy and dropping no more books in future
            (dropped books replaced with hourly opportunity cost)
            this is the total util over book_time hours
    """

    n_books = len(true_ratings)
    true_ratings = np.sort(true_ratings)  # easier to debug
    val_full = utility_value(true_ratings)
    val_partial = utility_value(np.maximum(true_ratings - PARTIAL_RATING_PENALTY, 1))

    book_time = READ_TIME_HOURS + SEARCH_COST_HOURS
    if not hourly_opportunity:
        hourly_opportunity = np.percentile(val_full, 50) / book_time  # TODO dynamically update

    best_cum_drop = np.zeros(len(F_GRID))
    best_cutoffs = np.zeros(len(F_GRID))
    true_avg_utils = np.zeros(len(F_GRID))
    dropped_books_utils = np.zeros(
        len(F_GRID)
    )  # Track cumulative projected utility of dropped books and replacment with hourly opportunity

    active_mask = np.ones(n_books, dtype=bool)  # books still being read
    est_matrix = simulate_estimates(true_ratings)  # initial estimates

    for idx_f, f in enumerate(F_GRID):
        est_now = est_matrix[:, idx_f]

        # Only evaluate books still active
        if active_mask.sum() == 0:
            print("WARNING: No active books left")
            best_cum_drop[idx_f:] = 1  # nothing left to drop, keep dropping all
            best_cutoffs[idx_f:] = 0
            true_avg_utils[idx_f:] = true_avg_utils[idx_f - 1]
            break

        # Get estimated utilities for active books
        a_est_full = utility_value(est_now[active_mask])
        a_est_partial = utility_value(np.maximum(est_now[active_mask] - PARTIAL_RATING_PENALTY, 1))

        # Estimated Hourly utilities for remaining books
        u_continue_est = a_est_full / book_time
        cum_value = f * a_est_partial  # if stop
        new_value = (1 - f) * READ_TIME_HOURS * hourly_opportunity
        u_stop_est = (cum_value + new_value) / book_time

        # Difference (negative ⇒ look worse than switching)
        diff = u_continue_est - u_stop_est
        sort_idx = np.argsort(diff)  # ascending: worst first
        # print(
        #     f"Sort idx: {sort_idx} , diff: {diff[sort_idx]} , u_continue_est:"
        #     f" {a_est_full[sort_idx]} , u_stop_est: {est_now[active_mask][sort_idx]}"
        # )

        # True Hourly utilities of remaining books
        active_true_full = val_full[active_mask]
        active_true_part = val_partial[active_mask]
        h_util_keep = active_true_full / book_time
        util_till_drop = f * active_true_part
        util_after_drop = (1 - f) * READ_TIME_HOURS * hourly_opportunity
        h_util_drop = (util_till_drop + util_after_drop) / book_time

        # values if no dropping
        best_drop = 0.0
        best_rating_cut = 0
        if idx_f > 0:
            best_u = true_avg_utils[idx_f - 1]
            best_drop_u = dropped_books_utils[idx_f - 1]
        else:
            best_u = val_full.sum()
            best_drop_u = 0

        for d in D_GRID:
            k_drop = int(np.floor(d * active_mask.sum()))  # number of books to drop

            # Get indices of books to keep and drop
            drop_set = sort_idx[:k_drop]
            keep_mask = np.ones(len(sort_idx), dtype=bool)
            keep_mask[drop_set] = False
            # include utility from previously dropped books

            h_total_u = h_util_keep[keep_mask].sum() + h_util_drop[~keep_mask].sum()
            total_u = h_total_u * book_time + dropped_books_utils[idx_f - 1]
            if total_u > best_u:
                best_u = total_u
                best_drop = d
                if active_mask.sum() >= k_drop:
                    best_rating_cut = est_now[active_mask][sort_idx[k_drop - 1]]
                else:
                    best_rating_cut = 5  # dropping more books than have left
                # util of now dropped books plus util of replacing them with hourly opportunity
                best_drop_u = (
                    h_util_drop[~keep_mask].sum() * book_time + dropped_books_utils[idx_f - 1]
                )

        best_cum_drop[idx_f] = best_drop  # of books that remain
        best_cutoffs[idx_f] = best_rating_cut
        true_avg_utils[idx_f] = best_u
        dropped_books_utils[idx_f] = best_drop_u

        # Update active mask: drop the chosen set
        k_drop = int(np.floor(best_drop * active_mask.sum()))
        if k_drop > 0:
            # print(f"Dropping {k_drop} books at f={f:.2f} , d={best_drop:.2f} , best_u={best_u:.2f}")
            drop_global_idx = np.where(active_mask)[0][sort_idx[:k_drop]]
            active_mask[drop_global_idx] = False

        if False:  # idx_f == 0:  # Print values for first step
            print(f"\nDebug - First step utilities:")
            print(f"Continue: {u_continue_est[:5]}")
            print(f"Stop: {u_stop_est[:5]}")
            print(f"Best drop fraction: {best_drop}")
            print(f"Best cut: {best_rating_cut}")

    # print(best_cum_drop, best_cutoffs, true_avg_utils, dropped_books_utils, sep="\n")
    true_avg_utils /= len(true_ratings)
    assert true_avg_utils[-1] >= val_full.mean(), (
        "Optimal strategy is worse than just reading all books"
        f" {true_avg_utils[-1]} {val_full.mean()}"
    )
    cum_drop = np.prod(1 - best_cum_drop)
    n_keep = math.ceil(len(val_full) * cum_drop)
    if n_keep > 0:
        u_if_drop_best = np.sort(val_full)[-n_keep:].mean()
        assert true_avg_utils[-1] <= u_if_drop_best + 1e-06, (
            "Optimal strategy is better than reading optimal number of books initially"
            f" {true_avg_utils[-1]} {u_if_drop_best} {n_keep}"
        )
    assert np.allclose(
        true_avg_utils, np.maximum.accumulate(true_avg_utils)
    ), "getting worse over time"
    return {"cur_drop": best_cum_drop, "cutoffs": best_cutoffs, "true_avg_utils": true_avg_utils}


# -------------- Wrapper per category --------------
def simulate_category(df_cat: pd.DataFrame, rating_col: str) -> Dict[str, np.ndarray]:
    """ """
    N_FOR_BASELINE_U = 5  # Run multiple times to get true values for baseline utility

    true_ratings_original = df_cat[rating_col].values  # Original ratings for the category
    current_u = np.mean(utility_value(true_ratings_original)) / (
        READ_TIME_HOURS + SEARCH_COST_HOURS
    )

    # if rating_col == "Usefulness /5 to Me":
    #    QUIT_TABLE
    # TODO handle already dropped books

    for j in range(N_FOR_BASELINE_U):
        # Store individual simulation paths and their true utilities
        cur_drop_acc = np.zeros(len(F_GRID))
        cutoff_acc = np.zeros(len(F_GRID))

        all_drop_paths = []  # of books remaining, drop fraction at each step
        all_cutoffs = []
        all_true_utils = []

        for i in range(N_SIM):
            if True:  # j == 0 and i == 0:
                # on first run, use original ratings to match emperical utility from real number of books dropped
                # this prevents error correction?
                bootstrapped_ratings = true_ratings_original
            else:
                bootstrapped_ratings = np.random.choice(
                    true_ratings_original, size=len(true_ratings_original), replace=True
                )
            res = optimise_schedule_greedy(bootstrapped_ratings, hourly_opportunity=current_u)
            cur_drop_acc += res["cur_drop"]
            cutoff_acc += res["cutoffs"]
            all_drop_paths.append(res["cur_drop"])
            all_cutoffs.append(res["cutoffs"])
            all_true_utils.append(res["true_avg_utils"])
        avg_of_optimal_path = np.mean([i[-1] for i in all_true_utils])
        current_u = avg_of_optimal_path

        cur_drop_acc /= N_SIM
        cutoff_acc /= N_SIM
        print(current_u, avg_of_optimal_path, np.mean(cur_drop_acc), np.mean(cutoff_acc))
    return {
        "cur_drop": cur_drop_acc,
        "cutoffs": cutoff_acc,
        "cur_drop_path": np.array(all_drop_paths),
        "cutoffs_all": np.array(all_cutoffs),
        "true_avg_utils": np.array(all_true_utils),
    }


def plot_simulation_paths(
    drop_paths: np.ndarray,
    f_grid: np.ndarray,
    true_utils: np.ndarray,
    cutoffs: np.ndarray,
    title: str = "Simulation Paths",
):
    """
    Plot simulation paths with three subplots:
    1. True Utilities over time
    2. Cumulative Drop Fraction over time
    3. Rating Cutoffs over time

    All plots include colorbars for utility values.

    Args:
        drop_paths: Array of shape (n_simulations, n_points) containing all simulation paths
        f_grid: Array of x-axis values (fraction read)
        true_utils: Array of shape (n_simulations, n_points) containing true utilities
        cutoffs: Array of shape (n_simulations, n_points) containing rating cutoffs
        title: Plot title
    """
    # Calculate final values for coloring
    final_values = true_utils[:, -1]

    # Calculate remaining books at each timestep
    remaining_books = np.zeros_like(drop_paths)
    for i, path in enumerate(drop_paths):
        # Start with 1 (all books)
        remaining = 1.0
        for j, drop_frac in enumerate(path):
            remaining *= 1 - drop_frac  # Multiply by fraction kept
            remaining_books[i, j] = remaining

    # Normalize final utilities for line colors
    norm_final = (final_values - final_values.min()) / (final_values.max() - final_values.min())

    # Normalize instantaneous utilities for scatter colors using global min/max
    global_min = true_utils.min()
    global_max = true_utils.max()
    norm_instant = (true_utils - global_min) / (global_max - global_min)
    instant_colors = plt.cm.RdYlGn(norm_instant)

    # Create a figure with three subplots side by side
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(20, 8))

    # Plot true utilities on the first subplot
    for i, utils in enumerate(true_utils):
        # Plot each line segment with its own color based on instantaneous utility
        for j in range(len(f_grid) - 1):
            ax1.plot(
                f_grid[j : j + 2],
                utils[j : j + 2],
                color=plt.cm.RdYlGn(norm_instant[i, j]),
                alpha=0.3,
                linewidth=1,
            )
        # Add scatter points
        ax1.scatter(f_grid, utils, c=utils, alpha=0.1, s=10)

    # Plot mean and median paths
    ax1.plot(f_grid, true_utils.mean(axis=0), "k--", linewidth=2, label="Mean Utility", alpha=0.7)
    ax1.plot(f_grid, np.median(true_utils, axis=0), "k-", linewidth=2, label="Median Utility")

    ax1.set_title(f"{title} - True Utilities")
    ax1.set_xlabel("Fraction Read")
    ax1.set_ylabel("True Utility")
    ax1.grid(True, alpha=0.3)
    ax1.legend()

    # Add colorbar for instantaneous utility (scatter colors)
    sm1 = plt.cm.ScalarMappable(cmap=plt.cm.RdYlGn, norm=plt.Normalize(global_min, global_max))
    plt.colorbar(sm1, ax=ax1, label="Instantaneous Utility")

    # Calculate cumulative drops
    cumulative_drops = np.zeros_like(drop_paths)
    for i, path in enumerate(drop_paths):
        cumulative = 0
        for j, drop_frac in enumerate(path):
            cumulative += drop_frac
            cumulative_drops[i, j] = cumulative

    # Plot cumulative drops on the second subplot
    for i, (cum_drop, utils) in enumerate(zip(cumulative_drops, true_utils)):
        # Color line based on final utility
        line_color = plt.cm.RdYlGn(norm_final[i])
        ax2.plot(f_grid, cum_drop, color=line_color, alpha=0.3, linewidth=1)
        # Color scatter points based on instantaneous utility
        ax2.scatter(f_grid, cum_drop, c=instant_colors[i], alpha=0.1, s=10)

    ax2.plot(
        f_grid,
        cumulative_drops.mean(axis=0),
        "k--",
        linewidth=2,
        label="Mean Cumulative Drop",
        alpha=0.7,
    )
    ax2.plot(
        f_grid,
        np.median(cumulative_drops, axis=0),
        "k-",
        linewidth=2,
        label="Median Cumulative Drop",
    )

    ax2.set_title(f"{title} - Cumulative Drop")
    ax2.set_xlabel("Fraction Read")
    ax2.set_ylabel("Cumulative Drop Fraction")
    ax2.grid(True, alpha=0.3)
    ax2.legend()

    # Plot rating cutoffs on the third subplot
    for i, (cutoff, utils) in enumerate(zip(cutoffs, true_utils)):
        # Color line based on final utility
        line_color = plt.cm.RdYlGn(norm_final[i])
        ax3.plot(f_grid, cutoff, color=line_color, alpha=0.3, linewidth=1)
        # Color scatter points based on instantaneous utility
        ax3.scatter(f_grid, cutoff, c=norm_instant[i], cmap=plt.cm.RdYlGn, alpha=0.1, s=10)

    ax3.plot(
        f_grid,
        cutoffs.mean(axis=0),
        "k--",
        linewidth=2,
        label="Mean Cutoff",
        alpha=0.7,
    )
    ax3.plot(
        f_grid,
        np.median(cutoffs, axis=0),
        "k-",
        linewidth=2,
        label="Median Cutoff",
    )

    ax3.set_title(f"{title} - Rating Cutoffs")
    ax3.set_xlabel("Fraction Read")
    ax3.set_ylabel("Rating Cutoff")
    ax3.grid(True, alpha=0.3)
    ax3.legend()

    plt.tight_layout()
    plt.show()


# ---------------- Main ----------------
if __name__ == "__main__":
    # Then run the main simulation
    DATA_PATH = Path("data/Books Read and their effects - Play Export.csv")
    if not DATA_PATH.exists():
        print("CSV not found – replace DATA_PATH with your local file path.")
        exit()
    df = pd.read_csv(DATA_PATH)
    df["Bookshelf"] = df["Bookshelf"].str.strip().str.replace("/", ",").str.replace("&", "and")

    shelf_counts = df["Bookshelf"].value_counts()
    shelves = shelf_counts[shelf_counts > 10].index.tolist()
    out = {}

    # Find indices closest to 10%, 30%, and 50% of reading
    target_fractions = [0.1, 0.3, 0.5]
    milestone_indices = [np.abs(F_GRID - target).argmin() for target in target_fractions]
    rating_col = "Usefulness /5 to Me"
    for shelf in shelves:
        sub = df[df["Bookshelf"] == shelf]
        if sub.empty:
            continue
        out[shelf] = simulate_category(sub, rating_col)

        print(f"\n{'=' * 80}")
        print(f"Optimising schedule for: {shelf}")
        print(f"{'=' * 80}")

        # Get the optimal path (path with highest final utility)
        optimal_idx = np.argmax(out[shelf]["true_avg_utils"][:, -1])
        out[shelf]["optimal_drops"] = out[shelf]["cur_drop_path"][optimal_idx]
        # Get the median path
        median_idx = np.argsort(out[shelf]["true_avg_utils"][:, -1])[
            len(out[shelf]["true_avg_utils"]) // 2
        ]
        out[shelf]["median_drops"] = out[shelf]["cur_drop_path"][median_idx]
        out[shelf]["cumulative_drop"] = 1 - np.cumprod(1 - out[shelf]["optimal_drops"])

        print("\nOptimal Drop Schedule:")
        print(f"{'Fraction Read':>12} {'Instant Drop %':>15} {'Cumulative Drop %':>20}")
        print("-" * 50)
        for i, (f, drop, cum_drop) in enumerate(
            zip(F_GRID, out[shelf]["optimal_drops"], out[shelf]["cumulative_drop"])
        ):
            if i in milestone_indices or i % 10 == 0:  # Print every 10th point plus milestones
                print(f"{f:>12.2f} {drop*100:>15.2f} {cum_drop*100:>20.2f}")

        # Plot simulation paths for this shelf
        plot_simulation_paths(
            out[shelf]["cur_drop_path"],
            F_GRID,
            out[shelf]["true_avg_utils"],
            out[shelf]["cutoffs_all"],
            f"Simulation Paths - {shelf}",
        )

    for shelf in shelves:
        print(f"\n{shelf}")
        print(f"{'Fraction Read':>12} {'Cumulative Drop %':>20}")
        print("-" * 50)
        for target, idx in zip(target_fractions, milestone_indices):
            print(f"{F_GRID[idx]:>12.2f} {out[shelf]['cumulative_drop'][idx]*100:>20.2f}")
        print(f"Final cumulative drop: {out[shelf]['cumulative_drop'][-1]*100:.1f}%")
        best_u = out[shelf]["true_avg_utils"][optimal_idx, -1]
        best_r = inverse_utility_value(best_u)
        median_idx = np.argsort(out[shelf]["true_avg_utils"][:, -1])[
            len(out[shelf]["true_avg_utils"]) // 2
        ]
        median_u = out[shelf]["true_avg_utils"][median_idx, -1]
        median_r = inverse_utility_value(median_u)
        current_u = utility_value(df[df["Bookshelf"] == shelf][rating_col]).mean()
        current_r = inverse_utility_value(
            current_u
        )  # convex fn so must be calculated in the same way
        print(f"Final utility: {median_u:.2f} , current: {current_u:.2f}")
        print(f"Final Rating: {median_r:.2f} , current: {current_r:.2f}")
# %%
# Dynamic where check all options: D^F: 100M here at 8**9
F_GRID = np.concatenate(
    [
        np.arange(0.01, 0.4, 0.08),  # more precise in first half
        np.arange(0.4, 1, 0.15),  # less precise in second half
    ]
)
D_GRID = np.concatenate(
    [
        np.arange(0.00, 0.10, 0.02),  # dropping up to 30% in 1 step. Depends on F_GRID size
        np.arange(0.0, 0.21, 0.07),
    ]
)
