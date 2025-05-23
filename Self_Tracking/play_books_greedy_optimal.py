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
VALUE_BASE = 2  # utility = VALUE_BASE ** rating

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
FINISHED_TO_STARTED_RATIO = {
    k: (
        (v["finished"] + 0.5 * v["might finish"])
        / (v["finished"] + v["might finish"] + v["wont finish"])
    )
    for k, v in QUIT_TABLE.items()
}
STARTED_TO_FINISHED_RATIO = {k: 1 / v for k, v in FINISHED_TO_STARTED_RATIO.items()}

QUIT_USEFULNESS = 1.2
QUIT_ENJOYMENT = 1.4
# I quit some books since mid or bored, not because I hated them
QUIT_AT_FRACTION = 0.15  # but this would vary a lot?

# Static: O(D*F)
F_GRID = np.concatenate(
    [
        np.arange(0.01, 0.4, 0.02),  # more precise in first half
        np.arange(0.4, 1.01, 0.1),  # less precise in second half, f=1 as temp hack for graphs
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
    """Transform rating (1‑5) into cardinal utility."""
    return VALUE_BASE ** (r - 1) - 1


def inverse_utility_value(u):
    """Transform utility into rating."""
    return np.log(u + 1) / np.log(VALUE_BASE) + 1


def util_if_stop(u, f):
    """Utility if I stop reading now, given that search costs are fixed.

    But I haven't standardized on  what hourly util means, if it includes search costs or not"""
    hourly_u = u / (READ_TIME_HOURS + SEARCH_COST_HOURS)
    hourly_u_while_reading = u / READ_TIME_HOURS
    return (1 - f) * READ_TIME_HOURS * hourly_u


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
        hourly_opportunity: utility per hour of the marginal book (includes search costs and quit rate)
    returns
        best_spot_drop: fraction of currently active books to drop at each reading fraction f
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
        hourly_opportunity = np.percentile(val_full, 30) / book_time

    n_steps = len([f for f in F_GRID if f != 1])
    best_spot_drop = np.zeros(n_steps)
    best_cutoffs = np.zeros(n_steps)
    true_avg_utils = np.zeros(n_steps)
    dropped_books_utils = np.zeros(n_steps)
    # Track cumulative projected utility of dropped books and replacment with hourly opportunity

    active_mask = np.ones(n_books, dtype=bool)  # books still being read
    est_matrix = simulate_estimates(true_ratings)  # initial estimates

    for idx_f, f in enumerate(F_GRID):
        if f == 1:  # already finished
            continue
        est_now = est_matrix[:, idx_f]

        # Only evaluate books still active
        if active_mask.sum() == 0:
            print("WARNING: No active books left")
            while idx_f < len(F_GRID):
                best_spot_drop[idx_f:] = 1  # nothing left to drop, keep dropping all
                best_cutoffs[idx_f:] = 0
                true_avg_utils[idx_f:] = true_avg_utils[idx_f - 1]
                idx_f += 1
            break

        # Get estimated utilities for active books
        a_est_full = utility_value(est_now[active_mask])
        a_est_partial = utility_value(np.maximum(est_now[active_mask] - PARTIAL_RATING_PENALTY, 1))

        # Estimated Hourly utilities for remaining book
        current_t = SEARCH_COST_HOURS + f * READ_TIME_HOURS
        finish_t = book_time
        remaining_t = finish_t - current_t
        current_u = f * a_est_partial
        finish_u = a_est_full
        est_marginal_hourly_u = (finish_u - current_u) / remaining_t

        # Old version where hourly opportunity represented average util of replacement book
        # not marginal util
        # u_continue_est = (
        #     a_est_full / book_time
        # )  # todo: ignore fixed costs
        # cum_value = f * a_est_partial  # if stop
        # new_value = (
        #     (1 - f) * READ_TIME_HOURS * hourly_opportunity
        # )  # hourly opportunity includes search costs
        # u_stop_est = (cum_value + new_value) / book_time

        sort_idx = np.argsort(est_marginal_hourly_u)  # ascending: worst first
        # print(
        #     f"Sort idx: {sort_idx} , diff: {diff[sort_idx]} , u_continue_est:"
        #     f" {a_est_full[sort_idx]} , u_stop_est: {est_now[active_mask][sort_idx]}"
        # )
        # True marginal Hourly utilities of remaining books
        active_true_full = val_full[active_mask]
        active_true_part = val_partial[active_mask]

        current_u = f * active_true_part
        finish_u = active_true_full
        h_util_keep = (finish_u - current_u) / remaining_t

        # util_after_drop = remaining_t * hourly_opportunity
        # h_util_drop = (current_u + util_after_drop) / book_time
        h_util_drop = hourly_opportunity

        # values if no dropping
        best_drop = 0.0
        best_u = 0.0
        best_rating_cut = 0
        if idx_f == 0:
            best_u = val_full.sum()
            best_drop_u = 0
        else:
            best_u = true_avg_utils[idx_f - 1]
            best_drop_u = dropped_books_utils[idx_f - 1]

        for d in D_GRID:
            k_drop = int(np.floor(d * active_mask.sum()))  # number of books to drop

            # Get indices of books to keep and drop
            drop_set = sort_idx[:k_drop]
            keep_mask = np.ones(len(sort_idx), dtype=bool)
            keep_mask[drop_set] = False
            # include utility from previously dropped books

            h_total_u = h_util_keep[keep_mask].sum() + h_util_drop * k_drop
            total_u = h_total_u * remaining_t + dropped_books_utils[idx_f - 1]
            if total_u > best_u:
                best_u = total_u
                best_drop = d
                if active_mask.sum() >= k_drop:
                    best_rating_cut = est_now[active_mask][sort_idx[k_drop - 1]]
                else:
                    best_rating_cut = 5  # dropping more books than have left
                # util of now dropped books plus util of replacing them with hourly opportunity
                best_drop_u = current_u[~keep_mask].sum() + h_util_drop * remaining_t * k_drop

        best_spot_drop[idx_f] = best_drop  # of books that remain
        best_cutoffs[idx_f] = best_rating_cut
        true_avg_utils[idx_f] = best_u
        dropped_books_utils[idx_f] = best_drop_u

        # Update active mask: drop the chosen set
        k_drop = int(np.floor(best_drop * active_mask.sum()))
        if k_drop > 0:
            # print(f"Dropping {k_drop} books at f={f:.2f} , d={best_drop:.2f} , best_u={best_u:.2f}")
            drop_global_idx = np.where(active_mask)[0][sort_idx[:k_drop]]
            active_mask[drop_global_idx] = False

    # print(best_cum_drop, best_cutoffs, true_avg_utils, dropped_books_utils, sep="\n")
    true_avg_utils /= len(true_ratings)
    assert true_avg_utils[-1] >= val_full.mean(), (
        "Optimal strategy is worse than just reading all books"
        f" {true_avg_utils[-1]} {val_full.mean()}"
    )
    cum_drop = np.prod(1 - best_spot_drop)
    n_keep = math.ceil(len(val_full) * cum_drop)
    if n_keep > 0:
        u_if_drop_best = np.sort(val_full)[-n_keep:].mean()
        print(true_avg_utils)
        print(best_spot_drop)
        assert true_avg_utils[-1] <= u_if_drop_best + 1e-06, (
            "Optimal strategy is better than reading optimal number of books initially"
            f" {true_avg_utils[-1]} {u_if_drop_best} {n_keep}/{len(val_full)}"
        )
    assert np.allclose(
        true_avg_utils, np.maximum.accumulate(true_avg_utils)
    ), "getting worse over time"
    return {"cur_drop": best_spot_drop, "cutoffs": best_cutoffs, "true_avg_utils": true_avg_utils}


def avg_hours_reading(cur_drop):
    """Expected Hours of reading if I follow the current instant drop schedule for 1 book"""
    # cum_drop = 1 - np.prod(1 - cur_drop)
    if len(cur_drop.shape) == 1:
        have = np.concatenate([[1], np.cumprod(1 - cur_drop)])
    else:
        # 2d, each row is a different run. Take average number remaining at each step
        have = np.concatenate([[1], np.mean(np.cumprod(1 - cur_drop, axis=1), axis=0)])
    n_dropped_at_step = [i - j for i, j in zip(have[:-1], have[1:])]
    assert np.isclose(F_GRID[-1], 1), "last fraction read is 1"
    assert len(n_dropped_at_step) == len(
        F_GRID
    ), f"n_dropped_at_step {len(n_dropped_at_step)} , F_GRID {len(F_GRID)}"
    n_dropped_at_step[-1] = 1 - np.sum(
        n_dropped_at_step[:-1]
    )  # last entry is fraction of 1, finished whole thing
    o = (
        np.sum([d * f * READ_TIME_HOURS for d, f in zip(n_dropped_at_step, F_GRID)])
        + SEARCH_COST_HOURS
    )
    assert o <= READ_TIME_HOURS + SEARCH_COST_HOURS, o
    return o


def quit_u_h(df_cat: pd.DataFrame, rating_col: str) -> float:
    """Total Hours and utility of books I never finsihed reading originally for category(s)
    problem is if quit at F=0.01 then because base utility =1
    I get way more utility from quiting many books faster
    so the utility is hourly rate of a book with that utility if read whole thing * hours actually read
    """
    category_quit_counts = dict(df_cat["Bookshelf"].value_counts())
    expected_num_quit = sum(
        [(STARTED_TO_FINISHED_RATIO[k] - 1) * v for k, v in category_quit_counts.items()]
    )
    if rating_col == "Usefulness /5 to Me":
        quit_u_if_read = utility_value(QUIT_USEFULNESS)
    else:
        quit_u_if_read = utility_value(QUIT_ENJOYMENT)
    quit_hourly_rate = quit_u_if_read / (READ_TIME_HOURS + SEARCH_COST_HOURS)
    quit_h = expected_num_quit * (QUIT_AT_FRACTION * READ_TIME_HOURS + SEARCH_COST_HOURS)
    quit_u = quit_hourly_rate * quit_h
    return quit_u, quit_h


def current_hourly_u(df_cat: pd.DataFrame, rating_col: str, cur_drop=None) -> float:
    """Utility per hour of current reading habits, including books I never finished
    Can't include cur_drop since then utility depents on which specific books I drop
    """
    true_ratings_original = df_cat[rating_col].values  # Original ratings for the category, finished
    assert np.all(
        true_ratings_original >= 1
    ), f"some books are rated below 1 {true_ratings_original}"
    assert np.all(
        true_ratings_original <= 5
    ), f"some books are rated above 5 {true_ratings_original}"

    finished_u = np.sum(utility_value(true_ratings_original))
    if cur_drop is None:
        finished_h = len(true_ratings_original) * (READ_TIME_HOURS + SEARCH_COST_HOURS)
    else:
        assert (
            False
        ), "Can't include cur_drop since then utility depents on which specific books I drop"
        finished_h = len(true_ratings_original) * avg_hours_reading(cur_drop)

    quit_u, quit_h = quit_u_h(df_cat, rating_col)
    hourly_u = (finished_u + quit_u) / (finished_h + quit_h)
    return hourly_u


# -------------- Wrapper per category --------------
def simulate_category(df_cat: pd.DataFrame, rating_col: str) -> Dict[str, np.ndarray]:
    """df_cat: dataframe of books FINISHED in category
    rating_col: column name of rating, e.g. "Usefulness /5 to Me" or "Enjoyment /5 to Me"
    baseline hourly u is the current hourly u of the category, then 30th percentile of hourly u of the end of the simulation
    """
    N_FOR_BASELINE_U = (
        7  # Run multiple times to get true values for baseline utility, generally converges here
    )
    true_ratings_original = df_cat[rating_col].values  # Original ratings for the category, finished
    baseline_hourly_u = current_hourly_u(df_cat, rating_col)
    quit_u, quit_h = quit_u_h(df_cat, rating_col)

    for j in range(N_FOR_BASELINE_U):
        # Store individual simulation paths and their true utilities
        cur_drop_acc = np.zeros(len(F_GRID))
        cutoff_acc = np.zeros(len(F_GRID))

        all_drop_paths = []  # of books remaining, drop fraction at each step
        all_cutoffs = []
        all_true_utils = []

        for i in range(N_SIM):
            if False:  # j == 0 and i == 0:
                # on first run, use original ratings to match emperical utility from real number of books dropped
                # this prevents error correction?
                bootstrapped_ratings = true_ratings_original
            else:
                bootstrapped_ratings = np.random.choice(
                    true_ratings_original, size=len(true_ratings_original), replace=True
                )
            res = optimise_schedule_greedy(
                bootstrapped_ratings, hourly_opportunity=baseline_hourly_u
            )
            cur_drop_acc += res["cur_drop"]
            cutoff_acc += res["cutoffs"]
            all_drop_paths.append(res["cur_drop"])
            all_cutoffs.append(res["cutoffs"])
            all_true_utils.append(res["true_avg_utils"])
        cur_drop_acc /= N_SIM
        cutoff_acc /= N_SIM

        # really we'd set the baseline as the average expected utility of following current strategy
        # , adjusting for already quit books, but better to be conservative
        new_baseline_u = np.percentile([i[-1] for i in all_true_utils], 20)
        hourly_avg_u = (len(true_ratings_original) * new_baseline_u + quit_u) / (
            len(true_ratings_original) * (READ_TIME_HOURS + SEARCH_COST_HOURS) + quit_h
        )
        print("end", baseline_hourly_u, hourly_avg_u, np.mean(cur_drop_acc), np.mean(cutoff_acc))
        baseline_hourly_u = hourly_avg_u

        if baseline_hourly_u >= np.mean(utility_value(true_ratings_original)) / (
            READ_TIME_HOURS + SEARCH_COST_HOURS
        ):
            print("WARNING: hourly_opportunity is greater than the mean utility of completedbooks")

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


# ---------------- Result Printing Functions ----------------
def print_drop_schedule_table(
    shelf_name: str,
    f_grid: np.ndarray,
    avg_instant_drops: np.ndarray,
    avg_cumulative_drop: np.ndarray,
    milestone_indices: List[int],
    # cumulative_drop: np.ndarray, # No longer needed as input
):
    """Prints the average drop schedule for a given shelf, where each step was greddily optimised"""
    print(f"\nAverage Optimal Drop Schedule for: {shelf_name}")  # Title updated
    print(
        f"{'Fraction Read':>12} {'Avg Instant Drop %':>20} {'Avg Cumulative Drop %':>25}"
    )  # Header updated

    print("-" * 60)  # Adjusted width
    for i, (f_val, avg_drop_val, avg_cum_drop_val) in enumerate(
        zip(f_grid, avg_instant_drops, avg_cumulative_drop)
    ):
        if i in milestone_indices or i % 10 == 0:  # Print every 10th point plus milestones
            print(f"{f_val:>12.2f} {avg_drop_val*100:>20.2f} {avg_cum_drop_val*100:>25.2f}")


def print_all_shelves_summary(
    shelves: List[str],
    out_results: Dict,
    f_grid: np.ndarray,
    milestone_indices: List[int],
    target_fractions: List[float],
    df_all_books: pd.DataFrame,
    rating_col_name: str,
):
    """Prints a summary of results across all shelves."""
    print("\n\n" + "=" * 30 + " FINAL SUMMARY " + "=" * 30)
    for shelf in shelves:
        if shelf not in out_results:
            continue
        shelf_data = out_results[shelf]
        print(f"\nSummary for: {shelf}")
        print(f"{'Fraction Read':>12} {'Cumulative Drop %':>20}")
        print("-" * 50)
        for target, idx in zip(target_fractions, milestone_indices):
            print(f"{f_grid[idx]:>12.2f} {shelf_data['cumulative_drop'][idx]*100:>20.2f}")
        print(f"Final cumulative drop: {shelf_data['cumulative_drop'][-1]*100:.1f}%")

        median_idx = shelf_data.get(
            "median_idx",
            np.argsort(shelf_data["true_avg_utils"][:, -1])[len(shelf_data["true_avg_utils"]) // 2],
        )

        median_u = shelf_data["true_avg_utils"][median_idx, -1]
        median_r = inverse_utility_value(median_u)
        mean_u = shelf_data["true_avg_utils"][:, -1].mean()
        mean_r = inverse_utility_value(mean_u)

        current_shelf_df = df_all_books[df_all_books["Bookshelf"] == shelf]
        current_u = utility_value(current_shelf_df[rating_col_name]).mean()
        current_r = inverse_utility_value(current_u)

        print(f"Median Final Utility (simulated): {median_u:.2f} (Rating: {median_r:.2f})")
        print(f"Mean Final Utility (simulated): {mean_u:.2f} (Rating: {mean_r:.2f})")
        print(f"Current Avg Utility (empirical): {current_u:.2f} (Rating: {current_r:.2f})")


# ---------------- Main ----------------
if __name__ == "__main__":
    rating_col = "Usefulness /5 to Me"
    DATA_PATH = Path("data/Books Read and their effects - Play Export.csv")
    if not DATA_PATH.exists():
        print("CSV not found – replace DATA_PATH with your local file path.")
    df = pd.read_csv(DATA_PATH)
    df["Bookshelf"] = df["Bookshelf"].str.strip().str.replace("/", ",").str.replace("&", "and")

    shelf_counts = df["Bookshelf"].value_counts()
    shelves = shelf_counts[shelf_counts > 10].index.tolist()
    out = {}

    # Find indices closest to 10%, 30%, and 50% of reading
    target_fractions = [0.1, 0.3, 0.5]
    milestone_indices = [np.abs(F_GRID - target).argmin() for target in target_fractions]

    # for shelf in shelves:
    for shelf in ["Computer Science"]:
        sub = df[df["Bookshelf"] == shelf]
        if sub.empty:
            continue
        out[shelf] = simulate_category(sub, rating_col)

        # Get the optimal path (path with highest final utility)
        shelf_results = out[shelf]
        # Calculate average cummulative remaining at each fraction read
        shelf_results["avg_cumulative_drop"] = 1 - np.mean(
            np.cumprod(1 - shelf_results["cur_drop_path"], axis=1), axis=0
        )
        # dont care about the specific path that got a specific utility, but what's best drop path

        print_drop_schedule_table(
            shelf_name=shelf,
            f_grid=F_GRID,
            avg_instant_drops=shelf_results["cur_drop_path"].mean(
                axis=0
            ),  # not correct but close enough. I'll be eyeballing drop path anyway
            avg_cumulative_drop=shelf_results["avg_cumulative_drop"],
            milestone_indices=milestone_indices,
        )

        # Plot simulation paths for this shelf
        plot_simulation_paths(
            shelf_results["cur_drop_path"],
            F_GRID,
            shelf_results["true_avg_utils"],
            shelf_results["cutoffs_all"],
            f"Simulation Paths - {shelf}",
        )
    print_all_shelves_summary(
        shelves, out, F_GRID, milestone_indices, target_fractions, df, rating_col
    )


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
