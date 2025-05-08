# %%

import pandas as pd
import numpy as np
import os
from pathlib import Path
import numpy.random as rnd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

# Parameters
READ_TIME_HOURS = 4.0  # full reading time, in units of lost utility
SEARCH_COST_HOURS = 0.25  # discovery cost, in units of lost utility
PARTIAL_RATING_PENALTY = 0.5  # rating loss when abandoning, assumed utility lose linear
VALUE_BASE = 2  # utility = VALUE_BASE ** rating
F_GRID = np.arange(0.01, 1.01, 0.01)

DATA_PATH = Path("data/Books Read and their effects - Play Export.csv")

# --------------------------
# 1. Load core CSV
# --------------------------
df = pd.read_csv(DATA_PATH)
df = df.drop(columns=[c for c in df.columns if "Unnamed:" in c])

# Normalise bookshelf names
df["Bookshelf"] = df["Bookshelf"].str.strip().str.replace("/", ",").str.replace("&", "and")

# --------------------------
# 2. Append synthetic rows
# --------------------------
add_table = {
    "Business, management": {"finished": 44, "might finish": 5, "wont finish": 6},
    "Computer Science": {"finished": 14, "might finish": 6, "wont finish": 10},
    "fiction": {"finished": 51, "might finish": 0, "wont finish": 6},
    "General Reading": {"finished": 40, "might finish": 5, "wont finish": 7},
    "Literature": {"finished": 50, "might finish": 3, "wont finish": 6},
    "Machine Learning": {"finished": 5, "might finish": 3, "wont finish": 2},
    "Math": {"finished": 3, "might finish": 6, "wont finish": 2},
}

synthetic_rows = []
for shelf, counts in add_table.items():
    for status, n in counts.items():
        if n == 0:
            continue
        if status == "finished":
            # finished books already in CSV; skip adding duplicates
            continue
        enjoyment = 2.0 if status == "might finish" else 1.4
        usefulness = 1.5 if status == "might finish" else 1.2
        for i in range(n):
            synthetic_rows.append(
                {
                    "title": f"synthetic_{status}_{i}",
                    "author": "synthetic",
                    "Bookshelf": shelf,
                    "earliest_modified": pd.NaT,
                    "latest_modified": pd.NaT,
                    "filename": "synthetic",
                    "Enjoyment (/5)": enjoyment,
                    "Usefulness /5 to Me": usefulness,
                    "Long Term Effects": status,
                }
            )
synthetic_df = pd.DataFrame(synthetic_rows)
df = pd.concat([df, synthetic_df], ignore_index=True)


# --------------------------
# 3. Helper functions
# --------------------------
def utility_value(rating):
    return VALUE_BASE**rating


def error_width(f):
    e = 0.4
    return 0.2 + 5**e - (5 * f) ** e
    # plt.plot(F_GRID, [error_width(f) for f in F_GRID])


def expected_decisions_sim(sub_df, rating_col):
    """
    Monte‑Carlo expected utility & drop fraction given estimation noise.
    Assumes hours to read are fixed.

    returns
        [(optimal % of time to drop, average hourly utility) for f in f_grid]
    """
    true_ratings = sub_df[rating_col].values
    n_books = len(true_ratings)

    val_full_true = utility_value(true_ratings)
    val_partial_true = utility_value(np.maximum(true_ratings - PARTIAL_RATING_PENALTY, 0))
    # hourly utility of replacement book
    hourly_opportunity_cost = np.percentile(val_full_true, 30) / (
        READ_TIME_HOURS + SEARCH_COST_HOURS
    )

    drop_fractions, net_utils = [], []

    N_SAMPLES = 100
    for f in F_GRID:

        sigma = error_width(f)
        # replicate each true rating N_SAMPLES times
        r_true_rep = np.repeat(true_ratings, N_SAMPLES)
        noise = rnd.uniform(-sigma, sigma, size=r_true_rep.shape[0])
        r_est = np.clip(r_true_rep + noise, 0.0, 5.0)

        u_full_est = utility_value(r_est) / (READ_TIME_HOURS + SEARCH_COST_HOURS)
        # utility if stopped book at fraction f and switched to another
        u_part_est = (
            # u got so far
            f
            * utility_value(np.maximum(r_est - PARTIAL_RATING_PENALTY, 0))
            / (f * READ_TIME_HOURS + SEARCH_COST_HOURS)
            # u got from switching, assume will finish this next book
            + (1 - f) * hourly_opportunity_cost
        )

        cont_mask = (u_full_est - u_part_est) > 0
        cont_mask = cont_mask.reshape(n_books, N_SAMPLES)

        # True utilities (no noise) replicated
        u_full_true = np.repeat(val_full_true, N_SAMPLES).reshape(n_books, N_SAMPLES)
        u_part_true = np.repeat(val_partial_true, N_SAMPLES).reshape(n_books, N_SAMPLES)

        # where I realize it's worse than default I lower utility from starting but utility of another book
        # I choose based on estimated value of hour
        util = np.where(
            cont_mask,
            u_full_true / (READ_TIME_HOURS + SEARCH_COST_HOURS),
            (f * utility_value(np.maximum(u_part_true - PARTIAL_RATING_PENALTY, 0)))
            / (f * READ_TIME_HOURS + SEARCH_COST_HOURS)
            + (1 - f) * hourly_opportunity_cost,
        )

        drop_fractions.append((~cont_mask).mean())
        net_utils.append(util.mean())

    return np.array(drop_fractions), np.array(net_utils)


def expected_decisions_sim(sub_df, rating_col):
    """
    Calculates the optimal drop fraction and expected utility based on TRUE utilities.
    Decision rule: Stop if the utility per hour of continuing is less than the
                   utility per hour of switching to an average book.

    Returns:
        tuple: (np.array of drop fractions for each f, np.array of net utilities for each f)
    """
    true_ratings = sub_df[rating_col].values
    n_books = len(true_ratings)
    if n_books == 0:
        print(f"Warning: No books found for category in column '{rating_col}'. Returning zeros.")
        return np.zeros_like(F_GRID), np.zeros_like(F_GRID)

    # Calculate true utilities
    val_full_true = utility_value(true_ratings)
    val_partial_true = utility_value(np.maximum(true_ratings - PARTIAL_RATING_PENALTY, 0))

    # Calculate the opportunity cost: average utility per hour of a replacement book
    # Using 30th percentile as per original code
    avg_replacement_utility = np.percentile(val_full_true, 30)
    # Utility per hour, assuming full read time + search cost for the replacement
    hourly_opportunity_cost = avg_replacement_utility / (READ_TIME_HOURS + SEARCH_COST_HOURS)

    drop_fractions, net_utils = [], []
    epsilon = 1e-9  # To avoid division by zero when f=1

    for f in F_GRID:
        # --- Optimal Decision based on True Values ---

        # 1. Calculate Utility Rate if Continuing:
        # Utility gain from finishing = val_full_true - val_partial_true
        # Time required to finish = (1 - f) * READ_TIME_HOURS
        utility_gain_if_continue = val_full_true - val_partial_true
        time_to_finish = (1 - f) * READ_TIME_HOURS
        # Rate of utility gain if continuing (utility per remaining hour)
        continue_utility_rate = utility_gain_if_continue / (time_to_finish + epsilon)

        # 2. Decision Mask: Continue if the rate is better than the opportunity cost
        # This mask is based on perfect information (true utilities)
        cont_mask_true = continue_utility_rate > hourly_opportunity_cost

        # 3. Calculate Drop Fraction: Proportion of books where stopping is optimal
        drop_fraction = (~cont_mask_true).mean()
        drop_fractions.append(drop_fraction)

        # --- Calculate Average Net Utility Achieved ---
        # Calculate the utility per hour achieved for each book, assuming the optimal decision is made

        # Utility per hour if the book is finished
        util_per_hour_if_continue = val_full_true / (READ_TIME_HOURS + SEARCH_COST_HOURS)

        # Utility per hour if stopped at f and switched
        # Utility gained = Utility from partial read + Utility from replacement book in remaining time
        # Total time = Time for partial read + search cost + Time for replacement read
        # Note: The total time base for comparison should be consistent.
        # Let's use the total time horizon (READ_TIME_HOURS + SEARCH_COST_HOURS)
        utility_if_stop = (
            val_partial_true * f / (f * READ_TIME_HOURS + SEARCH_COST_HOURS)
            + time_to_finish * hourly_opportunity_cost
        )
        util_per_hour_if_stop = utility_if_stop / (READ_TIME_HOURS + SEARCH_COST_HOURS)

        # Average utility across all books, applying the optimal decision mask
        util_achieved = np.where(cont_mask_true, util_per_hour_if_continue, util_per_hour_if_stop)
        net_utils.append(util_achieved.mean())

    return np.array(drop_fractions), np.array(net_utils)


# --------------------------
# 4. Compute per-category results
# --------------------------
results = {}
for shelf, sub in df.groupby("Bookshelf"):
    drop_U, util_U = expected_decisions_sim(sub, "Usefulness /5 to Me")
    drop_E, util_E = expected_decisions_sim(sub, "Enjoyment (/5)")
    results[shelf] = {
        "drop_usefulness": drop_U,
        "util_usefulness": util_U,
        "drop_enjoyment": drop_E,
        "util_enjoyment": util_E,
    }
    break

# --------------------------
# 5. Plot per-category histogram and drop fractions
# --------------------------
n_categories = len(results)

# Create a figure with n_categories rows
fig = plt.figure(figsize=(12, n_categories * 3.5))  # Increased height slightly for spacing

# Define the grid specification: More vertical space (hspace)
# We will create this grid structure for each category row
outer_grid = gridspec.GridSpec(n_categories, 1, wspace=0.2, hspace=0.6)  # Increased hspace

for i, (shelf, data) in enumerate(results.items()):
    # Create nested GridSpec with updated width ratio
    inner_grid = gridspec.GridSpecFromSubplotSpec(
        1, 2, subplot_spec=outer_grid[i], width_ratios=[1, 2.5], wspace=0.3
    )

    # --- Left Subplot: Histogram ---
    ax_hist = plt.Subplot(fig, inner_grid[0])
    # Get the ratings for this shelf
    shelf_ratings = df.loc[df["Bookshelf"] == shelf, "Usefulness /5 to Me"].dropna()
    ax_hist.hist(shelf_ratings, bins=np.arange(0, 5.5, 0.5), edgecolor="black", alpha=0.7)
    ax_hist.set_title(f"{shelf}\nRatings Histogram")
    # Plot histogram and get counts/bins
    counts, bins, patches = ax_hist.hist(
        shelf_ratings, bins=np.arange(0, 5.5, 0.5), edgecolor="black", alpha=0.7
    )
    ax_hist.set_xlabel("Usefulness Rating")
    ax_hist.set_ylabel("Count")
    ax_hist.set_xlim(df["Usefulness /5 to Me"].min(), df["Usefulness /5 to Me"].max())
    # Set y-limit based on max count
    max_count = counts.max()
    ax_hist.set_ylim(
        0, max_count * 1.05 if max_count > 0 else 1
    )  # Add 5% margin, handle empty case
    fig.add_subplot(ax_hist)

    # --- Right Subplot: Drop Fraction ---
    ax_drop = plt.Subplot(fig, inner_grid[1])
    ax_drop.plot(F_GRID, data["drop_usefulness"])
    # Optional: fill area
    # ax_drop.fill_between(F_GRID, 0, data["drop_usefulness"], alpha=0.3)
    ax_drop.set_title(f"{shelf}\nDrop Fraction vs. Read Fraction")
    ax_drop.set_xlabel("Fraction read f")
    ax_drop.set_ylabel("Drop fraction (Usefulness)")
    ax_drop.set_ylim(0, 1)  # Ensure y-axis is consistent for drop fraction
    ax_drop.grid(True, linestyle="--", alpha=0.6)
    fig.add_subplot(ax_drop)

# Add an overall title for the figure
fig.suptitle("Category Analysis: Ratings Distribution and Optimal Dropping Strategy", fontsize=16)
# Adjust layout rect: top value closer to 1 reduces space below suptitle
# fig.tight_layout(rect=[0, 0.03, 1, 0.95]) # Adjust top (0.95) to reduce space below suptitle
fig.tight_layout()  # Apply basic tight layout first
fig.subplots_adjust(top=0.94)  # Adjust top margin (decrease to reduce space below suptitle)

plt.show()

print("Finished computation.")

# %%
# Explciitly compute the optimal stopping fraction for each category at each reading fraction
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List

# ---------------- Parameters ----------------
READ_TIME_HOURS = 4.0
SEARCH_COST_HOURS = 0.25
PARTIAL_RATING_PENALTY = 0.25
VALUE_BASE = 1.75

F_GRID = np.arange(0.01, 1.01, 0.10)  # reading fractions
D_GRID = np.arange(0.00, 0.31, 0.01)  # cumulative drops (share of initial)

N_SIM = 200  # MC replications


# ---------------- Utility helpers ----------------
def utility_value(r):
    """Transform rating (0‑5) into cardinal utility."""
    return VALUE_BASE**r


def error_sigma(f):
    """Error half‑width at progress f of uniform noise"""
    return max(2.5 - 4 * f, 0.5)  # ±2.5 at start → ±0.5 by halfway


# Not sure which to go with
def utility_value2(r):
    return r**VALUE_BASE


def error_sigma2(f):
    return 1.5 * (1 - f**0.5)


def simulate_estimates(true_ratings: np.ndarray) -> np.ndarray:
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
    sigmas = np.array([error_sigma(f) for f in F_GRID])
    rho = 0.9  # autocorrelation coefficient

    # Initial error using uniform noise to match original implementation
    e_prev = np.random.uniform(-sigmas[0], sigmas[0], size=n_books)
    est[:, 0] = np.clip(true_ratings + e_prev, 0, 5)

    # Subsequent errors with autocorrelation
    for j in range(1, len(F_GRID)):
        # Scale previous error by rho
        scale_factor = sigmas[j] / sigmas[j-1] if sigmas[j-1] > 0 else 0
        e_autocorr = rho * e_prev * scale_factor

        # Add innovation term scaled by sqrt(1-rho^2) to maintain variance
        e_innovation = np.sqrt(1 - rho**2) * sigmas[j] * np.random.normal(0, 1, size=n_books)
        e_current = e_autocorr + e_innovation

        est[:, j] = np.clip(true_ratings + e_current, 0, 5)
        e_prev = e_current

    return est


# ---------------- Core optimiser ----------------
def optimise_schedule(true_ratings: np.ndarray) -> Dict[str, np.ndarray]:
    n_books = len(true_ratings)
    val_full = utility_value(true_ratings)
    val_partial = utility_value(np.maximum(true_ratings - PARTIAL_RATING_PENALTY, 0))
    hourly_opportunity = np.percentile(val_full, 30) / (READ_TIME_HOURS + SEARCH_COST_HOURS)

    best_cum_drop = np.zeros(len(F_GRID))
    best_cutoffs = np.zeros(len(F_GRID))

    active_mask = np.ones(n_books, dtype=bool)  # books still being read

    est_matrix = simulate_estimates(true_ratings)  # initial estimates

    for idx_f, f in enumerate(F_GRID):
        est_now = est_matrix[:, idx_f]

        # Only evaluate books still active
        active_idx = np.where(active_mask)[0]
        if active_idx.size == 0:
            best_cum_drop[idx_f:] = best_cum_drop[idx_f - 1]  # nothing left
            best_cutoffs[idx_f:] = best_cutoffs[idx_f - 1]
            break

        a_true_full = val_full[active_idx]
        a_true_part = val_partial[active_idx]
        a_est_full = utility_value(est_now[active_idx])
        a_est_partial = utility_value(np.maximum(est_now[active_idx] - PARTIAL_RATING_PENALTY, 0))

        # Hourly utilities
        u_continue_est = a_est_full / (READ_TIME_HOURS + SEARCH_COST_HOURS)
        u_stop_est = (f * a_est_partial) / (f * READ_TIME_HOURS + SEARCH_COST_HOURS) + (
            1 - f
        ) * hourly_opportunity

        # Difference (negative ⇒ look worse than switching)
        diff = u_continue_est - u_stop_est
        sort_idx = np.argsort(diff)  # ascending: worst first

        best_u = -np.inf
        best_d = 0.0
        best_cut = diff[sort_idx[int(0)]]  # initialise

        for d in D_GRID:
            k_drop = int(np.floor(d * n_books))
            drop_set = sort_idx[:k_drop]

            keep_mask_local = np.ones(active_idx.shape[0], dtype=bool)
            keep_mask_local[drop_set] = False

            # Compute real utility realised
            util_keep = val_full[active_idx][keep_mask_local] / (
                READ_TIME_HOURS + SEARCH_COST_HOURS
            )
            util_drop = (f * val_partial[active_idx][~keep_mask_local]) / (
                f * READ_TIME_HOURS + SEARCH_COST_HOURS
            ) + (1 - f) * hourly_opportunity

            total_u = util_keep.sum() + util_drop.sum()
            if total_u > best_u:
                best_u = total_u
                best_d = d
                if k_drop > 0:
                    best_cut = diff[sort_idx[k_drop - 1]]
                else:
                    best_cut = diff[sort_idx[0]]

        # record
        best_cum_drop[idx_f] = best_d
        best_cutoffs[idx_f] = best_cut

        # update active mask: drop the chosen set
        k_drop = int(np.floor(best_d * n_books))
        drop_global_idx = active_idx[sort_idx[:k_drop]]
        active_mask[drop_global_idx] = False

    return {"cum_drop": best_cum_drop, "cutoffs": best_cutoffs}


# -------------- Wrapper per category --------------
def simulate_category(df_cat: pd.DataFrame, rating_col: str) -> Dict[str, np.ndarray]:
    true_ratings = df_cat[rating_col].values
    cum_drop_acc = np.zeros(len(F_GRID))
    cutoff_acc = np.zeros(len(F_GRID))

    for _ in range(N_SIM):
        res = optimise_schedule(true_ratings)
        cum_drop_acc += res["cum_drop"]
        cutoff_acc += res["cutoffs"]

    cum_drop_acc /= N_SIM
    cutoff_acc /= N_SIM
    return {"cum_drop": cum_drop_acc, "cutoffs": cutoff_acc}


# ---------------- Main ----------------
def main():
    DATA_PATH = Path("data/Books Read and their effects - Play Export.csv")
    if not DATA_PATH.exists():
        print("CSV not found – replace DATA_PATH with your local file path.")
        return
    df = pd.read_csv(DATA_PATH)
    df["Bookshelf"] = df["Bookshelf"].str.strip().str.replace("/", ",").str.replace("&", "and")

    shelves = df["Bookshelf"].unique()
    out = {}

    for shelf in shelves:
        sub = df[df["Bookshelf"] == shelf]
        if sub.empty:
            continue
        print(f"Optimising schedule for: {shelf}")
        out[shelf] = simulate_category(sub, "Usefulness /5 to Me")
        print(f"{shelf} cumulative drop (avg over sims) at F_GRID:")
        for f, d in zip(F_GRID, out[shelf]["cum_drop"]):
            print(f"  f={f:.2f}  drop={d:.3f}")

    # Example: print Business cum_drop
    if "Business, management" in out:


if __name__ == "__main__":
    main()
