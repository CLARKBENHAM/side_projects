# %%
# first cell computes the stopping fraction given the assumed utility function, estimates, etc
import pandas as pd
import numpy as np
import os
import math
from pathlib import Path
import numpy.random as rnd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from typing import Dict, List

# Parameters
READ_TIME_HOURS = 4.0  # full reading time
SEARCH_COST_HOURS = 0.25  # discovery cost of getting a new book
PARTIAL_RATING_PENALTY = 0.1  # rating loss when abandoning, assumed utility loss linear
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
add_table = {  # might 28, wont 39 is slightly off from other time I did counts but I'm exactly sure on number anyway
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
    """
    returns error width at fraction f
    """
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
