from __future__ import annotations

import pandas as pd

from ai_books_tracking.scripts.temp import check_noise_band_questions as module


def synthetic_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "title": [f"Book {index}" for index in range(6)],
            "category": ["General Reading"] * 6,
            "source": ["Holdout 2026"] * 6,
            "true_enjoy": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            "true_useful": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            "enjoy_1st": [0.8, 2.2, 3.2, 3.8, 5.3, 5.7],
            "enjoy_2nd": [1.2, 1.8, 2.8, 4.2, 4.7, 6.3],
            "useful_1st": [0.8, 2.2, 3.2, 3.8, 5.3, 5.7],
            "useful_2nd": [1.2, 1.8, 2.8, 4.2, 4.7, 6.3],
            "ridge_full_enjoy": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            "ridge_full_useful": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            "rf_enjoy": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            "rf_useful": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            "external_sum": [2.0, 4.0, 6.0, 8.0, 10.0, 12.0],
        }
    )


def test_current_monte_carlo_degrades_oracle_more_than_fixed_model() -> None:
    frame = synthetic_frame()
    observed = module.calculate_drop_curve(frame, "oracle", "enjoy", 1.0)
    oracle_band = module.monte_carlo_current_band(
        frame,
        "oracle",
        "enjoy",
        1.0,
        enjoy_sd=1.0,
        useful_sd=1.0,
        n_iterations=4000,
        seed=0,
    )
    rf_band = module.monte_carlo_current_band(
        frame,
        "rf",
        "enjoy",
        1.0,
        enjoy_sd=1.0,
        useful_sd=1.0,
        n_iterations=4000,
        seed=1,
    )

    observed_50 = float(observed.loc[observed["drop_percent"].eq(50), "gain"].iloc[0])
    oracle_mean_50 = float(
        oracle_band.loc[oracle_band["drop_percent"].eq(50), "mean"].iloc[0]
    )
    rf_mean_50 = float(rf_band.loc[rf_band["drop_percent"].eq(50), "mean"].iloc[0])

    assert abs(oracle_mean_50 - observed_50) > 0.05
    assert abs(rf_mean_50 - observed_50) < 0.06


def test_fixed_ranking_band_recenters_oracle_curve() -> None:
    frame = synthetic_frame()
    observed = module.calculate_drop_curve(frame, "oracle", "enjoy", 1.0)
    fixed_band = module.monte_carlo_fixed_ranking_band(
        frame,
        "oracle",
        "enjoy",
        1.0,
        target_sd=1.0,
        n_iterations=4000,
        seed=2,
    )

    observed_50 = float(observed.loc[observed["drop_percent"].eq(50), "gain"].iloc[0])
    fixed_mean_50 = float(
        fixed_band.loc[fixed_band["drop_percent"].eq(50), "mean"].iloc[0]
    )

    assert abs(fixed_mean_50 - observed_50) < 0.06


def test_empirical_rerating_band_tracks_observed_curve_and_has_width() -> None:
    frame = synthetic_frame()
    observed = module.calculate_drop_curve(frame, "rf", "enjoy", 1.0)
    empirical = module.empirical_rerating_band(
        frame,
        "rf",
        "enjoy",
        1.0,
        n_iterations=4000,
        seed=3,
    )

    observed_50 = float(observed.loc[observed["drop_percent"].eq(50), "gain"].iloc[0])
    empirical_row = empirical.loc[empirical["drop_percent"].eq(50)].iloc[0]

    assert abs(float(empirical_row["mean"]) - observed_50) < 0.06
    assert float(empirical_row["p90"] - empirical_row["p10"]) > 0.0


def test_oracle_penalty_shrinks_when_noise_scale_is_reduced() -> None:
    frame = synthetic_frame()
    full_noise = module.monte_carlo_current_band(
        frame,
        "oracle",
        "enjoy",
        1.0,
        enjoy_sd=1.0,
        useful_sd=1.0,
        n_iterations=4000,
        seed=4,
    )
    half_noise = module.monte_carlo_current_band(
        frame,
        "oracle",
        "enjoy",
        1.0,
        enjoy_sd=0.5,
        useful_sd=0.5,
        n_iterations=4000,
        seed=5,
    )
    observed = module.calculate_drop_curve(frame, "oracle", "enjoy", 1.0)

    observed_50 = float(observed.loc[observed["drop_percent"].eq(50), "gain"].iloc[0])
    full_penalty = observed_50 - float(
        full_noise.loc[full_noise["drop_percent"].eq(50), "mean"].iloc[0]
    )
    half_penalty = observed_50 - float(
        half_noise.loc[half_noise["drop_percent"].eq(50), "mean"].iloc[0]
    )

    assert abs(half_penalty) < abs(full_penalty)
