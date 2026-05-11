from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
import sys

import numpy as np


SCRIPT_PATH = (
    Path(__file__).resolve().parent.parent
    / "ai_books_tracking"
    / "scripts"
    / "temp"
    / "generalized_screening_simulation.py"
)


def load_module():
    spec = spec_from_file_location("generalized_screening_simulation", SCRIPT_PATH)
    module = module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_truncated_normal_gain_matches_large_normal_simulation() -> None:
    module = load_module()
    rng = np.random.default_rng(0)
    rho = 0.30
    keep_share = 0.50

    latent = rng.normal(size=200_000)
    score = rho * latent + np.sqrt(1.0 - rho**2) * rng.normal(size=200_000)
    observed = module.standardize(latent)

    actual_gain = module.selection_gain(observed, score, keep_share)
    predicted_gain = module.theoretical_gain(
        float(np.std(observed, ddof=1)),
        module.safe_corr(observed, score),
        keep_share,
    )

    assert abs(actual_gain - predicted_gain) < 0.02


def test_integer_likert_formula_remains_close() -> None:
    module = load_module()
    rng = np.random.default_rng(1)
    empirical = {
        "empirical_book_enjoyment": np.array([1.0, 2.0, 3.0, 4.0, 5.0]),
        "empirical_book_usefulness": np.array([1.0, 2.0, 3.0, 4.0, 5.0]),
        "empirical_movie_rating": np.array([4.0, 6.0, 8.0, 10.0]),
    }

    latent = rng.normal(size=200_000)
    score = 0.30 * latent + np.sqrt(1.0 - 0.30**2) * rng.normal(size=200_000)
    observed = module.observed_outcome_from_latent(
        latent,
        "integer_1_to_5",
        empirical,
    )

    actual_gain = module.selection_gain(observed, score, 0.50)
    predicted_gain = module.theoretical_gain(
        float(np.std(observed, ddof=1)),
        module.safe_corr(observed, score),
        0.50,
    )

    assert abs(actual_gain - predicted_gain) < 0.05


def test_prefiltering_reduces_residual_external_signal() -> None:
    module = load_module()
    rng = np.random.default_rng(2)
    empirical = {
        "empirical_book_enjoyment": np.array([1.0, 2.0, 3.0, 4.0, 5.0]),
        "empirical_book_usefulness": np.array([1.0, 2.0, 3.0, 4.0, 5.0]),
        "empirical_movie_rating": np.array([4.0, 6.0, 8.0, 10.0]),
    }

    latent = rng.normal(size=200_000)
    observed = module.observed_outcome_from_latent(latent, "integer_1_to_5", empirical)
    external = 0.30 * latent + np.sqrt(1.0 - 0.30**2) * rng.normal(size=200_000)

    weak_internal = 0.20 * latent + np.sqrt(1.0 - 0.20**2) * rng.normal(size=200_000)
    strong_internal = 0.60 * latent + np.sqrt(1.0 - 0.60**2) * rng.normal(size=200_000)

    weak_idx = np.argsort(weak_internal)[-60_000:]
    strong_idx = np.argsort(strong_internal)[-60_000:]

    weak_corr = module.safe_corr(observed[weak_idx], external[weak_idx])
    strong_corr = module.safe_corr(observed[strong_idx], external[strong_idx])
    weak_gain = module.selection_gain(observed[weak_idx], external[weak_idx], 0.50)
    strong_gain = module.selection_gain(
        observed[strong_idx], external[strong_idx], 0.50
    )

    assert strong_corr < weak_corr
    assert strong_gain < weak_gain


def test_multisource_theory_and_simulation_track() -> None:
    module = load_module()
    rng = np.random.default_rng(3)
    rho = 0.30
    pairwise = module.pairwise_correlation_for_label(rho, "low_redundancy")
    theoretical = module.effective_corr_theory(3, rho, pairwise)

    target, sources = module.generate_multisource_data(
        n_items=200_000,
        n_sources=3,
        source_target_corr=rho,
        pairwise_source_corr=pairwise,
        rng=rng,
    )
    aggregate = np.mean(sources, axis=1)
    simulated = module.safe_corr(target, aggregate)

    assert abs(simulated - theoretical) < 0.03
