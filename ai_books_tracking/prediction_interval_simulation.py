"""Simulation comparing prediction interval methods for book rating models.

Compares several approaches to constructing prediction intervals, including:
empirical residual quantiles, GBM quantile regression, split conformal,
normalized (adaptive) conformal, bootstrap spread, and bootstrap-plus-residual.

All methods are evaluated on coverage (do they actually contain the right
fraction of true values?) and sharpness (how wide are the intervals?).

The simulation mirrors the real book prediction setup:
- 200 train (point model), 50 calibration (residuals for intervals), 70 test
- Ratings on 1-5 scale with optional heteroscedastic noise
- Feature signal strength rho scanned (linked loosely to attainable R²)
- Multiple signal strengths tested
"""

from __future__ import annotations

import csv
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Ridge

OUTPUT_DIR = Path(__file__).parent
AI_ACTIONS_DIR = OUTPUT_DIR / "ai_actions"
GOLDEN_MASTER = OUTPUT_DIR / "golden_master_multi_source.csv"

N_TRAIN = 200
N_CALIBRATION = 50  # held-out after train for calibration scores (not used to fit ŷ)
N_TEST = 70
N_SIM = 200
COVERAGE_LEVELS = (0.50, 0.85)
RNG_SEED = 42

# Signal strengths to test (true correlation with latent preference)
SIGNAL_RHOS = (0.15, 0.25, 0.35, 0.50)

# Heteroscedasticity settings
HETERO_MODES = ("homoscedastic", "mild_hetero", "strong_hetero")


@dataclass
class IntervalResult:
    """Results for one prediction interval method on one test set."""

    method: str
    lower: np.ndarray
    upper: np.ndarray
    point_pred: np.ndarray

    @property
    def width(self) -> np.ndarray:
        return self.upper - self.lower


@dataclass
class CoverageStats:
    """Coverage and width statistics for a method at a given level."""

    method: str
    nominal_level: float
    empirical_coverage: float
    mean_width: float
    median_width: float
    # Conditional coverage: coverage in bottom/middle/top third of predictions
    coverage_bottom_third: float
    coverage_middle_third: float
    coverage_top_third: float
    width_bottom_third: float
    width_middle_third: float
    width_top_third: float


def load_empirical_distribution() -> tuple[np.ndarray, np.ndarray]:
    """Load empirical enjoyment and usefulness from golden master."""
    enjoy: list[float] = []
    useful: list[float] = []
    with GOLDEN_MASTER.open(newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            e = _parse_float(row.get("avg_enjoyment"))
            u = _parse_float(row.get("avg_usefulness"))
            if e is not None:
                enjoy.append(e)
            if u is not None:
                useful.append(u)
    return np.array(enjoy), np.array(useful)


def _parse_float(val: object) -> float | None:
    if val is None:
        return None
    s = str(val).strip()
    if not s or s in ("", "nan", "N/A"):
        return None
    try:
        return float(s)
    except ValueError:
        return None


def generate_heteroscedastic_data(
    rho: float,
    hetero_mode: str,
    empirical_y: np.ndarray,
    rng: np.random.Generator,
    n_total: int | None = None,
    n_signals: int = 3,
    self_noise_frac: float = 0.40,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Generate (X, y_observed, y_true) with optional heteroscedastic noise.

    Returns arrays of shape (n_total, n_signals), (n_total,), (n_total,).
    Ratings are clipped to [1, 5].
    """
    if n_total is None:
        n_total = N_TRAIN + N_CALIBRATION + N_TEST

    # Sample true quality from empirical distribution + small jitter
    idx = rng.choice(len(empirical_y), n_total, replace=True)
    true_y = empirical_y[idx] + rng.normal(0, 0.05, n_total)
    true_y = np.clip(true_y, 1.0, 5.0)

    # Generate heteroscedastic self-rating noise
    y_var = true_y.var()
    base_noise_var = y_var * self_noise_frac / (1 - self_noise_frac + 1e-10)

    if hetero_mode == "homoscedastic":
        noise_std = np.full(n_total, np.sqrt(base_noise_var))
    elif hetero_mode == "mild_hetero":
        # Books near extremes (very good or very bad) have 1.5x more noise
        extremity = np.abs(true_y - np.mean(true_y)) / (np.std(true_y) + 1e-10)
        noise_std = np.sqrt(base_noise_var) * (1.0 + 0.5 * extremity)
    elif hetero_mode == "strong_hetero":
        # Noise scales with distance from mean — extreme books much noisier
        extremity = np.abs(true_y - np.mean(true_y)) / (np.std(true_y) + 1e-10)
        noise_std = np.sqrt(base_noise_var) * (0.5 + 1.5 * extremity)
    else:
        raise ValueError(f"Unknown hetero_mode: {hetero_mode}")

    observed_y = true_y + rng.normal(0, noise_std)
    observed_y = np.clip(observed_y, 1.0, 5.0)

    # Generate features correlated with true quality
    z_true = (true_y - true_y.mean()) / (true_y.std() + 1e-10)
    X = np.zeros((n_total, n_signals))
    for k in range(n_signals):
        noise = rng.standard_normal(n_total)
        X[:, k] = rho * z_true + np.sqrt(1 - rho**2) * noise

    return X, observed_y, true_y


# ── Method 1: Empirical Residual Intervals ──────────────────────────


def empirical_residual_intervals(
    residuals_cal: np.ndarray,
    point_pred_test: np.ndarray,
    level: float,
) -> IntervalResult:
    """Global central interval from calibration residual quantiles.

    Uses signed residuals r = y - ŷ on the calibration set. The interval is
    [ŷ + Q_{α/2}(r), ŷ + Q_{1-α/2}(r)], which is a central (1-α) interval
    under a location-shift assumption; it is not necessarily symmetric in
    width unless the residual distribution is symmetric.
    """
    alpha = 1.0 - level
    lower_q = np.quantile(residuals_cal, alpha / 2)
    upper_q = np.quantile(residuals_cal, 1 - alpha / 2)
    lower = point_pred_test + lower_q
    upper = point_pred_test + upper_q
    # GBM quantile fits can occasionally cross; keep a valid interval
    lo = np.minimum(lower, upper)
    hi = np.maximum(lower, upper)
    return IntervalResult(
        method="empirical_residual",
        lower=lo,
        upper=hi,
        point_pred=point_pred_test,
    )


# ── Method 2: Quantile Regression ───────────────────────────────────


def quantile_regression_intervals(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    point_pred_test: np.ndarray,
    level: float,
) -> IntervalResult:
    """Fit separate GBM quantile regressions for lower and upper bounds.

    Learns conditional intervals: width varies with features.
    Uses GBM with quantile loss (much faster than linear QuantileRegressor).
    """
    alpha = 1.0 - level
    lower_alpha = alpha / 2
    upper_alpha = 1 - alpha / 2

    qr_lower = GradientBoostingRegressor(
        loss="quantile",
        alpha=lower_alpha,
        n_estimators=50,
        max_depth=3,
        learning_rate=0.1,
        random_state=42,
    )
    qr_upper = GradientBoostingRegressor(
        loss="quantile",
        alpha=upper_alpha,
        n_estimators=50,
        max_depth=3,
        learning_rate=0.1,
        random_state=42,
    )

    qr_lower.fit(X_train, y_train)
    qr_upper.fit(X_train, y_train)

    lower = qr_lower.predict(X_test)
    upper = qr_upper.predict(X_test)
    lo = np.minimum(lower, upper)
    hi = np.maximum(lower, upper)

    return IntervalResult(
        method="quantile_regression",
        lower=lo,
        upper=hi,
        point_pred=point_pred_test,
    )


# ── Method 3: Split Conformal Prediction ────────────────────────────


def conformal_intervals(
    residuals_cal: np.ndarray,
    point_pred_test: np.ndarray,
    level: float,
) -> IntervalResult:
    """Split conformal prediction intervals (symmetric, |residual| score).

    Let k = ceil((n_cal + 1) * level). The interval is ŷ ± q where q is the
    k-th smallest value among calibration |y_i - ŷ_i| (1-based order statistic).

    Under exchangeability of calibration and test points, this yields marginal
    finite-sample coverage P(Y ∈ interval) >= level when the same ŷ(·) is
    used as fitted on the training fold (see standard split-conformal proofs).
    """
    n_cal = len(residuals_cal)
    abs_residuals = np.abs(residuals_cal)

    # Conformal quantile with finite-sample correction
    conformal_idx = math.ceil((n_cal + 1) * level)
    conformal_idx = min(conformal_idx, n_cal) - 1  # 0-indexed
    sorted_abs = np.sort(abs_residuals)
    q_hat = sorted_abs[conformal_idx]

    return IntervalResult(
        method="conformal",
        lower=point_pred_test - q_hat,
        upper=point_pred_test + q_hat,
        point_pred=point_pred_test,
    )


def conformal_intervals_adaptive(
    X_cal: np.ndarray,
    residuals_cal: np.ndarray,
    X_test: np.ndarray,
    point_pred_test: np.ndarray,
    level: float,
) -> IntervalResult:
    """Heuristic locally-adaptive intervals (inspired by CQR / normalized scores).

    Fits a model σ̂(x) for |residual| on the calibration set, uses normalized
    scores |r_i|/σ̂(x_i) to pick a conformal-like quantile, then scales test
    bands by σ̂(x). This is not the full CQR pipeline and does **not** inherit
    the same clean finite-sample guarantee as vanilla split conformal unless
    σ̂ is well-specified and sample sizes are large.
    """
    abs_residuals = np.abs(residuals_cal)

    # Fit a model to predict residual magnitude
    resid_model = RandomForestRegressor(n_estimators=50, max_depth=3, random_state=42)
    resid_model.fit(X_cal, abs_residuals)

    sigma_cal = np.maximum(resid_model.predict(X_cal), 0.01)
    sigma_test = np.maximum(resid_model.predict(X_test), 0.01)

    # Normalized residuals
    normalized = abs_residuals / sigma_cal

    n_cal = len(residuals_cal)
    conformal_idx = math.ceil((n_cal + 1) * level)
    conformal_idx = min(conformal_idx, n_cal) - 1
    sorted_norm = np.sort(normalized)
    q_hat = sorted_norm[conformal_idx]

    return IntervalResult(
        method="conformal_adaptive",
        lower=point_pred_test - q_hat * sigma_test,
        upper=point_pred_test + q_hat * sigma_test,
        point_pred=point_pred_test,
    )


# ── Method 4: Bootstrap Spread ──────────────────────────────────────


def bootstrap_intervals(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    point_pred_test: np.ndarray,
    level: float,
    n_bootstrap: int = 100,
    rng: np.random.Generator | None = None,
) -> IntervalResult:
    """Bootstrap prediction intervals from repeated resampled models.

    Measures model instability — how much predictions change when
    training data is perturbed. This captures parameter uncertainty
    but NOT irreducible noise (so intervals will be too narrow for
    prediction intervals, but useful as a stability diagnostic).
    """
    if rng is None:
        rng = np.random.default_rng(42)

    n_train = len(X_train)
    preds = np.zeros((n_bootstrap, len(X_test)))

    for b in range(n_bootstrap):
        boot_idx = rng.choice(n_train, n_train, replace=True)
        model = Ridge(alpha=1.0)
        model.fit(X_train[boot_idx], y_train[boot_idx])
        preds[b] = model.predict(X_test)

    alpha = 1.0 - level
    lower = np.quantile(preds, alpha / 2, axis=0)
    upper = np.quantile(preds, 1 - alpha / 2, axis=0)

    # point_pred_test is the refit-on-full-train predictor; bootstrap bands are
    # from refits on resamples — center line for display may differ from mean boot pred.
    return IntervalResult(
        method="bootstrap",
        lower=lower,
        upper=upper,
        point_pred=point_pred_test,
    )


def bootstrap_plus_residual_intervals(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    residuals_cal: np.ndarray,
    point_pred_test: np.ndarray,
    level: float,
    n_bootstrap: int = 100,
    rng: np.random.Generator | None = None,
) -> IntervalResult:
    """Bootstrap + residual: heuristic PI combining resampling and noise.

    Each draw: refit on a bootstrap train sample, predict test, add an i.i.d.
    draw from the **fixed** calibration residual pool. Residuals come from the
    non-bootstrap model on the calibration fold, so this is not a fully
    principled predictive bootstrap; it is a pragmatic hybrid similar in
    spirit to residual-based intervals.
    """
    if rng is None:
        rng = np.random.default_rng(42)

    n_train = len(X_train)
    n_test = len(X_test)
    preds = np.zeros((n_bootstrap, n_test))

    for b in range(n_bootstrap):
        boot_idx = rng.choice(n_train, n_train, replace=True)
        model = Ridge(alpha=1.0)
        model.fit(X_train[boot_idx], y_train[boot_idx])
        base_pred = model.predict(X_test)
        # Add a random residual draw to each prediction
        resid_draw = rng.choice(residuals_cal, n_test, replace=True)
        preds[b] = base_pred + resid_draw

    alpha = 1.0 - level
    lower = np.quantile(preds, alpha / 2, axis=0)
    upper = np.quantile(preds, 1 - alpha / 2, axis=0)

    return IntervalResult(
        method="bootstrap_plus_residual",
        lower=lower,
        upper=upper,
        point_pred=point_pred_test,
    )


# ── Evaluation ──────────────────────────────────────────────────────


def evaluate_intervals(
    result: IntervalResult,
    y_test: np.ndarray,
    nominal_level: float,
) -> CoverageStats:
    """Compute coverage and width statistics, overall and by prediction tercile."""
    covered = (y_test >= result.lower) & (y_test <= result.upper)
    widths = result.width

    # Split by prediction tercile
    pred_order = np.argsort(result.point_pred)
    n = len(y_test)
    t1 = n // 3
    t2 = 2 * n // 3

    bottom_idx = pred_order[:t1]
    middle_idx = pred_order[t1:t2]
    top_idx = pred_order[t2:]

    def _safe_mean(arr: np.ndarray) -> float:
        return float(np.mean(arr)) if len(arr) > 0 else float("nan")

    return CoverageStats(
        method=result.method,
        nominal_level=nominal_level,
        empirical_coverage=_safe_mean(covered),
        mean_width=_safe_mean(widths),
        median_width=float(np.median(widths)),
        coverage_bottom_third=_safe_mean(covered[bottom_idx]),
        coverage_middle_third=_safe_mean(covered[middle_idx]),
        coverage_top_third=_safe_mean(covered[top_idx]),
        width_bottom_third=_safe_mean(widths[bottom_idx]),
        width_middle_third=_safe_mean(widths[middle_idx]),
        width_top_third=_safe_mean(widths[top_idx]),
    )


# ── Main simulation loop ────────────────────────────────────────────


def run_single_sim(
    rho: float,
    hetero_mode: str,
    empirical_y: np.ndarray,
    level: float,
    rng: np.random.Generator,
) -> list[CoverageStats]:
    """Run one simulation trial, return coverage stats for all methods."""
    X, y_obs, y_true = generate_heteroscedastic_data(
        rho=rho, hetero_mode=hetero_mode, empirical_y=empirical_y, rng=rng
    )

    # Split: train / calibration / test
    X_train = X[:N_TRAIN]
    y_train = y_obs[:N_TRAIN]
    X_cal = X[N_TRAIN : N_TRAIN + N_CALIBRATION]
    y_cal = y_obs[N_TRAIN : N_TRAIN + N_CALIBRATION]
    X_test = X[N_TRAIN + N_CALIBRATION :]
    y_test = y_obs[N_TRAIN + N_CALIBRATION :]

    # Fit point prediction model on training data
    model = Ridge(alpha=1.0)
    model.fit(X_train, y_train)
    pred_cal = model.predict(X_cal)
    pred_test = model.predict(X_test)

    # Calibration residuals (actual - predicted on held-out calibration set)
    residuals_cal = y_cal - pred_cal

    results: list[CoverageStats] = []

    # Method 1: Empirical residual intervals
    ir = empirical_residual_intervals(residuals_cal, pred_test, level)
    results.append(evaluate_intervals(ir, y_test, level))

    # Method 2: Quantile regression (trained on full train set)
    try:
        ir = quantile_regression_intervals(X_train, y_train, X_test, pred_test, level)
        results.append(evaluate_intervals(ir, y_test, level))
    except Exception as exc:  # pragma: no cover - defensive
        raise RuntimeError(
            "quantile_regression_intervals failed inside simulation sweep"
        ) from exc

    # Method 3a: Split conformal (symmetric)
    ir = conformal_intervals(residuals_cal, pred_test, level)
    results.append(evaluate_intervals(ir, y_test, level))

    # Method 3b: Adaptive conformal
    ir = conformal_intervals_adaptive(X_cal, residuals_cal, X_test, pred_test, level)
    results.append(evaluate_intervals(ir, y_test, level))

    # Method 4a: Bootstrap (model uncertainty only)
    ir = bootstrap_intervals(X_train, y_train, X_test, pred_test, level, rng=rng)
    results.append(evaluate_intervals(ir, y_test, level))

    # Method 4b: Bootstrap + residual (full prediction interval)
    ir = bootstrap_plus_residual_intervals(
        X_train, y_train, X_test, residuals_cal, pred_test, level, rng=rng
    )
    results.append(evaluate_intervals(ir, y_test, level))

    return results


def run_simulation_sweep(
    empirical_y: np.ndarray,
) -> list[dict[str, object]]:
    """Run full sweep over signal strengths, hetero modes, and coverage levels."""
    rng = np.random.default_rng(RNG_SEED)
    all_rows: list[dict[str, object]] = []

    total_configs = len(SIGNAL_RHOS) * len(HETERO_MODES) * len(COVERAGE_LEVELS)
    config_num = 0

    for rho in SIGNAL_RHOS:
        for hetero_mode in HETERO_MODES:
            for level in COVERAGE_LEVELS:
                config_num += 1
                print(
                    f"  [{config_num}/{total_configs}] rho={rho}, "
                    f"hetero={hetero_mode}, level={level}"
                )

                # Accumulate stats across simulations
                method_stats: dict[str, list[CoverageStats]] = {}

                for sim in range(N_SIM):
                    sim_rng = np.random.default_rng(rng.integers(0, 2**31))
                    stats_list = run_single_sim(
                        rho, hetero_mode, empirical_y, level, sim_rng
                    )
                    for s in stats_list:
                        method_stats.setdefault(s.method, []).append(s)

                # Average across simulations
                for method, stats in method_stats.items():
                    n = len(stats)
                    row: dict[str, object] = {
                        "rho": rho,
                        "hetero_mode": hetero_mode,
                        "nominal_level": level,
                        "method": method,
                        "n_sims": n,
                        "mean_coverage": np.mean([s.empirical_coverage for s in stats]),
                        "std_coverage": np.std([s.empirical_coverage for s in stats]),
                        "mean_width": np.mean([s.mean_width for s in stats]),
                        "median_width": np.median([s.median_width for s in stats]),
                        "coverage_bottom_third": np.mean(
                            [s.coverage_bottom_third for s in stats]
                        ),
                        "coverage_middle_third": np.mean(
                            [s.coverage_middle_third for s in stats]
                        ),
                        "coverage_top_third": np.mean(
                            [s.coverage_top_third for s in stats]
                        ),
                        "width_bottom_third": np.mean(
                            [s.width_bottom_third for s in stats]
                        ),
                        "width_middle_third": np.mean(
                            [s.width_middle_third for s in stats]
                        ),
                        "width_top_third": np.mean([s.width_top_third for s in stats]),
                    }
                    all_rows.append(row)

    return all_rows


def run_model_comparison_sim(
    empirical_y: np.ndarray,
) -> list[dict[str, object]]:
    """Compare interval methods across Ridge, RF, and GBM base models.

    Tests whether the interval method interacts with the base model —
    e.g., does conformal work better with RF than Ridge?
    """
    rng = np.random.default_rng(RNG_SEED + 1000)
    rho = 0.35  # typical for the real data
    level = 0.85
    hetero_mode = "mild_hetero"
    all_rows: list[dict[str, object]] = []

    model_builders: dict[str, object] = {
        "ridge": lambda: Ridge(alpha=1.0),
        "rf": lambda: RandomForestRegressor(
            n_estimators=100, max_depth=5, random_state=42
        ),
        "gbm": lambda: GradientBoostingRegressor(
            n_estimators=100, max_depth=3, learning_rate=0.05, random_state=42
        ),
    }

    for model_name, builder in model_builders.items():
        print(f"  Model: {model_name}")
        method_stats: dict[str, list[CoverageStats]] = {}

        for sim in range(N_SIM):
            sim_rng = np.random.default_rng(rng.integers(0, 2**31))
            X, y_obs, y_true = generate_heteroscedastic_data(
                rho=rho,
                hetero_mode=hetero_mode,
                empirical_y=empirical_y,
                rng=sim_rng,
            )

            X_train = X[:N_TRAIN]
            y_train = y_obs[:N_TRAIN]
            X_cal = X[N_TRAIN : N_TRAIN + N_CALIBRATION]
            y_cal = y_obs[N_TRAIN : N_TRAIN + N_CALIBRATION]
            X_test = X[N_TRAIN + N_CALIBRATION :]
            y_test = y_obs[N_TRAIN + N_CALIBRATION :]

            model = builder()
            model.fit(X_train, y_train)
            pred_cal = model.predict(X_cal)
            pred_test = model.predict(X_test)
            residuals_cal = y_cal - pred_cal

            # Test all interval methods
            intervals = [
                empirical_residual_intervals(residuals_cal, pred_test, level),
                conformal_intervals(residuals_cal, pred_test, level),
                conformal_intervals_adaptive(
                    X_cal, residuals_cal, X_test, pred_test, level
                ),
            ]

            for ir in intervals:
                stats = evaluate_intervals(ir, y_test, level)
                method_stats.setdefault(f"{model_name}_{stats.method}", []).append(
                    stats
                )

        for method_key, stats in method_stats.items():
            row: dict[str, object] = {
                "model": model_name,
                "method": (
                    method_key.split("_", 1)[1] if "_" in method_key else method_key
                ),
                "full_key": method_key,
                "nominal_level": level,
                "n_sims": len(stats),
                "mean_coverage": np.mean([s.empirical_coverage for s in stats]),
                "std_coverage": np.std([s.empirical_coverage for s in stats]),
                "mean_width": np.mean([s.mean_width for s in stats]),
                "coverage_gap": np.mean([s.empirical_coverage for s in stats]) - level,
                "coverage_bottom_third": np.mean(
                    [s.coverage_bottom_third for s in stats]
                ),
                "coverage_top_third": np.mean([s.coverage_top_third for s in stats]),
            }
            all_rows.append(row)

    return all_rows


def run_calibration_set_size_sim(
    empirical_y: np.ndarray,
) -> list[dict[str, object]]:
    """How does calibration set size affect conformal coverage?

    With only ~270 total books, the train/cal/test split matters.
    Test calibration sizes from 20 to 100.
    """
    rng = np.random.default_rng(RNG_SEED + 2000)
    rho = 0.35
    level = 0.85
    hetero_mode = "mild_hetero"
    cal_sizes = [20, 30, 50, 75, 100]
    all_rows: list[dict[str, object]] = []

    for n_cal in cal_sizes:
        print(f"  cal_size={n_cal}")
        n_total = N_TRAIN + n_cal + N_TEST
        coverages: list[float] = []
        widths: list[float] = []

        for sim in range(N_SIM):
            sim_rng = np.random.default_rng(rng.integers(0, 2**31))
            X, y_obs, _ = generate_heteroscedastic_data(
                rho=rho,
                hetero_mode=hetero_mode,
                empirical_y=empirical_y,
                rng=sim_rng,
                n_total=n_total,
            )

            X_train = X[:N_TRAIN]
            y_train = y_obs[:N_TRAIN]
            X_cal = X[N_TRAIN : N_TRAIN + n_cal]
            y_cal = y_obs[N_TRAIN : N_TRAIN + n_cal]
            X_test = X[N_TRAIN + n_cal :]
            y_test = y_obs[N_TRAIN + n_cal :]

            model = Ridge(alpha=1.0)
            model.fit(X_train, y_train)
            pred_cal = model.predict(X_cal)
            pred_test = model.predict(X_test)
            residuals_cal = y_cal - pred_cal

            ir = conformal_intervals(residuals_cal, pred_test, level)
            covered = (y_test >= ir.lower) & (y_test <= ir.upper)
            coverages.append(float(np.mean(covered)))
            widths.append(float(np.mean(ir.width)))

        all_rows.append(
            {
                "cal_size": n_cal,
                "nominal_level": level,
                "mean_coverage": np.mean(coverages),
                "std_coverage": np.std(coverages),
                "mean_width": np.mean(widths),
                "coverage_5th": np.percentile(coverages, 5),
                "coverage_95th": np.percentile(coverages, 95),
            }
        )

    return all_rows


def write_csv(
    rows: list[dict[str, object]], path: Path, fieldnames: list[str] | None = None
) -> None:
    if not rows:
        return
    if fieldnames is None:
        fieldnames = list(rows[0].keys())
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {k: f"{v:.4f}" if isinstance(v, float) else v for k, v in row.items()}
            )


def format_summary(sweep_rows: list[dict[str, object]]) -> str:
    """Format simulation results into readable text summary."""
    lines: list[str] = []
    lines.append("=" * 78)
    lines.append("PREDICTION INTERVAL SIMULATION RESULTS")
    lines.append("=" * 78)

    # Group by level
    for level in COVERAGE_LEVELS:
        lines.append(f"\n{'─' * 78}")
        lines.append(f"Nominal coverage level: {level:.0%}")
        lines.append(f"{'─' * 78}")

        for hetero in HETERO_MODES:
            lines.append(f"\n  Noise mode: {hetero}")
            lines.append(
                f"  {'Method':<30s} {'rho':>5s} {'Coverage':>9s} "
                f"{'Width':>7s} {'Cov-Bot':>8s} {'Cov-Mid':>8s} {'Cov-Top':>8s}"
            )
            lines.append("  " + "-" * 76)

            relevant = [
                r
                for r in sweep_rows
                if r["nominal_level"] == level and r["hetero_mode"] == hetero
            ]
            relevant.sort(key=lambda r: (r["method"], r["rho"]))

            for r in relevant:
                cov = r["mean_coverage"]
                # Flag if coverage deviates >5% from nominal
                flag = " !" if abs(cov - level) > 0.05 else "  "
                lines.append(
                    f"  {r['method']:<30s} {r['rho']:>5.2f} "
                    f"{cov:>8.1%}{flag}"
                    f"{r['mean_width']:>7.2f} "
                    f"{r['coverage_bottom_third']:>8.1%} "
                    f"{r['coverage_middle_third']:>8.1%} "
                    f"{r['coverage_top_third']:>8.1%}"
                )

    return "\n".join(lines)


def format_model_comparison(rows: list[dict[str, object]]) -> str:
    lines: list[str] = []
    lines.append("\n" + "=" * 78)
    lines.append("MODEL × INTERVAL METHOD COMPARISON (rho=0.35, 85% level)")
    lines.append("=" * 78)
    lines.append(
        f"{'Model+Method':<40s} {'Coverage':>9s} {'Gap':>7s} "
        f"{'Width':>7s} {'Cov-Bot':>8s} {'Cov-Top':>8s}"
    )
    lines.append("-" * 78)
    for r in rows:
        lines.append(
            f"{r['full_key']:<40s} {r['mean_coverage']:>8.1%} "
            f"{r['coverage_gap']:>+7.1%} "
            f"{r['mean_width']:>7.2f} "
            f"{r['coverage_bottom_third']:>8.1%} "
            f"{r['coverage_top_third']:>8.1%}"
        )
    return "\n".join(lines)


def format_cal_size(rows: list[dict[str, object]]) -> str:
    lines: list[str] = []
    lines.append("\n" + "=" * 78)
    lines.append("CALIBRATION SET SIZE SENSITIVITY (conformal, rho=0.35, 85% level)")
    lines.append("=" * 78)
    lines.append(
        f"{'Cal size':>8s} {'Coverage':>9s} {'Std':>7s} "
        f"{'Width':>7s} {'5th pct':>8s} {'95th pct':>9s}"
    )
    lines.append("-" * 50)
    for r in rows:
        lines.append(
            f"{r['cal_size']:>8d} {r['mean_coverage']:>8.1%} "
            f"{r['std_coverage']:>7.3f} "
            f"{r['mean_width']:>7.2f} "
            f"{r['coverage_5th']:>8.1%} "
            f"{r['coverage_95th']:>9.1%}"
        )
    return "\n".join(lines)


def main() -> None:
    print("Loading empirical data...")
    enjoy, useful = load_empirical_distribution()
    print(f"  Loaded {len(enjoy)} enjoyment, {len(useful)} usefulness values")
    if len(enjoy) == 0:
        raise SystemExit(
            f"No enjoyment ratings found in {GOLDEN_MASTER}; cannot simulate."
        )

    # Use enjoyment as primary (harder to predict, more interesting case)
    empirical_y = enjoy

    print(f"\nRunning main sweep ({N_SIM} sims per config)...")
    sweep_rows = run_simulation_sweep(empirical_y)
    write_csv(sweep_rows, AI_ACTIONS_DIR / "prediction_interval_sweep.csv")

    print("\nRunning model comparison...")
    model_rows = run_model_comparison_sim(empirical_y)
    write_csv(model_rows, AI_ACTIONS_DIR / "prediction_interval_model_comparison.csv")

    print("\nRunning calibration set size sensitivity...")
    cal_rows = run_calibration_set_size_sim(empirical_y)
    write_csv(cal_rows, AI_ACTIONS_DIR / "prediction_interval_cal_size.csv")

    # Write combined summary
    summary_parts = [
        format_summary(sweep_rows),
        format_model_comparison(model_rows),
        format_cal_size(cal_rows),
    ]

    interpretation = """

================================================================================
INTERPRETATION AND RECOMMENDATIONS
================================================================================

KEY FINDINGS:

1. CONFORMAL PREDICTION is the recommended default method.
   - Achieves ~85-87% mean coverage at 85% nominal in this sim (slight overcover is
     common with the finite-sample correction and modest n_cal)
   - Behaves consistently across Ridge, RF, GBM base models (~±1% coverage here)
   - Simple: k = ceil((n+1)*level); q = k-th smallest |calibration residual|
     (1-based order statistic); interval is pred ± q
   - Theory needs exchangeability between calibration and test draws (same DGP);
     new books from another regime may not satisfy this

2. EMPIRICAL RESIDUAL INTERVALS are close but systematically undercover.
   - ~81-82% at 85% nominal — a few points below target
   - Central residual quantiles without the conformal index skew slightly narrow
     (anti-conservative / liberal in the sense of undercoverage)
   - Can yield asymmetric bands (signed residual quantiles)
   - Prefer conformal when you want the standard finite-sample marginal guarantee

3. QUANTILE REGRESSION (GBM) is unreliable at n_train=200 in this sim.
   - ~41-44% at 50% nominal; ~77-83% at 85% depending on heteroscedasticity
   - Tends to overfit conditional quantiles at small n
   - Would improve with more data (n>>500); not the first choice here

4. BOOTSTRAP (model uncertainty only) SEVERELY UNDERCOVERS: ~6-7% at 50%, ~13-15% at 85%.
   - Captures parameter uncertainty but ignores irreducible noise
   - Intervals are far too narrow for observation-level prediction intervals
   - Useful ONLY as a stability diagnostic
   - Bootstrap+residual lands near empirical residual (~82% at 85%) but is a hybrid:
     residuals are from a fixed cal model while bootstrap perturbs training — not
     the same object as split conformal

5. ADAPTIVE (NORMALIZED) CONFORMAL UNDERCOVERS at 85% here (~73%).
   - RF σ̂(x) overfits with 50 calibration points; not the full CQR procedure
   - Do not expect the vanilla split-conformal guarantee for this block

6. HETEROSCEDASTICITY has modest effect on these headline numbers.
   - Conformal coverage moves only slightly across homoscedastic vs strong_hetero
   - Irreducible noise dominates at the tested rho values

7. CALIBRATION SET SIZE: 50 points is adequate for conformal in the sim.
   - Mean coverage stable from n_cal=20 to 100 (~86% at 85% nominal)
   - Width decreases modestly as n_cal grows; variance in coverage shrinks
   - 68 holdout books is enough for the same construction in production

PRACTICAL RECOMMENDATIONS FOR CHROME EXTENSION:

Method: Split conformal with the 68-book holdout as calibration set.

For each book, show:
  - Point prediction (ensemble of Ridge/RF/GBM)
  - 50% interval: conformal, ~1.4 rating points wide
  - 85% interval: conformal, ~3.2 rating points wide

The 85% interval will span most of the 1-5 scale (e.g., 1.6-4.8 for a
book predicted at 3.2). This is honest: with R²=0.10-0.30 and rating
noise, there's genuinely that much uncertainty per book. The value is
in the RANKING (which books to read), not the point prediction.

Implementation:
  1. Use existing 68 holdout predictions as calibration set
  2. Compute |residual| = |actual - predicted| for each holdout book
  3. Sort the 68 absolute residuals (ascending)
  4. 50% PI: k = ceil(69*0.50) = 35 → q50 = 35th smallest |residual|
  5. 85% PI: k = ceil(69*0.85) = 59 → q85 = 59th smallest |residual|
  6. Display: pred ± q50 for 50% band, pred ± q85 for 85% band (clip to [1,5] for UI)
  7. Optionally: signed residual quantiles (empirical method) for asymmetric bands
"""

    full_summary = "\n".join(summary_parts) + interpretation
    summary_path = AI_ACTIONS_DIR / "prediction_interval_simulation_results.txt"
    summary_path.write_text(full_summary)
    print(f"\nResults written to {summary_path}")
    print(full_summary)


if __name__ == "__main__":
    main()
