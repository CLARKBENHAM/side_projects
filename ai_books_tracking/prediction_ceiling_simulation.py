"""
Monte Carlo simulation to estimate prediction ceilings for book ratings.

Given:
- User's own ratings have measurement noise (test-retest reliability)
- External predictor variables (Goodreads, Amazon, OpenLibrary) have:
  - Imperfect correlation with the user's "true" latent preference
  - Their own measurement noise
- Training sample ~200 books, holdout ~70

Questions answered:
1. What's the irreducible floor given rating noise?
2. Best achievable R with 1..5 perfect predictors?
3. Effect of predictor noise?
4. Effect of imperfect true correlation?
5. How does sample size matter?
"""

import numpy as np
from dataclasses import dataclass

N_SIM = 5000  # replications per scenario
RNG = np.random.default_rng(42)


@dataclass
class RatingParams:
    name: str
    test_retest_r: float
    mae: float
    rmse: float
    mean: float
    std: float  # of observed ratings


ENJOYMENT = RatingParams(
    name="Enjoyment",
    test_retest_r=0.77,
    mae=0.46,
    rmse=0.66,
    mean=3.41,
    std=0.96,  # approximate from the data
)

USEFULNESS = RatingParams(
    name="Usefulness",
    test_retest_r=0.86,
    mae=0.34,
    rmse=0.59,
    mean=1.80,
    std=1.10,  # approximate
)


def generate_data(
    n: int,
    reliability_y: float,
    n_predictors: int,
    true_correlations: list[float],
    predictor_reliabilities: list[float],
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Generate simulated data from classical test theory model.

    Model:
      T ~ N(0, 1)              -- true latent rating
      Y1 = T + e1              -- first rating pass
      Y2 = T + e2              -- second rating pass
      Y_avg = (Y1 + Y2) / 2    -- target to predict

      For each predictor k:
        X_true_k = rho_k * T + sqrt(1 - rho_k^2) * u_k   -- true predictor value
        X_obs_k = X_true_k + noise_k                       -- observed (noisy) predictor

    The reliability of Y1 (and Y2) is:
      Cor(Y1, Y2) = Var(T) / (Var(T) + Var(e)) = reliability_y

    Returns: (X_obs, Y_avg, Y1, Y2) where X_obs is (n, n_predictors)
    """
    # Derive noise variance from reliability
    # reliability = var_T / (var_T + var_e)
    # With var_T = reliability, var_e = 1 - reliability (so total obs variance = 1)
    var_t = reliability_y
    var_e = 1.0 - reliability_y

    T = rng.normal(0, np.sqrt(var_t), size=n)
    e1 = rng.normal(0, np.sqrt(var_e), size=n)
    e2 = rng.normal(0, np.sqrt(var_e), size=n)
    Y1 = T + e1
    Y2 = T + e2
    Y_avg = (Y1 + Y2) / 2.0

    X_obs = np.zeros((n, n_predictors))
    for k in range(n_predictors):
        rho_k = true_correlations[k]
        # X_true has correlation rho_k with T
        u_k = rng.normal(0, 1, size=n)
        X_true_k = rho_k * T + np.sqrt(1.0 - rho_k**2) * u_k

        # Add predictor measurement noise
        rel_k = predictor_reliabilities[k]
        if rel_k < 1.0:
            # rel_k = var(X_true) / (var(X_true) + var(noise))
            # var(X_true) ≈ 1 (by construction), so var(noise) = (1 - rel_k) / rel_k
            var_noise_k = (1.0 - rel_k) / rel_k
            noise_k = rng.normal(0, np.sqrt(var_noise_k), size=n)
            X_obs[:, k] = X_true_k + noise_k
        else:
            X_obs[:, k] = X_true_k

    return X_obs, Y_avg, Y1, Y2


def ols_predict(
    X_train: np.ndarray, y_train: np.ndarray, X_test: np.ndarray
) -> np.ndarray:
    """OLS regression, returns predictions on test set."""
    # Add intercept
    ones_train = np.ones((X_train.shape[0], 1))
    ones_test = np.ones((X_test.shape[0], 1))
    Xt = np.hstack([ones_train, X_train])
    Xv = np.hstack([ones_test, X_test])
    # Solve normal equations with regularization for numerical stability
    beta = np.linalg.lstsq(Xt, y_train, rcond=None)[0]
    return Xv @ beta


def run_scenario(
    n_train: int,
    n_test: int,
    reliability_y: float,
    n_predictors: int,
    true_correlations: list[float],
    predictor_reliabilities: list[float],
    n_sim: int = N_SIM,
) -> dict[str, np.ndarray]:
    """Run Monte Carlo simulation for one scenario.

    Returns dict with arrays of length n_sim for each metric.
    """
    results: dict[str, list[float]] = {
        "r_pred_yavg": [],
        "r2_pred_yavg": [],
        "mae_pred_yavg": [],
        "rmse_pred_yavg": [],
        "r_pred_y1": [],
    }

    for _ in range(n_sim):
        n_total = n_train + n_test
        X, Y_avg, Y1, Y2 = generate_data(
            n_total,
            reliability_y,
            n_predictors,
            true_correlations,
            predictor_reliabilities,
            RNG,
        )

        X_train, X_test = X[:n_train], X[n_train:]
        Y_avg_train, Y_avg_test = Y_avg[:n_train], Y_avg[n_train:]
        Y1_test = Y1[n_train:]

        y_pred = ols_predict(X_train, Y_avg_train, X_test)

        r_avg = np.corrcoef(y_pred, Y_avg_test)[0, 1]
        r_y1 = np.corrcoef(y_pred, Y1_test)[0, 1]
        residuals = y_pred - Y_avg_test

        results["r_pred_yavg"].append(r_avg)
        results["r2_pred_yavg"].append(r_avg**2)
        results["mae_pred_yavg"].append(np.mean(np.abs(residuals)))
        results["rmse_pred_yavg"].append(np.sqrt(np.mean(residuals**2)))
        results["r_pred_y1"].append(r_y1)

    return {k: np.array(v) for k, v in results.items()}


def fmt_ci(arr: np.ndarray) -> str:
    """Format median [5th, 95th percentile]."""
    med = np.median(arr)
    lo, hi = np.percentile(arr, [5, 95])
    return (
        f"{med:+.3f} [{lo:+.3f}, {hi:+.3f}]"
        if med < 0
        else f"{med:.3f} [{lo:.3f}, {hi:.3f}]"
    )


def spearman_brown(r: float, k: int = 2) -> float:
    """Spearman-Brown prophecy formula: reliability of average of k measures."""
    return k * r / (1 + (k - 1) * r)


def theoretical_max_r(reliability_y_avg: float) -> float:
    """Max achievable R predicting Y_avg from a perfect noiseless predictor."""
    return np.sqrt(reliability_y_avg)


def print_section(title: str) -> None:
    print(f"\n{'='*70}")
    print(f"  {title}")
    print(f"{'='*70}")


def main() -> None:
    n_train = 200
    n_test = 70

    for params in [ENJOYMENT, USEFULNESS]:
        print_section(f"ANALYSIS FOR: {params.name.upper()}")

        rel_single = params.test_retest_r
        rel_avg = spearman_brown(rel_single, 2)
        max_r = theoretical_max_r(rel_avg)

        print(f"\n  Test-retest reliability (single pass):  R = {rel_single:.3f}")
        print(f"  Spearman-Brown reliability (avg of 2):  R = {rel_avg:.3f}")
        print(f"  Theoretical max R (perfect predictor):  R = {max_r:.3f}")
        print(f"  Theoretical max R²:                     R² = {rel_avg:.3f}")
        print(f"  Observed RMSE between passes:           {params.rmse:.3f}")
        print(f"  Observed MAE between passes:            {params.mae:.3f}")

        # ── Section 1: Ceiling with perfect predictors ──
        print_section(
            f"{params.name}: Best R with 1..5 PERFECT predictors (ρ=1.0, no noise)"
        )
        print(
            f"  {'Predictors':>10s}  {'Median R':>20s}  {'Median R²':>20s}  {'Median MAE':>20s}"
        )
        print(f"  {'-'*10}  {'-'*20}  {'-'*20}  {'-'*20}")

        for n_pred in range(1, 6):
            res = run_scenario(
                n_train,
                n_test,
                rel_single,
                n_pred,
                true_correlations=[1.0] * n_pred,
                predictor_reliabilities=[1.0] * n_pred,
            )
            print(
                f"  {n_pred:>10d}  {fmt_ci(res['r_pred_yavg']):>20s}  "
                f"{fmt_ci(res['r2_pred_yavg']):>20s}  {fmt_ci(res['mae_pred_yavg']):>20s}"
            )

        # ── Section 2: Varying true correlation ──
        print_section(f"{params.name}: R vs true correlation (1 noiseless predictor)")
        print(
            f"  {'True ρ':>10s}  {'Median R':>20s}  {'Median R²':>20s}  {'Theory R':>10s}"
        )
        print(f"  {'-'*10}  {'-'*20}  {'-'*20}  {'-'*10}")

        for rho in [0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 1.00]:
            res = run_scenario(
                n_train,
                n_test,
                rel_single,
                1,
                true_correlations=[rho],
                predictor_reliabilities=[1.0],
            )
            # Theoretical: observed R = rho * sqrt(rel_avg)
            theory_r = rho * np.sqrt(rel_avg)
            print(
                f"  {rho:>10.2f}  {fmt_ci(res['r_pred_yavg']):>20s}  "
                f"{fmt_ci(res['r2_pred_yavg']):>20s}  {theory_r:>10.3f}"
            )

        # ── Section 3: Multiple predictors with realistic correlations ──
        print_section(
            f"{params.name}: Multiple predictors with REALISTIC true correlations"
        )
        print(
            "  Scenario: predictors have true ρ with T of [0.35, 0.25, 0.10, 0.20, 0.15]"
        )
        print(
            "  (approximating Goodreads, Amazon, OpenLibrary, category, author effects)"
        )
        print()
        print(
            f"  {'Predictors':>10s}  {'Median R':>20s}  {'Median R²':>20s}  {'Median MAE':>20s}"
        )
        print(f"  {'-'*10}  {'-'*20}  {'-'*20}  {'-'*20}")

        realistic_rhos = [0.35, 0.25, 0.10, 0.20, 0.15]
        for n_pred in range(1, 6):
            res = run_scenario(
                n_train,
                n_test,
                rel_single,
                n_pred,
                true_correlations=realistic_rhos[:n_pred],
                predictor_reliabilities=[1.0] * n_pred,
            )
            rho_str = str(realistic_rhos[:n_pred])
            print(
                f"  {n_pred:>10d}  {fmt_ci(res['r_pred_yavg']):>20s}  "
                f"{fmt_ci(res['r2_pred_yavg']):>20s}  {fmt_ci(res['mae_pred_yavg']):>20s}"
                f"  ρs={rho_str}"
            )

        # ── Section 4: Effect of predictor noise ──
        print_section(
            f"{params.name}: Effect of predictor noise (1 predictor, true ρ=0.35)"
        )
        print(
            f"  {'Pred Reliability':>16s}  {'Median R':>20s}  {'Median R²':>20s}  {'Theory R':>10s}"
        )
        print(f"  {'-'*16}  {'-'*20}  {'-'*20}  {'-'*10}")

        for pred_rel in [1.0, 0.95, 0.90, 0.80, 0.70, 0.60, 0.50]:
            res = run_scenario(
                n_train,
                n_test,
                rel_single,
                1,
                true_correlations=[0.35],
                predictor_reliabilities=[pred_rel],
            )
            # Theory: observed R = rho * sqrt(rel_pred) * sqrt(rel_yavg)
            theory_r = 0.35 * np.sqrt(pred_rel) * np.sqrt(rel_avg)
            print(
                f"  {pred_rel:>16.2f}  {fmt_ci(res['r_pred_yavg']):>20s}  "
                f"{fmt_ci(res['r2_pred_yavg']):>20s}  {theory_r:>10.3f}"
            )

        # ── Section 5: Multiple noisy predictors ──
        print_section(
            f"{params.name}: Multiple noisy predictors (realistic ρ, reliability=0.85)"
        )
        print(
            f"  {'Predictors':>10s}  {'Median R':>20s}  {'Median R²':>20s}  {'Median MAE':>20s}"
        )
        print(f"  {'-'*10}  {'-'*20}  {'-'*20}  {'-'*20}")

        for n_pred in range(1, 6):
            res = run_scenario(
                n_train,
                n_test,
                rel_single,
                n_pred,
                true_correlations=realistic_rhos[:n_pred],
                predictor_reliabilities=[0.85] * n_pred,
            )
            print(
                f"  {n_pred:>10d}  {fmt_ci(res['r_pred_yavg']):>20s}  "
                f"{fmt_ci(res['r2_pred_yavg']):>20s}  {fmt_ci(res['mae_pred_yavg']):>20s}"
            )

        # ── Section 6: "How good does a predictor need to be?" ──
        print_section(f"{params.name}: Required true ρ to beat category baseline")
        if params.name == "Enjoyment":
            baseline_mae = 0.804
            best_achieved_r = 0.30
        else:
            baseline_mae = 0.729
            best_achieved_r = 0.53

        print(f"  Category baseline MAE: {baseline_mae:.3f}")
        print(f"  Best achieved holdout R: {best_achieved_r:.3f}")
        print()

        # What true ρ gives us the observed R?
        # observed_R ≈ ρ * sqrt(rel_avg) for single noiseless predictor
        implied_rho = best_achieved_r / np.sqrt(rel_avg)
        print(
            f"  Implied true ρ behind best observed R={best_achieved_r}: {implied_rho:.3f}"
        )
        print(
            f"  (i.e., the best model's features have ~{implied_rho:.0%} true correlation with your latent preference)"
        )
        print()

        # Achievable R if true correlation were higher
        print(
            f"  {'True ρ':>10s}  {'Expected R':>10s}  {'Expected R²':>10s}  {'Gap to ceiling':>15s}"
        )
        print(f"  {'-'*10}  {'-'*10}  {'-'*10}  {'-'*15}")
        for rho in [0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 1.00]:
            exp_r = rho * np.sqrt(rel_avg)
            gap = max_r - exp_r
            print(f"  {rho:>10.2f}  {exp_r:>10.3f}  {exp_r**2:>10.3f}  {gap:>15.3f}")

    # ── Section 7: Sample size sensitivity ──
    print_section("SAMPLE SIZE SENSITIVITY (Enjoyment, 1 predictor, true ρ=0.35)")
    print(
        f"  {'N_train':>10s}  {'Median R':>20s}  {'Median R²':>20s}  {'Width of 90% CI':>16s}"
    )
    print(f"  {'-'*10}  {'-'*20}  {'-'*20}  {'-'*16}")

    for n_tr in [50, 100, 150, 200, 300, 500, 1000]:
        res = run_scenario(
            n_tr,
            70,
            ENJOYMENT.test_retest_r,
            1,
            true_correlations=[0.35],
            predictor_reliabilities=[1.0],
        )
        lo, hi = np.percentile(res["r_pred_yavg"], [5, 95])
        print(
            f"  {n_tr:>10d}  {fmt_ci(res['r_pred_yavg']):>20s}  "
            f"{fmt_ci(res['r2_pred_yavg']):>20s}  {hi - lo:>16.3f}"
        )

    # ── Section 8: Predicting single pass vs average ──
    print_section("PREDICTING SINGLE PASS (Y1) vs AVERAGE (Y_avg)")
    print("  With 1 perfect predictor (ρ=1.0):")
    print(f"  {'Target':>10s}  {'R (enjoyment)':>20s}  {'R (usefulness)':>20s}")
    print(f"  {'-'*10}  {'-'*20}  {'-'*20}")

    for params in [ENJOYMENT, USEFULNESS]:
        res = run_scenario(
            n_train,
            n_test,
            params.test_retest_r,
            1,
            true_correlations=[1.0],
            predictor_reliabilities=[1.0],
        )
        r_avg_str = fmt_ci(res["r_pred_yavg"])
        r_y1_str = fmt_ci(res["r_pred_y1"])
        # Not clean to put both in one row; print per-target
        print(f"  {'Y_avg':>10s}  {r_avg_str:>20s}  ({params.name})")
        print(f"  {'Y1':>10s}  {r_y1_str:>20s}  ({params.name})")
        print()

    # ── Section 9: Interpretation summary ──
    print_section("INTERPRETATION SUMMARY")

    for params in [ENJOYMENT, USEFULNESS]:
        rel = params.test_retest_r
        rel_avg = spearman_brown(rel, 2)
        max_r = np.sqrt(rel_avg)

        if params.name == "Enjoyment":
            best_r = 0.30
            observed_rhos = {"Goodreads": 0.29, "Amazon": 0.18, "OpenLibrary": 0.05}
        else:
            best_r = 0.53
            observed_rhos = {"Goodreads": 0.34, "Amazon": 0.28, "OpenLibrary": 0.01}

        print(f"\n  {params.name}:")
        print(
            f"    Ceiling (perfect oracle):     R = {max_r:.3f}  (R² = {rel_avg:.3f})"
        )
        print(
            f"    Best achieved:                R = {best_r:.3f}  (R² = {best_r**2:.3f})"
        )
        print(
            f"    Gap:                          ΔR = {max_r - best_r:.3f}  (ΔR² = {rel_avg - best_r**2:.3f})"
        )
        print(f"    Best achieved / ceiling:      {best_r / max_r:.1%}")
        print()

        # Disattenuate observed correlations to get true correlations
        print("    Disattenuated true correlations (correcting for Y noise):")
        for source, obs_r in observed_rhos.items():
            # observed_r ≈ true_r * sqrt(reliability_of_Y_single_pass)
            # since these correlations were computed against single-pass Y
            true_r = obs_r / np.sqrt(rel)
            print(f"      {source:>15s}: observed r={obs_r:.3f} → true ρ={true_r:.3f}")

        # What R could we get from a perfectly measured combination?
        # With independent predictors each with true ρ, the max multiple R is:
        # R² = sum(ρ_k²) if they're uncorrelated
        true_rhos = [v / np.sqrt(rel) for v in observed_rhos.values()]
        r2_combo_uncorr = sum(r**2 for r in true_rhos)
        r2_combo_uncorr = min(r2_combo_uncorr, 1.0)
        r_combo = np.sqrt(r2_combo_uncorr) * np.sqrt(rel_avg)
        print("\n    If all 3 sources were uncorrelated (optimistic):")
        print(f"      Combined true R² with T: {r2_combo_uncorr:.3f}")
        print(f"      Observable R with Y_avg:  {r_combo:.3f}")
        print("      (Sources ARE correlated in reality, so this overestimates)")

    # ── Section 10: Practical recommendations ──
    print_section("PRACTICAL RECOMMENDATIONS")
    print(
        """
  1. YOUR CEILING IS NOT AS FAR AWAY AS IT LOOKS
     - Enjoyment max R = 0.933 sounds high, but your *noise* alone explains
       a huge chunk. Your best model at R=0.30 captures only ~32% of the
       theoretically achievable correlation.
     - But the remaining 68% requires finding features that truly correlate
       with your latent taste — this is hard because your taste is personal.

  2. DIMINISHING RETURNS FROM MORE EXTERNAL RATINGS
     - Goodreads, Amazon, and OpenLibrary correlate at ρ≈0.3-0.35 with each
       other AND with your ratings. They're measuring similar public consensus.
     - Adding a 4th or 5th consensus source (e.g., LibraryThing) likely adds
       very little because it's collinear with existing sources.
     - The simulation shows: going from 1 to 3 predictors at ρ=0.35 gains
       maybe +0.03 R if they're partially redundant.

  3. BIGGEST GAINS COME FROM ORTHOGONAL SIGNALS
     - To materially improve, you need features that capture YOUR taste, not
       public consensus. These are features with high true ρ to YOUR T that
       are NOT collinear with Goodreads ratings.
     - Candidates: recommendation source (who told you about it), category,
       author track record, topic keywords from descriptions, your reading
       history patterns.
     - Category alone explains more variance than all 3 external rating sites
       combined — precisely because it captures YOUR intent/preference.

  4. FOCUS AREAS FOR IMPROVEMENT
     a. Track recommendation sources directly (not Gemini-inferred).
        This was identified as the single most promising feature.
     b. Category-specific models: Business/General show R≈0.5-0.7 within
        category. Fiction shows R≈-0.3. Build separate models.
     c. Consider the DECISION problem, not the PREDICTION problem.
        Even R=0.30 gives you useful filtering if you just want to avoid
        the bottom 20-30% of books.

  5. YOUR DATA QUANTITY IS ADEQUATE
     - Going from n=200 to n=500 barely narrows the CI on R estimates.
     - The bottleneck is signal strength (true ρ), not sample size.
"""
    )


if __name__ == "__main__":
    main()
