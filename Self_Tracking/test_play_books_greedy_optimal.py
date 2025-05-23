import pytest
import numpy as np
from play_books_greedy_optimal import (
    utility_value,
    inverse_utility_value,
    error_sigma,
    error_sigma2,
    simulate_estimates,
    optimise_schedule_greedy,
    avg_hours_reading,
    current_hourly_u,
    F_GRID,
    READ_TIME_HOURS,
    SEARCH_COST_HOURS,
)
import matplotlib.pyplot as plt


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


# Test utility value transformations
def test_utility_value_transformations():
    # Test basic utility value calculations
    ratings = np.array([1, 2, 3, 4, 5])
    utils = utility_value(ratings)

    # Utility should be increasing
    assert np.all(np.diff(utils) > 0)

    # Test inverse transformation
    recovered_ratings = inverse_utility_value(utils)
    np.testing.assert_allclose(ratings, recovered_ratings, rtol=1e-5)

    # Test edge cases
    assert utility_value(1) == 0  # Base case
    assert utility_value(5) > utility_value(4)  # Monotonicity


# Test error sigma function
def test_error_sigma():
    # Test error decreases as reading progress increases
    f_values = np.array([0.0, 0.25, 0.5, 0.75, 1.0])
    errors = np.array([error_sigma(f) for f in f_values])

    # Error should decrease monotonically
    assert np.all(np.diff(errors) <= 0)

    # Error should be positive
    assert np.all(errors > 0)

    # Error at f=1 should be small but positive
    assert 0 < error_sigma(1.0) < 1


# Test simulate_estimates
def test_simulate_estimates():
    # Test with fixed random seed for reproducibility
    np.random.seed(42)

    # Test with a single book
    true_rating = 4.0
    estimates = simulate_estimates(np.array([true_rating]))

    # Check shape
    assert estimates.shape == (1, len(F_GRID))

    # Estimates should be within reasonable bounds
    assert np.all(estimates >= 1) and np.all(estimates <= 5)

    # Test with multiple books
    true_ratings = np.array([2.0, 3.0, 4.0, 5.0])
    estimates = simulate_estimates(true_ratings)

    # Check shape
    assert estimates.shape == (4, len(F_GRID))

    # Check bounds
    assert np.all(estimates >= 1) and np.all(estimates <= 5)


# Test optimise_schedule_greedy
def test_optimise_schedule_greedy():
    # Test with a simple case
    np.random.seed(42)
    true_ratings = np.array([2.0, 3.0, 4.0, 5.0])
    result = optimise_schedule_greedy(true_ratings)

    # Check return structure
    assert "cur_drop" in result
    assert "cutoffs" in result
    assert "true_avg_utils" in result

    # Check shapes
    assert len(result["cur_drop"]) == len(F_GRID) - 1  # Excluding f=1
    assert len(result["cutoffs"]) == len(F_GRID) - 1
    assert len(result["true_avg_utils"]) == len(F_GRID) - 1

    # Check bounds
    assert np.all(result["cur_drop"] >= 0) and np.all(result["cur_drop"] <= 1)
    assert np.all(result["cutoffs"] >= 1) and np.all(result["cutoffs"] <= 5)

    # Test with all same ratings
    same_ratings = np.array([3.0, 3.0, 3.0, 3.0])
    result_same = optimise_schedule_greedy(same_ratings)

    # Should either keep all or drop all at each step
    assert np.all(np.isin(result_same["cur_drop"], [0, 1]))


# Test avg_hours_reading
def test_avg_hours_reading():
    # Test with no drops
    no_drops = np.zeros(len(F_GRID) - 1)
    hours = avg_hours_reading(no_drops)
    assert np.isclose(hours, READ_TIME_HOURS + SEARCH_COST_HOURS)

    # Test with all drops at start
    all_drops_start = np.ones(len(F_GRID) - 1)
    all_drops_start[1:] = 0
    hours = avg_hours_reading(all_drops_start)
    assert hours < READ_TIME_HOURS + SEARCH_COST_HOURS

    # Test with gradual drops
    gradual_drops = np.linspace(0, 1, len(F_GRID) - 1)
    hours = avg_hours_reading(gradual_drops)
    assert 0 < hours < READ_TIME_HOURS + SEARCH_COST_HOURS


# Test current_hourly_u
def test_current_hourly_u():
    # Create a simple test dataframe
    import pandas as pd

    test_df = pd.DataFrame({"Bookshelf": ["Test"] * 4, "Usefulness /5 to Me": [2.0, 3.0, 4.0, 5.0]})

    hourly_u = current_hourly_u(test_df, "Usefulness /5 to Me")

    # Should be positive
    assert hourly_u > 0

    # Test with different rating column
    hourly_u_enjoyment = current_hourly_u(test_df, "Enjoyment /5 to Me")
    assert hourly_u_enjoyment > 0


# Test edge cases and error handling
def test_edge_cases():
    # Test with empty ratings
    with pytest.raises(AssertionError):
        optimise_schedule_greedy(np.array([]))

    # Test with invalid ratings
    with pytest.raises(AssertionError):
        optimise_schedule_greedy(np.array([0.5, 5.5]))

    # Test with single book
    result = optimise_schedule_greedy(np.array([3.0]))
    assert len(result["cur_drop"]) == len(F_GRID) - 1


if __name__ == "__main__":
    pytest.main([__file__])
