"""Tests for llm_enhanced_drop_gain_analysis.py.

Verifies key correctness properties: matching logic, blend computation,
gain_at_drop behavior, and resampling diminishing-returns shape.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

# Add the script directory to path so we can import the module
SCRIPT_DIR = (
    Path(__file__).resolve().parent.parent / "ai_books_tracking" / "scripts" / "temp"
)
sys.path.insert(0, str(SCRIPT_DIR))

from llm_enhanced_drop_gain_analysis import (  # noqa: E402
    _build_holdout_to_wide,
    _match_to_wide_keys,
    _title_norm,
    compute_combined_pred,
    eval_predictor,
    sweep_blend_alpha,
)
from comprehensive_drop_gain_analysis import (  # noqa: E402
    gain_at_drop,
    W_ENJOY,
    W_USEFUL,
)


class TestTitleNorm:
    def test_strips_punctuation_and_lowercases(self) -> None:
        assert _title_norm("The Day of the Jackal!") == "thedayofthejackal"

    def test_handles_numbers(self) -> None:
        assert _title_norm("12 Rules for Life") == "12rulesforlife"

    def test_empty_string(self) -> None:
        assert _title_norm("") == ""


class TestMatchToWideKeys:
    def _make_wide(self, titles: list[str]) -> pd.DataFrame:
        keys = [t.lower() + " || author" for t in titles]
        return pd.DataFrame({"display_title": titles, "key": keys})

    def test_exact_match(self) -> None:
        wide = self._make_wide(["The Power Broker", "Dark Sun"])
        titles = pd.Series(["The Power Broker", "Dark Sun"])
        result = _match_to_wide_keys(titles, wide)
        assert len(result) == 2
        assert result[0] == "the power broker || author"
        assert result[1] == "dark sun || author"

    def test_substring_match(self) -> None:
        wide = self._make_wide(["Hyperion (Hyperion Cantos, #1)", "fall of Hyperion"])
        # The long filename-based title should match the shorter display title
        titles = pd.Series(["Hyperion (Hyperion Cantos, #1) extra stuff here"])
        result = _match_to_wide_keys(titles, wide)
        assert len(result) == 1
        assert "hyperion" in result[0].lower()

    def test_no_false_positive_short_titles(self) -> None:
        wide = self._make_wide(["ABC", "DEF"])
        titles = pd.Series(["GHI"])
        result = _match_to_wide_keys(titles, wide)
        assert len(result) == 0


class TestBuildHoldoutToWide:
    def test_all_68_holdout_books_match(self) -> None:
        """The actual data should produce 68/68 matches."""
        from llm_enhanced_drop_gain_analysis import (
            WIDE_CSV,
            load_and_prepare,
        )

        df = load_and_prepare()
        holdout = (
            df[df["source"].eq("Holdout 2026") & df["combined_target"].notna()]
            .copy()
            .reset_index(drop=True)
        )
        wide = pd.read_csv(WIDE_CSV)
        h2w = _build_holdout_to_wide(holdout, wide)
        assert len(h2w) == len(
            holdout
        ), f"Expected {len(holdout)} matches, got {len(h2w)}"

    def test_no_duplicate_wide_assignments(self) -> None:
        """Each wide row should be assigned to at most one holdout row."""
        from llm_enhanced_drop_gain_analysis import (
            WIDE_CSV,
            load_and_prepare,
        )

        df = load_and_prepare()
        holdout = (
            df[df["source"].eq("Holdout 2026") & df["combined_target"].notna()]
            .copy()
            .reset_index(drop=True)
        )
        wide = pd.read_csv(WIDE_CSV)
        h2w = _build_holdout_to_wide(holdout, wide)
        wide_indices = list(h2w.values())
        assert len(wide_indices) == len(set(wide_indices)), "Duplicate wide assignments"


class TestComputeCombinedPred:
    def test_weights_match_specification(self) -> None:
        enjoy = np.array([5.0, 1.0])
        useful = np.array([1.0, 5.0])
        result = compute_combined_pred(enjoy, useful)
        expected = W_ENJOY * enjoy + W_USEFUL * useful
        np.testing.assert_allclose(result, expected)

    def test_equal_inputs_give_same_output(self) -> None:
        arr = np.array([3.0, 3.0, 3.0])
        result = compute_combined_pred(arr, arr)
        np.testing.assert_allclose(result, arr)


class TestSweepBlendAlpha:
    def test_good_predictor_gets_high_alpha(self) -> None:
        """If pred_a is well-correlated and pred_b is noise, alpha should be high."""
        rng = np.random.default_rng(99)
        actual = np.arange(20, dtype=float)
        pred_a = actual + rng.standard_normal(20) * 0.3
        pred_b = rng.standard_normal(20) * 5  # pure noise
        alpha, rho = sweep_blend_alpha(pred_a, pred_b, actual)
        assert alpha >= 0.7
        assert rho > 0.8

    def test_noise_predictor_gets_alpha_0(self) -> None:
        """If pred_b is perfectly correlated and pred_a is noise, alpha ~ 0."""
        rng = np.random.default_rng(42)
        actual = np.arange(20, dtype=float)
        pred_b = actual.copy()
        pred_a = rng.standard_normal(20)
        alpha, rho = sweep_blend_alpha(pred_a, pred_b, actual)
        assert alpha <= 0.15  # should be close to 0


class TestGainAtDropCorrectness:
    def test_gain_equals_kept_mean_minus_overall_mean(self) -> None:
        """gain_at_drop should return mean(top k) - mean(all)."""
        pred = np.array([10.0, 8.0, 6.0, 4.0, 2.0])
        actual = np.array([5.0, 4.0, 3.0, 2.0, 1.0])
        gain, n_kept, n_total = gain_at_drop(pred, actual, 60)
        assert n_kept == 2
        assert n_total == 5
        expected_gain = np.mean([5.0, 4.0]) - np.mean(actual)
        assert abs(gain - expected_gain) < 1e-10

    def test_zero_drop_gives_zero_gain(self) -> None:
        pred = np.array([1.0, 2.0, 3.0])
        actual = np.array([1.0, 2.0, 3.0])
        gain, n_kept, _ = gain_at_drop(pred, actual, 0)
        assert n_kept == 3
        assert abs(gain) < 1e-10

    def test_bad_predictor_gives_negative_gain(self) -> None:
        """Reverse-ordering should give negative gain at high drop rates."""
        pred = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        actual = np.array([5.0, 4.0, 3.0, 2.0, 1.0])
        gain, _, _ = gain_at_drop(pred, actual, 60)
        assert gain < 0


class TestEvalPredictor:
    def test_returns_rows_for_all_drop_pcts(self) -> None:
        rng = np.random.default_rng(42)
        pred = np.arange(20, dtype=float)
        actual = pred + rng.standard_normal(20) * 0.5
        groups = np.array(["Business_Histories_General"] * 20)
        rows = eval_predictor(pred, actual, groups, rng, "test_model")
        drop_pcts_seen = {r["drop_pct"] for r in rows}
        assert 30 in drop_pcts_seen
        assert 60 in drop_pcts_seen
        assert 80 in drop_pcts_seen

    def test_flags_small_n_kept(self) -> None:
        rng = np.random.default_rng(42)
        pred = np.arange(5, dtype=float)
        actual = pred.copy()
        groups = np.array(["Technical_Other"] * 5)
        rows = eval_predictor(pred, actual, groups, rng, "test_small")
        # At 80% drop on 5 books, n_kept = 1 < 10
        for r in rows:
            if r["drop_pct"] == 80 and r["group"] == "Overall":
                assert r["flag"] == "n<10"


class TestResamplingDiminishingReturns:
    def test_more_samples_reduces_variance(self) -> None:
        """Averaging more runs should reduce the CI width."""
        rng = np.random.default_rng(42)
        n_books = 30
        true_quality = rng.standard_normal(n_books)
        # Simulate 10 noisy runs
        samples = np.column_stack(
            [true_quality + rng.standard_normal(n_books) * 0.5 for _ in range(10)]
        )

        widths = []
        for n_runs in [1, 5, 10]:
            rhos = []
            for _ in range(100):
                idx = rng.integers(0, 10, size=n_runs)
                avg = samples[:, idx].mean(axis=1)
                from scipy.stats import spearmanr

                rhos.append(spearmanr(avg, true_quality).statistic)
            rho_arr = np.array(rhos)
            width = np.percentile(rho_arr, 95) - np.percentile(rho_arr, 5)
            widths.append(width)

        # Width should decrease with more samples
        assert widths[0] > widths[1], "1-run should have wider CI than 5-run"
        assert widths[1] > widths[2], "5-run should have wider CI than 10-run"


class TestOutputCSVsExist:
    """After running main(), the output CSVs should exist and have expected structure."""

    @pytest.fixture(scope="class")
    def output_dir(self) -> Path:
        return (
            Path(__file__).resolve().parent.parent / "ai_books_tracking" / "ai_actions"
        )

    def test_blends_csv_has_required_columns(self, output_dir: Path) -> None:
        csv_path = output_dir / "llm_enhanced_drop_gain_blends.csv"
        if not csv_path.exists():
            pytest.skip("Run main() first to generate output CSVs")
        df = pd.read_csv(csv_path)
        required = {
            "group",
            "model",
            "drop_pct",
            "gain",
            "boot_p5",
            "boot_p95",
            "n_kept",
        }
        assert required.issubset(set(df.columns))

    def test_resampling_csv_has_all_run_counts(self, output_dir: Path) -> None:
        csv_path = output_dir / "llm_enhanced_drop_gain_resampling.csv"
        if not csv_path.exists():
            pytest.skip("Run main() first to generate output CSVs")
        df = pd.read_csv(csv_path)
        combined = df[(df["metric"] == "rho") & (df["scope"] == "combined")]
        assert set(range(1, 11)).issubset(set(combined["n_runs"]))

    def test_sources_csv_includes_68book_models(self, output_dir: Path) -> None:
        csv_path = output_dir / "llm_enhanced_drop_gain_sources.csv"
        if not csv_path.exists():
            pytest.skip("Run main() first to generate output CSVs")
        df = pd.read_csv(csv_path)
        models_68 = df[df["model"].str.contains("_68")]
        assert len(models_68) > 0, "Should have models evaluated on all 68 books"
