from __future__ import annotations

import json
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))


def best_average_threshold_on_10(cv_payload: dict[str, object]) -> float:
    classification = pd.DataFrame(cv_payload["classification_summary"])
    average_rows = classification.loc[
        classification["model"] == "average_threshold"
    ].sort_values("accuracy_mean", ascending=False)
    if average_rows.empty:
        return 7.0
    return float(average_rows.iloc[0]["threshold"]) / 10.0


def main() -> None:
    from analysis_core.movie_rt_second_pass_analysis import (
        LOCAL_TZ,
        apply_saved_legacy_models,
        build_cleanup_delta_table,
        build_cleanup_overlap,
        build_gain_curve,
        build_loocv_model_tables,
        build_residual_inspection_table,
        build_synopsis_dataset,
        build_time_window_metrics,
        compute_bias_summary,
        compute_group_effects,
        compute_repeat_watch_consistency,
        evaluate_legacy_predictions,
        extract_linear_pipeline_coefficients,
        fit_full_legacy_models,
        fit_full_cv_ridge_context_model,
        fit_residual_context_model,
        plot_ridge_coefficients,
        load_legacy_dataset,
        load_second_sheet_dataset,
        build_score_cutoff_tradeoff,
        plot_cleanup_comparison,
        plot_gain_curves,
        plot_period_residuals,
        plot_residuals_over_time,
        plot_score_cutoff_tradeoff,
        summarize_second_pass,
    )

    data_dir = ROOT / "data" / "summaries"
    movie_rt_dir = data_dir / "movie_rt_analysis"
    output_dir = movie_rt_dir / "second_pass_20260404"
    output_dir.mkdir(parents=True, exist_ok=True)

    legacy_path = movie_rt_dir / "movie_rt_scores_detailed.csv"
    raw_second_path = (
        data_dir / "Movie Ratings - new_calendar_movies_features-labeled.csv"
    )
    clean_second_path = (
        data_dir
        / "Movie Ratings - new_calendar_movies_features-labeled-verified-qc.csv"
    )
    calendar_dir = ROOT.parents[1] / "data" / "Takeout 5" / "Calendar"
    as_of_local = datetime(2026, 4, 4, 23, 59, 59, tzinfo=LOCAL_TZ)

    legacy_df = load_legacy_dataset(legacy_path)
    raw_second_df = load_second_sheet_dataset(
        raw_second_path,
        source_name="second_raw",
        include_unfinished_as_two=False,
        calendar_dir=calendar_dir,
        as_of_local=as_of_local,
    )
    clean_second_df = load_second_sheet_dataset(
        clean_second_path,
        source_name="second_clean",
        include_unfinished_as_two=True,
        calendar_dir=calendar_dir,
        as_of_local=as_of_local,
    )
    analysis_ready_path = (
        data_dir
        / "Movie Ratings - new_calendar_movies_features-labeled-verified-qc-unfinished-2.csv"
    )
    clean_second_df.to_csv(analysis_ready_path, index=False)

    metrics_payload = json.loads((movie_rt_dir / "movie_rt_metrics.json").read_text())
    cv_payload = json.loads((movie_rt_dir / "movie_rt_cv_metrics.json").read_text())
    legacy_threshold = best_average_threshold_on_10(cv_payload)
    full_legacy_models = fit_full_legacy_models(legacy_df)

    overlap_keys = set(raw_second_df["analysis_row_key"]) & set(
        clean_second_df["analysis_row_key"]
    )
    raw_overlap = raw_second_df.loc[
        raw_second_df["analysis_row_key"].isin(overlap_keys)
    ].copy()
    clean_overlap = clean_second_df.loc[
        clean_second_df["analysis_row_key"].isin(overlap_keys)
    ].copy()

    raw_overlap_predictions = apply_saved_legacy_models(
        raw_overlap,
        metrics_payload=metrics_payload,
        full_legacy_models=full_legacy_models,
        average_threshold=legacy_threshold,
    )
    clean_overlap_predictions = apply_saved_legacy_models(
        clean_overlap,
        metrics_payload=metrics_payload,
        full_legacy_models=full_legacy_models,
        average_threshold=legacy_threshold,
    )
    clean_full_predictions = apply_saved_legacy_models(
        clean_second_df,
        metrics_payload=metrics_payload,
        full_legacy_models=full_legacy_models,
        average_threshold=legacy_threshold,
    )

    legacy_metrics = pd.concat(
        [
            evaluate_legacy_predictions(
                raw_overlap_predictions, dataset_label="raw_overlap"
            ),
            evaluate_legacy_predictions(
                clean_overlap_predictions,
                dataset_label="clean_overlap",
            ),
            evaluate_legacy_predictions(
                clean_full_predictions,
                dataset_label="clean_full_with_unfinished2",
            ),
        ],
        ignore_index=True,
    )

    second_cv_metrics, second_cv_predictions = build_loocv_model_tables(
        clean_full_predictions,
        dataset_label="clean_full_with_unfinished2",
        include_context=True,
    )
    combined_df = pd.concat([legacy_df, clean_second_df], ignore_index=True, sort=False)
    combined_cv_metrics, combined_cv_predictions = build_loocv_model_tables(
        combined_df,
        dataset_label="combined_all",
        include_context=True,
    )

    cleanup_overlap = build_cleanup_overlap(
        raw_overlap_predictions, clean_overlap_predictions
    )
    cleanup_deltas = build_cleanup_delta_table(
        raw_overlap_predictions,
        clean_overlap_predictions,
    )

    gain_curves = pd.concat(
        [
            build_gain_curve(
                second_cv_predictions,
                prediction_column="legacy_average_prediction",
                model_name="legacy_average_rt",
                dataset_label="clean_full_with_unfinished2",
            ),
            build_gain_curve(
                second_cv_predictions,
                prediction_column="legacy_saved_linear_prediction",
                model_name="legacy_saved_linear",
                dataset_label="clean_full_with_unfinished2",
            ),
            build_gain_curve(
                second_cv_predictions,
                prediction_column="cv_linear_rt_prediction",
                model_name="cv_linear_rt",
                dataset_label="clean_full_with_unfinished2",
            ),
            build_gain_curve(
                second_cv_predictions,
                prediction_column="cv_linear_rt_imdb_prediction",
                model_name="cv_linear_rt_imdb",
                dataset_label="clean_full_with_unfinished2",
            ),
            build_gain_curve(
                second_cv_predictions,
                prediction_column="cv_ridge_context_prediction",
                model_name="cv_ridge_context",
                dataset_label="clean_full_with_unfinished2",
            ),
        ],
        ignore_index=True,
    )

    time_window_metrics = pd.concat(
        [
            build_time_window_metrics(
                second_cv_predictions,
                prediction_column="legacy_saved_linear_prediction",
                dataset_label="clean_full_with_unfinished2",
            ),
            build_time_window_metrics(
                second_cv_predictions,
                prediction_column="cv_linear_rt_imdb_prediction",
                dataset_label="clean_full_with_unfinished2",
            ),
            build_time_window_metrics(
                second_cv_predictions,
                prediction_column="cv_ridge_context_prediction",
                dataset_label="clean_full_with_unfinished2",
            ),
        ],
        ignore_index=True,
    )

    group_effects = compute_group_effects(second_cv_predictions)
    residual_context_model = fit_residual_context_model(second_cv_predictions)
    repeat_watch_consistency = compute_repeat_watch_consistency(second_cv_predictions)
    bias_summary = compute_bias_summary(second_cv_predictions)
    full_ridge_context = fit_full_cv_ridge_context_model(clean_full_predictions)
    ridge_context_coefficients = extract_linear_pipeline_coefficients(
        full_ridge_context,
        numeric_features=[
            "rt_critic_on_10",
            "rt_audience_on_10",
            "imdb_score",
            "start_hour",
            "month_index",
        ],
        binary_features=[
            "saw_in_theater",
            "drink_before_movie",
            "assigned_unfinished_two",
            "likely_with_amelia",
            "is_weekend",
        ],
        categorical_features=["life_period", "pressure_tier", "data_source"],
    )
    rt_cutoff_tradeoff = build_score_cutoff_tradeoff(
        second_cv_predictions,
        score_column="rt_average_rating",
        score_label="rt_average_rating",
    )
    residual_inspection = build_residual_inspection_table(
        second_cv_predictions,
        prediction_columns=[
            "legacy_saved_linear_prediction",
            "cv_linear_rt_imdb_prediction",
            "cv_ridge_context_prediction",
        ],
    )

    synopsis_cache_path = output_dir / "synopsis_cache.json"
    synopsis_dataset = build_synopsis_dataset(
        clean_second_df,
        cache_path=synopsis_cache_path,
    )
    synopsis_train = synopsis_dataset.loc[
        synopsis_dataset["random_half_split"] == "train_half"
    ].copy()
    synopsis_holdout = synopsis_dataset.loc[
        synopsis_dataset["random_half_split"] == "holdout_half"
    ].copy()

    source_year_gap_count = int(
        clean_second_df["quality_flags"]
        .fillna("")
        .str.contains("source_year_gap")
        .sum()
    )
    summary_text = summarize_second_pass(
        rated_row_count=int(len(clean_second_df)),
        legacy_metrics=legacy_metrics,
        second_cv_metrics=second_cv_metrics,
        combined_cv_metrics=combined_cv_metrics,
        cleanup_metrics=legacy_metrics.loc[
            legacy_metrics["dataset"].isin(["raw_overlap", "clean_overlap"])
        ],
        bias_summary=bias_summary,
        source_year_gap_count=source_year_gap_count,
    )

    legacy_metrics.to_csv(output_dir / "legacy_on_second_metrics.csv", index=False)
    raw_overlap_predictions.to_csv(
        output_dir / "raw_overlap_predictions.csv", index=False
    )
    clean_overlap_predictions.to_csv(
        output_dir / "clean_overlap_predictions.csv",
        index=False,
    )
    clean_full_predictions.to_csv(
        output_dir / "clean_full_predictions.csv",
        index=False,
    )
    second_cv_metrics.to_csv(output_dir / "second_sheet_cv_metrics.csv", index=False)
    second_cv_predictions.to_csv(
        output_dir / "second_sheet_cv_predictions.csv",
        index=False,
    )
    combined_cv_metrics.to_csv(output_dir / "combined_cv_metrics.csv", index=False)
    combined_cv_predictions.to_csv(
        output_dir / "combined_cv_predictions.csv",
        index=False,
    )
    cleanup_overlap.to_csv(output_dir / "cleanup_overlap_values.csv", index=False)
    cleanup_deltas.to_csv(output_dir / "cleanup_prediction_deltas.csv", index=False)
    gain_curves.to_csv(output_dir / "gain_curves.csv", index=False)
    time_window_metrics.to_csv(output_dir / "time_window_metrics.csv", index=False)
    group_effects.to_csv(output_dir / "context_group_effects.csv", index=False)
    residual_context_model.to_csv(
        output_dir / "residual_context_model.csv", index=False
    )
    repeat_watch_consistency.to_csv(
        output_dir / "repeat_watch_consistency.csv",
        index=False,
    )
    bias_summary.to_csv(output_dir / "bias_summary.csv", index=False)
    ridge_context_coefficients.to_csv(
        output_dir / "cv_ridge_context_full_coefficients.csv",
        index=False,
    )
    rt_cutoff_tradeoff.to_csv(output_dir / "rt_cutoff_tradeoff.csv", index=False)
    residual_inspection.to_csv(output_dir / "residual_inspection.csv", index=False)
    synopsis_dataset.to_csv(output_dir / "synopsis_dataset.csv", index=False)
    synopsis_train.to_csv(output_dir / "synopsis_train_half.csv", index=False)
    synopsis_holdout.to_csv(output_dir / "synopsis_holdout_half.csv", index=False)
    (output_dir / "summary.txt").write_text(summary_text + "\n")

    plot_gain_curves(gain_curves, output_dir / "gain_curves.png")
    plot_residuals_over_time(
        second_cv_predictions,
        prediction_columns=[
            "legacy_saved_linear_prediction",
            "cv_linear_rt_imdb_prediction",
            "cv_ridge_context_prediction",
        ],
        output_path=output_dir / "residuals_over_time.png",
    )
    plot_cleanup_comparison(
        legacy_metrics.loc[
            legacy_metrics["dataset"].isin(["raw_overlap", "clean_overlap"])
        ],
        output_dir / "cleanup_mae_comparison.png",
    )
    plot_period_residuals(group_effects, output_dir / "period_residuals.png")
    plot_score_cutoff_tradeoff(
        rt_cutoff_tradeoff,
        output_dir / "rt_cutoff_tradeoff.png",
    )
    plot_ridge_coefficients(
        ridge_context_coefficients,
        output_dir / "cv_ridge_context_coefficients.png",
    )

    print(summary_text)
    print()
    print(f"Analysis-ready cleaned sheet: {analysis_ready_path}")
    print(f"Output directory: {output_dir}")
    print(f"Synopsis dataset: {output_dir / 'synopsis_dataset.csv'}")


if __name__ == "__main__":
    main()
