from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))


def main() -> None:
    from analysis_core.movie_rt_followup_analysis import (
        bootstrap_threshold_uncertainty,
        build_direct_bucket_tables,
        build_gemini_comparison_table,
        build_provisional_full_imdb_dataset,
        build_verified_imdb_dataset,
        compare_thresholds_by_keep_rate,
        compare_gemini_rt_model_fits,
        compute_threshold_tradeoff,
        direct_bucket_accuracy,
        evaluate_bucket_models,
        evaluate_full_provisional_imdb_models,
        evaluate_imdb_rt_bucket_models,
        evaluate_imdb_rt_continuous_models,
        load_gemini_rating_tables,
        plot_audience_shape_comparison,
        plot_bucket_accuracy,
        plot_direct_bucket_accuracy,
        plot_gemini_deviation,
        plot_gated_combo_threshold_panels,
        plot_imdb_rt_correlation,
        plot_imdb_rt_model_bars,
        plot_threshold_gain_detailed,
        plot_threshold_practical_tradeoffs,
        plot_threshold_super_comparison,
        plot_threshold_tradeoff_curves,
        plot_threshold_uncertainty_panels,
        personal_bucket_5,
        plot_verified_predictors,
        summarize_conjunctive_threshold_rules,
        summarize_gemini_alignment,
        write_followup_summary,
        write_threshold_uncertainty_summary,
    )

    data_dir = ROOT / "data" / "summaries" / "movie_rt_analysis"
    output_dir = data_dir / "followup"
    output_dir.mkdir(parents=True, exist_ok=True)

    verified_df_path = data_dir / "movie_rt_scores_detailed.csv"
    provisional_imdb_path = data_dir / "movie_rt_imdb_scores_combined_gemin_est.csv"
    gemini_markdown_path = ROOT / "data" / "Gemini-Movie Ratings vs. Rotten Tomatoes.md"

    import pandas as pd

    verified_df = pd.read_csv(verified_df_path)
    gemini_tables = load_gemini_rating_tables(gemini_markdown_path)
    gemini_comparison = build_gemini_comparison_table(
        gemini_tables["approximate"],
        gemini_tables["corrected"],
        verified_df,
    )
    gemini_comparison_path = output_dir / "gemini_vs_verified_comparison.csv"
    gemini_comparison.to_csv(gemini_comparison_path, index=False)
    gemini_alignment_summary = summarize_gemini_alignment(gemini_comparison)
    gemini_alignment_summary_path = output_dir / "gemini_alignment_summary.csv"
    gemini_alignment_summary.to_csv(gemini_alignment_summary_path, index=False)
    gemini_rt_model_fits = compare_gemini_rt_model_fits(gemini_comparison)
    gemini_rt_model_fits_path = output_dir / "gemini_rt_model_fits.csv"
    gemini_rt_model_fits.to_csv(gemini_rt_model_fits_path, index=False)

    gemini_plot_path = output_dir / "gemini_vs_verified_scatter.png"
    gemini_metrics = plot_gemini_deviation(gemini_comparison, gemini_plot_path)

    predictor_plot_path = output_dir / "verified_predictors_vs_my_rating.png"
    predictor_metrics = plot_verified_predictors(verified_df, predictor_plot_path)
    audience_shape_plot_path = output_dir / "audience_shape_comparison.png"
    audience_shape_metrics = plot_audience_shape_comparison(
        verified_df,
        audience_shape_plot_path,
    )
    audience_shape_metrics_path = output_dir / "audience_shape_metrics.csv"
    audience_shape_metrics.to_csv(audience_shape_metrics_path, index=False)

    bucket_metrics = evaluate_bucket_models(verified_df)
    bucket_metrics_path = output_dir / "bucket_model_metrics.csv"
    bucket_metrics.to_csv(bucket_metrics_path, index=False)
    bucket_plot_path = output_dir / "bucket_model_accuracy.png"
    plot_bucket_accuracy(bucket_metrics, bucket_plot_path)

    direct_bucket_tables = build_direct_bucket_tables(verified_df)
    for name, table in direct_bucket_tables.items():
        table.to_csv(output_dir / f"{name}_confusion.csv")
    direct_bucket_metrics = direct_bucket_accuracy(direct_bucket_tables)
    direct_bucket_metrics_path = output_dir / "direct_bucket_accuracy.csv"
    direct_bucket_metrics.to_csv(direct_bucket_metrics_path, index=False)
    direct_bucket_plot_path = output_dir / "direct_bucket_accuracy.png"
    plot_direct_bucket_accuracy(direct_bucket_metrics, direct_bucket_plot_path)

    threshold_tables = {
        "Audience": compute_threshold_tradeoff(verified_df, "rt_audience_rating"),
        "Average": compute_threshold_tradeoff(verified_df, "rt_average_rating"),
    }
    for label, table in threshold_tables.items():
        table.to_csv(
            output_dir / f"{label.lower()}_threshold_tradeoff.csv", index=False
        )
    threshold_plot_path = output_dir / "threshold_tradeoff.png"
    plot_threshold_tradeoff_curves(threshold_tables, threshold_plot_path)
    baseline_mean_rating = float(verified_df["my_rating"].mean())
    baseline_liked_rate = float((verified_df["my_rating"] >= 7).mean())
    baseline_bucket5 = float(verified_df["my_rating"].map(personal_bucket_5).mean())

    audience_threshold_uncertainty = bootstrap_threshold_uncertainty(
        verified_df,
        "rt_audience_rating",
    )
    audience_threshold_uncertainty_path = (
        output_dir / "audience_threshold_uncertainty.csv"
    )
    audience_threshold_uncertainty.to_csv(
        audience_threshold_uncertainty_path,
        index=False,
    )
    audience_threshold_uncertainty_plot_path = (
        output_dir / "audience_threshold_uncertainty.png"
    )
    plot_threshold_uncertainty_panels(
        audience_threshold_uncertainty,
        audience_threshold_uncertainty_plot_path,
        score_label="RT audience",
        threshold_axis_label="Minimum RT audience score (%)",
        baseline_mean_rating=baseline_mean_rating,
        baseline_liked_rate=baseline_liked_rate,
        baseline_bucket5=baseline_bucket5,
    )
    audience_threshold_practical_plot_path = (
        output_dir / "audience_threshold_practical_tradeoff.png"
    )
    plot_threshold_practical_tradeoffs(
        audience_threshold_uncertainty,
        audience_threshold_practical_plot_path,
        score_label="RT audience",
        threshold_axis_label="RT audience threshold (%)",
        baseline_liked_rate=baseline_liked_rate,
    )
    audience_threshold_detailed_plot_path = (
        output_dir / "audience_threshold_detailed_curve.png"
    )
    plot_threshold_gain_detailed(
        audience_threshold_uncertainty,
        audience_threshold_detailed_plot_path,
        score_label="RT audience",
        threshold_axis_label="Minimum RT audience score (%)",
    )
    audience_threshold_summary_path = (
        output_dir / "audience_threshold_uncertainty_summary.txt"
    )
    write_threshold_uncertainty_summary(
        audience_threshold_summary_path,
        audience_threshold_uncertainty,
        score_label="RT audience",
    )

    verified_imdb_df = build_verified_imdb_dataset(
        gemini_tables["corrected"], verified_df
    )
    verified_imdb_path = output_dir / "verified_rt_imdb_dataset.csv"
    verified_imdb_df.to_csv(verified_imdb_path, index=False)
    imdb_plot_df = verified_imdb_df.copy()
    imdb_plot_df["imdb_on_10"] = imdb_plot_df["IMDb"] / 10.0
    imdb_plot_df["rt_audience_on_10"] = imdb_plot_df["rt_audience_rating"] / 10.0
    imdb_rt_correlation_plot_path = output_dir / "rt_imdb_correlation.png"
    imdb_rt_correlation = plot_imdb_rt_correlation(
        verified_imdb_df,
        imdb_rt_correlation_plot_path,
    )
    imdb_rt_continuous = evaluate_imdb_rt_continuous_models(verified_imdb_df)
    imdb_rt_continuous_path = output_dir / "rt_imdb_continuous_metrics.csv"
    imdb_rt_continuous.to_csv(imdb_rt_continuous_path, index=False)
    imdb_rt_bucket_3 = evaluate_imdb_rt_bucket_models(verified_imdb_df, scheme="3_step")
    imdb_rt_bucket_3_path = output_dir / "rt_imdb_3step_metrics.csv"
    imdb_rt_bucket_3.to_csv(imdb_rt_bucket_3_path, index=False)
    imdb_rt_bucket_5 = evaluate_imdb_rt_bucket_models(verified_imdb_df, scheme="5_step")
    imdb_rt_bucket_5_path = output_dir / "rt_imdb_5step_metrics.csv"
    imdb_rt_bucket_5.to_csv(imdb_rt_bucket_5_path, index=False)
    imdb_rt_model_plot_path = output_dir / "rt_imdb_model_comparison.png"
    plot_imdb_rt_model_bars(
        imdb_rt_continuous,
        imdb_rt_bucket_3,
        imdb_rt_bucket_5,
        imdb_rt_model_plot_path,
    )
    imdb_baseline_mean_rating = float(imdb_plot_df["my_rating"].mean())
    imdb_baseline_liked_rate = float((imdb_plot_df["my_rating"] >= 7).mean())
    imdb_baseline_bucket5 = float(
        imdb_plot_df["my_rating"].map(personal_bucket_5).mean()
    )
    imdb_threshold_uncertainty = bootstrap_threshold_uncertainty(
        imdb_plot_df,
        "imdb_on_10",
        thresholds=[5.0, 5.5, 6.0, 6.5, 7.0, 7.5, 8.0, 8.5, 9.0],
    )
    imdb_threshold_uncertainty_path = output_dir / "imdb_threshold_uncertainty.csv"
    imdb_threshold_uncertainty.to_csv(imdb_threshold_uncertainty_path, index=False)
    imdb_threshold_uncertainty_plot_path = output_dir / "imdb_threshold_uncertainty.png"
    plot_threshold_uncertainty_panels(
        imdb_threshold_uncertainty,
        imdb_threshold_uncertainty_plot_path,
        score_label="IMDb",
        threshold_axis_label="Minimum IMDb score (/10)",
        baseline_mean_rating=imdb_baseline_mean_rating,
        baseline_liked_rate=imdb_baseline_liked_rate,
        baseline_bucket5=imdb_baseline_bucket5,
    )
    imdb_threshold_practical_plot_path = (
        output_dir / "imdb_threshold_practical_tradeoff.png"
    )
    plot_threshold_practical_tradeoffs(
        imdb_threshold_uncertainty,
        imdb_threshold_practical_plot_path,
        score_label="IMDb",
        threshold_axis_label="IMDb threshold (/10)",
        baseline_liked_rate=imdb_baseline_liked_rate,
        practical_thresholds=(6.0, 6.5, 7.0, 7.5, 8.0, 8.5, 9.0),
    )
    imdb_threshold_detailed_plot_path = output_dir / "imdb_threshold_detailed_curve.png"
    plot_threshold_gain_detailed(
        imdb_threshold_uncertainty,
        imdb_threshold_detailed_plot_path,
        score_label="IMDb",
        threshold_axis_label="Minimum IMDb score (/10)",
        practical_thresholds=(6.0, 6.5, 7.0, 7.5, 8.0, 8.5, 9.0),
    )
    imdb_threshold_summary_path = output_dir / "imdb_threshold_uncertainty_summary.txt"
    write_threshold_uncertainty_summary(
        imdb_threshold_summary_path,
        imdb_threshold_uncertainty,
        score_label="IMDb",
        practical_thresholds=(6.0, 6.5, 7.0, 7.5, 8.0, 8.5, 9.0),
    )
    hybrid_rt_threshold_uncertainty = bootstrap_threshold_uncertainty(
        imdb_plot_df,
        "rt_audience_on_10",
        thresholds=[5.0, 5.5, 6.0, 6.5, 7.0, 7.5, 8.0, 8.5, 9.0],
    )
    hybrid_rt_threshold_uncertainty_path = (
        output_dir / "hybrid_rt_audience_threshold_uncertainty.csv"
    )
    hybrid_rt_threshold_uncertainty.to_csv(
        hybrid_rt_threshold_uncertainty_path,
        index=False,
    )
    hybrid_rt_threshold_detailed_plot_path = (
        output_dir / "hybrid_rt_audience_threshold_detailed_curve.png"
    )
    plot_threshold_gain_detailed(
        hybrid_rt_threshold_uncertainty,
        hybrid_rt_threshold_detailed_plot_path,
        score_label="RT audience (25-row subset)",
        threshold_axis_label="Minimum RT audience score (/10)",
        practical_thresholds=(6.0, 6.5, 7.0, 7.5, 8.0, 8.5, 9.0),
    )
    imdb_vs_rt_keep_path = output_dir / "imdb_vs_rt_matched_keep_comparison.csv"
    compare_thresholds_by_keep_rate(
        imdb_threshold_uncertainty,
        hybrid_rt_threshold_uncertainty,
        comparison_label="imdb",
        reference_label="rt_audience",
    ).to_csv(imdb_vs_rt_keep_path, index=False)
    super_plot_path = output_dir / "rt_vs_imdb_threshold_super_plot.png"
    plot_threshold_super_comparison(
        hybrid_rt_threshold_uncertainty,
        imdb_threshold_uncertainty,
        super_plot_path,
    )

    provisional_imdb_source = pd.read_csv(provisional_imdb_path)
    provisional_full_imdb_df = build_provisional_full_imdb_dataset(
        provisional_imdb_source,
        verified_df,
    )
    provisional_full_imdb_dataset_path = (
        output_dir / "provisional_full_rt_imdb_dataset.csv"
    )
    provisional_full_imdb_df.to_csv(provisional_full_imdb_dataset_path, index=False)

    provisional_quality = provisional_full_imdb_df.merge(
        verified_imdb_df[["movie_title", "IMDb"]],
        on="movie_title",
        how="inner",
    )
    provisional_quality["abs_error_vs_corrected25"] = (
        provisional_quality["IMDb Rating"] - provisional_quality["IMDb"]
    ).abs()
    provisional_quality_path = output_dir / "provisional_imdb_vs_corrected25.csv"
    provisional_quality.to_csv(provisional_quality_path, index=False)
    provisional_quality_summary = pd.DataFrame(
        [
            {
                "rows_compared": int(len(provisional_quality)),
                "mae": float(provisional_quality["abs_error_vs_corrected25"].mean()),
                "median_abs_error": float(
                    provisional_quality["abs_error_vs_corrected25"].median()
                ),
                "max_abs_error": float(
                    provisional_quality["abs_error_vs_corrected25"].max()
                ),
                "share_within_1": float(
                    (provisional_quality["abs_error_vs_corrected25"] <= 1).mean()
                ),
                "share_within_2": float(
                    (provisional_quality["abs_error_vs_corrected25"] <= 2).mean()
                ),
                "share_within_5": float(
                    (provisional_quality["abs_error_vs_corrected25"] <= 5).mean()
                ),
            }
        ]
    )
    provisional_quality_summary_path = (
        output_dir / "provisional_imdb_quality_summary.csv"
    )
    provisional_quality_summary.to_csv(provisional_quality_summary_path, index=False)

    provisional_continuous_metrics, provisional_bucket_metrics = (
        evaluate_full_provisional_imdb_models(provisional_full_imdb_df)
    )
    provisional_continuous_metrics_path = (
        output_dir / "provisional_full_rt_imdb_continuous_metrics.csv"
    )
    provisional_bucket_metrics_path = (
        output_dir / "provisional_full_rt_imdb_bucket_metrics.csv"
    )
    provisional_continuous_metrics.to_csv(
        provisional_continuous_metrics_path,
        index=False,
    )
    provisional_bucket_metrics.to_csv(
        provisional_bucket_metrics_path,
        index=False,
    )

    combo_avg_threshold_uncertainty = bootstrap_threshold_uncertainty(
        provisional_full_imdb_df,
        "combo_avg_on_10",
        thresholds=[5.0, 5.5, 6.0, 6.5, 7.0, 7.5, 8.0, 8.5, 9.0],
    )
    combo_avg_threshold_uncertainty_path = (
        output_dir / "combo_avg_threshold_uncertainty.csv"
    )
    combo_avg_threshold_uncertainty.to_csv(
        combo_avg_threshold_uncertainty_path,
        index=False,
    )
    combo_avg_threshold_uncertainty_plot_path = (
        output_dir / "combo_avg_threshold_uncertainty.png"
    )
    plot_threshold_uncertainty_panels(
        combo_avg_threshold_uncertainty,
        combo_avg_threshold_uncertainty_plot_path,
        score_label="Average of RT audience and IMDb",
        threshold_axis_label="Minimum average(RT audience, IMDb) (/10)",
        baseline_mean_rating=baseline_mean_rating,
        baseline_liked_rate=baseline_liked_rate,
        baseline_bucket5=baseline_bucket5,
    )
    combo_avg_threshold_practical_plot_path = (
        output_dir / "combo_avg_threshold_practical_tradeoff.png"
    )
    plot_threshold_practical_tradeoffs(
        combo_avg_threshold_uncertainty,
        combo_avg_threshold_practical_plot_path,
        score_label="Average of RT audience and IMDb",
        threshold_axis_label="Average(RT audience, IMDb) threshold (/10)",
        baseline_liked_rate=baseline_liked_rate,
        practical_thresholds=(6.0, 6.5, 7.0, 7.5, 8.0, 8.5, 9.0),
    )
    combo_avg_threshold_detailed_plot_path = (
        output_dir / "combo_avg_threshold_detailed_curve.png"
    )
    plot_threshold_gain_detailed(
        combo_avg_threshold_uncertainty,
        combo_avg_threshold_detailed_plot_path,
        score_label="Average of RT audience and IMDb",
        threshold_axis_label="Minimum average(RT audience, IMDb) (/10)",
        practical_thresholds=(6.0, 6.5, 7.0, 7.5, 8.0, 8.5, 9.0),
    )
    combo_avg_threshold_summary_path = (
        output_dir / "combo_avg_threshold_uncertainty_summary.txt"
    )
    write_threshold_uncertainty_summary(
        combo_avg_threshold_summary_path,
        combo_avg_threshold_uncertainty,
        score_label="Average of RT audience and IMDb",
        practical_thresholds=(6.0, 6.5, 7.0, 7.5, 8.0, 8.5, 9.0),
    )
    combo_avg_vs_rt_keep_path = (
        output_dir / "combo_avg_vs_rt_audience_matched_keep_comparison.csv"
    )
    compare_thresholds_by_keep_rate(
        combo_avg_threshold_uncertainty,
        audience_threshold_uncertainty,
        comparison_label="combo_avg",
        reference_label="rt_audience",
    ).to_csv(combo_avg_vs_rt_keep_path, index=False)

    conjunctive_rules_path = output_dir / "conjunctive_rt_imdb_threshold_rules.csv"
    summarize_conjunctive_threshold_rules(provisional_full_imdb_df).to_csv(
        conjunctive_rules_path,
        index=False,
    )

    rt_median = float(provisional_full_imdb_df["rt_audience_on_10"].median())
    imdb_median = float(provisional_full_imdb_df["imdb_on_10"].median())
    full_baseline_mean = float(provisional_full_imdb_df["my_rating"].mean())
    full_n = int(len(provisional_full_imdb_df))
    gated_tables: dict[str, pd.DataFrame] = {}
    gated_rows: list[pd.DataFrame] = []
    gate_specs = {
        f"Both >= medians ({rt_median:.2f}/{imdb_median:.2f})": (
            (provisional_full_imdb_df["rt_audience_on_10"] >= rt_median)
            & (provisional_full_imdb_df["imdb_on_10"] >= imdb_median)
        ),
        "Both >= 6.0": (
            (provisional_full_imdb_df["rt_audience_on_10"] >= 6.0)
            & (provisional_full_imdb_df["imdb_on_10"] >= 6.0)
        ),
        "Both >= 7.0": (
            (provisional_full_imdb_df["rt_audience_on_10"] >= 7.0)
            & (provisional_full_imdb_df["imdb_on_10"] >= 7.0)
        ),
    }
    for gate_label, gate_mask in gate_specs.items():
        gate_df = provisional_full_imdb_df.loc[gate_mask].copy()
        gate_table = bootstrap_threshold_uncertainty(
            gate_df,
            "combo_avg_on_10",
            thresholds=[6.0, 6.5, 7.0, 7.5, 8.0, 8.5, 9.0],
            baseline_mean=full_baseline_mean,
            denominator_n=full_n,
        )
        gate_table.insert(0, "gate", gate_label)
        gated_tables[gate_label] = gate_table
        gated_rows.append(gate_table)
    combo_avg_gated_thresholds_path = output_dir / "combo_avg_gated_thresholds.csv"
    pd.concat(gated_rows, ignore_index=True).to_csv(
        combo_avg_gated_thresholds_path,
        index=False,
    )
    combo_avg_gated_plot_path = output_dir / "combo_avg_gated_threshold_panels.png"
    plot_gated_combo_threshold_panels(gated_tables, combo_avg_gated_plot_path)

    summary_path = output_dir / "followup_summary.txt"
    write_followup_summary(
        summary_path,
        gemini_metrics=gemini_metrics,
        predictor_metrics=predictor_metrics,
        bucket_metrics=bucket_metrics,
        direct_bucket_metrics=direct_bucket_metrics,
        threshold_tables=threshold_tables,
    )
    imdb_summary_path = output_dir / "rt_imdb_summary.txt"
    imdb_summary_lines = [
        "Audience shape check:",
    ]
    for row in audience_shape_metrics.itertuples(index=False):
        imdb_summary_lines.append(
            f"  {row.model}: CV R2 {row.cv_r2:.3f}, MAE {row.mae:.3f}, "
            f"5-step acc {row.five_step_accuracy:.3f}, QWK {row.five_step_quadratic_kappa:.3f}"
        )
    imdb_summary_lines.extend(
        [
            "",
            "RT audience vs IMDb correlation on the 25-movie hybrid set:",
            (
                f"  r = {imdb_rt_correlation['correlation']:.3f}, "
                f"R2 = {imdb_rt_correlation['r2']:.3f}"
            ),
            "",
            "Best continuous RT/IMDb models:",
        ]
    )
    for row in imdb_rt_continuous.head(3).itertuples(index=False):
        imdb_summary_lines.append(
            f"  {row.model}: CV R2 {row.cv_r2:.3f}, MAE {row.mae:.3f}, RMSE {row.rmse:.3f}"
        )
    imdb_summary_lines.extend(
        [
            "",
            "Best 3-step RT/IMDb models:",
        ]
    )
    for row in imdb_rt_bucket_3.head(3).itertuples(index=False):
        imdb_summary_lines.append(
            f"  {row.model}: acc {row.accuracy:.3f}, macro F1 {row.macro_f1:.3f}, QWK {row.quadratic_kappa:.3f}"
        )
    imdb_summary_lines.extend(
        [
            "",
            "Best 5-step RT/IMDb models:",
        ]
    )
    for row in imdb_rt_bucket_5.head(3).itertuples(index=False):
        imdb_summary_lines.append(
            f"  {row.model}: acc {row.accuracy:.3f}, macro F1 {row.macro_f1:.3f}, QWK {row.quadratic_kappa:.3f}"
        )
    imdb_summary_path.write_text("\n".join(imdb_summary_lines) + "\n")

    print(f"Gemini comparison CSV: {gemini_comparison_path}")
    print(f"Gemini alignment summary CSV: {gemini_alignment_summary_path}")
    print(f"Gemini RT model fits CSV: {gemini_rt_model_fits_path}")
    print(f"Gemini drift plot: {gemini_plot_path}")
    print(f"Predictor plot: {predictor_plot_path}")
    print(f"Audience shape plot: {audience_shape_plot_path}")
    print(f"Bucket metrics CSV: {bucket_metrics_path}")
    print(f"Bucket accuracy plot: {bucket_plot_path}")
    print(f"Direct bucket accuracy CSV: {direct_bucket_metrics_path}")
    print(f"Direct bucket accuracy plot: {direct_bucket_plot_path}")
    print(f"Threshold tradeoff plot: {threshold_plot_path}")
    print(f"Audience threshold uncertainty CSV: {audience_threshold_uncertainty_path}")
    print(
        f"Audience threshold uncertainty plot: {audience_threshold_uncertainty_plot_path}"
    )
    print(
        f"Audience practical threshold plot: {audience_threshold_practical_plot_path}"
    )
    print(f"Audience detailed threshold plot: {audience_threshold_detailed_plot_path}")
    print(f"Audience threshold summary: {audience_threshold_summary_path}")
    print(f"Summary: {summary_path}")
    print(f"Verified RT+IMDb dataset: {verified_imdb_path}")
    print(f"RT+IMDb correlation plot: {imdb_rt_correlation_plot_path}")
    print(f"RT+IMDb model comparison plot: {imdb_rt_model_plot_path}")
    print(f"IMDb threshold uncertainty CSV: {imdb_threshold_uncertainty_path}")
    print(f"IMDb threshold uncertainty plot: {imdb_threshold_uncertainty_plot_path}")
    print(f"IMDb practical threshold plot: {imdb_threshold_practical_plot_path}")
    print(f"IMDb detailed threshold plot: {imdb_threshold_detailed_plot_path}")
    print(f"IMDb threshold summary: {imdb_threshold_summary_path}")
    print(f"Hybrid RT audience threshold CSV: {hybrid_rt_threshold_uncertainty_path}")
    print(f"Hybrid RT audience detailed plot: {hybrid_rt_threshold_detailed_plot_path}")
    print(f"RT vs IMDb super plot: {super_plot_path}")
    print(f"IMDb vs RT matched keep CSV: {imdb_vs_rt_keep_path}")
    print(f"Provisional full RT+IMDb dataset: {provisional_full_imdb_dataset_path}")
    print(f"Provisional IMDb quality CSV: {provisional_quality_path}")
    print(f"Provisional IMDb quality summary: {provisional_quality_summary_path}")
    print(
        f"Provisional continuous RT+IMDb metrics: {provisional_continuous_metrics_path}"
    )
    print(f"Provisional bucket RT+IMDb metrics: {provisional_bucket_metrics_path}")
    print(f"Combo-average threshold CSV: {combo_avg_threshold_uncertainty_path}")
    print(
        f"Combo-average threshold uncertainty plot: {combo_avg_threshold_uncertainty_plot_path}"
    )
    print(
        f"Combo-average practical threshold plot: {combo_avg_threshold_practical_plot_path}"
    )
    print(
        f"Combo-average detailed threshold plot: {combo_avg_threshold_detailed_plot_path}"
    )
    print(f"Combo-average threshold summary: {combo_avg_threshold_summary_path}")
    print(f"Combo-average vs RT matched keep CSV: {combo_avg_vs_rt_keep_path}")
    print(f"Conjunctive RT+IMDb threshold rules CSV: {conjunctive_rules_path}")
    print(f"Combo-average gated thresholds CSV: {combo_avg_gated_thresholds_path}")
    print(f"Combo-average gated panels plot: {combo_avg_gated_plot_path}")
    print(f"RT+IMDb summary: {imdb_summary_path}")


if __name__ == "__main__":
    main()
