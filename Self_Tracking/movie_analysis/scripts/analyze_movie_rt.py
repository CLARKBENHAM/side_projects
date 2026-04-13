from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))


def _cache_row_to_match_result(entry: Any, row: dict[str, object]) -> Any:
    from analysis_core.movie_rt_analysis import MatchResult

    return MatchResult(
        entry=entry,
        matched_title=str(row["matched_title"]),
        media_type=str(row["media_type"]),
        rt_url=str(row["rt_url"]),
        critic_score=None if row["critic_score"] is None else int(row["critic_score"]),
        audience_score=(
            None if row["audience_score"] is None else int(row["audience_score"])
        ),
        release_year=None if row["release_year"] is None else int(row["release_year"]),
        match_confidence=float(row["match_confidence"]),
        query=str(row["query"]),
    )


def _match_result_to_cache_row(result: Any) -> dict[str, object]:
    return {
        "matched_title": result.matched_title,
        "media_type": result.media_type,
        "rt_url": result.rt_url,
        "critic_score": result.critic_score,
        "audience_score": result.audience_score,
        "release_year": result.release_year,
        "match_confidence": result.match_confidence,
        "query": result.query,
    }


def main() -> None:
    from analysis_core.movie_rt_analysis import (
        build_disagreement_tables,
        build_ratings_dataframe,
        build_session,
        cross_validate_models,
        create_model_comparison_plot,
        create_rt_scatter_plot,
        evaluate_models,
        evaluate_threshold_baselines,
        fetch_match_result,
        parse_movie_entries,
        simulate_data_quality_impact,
        summarize_analysis,
    )

    notes_path = ROOT / "data" / "summaries" / "movie_rt_notes.txt"
    output_dir = ROOT / "data" / "summaries" / "movie_rt_analysis"
    output_dir.mkdir(parents=True, exist_ok=True)
    match_cache_path = output_dir / "movie_rt_match_cache.json"

    entries = parse_movie_entries(notes_path.read_text())
    session = build_session()
    cache: dict[str, dict[str, object]] = (
        json.loads(match_cache_path.read_text()) if match_cache_path.exists() else {}
    )
    matches: list[Any] = []
    for entry in entries:
        cached_row = cache.get(entry.title)
        if cached_row is not None:
            matches.append(_cache_row_to_match_result(entry, cached_row))
            continue

        result = fetch_match_result(session, entry)
        matches.append(result)
        cache[entry.title] = _match_result_to_cache_row(result)
        match_cache_path.write_text(json.dumps(cache, indent=2, sort_keys=True) + "\n")
        print(
            f"Matched {entry.index + 1:03d}/{len(entries)}: {entry.title} -> {result.matched_title}"
        )
        time.sleep(0.35)

    df = build_ratings_dataframe(matches)
    model_results = evaluate_models(df)
    threshold_results = evaluate_threshold_baselines(df)
    cv_results = cross_validate_models(df)
    data_quality_results = simulate_data_quality_impact(df)
    disagreements = build_disagreement_tables(model_results["full_predictions"])

    csv_path = output_dir / "movie_rt_scores.csv"
    df[["movie_title", "my_rating", "rt_audience_rating", "rt_critic_rating"]].to_csv(
        csv_path, index=False
    )

    detailed_csv_path = output_dir / "movie_rt_scores_detailed.csv"
    df.to_csv(detailed_csv_path, index=False)

    holdout_csv_path = output_dir / "movie_rt_holdout_predictions.csv"
    model_results["holdout_predictions"].to_csv(holdout_csv_path, index=False)

    threshold_csv_path = output_dir / "movie_rt_threshold_rules.csv"
    threshold_results.to_csv(threshold_csv_path, index=False)

    for name, table in disagreements.items():
        table.head(15).to_csv(output_dir / f"{name}.csv", index=False)

    plot_path = output_dir / "movie_rt_vs_my_rating.png"
    create_rt_scatter_plot(df, plot_path)
    model_plot_path = output_dir / "movie_rt_model_comparison.png"
    create_model_comparison_plot(df, model_results, model_plot_path)

    summary_text = summarize_analysis(df, model_results)
    summary_path = output_dir / "movie_rt_summary.txt"
    summary_path.write_text(summary_text + "\n")

    metrics_path = output_dir / "movie_rt_metrics.json"
    metrics_payload = {
        "summary": summary_text,
        "train_size": model_results["train_size"],
        "holdout_size": model_results["holdout_size"],
        "average_metrics": model_results["average_metrics"],
        "linear_metrics": model_results["linear_metrics"],
        "linear_coefficients": model_results["linear_coefficients"],
        "logistic_metrics": model_results["logistic_metrics"],
        "logistic_coefficients": model_results["logistic_coefficients"],
    }
    metrics_path.write_text(json.dumps(metrics_payload, indent=2) + "\n")

    cv_metrics_path = output_dir / "movie_rt_cv_metrics.json"
    cv_metrics_payload = {
        "n_splits": cv_results["n_splits"],
        "n_repeats": cv_results["n_repeats"],
        "regression_summary": cv_results["regression_summary"].to_dict(
            orient="records"
        ),
        "classification_summary": cv_results["classification_summary"].to_dict(
            orient="records"
        ),
    }
    cv_metrics_path.write_text(json.dumps(cv_metrics_payload, indent=2) + "\n")

    data_quality_path = output_dir / "movie_rt_data_quality_sensitivity.json"
    data_quality_path.write_text(json.dumps(data_quality_results, indent=2) + "\n")

    print(summary_text)
    print()
    print(f"CSV: {csv_path}")
    print(f"Detailed CSV: {detailed_csv_path}")
    print(f"Holdout predictions: {holdout_csv_path}")
    print(f"Threshold rules: {threshold_csv_path}")
    print(f"Plot: {plot_path}")
    print(f"Model plot: {model_plot_path}")
    print(f"Summary: {summary_path}")
    print(f"Metrics JSON: {metrics_path}")
    print(f"Cross-validation JSON: {cv_metrics_path}")
    print(f"Data quality sensitivity JSON: {data_quality_path}")


if __name__ == "__main__":
    main()
