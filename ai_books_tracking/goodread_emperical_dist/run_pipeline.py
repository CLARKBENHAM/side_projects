from __future__ import annotations

import argparse

from ai_books_tracking.goodread_emperical_dist.pipeline import run_pipeline


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the isolated Goodreads empirical distribution pipeline."
    )
    parser.add_argument("--minimum-public-ratings", type=int, default=100)
    parser.add_argument("--limit-profiles", type=int)
    parser.add_argument("--refresh-feeds", action="store_true")
    parser.add_argument("--max-pages", type=int)
    parser.add_argument("--sleep-seconds", type=float, default=0.1)
    parser.add_argument("--bootstrap-samples", type=int, default=300)
    parser.add_argument("--small-multiple-count", type=int, default=19)
    parser.add_argument("--run-network-expansion", action="store_true")
    parser.add_argument("--include-network-profiles-in-analysis", action="store_true")
    parser.add_argument("--analysis-target-total", type=int)
    parser.add_argument("--network-max-depth", type=int, default=2)
    parser.add_argument("--network-max-profiles", type=int, default=600)
    parser.add_argument("--network-target-pass-count", type=int)
    parser.add_argument("--network-profile-step-size", type=int, default=200)
    parser.add_argument("--network-max-profiles-cap", type=int, default=4000)
    parser.add_argument("--network-validation-sleep-seconds", type=float)
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    outputs = run_pipeline(
        minimum_public_ratings=args.minimum_public_ratings,
        limit_profiles=args.limit_profiles,
        refresh_feeds=args.refresh_feeds,
        max_pages=args.max_pages,
        sleep_seconds=args.sleep_seconds,
        verbose=args.verbose,
        bootstrap_samples=args.bootstrap_samples,
        small_multiple_count=args.small_multiple_count,
        run_network_expansion=args.run_network_expansion,
        include_network_profiles_in_analysis=args.include_network_profiles_in_analysis,
        analysis_target_total=args.analysis_target_total,
        network_max_depth=args.network_max_depth,
        network_max_profiles=args.network_max_profiles,
        network_target_pass_count=args.network_target_pass_count,
        network_profile_step_size=args.network_profile_step_size,
        network_max_profiles_cap=args.network_max_profiles_cap,
        network_validation_sleep_seconds=args.network_validation_sleep_seconds,
    )
    profile_summary = outputs["profile_summary"]
    evaluated = profile_summary[profile_summary["split_type"] != "not_evaluated"]
    print(
        "Pipeline finished: "
        f"{len(profile_summary)} profiles fetched, {len(evaluated)} evaluated."
    )


if __name__ == "__main__":
    main()
