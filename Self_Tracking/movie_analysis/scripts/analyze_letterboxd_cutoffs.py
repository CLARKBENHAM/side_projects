from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Fetch Letterboxd user RSS feeds, join to Letterboxd film averages, "
            "and measure how much users' mean ratings improve when filtering out "
            "low-average films."
        )
    )
    parser.add_argument(
        "users",
        nargs="+",
        help="Public Letterboxd profile slugs, for example 'brat'.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "data" / "summaries" / "letterboxd_cutoff_analysis",
        help="Directory for CSV outputs, plots, and the film metadata cache.",
    )
    parser.add_argument(
        "--refresh-film-cache",
        action="store_true",
        help="Refetch Letterboxd film average metadata even if cached.",
    )
    parser.add_argument(
        "--sleep-seconds",
        type=float,
        default=0.0,
        help="Optional sleep between fresh film page fetches.",
    )
    parser.add_argument(
        "--min-keep-rate",
        type=float,
        default=0.1,
        help="Minimum share of a user's rated films kept when selecting the best cutoff.",
    )
    return parser.parse_args()


def main() -> None:
    from analysis_core.letterboxd_cutoff_analysis import analyze_letterboxd_profiles

    args = parse_args()
    results = analyze_letterboxd_profiles(
        profile_slugs=args.users,
        output_dir=args.output_dir,
        refresh_film_cache=args.refresh_film_cache,
        sleep_seconds=args.sleep_seconds,
        min_keep_rate=args.min_keep_rate,
    )
    profile_summary = results["profile_summary"]
    if profile_summary.empty:
        print("No rated films with public RSS entries were found.")
        return
    print(profile_summary.to_string(index=False))
    print()
    print(f"Output directory: {args.output_dir}")


if __name__ == "__main__":
    main()
