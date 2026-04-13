from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from analysis_core.calendar_movie_scores import find_new_calendar_movies  # noqa: E402


def main() -> None:
    output_dir = ROOT / "data" / "summaries" / "movie_rt_analysis" / "calendar_movies"
    output_dir.mkdir(parents=True, exist_ok=True)

    rows, ambiguities = find_new_calendar_movies(
        calendar_dir=ROOT.parents[1] / "data" / "Takeout 5" / "Calendar",
        notes_path=ROOT / "data" / "summaries" / "movie_rt_notes.txt",
        existing_rt_csv_path=ROOT
        / "data"
        / "summaries"
        / "movie_rt_analysis"
        / "movie_rt_scores_detailed.csv",
        output_csv_path=output_dir / "new_calendar_movies_rt_imdb.csv",
        ambiguity_csv_path=output_dir / "new_calendar_movies_ambiguities.csv",
        cache_path=output_dir / "new_calendar_movies_match_cache.json",
    )

    print(f"Wrote {len(rows)} new movie rows")
    print(output_dir / "new_calendar_movies_rt_imdb.csv")
    print(f"Wrote {len(ambiguities)} ambiguity rows")
    print(output_dir / "new_calendar_movies_ambiguities.csv")


if __name__ == "__main__":
    main()
