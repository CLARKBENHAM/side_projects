Movie Rotten Tomatoes analysis work moved out of `/Users/clarkbenham/tree_leads`.

Contents:
- `analysis_core/movie_rt_analysis.py`: reusable matching, scoring, plotting, and modeling logic.
- `scripts/analyze_movie_rt.py`: runner script.
- `tests/test_movie_rt_analysis.py`: focused tests for parsing and holdout evaluation.
- `data/summaries/movie_rt_notes.txt`: source ratings/notes used for this run.
- `data/summaries/movie_rt_analysis/`: generated CSVs, plots, cache, and summary outputs.

The implementation package is named `analysis_core` to avoid confusion with the
separate `tree_leads` repo.
