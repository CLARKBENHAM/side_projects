"""Score unread books from the cleaned takeout CSV using both simple GR cutoff and ML models."""

from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

os.environ.setdefault("MPLCONFIGDIR", str(Path("/tmp/matplotlib-cache")))

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ai_books_tracking.future_prediction_evaluation import (
    FeatureSpec,
    model_predictions_for_split,
)
from ai_books_tracking.goodreads_followup_analysis import (
    derive_analysis_columns,
    load_data,
)
from ai_books_tracking.new_books_to_rate_analysis import (
    NEW_BOOKS_ENRICHED,
    add_centered_targets,
)

OUTPUT_DIR = Path(__file__).parent
DATA_DIR = OUTPUT_DIR.parent / "data"
UNREAD_CSV = (
    DATA_DIR
    / "Books Read and their effects - takeout_play_books_03_16_25_unread_for_external_search_cleaned.csv"
)


def load_training_frame() -> pd.DataFrame:
    historical = add_centered_targets(derive_analysis_columns(load_data()))
    historical["label_source"] = "historical_main"
    frames = [historical]
    if NEW_BOOKS_ENRICHED.exists():
        new_holdout = pd.read_csv(NEW_BOOKS_ENRICHED)
        new_holdout.columns = new_holdout.columns.str.strip()
        new_holdout = add_centered_targets(derive_analysis_columns(new_holdout))
        new_holdout["label_source"] = "new_books_holdout_2026"
        frames.append(new_holdout)
    combined = pd.concat(frames, ignore_index=True, sort=False)
    dedupe_key = (
        combined["title"].astype(str).str.strip().str.lower()
        + "||"
        + combined.get("filename", pd.Series("", index=combined.index))
        .astype(str)
        .str.strip()
        .str.lower()
    )
    combined = combined.loc[~dedupe_key.duplicated()].reset_index(drop=True)
    return combined


def parse_amazon_rating(val: object) -> tuple[float | None, int | None]:
    s = str(val).strip()
    if s in ("", "nan", "N/A", "N/A|N/A|N/A"):
        return None, None
    parts = s.split("|")
    try:
        rating = float(parts[0]) if parts[0] not in ("N/A", "") else None
    except ValueError:
        rating = None
    try:
        count = (
            int(parts[1]) if len(parts) > 1 and parts[1] not in ("N/A", "") else None
        )
    except ValueError:
        count = None
    return rating, count


def load_unread_books(csv_path: Path = UNREAD_CSV) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    df.columns = df.columns.str.strip()

    # Parse Goodreads rating
    df["goodreads_rating"] = pd.to_numeric(df["goodread ratings"], errors="coerce")
    df["goodreads_rating_count"] = pd.to_numeric(
        df["goodreads number ratings"], errors="coerce"
    )

    # Parse Amazon rating
    amz = df["Amazon combined with links"].apply(parse_amazon_rating)
    df["amazon_rating"] = amz.apply(lambda x: x[0])
    df["amazon_count"] = amz.apply(lambda x: x[1])

    # Parse OL rating
    df["ol_rating"] = pd.to_numeric(df["open library ratings"], errors="coerce")

    # Determine started vs not started
    # "finished" = already finished but in the unread list (odd), "unfinished" = started or not
    # Use earliest_modified: if it exists, the book was opened/started at some point
    df["earliest_modified"] = pd.to_datetime(df["earliest_modified"], errors="coerce")
    df["latest_modified"] = pd.to_datetime(df["latest_modified"], errors="coerce")

    # Classify: if play_status is "finished" or has timestamps, consider it "started"
    df["has_been_started"] = (
        df["play_status"].eq("finished") | df["earliest_modified"].notna()
    )

    return df


def prepare_for_scoring(df: pd.DataFrame) -> pd.DataFrame:
    """Adapt unread book columns to match what the model pipeline expects."""
    scored = df.copy()

    # The model needs these columns from the training pipeline
    for col in [
        "Enjoyment (/5)",
        "Enjoyment (/5)_ratings2",
        "Usefulness /5 to Me",
        "Usefulness /5 to Me_ratings2",
    ]:
        scored[col] = np.nan

    scored["Long Term Effects"] = ""
    scored["earliest_modified_ratings2"] = scored["earliest_modified"]
    scored["latest_modified_ratings2"] = scored["latest_modified"]
    scored["author_ratings2"] = scored.get("cleaned_author", scored.get("author", ""))
    scored["author_goodreads"] = ""

    # Map goodreads columns to what the conservative spec expects
    scored["goodreads_status"] = np.where(
        scored["goodreads_rating"].notna(), "matched", "unmatched"
    )
    scored["goodreads_match_method"] = "external_search"
    scored["goodreads_url"] = ""
    scored["goodreads_title"] = scored["title"]
    scored["goodreads_author"] = scored.get("cleaned_author", "")
    scored["goodreads_match_score"] = np.nan
    scored["goodreads_title_similarity"] = np.nan
    scored["goodreads_author_similarity"] = np.nan
    scored["goodreads_query"] = ""
    scored["goodreads_candidates_json"] = "[]"

    # For raw_best goodreads columns (same as conservative here since we have one source)
    scored["goodreads_rating_raw_best"] = scored["goodreads_rating"]
    scored["goodreads_rating_count_raw_best"] = scored["goodreads_rating_count"]
    scored["goodreads_url_raw_best"] = ""
    scored["goodreads_title_raw_best"] = scored["title"]
    scored["goodreads_author_raw_best"] = scored.get("cleaned_author", "")
    scored["goodreads_raw_best_source"] = "gsheets"
    scored["goodreads_raw_best_score"] = np.nan

    # Stub page count / pub year
    if "gb_page_count" not in scored.columns:
        scored["gb_page_count"] = np.nan
    if "pub_year" not in scored.columns:
        scored["pub_year"] = np.nan

    # Map Bookshelf values to training categories
    shelf_map = {
        "Histories": "General Reading",
        "Stats": "Computer Science",
        "Kids": "General Reading",
        "Unknown Shelf": "General Reading",
        "Advanced Finance": "Business, management",
    }
    scored["Bookshelf"] = scored["Bookshelf"].replace(shelf_map)

    scored = derive_analysis_columns(scored)
    scored = add_centered_targets(scored)
    return scored


def main() -> None:
    print("Loading training data...")
    training = load_training_frame()
    print(f"  Training: {len(training)} books")

    print("Loading unread books...")
    unread = load_unread_books()
    print(f"  Unread: {len(unread)} books")
    print(f"  Started: {unread['has_been_started'].sum()}")
    print(f"  Not started: {(~unread['has_been_started']).sum()}")
    print(f"  With Goodreads rating: {unread['goodreads_rating'].notna().sum()}")

    print("\nPreparing for scoring...")
    candidates = prepare_for_scoring(unread)

    # ── Model 1: Simple Goodreads cutoff ──
    candidates["gr_cutoff_4p2"] = candidates["goodreads_rating"] >= 4.2
    candidates["gr_cutoff_4p0"] = candidates["goodreads_rating"] >= 4.0

    # ── Model 2: Random Forest (best bootstrap-stable model) ──
    rf_spec = FeatureSpec(
        "preread_plus_goodreads_conservative", include_goodreads="conservative"
    )
    rf_enjoy_preds = model_predictions_for_split(
        train_df=training[training["avg_enjoyment"].notna()].copy(),
        test_df=candidates,
        target_col="avg_enjoyment",
        spec=rf_spec,
        model_name="Random Forest",
    )
    candidates["pred_enjoy_rf"] = rf_enjoy_preds

    rf_useful_preds = model_predictions_for_split(
        train_df=training[training["avg_usefulness"].notna()].copy(),
        test_df=candidates,
        target_col="avg_usefulness",
        spec=rf_spec,
        model_name="Random Forest",
    )
    candidates["pred_useful_rf"] = rf_useful_preds

    # ── Model 3: GBM (best MAE model) ──
    gbm_spec = FeatureSpec(
        "preread_plus_goodreads_conservative", include_goodreads="conservative"
    )
    gbm_enjoy_preds = model_predictions_for_split(
        train_df=training[training["avg_enjoyment"].notna()].copy(),
        test_df=candidates,
        target_col="avg_enjoyment",
        spec=gbm_spec,
        model_name="GBM",
    )
    candidates["pred_enjoy_gbm"] = gbm_enjoy_preds

    gbm_useful_preds = model_predictions_for_split(
        train_df=training[training["avg_usefulness"].notna()].copy(),
        test_df=candidates,
        target_col="avg_usefulness",
        spec=gbm_spec,
        model_name="GBM",
    )
    candidates["pred_useful_gbm"] = gbm_useful_preds

    # Decision thresholds (original holdout-calibrated)
    candidates["rf_keep_enjoy"] = candidates["pred_enjoy_rf"] >= 3.9
    candidates["rf_keep_useful"] = candidates["pred_useful_rf"] >= 2.5
    candidates["gbm_keep_enjoy"] = candidates["pred_enjoy_gbm"] >= 3.8

    # Percentile-based tiers (more useful for unread catalog where predictions cluster)
    for col in ["pred_enjoy_rf", "pred_useful_rf"]:
        candidates[f"{col}_pctile"] = candidates[col].rank(pct=True)

    # Multiple ranking modes
    ranking_modes: list[tuple[str, str, float, float]] = [
        ("balanced", "60% enjoyment + 40% usefulness", 0.6, 0.4),
        ("usefulness", "100% usefulness", 0.0, 1.0),
        ("weighted_useful", "2.5*usefulness + enjoyment (normalized)", 1.0, 2.5),
    ]

    for mode_name, mode_desc, enjoy_w, useful_w in ranking_modes:
        # Raw weighted score (for ranking — use raw predictions, not percentiles,
        # when weights are non-percentile like 2.5*U + E)
        if mode_name == "weighted_useful":
            candidates[f"raw_score_{mode_name}"] = (
                enjoy_w * candidates["pred_enjoy_rf"]
                + useful_w * candidates["pred_useful_rf"]
            )
        else:
            candidates[f"raw_score_{mode_name}"] = (
                enjoy_w * candidates["pred_enjoy_rf_pctile"]
                + useful_w * candidates["pred_useful_rf_pctile"]
            )

        # Rank the raw score to get percentiles for tier assignment
        candidates[f"composite_pctile_{mode_name}"] = candidates[
            f"raw_score_{mode_name}"
        ].rank(pct=True)

        # Tier assignment
        candidates[f"tier_{mode_name}"] = np.select(
            [
                candidates[f"composite_pctile_{mode_name}"] >= 0.80,
                candidates[f"composite_pctile_{mode_name}"] >= 0.60,
                candidates[f"composite_pctile_{mode_name}"] >= 0.40,
                candidates[f"composite_pctile_{mode_name}"] >= 0.20,
            ],
            ["A: Top 20%", "B: 60-80%", "C: 40-60%", "D: 20-40%"],
            default="F: Bottom 20%",
        )

    # Default tier/composite for backward compat
    candidates["composite_pctile"] = candidates["composite_pctile_balanced"]
    candidates["tier"] = candidates["tier_balanced"]

    # Composite decision (original thresholds)
    candidates["rf_decision"] = np.select(
        [
            candidates["rf_keep_enjoy"] & candidates["rf_keep_useful"],
            candidates["rf_keep_enjoy"],
            candidates["rf_keep_useful"],
        ],
        ["KEEP (high priority)", "KEEP (enjoyable)", "KEEP (useful)"],
        default="DISCARD",
    )

    # ── Print summary stats ──
    started = candidates[candidates["has_been_started"]]
    not_started = candidates[~candidates["has_been_started"]]

    print("\n" + "=" * 80)
    print("  PREDICTED RATINGS SUMMARY")
    print("=" * 80)

    for label, subset in [
        ("STARTED books", started),
        ("NOT STARTED books", not_started),
    ]:
        print(f"\n--- {label} (n={len(subset)}) ---")
        has_gr = subset["goodreads_rating"].notna()
        print(
            f"  Goodreads coverage: {has_gr.sum()}/{len(subset)} ({100*has_gr.mean():.0f}%)"
        )
        if has_gr.any():
            print(
                f"  Mean Goodreads rating: {subset.loc[has_gr, 'goodreads_rating'].mean():.2f}"
            )
        print(f"  Mean pred enjoyment (RF):  {subset['pred_enjoy_rf'].mean():.2f}")
        print(f"  Mean pred usefulness (RF): {subset['pred_useful_rf'].mean():.2f}")
        print(f"  Mean pred enjoyment (GBM): {subset['pred_enjoy_gbm'].mean():.2f}")
        print(f"  Mean pred usefulness (GBM):{subset['pred_useful_gbm'].mean():.2f}")
        print(f"  GR >= 4.2 keep: {subset['gr_cutoff_4p2'].sum()}/{len(subset)}")
        print(
            f"  RF keep (enjoy >= 3.9): {subset['rf_keep_enjoy'].sum()}/{len(subset)}"
        )
        print(
            f"  GBM keep (enjoy >= 3.8): {subset['gbm_keep_enjoy'].sum()}/{len(subset)}"
        )

    # ── Print per-category breakdown ──
    print("\n" + "=" * 80)
    print("  PREDICTIONS BY CATEGORY")
    print("=" * 80)
    for cat in sorted(candidates["Bookshelf"].unique()):
        cat_df = candidates[candidates["Bookshelf"] == cat]
        print(f"\n  {cat} (n={len(cat_df)}):")
        print(
            f"    Pred enjoy RF:  {cat_df['pred_enjoy_rf'].mean():.2f} ({cat_df['pred_enjoy_rf'].min():.2f}-{cat_df['pred_enjoy_rf'].max():.2f})"
        )
        print(
            f"    Pred useful RF: {cat_df['pred_useful_rf'].mean():.2f} ({cat_df['pred_useful_rf'].min():.2f}-{cat_df['pred_useful_rf'].max():.2f})"
        )
        print(f"    RF keep: {cat_df['rf_keep_enjoy'].sum()}/{len(cat_df)}")

    # ── Check generalization: books likely not cared for ──
    print("\n" + "=" * 80)
    print("  GENERALIZATION CHECK: LIKELY LOW-INTEREST BOOKS")
    print("=" * 80)
    print("  (Books with low Goodreads rating OR in categories historically rated low)")

    low_gr = candidates[
        candidates["goodreads_rating"].lt(3.8) & candidates["goodreads_rating"].notna()
    ]
    print(f"\n  Books with GR < 3.8 (n={len(low_gr)}):")
    print(f"    Mean pred enjoy RF: {low_gr['pred_enjoy_rf'].mean():.2f}")
    print(f"    Mean pred useful RF: {low_gr['pred_useful_rf'].mean():.2f}")
    print(f"    RF keep: {low_gr['rf_keep_enjoy'].sum()}/{len(low_gr)}")

    high_gr = candidates[
        candidates["goodreads_rating"].ge(4.2) & candidates["goodreads_rating"].notna()
    ]
    print(f"\n  Books with GR >= 4.2 (n={len(high_gr)}):")
    print(f"    Mean pred enjoy RF: {high_gr['pred_enjoy_rf'].mean():.2f}")
    print(f"    Mean pred useful RF: {high_gr['pred_useful_rf'].mean():.2f}")
    print(f"    RF keep: {high_gr['rf_keep_enjoy'].sum()}/{len(high_gr)}")

    no_gr = candidates[candidates["goodreads_rating"].isna()]
    print(f"\n  Books with NO Goodreads rating (n={len(no_gr)}):")
    print(f"    Mean pred enjoy RF: {no_gr['pred_enjoy_rf'].mean():.2f}")
    print(f"    Mean pred useful RF: {no_gr['pred_useful_rf'].mean():.2f}")
    print(f"    RF keep: {no_gr['rf_keep_enjoy'].sum()}/{len(no_gr)}")

    # ── Write recommendations files (one per ranking mode) ──

    def write_book_table(
        f,
        subset: pd.DataFrame,
        tier_col: str = "tier",
        pctile_col: str = "composite_pctile",
        include_decision: bool = True,
    ) -> None:
        if include_decision:
            f.write(
                f"{'Title':<60} {'Shelf':<20} {'GR':>5} {'AMZ':>5} {'E(RF)':>6} {'U(RF)':>6} {'E(GBM)':>7} {'Tier':<12} {'GR>=4.2':>7}\n"
            )
            f.write("-" * 140 + "\n")
        else:
            f.write(
                f"{'Title':<60} {'Shelf':<20} {'GR':>5} {'AMZ':>5} {'E(RF)':>6} {'U(RF)':>6} {'E(GBM)':>7}\n"
            )
            f.write("-" * 120 + "\n")
        for _, row in subset.iterrows():
            title = str(row["title"])[:58]
            shelf = str(row["Bookshelf"])[:18]
            gr = (
                f"{row['goodreads_rating']:.1f}"
                if pd.notna(row["goodreads_rating"])
                else "  N/A"
            )
            amz = (
                f"{row['amazon_rating']:.1f}"
                if pd.notna(row["amazon_rating"])
                else "  N/A"
            )
            e_rf = f"{row['pred_enjoy_rf']:.2f}"
            u_rf = f"{row['pred_useful_rf']:.2f}"
            e_gbm = f"{row['pred_enjoy_gbm']:.2f}"
            if include_decision:
                tier = str(row[tier_col])[:10]
                gr42 = "  yes" if row.get("gr_cutoff_4p2", False) else "   no"
                f.write(
                    f"{title:<60} {shelf:<20} {gr:>5} {amz:>5} {e_rf:>6} {u_rf:>6} {e_gbm:>7} {tier:<12} {gr42:>7}\n"
                )
            else:
                f.write(
                    f"{title:<60} {shelf:<20} {gr:>5} {amz:>5} {e_rf:>6} {u_rf:>6} {e_gbm:>7}\n"
                )

    for mode_name, mode_desc, _, _ in ranking_modes:
        tier_col = f"tier_{mode_name}"
        pctile_col = f"composite_pctile_{mode_name}"

        suffix = "" if mode_name == "balanced" else f"_{mode_name}"
        output_path = (
            OUTPUT_DIR / "ai_actions" / f"unread_book_recommendations{suffix}.txt"
        )

        # Recompute started/not_started views (they share the same DataFrame)
        started = candidates[candidates["has_been_started"]]
        not_started = candidates[~candidates["has_been_started"]]

        with open(output_path, "w") as f:
            f.write("=" * 100 + "\n")
            f.write("  UNREAD BOOK RECOMMENDATIONS\n")
            f.write(f"  Ranking: {mode_desc}\n")
            f.write(f"  Generated from {len(candidates)} books\n")
            f.write("  \n")
            f.write("  Models used:\n")
            f.write("    1. Simple cutoff: Goodreads >= 4.2\n")
            f.write(
                "    2. Random Forest: category + Goodreads (conservative) -> predicted enjoyment & usefulness\n"
            )
            f.write(
                "    3. GBM: category + Goodreads (conservative) -> predicted enjoyment\n"
            )
            f.write("  \n")
            f.write(f"  Tiers based on: {mode_desc}\n")
            f.write(
                "  Columns: GR=Goodreads, AMZ=Amazon, E(RF)=pred enjoyment RF, U(RF)=pred usefulness RF\n"
            )
            f.write("=" * 100 + "\n\n")

            for section_label, subset in [
                ("BOOKS I'VE STARTED", started),
                ("BOOKS I HAVEN'T STARTED", not_started),
            ]:
                f.write("=" * 100 + "\n")
                f.write(f"  {section_label} (n={len(subset)})\n")
                f.write("=" * 100 + "\n\n")

                keep = subset[subset[tier_col].str.startswith("A:")].sort_values(
                    pctile_col, ascending=False
                )
                f.write(
                    f"--- RECOMMENDED: Top 20% by model ({len(keep)} books) ---\n\n"
                )
                if len(keep) > 0:
                    write_book_table(f, keep, tier_col, pctile_col)
                else:
                    f.write("  (none in this section)\n")

                maybe = subset[subset[tier_col].str.startswith("B:")].sort_values(
                    pctile_col, ascending=False
                )
                f.write(
                    f"\n--- WORTH CONSIDERING: 60-80th percentile ({len(maybe)} books) ---\n\n"
                )
                if len(maybe) > 0:
                    write_book_table(f, maybe, tier_col, pctile_col)

                bottom = subset[
                    subset[tier_col].str.startswith("D:")
                    | subset[tier_col].str.startswith("F:")
                ].sort_values(pctile_col, ascending=True)
                f.write(
                    f"\n--- LIKELY DISCARD: Bottom 40% ({len(bottom)} books) ---\n\n"
                )
                if len(bottom) > 0:
                    write_book_table(
                        f, bottom, tier_col, pctile_col, include_decision=False
                    )

                f.write("\n\n")

            # Generalization check
            f.write("=" * 100 + "\n")
            f.write("  GENERALIZATION CHECK: MODEL VS SIMPLE CUTOFF DISAGREEMENTS\n")
            f.write("=" * 100 + "\n\n")

            model_yes_gr_no = candidates[
                (candidates[tier_col].str.startswith("A:"))
                & (~candidates["gr_cutoff_4p2"])
            ].sort_values(pctile_col, ascending=False)
            f.write(
                f"--- Model Top 20% but GR < 4.2 or missing ({len(model_yes_gr_no)} books) ---\n"
            )
            f.write(
                "  (Model relies on category/author signal rather than public ratings)\n\n"
            )
            if len(model_yes_gr_no) > 0:
                write_book_table(f, model_yes_gr_no, tier_col, pctile_col)

            f.write("\n")

            gr_yes_model_no = candidates[
                (candidates[tier_col].str.startswith(("D:", "F:")))
                & (candidates["gr_cutoff_4p2"])
            ].sort_values("goodreads_rating", ascending=False)
            f.write(
                f"--- GR >= 4.2 but model Bottom 40% ({len(gr_yes_model_no)} books) ---\n"
            )
            f.write("  (High public consensus but model predicts low personal fit)\n\n")
            if len(gr_yes_model_no) > 0:
                write_book_table(f, gr_yes_model_no, tier_col, pctile_col)

            f.write("\n")

            started_bottom = started[
                started[tier_col].str.startswith("F:")
            ].sort_values(pctile_col)
            f.write(f"--- Started but Bottom 20% ({len(started_bottom)} books) ---\n")
            f.write(
                "  (You started these but model predicts low enjoyment -- consider dropping)\n\n"
            )
            if len(started_bottom) > 0:
                write_book_table(
                    f, started_bottom, tier_col, pctile_col, include_decision=False
                )

        print(f"Wrote recommendations ({mode_name}) to: {output_path}")

    # Also save a CSV for further analysis
    csv_path = OUTPUT_DIR / "ai_actions" / "unread_book_scores.csv"
    out_cols = [
        "title",
        "cleaned_author",
        "Bookshelf",
        "play_status",
        "has_been_started",
        "goodreads_rating",
        "goodreads_rating_count",
        "amazon_rating",
        "amazon_count",
        "ol_rating",
        "pred_enjoy_rf",
        "pred_useful_rf",
        "pred_enjoy_gbm",
        "pred_useful_gbm",
        "tier_balanced",
        "composite_pctile_balanced",
        "tier_usefulness",
        "composite_pctile_usefulness",
        "tier_weighted_useful",
        "composite_pctile_weighted_useful",
        "raw_score_weighted_useful",
        "tier",
        "composite_pctile",
        "rf_decision",
        "gr_cutoff_4p2",
        "gr_cutoff_4p0",
        "rf_keep_enjoy",
        "rf_keep_useful",
        "gbm_keep_enjoy",
    ]
    existing = [c for c in out_cols if c in candidates.columns]
    candidates[existing].to_csv(csv_path, index=False)
    print(f"Wrote scores CSV to: {csv_path}")


if __name__ == "__main__":
    main()
