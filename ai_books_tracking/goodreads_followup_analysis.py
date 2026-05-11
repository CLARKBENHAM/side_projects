"""Follow-up analysis for Goodreads-enriched book ratings."""

from __future__ import annotations

import math
import re
import textwrap
import warnings
from dataclasses import dataclass
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import LeaveOneOut, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

matplotlib.use("Agg")
import matplotlib.pyplot as plt

CURRENT_YEAR = 2026
OUTPUT_DIR = Path(__file__).parent
INPUT_FILE = OUTPUT_DIR / "books_enriched_with_goodreads.csv"
CORRELATION_CSV = OUTPUT_DIR / "goodreads_correlations.csv"
MODEL_RESULTS_CSV = OUTPUT_DIR / "goodreads_model_results.csv"
BEST_MODELS_CSV = OUTPUT_DIR / "goodreads_model_best_by_target.csv"
MISSING_AUDIT_CSV = OUTPUT_DIR / "goodreads_missing_match_audit.csv"
MISSING_YEAR_CSV = OUTPUT_DIR / "goodreads_missing_match_by_year.csv"
MISSING_YEAR_PLOT = OUTPUT_DIR / "goodreads_missing_match_by_year.png"
RESCUE_AUDIT_CSV = OUTPUT_DIR / "goodreads_raw_rescue_audit.csv"
RESTRICTED_RANGE_PLOT = OUTPUT_DIR / "goodreads_raw_best_restricted_3p2_4p8.png"
RESIDUAL_SUMMARY_CSV = OUTPUT_DIR / "goodreads_residuals_by_category.csv"
RESIDUAL_PLOT = OUTPUT_DIR / "goodreads_residuals_by_category.png"
BOOKSHELF_CORRELATION_CSV = OUTPUT_DIR / "goodreads_bookshelf_correlations.csv"
BOOKSHELF_PLOT = OUTPUT_DIR / "goodreads_bookshelf_avg_targets.png"
SOURCE_SUMMARY_CSV = OUTPUT_DIR / "goodreads_inferred_source_summary.csv"
SOURCE_PLOT = OUTPUT_DIR / "goodreads_inferred_source_vs_avg_targets.png"
CUTOFF_SWEEP_CSV = OUTPUT_DIR / "goodreads_cutoff_policy_sweep.csv"
CUTOFF_BEST_CSV = OUTPUT_DIR / "goodreads_cutoff_policy_best.csv"
CUTOFF_TRADEOFF_PLOT = OUTPUT_DIR / "goodreads_cutoff_tradeoffs.png"
FILTER_EFFECT_CSV = OUTPUT_DIR / "goodreads_filter_effects.csv"
FILTER_EFFECT_PLOT = OUTPUT_DIR / "goodreads_filter_effects.png"
FILTER_EFFECT_ORIGINAL_SCALE_PLOT = (
    OUTPUT_DIR / "goodreads_filter_effects_original_scale.png"
)
FILTER_UTILITY_BEST_CSV = OUTPUT_DIR / "goodreads_filter_utility_best.csv"
FILTER_UTILITY_PLOT = OUTPUT_DIR / "goodreads_filter_effects_utility.png"
PRIMARY_RATING_COL = "goodreads_rating_raw_best"
PRIMARY_RANGE = (3.2, 4.8)
MIN_CATEGORY_PLOT_N = 5
MIN_BOOKSHELF_ANALYSIS_N = 10
MIN_SOURCE_ANALYSIS_N = 7
MIN_POLICY_SIDE_N = 15
BALANCED_KEEP_SHARE_RANGE = (0.25, 0.75)
UTILITY_TRANSFORMS: dict[str, dict[str, float | str]] = {
    "avg_enjoyment": {
        "kind": "power",
        "power": 1.3,
        "shift": 1.0,
        "label": "(rating - 1)^1.3",
    },
    "avg_usefulness": {
        "kind": "exponential",
        "base": 2.0,
        "shift": 1.0,
        "label": "2^(rating - 1) - 1",
    },
}
PLOT_CONFIGS = (
    ("goodreads_rating", "Goodreads rating (conservative)", "goodreads_rating"),
    ("goodreads_rating_raw", "Goodreads rating (raw)", "goodreads_rating_raw"),
    (
        "goodreads_rating_raw_best",
        "Goodreads rating (raw, rescued)",
        "goodreads_rating_raw_best",
    ),
    (
        "goodreads_rating_count",
        "log10(1 + Goodreads rating count, conservative)",
        "goodreads_rating_count",
    ),
    (
        "goodreads_rating_count_raw",
        "log10(1 + Goodreads rating count, raw)",
        "goodreads_rating_count_raw",
    ),
    (
        "goodreads_rating_count_raw_best",
        "log10(1 + Goodreads rating count, raw rescued)",
        "goodreads_rating_count_raw_best",
    ),
)
PLACEHOLDER_AUTHORS = {"", "by", "unknown", "nan", "none"}
FILE_SUFFIXES = (".pdf", ".epub", ".html", ".txt")

warnings.filterwarnings(
    "ignore",
    message="Found unknown categories in columns",
    category=UserWarning,
)


@dataclass(frozen=True)
class TargetSpec:
    column: str
    label: str
    slug: str
    gap_column: str | None = None


TARGET_SPECS = (
    TargetSpec("Enjoyment (/5)", "Enjoyment (pass 1)", "enjoyment_pass1"),
    TargetSpec("Usefulness /5 to Me", "Usefulness (pass 1)", "usefulness_pass1"),
    TargetSpec("Enjoyment (/5)_ratings2", "Enjoyment (pass 2)", "enjoyment_pass2"),
    TargetSpec(
        "Usefulness /5 to Me_ratings2",
        "Usefulness (pass 2)",
        "usefulness_pass2",
    ),
    TargetSpec(
        "avg_enjoyment",
        "Enjoyment average",
        "enjoyment_avg",
        gap_column="enjoyment_label_gap",
    ),
    TargetSpec(
        "avg_usefulness",
        "Usefulness average",
        "usefulness_avg",
        gap_column="usefulness_label_gap",
    ),
)
PRIMARY_TARGET_SPECS = tuple(
    spec for spec in TARGET_SPECS if spec.column in {"avg_enjoyment", "avg_usefulness"}
)


def load_data(path: Path = INPUT_FILE) -> pd.DataFrame:
    df = pd.read_csv(path)
    df.columns = df.columns.str.strip()
    for column in [
        "earliest_modified",
        "latest_modified",
        "earliest_modified_ratings2",
        "latest_modified_ratings2",
    ]:
        if column in df.columns:
            df[column] = pd.to_datetime(df[column], format="mixed", errors="coerce")
    return df


def normalize_text(value: object) -> str:
    if pd.isna(value):
        return ""
    return str(value).strip()


def normalize_author(value: object) -> str:
    text = normalize_text(value).lower()
    if not text or text in PLACEHOLDER_AUTHORS:
        return ""
    if re.fullmatch(r"\d{4}-\d{2}-\d{2}", text):
        return ""
    return text


def choose_canonical_author(row: pd.Series) -> str:
    for column in ["author_ratings2", "goodreads_author", "author_goodreads", "author"]:
        normalized = normalize_author(row.get(column))
        if normalized:
            return normalized
    return ""


def derive_analysis_columns(df: pd.DataFrame) -> pd.DataFrame:
    derived = df.copy()

    for column in [
        "earliest_modified",
        "latest_modified",
        "earliest_modified_ratings2",
        "latest_modified_ratings2",
    ]:
        if column in derived.columns:
            derived[column] = pd.to_datetime(
                derived[column], format="mixed", errors="coerce"
            )

    earliest = derived.get("earliest_modified")
    latest = derived.get("latest_modified")
    earliest_r2 = derived.get("earliest_modified_ratings2")
    latest_r2 = derived.get("latest_modified_ratings2")

    derived["earliest_modified_best"] = (
        earliest.combine_first(earliest_r2)
        if earliest is not None and earliest_r2 is not None
        else earliest if earliest is not None else earliest_r2
    )
    derived["latest_modified_best"] = (
        latest.combine_first(latest_r2)
        if latest is not None and latest_r2 is not None
        else latest if latest is not None else latest_r2
    )

    derived["reading_days"] = (
        derived["latest_modified_best"] - derived["earliest_modified_best"]
    ).dt.days
    derived["year_finished"] = derived["latest_modified_best"].dt.year
    derived["month_finished"] = derived["latest_modified_best"].dt.month
    derived["note_length"] = derived["Long Term Effects"].fillna("").str.len()
    derived["log_pages"] = np.log1p(
        pd.to_numeric(derived["gb_page_count"], errors="coerce")
    )
    derived["book_age"] = CURRENT_YEAR - pd.to_numeric(
        derived["pub_year"], errors="coerce"
    ).fillna(CURRENT_YEAR)
    derived["avg_enjoyment"] = derived[
        ["Enjoyment (/5)", "Enjoyment (/5)_ratings2"]
    ].mean(axis=1)
    derived["avg_usefulness"] = derived[
        ["Usefulness /5 to Me", "Usefulness /5 to Me_ratings2"]
    ].mean(axis=1)
    derived[utility_column_name("avg_enjoyment")] = transform_target_utility(
        derived["avg_enjoyment"], "avg_enjoyment"
    )
    derived[utility_column_name("avg_usefulness")] = transform_target_utility(
        derived["avg_usefulness"], "avg_usefulness"
    )
    derived["enjoyment_label_gap"] = (
        derived["Enjoyment (/5)"] - derived["Enjoyment (/5)_ratings2"]
    ).abs()
    derived["usefulness_label_gap"] = (
        derived["Usefulness /5 to Me"] - derived["Usefulness /5 to Me_ratings2"]
    ).abs()
    derived["canonical_author"] = derived.apply(choose_canonical_author, axis=1)
    if "goodreads_rating_raw_best" not in derived.columns:
        derived["goodreads_rating_raw_best"] = derived.get("goodreads_rating_raw")
    if "goodreads_rating_count_raw_best" not in derived.columns:
        derived["goodreads_rating_count_raw_best"] = derived.get(
            "goodreads_rating_count_raw"
        )
    if "goodreads_raw_best_source" not in derived.columns:
        derived["goodreads_raw_best_source"] = np.where(
            derived["goodreads_rating_raw_best"].notna(),
            "chosen_candidate",
            "missing",
        )
    return derived


def classify_missing_goodreads_reason(row: pd.Series) -> str:
    title = normalize_text(row.get("title"))
    author = normalize_author(row.get("author"))
    reasons: list[str] = []

    if not title:
        reasons.append("blank_title")
    if re.fullmatch(r"[a-z0-9]{12,}", title):
        reasons.append("opaque_id")
    if any(suffix in title.lower() for suffix in FILE_SUFFIXES):
        reasons.append("filename_or_file_suffix")
    if not author:
        reasons.append("missing_or_placeholder_author")

    token_count = len(re.findall(r"[A-Za-z0-9]+", title))
    if token_count <= 3:
        reasons.append("very_short_or_ambiguous_title")

    if not reasons:
        reasons.append("other")
    return "|".join(reasons)


def add_target_author_loo(df: pd.DataFrame, target: str) -> pd.DataFrame:
    enriched = df.copy()
    target_values = pd.to_numeric(enriched[target], errors="coerce")
    global_mean = target_values.mean()
    author_counts = enriched["canonical_author"].value_counts()

    valid_author = enriched["canonical_author"].ne("")
    author_sums = (
        enriched.loc[valid_author].groupby("canonical_author")[target].transform("sum")
    )
    author_count_series = enriched["canonical_author"].map(author_counts).fillna(0)

    loo = pd.Series(global_mean, index=enriched.index, dtype=float)
    repeat_mask = valid_author & author_count_series.gt(1) & target_values.notna()
    loo.loc[repeat_mask] = (
        author_sums.loc[repeat_mask] - target_values.loc[repeat_mask]
    ) / (author_count_series.loc[repeat_mask] - 1)

    enriched["author_target_mean_loo"] = loo.fillna(global_mean)
    enriched["author_book_count"] = author_count_series.fillna(0)
    return enriched


def spearman_summary(x: pd.Series, y: pd.Series) -> tuple[float, float]:
    if x.nunique() < 2 or y.nunique() < 2:
        return math.nan, math.nan
    rho, p_value = stats.spearmanr(x, y)
    return float(rho), float(p_value)


def utility_column_name(target_col: str) -> str:
    return f"{target_col}_utility"


def transform_target_utility(values: pd.Series, target_col: str) -> pd.Series:
    spec = UTILITY_TRANSFORMS[target_col]
    numeric = pd.to_numeric(values, errors="coerce")
    shift = float(spec["shift"])
    shifted = np.clip(numeric - shift, a_min=0, a_max=None)
    if spec["kind"] == "power":
        return pd.Series(np.power(shifted, float(spec["power"])), index=numeric.index)
    return pd.Series(
        np.power(float(spec["base"]), shifted) - 1,
        index=numeric.index,
    )


def utility_transform_label(target_col: str) -> str:
    return str(UTILITY_TRANSFORMS[target_col]["label"])


def regression_summary(x: pd.Series, y: pd.Series) -> dict[str, float]:
    if len(x) < 2 or x.nunique() < 2 or y.nunique() < 2:
        return {
            "slope": math.nan,
            "intercept": math.nan,
            "pearson_r": math.nan,
            "pearson_p": math.nan,
        }
    slope, intercept, r_value, p_value, _ = stats.linregress(x, y)
    return {
        "slope": float(slope),
        "intercept": float(intercept),
        "pearson_r": float(r_value),
        "pearson_p": float(p_value),
    }


def summarize_correlations(df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for x_col, _, _ in PLOT_CONFIGS:
        x_series = pd.to_numeric(df[x_col], errors="coerce")
        x_metric = np.log10(1 + x_series) if "count" in x_col else x_series
        for spec in TARGET_SPECS:
            subset = pd.DataFrame({"x": x_metric, "y": df[spec.column]}).dropna()
            rho, p_value = spearman_summary(subset["x"], subset["y"])
            rows.append(
                {
                    "x_column": x_col,
                    "target": spec.column,
                    "n": len(subset),
                    "spearman_rho": rho,
                    "p_value": p_value,
                    "mean_target": subset["y"].mean() if len(subset) else math.nan,
                }
            )
    result = pd.DataFrame(rows)
    result.to_csv(CORRELATION_CSV, index=False)
    return result


def linear_fit_summary(
    df: pd.DataFrame,
    x_col: str,
    y_col: str,
    x_range: tuple[float, float] | None = None,
) -> tuple[pd.DataFrame, dict[str, float]]:
    subset = df[[x_col, y_col, "Bookshelf"]].dropna().copy()
    subset[x_col] = pd.to_numeric(subset[x_col], errors="coerce")
    subset[y_col] = pd.to_numeric(subset[y_col], errors="coerce")
    subset = subset.dropna()

    if x_range is not None:
        subset = subset[subset[x_col].between(x_range[0], x_range[1])].copy()

    fit = regression_summary(subset[x_col], subset[y_col])
    if math.isnan(fit["slope"]):
        subset["predicted"] = math.nan
        subset["residual"] = math.nan
    else:
        subset["predicted"] = fit["intercept"] + fit["slope"] * subset[x_col]
        subset["residual"] = subset[y_col] - subset["predicted"]
    return subset, {**fit, "n": float(len(subset))}


def add_fit(ax: plt.Axes, x: pd.Series, y: pd.Series) -> None:
    if len(x) < 3 or x.nunique() < 2:
        return
    slope, intercept, _, _, _ = stats.linregress(x, y)
    x_line = np.linspace(float(x.min()), float(x.max()), 100)
    ax.plot(x_line, slope * x_line + intercept, color="firebrick", linewidth=1.5)


def make_target_grid(
    df: pd.DataFrame,
    x_col: str,
    x_label: str,
    output_path: Path,
) -> None:
    fig, axes = plt.subplots(3, 2, figsize=(15, 16))
    axes = axes.flatten()

    raw_x = pd.to_numeric(df[x_col], errors="coerce")
    x_values = np.log10(1 + raw_x) if "count" in x_col else raw_x

    for axis, spec in zip(axes, TARGET_SPECS, strict=False):
        subset = pd.DataFrame({"x": x_values, "y": df[spec.column]}).dropna()
        axis.scatter(subset["x"], subset["y"], alpha=0.6, s=26, color="steelblue")
        add_fit(axis, subset["x"], subset["y"])

        rho, p_value = spearman_summary(subset["x"], subset["y"])
        annotation = [
            f"n={len(subset)}",
            f"rho={rho:.2f}" if not math.isnan(rho) else "rho=NA",
            f"p={p_value:.3g}" if not math.isnan(p_value) else "p=NA",
        ]
        if spec.gap_column:
            gap = pd.to_numeric(df[spec.gap_column], errors="coerce").dropna()
            if len(gap):
                annotation.append(f"mean |pass1-pass2|={gap.mean():.2f}")

        axis.text(
            0.03,
            0.97,
            "\n".join(annotation),
            transform=axis.transAxes,
            va="top",
            fontsize=9,
            bbox={"facecolor": "white", "alpha": 0.8, "edgecolor": "none"},
        )
        axis.set_title(spec.label)
        axis.set_xlabel(x_label)
        axis.set_ylabel("My rating")
        axis.set_ylim(0.5, 5.5)
        axis.grid(alpha=0.25)

    fig.suptitle(f"{x_label} vs personal ratings", fontsize=16)
    plt.tight_layout()
    fig.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def plot_primary_restricted_range(df: pd.DataFrame) -> pd.DataFrame:
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
    rows: list[dict[str, object]] = []

    for axis, spec in zip(axes, PRIMARY_TARGET_SPECS, strict=False):
        subset, fit = linear_fit_summary(
            df,
            x_col=PRIMARY_RATING_COL,
            y_col=spec.column,
            x_range=PRIMARY_RANGE,
        )
        axis.scatter(
            subset[PRIMARY_RATING_COL],
            subset[spec.column],
            alpha=0.65,
            s=28,
            color="steelblue",
        )
        add_fit(axis, subset[PRIMARY_RATING_COL], subset[spec.column])
        rho, rho_p = spearman_summary(subset[PRIMARY_RATING_COL], subset[spec.column])
        axis.text(
            0.03,
            0.97,
            f"n={len(subset)}\nrho={rho:.2f}\np={rho_p:.3g}",
            transform=axis.transAxes,
            va="top",
            fontsize=9,
            bbox={"facecolor": "white", "alpha": 0.8, "edgecolor": "none"},
        )
        axis.set_title(spec.label)
        axis.set_xlabel("Goodreads rating (raw, rescued)")
        axis.set_ylabel("My rating")
        axis.set_xlim(*PRIMARY_RANGE)
        axis.set_ylim(0.5, 5.5)
        axis.grid(alpha=0.25)
        rows.append(
            {
                "target": spec.column,
                "range_min": PRIMARY_RANGE[0],
                "range_max": PRIMARY_RANGE[1],
                "n": len(subset),
                "spearman_rho": rho,
                "spearman_p": rho_p,
                **fit,
            }
        )

    fig.suptitle("Restricted Goodreads range: 3.2 to 4.8", fontsize=15)
    plt.tight_layout()
    fig.savefig(RESTRICTED_RANGE_PLOT, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return pd.DataFrame(rows)


def wrap_axis_label(text: str, width: int = 18) -> str:
    return textwrap.fill(str(text), width=width, break_long_words=False)


def short_source_label(text: str) -> str:
    mapping = {
        "Tyler Cowen / Marginal Revolution": "Cowen / MR",
        "Tanner Greer / Scholar's Stage": "Greer",
        "The Last Psychiatrist / Alone": "TLP / Alone",
        "Scott Alexander / SSC / ACX / LessWrong / Rationalist community": (
            "Scott / ACX / LW"
        ),
        "Twitter / X general": "Twitter / X",
        "College / school assignment": "School",
        "Self-discovered / browsing": "Self-found",
        "Classic canon / Great Books list": "Great Books",
        "Professional / work-related need": "Professional",
        "Friend recommendation": "Friend",
        "Author's other work (already read another by same author)": "Same author",
        "Unknown / can't determine": "Unknown",
        "missing": "Missing",
    }
    return mapping.get(text, wrap_axis_label(text, width=14))


def summarize_group_relationships(
    df: pd.DataFrame,
    group_col: str,
    target_specs: tuple[TargetSpec, ...],
    min_n: int,
    x_col: str = PRIMARY_RATING_COL,
    x_range: tuple[float, float] | None = None,
    range_label: str = "full",
) -> pd.DataFrame:
    frame = df[[group_col, x_col, *(spec.column for spec in target_specs)]].copy()
    frame[group_col] = frame[group_col].fillna("missing")
    frame[x_col] = pd.to_numeric(frame[x_col], errors="coerce")

    if x_range is not None:
        frame = frame[frame[x_col].between(x_range[0], x_range[1])].copy()

    rows: list[dict[str, object]] = []
    for spec in target_specs:
        pair = frame[[group_col, x_col, spec.column]].dropna().copy()
        pair[spec.column] = pd.to_numeric(pair[spec.column], errors="coerce")
        pair = pair.dropna()
        counts = pair[group_col].value_counts()

        for group_name, count in counts.items():
            if count < min_n:
                continue
            group = pair[pair[group_col] == group_name].copy()
            rho, p_value = spearman_summary(group[x_col], group[spec.column])
            fit = regression_summary(group[x_col], group[spec.column])
            rows.append(
                {
                    "group_column": group_col,
                    "group_name": group_name,
                    "target": spec.column,
                    "range_label": range_label,
                    "n": len(group),
                    "mean_goodreads": group[x_col].mean(),
                    "mean_target": group[spec.column].mean(),
                    "target_std": group[spec.column].std(ddof=0),
                    "spearman_rho": rho,
                    "p_value": p_value,
                    **fit,
                }
            )

    return pd.DataFrame(rows)


def summarize_bookshelf_relationships(df: pd.DataFrame) -> pd.DataFrame:
    full = summarize_group_relationships(
        df,
        group_col="Bookshelf",
        target_specs=PRIMARY_TARGET_SPECS,
        min_n=MIN_BOOKSHELF_ANALYSIS_N,
        range_label="full",
    )
    restricted = summarize_group_relationships(
        df,
        group_col="Bookshelf",
        target_specs=PRIMARY_TARGET_SPECS,
        min_n=MIN_BOOKSHELF_ANALYSIS_N,
        x_range=PRIMARY_RANGE,
        range_label="restricted_3.2_4.8",
    )
    combined = pd.concat([full, restricted], ignore_index=True)
    combined.to_csv(BOOKSHELF_CORRELATION_CSV, index=False)
    return combined


def plot_bookshelf_relationships(df: pd.DataFrame, summary: pd.DataFrame) -> None:
    eligible = summary[
        (summary["range_label"] == "restricted_3.2_4.8")
        & (summary["target"] == "avg_enjoyment")
    ].sort_values(["n", "group_name"], ascending=[False, True])
    categories = eligible["group_name"].tolist()
    if not categories:
        return

    fig, axes = plt.subplots(
        len(PRIMARY_TARGET_SPECS),
        len(categories),
        figsize=(4.0 * len(categories), 8.5),
        sharex=True,
        sharey="row",
    )
    axes = np.asarray(axes).reshape(len(PRIMARY_TARGET_SPECS), len(categories))

    for row_index, spec in enumerate(PRIMARY_TARGET_SPECS):
        target_summary = summary[
            (summary["range_label"] == "restricted_3.2_4.8")
            & (summary["target"] == spec.column)
        ].set_index("group_name")

        for col_index, category in enumerate(categories):
            axis = axes[row_index, col_index]
            subset = df[
                df["Bookshelf"].fillna("missing").eq(category)
                & df[PRIMARY_RATING_COL].between(PRIMARY_RANGE[0], PRIMARY_RANGE[1])
            ][[PRIMARY_RATING_COL, spec.column]].dropna()

            axis.scatter(
                subset[PRIMARY_RATING_COL],
                subset[spec.column],
                alpha=0.68,
                s=26,
                color="steelblue",
            )
            add_fit(axis, subset[PRIMARY_RATING_COL], subset[spec.column])

            row = target_summary.loc[category]
            axis.text(
                0.03,
                0.97,
                f"n={int(row['n'])}\nrho={row['spearman_rho']:.2f}",
                transform=axis.transAxes,
                va="top",
                fontsize=9,
                bbox={"facecolor": "white", "alpha": 0.82, "edgecolor": "none"},
            )
            axis.set_title(wrap_axis_label(category))
            axis.set_xlim(*PRIMARY_RANGE)
            axis.set_ylim(0.5, 5.5)
            axis.grid(alpha=0.25)
            if row_index == len(PRIMARY_TARGET_SPECS) - 1:
                axis.set_xlabel("Goodreads rating (raw, rescued)")
            if col_index == 0:
                axis.set_ylabel(spec.label)

    fig.suptitle(
        "Goodreads vs averaged personal ratings within larger bookshelf categories",
        fontsize=15,
    )
    plt.tight_layout()
    fig.savefig(BOOKSHELF_PLOT, dpi=160, bbox_inches="tight")
    plt.close(fig)


def dominant_category(series: pd.Series) -> str:
    counts = series.dropna().astype(str).value_counts()
    if counts.empty:
        return "missing"
    return str(counts.index[0])


def summarize_inferred_sources(
    df: pd.DataFrame, min_n: int = MIN_SOURCE_ANALYSIS_N
) -> pd.DataFrame:
    frame = df.copy()
    frame["inferred_source"] = frame["inferred_source"].fillna("missing")
    frame[PRIMARY_RATING_COL] = pd.to_numeric(
        frame[PRIMARY_RATING_COL], errors="coerce"
    )
    matched = frame[frame[PRIMARY_RATING_COL].notna()].copy()

    grouped = (
        matched.groupby("inferred_source")
        .agg(
            matched_goodreads_n=("title", "size"),
            mean_goodreads=(PRIMARY_RATING_COL, "mean"),
            mean_avg_enjoyment=("avg_enjoyment", "mean"),
            mean_avg_usefulness=("avg_usefulness", "mean"),
            top_bookshelf=("Bookshelf", dominant_category),
        )
        .reset_index()
    )
    total_counts = frame["inferred_source"].value_counts().rename("source_total_n")
    grouped["source_total_n"] = grouped["inferred_source"].map(total_counts).fillna(0)
    grouped = grouped[grouped["matched_goodreads_n"] >= min_n].copy()

    rows: list[dict[str, object]] = []
    for _, row in grouped.iterrows():
        source = row["inferred_source"]
        subset = matched[matched["inferred_source"] == source].copy()
        record = row.to_dict()
        for spec in PRIMARY_TARGET_SPECS:
            pair = subset[[PRIMARY_RATING_COL, spec.column]].dropna().copy()
            pair[PRIMARY_RATING_COL] = pd.to_numeric(
                pair[PRIMARY_RATING_COL], errors="coerce"
            )
            pair[spec.column] = pd.to_numeric(pair[spec.column], errors="coerce")
            pair = pair.dropna()
            rho, p_value = spearman_summary(pair[PRIMARY_RATING_COL], pair[spec.column])
            record[f"{spec.column}_rho"] = rho
            record[f"{spec.column}_p_value"] = p_value
            record[f"{spec.column}_n"] = len(pair)
        rows.append(record)

    summary = pd.DataFrame(rows).sort_values(
        ["mean_goodreads", "matched_goodreads_n"], ascending=[False, False]
    )
    summary.to_csv(SOURCE_SUMMARY_CSV, index=False)
    return summary


def plot_inferred_source_summary(summary: pd.DataFrame) -> None:
    if summary.empty:
        return

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    plot_specs = [
        ("mean_avg_enjoyment", PRIMARY_TARGET_SPECS[0].label),
        ("mean_avg_usefulness", PRIMARY_TARGET_SPECS[1].label),
    ]
    base_x = summary["mean_goodreads"]

    for axis, (target_col, label) in zip(axes, plot_specs, strict=False):
        axis.scatter(
            base_x,
            summary[target_col],
            s=25 + summary["matched_goodreads_n"] * 6,
            alpha=0.75,
            color="steelblue",
        )
        add_fit(axis, base_x, summary[target_col])
        rho, p_value = spearman_summary(base_x, summary[target_col])
        axis.text(
            0.03,
            0.97,
            f"sources={len(summary)}\nrho={rho:.2f}\np={p_value:.3g}",
            transform=axis.transAxes,
            va="top",
            fontsize=9,
            bbox={"facecolor": "white", "alpha": 0.82, "edgecolor": "none"},
        )
        for _, row in summary.iterrows():
            axis.annotate(
                short_source_label(row["inferred_source"]),
                (row["mean_goodreads"], row[target_col]),
                textcoords="offset points",
                xytext=(5, 4),
                fontsize=8,
            )
        axis.set_xlabel("Mean Goodreads rating (raw, rescued)")
        axis.set_ylabel(label)
        axis.set_title(f"Inferred source means: Goodreads vs {label.lower()}")
        axis.grid(alpha=0.25)
        axis.set_xlim(min(3.5, float(base_x.min()) - 0.05), float(base_x.max()) + 0.05)
        axis.set_ylim(0.5, 5.1)

    plt.tight_layout()
    fig.savefig(SOURCE_PLOT, dpi=160, bbox_inches="tight")
    plt.close(fig)


def simulate_filter_effects(
    df: pd.DataFrame,
    x_col: str = PRIMARY_RATING_COL,
    min_keep_n: int = MIN_POLICY_SIDE_N,
) -> pd.DataFrame:
    ratings = pd.to_numeric(df[x_col], errors="coerce")
    rows: list[dict[str, object]] = []
    thresholds = [
        round(value, 1)
        for value in np.arange(PRIMARY_RANGE[0], PRIMARY_RANGE[1] + 0.001, 0.1)
    ]

    for spec in PRIMARY_TARGET_SPECS:
        target = pd.to_numeric(df[spec.column], errors="coerce")
        subset = pd.DataFrame({"rating": ratings, "target": target}).dropna()
        if subset.empty:
            continue

        overall_rho, overall_p = spearman_summary(subset["rating"], subset["target"])
        overall_mean = subset["target"].mean()
        overall_utility = transform_target_utility(subset["target"], spec.column)
        overall_utility_mean = overall_utility.mean()
        for threshold in thresholds:
            keep = subset[subset["rating"] >= threshold].copy()
            skip = subset[subset["rating"] < threshold].copy()
            if len(keep) < min_keep_n:
                continue
            keep_rho, keep_p = spearman_summary(keep["rating"], keep["target"])
            keep_utility = transform_target_utility(keep["target"], spec.column)
            skip_utility = transform_target_utility(skip["target"], spec.column)
            rows.append(
                {
                    "target": spec.column,
                    "utility_transform": utility_transform_label(spec.column),
                    "threshold": threshold,
                    "n_total": len(subset),
                    "n_keep": len(keep),
                    "n_skip": len(skip),
                    "keep_share": len(keep) / len(subset),
                    "keep_mean": keep["target"].mean(),
                    "skip_mean": skip["target"].mean() if len(skip) else math.nan,
                    "delta_keep_vs_all": keep["target"].mean() - overall_mean,
                    "delta_keep_vs_skip": (
                        keep["target"].mean() - skip["target"].mean()
                        if len(skip)
                        else math.nan
                    ),
                    "overall_utility_mean": overall_utility_mean,
                    "keep_utility_mean": keep_utility.mean(),
                    "skip_utility_mean": skip_utility.mean() if len(skip) else math.nan,
                    "delta_keep_utility_vs_all": keep_utility.mean()
                    - overall_utility_mean,
                    "delta_keep_utility_vs_skip": (
                        keep_utility.mean() - skip_utility.mean()
                        if len(skip)
                        else math.nan
                    ),
                    "overall_spearman_rho": overall_rho,
                    "overall_spearman_p": overall_p,
                    "keep_spearman_rho": keep_rho,
                    "keep_spearman_p": keep_p,
                }
            )

    effect_df = pd.DataFrame(rows)
    effect_df.to_csv(FILTER_EFFECT_CSV, index=False)
    return effect_df


def plot_filter_effects(effect_df: pd.DataFrame) -> None:
    if effect_df.empty:
        return

    fig, axes = plt.subplots(2, 2, figsize=(14, 9), sharex="col")
    for col_index, spec in enumerate(PRIMARY_TARGET_SPECS):
        subset = effect_df[effect_df["target"] == spec.column].sort_values("threshold")
        top_axis = axes[0, col_index]
        bottom_axis = axes[1, col_index]

        top_axis.plot(
            subset["threshold"],
            subset["delta_keep_vs_all"],
            marker="o",
            color="steelblue",
            label="Lift vs overall mean",
        )
        top_axis.axhline(0, color="gray", linestyle="--", linewidth=1, label="No lift")
        share_axis = top_axis.twinx()
        share_axis.plot(
            subset["threshold"],
            subset["keep_share"],
            marker="s",
            color="darkorange",
            label="Share kept",
        )
        top_axis.axvspan(4.0, 4.5, color="lightgreen", alpha=0.15)
        top_axis.axvline(3.7, color="gray", linestyle=":", linewidth=1)
        top_axis.set_title(spec.label)
        top_axis.set_ylabel("Lift in kept mean rating")
        share_axis.set_ylabel("Share kept")
        y_min = min(float(subset["delta_keep_vs_all"].min()), 0.0)
        y_max = max(float(subset["delta_keep_vs_all"].max()), 0.0)
        y_pad = max(0.03, (y_max - y_min) * 0.15)
        top_axis.set_ylim(y_min - y_pad, y_max + y_pad)
        share_axis.set_ylim(0, 1)
        top_axis.grid(alpha=0.2)
        handles1, labels1 = top_axis.get_legend_handles_labels()
        handles2, labels2 = share_axis.get_legend_handles_labels()
        top_axis.legend(handles1 + handles2, labels1 + labels2, loc="lower right")

        bottom_axis.plot(
            subset["threshold"],
            subset["keep_spearman_rho"],
            marker="o",
            color="seagreen",
            label="Kept-subset rho",
        )
        n_axis = bottom_axis.twinx()
        n_axis.plot(
            subset["threshold"],
            subset["n_keep"],
            marker="s",
            color="mediumpurple",
            label="Kept n",
        )
        bottom_axis.axhline(
            subset["overall_spearman_rho"].iloc[0],
            color="gray",
            linestyle="--",
            linewidth=1,
            label="Overall rho",
        )
        bottom_axis.axvspan(4.0, 4.5, color="lightgreen", alpha=0.15)
        bottom_axis.axvline(3.7, color="gray", linestyle=":", linewidth=1)
        bottom_axis.set_xlabel("Goodreads cutoff: keep if rating >= threshold")
        bottom_axis.set_ylabel("Spearman rho")
        n_axis.set_ylabel("Books kept")
        bottom_axis.set_ylim(-0.1, 0.8)
        bottom_axis.grid(alpha=0.2)
        handles1, labels1 = bottom_axis.get_legend_handles_labels()
        handles2, labels2 = n_axis.get_legend_handles_labels()
        bottom_axis.legend(handles1 + handles2, labels1 + labels2, loc="upper left")

    plt.tight_layout()
    fig.savefig(FILTER_EFFECT_PLOT, dpi=160, bbox_inches="tight")
    plt.close(fig)


def plot_filter_effects_original_scale(effect_df: pd.DataFrame) -> None:
    if effect_df.empty:
        return

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
    for axis, spec in zip(axes, PRIMARY_TARGET_SPECS, strict=False):
        subset = effect_df[effect_df["target"] == spec.column].sort_values("threshold")
        axis.plot(
            subset["threshold"],
            subset["keep_mean"],
            marker="o",
            color="steelblue",
            label="Mean if kept",
        )
        axis.axhline(
            subset["keep_mean"].iloc[0] - subset["delta_keep_vs_all"].iloc[0],
            color="gray",
            linestyle="--",
            linewidth=1,
            label="Overall mean",
        )
        share_axis = axis.twinx()
        share_axis.plot(
            subset["threshold"],
            subset["keep_share"],
            marker="s",
            color="darkorange",
            label="Share kept",
        )
        axis.axvspan(4.0, 4.5, color="lightgreen", alpha=0.15)
        axis.axvline(3.7, color="gray", linestyle=":", linewidth=1)
        axis.set_title(f"{spec.label} on original scale")
        axis.set_xlabel("Goodreads cutoff: keep if rating >= threshold")
        axis.set_ylabel("Average kept rating")
        share_axis.set_ylabel("Share kept")
        axis.set_ylim(0.5, 5.1)
        share_axis.set_ylim(0, 1)
        axis.grid(alpha=0.2)
        handles1, labels1 = axis.get_legend_handles_labels()
        handles2, labels2 = share_axis.get_legend_handles_labels()
        axis.legend(handles1 + handles2, labels1 + labels2, loc="lower right")

    plt.tight_layout()
    fig.savefig(FILTER_EFFECT_ORIGINAL_SCALE_PLOT, dpi=160, bbox_inches="tight")
    plt.close(fig)


def summarize_best_utility_filters(effect_df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for spec in PRIMARY_TARGET_SPECS:
        sub = effect_df[effect_df["target"] == spec.column].copy()
        if sub.empty:
            continue
        rows.append(
            {
                "target": spec.column,
                "selection": "best_overall_utility",
                **sub.sort_values("delta_keep_utility_vs_all", ascending=False)
                .iloc[0]
                .to_dict(),
            }
        )
        balanced = sub[
            sub["keep_share"].between(
                BALANCED_KEEP_SHARE_RANGE[0], BALANCED_KEEP_SHARE_RANGE[1]
            )
        ].copy()
        if not balanced.empty:
            rows.append(
                {
                    "target": spec.column,
                    "selection": "best_balanced_utility",
                    **balanced.sort_values("delta_keep_utility_vs_all", ascending=False)
                    .iloc[0]
                    .to_dict(),
                }
            )
    best = pd.DataFrame(rows)
    best.to_csv(FILTER_UTILITY_BEST_CSV, index=False)
    return best


def plot_filter_effects_utility(effect_df: pd.DataFrame) -> None:
    if effect_df.empty:
        return

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
    for axis, spec in zip(axes, PRIMARY_TARGET_SPECS, strict=False):
        subset = effect_df[effect_df["target"] == spec.column].sort_values("threshold")
        axis.plot(
            subset["threshold"],
            subset["delta_keep_utility_vs_all"],
            marker="o",
            color="firebrick",
            label="Utility lift vs overall",
        )
        share_axis = axis.twinx()
        share_axis.plot(
            subset["threshold"],
            subset["keep_share"],
            marker="s",
            color="darkorange",
            label="Share kept",
        )
        axis.axhline(0, color="gray", linestyle="--", linewidth=1)
        axis.axvspan(4.0, 4.5, color="lightgreen", alpha=0.15)
        axis.axvline(3.7, color="gray", linestyle=":", linewidth=1)
        axis.set_title(f"{spec.label} utility")
        axis.set_xlabel("Goodreads cutoff: keep if rating >= threshold")
        axis.set_ylabel(f"Utility lift ({utility_transform_label(spec.column)})")
        share_axis.set_ylabel("Share kept")
        y_min = min(float(subset["delta_keep_utility_vs_all"].min()), 0.0)
        y_max = max(float(subset["delta_keep_utility_vs_all"].max()), 0.0)
        y_pad = max(0.05, (y_max - y_min) * 0.15)
        axis.set_ylim(y_min - y_pad, y_max + y_pad)
        share_axis.set_ylim(0, 1)
        axis.grid(alpha=0.2)
        handles1, labels1 = axis.get_legend_handles_labels()
        handles2, labels2 = share_axis.get_legend_handles_labels()
        axis.legend(handles1 + handles2, labels1 + labels2, loc="upper left")

    plt.tight_layout()
    fig.savefig(FILTER_UTILITY_PLOT, dpi=160, bbox_inches="tight")
    plt.close(fig)


def summarize_residuals_by_category(
    df: pd.DataFrame,
    x_range: tuple[float, float] | None = None,
    label: str = "full",
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for spec in PRIMARY_TARGET_SPECS:
        subset, fit = linear_fit_summary(
            df, x_col=PRIMARY_RATING_COL, y_col=spec.column, x_range=x_range
        )
        grouped = (
            subset.groupby("Bookshelf")
            .agg(
                n=("residual", "size"),
                mean_residual=("residual", "mean"),
                median_residual=("residual", "median"),
                mean_actual=(spec.column, "mean"),
                mean_predicted=("predicted", "mean"),
            )
            .reset_index()
        )
        grouped["target"] = spec.column
        grouped["range_label"] = label
        grouped["slope"] = fit["slope"]
        grouped["pearson_r"] = fit["pearson_r"]
        rows.extend(grouped.to_dict("records"))
    return pd.DataFrame(rows)


def plot_residuals_by_category(df: pd.DataFrame) -> pd.DataFrame:
    full_summary = summarize_residuals_by_category(df, x_range=None, label="full")
    restricted_summary = summarize_residuals_by_category(
        df, x_range=PRIMARY_RANGE, label="restricted_3.2_4.8"
    )
    combined = pd.concat([full_summary, restricted_summary], ignore_index=True)
    combined.to_csv(RESIDUAL_SUMMARY_CSV, index=False)

    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    plot_plan = [
        ("full", PRIMARY_TARGET_SPECS[0], axes[0, 0]),
        ("full", PRIMARY_TARGET_SPECS[1], axes[0, 1]),
        ("restricted_3.2_4.8", PRIMARY_TARGET_SPECS[0], axes[1, 0]),
        ("restricted_3.2_4.8", PRIMARY_TARGET_SPECS[1], axes[1, 1]),
    ]

    for range_label, spec, axis in plot_plan:
        subset, fit = linear_fit_summary(
            df,
            x_col=PRIMARY_RATING_COL,
            y_col=spec.column,
            x_range=PRIMARY_RANGE if range_label != "full" else None,
        )
        category_order = (
            subset.groupby("Bookshelf")
            .size()
            .loc[lambda series: series >= MIN_CATEGORY_PLOT_N]
            .sort_values(ascending=False)
            .index.tolist()
        )
        box_data = [
            subset.loc[subset["Bookshelf"] == category, "residual"].values
            for category in category_order
        ]
        axis.boxplot(box_data, tick_labels=category_order, patch_artist=True)
        axis.axhline(0, color="firebrick", linewidth=1.2, linestyle="--")
        axis.set_title(
            f"{spec.label} residuals ({'full range' if range_label == 'full' else '3.2-4.8'})"
        )
        axis.set_ylabel("Residual: actual - fitted")
        axis.tick_params(axis="x", rotation=20)
        axis.grid(alpha=0.2, axis="y")
        axis.text(
            0.02,
            0.95,
            f"n={len(subset)}\nslope={fit['slope']:.2f}\nr={fit['pearson_r']:.2f}",
            transform=axis.transAxes,
            va="top",
            fontsize=9,
            bbox={"facecolor": "white", "alpha": 0.8, "edgecolor": "none"},
        )

    plt.tight_layout()
    fig.savefig(RESIDUAL_PLOT, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return combined


def missing_match_audit(
    df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    missing = df[df["goodreads_rating_raw_best"].isna()].copy()
    missing["missing_reason"] = missing.apply(classify_missing_goodreads_reason, axis=1)

    missing.to_csv(MISSING_AUDIT_CSV, index=False)
    rescued = df[df["goodreads_raw_best_source"] == "rescued_candidate"].copy()
    rescued.to_csv(RESCUE_AUDIT_CSV, index=False)

    with_dates = df.dropna(subset=["year_finished"]).copy()
    yearly = (
        with_dates.assign(missing_raw=with_dates["goodreads_rating_raw_best"].isna())
        .groupby("year_finished")
        .agg(n_books=("title", "count"), missing_books=("missing_raw", "sum"))
        .reset_index()
    )
    yearly["missing_rate"] = yearly["missing_books"] / yearly["n_books"]
    yearly.to_csv(MISSING_YEAR_CSV, index=False)
    return missing, yearly, rescued


def plot_missing_match_by_year(yearly: pd.DataFrame) -> None:
    fig, ax1 = plt.subplots(figsize=(10, 5.5))
    ax2 = ax1.twinx()

    years = yearly["year_finished"].astype(int)
    ax1.bar(years, yearly["n_books"], color="lightgray", alpha=0.8, label="All books")
    ax1.bar(
        years,
        yearly["missing_books"],
        color="tomato",
        alpha=0.8,
        label="No Goodreads best-raw rating",
    )
    ax2.plot(
        years,
        yearly["missing_rate"],
        color="navy",
        marker="o",
        linewidth=2,
        label="Missing rate",
    )

    ax1.set_xlabel("Finished year")
    ax1.set_ylabel("Book count")
    ax2.set_ylabel("Missing Goodreads raw rate")
    ax2.set_ylim(0, max(0.2, yearly["missing_rate"].max() * 1.2))
    ax1.set_title("Missing Goodreads best-raw matches by year finished")
    ax1.grid(alpha=0.2, axis="y")

    handles1, labels1 = ax1.get_legend_handles_labels()
    handles2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(handles1 + handles2, labels1 + labels2, loc="upper right")

    plt.tight_layout()
    fig.savefig(MISSING_YEAR_PLOT, dpi=160, bbox_inches="tight")
    plt.close(fig)


def sweep_cutoff_policies(
    df: pd.DataFrame,
    target_col: str,
    x_col: str = PRIMARY_RATING_COL,
) -> pd.DataFrame:
    subset = df[[x_col, target_col]].dropna().copy()
    subset[x_col] = pd.to_numeric(subset[x_col], errors="coerce")
    subset[target_col] = pd.to_numeric(subset[target_col], errors="coerce")
    subset = subset.dropna()
    overall_mean = subset[target_col].mean()

    rows: list[dict[str, object]] = []
    thresholds = [
        round(value, 1)
        for value in np.arange(PRIMARY_RANGE[0], PRIMARY_RANGE[1] + 0.001, 0.1)
    ]

    for threshold in thresholds:
        keep = subset[subset[x_col] >= threshold]
        skip = subset[subset[x_col] < threshold]
        if len(keep) < MIN_POLICY_SIDE_N or len(skip) < MIN_POLICY_SIDE_N:
            continue
        rows.append(
            {
                "target": target_col,
                "rule": "keep_if_ge",
                "threshold_lo": threshold,
                "threshold_hi": math.nan,
                "n_keep": len(keep),
                "n_skip": len(skip),
                "keep_share": len(keep) / len(subset),
                "keep_mean": keep[target_col].mean(),
                "skip_mean": skip[target_col].mean(),
                "delta_keep_vs_all": keep[target_col].mean() - overall_mean,
                "delta_keep_vs_skip": keep[target_col].mean() - skip[target_col].mean(),
            }
        )

    for threshold_lo in thresholds:
        for threshold_hi in thresholds:
            if threshold_hi <= threshold_lo:
                continue
            keep = subset[subset[x_col].between(threshold_lo, threshold_hi)]
            skip = subset[~subset[x_col].between(threshold_lo, threshold_hi)]
            if len(keep) < MIN_POLICY_SIDE_N or len(skip) < MIN_POLICY_SIDE_N:
                continue
            rows.append(
                {
                    "target": target_col,
                    "rule": "keep_if_between",
                    "threshold_lo": threshold_lo,
                    "threshold_hi": threshold_hi,
                    "n_keep": len(keep),
                    "n_skip": len(skip),
                    "keep_share": len(keep) / len(subset),
                    "keep_mean": keep[target_col].mean(),
                    "skip_mean": skip[target_col].mean(),
                    "delta_keep_vs_all": keep[target_col].mean() - overall_mean,
                    "delta_keep_vs_skip": keep[target_col].mean()
                    - skip[target_col].mean(),
                }
            )

    return pd.DataFrame(rows)


def evaluate_cutoff_policies(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    sweeps = []
    for spec in PRIMARY_TARGET_SPECS:
        sweeps.append(sweep_cutoff_policies(df, target_col=spec.column))
    sweep_df = pd.concat(sweeps, ignore_index=True)
    sweep_df.to_csv(CUTOFF_SWEEP_CSV, index=False)

    best_rows: list[dict[str, object]] = []
    for spec in PRIMARY_TARGET_SPECS:
        for rule in ["keep_if_ge", "keep_if_between"]:
            sub = sweep_df[
                (sweep_df["target"] == spec.column) & (sweep_df["rule"] == rule)
            ].copy()
            if sub.empty:
                continue
            best_rows.append(
                {
                    "target": spec.column,
                    "selection": "best_overall",
                    **sub.sort_values("delta_keep_vs_all", ascending=False)
                    .iloc[0]
                    .to_dict(),
                }
            )
            balanced = sub[
                sub["keep_share"].between(
                    BALANCED_KEEP_SHARE_RANGE[0], BALANCED_KEEP_SHARE_RANGE[1]
                )
            ].copy()
            if not balanced.empty:
                best_rows.append(
                    {
                        "target": spec.column,
                        "selection": "best_balanced",
                        **balanced.sort_values("delta_keep_vs_all", ascending=False)
                        .iloc[0]
                        .to_dict(),
                    }
                )

    best_df = pd.DataFrame(best_rows)
    best_df.to_csv(CUTOFF_BEST_CSV, index=False)
    return sweep_df, best_df


def plot_cutoff_tradeoffs(sweep_df: pd.DataFrame) -> None:
    ge_only = sweep_df[sweep_df["rule"] == "keep_if_ge"].copy()
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))

    for axis, spec in zip(axes, PRIMARY_TARGET_SPECS, strict=False):
        subset = ge_only[ge_only["target"] == spec.column].sort_values("threshold_lo")
        axis.plot(
            subset["threshold_lo"],
            subset["keep_mean"],
            marker="o",
            color="steelblue",
            label="Mean if kept",
        )
        axis2 = axis.twinx()
        axis2.plot(
            subset["threshold_lo"],
            subset["keep_share"],
            marker="s",
            color="darkorange",
            label="Share kept",
        )
        axis.axvspan(4.0, 4.5, color="lightgreen", alpha=0.15)
        axis.axvline(3.7, color="gray", linestyle="--", linewidth=1)
        axis.set_title(spec.label)
        axis.set_xlabel("Goodreads cutoff: keep if rating >= threshold")
        axis.set_ylabel("Average kept rating")
        axis2.set_ylabel("Share kept")
        axis.set_ylim(0.5, 5.1)
        axis2.set_ylim(0, 1)
        axis.grid(alpha=0.2)
        handles1, labels1 = axis.get_legend_handles_labels()
        handles2, labels2 = axis2.get_legend_handles_labels()
        axis.legend(handles1 + handles2, labels1 + labels2, loc="lower right")

    plt.tight_layout()
    fig.savefig(CUTOFF_TRADEOFF_PLOT, dpi=160, bbox_inches="tight")
    plt.close(fig)


def category_mean_predictions(df: pd.DataFrame, target: str) -> np.ndarray:
    values = pd.to_numeric(df[target], errors="coerce").values
    global_mean = np.nanmean(values)
    predictions = np.zeros(len(df))
    for index in range(len(df)):
        category = df.iloc[index]["Bookshelf"]
        other_rows = df.drop(df.index[index])
        category_mean = other_rows.loc[
            other_rows["Bookshelf"] == category, target
        ].mean()
        predictions[index] = (
            category_mean if not np.isnan(category_mean) else global_mean
        )
    return predictions


def build_feature_lists(
    frame: pd.DataFrame,
    rating_col: str,
    count_col: str,
    include_goodreads: bool,
) -> tuple[list[str], list[str]]:
    numeric = [
        "year_finished",
        "reading_days",
        "note_length",
        "log_pages",
        "book_age",
        "author_target_mean_loo",
        "author_book_count",
    ]
    categorical = ["Bookshelf", "inferred_source"]

    if include_goodreads:
        numeric.append(rating_col)
        frame[f"log_{count_col}"] = np.log10(
            1 + pd.to_numeric(frame[count_col], errors="coerce").fillna(0)
        )
        numeric.append(f"log_{count_col}")

    numeric = [column for column in numeric if column in frame.columns]
    categorical = [column for column in categorical if column in frame.columns]
    return numeric, categorical


def cross_validated_metrics(
    df: pd.DataFrame,
    target: str,
    numeric_features: list[str],
    categorical_features: list[str],
    model_name: str,
    model,
) -> dict[str, object]:
    frame = df.copy()
    for column in numeric_features:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
        frame[column] = frame[column].fillna(frame[column].median())
    for column in categorical_features:
        frame[column] = frame[column].fillna("Unknown")

    y = pd.to_numeric(frame[target], errors="coerce").values
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numeric_features),
            (
                "cat",
                OneHotEncoder(
                    drop="first",
                    sparse_output=False,
                    handle_unknown="infrequent_if_exist",
                ),
                categorical_features,
            ),
        ]
    )

    pipeline = Pipeline([("prep", preprocessor), ("model", model)])
    predictions = cross_val_predict(pipeline, frame, y, cv=LeaveOneOut())
    mae = mean_absolute_error(y, predictions)
    rmse = math.sqrt(mean_squared_error(y, predictions))
    rho, p_value = spearman_summary(pd.Series(predictions), pd.Series(y))
    return {
        "model": model_name,
        "mae": mae,
        "rmse": rmse,
        "spearman_rho": rho,
        "p_value": p_value,
    }


def evaluate_models(df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    configs = (
        ("goodreads", "goodreads_rating", "goodreads_rating_count"),
        ("goodreads_raw", "goodreads_rating_raw", "goodreads_rating_count_raw"),
        (
            "goodreads_raw_best",
            "goodreads_rating_raw_best",
            "goodreads_rating_count_raw_best",
        ),
    )

    for subset_name, rating_col, count_col in configs:
        subset = df[df[rating_col].notna()].copy()
        if len(subset) < 25:
            continue

        for spec in TARGET_SPECS:
            target_frame = subset[subset[spec.column].notna()].copy()
            if len(target_frame) < 25:
                continue

            target_frame = add_target_author_loo(target_frame, spec.column)
            cat_predictions = category_mean_predictions(target_frame, spec.column)
            y = pd.to_numeric(target_frame[spec.column], errors="coerce").values
            cat_mae = mean_absolute_error(y, cat_predictions)
            cat_rmse = math.sqrt(mean_squared_error(y, cat_predictions))
            rows.append(
                {
                    "subset": subset_name,
                    "target": spec.column,
                    "model": "Category mean LOO",
                    "n": len(target_frame),
                    "mae": cat_mae,
                    "rmse": cat_rmse,
                    "spearman_rho": spearman_summary(
                        pd.Series(cat_predictions), pd.Series(y)
                    )[0],
                    "p_value": spearman_summary(
                        pd.Series(cat_predictions), pd.Series(y)
                    )[1],
                    "improvement_vs_category_mae_pct": 0.0,
                }
            )

            for include_goodreads, model_name, model in (
                (False, "Ridge base", Ridge(alpha=10.0)),
                (True, "Ridge base + Goodreads", Ridge(alpha=10.0)),
                (
                    True,
                    "GBM base + Goodreads",
                    GradientBoostingRegressor(
                        n_estimators=100,
                        learning_rate=0.05,
                        max_depth=3,
                        random_state=42,
                    ),
                ),
            ):
                frame = target_frame.copy()
                numeric_features, categorical_features = build_feature_lists(
                    frame, rating_col, count_col, include_goodreads=include_goodreads
                )
                metrics = cross_validated_metrics(
                    frame,
                    spec.column,
                    numeric_features,
                    categorical_features,
                    model_name,
                    model,
                )
                rows.append(
                    {
                        "subset": subset_name,
                        "target": spec.column,
                        "n": len(frame),
                        **metrics,
                        "improvement_vs_category_mae_pct": (
                            (cat_mae - metrics["mae"]) / cat_mae * 100
                        ),
                    }
                )

    results = pd.DataFrame(rows)
    results.to_csv(MODEL_RESULTS_CSV, index=False)

    best = (
        results.sort_values(["subset", "target", "mae"])
        .groupby(["subset", "target"], as_index=False)
        .first()
    )
    best.to_csv(BEST_MODELS_CSV, index=False)
    return results


def print_summary(
    correlations: pd.DataFrame,
    restricted: pd.DataFrame,
    bookshelf_summary: pd.DataFrame,
    source_summary: pd.DataFrame,
    residuals: pd.DataFrame,
    cutoff_best: pd.DataFrame,
    filter_effects: pd.DataFrame,
    utility_best: pd.DataFrame,
    missing: pd.DataFrame,
    yearly: pd.DataFrame,
    rescued: pd.DataFrame,
    models: pd.DataFrame,
) -> None:
    print("=" * 80)
    print("GOODREADS FOLLOW-UP")
    print("=" * 80)

    for x_col in [
        "goodreads_rating",
        "goodreads_rating_raw",
        "goodreads_rating_raw_best",
    ]:
        subset = correlations[correlations["x_column"] == x_col].copy()
        subset = subset.sort_values("spearman_rho", ascending=False)
        print(f"\nTop correlations for {x_col}:")
        print(
            subset[["target", "n", "spearman_rho", "p_value"]]
            .head(6)
            .to_string(index=False)
        )

    print("\nRestricted to Goodreads ratings 3.2-4.8:")
    print(
        restricted[
            ["target", "n", "spearman_rho", "spearman_p", "slope", "pearson_r"]
        ].to_string(index=False)
    )

    restricted_bookshelf = bookshelf_summary[
        bookshelf_summary["range_label"] == "restricted_3.2_4.8"
    ].copy()
    if not restricted_bookshelf.empty:
        print("\nWithin-bookshelf Goodreads correlations (restricted 3.2-4.8):")
        print(
            restricted_bookshelf[
                ["target", "group_name", "n", "spearman_rho", "p_value", "mean_target"]
            ]
            .sort_values(["target", "spearman_rho"], ascending=[True, False])
            .to_string(index=False)
        )

    if not source_summary.empty:
        print("\nInferred source means on Goodreads-matched subset:")
        print(
            source_summary[
                [
                    "inferred_source",
                    "matched_goodreads_n",
                    "mean_goodreads",
                    "mean_avg_enjoyment",
                    "mean_avg_usefulness",
                    "top_bookshelf",
                ]
            ].to_string(index=False)
        )

    print("\nResiduals by category (mean residual, restricted 3.2-4.8):")
    restricted_residuals = residuals[
        residuals["range_label"] == "restricted_3.2_4.8"
    ].copy()
    print(
        restricted_residuals[
            ["target", "Bookshelf", "n", "mean_residual", "median_residual"]
        ]
        .sort_values(["target", "mean_residual"], ascending=[True, False])
        .to_string(index=False)
    )

    print("\nMissing-match heuristics:")
    print(missing["missing_reason"].value_counts().head(10).to_string())
    print(f"\nRescued raw-rating rows: {len(rescued)}")
    if len(rescued):
        print(
            rescued[
                [
                    "title",
                    "goodreads_title",
                    "goodreads_title_raw_best",
                    "goodreads_rating_raw_best",
                    "goodreads_rating_count_raw_best",
                ]
            ]
            .head(10)
            .to_string(index=False)
        )

    if not cutoff_best.empty:
        print("\nBest cutoff policies:")
        print(
            cutoff_best[
                [
                    "target",
                    "selection",
                    "rule",
                    "threshold_lo",
                    "threshold_hi",
                    "n_keep",
                    "keep_share",
                    "keep_mean",
                    "delta_keep_vs_all",
                    "delta_keep_vs_skip",
                ]
            ].to_string(index=False)
        )

    if not filter_effects.empty:
        print("\nFiltering simulation at key Goodreads cutoffs:")
        print(
            filter_effects[filter_effects["threshold"].isin([3.7, 4.0, 4.2, 4.5])][
                [
                    "target",
                    "utility_transform",
                    "threshold",
                    "n_keep",
                    "keep_share",
                    "keep_mean",
                    "delta_keep_vs_all",
                    "keep_utility_mean",
                    "delta_keep_utility_vs_all",
                    "keep_spearman_rho",
                ]
            ].to_string(index=False)
        )

    if not utility_best.empty:
        print("\nBest utility-adjusted Goodreads cutoffs:")
        print(
            utility_best[
                [
                    "target",
                    "selection",
                    "utility_transform",
                    "threshold",
                    "n_keep",
                    "keep_share",
                    "keep_mean",
                    "keep_utility_mean",
                    "delta_keep_utility_vs_all",
                ]
            ].to_string(index=False)
        )

    if not yearly.empty:
        year_rho, year_p = spearman_summary(
            yearly["year_finished"], yearly["missing_rate"]
        )
        print("\nMissing-rate vs year finished:" f" rho={year_rho:.3f}, p={year_p:.3f}")
        print(yearly.to_string(index=False))

    if not models.empty:
        print("\nBest model per subset/target:")
        best = (
            models.sort_values(["subset", "target", "mae"])
            .groupby(["subset", "target"], as_index=False)
            .first()
        )
        print(
            best[
                [
                    "subset",
                    "target",
                    "model",
                    "n",
                    "mae",
                    "rmse",
                    "improvement_vs_category_mae_pct",
                ]
            ].to_string(index=False)
        )


def main() -> None:
    df = derive_analysis_columns(load_data())

    for x_col, x_label, slug in PLOT_CONFIGS:
        make_target_grid(
            df,
            x_col=x_col,
            x_label=x_label,
            output_path=OUTPUT_DIR / f"{slug}_vs_personal_ratings.png",
        )

    missing, yearly, rescued = missing_match_audit(df)
    plot_missing_match_by_year(yearly)
    correlations = summarize_correlations(df)
    restricted = plot_primary_restricted_range(df)
    bookshelf_summary = summarize_bookshelf_relationships(df)
    plot_bookshelf_relationships(df, bookshelf_summary)
    source_summary = summarize_inferred_sources(df)
    plot_inferred_source_summary(source_summary)
    residuals = plot_residuals_by_category(df)
    cutoff_sweep, cutoff_best = evaluate_cutoff_policies(df)
    plot_cutoff_tradeoffs(cutoff_sweep)
    filter_effects = simulate_filter_effects(df)
    plot_filter_effects(filter_effects)
    plot_filter_effects_original_scale(filter_effects)
    utility_best = summarize_best_utility_filters(filter_effects)
    plot_filter_effects_utility(filter_effects)
    models = evaluate_models(df)
    print_summary(
        correlations,
        restricted,
        bookshelf_summary,
        source_summary,
        residuals,
        cutoff_best,
        filter_effects,
        utility_best,
        missing,
        yearly,
        rescued,
        models,
    )


if __name__ == "__main__":
    main()
