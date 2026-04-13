from __future__ import annotations

import difflib
import html
import json
import math
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

os.environ.setdefault(
    "MPLCONFIGDIR",
    str(Path(__file__).resolve().parents[1] / "data" / "mpl_cache"),
)

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import accuracy_score, mean_absolute_error, mean_squared_error
from sklearn.model_selection import RepeatedKFold, RepeatedStratifiedKFold

matplotlib.use("Agg")

ENTRY_PATTERN = re.compile(r"^(?P<title>.+?)\s+(?P<rating>\d+)\s*;\s*(?P<notes>.*)$")
YEAR_PATTERN = re.compile(r"\b(19\d{2}|20\d{2})\b")
TOKEN_PATTERN = re.compile(r"[a-z0-9]+")
SEARCH_SECTION_PATTERN = re.compile(
    r'<search-page-result skeleton="panel"'
    r' type="(?P<media_type>movie|tvSeries)"[^>]*>(?P<body>.*?)</search-page-result>',
    re.DOTALL,
)
SEARCH_ROW_PATTERN = re.compile(
    r"<search-page-media-row(?P<attrs>.*?)>"
    r'.*?<a href="(?P<href>https://www\.rottentomatoes\.com/[^"]+)" class="unset"'
    r' data-qa="thumbnail-link" slot="thumbnail">'
    r'.*?<a href="https://www\.rottentomatoes\.com/[^"]+" class="unset" data-qa="info-name"'
    r' slot="title">\s*(?P<title>.*?)\s*</a>',
    re.DOTALL,
)
JSON_SCRIPT_PATTERN = re.compile(
    r'<script[^>]+data-json="(?P<name>[^"]+)"[^>]*>(?P<body>.*?)</script>',
    re.DOTALL,
)

STOPWORDS = {
    "a",
    "an",
    "and",
    "as",
    "at",
    "by",
    "for",
    "from",
    "in",
    "of",
    "on",
    "the",
    "to",
    "vs",
}
REQUEST_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/134.0.0.0 Safari/537.36"
    )
}
LIKE_THRESHOLD = 7.0
MATCH_OVERRIDES: dict[str, dict[str, str]] = {
    "Trainspotters": {"query": "Trainspotting"},
    "Pi": {"query": "Pi 1998"},
    "The inspectors Call": {"query": "An Inspector Calls 1954"},
    "Justice League The Synder Cut": {"query": "Zack Snyder's Justice League"},
    "Batman vs Superman: Dawn of Justice": {
        "query": "Batman v Superman: Dawn of Justice"
    },
    "The Watchmen": {"query": "Watchmen"},
    "Space Odyssey 2001": {"query": "2001: A Space Odyssey"},
    "Everything Everywhere all at Once": {"query": "Everything Everywhere All at Once"},
    "The first and last men": {"query": "Last and First Men"},
    "The 5th element": {"query": "The Fifth Element"},
    "Max Max (1980)": {"query": "Mad Max 1979"},
    "A Lover and a Gentleman": {"query": "An Officer and a Gentleman"},
    "children of men": {"query": "Children of Men"},
    "the league of extraordinary gentlemen": {
        "query": "The League of Extraordinary Gentlemen"
    },
    "The outrun": {"query": "The Outrun"},
    "Rocky I": {"query": "Rocky"},
    "Rocky VI": {"query": "Rocky Balboa"},
    "The Edge (wilderness, grit)": {"url": "https://www.rottentomatoes.com/m/edge"},
    "Tropa de Elite (brazilian police squad, lead to support of Bolsonaro)": {
        "query": "Elite Squad"
    },
    "Untouchables": {"query": "The Untouchables"},
    "I love you man": {"query": "I Love You, Man"},
    "Kill Bill I": {"query": "Kill Bill: Vol. 1"},
    "Kill Bill II": {"query": "Kill Bill: Vol. 2"},
    "Ghost in the Shell (2018)": {"query": "Ghost in the Shell 2017"},
    "Tyson (documentary)": {"url": "https://www.rottentomatoes.com/m/1208128-tyson"},
    "Megapolis": {"query": "Megalopolis"},
    "Dr Stangelove": {"url": "https://www.rottentomatoes.com/m/dr_strangelove"},
    "The 5th element (hanson, thought it was like star wars)": {
        "url": "https://www.rottentomatoes.com/m/fifth_element"
    },
    "Generation Kill (10 episodes about invasion of Iraq, most accurate tv series about military per Richards Brother)": {
        "url": "https://www.rottentomatoes.com/tv/generation_kill"
    },
    "Go": {"url": "https://www.rottentomatoes.com/m/1087053-go"},
}


@dataclass(frozen=True)
class MovieEntry:
    index: int
    title: str
    rating: float
    notes: str


@dataclass(frozen=True)
class SearchCandidate:
    media_type: str
    title: str
    url: str
    release_year: int | None
    cast: tuple[str, ...]
    tomatometer_score: int | None


@dataclass(frozen=True)
class MatchResult:
    entry: MovieEntry
    matched_title: str
    media_type: str
    rt_url: str
    critic_score: int | None
    audience_score: int | None
    release_year: int | None
    match_confidence: float
    query: str


def parse_movie_entries(raw_text: str) -> list[MovieEntry]:
    entries: list[MovieEntry] = []
    current_title: str | None = None
    current_rating: float | None = None
    current_notes: list[str] = []

    def flush() -> None:
        if current_title is None or current_rating is None:
            return
        notes = "\n".join(line for line in current_notes if line).strip()
        entries.append(
            MovieEntry(
                index=len(entries),
                title=current_title.strip(),
                rating=float(current_rating),
                notes=notes,
            )
        )

    for raw_line in raw_text.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        match = ENTRY_PATTERN.match(line)
        if match:
            flush()
            current_title = match.group("title").strip()
            current_rating = float(match.group("rating"))
            current_notes = [match.group("notes").strip()]
            continue
        if current_title is not None:
            current_notes.append(line)

    flush()
    return entries


def _extract_attr(attrs: str, attr_name: str) -> str | None:
    match = re.search(rf'\b{re.escape(attr_name)}="([^"]*)"', attrs)
    if not match:
        return None
    return html.unescape(match.group(1).strip())


def parse_search_results(search_html: str) -> list[SearchCandidate]:
    candidates: list[SearchCandidate] = []
    for section_match in SEARCH_SECTION_PATTERN.finditer(search_html):
        media_type = section_match.group("media_type")
        body = section_match.group("body")
        for row_match in SEARCH_ROW_PATTERN.finditer(body):
            attrs = row_match.group("attrs")
            year_text = _extract_attr(attrs, "release-year")
            tomatometer_text = _extract_attr(attrs, "tomatometer-score")
            cast_text = _extract_attr(attrs, "cast") or ""
            candidates.append(
                SearchCandidate(
                    media_type=media_type,
                    title=html.unescape(" ".join(row_match.group("title").split())),
                    url=row_match.group("href"),
                    release_year=int(year_text) if year_text else None,
                    cast=tuple(
                        part.strip() for part in cast_text.split(",") if part.strip()
                    ),
                    tomatometer_score=(
                        int(tomatometer_text) if tomatometer_text else None
                    ),
                )
            )
    return candidates


def parse_reviews_data(movie_html: str) -> dict[str, Any]:
    json_blobs: dict[str, dict[str, Any]] = {}
    for match in JSON_SCRIPT_PATTERN.finditer(movie_html):
        name = match.group("name")
        try:
            json_blobs[name] = json.loads(html.unescape(match.group("body")))
        except json.JSONDecodeError:
            continue

    reviews_data = json_blobs.get("reviewsData", {}).copy()
    media_scorecard = json_blobs.get("mediaScorecard", {})
    if media_scorecard:
        if not reviews_data.get("audienceScore"):
            reviews_data["audienceScore"] = media_scorecard.get("audienceScore", {})
        else:
            reviews_data["audienceScore"] = {
                **media_scorecard.get("audienceScore", {}),
                **reviews_data["audienceScore"],
            }
        if not reviews_data.get("criticsScore"):
            reviews_data["criticsScore"] = media_scorecard.get("criticsScore", {})
        else:
            reviews_data["criticsScore"] = {
                **media_scorecard.get("criticsScore", {}),
                **reviews_data["criticsScore"],
            }
        for key in ("title", "description"):
            if not reviews_data.get(key) and media_scorecard.get(key):
                reviews_data[key] = media_scorecard[key]

    for score_key in ("audienceScore", "criticsScore"):
        score_blob = reviews_data.get(score_key) or {}
        if score_blob.get("score") is None:
            liked_count = score_blob.get("likedCount")
            rating_count = score_blob.get("ratingCount")
            if liked_count is not None and rating_count:
                score_blob = score_blob.copy()
                score_blob["score"] = str(round(100 * liked_count / rating_count))
                reviews_data[score_key] = score_blob

    if reviews_data:
        return reviews_data
    raise ValueError("Could not find Rotten Tomatoes score JSON in page HTML")


def normalize_text(text: str) -> str:
    lowered = text.lower().replace("&", " and ")
    tokens = TOKEN_PATTERN.findall(lowered)
    return " ".join(tokens)


def significant_tokens(text: str) -> set[str]:
    return {
        token for token in TOKEN_PATTERN.findall(text.lower()) if token not in STOPWORDS
    }


def extract_year_hints(text: str) -> set[int]:
    return {int(year) for year in YEAR_PATTERN.findall(text)}


def default_query_from_title(title: str) -> str:
    return " ".join(re.sub(r"\([^)]*\)", "", title).split())


def candidate_match_score(
    entry: MovieEntry, candidate: SearchCandidate, query: str
) -> float:
    normalized_query = normalize_text(query)
    normalized_entry_title = normalize_text(entry.title)
    normalized_candidate_title = normalize_text(candidate.title)
    similarity = max(
        difflib.SequenceMatcher(
            None, normalized_query, normalized_candidate_title
        ).ratio(),
        difflib.SequenceMatcher(
            None, normalized_entry_title, normalized_candidate_title
        ).ratio(),
    )
    query_tokens = significant_tokens(query)
    title_tokens = significant_tokens(entry.title)
    candidate_tokens = significant_tokens(candidate.title)
    token_overlap = 0.0
    if query_tokens | candidate_tokens:
        token_overlap = len(query_tokens & candidate_tokens) / len(
            query_tokens | candidate_tokens
        )
    if title_tokens | candidate_tokens:
        token_overlap = max(
            token_overlap,
            len(title_tokens & candidate_tokens) / len(title_tokens | candidate_tokens),
        )

    year_hints = extract_year_hints(f"{entry.title} {entry.notes}")
    year_bonus = 0.0
    if candidate.release_year is not None and candidate.release_year in year_hints:
        year_bonus = 0.2

    note_tokens = significant_tokens(entry.notes)
    cast_tokens = significant_tokens(" ".join(candidate.cast))
    cast_overlap = len(note_tokens & cast_tokens)
    cast_bonus = min(cast_overlap * 0.05, 0.2)

    media_bonus = 0.0 if candidate.media_type == "movie" else -0.1
    return similarity + token_overlap + year_bonus + cast_bonus + media_bonus


def pick_best_candidate(
    entry: MovieEntry, candidates: list[SearchCandidate], query: str
) -> tuple[SearchCandidate, float]:
    if not candidates:
        raise ValueError(
            f"No Rotten Tomatoes search candidates found for {entry.title!r}"
        )
    scored = [
        (candidate_match_score(entry, candidate, query), candidate)
        for candidate in candidates
    ]
    scored.sort(key=lambda item: item[0], reverse=True)
    best_score, best_candidate = scored[0]
    return best_candidate, best_score


def build_session() -> requests.Session:
    session = requests.Session()
    session.headers.update(REQUEST_HEADERS)
    return session


def fetch_search_candidates(
    session: requests.Session, query: str
) -> list[SearchCandidate]:
    response = session.get(
        "https://www.rottentomatoes.com/search",
        params={"search": query},
        timeout=30,
    )
    response.raise_for_status()
    return parse_search_results(response.text)


def fetch_match_result(session: requests.Session, entry: MovieEntry) -> MatchResult:
    override = MATCH_OVERRIDES.get(entry.title, {})
    query = override.get("query", default_query_from_title(entry.title))
    override_url = override.get("url")
    if override_url:
        best_candidate = SearchCandidate(
            media_type="tvSeries" if "/tv/" in override_url else "movie",
            title=entry.title,
            url=override_url,
            release_year=None,
            cast=(),
            tomatometer_score=None,
        )
        confidence = 2.5
    else:
        candidates = fetch_search_candidates(session, query)
        best_candidate, confidence = pick_best_candidate(entry, candidates, query=query)

    response = session.get(best_candidate.url, timeout=30)
    response.raise_for_status()
    reviews_data = parse_reviews_data(response.text)

    critic_score = reviews_data.get("criticsScore", {}).get("score")
    audience_score = reviews_data.get("audienceScore", {}).get("score")
    return MatchResult(
        entry=entry,
        matched_title=reviews_data.get("title", best_candidate.title),
        media_type=best_candidate.media_type,
        rt_url=best_candidate.url,
        critic_score=int(critic_score) if critic_score is not None else None,
        audience_score=int(audience_score) if audience_score is not None else None,
        release_year=best_candidate.release_year,
        match_confidence=confidence,
        query=query,
    )


def build_ratings_dataframe(match_results: list[MatchResult]) -> pd.DataFrame:
    rows = []
    for result in match_results:
        rows.append(
            {
                "watch_order": result.entry.index + 1,
                "movie_title": result.entry.title,
                "my_rating": result.entry.rating,
                "rt_audience_rating": result.audience_score,
                "rt_critic_rating": result.critic_score,
                "matched_title": result.matched_title,
                "rt_media_type": result.media_type,
                "rt_release_year": result.release_year,
                "rt_url": result.rt_url,
                "match_confidence": result.match_confidence,
                "match_query": result.query,
                "notes": result.entry.notes,
            }
        )
    df = pd.DataFrame(rows)
    df["rt_average_rating"] = df[["rt_audience_rating", "rt_critic_rating"]].mean(
        axis=1
    )
    df["rt_audience_on_10"] = df["rt_audience_rating"] / 10.0
    df["rt_critic_on_10"] = df["rt_critic_rating"] / 10.0
    df["rt_average_on_10"] = df["rt_average_rating"] / 10.0
    df["liked"] = df["my_rating"] >= LIKE_THRESHOLD
    return df


def create_rt_scatter_plot(df: pd.DataFrame, output_path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharey=True)
    plot_specs = [
        ("rt_critic_rating", "RT critic score", "#c0392b"),
        ("rt_audience_rating", "RT audience score", "#2980b9"),
    ]
    for ax, (column, x_label, color) in zip(axes, plot_specs, strict=True):
        x = df[column].to_numpy(dtype=float)
        y = df["my_rating"].to_numpy(dtype=float)
        ax.scatter(x, y, color=color, alpha=0.75)
        slope, intercept = np.polyfit(x, y, 1)
        xs = np.linspace(0, 100, 200)
        ax.plot(xs, slope * xs + intercept, color="#2d3436", linewidth=2)
        corr = np.corrcoef(x, y)[0, 1]
        ax.set_title(f"{x_label} vs my rating\nPearson r = {corr:.2f}")
        ax.set_xlabel(f"{x_label} (%)")
        ax.set_xlim(0, 100)
        ax.set_ylim(0, 10.5)
        ax.grid(alpha=0.25)
    axes[0].set_ylabel("My rating (/10)")
    fig.suptitle("Rotten Tomatoes vs my ratings", fontsize=14)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def create_model_comparison_plot(
    df: pd.DataFrame,
    model_results: dict[str, Any],
    output_path: Path,
) -> None:
    holdout = model_results["holdout_predictions"]
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    avg_x = df["rt_average_on_10"].to_numpy(dtype=float)
    avg_y = df["my_rating"].to_numpy(dtype=float)
    axes[0].scatter(avg_x, avg_y, color="#8e44ad", alpha=0.75)
    axes[0].plot([0, 10], [0, 10], color="#2d3436", linewidth=2, linestyle="--")
    axes[0].set_title(
        f"Average RT vs my rating\nr = {df['my_rating'].corr(df['rt_average_on_10']):.2f}"
    )
    axes[0].set_xlabel("Average RT (/10)")
    axes[0].set_ylabel("My rating (/10)")
    axes[0].set_xlim(0, 10)
    axes[0].set_ylim(0, 10.5)
    axes[0].grid(alpha=0.25)

    linear_x = holdout["linear_prediction"].to_numpy(dtype=float)
    linear_y = holdout["my_rating"].to_numpy(dtype=float)
    axes[1].scatter(linear_x, linear_y, color="#16a085", alpha=0.75)
    axes[1].plot([0, 10], [0, 10], color="#2d3436", linewidth=2, linestyle="--")
    axes[1].set_title(
        "Holdout linear combo\n"
        f"MAE {model_results['linear_metrics']['mae']:.2f}, "
        f"RMSE {model_results['linear_metrics']['rmse']:.2f}"
    )
    axes[1].set_xlabel("Predicted rating (/10)")
    axes[1].set_ylabel("My rating (/10)")
    axes[1].set_xlim(0, 10)
    axes[1].set_ylim(0, 10.5)
    axes[1].grid(alpha=0.25)

    logistic_x = holdout["logistic_like_probability"].to_numpy(dtype=float)
    logistic_y = holdout["liked"].astype(int).to_numpy()
    jitter = np.linspace(-0.03, 0.03, num=len(holdout))
    axes[2].scatter(logistic_x, logistic_y + jitter, color="#d35400", alpha=0.75)
    axes[2].axvline(0.5, color="#2d3436", linewidth=2, linestyle="--")
    axes[2].set_title(
        f"Holdout logistic regression\nAccuracy {model_results['logistic_metrics']['accuracy']:.2f}"
    )
    axes[2].set_xlabel("Predicted P(like)")
    axes[2].set_ylabel("Actual like")
    axes[2].set_xlim(0, 1)
    axes[2].set_ylim(-0.2, 1.2)
    axes[2].set_yticks([0, 1], labels=["No", "Yes"])
    axes[2].grid(alpha=0.25)

    fig.suptitle("Average RT and fitted models", fontsize=14)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def _regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    return {
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": float(math.sqrt(mean_squared_error(y_true, y_pred))),
        "correlation": (
            float(np.corrcoef(y_true, y_pred)[0, 1])
            if len(y_true) > 1
            else float("nan")
        ),
    }


def evaluate_threshold_baselines(
    df: pd.DataFrame,
    *,
    thresholds: tuple[int, ...] = (50, 55, 60, 65, 70, 75, 80),
) -> pd.DataFrame:
    liked = df["liked"].astype(bool)
    rows: list[dict[str, float | int | str]] = []
    score_columns = {
        "audience": "rt_audience_rating",
        "critic": "rt_critic_rating",
        "average": "rt_average_rating",
    }
    for score_name, column in score_columns.items():
        scores = df[column].to_numpy(dtype=float)
        for threshold in thresholds:
            predicted_like = scores >= threshold
            rows.append(
                {
                    "score_source": score_name,
                    "threshold": threshold,
                    "accuracy": float((predicted_like == liked).mean()),
                    "true_positive": int((predicted_like & liked).sum()),
                    "true_negative": int((~predicted_like & ~liked).sum()),
                    "false_positive": int((predicted_like & ~liked).sum()),
                    "false_negative": int((~predicted_like & liked).sum()),
                }
            )
    result = pd.DataFrame(rows)
    return result.sort_values(
        ["accuracy", "score_source", "threshold"],
        ascending=[False, True, True],
    ).reset_index(drop=True)


def evaluate_models(df: pd.DataFrame) -> dict[str, Any]:
    split_index = max(1, math.floor(len(df) * 0.8))
    train_df = df.iloc[:split_index].copy()
    holdout_df = df.iloc[split_index:].copy()
    if holdout_df.empty:
        raise ValueError("Need at least one holdout row for evaluation")

    feature_columns = ["rt_critic_on_10", "rt_audience_on_10"]
    X_train = train_df[feature_columns].to_numpy(dtype=float)
    X_holdout = holdout_df[feature_columns].to_numpy(dtype=float)
    y_train = train_df["my_rating"].to_numpy(dtype=float)
    y_holdout = holdout_df["my_rating"].to_numpy(dtype=float)

    avg_holdout_pred = holdout_df["rt_average_on_10"].to_numpy(dtype=float)
    average_metrics = _regression_metrics(y_holdout, avg_holdout_pred)

    holdout_linear_model = LinearRegression()
    holdout_linear_model.fit(X_train, y_train)
    linear_holdout_pred = holdout_linear_model.predict(X_holdout)
    linear_metrics = _regression_metrics(y_holdout, linear_holdout_pred)

    y_train_like = train_df["liked"].to_numpy(dtype=int)
    y_holdout_like = holdout_df["liked"].to_numpy(dtype=int)
    if len(set(y_train_like.tolist())) < 2:
        raise ValueError(
            "Training split does not contain both liked and disliked examples"
        )

    holdout_logistic_model = LogisticRegression(random_state=0, max_iter=1000)
    holdout_logistic_model.fit(X_train, y_train_like)
    logistic_holdout_prob = holdout_logistic_model.predict_proba(X_holdout)[:, 1]
    logistic_holdout_pred = (logistic_holdout_prob >= 0.5).astype(int)
    cutoff_holdout_pred = (
        holdout_df["rt_average_rating"].to_numpy(dtype=float) >= 60.0
    ).astype(int)

    df = df.copy()
    full_X = df[feature_columns].to_numpy(dtype=float)
    full_linear_model = LinearRegression()
    full_linear_model.fit(full_X, df["my_rating"].to_numpy(dtype=float))
    df["linear_prediction_full"] = full_linear_model.predict(full_X)

    full_logistic_model = LogisticRegression(random_state=0, max_iter=1000)
    full_logistic_model.fit(full_X, df["liked"].to_numpy(dtype=int))
    df["logistic_like_probability_full"] = full_logistic_model.predict_proba(full_X)[
        :, 1
    ]
    df["logistic_like_prediction_full"] = df["logistic_like_probability_full"] >= 0.5

    return {
        "split_index": split_index,
        "train_size": int(len(train_df)),
        "holdout_size": int(len(holdout_df)),
        "average_metrics": average_metrics,
        "linear_metrics": linear_metrics,
        "linear_coefficients": {
            "intercept": float(holdout_linear_model.intercept_),
            "critic_weight": float(holdout_linear_model.coef_[0]),
            "audience_weight": float(holdout_linear_model.coef_[1]),
        },
        "logistic_metrics": {
            "accuracy": float(accuracy_score(y_holdout_like, logistic_holdout_pred)),
            "baseline_60_accuracy": float(
                accuracy_score(y_holdout_like, cutoff_holdout_pred)
            ),
        },
        "logistic_coefficients": {
            "intercept": float(holdout_logistic_model.intercept_[0]),
            "critic_weight": float(holdout_logistic_model.coef_[0][0]),
            "audience_weight": float(holdout_logistic_model.coef_[0][1]),
        },
        "holdout_predictions": holdout_df.assign(
            average_prediction=avg_holdout_pred,
            linear_prediction=linear_holdout_pred,
            logistic_like_probability=logistic_holdout_prob,
            logistic_like_prediction=logistic_holdout_pred.astype(bool),
            baseline_60_like_prediction=cutoff_holdout_pred.astype(bool),
        ),
        "full_predictions": df,
    }


def cross_validate_models(
    df: pd.DataFrame,
    *,
    n_splits: int = 5,
    n_repeats: int = 100,
    random_state: int = 0,
    thresholds: tuple[int, ...] = (60, 65, 70),
) -> dict[str, Any]:
    feature_columns = ["rt_critic_on_10", "rt_audience_on_10"]
    X = df[feature_columns].to_numpy(dtype=float)
    y = df["my_rating"].to_numpy(dtype=float)
    liked = df["liked"].to_numpy(dtype=int)

    regression_rows: list[dict[str, float | str]] = []
    regression_splitter = RepeatedKFold(
        n_splits=n_splits,
        n_repeats=n_repeats,
        random_state=random_state,
    )
    for train_index, test_index in regression_splitter.split(X):
        y_test = y[test_index]
        holdout = df.iloc[test_index]

        average_pred = holdout["rt_average_on_10"].to_numpy(dtype=float)
        average_metrics = _regression_metrics(y_test, average_pred)
        regression_rows.append({"model": "average_rt", **average_metrics})

        linear_model = LinearRegression()
        linear_model.fit(X[train_index], y[train_index])
        linear_metrics = _regression_metrics(
            y_test, linear_model.predict(X[test_index])
        )
        regression_rows.append({"model": "linear_rt", **linear_metrics})

    classification_rows: list[dict[str, float | int | str]] = []
    classification_splitter = RepeatedStratifiedKFold(
        n_splits=n_splits,
        n_repeats=n_repeats,
        random_state=random_state,
    )
    for train_index, test_index in classification_splitter.split(X, liked):
        y_test = liked[test_index]
        holdout = df.iloc[test_index]

        logistic_model = LogisticRegression(random_state=0, max_iter=1000)
        logistic_model.fit(X[train_index], liked[train_index])
        logistic_pred = logistic_model.predict(X[test_index])
        classification_rows.append(
            {
                "model": "logistic_rt",
                "accuracy": float(accuracy_score(y_test, logistic_pred)),
            }
        )

        for source_name, column in (
            ("audience", "rt_audience_rating"),
            ("average", "rt_average_rating"),
            ("critic", "rt_critic_rating"),
        ):
            for threshold in thresholds:
                threshold_pred = (
                    holdout[column].to_numpy(dtype=float) >= threshold
                ).astype(int)
                classification_rows.append(
                    {
                        "model": f"{source_name}_threshold",
                        "threshold": threshold,
                        "accuracy": float(accuracy_score(y_test, threshold_pred)),
                    }
                )

    regression_summary = (
        pd.DataFrame(regression_rows)
        .groupby("model", as_index=False)
        .agg(
            mae_mean=("mae", "mean"),
            mae_std=("mae", "std"),
            rmse_mean=("rmse", "mean"),
            rmse_std=("rmse", "std"),
            correlation_mean=("correlation", "mean"),
            correlation_std=("correlation", "std"),
        )
        .sort_values("mae_mean")
        .reset_index(drop=True)
    )
    classification_summary = (
        pd.DataFrame(classification_rows)
        .groupby(["model", "threshold"], dropna=False, as_index=False)
        .agg(
            accuracy_mean=("accuracy", "mean"),
            accuracy_std=("accuracy", "std"),
        )
        .sort_values("accuracy_mean", ascending=False)
        .reset_index(drop=True)
    )
    return {
        "n_splits": n_splits,
        "n_repeats": n_repeats,
        "regression_summary": regression_summary,
        "classification_summary": classification_summary,
    }


def simulate_data_quality_impact(
    df: pd.DataFrame,
    *,
    deltas: tuple[int, ...] = (5, 10),
    iterations: int = 5000,
    random_state: int = 0,
) -> list[dict[str, Any]]:
    liked = df["liked"].to_numpy(dtype=bool)
    audience = df["rt_audience_rating"].to_numpy(dtype=float)
    critic = df["rt_critic_rating"].to_numpy(dtype=float)
    my_rating = df["my_rating"].to_numpy(dtype=float)
    rng = np.random.default_rng(random_state)
    rows: list[dict[str, Any]] = []
    for delta in deltas:
        audience_corr: list[float] = []
        audience_60_accuracy: list[float] = []
        audience_70_accuracy: list[float] = []
        average_60_accuracy: list[float] = []
        for _ in range(iterations):
            audience_noise = rng.integers(-delta, delta + 1, size=len(df))
            critic_noise = rng.integers(-delta, delta + 1, size=len(df))
            perturbed_audience = np.clip(audience + audience_noise, 0, 100)
            perturbed_critic = np.clip(critic + critic_noise, 0, 100)
            perturbed_average = (perturbed_audience + perturbed_critic) / 2.0

            audience_corr.append(
                float(np.corrcoef(my_rating, perturbed_audience / 10.0)[0, 1])
            )
            audience_60_accuracy.append(
                float(((perturbed_audience >= 60) == liked).mean())
            )
            audience_70_accuracy.append(
                float(((perturbed_audience >= 70) == liked).mean())
            )
            average_60_accuracy.append(
                float(((perturbed_average >= 60) == liked).mean())
            )

        rows.append(
            {
                "delta": delta,
                "iterations": iterations,
                "audience_correlation_percentiles": {
                    "p05": float(np.percentile(audience_corr, 5)),
                    "p50": float(np.percentile(audience_corr, 50)),
                    "p95": float(np.percentile(audience_corr, 95)),
                },
                "audience_60_accuracy_percentiles": {
                    "p05": float(np.percentile(audience_60_accuracy, 5)),
                    "p50": float(np.percentile(audience_60_accuracy, 50)),
                    "p95": float(np.percentile(audience_60_accuracy, 95)),
                },
                "audience_70_accuracy_percentiles": {
                    "p05": float(np.percentile(audience_70_accuracy, 5)),
                    "p50": float(np.percentile(audience_70_accuracy, 50)),
                    "p95": float(np.percentile(audience_70_accuracy, 95)),
                },
                "average_60_accuracy_percentiles": {
                    "p05": float(np.percentile(average_60_accuracy, 5)),
                    "p50": float(np.percentile(average_60_accuracy, 50)),
                    "p95": float(np.percentile(average_60_accuracy, 95)),
                },
            }
        )
    return rows


def build_disagreement_tables(
    df_with_predictions: pd.DataFrame,
) -> dict[str, pd.DataFrame]:
    df = df_with_predictions.copy()
    df["critic_gap"] = df["my_rating"] - df["rt_critic_on_10"]
    df["audience_gap"] = df["my_rating"] - df["rt_audience_on_10"]
    df["average_gap"] = df["my_rating"] - df["rt_average_on_10"]
    df["linear_gap"] = df["my_rating"] - df["linear_prediction_full"]
    df["logistic_confidence_gap"] = np.where(
        df["liked"],
        1.0 - df["logistic_like_probability_full"],
        df["logistic_like_probability_full"],
    )

    core_columns = [
        "movie_title",
        "my_rating",
        "rt_critic_rating",
        "rt_audience_rating",
        "rt_average_rating",
        "linear_prediction_full",
        "logistic_like_probability_full",
        "liked",
    ]
    return {
        "rt_average_disagreements": df.loc[
            df["average_gap"].abs().sort_values(ascending=False).index,
            core_columns + ["average_gap"],
        ],
        "critic_disagreements": df.loc[
            df["critic_gap"].abs().sort_values(ascending=False).index,
            core_columns + ["critic_gap"],
        ],
        "audience_disagreements": df.loc[
            df["audience_gap"].abs().sort_values(ascending=False).index,
            core_columns + ["audience_gap"],
        ],
        "linear_model_disagreements": df.loc[
            df["linear_gap"].abs().sort_values(ascending=False).index,
            core_columns + ["linear_gap"],
        ],
        "logistic_model_disagreements": df.loc[
            df["logistic_confidence_gap"].sort_values(ascending=False).index,
            core_columns + ["logistic_confidence_gap"],
        ],
    }


def summarize_analysis(df: pd.DataFrame, model_results: dict[str, Any]) -> str:
    audience_corr = df["my_rating"].corr(df["rt_audience_rating"])
    critic_corr = df["my_rating"].corr(df["rt_critic_rating"])
    average_corr = df["my_rating"].corr(df["rt_average_rating"])

    holdout = model_results["holdout_predictions"]
    rule_60_all = float(((df["rt_average_rating"] >= 60.0) == df["liked"]).mean())
    rule_60_holdout = float(
        ((holdout["rt_average_rating"] >= 60.0) == holdout["liked"]).mean()
    )
    lines = [
        f"Movies analyzed: {len(df)}",
        f"Like threshold: my rating >= {LIKE_THRESHOLD:.0f}/10",
        "",
        "Raw RT correlations against my rating (/10 vs %):",
        f"  Critic correlation: {critic_corr:.3f}",
        f"  Audience correlation: {audience_corr:.3f}",
        f"  Average RT correlation: {average_corr:.3f}",
        "",
        "Holdout evaluation on last 20% of entries:",
        f"  Holdout size: {model_results['holdout_size']}",
        f"  Average RT/10 MAE: {model_results['average_metrics']['mae']:.3f}",
        f"  Average RT/10 RMSE: {model_results['average_metrics']['rmse']:.3f}",
        (
            "  Linear combo MAE/RMSE: "
            f"{model_results['linear_metrics']['mae']:.3f} / "
            f"{model_results['linear_metrics']['rmse']:.3f}"
        ),
        (
            "  Linear combo weights: "
            f"{model_results['linear_coefficients']['critic_weight']:.3f} * critic/10 + "
            f"{model_results['linear_coefficients']['audience_weight']:.3f} * audience/10 + "
            f"{model_results['linear_coefficients']['intercept']:.3f}"
        ),
        f"  Logistic holdout accuracy: {model_results['logistic_metrics']['accuracy']:.3f}",
        (
            "  Average RT >= 60 baseline accuracy "
            f"(holdout / all): {rule_60_holdout:.3f} / {rule_60_all:.3f}"
        ),
    ]
    return "\n".join(lines)
