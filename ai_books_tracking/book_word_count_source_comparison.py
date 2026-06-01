"""Compare local extracted book word counts with online/proxy sources."""

from __future__ import annotations

import argparse
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from ai_books_tracking.book_wpm_calendar_notes import (  # noqa: E402
    DEFAULT_OUTPUT_DIR,
    normalize_title,
)


DEFAULT_ALL_FINISHED_PROJECTION = (
    DEFAULT_OUTPUT_DIR / "book_word_count_all_finished_projection.csv"
)
DEFAULT_GOLDEN_MASTER = (
    REPO_ROOT / "ai_books_tracking" / "golden_master_multi_source.csv"
)
DEFAULT_GOOGLE_CACHE = (
    REPO_ROOT / "ai_books_tracking" / "books_enriched_with_goodreads.csv"
)
DEFAULT_OPENLIBRARY_CACHE = (
    DEFAULT_OUTPUT_DIR / "book_word_count_openlibrary_live_cache.json"
)
USER_AGENT = "Codex book word-count audit (codex-ai@local)"


def clean_query_title(title: object) -> str:
    text = str(title or "").strip()
    for suffix in [".pdf", ".epub", ".html", ".mobi", ".azw3"]:
        if text.lower().endswith(suffix):
            text = text[: -len(suffix)]
    text = text.replace("_", " ").replace("-", " ")
    return " ".join(text.split())


def title_overlap_score(left: object, right: object) -> float:
    left_words = set(normalize_title(str(left)).split())
    right_words = set(normalize_title(str(right)).split())
    if not left_words or not right_words:
        return 0.0
    return len(left_words & right_words) / min(len(left_words), len(right_words))


def best_table_match(title: object, table: pd.DataFrame) -> pd.Series | None:
    if table.empty or "title" not in table:
        return None
    title_norm = normalize_title(str(title))
    if not title_norm:
        return None
    exact = table[table["_title_norm"].eq(title_norm)]
    if not exact.empty:
        return exact.iloc[0]
    scored = table.copy()
    scored["_score"] = scored["title"].map(
        lambda value: title_overlap_score(title, value)
    )
    scored = scored[scored["_score"].ge(0.75)].sort_values("_score", ascending=False)
    if scored.empty:
        return None
    return scored.iloc[0]


def load_match_table(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    frame = pd.read_csv(path)
    if "title" in frame:
        frame["_title_norm"] = frame["title"].map(
            lambda value: normalize_title(str(value))
        )
    return frame


def load_openlibrary_cache(path: Path) -> dict[str, dict[str, object]]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def write_openlibrary_cache(path: Path, cache: dict[str, dict[str, object]]) -> None:
    path.write_text(json.dumps(cache, indent=2, sort_keys=True), encoding="utf-8")


def openlibrary_query(
    *,
    title: str,
    author: str = "",
    cache: dict[str, dict[str, object]],
    delay_seconds: float,
) -> dict[str, object]:
    query_key = "|".join([normalize_title(title), normalize_title(author)])
    if query_key in cache:
        return cache[query_key]
    params = {
        "title": clean_query_title(title),
        "limit": 8,
        "fields": ",".join(
            [
                "title",
                "author_name",
                "number_of_pages_median",
                "number_of_pages",
                "first_publish_year",
                "edition_count",
                "key",
            ]
        ),
    }
    if author and str(author).strip().lower() not in {"by", "unknown", "nan"}:
        params["author"] = str(author)
    result: dict[str, object] = {
        "query_title": params["title"],
        "query_author": params.get("author", ""),
        "status": "not_requested",
        "match_title": "",
        "match_author": "",
        "match_score": np.nan,
        "number_of_pages_median": np.nan,
        "number_of_pages": np.nan,
        "first_publish_year": np.nan,
        "edition_count": np.nan,
        "key": "",
    }
    try:
        response = requests.get(
            "https://openlibrary.org/search.json",
            params=params,
            timeout=8,
            headers={"User-Agent": USER_AGENT},
        )
        response.raise_for_status()
        docs = response.json().get("docs", [])
        candidates: list[dict[str, object]] = []
        for doc in docs:
            pages = doc.get("number_of_pages_median") or doc.get("number_of_pages")
            if not pages:
                continue
            score = title_overlap_score(title, doc.get("title", ""))
            author_names = " ; ".join(doc.get("author_name", [])[:3])
            if author and author_names:
                score = max(score, 0.8 * title_overlap_score(author, author_names))
            candidates.append({"score": score, **doc})
        if candidates:
            best = sorted(candidates, key=lambda item: item["score"], reverse=True)[0]
            result.update(
                {
                    "status": "matched" if best["score"] >= 0.5 else "weak_match",
                    "match_title": best.get("title", ""),
                    "match_author": " ; ".join(best.get("author_name", [])[:3]),
                    "match_score": best["score"],
                    "number_of_pages_median": best.get(
                        "number_of_pages_median", np.nan
                    ),
                    "number_of_pages": best.get("number_of_pages", np.nan),
                    "first_publish_year": best.get("first_publish_year", np.nan),
                    "edition_count": best.get("edition_count", np.nan),
                    "key": best.get("key", ""),
                }
            )
        else:
            result["status"] = "no_page_count_match"
    except Exception as exc:
        result["status"] = f"error:{type(exc).__name__}"
        result["error"] = str(exc)
    cache[query_key] = result
    if delay_seconds > 0:
        time.sleep(delay_seconds)
    return result


def add_online_sources(
    base: pd.DataFrame,
    *,
    golden_master: pd.DataFrame,
    google_cache: pd.DataFrame,
    openlibrary_cache_path: Path,
    refresh_openlibrary: bool,
    delay_seconds: float,
    openlibrary_workers: int,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    cache = (
        {} if refresh_openlibrary else load_openlibrary_cache(openlibrary_cache_path)
    )
    openlibrary_results: dict[int, dict[str, object]] = {}

    def fetch_openlibrary(
        row_item: tuple[int, pd.Series],
    ) -> tuple[int, dict[str, object]]:
        row_index, row = row_item
        title = row.get("title", "")
        golden = best_table_match(title, golden_master)
        author = (
            str(golden.get("author", ""))
            if golden is not None and "author" in golden
            else ""
        )
        return (
            row_index,
            openlibrary_query(
                title=str(title),
                author=author,
                cache=cache,
                delay_seconds=delay_seconds,
            ),
        )

    row_items = list(base.iterrows())
    if openlibrary_workers > 1:
        with ThreadPoolExecutor(max_workers=openlibrary_workers) as executor:
            futures = [executor.submit(fetch_openlibrary, item) for item in row_items]
            for done_count, future in enumerate(as_completed(futures), start=1):
                row_index, openlibrary = future.result()
                openlibrary_results[row_index] = openlibrary
                if done_count % 25 == 0:
                    write_openlibrary_cache(openlibrary_cache_path, cache)
                    print(
                        f"Checked Open Library for {done_count}/{len(base)} titles",
                        flush=True,
                    )
    else:
        for row_index, row in row_items:
            _, openlibrary = fetch_openlibrary((row_index, row))
            openlibrary_results[row_index] = openlibrary
            if (row_index + 1) % 25 == 0:
                write_openlibrary_cache(openlibrary_cache_path, cache)
                print(
                    f"Checked Open Library for {row_index + 1}/{len(base)} titles",
                    flush=True,
                )

    for row_index, row in base.iterrows():
        title = row.get("title", "")
        golden = best_table_match(title, golden_master)
        google = best_table_match(title, google_cache)
        author = (
            str(golden.get("author", ""))
            if golden is not None and "author" in golden
            else ""
        )
        openlibrary = openlibrary_results.get(row_index, {})
        rows.append(
            {
                "finish_id": row.get("finish_id", ""),
                "title": title,
                "finish_date": row.get("finish_date", ""),
                "local_file_word_count": row.get("local_file_word_count", np.nan),
                "local_raw_word_count": row.get("local_raw_word_count", np.nan),
                "local_excluded_word_count": row.get(
                    "local_excluded_word_count", np.nan
                ),
                "local_word_count_method": row.get("local_word_count_method", ""),
                "local_file_path": row.get("local_file_path", ""),
                "readinglength_word_count": row.get("external_word_count", np.nan),
                "readinglength_source": row.get("external_word_count_source", ""),
                "current_metadata_page_count": row.get("metadata_page_count", np.nan),
                "projected_word_count": row.get("projected_word_count", np.nan),
                "projected_word_count_method": row.get(
                    "projected_word_count_method", ""
                ),
                "golden_master_page_count": (
                    golden.get("page_count", np.nan) if golden is not None else np.nan
                ),
                "golden_master_author": author,
                "cached_google_books_page_count": (
                    google.get("gb_page_count", np.nan)
                    if google is not None
                    else np.nan
                ),
                "cached_google_books_title": (
                    google.get("goodreads_title", google.get("title", ""))
                    if google is not None
                    else ""
                ),
                "openlibrary_page_count": openlibrary.get(
                    "number_of_pages_median", np.nan
                )
                or openlibrary.get("number_of_pages", np.nan),
                "openlibrary_status": openlibrary.get("status", ""),
                "openlibrary_match_title": openlibrary.get("match_title", ""),
                "openlibrary_match_author": openlibrary.get("match_author", ""),
                "openlibrary_match_score": openlibrary.get("match_score", np.nan),
                "openlibrary_key": openlibrary.get("key", ""),
            }
        )
    write_openlibrary_cache(openlibrary_cache_path, cache)
    return pd.DataFrame(rows)


def add_word_estimate_columns(
    frame: pd.DataFrame, words_per_page: float
) -> pd.DataFrame:
    result = frame.copy()
    for column in [
        "current_metadata_page_count",
        "golden_master_page_count",
        "cached_google_books_page_count",
        "openlibrary_page_count",
    ]:
        result[f"{column}_word_estimate"] = (
            pd.to_numeric(result[column], errors="coerce") * words_per_page
        )
    return result


def build_long_comparison(frame: pd.DataFrame) -> pd.DataFrame:
    sources = [
        ("readinglength_direct_words", "readinglength_word_count", "direct_words"),
        (
            "current_metadata_pages_x_local_wpp",
            "current_metadata_page_count_word_estimate",
            "page_proxy",
        ),
        (
            "golden_master_pages_x_local_wpp",
            "golden_master_page_count_word_estimate",
            "page_proxy",
        ),
        (
            "cached_google_books_pages_x_local_wpp",
            "cached_google_books_page_count_word_estimate",
            "page_proxy",
        ),
        (
            "openlibrary_pages_x_local_wpp",
            "openlibrary_page_count_word_estimate",
            "page_proxy",
        ),
        ("full_projection", "projected_word_count", "mixed_projection"),
    ]
    rows: list[dict[str, object]] = []
    local = pd.to_numeric(frame["local_file_word_count"], errors="coerce")
    for source_name, column, source_kind in sources:
        values = pd.to_numeric(frame[column], errors="coerce")
        mask = local.gt(0) & values.gt(0)
        for idx in frame[mask].index:
            rows.append(
                {
                    "finish_id": frame.loc[idx, "finish_id"],
                    "title": frame.loc[idx, "title"],
                    "source": source_name,
                    "source_kind": source_kind,
                    "local_file_word_count": float(local.loc[idx]),
                    "source_word_estimate": float(values.loc[idx]),
                    "source_minus_local_words": float(values.loc[idx] - local.loc[idx]),
                    "source_error_rate_vs_local": float(
                        (values.loc[idx] - local.loc[idx]) / local.loc[idx]
                    ),
                }
            )
    return pd.DataFrame(rows)


def short_label(value: object, max_length: int = 28) -> str:
    text = " ".join(str(value or "").split())
    if len(text) <= max_length:
        return text
    return text[: max_length - 1].rstrip() + "..."


def build_outlier_report(
    long_frame: pd.DataFrame,
    wide_frame: pd.DataFrame,
    *,
    abs_error_threshold: float = 0.50,
    top_per_source: int = 5,
) -> pd.DataFrame:
    if long_frame.empty:
        return pd.DataFrame()
    wide_lookup = wide_frame.set_index("finish_id", drop=False)
    candidates = long_frame.copy()
    candidates["abs_error_rate_vs_local"] = candidates[
        "source_error_rate_vs_local"
    ].abs()
    threshold_rows = candidates[
        candidates["abs_error_rate_vs_local"].ge(abs_error_threshold)
    ]
    top_rows = (
        candidates[candidates["abs_error_rate_vs_local"].gt(1e-9)]
        .sort_values("abs_error_rate_vs_local", ascending=False)
        .groupby("source", group_keys=False)
        .head(top_per_source)
    )
    outliers = (
        pd.concat([threshold_rows, top_rows], ignore_index=True)
        .drop_duplicates(["finish_id", "source"])
        .sort_values(["source", "abs_error_rate_vs_local"], ascending=[True, False])
    )
    rows: list[dict[str, object]] = []
    for _, row in outliers.iterrows():
        wide = wide_lookup.loc[row["finish_id"]]
        excluded_pct = (
            float(wide["local_excluded_word_count"])
            / float(wide["local_raw_word_count"])
            if pd.notna(wide.get("local_excluded_word_count"))
            and pd.notna(wide.get("local_raw_word_count"))
            and float(wide["local_raw_word_count"]) > 0
            else np.nan
        )
        notes: list[str] = []
        source = str(row["source"])
        error = float(row["source_error_rate_vs_local"])
        if source == "readinglength_direct_words":
            direction = "under" if error < 0 else "over"
            notes.append(
                f"ReadingLength direct count {direction} local extracted text; "
                "treat as edition/vendor-specific, not authoritative."
            )
        elif source == "openlibrary_pages_x_local_wpp":
            openlibrary_score = wide.get("openlibrary_match_score", np.nan)
            openlibrary_score_text = (
                f"{float(openlibrary_score):.2f}" if pd.notna(openlibrary_score) else ""
            )
            notes.append(
                "Open Library page proxy depends on matched edition page count; "
                f"matched {wide.get('openlibrary_match_title', '')!r} "
                f"with score {openlibrary_score_text}."
            )
        elif source == "cached_google_books_pages_x_local_wpp":
            notes.append(
                "Cached Google Books page proxy depends on cached edition; "
                f"matched {wide.get('cached_google_books_title', '')!r}."
            )
        elif source in {
            "current_metadata_pages_x_local_wpp",
            "golden_master_pages_x_local_wpp",
        }:
            notes.append(
                "Page-count proxy is calibrated on average words/page and can fail "
                "when page metadata is from a different edition or format."
            )
        if pd.notna(excluded_pct) and excluded_pct >= 0.20:
            notes.append(
                f"Local extractor excluded {excluded_pct:.1%} of raw words; "
                "inspect if this is front/back matter or body text."
            )
        rows.append(
            {
                "finish_id": row["finish_id"],
                "title": row["title"],
                "source": source,
                "local_file_word_count": row["local_file_word_count"],
                "source_word_estimate": row["source_word_estimate"],
                "source_error_rate_vs_local": error,
                "abs_error_rate_vs_local": row["abs_error_rate_vs_local"],
                "local_word_count_method": wide.get("local_word_count_method", ""),
                "local_excluded_pct": excluded_pct,
                "local_file_path": wide.get("local_file_path", ""),
                "source_match_or_page_count": source_match_detail(source, wide),
                "investigation_note": " ".join(notes),
            }
        )
    return pd.DataFrame(rows)


def source_match_detail(source: str, wide: pd.Series) -> object:
    def pages(value: object) -> str:
        return f"{float(value):.0f}" if pd.notna(value) else ""

    def score(value: object) -> str:
        return f"{float(value):.2f}" if pd.notna(value) else ""

    if source == "openlibrary_pages_x_local_wpp":
        return (
            f"pages={pages(wide.get('openlibrary_page_count', np.nan))}; "
            f"status={wide.get('openlibrary_status', '')}; "
            f"match={wide.get('openlibrary_match_title', '')}; "
            f"author={wide.get('openlibrary_match_author', '')}; "
            f"score={score(wide.get('openlibrary_match_score', np.nan))}"
        )
    if source == "cached_google_books_pages_x_local_wpp":
        return (
            f"pages={pages(wide.get('cached_google_books_page_count', np.nan))}; "
            f"match={wide.get('cached_google_books_title', '')}"
        )
    if source == "golden_master_pages_x_local_wpp":
        return (
            f"pages={pages(wide.get('golden_master_page_count', np.nan))}; "
            f"author={wide.get('golden_master_author', '')}"
        )
    if source == "current_metadata_pages_x_local_wpp":
        return f"pages={pages(wide.get('current_metadata_page_count', np.nan))}"
    if source == "readinglength_direct_words":
        return str(wide.get("readinglength_source", ""))
    return ""


def markdown_escape(value: object) -> str:
    if pd.isna(value):
        return ""
    return str(value).replace("\n", " ").replace("|", "\\|")


def write_markdown_table(frame: pd.DataFrame, path: Path) -> None:
    if frame.empty:
        path.write_text("_No outliers found._\n", encoding="utf-8")
        return
    display = frame.copy()
    for column in [
        "source_error_rate_vs_local",
        "abs_error_rate_vs_local",
        "local_excluded_pct",
    ]:
        if column in display:
            display[column] = pd.to_numeric(display[column], errors="coerce").map(
                lambda value: f"{value:.1%}" if pd.notna(value) else ""
            )
    for column in ["local_file_word_count", "source_word_estimate"]:
        if column in display:
            display[column] = pd.to_numeric(display[column], errors="coerce").map(
                lambda value: f"{value:,.0f}" if pd.notna(value) else ""
            )
    headers = list(display.columns)
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for _, row in display.iterrows():
        lines.append(
            "| " + " | ".join(markdown_escape(row[column]) for column in headers) + " |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_summary(long_frame: pd.DataFrame, wide_frame: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    if not long_frame.empty:
        for source, group in long_frame.groupby("source"):
            rows.append(
                {
                    "source": source,
                    "n_with_local_pair": len(group),
                    "local_total_words_for_paired_rows": group[
                        "local_file_word_count"
                    ].sum(),
                    "source_total_words_for_paired_rows": group[
                        "source_word_estimate"
                    ].sum(),
                    "aggregate_error_rate_vs_local": (
                        group["source_word_estimate"].sum()
                        / group["local_file_word_count"].sum()
                        - 1
                    ),
                    "median_error_rate_vs_local": group[
                        "source_error_rate_vs_local"
                    ].median(),
                    "mean_abs_error_rate_vs_local": group["source_error_rate_vs_local"]
                    .abs()
                    .mean(),
                }
            )
    coverage = {
        "source": "COVERAGE_COUNTS",
        "n_finished_rows": len(wide_frame),
        "n_with_local_file_words": pd.to_numeric(
            wide_frame["local_file_word_count"], errors="coerce"
        )
        .gt(0)
        .sum(),
        "n_with_readinglength_words": pd.to_numeric(
            wide_frame["readinglength_word_count"], errors="coerce"
        )
        .gt(0)
        .sum(),
        "n_with_current_metadata_pages": pd.to_numeric(
            wide_frame["current_metadata_page_count"], errors="coerce"
        )
        .gt(0)
        .sum(),
        "n_with_golden_master_pages": pd.to_numeric(
            wide_frame["golden_master_page_count"], errors="coerce"
        )
        .gt(0)
        .sum(),
        "n_with_cached_google_books_pages": pd.to_numeric(
            wide_frame["cached_google_books_page_count"], errors="coerce"
        )
        .gt(0)
        .sum(),
        "n_with_openlibrary_pages": pd.to_numeric(
            wide_frame["openlibrary_page_count"], errors="coerce"
        )
        .gt(0)
        .sum(),
    }
    rows.append(coverage)
    return pd.DataFrame(rows)


def plot_source_scatter(long_frame: pd.DataFrame, output_path: Path) -> None:
    sources = [
        "readinglength_direct_words",
        "current_metadata_pages_x_local_wpp",
        "cached_google_books_pages_x_local_wpp",
        "openlibrary_pages_x_local_wpp",
        "golden_master_pages_x_local_wpp",
        "full_projection",
    ]
    fig, axes = plt.subplots(2, 3, figsize=(15, 9), sharex=True, sharey=True)
    all_values = pd.concat(
        [
            long_frame["local_file_word_count"],
            long_frame["source_word_estimate"],
        ]
    )
    min_value = max(10_000, float(all_values.min()) * 0.8)
    max_value = float(all_values.max()) * 1.2
    for ax, source in zip(axes.flat, sources, strict=False):
        sub = long_frame[long_frame["source"].eq(source)]
        ax.scatter(
            sub["local_file_word_count"],
            sub["source_word_estimate"],
            alpha=0.7,
            s=26,
        )
        labels = sub[sub["source_error_rate_vs_local"].abs().ge(0.50)].sort_values(
            "source_error_rate_vs_local", key=lambda series: series.abs()
        )
        labels = labels.tail(5)
        for _, point in labels.iterrows():
            ax.annotate(
                short_label(point["title"]),
                (
                    point["local_file_word_count"],
                    point["source_word_estimate"],
                ),
                textcoords="offset points",
                xytext=(5, 4),
                fontsize=7,
                alpha=0.85,
            )
        ax.plot([min_value, max_value], [min_value, max_value], color="black", lw=1)
        ax.plot(
            [min_value, max_value],
            [min_value * 0.8, max_value * 0.8],
            color="#dc2626",
            lw=0.8,
            linestyle="--",
        )
        ax.plot(
            [min_value, max_value],
            [min_value * 1.2, max_value * 1.2],
            color="#dc2626",
            lw=0.8,
            linestyle="--",
        )
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_title(f"{source}\npaired n={len(sub)}", fontsize=10)
        ax.grid(True, alpha=0.25)
    for ax in axes[-1, :]:
        ax.set_xlabel("Local extracted reading words")
    for ax in axes[:, 0]:
        ax.set_ylabel("Source word estimate")
    fig.suptitle("Book Word-Count Sources vs Local Extracted Text")
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def plot_error_by_source(long_frame: pd.DataFrame, output_path: Path) -> None:
    if long_frame.empty:
        return
    sources = (
        long_frame.groupby("source")["source_error_rate_vs_local"]
        .count()
        .sort_values(ascending=False)
        .index.tolist()
    )
    data = [
        long_frame.loc[long_frame["source"].eq(source), "source_error_rate_vs_local"]
        * 100
        for source in sources
    ]
    fig, ax = plt.subplots(figsize=(13, 7))
    ax.boxplot(data, tick_labels=sources, showfliers=False)
    for idx, values in enumerate(data, start=1):
        jitter = np.linspace(-0.18, 0.18, len(values)) if len(values) else []
        ax.scatter(np.array(jitter) + idx, values, alpha=0.55, s=18)
    ax.axhline(0, color="black", lw=1)
    ax.axhline(-20, color="#dc2626", lw=0.8, linestyle="--")
    ax.axhline(20, color="#dc2626", lw=0.8, linestyle="--")
    ax.set_ylabel("Error vs local extracted text (%)")
    ax.set_title("Online/Proxy Word-Count Error by Source")
    ax.tick_params(axis="x", rotation=25)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def plot_source_totals(wide_frame: pd.DataFrame, output_path: Path) -> None:
    totals = []
    source_columns = [
        ("local_file_word_count", "local extracted"),
        ("readinglength_word_count", "ReadingLength direct"),
        ("current_metadata_page_count_word_estimate", "current metadata pages"),
        ("golden_master_page_count_word_estimate", "golden master pages"),
        ("cached_google_books_page_count_word_estimate", "cached Google pages"),
        ("openlibrary_page_count_word_estimate", "Open Library pages"),
        ("projected_word_count", "full projection"),
    ]
    for column, label in source_columns:
        values = pd.to_numeric(wide_frame[column], errors="coerce")
        totals.append(
            {"source": label, "rows": int(values.gt(0).sum()), "words": values.sum()}
        )
    frame = pd.DataFrame(totals)
    fig, ax = plt.subplots(figsize=(12, 6))
    bars = ax.bar(frame["source"], frame["words"] / 1_000_000, color="#4f6f8f")
    for bar, rows in zip(bars, frame["rows"], strict=False):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height(),
            f"n={rows}",
            ha="center",
            va="bottom",
            fontsize=9,
        )
    ax.set_ylabel("Total words represented (millions)")
    ax.set_title("Total Word Coverage by Source")
    ax.tick_params(axis="x", rotation=25)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--all-finished-projection",
        type=Path,
        default=DEFAULT_ALL_FINISHED_PROJECTION,
    )
    parser.add_argument("--golden-master", type=Path, default=DEFAULT_GOLDEN_MASTER)
    parser.add_argument("--google-cache", type=Path, default=DEFAULT_GOOGLE_CACHE)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--openlibrary-cache", type=Path, default=DEFAULT_OPENLIBRARY_CACHE
    )
    parser.add_argument("--refresh-openlibrary", action="store_true")
    parser.add_argument("--delay-seconds", type=float, default=0.15)
    parser.add_argument("--openlibrary-workers", type=int, default=8)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    base = pd.read_csv(args.all_finished_projection)
    words_per_page = float(base["projection_local_calibrated_words_per_page"].iloc[0])
    golden_master = load_match_table(args.golden_master)
    google_cache = load_match_table(args.google_cache)
    wide = add_online_sources(
        base,
        golden_master=golden_master,
        google_cache=google_cache,
        openlibrary_cache_path=args.openlibrary_cache,
        refresh_openlibrary=args.refresh_openlibrary,
        delay_seconds=args.delay_seconds,
        openlibrary_workers=args.openlibrary_workers,
    )
    wide = add_word_estimate_columns(wide, words_per_page)
    long = build_long_comparison(wide)
    summary = build_summary(long, wide)
    wide.to_csv(
        args.output_dir / "book_word_count_source_comparison_wide.csv", index=False
    )
    long.to_csv(
        args.output_dir / "book_word_count_source_comparison_long.csv", index=False
    )
    summary.to_csv(
        args.output_dir / "book_word_count_source_comparison_summary.csv",
        index=False,
    )
    outliers = build_outlier_report(long, wide)
    outliers.to_csv(
        args.output_dir / "book_word_count_source_comparison_outliers.csv",
        index=False,
    )
    write_markdown_table(
        outliers, args.output_dir / "book_word_count_source_comparison_outliers.md"
    )
    plot_source_scatter(
        long, args.output_dir / "book_word_count_source_scatter_local_vs_sources.png"
    )
    plot_error_by_source(
        long, args.output_dir / "book_word_count_source_error_by_source.png"
    )
    plot_source_totals(wide, args.output_dir / "book_word_count_source_totals.png")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
