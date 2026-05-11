"""Analyze the tradeoff between test-set signal and cutoff gains.

Outputs:
- empirical and simulation CSVs
- SVG scatter grids for rating-gain and utility-gain tradeoffs
- short markdown summary in ai_actions/

The x-axis is squared Pearson correlation on the holdout (`rank_r2`), which is
the most comparable notion of "R²" when mixing raw site scores with model
predictions for ranking / cutoff decisions.
"""

from __future__ import annotations

import colorsys
import csv
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

OUTPUT_DIR = Path(__file__).parent
AI_ACTIONS_DIR = OUTPUT_DIR / "ai_actions"
PREDICTIONS_CSV = OUTPUT_DIR / "multi_source_holdout_predictions_long.csv"
GOLDEN_MASTER_CSV = OUTPUT_DIR / "golden_master.csv"

GROUP_ORDER = ("All", "Fiction/Literature", "Other")
TARGET_ORDER = ("avg_enjoyment", "avg_usefulness")
DROP_FRACTIONS = tuple(round(value, 2) for value in np.arange(0.05, 0.81, 0.05))
FIXED_DROP_FRACTIONS = (0.50, 0.80)
SIM_SIGNAL_COUNTS = (1, 2, 3, 4, 5)
SIM_RHOS = (0.15, 0.25, 0.35, 0.45, 0.50)
SIM_N = 500
MIN_SIDE_N = 5
RNG = np.random.default_rng(42)

TARGET_CONFIGS = {
    "avg_enjoyment": {
        "label": "Enjoyment",
        "utility_base": 1.3,
        "reliability_single": 0.77,
    },
    "avg_usefulness": {
        "label": "Usefulness",
        "utility_base": 1.8,
        "reliability_single": 0.86,
    },
}

DIRECT_SCORE_SPECS = (
    ("goodreads_rating_verified", "GR"),
    ("ol_rating_consensus", "OL"),
    ("amazon_rating_consensus", "AMZ"),
    ("mean_site_rating", "Mean"),
)

MODEL_SCORE_SPECS = (
    ("preread_base", "Category mean", "Category"),
    ("preread_base", "Global mean", "Global"),
    ("preread_goodreads", "Ridge", "GR Ridge"),
    ("preread_openlibrary", "Ridge", "OL Ridge"),
    ("preread_amazon", "Ridge", "AMZ Ridge"),
    ("preread_goodreads_amazon", "Ridge", "GR+AMZ"),
    ("preread_all_sources", "Ridge", "All Ridge"),
    ("preread_all_sources", "GBM", "All GBM"),
)

EMPIRICAL_POINT_COLORS = {
    "GR": "#2563eb",
    "OL": "#0891b2",
    "AMZ": "#f59e0b",
    "Mean": "#7c3aed",
    "Category": "#374151",
    "Global": "#6b7280",
    "GR Ridge": "#1d4ed8",
    "OL Ridge": "#0f766e",
    "AMZ Ridge": "#d97706",
    "GR+AMZ": "#db2777",
    "All Ridge": "#16a34a",
    "All GBM": "#dc2626",
}


@dataclass(frozen=True)
class HoldoutRecord:
    key: str
    category: str
    target: str
    actual: float
    goodreads: float | None
    openlibrary: float | None
    amazon: float | None


def parse_float(value: Any) -> float | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    try:
        return float(text)
    except ValueError:
        return None


def family_group(category: str) -> str:
    text = (category or "").strip()
    if text in {"fiction", "Literature"}:
        return "Fiction/Literature"
    return "Other"


def standard_normal_cdf(values: np.ndarray) -> np.ndarray:
    array = np.asarray(values, dtype=float)
    scalar_input = array.ndim == 0
    flat = np.atleast_1d(array).astype(float)
    probs = np.array(
        [0.5 * (1.0 + math.erf(value / math.sqrt(2.0))) for value in flat],
        dtype=float,
    )
    if scalar_input:
        return np.array(probs[0])
    return probs.reshape(array.shape)


def empirical_quantile_map(
    latent_values: np.ndarray,
    target_std: float,
    empirical_values: np.ndarray,
) -> np.ndarray:
    standardized = latent_values / target_std
    probs = np.clip(standard_normal_cdf(standardized), 1e-6, 1 - 1e-6)
    return np.quantile(empirical_values, probs, method="linear")


def safe_corr(x: np.ndarray, y: np.ndarray) -> float:
    if x.size < 2:
        return math.nan
    if np.allclose(np.std(x), 0.0) or np.allclose(np.std(y), 0.0):
        return math.nan
    return float(np.corrcoef(x, y)[0, 1])


def correlation_r2(x: np.ndarray, y: np.ndarray) -> float:
    corr = safe_corr(x, y)
    return corr * corr if not math.isnan(corr) else math.nan


def regression_r2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    if y_true.size < 2:
        return math.nan
    total = float(np.sum((y_true - np.mean(y_true)) ** 2))
    if np.isclose(total, 0.0):
        return math.nan
    residual = float(np.sum((y_true - y_pred) ** 2))
    return 1.0 - (residual / total)


def utility_transform(values: np.ndarray, target: str) -> np.ndarray:
    base = float(TARGET_CONFIGS[target]["utility_base"])
    exponent = np.clip(values - 1.0, a_min=0.0, a_max=None)
    return np.power(base, exponent) - 1.0


def score_to_prediction(score: np.ndarray, actual: np.ndarray) -> np.ndarray:
    if score.size == 0:
        return score
    if np.allclose(np.std(score), 0.0):
        return np.full_like(score, np.mean(actual))
    x = np.column_stack([np.ones(len(score)), score])
    beta = np.linalg.lstsq(x, actual, rcond=None)[0]
    return x @ beta


def evaluate_cut_strategy(
    scores: np.ndarray,
    actual: np.ndarray,
    target: str,
    drop_fraction: float,
) -> dict[str, float]:
    n = len(actual)
    drop_n = int(round(drop_fraction * n))
    keep_n = max(1, n - drop_n)
    order = np.argsort(scores)[::-1]
    keep_idx = order[:keep_n]
    keep = actual[keep_idx]
    utility = utility_transform(actual, target)
    keep_utility = utility_transform(keep, target)
    return {
        "drop_fraction": drop_fraction,
        "keep_n": keep_n,
        "rating_gain": float(np.mean(keep) - np.mean(actual)),
        "utility_gain": float(np.mean(keep_utility) - np.mean(utility)),
        "keep_mean": float(np.mean(keep)),
        "keep_utility_mean": float(np.mean(keep_utility)),
    }


def evaluate_scorer(
    scores: np.ndarray,
    actual: np.ndarray,
    target: str,
) -> dict[str, float]:
    predicted = score_to_prediction(scores, actual)
    rank_r2 = correlation_r2(scores, actual)
    reg_r2 = regression_r2(actual, predicted)

    optimal_rows = []
    for drop_fraction in DROP_FRACTIONS:
        drop_n = int(round(drop_fraction * len(actual)))
        keep_n = len(actual) - drop_n
        if keep_n < MIN_SIDE_N or drop_n < MIN_SIDE_N:
            continue
        optimal_rows.append(evaluate_cut_strategy(scores, actual, target, drop_fraction))
    if optimal_rows:
        optimal = max(optimal_rows, key=lambda row: (row["utility_gain"], row["rating_gain"]))
    else:
        optimal = evaluate_cut_strategy(scores, actual, target, 0.0)

    drop50 = evaluate_cut_strategy(scores, actual, target, 0.50)
    drop80 = evaluate_cut_strategy(scores, actual, target, 0.80)

    return {
        "rank_r2": rank_r2,
        "regression_r2": reg_r2,
        "optimal_drop_fraction": float(optimal["drop_fraction"]),
        "optimal_rating_gain": float(optimal["rating_gain"]),
        "optimal_utility_gain": float(optimal["utility_gain"]),
        "drop50_rating_gain": float(drop50["rating_gain"]),
        "drop50_utility_gain": float(drop50["utility_gain"]),
        "drop80_rating_gain": float(drop80["rating_gain"]),
        "drop80_utility_gain": float(drop80["utility_gain"]),
    }


def load_holdout_records() -> tuple[
    dict[str, dict[str, HoldoutRecord]],
    dict[str, dict[tuple[str, str], dict[str, float]]],
]:
    records_by_target: dict[str, dict[str, HoldoutRecord]] = {target: {} for target in TARGET_ORDER}
    model_scores: dict[str, dict[tuple[str, str], dict[str, float]]] = {
        target: {} for target in TARGET_ORDER
    }

    with PREDICTIONS_CSV.open(newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            target = row["target"]
            if target not in TARGET_ORDER:
                continue
            key = f"{row['title']}|{row['estimated_finish']}"
            actual = parse_float(row[target])
            if actual is None:
                continue
            record = HoldoutRecord(
                key=key,
                category=row["category"],
                target=target,
                actual=actual,
                goodreads=parse_float(row.get("goodreads_rating_verified")),
                openlibrary=parse_float(row.get("ol_rating_consensus")),
                amazon=parse_float(row.get("amazon_rating_consensus")),
            )
            records_by_target[target][key] = record

            scorer_key = (row["feature_spec"], row["model"])
            prediction = parse_float(row.get("prediction"))
            if prediction is not None:
                model_scores[target].setdefault(scorer_key, {})[key] = prediction
    return records_by_target, model_scores


def compute_train_counts() -> dict[str, int]:
    counts = {"All": 0, "Fiction/Literature": 0, "Other": 0}
    with GOLDEN_MASTER_CSV.open(newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            source = row.get("source", "")
            if "Holdout 2026" in source:
                continue
            counts["All"] += 1
            counts[family_group(row.get("category", ""))] += 1
    return counts


def available_records_for_group(
    records: dict[str, HoldoutRecord], group: str
) -> list[HoldoutRecord]:
    if group == "All":
        return list(records.values())
    return [record for record in records.values() if family_group(record.category) == group]


def build_empirical_points(
    records_by_target: dict[str, dict[str, HoldoutRecord]],
    model_scores: dict[str, dict[tuple[str, str], dict[str, float]]],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []

    for target in TARGET_ORDER:
        target_records = records_by_target[target]
        for group in GROUP_ORDER:
            group_records = available_records_for_group(target_records, group)
            actual_map = {record.key: record.actual for record in group_records}
            if len(actual_map) < 3:
                continue

            direct_scores: dict[str, dict[str, float]] = {}
            for record in group_records:
                site_values = [
                    value
                    for value in (record.goodreads, record.openlibrary, record.amazon)
                    if value is not None
                ]
                direct_scores.setdefault("goodreads_rating_verified", {})
                if record.goodreads is not None:
                    direct_scores["goodreads_rating_verified"][record.key] = record.goodreads
                direct_scores.setdefault("ol_rating_consensus", {})
                if record.openlibrary is not None:
                    direct_scores["ol_rating_consensus"][record.key] = record.openlibrary
                direct_scores.setdefault("amazon_rating_consensus", {})
                if record.amazon is not None:
                    direct_scores["amazon_rating_consensus"][record.key] = record.amazon
                if site_values:
                    direct_scores.setdefault("mean_site_rating", {})[record.key] = float(
                        np.mean(site_values)
                    )

            for column, short_label in DIRECT_SCORE_SPECS:
                score_map = direct_scores.get(column, {})
                common_keys = [key for key in actual_map if key in score_map]
                if len(common_keys) < 3:
                    continue
                actual = np.array([actual_map[key] for key in common_keys], dtype=float)
                scores = np.array([score_map[key] for key in common_keys], dtype=float)
                metrics = evaluate_scorer(scores, actual, target)
                rows.append(
                    {
                        "source": "empirical",
                        "group": group,
                        "target": target,
                        "target_label": TARGET_CONFIGS[target]["label"],
                        "scorer": short_label,
                        "scorer_type": "direct",
                        "n_eval": len(common_keys),
                        **metrics,
                    }
                )

            for feature_spec, model, short_label in MODEL_SCORE_SPECS:
                score_map = model_scores[target].get((feature_spec, model), {})
                common_keys = [key for key in actual_map if key in score_map]
                if len(common_keys) < 3:
                    continue
                actual = np.array([actual_map[key] for key in common_keys], dtype=float)
                scores = np.array([score_map[key] for key in common_keys], dtype=float)
                metrics = evaluate_scorer(scores, actual, target)
                rows.append(
                    {
                        "source": "empirical",
                        "group": group,
                        "target": target,
                        "target_label": TARGET_CONFIGS[target]["label"],
                        "scorer": short_label,
                        "feature_spec": feature_spec,
                        "model": model,
                        "scorer_type": "model",
                        "n_eval": len(common_keys),
                        **metrics,
                    }
                )
    return rows


def generate_simulation_dataset(
    empirical_values: np.ndarray,
    reliability_single: float,
    rho: float,
    n_signals: int,
    n_train: int,
    n_test: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    n_total = n_train + n_test
    latent = rng.normal(size=n_total)
    rel = reliability_single
    y1_latent = (math.sqrt(rel) * latent) + (
        math.sqrt(1.0 - rel) * rng.normal(size=n_total)
    )
    y2_latent = (math.sqrt(rel) * latent) + (
        math.sqrt(1.0 - rel) * rng.normal(size=n_total)
    )
    y_avg_latent = (y1_latent + y2_latent) / 2.0
    y_avg_std = math.sqrt((1.0 + rel) / 2.0)
    y = empirical_quantile_map(y_avg_latent, y_avg_std, empirical_values)

    X = np.zeros((n_total, n_signals), dtype=float)
    for idx in range(n_signals):
        X[:, idx] = (rho * latent) + (math.sqrt(1.0 - rho**2) * rng.normal(size=n_total))
    return X[:n_train], X[n_train:], y[:n_train], y[n_train:]


def summarize_numeric(values: list[float]) -> tuple[float, float, float]:
    array = np.array(
        [value for value in values if value is not None and math.isfinite(float(value))],
        dtype=float,
    )
    if array.size == 0:
        return (math.nan, math.nan, math.nan)
    return (
        float(np.percentile(array, 10)),
        float(np.percentile(array, 50)),
        float(np.percentile(array, 90)),
    )


def build_simulation_points(
    records_by_target: dict[str, dict[str, HoldoutRecord]],
    train_counts: dict[str, int],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for target in TARGET_ORDER:
        for group in GROUP_ORDER:
            group_records = available_records_for_group(records_by_target[target], group)
            actual_values = np.array([record.actual for record in group_records], dtype=float)
            if actual_values.size < 3:
                continue
            n_train = train_counts[group]
            n_test = len(group_records)
            reliability = float(TARGET_CONFIGS[target]["reliability_single"])

            for n_signals in SIM_SIGNAL_COUNTS:
                for rho in SIM_RHOS:
                    sim_metrics: list[dict[str, float]] = []
                    for _ in range(SIM_N):
                        X_train, X_test, y_train, y_test = generate_simulation_dataset(
                            empirical_values=actual_values,
                            reliability_single=reliability,
                            rho=rho,
                            n_signals=n_signals,
                            n_train=n_train,
                            n_test=n_test,
                            rng=RNG,
                        )
                        train_design = np.column_stack([np.ones(len(X_train)), X_train])
                        test_design = np.column_stack([np.ones(len(X_test)), X_test])
                        beta = np.linalg.lstsq(train_design, y_train, rcond=None)[0]
                        test_pred = test_design @ beta
                        sim_metrics.append(evaluate_scorer(test_pred, y_test, target))

                    summary: dict[str, Any] = {
                        "source": "simulation",
                        "group": group,
                        "target": target,
                        "target_label": TARGET_CONFIGS[target]["label"],
                        "scenario": f"{n_signals}x{rho:.2f}",
                        "n_signals": n_signals,
                        "rho": rho,
                        "n_train": n_train,
                        "n_eval": n_test,
                    }
                    for metric in [
                        "rank_r2",
                        "regression_r2",
                        "optimal_drop_fraction",
                        "optimal_rating_gain",
                        "optimal_utility_gain",
                        "drop50_rating_gain",
                        "drop50_utility_gain",
                        "drop80_rating_gain",
                        "drop80_utility_gain",
                    ]:
                        p10, median, p90 = summarize_numeric([row[metric] for row in sim_metrics])
                        summary[f"{metric}_p10"] = p10
                        summary[f"{metric}_median"] = median
                        summary[f"{metric}_p90"] = p90
                    rows.append(summary)
    return rows


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def escape_xml(text: str) -> str:
    return (
        str(text)
        .replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
    )


def draw_text(
    body: list[str],
    x: float,
    y: float,
    text: str,
    size: int = 12,
    weight: str = "400",
    anchor: str = "start",
    fill: str = "#111827",
) -> None:
    body.append(
        f'<text x="{x:.1f}" y="{y:.1f}" font-family="Helvetica, Arial, sans-serif" '
        f'font-size="{size}" font-weight="{weight}" text-anchor="{anchor}" fill="{fill}">'
        f"{escape_xml(text)}</text>"
    )


def draw_rect(
    body: list[str],
    x: float,
    y: float,
    width: float,
    height: float,
    fill: str = "#ffffff",
    stroke: str = "#d1d5db",
    stroke_width: float = 1.0,
) -> None:
    body.append(
        f'<rect x="{x:.1f}" y="{y:.1f}" width="{width:.1f}" height="{height:.1f}" '
        f'fill="{fill}" stroke="{stroke}" stroke-width="{stroke_width:.1f}"/>'
    )


def simulation_color(n_signals: int) -> str:
    hue = 0.62 - 0.09 * (n_signals - 1)
    r, g, b = colorsys.hls_to_rgb(hue, 0.48, 0.65)
    return f"#{int(r * 255):02x}{int(g * 255):02x}{int(b * 255):02x}"


def svg_wrap(width: int, height: int, body: list[str]) -> str:
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}" style="background:#ffffff">'
        + "".join(body)
        + "</svg>"
    )


def scatter_panel(
    body: list[str],
    x0: int,
    y0: int,
    width: int,
    height: int,
    title: str,
    empirical_rows: list[dict[str, Any]],
    simulation_rows: list[dict[str, Any]],
    x_metric: str,
    y_metric: str,
    annotate_optimal_drop: bool,
) -> None:
    padding_left = 58
    padding_bottom = 40
    plot_w = width - padding_left - 18
    plot_h = height - 30 - padding_bottom
    draw_text(body, x0, y0 - 12, title, size=13, weight="700")
    draw_rect(body, x0 + padding_left, y0 + 10, plot_w, plot_h)

    x_values = []
    y_values = []
    for row in empirical_rows:
        x_values.append(float(row[x_metric]))
        y_values.append(float(row[y_metric]))
    for row in simulation_rows:
        x_values.append(float(row[f"{x_metric}_median"]))
        y_values.append(float(row[f"{y_metric}_median"]))

    xmin = min(0.0, min(x_values) if x_values else 0.0)
    xmax = max(0.01, max(x_values) if x_values else 0.01)
    ymin = min(0.0, min(y_values) if y_values else 0.0)
    ymax = max(0.01, max(y_values) if y_values else 0.01)
    if np.isclose(xmax, xmin):
        xmax = xmin + 0.1
    if np.isclose(ymax, ymin):
        ymax = ymin + 0.1
    xpad = 0.04 * (xmax - xmin)
    ypad = 0.08 * (ymax - ymin)
    xmin -= xpad
    xmax += xpad
    ymin -= ypad
    ymax += ypad

    zero_y = y0 + 10 + plot_h - ((0.0 - ymin) / (ymax - ymin)) * plot_h
    if y0 + 10 <= zero_y <= y0 + 10 + plot_h:
        body.append(
            f'<line x1="{x0 + padding_left:.1f}" y1="{zero_y:.1f}" '
            f'x2="{x0 + padding_left + plot_w:.1f}" y2="{zero_y:.1f}" '
            f'stroke="#d1d5db" stroke-width="1.2"/>'
        )

    for frac in np.linspace(0, 1, 5):
        x_value = xmin + frac * (xmax - xmin)
        x = x0 + padding_left + frac * plot_w
        draw_text(body, x, y0 + 10 + plot_h + 22, f"{x_value:.2f}", size=10, anchor="middle")

        y_value = ymin + frac * (ymax - ymin)
        y = y0 + 10 + plot_h - frac * plot_h
        body.append(
            f'<line x1="{x0 + padding_left:.1f}" y1="{y:.1f}" '
            f'x2="{x0 + padding_left + plot_w:.1f}" y2="{y:.1f}" '
            f'stroke="#f3f4f6" stroke-width="1"/>'
        )
        draw_text(body, x0 + padding_left - 8, y + 4, f"{y_value:.2f}", size=10, anchor="end")

    for row in simulation_rows:
        x_value = float(row[f"{x_metric}_median"])
        y_value = float(row[f"{y_metric}_median"])
        if not math.isfinite(x_value) or not math.isfinite(y_value):
            continue
        x = x0 + padding_left + ((x_value - xmin) / (xmax - xmin)) * plot_w
        y = y0 + 10 + plot_h - ((y_value - ymin) / (ymax - ymin)) * plot_h
        color = simulation_color(int(row["n_signals"]))
        body.append(
            f'<circle cx="{x:.1f}" cy="{y:.1f}" r="4.2" fill="{color}" opacity="0.85"/>'
        )
        if annotate_optimal_drop:
            drop_pct = int(round(float(row["optimal_drop_fraction_median"]) * 100))
            draw_text(body, x + 6, y - 4, f"{drop_pct}%", size=9, fill=color)

    for row in empirical_rows:
        x_value = float(row[x_metric])
        y_value = float(row[y_metric])
        if not math.isfinite(x_value) or not math.isfinite(y_value):
            continue
        x = x0 + padding_left + ((x_value - xmin) / (xmax - xmin)) * plot_w
        y = y0 + 10 + plot_h - ((y_value - ymin) / (ymax - ymin)) * plot_h
        color = EMPIRICAL_POINT_COLORS.get(row["scorer"], "#111827")
        body.append(
            f'<rect x="{x - 4:.1f}" y="{y - 4:.1f}" width="8" height="8" '
            f'fill="{color}" opacity="0.95"/>'
        )
        if annotate_optimal_drop:
            label = f"{row['scorer']} {int(round(float(row['optimal_drop_fraction']) * 100))}%"
        else:
            label = row["scorer"]
        draw_text(body, x + 6, y + 4, label, size=9, fill=color)

    draw_text(
        body,
        x0 + padding_left + plot_w / 2,
        y0 + height - 5,
        "Test rank R²",
        size=10,
        anchor="middle",
    )


def plot_tradeoff_grids(
    empirical_rows: list[dict[str, Any]],
    simulation_rows: list[dict[str, Any]],
) -> None:
    plot_specs = [
        ("rating_gain", "Rating gain"),
        ("utility_gain", "Utility gain"),
    ]
    cutoff_specs = [
        ("optimal", "Optimal drop", True),
        ("drop50", "Drop 50%", False),
        ("drop80", "Drop 80%", False),
    ]

    for target in TARGET_ORDER:
        target_label = TARGET_CONFIGS[target]["label"]
        for gain_kind, gain_label in plot_specs:
            body: list[str] = []
            draw_text(
                body,
                22,
                30,
                f"{target_label}: test rank R² vs {gain_label.lower()}",
                size=22,
                weight="700",
            )
            for row_idx, group in enumerate(GROUP_ORDER):
                draw_text(body, 20, 95 + row_idx * 245, group, size=14, weight="700")
                for col_idx, (cutoff_key, panel_title, annotate) in enumerate(cutoff_specs):
                    x = 120 + col_idx * 360
                    y = 70 + row_idx * 245
                    empirical_subset = [
                        row
                        for row in empirical_rows
                        if row["target"] == target and row["group"] == group
                    ]
                    simulation_subset = [
                        row
                        for row in simulation_rows
                        if row["target"] == target and row["group"] == group
                    ]
                    scatter_panel(
                        body=body,
                        x0=x,
                        y0=y,
                        width=330,
                        height=210,
                        title=panel_title,
                        empirical_rows=empirical_subset,
                        simulation_rows=simulation_subset,
                        x_metric="rank_r2",
                        y_metric=f"{cutoff_key}_{gain_kind}",
                        annotate_optimal_drop=annotate,
                    )

            legend_y = 820
            draw_text(body, 20, legend_y, "Simulation colors = number of signals", size=11)
            for idx, n_signals in enumerate(SIM_SIGNAL_COUNTS):
                color = simulation_color(n_signals)
                draw_rect(body, 235 + idx * 80, legend_y - 12, 14, 14, fill=color, stroke=color)
                draw_text(body, 255 + idx * 80, legend_y, f"{n_signals} signal", size=10)
            draw_text(
                body,
                20,
                legend_y + 24,
                "Empirical squares = real holdout scorers; optimal panels label drop fraction.",
                size=10,
            )

            svg = svg_wrap(1230, 860, body)
            filename = (
                f"r2_vs_{gain_kind}_tradeoff_{TARGET_CONFIGS[target]['label'].lower()}.svg"
            )
            (OUTPUT_DIR / filename).write_text(svg)


def combine_for_plot(
    empirical_rows: list[dict[str, Any]],
    simulation_rows: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    empirical_plot_rows = []
    for row in empirical_rows:
        empirical_plot_rows.append(
            {
                **row,
                "optimal_rating_gain": row["optimal_rating_gain"],
                "optimal_utility_gain": row["optimal_utility_gain"],
                "drop50_rating_gain": row["drop50_rating_gain"],
                "drop50_utility_gain": row["drop50_utility_gain"],
                "drop80_rating_gain": row["drop80_rating_gain"],
                "drop80_utility_gain": row["drop80_utility_gain"],
            }
        )

    simulation_plot_rows = []
    for row in simulation_rows:
        simulation_plot_rows.append(
            {
                **row,
                "optimal_rating_gain": row["optimal_rating_gain_median"],
                "optimal_utility_gain": row["optimal_utility_gain_median"],
                "drop50_rating_gain": row["drop50_rating_gain_median"],
                "drop50_utility_gain": row["drop50_utility_gain_median"],
                "drop80_rating_gain": row["drop80_rating_gain_median"],
                "drop80_utility_gain": row["drop80_utility_gain_median"],
            }
        )
    return empirical_plot_rows, simulation_plot_rows


def write_summary_note(
    empirical_rows: list[dict[str, Any]],
    simulation_rows: list[dict[str, Any]],
) -> None:
    lines = ["# 2026-03-22 R² vs Cutoff Tradeoff", ""]
    lines.append("Built `r2_cutoff_tradeoff_analysis.py` to connect holdout signal strength to cutoff gains.")
    lines.append("")
    lines.append("Notes:")
    lines.append("- x-axis uses squared Pearson correlation on the holdout (`rank_r2`), so raw site ratings and model predictions are comparable as ranking signals.")
    lines.append("- Groups are `All`, `Fiction/Literature`, and `Other`.")
    lines.append("- Utility uses `1.3^(rating-1)-1` for enjoyment and `1.8^(rating-1)-1` for usefulness.")
    lines.append("")

    for target in TARGET_ORDER:
        lines.append(f"## {TARGET_CONFIGS[target]['label']}")
        lines.append("")
        for group in GROUP_ORDER:
            empirical_subset = [
                row
                for row in empirical_rows
                if row["target"] == target
                and row["group"] == group
                and math.isfinite(float(row["rank_r2"]))
            ]
            simulation_subset = [
                row
                for row in simulation_rows
                if row["target"] == target
                and row["group"] == group
                and math.isfinite(float(row["rank_r2_median"]))
            ]
            if not empirical_subset or not simulation_subset:
                continue
            best_empirical = max(
                empirical_subset,
                key=lambda row: row["optimal_utility_gain"],
            )
            best_sim = max(
                simulation_subset,
                key=lambda row: row["optimal_utility_gain_median"],
            )
            lines.append(f"### {group}")
            lines.append(
                f"- Best empirical scorer by optimal utility gain: `{best_empirical['scorer']}` with `rank_r2={best_empirical['rank_r2']:.3f}`, drop `{best_empirical['optimal_drop_fraction']:.0%}`, utility gain `+{best_empirical['optimal_utility_gain']:.3f}`."
            )
            lines.append(
                f"- Best simulation point in the grid: `{best_sim['scenario']}` with median `rank_r2={best_sim['rank_r2_median']:.3f}`, drop `{best_sim['optimal_drop_fraction_median']:.0%}`, utility gain `+{best_sim['optimal_utility_gain_median']:.3f}`."
            )
            if group == "Fiction/Literature":
                other = max(
                    [
                        row
                        for row in empirical_rows
                        if row["target"] == target and row["group"] == "Other"
                        and math.isfinite(float(row["rank_r2"]))
                    ],
                    key=lambda row: row["optimal_utility_gain"],
                )
                lines.append(
                    f"- Fiction/Literature is the main weak spot: its best empirical utility gain is `+{best_empirical['optimal_utility_gain']:.3f}` versus `+{other['optimal_utility_gain']:.3f}` for `Other`."
                )
            lines.append("")

    lines.append("## Output Files")
    lines.append("")
    lines.append("- `r2_tradeoff_empirical_points.csv`")
    lines.append("- `r2_tradeoff_simulation_points.csv`")
    lines.append("- `r2_vs_rating_gain_tradeoff_enjoyment.svg`")
    lines.append("- `r2_vs_utility_gain_tradeoff_enjoyment.svg`")
    lines.append("- `r2_vs_rating_gain_tradeoff_usefulness.svg`")
    lines.append("- `r2_vs_utility_gain_tradeoff_usefulness.svg`")
    lines.append("")

    (AI_ACTIONS_DIR / "2026-03-22_r2_cutoff_tradeoff.md").write_text("\n".join(lines))


def main() -> None:
    records_by_target, model_scores = load_holdout_records()
    train_counts = compute_train_counts()

    empirical_rows = build_empirical_points(records_by_target, model_scores)
    write_csv(OUTPUT_DIR / "r2_tradeoff_empirical_points.csv", empirical_rows)

    simulation_rows = build_simulation_points(records_by_target, train_counts)
    write_csv(OUTPUT_DIR / "r2_tradeoff_simulation_points.csv", simulation_rows)

    empirical_plot_rows, simulation_plot_rows = combine_for_plot(
        empirical_rows, simulation_rows
    )
    plot_tradeoff_grids(empirical_plot_rows, simulation_plot_rows)
    write_summary_note(empirical_rows, simulation_rows)
    print("Saved R² cutoff tradeoff outputs.")


if __name__ == "__main__":
    main()
