"""Apples-to-apples R² tradeoff plots matching `rating_sim_focused.py`.

Matches the older plot semantics:
- x-axis: test regression R², floored at 0
- fixed drop fractions (50% and 80%)
- utility shown as percent gain over baseline utility

Adds separate panels for:
- All books
- Fiction/Literature
- Other
"""

from __future__ import annotations

import colorsys
import csv
import math
from pathlib import Path
from typing import Any

import numpy as np

OUTPUT_DIR = Path(__file__).parent
AI_ACTIONS_DIR = OUTPUT_DIR / "ai_actions"
PLOTS_DIR = OUTPUT_DIR / "plots"
PREDICTIONS_CSV = OUTPUT_DIR / "multi_source_holdout_predictions_long.csv"
GOLDEN_MASTER_CSV = OUTPUT_DIR / "golden_master.csv"

PLOTS_DIR.mkdir(exist_ok=True)

GROUPS = ("All", "Fiction/Literature", "Other")
TARGETS = ("avg_enjoyment", "avg_usefulness")
DROP_FRACTIONS = (0.50, 0.80)
SELF_NOISE = 0.40
N_SIMS = 1200
NOISE_FRACS = (
    0.03,
    0.07,
    0.10,
    0.15,
    0.20,
    0.30,
    0.50,
    0.75,
    1.0,
    1.5,
    2.0,
    3.0,
    5.0,
    8.0,
    12.0,
    20.0,
)
SIGNAL_RS = tuple(1.0 / math.sqrt(1.0 + value) for value in NOISE_FRACS)
MAIN_SIGNAL_COUNT = 3
RNG = np.random.default_rng(42)

TARGET_SPECS = {
    "avg_enjoyment": {
        "label": "Enjoyment",
        "utility_base": 1.3,
    },
    "avg_usefulness": {
        "label": "Usefulness",
        "utility_base": 1.8,
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

EMPIRICAL_COLORS = {
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


def regression_r2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    if y_true.size < 2:
        return 0.0
    ss_tot = float(np.sum((y_true - np.mean(y_true)) ** 2))
    if np.isclose(ss_tot, 0.0):
        return 0.0
    ss_res = float(np.sum((y_true - y_pred) ** 2))
    return max(0.0, 1.0 - ss_res / ss_tot)


def utility_values(values: np.ndarray, target: str) -> np.ndarray:
    base = float(TARGET_SPECS[target]["utility_base"])
    exponent = np.clip(values - 1.0, a_min=0.0, a_max=None)
    return np.power(base, exponent) - 1.0


def fixed_drop_metrics(scores: np.ndarray, actual: np.ndarray, target: str, drop_fraction: float) -> dict[str, float]:
    n = len(actual)
    drop_n = int(round(drop_fraction * n))
    keep_n = max(1, n - drop_n)
    order = np.argsort(scores)[::-1]
    keep = actual[order[:keep_n]]
    baseline_mean = float(np.mean(actual))
    keep_mean = float(np.mean(keep))
    util_all = utility_values(actual, target)
    util_keep = utility_values(keep, target)
    baseline_utility = float(np.mean(util_all))
    keep_utility = float(np.mean(util_keep))
    utility_pct_gain = (
        ((keep_utility - baseline_utility) / baseline_utility) * 100.0
        if baseline_utility > 0
        else 0.0
    )
    return {
        "rating_gain": keep_mean - baseline_mean,
        "utility_pct_gain": utility_pct_gain,
    }


def load_empirical_distributions() -> dict[tuple[str, str], np.ndarray]:
    values: dict[tuple[str, str], list[float]] = {}
    with GOLDEN_MASTER_CSV.open(newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            for target in TARGETS:
                value = parse_float(row.get(target))
                if value is None:
                    continue
                group = family_group(row.get("category", ""))
                values.setdefault((target, "All"), []).append(value)
                values.setdefault((target, group), []).append(value)
    return {
        key: np.array(value_list, dtype=float)
        for key, value_list in values.items()
        if value_list
    }


def load_empirical_points() -> list[dict[str, Any]]:
    actual_rows: dict[tuple[str, str], dict[str, float]] = {}
    score_maps: dict[tuple[str, str, str], dict[str, float]] = {}

    with PREDICTIONS_CSV.open(newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            target = row["target"]
            if target not in TARGETS:
                continue
            key = f"{row['title']}|{row['estimated_finish']}"
            group = family_group(row["category"])
            actual = parse_float(row[target])
            if actual is None:
                continue
            actual_rows.setdefault((target, "All"), {})[key] = actual
            actual_rows.setdefault((target, group), {})[key] = actual

            direct = {
                "goodreads_rating_verified": parse_float(row.get("goodreads_rating_verified")),
                "ol_rating_consensus": parse_float(row.get("ol_rating_consensus")),
                "amazon_rating_consensus": parse_float(row.get("amazon_rating_consensus")),
            }
            direct_mean_values = [value for value in direct.values() if value is not None]
            if direct_mean_values:
                direct["mean_site_rating"] = float(np.mean(direct_mean_values))

            for column, label in DIRECT_SCORE_SPECS:
                value = direct.get(column)
                if value is None:
                    continue
                score_maps.setdefault((target, "All", label), {})[key] = value
                score_maps.setdefault((target, group, label), {})[key] = value

            prediction = parse_float(row.get("prediction"))
            if prediction is not None:
                for feature_spec, model, label in MODEL_SCORE_SPECS:
                    if row["feature_spec"] == feature_spec and row["model"] == model:
                        score_maps.setdefault((target, "All", label), {})[key] = prediction
                        score_maps.setdefault((target, group, label), {})[key] = prediction

    rows: list[dict[str, Any]] = []
    for (target, group, scorer), score_map in score_maps.items():
        actual_map = actual_rows.get((target, group), {})
        common_keys = [key for key in actual_map if key in score_map]
        if len(common_keys) < 3:
            continue
        actual = np.array([actual_map[key] for key in common_keys], dtype=float)
        scores = np.array([score_map[key] for key in common_keys], dtype=float)
        r2 = regression_r2(actual, scores)
        row: dict[str, Any] = {
            "source": "empirical",
            "target": target,
            "group": group,
            "scorer": scorer,
            "test_r2": r2,
            "n_eval": len(common_keys),
        }
        for drop_fraction in DROP_FRACTIONS:
            metrics = fixed_drop_metrics(scores, actual, target, drop_fraction)
            label = int(round(drop_fraction * 100))
            row[f"drop{label}_rating_gain"] = metrics["rating_gain"]
            row[f"drop{label}_utility_pct_gain"] = metrics["utility_pct_gain"]
        rows.append(row)
    return rows


def simulate_one_config(
    empirical_values: np.ndarray,
    signal_r: float,
    target: str,
    n_signals: int,
    n_sims: int,
    rng: np.random.Generator,
) -> dict[str, np.ndarray]:
    n_total = 207 + 68
    test_r2 = np.zeros(n_sims, dtype=float)
    gains_50_rating = np.zeros(n_sims, dtype=float)
    gains_50_utility = np.zeros(n_sims, dtype=float)
    gains_80_rating = np.zeros(n_sims, dtype=float)
    gains_80_utility = np.zeros(n_sims, dtype=float)

    for sim in range(n_sims):
        idx = rng.choice(len(empirical_values), n_total, replace=True)
        true_y = empirical_values[idx] + rng.normal(0.0, 0.05, n_total)
        true_y = np.clip(true_y, 1.0, 5.0)

        y_var = float(np.var(true_y))
        noise_var = y_var * SELF_NOISE / (1.0 - SELF_NOISE + 1e-10)
        observed_y = true_y + rng.normal(0.0, math.sqrt(noise_var), n_total)
        observed_y = np.clip(observed_y, 1.0, 5.0)

        z_true = (true_y - np.mean(true_y)) / (np.std(true_y) + 1e-10)
        X = np.zeros((n_total, n_signals), dtype=float)
        for idx_signal in range(n_signals):
            noise = rng.standard_normal(n_total)
            X[:, idx_signal] = signal_r * z_true + math.sqrt(1.0 - signal_r**2) * noise

        X_train = X[:207]
        X_test = X[207:]
        y_train = observed_y[:207]
        y_test = observed_y[207:]
        design_train = np.column_stack([np.ones(len(X_train)), X_train])
        design_test = np.column_stack([np.ones(len(X_test)), X_test])
        beta = np.linalg.lstsq(design_train, y_train, rcond=None)[0]
        pred_test = design_test @ beta

        test_r2[sim] = regression_r2(y_test, pred_test)
        metrics50 = fixed_drop_metrics(pred_test, y_test, target, 0.50)
        metrics80 = fixed_drop_metrics(pred_test, y_test, target, 0.80)
        gains_50_rating[sim] = metrics50["rating_gain"]
        gains_50_utility[sim] = metrics50["utility_pct_gain"]
        gains_80_rating[sim] = metrics80["rating_gain"]
        gains_80_utility[sim] = metrics80["utility_pct_gain"]

    return {
        "test_r2": test_r2,
        "drop50_rating_gain": gains_50_rating,
        "drop50_utility_pct_gain": gains_50_utility,
        "drop80_rating_gain": gains_80_rating,
        "drop80_utility_pct_gain": gains_80_utility,
    }


def build_simulation_rows(empirical_distributions: dict[tuple[str, str], np.ndarray]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for target in TARGETS:
        for group in GROUPS:
            empirical_values = empirical_distributions[(target, group)]
            for noise_frac, signal_r in zip(NOISE_FRACS, SIGNAL_RS, strict=True):
                result = simulate_one_config(
                    empirical_values=empirical_values,
                    signal_r=signal_r,
                    target=target,
                    n_signals=MAIN_SIGNAL_COUNT,
                    n_sims=N_SIMS,
                    rng=RNG,
                )
                row: dict[str, Any] = {
                    "source": "simulation",
                    "target": target,
                    "group": group,
                    "noise_frac": noise_frac,
                    "signal_r": signal_r,
                    "mean_test_r2": float(np.mean(result["test_r2"])),
                }
                for metric in (
                    "drop50_rating_gain",
                    "drop50_utility_pct_gain",
                    "drop80_rating_gain",
                    "drop80_utility_pct_gain",
                ):
                    values = result[metric]
                    row[f"{metric}_mean"] = float(np.mean(values))
                    row[f"{metric}_p10"] = float(np.percentile(values, 10))
                    row[f"{metric}_p90"] = float(np.percentile(values, 90))
                rows.append(row)
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


def simulation_color(index: int, count: int) -> str:
    t = index / max(1, count - 1)
    hue = 0.65 - 0.55 * t
    r, g, b = colorsys.hls_to_rgb(hue, 0.48, 0.65)
    return f"#{int(r * 255):02x}{int(g * 255):02x}{int(b * 255):02x}"


def svg_wrap(width: int, height: int, body: list[str]) -> str:
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}" style="background:#ffffff">'
        + "".join(body)
        + "</svg>"
    )


def draw_line_points(body: list[str], points: list[tuple[float, float]], color: str) -> None:
    if not points:
        return
    path = " ".join(
        [
            ("M" if idx == 0 else "L") + f" {point[0]:.1f} {point[1]:.1f}"
            for idx, point in enumerate(points)
        ]
    )
    body.append(
        f'<path d="{path}" fill="none" stroke="{color}" stroke-width="2"/>'
    )


def scatter_panel(
    body: list[str],
    x0: int,
    y0: int,
    width: int,
    height: int,
    title: str,
    sim_rows: list[dict[str, Any]],
    empirical_rows: list[dict[str, Any]],
    sim_x_key: str,
    empirical_x_key: str,
    y_key: str,
    y_label_suffix: str,
) -> None:
    padding_left = 46
    padding_bottom = 36
    plot_w = width - padding_left - 12
    plot_h = height - 28 - padding_bottom
    draw_text(body, x0, y0 - 10, title, size=12, weight="700")
    draw_rect(body, x0 + padding_left, y0 + 10, plot_w, plot_h)

    x_values = [float(row[sim_x_key]) for row in sim_rows]
    x_values.extend(float(row[empirical_x_key]) for row in empirical_rows)
    y_values = [float(row[f"{y_key}_mean"]) for row in sim_rows]
    y_values.extend(float(row[f"{y_key}_p10"]) for row in sim_rows)
    y_values.extend(float(row[f"{y_key}_p90"]) for row in sim_rows)
    y_values.extend(float(row[y_key]) for row in empirical_rows)

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

    for frac in np.linspace(0, 1, 5):
        x_value = xmin + frac * (xmax - xmin)
        x = x0 + padding_left + frac * plot_w
        draw_text(body, x, y0 + 10 + plot_h + 22, f"{x_value:.2f}", size=9, anchor="middle")
        y_value = ymin + frac * (ymax - ymin)
        y = y0 + 10 + plot_h - frac * plot_h
        body.append(
            f'<line x1="{x0 + padding_left:.1f}" y1="{y:.1f}" '
            f'x2="{x0 + padding_left + plot_w:.1f}" y2="{y:.1f}" '
            f'stroke="#f3f4f6" stroke-width="1"/>'
        )
        draw_text(body, x0 + padding_left - 6, y + 4, f"{y_value:.1f}", size=9, anchor="end")

    zero_y = y0 + 10 + plot_h - ((0.0 - ymin) / (ymax - ymin)) * plot_h
    if y0 + 10 <= zero_y <= y0 + 10 + plot_h:
        body.append(
            f'<line x1="{x0 + padding_left:.1f}" y1="{zero_y:.1f}" '
            f'x2="{x0 + padding_left + plot_w:.1f}" y2="{zero_y:.1f}" '
            f'stroke="#9ca3af" stroke-width="1.2" stroke-dasharray="3 3"/>'
        )

    band_points_top: list[tuple[float, float]] = []
    band_points_bottom: list[tuple[float, float]] = []
    mean_points: list[tuple[float, float]] = []
    for row in sim_rows:
        x_val = float(row[sim_x_key])
        x = x0 + padding_left + ((x_val - xmin) / (xmax - xmin)) * plot_w
        y_mean = y0 + 10 + plot_h - ((float(row[f"{y_key}_mean"]) - ymin) / (ymax - ymin)) * plot_h
        y_p10 = y0 + 10 + plot_h - ((float(row[f"{y_key}_p10"]) - ymin) / (ymax - ymin)) * plot_h
        y_p90 = y0 + 10 + plot_h - ((float(row[f"{y_key}_p90"]) - ymin) / (ymax - ymin)) * plot_h
        band_points_top.append((x, y_p90))
        band_points_bottom.append((x, y_p10))
        mean_points.append((x, y_mean))

    if band_points_top and band_points_bottom:
        poly_points = band_points_top + list(reversed(band_points_bottom))
        point_str = " ".join(f"{x:.1f},{y:.1f}" for x, y in poly_points)
        body.append(
            f'<polygon points="{point_str}" fill="#60a5fa" opacity="0.18" stroke="none"/>'
        )
    draw_line_points(body, mean_points, "#2563eb")
    for x, y in mean_points:
        body.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="3.2" fill="#2563eb"/>')

    for row in empirical_rows:
        x_val = float(row[empirical_x_key])
        y_val = float(row[y_key])
        x = x0 + padding_left + ((x_val - xmin) / (xmax - xmin)) * plot_w
        y = y0 + 10 + plot_h - ((y_val - ymin) / (ymax - ymin)) * plot_h
        color = EMPIRICAL_COLORS.get(row["scorer"], "#111827")
        body.append(
            f'<rect x="{x - 4:.1f}" y="{y - 4:.1f}" width="8" height="8" fill="{color}"/>'
        )
        draw_text(body, x + 6, y + 3, row["scorer"], size=8, fill=color)

    draw_text(body, x0 + padding_left + plot_w / 2, y0 + height - 2, "Test regression R²", size=9, anchor="middle")
    draw_text(body, x0 + 8, y0 + 20, y_label_suffix, size=9, weight="700")


def plot_drop_figure(
    empirical_rows: list[dict[str, Any]],
    simulation_rows: list[dict[str, Any]],
    drop_fraction: float,
) -> Path:
    drop_label = int(round(drop_fraction * 100))
    body: list[str] = []
    draw_text(
        body,
        20,
        28,
        f"Fixed drop {drop_label}%: gain vs test regression R²",
        size=22,
        weight="700",
    )
    draw_text(
        body,
        20,
        48,
        "Simulation line/band matches the older PNG semantics; colored squares are real holdout scorers.",
        size=10,
    )

    col_specs = [
        ("avg_enjoyment", f"drop{drop_label}_rating_gain", "Enjoyment rating gain"),
        ("avg_enjoyment", f"drop{drop_label}_utility_pct_gain", "Enjoyment utility % gain"),
        ("avg_usefulness", f"drop{drop_label}_rating_gain", "Usefulness rating gain"),
        ("avg_usefulness", f"drop{drop_label}_utility_pct_gain", "Usefulness utility % gain"),
    ]

    for row_idx, group in enumerate(GROUPS):
        draw_text(body, 18, 105 + row_idx * 235, group, size=13, weight="700")
        for col_idx, (target, metric, title) in enumerate(col_specs):
            x = 110 + col_idx * 285
            y = 76 + row_idx * 235
            sim_subset = [
                row for row in simulation_rows if row["target"] == target and row["group"] == group
            ]
            emp_subset = [
                row for row in empirical_rows if row["target"] == target and row["group"] == group
            ]
            scatter_panel(
                body=body,
                x0=x,
                y0=y,
                width=255,
                height=195,
                title=title,
                sim_rows=sim_subset,
                empirical_rows=emp_subset,
                sim_x_key="mean_test_r2",
                empirical_x_key="test_r2",
                y_key=metric,
                y_label_suffix="gain",
            )

    svg = svg_wrap(1260, 810, body)
    output_path = OUTPUT_DIR / f"matched_sim_drop{drop_label}_vs_r2.svg"
    output_path.write_text(svg)
    return output_path


def write_summary_note(empirical_rows: list[dict[str, Any]], simulation_rows: list[dict[str, Any]], outputs: list[Path]) -> None:
    lines = ["# 2026-03-22 Matched R² Tradeoff Plots", ""]
    lines.append("Built `matched_r2_tradeoff_plots.py` to match the old `rating_sim_focused.py` semantics:")
    lines.append("")
    lines.append("- x-axis: test regression `R²`, floored at `0`")
    lines.append("- fixed drop rules: `50%` and `80%`")
    lines.append("- utility shown as percent gain over baseline utility")
    lines.append("- rows split into `All`, `Fiction/Literature`, and `Other`")
    lines.append("")
    lines.append("## Quick Readout")
    lines.append("")
    for target in TARGETS:
        label = TARGET_SPECS[target]["label"]
        lines.append(f"### {label}")
        for group in GROUPS:
            subset = [row for row in empirical_rows if row["target"] == target and row["group"] == group]
            if not subset:
                continue
            best50 = max(subset, key=lambda row: row["drop50_utility_pct_gain"])
            best80 = max(subset, key=lambda row: row["drop80_utility_pct_gain"])
            lines.append(
                f"- `{group}` best empirical scorer at drop `50%`: `{best50['scorer']}` with `R²={best50['test_r2']:.3f}` and utility gain `{best50['drop50_utility_pct_gain']:.1f}%`."
            )
            lines.append(
                f"- `{group}` best empirical scorer at drop `80%`: `{best80['scorer']}` with `R²={best80['test_r2']:.3f}` and utility gain `{best80['drop80_utility_pct_gain']:.1f}%`."
            )
        lines.append("")

    lines.append("## Output Files")
    lines.append("")
    for output in outputs:
        lines.append(f"- `{output.name}`")
    lines.append("- `matched_r2_tradeoff_empirical_points.csv`")
    lines.append("- `matched_r2_tradeoff_simulation_points.csv`")
    lines.append("")

    (AI_ACTIONS_DIR / "2026-03-22_matched_r2_tradeoff_plots.md").write_text("\n".join(lines))


def main() -> None:
    empirical_distributions = load_empirical_distributions()
    empirical_rows = load_empirical_points()
    simulation_rows = build_simulation_rows(empirical_distributions)

    write_csv(OUTPUT_DIR / "matched_r2_tradeoff_empirical_points.csv", empirical_rows)
    write_csv(OUTPUT_DIR / "matched_r2_tradeoff_simulation_points.csv", simulation_rows)

    outputs = [
        plot_drop_figure(empirical_rows, simulation_rows, 0.50),
        plot_drop_figure(empirical_rows, simulation_rows, 0.80),
    ]
    write_summary_note(empirical_rows, simulation_rows, outputs)
    print("Saved matched R² tradeoff outputs.")


if __name__ == "__main__":
    main()
