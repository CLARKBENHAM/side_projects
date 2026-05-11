"""Detailed Monte Carlo expectations for noisy book-rating prediction.

This version is intentionally light on dependencies so it can run in the
current project environment. It writes CSV summaries plus simple SVG charts.
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
EMPIRICAL_SOURCE_CSV = OUTPUT_DIR / "golden_master.csv"

N_TRAIN = 200
N_TEST = 68
N_SIM = 1500
KEEP_SHARES = (0.20, 0.50, 0.80)
RHO_GRID = tuple(round(value, 2) for value in np.arange(0.15, 0.501, 0.05))
MIXED_SIGNAL_LADDERS = (
    (0.15,),
    (0.15, 0.30),
    (0.15, 0.30, 0.45),
    (0.15, 0.30, 0.45, 0.50),
    (0.15, 0.30, 0.45, 0.50, 0.50),
)
DISTRIBUTION_COMPARISON_SCENARIOS = (
    (1, 0.15),
    (1, 0.30),
    (1, 0.50),
    (3, 0.15),
    (3, 0.30),
    (3, 0.50),
    (5, 0.15),
    (5, 0.30),
    (5, 0.50),
)
RNG = np.random.default_rng(42)


@dataclass(frozen=True)
class TargetConfig:
    name: str
    slug: str
    reliability_single: float
    mean: float
    std: float
    empirical_values: np.ndarray
    utility_kind: str
    utility_param: float


def load_target_configs() -> list[TargetConfig]:
    columns = {"avg_enjoyment": [], "avg_usefulness": []}
    with EMPIRICAL_SOURCE_CSV.open(newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            for column in columns:
                value = parse_float(row.get(column))
                if value is not None:
                    columns[column].append(value)

    configs: list[TargetConfig] = []
    specs = [
        ("Enjoyment", "enjoyment", "avg_enjoyment", 0.77, "power", 1.3),
        ("Usefulness", "usefulness", "avg_usefulness", 0.86, "exponential", 2.0),
    ]
    for name, slug, column, reliability, utility_kind, utility_param in specs:
        values = np.array(columns[column], dtype=float)
        if values.size == 0:
            raise ValueError(f"No empirical values found for {column}")
        configs.append(
            TargetConfig(
                name=name,
                slug=slug,
                reliability_single=reliability,
                mean=float(np.mean(values)),
                std=float(np.std(values, ddof=1)),
                empirical_values=np.sort(values),
                utility_kind=utility_kind,
                utility_param=utility_param,
            )
        )
    return configs


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


def spearman_brown(r: float, k: int = 2) -> float:
    return (k * r) / (1 + (k - 1) * r)


def safe_corr(x: np.ndarray, y: np.ndarray) -> float:
    if x.size < 2:
        return math.nan
    if np.allclose(np.std(x), 0.0) or np.allclose(np.std(y), 0.0):
        return math.nan
    return float(np.corrcoef(x, y)[0, 1])


def regression_r2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    if y_true.size < 2:
        return math.nan
    total = float(np.sum((y_true - np.mean(y_true)) ** 2))
    if np.isclose(total, 0.0):
        return math.nan
    residual = float(np.sum((y_true - y_pred) ** 2))
    return 1.0 - (residual / total)


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(np.abs(y_true - y_pred)))


def ols_predict(
    X_train: np.ndarray, y_train: np.ndarray, X_eval: np.ndarray
) -> np.ndarray:
    train_design = np.column_stack([np.ones(len(X_train)), X_train])
    eval_design = np.column_stack([np.ones(len(X_eval)), X_eval])
    beta = np.linalg.lstsq(train_design, y_train, rcond=None)[0]
    return eval_design @ beta


def utility_transform(values: np.ndarray, config: TargetConfig) -> np.ndarray:
    shifted = np.clip(values - 1.0, a_min=0.0, a_max=None)
    if config.utility_kind == "power":
        return np.power(shifted, config.utility_param)
    return np.power(config.utility_param, shifted) - 1.0


def utility_inverse(values: np.ndarray, config: TargetConfig) -> np.ndarray:
    clipped = np.clip(values, a_min=0.0, a_max=None)
    if config.utility_kind == "power":
        return np.power(clipped, 1.0 / config.utility_param) + 1.0
    return (np.log1p(clipped) / math.log(config.utility_param)) + 1.0


def zscore_transform(values: np.ndarray) -> tuple[np.ndarray, float, float]:
    mean = float(np.mean(values))
    std = float(np.std(values, ddof=1))
    if np.isclose(std, 0.0):
        return np.zeros_like(values), mean, 1.0
    return (values - mean) / std, mean, std


def zscore_inverse(values: np.ndarray, mean: float, std: float) -> np.ndarray:
    return (values * std) + mean


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


def gaussian_scale_map(
    latent_values: np.ndarray,
    target_std: float,
    mean: float,
    std: float,
) -> np.ndarray:
    standardized = latent_values / target_std
    scaled = mean + (std * standardized)
    return np.clip(scaled, 1.0, 5.0)


def generate_dataset(
    config: TargetConfig,
    rhos: tuple[float, ...],
    distribution_mode: str,
    n_train: int = N_TRAIN,
    n_test: int = N_TEST,
    predictor_reliability: float = 1.0,
    rng: np.random.Generator = RNG,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    n_total = n_train + n_test
    latent = rng.normal(size=n_total)

    rel = config.reliability_single
    y1_latent = (math.sqrt(rel) * latent) + (
        math.sqrt(1.0 - rel) * rng.normal(size=n_total)
    )
    y2_latent = (math.sqrt(rel) * latent) + (
        math.sqrt(1.0 - rel) * rng.normal(size=n_total)
    )
    y_avg_latent = (y1_latent + y2_latent) / 2.0
    y_avg_std = math.sqrt((1.0 + rel) / 2.0)

    X = np.zeros((n_total, len(rhos)), dtype=float)
    for idx, rho in enumerate(rhos):
        x_true = (rho * latent) + (math.sqrt(1.0 - rho**2) * rng.normal(size=n_total))
        if predictor_reliability < 1.0:
            X[:, idx] = (math.sqrt(predictor_reliability) * x_true) + (
                math.sqrt(1.0 - predictor_reliability) * rng.normal(size=n_total)
            )
        else:
            X[:, idx] = x_true

    if distribution_mode == "empirical":
        y = empirical_quantile_map(y_avg_latent, y_avg_std, config.empirical_values)
    elif distribution_mode == "gaussian":
        y = gaussian_scale_map(y_avg_latent, y_avg_std, config.mean, config.std)
    else:
        raise ValueError(f"Unsupported distribution mode: {distribution_mode}")

    return X[:n_train], X[n_train:], y[:n_train], y[n_train:]


def collect_prediction_metrics(
    config: TargetConfig,
    y_train: np.ndarray,
    y_test: np.ndarray,
    pred_train: np.ndarray,
    pred_test: np.ndarray,
    keep_shares: tuple[float, ...] = KEEP_SHARES,
) -> dict[str, float]:
    train_r = safe_corr(pred_train, y_train)
    test_r = safe_corr(pred_test, y_test)
    train_regression = regression_r2(y_train, pred_train)
    test_regression = regression_r2(y_test, pred_test)

    metrics = {
        "train_r": train_r,
        "test_r": test_r,
        "train_rank_r2": train_r**2 if not math.isnan(train_r) else math.nan,
        "test_rank_r2": test_r**2 if not math.isnan(test_r) else math.nan,
        "train_regression_r2": train_regression,
        "test_regression_r2": test_regression,
        "train_rmse": rmse(y_train, pred_train),
        "test_rmse": rmse(y_test, pred_test),
        "train_mae": mae(y_train, pred_train),
        "test_mae": mae(y_test, pred_test),
        "train_test_r_gap": (
            train_r - test_r if not math.isnan(train_r) and not math.isnan(test_r) else math.nan
        ),
        "train_test_regression_r2_gap": (
            train_regression - test_regression
            if not math.isnan(train_regression) and not math.isnan(test_regression)
            else math.nan
        ),
    }

    overall_mean = float(np.mean(y_test))
    overall_utility = float(np.mean(utility_transform(y_test, config)))
    order = np.argsort(pred_test)[::-1]
    for keep_share in keep_shares:
        keep_n = max(1, int(round(keep_share * len(y_test))))
        keep_idx = order[:keep_n]
        kept = y_test[keep_idx]
        kept_utility = utility_transform(kept, config)
        label = int(round(keep_share * 100))
        metrics[f"keep_{label}_mean"] = float(np.mean(kept))
        metrics[f"keep_{label}_gain"] = float(np.mean(kept) - overall_mean)
        metrics[f"keep_{label}_utility_mean"] = float(np.mean(kept_utility))
        metrics[f"keep_{label}_utility_gain"] = float(
            np.mean(kept_utility) - overall_utility
        )
    return metrics


def summarize_metrics(
    rows: list[dict[str, Any]],
    group_cols: list[str],
) -> list[dict[str, Any]]:
    groups: dict[tuple[Any, ...], list[dict[str, Any]]] = {}
    for row in rows:
        key = tuple(row[col] for col in group_cols)
        groups.setdefault(key, []).append(row)

    metric_cols = [
        key
        for key in rows[0].keys()
        if key not in group_cols and isinstance(rows[0][key], (int, float, np.floating))
    ]

    summary_rows: list[dict[str, Any]] = []
    for key in sorted(groups):
        subset = groups[key]
        summary: dict[str, Any] = dict(zip(group_cols, key, strict=True))
        for metric in metric_cols:
            values = np.array(
                [
                    float(row[metric])
                    for row in subset
                    if row[metric] is not None and not math.isnan(float(row[metric]))
                ],
                dtype=float,
            )
            if values.size == 0:
                summary[f"{metric}_p10"] = math.nan
                summary[f"{metric}_median"] = math.nan
                summary[f"{metric}_p90"] = math.nan
            else:
                summary[f"{metric}_p10"] = float(np.percentile(values, 10))
                summary[f"{metric}_median"] = float(np.percentile(values, 50))
                summary[f"{metric}_p90"] = float(np.percentile(values, 90))
        summary_rows.append(summary)
    return summary_rows


def run_base_simulation(
    config: TargetConfig,
    rhos: tuple[float, ...],
    distribution_mode: str,
    n_sim: int = N_SIM,
) -> list[dict[str, float]]:
    rows: list[dict[str, float]] = []
    for _ in range(n_sim):
        X_train, X_test, y_train, y_test = generate_dataset(
            config=config,
            rhos=rhos,
            distribution_mode=distribution_mode,
        )
        pred_train = ols_predict(X_train, y_train, X_train)
        pred_test = ols_predict(X_train, y_train, X_test)
        rows.append(
            {
                **collect_prediction_metrics(config, y_train, y_test, pred_train, pred_test),
                "mean_rho": float(np.mean(rhos)),
                "max_rho": float(np.max(rhos)),
                "min_rho": float(np.min(rhos)),
            }
        )
    return rows


def fit_with_target_variant(
    X_train: np.ndarray,
    X_test: np.ndarray,
    y_train: np.ndarray,
    variant: str,
    config: TargetConfig,
) -> tuple[np.ndarray, np.ndarray]:
    if variant == "raw":
        return ols_predict(X_train, y_train, X_train), ols_predict(X_train, y_train, X_test)

    if variant.startswith("zscore"):
        y_train_transformed, mean, std = zscore_transform(y_train)
        pred_train = ols_predict(X_train, y_train_transformed, X_train)
        pred_test = ols_predict(X_train, y_train_transformed, X_test)
        if variant.endswith("correct_inverse"):
            return zscore_inverse(pred_train, mean, std), zscore_inverse(pred_test, mean, std)
        return pred_train, pred_test

    if variant.startswith("utility"):
        y_train_transformed = utility_transform(y_train, config)
        pred_train = ols_predict(X_train, y_train_transformed, X_train)
        pred_test = ols_predict(X_train, y_train_transformed, X_test)
        if variant.endswith("correct_inverse"):
            return utility_inverse(pred_train, config), utility_inverse(pred_test, config)
        return pred_train, pred_test

    raise ValueError(f"Unsupported target variant: {variant}")


def run_homogeneous_grid(configs: list[TargetConfig]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for config in configs:
        for n_signals in range(1, 6):
            for rho in RHO_GRID:
                metrics = run_base_simulation(
                    config=config,
                    rhos=(rho,) * n_signals,
                    distribution_mode="gaussian",
                )
                for metric_row in metrics:
                    metric_row["target"] = config.name
                    metric_row["distribution_mode"] = "gaussian"
                    metric_row["n_signals"] = n_signals
                    metric_row["rho"] = rho
                rows.extend(metrics)
    return summarize_metrics(rows, ["target", "distribution_mode", "n_signals", "rho"])


def run_random_rho_ranges(configs: list[TargetConfig]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for config in configs:
        for n_signals in range(1, 6):
            for _ in range(N_SIM):
                rhos = tuple(float(RNG.uniform(RHO_GRID[0], RHO_GRID[-1])) for _ in range(n_signals))
                X_train, X_test, y_train, y_test = generate_dataset(
                    config=config,
                    rhos=rhos,
                    distribution_mode="gaussian",
                )
                pred_train = ols_predict(X_train, y_train, X_train)
                pred_test = ols_predict(X_train, y_train, X_test)
                rows.append(
                    {
                        "target": config.name,
                        "n_signals": n_signals,
                        "rho_min_draw": float(np.min(rhos)),
                        "rho_mean_draw": float(np.mean(rhos)),
                        "rho_max_draw": float(np.max(rhos)),
                        **collect_prediction_metrics(
                            config, y_train, y_test, pred_train, pred_test
                        ),
                    }
                )
    return summarize_metrics(rows, ["target", "n_signals"])


def run_mixed_signal_ladder(configs: list[TargetConfig]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for config in configs:
        for ladder in MIXED_SIGNAL_LADDERS:
            metrics = run_base_simulation(
                config=config,
                rhos=ladder,
                distribution_mode="gaussian",
            )
            vector_label = ", ".join(f"{rho:.2f}" for rho in ladder)
            for metric_row in metrics:
                metric_row["target"] = config.name
                metric_row["n_signals"] = len(ladder)
                metric_row["rho_vector"] = vector_label
            rows.extend(metrics)
    return summarize_metrics(rows, ["target", "n_signals", "rho_vector"])


def run_transform_mistake_simulation(
    config: TargetConfig,
    rhos: tuple[float, ...],
    n_sim: int = N_SIM,
) -> list[dict[str, Any]]:
    variants = (
        ("raw", "Raw target"),
        ("zscore_correct_inverse", "Z-score target, invert correctly"),
        ("zscore_no_inverse", "Z-score target, no inverse"),
        ("utility_correct_inverse", "Utility target, invert correctly"),
        ("utility_no_inverse", "Utility target, no inverse"),
    )
    rows: list[dict[str, Any]] = []
    for _ in range(n_sim):
        X_train, X_test, y_train, y_test = generate_dataset(
            config=config,
            rhos=rhos,
            distribution_mode="empirical",
        )
        for variant, label in variants:
            pred_train, pred_test = fit_with_target_variant(
                X_train=X_train,
                X_test=X_test,
                y_train=y_train,
                variant=variant,
                config=config,
            )
            rows.append(
                {
                    "target": config.name,
                    "variant": variant,
                    "variant_label": label,
                    **collect_prediction_metrics(
                        config, y_train, y_test, pred_train, pred_test
                    ),
                }
            )
    return summarize_metrics(rows, ["target", "variant", "variant_label"])


def run_distribution_comparison(configs: list[TargetConfig]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for config in configs:
        for n_signals, rho in DISTRIBUTION_COMPARISON_SCENARIOS:
            for distribution_mode in ("gaussian", "empirical"):
                metrics = run_base_simulation(
                    config=config,
                    rhos=(rho,) * n_signals,
                    distribution_mode=distribution_mode,
                )
                for metric_row in metrics:
                    metric_row["target"] = config.name
                    metric_row["distribution_mode"] = distribution_mode
                    metric_row["n_signals"] = n_signals
                    metric_row["rho"] = rho
                rows.extend(metrics)
    return summarize_metrics(rows, ["target", "distribution_mode", "n_signals", "rho"])


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    fieldnames: list[str] = []
    for row in rows:
        for key in row.keys():
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def value_to_color(value: float, vmin: float, vmax: float) -> str:
    if math.isnan(value):
        return "#e5e7eb"
    if np.isclose(vmax, vmin):
        t = 0.5
    else:
        t = min(1.0, max(0.0, (value - vmin) / (vmax - vmin)))
    hue = 0.62 - (0.52 * t)
    lightness = 0.28 + (0.38 * t)
    r, g, b = colorsys.hls_to_rgb(hue, lightness, 0.7)
    return f"#{int(r * 255):02x}{int(g * 255):02x}{int(b * 255):02x}"


def svg_wrap(width: int, height: int, body: list[str]) -> str:
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}" style="background:#ffffff">'
        + "".join(body)
        + "</svg>"
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


def escape_xml(text: str) -> str:
    return (
        str(text)
        .replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
    )


def draw_rect(
    body: list[str],
    x: float,
    y: float,
    width: float,
    height: float,
    fill: str,
    stroke: str = "#ffffff",
    stroke_width: float = 1.0,
) -> None:
    body.append(
        f'<rect x="{x:.1f}" y="{y:.1f}" width="{width:.1f}" height="{height:.1f}" '
        f'fill="{fill}" stroke="{stroke}" stroke-width="{stroke_width:.1f}"/>'
    )


def draw_heatmap_panel(
    body: list[str],
    x0: int,
    y0: int,
    title: str,
    row_labels: list[str],
    col_labels: list[str],
    values: dict[tuple[str, str], float],
) -> None:
    cell_w = 66
    cell_h = 38
    label_w = 88
    label_h = 52

    numeric_values = [value for value in values.values() if not math.isnan(value)]
    vmin = min(numeric_values) if numeric_values else 0.0
    vmax = max(numeric_values) if numeric_values else 1.0

    draw_text(body, x0, y0 - 14, title, size=14, weight="700")
    for idx, col_label in enumerate(col_labels):
        draw_text(
            body,
            x0 + label_w + (idx + 0.5) * cell_w,
            y0 + 22,
            col_label,
            size=11,
            anchor="middle",
        )
    for idx, row_label in enumerate(row_labels):
        draw_text(
            body,
            x0 + label_w - 8,
            y0 + label_h + (idx + 0.6) * cell_h,
            row_label,
            size=11,
            anchor="end",
        )
    for row_idx, row_label in enumerate(row_labels):
        for col_idx, col_label in enumerate(col_labels):
            value = values.get((row_label, col_label), math.nan)
            fill = value_to_color(value, vmin, vmax)
            x = x0 + label_w + col_idx * cell_w
            y = y0 + label_h + row_idx * cell_h
            draw_rect(body, x, y, cell_w, cell_h, fill=fill)
            draw_text(
                body,
                x + (cell_w / 2),
                y + 23,
                "NA" if math.isnan(value) else f"{value:.2f}",
                size=11,
                weight="700",
                anchor="middle",
                fill="#ffffff" if not math.isnan(value) else "#374151",
            )


def plot_homogeneous_grid(summary: list[dict[str, Any]]) -> None:
    metric_specs = [
        ("test_r_median", "Median test R"),
        ("test_regression_r2_median", "Median test regression R²"),
        ("train_test_regression_r2_gap_median", "Median train-test R² gap"),
        ("keep_20_gain_median", "Top 20% mean-rating uplift"),
    ]
    for target in sorted({row["target"] for row in summary}):
        subset = [row for row in summary if row["target"] == target]
        body: list[str] = []
        row_labels = [str(value) for value in range(1, 6)]
        col_labels = [f"{value:.2f}" for value in RHO_GRID]
        draw_text(body, 20, 32, f"{target}: homogeneous signal grid", size=22, weight="700")
        panel_positions = [(20, 90), (720, 90), (20, 400), (720, 400)]
        for (metric, title), (x0, y0) in zip(metric_specs, panel_positions, strict=True):
            values = {
                (str(row["n_signals"]), f"{float(row['rho']):.2f}"): float(row[metric])
                for row in subset
            }
            draw_heatmap_panel(body, x0, y0, title, row_labels, col_labels, values)
        svg = svg_wrap(1420, 720, body)
        (OUTPUT_DIR / f"prediction_simulation_{target.lower()}_homogeneous_grid.svg").write_text(svg)


def draw_line_chart_panel(
    body: list[str],
    x0: int,
    y0: int,
    width: int,
    height: int,
    title: str,
    series: list[dict[str, Any]],
    metric: str,
) -> None:
    padding_left = 55
    padding_bottom = 35
    plot_w = width - padding_left - 20
    plot_h = height - 35 - padding_bottom
    draw_text(body, x0, y0 - 12, title, size=14, weight="700")
    draw_rect(body, x0 + padding_left, y0 + 10, plot_w, plot_h, fill="#ffffff", stroke="#d1d5db")

    values = []
    for row in series:
        for suffix in ("_p10", "_median", "_p90"):
            value = float(row[f"{metric}{suffix}"])
            if not math.isnan(value):
                values.append(value)
    ymin = min(values) if values else 0.0
    ymax = max(values) if values else 1.0
    if np.isclose(ymax, ymin):
        ymax = ymin + 1.0

    color_map = {"Enjoyment": "#2563eb", "Usefulness": "#d97706"}
    for target in ["Enjoyment", "Usefulness"]:
        subset = [row for row in series if row["target"] == target]
        subset.sort(key=lambda row: int(row["n_signals"]))
        path_points = []
        for row in subset:
            x = x0 + padding_left + ((int(row["n_signals"]) - 1) / 4.0) * plot_w
            value = float(row[f"{metric}_median"])
            y = y0 + 10 + plot_h - ((value - ymin) / (ymax - ymin)) * plot_h
            low = float(row[f"{metric}_p10"])
            high = float(row[f"{metric}_p90"])
            y_low = y0 + 10 + plot_h - ((low - ymin) / (ymax - ymin)) * plot_h
            y_high = y0 + 10 + plot_h - ((high - ymin) / (ymax - ymin)) * plot_h
            body.append(
                f'<line x1="{x:.1f}" y1="{y_low:.1f}" x2="{x:.1f}" y2="{y_high:.1f}" '
                f'stroke="{color_map[target]}" stroke-width="2"/>'
            )
            body.append(
                f'<circle cx="{x:.1f}" cy="{y:.1f}" r="4" fill="{color_map[target]}"/>'
            )
            path_points.append((x, y))
        if path_points:
            path_d = " ".join(
                [
                    ("M" if idx == 0 else "L") + f" {point[0]:.1f} {point[1]:.1f}"
                    for idx, point in enumerate(path_points)
                ]
            )
            body.append(
                f'<path d="{path_d}" fill="none" stroke="{color_map[target]}" stroke-width="2.5"/>'
            )

    for n_signals in range(1, 6):
        x = x0 + padding_left + ((n_signals - 1) / 4.0) * plot_w
        draw_text(body, x, y0 + 10 + plot_h + 22, str(n_signals), size=11, anchor="middle")
    for fraction in np.linspace(0, 1, 5):
        value = ymin + fraction * (ymax - ymin)
        y = y0 + 10 + plot_h - fraction * plot_h
        body.append(
            f'<line x1="{x0 + padding_left:.1f}" y1="{y:.1f}" x2="{x0 + padding_left + plot_w:.1f}" y2="{y:.1f}" '
            f'stroke="#e5e7eb" stroke-width="1"/>'
        )
        draw_text(body, x0 + padding_left - 8, y + 4, f"{value:.2f}", size=10, anchor="end")


def plot_random_range_summary(summary: list[dict[str, Any]]) -> None:
    body: list[str] = []
    draw_text(body, 20, 30, "Random rho draws in [0.15, 0.50]", size=22, weight="700")
    specs = [
        ("test_r", "Median test R"),
        ("test_regression_r2", "Median test regression R²"),
        ("keep_20_gain", "Top 20% mean-rating uplift"),
    ]
    positions = [(20, 80), (430, 80), (840, 80)]
    for (metric, title), (x0, y0) in zip(specs, positions, strict=True):
        draw_line_chart_panel(body, x0, y0, 380, 300, title, summary, metric)
    draw_text(body, 1130, 115, "Enjoyment", size=12, fill="#2563eb")
    draw_text(body, 1130, 140, "Usefulness", size=12, fill="#d97706")
    svg = svg_wrap(1240, 420, body)
    (OUTPUT_DIR / "prediction_simulation_random_rho_ranges.svg").write_text(svg)


def draw_bar_chart_panel(
    body: list[str],
    x0: int,
    y0: int,
    width: int,
    height: int,
    title: str,
    rows: list[dict[str, Any]],
    metric: str,
) -> None:
    padding_left = 45
    padding_bottom = 90
    plot_w = width - padding_left - 10
    plot_h = height - 30 - padding_bottom
    values = [float(row[metric]) for row in rows if not math.isnan(float(row[metric]))]
    ymin = min(0.0, min(values) if values else 0.0)
    ymax = max(values) if values else 1.0
    if np.isclose(ymax, ymin):
        ymax = ymin + 1.0

    draw_text(body, x0, y0 - 12, title, size=14, weight="700")
    draw_rect(body, x0 + padding_left, y0 + 10, plot_w, plot_h, fill="#ffffff", stroke="#d1d5db")

    bar_w = max(20.0, plot_w / max(1, len(rows)) - 12.0)
    for idx, row in enumerate(rows):
        value = float(row[metric])
        x = x0 + padding_left + 8 + idx * (plot_w / max(1, len(rows)))
        zero_y = y0 + 10 + plot_h - ((0.0 - ymin) / (ymax - ymin)) * plot_h
        value_y = y0 + 10 + plot_h - ((value - ymin) / (ymax - ymin)) * plot_h
        rect_y = min(zero_y, value_y)
        rect_h = abs(zero_y - value_y)
        draw_rect(body, x, rect_y, bar_w, rect_h, fill="#4f46e5", stroke="#4f46e5")
        draw_text(body, x + (bar_w / 2), rect_y - 6, f"{value:.2f}", size=10, anchor="middle")
        label = row["variant_label"].replace(", ", "\n")
        for line_idx, line in enumerate(label.split("\n")):
            draw_text(
                body,
                x + (bar_w / 2),
                y0 + 10 + plot_h + 18 + 12 * line_idx,
                line,
                size=10,
                anchor="middle",
            )


def plot_transform_summary(summary: list[dict[str, Any]]) -> None:
    specs = [
        ("test_r_median", "Median test R"),
        ("test_regression_r2_median", "Median test regression R²"),
        ("keep_20_gain_median", "Top 20% mean-rating uplift"),
    ]
    for target in sorted({row["target"] for row in summary}):
        subset = [row for row in summary if row["target"] == target]
        body: list[str] = []
        draw_text(body, 20, 30, f"{target}: transform mistakes", size=22, weight="700")
        positions = [(20, 80), (450, 80), (880, 80)]
        for (metric, title), (x0, y0) in zip(specs, positions, strict=True):
            draw_bar_chart_panel(body, x0, y0, 390, 370, title, subset, metric)
        svg = svg_wrap(1300, 520, body)
        (OUTPUT_DIR / f"prediction_simulation_{target.lower()}_transform_mistakes.svg").write_text(svg)


def draw_scatter_panel(
    body: list[str],
    x0: int,
    y0: int,
    width: int,
    height: int,
    title: str,
    rows: list[dict[str, Any]],
    metric: str,
) -> None:
    padding_left = 45
    padding_bottom = 35
    plot_w = width - padding_left - 15
    plot_h = height - 35 - padding_bottom
    draw_text(body, x0, y0 - 12, title, size=14, weight="700")
    draw_rect(body, x0 + padding_left, y0 + 10, plot_w, plot_h, fill="#ffffff", stroke="#d1d5db")

    xs = [float(row["rho"]) for row in rows]
    ys = [float(row[metric]) for row in rows if not math.isnan(float(row[metric]))]
    xmin = min(xs) if xs else 0.0
    xmax = max(xs) if xs else 1.0
    ymin = min(ys) if ys else 0.0
    ymax = max(ys) if ys else 1.0
    if np.isclose(ymax, ymin):
        ymax = ymin + 1.0

    color_map = {"gaussian": "#2563eb", "empirical": "#dc2626"}
    for row in rows:
        x_value = float(row["rho"])
        y_value = float(row[metric])
        x = x0 + padding_left + ((x_value - xmin) / (xmax - xmin)) * plot_w
        y = y0 + 10 + plot_h - ((y_value - ymin) / (ymax - ymin)) * plot_h
        radius = 4 + int(row["n_signals"])
        body.append(
            f'<circle cx="{x:.1f}" cy="{y:.1f}" r="{radius / 2:.1f}" fill="{color_map[row["distribution_mode"]]}" opacity="0.8"/>'
        )
        draw_text(body, x + 8, y + 4, str(row["n_signals"]), size=10)

    for rho in sorted({float(row["rho"]) for row in rows}):
        x = x0 + padding_left + ((rho - xmin) / (xmax - xmin)) * plot_w
        draw_text(body, x, y0 + 10 + plot_h + 22, f"{rho:.2f}", size=10, anchor="middle")


def plot_distribution_comparison(summary: list[dict[str, Any]]) -> None:
    specs = [
        ("test_regression_r2_median", "Median test regression R²"),
        ("keep_20_gain_median", "Top 20% mean-rating uplift"),
    ]
    for target in sorted({row["target"] for row in summary}):
        subset = [row for row in summary if row["target"] == target]
        body: list[str] = []
        draw_text(body, 20, 30, f"{target}: Gaussian vs empirical target shape", size=22, weight="700")
        positions = [(20, 80), (540, 80)]
        for (metric, title), (x0, y0) in zip(specs, positions, strict=True):
            draw_scatter_panel(body, x0, y0, 470, 330, title, subset, metric)
        draw_text(body, 1030, 110, "Gaussian", size=12, fill="#2563eb")
        draw_text(body, 1030, 135, "Empirical", size=12, fill="#dc2626")
        svg = svg_wrap(1150, 450, body)
        (OUTPUT_DIR / f"prediction_simulation_{target.lower()}_distribution_comparison.svg").write_text(svg)


def find_one(rows: list[dict[str, Any]], **criteria: Any) -> dict[str, Any]:
    for row in rows:
        if all(row.get(key) == value for key, value in criteria.items()):
            return row
    raise KeyError(f"No row found for {criteria}")


def write_summary_note(
    configs: list[TargetConfig],
    homogeneous: list[dict[str, Any]],
    random_ranges: list[dict[str, Any]],
    ladder: list[dict[str, Any]],
    transform_summary: list[dict[str, Any]],
    distribution_summary: list[dict[str, Any]],
) -> None:
    lines = ["# 2026-03-22 Detailed Prediction Simulation", ""]
    lines.append("Built `detailed_prediction_simulation.py` to generate direct Monte Carlo tables and SVG plots for:")
    lines.append("")
    lines.append("- 1..5 signals with per-signal `rho` between `0.15` and `0.50`")
    lines.append("- train vs test `R`, correlation `R^2`, and regression `R^2`")
    lines.append("- cutoff gains for top 20%, 50%, and 80% kept")
    lines.append("- transform mistakes where the target is transformed but not inverted")
    lines.append("- Gaussian targets versus the empirical 1-5 distribution from `golden_master.csv`")
    lines.append("")
    lines.append("## Quick Readout")
    lines.append("")

    for config in configs:
        target = config.name
        weakest = find_one(
            homogeneous,
            target=target,
            distribution_mode="gaussian",
            n_signals=1,
            rho=0.15,
        )
        strongest = find_one(
            homogeneous,
            target=target,
            distribution_mode="gaussian",
            n_signals=5,
            rho=0.50,
        )
        random_mid = find_one(random_ranges, target=target, n_signals=3)
        ladder_mid = find_one(
            ladder,
            target=target,
            n_signals=3,
            rho_vector="0.15, 0.30, 0.45",
        )
        good_transform = find_one(
            transform_summary,
            target=target,
            variant="utility_correct_inverse",
            variant_label="Utility target, invert correctly",
        )
        bad_transform = find_one(
            transform_summary,
            target=target,
            variant="utility_no_inverse",
            variant_label="Utility target, no inverse",
        )
        gaussian_mid = find_one(
            distribution_summary,
            target=target,
            distribution_mode="gaussian",
            n_signals=3,
            rho=0.30,
        )
        empirical_mid = find_one(
            distribution_summary,
            target=target,
            distribution_mode="empirical",
            n_signals=3,
            rho=0.30,
        )

        lines.append(f"### {target}")
        lines.append(
            f"- Weak case (`1` signal at `0.15`): median test `R={weakest['test_r_median']:.3f}`, regression `R^2={weakest['test_regression_r2_median']:.3f}`."
        )
        lines.append(
            f"- Strong case (`5` signals at `0.50`): median test `R={strongest['test_r_median']:.3f}`, regression `R^2={strongest['test_regression_r2_median']:.3f}`, top-20% gain `+{strongest['keep_20_gain_median']:.3f}`."
        )
        lines.append(
            f"- Random-rho `3`-signal case: median test `R={random_mid['test_r_median']:.3f}` with p10-p90 `{random_mid['test_r_p10']:.3f}` to `{random_mid['test_r_p90']:.3f}`."
        )
        lines.append(
            f"- Mixed ladder `0.15/0.30/0.45`: median test `R={ladder_mid['test_r_median']:.3f}`, top-20% gain `+{ladder_mid['keep_20_gain_median']:.3f}`."
        )
        lines.append(
            f"- Utility transform inverted correctly: median regression `R^2={good_transform['test_regression_r2_median']:.3f}`."
        )
        lines.append(
            f"- Utility transform not inverted: median test `R={bad_transform['test_r_median']:.3f}` but regression `R^2={bad_transform['test_regression_r2_median']:.3f}`."
        )
        lines.append(
            f"- Empirical shape vs Gaussian at `3` signals and `0.30`: regression `R^2` `{gaussian_mid['test_regression_r2_median']:.3f} -> {empirical_mid['test_regression_r2_median']:.3f}`, top-20% gain `{gaussian_mid['keep_20_gain_median']:.3f} -> {empirical_mid['keep_20_gain_median']:.3f}`."
        )
        lines.append("")

    lines.append("## Output Files")
    lines.append("")
    lines.append("- `prediction_simulation_homogeneous_grid.csv`")
    lines.append("- `prediction_simulation_random_rho_ranges.csv`")
    lines.append("- `prediction_simulation_mixed_signal_ladder.csv`")
    lines.append("- `prediction_simulation_transform_mistakes.csv`")
    lines.append("- `prediction_simulation_distribution_comparison.csv`")
    lines.append("- `prediction_simulation_*_homogeneous_grid.svg`")
    lines.append("- `prediction_simulation_random_rho_ranges.svg`")
    lines.append("- `prediction_simulation_*_transform_mistakes.svg`")
    lines.append("- `prediction_simulation_*_distribution_comparison.svg`")
    lines.append("")

    (AI_ACTIONS_DIR / "2026-03-22_detailed_prediction_simulation.md").write_text(
        "\n".join(lines)
    )


def main() -> None:
    configs = load_target_configs()

    homogeneous = run_homogeneous_grid(configs)
    write_csv(OUTPUT_DIR / "prediction_simulation_homogeneous_grid.csv", homogeneous)

    random_ranges = run_random_rho_ranges(configs)
    write_csv(OUTPUT_DIR / "prediction_simulation_random_rho_ranges.csv", random_ranges)

    ladder = run_mixed_signal_ladder(configs)
    write_csv(OUTPUT_DIR / "prediction_simulation_mixed_signal_ladder.csv", ladder)

    transform_summary: list[dict[str, Any]] = []
    for config in configs:
        transform_summary.extend(
            run_transform_mistake_simulation(config=config, rhos=(0.15, 0.30, 0.45))
        )
    write_csv(OUTPUT_DIR / "prediction_simulation_transform_mistakes.csv", transform_summary)

    distribution_summary = run_distribution_comparison(configs)
    write_csv(
        OUTPUT_DIR / "prediction_simulation_distribution_comparison.csv",
        distribution_summary,
    )

    plot_homogeneous_grid(homogeneous)
    plot_random_range_summary(random_ranges)
    plot_transform_summary(transform_summary)
    plot_distribution_comparison(distribution_summary)

    write_summary_note(
        configs=configs,
        homogeneous=homogeneous,
        random_ranges=random_ranges,
        ladder=ladder,
        transform_summary=transform_summary,
        distribution_summary=distribution_summary,
    )

    print("Saved prediction simulation outputs.")


if __name__ == "__main__":
    main()
