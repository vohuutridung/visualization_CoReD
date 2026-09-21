from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence


GROUP_ORDER = ("High-W", "Low-W")
COLORS = ("#2F6B9A", "#A7ADB4")


def _draw_distribution(axis: Any, grouped: dict[str, list[float]], spec: dict[str, Any]) -> None:
    data = [grouped[group] for group in GROUP_ORDER]
    if any(not values for values in data):
        axis.text(0.5, 0.5, "Insufficient data", ha="center", va="center")
        axis.set_axis_off()
        return
    can_estimate_density = all(len(values) >= 2 and len(set(values)) >= 2 for values in data)
    if can_estimate_density:
        violin = axis.violinplot(data, positions=(1, 2), showmeans=False, showmedians=False)
        for body, color in zip(violin["bodies"], COLORS):
            body.set_facecolor(color)
            body.set_edgecolor("black")
            body.set_alpha(0.65)
            body.set_linewidth(0.7)
        for key in ("cbars", "cmins", "cmaxes"):
            violin[key].set_color("#333333")
            violin[key].set_linewidth(0.7)
    else:
        for position, values, color in zip((1, 2), data, COLORS):
            shown = values[:500]
            offsets = [((index % 17) - 8) / 100 for index in range(len(shown))]
            axis.scatter(
                [position + offset for offset in offsets],
                shown,
                s=9,
                color=color,
                alpha=0.45,
                edgecolors="none",
            )
    box = axis.boxplot(
        data,
        positions=(1, 2),
        widths=0.22,
        showfliers=False,
        patch_artist=True,
        medianprops={"color": "white", "linewidth": 1.8},
        whiskerprops={"color": "#222222"},
        capprops={"color": "#222222"},
    )
    for patch, color in zip(box["boxes"], COLORS):
        patch.set_facecolor(color)
        patch.set_edgecolor("#222222")
    axis.set_xticks((1, 2), GROUP_ORDER)
    axis.set_ylabel(spec["ylabel"])
    axis.set_title(spec["title"], loc="left", fontweight="bold")
    axis.set_ylim(*spec["ylim"])
    axis.axhline(0, color="#555555", linewidth=0.7, alpha=0.5)
    axis.grid(axis="y", color="#D8DCE0", linewidth=0.6, alpha=0.7)
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)


def _save(figure: Any, stem: Path, dpi: int) -> None:
    stem.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(stem.with_suffix(".pdf"), bbox_inches="tight")
    figure.savefig(stem.with_suffix(".png"), dpi=dpi, bbox_inches="tight")


def make_figures(
    metric_groups: dict[str, dict[str, list[float]]],
    specs: dict[str, dict[str, Any]],
    output_dir: str | Path,
    run_name: str,
    *,
    figure_size: Sequence[float] = (11.0, 8.0),
    dpi: int = 300,
) -> None:
    import matplotlib.pyplot as plt

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "axes.labelsize": 10,
            "axes.titlesize": 11,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )
    destination = Path(output_dir)
    for metric, grouped in metric_groups.items():
        figure, axis = plt.subplots(figsize=(4.2, 3.5), constrained_layout=True)
        _draw_distribution(axis, grouped, specs[metric])
        _save(figure, destination / f"{metric}_{run_name}", dpi)
        plt.close(figure)

    main_metrics = (
        "answer_gain",
        "removal_damage",
        "semantic_repetition",
        "expert_branching",
    )
    if all(metric in metric_groups for metric in main_metrics):
        figure, axes = plt.subplots(2, 2, figsize=tuple(figure_size), constrained_layout=True)
        for panel, metric, axis in zip("ABCD", main_metrics, axes.flat):
            panel_spec = dict(specs[metric])
            panel_spec["title"] = f"{panel}  {panel_spec['title']}"
            _draw_distribution(axis, metric_groups[metric], panel_spec)
        _save(figure, destination / f"four_panel_{run_name}", dpi)
        plt.close(figure)
