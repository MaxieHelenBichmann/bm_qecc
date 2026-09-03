"""Render A6 as two complementary pairs of SAT runtime maps."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt

from paper.visualizations.common import (
    RESULTS_DIR,
    RUNTIME_CMAP,
    aggregate_cells,
    decimal_ticks,
    failure_marks,
    load_rows,
    mark_timeout,
    parameter_axis,
    partition_cell,
    runtime_norm,
    save_png,
    scalar_mappable,
    use_style,
)

INPUT = RESULTS_DIR / "a6" / "by_cell.csv"
OUTPUT = RESULTS_DIR / "a6" / "a6.png"
CSS_OUTPUT = RESULTS_DIR / "a6" / "a6_css.png"
NMAX = 25
GENERAL_AND_CSS_PANELS = (
    ("pm_stb_sat_on_stabilizer", "Tableau Encoding\non General Stabilizer Codes"),
    ("pm_css_sat_on_css", "Check-Matrix Encoding\non CSS Codes"),
)
CSS_PANELS = (
    ("pm_css_sat_on_css", "Check-Matrix Encoding\non CSS Codes"),
    ("pm_stb_sat_on_css", "Tableau Encoding\non CSS Codes"),
)
PANELS = (*GENERAL_AND_CSS_PANELS, CSS_PANELS[1])


def _render_panels(
    aggregated,
    norm,
    panels,
    output: Path,
    title: str,
    subtitle: str | None = None,
) -> Path:
    """Render one coordinated two-panel A6 figure."""
    use_style()
    figure, axes = plt.subplots(1, 2, figsize=(10.2, 5.5))
    figure.subplots_adjust(
        left=0.07,
        right=0.86,
        bottom=0.08,
        top=0.79 if subtitle else 0.84,
        wspace=0.18,
    )
    for ax, (variant, panel_title) in zip(axes, panels):
        parameter_axis(ax, panel_title, nmax=NMAX)
        for (n, r), cell in aggregated[variant].items():
            if int(cell["num_successful"]):
                partition_cell(
                    ax,
                    n,
                    r,
                    0,
                    1,
                    RUNTIME_CMAP(norm(float(cell["mean_value"]))),
                )
            failure_marks(ax, n, r, cell)
    figure.suptitle(title, fontsize=12)
    if subtitle:
        panels_center = (axes[0].get_position().x0 + axes[-1].get_position().x1) / 2
        figure.text(
            panels_center,
            0.885,
            subtitle,
            ha="center",
            va="center",
            fontsize=8,
        )
    bar = figure.colorbar(
        scalar_mappable(RUNTIME_CMAP, norm), ax=axes, fraction=0.025, pad=0.02
    )
    decimal_ticks(bar)
    mark_timeout(bar)
    bar.set_label("Mean runtime [s]")
    return save_png(figure, output)


def render(input_file: Path = INPUT, output: Path = OUTPUT, css_output: Path | None = None) -> Path:
    required = (
        "variant",
        "n",
        "r",
        "mean_seconds",
        "hx_hz_log_scale_improvement_percentage",
        "num_successful",
        "num_timeouts",
        "num_memory_limited",
        "num_errors",
    )
    rows = load_rows(input_file, required)
    rows = [row for row in rows if int(row["n"]) <= NMAX]
    for row in rows:
        row["num_runtime_samples"] = str(
            int(row["num_successful"])
            + int(row["num_timeouts"])
            + int(row.get("num_unexpected", 0) or 0)
        )
    aggregated = {
        variant: aggregate_cells(
            [row for row in rows if row["variant"] == variant],
            "mean_seconds",
            "num_runtime_samples",
        )
        for variant, _ in PANELS
    }
    improvements = [
        float(row["hx_hz_log_scale_improvement_percentage"])
        for row in rows
        if row["variant"] == "pm_stb_sat_on_css"
        and row["hx_hz_log_scale_improvement_percentage"].strip()
    ]
    mean_improvement = sum(improvements) / len(improvements) if improvements else None
    norm = runtime_norm(
        float(cell["mean_value"])
        for cells in aggregated.values()
        for cell in cells.values()
        if int(cell["num_successful"])
    )

    main_output = _render_panels(
        aggregated,
        norm,
        GENERAL_AND_CSS_PANELS,
        output,
        "SAT Encoding Performance on Stabilizer and CSS Codes",
    )
    if css_output is None:
        css_output = (
            CSS_OUTPUT
            if output == OUTPUT
            else output.with_name(f"{output.stem}_css{output.suffix}")
        )
    subtitle = None
    if mean_improvement is not None:
        subtitle = (
            "Mean log-scale improvement of check-matrix encoding over tableau "
            f"for CSS Codes: {mean_improvement:.2f}%"
        )
    _render_panels(
        aggregated,
        norm,
        CSS_PANELS,
        css_output,
        "SAT Encoding Performance on CSS Codes",
        subtitle,
    )
    return main_output


if __name__ == "__main__":
    render()
