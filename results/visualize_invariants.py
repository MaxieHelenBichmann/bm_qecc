"""Plot invariant benchmark runtimes as multi-invariant scatter plots."""

from __future__ import annotations

import argparse
import re
from collections.abc import Sequence
from pathlib import Path

import matplotlib.pyplot as plt

from visual_utils import (
    AXIS_LABELS,
    Axis,
    InvariantRow,
    configure_xaxis_ticks,
    draw_minute_guides,
    read_invariant_csv,
)


def invariant_algorithm_base(algorithm: str) -> str:
    """Return the base invariant name without a trailing subset-size suffix."""
    return re.sub(r"_s\d+$", "", algorithm)


def invariant_algorithm_variant(algorithm: str) -> int | None:
    """Return the subset-size suffix of an invariant algorithm, if present."""
    match = re.search(r"_s(\d+)$", algorithm)
    if match is None:
        return None
    return int(match.group(1))


def invariant_marker(algorithm: str) -> str:
    """Return a marker shape that distinguishes full and subset-sized variants."""
    variant = invariant_algorithm_variant(algorithm)
    if variant is None:
        return "o"
    markers = ("s", "^", "D", "P", "v", "X")
    return markers[(variant - 1) % len(markers)]


def invariant_alpha(algorithm: str) -> float:
    """Use transparency to keep stacked invariant dots readable."""
    return 0.5 if invariant_algorithm_variant(algorithm) is None else 0.62


def invariant_tone(base_color, algorithm: str):
    """Return a related but distinct tone for subset-sized invariant variants."""
    variant = invariant_algorithm_variant(algorithm)
    if variant is None:
        return base_color

    import colorsys

    red, green, blue, alpha = base_color
    hue, lightness, saturation = colorsys.rgb_to_hls(red, green, blue)
    direction = -1 if variant % 2 == 0 else 1
    amount = min(0.1 + 0.06 * ((variant - 1) // 2), 0.28)
    lightness = max(0.18, min(0.82, lightness + direction * amount))
    red, green, blue = colorsys.hls_to_rgb(hue, lightness, saturation)
    return (red, green, blue, alpha)


def plot_invariant_rows(
    rows: Sequence[InvariantRow],
    axis: Axis,
    output: Path | None,
    title: str | None,
) -> None:
    """Render invariant benchmark rows as a scatter plot."""
    ordered_rows = tuple(sorted(rows, key=lambda row: (row.axis_value(axis), row.algorithm, row.case)))
    x_values = [row.axis_value(axis) for row in ordered_rows]
    algorithms = sorted({row.algorithm for row in ordered_rows})
    algorithm_bases = sorted({invariant_algorithm_base(algorithm) for algorithm in algorithms})

    cmap = plt.get_cmap("tab10" if len(algorithm_bases) <= 10 else "tab20")
    base_colors = {base: cmap(index % cmap.N) for index, base in enumerate(algorithm_bases)}
    colors = {
        algorithm: invariant_tone(base_colors[invariant_algorithm_base(algorithm)], algorithm)
        for algorithm in algorithms
    }

    from matplotlib.lines import Line2D

    fig, ax = plt.subplots(figsize=(10.5, 4.8), constrained_layout=True)
    for algorithm in algorithms:
        points = [
            (x, row.seconds, row.success, row.expected)
            for x, row in zip(x_values, ordered_rows)
            if row.algorithm == algorithm
        ]
        if not points:
            continue

        label_algorithm = True
        for expected, facecolors, edgecolors, linewidths, alpha in (
            (True, colors[algorithm], "none", 0.0, invariant_alpha(algorithm)),
            (False, "none", colors[algorithm], 1.35, 0.9),
            (None, colors[algorithm], "black", 0.7, invariant_alpha(algorithm)),
        ):
            sign_points = [(px, py) for px, py, _, sign in points if sign is expected]
            if not sign_points:
                continue
            x, y = zip(*sign_points)
            ax.scatter(
                x,
                y,
                s=46,
                facecolors=facecolors,
                edgecolors=edgecolors,
                linewidths=linewidths,
                alpha=alpha,
                marker=invariant_marker(algorithm),
                label=algorithm if label_algorithm else "_nolegend_",
                zorder=3,
            )
            label_algorithm = False

        failed_points = [(px, py) for px, py, ok, _ in points if not ok]
        if failed_points:
            failed_x, failed_y = zip(*failed_points)
            ax.scatter(
                failed_x,
                failed_y,
                s=55,
                color=colors[algorithm],
                marker="x",
                linewidths=1.3,
                label="_nolegend_",
                zorder=4,
            )

    ax.set_xlabel(AXIS_LABELS[axis])
    ax.set_ylabel("runtime [s]")
    ax.margins(x=0.12, y=0.18)
    ax.set_ylim(bottom=0)
    configure_xaxis_ticks(axis, ax)
    draw_minute_guides(ax)
    ax.grid(True, which="major", alpha=0.15)
    algorithm_legend = ax.legend(
        title="Invariant",
        loc="upper left",
        bbox_to_anchor=(1.02, 1.0),
        borderaxespad=0.0,
    )
    ax.add_artist(algorithm_legend)
    sign_handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            color="black",
            markerfacecolor="black",
            markeredgecolor="black",
            linestyle="none",
            markersize=6,
            label="positive",
            alpha=0.65,
        ),
        Line2D(
            [0],
            [0],
            marker="o",
            color="black",
            markerfacecolor="none",
            markeredgecolor="black",
            linestyle="none",
            markersize=6,
            label="negative",
        ),
    ]
    ax.legend(
        handles=sign_handles,
        title="Case",
        loc="upper left",
        bbox_to_anchor=(1.02, 0.38),
        borderaxespad=0.0,
    )
    ax.set_title(title or "Invariant benchmark runtimes")

    if output is None:
        plt.show()
        return

    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=200)
    print(f"Saved diagram to {output}.")


def build_parser() -> argparse.ArgumentParser:
    """Create the command-line parser."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("csv", type=Path, help="Invariant CSV from benchmarks/run.py --inv.")
    parser.add_argument("--x", choices=("n", "k", "r"), required=True, help="Parameter used for the x-axis.")
    parser.add_argument("--output", type=Path, help="Where to save the diagram. Shows an interactive window if omitted.")
    parser.add_argument("--title", help="Optional diagram title.")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run the invariant visualization CLI."""
    args = build_parser().parse_args(argv)
    all_rows = read_invariant_csv(args.csv)
    if not all_rows:
        raise SystemExit("No rows found in the invariant CSV.")

    plot_invariant_rows(all_rows, axis=args.x, output=args.output, title=args.title)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
