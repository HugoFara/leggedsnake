"""
Visualization and reporting for walking mechanism analysis.

Provides matplotlib-based plotting functions for:
- Pareto front scatter plots (2D and 3D)
- Gait phase / timing diagrams
- Stability time series (tip-over margin, ZMP, body angle)
- Center-of-mass trajectory overlays
- Foot trajectory shape analysis
- Combined dashboard views

All functions return ``matplotlib.figure.Figure`` so callers can
further customize or save to file.

Example::

    from leggedsnake.plotting import (
        plot_pareto_front,
        plot_gait_diagram,
        plot_stability_timeseries,
    )

    fig = plot_pareto_front(nsga_result)
    fig.savefig("pareto.png")

    fig = plot_gait_diagram(gait_result)
    fig = plot_stability_timeseries(stability_series)
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    import matplotlib.figure
    from matplotlib.axes import Axes

    from .gait_analysis import GaitAnalysisResult
    from .nsga_optimizer import NsgaWalkingResult
    from .stability import StabilityTimeSeries


def _import_plt() -> Any:
    """Lazy-import matplotlib.pyplot."""
    import matplotlib.pyplot as plt

    return plt


# ---------------------------------------------------------------------------
# Pareto front
# ---------------------------------------------------------------------------


def plot_pareto_front(
    result: NsgaWalkingResult,
    objective_indices: tuple[int, int] | tuple[int, int, int] = (0, 1),
    highlight_best: bool = True,
    figsize: tuple[float, float] = (8, 6),
) -> matplotlib.figure.Figure:
    """Scatter plot of the Pareto front.

    Supports 2D (two objectives) and 3D (three objectives).

    Parameters
    ----------
    result : NsgaWalkingResult
        Output of ``nsga_walking_optimization``.
    objective_indices : tuple of int
        Which objectives to plot (indices into scores tuple).
    highlight_best : bool
        If True, mark the best-compromise solution.
    figsize : tuple[float, float]
        Figure size in inches.

    Returns
    -------
    matplotlib.figure.Figure
    """
    plt = _import_plt()
    solutions = result.pareto_front.solutions
    names = result.pareto_front.objective_names

    if len(objective_indices) == 3:
        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(111, projection="3d")
        i, j, k = objective_indices
        xs = [s.scores[i] for s in solutions]
        ys = [s.scores[j] for s in solutions]
        zs = [s.scores[k] for s in solutions]
        ax.scatter(xs, ys, zs, c="steelblue", alpha=0.7, edgecolors="k", linewidths=0.5)
        ax.set_xlabel(names[i] if names else f"Objective {i}")
        ax.set_ylabel(names[j] if names else f"Objective {j}")
        ax.set_zlabel(names[k] if names else f"Objective {k}")

        if highlight_best and solutions:
            best = result.best_compromise()
            ax.scatter(
                [best.scores[i]], [best.scores[j]], [best.scores[k]],
                c="red", s=120, marker="*", zorder=5, label="Best compromise",
            )
            ax.legend()
    else:
        fig, ax = plt.subplots(figsize=figsize)
        i, j = objective_indices[:2]
        xs = [s.scores[i] for s in solutions]
        ys = [s.scores[j] for s in solutions]
        ax.scatter(xs, ys, c="steelblue", alpha=0.7, edgecolors="k", linewidths=0.5)
        ax.set_xlabel(names[i] if names else f"Objective {i}")
        ax.set_ylabel(names[j] if names else f"Objective {j}")
        ax.grid(True, alpha=0.3)

        if highlight_best and solutions:
            best = result.best_compromise()
            ax.scatter(
                [best.scores[i]], [best.scores[j]],
                c="red", s=120, marker="*", zorder=5, label="Best compromise",
            )
            ax.legend()

    ax.set_title("Pareto Front")
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Gait diagram
# ---------------------------------------------------------------------------


def plot_gait_diagram(
    gait: GaitAnalysisResult,
    figsize: tuple[float, float] = (10, 4),
) -> matplotlib.figure.Figure:
    """Gait timing diagram showing stance/swing phases per foot.

    Each foot gets a horizontal bar: dark = stance, light = swing.

    Parameters
    ----------
    gait : GaitAnalysisResult
        Output of ``analyze_gait``.
    figsize : tuple[float, float]
        Figure size in inches.

    Returns
    -------
    matplotlib.figure.Figure
    """
    plt = _import_plt()
    fig, ax = plt.subplots(figsize=figsize)

    foot_ids = sorted(gait.gait_cycles.keys())
    if not foot_ids:
        ax.text(0.5, 0.5, "No gait cycles detected", ha="center", va="center",
                transform=ax.transAxes)
        ax.set_title("Gait Timing Diagram")
        fig.tight_layout()
        return fig

    colors_stance = plt.cm.Set2(np.linspace(0, 1, max(len(foot_ids), 1)))

    for row, fid in enumerate(foot_ids):
        for cycle in gait.gait_cycles[fid]:
            # Stance phase (solid bar)
            ax.barh(
                row, cycle.stance_duration, left=cycle.stance_start,
                height=0.6, color=colors_stance[row % len(colors_stance)],
                edgecolor="k", linewidth=0.5,
            )
            # Swing phase (hatched bar)
            ax.barh(
                row, cycle.swing_duration, left=cycle.swing_start,
                height=0.6, color=colors_stance[row % len(colors_stance)],
                alpha=0.3, edgecolor="k", linewidth=0.5, hatch="//",
            )

    ax.set_yticks(range(len(foot_ids)))
    ax.set_yticklabels(foot_ids)
    ax.set_xlabel("Time (s)")
    ax.set_title("Gait Timing Diagram")
    ax.invert_yaxis()

    # Legend
    from matplotlib.patches import Patch
    ax.legend(
        handles=[
            Patch(facecolor="gray", edgecolor="k", label="Stance"),
            Patch(facecolor="gray", alpha=0.3, edgecolor="k", hatch="//", label="Swing"),
        ],
        loc="upper right",
    )
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Stability time series
# ---------------------------------------------------------------------------


def plot_stability_timeseries(
    series: StabilityTimeSeries,
    figsize: tuple[float, float] = (12, 8),
) -> matplotlib.figure.Figure:
    """Multi-panel stability time series plot.

    Four subplots: tip-over margin, ZMP x-coordinate, body angle,
    and CoM height.

    Parameters
    ----------
    series : StabilityTimeSeries
        Output of stability recording.
    figsize : tuple[float, float]
        Figure size in inches.

    Returns
    -------
    matplotlib.figure.Figure
    """
    plt = _import_plt()
    fig, axes = plt.subplots(4, 1, figsize=figsize, sharex=True)

    if not series.snapshots:
        for ax in axes:
            ax.text(0.5, 0.5, "No data", ha="center", va="center",
                    transform=ax.transAxes)
        fig.suptitle("Stability Time Series")
        fig.tight_layout()
        return fig

    times = [s.time for s in series.snapshots]

    # Tip-over margin
    margins = [s.tip_over_margin for s in series.snapshots]
    axes[0].plot(times, margins, color="steelblue", linewidth=1)
    axes[0].axhline(0, color="red", linestyle="--", alpha=0.5, linewidth=0.8)
    axes[0].fill_between(
        times, margins, 0,
        where=[m >= 0 for m in margins], alpha=0.15, color="green",
    )
    axes[0].fill_between(
        times, margins, 0,
        where=[m < 0 for m in margins], alpha=0.15, color="red",
    )
    axes[0].set_ylabel("Tip-over margin")
    axes[0].set_title("Stability Time Series")

    # ZMP
    zmp_xs = [s.zmp_x for s in series.snapshots]
    axes[1].plot(times, zmp_xs, color="darkorange", linewidth=1)
    axes[1].set_ylabel("ZMP x")

    # Body angle
    angles = [s.body_angle for s in series.snapshots]
    axes[2].plot(times, angles, color="purple", linewidth=1)
    axes[2].axhline(0, color="gray", linestyle="--", alpha=0.5, linewidth=0.8)
    axes[2].set_ylabel("Body angle (rad)")

    # CoM height
    com_ys = [s.com[1] for s in series.snapshots]
    axes[3].plot(times, com_ys, color="teal", linewidth=1)
    axes[3].set_ylabel("CoM height")
    axes[3].set_xlabel("Time (s)")

    for ax in axes:
        ax.grid(True, alpha=0.3)

    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# CoM trajectory
# ---------------------------------------------------------------------------


def plot_com_trajectory(
    series: StabilityTimeSeries,
    figsize: tuple[float, float] = (10, 4),
) -> matplotlib.figure.Figure:
    """2D plot of center-of-mass trajectory with support polygon snapshots.

    Parameters
    ----------
    series : StabilityTimeSeries
        Stability recording with CoM and support polygon data.
    figsize : tuple[float, float]
        Figure size in inches.

    Returns
    -------
    matplotlib.figure.Figure
    """
    plt = _import_plt()
    fig, ax = plt.subplots(figsize=figsize)

    if not series.snapshots:
        ax.text(0.5, 0.5, "No data", ha="center", va="center",
                transform=ax.transAxes)
        fig.tight_layout()
        return fig

    com_xs = [s.com[0] for s in series.snapshots]
    com_ys = [s.com[1] for s in series.snapshots]

    # Color by tip-over margin
    margins = [s.tip_over_margin for s in series.snapshots]
    sc = ax.scatter(
        com_xs, com_ys, c=margins, cmap="RdYlGn", s=4, alpha=0.8,
        edgecolors="none",
    )
    fig.colorbar(sc, ax=ax, label="Tip-over margin")

    # Draw support polygon at sampled intervals
    n_snaps = len(series.snapshots)
    sample_interval = max(1, n_snaps // 10)
    for idx in range(0, n_snaps, sample_interval):
        snap = series.snapshots[idx]
        if len(snap.support_polygon) >= 2:
            sp_xs = [p[0] for p in snap.support_polygon]
            ax.plot(sp_xs, [0] * len(sp_xs), "k-", alpha=0.2, linewidth=2)

    ax.set_xlabel("X position")
    ax.set_ylabel("Y position")
    ax.set_title("Center of Mass Trajectory")
    ax.set_aspect("equal", adjustable="datalim")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Foot trajectories
# ---------------------------------------------------------------------------


def plot_foot_trajectories(
    gait: GaitAnalysisResult,
    figsize: tuple[float, float] = (10, 6),
) -> matplotlib.figure.Figure:
    """Foot trajectory shape plot for each tracked foot.

    Each foot trajectory is drawn with contact regions highlighted.

    Parameters
    ----------
    gait : GaitAnalysisResult
        Gait analysis with foot trajectory data.
    figsize : tuple[float, float]
        Figure size in inches.

    Returns
    -------
    matplotlib.figure.Figure
    """
    plt = _import_plt()
    foot_ids = sorted(gait.foot_trajectories.keys())
    n_feet = len(foot_ids)

    if n_feet == 0:
        fig, ax = plt.subplots(figsize=figsize)
        ax.text(0.5, 0.5, "No foot trajectories", ha="center", va="center",
                transform=ax.transAxes)
        fig.tight_layout()
        return fig

    cols = min(n_feet, 3)
    rows = (n_feet + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=figsize, squeeze=False)

    colors = plt.cm.tab10(np.linspace(0, 1, max(n_feet, 1)))

    for idx, fid in enumerate(foot_ids):
        row, col = divmod(idx, cols)
        ax = axes[row][col]
        traj = gait.foot_trajectories[fid]
        if not traj:
            continue

        xs = [p[0] for p in traj]
        ys = [p[1] for p in traj]
        ax.plot(xs, ys, color=colors[idx], linewidth=1.2, alpha=0.8)
        ax.scatter(xs[0], ys[0], c="green", s=30, zorder=5, marker="o", label="Start")
        ax.scatter(xs[-1], ys[-1], c="red", s=30, zorder=5, marker="x", label="End")
        ax.set_title(fid)
        ax.set_aspect("equal", adjustable="datalim")
        ax.grid(True, alpha=0.3)

    # Hide unused subplots
    for idx in range(n_feet, rows * cols):
        row, col = divmod(idx, cols)
        axes[row][col].set_visible(False)

    fig.suptitle("Foot Trajectories")
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Combined dashboard
# ---------------------------------------------------------------------------


def plot_optimization_dashboard(
    result: NsgaWalkingResult,
    solution_index: int = 0,
    figsize: tuple[float, float] = (16, 10),
) -> matplotlib.figure.Figure:
    """Combined dashboard for a single Pareto-front solution.

    Shows Pareto front (top-left), gait diagram (top-right),
    stability series (bottom-left), and foot trajectories (bottom-right).

    Parameters
    ----------
    result : NsgaWalkingResult
        Full NSGA optimization result with gait and stability data.
    solution_index : int
        Index of the Pareto solution to detail.
    figsize : tuple[float, float]
        Figure size in inches.

    Returns
    -------
    matplotlib.figure.Figure
    """
    plt = _import_plt()
    fig = plt.figure(figsize=figsize)

    # Top-left: Pareto front
    ax_pareto = fig.add_subplot(2, 2, 1)
    _draw_pareto_on_axes(ax_pareto, result, solution_index)

    # Top-right: Gait diagram
    ax_gait = fig.add_subplot(2, 2, 2)
    if result.gait_analyses and solution_index in result.gait_analyses:
        _draw_gait_on_axes(ax_gait, result.gait_analyses[solution_index])
    else:
        ax_gait.text(0.5, 0.5, "No gait data\n(run with include_gait=True)",
                     ha="center", va="center", transform=ax_gait.transAxes)
        ax_gait.set_title("Gait Timing")

    # Bottom-left: Stability
    ax_stab = fig.add_subplot(2, 2, 3)
    if result.stability_series and solution_index in result.stability_series:
        _draw_stability_on_axes(ax_stab, result.stability_series[solution_index])
    else:
        ax_stab.text(0.5, 0.5, "No stability data\n(run with include_stability=True)",
                     ha="center", va="center", transform=ax_stab.transAxes)
        ax_stab.set_title("Stability")

    # Bottom-right: Metrics summary table
    ax_metrics = fig.add_subplot(2, 2, 4)
    _draw_metrics_table(ax_metrics, result, solution_index)

    fig.suptitle(f"Optimization Dashboard - Solution {solution_index}", fontsize=14)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    return fig


# ---------------------------------------------------------------------------
# Internal helpers for dashboard subplots
# ---------------------------------------------------------------------------


def _draw_pareto_on_axes(
    ax: Axes, result: NsgaWalkingResult, highlight_idx: int,
) -> None:
    """Draw Pareto scatter on existing axes."""
    solutions = result.pareto_front.solutions
    names = result.pareto_front.objective_names

    if len(solutions) == 0 or len(solutions[0].scores) < 2:
        ax.text(0.5, 0.5, "Insufficient data", ha="center", va="center",
                transform=ax.transAxes)
        return

    xs = [s.scores[0] for s in solutions]
    ys = [s.scores[1] for s in solutions]
    ax.scatter(xs, ys, c="steelblue", alpha=0.6, edgecolors="k", linewidths=0.5, s=30)

    if 0 <= highlight_idx < len(solutions):
        s = solutions[highlight_idx]
        ax.scatter([s.scores[0]], [s.scores[1]], c="red", s=100, marker="*", zorder=5)

    ax.set_xlabel(names[0] if names else "Objective 0")
    ax.set_ylabel(names[1] if names else "Objective 1")
    ax.set_title("Pareto Front")
    ax.grid(True, alpha=0.3)


def _draw_gait_on_axes(ax: Axes, gait: GaitAnalysisResult) -> None:
    """Draw gait timing on existing axes."""
    plt = _import_plt()
    foot_ids = sorted(gait.gait_cycles.keys())
    if not foot_ids:
        ax.text(0.5, 0.5, "No gait cycles", ha="center", va="center",
                transform=ax.transAxes)
        ax.set_title("Gait Timing")
        return

    colors = plt.cm.Set2(np.linspace(0, 1, max(len(foot_ids), 1)))
    for row, fid in enumerate(foot_ids):
        for cycle in gait.gait_cycles[fid]:
            ax.barh(row, cycle.stance_duration, left=cycle.stance_start,
                    height=0.6, color=colors[row % len(colors)],
                    edgecolor="k", linewidth=0.5)
            ax.barh(row, cycle.swing_duration, left=cycle.swing_start,
                    height=0.6, color=colors[row % len(colors)],
                    alpha=0.3, edgecolor="k", linewidth=0.5, hatch="//")

    ax.set_yticks(range(len(foot_ids)))
    ax.set_yticklabels(foot_ids, fontsize=7)
    ax.set_xlabel("Time (s)")
    ax.set_title("Gait Timing")
    ax.invert_yaxis()


def _draw_stability_on_axes(ax: Axes, series: StabilityTimeSeries) -> None:
    """Draw tip-over margin on existing axes."""
    if not series.snapshots:
        ax.text(0.5, 0.5, "No data", ha="center", va="center",
                transform=ax.transAxes)
        return

    times = [s.time for s in series.snapshots]
    margins = [s.tip_over_margin for s in series.snapshots]
    ax.plot(times, margins, color="steelblue", linewidth=1)
    ax.axhline(0, color="red", linestyle="--", alpha=0.5, linewidth=0.8)
    ax.fill_between(times, margins, 0,
                    where=[m >= 0 for m in margins], alpha=0.15, color="green")
    ax.fill_between(times, margins, 0,
                    where=[m < 0 for m in margins], alpha=0.15, color="red")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Tip-over margin")
    ax.set_title("Stability")
    ax.grid(True, alpha=0.3)


def _draw_metrics_table(
    ax: Axes, result: NsgaWalkingResult, idx: int,
) -> None:
    """Draw a text-based metrics summary table."""
    ax.axis("off")
    solutions = result.pareto_front.solutions
    if not solutions or idx >= len(solutions):
        ax.text(0.5, 0.5, "No solution data", ha="center", va="center",
                transform=ax.transAxes)
        return

    sol = solutions[idx]
    names = result.pareto_front.objective_names

    rows: list[list[str]] = []
    for i, score in enumerate(sol.scores):
        name = names[i] if names and i < len(names) else f"Objective {i}"
        rows.append([name, f"{score:.4f}"])

    # Add gait metrics if available
    if result.gait_analyses and idx in result.gait_analyses:
        gait = result.gait_analyses[idx]
        metrics = gait.summary_metrics()
        for key, val in metrics.items():
            rows.append([key, f"{val:.4f}"])

    # Add stability metrics if available
    if result.stability_series and idx in result.stability_series:
        stab = result.stability_series[idx]
        for key, val in stab.summary_metrics().items():
            rows.append([key, f"{val:.4f}"])

    table = ax.table(
        cellText=rows,
        colLabels=["Metric", "Value"],
        loc="center",
        cellLoc="left",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 1.3)
    ax.set_title("Metrics Summary")


# ---------------------------------------------------------------------------
# Interactive / SVG renderings via pylinkage's visualizer
# ---------------------------------------------------------------------------


def plot_walker_plotly(
    walker: Any,
    iterations: int | None = None,
    skip_unbuildable: bool = True,
    *,
    title: str | None = None,
    show_loci: bool = True,
    show_labels: bool = True,
    width: int = 800,
    height: int = 600,
) -> Any:
    """Interactive Plotly diagram of a Walker's kinematic trajectory.

    Delegates to :func:`pylinkage.visualizer.plot_linkage_plotly` with a
    pre-computed locus from :meth:`Walker.step`. Useful for inspecting
    optimization-run outputs in notebooks or dashboards.

    Parameters
    ----------
    walker : Walker
        Mechanism to render.
    iterations : int | None
        Simulation steps. Defaults to one full rotation period.
    skip_unbuildable : bool
        Forwarded to ``Walker.step``. Dead-zone frames are drawn with
        their ``(None, None)`` positions, which pylinkage's plotly
        renderer tolerates gracefully.
    title, show_loci, show_labels, width, height : ...
        Forwarded to ``plot_linkage_plotly``.

    Returns
    -------
    plotly.graph_objects.Figure
    """
    from pylinkage.visualizer.plotly_viz import plot_linkage_plotly

    loci = list(walker.step(
        iterations=iterations, skip_unbuildable=skip_unbuildable,
    ))
    return plot_linkage_plotly(
        walker.to_mechanism(),
        loci=loci,
        title=title or walker.name or "walker",
        show_loci=show_loci,
        show_labels=show_labels,
        width=width,
        height=height,
    )


def save_walker_svg(
    walker: Any,
    path: str,
    iterations: int | None = None,
    skip_unbuildable: bool = True,
    **kwargs: Any,
) -> None:
    """Save a Walker's kinematic trajectory as an SVG file.

    Delegates to :func:`pylinkage.visualizer.save_linkage_svg` with a
    pre-computed locus. Useful for embedding optimizer outputs in
    papers or reports.

    Parameters
    ----------
    walker : Walker
        Mechanism to render.
    path : str
        Output SVG file path.
    iterations : int | None
        Simulation steps. Defaults to one full rotation period.
    skip_unbuildable : bool
        Forwarded to ``Walker.step``.
    **kwargs
        Forwarded to ``save_linkage_svg`` (``show_loci``, ``width``,
        ``height``, etc.).
    """
    from pylinkage.visualizer.drawsvg_viz import save_linkage_svg

    loci = list(walker.step(
        iterations=iterations, skip_unbuildable=skip_unbuildable,
    ))
    save_linkage_svg(walker.to_mechanism(), path, loci=loci, **kwargs)
