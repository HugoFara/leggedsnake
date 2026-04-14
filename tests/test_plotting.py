#!/usr/bin/env python3
"""Tests for the plotting / visualization module."""

import unittest

import matplotlib
matplotlib.use("Agg")  # Non-interactive backend for testing

from matplotlib.figure import Figure

from pylinkage.optimization.collections import ParetoFront, ParetoSolution

from leggedsnake.gait_analysis import (
    FootEvent,
    GaitAnalysisResult,
    GaitCycle,
)
from leggedsnake.nsga_optimizer import NsgaWalkingConfig, NsgaWalkingResult
from leggedsnake.plotting import (
    plot_com_trajectory,
    plot_foot_trajectories,
    plot_gait_diagram,
    plot_optimization_dashboard,
    plot_pareto_front,
    plot_stability_timeseries,
)
from leggedsnake.stability import StabilitySnapshot, StabilityTimeSeries


def _make_pareto_front(n_solutions=10, n_obj=2):
    """Create a synthetic ParetoFront."""
    import numpy as np
    rng = np.random.RandomState(42)
    solutions = []
    for _ in range(n_solutions):
        scores = tuple(rng.uniform(0, 10, n_obj).tolist())
        solutions.append(ParetoSolution(
            scores=scores,
            dimensions=rng.uniform(0.5, 2.0, 3),
            init_positions=[(0.0, 0.0), (1.0, 0.0)],
        ))
    names = tuple(f"obj_{i}" for i in range(n_obj))
    return ParetoFront(solutions, names)


def _make_nsga_result(n_obj=2, with_gait=False, with_stability=False):
    """Create a synthetic NsgaWalkingResult."""
    pf = _make_pareto_front(n_obj=n_obj)
    gait_analyses = None
    stability_series = None

    if with_gait:
        gait_analyses = {0: _make_gait_result()}

    if with_stability:
        stability_series = {0: _make_stability_series()}

    return NsgaWalkingResult(
        pareto_front=pf,
        gait_analyses=gait_analyses,
        stability_series=stability_series,
        config=NsgaWalkingConfig(n_generations=5, pop_size=10),
    )


def _make_gait_result():
    """Create a synthetic GaitAnalysisResult."""
    events = [
        FootEvent("foot_a", 0.1, "touchdown", (0.0, 0.0)),
        FootEvent("foot_a", 0.3, "liftoff", (1.0, 0.5)),
        FootEvent("foot_a", 0.5, "touchdown", (2.0, 0.0)),
        FootEvent("foot_b", 0.2, "touchdown", (0.5, 0.0)),
        FootEvent("foot_b", 0.4, "liftoff", (1.5, 0.5)),
        FootEvent("foot_b", 0.6, "touchdown", (2.5, 0.0)),
    ]
    cycles = {
        "foot_a": [GaitCycle("foot_a", 0.1, 0.3, 0.3, 0.5)],
        "foot_b": [GaitCycle("foot_b", 0.2, 0.4, 0.4, 0.6)],
    }
    import math
    trajectories = {
        "foot_a": [
            (i * 0.05, 0.5 * abs(math.sin(i * 0.1))) for i in range(100)
        ],
        "foot_b": [
            (i * 0.05 + 0.5, 0.5 * abs(math.sin(i * 0.1 + 1.0))) for i in range(100)
        ],
    }
    return GaitAnalysisResult(
        foot_events=events,
        gait_cycles=cycles,
        foot_trajectories=trajectories,
    )


def _make_stability_series(n_steps=50):
    """Create a synthetic StabilityTimeSeries."""
    import math
    snapshots = []
    for i in range(n_steps):
        t = i * 0.02
        margin = 0.5 * math.sin(t * 10) + 0.3
        snapshots.append(StabilitySnapshot(
            time=t,
            com=(i * 0.1, 1.5 + 0.1 * math.sin(t * 5)),
            com_velocity=(0.5, 0.01 * math.cos(t * 5)),
            zmp_x=i * 0.1 + 0.05 * math.sin(t * 8),
            support_polygon=[(i * 0.1 - 0.5, 0), (i * 0.1 + 0.5, 0)],
            tip_over_margin=margin,
            body_angle=0.02 * math.sin(t * 3),
        ))
    return StabilityTimeSeries(snapshots)


class TestPlotParetoFront(unittest.TestCase):

    def test_2d_pareto(self):
        result = _make_nsga_result(n_obj=2)
        fig = plot_pareto_front(result)
        self.assertIsInstance(fig, Figure)

    def test_3d_pareto(self):
        result = _make_nsga_result(n_obj=3)
        fig = plot_pareto_front(result, objective_indices=(0, 1, 2))
        self.assertIsInstance(fig, Figure)

    def test_no_highlight(self):
        result = _make_nsga_result(n_obj=2)
        fig = plot_pareto_front(result, highlight_best=False)
        self.assertIsInstance(fig, Figure)

    def test_empty_pareto(self):
        result = NsgaWalkingResult(
            pareto_front=ParetoFront([], ("a", "b")),
        )
        fig = plot_pareto_front(result)
        self.assertIsInstance(fig, Figure)


class TestPlotGaitDiagram(unittest.TestCase):

    def test_with_cycles(self):
        gait = _make_gait_result()
        fig = plot_gait_diagram(gait)
        self.assertIsInstance(fig, Figure)

    def test_empty_gait(self):
        gait = GaitAnalysisResult()
        fig = plot_gait_diagram(gait)
        self.assertIsInstance(fig, Figure)


class TestPlotStabilityTimeseries(unittest.TestCase):

    def test_normal(self):
        series = _make_stability_series()
        fig = plot_stability_timeseries(series)
        self.assertIsInstance(fig, Figure)

    def test_empty(self):
        series = StabilityTimeSeries()
        fig = plot_stability_timeseries(series)
        self.assertIsInstance(fig, Figure)


class TestPlotComTrajectory(unittest.TestCase):

    def test_normal(self):
        series = _make_stability_series()
        fig = plot_com_trajectory(series)
        self.assertIsInstance(fig, Figure)

    def test_empty(self):
        series = StabilityTimeSeries()
        fig = plot_com_trajectory(series)
        self.assertIsInstance(fig, Figure)


class TestPlotFootTrajectories(unittest.TestCase):

    def test_with_data(self):
        gait = _make_gait_result()
        fig = plot_foot_trajectories(gait)
        self.assertIsInstance(fig, Figure)

    def test_empty(self):
        gait = GaitAnalysisResult()
        fig = plot_foot_trajectories(gait)
        self.assertIsInstance(fig, Figure)


class TestPlotOptimizationDashboard(unittest.TestCase):

    def test_full_dashboard(self):
        result = _make_nsga_result(n_obj=2, with_gait=True, with_stability=True)
        fig = plot_optimization_dashboard(result, solution_index=0)
        self.assertIsInstance(fig, Figure)

    def test_dashboard_no_analysis(self):
        result = _make_nsga_result(n_obj=2)
        fig = plot_optimization_dashboard(result, solution_index=0)
        self.assertIsInstance(fig, Figure)

    def test_dashboard_empty(self):
        result = NsgaWalkingResult(
            pareto_front=ParetoFront([], ("a", "b")),
        )
        fig = plot_optimization_dashboard(result, solution_index=0)
        self.assertIsInstance(fig, Figure)


class TestWalkerRenderings(unittest.TestCase):
    """Plotly / SVG renderings that delegate to pylinkage's visualizer."""

    def _make_walker(self):
        from math import tau
        from pylinkage.dimensions import Dimensions, DriverAngle
        from pylinkage.hypergraph import (
            Edge, HypergraphLinkage, Node, NodeRole,
        )
        from leggedsnake.walker import Walker

        hg = HypergraphLinkage(name="fourbar")
        hg.add_node(Node("frame", role=NodeRole.GROUND))
        hg.add_node(Node("crank", role=NodeRole.DRIVER))
        hg.add_node(Node("follower", role=NodeRole.DRIVEN))
        hg.add_edge(Edge("e0", "frame", "crank"))
        hg.add_edge(Edge("e1", "frame", "follower"))
        hg.add_edge(Edge("e2", "crank", "follower"))
        dims = Dimensions(
            node_positions={
                "frame": (0, 0), "crank": (1, 0), "follower": (0, 2),
            },
            driver_angles={"crank": DriverAngle(angular_velocity=-tau / 12)},
            edge_distances={"e0": 1.0, "e1": 2.0, "e2": 1.5},
        )
        return Walker(hg, dims, name="fourbar")

    def test_plot_walker_plotly_returns_figure(self):
        from leggedsnake.plotting import plot_walker_plotly

        walker = self._make_walker()
        fig = plot_walker_plotly(walker, iterations=12)
        # plotly.graph_objects.Figure — structural check
        self.assertEqual(type(fig).__name__, "Figure")
        self.assertTrue(hasattr(fig, "to_dict"))

    def test_save_walker_svg_writes_file(self):
        import os
        import tempfile
        from leggedsnake.plotting import save_walker_svg

        walker = self._make_walker()
        with tempfile.NamedTemporaryFile(
            suffix=".svg", delete=False,
        ) as f:
            path = f.name
        try:
            save_walker_svg(walker, path, iterations=12)
            self.assertGreater(os.path.getsize(path), 0)
        finally:
            os.unlink(path)


if __name__ == "__main__":
    unittest.main()
