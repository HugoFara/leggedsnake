#!/usr/bin/env python3
"""Tests for the NSGA-II/III walking optimizer."""

import unittest
from math import tau

from pylinkage.dimensions import Dimensions, DriverAngle
from pylinkage.hypergraph import HypergraphLinkage, Node, Edge, NodeRole
from pylinkage.optimization.collections import ParetoFront

from leggedsnake.fitness import (
    CompositeFitness,
    DistanceFitness,
    StabilityFitness,
    as_ga_fitness,
)
from leggedsnake.geneticoptimizer import GeneticOptimization
from leggedsnake.nsga_optimizer import (
    NsgaWalkingConfig,
    NsgaWalkingResult,
    WalkingNsgaProblem,
    nsga_walking_optimization,
)
from leggedsnake.walker import Walker


def _make_fourbar_walker():
    hg = HypergraphLinkage(name="fourbar")
    hg.add_node(Node("frame", role=NodeRole.GROUND))
    hg.add_node(Node("crank", role=NodeRole.DRIVER))
    hg.add_node(Node("follower", role=NodeRole.DRIVEN))
    hg.add_edge(Edge("frame_crank", "frame", "crank"))
    hg.add_edge(Edge("frame_follower", "frame", "follower"))
    hg.add_edge(Edge("crank_follower", "crank", "follower"))
    dims = Dimensions(
        node_positions={"frame": (0, 0), "crank": (1, 0), "follower": (0, 2)},
        driver_angles={"crank": DriverAngle(angular_velocity=-tau / 12)},
        edge_distances={
            "frame_crank": 1.0, "frame_follower": 2.0, "crank_follower": 1.5,
        },
    )
    return Walker(hg, dims, name="fourbar")


class TestNsgaWalkingConfig(unittest.TestCase):

    def test_defaults(self):
        cfg = NsgaWalkingConfig()
        self.assertEqual(cfg.n_generations, 100)
        self.assertEqual(cfg.pop_size, 100)
        self.assertEqual(cfg.algorithm, "nsga2")
        self.assertIsNone(cfg.seed)
        self.assertTrue(cfg.verbose)

    def test_custom_values(self):
        cfg = NsgaWalkingConfig(
            n_generations=50, pop_size=20, seed=42, verbose=False,
        )
        self.assertEqual(cfg.n_generations, 50)
        self.assertEqual(cfg.pop_size, 20)
        self.assertEqual(cfg.seed, 42)
        self.assertFalse(cfg.verbose)


class TestWalkingNsgaProblem(unittest.TestCase):

    def test_problem_creation(self):
        """Problem has correct n_var and n_obj."""
        objectives = [DistanceFitness(duration=1), DistanceFitness(duration=1)]
        bounds = ([0.5, 1.0, 0.5], [2.0, 4.0, 3.0])
        problem = WalkingNsgaProblem(
            walker_factory=_make_fourbar_walker,
            objectives=objectives,
            bounds=bounds,
        )
        self.assertEqual(problem.problem.n_var, 3)
        self.assertEqual(problem.problem.n_obj, 2)

    def test_evaluate_candidate(self):
        """_evaluate_candidate returns correct number of scores."""
        objectives = [
            DistanceFitness(duration=1, n_legs=1),
            DistanceFitness(duration=1, n_legs=1),
        ]
        bounds = ([0.5, 1.0, 0.5], [2.0, 4.0, 3.0])
        problem = WalkingNsgaProblem(
            walker_factory=_make_fourbar_walker,
            objectives=objectives,
            bounds=bounds,
        )
        scores = problem._evaluate_candidate([1.0, 2.0, 1.5])
        self.assertEqual(len(scores), 2)
        for s in scores:
            self.assertIsInstance(s, float)


class TestNsgaWalkingOptimization(unittest.TestCase):
    """Integration tests for the full NSGA-II pipeline.

    Uses very small population and generations to keep tests fast.
    """

    def test_basic_run(self):
        """Optimization completes and returns valid result structure."""
        result = nsga_walking_optimization(
            walker_factory=_make_fourbar_walker,
            objectives=[
                DistanceFitness(duration=2, n_legs=1),
                DistanceFitness(duration=2, n_legs=1),
            ],
            bounds=([0.5, 1.0, 0.5], [2.0, 4.0, 3.0]),
            objective_names=["dist_a", "dist_b"],
            nsga_config=NsgaWalkingConfig(
                n_generations=3, pop_size=6, seed=42, verbose=False,
            ),
        )
        self.assertIsInstance(result, NsgaWalkingResult)
        self.assertIsInstance(result.pareto_front, ParetoFront)
        self.assertIsNone(result.gait_analyses)
        self.assertIsNone(result.stability_series)

    def test_result_has_solutions(self):
        """Pareto front should contain at least one solution."""
        result = nsga_walking_optimization(
            walker_factory=_make_fourbar_walker,
            objectives=[
                DistanceFitness(duration=2, n_legs=1),
            ],
            bounds=([0.5, 1.0, 0.5], [2.0, 4.0, 3.0]),
            nsga_config=NsgaWalkingConfig(
                n_generations=3, pop_size=6, seed=42, verbose=False,
            ),
        )
        self.assertGreater(len(result.pareto_front.solutions), 0)

    def test_best_compromise(self):
        """best_compromise should return a ParetoSolution."""
        result = nsga_walking_optimization(
            walker_factory=_make_fourbar_walker,
            objectives=[
                DistanceFitness(duration=2, n_legs=1),
                DistanceFitness(duration=2, n_legs=1),
            ],
            bounds=([0.5, 1.0, 0.5], [2.0, 4.0, 3.0]),
            nsga_config=NsgaWalkingConfig(
                n_generations=3, pop_size=6, seed=42, verbose=False,
            ),
        )
        best = result.best_compromise()
        self.assertIsNotNone(best)
        self.assertEqual(len(best.scores), 2)

    def test_with_gait_analysis(self):
        """include_gait=True produces gait analysis for Pareto solutions."""
        result = nsga_walking_optimization(
            walker_factory=_make_fourbar_walker,
            objectives=[DistanceFitness(duration=2, n_legs=1)],
            bounds=([0.5, 1.0, 0.5], [2.0, 4.0, 3.0]),
            nsga_config=NsgaWalkingConfig(
                n_generations=3, pop_size=6, seed=42, verbose=False,
            ),
            include_gait=True,
        )
        self.assertIsNotNone(result.gait_analyses)
        self.assertIsInstance(result.gait_analyses, dict)

    def test_with_stability(self):
        """include_stability=True produces stability series."""
        result = nsga_walking_optimization(
            walker_factory=_make_fourbar_walker,
            objectives=[DistanceFitness(duration=2, n_legs=1)],
            bounds=([0.5, 1.0, 0.5], [2.0, 4.0, 3.0]),
            nsga_config=NsgaWalkingConfig(
                n_generations=3, pop_size=6, seed=42, verbose=False,
            ),
            include_stability=True,
        )
        self.assertIsNotNone(result.stability_series)
        self.assertIsInstance(result.stability_series, dict)

    def test_best_for_objective(self):
        """best_for_objective returns best per-objective solution."""
        result = nsga_walking_optimization(
            walker_factory=_make_fourbar_walker,
            objectives=[
                DistanceFitness(duration=2, n_legs=1),
                DistanceFitness(duration=2, n_legs=1),
            ],
            bounds=([0.5, 1.0, 0.5], [2.0, 4.0, 3.0]),
            nsga_config=NsgaWalkingConfig(
                n_generations=3, pop_size=6, seed=42, verbose=False,
            ),
        )
        best_0 = result.best_for_objective(0)
        self.assertIsNotNone(best_0)


class TestBackwardCompatibility(unittest.TestCase):
    """Existing as_ga_fitness + GeneticOptimization still works."""

    def test_as_ga_fitness_unchanged(self):
        """as_ga_fitness still produces callable with correct signature."""
        fitness = DistanceFitness(duration=1, n_legs=1)
        ga_fn = as_ga_fitness(fitness, _make_fourbar_walker)
        self.assertTrue(callable(ga_fn))


class TestCompositeFitness(unittest.TestCase):

    def test_composite_returns_metrics(self):
        """CompositeFitness returns metrics for all requested objectives."""
        composite = CompositeFitness(
            duration=2, n_legs=1,
            objectives=("distance", "efficiency", "stability"),
        )
        walker = _make_fourbar_walker()
        result = composite(walker.topology, walker.dimensions)
        self.assertIsInstance(result.score, float)
        self.assertIn("distance", result.metrics)
        self.assertIn("efficiency", result.metrics)
        # Stability metrics
        self.assertIn("mean_tip_over_margin", result.metrics)

    def test_composite_subset(self):
        """CompositeFitness with only distance+efficiency skips stability."""
        composite = CompositeFitness(
            duration=2, n_legs=1,
            objectives=("distance", "efficiency"),
        )
        walker = _make_fourbar_walker()
        result = composite(walker.topology, walker.dimensions)
        self.assertIn("distance", result.metrics)
        self.assertIn("efficiency", result.metrics)
        self.assertNotIn("mean_tip_over_margin", result.metrics)


class TestStabilityFitness(unittest.TestCase):

    def test_stability_fitness_runs(self):
        """StabilityFitness returns a valid FitnessResult."""
        fitness = StabilityFitness(duration=2, n_legs=1, min_distance=0.0)
        walker = _make_fourbar_walker()
        result = fitness(walker.topology, walker.dimensions)
        self.assertIsInstance(result.score, float)
        self.assertTrue(result.valid)


if __name__ == "__main__":
    unittest.main()
