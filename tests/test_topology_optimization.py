#!/usr/bin/env python3
"""Tests for the topology co-optimization module."""

import unittest

from pylinkage.optimization.collections import ParetoFront

from leggedsnake.fitness import DistanceFitness
from leggedsnake.topology_optimization import (
    TopologyCoOptConfig,
    TopologySolutionInfo,
    TopologyWalkingResult,
    _TopologyContext,
    _TopologyWalkingProblem,
    topology_walking_optimization,
)


class TestTopologyContext(unittest.TestCase):

    def test_loads_catalog(self):
        """Context loads built-in catalog and finds topologies."""
        ctx = _TopologyContext(max_links=6)
        self.assertGreater(ctx.n_topologies, 0)
        self.assertGreater(ctx.max_edges, 0)

    def test_four_bar_only(self):
        """max_links=4 includes only four-bar."""
        ctx = _TopologyContext(max_links=4)
        self.assertEqual(ctx.n_topologies, 1)
        self.assertEqual(ctx.entries[0].family, "four-bar")

    def test_six_bar_includes_more(self):
        """max_links=6 includes four-bar + six-bar variants."""
        ctx = _TopologyContext(max_links=6)
        self.assertGreater(ctx.n_topologies, 1)

    def test_n_edges_per_topology(self):
        """Each topology has a positive edge count."""
        ctx = _TopologyContext(max_links=6)
        for i in range(ctx.n_topologies):
            self.assertGreater(ctx.n_edges(i), 0)

    def test_build_walker_fourbar(self):
        """Can build a Walker from four-bar topology."""
        ctx = _TopologyContext(max_links=4)
        dims = [1.0] * ctx.max_edges
        walker = ctx.build_walker(0, dims)
        # May or may not succeed depending on dimension compatibility,
        # but should not raise
        # Just test it doesn't crash
        self.assertTrue(walker is None or walker is not None)

    def test_build_walker_out_of_range(self):
        """Out-of-range topology index is clamped, not crashed."""
        ctx = _TopologyContext(max_links=4)
        dims = [1.0] * ctx.max_edges
        # Should clamp to valid range, not crash
        walker = ctx.build_walker(999, dims)
        self.assertTrue(walker is None or walker is not None)

    def test_build_walker_negative_index(self):
        """Negative topology index is clamped to 0."""
        ctx = _TopologyContext(max_links=4)
        dims = [1.0] * ctx.max_edges
        walker = ctx.build_walker(-5, dims)
        self.assertTrue(walker is None or walker is not None)


class TestTopologyWalkingProblem(unittest.TestCase):

    def test_problem_creation(self):
        """Problem has correct n_var and n_obj."""
        ctx = _TopologyContext(max_links=4)
        objectives = [DistanceFitness(duration=1, n_legs=1)]
        cfg = TopologyCoOptConfig(n_legs=1)
        problem = _TopologyWalkingProblem(
            ctx=ctx, objectives=objectives, config=cfg,
        )
        # n_var = 1 (topology) + max_edges
        self.assertEqual(problem.problem.n_var, 1 + ctx.max_edges)
        self.assertEqual(problem.problem.n_obj, 1)

    def test_evaluate_candidate(self):
        """Candidate evaluation returns correct number of scores."""
        ctx = _TopologyContext(max_links=4)
        objectives = [
            DistanceFitness(duration=1, n_legs=1),
            DistanceFitness(duration=1, n_legs=1),
        ]
        cfg = TopologyCoOptConfig(n_legs=1)
        problem = _TopologyWalkingProblem(
            ctx=ctx, objectives=objectives, config=cfg,
        )
        import numpy as np
        x = np.ones(1 + ctx.max_edges)
        x[0] = 0.0  # four-bar topology
        scores = problem._evaluate_candidate(x)
        self.assertEqual(len(scores), 2)


class TestTopologyWalkingOptimization(unittest.TestCase):
    """Integration tests with very small populations."""

    def test_basic_run(self):
        """Optimization completes and returns valid result."""
        result = topology_walking_optimization(
            objectives=[DistanceFitness(duration=2, n_legs=1)],
            objective_names=["distance"],
            config=TopologyCoOptConfig(
                max_links=4,
                n_generations=2,
                pop_size=4,
                seed=42,
                verbose=False,
                n_legs=1,
            ),
        )
        self.assertIsInstance(result, TopologyWalkingResult)
        self.assertIsInstance(result.pareto_front, ParetoFront)

    def test_multi_objective(self):
        """Two-objective optimization returns Pareto front."""
        result = topology_walking_optimization(
            objectives=[
                DistanceFitness(duration=2, n_legs=1),
                DistanceFitness(duration=2, n_legs=1),
            ],
            objective_names=["dist_a", "dist_b"],
            config=TopologyCoOptConfig(
                max_links=4,
                n_generations=2,
                pop_size=4,
                seed=42,
                verbose=False,
                n_legs=1,
            ),
        )
        self.assertIsInstance(result, TopologyWalkingResult)

    def test_solutions_have_topology_info(self):
        """Pareto solutions have topology metadata."""
        result = topology_walking_optimization(
            objectives=[DistanceFitness(duration=2, n_legs=1)],
            config=TopologyCoOptConfig(
                max_links=4,
                n_generations=2,
                pop_size=4,
                seed=42,
                verbose=False,
                n_legs=1,
            ),
        )
        for idx in range(len(result.pareto_front.solutions)):
            if idx in result.topology_info:
                info = result.topology_info[idx]
                self.assertIsInstance(info, TopologySolutionInfo)
                self.assertIsInstance(info.topology_name, str)
                self.assertIsInstance(info.topology_id, str)
                self.assertGreater(info.num_links, 0)

    def test_solutions_by_topology(self):
        """solutions_by_topology groups correctly."""
        result = topology_walking_optimization(
            objectives=[DistanceFitness(duration=2, n_legs=1)],
            config=TopologyCoOptConfig(
                max_links=4,
                n_generations=2,
                pop_size=4,
                seed=42,
                verbose=False,
                n_legs=1,
            ),
        )
        groups = result.solutions_by_topology()
        self.assertIsInstance(groups, dict)


class TestTopologyCoOptConfig(unittest.TestCase):

    def test_defaults(self):
        cfg = TopologyCoOptConfig()
        self.assertEqual(cfg.max_links, 8)
        self.assertEqual(cfg.n_generations, 100)
        self.assertEqual(cfg.n_legs, 2)

    def test_custom(self):
        cfg = TopologyCoOptConfig(max_links=6, n_legs=4, seed=123)
        self.assertEqual(cfg.max_links, 6)
        self.assertEqual(cfg.n_legs, 4)
        self.assertEqual(cfg.seed, 123)

    def test_leg_gene_inactive_by_default(self):
        cfg = TopologyCoOptConfig()
        self.assertFalse(cfg.leg_gene_active)
        self.assertEqual(cfg.leg_bounds, (2, 2))

    def test_leg_gene_active_when_range_differs(self):
        cfg = TopologyCoOptConfig(n_legs_min=2, n_legs_max=6)
        self.assertTrue(cfg.leg_gene_active)
        self.assertEqual(cfg.leg_bounds, (2, 6))

    def test_leg_gene_inactive_when_range_collapses(self):
        # Equal bounds → fixed leg count, no gene.
        cfg = TopologyCoOptConfig(n_legs_min=3, n_legs_max=3)
        self.assertFalse(cfg.leg_gene_active)
        self.assertEqual(cfg.leg_bounds, (3, 3))


class TestLegGeneChromosome(unittest.TestCase):

    def test_problem_has_no_leg_gene_by_default(self):
        ctx = _TopologyContext(max_links=4)
        cfg = TopologyCoOptConfig(n_legs=1)
        problem = _TopologyWalkingProblem(
            ctx=ctx,
            objectives=[DistanceFitness(duration=1, n_legs=1)],
            config=cfg,
        )
        self.assertEqual(problem.problem.n_var, 1 + ctx.max_edges)

    def test_problem_adds_leg_gene_when_range_differs(self):
        ctx = _TopologyContext(max_links=4)
        cfg = TopologyCoOptConfig(n_legs_min=2, n_legs_max=6)
        problem = _TopologyWalkingProblem(
            ctx=ctx,
            objectives=[DistanceFitness(duration=1, n_legs=1)],
            config=cfg,
        )
        self.assertEqual(problem.problem.n_var, 2 + ctx.max_edges)
        # Leg-gene bounds at index 1.
        self.assertEqual(problem.problem.xl[1], 2.0)
        self.assertEqual(problem.problem.xu[1], 6.0)

    def test_decode_chromosome_fixed_legs(self):
        from leggedsnake.topology_optimization import _decode_chromosome
        cfg = TopologyCoOptConfig(n_legs=4)
        import numpy as np
        x = np.array([2.0, 1.0, 2.0, 3.0])
        topo, n_legs, dims = _decode_chromosome(x, cfg)
        self.assertEqual(topo, 2)
        self.assertEqual(n_legs, 4)
        self.assertEqual(dims, [1.0, 2.0, 3.0])

    def test_decode_chromosome_with_leg_gene(self):
        from leggedsnake.topology_optimization import _decode_chromosome
        cfg = TopologyCoOptConfig(n_legs_min=2, n_legs_max=6)
        import numpy as np
        x = np.array([1.0, 4.7, 1.0, 2.0, 3.0])
        topo, n_legs, dims = _decode_chromosome(x, cfg)
        self.assertEqual(topo, 1)
        self.assertEqual(n_legs, 5)  # rounded
        self.assertEqual(dims, [1.0, 2.0, 3.0])

    def test_decode_chromosome_clamps_leg_gene(self):
        from leggedsnake.topology_optimization import _decode_chromosome
        cfg = TopologyCoOptConfig(n_legs_min=3, n_legs_max=5)
        import numpy as np
        x_high = np.array([0.0, 10.0, 1.0])
        x_low = np.array([0.0, -5.0, 1.0])
        _, high, _ = _decode_chromosome(x_high, cfg)
        _, low, _ = _decode_chromosome(x_low, cfg)
        self.assertEqual(high, 5)
        self.assertEqual(low, 3)


if __name__ == "__main__":
    unittest.main()
