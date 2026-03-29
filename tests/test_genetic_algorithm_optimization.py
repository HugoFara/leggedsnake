#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Tests for the genetic_algorithm_optimization standard-signature wrapper.
"""

import inspect
import unittest
import warnings

import pymunk as pm

with warnings.catch_warnings():
    warnings.filterwarnings(
        "ignore", category=DeprecationWarning, message=r"pylinkage\.joints"
    )
    from pylinkage import Static, Crank, Pivot, Linkage

from pylinkage.optimization.collections import Agent

from leggedsnake.geneticoptimizer import genetic_algorithm_optimization


def _make_linkage():
    """Create a minimal kinematic linkage for testing."""
    base = Static(0, 0, name="base")
    crank = Crank(1, 0, joint0=base, distance=1, angle=0.1, name="crank")
    follower = Pivot(
        0, 2, joint0=base, joint1=crank,
        distance0=2, distance1=1.5, name="follower",
    )
    return Linkage(joints=(base, crank, follower), name="test")


def _simple_evaluator(linkage, dims, pos):
    """Maximize negative sum-of-squares (optimum at dims = [0, 0, ...])."""
    return -sum(d ** 2 for d in dims)


def _simple_minimizer(linkage, dims, pos):
    """Minimize sum-of-squares (optimum at dims = [0, 0, ...])."""
    return sum(d ** 2 for d in dims)


class TestGeneticAlgorithmOptimization(unittest.TestCase):
    """Tests for the standard-signature GA wrapper."""

    def test_returns_list_of_agents(self):
        """Result must be a list of Agent namedtuples."""
        linkage = _make_linkage()
        results = genetic_algorithm_optimization(
            _simple_evaluator, linkage,
            center=[1.0, 0.1, 2.0, 1.5],
            iters=3, max_pop=5, verbose=False,
        )
        self.assertIsInstance(results, list)
        self.assertGreater(len(results), 0)
        for agent in results:
            self.assertIsInstance(agent, Agent)
            self.assertIsInstance(agent.score, float)
            self.assertIsNotNone(agent.dimensions)
            self.assertIsNotNone(agent.init_positions)

    def test_agent_indexing_backward_compat(self):
        """Agent[0], Agent[1], Agent[2] must still work."""
        linkage = _make_linkage()
        results = genetic_algorithm_optimization(
            _simple_evaluator, linkage,
            center=[1.0, 0.1, 2.0, 1.5],
            iters=2, max_pop=5, verbose=False,
        )
        best = results[0]
        self.assertEqual(best[0], best.score)
        self.assertEqual(best[1], best.dimensions)
        self.assertEqual(best[2], best.init_positions)

    def test_maximization(self):
        """Default order_relation=max should maximize the score."""
        linkage = _make_linkage()
        results = genetic_algorithm_optimization(
            _simple_evaluator, linkage,
            center=[5.0, 5.0, 5.0, 5.0],
            iters=5, max_pop=8, verbose=False,
        )
        # Best score should be better (less negative) than initial
        initial_score = _simple_evaluator(linkage, [5, 5, 5, 5], [])
        self.assertGreaterEqual(results[0].score, initial_score)

    def test_minimization(self):
        """order_relation=min should minimize."""
        linkage = _make_linkage()
        results = genetic_algorithm_optimization(
            _simple_minimizer, linkage,
            center=[5.0, 5.0, 5.0, 5.0],
            order_relation=min,
            iters=5, max_pop=8, verbose=False,
        )
        initial_score = _simple_minimizer(linkage, [5, 5, 5, 5], [])
        self.assertLessEqual(results[0].score, initial_score)

    def test_center_parameter_used(self):
        """The center parameter should seed the initial population."""
        linkage = _make_linkage()
        center = [2.0, 0.5, 3.0, 2.0]
        results = genetic_algorithm_optimization(
            _simple_evaluator, linkage,
            center=center,
            iters=1, max_pop=3, verbose=False,
        )
        # At least the seed DNA should use the provided center
        self.assertEqual(len(results[0].dimensions), len(center))

    def test_chainable_signature(self):
        """Function must have 'center' param for chain_optimizers introspection."""
        sig = inspect.signature(genetic_algorithm_optimization)
        self.assertIn("center", sig.parameters)
        self.assertIn("eval_func", sig.parameters)
        self.assertIn("linkage", sig.parameters)
        self.assertIn("order_relation", sig.parameters)
        self.assertIn("bounds", sig.parameters)

    def test_sorted_by_score(self):
        """Results should be sorted by score, best first."""
        linkage = _make_linkage()
        results = genetic_algorithm_optimization(
            _simple_evaluator, linkage,
            center=[3.0, 3.0, 3.0, 3.0],
            iters=3, max_pop=6, verbose=False,
        )
        if len(results) > 1:
            for i in range(len(results) - 1):
                self.assertGreaterEqual(results[i].score, results[i + 1].score)


class TestGeneticOptimizationRunReturnsAgent(unittest.TestCase):
    """Test that GeneticOptimization.run() returns list[Agent]."""

    def test_run_returns_agents(self):
        """Direct GeneticOptimization usage should also return Agents."""
        from leggedsnake.geneticoptimizer import GeneticOptimization

        def fitness(dna):
            return -sum(d ** 2 for d in dna[1]), dna[2]

        dna = [0, [1.0, 2.0, 3.0], [(0, 0), (1, 0), (0, 2)]]
        dna[0] = fitness(dna)[0]
        optimizer = GeneticOptimization(
            dna=dna, fitness=fitness, prob=0.1, max_pop=5, verbose=0,
        )
        results = optimizer.run(iters=3)
        self.assertIsInstance(results, list)
        self.assertIsInstance(results[0], Agent)


if __name__ == "__main__":
    unittest.main()
