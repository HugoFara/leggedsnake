#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Tests for the genetic_algorithm_optimization standard-signature wrapper."""

import inspect
import unittest
from math import tau

from pylinkage.dimensions import Dimensions, DriverAngle
from pylinkage.hypergraph import HypergraphLinkage, Node, Edge, NodeRole
from pylinkage.optimization.collections import Agent
from pylinkage.population import Ensemble, Member

from leggedsnake.genetic_optimizer import genetic_algorithm_optimization
from leggedsnake.walker import Walker


def _make_walker():
    """Create a minimal Walker for testing."""
    hg = HypergraphLinkage(name="test")
    hg.add_node(Node("frame", role=NodeRole.GROUND))
    hg.add_node(Node("crank", role=NodeRole.DRIVER))
    hg.add_node(Node("follower", role=NodeRole.DRIVEN))
    hg.add_edge(Edge("edge_0", "frame", "crank"))
    hg.add_edge(Edge("edge_1", "frame", "follower"))
    hg.add_edge(Edge("edge_2", "crank", "follower"))
    dims = Dimensions(
        node_positions={"frame": (0, 0), "crank": (1, 0), "follower": (0, 2)},
        driver_angles={"crank": DriverAngle(angular_velocity=-tau / 12)},
        edge_distances={"edge_0": 1.0, "edge_1": 2.0, "edge_2": 1.5},
    )
    return Walker(hg, dims, name="test")


def _simple_evaluator(linkage, dims, pos):
    """Maximize negative sum-of-squares (optimum at dims = [0, 0, ...])."""
    return -sum(d ** 2 for d in dims)


def _simple_minimizer(linkage, dims, pos):
    """Minimize sum-of-squares (optimum at dims = [0, 0, ...])."""
    return sum(d ** 2 for d in dims)


class TestGeneticAlgorithmOptimization(unittest.TestCase):
    def test_returns_ensemble_of_members(self):
        walker = _make_walker()
        results = genetic_algorithm_optimization(
            _simple_evaluator, walker,
            center=[1.0, 2.0, 1.5],
            iters=3, max_pop=5, verbose=False,
        )
        self.assertIsInstance(results, Ensemble)
        self.assertGreater(results.n_members, 0)
        for member in results:
            self.assertIsInstance(member, Member)
            self.assertIsInstance(member.score, float)
            self.assertEqual(member.dimensions.shape, (3,))
            self.assertEqual(member.initial_positions.shape, (3, 2))

    def test_agent_backward_compat_conversion(self):
        walker = _make_walker()
        results = genetic_algorithm_optimization(
            _simple_evaluator, walker,
            center=[1.0, 2.0, 1.5],
            iters=2, max_pop=5, verbose=False,
        )
        best = results[0]
        agent = best.to_agent()
        self.assertIsInstance(agent, Agent)
        self.assertEqual(agent.score, best.score)

    def test_maximization(self):
        walker = _make_walker()
        results = genetic_algorithm_optimization(
            _simple_evaluator, walker,
            center=[5.0, 5.0, 5.0],
            iters=5, max_pop=8, verbose=False,
        )
        initial_score = _simple_evaluator(walker, [5, 5, 5], [])
        self.assertGreaterEqual(results[0].score, initial_score)

    def test_minimization(self):
        walker = _make_walker()
        results = genetic_algorithm_optimization(
            _simple_minimizer, walker,
            center=[5.0, 5.0, 5.0],
            order_relation=min,
            iters=5, max_pop=8, verbose=False,
        )
        initial_score = _simple_minimizer(walker, [5, 5, 5], [])
        self.assertLessEqual(results[0].score, initial_score)

    def test_center_parameter_used(self):
        walker = _make_walker()
        center = [2.0, 3.0, 2.0]
        results = genetic_algorithm_optimization(
            _simple_evaluator, walker,
            center=center,
            iters=1, max_pop=3, verbose=False,
        )
        self.assertEqual(results[0].dimensions.shape[0], len(center))

    def test_chainable_signature(self):
        sig = inspect.signature(genetic_algorithm_optimization)
        self.assertIn("center", sig.parameters)
        self.assertIn("eval_func", sig.parameters)
        self.assertIn("linkage", sig.parameters)
        self.assertIn("order_relation", sig.parameters)
        self.assertIn("bounds", sig.parameters)

    def test_sorted_by_score(self):
        walker = _make_walker()
        results = genetic_algorithm_optimization(
            _simple_evaluator, walker,
            center=[3.0, 3.0, 3.0],
            iters=3, max_pop=6, verbose=False,
        )
        if results.n_members > 1:
            for i in range(results.n_members - 1):
                self.assertGreaterEqual(results[i].score, results[i + 1].score)

    def test_top_and_rank(self):
        """Ensemble exposes .top() and .rank() from pylinkage 0.9."""
        walker = _make_walker()
        results = genetic_algorithm_optimization(
            _simple_evaluator, walker,
            center=[3.0, 3.0, 3.0],
            iters=3, max_pop=6, verbose=False,
        )
        top2 = results.top(2, key="score", ascending=False)
        self.assertIsInstance(top2, Ensemble)
        self.assertLessEqual(top2.n_members, 2)


class TestGeneticOptimizationRunReturnsAgent(unittest.TestCase):
    def test_run_returns_agents(self):
        from leggedsnake.genetic_optimizer import GeneticOptimization

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
