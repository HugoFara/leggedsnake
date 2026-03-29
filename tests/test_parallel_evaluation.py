#!/usr/bin/env python3
"""Tests for parallel fitness evaluation in NSGA optimizers."""

import unittest
from math import tau

import numpy as np

from pylinkage.dimensions import Dimensions, DriverAngle
from pylinkage.hypergraph import HypergraphLinkage, Node, Edge, NodeRole

from leggedsnake.fitness import DistanceFitness
from leggedsnake.nsga_optimizer import (
    NsgaWalkingConfig,
    WalkingNsgaProblem,
    nsga_walking_optimization,
)
from leggedsnake.topology_optimization import (
    TopologyCoOptConfig,
    _TopologyContext,
    _TopologyWalkingProblem,
    topology_walking_optimization,
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


class TestWalkingNsgaProblemParallel(unittest.TestCase):

    def test_sequential_batch(self):
        """n_workers=1 evaluates batch sequentially."""
        problem = WalkingNsgaProblem(
            walker_factory=_make_fourbar_walker,
            objectives=[DistanceFitness(duration=1, n_legs=1)],
            bounds=([0.5, 1.0, 0.5], [2.0, 4.0, 3.0]),
            n_workers=1,
        )
        X = np.array([[1.0, 2.0, 1.5], [0.8, 1.5, 1.2]])
        F = problem._evaluate_batch(X)
        self.assertEqual(F.shape, (2, 1))

    def test_parallel_batch(self):
        """n_workers=2 evaluates batch in parallel."""
        problem = WalkingNsgaProblem(
            walker_factory=_make_fourbar_walker,
            objectives=[DistanceFitness(duration=1, n_legs=1)],
            bounds=([0.5, 1.0, 0.5], [2.0, 4.0, 3.0]),
            n_workers=2,
        )
        X = np.array([[1.0, 2.0, 1.5], [0.8, 1.5, 1.2]])
        F = problem._evaluate_batch(X)
        self.assertEqual(F.shape, (2, 1))

    def test_config_n_workers_passed(self):
        """NsgaWalkingConfig.n_workers flows through to optimization."""
        result = nsga_walking_optimization(
            walker_factory=_make_fourbar_walker,
            objectives=[DistanceFitness(duration=1, n_legs=1)],
            bounds=([0.5, 1.0, 0.5], [2.0, 4.0, 3.0]),
            nsga_config=NsgaWalkingConfig(
                n_generations=2,
                pop_size=4,
                seed=42,
                verbose=False,
                n_workers=1,  # sequential for speed in test
            ),
        )
        self.assertGreater(len(result.pareto_front.solutions), 0)


class TestTopologyProblemParallel(unittest.TestCase):

    def test_sequential_batch(self):
        """n_workers=1 evaluates topology batch sequentially."""
        ctx = _TopologyContext(max_links=4)
        cfg = TopologyCoOptConfig(n_legs=1, n_workers=1)
        problem = _TopologyWalkingProblem(
            ctx=ctx,
            objectives=[DistanceFitness(duration=1, n_legs=1)],
            config=cfg,
        )
        n_var = 1 + ctx.max_edges
        X = np.ones((3, n_var))
        X[:, 0] = 0.0
        F = problem._evaluate_batch(X)
        self.assertEqual(F.shape, (3, 1))

    def test_parallel_batch(self):
        """n_workers=2 evaluates topology batch in parallel."""
        ctx = _TopologyContext(max_links=4)
        cfg = TopologyCoOptConfig(n_legs=1, n_workers=2)
        problem = _TopologyWalkingProblem(
            ctx=ctx,
            objectives=[DistanceFitness(duration=1, n_legs=1)],
            config=cfg,
        )
        n_var = 1 + ctx.max_edges
        X = np.ones((3, n_var))
        X[:, 0] = 0.0
        F = problem._evaluate_batch(X)
        self.assertEqual(F.shape, (3, 1))

    def test_topology_config_n_workers(self):
        """TopologyCoOptConfig.n_workers flows through."""
        result = topology_walking_optimization(
            objectives=[DistanceFitness(duration=1, n_legs=1)],
            config=TopologyCoOptConfig(
                max_links=4,
                n_generations=2,
                pop_size=4,
                seed=42,
                verbose=False,
                n_legs=1,
                n_workers=1,
            ),
        )
        self.assertIsNotNone(result.pareto_front)


if __name__ == "__main__":
    unittest.main()
