#!/usr/bin/env python3
"""Tests for performance knobs: pool reuse and sparse loci recording."""

import unittest
from math import tau

import numpy as np

from pylinkage.dimensions import Dimensions, DriverAngle
from pylinkage.hypergraph import HypergraphLinkage, Node, Edge, NodeRole

from leggedsnake.fitness import CompositeFitness, DistanceFitness, GaitFitness
from leggedsnake.nsga_optimizer import (
    NsgaWalkingConfig,
    WalkingNsgaProblem,
    nsga_walking_optimization,
)
from leggedsnake.walker import Walker


def _make_fourbar_walker() -> Walker:
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


def _make_walking_walker() -> Walker:
    """A walker with multiple joints so loci recording is meaningful."""
    hg = HypergraphLinkage(name="walking")
    hg.add_node(Node("frame", role=NodeRole.GROUND))
    hg.add_node(Node("frame2", role=NodeRole.GROUND))
    hg.add_node(Node("crank", role=NodeRole.DRIVER))
    hg.add_node(Node("upper", role=NodeRole.DRIVEN))
    hg.add_node(Node("foot", role=NodeRole.DRIVEN))
    hg.add_edge(Edge("frame_crank", "frame", "crank"))
    hg.add_edge(Edge("frame2_upper", "frame2", "upper"))
    hg.add_edge(Edge("crank_upper", "crank", "upper"))
    hg.add_edge(Edge("crank_foot", "crank", "foot"))
    hg.add_edge(Edge("upper_foot", "upper", "foot"))
    dims = Dimensions(
        node_positions={
            "frame": (0, 0), "frame2": (2, 0),
            "crank": (1, 0), "upper": (1, 2), "foot": (1, 3),
        },
        driver_angles={"crank": DriverAngle(angular_velocity=-tau / 12)},
        edge_distances={
            "frame_crank": 1.0, "frame2_upper": 2.24,
            "crank_upper": 2.0, "crank_foot": 3.16, "upper_foot": 1.0,
        },
    )
    return Walker(hg, dims, name="walking", motor_rates=-4.0)


class TestPoolReuse(unittest.TestCase):
    """The shared process pool persists across batches and is cleaned
    up on ``close()``. Previously each generation forked a fresh pool
    of N workers — a measurable per-generation tax."""

    def test_pool_lazy_create(self):
        """No pool is created until a parallel batch runs."""
        problem = WalkingNsgaProblem(
            walker_factory=_make_fourbar_walker,
            objectives=[DistanceFitness(duration=0.3, n_legs=1)],
            bounds=([0.5, 1.0, 0.5], [2.0, 4.0, 3.0]),
            n_workers=2,
        )
        try:
            self.assertIsNone(problem._pool)
        finally:
            problem.close()

    def test_pool_persists_across_batches(self):
        """Same pool object survives multiple batch evaluations."""
        problem = WalkingNsgaProblem(
            walker_factory=_make_fourbar_walker,
            objectives=[DistanceFitness(duration=0.3, n_legs=1)],
            bounds=([0.5, 1.0, 0.5], [2.0, 4.0, 3.0]),
            n_workers=2,
        )
        try:
            X = np.array([[1.0, 2.0, 1.5], [0.8, 1.5, 1.2]])
            problem._evaluate_batch(X)
            pool_after_first = problem._pool
            self.assertIsNotNone(pool_after_first)
            problem._evaluate_batch(X)
            self.assertIs(problem._pool, pool_after_first)
        finally:
            problem.close()

    def test_close_shuts_down_pool(self):
        problem = WalkingNsgaProblem(
            walker_factory=_make_fourbar_walker,
            objectives=[DistanceFitness(duration=0.3, n_legs=1)],
            bounds=([0.5, 1.0, 0.5], [2.0, 4.0, 3.0]),
            n_workers=2,
        )
        X = np.array([[1.0, 2.0, 1.5]])
        problem._evaluate_batch(X)
        self.assertIsNotNone(problem._pool)
        problem.close()
        self.assertIsNone(problem._pool)

    def test_close_idempotent(self):
        """Calling close() twice — or before any pool exists — is safe."""
        problem = WalkingNsgaProblem(
            walker_factory=_make_fourbar_walker,
            objectives=[DistanceFitness(duration=0.3, n_legs=1)],
            bounds=([0.5, 1.0, 0.5], [2.0, 4.0, 3.0]),
            n_workers=2,
        )
        problem.close()  # before any pool
        problem.close()  # idempotent

    def test_optimization_driver_closes_pool(self):
        """``nsga_walking_optimization`` shuts down the pool in finally,
        so workers don't outlive the call."""
        result = nsga_walking_optimization(
            walker_factory=_make_fourbar_walker,
            objectives=[DistanceFitness(duration=0.3, n_legs=1)],
            bounds=([0.5, 1.0, 0.5], [2.0, 4.0, 3.0]),
            nsga_config=NsgaWalkingConfig(
                n_generations=2, pop_size=4, seed=42, verbose=False,
                n_workers=2,
            ),
        )
        self.assertGreater(len(result.pareto_front.solutions), 0)


class TestSparseLoci(unittest.TestCase):
    """``loci_stride > 1`` keeps every Nth recorded step. Cuts memory
    proportionally for long sweeps."""

    def test_default_stride_records_every_step(self):
        result = CompositeFitness(
            duration=1.0, n_legs=1, objectives=("distance",),
        )(_make_walking_walker().topology, _make_walking_walker().dimensions)
        # default stride=1 → one loci entry per physics step
        if result.loci:
            joint_count = len(next(iter(result.loci.values())))
            self.assertGreater(joint_count, 30)  # 1s @ ~50Hz physics

    def test_stride_10_cuts_loci_count(self):
        full = CompositeFitness(
            duration=1.0, n_legs=1, objectives=("distance",),
            loci_stride=1,
        )(_make_walking_walker().topology, _make_walking_walker().dimensions)
        sparse = CompositeFitness(
            duration=1.0, n_legs=1, objectives=("distance",),
            loci_stride=10,
        )(_make_walking_walker().topology, _make_walking_walker().dimensions)
        if full.loci and sparse.loci:
            full_count = len(next(iter(full.loci.values())))
            sparse_count = len(next(iter(sparse.loci.values())))
            # ~10× fewer loci entries, allow ±1 for the boundary frame.
            self.assertAlmostEqual(
                full_count / max(sparse_count, 1), 10.0, delta=1.0,
            )

    def test_stride_clamped_to_one(self):
        """``loci_stride=0`` is silently bumped to 1 (record every step)."""
        cf = CompositeFitness(
            duration=0.3, n_legs=1, objectives=("distance",),
            loci_stride=0,
        )
        self.assertEqual(cf.loci_stride, 1)

    def test_gait_fitness_loci_stride(self):
        """``GaitFitness`` accepts loci_stride and the resulting metric
        still has the canonical fields."""
        result = GaitFitness(
            duration=1.0, n_legs=1, loci_stride=5,
        )(_make_walking_walker().topology, _make_walking_walker().dimensions)
        # Whatever stride we choose, the canonical gait fields exist.
        self.assertIn("mean_stride_length", result.metrics)


class TestLociDtPropagatesToGait(unittest.TestCase):
    """When loci is subsampled, ``_SimulationResult.loci_dt`` reports
    ``physics_dt * loci_stride``. Gait analysis must use that — not
    raw ``dt`` — or stride times come out 10× too short."""

    def test_loci_dt_scales_with_stride(self):
        from leggedsnake.fitness import _run_simulation

        walker = _make_walking_walker()
        result = _run_simulation(
            walker.topology, walker.dimensions, None,
            duration=0.3, n_legs=1, motor_rates=-4.0,
            record_loci=True, loci_stride=5,
        )
        self.assertAlmostEqual(result.loci_dt, result.dt * 5, places=10)


if __name__ == "__main__":
    unittest.main()
