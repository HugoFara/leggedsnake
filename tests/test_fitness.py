#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Tests for the physics-aware fitness protocol (fitness.py).

Covers FitnessResult, DynamicFitness Protocol conformance, built-in
fitness classes, and adapter functions.
"""
import unittest
from math import tau

from pylinkage.dimensions import Dimensions, DriverAngle
from pylinkage.hypergraph import HypergraphLinkage, Node, Edge, NodeRole

import leggedsnake as ls
from leggedsnake.fitness import (
    DynamicFitness,
    DistanceFitness,
    EfficiencyFitness,
    FitnessResult,
    StrideFitness,
    as_eval_func,
    as_ga_fitness,
)
from leggedsnake.physics_engine import WorldConfig
from leggedsnake.walker import Walker


def _make_simple_walker() -> Walker:
    """Create a 5-node walker suitable for physics simulation."""
    hg = HypergraphLinkage(name="simple")
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
    return Walker(hg, dims, name="simple", motor_rates=-4.0)


def _make_fourbar_walker() -> Walker:
    """Simple 3-node fourbar with semantic edge IDs."""
    hg = HypergraphLinkage(name="fourbar")
    hg.add_node(Node("A", role=NodeRole.GROUND))
    hg.add_node(Node("B", role=NodeRole.DRIVER))
    hg.add_node(Node("C", role=NodeRole.DRIVEN))
    hg.add_edge(Edge("A_B", "A", "B"))
    hg.add_edge(Edge("A_C", "A", "C"))
    hg.add_edge(Edge("B_C", "B", "C"))
    dims = Dimensions(
        node_positions={"A": (0, 0), "B": (1, 0), "C": (0, 2)},
        driver_angles={"B": DriverAngle(angular_velocity=-tau / 12)},
        edge_distances={"A_B": 1.0, "A_C": 2.0, "B_C": 1.5},
    )
    return Walker(hg, dims, name="fourbar")


class TestFitnessResult(unittest.TestCase):
    """Test the FitnessResult dataclass."""

    def test_defaults(self):
        """FitnessResult has sensible defaults."""
        r = FitnessResult(score=1.5)
        self.assertEqual(r.score, 1.5)
        self.assertEqual(r.metrics, {})
        self.assertTrue(r.valid)
        self.assertEqual(r.loci, {})

    def test_with_metrics(self):
        """FitnessResult stores arbitrary metrics."""
        r = FitnessResult(
            score=3.0,
            metrics={"distance": 12.3, "energy": 45.6},
            valid=True,
        )
        self.assertAlmostEqual(r.metrics["distance"], 12.3)
        self.assertAlmostEqual(r.metrics["energy"], 45.6)

    def test_invalid_result(self):
        """FitnessResult can represent invalid evaluations."""
        r = FitnessResult(score=0.0, valid=False)
        self.assertFalse(r.valid)

    def test_with_loci(self):
        """FitnessResult stores joint trajectories."""
        r = FitnessResult(
            score=1.0,
            loci={"foot": [(0.0, 0.0), (1.0, 0.5), (2.0, 0.0)]},
        )
        self.assertEqual(len(r.loci["foot"]), 3)


class TestProtocolConformance(unittest.TestCase):
    """Verify built-in classes satisfy DynamicFitness Protocol."""

    def test_distance_fitness_is_dynamic_fitness(self):
        self.assertIsInstance(DistanceFitness(), DynamicFitness)

    def test_efficiency_fitness_is_dynamic_fitness(self):
        self.assertIsInstance(EfficiencyFitness(), DynamicFitness)

    def test_stride_fitness_is_dynamic_fitness(self):
        self.assertIsInstance(StrideFitness(), DynamicFitness)

    def test_custom_callable_satisfies_protocol(self):
        """A plain callable with matching signature satisfies Protocol."""
        def my_fitness(topology, dimensions, config=None):
            return FitnessResult(score=42.0)
        self.assertIsInstance(my_fitness, DynamicFitness)


class TestDistanceFitness(unittest.TestCase):
    """Test DistanceFitness evaluation."""

    def test_returns_fitness_result(self):
        """DistanceFitness returns a FitnessResult with distance metric."""
        walker = _make_simple_walker()
        fitness = DistanceFitness(duration=0.5, n_legs=1)
        result = fitness(walker.topology, walker.dimensions)
        self.assertIsInstance(result, FitnessResult)
        self.assertIsInstance(result.score, float)
        self.assertIn("distance", result.metrics)
        self.assertIn("total_energy", result.metrics)
        self.assertTrue(result.valid)

    def test_with_custom_config(self):
        """DistanceFitness respects custom WorldConfig."""
        walker = _make_simple_walker()
        cfg = WorldConfig(gravity=(0, -5.0), physics_period=0.05)
        fitness = DistanceFitness(duration=0.5, n_legs=1)
        result = fitness(walker.topology, walker.dimensions, config=cfg)
        self.assertIsInstance(result, FitnessResult)

    def test_with_multiple_legs(self):
        """DistanceFitness works with n_legs > 1."""
        walker = _make_simple_walker()
        fitness = DistanceFitness(duration=0.5, n_legs=2)
        result = fitness(walker.topology, walker.dimensions)
        self.assertIsInstance(result, FitnessResult)
        self.assertTrue(result.valid)

    def test_record_loci(self):
        """DistanceFitness records joint trajectories when requested."""
        walker = _make_simple_walker()
        fitness = DistanceFitness(duration=0.5, n_legs=1, record_loci=True)
        result = fitness(walker.topology, walker.dimensions)
        self.assertGreater(len(result.loci), 0)
        # Each locus should have multiple points
        for name, points in result.loci.items():
            self.assertGreater(len(points), 0)
            self.assertEqual(len(points[0]), 2)  # (x, y) tuples


class TestEfficiencyFitness(unittest.TestCase):
    """Test EfficiencyFitness evaluation."""

    def test_returns_fitness_result(self):
        """EfficiencyFitness returns FitnessResult with efficiency metrics."""
        walker = _make_simple_walker()
        fitness = EfficiencyFitness(duration=0.5, n_legs=1, min_distance=0.0)
        result = fitness(walker.topology, walker.dimensions)
        self.assertIsInstance(result, FitnessResult)
        self.assertIn("efficiency_ratio", result.metrics)
        self.assertIn("distance", result.metrics)

    def test_min_distance_filter(self):
        """EfficiencyFitness returns 0 when min_distance not met."""
        walker = _make_simple_walker()
        # Require an impossibly large distance for a 0.1s sim
        fitness = EfficiencyFitness(duration=0.1, n_legs=1, min_distance=1000.0)
        result = fitness(walker.topology, walker.dimensions)
        self.assertEqual(result.score, 0.0)


class TestStrideFitness(unittest.TestCase):
    """Test StrideFitness (kinematic, no physics)."""

    def test_returns_fitness_result(self):
        """StrideFitness returns a FitnessResult."""
        walker = _make_simple_walker()
        fitness = StrideFitness(lap_points=6, foot_index=-1)
        result = fitness(walker.topology, walker.dimensions)
        self.assertIsInstance(result, FitnessResult)
        self.assertIsInstance(result.score, float)


class TestAsEvalFunc(unittest.TestCase):
    """Test the as_eval_func adapter."""

    def test_adapts_to_standard_contract(self):
        """as_eval_func returns (linkage, dims, pos) -> float callable."""
        fitness = DistanceFitness(duration=0.5, n_legs=1)
        eval_fn = as_eval_func(fitness)
        walker = _make_simple_walker()
        dims = walker.get_num_constraints()
        pos = walker.get_coords()
        score = eval_fn(walker, dims, pos)
        self.assertIsInstance(score, float)

    def test_adapts_with_config(self):
        """as_eval_func passes config through to fitness."""
        cfg = WorldConfig(gravity=(0, -5.0))
        fitness = DistanceFitness(duration=0.5, n_legs=1)
        eval_fn = as_eval_func(fitness, config=cfg)
        walker = _make_simple_walker()
        dims = walker.get_num_constraints()
        pos = walker.get_coords()
        score = eval_fn(walker, dims, pos)
        self.assertIsInstance(score, float)

    def test_stride_adapter(self):
        """as_eval_func works with StrideFitness (kinematic)."""
        fitness = StrideFitness(lap_points=6, foot_index=-1)
        eval_fn = as_eval_func(fitness)
        walker = _make_simple_walker()
        dims = walker.get_num_constraints()
        pos = walker.get_coords()
        score = eval_fn(walker, dims, pos)
        self.assertIsInstance(score, float)

    def test_walker_factory_mode_ignores_linkage(self):
        """walker_factory-based adapter builds fresh walkers per call."""
        fitness = DistanceFitness(duration=0.5, n_legs=1)
        eval_fn = as_eval_func(fitness, walker_factory=_make_simple_walker)
        walker = _make_simple_walker()
        dims = walker.get_num_constraints()
        # linkage and pos are ignored in walker_factory mode.
        score = eval_fn(None, dims, ())
        self.assertIsInstance(score, float)

    def test_negate_flips_sign(self):
        """negate=True returns -score for pylinkage's minimization optimizers."""
        def always_two(topology, dimensions, config=None):
            return FitnessResult(score=2.0, valid=True)

        eval_fn = as_eval_func(
            always_two,
            walker_factory=_make_simple_walker,
            negate=True,
        )
        walker = _make_simple_walker()
        score = eval_fn(None, walker.get_num_constraints(), ())
        self.assertAlmostEqual(score, -2.0)

    def test_pylinkage_moo_accepts_walking_fitness(self):
        """Adapted walking fitness feeds pylinkage's multi_objective_optimization."""
        from pylinkage.optimization import multi_objective_optimization

        fitness = DistanceFitness(duration=0.5, n_legs=1)
        eval_fn = as_eval_func(
            fitness, walker_factory=_make_simple_walker, negate=True,
        )
        walker = _make_simple_walker()
        ensemble = multi_objective_optimization(
            objectives=[eval_fn, eval_fn],
            linkage=walker,
            bounds=([0.5] * 5, [3.0] * 5),
            objective_names=["a", "b"],
            n_generations=2, pop_size=4, seed=42, verbose=False,
        )
        self.assertGreater(ensemble.n_members, 0)
        self.assertIn("a", ensemble.scores)
        self.assertIn("b", ensemble.scores)


class TestAsGaFitness(unittest.TestCase):
    """Test the as_ga_fitness adapter."""

    def test_adapts_to_ga_contract(self):
        """as_ga_fitness returns (dna) -> (score, positions) callable."""
        fitness = DistanceFitness(duration=0.5, n_legs=1)
        ga_fn = as_ga_fitness(fitness, walker_factory=_make_simple_walker)
        walker = _make_simple_walker()
        dna = [0, walker.get_num_constraints(), walker.get_coords()]
        score, positions = ga_fn(dna)
        self.assertIsInstance(score, float)
        self.assertIsInstance(positions, list)

    def test_minimize_negates_score(self):
        """as_ga_fitness with minimize=True negates the score."""
        # Use a custom fitness that always returns score=1.0
        def always_one(topology, dimensions, config=None):
            return FitnessResult(score=1.0)

        ga_fn = as_ga_fitness(
            always_one, walker_factory=_make_fourbar_walker, minimize=True,
        )
        walker = _make_fourbar_walker()
        dna = [0, walker.get_num_constraints(), walker.get_coords()]
        score, _ = ga_fn(dna)
        self.assertAlmostEqual(score, -1.0)


class TestExports(unittest.TestCase):
    """Verify fitness protocol types are accessible from leggedsnake."""

    def test_fitness_result_exported(self):
        self.assertTrue(hasattr(ls, 'FitnessResult'))

    def test_dynamic_fitness_exported(self):
        self.assertTrue(hasattr(ls, 'DynamicFitness'))

    def test_distance_fitness_exported(self):
        self.assertTrue(hasattr(ls, 'DistanceFitness'))

    def test_efficiency_fitness_exported(self):
        self.assertTrue(hasattr(ls, 'EfficiencyFitness'))

    def test_stride_fitness_exported(self):
        self.assertTrue(hasattr(ls, 'StrideFitness'))

    def test_as_eval_func_exported(self):
        self.assertTrue(hasattr(ls, 'as_eval_func'))

    def test_as_ga_fitness_exported(self):
        self.assertTrue(hasattr(ls, 'as_ga_fitness'))

    def test_chain_walking_optimizers_exported(self):
        self.assertTrue(hasattr(ls, 'chain_walking_optimizers'))


class TestChainWalkingOptimizers(unittest.TestCase):
    """Test the walking-fitness wrapper around chain_optimizers."""

    def test_runs_two_stage_pipeline(self):
        """Chaining trials_and_errors + minimize_linkage produces a result."""
        from pylinkage.optimization import (
            minimize_linkage,
            trials_and_errors_optimization,
        )
        from leggedsnake import chain_walking_optimizers

        # Custom fitness: maximize sum of crank distance constraints.
        def linear_fitness(topology, dimensions, config=None):
            score = sum(dimensions.edge_distances.values())
            return FitnessResult(score=score, valid=True)

        walker = _make_fourbar_walker()
        result = chain_walking_optimizers(
            linear_fitness,
            walker,
            stages=[
                (trials_and_errors_optimization, {
                    "n_results": 3,
                    "divisions": 3,
                    "bounds": ([0.5, 1.0, 0.5], [2.0, 3.0, 2.0]),
                }),
                (minimize_linkage, {
                    "method": "Nelder-Mead",
                    "bounds": ([0.5, 1.0, 0.5], [2.0, 3.0, 2.0]),
                }),
            ],
            verbose=False,
        )
        self.assertGreater(result.n_members, 0)


if __name__ == "__main__":
    unittest.main()
