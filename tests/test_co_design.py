#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Tests for Phase 6: Dynamic Co-Design.

Covers Walker factory methods (from_catalog, from_hierarchy, from_synthesis),
co_optimize_objective adapter, and end-to-end pipeline smoke tests.
"""
import unittest
from math import tau

from pylinkage.dimensions import Dimensions, DriverAngle
from pylinkage.hypergraph import (
    HypergraphLinkage,
    Node,
    Edge,
    Hyperedge,
    NodeRole,
    HierarchicalLinkage,
    ComponentInstance,
    Connection,
)
from pylinkage.topology import load_catalog

import leggedsnake as ls
from leggedsnake.fitness import (
    DynamicFitness,
    DistanceFitness,
    EfficiencyFitness,
    FitnessResult,
    StrideFitness,
    co_optimize_objective,
)
from leggedsnake.co_design import (
    WalkingDesignSpec,
    WalkingDesignResult,
    optimize_walking_mechanism,
)
from leggedsnake.physicsengine import WorldConfig
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


def _make_fourbar_dims() -> Dimensions:
    """Dimensions compatible with a four-bar catalog entry."""
    return Dimensions(
        node_positions={
            "ground_0": (0, 0),
            "ground_1": (3, 0),
            "driver_0": (1, 0),
            "driven_0": (2, 2),
        },
        driver_angles={
            "driver_0": DriverAngle(angular_velocity=-tau / 12),
        },
        edge_distances={
            "ground_0-driver_0": 1.0,
            "ground_1-driven_0": 2.24,
            "driver_0-driven_0": 2.24,
        },
    )


class TestWalkerFromCatalog(unittest.TestCase):
    """Test Walker.from_catalog() factory."""

    def test_creates_walker(self):
        """from_catalog returns a Walker with the catalog topology."""
        catalog = load_catalog()
        entry = catalog.get("four-bar")
        self.assertIsNotNone(entry)
        graph = entry.to_graph()
        # Build dimensions that match the topology's nodes/edges
        dims = Dimensions(
            node_positions={nid: (float(i), 0.0) for i, nid in enumerate(graph.nodes)},
            driver_angles={
                n.id: DriverAngle(angular_velocity=-tau / 12)
                for n in graph.driver_nodes()
            },
            edge_distances={eid: 1.0 for eid in graph.edges},
        )
        walker = Walker.from_catalog(entry, dims)
        self.assertIsInstance(walker, Walker)
        self.assertEqual(walker.name, entry.name)
        self.assertEqual(len(walker.topology.nodes), len(graph.nodes))

    def test_custom_motor_rates(self):
        """from_catalog passes motor_rates to the Walker."""
        catalog = load_catalog()
        entry = catalog.get("four-bar")
        graph = entry.to_graph()
        dims = Dimensions(
            node_positions={nid: (float(i), 0.0) for i, nid in enumerate(graph.nodes)},
            driver_angles={
                n.id: DriverAngle(angular_velocity=-tau / 12)
                for n in graph.driver_nodes()
            },
            edge_distances={eid: 1.0 for eid in graph.edges},
        )
        walker = Walker.from_catalog(entry, dims, motor_rates=-6.0)
        self.assertEqual(walker.motor_rates, -6.0)


class TestWalkerFromHierarchy(unittest.TestCase):
    """Test Walker.from_hierarchy() factory."""

    def test_creates_walker_from_hierarchy(self):
        """from_hierarchy flattens and returns a Walker."""
        # Build a simple component
        leg = HypergraphLinkage(name="leg")
        leg.add_node(Node("hip", role=NodeRole.GROUND))
        leg.add_node(Node("crank", role=NodeRole.DRIVER))
        leg.add_node(Node("foot", role=NodeRole.DRIVEN))
        leg.add_edge(Edge("hip_crank", "hip", "crank"))
        leg.add_edge(Edge("crank_foot", "crank", "foot"))
        leg.add_edge(Edge("hip_foot", "hip", "foot"))

        frame = HypergraphLinkage(name="frame")
        frame.add_node(Node("mount", role=NodeRole.GROUND))

        hl = HierarchicalLinkage(name="two_leg_walker")
        hl.add_instance(ComponentInstance("frame", frame, ports={"m0": "mount", "m1": "mount"}))
        hl.add_instance(ComponentInstance("leg_0", leg, ports={"hip": "hip"}))
        hl.add_instance(ComponentInstance("leg_1", leg, ports={"hip": "hip"}))
        hl.add_connection(Connection("frame", "m0", "leg_0", "hip"))
        hl.add_connection(Connection("frame", "m1", "leg_1", "hip"))

        flat = hl.flatten()
        dims = Dimensions(
            node_positions={nid: (float(i), 0.0) for i, nid in enumerate(flat.nodes)},
            driver_angles={
                n.id: DriverAngle(angular_velocity=-tau / 12)
                for n in flat.driver_nodes()
            },
            edge_distances={eid: 1.0 for eid in flat.edges},
        )

        walker = Walker.from_hierarchy(hl, dims)
        self.assertIsInstance(walker, Walker)
        self.assertEqual(walker.name, "two_leg_walker")
        # Flattened: 1 shared mount + 2 cranks + 2 feet = 5 nodes
        self.assertEqual(len(walker.topology.nodes), 5)


class TestWalkerFromSynthesis(unittest.TestCase):
    """Test Walker.from_synthesis() factory."""

    def test_from_synthesis_with_linkage(self):
        """from_synthesis converts a solution with .linkage attribute."""
        # Create a mock solution object with a .linkage attribute
        from pylinkage.joints.joint import Static
        from pylinkage.joints.crank import Crank
        from pylinkage.joints.revolute import Revolute
        from pylinkage.linkage import Linkage

        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            base = Static(0, 0, name="base")
            crank = Crank(1, 0, joint0=base, distance=1, angle=-tau / 12, name="crank")
            follower = Revolute(0, 2, joint0=base, joint1=crank, distance0=2, distance1=1.5, name="follower")
            linkage = Linkage(joints=[base, crank, follower], name="test")

        class MockSolution:
            def __init__(self, lk):
                self.linkage = lk

        sol = MockSolution(linkage)
        walker = Walker.from_synthesis(sol, motor_rates=-5.0)
        self.assertIsInstance(walker, Walker)
        self.assertEqual(walker.motor_rates, -5.0)

    def test_from_synthesis_with_n_legs(self):
        """from_synthesis adds legs when n_legs > 1."""
        from pylinkage.joints.joint import Static
        from pylinkage.joints.crank import Crank
        from pylinkage.joints.revolute import Revolute
        from pylinkage.linkage import Linkage

        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            base = Static(0, 0, name="base")
            crank = Crank(1, 0, joint0=base, distance=1, angle=-tau / 12, name="crank")
            follower = Revolute(0, 2, joint0=base, joint1=crank, distance0=2, distance1=1.5, name="follower")
            linkage = Linkage(joints=[base, crank, follower], name="test")

        class MockSolution:
            def __init__(self, lk):
                self.linkage = lk

        sol = MockSolution(linkage)
        walker = Walker.from_synthesis(sol, n_legs=2)
        # Should have more nodes than base 3 (ground + crank + follower)
        self.assertGreater(len(walker.topology.nodes), 3)

    def test_from_synthesis_none_linkage_raises(self):
        """from_synthesis raises ValueError when linkage is None."""
        class MockSolution:
            linkage = None

        with self.assertRaises(ValueError):
            Walker.from_synthesis(MockSolution())

    def test_from_synthesis_missing_linkage_raises(self):
        """from_synthesis raises ValueError when no .linkage attribute."""
        class MockSolution:
            pass

        with self.assertRaises(ValueError):
            Walker.from_synthesis(MockSolution())


class TestCoOptimizeObjective(unittest.TestCase):
    """Test co_optimize_objective() adapter."""

    def test_returns_callable(self):
        """co_optimize_objective returns a callable."""
        fitness = DistanceFitness(duration=0.5, n_legs=1)
        obj = co_optimize_objective(fitness)
        self.assertTrue(callable(obj))

    def test_negates_score(self):
        """Returned objective negates the DynamicFitness score."""
        # Custom fitness that always returns score=5.0
        def constant_fitness(topology, dimensions, config=None):
            return FitnessResult(score=5.0)

        obj = co_optimize_objective(constant_fitness)

        # Build a real Linkage for the objective to convert
        from pylinkage.joints.joint import Static
        from pylinkage.joints.crank import Crank
        from pylinkage.joints.revolute import Revolute
        from pylinkage.linkage import Linkage

        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            base = Static(0, 0, name="base")
            crank = Crank(1, 0, joint0=base, distance=1, angle=-tau / 12, name="crank")
            follower = Revolute(0, 2, joint0=base, joint1=crank, distance0=2, distance1=1.5, name="follower")
            linkage = Linkage(joints=[base, crank, follower], name="test")

        result = obj(linkage)
        self.assertAlmostEqual(result, -5.0)

    def test_invalid_linkage_returns_inf(self):
        """Objective returns inf for objects that can't be converted."""
        fitness = DistanceFitness(duration=0.5, n_legs=1)
        obj = co_optimize_objective(fitness)
        result = obj("not a linkage")
        self.assertEqual(result, float("inf"))

    def test_prefilter_rejects(self):
        """When pre-filter returns score=0, objective returns inf."""
        def always_zero(topology, dimensions, config=None):
            return FitnessResult(score=0.0)

        def always_ten(topology, dimensions, config=None):
            return FitnessResult(score=10.0)

        obj = co_optimize_objective(
            always_ten,
            kinematic_prefilter=always_zero,
        )

        from pylinkage.joints.joint import Static
        from pylinkage.joints.crank import Crank
        from pylinkage.joints.revolute import Revolute
        from pylinkage.linkage import Linkage

        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            base = Static(0, 0, name="base")
            crank = Crank(1, 0, joint0=base, distance=1, angle=-tau / 12, name="crank")
            follower = Revolute(0, 2, joint0=base, joint1=crank, distance0=2, distance1=1.5, name="follower")
            linkage = Linkage(joints=[base, crank, follower], name="test")

        result = obj(linkage)
        self.assertEqual(result, float("inf"))

    def test_prefilter_passes(self):
        """When pre-filter score > 0, full fitness runs."""
        def always_five(topology, dimensions, config=None):
            return FitnessResult(score=5.0)

        def always_ten(topology, dimensions, config=None):
            return FitnessResult(score=10.0)

        obj = co_optimize_objective(
            always_ten,
            kinematic_prefilter=always_five,
        )

        from pylinkage.joints.joint import Static
        from pylinkage.joints.crank import Crank
        from pylinkage.joints.revolute import Revolute
        from pylinkage.linkage import Linkage

        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            base = Static(0, 0, name="base")
            crank = Crank(1, 0, joint0=base, distance=1, angle=-tau / 12, name="crank")
            follower = Revolute(0, 2, joint0=base, joint1=crank, distance0=2, distance1=1.5, name="follower")
            linkage = Linkage(joints=[base, crank, follower], name="test")

        result = obj(linkage)
        self.assertAlmostEqual(result, -10.0)


class TestWalkingDesignSpec(unittest.TestCase):
    """Test WalkingDesignSpec dataclass."""

    def test_defaults(self):
        """WalkingDesignSpec has sensible defaults."""
        spec = WalkingDesignSpec()
        self.assertEqual(spec.objectives, [])
        self.assertEqual(spec.n_legs, 4)
        self.assertEqual(spec.max_links, 8)
        self.assertFalse(spec.use_warm_start)

    def test_with_objectives(self):
        """WalkingDesignSpec stores objectives."""
        fitness = DistanceFitness(duration=10.0, n_legs=2)
        spec = WalkingDesignSpec(
            objectives=[fitness],
            objective_names=["distance"],
            n_legs=2,
        )
        self.assertEqual(len(spec.objectives), 1)
        self.assertEqual(spec.n_legs, 2)


class TestOptimizeWalkingMechanismValidation(unittest.TestCase):
    """Test optimize_walking_mechanism validation."""

    def test_empty_objectives_raises(self):
        """optimize_walking_mechanism raises with no objectives."""
        spec = WalkingDesignSpec(objectives=[])
        with self.assertRaises(ValueError):
            optimize_walking_mechanism(spec)


class TestExports(unittest.TestCase):
    """Verify Phase 6 types are accessible from leggedsnake."""

    def test_co_optimize_objective_exported(self):
        self.assertTrue(hasattr(ls, 'co_optimize_objective'))

    def test_walking_design_spec_exported(self):
        self.assertTrue(hasattr(ls, 'WalkingDesignSpec'))

    def test_walking_design_result_exported(self):
        self.assertTrue(hasattr(ls, 'WalkingDesignResult'))

    def test_optimize_walking_mechanism_exported(self):
        self.assertTrue(hasattr(ls, 'optimize_walking_mechanism'))

    def test_walker_from_catalog(self):
        self.assertTrue(hasattr(ls.Walker, 'from_catalog'))

    def test_walker_from_hierarchy(self):
        self.assertTrue(hasattr(ls.Walker, 'from_hierarchy'))

    def test_walker_from_synthesis(self):
        self.assertTrue(hasattr(ls.Walker, 'from_synthesis'))


if __name__ == "__main__":
    unittest.main()
