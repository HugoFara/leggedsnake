#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Integration tests for the end-to-end dynamic simulation pipeline.

Tests that exercise: build Walker → add legs → run physics → measure result.
Also covers walking objectives and GA+physics integration.
"""
import unittest
from math import tau, pi

from pylinkage.dimensions import Dimensions, DriverAngle
from pylinkage.hypergraph import HypergraphLinkage, Node, Edge, NodeRole

import leggedsnake as ls
from leggedsnake.walker import Walker
from leggedsnake.physics_engine import World, WorldConfig, TerrainConfig, DEFAULT_CONFIG


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


class TestSemanticEdgeIDs(unittest.TestCase):
    """Verify add_legs/add_opposite_leg work with non-numeric edge IDs."""

    def test_add_legs_semantic_ids(self):
        """add_legs works with edge IDs like 'A_B', 'frame_crank'."""
        walker = _make_fourbar_walker()
        walker.add_legs(2)
        self.assertEqual(len(walker.topology.nodes), 7)  # 1 ground + 3*2 non-ground

    def test_add_opposite_leg_semantic_ids(self):
        """add_opposite_leg works with semantic edge IDs."""
        walker = _make_simple_walker()
        initial = len(walker.topology.nodes)
        walker.add_opposite_leg(axis_x=1.0)
        self.assertGreater(len(walker.topology.nodes), initial)

    def test_add_opposite_then_add_legs(self):
        """Combining add_opposite_leg + add_legs doesn't crash."""
        walker = _make_simple_walker()
        walker.add_opposite_leg(axis_x=1.0)
        walker.add_legs(1)
        # Should have original + opposite + 2 more leg copies
        self.assertGreater(len(walker.topology.nodes), 10)

    def test_add_legs_phase_offsets(self):
        """Cloned drivers get correct phase offsets in DriverAngle."""
        walker = _make_fourbar_walker()
        walker.add_legs(2)
        # Original driver "B" at initial_angle=0
        # Copy 1 "B (1)" at initial_angle=tau/3
        # Copy 2 "B (2)" at initial_angle=2*tau/3
        da1 = walker.dimensions.driver_angles.get("B (1)")
        da2 = walker.dimensions.driver_angles.get("B (2)")
        self.assertIsNotNone(da1)
        self.assertIsNotNone(da2)
        self.assertAlmostEqual(da1.initial_angle, tau / 3, places=5)
        self.assertAlmostEqual(da2.initial_angle, 2 * tau / 3, places=5)

    def test_add_opposite_leg_phase_offset(self):
        """Opposite-leg driver gets pi phase offset."""
        walker = _make_fourbar_walker()
        walker.add_opposite_leg(axis_x=0.0)
        da_opp = walker.dimensions.driver_angles.get("B (opposite)")
        self.assertIsNotNone(da_opp)
        self.assertAlmostEqual(da_opp.initial_angle, pi, places=5)


class TestWalkerPhysicsPipeline(unittest.TestCase):
    """Test Walker → World → physics simulation end-to-end."""

    def test_walker_to_world(self):
        """Walker can be added to World and stepped."""
        walker = _make_simple_walker()
        world = World()
        world.add_linkage(walker)
        self.assertEqual(len(world.linkages), 1)
        for _ in range(10):
            world.update()

    def test_walker_with_legs_in_physics(self):
        """Walker with add_legs can run physics without errors."""
        walker = _make_simple_walker()
        walker.add_legs(1)
        world = World()
        world.add_linkage(walker)
        for _ in range(50):
            world.update()

    def test_walker_with_opposite_leg_in_physics(self):
        """Walker with add_opposite_leg can run physics without errors."""
        walker = _make_simple_walker()
        walker.add_opposite_leg(axis_x=1.0)
        world = World()
        world.add_linkage(walker)
        for _ in range(50):
            world.update()

    def test_physics_returns_energy(self):
        """After motors enable, update() returns (efficiency, energy)."""
        walker = _make_simple_walker()
        world = World()
        world.add_linkage(walker)
        # Run enough steps for the linkage to settle and motors to enable
        results = []
        for _ in range(200):
            r = world.update()
            if r is not None:
                results.append(r)
        # Should have gotten some non-None results
        self.assertGreater(len(results), 0)

    def test_multi_motor_energy_accounting(self):
        """Walker with multiple drivers sums power from all motors."""
        hg = HypergraphLinkage(name="multi_dof")
        hg.add_node(Node("G1", role=NodeRole.GROUND))
        hg.add_node(Node("G2", role=NodeRole.GROUND))
        hg.add_node(Node("D1", role=NodeRole.DRIVER))
        hg.add_node(Node("D2", role=NodeRole.DRIVER))
        hg.add_node(Node("P", role=NodeRole.DRIVEN))
        hg.add_edge(Edge("G1_D1", "G1", "D1"))
        hg.add_edge(Edge("G2_D2", "G2", "D2"))
        hg.add_edge(Edge("D1_P", "D1", "P"))
        hg.add_edge(Edge("D2_P", "D2", "P"))
        dims = Dimensions(
            node_positions={
                "G1": (0, 0), "G2": (3, 0),
                "D1": (1, 0), "D2": (2, 0), "P": (1.5, 2),
            },
            driver_angles={
                "D1": DriverAngle(angular_velocity=-tau / 12),
                "D2": DriverAngle(angular_velocity=-tau / 15),
            },
            edge_distances={
                "G1_D1": 1.0, "G2_D2": 1.0,
                "D1_P": 2.24, "D2_P": 2.24,
            },
        )
        walker = Walker(hg, dims, motor_rates={"D1": -3.0, "D2": -5.0})
        world = World()
        world.add_linkage(walker)
        # Verify both motors exist
        dl = world.linkages[0]
        self.assertEqual(len(dl.physics_mapping.motors), 2)
        # Run physics
        for _ in range(50):
            world.update()


class TestWalkingObjectives(unittest.TestCase):
    """Test walking objective functions with real physics."""

    def test_stride_length_objective(self):
        """stride_length_objective returns a float score."""
        walker = _make_simple_walker()
        obj = ls.stride_length_objective(lap_points=6, foot_index=-1)
        dims = walker.get_num_constraints()
        pos = walker.get_coords()
        score = obj(walker, dims, pos)
        self.assertIsInstance(score, float)

    def test_total_distance_objective(self):
        """total_distance_objective runs physics and returns distance."""
        walker = _make_simple_walker()
        obj = ls.total_distance_objective(duration=1.0, n_legs=1)
        dims = walker.get_num_constraints()
        pos = walker.get_coords()
        score = obj(walker, dims, pos)
        self.assertIsInstance(score, float)

    def test_energy_efficiency_objective(self):
        """energy_efficiency_objective runs physics and returns efficiency."""
        walker = _make_simple_walker()
        obj = ls.energy_efficiency_objective(duration=1.0, n_legs=1, min_distance=0.0)
        dims = walker.get_num_constraints()
        pos = walker.get_coords()
        score = obj(walker, dims, pos)
        self.assertIsInstance(score, float)


class TestGAPhysicsPipeline(unittest.TestCase):
    """Test that GA optimization works with Walker + physics."""

    def test_ga_with_walker(self):
        """GeneticOptimization can optimize a Walker with physics fitness."""
        walker = _make_fourbar_walker()
        dims = walker.get_num_constraints()
        coords = walker.get_coords()
        dna = [0, dims, coords]

        def fitness(dna):
            w = _make_fourbar_walker()
            w.set_num_constraints(dna[1])
            w.set_coords(dna[2])
            # Simple kinematic score — no physics needed for smoke test
            try:
                positions = list(w.step(iterations=5))
                return len(positions), dna[2]
            except Exception:
                return 0, dna[2]

        optimizer = ls.GeneticOptimization(
            dna=dna,
            fitness=fitness,
            max_pop=4,
        )
        results = optimizer.run(iters=2, processes=1)
        self.assertIsInstance(results, list)
        self.assertGreater(len(results), 0)


class TestWorldConfig(unittest.TestCase):
    """Test WorldConfig dataclass and its integration with World."""

    def test_default_config(self):
        """DEFAULT_CONFIG has sensible defaults."""
        cfg = DEFAULT_CONFIG
        self.assertAlmostEqual(cfg.gravity[1], -9.80665)
        self.assertAlmostEqual(cfg.physics_period, 0.02)
        self.assertGreater(cfg.torque, 0)

    def test_world_uses_default_config(self):
        """World without config uses DEFAULT_CONFIG."""
        world = World()
        self.assertIs(world.config, DEFAULT_CONFIG)

    def test_world_accepts_custom_config(self):
        """World stores the custom config passed at construction."""
        cfg = WorldConfig(gravity=(0, -5.0), physics_period=0.01)
        world = World(config=cfg)
        self.assertIs(world.config, cfg)
        self.assertEqual(world.space.gravity[1], -5.0)

    def test_custom_gravity_affects_physics(self):
        """A world with weaker gravity should let bodies fall slower."""
        walker = _make_simple_walker()

        # Normal gravity
        world_normal = World(config=WorldConfig(gravity=(0, -9.81)))
        world_normal.add_linkage(walker)
        for _ in range(20):
            world_normal.update()
        y_normal = world_normal.linkages[0].body.position.y

        # Weak gravity
        walker2 = _make_simple_walker()
        world_weak = World(config=WorldConfig(gravity=(0, -1.0)))
        world_weak.add_linkage(walker2)
        for _ in range(20):
            world_weak.update()
        y_weak = world_weak.linkages[0].body.position.y

        # Weaker gravity → body should be higher (less fallen)
        self.assertGreater(y_weak, y_normal)

    def test_custom_physics_period(self):
        """World.update() uses config.physics_period as default dt."""
        cfg = WorldConfig(physics_period=0.05)
        world = World(config=cfg)
        walker = _make_simple_walker()
        world.add_linkage(walker)
        # Should not raise — just verify it runs with custom period
        for _ in range(10):
            world.update()

    def test_terrain_config(self):
        """TerrainConfig fields are accessible."""
        terrain = TerrainConfig(slope=0.1, max_step=1.0, friction=0.8)
        self.assertAlmostEqual(terrain.slope, 0.1)
        self.assertAlmostEqual(terrain.max_step, 1.0)
        cfg = WorldConfig(terrain=terrain)
        self.assertAlmostEqual(cfg.terrain.slope, 0.1)

    def test_objective_with_custom_config(self):
        """Walking objectives accept WorldConfig."""
        walker = _make_simple_walker()
        cfg = WorldConfig(gravity=(0, -5.0), physics_period=0.05)
        obj = ls.total_distance_objective(duration=0.5, n_legs=1, config=cfg)
        dims = walker.get_num_constraints()
        pos = walker.get_coords()
        score = obj(walker, dims, pos)
        self.assertIsInstance(score, float)

    def test_config_exported(self):
        """WorldConfig and TerrainConfig are accessible from leggedsnake."""
        self.assertTrue(hasattr(ls, 'WorldConfig'))
        self.assertTrue(hasattr(ls, 'TerrainConfig'))
        self.assertTrue(hasattr(ls, 'DEFAULT_CONFIG'))


if __name__ == "__main__":
    unittest.main()
