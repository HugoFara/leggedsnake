#!/usr/bin/env python3
"""Tests for foot-ground contact force sampling (Phase 8.3).

Covers:
- ``sample_ground_reaction_force`` — raw arbiter sum on a linkage.
- ``StabilitySnapshot`` / ``StabilityTimeSeries`` force fields.
- Integration: ``StabilityFitness`` now surfaces ``peak_ground_reaction_force``
  in ``FitnessResult.metrics``.
"""
import unittest
from math import tau

import pymunk as pm

from pylinkage.dimensions import Dimensions, DriverAngle
from pylinkage.hypergraph import HypergraphLinkage, Node, Edge, NodeRole

from leggedsnake.dynamic_linkage import DynamicLinkage
from leggedsnake.fitness import CompositeFitness, StabilityFitness
from leggedsnake.physics_engine import World, WorldConfig, TerrainConfig
from leggedsnake.stability import (
    StabilitySnapshot,
    StabilityTimeSeries,
    compute_stability_snapshot,
    sample_ground_reaction_force,
)
from leggedsnake.walker import Walker


def _make_simple_walker() -> Walker:
    """Same 5-node walker used elsewhere in the test suite."""
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


class TestSampleGRF(unittest.TestCase):
    """Low-level contact impulse sampling."""

    def test_zero_dt_returns_zero(self):
        """Guard against division by zero — dt <= 0 → (0, 0)."""
        space = pm.Space()
        hg = HypergraphLinkage(name="x")
        hg.add_node(Node("g", role=NodeRole.GROUND))
        hg.add_node(Node("c", role=NodeRole.DRIVER))
        hg.add_node(Node("d", role=NodeRole.DRIVEN))
        hg.add_edge(Edge("g_c", "g", "c"))
        hg.add_edge(Edge("g_d", "g", "d"))
        hg.add_edge(Edge("c_d", "c", "d"))
        dims = Dimensions(
            node_positions={"g": (0, 0), "c": (1, 0), "d": (0, 2)},
            driver_angles={"c": DriverAngle(angular_velocity=-tau / 12)},
            edge_distances={"g_c": 1.0, "g_d": 2.0, "c_d": 1.5},
        )
        dl = DynamicLinkage(hg, dims, space, name="x")
        total, peak = sample_ground_reaction_force(dl, space.static_body, 0.0)
        self.assertEqual(total, 0.0)
        self.assertEqual(peak, 0.0)

    def test_no_contact_yields_zero(self):
        """Before any stepping the arbiter list is empty."""
        walker = _make_simple_walker()
        world = World(config=WorldConfig(), road_y=-10.0)
        world.add_linkage(walker)
        dl = world.linkages[0]
        total, peak = sample_ground_reaction_force(
            dl, world.space.static_body, 0.02,
        )
        self.assertEqual(total, 0.0)
        self.assertEqual(peak, 0.0)

    def test_contact_generates_force(self):
        """Walker landing on the ground produces a positive GRF."""
        walker = _make_simple_walker()
        # Place the road right under the walker so it lands quickly.
        world = World(config=WorldConfig(), road_y=-0.1)
        world.add_linkage(walker)
        dl = world.linkages[0]

        # Step long enough for the falling walker to hit the ground.
        peak_over_run = 0.0
        total_seen = 0.0
        for _ in range(30):
            world.update()
            total, peak = sample_ground_reaction_force(
                dl, world.space.static_body, world.config.physics_period,
            )
            total_seen = max(total_seen, total)
            peak_over_run = max(peak_over_run, peak)

        # At least one step should show a nonzero force once contact happens.
        self.assertGreater(peak_over_run, 0.0)
        self.assertGreaterEqual(total_seen, peak_over_run)


class TestStabilitySnapshotGRF(unittest.TestCase):
    """StabilitySnapshot force fields + snapshot helper wiring."""

    def test_default_fields_zero(self):
        snap = StabilitySnapshot(
            time=0.0, com=(0.0, 0.0), com_velocity=(0.0, 0.0),
            zmp_x=0.0, support_polygon=[], tip_over_margin=0.0,
            body_angle=0.0,
        )
        self.assertEqual(snap.ground_reaction_force, 0.0)
        self.assertEqual(snap.peak_contact_force, 0.0)

    def test_snapshot_without_static_body_leaves_zero(self):
        """Omitting static_body keeps the force fields zero."""
        walker = _make_simple_walker()
        world = World(config=WorldConfig(), road_y=-0.1)
        world.add_linkage(walker)
        dl = world.linkages[0]
        world.update()

        snap = compute_stability_snapshot(
            dl, None, time=0.0, dt=0.02, gravity=9.81,
        )
        self.assertEqual(snap.ground_reaction_force, 0.0)
        self.assertEqual(snap.peak_contact_force, 0.0)

    def test_snapshot_with_static_body_populates(self):
        """Providing static_body feeds GRF into the snapshot."""
        walker = _make_simple_walker()
        world = World(config=WorldConfig(), road_y=-0.1)
        world.add_linkage(walker)
        dl = world.linkages[0]

        # Step until we see contact.
        saw_force = False
        for _ in range(30):
            world.update()
            snap = compute_stability_snapshot(
                dl, None,
                time=0.0, dt=world.config.physics_period, gravity=9.81,
                static_body=world.space.static_body,
            )
            if snap.ground_reaction_force > 0:
                saw_force = True
                self.assertGreaterEqual(
                    snap.ground_reaction_force, snap.peak_contact_force,
                )
                break

        self.assertTrue(saw_force, "expected walker to contact ground")


class TestStabilityTimeSeriesGRF(unittest.TestCase):
    """Aggregate GRF properties on StabilityTimeSeries."""

    def _snap(self, grf: float, peak: float = 0.0) -> StabilitySnapshot:
        return StabilitySnapshot(
            time=0.0, com=(0.0, 0.0), com_velocity=(0.0, 0.0),
            zmp_x=0.0, support_polygon=[], tip_over_margin=0.0,
            body_angle=0.0,
            ground_reaction_force=grf,
            peak_contact_force=peak,
        )

    def test_empty_series(self):
        ts = StabilityTimeSeries()
        self.assertEqual(ts.peak_ground_reaction_force, 0.0)
        self.assertEqual(ts.mean_ground_reaction_force, 0.0)
        self.assertEqual(ts.peak_contact_force, 0.0)

    def test_peak_is_max(self):
        ts = StabilityTimeSeries(snapshots=[
            self._snap(10.0, 4.0),
            self._snap(25.0, 8.0),
            self._snap(15.0, 7.0),
        ])
        self.assertAlmostEqual(ts.peak_ground_reaction_force, 25.0)
        self.assertAlmostEqual(ts.peak_contact_force, 8.0)

    def test_mean_includes_zero_steps(self):
        ts = StabilityTimeSeries(snapshots=[
            self._snap(0.0), self._snap(30.0), self._snap(0.0),
        ])
        self.assertAlmostEqual(ts.mean_ground_reaction_force, 10.0)

    def test_summary_contains_grf_keys(self):
        ts = StabilityTimeSeries(snapshots=[self._snap(20.0, 5.0)])
        summary = ts.summary_metrics()
        self.assertIn("peak_ground_reaction_force", summary)
        self.assertIn("mean_ground_reaction_force", summary)
        self.assertIn("peak_contact_force", summary)
        self.assertAlmostEqual(summary["peak_ground_reaction_force"], 20.0)


class TestFitnessIntegration(unittest.TestCase):
    """Fitness classes surface GRF via StabilityTimeSeries.summary_metrics."""

    def test_stability_fitness_has_grf_metrics(self):
        walker = _make_simple_walker()
        fitness = StabilityFitness(duration=0.5, n_legs=1, min_distance=0.0)
        result = fitness(walker.topology, walker.dimensions)
        self.assertIn("peak_ground_reaction_force", result.metrics)
        self.assertIn("mean_ground_reaction_force", result.metrics)
        self.assertIn("peak_contact_force", result.metrics)

    def test_composite_stability_objective_has_grf_metrics(self):
        walker = _make_simple_walker()
        fitness = CompositeFitness(
            duration=0.5, n_legs=1, objectives=("distance", "stability"),
        )
        result = fitness(walker.topology, walker.dimensions)
        self.assertIn("peak_ground_reaction_force", result.metrics)

    def test_deterministic_flat_terrain_gives_positive_grf(self):
        """Walker on flat terrain eventually develops measurable GRF."""
        walker = _make_simple_walker()
        cfg = WorldConfig(terrain=TerrainConfig(
            slope=0.0, noise=0.0, step_freq=0.0, seed=1,
        ))
        # StabilityFitness runs its own World; verify the recorded series
        # has at least one nonzero GRF step.
        fitness = StabilityFitness(
            duration=2.0, n_legs=1, min_distance=0.0, record_loci=False,
        )
        result = fitness(walker.topology, walker.dimensions, config=cfg)
        # With the short sim the walker should contact the ground at least
        # briefly, making peak > 0.
        self.assertGreater(result.metrics["peak_ground_reaction_force"], 0.0)


if __name__ == "__main__":
    unittest.main()
