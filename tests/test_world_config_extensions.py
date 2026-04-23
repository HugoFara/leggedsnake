#!/usr/bin/env python3
"""Tests for WorldConfig payload / wind / drag extensions."""
import unittest
from math import tau

from pylinkage.dimensions import Dimensions, DriverAngle
from pylinkage.hypergraph import HypergraphLinkage, Node, Edge, NodeRole

from leggedsnake.physicsengine import World, WorldConfig
from leggedsnake.walker import Walker


def _make_simple_walker() -> Walker:
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


class TestPayloadOffset(unittest.TestCase):
    """payload_offset shifts the chassis centre of gravity."""

    def test_default_leaves_cog_at_origin(self):
        walker = _make_simple_walker()
        world = World(config=WorldConfig(), road_y=-10.0)
        world.add_linkage(walker)
        cog = world.linkages[0].body.center_of_gravity
        self.assertAlmostEqual(cog.x, 0.0)
        self.assertAlmostEqual(cog.y, 0.0)

    def test_offset_sets_center_of_gravity(self):
        walker = _make_simple_walker()
        cfg = WorldConfig(payload_offset=(0.5, -0.25))
        world = World(config=cfg, road_y=-10.0)
        world.add_linkage(walker)
        cog = world.linkages[0].body.center_of_gravity
        self.assertAlmostEqual(cog.x, 0.5)
        self.assertAlmostEqual(cog.y, -0.25)

    def test_negative_offset(self):
        walker = _make_simple_walker()
        cfg = WorldConfig(payload_offset=(-1.0, 0.3))
        world = World(config=cfg, road_y=-10.0)
        world.add_linkage(walker)
        cog = world.linkages[0].body.center_of_gravity
        self.assertAlmostEqual(cog.x, -1.0)
        self.assertAlmostEqual(cog.y, 0.3)


class TestWindForce(unittest.TestCase):
    """wind_force applies a constant force to the chassis each step."""

    def test_no_wind_by_default(self):
        walker = _make_simple_walker()
        world = World(config=WorldConfig(), road_y=-1000.0)
        world.add_linkage(walker)
        body = world.linkages[0].body
        for _ in range(10):
            world.update()
        # No horizontal force applied; chassis shouldn't drift sideways
        # from a standing start (constraint chatter is small).
        self.assertAlmostEqual(body.velocity.x, 0.0, places=1)

    def test_positive_wind_accelerates_rightward(self):
        """Comparing wind vs no-wind isolates the effect from linkage dynamics."""
        walker1 = _make_simple_walker()
        walker2 = _make_simple_walker()
        no_wind = World(config=WorldConfig(), road_y=-1000.0)
        windy = World(
            config=WorldConfig(wind_force=(80.0, 0.0)), road_y=-1000.0,
        )
        no_wind.add_linkage(walker1)
        windy.add_linkage(walker2)

        for _ in range(20):
            no_wind.update()
            windy.update()

        # The windy chassis should have measurably more +x velocity.
        self.assertGreater(
            windy.linkages[0].body.velocity.x,
            no_wind.linkages[0].body.velocity.x + 0.5,
        )

    def test_vertical_wind(self):
        walker1 = _make_simple_walker()
        walker2 = _make_simple_walker()
        base = World(config=WorldConfig(), road_y=-1000.0)
        updraft = World(
            config=WorldConfig(wind_force=(0.0, 150.0)), road_y=-1000.0,
        )
        base.add_linkage(walker1)
        updraft.add_linkage(walker2)

        for _ in range(20):
            base.update()
            updraft.update()

        # Updraft should make the chassis fall less (or rise).
        self.assertGreater(
            updraft.linkages[0].body.velocity.y,
            base.linkages[0].body.velocity.y,
        )


class TestDragCoefficient(unittest.TestCase):
    """drag_coefficient applies velocity-proportional resistance to the chassis."""

    def test_no_drag_by_default(self):
        walker = _make_simple_walker()
        world = World(config=WorldConfig(), road_y=-1000.0)
        world.add_linkage(walker)
        body = world.linkages[0].body
        body.velocity = (5.0, 0.0)
        v0 = body.velocity.x
        world.update()
        # First step: no drag force → small velocity change is only from
        # pivot constraint corrections, not forced deceleration.
        self.assertAlmostEqual(body.velocity.x, v0, places=0)

    def test_drag_decelerates_moving_chassis(self):
        """Drag > 0 produces more deceleration than drag = 0."""
        walker1 = _make_simple_walker()
        walker2 = _make_simple_walker()
        no_drag = World(config=WorldConfig(), road_y=-1000.0)
        with_drag = World(
            config=WorldConfig(drag_coefficient=50.0), road_y=-1000.0,
        )
        no_drag.add_linkage(walker1)
        with_drag.add_linkage(walker2)
        no_drag.linkages[0].body.velocity = (5.0, 0.0)
        with_drag.linkages[0].body.velocity = (5.0, 0.0)

        for _ in range(20):
            no_drag.update()
            with_drag.update()

        self.assertLess(
            with_drag.linkages[0].body.velocity.x,
            no_drag.linkages[0].body.velocity.x - 0.2,
        )

    def test_drag_does_not_reverse_velocity(self):
        """Drag force shouldn't flip the sign of velocity in a single step."""
        walker = _make_simple_walker()
        cfg = WorldConfig(drag_coefficient=50.0)
        world = World(config=cfg, road_y=-1000.0)
        world.add_linkage(walker)
        body = world.linkages[0].body
        body.velocity = (2.0, 0.0)
        for _ in range(5):
            world.update()
        self.assertGreater(body.velocity.x, 0.0)


if __name__ == "__main__":
    unittest.main()
