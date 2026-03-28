#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Tests for the physicsengine module.
"""

import unittest
from math import pi
import pymunk as pm
from pylinkage import Static, Crank, Fixed, Pivot, Linkage

from leggedsnake.physicsengine import (
    World, set_space_constraints, recalc_linkage, linkage_bb, params
)
from leggedsnake.dynamiclinkage import DynamicLinkage


class TestParams(unittest.TestCase):
    """Test the params configuration dict."""

    def test_params_structure(self):
        """Test that params has expected keys."""
        self.assertIn("ground", params)
        self.assertIn("linkage", params)
        self.assertIn("physics", params)
        self.assertIn("simul", params)

    def test_ground_params(self):
        """Test ground parameters."""
        ground = params["ground"]
        self.assertIn("slope", ground)
        self.assertIn("max_step", ground)
        self.assertIn("step_freq", ground)
        self.assertIn("noise", ground)
        self.assertIn("section_len", ground)
        self.assertIn("friction", ground)

    def test_physics_params(self):
        """Test physics parameters."""
        physics = params["physics"]
        self.assertIn("gravity", physics)
        self.assertIn("max_force", physics)
        # Gravity should be a tuple with y pointing down
        self.assertEqual(len(physics["gravity"]), 2)
        self.assertLess(physics["gravity"][1], 0)


class TestSetSpaceConstraints(unittest.TestCase):
    """Test set_space_constraints function."""

    def test_empty_space(self):
        """Test with empty space."""
        space = pm.Space()
        # Should not raise any exception
        set_space_constraints(space)
        self.assertGreater(space.iterations, 0)

    def test_space_with_constraints(self):
        """Test with space containing constraints."""
        space = pm.Space()
        body1 = pm.Body(1, 1)
        body2 = pm.Body(1, 1)
        pivot = pm.PivotJoint(body1, body2, (0, 0))
        space.add(body1, body2, pivot)
        set_space_constraints(space)
        self.assertGreater(space.iterations, 0)


class TestWorld(unittest.TestCase):
    """Test World class."""

    def test_world_creation_default(self):
        """Test creating a World with default parameters."""
        world = World()
        self.assertIsInstance(world.space, pm.Space)
        self.assertEqual(world.space.gravity, params["physics"]["gravity"])
        self.assertEqual(len(world.linkages), 0)
        self.assertEqual(len(world.road), 2)

    def test_world_creation_custom_space(self):
        """Test creating a World with custom space."""
        custom_space = pm.Space()
        custom_space.gravity = (0, -5)
        world = World(space=custom_space)
        self.assertEqual(world.space.gravity, (0, -5))

    def test_world_creation_custom_road_y(self):
        """Test creating a World with custom road_y."""
        world = World(road_y=-10)
        # Road points should have y = -10
        self.assertEqual(world.road[0][1], -10)
        self.assertEqual(world.road[-1][1], -10)

    def test_add_linkage_kinematic(self):
        """Test adding a kinematic linkage."""
        world = World()
        base = Static(0, 0, name="base")
        crank = Crank(1, 0, joint0=base, distance=1, angle=0, name="crank")
        follower = Pivot(
            0, 2, joint0=base, joint1=crank,
            distance0=2, distance1=1.5, name="follower"
        )
        output = Fixed(
            joint0=crank, joint1=follower,
            distance=1, angle=-pi/2, name="output"
        )
        linkage = Linkage(
            joints=(base, crank, follower, output),
            name="test_linkage"
        )
        world.add_linkage(linkage, load=5)
        self.assertEqual(len(world.linkages), 1)
        self.assertIsInstance(world.linkages[0], DynamicLinkage)

    def test_add_dynamic_linkage(self):
        """Test adding a DynamicLinkage directly."""
        space = pm.Space()
        space.gravity = params["physics"]["gravity"]
        base = Static(0, 0, name="base")
        crank = Crank(1, 0, joint0=base, distance=1, angle=0, name="crank")
        follower = Pivot(
            0, 2, joint0=base, joint1=crank,
            distance0=2, distance1=1.5, name="follower"
        )
        output = Fixed(
            joint0=crank, joint1=follower,
            distance=1, angle=-pi/2, name="output"
        )
        dynamic_linkage = DynamicLinkage(
            joints=(base, crank, follower, output),
            space=space,
            name="test_dynamic"
        )
        world = World(space=space)
        world.add_linkage(dynamic_linkage)
        self.assertEqual(len(world.linkages), 1)

    def test_world_update(self):
        """Test World update method."""
        world = World()
        base = Static(0, 0, name="base")
        crank = Crank(1, 0, joint0=base, distance=1, angle=0, name="crank")
        follower = Pivot(
            0, 2, joint0=base, joint1=crank,
            distance0=2, distance1=1.5, name="follower"
        )
        output = Fixed(
            joint0=crank, joint1=follower,
            distance=1, angle=-pi/2, name="output"
        )
        linkage = Linkage(
            joints=(base, crank, follower, output),
            name="test_linkage"
        )
        world.add_linkage(linkage, load=5)
        # Update should return efficiency and energy
        result = world.update()
        self.assertIsNotNone(result)

    def test_world_update_custom_dt(self):
        """Test World update with custom dt."""
        world = World()
        base = Static(0, 0, name="base")
        crank = Crank(1, 0, joint0=base, distance=1, angle=0, name="crank")
        follower = Pivot(
            0, 2, joint0=base, joint1=crank,
            distance0=2, distance1=1.5, name="follower"
        )
        output = Fixed(
            joint0=crank, joint1=follower,
            distance=1, angle=-pi/2, name="output"
        )
        linkage = Linkage(
            joints=(base, crank, follower, output),
            name="test_linkage"
        )
        world.add_linkage(linkage, load=5)
        result = world.update(dt=0.01)
        self.assertIsNotNone(result)

    def test_build_road_positive(self):
        """Test building road in positive direction."""
        world = World()
        initial_road_len = len(world.road)
        world.build_road(positive=True)
        self.assertGreater(len(world.road), initial_road_len)

    def test_build_road_negative(self):
        """Test building road in negative direction."""
        world = World()
        initial_road_len = len(world.road)
        world.build_road(positive=False)
        self.assertGreater(len(world.road), initial_road_len)


class TestRecalcLinkage(unittest.TestCase):
    """Test recalc_linkage function."""

    def test_recalc_linkage(self):
        """Test that recalc_linkage doesn't raise errors."""
        space = pm.Space()
        space.gravity = params["physics"]["gravity"]
        base = Static(0, 0, name="base")
        crank = Crank(1, 0, joint0=base, distance=1, angle=0, name="crank")
        follower = Pivot(
            0, 2, joint0=base, joint1=crank,
            distance0=2, distance1=1.5, name="follower"
        )
        output = Fixed(
            joint0=crank, joint1=follower,
            distance=1, angle=-pi/2, name="output"
        )
        linkage = DynamicLinkage(
            joints=(base, crank, follower, output),
            space=space,
            name="test_dynamic"
        )
        # Should not raise exception
        recalc_linkage(linkage)


class TestLinkageBb(unittest.TestCase):
    """Test linkage_bb function."""

    def test_linkage_bb_kinematic(self):
        """Test bounding box for kinematic linkage with initialized joints."""
        # Create a simple linkage with explicit coordinates
        base = Static(0, 0, name="base")
        crank = Crank(1, 0, joint0=base, distance=1, angle=1, name="crank")
        # Manually trigger position calculation for crank
        crank.reload(0)
        # Create a simple 2-joint linkage since Fixed joints require rebuild
        linkage = Linkage(
            joints=(base, crank),
            name="test_linkage"
        )
        bb = linkage_bb(linkage)
        # Bounding box should be (min_y, max_x, max_y, min_x)
        self.assertEqual(len(bb), 4)

    def test_linkage_bb_dynamic(self):
        """Test bounding box for dynamic linkage."""
        space = pm.Space()
        space.gravity = params["physics"]["gravity"]
        base = Static(0, 0, name="base")
        crank = Crank(1, 0, joint0=base, distance=1, angle=0, name="crank")
        follower = Pivot(
            0, 2, joint0=base, joint1=crank,
            distance0=2, distance1=1.5, name="follower"
        )
        output = Fixed(
            joint0=crank, joint1=follower,
            distance=1, angle=-pi/2, name="output"
        )
        linkage = DynamicLinkage(
            joints=(base, crank, follower, output),
            space=space,
            name="test_dynamic"
        )
        bb = linkage_bb(linkage)
        self.assertEqual(len(bb), 4)


if __name__ == "__main__":
    unittest.main()
