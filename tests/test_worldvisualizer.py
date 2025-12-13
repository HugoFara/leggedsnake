#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Tests for the worldvisualizer module.
"""

import unittest
from math import pi
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend for testing
import matplotlib.pyplot as plt
import pymunk as pm
from pylinkage import Static, Crank, Fixed, Pivot, Linkage

from leggedsnake.worldvisualizer import (
    smooth_transition, VisualWorld, CAMERA
)
from leggedsnake.dynamiclinkage import DynamicLinkage
from leggedsnake.physicsengine import params


class TestSmoothTransition(unittest.TestCase):
    """Test smooth_transition function."""

    def test_no_change_needed(self):
        """Test when target equals prev_view."""
        target = ((-10, 10), (-5, 5))
        prev_view = ((-10, 10), (-5, 5))
        result = smooth_transition(target, prev_view)
        # Result should be close to target/prev_view
        self.assertEqual(len(result), 2)
        self.assertEqual(len(result[0]), 2)
        self.assertEqual(len(result[1]), 2)

    def test_smooth_transition_expands(self):
        """Test when target is larger than prev_view."""
        target = ((-20, 20), (-10, 10))
        prev_view = ((-10, 10), (-5, 5))
        result = smooth_transition(target, prev_view)
        # Should return new bounds
        self.assertEqual(len(result), 2)

    def test_smooth_transition_contracts(self):
        """Test when target is smaller than prev_view."""
        target = ((-5, 5), (-2, 2))
        prev_view = ((-10, 10), (-5, 5))
        result = smooth_transition(target, prev_view)
        self.assertEqual(len(result), 2)

    def test_smooth_transition_custom_dampers(self):
        """Test with custom dampers."""
        target = ((-15, 15), (-8, 8))
        prev_view = ((-10, 10), (-5, 5))
        dampers = ((-5, -3), (5, 3))
        result = smooth_transition(target, prev_view, dampers=dampers)
        self.assertEqual(len(result), 2)


class TestVisualWorld(unittest.TestCase):
    """Test VisualWorld class."""

    def setUp(self):
        """Set up test fixtures."""
        # Force non-interactive backend
        matplotlib.use('Agg')

    def tearDown(self):
        """Clean up matplotlib figures."""
        plt.close('all')

    def test_visual_world_creation(self):
        """Test creating a VisualWorld."""
        world = VisualWorld()
        self.assertIsNotNone(world.fig)
        self.assertIsNotNone(world.ax)
        self.assertEqual(len(world.linkages), 0)

    def test_visual_world_custom_road_y(self):
        """Test VisualWorld with custom road_y."""
        world = VisualWorld(road_y=-10)
        self.assertEqual(world.road[0][1], -10)

    def test_visual_world_add_linkage(self):
        """Test adding a linkage to VisualWorld."""
        world = VisualWorld()
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
        self.assertEqual(len(world.linkage_im), 1)

    def test_visual_world_init_visuals(self):
        """Test init_visuals method."""
        world = VisualWorld()
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
        world.add_linkage(linkage)
        result = world.init_visuals()
        self.assertIsNotNone(result)

    def test_visual_world_init_visuals_with_colors(self):
        """Test init_visuals with colors."""
        world = VisualWorld()
        base = Static(0, 0, name="base")
        crank = Crank(1, 0, joint0=base, distance=1, angle=0, name="crank")
        linkage = Linkage(joints=(base, crank), name="test")
        world.add_linkage(linkage)
        colors = [0.5]  # opacity
        result = world.init_visuals(colors=colors)
        self.assertIsNotNone(result)

    def test_visual_world_reload_visuals(self):
        """Test reload_visuals method."""
        world = VisualWorld()
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
        world.add_linkage(linkage)
        result = world.reload_visuals()
        self.assertIsNotNone(result)

    def test_visual_world_draw_linkage(self):
        """Test draw_linkage method."""
        world = VisualWorld()
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
        world.add_linkage(linkage)
        dynamic_linkage = world.linkages[0]
        result = world.draw_linkage(world.linkage_im[0], dynamic_linkage.joints)
        self.assertIsNotNone(result)

    def test_visual_world_visual_update(self):
        """Test visual_update method."""
        world = VisualWorld()
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
        world.add_linkage(linkage)
        result = world.visual_update()
        self.assertIsNotNone(result)

    def test_visual_world_visual_update_with_time(self):
        """Test visual_update with custom time."""
        world = VisualWorld()
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
        world.add_linkage(linkage)
        result = world.visual_update(time=0.01)
        self.assertIsNotNone(result)

    def test_visual_world_visual_update_with_time_tuple(self):
        """Test visual_update with time as tuple (dt, fps)."""
        world = VisualWorld()
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
        world.add_linkage(linkage)
        result = world.visual_update(time=[0.02, 20])
        self.assertIsNotNone(result)


class TestCAMERASettings(unittest.TestCase):
    """Test CAMERA configuration dict."""

    def test_camera_settings_structure(self):
        """Test CAMERA has expected keys."""
        self.assertIn("dynamic_camera", CAMERA)
        self.assertIn("fps", CAMERA)

    def test_camera_fps_is_positive(self):
        """Test fps is a positive number."""
        self.assertGreater(CAMERA["fps"], 0)


if __name__ == "__main__":
    unittest.main()
