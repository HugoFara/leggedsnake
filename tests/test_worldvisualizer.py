#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Tests for the worldvisualizer module.
"""

import unittest
from math import pi
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

    def test_visual_world_creation(self):
        """Test creating a VisualWorld in headless mode."""
        world = VisualWorld(headless=True)
        self.assertIsNone(world.window)
        self.assertEqual(len(world.linkages), 0)

    def test_visual_world_custom_road_y(self):
        """Test VisualWorld with custom road_y."""
        world = VisualWorld(road_y=-10, headless=True)
        self.assertEqual(world.road[0][1], -10)

    def test_visual_world_add_linkage(self):
        """Test adding a linkage to VisualWorld."""
        world = VisualWorld(headless=True)
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

    def test_visual_world_init_visuals(self):
        """Test init_visuals method."""
        world = VisualWorld(headless=True)
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
        world = VisualWorld(headless=True)
        base = Static(0, 0, name="base")
        crank = Crank(1, 0, joint0=base, distance=1, angle=0, name="crank")
        linkage = Linkage(joints=(base, crank), name="test")
        world.add_linkage(linkage)
        colors = [0.5]  # opacity
        result = world.init_visuals(colors=colors)
        self.assertIsNotNone(result)
        # Check that color was processed
        self.assertIsNotNone(world._linkage_colors)

    def test_visual_world_init_visuals_with_rgb_colors(self):
        """Test init_visuals with RGB colors."""
        world = VisualWorld(headless=True)
        base = Static(0, 0, name="base")
        crank = Crank(1, 0, joint0=base, distance=1, angle=0, name="crank")
        linkage = Linkage(joints=(base, crank), name="test")
        world.add_linkage(linkage)
        colors = [[0.5, 0.3, 0.8]]  # RGB color
        result = world.init_visuals(colors=colors)
        self.assertIsNotNone(result)
        self.assertIsNotNone(world._linkage_colors)

    def test_visual_world_reload_visuals(self):
        """Test reload_visuals method."""
        world = VisualWorld(headless=True)
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

    def test_visual_world_visual_update(self):
        """Test visual_update method."""
        world = VisualWorld(headless=True)
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
        world = VisualWorld(headless=True)
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
        world = VisualWorld(headless=True)
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

    def test_visual_world_view_bounds(self):
        """Test view bounds are updated correctly."""
        world = VisualWorld(headless=True)
        base = Static(0, 0, name="base")
        crank = Crank(1, 0, joint0=base, distance=1, angle=0, name="crank")
        linkage = Linkage(joints=(base, crank), name="test")
        world.add_linkage(linkage)
        world.reload_visuals()
        # View bounds should have been updated
        self.assertIsNotNone(world._view_bounds)
        self.assertEqual(len(world._view_bounds), 2)

    def test_visual_world_world_to_screen(self):
        """Test world to screen coordinate conversion."""
        world = VisualWorld(headless=True)
        # Test coordinate conversion
        screen_x, screen_y = world._world_to_screen(0, 0)
        # Should return some screen coordinates
        self.assertIsInstance(screen_x, float)
        self.assertIsInstance(screen_y, float)

    def test_visual_world_get_joint_color(self):
        """Test joint color assignment based on type."""
        world = VisualWorld(headless=True)
        base = Static(0, 0, name="base")
        crank = Crank(1, 0, joint0=base, distance=1, angle=0, name="crank")
        follower = Pivot(0, 2, joint0=base, joint1=crank, distance0=2, distance1=1.5)
        output = Fixed(joint0=crank, joint1=follower, distance=1, angle=-pi/2)

        # Test different joint types get different colors
        static_color = world._get_joint_color(base)
        crank_color = world._get_joint_color(crank)
        pivot_color = world._get_joint_color(follower)
        fixed_color = world._get_joint_color(output)

        # All should return RGBA tuples
        self.assertEqual(len(static_color), 4)
        self.assertEqual(len(crank_color), 4)
        self.assertEqual(len(pivot_color), 4)
        self.assertEqual(len(fixed_color), 4)


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
