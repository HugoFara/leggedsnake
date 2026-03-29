#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Tests for the worldvisualizer module.
"""

import unittest
from math import tau
from unittest.mock import MagicMock

import pymunk as pm
from pylinkage.dimensions import Dimensions, DriverAngle
from pylinkage.hypergraph import HypergraphLinkage, Node, Edge, NodeRole

from leggedsnake.worldvisualizer import (
    smooth_transition, VisualWorld, CAMERA
)
from leggedsnake.dynamiclinkage import DynamicLinkage
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
        edge_distances={"frame_crank": 1.0, "frame_follower": 2.0, "crank_follower": 1.5},
    )
    return Walker(hg, dims, name="fourbar")


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
        """Test adding a Walker to VisualWorld."""
        world = VisualWorld(headless=True)
        walker = _make_fourbar_walker()
        world.add_linkage(walker, load=5)
        self.assertEqual(len(world.linkages), 1)

    def test_visual_world_init_visuals(self):
        """Test init_visuals method."""
        world = VisualWorld(headless=True)
        walker = _make_fourbar_walker()
        world.add_linkage(walker)
        result = world.init_visuals()
        self.assertIsNotNone(result)

    def test_visual_world_init_visuals_with_colors(self):
        """Test init_visuals with opacity colors."""
        world = VisualWorld(headless=True)
        walker = _make_fourbar_walker()
        world.add_linkage(walker)
        colors = [0.5]  # opacity
        result = world.init_visuals(colors=colors)
        self.assertIsNotNone(result)
        self.assertIsNotNone(world._linkage_colors)

    def test_visual_world_init_visuals_with_rgb_colors(self):
        """Test init_visuals with RGB colors."""
        world = VisualWorld(headless=True)
        walker = _make_fourbar_walker()
        world.add_linkage(walker)
        colors = [[0.5, 0.3, 0.8]]  # RGB color
        result = world.init_visuals(colors=colors)
        self.assertIsNotNone(result)
        self.assertIsNotNone(world._linkage_colors)

    def test_visual_world_reload_visuals(self):
        """Test reload_visuals method."""
        world = VisualWorld(headless=True)
        walker = _make_fourbar_walker()
        world.add_linkage(walker)
        result = world.reload_visuals()
        self.assertIsNotNone(result)

    def test_visual_world_visual_update(self):
        """Test visual_update method."""
        world = VisualWorld(headless=True)
        walker = _make_fourbar_walker()
        world.add_linkage(walker)
        result = world.visual_update()
        self.assertIsNotNone(result)

    def test_visual_world_visual_update_with_time(self):
        """Test visual_update with custom time."""
        world = VisualWorld(headless=True)
        walker = _make_fourbar_walker()
        world.add_linkage(walker)
        result = world.visual_update(time=0.01)
        self.assertIsNotNone(result)

    def test_visual_world_visual_update_with_time_tuple(self):
        """Test visual_update with time as tuple (dt, fps)."""
        world = VisualWorld(headless=True)
        walker = _make_fourbar_walker()
        world.add_linkage(walker)
        result = world.visual_update(time=[0.02, 20])
        self.assertIsNotNone(result)

    def test_visual_world_view_bounds(self):
        """Test view bounds are updated correctly."""
        world = VisualWorld(headless=True)
        walker = _make_fourbar_walker()
        world.add_linkage(walker)
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

    def test_visual_world_get_joint_color_ground(self):
        """Test joint color for GROUND role via mock NodeProxy."""
        world = VisualWorld(headless=True)
        mock_joint = MagicMock()
        mock_joint.role = NodeRole.GROUND
        color = world._get_joint_color(mock_joint)
        self.assertEqual(len(color), 4)
        # Ground color is gray (100, 100, 100, 255)
        self.assertEqual(color, (100, 100, 100, 255))

    def test_visual_world_get_joint_color_driver(self):
        """Test joint color for DRIVER role via mock NodeProxy."""
        world = VisualWorld(headless=True)
        mock_joint = MagicMock()
        mock_joint.role = NodeRole.DRIVER
        color = world._get_joint_color(mock_joint)
        self.assertEqual(len(color), 4)
        # Driver color is green (50, 200, 50, 255)
        self.assertEqual(color, (50, 200, 50, 255))

    def test_visual_world_get_joint_color_driven(self):
        """Test joint color for DRIVEN role via mock NodeProxy."""
        world = VisualWorld(headless=True)
        mock_joint = MagicMock()
        mock_joint.role = NodeRole.DRIVEN
        color = world._get_joint_color(mock_joint)
        self.assertEqual(len(color), 4)
        # Driven color is blue (50, 100, 200, 255)
        self.assertEqual(color, (50, 100, 200, 255))

    def test_visual_world_get_joint_color_no_role(self):
        """Test joint color when object has no role attribute."""
        world = VisualWorld(headless=True)
        mock_joint = object()  # no .role attribute
        color = world._get_joint_color(mock_joint)
        self.assertEqual(len(color), 4)
        # Falls back to ground color
        self.assertEqual(color, (100, 100, 100, 255))


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
