#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Tests for the URDF export module."""

import math
import unittest
import xml.etree.ElementTree as ET

from pylinkage.dimensions import Dimensions, DriverAngle
from pylinkage.hypergraph import (
    Edge,
    Hyperedge,
    HypergraphLinkage,
    Node,
    NodeRole,
)

from leggedsnake.urdf_export import URDFConfig, to_urdf, to_urdf_file
from leggedsnake.walker import Walker


def _make_fourbar_walker() -> Walker:
    """Simple 4-bar: frame, crank, follower, 3 edges."""
    hg = HypergraphLinkage(name="fourbar")
    hg.add_node(Node("frame", role=NodeRole.GROUND))
    hg.add_node(Node("crank", role=NodeRole.DRIVER))
    hg.add_node(Node("follower", role=NodeRole.DRIVEN))
    hg.add_edge(Edge("e0", "frame", "crank"))
    hg.add_edge(Edge("e1", "frame", "follower"))
    hg.add_edge(Edge("e2", "crank", "follower"))
    dims = Dimensions(
        node_positions={"frame": (0, 0), "crank": (1, 0), "follower": (0, 2)},
        driver_angles={"crank": DriverAngle(angular_velocity=-math.tau / 12)},
        edge_distances={"e0": 1.0, "e1": 2.0, "e2": 1.5},
    )
    return Walker(hg, dims, name="fourbar")


def _make_sixbar_with_hyperedge() -> Walker:
    """6-bar with a ternary link (hyperedge grouping 3 nodes)."""
    hg = HypergraphLinkage(name="sixbar")
    hg.add_node(Node("G1", role=NodeRole.GROUND))
    hg.add_node(Node("G2", role=NodeRole.GROUND))
    hg.add_node(Node("D", role=NodeRole.DRIVER))
    hg.add_node(Node("A", role=NodeRole.DRIVEN))
    hg.add_node(Node("B", role=NodeRole.DRIVEN))
    hg.add_node(Node("C", role=NodeRole.DRIVEN))
    hg.add_edge(Edge("e0", "G1", "D"))
    hg.add_edge(Edge("e1", "D", "A"))
    hg.add_edge(Edge("e2", "A", "B"))
    hg.add_edge(Edge("e3", "B", "C"))
    hg.add_edge(Edge("e4", "G2", "C"))
    hg.add_edge(Edge("e5", "A", "C"))  # Part of triangle A-B-C
    # Triangle A-B-C as a hyperedge
    hg.add_hyperedge(Hyperedge("tri", nodes=("A", "B", "C")))
    dims = Dimensions(
        node_positions={
            "G1": (0, 0), "G2": (4, 0), "D": (1, 0),
            "A": (2, 1), "B": (3, 2), "C": (3, 0),
        },
        driver_angles={"D": DriverAngle(angular_velocity=-1.0)},
        edge_distances={
            "e0": 1.0, "e1": 1.41, "e2": 1.41,
            "e3": 2.0, "e4": 1.0, "e5": 1.41,
        },
    )
    return Walker(hg, dims, name="sixbar")


class TestToUrdf(unittest.TestCase):
    """Test basic URDF generation."""

    def test_returns_string(self):
        walker = _make_fourbar_walker()
        result = to_urdf(walker)
        self.assertIsInstance(result, str)

    def test_valid_xml(self):
        walker = _make_fourbar_walker()
        result = to_urdf(walker)
        root = ET.fromstring(result)
        self.assertEqual(root.tag, "robot")

    def test_robot_name(self):
        walker = _make_fourbar_walker()
        root = ET.fromstring(to_urdf(walker))
        self.assertEqual(root.get("name"), "fourbar")

    def test_has_base_link(self):
        walker = _make_fourbar_walker()
        root = ET.fromstring(to_urdf(walker))
        links = {el.get("name") for el in root.findall("link")}
        self.assertIn("base_link", links)

    def test_has_links_for_edges(self):
        walker = _make_fourbar_walker()
        root = ET.fromstring(to_urdf(walker))
        links = {el.get("name") for el in root.findall("link")}
        # 3 edges + base_link = at least 4 links
        self.assertGreaterEqual(len(links), 4)

    def test_has_joints(self):
        walker = _make_fourbar_walker()
        root = ET.fromstring(to_urdf(walker))
        joints = root.findall("joint")
        self.assertGreater(len(joints), 0)

    def test_ground_joints_are_fixed(self):
        walker = _make_fourbar_walker()
        root = ET.fromstring(to_urdf(walker))
        ground_joints = [
            j for j in root.findall("joint")
            if j.get("name", "").startswith("ground_")
        ]
        for j in ground_joints:
            self.assertEqual(j.get("type"), "fixed")

    def test_driver_joint_is_continuous(self):
        walker = _make_fourbar_walker()
        root = ET.fromstring(to_urdf(walker))
        motor_joints = [
            j for j in root.findall("joint")
            if j.get("name", "").startswith("motor_")
        ]
        self.assertGreater(len(motor_joints), 0)
        self.assertEqual(motor_joints[0].get("type"), "continuous")

    def test_revolute_joints_have_limits(self):
        walker = _make_fourbar_walker()
        root = ET.fromstring(to_urdf(walker))
        revolute_joints = [
            j for j in root.findall("joint")
            if j.get("type") == "revolute"
        ]
        for j in revolute_joints:
            limit = j.find("limit")
            self.assertIsNotNone(limit, f"Joint {j.get('name')} missing <limit>")

    def test_links_have_inertial(self):
        walker = _make_fourbar_walker()
        root = ET.fromstring(to_urdf(walker))
        for link_el in root.findall("link"):
            inertial = link_el.find("inertial")
            self.assertIsNotNone(
                inertial, f"Link {link_el.get('name')} missing <inertial>"
            )


class TestHyperedgeGrouping(unittest.TestCase):
    """Test that hyperedge members are merged into a single URDF link."""

    def test_hyperedge_creates_single_link(self):
        walker = _make_sixbar_with_hyperedge()
        root = ET.fromstring(to_urdf(walker))
        links = {el.get("name") for el in root.findall("link")}
        # The triangle edges (e2, e3, e5) should all map to "link_tri"
        self.assertIn("link_tri", links)

    def test_fewer_links_with_hyperedge(self):
        walker = _make_sixbar_with_hyperedge()
        root = ET.fromstring(to_urdf(walker))
        links = root.findall("link")
        # Without hyperedge: 6 edges + base = 7 links
        # With hyperedge: edges e2,e3,e5 merge into 1 link = 4 standalone + 1 merged + base = 6
        self.assertLess(len(links), 7)


class TestURDFConfig(unittest.TestCase):
    """Test configuration options."""

    def test_custom_base_link_name(self):
        walker = _make_fourbar_walker()
        cfg = URDFConfig(base_link_name="chassis")
        root = ET.fromstring(to_urdf(walker, config=cfg))
        links = {el.get("name") for el in root.findall("link")}
        self.assertIn("chassis", links)
        self.assertNotIn("base_link", links)

    def test_custom_color(self):
        walker = _make_fourbar_walker()
        cfg = URDFConfig(mesh_color=(1.0, 0.0, 0.0, 1.0))
        root = ET.fromstring(to_urdf(walker, config=cfg))
        # Find a material color element
        for material in root.iter("color"):
            rgba = material.get("rgba", "")
            self.assertTrue(rgba.startswith("1.0 0.0 0.0"))

    def test_custom_link_radius(self):
        walker = _make_fourbar_walker()
        cfg = URDFConfig(link_radius=0.05)
        root = ET.fromstring(to_urdf(walker, config=cfg))
        for cyl in root.iter("cylinder"):
            self.assertEqual(cyl.get("radius"), "0.0500")


class TestToUrdfFile(unittest.TestCase):
    """Test file output."""

    def test_writes_file(self):
        import tempfile
        import os

        walker = _make_fourbar_walker()
        with tempfile.NamedTemporaryFile(suffix=".urdf", delete=False) as f:
            path = f.name

        try:
            to_urdf_file(walker, path)
            self.assertTrue(os.path.exists(path))
            with open(path) as f:
                content = f.read()
            self.assertIn("<robot", content)
            # Validate XML
            ET.fromstring(content)
        finally:
            os.unlink(path)


class TestMultiLegWalker(unittest.TestCase):
    """Test URDF export with multi-leg walkers."""

    def test_walker_with_added_legs(self):
        walker = _make_fourbar_walker()
        walker.add_legs(1)  # Add one more leg
        result = to_urdf(walker)
        root = ET.fromstring(result)
        # Should have more links than original
        links = root.findall("link")
        self.assertGreater(len(links), 4)  # base + original 3 + cloned

    def test_valid_xml_with_legs(self):
        walker = _make_fourbar_walker()
        walker.add_legs(2)
        result = to_urdf(walker)
        # Should parse without error
        ET.fromstring(result)


class TestExports(unittest.TestCase):
    """Test that URDF exports are accessible from the package."""

    def test_to_urdf_importable(self):
        import leggedsnake
        self.assertTrue(hasattr(leggedsnake, "to_urdf"))

    def test_to_urdf_file_importable(self):
        import leggedsnake
        self.assertTrue(hasattr(leggedsnake, "to_urdf_file"))

    def test_urdf_config_importable(self):
        import leggedsnake
        self.assertTrue(hasattr(leggedsnake, "URDFConfig"))


if __name__ == "__main__":
    unittest.main()
