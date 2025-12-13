#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Tests for the dynamiclinkage module.
"""

import unittest
from math import pi
import pymunk as pm
from pylinkage import Static, Crank, Fixed, Pivot, Linkage

from leggedsnake.dynamiclinkage import (
    DynamicJoint, Nail, PinUp, DynamicPivot, Motor,
    DynamicLinkage, convert_to_dynamic_linkage
)


class TestNail(unittest.TestCase):
    """Test Nail class (dynamic Static joint)."""

    def test_nail_creation_basic(self):
        """Test creating a basic Nail."""
        nail = Nail(x=0, y=0, name="test_nail")
        self.assertEqual(nail.x, 0)
        self.assertEqual(nail.y, 0)
        self.assertEqual(nail.name, "test_nail")

    def test_nail_with_body(self):
        """Test creating a Nail attached to a body."""
        space = pm.Space()
        body = pm.Body(1, 1)
        body.position = (0, 0)
        space.add(body)
        shape_filter = pm.ShapeFilter(group=1)
        nail = Nail(
            x=0, y=0, name="test_nail", body=body, space=space,
            shape_filter=shape_filter
        )
        self.assertEqual(nail._a, body)
        self.assertEqual(nail._b, body)

    def test_nail_coord(self):
        """Test that Nail returns correct coordinates."""
        nail = Nail(x=5, y=10, name="test_nail")
        coord = nail.coord()
        self.assertEqual(coord[0], 5)
        self.assertEqual(coord[1], 10)


class TestMotor(unittest.TestCase):
    """Test Motor class (dynamic Crank)."""

    def test_motor_creation(self):
        """Test creating a Motor."""
        space = pm.Space()
        body = pm.Body(1, 1)
        body.position = (0, 0)
        space.add(body)
        shape_filter = pm.ShapeFilter(group=1)
        nail = Nail(
            x=0, y=0, name="base", body=body, space=space,
            shape_filter=shape_filter
        )
        motor = Motor(
            x=1, y=0, joint0=nail, space=space,
            distance=1, angle=0, name="motor",
            shape_filter=shape_filter
        )
        self.assertEqual(motor.name, "motor")
        self.assertIsNotNone(motor._b)


class TestDynamicPivot(unittest.TestCase):
    """Test DynamicPivot class."""

    def test_dynamic_pivot_creation(self):
        """Test creating a DynamicPivot."""
        space = pm.Space()
        body = pm.Body(1, 1)
        body.position = (0, 0)
        space.add(body)
        shape_filter = pm.ShapeFilter(group=1)
        nail = Nail(
            x=0, y=0, name="base", body=body, space=space,
            shape_filter=shape_filter
        )
        motor = Motor(
            x=1, y=0, joint0=nail, space=space,
            distance=1, angle=0, name="motor",
            shape_filter=shape_filter
        )
        pivot = DynamicPivot(
            x=0, y=2, joint0=nail, joint1=motor, space=space,
            distance0=2, distance1=1.5, name="pivot",
            shape_filter=shape_filter
        )
        self.assertEqual(pivot.name, "pivot")


class TestPinUp(unittest.TestCase):
    """Test PinUp class (dynamic Fixed joint)."""

    def test_pinup_creation(self):
        """Test creating a PinUp."""
        space = pm.Space()
        body = pm.Body(1, 1)
        body.position = (0, 0)
        space.add(body)
        shape_filter = pm.ShapeFilter(group=1)
        nail = Nail(
            x=0, y=0, name="base", body=body, space=space,
            shape_filter=shape_filter
        )
        motor = Motor(
            x=1, y=0, joint0=nail, space=space,
            distance=1, angle=0, name="motor",
            shape_filter=shape_filter
        )
        pivot = DynamicPivot(
            x=0, y=2, joint0=nail, joint1=motor, space=space,
            distance0=2, distance1=1.5, name="pivot",
            shape_filter=shape_filter
        )
        pinup = PinUp(
            joint0=motor, joint1=pivot, space=space,
            distance=1, angle=-pi/2, name="pinup",
            shape_filter=shape_filter
        )
        self.assertEqual(pinup.name, "pinup")


class TestDynamicLinkage(unittest.TestCase):
    """Test DynamicLinkage class."""

    def setUp(self):
        """Create a basic linkage setup for testing."""
        self.space = pm.Space()
        self.space.gravity = (0, -9.8)
        self.base = Static(0, 0, name="base")
        self.crank = Crank(1, 0, joint0=self.base, distance=1, angle=0, name="crank")
        self.follower = Pivot(
            0, 2, joint0=self.base, joint1=self.crank,
            distance0=2, distance1=1.5, name="follower"
        )
        self.output = Fixed(
            joint0=self.crank, joint1=self.follower,
            distance=1, angle=-pi/2, name="output"
        )

    def test_dynamic_linkage_creation(self):
        """Test creating a DynamicLinkage."""
        linkage = DynamicLinkage(
            joints=(self.base, self.crank, self.follower, self.output),
            space=self.space,
            name="test_dynamic"
        )
        self.assertEqual(linkage.name, "test_dynamic")
        self.assertIsInstance(linkage.space, pm.Space)

    def test_dynamic_linkage_joints_converted(self):
        """Test that kinematic joints are converted to dynamic equivalents."""
        linkage = DynamicLinkage(
            joints=(self.base, self.crank, self.follower, self.output),
            space=self.space,
            name="test_dynamic"
        )
        # All joints should now be DynamicJoint instances
        for joint in linkage.joints:
            self.assertIsInstance(joint, DynamicJoint)

    def test_dynamic_linkage_has_cranks(self):
        """Test that DynamicLinkage identifies cranks."""
        linkage = DynamicLinkage(
            joints=(self.base, self.crank, self.follower, self.output),
            space=self.space,
            name="test_dynamic"
        )
        self.assertGreater(len(linkage._cranks), 0)

    def test_dynamic_linkage_density(self):
        """Test setting custom density."""
        linkage = DynamicLinkage(
            joints=(self.base, self.crank, self.follower, self.output),
            space=self.space,
            density=2.5,
            name="test_dynamic"
        )
        self.assertEqual(linkage.density, 2.5)

    def test_dynamic_linkage_load(self):
        """Test setting custom load."""
        linkage = DynamicLinkage(
            joints=(self.base, self.crank, self.follower, self.output),
            space=self.space,
            load=10,
            name="test_dynamic"
        )
        self.assertIsNotNone(linkage.body)

    def test_dynamic_linkage_thickness(self):
        """Test setting custom thickness."""
        linkage = DynamicLinkage(
            joints=(self.base, self.crank, self.follower, self.output),
            space=self.space,
            thickness=0.2,
            name="test_dynamic"
        )
        self.assertEqual(linkage._thickness, 0.2)

    def test_build_load(self):
        """Test build_load method."""
        linkage = DynamicLinkage(
            joints=(self.base, self.crank, self.follower, self.output),
            space=self.space,
            name="test_dynamic"
        )
        load_body = linkage.build_load((5, 5), 10)
        self.assertIsInstance(load_body, pm.Body)
        self.assertEqual(load_body.position, (5, 5))


class TestConvertToDynamicLinkage(unittest.TestCase):
    """Test convert_to_dynamic_linkage function."""

    def test_convert_kinematic_linkage(self):
        """Test converting a kinematic Linkage to DynamicLinkage."""
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
        kinematic = Linkage(
            joints=(base, crank, follower, output),
            name="kinematic"
        )
        space = pm.Space()
        space.gravity = (0, -9.8)
        dynamic = convert_to_dynamic_linkage(kinematic, space, density=1, load=5)
        self.assertIsInstance(dynamic, DynamicLinkage)
        self.assertEqual(dynamic.name, "kinematic")


class TestDynamicJointReload(unittest.TestCase):
    """Test reload methods of dynamic joints."""

    def test_nail_reload(self):
        """Test Nail reload method."""
        space = pm.Space()
        body = pm.Body(1, 1)
        body.position = (0, 0)
        space.add(body)
        shape_filter = pm.ShapeFilter(group=1)
        nail = Nail(
            x=0, y=0, name="test_nail", body=body, space=space,
            shape_filter=shape_filter
        )
        # Move body
        body.position = (5, 5)
        nail.reload()
        # Nail should update its position based on body
        coord = nail.coord()
        # Position should change based on body movement
        self.assertIsNotNone(coord)

    def test_motor_reload(self):
        """Test Motor reload method."""
        space = pm.Space()
        body = pm.Body(1, 1)
        body.position = (0, 0)
        space.add(body)
        shape_filter = pm.ShapeFilter(group=1)
        nail = Nail(
            x=0, y=0, name="base", body=body, space=space,
            shape_filter=shape_filter
        )
        motor = Motor(
            x=1, y=0, joint0=nail, space=space,
            distance=1, angle=0, name="motor",
            shape_filter=shape_filter
        )
        # Should not raise exception
        motor.reload()


if __name__ == "__main__":
    unittest.main()
