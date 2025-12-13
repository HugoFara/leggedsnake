#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Tests for the walker module.
"""

import unittest
from math import pi
from pylinkage import Static, Crank, Fixed, Pivot
from leggedsnake.walker import Walker


class TestWalker(unittest.TestCase):
    """Test suite for Walker class."""

    def setUp(self):
        """Create a simple linkage for testing."""
        # Create a simple four-bar linkage
        self.base = Static(0, 0, name="base")
        # Use non-zero angle to avoid ZeroDivisionError in get_rotation_period
        self.crank = Crank(1, 0, joint0=self.base, distance=1, angle=1, name="crank")
        self.follower = Pivot(
            0, 2, joint0=self.base, joint1=self.crank,
            distance0=2, distance1=1.5, name="follower"
        )
        self.output = Fixed(
            joint0=self.crank, joint1=self.follower,
            distance=1, angle=-pi/2, name="output"
        )
        self.walker = Walker(
            joints=(self.base, self.crank, self.follower, self.output),
            name="test_walker"
        )

    def test_walker_creation(self):
        """Test that a Walker can be created."""
        self.assertIsInstance(self.walker, Walker)
        self.assertEqual(len(self.walker.joints), 4)
        self.assertEqual(self.walker.name, "test_walker")

    def test_get_foots_returns_terminal_joints(self):
        """Test that get_foots returns joints without children."""
        foots = self.walker.get_foots()
        # The output joint should be a foot (no other joints reference it)
        self.assertIn(self.output, foots)
        # The base should NOT be a foot (crank and follower reference it)
        self.assertNotIn(self.base, foots)

    def test_get_foots_count(self):
        """Test that get_foots returns correct number of terminal joints."""
        foots = self.walker.get_foots()
        # In a four-bar linkage, the output (Fixed) and follower (Pivot)
        # may be terminal depending on linkage structure
        self.assertGreater(len(foots), 0)

    def test_add_legs_increases_joints(self):
        """Test that add_legs adds new joints to the walker."""
        initial_joint_count = len(self.walker.joints)
        self.walker.add_legs(number=1)
        # Adding 1 leg should add more joints
        self.assertGreater(len(self.walker.joints), initial_joint_count)

    def test_add_legs_default_two(self):
        """Test that add_legs with default parameter adds two legs."""
        # Create a fresh walker for this test
        base = Static(0, 0, name="base2")
        # Use non-zero angle to avoid ZeroDivisionError
        crank = Crank(1, 0, joint0=base, distance=1, angle=1, name="crank2")
        follower = Pivot(
            0, 2, joint0=base, joint1=crank,
            distance0=2, distance1=1.5, name="follower2"
        )
        output = Fixed(
            joint0=crank, joint1=follower,
            distance=1, angle=-pi/2, name="output2"
        )
        walker = Walker(
            joints=(base, crank, follower, output),
            name="test_walker2"
        )
        initial_joint_count = len(walker.joints)
        walker.add_legs(number=2)
        # Adding 2 legs should add even more joints
        self.assertGreater(len(walker.joints), initial_joint_count)


class TestWalkerWithSimpleLinkage(unittest.TestCase):
    """Test Walker with minimal linkage."""

    def test_simple_walker_get_foots(self):
        """Test get_foots with a very simple linkage."""
        base = Static(0, 0, name="simple_base")
        crank = Crank(1, 0, joint0=base, distance=1, angle=0, name="simple_crank")
        walker = Walker(joints=(base, crank), name="simple_walker")

        foots = walker.get_foots()
        # Crank should be a foot since nothing references it
        self.assertIn(crank, foots)


class TestAddOppositeLeg(unittest.TestCase):
    """Test suite for add_opposite_leg functionality."""

    def setUp(self):
        """Create a simple linkage for testing."""
        # Create a simple four-bar linkage with explicit coordinates
        self.base = Static(0, 0, name="base")
        self.crank = Crank(1, 0, joint0=self.base, distance=1, angle=0.1, name="crank")
        self.follower = Pivot(
            x=0, y=2, joint0=self.base, joint1=self.crank,
            distance0=2, distance1=1.5, name="follower"
        )
        self.output = Fixed(
            x=1, y=1,  # Explicit initial coordinates
            joint0=self.crank, joint1=self.follower,
            distance=1, angle=-pi/2, name="output"
        )
        joints = (self.base, self.crank, self.follower, self.output)
        self.walker = Walker(
            joints=joints,
            order=joints,  # Required for add_opposite_leg and add_legs
            name="test_walker"
        )
        # Step to compute actual joint positions
        list(self.walker.step())

    def test_add_opposite_leg_increases_joints(self):
        """Test that add_opposite_leg adds new joints."""
        initial_count = len(self.walker.joints)
        self.walker.add_opposite_leg()
        # Should add joints (base at x=0 is shared, so not doubled)
        self.assertGreater(len(self.walker.joints), initial_count)

    def test_add_opposite_leg_coordinates_reflected(self):
        """Test that opposite joints have reflected X coordinates."""
        self.walker.add_opposite_leg(axis_x=0.0)

        # Find original and opposite crank
        original_crank = self.crank
        opposite_crank = None
        for j in self.walker.joints:
            if "crank" in j.name.lower() and "(opposite)" in j.name:
                opposite_crank = j
                break

        self.assertIsNotNone(opposite_crank)
        # X coordinate should be negated (mirrored across x=0)
        self.assertEqual(opposite_crank.x, -original_crank.x)
        # Y coordinate should be the same
        self.assertEqual(opposite_crank.y, original_crank.y)

    def test_add_opposite_leg_custom_axis(self):
        """Test that add_opposite_leg works with custom axis."""
        self.walker.add_opposite_leg(axis_x=2.0)

        # Find opposite crank
        opposite_crank = None
        for j in self.walker.joints:
            if "crank" in j.name.lower() and "(opposite)" in j.name:
                opposite_crank = j
                break

        self.assertIsNotNone(opposite_crank)
        # Original crank at x=1, mirrored across x=2 should be at x=3
        self.assertEqual(opposite_crank.x, 2 * 2.0 - self.crank.x)

    def test_add_opposite_leg_fixed_angle_negated(self):
        """Test that Fixed joints have negated angles when mirrored."""
        original_angle = self.output.angle
        self.walker.add_opposite_leg()

        # Find opposite output (Fixed joint)
        opposite_output = None
        for j in self.walker.joints:
            if "output" in j.name.lower() and "(opposite)" in j.name:
                opposite_output = j
                break

        self.assertIsNotNone(opposite_output)
        # Angle should be negated
        self.assertEqual(opposite_output.angle, -original_angle)

    def test_add_opposite_leg_names_have_opposite_suffix(self):
        """Test that opposite joints have '(opposite)' in name."""
        self.walker.add_opposite_leg()

        opposite_count = sum(
            1 for j in self.walker.joints if "(opposite)" in j.name
        )
        # Should have some opposite joints (not all, as base is shared)
        self.assertGreater(opposite_count, 0)

    def test_add_opposite_leg_solve_order_updated(self):
        """Test that solve order is updated after adding opposite leg."""
        initial_solve_count = len(self.walker._solve_order)
        self.walker.add_opposite_leg()
        # Solve order should increase
        self.assertGreater(len(self.walker._solve_order), initial_solve_count)

    def test_add_opposite_then_add_legs(self):
        """Test that add_opposite_leg and add_legs can be combined."""
        # Add opposite first to create left/right legs
        self.walker.add_opposite_leg()
        opposite_count = len(self.walker.joints)

        # Then add phase-offset copies
        self.walker.add_legs(1)

        # Should have more joints after adding legs
        self.assertGreater(len(self.walker.joints), opposite_count)

    def test_static_joints_on_axis_are_shared(self):
        """Test that Static joints on the mirror axis are not duplicated."""
        # Base is at x=0, which is on the default axis
        initial_base_count = sum(
            1 for j in self.walker.joints
            if isinstance(j, Static) and j.name == "base"
        )
        self.walker.add_opposite_leg(axis_x=0.0)

        # Base should still only appear once (not duplicated)
        final_base_count = sum(
            1 for j in self.walker.joints
            if isinstance(j, Static) and j.name == "base"
        )
        self.assertEqual(initial_base_count, final_base_count)

    def test_mirror_leg_alias_works(self):
        """Test that mirror_leg is an alias for add_opposite_leg."""
        initial_count = len(self.walker.joints)
        self.walker.mirror_leg()  # Should work as alias
        self.assertGreater(len(self.walker.joints), initial_count)


if __name__ == "__main__":
    unittest.main()
