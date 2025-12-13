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


if __name__ == "__main__":
    unittest.main()
