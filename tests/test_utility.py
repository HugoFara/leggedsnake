#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 21 13:55:59 2021.

@author: HugoFara
"""

import unittest
from leggedsnake.utility import stride, step


class TestStride(unittest.TestCase):
    """Test suite for stride function."""
    # A square
    locus = [(0, 0), (-1, 0), (-1, 1), (0, 1), (0, .5)]

    def test_minimal_stride(self):
        """Test if we only get the three lowest points."""
        result = stride(self.locus, height=.1)
        self.assertEqual(len(result), 3)
        self.assertEqual(result, self.locus[-1:] + self.locus[:2])

    def test_ambiguous_stride(self):
        """
        Test if we only get all points but not the highest.

        A point on the limit shoud be discarded when some points are on the
        limit.
        """
        result = stride(self.locus, height=.5)
        self.assertEqual(result, self.locus[-2:] + self.locus[0:2])

    def test_maximal_stride(self):
        """Test if all points are retrivewed."""
        result = stride(self.locus, height=2)
        self.assertEqual(result, self.locus)


class TestStep(unittest.TestCase):
    """Test suite for step function."""
    # A square
    locus = [(0, 0), (-1, 0), (-1, 1), (0, 1), (0, .5)]

    def test_minimal_step(self):
        """Test if we can pass an obstacle small enough."""
        result = step(self.locus, height=0, width=.5)
        self.assertTrue(result)

    def test_ambiguous_step(self):
        """Test if successfully to pass an obstacle of the size of the locus."""
        result = step(self.locus, height=1, width=1)
        self.assertTrue(result)

    def test_streched_step(self):
        """Test if we fail to pass an obstacle to big."""
        result = step(self.locus, height=1, width=2)
        self.assertFalse(result)

    def test_dwarf_step(self):
        """Test if we fail to pass an obstacle to high."""
        result = step(self.locus, height=2, width=.5)
        self.assertFalse(result)

    def test_step_empty_points(self):
        """Test step with empty points list."""
        result = step([], height=1, width=1)
        self.assertEqual(result, [])

    def test_step_return_res_true(self):
        """Test step with return_res=True."""
        result = step(self.locus, height=0.5, width=0.5, return_res=True)
        # Should return the subset of points that can cross
        self.assertIsNotNone(result)

    def test_step_with_y_min(self):
        """Test step with explicit y_min."""
        result = step(self.locus, height=0.5, width=0.5, y_min=0)
        self.assertIsNotNone(result)


class TestStrideEdgeCases(unittest.TestCase):
    """Test edge cases for stride function."""

    def test_stride_single_point(self):
        """Test stride with single point."""
        locus = [(0, 0)]
        result = stride(locus, height=1)
        self.assertEqual(result, locus)

    def test_stride_two_points(self):
        """Test stride with two points."""
        locus = [(0, 0), (1, 0)]
        result = stride(locus, height=1)
        self.assertEqual(len(result), 2)

    def test_stride_negative_heights(self):
        """Test stride with points at negative heights."""
        locus = [(0, -2), (1, -1), (2, 0), (3, -1)]
        result = stride(locus, height=0.5)
        self.assertIsNotNone(result)


class TestStepEdgeCases(unittest.TestCase):
    """Test edge cases for step function."""

    def test_step_tall_locus(self):
        """Test step with tall locus."""
        locus = [(0, 0), (0, 1), (0, 2), (0, 3)]
        result = step(locus, height=2, width=0.1)
        # Can't cross because width is too small
        self.assertFalse(result)

    def test_step_wide_locus(self):
        """Test step with wide locus."""
        locus = [(0, 0), (1, 0), (2, 0), (3, 0)]
        result = step(locus, height=0.1, width=2)
        # Can't cross because height is too small
        self.assertFalse(result)

    def test_step_return_res_no_crossing(self):
        """Test step return_res when can't cross."""
        locus = [(0, 0), (1, 0)]
        result = step(locus, height=1, width=1, return_res=True)
        self.assertFalse(result)
