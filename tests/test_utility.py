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
