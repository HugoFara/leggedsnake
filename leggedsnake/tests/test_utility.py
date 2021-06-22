#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 21 13:55:59 2021.

@author: HugoFara
"""

import unittest
from ..utility import stride, step


class TestStride(unittest.TestCase):
    """Test suite for stride function."""
    # A square
    locus = [(0, 0), (-1, 0), (-1, 1), (0, 1), (0, .5)]

    def test_minimal_stride(self):
        result = stride(self.locus, height=.1)
        self.assertEqual(len(result), 2)
        self.assertEqual(result, self.locus[:2])

    def test_ambiguous_stride(self):
        result = stride(self.locus, height=.5)
        self.assertEqual(result, self.locus[:2] + self.locus[-1:])

    def test_maximal_stride(self):
        result = stride(self.locus, height=2)
        self.assertEqual(result, self.locus)


class TestStep(unittest.TestCase):
    """Test suite for step function."""
    # A square
    locus = [(0, 0), (-1, 0), (-1, 1), (0, 1), (0, .5)]

    def test_minimal_step(self):
        result = step(self.locus, height=0, size=.5)
        self.assertTrue(result)

    def test_ambiguous_step(self):
        result = step(self.locus, height=1, size=1)
        self.assertTrue(result)

    def test_maximal_step(self):
        result = stride(self.locus, height=1, size=2)
        self.assertFalse(result)


if __name__ == '__main__':
    unittest.main()
