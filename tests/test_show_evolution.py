#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Tests for the show_evolution module.
"""

import unittest
import os
import tempfile
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend for testing
import matplotlib.pyplot as plt

from leggedsnake.show_evolution import (
    draw_any_func, draw_median_score, draw_best_score,
    draw_standard_deviation, draw_population, draw_diversity,
    load_data
)


class TestDrawFunctions(unittest.TestCase):
    """Test drawing functions."""

    def setUp(self):
        """Create sample data and figure for testing."""
        self.fig, self.ax = plt.subplots()
        self.scores = [
            [1.0, 2.0, 3.0],
            [2.0, 3.0, 4.0],
            [3.0, 4.0, 5.0]
        ]
        self.dimensions = [
            [[1.0, 2.0], [1.5, 2.5]],
            [[1.1, 2.1], [1.6, 2.6]],
            [[1.2, 2.2], [1.7, 2.7]]
        ]

    def tearDown(self):
        """Close the figure after each test."""
        plt.close(self.fig)

    def test_draw_any_func(self):
        """Test draw_any_func with max function."""
        draw_any_func(self.ax, self.scores, max)
        # Should have added a line to the axis
        self.assertEqual(len(self.ax.lines), 1)

    def test_draw_any_func_min(self):
        """Test draw_any_func with min function."""
        draw_any_func(self.ax, self.scores, min)
        self.assertEqual(len(self.ax.lines), 1)

    def test_draw_median_score(self):
        """Test draw_median_score."""
        draw_median_score(self.ax, self.scores)
        self.assertEqual(len(self.ax.lines), 1)

    def test_draw_best_score(self):
        """Test draw_best_score."""
        draw_best_score(self.ax, self.scores)
        self.assertEqual(len(self.ax.lines), 1)

    def test_draw_standard_deviation(self):
        """Test draw_standard_deviation."""
        draw_standard_deviation(self.ax, self.scores)
        self.assertEqual(len(self.ax.lines), 1)

    def test_draw_population(self):
        """Test draw_population."""
        populations = [[1, 2, 3], [1, 2], [1, 2, 3, 4]]
        draw_population(self.ax, populations)
        self.assertEqual(len(self.ax.lines), 1)

    def test_draw_diversity(self):
        """Test draw_diversity."""
        draw_diversity(self.ax, self.dimensions)
        self.assertEqual(len(self.ax.lines), 1)


class TestLoadData(unittest.TestCase):
    """Test load_data function."""

    def setUp(self):
        """Create a temporary JSON file for testing."""
        self.temp_file = tempfile.NamedTemporaryFile(
            mode='w', suffix='.json', delete=False
        )
        # Use lists instead of tuples since JSON converts tuples to lists
        self.data = [
            {
                "turn": 0,
                "population": [
                    [1.0, [1.0, 2.0], [[0, 0], [1, 1]]],
                    [2.0, [2.0, 3.0], [[1, 1], [2, 2]]]
                ]
            },
            {
                "turn": 1,
                "population": [
                    [1.5, [1.2, 2.2], [[0, 0], [1, 1]]],
                    [2.5, [2.2, 3.2], [[1, 1], [2, 2]]]
                ]
            }
        ]
        json.dump(self.data, self.temp_file)
        self.temp_file.close()
        self.file_path = self.temp_file.name

    def tearDown(self):
        """Clean up temporary file."""
        if os.path.exists(self.file_path):
            os.remove(self.file_path)

    def test_load_data(self):
        """Test loading data from JSON file."""
        loaded = load_data(self.file_path)
        self.assertEqual(loaded, self.data)

    def test_load_data_structure(self):
        """Test that loaded data has correct structure."""
        loaded = load_data(self.file_path)
        self.assertEqual(len(loaded), 2)
        self.assertIn("turn", loaded[0])
        self.assertIn("population", loaded[0])


class TestDrawFunctionsWithNaN(unittest.TestCase):
    """Test drawing functions with NaN values."""

    def setUp(self):
        """Create sample data with NaN values."""
        self.fig, self.ax = plt.subplots()
        self.scores_with_nan = [
            [1.0, float('nan'), 3.0],
            [2.0, 3.0, float('nan')],
            [float('nan'), 4.0, 5.0]
        ]

    def tearDown(self):
        """Close the figure after each test."""
        plt.close(self.fig)

    def test_draw_median_score_with_nan(self):
        """Test draw_median_score handles NaN values."""
        # Should not raise exception
        draw_median_score(self.ax, self.scores_with_nan)
        self.assertEqual(len(self.ax.lines), 1)

    def test_draw_best_score_with_nan(self):
        """Test draw_best_score handles NaN values."""
        draw_best_score(self.ax, self.scores_with_nan)
        self.assertEqual(len(self.ax.lines), 1)

    def test_draw_standard_deviation_with_nan(self):
        """Test draw_standard_deviation handles NaN values."""
        draw_standard_deviation(self.ax, self.scores_with_nan)
        self.assertEqual(len(self.ax.lines), 1)


class TestDrawFunctionsWithInf(unittest.TestCase):
    """Test drawing functions with infinite values."""

    def setUp(self):
        """Create sample data with inf values."""
        self.fig, self.ax = plt.subplots()
        self.scores_with_inf = [
            [1.0, float('inf'), 3.0],
            [2.0, 3.0, float('-inf')],
            [4.0, 5.0, 6.0]
        ]

    def tearDown(self):
        """Close the figure after each test."""
        plt.close(self.fig)

    def test_draw_any_func_with_inf(self):
        """Test draw_any_func with inf values."""
        # Using np.nanmax should handle inf
        draw_any_func(self.ax, self.scores_with_inf, np.nanmax)
        self.assertEqual(len(self.ax.lines), 1)


if __name__ == "__main__":
    unittest.main()
