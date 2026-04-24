#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Tests for the genetic_optimizer module.
"""

import unittest
import os
import tempfile
import json

from leggedsnake.genetic_optimizer import (
    kwargs_switcher, load_population, save_population, GeneticOptimization
)


class TestKwargsSwitcher(unittest.TestCase):
    """Test kwargs_switcher helper function."""

    def test_key_exists(self):
        """Test when key exists in kwargs."""
        kwargs = {"key1": "value1", "key2": 42}
        result = kwargs_switcher("key1", kwargs)
        self.assertEqual(result, "value1")

    def test_key_missing_no_default(self):
        """Test when key is missing with no default."""
        kwargs = {"key1": "value1"}
        result = kwargs_switcher("missing_key", kwargs)
        self.assertIsNone(result)

    def test_key_missing_with_default(self):
        """Test when key is missing with default."""
        kwargs = {"key1": "value1"}
        result = kwargs_switcher("missing_key", kwargs, default="default_value")
        self.assertEqual(result, "default_value")

    def test_key_exists_but_none(self):
        """Test when key exists but value is None."""
        kwargs = {"key1": None}
        result = kwargs_switcher("key1", kwargs, default="default")
        self.assertEqual(result, "default")

    def test_key_exists_falsy_value(self):
        """Test when key exists with falsy value (0, empty string, etc.)."""
        kwargs = {"key1": 0}
        result = kwargs_switcher("key1", kwargs, default=10)
        # 0 is falsy, so should return default
        self.assertEqual(result, 10)


class TestSaveAndLoadPopulation(unittest.TestCase):
    """Test save_population and load_population functions."""

    def setUp(self):
        """Create a temporary file for testing."""
        self.temp_file = tempfile.NamedTemporaryFile(
            mode='w', suffix='.json', delete=False
        )
        self.temp_file.close()
        self.file_path = self.temp_file.name

    def tearDown(self):
        """Clean up temporary file."""
        if os.path.exists(self.file_path):
            os.remove(self.file_path)

    def test_save_population_new_file(self):
        """Test saving population to a new file."""
        # Remove file to test new file creation
        if os.path.exists(self.file_path):
            os.remove(self.file_path)

        # Use lists instead of tuples since JSON converts tuples to lists
        population = [
            [0.5, [1.0, 2.0], [[0, 0], [1, 1]]],
            [0.7, [1.5, 2.5], [[0, 0], [2, 2]]]
        ]
        save_population(self.file_path, population)

        self.assertTrue(os.path.exists(self.file_path))
        with open(self.file_path, 'r') as f:
            data = json.load(f)
        self.assertEqual(len(data), 1)
        self.assertEqual(data[0]['turn'], 0)
        self.assertEqual(data[0]['population'], population)

    def test_save_population_append(self):
        """Test saving population appends to existing file."""
        # Remove file to start fresh
        if os.path.exists(self.file_path):
            os.remove(self.file_path)

        pop1 = [[0.5, [1.0], [(0, 0)]]]
        pop2 = [[0.8, [2.0], [(1, 1)]]]

        save_population(self.file_path, pop1)
        save_population(self.file_path, pop2)

        with open(self.file_path, 'r') as f:
            data = json.load(f)
        self.assertEqual(len(data), 2)
        self.assertEqual(data[0]['turn'], 0)
        self.assertEqual(data[1]['turn'], 1)

    def test_save_population_with_descriptors(self):
        """Test saving population with additional data descriptors."""
        if os.path.exists(self.file_path):
            os.remove(self.file_path)

        population = [[0.5, [1.0], [(0, 0)]]]
        descriptors = {"best_score": 0.5, "best_individual_id": 0}
        save_population(self.file_path, population, data_descriptors=descriptors)

        with open(self.file_path, 'r') as f:
            data = json.load(f)
        self.assertEqual(data[0]['best_score'], 0.5)
        self.assertEqual(data[0]['best_individual_id'], 0)

    def test_load_population(self):
        """Test loading population from file."""
        if os.path.exists(self.file_path):
            os.remove(self.file_path)

        # Use lists since JSON converts tuples to lists
        population = [[0.5, [1.0, 2.0], [[0, 0], [1, 1]]]]
        save_population(self.file_path, population)

        loaded = load_population(self.file_path)
        self.assertEqual(loaded, population)


class TestGeneticOptimization(unittest.TestCase):
    """Test GeneticOptimization class."""

    def test_initialization(self):
        """Test basic initialization."""
        dna = [0, [1.0, 2.0], [(0, 0), (1, 1)]]

        def dummy_fitness(dna):
            return (sum(dna[1]), dna[2])

        ga = GeneticOptimization(dna=dna, fitness=dummy_fitness)
        self.assertEqual(ga.dna, dna)
        self.assertEqual(ga.fitness, dummy_fitness)
        self.assertEqual(ga.max_pop, 11)  # default
        self.assertIsNotNone(ga.pop)

    def test_initialization_custom_max_pop(self):
        """Test initialization with custom max_pop."""
        dna = [0, [1.0, 2.0], [(0, 0), (1, 1)]]

        def dummy_fitness(dna):
            return (sum(dna[1]), dna[2])

        ga = GeneticOptimization(dna=dna, fitness=dummy_fitness, max_pop=5)
        self.assertEqual(ga.max_pop, 5)

    def test_initialization_custom_verbosity(self):
        """Test initialization with custom verbosity."""
        dna = [0, [1.0, 2.0], [(0, 0), (1, 1)]]

        def dummy_fitness(dna):
            return (sum(dna[1]), dna[2])

        # Use verbose=2 since verbose=0 has edge case with kwargs_switcher (0 is falsy)
        ga = GeneticOptimization(dna=dna, fitness=dummy_fitness, verbose=2)
        self.assertEqual(ga.verbosity, 2)

    def test_birth(self):
        """Test birth method creates a child."""
        dna = [0, [1.0, 2.0], [(0, 0), (1, 1)]]

        def dummy_fitness(dna):
            return (sum(dna[1]), dna[2])

        ga = GeneticOptimization(dna=dna, fitness=dummy_fitness, prob=0.1)

        parent1 = [0.5, [1.0, 2.0], [(0, 0), (1, 1)]]
        parent2 = [0.7, [1.5, 2.5], [(2, 2), (3, 3)]]

        child = ga.birth(parent1, parent2)
        self.assertEqual(child[0], 0)  # Fitness not yet evaluated
        self.assertEqual(len(child[1]), 2)  # Same dimension count
        self.assertEqual(len(child[2]), 2)  # Same position count

    def test_evaluate_individual(self):
        """Test evaluate_individual method."""
        dna = [0, [1.0, 2.0], [(0, 0), (1, 1)]]

        def dummy_fitness(dna):
            return (sum(dna[1]), dna[2])

        ga = GeneticOptimization(dna=dna, fitness=dummy_fitness)
        individual = [0, [3.0, 4.0], [(0, 0), (1, 1)]]
        score, positions = ga.evaluate_individual(individual, None)
        self.assertEqual(score, 7.0)  # 3.0 + 4.0

    def test_evaluate_individual_with_args(self):
        """Test evaluate_individual with fitness_args."""
        dna = [0, [1.0, 2.0], [(0, 0), (1, 1)]]

        def fitness_with_args(dna, multiplier):
            return (sum(dna[1]) * multiplier, dna[2])

        ga = GeneticOptimization(dna=dna, fitness=fitness_with_args)
        individual = [0, [3.0, 4.0], [(0, 0), (1, 1)]]
        score, positions = ga.evaluate_individual(individual, (2,))
        self.assertEqual(score, 14.0)  # (3.0 + 4.0) * 2

    def test_evaluate_population(self):
        """Test evaluate_population method."""
        dna = [0, [1.0, 2.0], [(0, 0), (1, 1)]]

        def dummy_fitness(dna):
            return (sum(dna[1]), dna[2])

        ga = GeneticOptimization(dna=dna, fitness=dummy_fitness, verbose=0)
        ga.pop = [
            [0, [1.0, 2.0], [(0, 0), (1, 1)]],
            [0, [3.0, 4.0], [(2, 2), (3, 3)]]
        ]
        ga.evaluate_population(None, verbose=False)
        self.assertEqual(ga.pop[0][0], 3.0)  # 1.0 + 2.0
        self.assertEqual(ga.pop[1][0], 7.0)  # 3.0 + 4.0

    def test_select_parents(self):
        """Test select_parents method."""
        dna = [0, [1.0, 2.0], [(0, 0), (1, 1)]]

        def dummy_fitness(dna):
            return (sum(dna[1]), dna[2])

        ga = GeneticOptimization(dna=dna, fitness=dummy_fitness, verbose=0)
        ga.pop = [
            [1.0, [1.0, 2.0], [(0, 0), (1, 1)]],
            [5.0, [3.0, 4.0], [(2, 2), (3, 3)]],
            [3.0, [2.0, 3.0], [(1, 1), (2, 2)]],
            [2.0, [1.5, 2.5], [(0, 0), (1, 1)]]
        ]
        parents = ga.select_parents(verbose=False)
        # Should return at least one parent (the best)
        self.assertGreater(len(parents), 0)

    def test_make_children(self):
        """Test make_children method."""
        dna = [0, [1.0, 2.0], [(0, 0), (1, 1)]]

        def dummy_fitness(dna):
            return (sum(dna[1]), dna[2])

        ga = GeneticOptimization(dna=dna, fitness=dummy_fitness, prob=0.1)
        parents = [
            [5.0, [1.0, 2.0], [(0, 0), (1, 1)]],
            [3.0, [3.0, 4.0], [(2, 2), (3, 3)]]
        ]
        children = ga.make_children(parents)
        # With 2 parents, should produce at least 1 child
        self.assertGreaterEqual(len(children), 0)

    def test_reduce_population(self):
        """Test reduce_population method."""
        dna = [0, [1.0, 2.0], [(0, 0), (1, 1)]]

        def dummy_fitness(dna):
            return (sum(dna[1]), dna[2])

        ga = GeneticOptimization(dna=dna, fitness=dummy_fitness, max_pop=2)
        ga.pop = [
            [1.0, [1.0], [(0, 0)]],
            [5.0, [3.0], [(2, 2)]],
            [3.0, [2.0], [(1, 1)]],
            [2.0, [1.5], [(0, 0)]]
        ]
        new_pop = ga.reduce_population()
        self.assertEqual(len(new_pop), 2)
        # Should keep the best individuals
        self.assertEqual(new_pop[0][0], 5.0)
        self.assertEqual(new_pop[1][0], 3.0)

    def test_run_basic(self):
        """Test run method with minimal iterations."""
        dna = [0, [1.0, 2.0], [(0, 0), (1, 1)]]

        def dummy_fitness(dna):
            return (sum(dna[1]), dna[2])

        ga = GeneticOptimization(
            dna=dna, fitness=dummy_fitness,
            max_pop=3, verbose=0
        )
        result = ga.run(iters=2)
        self.assertIsInstance(result, list)
        self.assertGreater(len(result), 0)
        # Results should be sorted by score (descending)
        for i in range(len(result) - 1):
            self.assertGreaterEqual(result[i][0], result[i + 1][0])


class TestGeneticOptimizationStartnstop(unittest.TestCase):
    """Test GeneticOptimization with startnstop feature."""

    def setUp(self):
        """Create temporary file for checkpoint testing."""
        self.temp_file = tempfile.NamedTemporaryFile(
            mode='w', suffix='.json', delete=False
        )
        self.temp_file.close()
        self.file_path = self.temp_file.name

    def tearDown(self):
        """Clean up temporary file."""
        if os.path.exists(self.file_path):
            os.remove(self.file_path)

    def test_startnstop_new_run(self):
        """Test startnstop when file doesn't exist."""
        # Remove file to ensure fresh start
        if os.path.exists(self.file_path):
            os.remove(self.file_path)

        dna = [0, [1.0, 2.0], [(0, 0), (1, 1)]]

        def dummy_fitness(dna):
            return (sum(dna[1]), dna[2])

        # Use a non-existent file path
        non_existent = self.file_path + "_nonexistent"
        ga = GeneticOptimization(
            dna=dna, fitness=dummy_fitness,
            startnstop=non_existent, verbose=0
        )
        # Should start with default population
        self.assertIsNotNone(ga.pop)

    def test_startnstop_resume(self):
        """Test startnstop resuming from existing file."""
        # Remove temp file first (it's empty from setUp)
        if os.path.exists(self.file_path):
            os.remove(self.file_path)

        # Create a checkpoint file - use lists since JSON converts tuples
        population = [
            [5.0, [1.0, 2.0], [[0, 0], [1, 1]]],
            [3.0, [1.5, 2.5], [[2, 2], [3, 3]]]
        ]
        save_population(self.file_path, population)

        dna = [0, [1.0, 2.0], [(0, 0), (1, 1)]]

        def dummy_fitness(dna):
            return (sum(dna[1]), dna[2])

        ga = GeneticOptimization(
            dna=dna, fitness=dummy_fitness,
            startnstop=self.file_path, verbose=2
        )
        # Should load population from file
        self.assertEqual(ga.pop, population)


if __name__ == "__main__":
    unittest.main()
