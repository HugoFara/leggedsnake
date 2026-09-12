"""Tests for the walking-specific objective wrappers."""
import unittest

from pylinkage.optimization.collections import ParetoFront, ParetoSolution

import leggedsnake as ls
from leggedsnake.walking_objectives import multi_objective_walking_optimization


class TestMultiObjectiveWalkingOptimization(unittest.TestCase):
    """The wrapper honours its documented ``ParetoFront`` contract."""

    def test_returns_pareto_front(self):
        walker = ls.Walker.from_jansen()
        dims = walker.get_constraints()

        def total_length(linkage, d, pos):
            return float(sum(d))

        def spread(linkage, d, pos):
            return float(max(d) - min(d))

        front = multi_objective_walking_optimization(
            walker,
            objectives=[total_length, spread],
            bounds=([x * 0.8 for x in dims], [x * 1.2 for x in dims]),
            objective_names=["length", "spread"],
            n_generations=2,
            pop_size=8,
            seed=0,
            verbose=False,
        )

        self.assertIsInstance(front, ParetoFront)
        self.assertEqual(front.objective_names, ("length", "spread"))
        self.assertGreater(len(front.solutions), 0)
        best = front.best_compromise()
        self.assertIsInstance(best, ParetoSolution)
        self.assertEqual(len(best.dimensions), len(dims))
        self.assertEqual(len(best.scores), 2)


if __name__ == "__main__":
    unittest.main()
