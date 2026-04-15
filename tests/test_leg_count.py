#!/usr/bin/env python3
"""Tests for sweep_leg_counts."""
import unittest
from copy import deepcopy

from leggedsnake import DistanceFitness, Walker, sweep_leg_counts
from leggedsnake.fitness import FitnessResult


class TestSweepLegCounts(unittest.TestCase):
    """Verify the post-design leg-count sweep utility."""

    def _walker(self) -> Walker:
        # Chebyshev gives a fast, stable 4-bar baseline.
        return Walker.from_chebyshev()

    def test_returns_ordered_dict_keyed_by_leg_count(self):
        class StubObjective:
            n_legs = 1

            def __call__(self, topology, dimensions, config=None):
                return FitnessResult(
                    score=float(len(topology.driver_nodes())),
                    valid=True,
                )

        walker = self._walker()
        objective = StubObjective()
        results = sweep_leg_counts(
            walker, objective, n_legs_range=range(1, 4),
        )
        self.assertEqual(list(results.keys()), [1, 2, 3])
        for n, res in results.items():
            self.assertIsInstance(res, FitnessResult)
            # Each extra leg adds one DRIVER.
            self.assertEqual(res.score, float(n))

    def test_restores_objective_n_legs(self):
        fitness = DistanceFitness(duration=1.0, n_legs=4)
        walker = self._walker()
        # Swap in a stub call so we don't actually simulate.
        captured: list[int] = []

        def _stub(self, topology, dimensions, config=None):
            captured.append(self.n_legs)
            return FitnessResult(score=0.0, valid=True)

        original_call = type(fitness).__call__
        type(fitness).__call__ = _stub  # type: ignore[method-assign]
        try:
            sweep_leg_counts(walker, fitness, n_legs_range=(2, 3))
        finally:
            type(fitness).__call__ = original_call  # type: ignore[method-assign]

        # The sweep temporarily forces n_legs=1 to avoid double-adding.
        self.assertEqual(captured, [1, 1])
        # Original value is restored after the sweep.
        self.assertEqual(fitness.n_legs, 4)

    def test_does_not_mutate_input_walker(self):
        class StubObjective:
            n_legs = 1

            def __call__(self, topology, dimensions, config=None):
                return FitnessResult(score=0.0, valid=True)

        walker = self._walker()
        before = deepcopy(walker.topology)
        sweep_leg_counts(walker, StubObjective(), n_legs_range=(2, 3))
        self.assertEqual(len(walker.topology.nodes), len(before.nodes))
        self.assertEqual(len(walker.topology.edges), len(before.edges))

    def test_opposite_leg_flag_doubles_legs(self):
        class StubObjective:
            n_legs = 1

            def __call__(self, topology, dimensions, config=None):
                return FitnessResult(
                    score=float(len(topology.driver_nodes())),
                    valid=True,
                )

        walker = self._walker()
        results = sweep_leg_counts(
            walker, StubObjective(),
            n_legs_range=(1,),
            opposite_leg=True,
        )
        # 1 driver + 1 opposite mirror = 2.
        self.assertEqual(results[1].score, 2.0)


if __name__ == "__main__":
    unittest.main()
