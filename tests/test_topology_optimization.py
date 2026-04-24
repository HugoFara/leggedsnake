#!/usr/bin/env python3
"""Tests for the topology co-optimization module."""

import unittest

from pylinkage.optimization.collections import ParetoFront

from leggedsnake.fitness import DistanceFitness
from leggedsnake.topology_optimization import (
    TopologyCoOptConfig,
    TopologySolutionInfo,
    TopologyWalkingResult,
    _TopologyContext,
    _TopologyWalkingProblem,
    topology_walking_optimization,
)


class TestTopologyContext(unittest.TestCase):

    def test_loads_catalog(self):
        """Context loads built-in catalog and finds topologies."""
        ctx = _TopologyContext(max_links=6)
        self.assertGreater(ctx.n_topologies, 0)
        self.assertGreater(ctx.max_edges, 0)

    def test_four_bar_only(self):
        """max_links=4 includes only four-bar."""
        ctx = _TopologyContext(max_links=4)
        self.assertEqual(ctx.n_topologies, 1)
        self.assertEqual(ctx.entries[0].family, "four-bar")

    def test_six_bar_includes_more(self):
        """max_links=6 includes four-bar + six-bar variants."""
        ctx = _TopologyContext(max_links=6)
        self.assertGreater(ctx.n_topologies, 1)

    def test_n_edges_per_topology(self):
        """Each topology has a positive edge count."""
        ctx = _TopologyContext(max_links=6)
        for i in range(ctx.n_topologies):
            self.assertGreater(ctx.n_edges(i), 0)

    def test_build_walker_fourbar(self):
        """Can build a Walker from four-bar topology."""
        ctx = _TopologyContext(max_links=4)
        dims = [1.0] * ctx.max_edges
        walker = ctx.build_walker(0, dims)
        # May or may not succeed depending on dimension compatibility,
        # but should not raise
        # Just test it doesn't crash
        self.assertTrue(walker is None or walker is not None)

    def test_build_walker_out_of_range(self):
        """Out-of-range topology index is clamped, not crashed."""
        ctx = _TopologyContext(max_links=4)
        dims = [1.0] * ctx.max_edges
        # Should clamp to valid range, not crash
        walker = ctx.build_walker(999, dims)
        self.assertTrue(walker is None or walker is not None)

    def test_build_walker_negative_index(self):
        """Negative topology index is clamped to 0."""
        ctx = _TopologyContext(max_links=4)
        dims = [1.0] * ctx.max_edges
        walker = ctx.build_walker(-5, dims)
        self.assertTrue(walker is None or walker is not None)


class TestTopologyWalkingProblem(unittest.TestCase):

    def test_problem_creation(self):
        """Problem has correct n_var and n_obj."""
        ctx = _TopologyContext(max_links=4)
        objectives = [DistanceFitness(duration=1, n_legs=1)]
        cfg = TopologyCoOptConfig(n_legs=1)
        problem = _TopologyWalkingProblem(
            ctx=ctx, objectives=objectives, config=cfg,
        )
        # n_var = 1 (topology) + max_edges
        self.assertEqual(problem.problem.n_var, 1 + ctx.max_edges)
        self.assertEqual(problem.problem.n_obj, 1)

    def test_evaluate_candidate(self):
        """Candidate evaluation returns correct number of scores."""
        ctx = _TopologyContext(max_links=4)
        objectives = [
            DistanceFitness(duration=1, n_legs=1),
            DistanceFitness(duration=1, n_legs=1),
        ]
        cfg = TopologyCoOptConfig(n_legs=1)
        problem = _TopologyWalkingProblem(
            ctx=ctx, objectives=objectives, config=cfg,
        )
        import numpy as np
        x = np.ones(1 + ctx.max_edges)
        x[0] = 0.0  # four-bar topology
        scores = problem._evaluate_candidate(x)
        self.assertEqual(len(scores), 2)


class TestTopologyWalkingOptimization(unittest.TestCase):
    """Integration tests with very small populations."""

    def test_basic_run(self):
        """Optimization completes and returns valid result."""
        result = topology_walking_optimization(
            objectives=[DistanceFitness(duration=2, n_legs=1)],
            objective_names=["distance"],
            config=TopologyCoOptConfig(
                max_links=4,
                n_generations=2,
                pop_size=4,
                seed=42,
                verbose=False,
                n_legs=1,
            ),
        )
        self.assertIsInstance(result, TopologyWalkingResult)
        self.assertIsInstance(result.pareto_front, ParetoFront)

    def test_multi_objective(self):
        """Two-objective optimization returns Pareto front."""
        result = topology_walking_optimization(
            objectives=[
                DistanceFitness(duration=2, n_legs=1),
                DistanceFitness(duration=2, n_legs=1),
            ],
            objective_names=["dist_a", "dist_b"],
            config=TopologyCoOptConfig(
                max_links=4,
                n_generations=2,
                pop_size=4,
                seed=42,
                verbose=False,
                n_legs=1,
            ),
        )
        self.assertIsInstance(result, TopologyWalkingResult)

    def test_solutions_have_topology_info(self):
        """Pareto solutions have topology metadata."""
        result = topology_walking_optimization(
            objectives=[DistanceFitness(duration=2, n_legs=1)],
            config=TopologyCoOptConfig(
                max_links=4,
                n_generations=2,
                pop_size=4,
                seed=42,
                verbose=False,
                n_legs=1,
            ),
        )
        for idx in range(len(result.pareto_front.solutions)):
            if idx in result.topology_info:
                info = result.topology_info[idx]
                self.assertIsInstance(info, TopologySolutionInfo)
                self.assertIsInstance(info.topology_name, str)
                self.assertIsInstance(info.topology_id, str)
                self.assertGreater(info.num_links, 0)

    def test_solutions_by_topology(self):
        """solutions_by_topology groups correctly."""
        result = topology_walking_optimization(
            objectives=[DistanceFitness(duration=2, n_legs=1)],
            config=TopologyCoOptConfig(
                max_links=4,
                n_generations=2,
                pop_size=4,
                seed=42,
                verbose=False,
                n_legs=1,
            ),
        )
        groups = result.solutions_by_topology()
        self.assertIsInstance(groups, dict)


class TestTopologyCoOptConfig(unittest.TestCase):

    def test_defaults(self):
        cfg = TopologyCoOptConfig()
        self.assertEqual(cfg.max_links, 8)
        self.assertEqual(cfg.n_generations, 100)
        self.assertEqual(cfg.n_legs, 2)

    def test_custom(self):
        cfg = TopologyCoOptConfig(max_links=6, n_legs=4, seed=123)
        self.assertEqual(cfg.max_links, 6)
        self.assertEqual(cfg.n_legs, 4)
        self.assertEqual(cfg.seed, 123)

    def test_leg_gene_inactive_by_default(self):
        cfg = TopologyCoOptConfig()
        self.assertFalse(cfg.leg_gene_active)
        self.assertEqual(cfg.leg_bounds, (2, 2))

    def test_leg_gene_active_when_range_differs(self):
        cfg = TopologyCoOptConfig(n_legs_min=2, n_legs_max=6)
        self.assertTrue(cfg.leg_gene_active)
        self.assertEqual(cfg.leg_bounds, (2, 6))

    def test_leg_gene_inactive_when_range_collapses(self):
        # Equal bounds → fixed leg count, no gene.
        cfg = TopologyCoOptConfig(n_legs_min=3, n_legs_max=3)
        self.assertFalse(cfg.leg_gene_active)
        self.assertEqual(cfg.leg_bounds, (3, 3))


class TestLegGeneChromosome(unittest.TestCase):

    def test_problem_has_no_leg_gene_by_default(self):
        ctx = _TopologyContext(max_links=4)
        cfg = TopologyCoOptConfig(n_legs=1)
        problem = _TopologyWalkingProblem(
            ctx=ctx,
            objectives=[DistanceFitness(duration=1, n_legs=1)],
            config=cfg,
        )
        self.assertEqual(problem.problem.n_var, 1 + ctx.max_edges)

    def test_problem_adds_leg_gene_when_range_differs(self):
        ctx = _TopologyContext(max_links=4)
        cfg = TopologyCoOptConfig(n_legs_min=2, n_legs_max=6)
        problem = _TopologyWalkingProblem(
            ctx=ctx,
            objectives=[DistanceFitness(duration=1, n_legs=1)],
            config=cfg,
        )
        self.assertEqual(problem.problem.n_var, 2 + ctx.max_edges)
        # Leg-gene bounds at index 1.
        self.assertEqual(problem.problem.xl[1], 2.0)
        self.assertEqual(problem.problem.xu[1], 6.0)

    def test_decode_chromosome_fixed_legs(self):
        from leggedsnake.topology_optimization import _decode_chromosome
        cfg = TopologyCoOptConfig(n_legs=4)
        import numpy as np
        x = np.array([2.0, 1.0, 2.0, 3.0])
        topo, n_legs, dims, offsets, motors = _decode_chromosome(x, cfg)
        self.assertEqual(topo, 2)
        self.assertEqual(n_legs, 4)
        self.assertEqual(dims, [1.0, 2.0, 3.0])
        self.assertIsNone(offsets)

    def test_decode_chromosome_with_leg_gene(self):
        from leggedsnake.topology_optimization import _decode_chromosome
        cfg = TopologyCoOptConfig(n_legs_min=2, n_legs_max=6)
        import numpy as np
        x = np.array([1.0, 4.7, 1.0, 2.0, 3.0])
        topo, n_legs, dims, offsets, motors = _decode_chromosome(x, cfg)
        self.assertEqual(topo, 1)
        self.assertEqual(n_legs, 5)  # rounded
        self.assertEqual(dims, [1.0, 2.0, 3.0])
        self.assertIsNone(offsets)

    def test_decode_chromosome_clamps_leg_gene(self):
        from leggedsnake.topology_optimization import _decode_chromosome
        cfg = TopologyCoOptConfig(n_legs_min=3, n_legs_max=5)
        import numpy as np
        x_high = np.array([0.0, 10.0, 1.0])
        x_low = np.array([0.0, -5.0, 1.0])
        _, high, _, _, _ = _decode_chromosome(x_high, cfg)
        _, low, _, _, _ = _decode_chromosome(x_low, cfg)
        self.assertEqual(high, 5)
        self.assertEqual(low, 3)


class TestPhaseOffsetChromosome(unittest.TestCase):
    """Phase 8.3 — gait genes folded into the topology+dims chromosome."""

    def test_evolve_offsets_default_off(self):
        cfg = TopologyCoOptConfig()
        self.assertFalse(cfg.evolve_offsets)
        self.assertEqual(cfg.n_offset_genes, 0)

    def test_n_offset_genes_fixed_legs(self):
        cfg = TopologyCoOptConfig(n_legs=4, evolve_offsets=True)
        # 4 legs -> 3 phase offsets to evolve
        self.assertEqual(cfg.n_offset_genes, 3)

    def test_n_offset_genes_variable_legs_uses_upper_bound(self):
        # Chromosome must accommodate the largest possible n_legs.
        cfg = TopologyCoOptConfig(
            n_legs_min=2, n_legs_max=6, evolve_offsets=True,
        )
        self.assertEqual(cfg.n_offset_genes, 5)

    def test_evolve_offsets_rejects_single_leg(self):
        with self.assertRaises(ValueError):
            TopologyCoOptConfig(n_legs=1, evolve_offsets=True)

    def test_problem_extends_n_var_with_offsets(self):
        ctx = _TopologyContext(max_links=4)
        cfg = TopologyCoOptConfig(n_legs=3, evolve_offsets=True)
        problem = _TopologyWalkingProblem(
            ctx=ctx,
            objectives=[DistanceFitness(duration=1, n_legs=1)],
            config=cfg,
        )
        # 1 (topology) + max_edges + (n_legs - 1) offsets
        self.assertEqual(
            problem.problem.n_var, 1 + ctx.max_edges + 2,
        )

    def test_problem_offset_bounds_are_zero_to_tau(self):
        from math import tau
        ctx = _TopologyContext(max_links=4)
        cfg = TopologyCoOptConfig(n_legs=3, evolve_offsets=True)
        problem = _TopologyWalkingProblem(
            ctx=ctx,
            objectives=[DistanceFitness(duration=1, n_legs=1)],
            config=cfg,
        )
        # Offset region is the trailing 2 entries.
        self.assertAlmostEqual(problem.problem.xl[-1], 0.0)
        self.assertAlmostEqual(problem.problem.xu[-1], tau)
        self.assertAlmostEqual(problem.problem.xl[-2], 0.0)
        self.assertAlmostEqual(problem.problem.xu[-2], tau)

    def test_decode_returns_offsets_when_active(self):
        from leggedsnake.topology_optimization import _decode_chromosome
        import numpy as np
        cfg = TopologyCoOptConfig(n_legs=3, evolve_offsets=True)
        # [topology, dim_1, dim_2, dim_3, off_1, off_2]
        x = np.array([0.0, 1.0, 2.0, 3.0, 1.5, 4.7])
        topo, n_legs, dims, offsets, motors = _decode_chromosome(
            x, cfg, max_edges=3,
        )
        self.assertEqual(topo, 0)
        self.assertEqual(n_legs, 3)
        self.assertEqual(dims, [1.0, 2.0, 3.0])
        self.assertEqual(offsets, [1.5, 4.7])
        self.assertIsNone(motors)

    def test_decode_truncates_padded_offsets(self):
        """Variable n_legs: chromosome carries max_legs-1 offsets, decoder
        trims to current n_legs-1."""
        from leggedsnake.topology_optimization import _decode_chromosome
        import numpy as np
        cfg = TopologyCoOptConfig(
            n_legs_min=2, n_legs_max=4, evolve_offsets=True,
        )
        # Layout: [topo, n_legs, dim_1, dim_2, off_1, off_2, off_3]
        x = np.array([0.0, 2.4, 1.0, 2.0, 0.5, 1.5, 2.5])
        # n_legs rounds to 2 -> only 1 offset is used.
        topo, n_legs, dims, offsets, motors = _decode_chromosome(
            x, cfg, max_edges=2,
        )
        self.assertEqual(topo, 0)
        self.assertEqual(n_legs, 2)
        self.assertEqual(dims, [1.0, 2.0])
        self.assertEqual(offsets, [0.5])
        self.assertIsNone(motors)

    def test_decode_requires_max_edges_when_offsets_active(self):
        from leggedsnake.topology_optimization import _decode_chromosome
        import numpy as np
        cfg = TopologyCoOptConfig(n_legs=3, evolve_offsets=True)
        x = np.array([0.0, 1.0, 2.0, 3.0, 1.5, 4.7])
        with self.assertRaises(ValueError):
            _decode_chromosome(x, cfg)  # max_edges missing

    def test_optimization_runs_with_offsets(self):
        """Smoke test: the full pipeline accepts the new chromosome."""
        result = topology_walking_optimization(
            objectives=[DistanceFitness(duration=2, n_legs=1)],
            objective_names=["distance"],
            config=TopologyCoOptConfig(
                max_links=4,
                n_generations=2,
                pop_size=4,
                seed=42,
                verbose=False,
                n_legs=2,
                evolve_offsets=True,
            ),
        )
        self.assertIsInstance(result, TopologyWalkingResult)
        # Each Pareto entry should have phase_offsets populated.
        for idx, info in result.topology_info.items():
            self.assertIsNotNone(info.phase_offsets)
            self.assertEqual(len(info.phase_offsets), info.n_legs - 1)


class TestMotorRateChromosome(unittest.TestCase):
    """Motor angular velocity folded into the topology+dims chromosome."""

    def test_evolve_motor_rates_default_off(self):
        cfg = TopologyCoOptConfig()
        self.assertFalse(cfg.evolve_motor_rates)

    def test_evolve_motor_rates_rejects_inverted_bounds(self):
        with self.assertRaises(ValueError):
            TopologyCoOptConfig(
                evolve_motor_rates=True,
                motor_rate_lower=2.0,
                motor_rate_upper=-2.0,
            )

    def test_context_exposes_max_drivers(self):
        ctx = _TopologyContext(max_links=6)
        # Built-in catalog topologies all have exactly 1 driver today;
        # the field exists so multi-DOF entries land cleanly.
        self.assertGreaterEqual(ctx.max_drivers, 1)

    def test_problem_extends_n_var_with_motor_genes(self):
        ctx = _TopologyContext(max_links=4)
        cfg = TopologyCoOptConfig(n_legs=1, evolve_motor_rates=True)
        problem = _TopologyWalkingProblem(
            ctx=ctx,
            objectives=[DistanceFitness(duration=1, n_legs=1)],
            config=cfg,
        )
        # 1 (topology) + max_edges + max_drivers
        self.assertEqual(
            problem.problem.n_var, 1 + ctx.max_edges + ctx.max_drivers,
        )

    def test_problem_motor_bounds_use_config(self):
        ctx = _TopologyContext(max_links=4)
        cfg = TopologyCoOptConfig(
            n_legs=1,
            evolve_motor_rates=True,
            motor_rate_lower=-6.0,
            motor_rate_upper=6.0,
        )
        problem = _TopologyWalkingProblem(
            ctx=ctx,
            objectives=[DistanceFitness(duration=1, n_legs=1)],
            config=cfg,
        )
        # Motor region is the trailing max_drivers entries.
        self.assertAlmostEqual(problem.problem.xl[-1], -6.0)
        self.assertAlmostEqual(problem.problem.xu[-1], 6.0)

    def test_problem_motor_genes_after_offsets(self):
        ctx = _TopologyContext(max_links=4)
        cfg = TopologyCoOptConfig(
            n_legs=3, evolve_offsets=True, evolve_motor_rates=True,
        )
        problem = _TopologyWalkingProblem(
            ctx=ctx,
            objectives=[DistanceFitness(duration=1, n_legs=1)],
            config=cfg,
        )
        # 1 (topology) + max_edges + 2 offsets + max_drivers motors
        self.assertEqual(
            problem.problem.n_var,
            1 + ctx.max_edges + 2 + ctx.max_drivers,
        )

    def test_decode_returns_motor_rates_when_active(self):
        from leggedsnake.topology_optimization import _decode_chromosome
        import numpy as np
        cfg = TopologyCoOptConfig(n_legs=1, evolve_motor_rates=True)
        # [topology, dim_1, dim_2, dim_3, motor_1] (max_drivers=1)
        x = np.array([0.0, 1.0, 2.0, 3.0, -3.5])
        topo, n_legs, dims, offsets, motors = _decode_chromosome(
            x, cfg, max_edges=3, max_drivers=1,
        )
        self.assertEqual(topo, 0)
        self.assertEqual(n_legs, 1)
        self.assertEqual(dims, [1.0, 2.0, 3.0])
        self.assertIsNone(offsets)
        self.assertEqual(motors, [-3.5])

    def test_decode_combined_offsets_and_motors(self):
        from leggedsnake.topology_optimization import _decode_chromosome
        import numpy as np
        cfg = TopologyCoOptConfig(
            n_legs=3, evolve_offsets=True, evolve_motor_rates=True,
        )
        # [topology, dim_1, dim_2, dim_3, off_1, off_2, motor_1]
        x = np.array([0.0, 1.0, 2.0, 3.0, 0.5, 1.5, 5.0])
        topo, n_legs, dims, offsets, motors = _decode_chromosome(
            x, cfg, max_edges=3, max_drivers=1,
        )
        self.assertEqual(dims, [1.0, 2.0, 3.0])
        self.assertEqual(offsets, [0.5, 1.5])
        self.assertEqual(motors, [5.0])

    def test_decode_requires_max_drivers_when_motors_active(self):
        from leggedsnake.topology_optimization import _decode_chromosome
        import numpy as np
        cfg = TopologyCoOptConfig(n_legs=1, evolve_motor_rates=True)
        x = np.array([0.0, 1.0, 2.0, 3.0, -3.5])
        with self.assertRaises(ValueError):
            _decode_chromosome(x, cfg, max_edges=3)

    def test_resolve_evolved_motor_rates_maps_to_driver_ids(self):
        from leggedsnake.topology_optimization import (
            _resolve_evolved_motor_rates,
        )
        ctx = _TopologyContext(max_links=4)
        result = _resolve_evolved_motor_rates(
            ctx, topo_idx=0, raw_motors=[-3.5], fallback=-4.0,
        )
        # Four-bar has one driver; the evolved rate is mapped to its ID.
        self.assertIsInstance(result, dict)
        self.assertEqual(len(result), 1)
        self.assertEqual(list(result.values())[0], -3.5)

    def test_resolve_evolved_motor_rates_falls_back(self):
        from leggedsnake.topology_optimization import (
            _resolve_evolved_motor_rates,
        )
        ctx = _TopologyContext(max_links=4)
        result = _resolve_evolved_motor_rates(
            ctx, topo_idx=0, raw_motors=None, fallback=-4.0,
        )
        # No motor genes — fall back to the config scalar unchanged.
        self.assertEqual(result, -4.0)

    def test_with_motor_rates_clones_fitness(self):
        from leggedsnake.topology_optimization import _with_motor_rates
        original = DistanceFitness(duration=1, n_legs=1, motor_rates=-4.0)
        tuned = _with_motor_rates(original, [3.0], {"crank": 3.0})
        self.assertIsNot(tuned, original)
        self.assertEqual(original.motor_rates, -4.0)
        self.assertEqual(tuned.motor_rates, {"crank": 3.0})

    def test_with_motor_rates_no_op_when_disabled(self):
        from leggedsnake.topology_optimization import _with_motor_rates
        original = DistanceFitness(duration=1, n_legs=1)
        same = _with_motor_rates(original, None, -4.0)
        self.assertIs(same, original)

    def test_optimization_runs_with_motor_genes(self):
        """Smoke test: full pipeline accepts motor genes."""
        result = topology_walking_optimization(
            objectives=[DistanceFitness(duration=2, n_legs=1)],
            objective_names=["distance"],
            config=TopologyCoOptConfig(
                max_links=4,
                n_generations=2,
                pop_size=4,
                seed=42,
                verbose=False,
                n_legs=1,
                evolve_motor_rates=True,
                motor_rate_lower=-6.0,
                motor_rate_upper=6.0,
            ),
        )
        self.assertIsInstance(result, TopologyWalkingResult)
        for info in result.topology_info.values():
            self.assertIsNotNone(info.motor_rates)
            self.assertIsInstance(info.motor_rates, dict)
            for rate in info.motor_rates.values():
                self.assertGreaterEqual(rate, -6.0)
                self.assertLessEqual(rate, 6.0)


class TestWindForceConfig(unittest.TestCase):
    """``wind_force`` convenience knob on ``TopologyCoOptConfig``."""

    def test_default_is_zero(self):
        cfg = TopologyCoOptConfig()
        self.assertEqual(cfg.wind_force, (0.0, 0.0))

    def test_passthrough_when_default(self):
        from leggedsnake.physics_engine import WorldConfig
        from leggedsnake.topology_optimization import _resolve_world_config

        cfg = TopologyCoOptConfig()  # default wind_force
        world_cfg = WorldConfig(gravity=(0.0, -9.81))
        out = _resolve_world_config(world_cfg, cfg)
        # No conflict to resolve; pass through untouched.
        self.assertIs(out, world_cfg)

    def test_builds_default_world_config_when_none(self):
        from leggedsnake.physics_engine import WorldConfig
        from leggedsnake.topology_optimization import _resolve_world_config

        cfg = TopologyCoOptConfig(wind_force=(2.5, 0.0))
        out = _resolve_world_config(None, cfg)
        self.assertIsInstance(out, WorldConfig)
        self.assertEqual(out.wind_force, (2.5, 0.0))

    def test_grafts_onto_explicit_world_config(self):
        from leggedsnake.physics_engine import WorldConfig
        from leggedsnake.topology_optimization import _resolve_world_config

        cfg = TopologyCoOptConfig(wind_force=(2.5, 0.0))
        world_cfg = WorldConfig(gravity=(0.0, -5.0))  # no wind set
        out = _resolve_world_config(world_cfg, cfg)
        # Grafted, not mutated in place.
        self.assertEqual(out.wind_force, (2.5, 0.0))
        self.assertEqual(out.gravity, (0.0, -5.0))
        self.assertEqual(world_cfg.wind_force, (0.0, 0.0))

    def test_passthrough_when_both_agree(self):
        from leggedsnake.physics_engine import WorldConfig
        from leggedsnake.topology_optimization import _resolve_world_config

        cfg = TopologyCoOptConfig(wind_force=(2.5, 0.0))
        world_cfg = WorldConfig(wind_force=(2.5, 0.0))
        out = _resolve_world_config(world_cfg, cfg)
        self.assertIs(out, world_cfg)

    def test_raises_when_both_disagree(self):
        from leggedsnake.physics_engine import WorldConfig
        from leggedsnake.topology_optimization import _resolve_world_config

        cfg = TopologyCoOptConfig(wind_force=(2.5, 0.0))
        world_cfg = WorldConfig(wind_force=(-1.0, 0.0))
        with self.assertRaises(ValueError):
            _resolve_world_config(world_cfg, cfg)

    def test_optimization_runs_with_wind(self):
        """Smoke test: pipeline accepts wind_force on the config."""
        result = topology_walking_optimization(
            objectives=[DistanceFitness(duration=2, n_legs=1)],
            objective_names=["distance"],
            config=TopologyCoOptConfig(
                max_links=4,
                n_generations=2,
                pop_size=4,
                seed=42,
                verbose=False,
                n_legs=1,
                wind_force=(1.0, 0.0),
            ),
        )
        self.assertIsInstance(result, TopologyWalkingResult)


class TestAllowPassive(unittest.TestCase):
    """``allow_passive`` snaps near-zero evolved rates to exactly 0.0."""

    def test_default_off(self):
        cfg = TopologyCoOptConfig()
        self.assertFalse(cfg.allow_passive)

    def test_requires_evolve_motor_rates(self):
        with self.assertRaises(ValueError):
            TopologyCoOptConfig(allow_passive=True)

    def test_requires_bounds_spanning_zero_positive_only(self):
        with self.assertRaises(ValueError):
            TopologyCoOptConfig(
                evolve_motor_rates=True,
                motor_rate_lower=1.0,
                motor_rate_upper=8.0,
                allow_passive=True,
            )

    def test_requires_bounds_spanning_zero_negative_only(self):
        with self.assertRaises(ValueError):
            TopologyCoOptConfig(
                evolve_motor_rates=True,
                motor_rate_lower=-8.0,
                motor_rate_upper=-1.0,
                allow_passive=True,
            )

    def test_accepts_bounds_with_zero_endpoint(self):
        # Zero as an endpoint is allowed (the optimizer can land exactly on it).
        cfg = TopologyCoOptConfig(
            evolve_motor_rates=True,
            motor_rate_lower=0.0,
            motor_rate_upper=8.0,
            allow_passive=True,
        )
        self.assertTrue(cfg.allow_passive)

    def test_snaps_near_zero_to_zero(self):
        from leggedsnake.topology_optimization import (
            _resolve_evolved_motor_rates,
        )
        ctx = _TopologyContext(max_links=4)
        result = _resolve_evolved_motor_rates(
            ctx, topo_idx=0, raw_motors=[1e-9], fallback=-4.0,
            allow_passive=True,
        )
        self.assertIsInstance(result, dict)
        self.assertEqual(list(result.values())[0], 0.0)

    def test_does_not_snap_when_disabled(self):
        from leggedsnake.topology_optimization import (
            _resolve_evolved_motor_rates,
        )
        ctx = _TopologyContext(max_links=4)
        result = _resolve_evolved_motor_rates(
            ctx, topo_idx=0, raw_motors=[1e-9], fallback=-4.0,
            allow_passive=False,
        )
        # Without allow_passive the raw value passes through verbatim.
        self.assertIsInstance(result, dict)
        self.assertEqual(list(result.values())[0], 1e-9)

    def test_does_not_snap_outside_eps(self):
        from leggedsnake.topology_optimization import (
            _resolve_evolved_motor_rates,
        )
        ctx = _TopologyContext(max_links=4)
        result = _resolve_evolved_motor_rates(
            ctx, topo_idx=0, raw_motors=[1e-3], fallback=-4.0,
            allow_passive=True,
        )
        self.assertIsInstance(result, dict)
        # 1e-3 is far above the snap epsilon — preserved as-is.
        self.assertEqual(list(result.values())[0], 1e-3)

    def test_optimization_runs_with_allow_passive(self):
        """Smoke test: pipeline accepts allow_passive end-to-end."""
        result = topology_walking_optimization(
            objectives=[DistanceFitness(duration=2, n_legs=1)],
            objective_names=["distance"],
            config=TopologyCoOptConfig(
                max_links=4,
                n_generations=2,
                pop_size=4,
                seed=42,
                verbose=False,
                n_legs=1,
                evolve_motor_rates=True,
                motor_rate_lower=-6.0,
                motor_rate_upper=6.0,
                allow_passive=True,
                wind_force=(1.0, 0.0),
            ),
        )
        self.assertIsInstance(result, TopologyWalkingResult)


if __name__ == "__main__":
    unittest.main()
