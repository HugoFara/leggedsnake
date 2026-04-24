#!/usr/bin/env python3
"""Tests for expanded terrain generation features."""

import unittest

import numpy as np

from leggedsnake.physics_engine import (
    SLOPE_PROFILES,
    SlopeProfile,
    TerrainConfig,
    TerrainPreset,
    World,
    WorldConfig,
)


class TestSeed(unittest.TestCase):
    """Seeded terrain must be reproducible."""

    def test_same_seed_same_road(self):
        cfg = WorldConfig(terrain=TerrainConfig(seed=42))
        w1 = World(config=cfg)
        w2 = World(config=cfg)
        for _ in range(20):
            w1.build_road(positive=True)
            w2.build_road(positive=True)
        self.assertEqual(w1.road, w2.road)

    def test_different_seed_different_road(self):
        w1 = World(config=WorldConfig(terrain=TerrainConfig(seed=1)))
        w2 = World(config=WorldConfig(terrain=TerrainConfig(seed=2)))
        for _ in range(20):
            w1.build_road(positive=True)
            w2.build_road(positive=True)
        self.assertNotEqual(w1.road, w2.road)

    def test_none_seed_nondeterministic(self):
        """Two worlds with seed=None should (almost certainly) diverge."""
        w1 = World(config=WorldConfig(terrain=TerrainConfig(seed=None)))
        w2 = World(config=WorldConfig(terrain=TerrainConfig(seed=None)))
        for _ in range(50):
            w1.build_road(positive=True)
            w2.build_road(positive=True)
        # Extremely unlikely to be identical
        self.assertNotEqual(w1.road, w2.road)


class TestStepGeneration(unittest.TestCase):
    """Discrete step generation (re-enabled)."""

    def test_steps_extend_road(self):
        terrain = TerrainConfig(step_freq=1.0, seed=7)
        w = World(config=WorldConfig(terrain=terrain))
        initial = len(w.road)
        w.build_road(positive=True)
        # Steps add 2 points
        self.assertEqual(len(w.road), initial + 2)

    def test_steps_go_upward(self):
        terrain = TerrainConfig(step_freq=1.0, max_step=1.0, seed=10)
        w = World(config=WorldConfig(terrain=terrain))
        base_y = w.road[-1][1]
        w.build_road(positive=True)
        # The step riser should be above the base
        self.assertGreaterEqual(w.road[-1][1], base_y)


class TestFrictionVariation(unittest.TestCase):
    """Per-segment friction randomization."""

    def test_friction_within_range(self):
        terrain = TerrainConfig(friction_range=(0.2, 0.8), seed=5)
        w = World(config=WorldConfig(terrain=terrain))
        for _ in range(30):
            w.build_road(positive=True)
        # Check that road segments have friction in range
        for shape in w.space.shapes:
            self.assertGreaterEqual(shape.friction, 0.2)
            self.assertLessEqual(shape.friction, 0.8)

    def test_friction_varies(self):
        terrain = TerrainConfig(friction_range=(0.1, 0.9), seed=3)
        w = World(config=WorldConfig(terrain=terrain))
        for _ in range(30):
            w.build_road(positive=True)
        frictions = {s.friction for s in w.space.shapes}
        # Should have more than one distinct value
        self.assertGreater(len(frictions), 1)

    def test_no_range_uses_default(self):
        terrain = TerrainConfig(friction_range=None, seed=1)
        cfg = WorldConfig(terrain=terrain, ground_friction=0.42)
        w = World(config=cfg)
        w.build_road(positive=True)
        # New segment should use the default ground_friction
        frictions = {round(s.friction, 4) for s in w.space.shapes}
        self.assertEqual(frictions, {0.42})


class TestTerrainPresets(unittest.TestCase):
    """Named terrain presets."""

    def test_all_presets_create_valid_config(self):
        for preset in TerrainPreset:
            cfg = TerrainConfig.from_preset(preset)
            self.assertIsInstance(cfg, TerrainConfig)

    def test_from_string(self):
        cfg = TerrainConfig.from_preset("flat")
        self.assertEqual(cfg.slope, 0.0)
        self.assertEqual(cfg.noise, 0.0)

    def test_flat_is_flat(self):
        cfg = TerrainConfig.from_preset(TerrainPreset.FLAT)
        cfg.seed = 99
        w = World(config=WorldConfig(terrain=cfg))
        for _ in range(20):
            w.build_road(positive=True)
        ys = [p[1] for p in w.road]
        # All heights should be identical on a flat road
        self.assertTrue(all(y == ys[0] for y in ys))

    def test_stairs_all_steps(self):
        cfg = TerrainConfig.from_preset(TerrainPreset.STAIRS)
        cfg.seed = 11
        w = World(config=WorldConfig(terrain=cfg))
        initial = len(w.road)
        for _ in range(5):
            w.build_road(positive=True)
        # Steps add 2 points each → 5 builds × 2 = 10
        self.assertEqual(len(w.road), initial + 10)


class TestGaps(unittest.TestCase):
    """Gap / chasm generation."""

    def test_gap_extends_road_without_segment(self):
        terrain = TerrainConfig(gap_freq=1.0, gap_width=2.0, seed=4)
        w = World(config=WorldConfig(terrain=terrain))
        shapes_before = len(w.space.shapes)
        initial_road = len(w.road)
        w.build_road(positive=True)
        self.assertEqual(len(w.road), initial_road + 1)
        # No new physics segment for a gap
        self.assertEqual(len(w.space.shapes), shapes_before)

    def test_gap_width(self):
        terrain = TerrainConfig(gap_freq=1.0, gap_width=3.0, seed=6)
        w = World(config=WorldConfig(terrain=terrain))
        last_x = w.road[-1][0]
        w.build_road(positive=True)
        self.assertAlmostEqual(abs(w.road[-1][0] - last_x), 3.0)


class TestObstacles(unittest.TestCase):
    """Obstacle / bump placement."""

    def test_obstacle_adds_segments(self):
        terrain = TerrainConfig(
            obstacle_freq=1.0, gap_freq=0.0, step_freq=0.0,
            obstacle_height=0.5, obstacle_width=0.4, seed=8,
        )
        w = World(config=WorldConfig(terrain=terrain))
        shapes_before = len(w.space.shapes)
        w.build_road(positive=True)
        # 3 new segments for left wall, top, right wall
        self.assertEqual(len(w.space.shapes), shapes_before + 3)

    def test_obstacle_extends_road(self):
        terrain = TerrainConfig(
            obstacle_freq=1.0, gap_freq=0.0, step_freq=0.0,
            obstacle_width=0.6, seed=9,
        )
        w = World(config=WorldConfig(terrain=terrain))
        last_x = w.road[-1][0]
        w.build_road(positive=True)
        self.assertAlmostEqual(abs(w.road[-1][0] - last_x), 0.6)


class TestSlopeProfiles(unittest.TestCase):
    """Deterministic slope profile generators."""

    def test_all_builtin_profiles_registered(self):
        for sp in SlopeProfile:
            self.assertIn(sp.value, SLOPE_PROFILES)

    def test_constant_profile(self):
        terrain = TerrainConfig(
            slope=0.2, slope_profile=SlopeProfile.CONSTANT, seed=1,
        )
        w = World(config=WorldConfig(terrain=terrain))
        for _ in range(10):
            w.build_road(positive=True)
        # All segments should have the same slope → monotonically rising
        ys = [p[1] for p in w.road]
        for i in range(1, len(ys)):
            self.assertGreaterEqual(ys[i], ys[i - 1] - 1e-9)

    def test_valley_profile_descends_then_ascends(self):
        terrain = TerrainConfig(
            slope=0.3, slope_profile=SlopeProfile.VALLEY, seed=2,
        )
        w = World(config=WorldConfig(terrain=terrain))
        for _ in range(40):
            w.build_road(positive=True)
        ys = [p[1] for p in w.road]
        # Should not be monotonic
        diffs = [ys[i + 1] - ys[i] for i in range(len(ys) - 1)]
        has_down = any(d < -1e-6 for d in diffs)
        has_up = any(d > 1e-6 for d in diffs)
        self.assertTrue(has_down and has_up)

    def test_custom_callable_profile(self):
        def always_flat(_t, _rng, _s):
            return 0.0

        terrain = TerrainConfig(
            slope_profile=always_flat, step_freq=0.0, seed=1,
        )
        w = World(config=WorldConfig(terrain=terrain))
        for _ in range(10):
            w.build_road(positive=True)
        ys = [p[1] for p in w.road]
        self.assertTrue(all(abs(y - ys[0]) < 1e-9 for y in ys))

    def test_string_profile_key(self):
        terrain = TerrainConfig(slope_profile="flat", step_freq=0.0, seed=1)
        w = World(config=WorldConfig(terrain=terrain))
        for _ in range(10):
            w.build_road(positive=True)
        ys = [p[1] for p in w.road]
        self.assertTrue(all(abs(y - ys[0]) < 1e-9 for y in ys))

    def test_sawtooth_profile(self):
        terrain = TerrainConfig(
            slope=0.3, slope_profile=SlopeProfile.SAWTOOTH, seed=3,
        )
        w = World(config=WorldConfig(terrain=terrain))
        for _ in range(25):
            w.build_road(positive=True)
        ys = [p[1] for p in w.road]
        diffs = [ys[i + 1] - ys[i] for i in range(len(ys) - 1)]
        has_down = any(d < -1e-6 for d in diffs)
        has_up = any(d > 1e-6 for d in diffs)
        self.assertTrue(has_down and has_up)


class TestSinusoidalProfile(unittest.TestCase):
    """Smooth sinusoidal slope generator."""

    def test_zero_at_origin(self):
        terrain = TerrainConfig(
            slope=0.5, wave_period=10.0, section_len=1.0,
        )
        gen = SLOPE_PROFILES[SlopeProfile.SINUSOIDAL.value]
        rng = np.random.default_rng(0)
        self.assertAlmostEqual(gen(terrain, rng, 0), 0.0, places=12)

    def test_quarter_period_is_amplitude(self):
        """At x = period/4 the sine peaks at +slope."""
        terrain = TerrainConfig(
            slope=0.5, wave_period=8.0, section_len=2.0,
        )
        gen = SLOPE_PROFILES[SlopeProfile.SINUSOIDAL.value]
        rng = np.random.default_rng(0)
        # x = step * section_len = 1 * 2 = 2 = wave_period / 4
        self.assertAlmostEqual(gen(terrain, rng, 1), 0.5, places=10)

    def test_half_period_returns_to_zero(self):
        terrain = TerrainConfig(
            slope=0.5, wave_period=8.0, section_len=2.0,
        )
        gen = SLOPE_PROFILES[SlopeProfile.SINUSOIDAL.value]
        rng = np.random.default_rng(0)
        # x = 2 * 2 = 4 = wave_period / 2
        self.assertAlmostEqual(gen(terrain, rng, 2), 0.0, places=10)

    def test_zero_period_collapses_to_flat(self):
        """Degenerate wave_period <= 0 yields no slope (no division blow-up)."""
        terrain = TerrainConfig(
            slope=0.5, wave_period=0.0, section_len=1.0,
        )
        gen = SLOPE_PROFILES[SlopeProfile.SINUSOIDAL.value]
        rng = np.random.default_rng(0)
        for step in (0, 1, 5, 17):
            self.assertEqual(gen(terrain, rng, step), 0.0)

    def test_road_undulates(self):
        """Built road has both up and down segments."""
        terrain = TerrainConfig(
            slope=0.3, wave_period=4.0, section_len=0.5,
            step_freq=0.0, slope_profile=SlopeProfile.SINUSOIDAL, seed=1,
        )
        w = World(config=WorldConfig(terrain=terrain))
        for _ in range(40):
            w.build_road(positive=True)
        ys = [p[1] for p in w.road]
        diffs = [ys[i + 1] - ys[i] for i in range(len(ys) - 1)]
        self.assertTrue(any(d < -1e-6 for d in diffs))
        self.assertTrue(any(d > 1e-6 for d in diffs))


class TestFrequencySweepProfile(unittest.TestCase):
    """Linear-chirp slope generator."""

    def test_degenerates_to_sinusoid(self):
        """sweep_rate = 0 → identical to SINUSOIDAL."""
        terrain = TerrainConfig(
            slope=0.4, wave_period=6.0, section_len=1.0, wave_sweep_rate=0.0,
        )
        sweep = SLOPE_PROFILES[SlopeProfile.FREQUENCY_SWEEP.value]
        sine = SLOPE_PROFILES[SlopeProfile.SINUSOIDAL.value]
        rng = np.random.default_rng(0)
        for step in range(10):
            self.assertAlmostEqual(
                sweep(terrain, rng, step), sine(terrain, rng, step), places=12,
            )

    def test_frequency_increases_with_distance(self):
        """Zero-crossings occur more often as x grows (chirp behaviour).

        Parameters chosen so the second half stays well above Nyquist:
        with ``section_len=0.5`` and a final wavelength near 1.6 m, each
        oscillation is sampled ~3 times.
        """
        terrain = TerrainConfig(
            slope=0.3, wave_period=8.0, section_len=0.5, wave_sweep_rate=0.005,
        )
        gen = SLOPE_PROFILES[SlopeProfile.FREQUENCY_SWEEP.value]
        rng = np.random.default_rng(0)
        n = 200
        values = [gen(terrain, rng, step) for step in range(n)]

        def zero_crossings(seq: list[float]) -> int:
            return sum(
                1 for i in range(1, len(seq))
                if seq[i - 1] == 0 or (seq[i] * seq[i - 1] < 0)
            )

        first_half = zero_crossings(values[: n // 2])
        second_half = zero_crossings(values[n // 2:])
        self.assertGreater(second_half, first_half)

    def test_zero_period_collapses_to_flat(self):
        terrain = TerrainConfig(
            slope=0.5, wave_period=0.0, wave_sweep_rate=0.1,
        )
        gen = SLOPE_PROFILES[SlopeProfile.FREQUENCY_SWEEP.value]
        rng = np.random.default_rng(0)
        for step in (0, 5, 17):
            self.assertEqual(gen(terrain, rng, step), 0.0)


class TestNewTerrainPresets(unittest.TestCase):
    """SLOPE_UP / SLOPE_DOWN / SINUSOIDAL terrain presets."""

    def test_all_new_presets_build(self):
        for preset in (
            TerrainPreset.SLOPE_UP,
            TerrainPreset.SLOPE_DOWN,
            TerrainPreset.SINUSOIDAL,
        ):
            cfg = TerrainConfig.from_preset(preset)
            self.assertIsInstance(cfg, TerrainConfig)

    def test_slope_up_climbs(self):
        cfg = TerrainConfig.from_preset(TerrainPreset.SLOPE_UP)
        cfg.seed = 5
        w = World(config=WorldConfig(terrain=cfg))
        for _ in range(20):
            w.build_road(positive=True)
        ys = [p[1] for p in w.road]
        # Constant uphill → strictly non-decreasing on the right side
        # (allowing floating noise).
        for i in range(1, len(ys)):
            self.assertGreaterEqual(ys[i], ys[i - 1] - 1e-9)
        self.assertGreater(ys[-1], ys[0])

    def test_slope_down_descends(self):
        cfg = TerrainConfig.from_preset(TerrainPreset.SLOPE_DOWN)
        cfg.seed = 6
        w = World(config=WorldConfig(terrain=cfg))
        for _ in range(20):
            w.build_road(positive=True)
        ys = [p[1] for p in w.road]
        self.assertLess(ys[-1], ys[0])

    def test_sinusoidal_preset_uses_wave_profile(self):
        cfg = TerrainConfig.from_preset(TerrainPreset.SINUSOIDAL)
        self.assertEqual(cfg.slope_profile, SlopeProfile.SINUSOIDAL)
        self.assertGreater(cfg.wave_period, 0)


class TestMixedBuildRoad(unittest.TestCase):
    """Verify build_road dispatches to different generators."""

    def test_mixed_generation(self):
        terrain = TerrainConfig(
            gap_freq=0.2, obstacle_freq=0.2, step_freq=0.2,
            seed=42,
        )
        w = World(config=WorldConfig(terrain=terrain))
        for _ in range(100):
            w.build_road(positive=True)
        # Road should have grown substantially
        self.assertGreater(len(w.road), 50)

    def test_backward_extension(self):
        terrain = TerrainConfig(seed=42)
        w = World(config=WorldConfig(terrain=terrain))
        left_x = w.road[0][0]
        for _ in range(10):
            w.build_road(positive=False)
        self.assertLess(w.road[0][0], left_x)


if __name__ == "__main__":
    unittest.main()
