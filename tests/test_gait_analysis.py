#!/usr/bin/env python3
"""Tests for the gait analysis module."""

import unittest
import math

from leggedsnake.gait_analysis import (
    FootEvent,
    GaitAnalysisResult,
    GaitCycle,
    analyze_gait,
    compute_foot_trajectory_metrics,
    compute_phase_offsets,
    detect_foot_events,
    extract_gait_cycles,
)


def _sinusoidal_trajectory(
    n_points: int = 200,
    period: int = 50,
    amplitude: float = 0.5,
    offset: float = 0.05,
    x_speed: float = 0.1,
) -> list[tuple[float, float]]:
    """Generate a sinusoidal foot trajectory that periodically touches ground."""
    traj = []
    for i in range(n_points):
        x = i * x_speed
        y = offset + amplitude * (1 + math.sin(2 * math.pi * i / period)) / 2
        traj.append((x, y))
    return traj


class TestDetectFootEvents(unittest.TestCase):

    def test_simple_sinusoidal(self):
        """Sinusoidal trajectory should produce touchdown/liftoff pairs."""
        traj = _sinusoidal_trajectory(n_points=200, period=50, offset=-0.1)
        events = detect_foot_events({"foot_a": traj}, contact_threshold=0.1, dt=0.02)
        tds = [e for e in events if e.event_type == "touchdown"]
        los = [e for e in events if e.event_type == "liftoff"]
        self.assertGreater(len(tds), 0)
        self.assertGreater(len(los), 0)

    def test_always_on_ground(self):
        """Trajectory always below threshold → no events (already on ground)."""
        traj = [(i * 0.1, 0.0) for i in range(100)]
        events = detect_foot_events({"foot": traj}, contact_threshold=0.5, dt=0.02)
        self.assertEqual(len(events), 0)

    def test_always_airborne(self):
        """Trajectory always above threshold → no events."""
        traj = [(i * 0.1, 5.0) for i in range(100)]
        events = detect_foot_events({"foot": traj}, contact_threshold=0.1, dt=0.02)
        self.assertEqual(len(events), 0)

    def test_single_point(self):
        """Single point trajectory → no events."""
        events = detect_foot_events({"foot": [(0, 0)]}, dt=0.02)
        self.assertEqual(len(events), 0)

    def test_multiple_feet(self):
        """Events from multiple feet are sorted by time."""
        traj_a = _sinusoidal_trajectory(100, 40, offset=-0.1)
        traj_b = _sinusoidal_trajectory(100, 40, offset=-0.1)
        events = detect_foot_events(
            {"a": traj_a, "b": traj_b}, contact_threshold=0.1, dt=0.02,
        )
        times = [e.time for e in events]
        self.assertEqual(times, sorted(times))


class TestExtractGaitCycles(unittest.TestCase):

    def test_from_events(self):
        """Three touchdowns and two liftoffs should produce one cycle."""
        events = [
            FootEvent("f", 0.1, "touchdown", (0, 0)),
            FootEvent("f", 0.3, "liftoff", (1, 0.5)),
            FootEvent("f", 0.5, "touchdown", (2, 0)),
        ]
        cycles = extract_gait_cycles(events)
        self.assertIn("f", cycles)
        self.assertEqual(len(cycles["f"]), 1)
        c = cycles["f"][0]
        self.assertAlmostEqual(c.stance_start, 0.1)
        self.assertAlmostEqual(c.stance_end, 0.3)
        self.assertAlmostEqual(c.swing_end, 0.5)

    def test_duty_factor(self):
        """Duty factor = stance / period."""
        events = [
            FootEvent("f", 0.0, "touchdown", (0, 0)),
            FootEvent("f", 0.5, "liftoff", (1, 0.5)),
            FootEvent("f", 1.0, "touchdown", (2, 0)),
        ]
        cycles = extract_gait_cycles(events)
        c = cycles["f"][0]
        self.assertAlmostEqual(c.duty_factor, 0.5)
        self.assertAlmostEqual(c.stride_period, 1.0)

    def test_no_complete_cycle(self):
        """Incomplete events → no cycles."""
        events = [
            FootEvent("f", 0.1, "touchdown", (0, 0)),
            FootEvent("f", 0.3, "liftoff", (1, 0.5)),
            # No second touchdown
        ]
        cycles = extract_gait_cycles(events)
        self.assertEqual(len(cycles), 0)

    def test_multiple_cycles(self):
        """Multiple touchdown-liftoff-touchdown sequences."""
        events = [
            FootEvent("f", 0.0, "touchdown", (0, 0)),
            FootEvent("f", 0.3, "liftoff", (1, 0.5)),
            FootEvent("f", 0.5, "touchdown", (2, 0)),
            FootEvent("f", 0.8, "liftoff", (3, 0.5)),
            FootEvent("f", 1.0, "touchdown", (4, 0)),
        ]
        cycles = extract_gait_cycles(events)
        self.assertEqual(len(cycles["f"]), 2)


class TestPhaseOffsets(unittest.TestCase):

    def test_in_phase(self):
        """Two feet with identical timing → phase offset 0."""
        cycles = {
            "a": [GaitCycle("a", 0.0, 0.3, 0.3, 0.5)],
            "b": [GaitCycle("b", 0.0, 0.3, 0.3, 0.5)],
        }
        offsets = compute_phase_offsets(cycles)
        self.assertIn(("a", "b"), offsets)
        self.assertAlmostEqual(offsets[("a", "b")], 0.0)

    def test_alternating(self):
        """Two feet offset by half a period → phase offset 0.5."""
        cycles = {
            "a": [GaitCycle("a", 0.0, 0.25, 0.25, 0.5)],
            "b": [GaitCycle("b", 0.25, 0.5, 0.5, 0.75)],
        }
        offsets = compute_phase_offsets(cycles)
        self.assertAlmostEqual(offsets[("a", "b")], 0.5, places=1)

    def test_empty_cycles(self):
        offsets = compute_phase_offsets({})
        self.assertEqual(len(offsets), 0)


class TestFootTrajectoryMetrics(unittest.TestCase):

    def test_horizontal_line(self):
        traj = [(i * 0.1, 0.0) for i in range(10)]
        m = compute_foot_trajectory_metrics(traj)
        self.assertAlmostEqual(m["max_height"], 0.0)
        self.assertAlmostEqual(m["horizontal_range"], 0.9)
        self.assertGreater(m["path_length"], 0)

    def test_single_point(self):
        m = compute_foot_trajectory_metrics([(0, 0)])
        self.assertEqual(m["max_height"], 0.0)
        self.assertEqual(m["path_length"], 0.0)

    def test_circular_trajectory(self):
        """Circular path should have equal height and width range."""
        import math
        traj = [
            (math.cos(2 * math.pi * i / 100), math.sin(2 * math.pi * i / 100))
            for i in range(101)
        ]
        m = compute_foot_trajectory_metrics(traj)
        self.assertAlmostEqual(m["max_height"], 2.0, places=1)
        self.assertAlmostEqual(m["horizontal_range"], 2.0, places=1)


class TestAnalyzeGait(unittest.TestCase):

    def test_end_to_end_synthetic(self):
        """Full analysis pipeline with synthetic data."""
        traj = _sinusoidal_trajectory(200, 40, offset=-0.1)
        loci = {"foot_a": traj, "other_joint": [(0, 5)] * 200}

        result = analyze_gait(
            loci=loci,
            foot_ids=["foot_a"],
            dt=0.02,
            contact_threshold=0.1,
        )
        self.assertIsInstance(result, GaitAnalysisResult)
        self.assertGreater(len(result.foot_events), 0)
        self.assertIn("foot_a", result.foot_trajectories)

    def test_no_foot_data(self):
        """Missing foot IDs in loci → empty result."""
        result = analyze_gait(
            loci={"other": [(0, 0)] * 10},
            foot_ids=["foot_a"],
            dt=0.02,
        )
        self.assertEqual(len(result.foot_events), 0)
        self.assertEqual(len(result.foot_trajectories), 0)

    def test_summary_metrics(self):
        """summary_metrics returns expected keys."""
        traj = _sinusoidal_trajectory(200, 40, offset=-0.1)
        result = analyze_gait(
            loci={"foot_a": traj},
            foot_ids=["foot_a"],
            dt=0.02,
            contact_threshold=0.1,
        )
        m = result.summary_metrics()
        self.assertIn("mean_duty_factor", m)
        self.assertIn("mean_stride_frequency", m)
        self.assertIn("mean_stride_length", m)


class TestGaitAnalysisResult(unittest.TestCase):

    def test_empty_result(self):
        result = GaitAnalysisResult()
        self.assertEqual(result.mean_duty_factor, 0.0)
        self.assertEqual(result.mean_stride_frequency, 0.0)
        self.assertEqual(result.mean_stride_length, 0.0)
        self.assertEqual(len(result.phase_offsets), 0)


if __name__ == "__main__":
    unittest.main()
