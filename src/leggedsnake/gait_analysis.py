"""
Gait analysis for walking mechanism simulations.

Transforms raw joint trajectory data (loci) into biomechanics metrics:
duty factor, stride frequency, phase offsets between feet, foot trajectory
shape analysis, and gait cycle decomposition.

Example::

    from leggedsnake.gait_analysis import analyze_gait

    # After running a simulation with record_loci=True
    result = fitness(topology, dimensions)
    gait = analyze_gait(
        loci=result.loci,
        foot_ids=walker.get_feet(),
        dt=0.02,
    )
    print(gait.mean_duty_factor, gait.mean_stride_frequency)
    print(gait.phase_offsets)
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Literal

import numpy as np

if TYPE_CHECKING:
    from .stability import StabilityTimeSeries


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class FootEvent:
    """A single foot touchdown or liftoff event."""

    foot_id: str
    time: float
    event_type: Literal["touchdown", "liftoff"]
    position: tuple[float, float]


@dataclass(slots=True)
class GaitCycle:
    """Metrics for one stride cycle of a single foot."""

    foot_id: str
    stance_start: float
    stance_end: float
    swing_start: float
    swing_end: float

    @property
    def stance_duration(self) -> float:
        return self.stance_end - self.stance_start

    @property
    def swing_duration(self) -> float:
        return self.swing_end - self.swing_start

    @property
    def stride_period(self) -> float:
        return self.swing_end - self.stance_start

    @property
    def duty_factor(self) -> float:
        period = self.stride_period
        if period <= 0:
            return 0.0
        return self.stance_duration / period


@dataclass
class GaitAnalysisResult:
    """Complete gait analysis for a simulation run.

    Attributes
    ----------
    foot_events : list[FootEvent]
        All detected touchdown/liftoff events.
    gait_cycles : dict[str, list[GaitCycle]]
        Stride cycles keyed by foot ID.
    foot_trajectories : dict[str, list[tuple[float, float]]]
        Raw foot trajectories keyed by foot ID.
    stability : StabilityTimeSeries | None
        Stability data if available.
    dt : float
        Simulation timestep used.
    """

    foot_events: list[FootEvent] = field(default_factory=list)
    gait_cycles: dict[str, list[GaitCycle]] = field(default_factory=dict)
    foot_trajectories: dict[str, list[tuple[float, float]]] = field(
        default_factory=dict,
    )
    stability: StabilityTimeSeries | None = None
    dt: float = 0.02

    @property
    def mean_duty_factor(self) -> float:
        """Average duty factor across all feet and cycles."""
        all_cycles = [c for cycles in self.gait_cycles.values() for c in cycles]
        if not all_cycles:
            return 0.0
        return sum(c.duty_factor for c in all_cycles) / len(all_cycles)

    @property
    def mean_stride_frequency(self) -> float:
        """Average stride frequency in Hz across all feet."""
        all_cycles = [c for cycles in self.gait_cycles.values() for c in cycles]
        if not all_cycles:
            return 0.0
        periods = [c.stride_period for c in all_cycles if c.stride_period > 0]
        if not periods:
            return 0.0
        return 1.0 / (sum(periods) / len(periods))

    @property
    def mean_stride_length(self) -> float:
        """Average horizontal distance between consecutive touchdowns."""
        lengths: list[float] = []
        td_events: dict[str, list[FootEvent]] = {}
        for ev in self.foot_events:
            if ev.event_type == "touchdown":
                td_events.setdefault(ev.foot_id, []).append(ev)

        for foot_tds in td_events.values():
            for i in range(1, len(foot_tds)):
                dx = foot_tds[i].position[0] - foot_tds[i - 1].position[0]
                lengths.append(abs(dx))

        if not lengths:
            return 0.0
        return sum(lengths) / len(lengths)

    @property
    def phase_offsets(self) -> dict[tuple[str, str], float]:
        """Phase difference between each pair of feet.

        Normalized to [0, 1) where 0 = in-phase, 0.5 = alternating.
        """
        return compute_phase_offsets(self.gait_cycles)

    def summary_metrics(self) -> dict[str, float]:
        """Flat dictionary of all scalar gait metrics."""
        metrics: dict[str, float] = {
            "mean_duty_factor": self.mean_duty_factor,
            "mean_stride_frequency": self.mean_stride_frequency,
            "mean_stride_length": self.mean_stride_length,
        }
        if self.stability is not None:
            metrics.update(self.stability.summary_metrics())
        return metrics


# ---------------------------------------------------------------------------
# Core analysis functions
# ---------------------------------------------------------------------------


def detect_foot_events(
    foot_trajectories: dict[str, list[tuple[float, float]]],
    contact_threshold: float = 0.1,
    dt: float = 0.02,
) -> list[FootEvent]:
    """Detect touchdown/liftoff events from foot y-position threshold crossings.

    A *touchdown* occurs when the foot y-coordinate drops below
    ``contact_threshold``. A *liftoff* occurs when it rises back above.

    Parameters
    ----------
    foot_trajectories : dict[str, list[tuple[float, float]]]
        Foot trajectories keyed by foot ID, each a list of (x, y) points.
    contact_threshold : float
        Maximum y-coordinate for ground contact.
    dt : float
        Timestep between trajectory points.

    Returns
    -------
    list[FootEvent]
        Chronologically sorted events across all feet.
    """
    events: list[FootEvent] = []

    for foot_id, traj in foot_trajectories.items():
        if len(traj) < 2:
            continue

        was_on_ground = traj[0][1] <= contact_threshold

        for i in range(1, len(traj)):
            is_on_ground = traj[i][1] <= contact_threshold
            time = i * dt

            if is_on_ground and not was_on_ground:
                events.append(FootEvent(
                    foot_id=foot_id, time=time,
                    event_type="touchdown", position=traj[i],
                ))
            elif not is_on_ground and was_on_ground:
                events.append(FootEvent(
                    foot_id=foot_id, time=time,
                    event_type="liftoff", position=traj[i],
                ))

            was_on_ground = is_on_ground

    events.sort(key=lambda e: e.time)
    return events


def extract_gait_cycles(
    events: list[FootEvent],
) -> dict[str, list[GaitCycle]]:
    """Group foot events into stride cycles per foot.

    A cycle is: touchdown → liftoff → next touchdown.

    Parameters
    ----------
    events : list[FootEvent]
        Sorted foot events from ``detect_foot_events``.

    Returns
    -------
    dict[str, list[GaitCycle]]
        Gait cycles keyed by foot ID.
    """
    # Collect events per foot in order
    per_foot: dict[str, list[FootEvent]] = {}
    for ev in events:
        per_foot.setdefault(ev.foot_id, []).append(ev)

    cycles: dict[str, list[GaitCycle]] = {}

    for foot_id, foot_events in per_foot.items():
        foot_cycles: list[GaitCycle] = []

        # Find touchdown-liftoff-touchdown triplets
        i = 0
        while i < len(foot_events):
            # Find next touchdown
            if foot_events[i].event_type != "touchdown":
                i += 1
                continue
            td1 = foot_events[i]

            # Find next liftoff after this touchdown
            lo = None
            j = i + 1
            while j < len(foot_events):
                if foot_events[j].event_type == "liftoff":
                    lo = foot_events[j]
                    break
                j += 1

            if lo is None:
                break

            # Find next touchdown after liftoff
            td2 = None
            k = j + 1
            while k < len(foot_events):
                if foot_events[k].event_type == "touchdown":
                    td2 = foot_events[k]
                    break
                k += 1

            if td2 is None:
                break

            foot_cycles.append(GaitCycle(
                foot_id=foot_id,
                stance_start=td1.time,
                stance_end=lo.time,
                swing_start=lo.time,
                swing_end=td2.time,
            ))

            # Next cycle starts at td2
            i = k

        if foot_cycles:
            cycles[foot_id] = foot_cycles

    return cycles


def compute_phase_offsets(
    cycles: dict[str, list[GaitCycle]],
) -> dict[tuple[str, str], float]:
    """Phase difference between each pair of feet.

    Compares the mean stance_start times, normalized by the mean stride
    period. Result is in [0, 1) where 0 = in-phase, 0.5 = alternating.

    Parameters
    ----------
    cycles : dict[str, list[GaitCycle]]
        Gait cycles keyed by foot ID.

    Returns
    -------
    dict[tuple[str, str], float]
        Phase offset for each foot pair.
    """
    offsets: dict[tuple[str, str], float] = {}
    foot_ids = sorted(cycles.keys())

    # Compute mean stance start and mean period per foot
    means: dict[str, tuple[float, float]] = {}  # foot_id -> (mean_start, mean_period)
    for fid in foot_ids:
        if not cycles[fid]:
            continue
        starts = [c.stance_start for c in cycles[fid]]
        periods = [c.stride_period for c in cycles[fid]]
        means[fid] = (
            sum(starts) / len(starts),
            sum(periods) / len(periods),
        )

    for i, fid_a in enumerate(foot_ids):
        for fid_b in foot_ids[i + 1:]:
            if fid_a not in means or fid_b not in means:
                continue
            mean_start_a, period_a = means[fid_a]
            mean_start_b, period_b = means[fid_b]
            avg_period = (period_a + period_b) / 2
            if avg_period <= 0:
                offsets[(fid_a, fid_b)] = 0.0
                continue
            dt = abs(mean_start_b - mean_start_a)
            phase = (dt / avg_period) % 1.0
            offsets[(fid_a, fid_b)] = phase

    return offsets


def compute_foot_trajectory_metrics(
    trajectory: list[tuple[float, float]],
) -> dict[str, float]:
    """Per-foot trajectory shape metrics.

    Parameters
    ----------
    trajectory : list[tuple[float, float]]
        A single foot's (x, y) trajectory.

    Returns
    -------
    dict[str, float]
        ``max_height``, ``horizontal_range``, ``path_length``,
        ``smoothness`` (inverse mean curvature).
    """
    if len(trajectory) < 2:
        return {
            "max_height": 0.0,
            "horizontal_range": 0.0,
            "path_length": 0.0,
            "smoothness": 0.0,
        }

    pts = np.array(trajectory)
    ys = pts[:, 1]
    xs = pts[:, 0]

    max_height = float(np.max(ys) - np.min(ys))
    horizontal_range = float(np.max(xs) - np.min(xs))

    # Path length
    diffs = np.diff(pts, axis=0)
    segment_lengths = np.sqrt(np.sum(diffs**2, axis=1))
    path_length = float(np.sum(segment_lengths))

    # Smoothness: inverse of mean absolute curvature
    # Curvature approximated from angle change between consecutive segments
    if len(trajectory) < 3:
        smoothness = float("inf")
    else:
        angles = np.arctan2(diffs[1:, 1], diffs[1:, 0]) - np.arctan2(
            diffs[:-1, 1], diffs[:-1, 0],
        )
        # Wrap to [-pi, pi]
        angles = (angles + np.pi) % (2 * np.pi) - np.pi
        mean_curvature = float(np.mean(np.abs(angles)))
        smoothness = 1.0 / mean_curvature if mean_curvature > 1e-9 else float("inf")

    return {
        "max_height": max_height,
        "horizontal_range": horizontal_range,
        "path_length": path_length,
        "smoothness": smoothness,
    }


# ---------------------------------------------------------------------------
# High-level entry point
# ---------------------------------------------------------------------------


def analyze_gait(
    loci: dict[str, list[tuple[float, float]]],
    foot_ids: list[str],
    stability: StabilityTimeSeries | None = None,
    dt: float = 0.02,
    contact_threshold: float = 0.1,
) -> GaitAnalysisResult:
    """One-call gait analysis from simulation loci.

    Parameters
    ----------
    loci : dict[str, list[tuple[float, float]]]
        Joint trajectories keyed by node ID (from ``FitnessResult.loci``).
    foot_ids : list[str]
        Node IDs of feet (from ``Walker.get_feet()``).
    stability : StabilityTimeSeries | None
        Optional stability data to attach.
    dt : float
        Simulation timestep.
    contact_threshold : float
        Maximum y-coordinate for ground contact detection.

    Returns
    -------
    GaitAnalysisResult
    """
    # Extract foot trajectories from full loci
    foot_trajectories = {
        fid: loci[fid] for fid in foot_ids if fid in loci
    }

    events = detect_foot_events(foot_trajectories, contact_threshold, dt)
    cycles = extract_gait_cycles(events)

    return GaitAnalysisResult(
        foot_events=events,
        gait_cycles=cycles,
        foot_trajectories=foot_trajectories,
        stability=stability,
        dt=dt,
    )
