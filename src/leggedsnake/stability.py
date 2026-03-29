"""
Stability metrics for walking mechanism evaluation.

Provides center-of-mass (CoM) tracking, zero-moment-point (ZMP) approximation,
support polygon computation, and tip-over margin for planar walkers.

These metrics run on top of pymunk simulation data extracted from
``DynamicLinkage`` and are collected into ``StabilityTimeSeries`` for
post-simulation analysis.

Example::

    from leggedsnake.stability import compute_stability_snapshot, StabilityTimeSeries

    snapshots = []
    prev = None
    for t in range(steps):
        world.update()
        snap = compute_stability_snapshot(dl, prev, t * dt, dt, gravity_y)
        snapshots.append(snap)
        prev = snap
    series = StabilityTimeSeries(snapshots)
    print(series.summary_metrics())
"""
from __future__ import annotations

from dataclasses import dataclass, field
from math import sqrt
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from .dynamiclinkage import DynamicLinkage


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class StabilitySnapshot:
    """Single-timestep stability measurement."""

    time: float
    com: tuple[float, float]
    com_velocity: tuple[float, float]
    zmp_x: float
    support_polygon: list[tuple[float, float]]
    tip_over_margin: float
    body_angle: float


@dataclass
class StabilityTimeSeries:
    """Full stability history across a simulation run.

    Attributes
    ----------
    snapshots : list[StabilitySnapshot]
        Chronologically ordered stability measurements.
    """

    snapshots: list[StabilitySnapshot] = field(default_factory=list)

    @property
    def com_trajectory(self) -> list[tuple[float, float]]:
        """Center-of-mass (x, y) at each timestep."""
        return [s.com for s in self.snapshots]

    @property
    def mean_tip_over_margin(self) -> float:
        """Average tip-over margin across the simulation."""
        if not self.snapshots:
            return 0.0
        return sum(s.tip_over_margin for s in self.snapshots) / len(self.snapshots)

    @property
    def min_tip_over_margin(self) -> float:
        """Worst-case tip-over margin (most unstable instant)."""
        if not self.snapshots:
            return 0.0
        return min(s.tip_over_margin for s in self.snapshots)

    @property
    def zmp_excursion(self) -> float:
        """Maximum deviation of ZMP from the support polygon center."""
        if not self.snapshots:
            return 0.0
        zmp_xs = [s.zmp_x for s in self.snapshots]
        return max(zmp_xs) - min(zmp_xs)

    @property
    def angular_stability(self) -> float:
        """RMS body angle — lower means more upright."""
        if not self.snapshots:
            return 0.0
        angles = [s.body_angle for s in self.snapshots]
        return float(np.sqrt(np.mean(np.square(angles))))

    def summary_metrics(self) -> dict[str, float]:
        """Flat dictionary of all scalar stability metrics."""
        return {
            "mean_tip_over_margin": self.mean_tip_over_margin,
            "min_tip_over_margin": self.min_tip_over_margin,
            "zmp_excursion": self.zmp_excursion,
            "angular_stability": self.angular_stability,
        }


# ---------------------------------------------------------------------------
# Core computation functions
# ---------------------------------------------------------------------------


def compute_com(linkage: DynamicLinkage) -> tuple[float, float]:
    """Mass-weighted center of mass of all rigid bodies.

    Parameters
    ----------
    linkage : DynamicLinkage
        The physics-enabled linkage.

    Returns
    -------
    tuple[float, float]
        (x, y) center of mass in world coordinates.
    """
    total_mass = 0.0
    wx = 0.0
    wy = 0.0
    for body in linkage.rigidbodies:
        m = body.mass
        if m <= 0 or not np.isfinite(m):
            continue
        total_mass += m
        wx += m * body.position.x
        wy += m * body.position.y

    if total_mass == 0:
        return (0.0, 0.0)
    return (wx / total_mass, wy / total_mass)


def compute_com_velocity(linkage: DynamicLinkage) -> tuple[float, float]:
    """Mass-weighted center-of-mass velocity.

    Parameters
    ----------
    linkage : DynamicLinkage
        The physics-enabled linkage.

    Returns
    -------
    tuple[float, float]
        (vx, vy) CoM velocity.
    """
    total_mass = 0.0
    wvx = 0.0
    wvy = 0.0
    for body in linkage.rigidbodies:
        m = body.mass
        if m <= 0 or not np.isfinite(m):
            continue
        total_mass += m
        wvx += m * body.velocity.x
        wvy += m * body.velocity.y

    if total_mass == 0:
        return (0.0, 0.0)
    return (wvx / total_mass, wvy / total_mass)


def approximate_zmp(
    com: tuple[float, float],
    com_velocity: tuple[float, float],
    prev_velocity: tuple[float, float],
    dt: float,
    gravity: float,
) -> float:
    """Zero-moment-point x-coordinate via inverted-pendulum model.

    Uses the linear inverted pendulum approximation::

        x_zmp = x_com - (z_com * a_x) / (g + a_z)

    where *a* is the CoM acceleration computed by finite difference
    from consecutive velocity readings, and *g* is the magnitude of
    gravitational acceleration (positive downward).

    Parameters
    ----------
    com : tuple[float, float]
        Current center-of-mass position.
    com_velocity : tuple[float, float]
        Current CoM velocity.
    prev_velocity : tuple[float, float]
        CoM velocity at the previous timestep.
    dt : float
        Timestep duration (seconds).
    gravity : float
        Gravitational acceleration magnitude (positive, e.g. 9.81).

    Returns
    -------
    float
        ZMP x-coordinate.
    """
    if dt <= 0:
        return com[0]

    ax = (com_velocity[0] - prev_velocity[0]) / dt
    ay = (com_velocity[1] - prev_velocity[1]) / dt

    denom = gravity + ay
    if abs(denom) < 1e-9:
        return com[0]

    # com[1] is the height above ground (z_com in the formula)
    return com[0] - (com[1] * ax) / denom


def get_support_polygon(
    linkage: DynamicLinkage,
    foot_ids: list[str] | None = None,
    ground_threshold: float = 0.1,
) -> list[tuple[float, float]]:
    """Convex hull of foot positions near ground level.

    Parameters
    ----------
    linkage : DynamicLinkage
        The physics-enabled linkage.
    foot_ids : list[str] | None
        Node IDs of feet. If None, uses all joints.
    ground_threshold : float
        Maximum y-coordinate for a foot to be considered in ground contact.

    Returns
    -------
    list[tuple[float, float]]
        Convex hull vertices of the support polygon (may be empty,
        a single point, or a line segment for 2D walkers).
    """
    contacts: list[tuple[float, float]] = []
    for proxy in linkage.joints:
        if foot_ids is not None and proxy.name not in foot_ids:
            # Also check by _node_id for internal lookups
            if proxy._node_id not in foot_ids:
                continue
        proxy.reload()
        if proxy.y <= ground_threshold:
            contacts.append((proxy.x, proxy.y))

    if len(contacts) <= 2:
        return contacts

    # 2D convex hull (sort by x for simple planar case)
    contacts.sort(key=lambda p: p[0])
    return contacts


def compute_tip_over_margin(
    com: tuple[float, float],
    support_polygon: list[tuple[float, float]],
) -> float:
    """Signed distance from CoM projection to nearest support boundary.

    For a 2D planar walker, the "support polygon" is typically a line
    segment on the x-axis. The margin is the minimum horizontal distance
    from the CoM x-projection to the support boundary edges.

    Positive means the CoM is inside the support; negative means tipping.

    Parameters
    ----------
    com : tuple[float, float]
        Center-of-mass position.
    support_polygon : list[tuple[float, float]]
        Ground contact points (sorted by x).

    Returns
    -------
    float
        Signed tip-over margin. Positive = stable, negative = tipping.
    """
    if not support_polygon:
        return -1.0  # No support at all

    if len(support_polygon) == 1:
        # Single contact point — margin is distance to that point
        dx = com[0] - support_polygon[0][0]
        return -abs(dx) if abs(dx) > 1e-9 else 0.0

    # For planar walkers: margin is distance from CoM x-projection
    # to the nearest edge of the x-interval [min_x, max_x]
    xs = [p[0] for p in support_polygon]
    min_x = min(xs)
    max_x = max(xs)
    com_x = com[0]

    if com_x < min_x:
        return com_x - min_x  # negative
    elif com_x > max_x:
        return max_x - com_x  # negative
    else:
        # Inside: margin is distance to nearest edge
        return min(com_x - min_x, max_x - com_x)


# ---------------------------------------------------------------------------
# Convenience: one-call snapshot assembly
# ---------------------------------------------------------------------------


def compute_stability_snapshot(
    linkage: DynamicLinkage,
    prev_snapshot: StabilitySnapshot | None,
    time: float,
    dt: float,
    gravity: float,
    foot_ids: list[str] | None = None,
    ground_threshold: float = 0.1,
) -> StabilitySnapshot:
    """Assemble a full stability snapshot from current physics state.

    Parameters
    ----------
    linkage : DynamicLinkage
        The physics-enabled linkage (call after ``world.update()``).
    prev_snapshot : StabilitySnapshot | None
        Previous snapshot (for velocity finite-differencing). If None,
        ZMP defaults to CoM x-position.
    time : float
        Current simulation time.
    dt : float
        Physics timestep.
    gravity : float
        Gravitational acceleration magnitude (positive).
    foot_ids : list[str] | None
        Node IDs of feet for support polygon computation.
    ground_threshold : float
        Maximum y-coordinate for ground contact detection.

    Returns
    -------
    StabilitySnapshot
    """
    # Reload all joint positions from physics
    for proxy in linkage.joints:
        proxy.reload()

    com = compute_com(linkage)
    com_vel = compute_com_velocity(linkage)

    prev_vel = prev_snapshot.com_velocity if prev_snapshot is not None else (0.0, 0.0)
    zmp_x = approximate_zmp(com, com_vel, prev_vel, dt, gravity)

    support = get_support_polygon(linkage, foot_ids, ground_threshold)
    margin = compute_tip_over_margin(com, support)
    body_angle = float(linkage.body.angle)

    return StabilitySnapshot(
        time=time,
        com=com,
        com_velocity=com_vel,
        zmp_x=zmp_x,
        support_polygon=support,
        tip_over_margin=margin,
        body_angle=body_angle,
    )
