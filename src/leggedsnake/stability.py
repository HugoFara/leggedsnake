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
from typing import TYPE_CHECKING

import numpy as np
import pymunk as pm

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
    ground_reaction_force: float = 0.0
    """Sum of contact-force magnitudes between the linkage and the ground
    (Newtons) — zero when no foot is touching."""
    peak_contact_force: float = 0.0
    """Largest single-contact force magnitude this step."""


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

    @property
    def mean_speed(self) -> float:
        """Mean forward (x-axis) CoM speed across the run."""
        if not self.snapshots:
            return 0.0
        vxs = [s.com_velocity[0] for s in self.snapshots]
        return float(np.mean(vxs))

    @property
    def speed_variance(self) -> float:
        """Population variance of the forward CoM speed.

        Captures how steady the walker's progress is under the current
        terrain — a smooth gait on flat ground has near-zero variance,
        jerky gaits or rough terrain raise it.
        """
        if len(self.snapshots) < 2:
            return 0.0
        vxs = [s.com_velocity[0] for s in self.snapshots]
        return float(np.var(vxs))

    @property
    def peak_ground_reaction_force(self) -> float:
        """Maximum summed foot-ground contact force across the run (N)."""
        if not self.snapshots:
            return 0.0
        return max(s.ground_reaction_force for s in self.snapshots)

    @property
    def mean_ground_reaction_force(self) -> float:
        """Mean summed foot-ground contact force across the run (N).

        Averages over *every* step — steps with no foot contact count
        as zero. A well-footed walker's mean should approximate the
        weight of the walker.
        """
        if not self.snapshots:
            return 0.0
        return (
            sum(s.ground_reaction_force for s in self.snapshots)
            / len(self.snapshots)
        )

    @property
    def peak_contact_force(self) -> float:
        """Largest single foot-ground contact force across the run (N).

        Useful for spotting impulsive spikes that would shatter a real
        mechanism even when the summed force looks benign.
        """
        if not self.snapshots:
            return 0.0
        return max(s.peak_contact_force for s in self.snapshots)

    def summary_metrics(self) -> dict[str, float]:
        """Flat dictionary of all scalar stability metrics."""
        return {
            "mean_tip_over_margin": self.mean_tip_over_margin,
            "min_tip_over_margin": self.min_tip_over_margin,
            "zmp_excursion": self.zmp_excursion,
            "angular_stability": self.angular_stability,
            "mean_speed": self.mean_speed,
            "speed_variance": self.speed_variance,
            "peak_ground_reaction_force": self.peak_ground_reaction_force,
            "mean_ground_reaction_force": self.mean_ground_reaction_force,
            "peak_contact_force": self.peak_contact_force,
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


def sample_ground_reaction_force(
    linkage: DynamicLinkage,
    static_body: pm.Body,
    dt: float,
) -> tuple[float, float]:
    """Sample foot–ground contact forces right after a physics step.

    Iterates active pymunk arbiters on every linkage body and sums the
    impulse magnitude of arbiters that involve the space's static body
    (the ground / road). Call **after** ``space.step(dt)``; arbiters are
    only valid then.

    Parameters
    ----------
    linkage : DynamicLinkage
        The linkage whose contacts to sample.
    static_body : pm.Body
        The pymunk static body that owns all ground segments — typically
        ``world.space.static_body``.
    dt : float
        The timestep just taken. Used to convert accumulated impulse to
        an average force (``F = J / dt``). Must be > 0.

    Returns
    -------
    tuple[float, float]
        ``(total_force, peak_force)`` — sum of all foot-ground contact
        forces this step, and the largest single-contact force. Both in
        the same units as pymunk mass × m / s² (usually Newtons).
    """
    if dt <= 0:
        return (0.0, 0.0)

    total = 0.0
    peak = 0.0

    def _collect(arbiter: pm.Arbiter, *args: object) -> None:
        nonlocal total, peak
        if static_body not in arbiter.bodies:
            return
        mag = float(arbiter.total_impulse.length) / dt
        total += mag
        if mag > peak:
            peak = mag

    for body in linkage.rigidbodies:
        body.each_arbiter(_collect)

    return total, peak


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
    static_body: pm.Body | None = None,
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
    static_body : pm.Body | None
        If provided, sample foot–ground contact forces via
        :func:`sample_ground_reaction_force` and populate the
        ``ground_reaction_force`` and ``peak_contact_force`` fields.
        Pass ``world.space.static_body`` to enable.

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

    if static_body is not None:
        grf_total, grf_peak = sample_ground_reaction_force(
            linkage, static_body, dt,
        )
    else:
        grf_total = 0.0
        grf_peak = 0.0

    return StabilitySnapshot(
        time=time,
        com=com,
        com_velocity=com_vel,
        zmp_x=zmp_x,
        support_polygon=support,
        tip_over_margin=margin,
        body_angle=body_angle,
        ground_reaction_force=grf_total,
        peak_contact_force=grf_peak,
    )
