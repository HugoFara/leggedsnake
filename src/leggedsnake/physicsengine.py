# -*- coding: utf-8 -*-
"""
Physics engine for dynamic walking simulation.

Uses the 2D physics engine pymunk (chipmunk) for planar mechanism simulation.
Manages the simulation space, road generation, and energy tracking.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, TypedDict

import numpy as np
import pymunk as pm

from pylinkage.geometry import norm, cyl_to_cart

from . import dynamiclinkage


# ---------------------------------------------------------------------------
# WorldConfig — structured replacement for the global ``params`` dict
# ---------------------------------------------------------------------------

@dataclass
class TerrainConfig:
    """Terrain generation parameters."""
    slope: float = 10 * np.pi / 180
    """Nominal slope in radians."""
    max_step: float = 0.5
    """Maximum step height."""
    step_freq: float = 0.1
    """Probability of a step vs. a slope segment."""
    noise: float = 0.9
    """Terrain variation factor (should be ≤ 1)."""
    section_len: float = 1.0
    """Length of each road section."""
    friction: float = 0.5 ** 0.5
    """Ground friction coefficient (square root of mu)."""


@dataclass
class WorldConfig:
    """Complete simulation configuration.

    Replaces the global ``params`` dict with a structured, immutable-by-default
    configuration object.  Pass to ``World(config=...)`` to parameterize a
    simulation; or omit to use ``DEFAULT_CONFIG``.

    Examples
    --------
    >>> cfg = WorldConfig(gravity=(0, -5.0), physics_period=0.01)
    >>> world = World(config=cfg)
    """
    gravity: tuple[float, float] = (0, -9.80665)
    """Gravity vector (m/s²)."""
    physics_period: float = 0.02
    """Time step for each physics computation (s)."""
    torque: float = 1e3
    """Maximum motor torque (N·m)."""
    load_mass: float = 10.0
    """Default load/chassis mass (kg)."""
    ground_friction: float = 0.5 ** 0.5
    """Ground friction coefficient (square root of mu)."""
    terrain: TerrainConfig = field(default_factory=TerrainConfig)
    """Terrain generation parameters."""


DEFAULT_CONFIG = WorldConfig()
"""Module-level default configuration, used when no config is passed."""


# ---------------------------------------------------------------------------
# Legacy ``params`` dict — kept for backward compatibility
# ---------------------------------------------------------------------------

class GroundParams(TypedDict):
    slope: float
    max_step: float
    step_freq: float
    noise: float
    section_len: float
    friction: float


class LinkageParams(TypedDict):
    torque: float
    crank_len: float
    masses: float
    load: float


class PhysicsParams(TypedDict):
    gravity: tuple[float, float]
    max_force: float


class SimulParams(TypedDict):
    physics_period: float


class Params(TypedDict):
    ground: GroundParams
    linkage: LinkageParams
    physics: PhysicsParams
    simul: SimulParams

# Simulation parameters
params: Params = {
    # Ground parameters
    "ground": {
        # Nominal slope (radian)
        "slope": 10 * np.pi / 180,
        # Maximal step height
        "max_step": .5,
        # Steps frequency
        "step_freq": .1,
        # Terrain variations should not be above 1
        "noise": .9,
        # Road trunks length
        "section_len": 1,
        # Ground friction coefficient root
        "friction": .5 ** .5
    },
    # Studied system parameters
    "linkage": {
        # Maximal torque (N.m)
        "torque": 1e3,
        # Crank length (m) (unused for now)
        "crank_len": .05,
        # Linear mass of bars (kg/m)
        "masses": 1,
        # Load mass (kg)
        "load": 10,
    },
    # Physics engine parameters
    "physics": {
        "gravity": (0, -9.80665),
        # Maximal value of forces (N)
        "max_force": 1e10,
    },
    # Study hypothesis
    "simul": {
        # Time between two physics computations
        "physics_period": 0.02,
    }
}


def set_space_constraints(space: pm.Space) -> None:
    """Auto-tune solver parameters based on constraint count."""
    constraints = space.constraints
    len_c = len(constraints)
    if len_c == 0:
        return

    space.iterations = max(20, int(15 + len_c * 0.5))

    correction_rate = 0.1 * np.exp(-len_c / 50)
    error_bias = pow(1 - correction_rate, 60)

    for constraint in constraints:
        constraint.error_bias = error_bias


class World:
    """Simulation world containing a pymunk space, linkages, and a road.

    Not intended to be rendered visually per se, see VisualWorld for that.

    Parameters
    ----------
    space : pm.Space | None
        Pymunk space. Created automatically if *None*.
    road_y : float
        Initial road height.
    config : WorldConfig | None
        Simulation parameters. Uses ``DEFAULT_CONFIG`` when *None*.
    """

    space: pm.Space
    road: list[tuple[float, float]]
    linkages: list[dynamiclinkage.DynamicLinkage]
    config: WorldConfig

    def __init__(
        self,
        space: pm.Space | None = None,
        road_y: float = -5,
        config: WorldConfig | None = None,
    ) -> None:
        self.config = config if config is not None else DEFAULT_CONFIG

        if isinstance(space, pm.Space):
            self.space = space
        else:
            self.space = pm.Space()
            self.space.gravity = self.config.gravity

        set_space_constraints(self.space)

        self.road = [(-15, road_y), (15, road_y)]
        seg = pm.Segment(
            self.space.static_body, self.road[0], self.road[-1], .1
        )
        seg.friction = self.config.ground_friction
        self.space.add(seg)
        self.linkages = []

    def add_linkage(
        self,
        source: Any,
        load: float = 0,
    ) -> None:
        """Add a linkage to the simulation.

        Parameters
        ----------
        source : Walker, DynamicLinkage, or legacy Linkage
            The mechanism to simulate.
        load : float
            Load mass (only used when converting from Walker/Linkage).
        """
        if isinstance(source, dynamiclinkage.DynamicLinkage):
            dl = source
        else:
            dl = dynamiclinkage.convert_to_dynamic_linkage(
                source, self.space, load=load
            )

        # Disable motors initially (enable when settled)
        for motor in dl.physics_mapping.motors:
            motor.max_force = 0

        self.linkages.append(dl)
        for s in self.space.shapes:
            s.friction = self.config.ground_friction

        self.tune_solver()

    def tune_solver(self) -> None:
        """Auto-tune solver parameters for stability."""
        set_space_constraints(self.space)

    def __update_linkage__(
        self, linkage: dynamiclinkage.DynamicLinkage, power: float
    ) -> tuple[float, float]:
        """Update a specific linkage."""
        motors = linkage.physics_mapping.motors
        vel = linkage.body.velocity

        # Check if motors need enabling (use first motor as reference)
        if motors and motors[0].max_force == 0:
            if norm(vel.x, vel.y) < .1:
                # Enable ALL motors when linkage settles
                for motor in motors:
                    motor.max_force = self.config.torque
                linkage.height = linkage.body.position.y
                linkage.mechanical_energy = (
                    .5 * linkage.mass * norm(vel.x, vel.y) ** 2
                )

        # Energy from the motor in this step
        energy = power * self.config.physics_period
        if hasattr(linkage, 'height') and linkage.height != 0.0 and energy != 0.:
            vel = linkage.body.velocity
            v = norm(vel.x, vel.y)
            g = norm(*self.config.gravity)
            m = linkage.mass
            new_mechanical_energy = m * (
                .5 * v ** 2 + g * (linkage.body.position.y - linkage.height)
            )
            efficiency = (
                new_mechanical_energy - linkage.mechanical_energy
            ) / energy
            linkage.mechanical_energy = new_mechanical_energy
            return energy, efficiency
        return 0, 0

    def update(self, dt: float | None = None) -> tuple[float, float] | None:
        """Update simulation by one time step.

        Parameters
        ----------
        dt : float | None
            Time step. Uses ``config.physics_period`` if None.
        """
        if dt is None:
            dt = self.config.physics_period

        # Compute motor power for each linkage
        powers: list[list[float]] = [
            [0.0] * len(lin.physics_mapping.motors)
            for lin in self.linkages
        ]
        self.space.step(dt)

        for i, linkage in enumerate(self.linkages):
            for j, motor in enumerate(linkage.physics_mapping.motors):
                # Motor angular velocity relative to chassis
                driver_body = motor.a  # type: ignore[attr-defined]
                w = driver_body.angular_velocity - linkage.body.angular_velocity
                powers[i][j] = abs(w) * abs(motor.impulse) / dt

        bounds: tuple[float, float] = (0.0, 0.0)
        energies: list[float] = [0.0] * len(self.linkages)
        efficiencies: list[float] = [0.0] * len(self.linkages)

        for idx, (linkage, power) in enumerate(zip(self.linkages, powers)):
            recalc_linkage(linkage)
            # Sum power from ALL motors (fixes multi-motor energy accounting)
            energies[idx], efficiencies[idx] = self.__update_linkage__(
                linkage, sum(power)
            )
            # Compute position bounds for road generation
            for proxy in linkage.joints:
                bounds = (
                    min(bounds[0], proxy.x),
                    max(bounds[1], proxy.x),
                )

        while self.road[-1][0] < bounds[1] + 10:
            self.build_road(True)
        while self.road[0][0] > bounds[0] - 10:
            self.build_road(False)

        for linkage, energy, efficiency in zip(
                self.linkages, energies, efficiencies
        ):
            return efficiency, energy * dt
        return None

    def __build_road_step__(self, terrain: TerrainConfig, index: int) -> None:
        """Add a step (two points)."""
        high = np.random.rand() * terrain.max_step
        a = self.road[index][0], self.road[index][1] + high
        b = (
            self.road[index][0] + terrain.section_len * (1 - index),
            self.road[index][1] + high
        )

        s = pm.Segment(self.space.static_body, a, b, .1)
        s.friction = self.config.ground_friction
        self.space.add(s)
        s = pm.Segment(self.space.static_body, a, self.road[index], .1)
        s.friction = self.config.ground_friction
        self.space.add(s)
        self.road.insert(-index * len(self.road), a)
        self.road.insert(-index * len(self.road), b)

    def __build_road_segment__(self, terrain: TerrainConfig, index: int) -> None:
        """Add a segment (one point)."""
        angle = np.random.normal(
            terrain.slope / 2, terrain.noise * terrain.slope / 2
        )
        if not index:
            angle = np.pi - angle
        a = pm.Vec2d(*cyl_to_cart(terrain.section_len, angle,
                                  *self.road[index]))
        s = pm.Segment(self.space.static_body, a, self.road[index], .1)
        s.friction = self.config.ground_friction
        self.space.add(s)
        self.road.insert(-index * len(self.road), a)

    def build_road(self, positive: bool = False) -> None:
        """Build a road part.

        Arguments
        ---------
        positive: if False (default), the road part will be added on the left.
        """
        terrain = self.config.terrain
        if np.random.rand() < terrain.step_freq and False:
            self.__build_road_step__(terrain, -positive)
        else:
            self.__build_road_segment__(terrain, -positive)


def recalc_linkage(linkage: dynamiclinkage.DynamicLinkage) -> None:
    """Update all joint proxy coordinates from physics bodies."""
    for j in linkage.joints:
        j.reload()


def linkage_bb(
    linkage: Any,
) -> tuple[float, float, float, float]:
    """Return the bounding box (min_y, max_x, max_y, min_x) for a linkage."""
    from .walker import Walker

    if isinstance(linkage, dynamiclinkage.DynamicLinkage):
        positions = list(linkage.get_all_positions().values())
        positions.extend(tuple(b.position) for b in linkage.rigidbodies)
    elif isinstance(linkage, Walker):
        positions = list(linkage.dimensions.node_positions.values())
    else:
        # Legacy Linkage
        from pylinkage import bounding_box
        data = [i.coord() for i in linkage.joints]
        bbox = bounding_box(data)
        return (float(bbox[0]), float(bbox[1]), float(bbox[2]), float(bbox[3]))

    if not positions:
        return (0.0, 0.0, 0.0, 0.0)

    xs = [p[0] for p in positions]
    ys = [p[1] for p in positions]
    return (float(min(ys)), float(max(xs)), float(max(ys)), float(min(xs)))
