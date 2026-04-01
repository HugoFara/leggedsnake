# -*- coding: utf-8 -*-
"""
Physics engine for dynamic walking simulation.

Uses the 2D physics engine pymunk (chipmunk) for planar mechanism simulation.
Manages the simulation space, road generation, and energy tracking.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, TypedDict

import numpy as np
import pymunk as pm

from pylinkage.geometry import norm, cyl_to_cart

from . import dynamiclinkage


# ---------------------------------------------------------------------------
# Terrain profiles — deterministic slope generators for benchmarking
# ---------------------------------------------------------------------------

class SlopeProfile(Enum):
    """Named slope profiles for repeatable terrain generation.

    Each value is a string key; the actual generator is looked up in
    :data:`SLOPE_PROFILES`.
    """
    RANDOM = "random"
    """Default Gaussian-distributed random slopes."""
    FLAT = "flat"
    """Perfectly flat ground (slope = 0)."""
    CONSTANT = "constant"
    """Constant uphill at the configured slope angle."""
    VALLEY = "valley"
    """V-shaped descent then ascent."""
    SAWTOOTH = "sawtooth"
    """Repeating climb-then-drop pattern."""


def _slope_random(
    terrain: "TerrainConfig", rng: np.random.Generator, _step: int,
) -> float:
    """Gaussian-distributed slope (original behaviour)."""
    return float(rng.normal(terrain.slope / 2, terrain.noise * terrain.slope / 2))


def _slope_flat(
    _terrain: "TerrainConfig", _rng: np.random.Generator, _step: int,
) -> float:
    return 0.0


def _slope_constant(
    terrain: "TerrainConfig", _rng: np.random.Generator, _step: int,
) -> float:
    return terrain.slope


def _slope_valley(
    terrain: "TerrainConfig", _rng: np.random.Generator, step: int,
) -> float:
    """V-shape: descend for 20 segments, then ascend."""
    half = 20
    if step % (2 * half) < half:
        return -terrain.slope
    return terrain.slope


def _slope_sawtooth(
    terrain: "TerrainConfig", _rng: np.random.Generator, step: int,
) -> float:
    """Climb for 10 segments, then a steep single-segment drop."""
    cycle = 11
    if step % cycle < cycle - 1:
        return terrain.slope
    return -terrain.slope * (cycle - 1)


#: Registry mapping profile keys to generator callables.
#: Signature: ``(terrain, rng, step_counter) -> angle_in_radians``
SlopeGenerator = Callable[["TerrainConfig", np.random.Generator, int], float]

SLOPE_PROFILES: dict[str, SlopeGenerator] = {
    SlopeProfile.RANDOM.value: _slope_random,
    SlopeProfile.FLAT.value: _slope_flat,
    SlopeProfile.CONSTANT.value: _slope_constant,
    SlopeProfile.VALLEY.value: _slope_valley,
    SlopeProfile.SAWTOOTH.value: _slope_sawtooth,
}


# ---------------------------------------------------------------------------
# Terrain presets
# ---------------------------------------------------------------------------

class TerrainPreset(Enum):
    """Ready-made terrain configurations."""
    FLAT = "flat"
    HILLY = "hilly"
    ROUGH = "rough"
    STAIRS = "stairs"
    MIXED = "mixed"


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
    friction_range: tuple[float, float] | None = None
    """If set, friction is randomized uniformly within this (min, max) range
    for each new road segment, overriding :attr:`friction`."""
    seed: int | None = None
    """Random seed for reproducible terrain generation.  *None* means
    non-deterministic."""
    gap_freq: float = 0.0
    """Probability of a gap (chasm) instead of a solid segment."""
    gap_width: float = 1.0
    """Width of gaps in meters."""
    obstacle_freq: float = 0.0
    """Probability of placing a rectangular obstacle on a segment."""
    obstacle_height: float = 0.3
    """Maximum obstacle height in meters."""
    obstacle_width: float = 0.4
    """Obstacle width in meters."""
    slope_profile: SlopeProfile | SlopeGenerator | str = SlopeProfile.RANDOM
    """Slope generation strategy.  Accepts a :class:`SlopeProfile` enum, a
    string key from :data:`SLOPE_PROFILES`, or a custom callable with
    signature ``(terrain, rng, step_counter) -> angle``."""

    @staticmethod
    def from_preset(preset: TerrainPreset | str) -> "TerrainConfig":
        """Create a :class:`TerrainConfig` from a named preset.

        Parameters
        ----------
        preset : TerrainPreset or str
            One of the built-in terrain presets.
        """
        if isinstance(preset, str):
            preset = TerrainPreset(preset)
        return dict(_TERRAIN_PRESETS)[preset]()


def _preset_flat() -> TerrainConfig:
    return TerrainConfig(
        slope=0.0, noise=0.0, step_freq=0.0,
        slope_profile=SlopeProfile.FLAT,
    )


def _preset_hilly() -> TerrainConfig:
    return TerrainConfig(
        slope=15 * np.pi / 180, noise=0.6, step_freq=0.0,
        section_len=1.5,
        slope_profile=SlopeProfile.RANDOM,
    )


def _preset_rough() -> TerrainConfig:
    return TerrainConfig(
        slope=5 * np.pi / 180, noise=1.0, step_freq=0.0,
        section_len=0.4,
        friction_range=(0.3, 0.9),
        obstacle_freq=0.15, obstacle_height=0.2, obstacle_width=0.3,
        slope_profile=SlopeProfile.RANDOM,
    )


def _preset_stairs() -> TerrainConfig:
    return TerrainConfig(
        slope=0.0, noise=0.0, step_freq=1.0,
        max_step=0.4, section_len=1.2,
    )


def _preset_mixed() -> TerrainConfig:
    return TerrainConfig(
        slope=10 * np.pi / 180, noise=0.9, step_freq=0.15,
        friction_range=(0.4, 0.8),
        gap_freq=0.05, gap_width=0.8,
        obstacle_freq=0.1, obstacle_height=0.25,
        slope_profile=SlopeProfile.RANDOM,
    )


_TERRAIN_PRESETS: list[tuple[TerrainPreset, Callable[[], TerrainConfig]]] = [
    (TerrainPreset.FLAT, _preset_flat),
    (TerrainPreset.HILLY, _preset_hilly),
    (TerrainPreset.ROUGH, _preset_rough),
    (TerrainPreset.STAIRS, _preset_stairs),
    (TerrainPreset.MIXED, _preset_mixed),
]


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
    _rng: np.random.Generator
    _slope_fn: SlopeGenerator
    _segment_counter: int

    def __init__(
        self,
        space: pm.Space | None = None,
        road_y: float = -5,
        config: WorldConfig | None = None,
    ) -> None:
        self.config = config if config is not None else DEFAULT_CONFIG

        # Seeded RNG for reproducible terrain
        terrain = self.config.terrain
        self._rng = np.random.default_rng(terrain.seed)
        self._segment_counter = 0

        # Resolve slope profile to a callable
        sp = terrain.slope_profile
        if callable(sp) and not isinstance(sp, (SlopeProfile, str)):
            self._slope_fn = sp
        else:
            key = sp.value if isinstance(sp, SlopeProfile) else str(sp)
            self._slope_fn = SLOPE_PROFILES[key]

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
        self._has_foot_filtering = False

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

        # When the linkage uses selective foot collision, tag road
        # segments with the ground filter so only foot edges touch them.
        if dl._non_foot_filter is not None:
            self._has_foot_filtering = True
            self._apply_ground_filter()

        for s in self.space.shapes:
            s.friction = self.config.ground_friction

        self.tune_solver()

    def _apply_ground_filter(self) -> None:
        """Tag all static-body shapes with the ground collision filter.

        Called once when a linkage with foot-edge filtering is added.
        Also patches newly built road segments in the road-building
        methods via ``_tag_ground_segment``.
        """
        from . import dynamiclinkage as _dl

        for shape in self.space.shapes:
            if shape.body is self.space.static_body:
                shape.filter = _dl.GROUND_FILTER

    def _tag_ground_segment(self, seg: pm.Segment) -> None:
        """Apply ground collision filter to *seg* when foot filtering is on."""
        if self._has_foot_filtering:
            from . import dynamiclinkage as _dl
            seg.filter = _dl.GROUND_FILTER

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

    def _segment_friction(self, terrain: TerrainConfig) -> float:
        """Return friction for a new road segment."""
        if terrain.friction_range is not None:
            lo, hi = terrain.friction_range
            return float(self._rng.uniform(lo, hi))
        return self.config.ground_friction

    def __build_road_step__(self, terrain: TerrainConfig, index: int) -> None:
        """Add a step (two points): vertical riser + horizontal plateau."""
        high = float(self._rng.random()) * terrain.max_step
        origin = self.road[index]
        # Vertical riser
        a = (origin[0], origin[1] + high)
        # Horizontal plateau
        direction = 1 if index else -1
        b = (a[0] + terrain.section_len * direction, a[1])

        friction = self._segment_friction(terrain)
        for p, q in [(origin, a), (a, b)]:
            s = pm.Segment(self.space.static_body, p, q, .1)
            s.friction = friction
            self.space.add(s)
            self._tag_ground_segment(s)
        self.road.insert(-index * len(self.road), a)
        self.road.insert(-index * len(self.road), b)

    def __build_road_segment__(self, terrain: TerrainConfig, index: int) -> None:
        """Add a slope segment (one point)."""
        angle = self._slope_fn(terrain, self._rng, self._segment_counter)
        if not index:
            angle = np.pi - angle
        a = pm.Vec2d(*cyl_to_cart(terrain.section_len, angle,
                                  *self.road[index]))
        s = pm.Segment(self.space.static_body, a, self.road[index], .1)
        s.friction = self._segment_friction(terrain)
        self.space.add(s)
        self._tag_ground_segment(s)
        self.road.insert(-index * len(self.road), a)

    def __build_road_gap__(self, terrain: TerrainConfig, index: int) -> None:
        """Add a gap (chasm) — road resumes at the same height after a gap."""
        origin = self.road[index]
        direction = 1 if index else -1
        far = (origin[0] + terrain.gap_width * direction, origin[1])
        # No segment added — the gap is empty space
        self.road.insert(-index * len(self.road), far)

    def __build_road_obstacle__(
        self, terrain: TerrainConfig, index: int,
    ) -> None:
        """Place a rectangular bump on the road surface."""
        origin = self.road[index]
        direction = 1 if index else -1
        h = float(self._rng.random()) * terrain.obstacle_height
        w = terrain.obstacle_width

        # Four corners of the bump
        bl = origin
        tl = (bl[0], bl[1] + h)
        tr = (tl[0] + w * direction, tl[1])
        br = (tr[0], bl[1])

        friction = self._segment_friction(terrain)
        for p, q in [(bl, tl), (tl, tr), (tr, br)]:
            s = pm.Segment(self.space.static_body, p, q, .1)
            s.friction = friction
            self.space.add(s)
            self._tag_ground_segment(s)

        # Road continues from the far base of the obstacle
        self.road.insert(-index * len(self.road), br)

    def build_road(self, positive: bool = False) -> None:
        """Build a road part.

        Parameters
        ----------
        positive : bool
            If *False* (default) the road is extended to the left.
        """
        terrain = self.config.terrain
        index = -positive  # 0 for left, -1 (True) for right
        self._segment_counter += 1

        roll = float(self._rng.random())
        if roll < terrain.gap_freq:
            self.__build_road_gap__(terrain, index)
        elif roll < terrain.gap_freq + terrain.obstacle_freq:
            self.__build_road_obstacle__(terrain, index)
        elif roll < terrain.gap_freq + terrain.obstacle_freq + terrain.step_freq:
            self.__build_road_step__(terrain, index)
        else:
            self.__build_road_segment__(terrain, index)


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
