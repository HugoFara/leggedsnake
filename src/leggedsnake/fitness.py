"""
Physics-aware fitness protocol for walking mechanism evaluation.

Provides a standardized ``DynamicFitness`` protocol and ``FitnessResult``
dataclass for evaluating walking mechanisms. Built-in implementations
cover common objectives (distance, efficiency, stride length). Adapter
functions bridge to pylinkage's optimizer contracts.

Example::

    from leggedsnake import (
        DistanceFitness, EfficiencyFitness, StrideFitness,
        as_eval_func, multi_objective_optimization,
    )

    # Use directly
    fitness = DistanceFitness(duration=10.0, n_legs=2)
    result = fitness(walker.topology, walker.dimensions)
    print(result.score, result.metrics)

    # Adapt to pylinkage optimizer contract
    objectives = [
        as_eval_func(DistanceFitness(duration=40.0, n_legs=4)),
        as_eval_func(EfficiencyFitness(duration=40.0, n_legs=4)),
    ]
"""
from __future__ import annotations

from collections.abc import Sequence
from copy import deepcopy
from dataclasses import dataclass, field
from typing import Any, Callable, Protocol, runtime_checkable

from pylinkage.dimensions import Dimensions
from pylinkage.hypergraph import HypergraphLinkage

from .physicsengine import World, WorldConfig, linkage_bb
from .stability import StabilityTimeSeries, compute_stability_snapshot
from .utility import step as step_check, stride, Point


@dataclass
class FitnessResult:
    """Rich result from a dynamic fitness evaluation.

    Attributes
    ----------
    score : float
        Primary fitness value (higher is better by convention).
    metrics : dict[str, float]
        Secondary metrics (e.g., ``{"distance": 12.3, "energy": 45.6}``).
        These feed directly into multi-objective optimization.
    valid : bool
        Whether the simulation completed without collapse or unbuildable error.
    loci : dict[str, list[Point]]
        Joint trajectories keyed by node ID. Empty when ``record_loci=False``.
    """

    score: float
    metrics: dict[str, float] = field(default_factory=dict)
    valid: bool = True
    loci: dict[str, list[Point]] = field(default_factory=dict)


@runtime_checkable
class DynamicFitness(Protocol):
    """Protocol for physics-based fitness evaluation of walking mechanisms.

    Implementations accept a topology and dimensions (the mechanism
    specification) plus an optional simulation configuration, and return
    a ``FitnessResult`` with a primary score and secondary metrics.

    The ``topology`` + ``dimensions`` signature means the same fitness
    works for any mechanism — not just a specific hand-coded linkage.
    """

    def __call__(
        self,
        topology: HypergraphLinkage,
        dimensions: Dimensions,
        config: WorldConfig | None = None,
    ) -> FitnessResult: ...


# ---------------------------------------------------------------------------
# Built-in fitness implementations
# ---------------------------------------------------------------------------


class DistanceFitness:
    """Evaluate total walking distance via physics simulation.

    Parameters
    ----------
    duration : float
        Simulation duration in seconds.
    n_legs : int
        Total number of legs when ``mirror=False`` (one-sided stack).
        Legs per side when ``mirror=True`` — total is ``2 * n_legs``.
    mirror : bool
        If True, call ``add_opposite_leg()`` before phase-offset copies,
        producing a left/right-symmetric walker. The canonical Jansen
        Strandbeest is ``n_legs=4, mirror=True`` (4 per side = 8 total).
        Default False preserves the one-sided behavior that predates
        this flag.
    motor_rates : float | dict[str, float]
        Motor angular velocity passed to Walker.
    record_loci : bool
        If True, record joint positions at each physics step.
    """

    def __init__(
        self,
        duration: float = 40.0,
        n_legs: int = 4,
        motor_rates: float | dict[str, float] = -4.0,
        record_loci: bool = False,
        mirror: bool = False,
    ) -> None:
        self.duration = duration
        self.n_legs = n_legs
        self.motor_rates = motor_rates
        self.record_loci = record_loci
        self.mirror = mirror

    def __call__(
        self,
        topology: HypergraphLinkage,
        dimensions: Dimensions,
        config: WorldConfig | None = None,
    ) -> FitnessResult:
        result = _run_simulation(
            topology, dimensions, config,
            duration=self.duration,
            n_legs=self.n_legs,
            motor_rates=self.motor_rates,
            record_loci=self.record_loci,
            mirror=self.mirror,
        )
        return FitnessResult(
            score=result.distance,
            metrics={
                "distance": result.distance,
                "total_efficiency": result.total_efficiency,
                "total_energy": result.total_energy,
            },
            valid=result.valid,
            loci=result.loci,
        )


class EfficiencyFitness:
    """Evaluate energy efficiency via physics simulation.

    Returns ``total_efficiency / total_energy`` as the primary score.
    Returns 0 if the walker doesn't travel at least ``min_distance``.

    Parameters
    ----------
    duration : float
        Simulation duration in seconds.
    n_legs : int
        Total legs (``mirror=False``) or legs per side (``mirror=True``).
    mirror : bool
        If True, build a symmetric walker via ``add_opposite_leg()``
        before phase-offset copies. See :class:`DistanceFitness`.
    motor_rates : float | dict[str, float]
        Motor angular velocity passed to Walker.
    min_distance : float
        Minimum distance the walker must cover for a non-zero score.
    record_loci : bool
        If True, record joint positions at each physics step.
    """

    def __init__(
        self,
        duration: float = 40.0,
        n_legs: int = 4,
        motor_rates: float | dict[str, float] = -4.0,
        min_distance: float = 5.0,
        record_loci: bool = False,
        mirror: bool = False,
    ) -> None:
        self.duration = duration
        self.n_legs = n_legs
        self.motor_rates = motor_rates
        self.min_distance = min_distance
        self.record_loci = record_loci
        self.mirror = mirror

    def __call__(
        self,
        topology: HypergraphLinkage,
        dimensions: Dimensions,
        config: WorldConfig | None = None,
    ) -> FitnessResult:
        result = _run_simulation(
            topology, dimensions, config,
            duration=self.duration,
            n_legs=self.n_legs,
            motor_rates=self.motor_rates,
            record_loci=self.record_loci,
            mirror=self.mirror,
        )
        if result.distance < self.min_distance or result.total_energy == 0:
            score = 0.0
        else:
            score = result.total_efficiency / result.total_energy

        return FitnessResult(
            score=score,
            metrics={
                "distance": result.distance,
                "total_efficiency": result.total_efficiency,
                "total_energy": result.total_energy,
                "efficiency_ratio": score,
            },
            valid=result.valid,
            loci=result.loci,
        )


class StrideFitness:
    """Evaluate kinematic stride length (no physics simulation).

    A fast, physics-free evaluation that measures horizontal travel of
    the foot locus. Suitable for initial exploration before expensive
    dynamic simulation.

    Parameters
    ----------
    lap_points : int
        Points per crank revolution for kinematic simulation.
    step_height : float
        Minimum height the foot must clear (obstacle clearance).
    step_width : float
        Minimum width the foot must clear.
    stride_height : float
        Height threshold for stride extraction.
    foot_index : int
        Index of the foot joint in the locus output.
    """

    def __init__(
        self,
        lap_points: int = 12,
        step_height: float = 0.5,
        step_width: float = 0.2,
        stride_height: float = 0.2,
        foot_index: int = -2,
    ) -> None:
        self.lap_points = lap_points
        self.step_height = step_height
        self.step_width = step_width
        self.stride_height = stride_height
        self.foot_index = foot_index

    def __call__(
        self,
        topology: HypergraphLinkage,
        dimensions: Dimensions,
        config: WorldConfig | None = None,
    ) -> FitnessResult:
        from .walker import Walker

        walker = Walker(
            deepcopy(topology), deepcopy(dimensions), motor_rates=-4.0,
        )

        try:
            loci = tuple(
                tuple(i) for i in walker.step(
                    iterations=self.lap_points,
                    dt=self.lap_points / self.lap_points,
                    skip_unbuildable=True,
                )
            )
        except Exception:
            return FitnessResult(score=0.0, valid=False)

        from pylinkage import extract_trajectory

        xs, ys = extract_trajectory(loci, self.foot_index)
        if xs.size == 0:
            return FitnessResult(score=0.0, valid=False)
        foot_locus = list(zip(xs.tolist(), ys.tolist()))
        if not step_check(foot_locus, self.step_height, self.step_width):
            return FitnessResult(
                score=0.0,
                metrics={"obstacle_clearance": 0.0},
                valid=True,
            )

        locus = stride(foot_locus, self.stride_height)
        score = max(k[0] for k in locus) - min(k[0] for k in locus)

        return FitnessResult(
            score=score,
            metrics={
                "stride_length": score,
                "obstacle_clearance": 1.0,
            },
            valid=True,
        )


class StabilityFitness:
    """Evaluate walking stability via physics simulation.

    Primary score is the mean tip-over margin. Stability metrics are
    included in ``FitnessResult.metrics``.

    Parameters
    ----------
    duration : float
        Simulation duration in seconds.
    n_legs : int
        Number of leg pairs.
    motor_rates : float | dict[str, float]
        Motor angular velocity.
    min_distance : float
        Minimum travel distance for a non-zero score.
    record_loci : bool
        If True, record joint trajectories.
    """

    def __init__(
        self,
        duration: float = 40.0,
        n_legs: int = 4,
        motor_rates: float | dict[str, float] = -4.0,
        min_distance: float = 2.0,
        record_loci: bool = False,
        mirror: bool = False,
    ) -> None:
        self.duration = duration
        self.n_legs = n_legs
        self.motor_rates = motor_rates
        self.min_distance = min_distance
        self.record_loci = record_loci
        self.mirror = mirror

    def __call__(
        self,
        topology: HypergraphLinkage,
        dimensions: Dimensions,
        config: WorldConfig | None = None,
    ) -> FitnessResult:
        result = _run_simulation(
            topology, dimensions, config,
            duration=self.duration,
            n_legs=self.n_legs,
            motor_rates=self.motor_rates,
            record_loci=self.record_loci,
            record_stability=True,
            mirror=self.mirror,
        )
        if not result.valid or result.distance < self.min_distance:
            return FitnessResult(score=0.0, valid=result.valid, loci=result.loci)

        stability = result.stability
        if stability is None or not stability.snapshots:
            return FitnessResult(score=0.0, valid=True, loci=result.loci)

        score = stability.mean_tip_over_margin
        metrics = {
            "distance": result.distance,
            **stability.summary_metrics(),
        }
        return FitnessResult(
            score=score, metrics=metrics, valid=True, loci=result.loci,
        )


class CompositeFitness:
    """Evaluate multiple objectives in a single simulation run.

    Runs physics once, then extracts distance, efficiency, stability,
    and gait metrics from the shared simulation data. This avoids
    redundant simulation when optimizing multiple objectives.

    The primary ``score`` is the distance; all requested objectives
    appear in ``FitnessResult.metrics``.

    Parameters
    ----------
    duration : float
        Simulation duration in seconds.
    n_legs : int
        Number of leg pairs.
    motor_rates : float | dict[str, float]
        Motor angular velocity.
    objectives : sequence of str
        Which metrics to compute. Supported: ``"distance"``,
        ``"efficiency"``, ``"stability"``, ``"gait"``.
    contact_threshold : float
        Y-coordinate below which a foot counts as in ground contact.
        Used when ``"gait"`` is in ``objectives``.
    """

    def __init__(
        self,
        duration: float = 40.0,
        n_legs: int = 4,
        motor_rates: float | dict[str, float] = -4.0,
        objectives: Sequence[str] = ("distance", "efficiency", "stability"),
        mirror: bool = False,
        contact_threshold: float = 0.1,
    ) -> None:
        self.duration = duration
        self.n_legs = n_legs
        self.motor_rates = motor_rates
        self.objectives = tuple(objectives)
        self.mirror = mirror
        self.contact_threshold = contact_threshold

    def __call__(
        self,
        topology: HypergraphLinkage,
        dimensions: Dimensions,
        config: WorldConfig | None = None,
    ) -> FitnessResult:
        needs_stability = "stability" in self.objectives
        result = _run_simulation(
            topology, dimensions, config,
            duration=self.duration,
            n_legs=self.n_legs,
            motor_rates=self.motor_rates,
            record_loci=True,
            record_stability=needs_stability,
            mirror=self.mirror,
        )
        if not result.valid:
            return FitnessResult(score=0.0, valid=False, loci=result.loci)

        metrics: dict[str, float] = {}

        if "distance" in self.objectives:
            metrics["distance"] = result.distance

        if "efficiency" in self.objectives:
            if result.total_energy > 0:
                metrics["efficiency"] = result.total_efficiency / result.total_energy
            else:
                metrics["efficiency"] = 0.0

        if needs_stability and result.stability is not None:
            metrics.update(result.stability.summary_metrics())

        if "gait" in self.objectives:
            from .gait_analysis import analyze_gait

            gait = analyze_gait(
                loci=result.loci,
                foot_ids=result.foot_ids,
                stability=result.stability,
                dt=result.dt,
                contact_threshold=self.contact_threshold,
            )
            gait_metrics = gait.summary_metrics()
            gait_metrics["energy_per_cycle"] = gait.energy_per_cycle(
                result.total_energy,
            )
            metrics.update(gait_metrics)

        return FitnessResult(
            score=result.distance,
            metrics=metrics,
            valid=True,
            loci=result.loci,
        )


class GaitFitness:
    """Evaluate gait quality via physics simulation.

    Runs the full dynamic sim, analyzes the foot trajectories into touchdown
    / liftoff events, and reports stride, duty-factor, asymmetry, and
    energy-per-cycle metrics. The primary score is ``mean_stride_length``
    (simulation-frame distance between consecutive touchdowns of the same
    foot), so higher is better.

    Parameters
    ----------
    duration : float
        Simulation duration in seconds.
    n_legs : int
        Total legs (``mirror=False``) or legs per side (``mirror=True``).
    mirror : bool
        Mirror the leg across the chassis via ``add_opposite_leg()``.
    motor_rates : float | dict[str, float]
        Motor angular velocity.
    contact_threshold : float
        Y-coordinate below which a foot counts as in ground contact.
    record_loci : bool
        If True, return joint trajectories in ``FitnessResult.loci``.
        Gait analysis runs regardless — this only controls what's returned.
    """

    def __init__(
        self,
        duration: float = 40.0,
        n_legs: int = 4,
        motor_rates: float | dict[str, float] = -4.0,
        mirror: bool = False,
        contact_threshold: float = 0.1,
        record_loci: bool = False,
    ) -> None:
        self.duration = duration
        self.n_legs = n_legs
        self.motor_rates = motor_rates
        self.mirror = mirror
        self.contact_threshold = contact_threshold
        self.record_loci = record_loci

    def __call__(
        self,
        topology: HypergraphLinkage,
        dimensions: Dimensions,
        config: WorldConfig | None = None,
    ) -> FitnessResult:
        from .gait_analysis import analyze_gait

        result = _run_simulation(
            topology, dimensions, config,
            duration=self.duration,
            n_legs=self.n_legs,
            motor_rates=self.motor_rates,
            record_loci=True,
            mirror=self.mirror,
        )
        if not result.valid:
            return FitnessResult(score=0.0, valid=False)

        gait = analyze_gait(
            loci=result.loci,
            foot_ids=result.foot_ids,
            dt=result.dt,
            contact_threshold=self.contact_threshold,
        )
        metrics = gait.summary_metrics()
        metrics["energy_per_cycle"] = gait.energy_per_cycle(result.total_energy)
        metrics["distance"] = result.distance
        metrics["total_energy"] = result.total_energy

        return FitnessResult(
            score=metrics["mean_stride_length"],
            metrics=metrics,
            valid=True,
            loci=result.loci if self.record_loci else {},
        )


# ---------------------------------------------------------------------------
# Adapter functions for optimizer compatibility
# ---------------------------------------------------------------------------


def as_eval_func(
    fitness: DynamicFitness,
    config: WorldConfig | None = None,
    walker_factory: Callable[[], Any] | None = None,
    negate: bool = False,
) -> Callable[..., float]:
    """Adapt a ``DynamicFitness`` to pylinkage's optimizer contract.

    Returns a function with signature ``(linkage, dims, pos) -> float``
    compatible with ``multi_objective_optimization``, ``chain_optimizers``,
    and the walking objective factories.

    Parameters
    ----------
    fitness : DynamicFitness
        The fitness evaluator to adapt.
    config : WorldConfig | None
        Simulation config override.  If None, the fitness uses its own default.
    walker_factory : callable, optional
        Zero-argument callable returning a fresh Walker. If provided, a new
        walker is built per evaluation (ignoring the ``linkage`` argument) —
        required for thread/process safety in multi-worker NSGA evaluation.
        If None, the adapter mutates the ``linkage`` passed in (pylinkage's
        default for sequential optimizers).
    negate : bool
        If True, return ``-score`` so pylinkage's minimization-based
        optimizers (``multi_objective_optimization``, ``minimize_linkage``,
        etc.) treat walking scores as maximization. Default False.

    Returns
    -------
    callable
        ``eval_func(linkage, dims, pos) -> float``

    Example
    -------
    Feed a walking fitness into pylinkage's multi-objective optimizer::

        from leggedsnake import DistanceFitness, as_eval_func
        from pylinkage.optimization import multi_objective_optimization

        eval_fn = as_eval_func(
            DistanceFitness(duration=20, n_legs=4),
            walker_factory=make_walker,
            negate=True,  # pylinkage minimizes
        )
        ensemble = multi_objective_optimization(
            objectives=[eval_fn],
            linkage=make_walker(),
            bounds=bounds,
        )
    """

    def _eval(linkage: Any, dims: Sequence[float], pos: Sequence[Any]) -> float:
        if walker_factory is not None:
            walker = walker_factory()
            try:
                walker.set_num_constraints(list(dims))
                result = fitness(
                    deepcopy(walker.topology),
                    deepcopy(walker.dimensions),
                    config,
                )
                score = result.score if result.valid else 0.0
            except Exception:
                score = 0.0
        else:
            linkage.set_num_constraints(dims)
            linkage.set_coords(pos)
            result = fitness(linkage.topology, linkage.dimensions, config)
            score = result.score
        return -score if negate else score

    return _eval


def as_ga_fitness(
    fitness: DynamicFitness,
    walker_factory: Callable[[], Any],
    config: WorldConfig | None = None,
    minimize: bool = False,
) -> Callable[..., tuple[float, list[Any]]]:
    """Adapt a ``DynamicFitness`` to the GA optimizer's DNA contract.

    Returns a function with signature ``(dna) -> (score, positions)``
    compatible with ``GeneticOptimization``.

    Parameters
    ----------
    fitness : DynamicFitness
        The fitness evaluator to adapt.
    walker_factory : callable
        Zero-argument callable that returns a fresh Walker instance.
        Called each evaluation to avoid mutation across generations.
    config : WorldConfig | None
        Simulation config override.
    minimize : bool
        If True, negate the score (GA maximizes by default).

    Returns
    -------
    callable
        ``ga_fitness(dna) -> (float, list[tuple[float, float]])``
    """

    def _ga(dna: list[Any]) -> tuple[float, list[Any]]:
        walker = walker_factory()
        walker.set_num_constraints(dna[1])
        walker.set_coords(dna[2])
        result = fitness(walker.topology, walker.dimensions, config)
        score = -result.score if minimize else result.score
        return score, list(walker.get_coords())

    return _ga


def chain_walking_optimizers(
    fitness: DynamicFitness,
    linkage: Any,
    stages: Sequence[tuple[Callable[..., Any], dict[str, Any]]],
    config: WorldConfig | None = None,
    order_relation: Callable[[float, float], float] = max,
    verbose: bool = True,
) -> Any:
    """Chain optimizer stages on a walking ``DynamicFitness``.

    Thin wrapper around :func:`pylinkage.optimization.chain_optimizers`
    that adapts a leggedsnake ``DynamicFitness`` via :func:`as_eval_func`
    and forwards to pylinkage. Each stage receives the previous stage's
    best solution as its starting point (``center`` for population
    methods, ``x0`` for local ones) — same contract as pylinkage.

    Parameters
    ----------
    fitness : DynamicFitness
        Walking fitness evaluator (e.g., ``DistanceFitness``).
    linkage : Walker
        Mechanism to optimize. Must implement ``set_num_constraints``
        / ``set_coords`` / ``get_coords`` (Walker does).
    stages : sequence of (optimizer, kwargs)
        Each optimizer is one of pylinkage's
        (``particle_swarm_optimization``, ``differential_evolution_optimization``,
        ``dual_annealing_optimization``, ``minimize_linkage``, ...) or
        leggedsnake's ``genetic_algorithm_optimization``. ``kwargs`` are
        passed through, minus ``eval_func`` / ``linkage``.
    config : WorldConfig | None
        Simulation config passed into the fitness.
    order_relation : callable
        ``max`` for maximize (default), ``min`` for minimize.
    verbose : bool
        Print stage headers.

    Returns
    -------
    Ensemble
        Result of the final stage — same type pylinkage returns.

    Example
    -------
    A global → local pipeline for a walker:

        from leggedsnake import (
            DistanceFitness, chain_walking_optimizers,
            differential_evolution_optimization, minimize_linkage,
            dual_annealing_optimization,
        )

        result = chain_walking_optimizers(
            DistanceFitness(duration=20.0, n_legs=4),
            my_walker,
            stages=[
                (differential_evolution_optimization, {"maxiter": 200}),
                (dual_annealing_optimization, {"maxiter": 100}),
                (minimize_linkage, {"method": "Nelder-Mead", "maxiter": 500}),
            ],
        )
        best = result[0]  # Member from the final stage
    """
    from pylinkage.optimization import chain_optimizers

    return chain_optimizers(
        eval_func=as_eval_func(fitness, config=config),
        linkage=linkage,
        stages=stages,
        order_relation=order_relation,
        verbose=verbose,
    )


def co_optimize_objective(
    fitness: DynamicFitness,
    config: WorldConfig | None = None,
    motor_rates: float | dict[str, float] = -4.0,
    kinematic_prefilter: DynamicFitness | None = None,
) -> Callable[..., float]:
    """Adapt a ``DynamicFitness`` to pylinkage's ``co_optimize()`` contract.

    Returns ``Callable[[Linkage], float]`` where the result is **minimized**
    (``co_optimize`` minimizes). Walking performance scores (higher = better)
    are negated so that better walkers have lower objective values.

    Supports an optional two-stage pipeline: a fast kinematic pre-filter
    rejects mechanisms with poor foot paths before the expensive physics
    simulation runs.

    Parameters
    ----------
    fitness : DynamicFitness
        The physics-based fitness evaluator (e.g., ``DistanceFitness``).
    config : WorldConfig | None
        Simulation config override.
    motor_rates : float | dict[str, float]
        Motor angular velocity applied to each Walker. Default -4.0.
    kinematic_prefilter : DynamicFitness | None
        Optional fast kinematic check (e.g., ``StrideFitness``).
        If the pre-filter returns ``score <= 0`` or ``valid=False``,
        the mechanism is rejected (returns ``inf``) without running physics.

    Returns
    -------
    callable
        ``objective(linkage: Linkage) -> float`` for ``co_optimize()``.
    """

    # Bridges a ``simulation.Linkage`` (SimLinkage) — what pylinkage's
    # ``co_optimize`` hands in — to a Walker through the temporary
    # ``_walker_from_sim_linkage`` shim. Swap to a pylinkage-native
    # bridge once one ships (targeted for 1.0).

    def _objective(linkage: Any) -> float:
        from .walker import _walker_from_sim_linkage

        try:
            walker = _walker_from_sim_linkage(linkage, motor_rates=motor_rates)
        except Exception:
            return float("inf")

        # Fast kinematic pre-filter
        if kinematic_prefilter is not None:
            pre_result = kinematic_prefilter(
                walker.topology, walker.dimensions, config,
            )
            if not pre_result.valid or pre_result.score <= 0:
                return float("inf")

        # Full dynamic evaluation
        result = fitness(walker.topology, walker.dimensions, config)

        if not result.valid or result.score <= 0:
            return float("inf")

        # Negate: co_optimize minimizes, but higher walking score is better
        return -result.score

    return _objective


# ---------------------------------------------------------------------------
# Internal simulation helper
# ---------------------------------------------------------------------------


@dataclass
class _SimulationResult:
    """Internal container for raw simulation outputs."""

    distance: float = 0.0
    total_efficiency: float = 0.0
    total_energy: float = 0.0
    valid: bool = True
    loci: dict[str, list[Point]] = field(default_factory=dict)
    stability: StabilityTimeSeries | None = None
    foot_ids: list[str] = field(default_factory=list)
    dt: float = 0.02


def _run_simulation(
    topology: HypergraphLinkage,
    dimensions: Dimensions,
    config: WorldConfig | None,
    *,
    duration: float,
    n_legs: int,
    motor_rates: float | dict[str, float],
    record_loci: bool,
    record_stability: bool = False,
    mirror: bool = False,
) -> _SimulationResult:
    """Build a Walker, run physics, and collect results.

    Parameters
    ----------
    record_stability : bool
        If True, collect ``StabilitySnapshot`` at each physics step.
    mirror : bool
        If True, mirror the leg across the chassis via
        ``add_opposite_leg()`` before phase-offset copies.
    """
    from .walker import Walker

    walker = Walker(
        deepcopy(topology), deepcopy(dimensions),
        motor_rates=motor_rates,
    )

    # Verify the mechanism is buildable
    try:
        from pylinkage import UnbuildableError
        tuple(walker.step())
    except UnbuildableError:
        return _SimulationResult(valid=False)

    foot_ids = walker.get_feet()

    # Add legs. With mirror=True, ``n_legs`` counts legs per side; the
    # ``add_opposite_leg()`` call doubles the total. Without mirroring,
    # ``n_legs`` is the total leg count (all on one side of the chassis).
    if mirror:
        walker.add_opposite_leg()
    if n_legs > 1:
        walker.add_legs(n_legs - 1)
    if mirror or n_legs > 1:
        foot_ids = walker.get_feet()

    # Place the road just beneath the walker's feet so it starts resting on
    # the ground instead of intersecting or free-falling onto it — matches
    # worldvisualizer and the nb02 ``simulate`` helper.
    road_y = linkage_bb(walker)[0] - 1
    world = World(config=config, road_y=road_y)
    world.add_linkage(walker)

    dt = world.config.physics_period
    steps = int(duration / dt)
    total_efficiency = 0.0
    total_energy = 0.0

    # Set up loci recording
    loci: dict[str, list[Point]] = {}
    if record_loci:
        dl = world.linkages[0]
        for proxy in dl.joints:
            loci[proxy.name] = []

    # Set up stability recording
    stability_series: StabilityTimeSeries | None = None
    prev_snap = None
    if record_stability:
        stability_series = StabilityTimeSeries()
        gravity_mag = abs(world.space.gravity[1])

    for step_i in range(steps):
        result = world.update()
        if result is not None:
            total_efficiency += result[0]
            total_energy += result[1]

        dl = world.linkages[0]

        if record_loci:
            for proxy in dl.joints:
                loci[proxy.name].append((proxy.x, proxy.y))

        if record_stability:
            snap = compute_stability_snapshot(
                dl, prev_snap,
                time=step_i * dt,
                dt=dt,
                gravity=gravity_mag,
                foot_ids=foot_ids,
                static_body=world.space.static_body,
            )
            stability_series.snapshots.append(snap)  # type: ignore[union-attr]
            prev_snap = snap

    distance = float(world.linkages[0].body.position.x)

    return _SimulationResult(
        distance=distance,
        total_efficiency=total_efficiency,
        total_energy=total_energy,
        valid=True,
        loci=loci,
        stability=stability_series,
        foot_ids=foot_ids,
        dt=dt,
    )
