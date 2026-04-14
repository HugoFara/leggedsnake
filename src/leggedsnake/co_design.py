"""
End-to-end walking mechanism design pipeline.

Connects pylinkage's topology co-optimization (``co_optimize``) with
leggedsnake's physics simulation, enabling automatic discovery of both
the mechanism topology and its dimensions for optimal walking performance.

Example::

    from leggedsnake import (
        DistanceFitness, StrideFitness,
        WalkingDesignSpec, optimize_walking_mechanism,
    )

    spec = WalkingDesignSpec(
        objectives=[DistanceFitness(duration=10.0, n_legs=4)],
        objective_names=["walking distance"],
        kinematic_prefilter=StrideFitness(),
        max_links=6,
    )
    result = optimize_walking_mechanism(spec)
    for walker, metrics in zip(result.walkers, result.fitness_results):
        print(walker.name, metrics)
"""
from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from .fitness import (
    DynamicFitness,
    FitnessResult,
    co_optimize_objective,
)
from .physicsengine import TerrainConfig, WorldConfig
from .walker import Walker

if TYPE_CHECKING:
    from pylinkage.optimization.co_optimization_types import CoOptimizationConfig
    from pylinkage.topology.catalog import TopologyCatalog


@dataclass
class WalkingDesignSpec:
    """Specification for a walking mechanism design problem.

    Attributes
    ----------
    objectives : list[DynamicFitness]
        Fitness evaluators to optimize (e.g., ``DistanceFitness``).
    objective_names : list[str] | None
        Human-readable names for objectives.
    world_config : WorldConfig | None
        Full simulation config. Overrides *terrain* if both given.
    terrain : TerrainConfig | None
        Terrain parameters. Ignored when *world_config* is provided.
    n_legs : int
        Number of legs for each walker candidate.
    motor_rates : float | dict[str, float]
        Motor angular velocity.
    max_links : int
        Maximum linkage complexity to explore in the catalog.
    kinematic_prefilter : DynamicFitness | None
        Optional fast pre-filter (e.g., ``StrideFitness``).
    catalog : TopologyCatalog | None
        Topology catalog. ``None`` loads the built-in catalog.
    co_opt_config : CoOptimizationConfig | None
        Co-optimization parameters. ``None`` uses defaults.
    use_warm_start : bool
        If True, run synthesis first to seed the optimizer.
    precision_points : list[tuple[float, float]] | None
        Target foot-path points for warm-start synthesis.
    """

    objectives: list[DynamicFitness] = field(default_factory=list)
    objective_names: list[str] | None = None
    world_config: WorldConfig | None = None
    terrain: TerrainConfig | None = None
    n_legs: int = 4
    motor_rates: float | dict[str, float] = -4.0
    max_links: int = 8
    kinematic_prefilter: DynamicFitness | None = None
    catalog: TopologyCatalog | None = None
    co_opt_config: CoOptimizationConfig | None = None
    use_warm_start: bool = False
    precision_points: list[tuple[float, float]] | None = None


@dataclass
class WalkingDesignResult:
    """Result of a full-stack walking mechanism design run.

    Attributes
    ----------
    walkers : list[Walker]
        Ranked Walker solutions (best first by primary objective).
    fitness_results : list[dict[str, FitnessResult]]
        Per-walker fitness results keyed by objective name.
    co_opt_result : CoOptimizationResult
        Raw co-optimization result with Pareto front and convergence data.
    spec : WalkingDesignSpec
        The design specification used.
    """

    walkers: list[Walker]
    fitness_results: list[dict[str, FitnessResult]]
    co_opt_result: Any  # CoOptimizationResult (lazy import)
    spec: WalkingDesignSpec


def optimize_walking_mechanism(
    spec: WalkingDesignSpec,
) -> WalkingDesignResult:
    """End-to-end walking mechanism design pipeline.

    Pipeline stages:

    1. Build ``WorldConfig`` from *spec* (terrain, gravity, etc.).
    2. Wrap each ``DynamicFitness`` objective via ``co_optimize_objective()``
       to match ``co_optimize()``'s minimization contract.
    3. Run ``warm_start_co_optimization()`` or ``co_optimize()``
       (depending on ``spec.use_warm_start``).
    4. Convert Pareto-front solutions to ``Walker`` instances via
       ``Walker.from_synthesis()``.
    5. Re-evaluate each walker with the original fitness functions to
       produce full ``FitnessResult`` dicts (with metrics and loci).
    6. Rank by primary objective score (highest first).

    Parameters
    ----------
    spec : WalkingDesignSpec
        Complete design specification.

    Returns
    -------
    WalkingDesignResult
        Ranked walker solutions with metrics and the raw Pareto front.

    Raises
    ------
    ValueError
        If *spec.objectives* is empty.
    """
    from pylinkage.optimization import co_optimize
    from pylinkage.optimization.co_optimization_types import CoOptimizationConfig
    from pylinkage.optimization.warm_start import warm_start_co_optimization

    if not spec.objectives:
        raise ValueError("At least one objective is required.")

    # 1. Build world config
    world_config = spec.world_config
    if world_config is None and spec.terrain is not None:
        world_config = WorldConfig(terrain=spec.terrain)

    # 2. Build co_optimize objectives (Linkage -> float, minimized)
    adapted: list[Any] = []
    names: list[str] = []
    for i, fitness in enumerate(spec.objectives):
        obj = co_optimize_objective(
            fitness,
            config=world_config,
            motor_rates=spec.motor_rates,
            kinematic_prefilter=spec.kinematic_prefilter,
        )
        adapted.append(obj)
        if spec.objective_names and i < len(spec.objective_names):
            names.append(spec.objective_names[i])
        else:
            names.append(f"objective_{i}")

    # 3. Configure and run co-optimization
    co_config = spec.co_opt_config
    if co_config is None:
        co_config = CoOptimizationConfig(max_links=spec.max_links)

    if spec.use_warm_start and spec.precision_points:
        co_result = warm_start_co_optimization(
            precision_points=spec.precision_points,
            objectives=adapted,
            catalog=spec.catalog,
            config=co_config,
            objective_names=names,
        )
    else:
        co_result = co_optimize(
            objectives=adapted,
            catalog=spec.catalog,
            config=co_config,
            objective_names=names,
        )

    # 4. Convert solutions to Walkers via the temporary SimLinkage shim.
    # Drop this routing in favor of a direct pylinkage call once 1.0
    # exposes a supported SimLinkage → Walker bridge.
    from .walker import _walker_from_sim_linkage

    walkers: list[Walker] = []
    fitness_results_list: list[dict[str, FitnessResult]] = []

    for sol in co_result.solutions:
        sim_linkage = getattr(sol, "linkage", None)
        if sim_linkage is None:
            continue
        try:
            walker = _walker_from_sim_linkage(
                sim_linkage, motor_rates=spec.motor_rates,
            )
        except Exception:
            continue
        if spec.n_legs > 1:
            try:
                walker.add_legs(spec.n_legs - 1)
            except Exception:
                continue

        # 5. Re-evaluate with full fitness for rich metrics
        result_dict: dict[str, FitnessResult] = {}
        for j, fitness in enumerate(spec.objectives):
            name = names[j] if j < len(names) else f"objective_{j}"
            fr = fitness(
                deepcopy(walker.topology),
                deepcopy(walker.dimensions),
                world_config,
            )
            result_dict[name] = fr

        walkers.append(walker)
        fitness_results_list.append(result_dict)

    # 6. Rank by primary objective (highest score first)
    if walkers:
        primary = names[0]
        ranked = sorted(
            range(len(walkers)),
            key=lambda i: fitness_results_list[i].get(
                primary, FitnessResult(score=0.0)
            ).score,
            reverse=True,
        )
        walkers = [walkers[i] for i in ranked]
        fitness_results_list = [fitness_results_list[i] for i in ranked]

    return WalkingDesignResult(
        walkers=walkers,
        fitness_results=fitness_results_list,
        co_opt_result=co_result,
        spec=spec,
    )
