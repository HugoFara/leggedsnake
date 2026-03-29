"""
Multi-objective NSGA-II/III optimizer for walking mechanisms.

Wraps pymoo's NSGA-II/III to optimize walking linkages against multiple
physics-based objectives simultaneously (e.g., distance + efficiency +
stability). Returns a ``ParetoFront`` of non-dominated solutions with
optional gait analysis and stability data.

Example::

    from leggedsnake import (
        DistanceFitness, StabilityFitness, Walker,
        nsga_walking_optimization, NsgaWalkingConfig,
    )

    def make_walker():
        return Walker(topology, dimensions)

    result = nsga_walking_optimization(
        walker_factory=make_walker,
        objectives=[DistanceFitness(duration=10), StabilityFitness(duration=10)],
        bounds=(lower, upper),
        objective_names=["distance", "stability"],
        nsga_config=NsgaWalkingConfig(n_generations=50, pop_size=40),
    )

    best = result.pareto_front.best_compromise()
    print(best.scores)
"""
from __future__ import annotations

from collections.abc import Sequence
from copy import deepcopy
from dataclasses import dataclass, field
from typing import Any, Callable, Literal

import numpy as np

from pylinkage.optimization.collections import ParetoFront, ParetoSolution

from .fitness import CompositeFitness, DynamicFitness, FitnessResult
from .gait_analysis import GaitAnalysisResult, analyze_gait
from .stability import StabilityTimeSeries


def _check_pymoo() -> None:
    """Raise ImportError with install hint if pymoo is missing."""
    try:
        import pymoo  # noqa: F401
    except ImportError:
        raise ImportError(
            "pymoo is required for multi-objective optimization. "
            "Install it with: pip install pymoo"
        ) from None


# ---------------------------------------------------------------------------
# Configuration and result dataclasses
# ---------------------------------------------------------------------------


@dataclass
class NsgaWalkingConfig:
    """Configuration for NSGA-II/III walking optimization.

    Attributes
    ----------
    n_generations : int
        Number of evolutionary generations.
    pop_size : int
        Population size per generation.
    algorithm : {"nsga2", "nsga3"}
        Multi-objective algorithm. NSGA-II is best for 2–3 objectives;
        NSGA-III handles more.
    seed : int | None
        Random seed for reproducibility.
    verbose : bool
        Show progress during optimization.
    crossover_prob : float
        Simulated binary crossover probability.
    mutation_eta : float
        Polynomial mutation distribution index.
    """

    n_generations: int = 100
    pop_size: int = 100
    algorithm: Literal["nsga2", "nsga3"] = "nsga2"
    seed: int | None = None
    verbose: bool = True
    crossover_prob: float = 0.9
    mutation_eta: float = 20.0


@dataclass
class NsgaWalkingResult:
    """Result of multi-objective walking optimization.

    Attributes
    ----------
    pareto_front : ParetoFront
        Non-dominated solutions with scores and dimensions.
    gait_analyses : dict[int, GaitAnalysisResult] | None
        Gait analysis for each Pareto solution (index → analysis).
        Only populated when ``include_gait=True``.
    stability_series : dict[int, StabilityTimeSeries] | None
        Stability time series for each Pareto solution.
        Only populated when ``include_stability=True``.
    config : NsgaWalkingConfig
        The configuration used.
    """

    pareto_front: ParetoFront
    gait_analyses: dict[int, GaitAnalysisResult] | None = None
    stability_series: dict[int, StabilityTimeSeries] | None = None
    config: NsgaWalkingConfig = field(default_factory=NsgaWalkingConfig)

    def best_for_objective(self, objective_index: int) -> ParetoSolution:
        """Return the Pareto solution that is best for a single objective.

        Parameters
        ----------
        objective_index : int
            Index into the scores tuple (0-based).
        """
        return min(
            self.pareto_front.solutions,
            key=lambda s: s.scores[objective_index],
        )

    def best_compromise(
        self, weights: Sequence[float] | None = None,
    ) -> ParetoSolution:
        """Return the best compromise solution (delegates to ParetoFront)."""
        return self.pareto_front.best_compromise(weights)


# ---------------------------------------------------------------------------
# Pymoo Problem wrapper
# ---------------------------------------------------------------------------


class WalkingNsgaProblem:
    """Pymoo Problem for multi-objective walking optimization.

    Each candidate is a vector of linkage constraint dimensions.
    Evaluation builds a Walker, runs physics for each objective,
    and returns the negated scores (pymoo minimizes; our fitness
    functions maximize).

    Parameters
    ----------
    walker_factory : callable
        Zero-argument callable returning a fresh Walker.
    objectives : sequence of DynamicFitness
        Fitness evaluators. Each produces a ``FitnessResult.score``.
    bounds : tuple of (lower, upper)
        Parameter bounds as sequences of floats.
    config : WorldConfig | None
        Simulation config override.
    """

    def __init__(
        self,
        walker_factory: Callable[[], Any],
        objectives: Sequence[DynamicFitness],
        bounds: tuple[Sequence[float], Sequence[float]],
        config: Any | None = None,
    ) -> None:
        _check_pymoo()
        from pymoo.core.problem import Problem

        self.walker_factory = walker_factory
        self.objectives = list(objectives)
        self.config = config
        self._n_obj = len(objectives)

        xl = np.array(bounds[0], dtype=float)
        xu = np.array(bounds[1], dtype=float)
        n_var = len(xl)

        # Capture reference for the nested class
        outer = self

        class _Problem(Problem):
            def __init__(self_inner) -> None:
                super().__init__(
                    n_var=n_var,
                    n_obj=outer._n_obj,
                    n_ieq_constr=0,
                    xl=xl,
                    xu=xu,
                )

            def _evaluate(
                self_inner, X: np.ndarray, out: dict, *args: Any, **kwargs: Any,
            ) -> None:
                F = np.full((X.shape[0], outer._n_obj), float("inf"))
                for i in range(X.shape[0]):
                    dims = X[i].tolist()
                    scores = outer._evaluate_candidate(dims)
                    # Negate: pymoo minimizes, our fitness maximizes
                    F[i] = [-s for s in scores]
                out["F"] = F

        self._problem = _Problem()

    @property
    def problem(self) -> Any:
        """The pymoo Problem instance."""
        return self._problem

    def _evaluate_candidate(
        self, dims: list[float],
    ) -> list[float]:
        """Evaluate a single candidate against all objectives."""
        scores: list[float] = []
        walker = self.walker_factory()
        walker.set_num_constraints(dims)

        for objective in self.objectives:
            try:
                result = objective(
                    deepcopy(walker.topology),
                    deepcopy(walker.dimensions),
                    self.config,
                )
                scores.append(result.score if result.valid else 0.0)
            except Exception:
                scores.append(0.0)

        return scores


# ---------------------------------------------------------------------------
# High-level entry point
# ---------------------------------------------------------------------------


def nsga_walking_optimization(
    walker_factory: Callable[[], Any],
    objectives: Sequence[DynamicFitness],
    bounds: tuple[Sequence[float], Sequence[float]],
    objective_names: Sequence[str] | None = None,
    nsga_config: NsgaWalkingConfig | None = None,
    world_config: Any | None = None,
    include_gait: bool = False,
    include_stability: bool = False,
) -> NsgaWalkingResult:
    """Multi-objective walking optimization via NSGA-II/III.

    Optimizes a walking mechanism against multiple ``DynamicFitness``
    objectives simultaneously using pymoo. Returns a Pareto front of
    non-dominated solutions with optional gait and stability analysis.

    Parameters
    ----------
    walker_factory : callable
        Zero-argument callable returning a fresh Walker instance.
    objectives : sequence of DynamicFitness
        Fitness evaluators (e.g., ``DistanceFitness``, ``StabilityFitness``).
    bounds : tuple of (lower, upper)
        Parameter bounds as sequences of floats, one per constraint dimension.
    objective_names : sequence of str, optional
        Human-readable names for each objective.
    nsga_config : NsgaWalkingConfig, optional
        Algorithm configuration. Uses defaults if None.
    world_config : WorldConfig, optional
        Simulation config passed to fitness evaluators.
    include_gait : bool
        If True, run gait analysis on Pareto-front solutions after
        optimization completes.
    include_stability : bool
        If True, collect stability time series for Pareto-front solutions.

    Returns
    -------
    NsgaWalkingResult
        Pareto front with optional gait/stability analysis.
    """
    _check_pymoo()
    from pymoo.algorithms.moo.nsga2 import NSGA2
    from pymoo.optimize import minimize as pymoo_minimize

    cfg = nsga_config or NsgaWalkingConfig()
    n_obj = len(objectives)

    if objective_names is None:
        objective_names = [f"objective_{i}" for i in range(n_obj)]

    # Build problem
    problem = WalkingNsgaProblem(
        walker_factory=walker_factory,
        objectives=objectives,
        bounds=bounds,
        config=world_config,
    )

    # Build algorithm
    if cfg.algorithm == "nsga3" and n_obj > 2:
        from pymoo.algorithms.moo.nsga3 import NSGA3
        from pymoo.util.ref_dirs import get_reference_directions

        n_partitions = max(4, 12 - n_obj)
        ref_dirs = get_reference_directions("das-dennis", n_obj, n_partitions=n_partitions)
        algo = NSGA3(pop_size=cfg.pop_size, ref_dirs=ref_dirs)
    else:
        algo = NSGA2(pop_size=cfg.pop_size)

    # Run optimization
    res = pymoo_minimize(
        problem.problem,
        algo,
        ("n_gen", cfg.n_generations),
        seed=cfg.seed,
        verbose=cfg.verbose,
    )

    # Package results into ParetoFront
    if res.F is None or res.X is None:
        return NsgaWalkingResult(
            pareto_front=ParetoFront([], tuple(objective_names)),
            config=cfg,
        )

    # Get initial positions from a fresh walker for ParetoSolution
    ref_walker = walker_factory()
    init_pos = ref_walker.get_coords()

    solutions: list[ParetoSolution] = []
    for i in range(res.F.shape[0]):
        # Negate back: pymoo minimized negated scores
        scores = tuple(-float(v) for v in res.F[i])
        solutions.append(ParetoSolution(
            scores=scores,
            dimensions=res.X[i],
            init_positions=init_pos,
        ))

    pareto = ParetoFront(solutions, tuple(objective_names))

    # Optional post-hoc analysis on Pareto front solutions
    gait_analyses: dict[int, GaitAnalysisResult] | None = None
    stability_map: dict[int, StabilityTimeSeries] | None = None

    if include_gait or include_stability:
        gait_analyses = {} if include_gait else None
        stability_map = {} if include_stability else None

        for idx, sol in enumerate(solutions):
            walker = walker_factory()
            walker.set_num_constraints(sol.dimensions.tolist())

            # Use CompositeFitness for a single efficient simulation
            composite = CompositeFitness(
                objectives=("distance", "efficiency", "stability"),
            )
            result = composite(
                deepcopy(walker.topology),
                deepcopy(walker.dimensions),
                world_config,
            )

            if include_gait and result.valid and result.loci:
                foot_ids = walker.get_feet()
                gait = analyze_gait(
                    loci=result.loci,
                    foot_ids=foot_ids,
                    dt=0.02,
                )
                gait_analyses[idx] = gait  # type: ignore[index]

            # For stability, re-run with stability recording
            if include_stability and result.valid:
                from .fitness import _run_simulation
                stab_result = _run_simulation(
                    deepcopy(walker.topology),
                    deepcopy(walker.dimensions),
                    world_config,
                    duration=composite.duration,
                    n_legs=composite.n_legs,
                    motor_rates=composite.motor_rates,
                    record_loci=False,
                    record_stability=True,
                )
                if stab_result.stability is not None:
                    stability_map[idx] = stab_result.stability  # type: ignore[index]

    return NsgaWalkingResult(
        pareto_front=pareto,
        gait_analyses=gait_analyses,
        stability_series=stability_map,
        config=cfg,
    )
