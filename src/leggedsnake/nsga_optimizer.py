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

from .fitness import CompositeFitness, DynamicFitness, as_eval_func
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
    n_workers: int = 1
    """Number of parallel workers for fitness evaluation.
    1 = sequential (default). >1 uses a process pool."""


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

        Walking fitnesses maximize (distance, efficiency, stability all
        larger-is-better), and scores are stored un-negated after
        ``_ensemble_to_pareto_front``, so ``max`` is correct here.

        Parameters
        ----------
        objective_index : int
            Index into the scores tuple (0-based).
        """
        return max(
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
        n_workers: int = 1,
    ) -> None:
        _check_pymoo()
        from pymoo.core.problem import Problem

        self.walker_factory = walker_factory
        self.objectives = list(objectives)
        self.config = config
        self._n_obj = len(objectives)
        self._n_workers = max(1, n_workers)
        # Lazy-created on first parallel batch and reused across
        # generations. Pool startup forks N worker processes — doing
        # that per generation wastes ~50–500 ms per generation depending
        # on import weight. ``close()`` shuts it down.
        self._pool: Any = None

        xl = np.array(bounds[0], dtype=float)
        xu = np.array(bounds[1], dtype=float)
        n_var = len(xl)

        # Capture reference for the nested class
        outer = self

        class _Problem(Problem):  # type: ignore[misc]  # pymoo Problem is Any-typed
            def __init__(self_inner) -> None:
                super().__init__(
                    n_var=n_var,
                    n_obj=outer._n_obj,
                    n_ieq_constr=0,
                    xl=xl,
                    xu=xu,
                )

            def _evaluate(
                self_inner,
                X: np.ndarray,
                out: dict[str, Any],
                *args: Any,
                **kwargs: Any,
            ) -> None:
                out["F"] = outer._evaluate_batch(X)

        self._problem = _Problem()

    @property
    def problem(self) -> Any:
        """The pymoo Problem instance."""
        return self._problem

    def _evaluate_batch(self, X: np.ndarray) -> np.ndarray:
        """Evaluate a batch of candidates, optionally in parallel."""
        n_pop = X.shape[0]
        F = np.full((n_pop, self._n_obj), float("inf"))

        if self._n_workers <= 1:
            for i in range(n_pop):
                scores = self._evaluate_candidate(X[i].tolist())
                F[i] = [-s for s in scores]
        else:
            from concurrent.futures import as_completed

            pool = self._get_pool()
            candidates = [X[i].tolist() for i in range(n_pop)]
            futures = {
                pool.submit(
                    _evaluate_candidate_worker,
                    self.walker_factory,
                    self.objectives,
                    self.config,
                    candidate,
                ): idx
                for idx, candidate in enumerate(candidates)
            }
            for future in as_completed(futures):
                idx = futures[future]
                try:
                    scores = future.result()
                    F[idx] = [-s for s in scores]
                except Exception:
                    F[idx] = [float("inf")] * self._n_obj

        return F

    def _get_pool(self) -> Any:
        """Return the shared process pool, creating it on first use.

        Uses the ``spawn`` start method explicitly. The default ``fork``
        emits a ``DeprecationWarning`` under Python 3.12+ when the
        parent has live threads (which pytest, matplotlib, and many
        Jupyter setups do) and risks deadlocks in the child. Spawn pays
        a one-time module-import cost per worker — already amortized
        because the pool is reused across generations.
        """
        if self._pool is None:
            import multiprocessing as mp
            from concurrent.futures import ProcessPoolExecutor

            self._pool = ProcessPoolExecutor(
                max_workers=self._n_workers,
                mp_context=mp.get_context("spawn"),
            )
        return self._pool

    def close(self) -> None:
        """Shut down the shared process pool, if one was created.

        Safe to call repeatedly. The driver in
        :func:`nsga_walking_optimization` invokes this in a finally
        block so workers don't outlive the optimization.
        """
        if self._pool is not None:
            self._pool.shutdown(wait=True)
            self._pool = None

    def __del__(self) -> None:  # pragma: no cover — best-effort cleanup
        # Defensive cleanup if the caller forgot ``close()``. Pool
        # workers otherwise linger until the parent process exits.
        try:
            self.close()
        except Exception:
            pass

    def _evaluate_candidate(
        self, dims: list[float],
    ) -> list[float]:
        """Evaluate a single candidate against all objectives.

        Delegates to :func:`as_eval_func` (walker_factory variant) so the
        same adapter works with pylinkage's ``multi_objective_optimization``,
        ``chain_optimizers``, and standalone optimizers.
        """
        return _evaluate_candidate_worker(
            self.walker_factory, self.objectives, self.config, dims,
        )


# ---------------------------------------------------------------------------
# Top-level worker for parallel evaluation (must be picklable)
# ---------------------------------------------------------------------------


def _evaluate_candidate_worker(
    walker_factory: Callable[[], Any],
    objectives: list[DynamicFitness],
    config: Any | None,
    dims: list[float],
) -> list[float]:
    """Evaluate a single candidate via :func:`as_eval_func`.

    Must be top-level for multiprocessing pickling.
    """
    eval_funcs = [
        as_eval_func(obj, config=config, walker_factory=walker_factory)
        for obj in objectives
    ]
    # eval_func built with walker_factory ignores the linkage/pos args.
    return [fn(None, dims, ()) for fn in eval_funcs]


# ---------------------------------------------------------------------------
# Ensemble → ParetoFront adapter and NSGA backends
# ---------------------------------------------------------------------------


def _ensemble_to_pareto_front(
    ensemble: Any,
    objective_names: Sequence[str],
    init_positions: Sequence[Any],
    negate_scores: bool = True,
) -> ParetoFront:
    """Bridge pylinkage's ``Ensemble`` → leggedsnake's ``ParetoFront``.

    The Ensemble stores dimensions and scores columnarly; ParetoFront
    holds per-solution ``ParetoSolution`` objects. Walking fitnesses
    maximize while pylinkage minimizes, so scores are negated back when
    ``negate_scores`` is True.
    """
    n = ensemble.n_members
    dims = ensemble.dimensions
    score_cols = [np.asarray(ensemble.scores[name]) for name in objective_names]

    solutions: list[ParetoSolution] = []
    for i in range(n):
        raw = tuple(float(col[i]) for col in score_cols)
        scores = tuple(-v for v in raw) if negate_scores else raw
        solutions.append(ParetoSolution(
            scores=scores,
            dimensions=np.asarray(dims[i], dtype=float),
            init_positions=init_positions,
        ))
    return ParetoFront(solutions, tuple(objective_names))


def _run_via_pylinkage(
    walker_factory: Callable[[], Any],
    ref_walker: Any,
    objectives: Sequence[DynamicFitness],
    bounds: tuple[Sequence[float], Sequence[float]],
    objective_names: Sequence[str],
    init_pos: Sequence[Any],
    cfg: NsgaWalkingConfig,
    world_config: Any | None,
) -> ParetoFront:
    """Sequential NSGA via ``pylinkage.optimization.multi_objective_optimization``."""
    from pylinkage.optimization import multi_objective_optimization

    eval_funcs = [
        as_eval_func(
            obj, config=world_config,
            walker_factory=walker_factory, negate=True,
        )
        for obj in objectives
    ]

    ensemble = multi_objective_optimization(
        objectives=eval_funcs,
        linkage=ref_walker,
        bounds=bounds,
        objective_names=objective_names,
        algorithm=cfg.algorithm,
        n_generations=cfg.n_generations,
        pop_size=cfg.pop_size,
        seed=cfg.seed,
        verbose=cfg.verbose,
    )
    return _ensemble_to_pareto_front(
        ensemble, objective_names, init_pos, negate_scores=True,
    )


def _run_via_parallel_problem(
    walker_factory: Callable[[], Any],
    objectives: Sequence[DynamicFitness],
    bounds: tuple[Sequence[float], Sequence[float]],
    objective_names: Sequence[str],
    init_pos: Sequence[Any],
    cfg: NsgaWalkingConfig,
    world_config: Any | None,
) -> ParetoFront:
    """Parallel NSGA via custom ``WalkingNsgaProblem`` (pymoo has no built-in parallelism)."""
    from pymoo.algorithms.moo.nsga2 import NSGA2
    from pymoo.optimize import minimize as pymoo_minimize

    problem = WalkingNsgaProblem(
        walker_factory=walker_factory,
        objectives=objectives,
        bounds=bounds,
        config=world_config,
        n_workers=cfg.n_workers,
    )

    n_obj = len(objective_names)
    if cfg.algorithm == "nsga3" and n_obj > 2:
        from pymoo.algorithms.moo.nsga3 import NSGA3
        from pymoo.util.ref_dirs import get_reference_directions

        n_partitions = max(4, 12 - n_obj)
        ref_dirs = get_reference_directions("das-dennis", n_obj, n_partitions=n_partitions)
        algo = NSGA3(pop_size=cfg.pop_size, ref_dirs=ref_dirs)
    else:
        algo = NSGA2(pop_size=cfg.pop_size)

    try:
        res = pymoo_minimize(
            problem.problem,
            algo,
            ("n_gen", cfg.n_generations),
            seed=cfg.seed,
            verbose=cfg.verbose,
        )
    finally:
        problem.close()

    if res.F is None or res.X is None:
        return ParetoFront([], tuple(objective_names))

    # pymoo may return 1-D for a single solution (shape (n_obj,)) or a
    # single-objective multi-solution run (shape (n_sol,)). Reshape against
    # the known objective count to disambiguate.
    F = np.asarray(res.F).reshape(-1, n_obj)
    X = np.asarray(res.X).reshape(F.shape[0], -1)

    solutions: list[ParetoSolution] = []
    for i in range(F.shape[0]):
        scores = tuple(-float(v) for v in F[i])
        solutions.append(ParetoSolution(
            scores=scores,
            dimensions=X[i],
            init_positions=init_pos,
        ))
    return ParetoFront(solutions, tuple(objective_names))


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

    cfg = nsga_config or NsgaWalkingConfig()
    n_obj = len(objectives)

    if objective_names is None:
        objective_names = [f"objective_{i}" for i in range(n_obj)]

    ref_walker = walker_factory()
    init_pos = ref_walker.get_coords()

    # Delegate to pylinkage for multi-objective sequential runs. Fall back
    # to our pymoo wrapper for parallel evaluation or single-objective
    # (pylinkage 0.9's multi_objective_optimization assumes 2-D ``res.F``
    # and crashes on pymoo's 1-D single-objective output).
    if cfg.n_workers <= 1 and n_obj > 1:
        pareto = _run_via_pylinkage(
            walker_factory=walker_factory,
            ref_walker=ref_walker,
            objectives=objectives,
            bounds=bounds,
            objective_names=objective_names,
            init_pos=init_pos,
            cfg=cfg,
            world_config=world_config,
        )
    else:
        pareto = _run_via_parallel_problem(
            walker_factory=walker_factory,
            objectives=objectives,
            bounds=bounds,
            objective_names=objective_names,
            init_pos=init_pos,
            cfg=cfg,
            world_config=world_config,
        )

    solutions = pareto.solutions

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
