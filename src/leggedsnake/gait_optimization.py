"""Phase-offset optimization for multi-leg walking mechanisms.

Evolves the per-leg phase offsets of an N-leg walker — the discrete
parameters that distinguish a rotating-stack gait (evenly spaced) from
trot (opposite pairs in phase), pace, canter, bound, and other
asymmetric gaits that emerge in nature but that the classical
``Walker.add_legs(n)`` constructor cannot express.

The optimizer wraps ``scipy.optimize.differential_evolution`` on an
``(n_legs - 1)``-dimensional continuous search space (one offset per
added leg, in radians). Each candidate rebuilds the walker via
``Walker.add_legs(offsets=...)`` and evaluates a user-supplied
``DynamicFitness``.

Example
-------
::

    from leggedsnake import DistanceFitness, optimize_gait, GaitOptimizationConfig

    def template_factory():
        return Walker.from_jansen()  # single-leg template

    config = GaitOptimizationConfig(
        walker_factory=template_factory,
        n_legs=4,
        fitness=DistanceFitness(duration=20.0),
        popsize=15,
        maxiter=30,
        seed=42,
    )
    result = optimize_gait(config)
    print("best offsets:", result.best_offsets)
    print("best score:", result.best_score)
"""
from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field
from math import tau
from typing import Callable

import numpy as np
from scipy.optimize import differential_evolution  # type: ignore[import-untyped]

from .fitness import DynamicFitness
from .physics_engine import WorldConfig
from .walker import Walker


@dataclass
class GaitOptimizationConfig:
    """Configuration for :func:`optimize_gait`.

    Attributes
    ----------
    walker_factory : callable
        Zero-argument callable returning a fresh single-leg Walker
        template. Called once per candidate evaluation so no state
        leaks across generations.
    n_legs : int
        Total number of legs in the evaluated walker (template + added).
        The optimizer searches ``n_legs - 1`` phase offsets.
    fitness : DynamicFitness
        The fitness evaluator applied to each candidate walker after
        phase-offset assembly.
    world_config : WorldConfig | None
        Simulation environment passed through to ``fitness``.
    popsize : int
        Differential-evolution population size multiplier (scipy's
        ``popsize`` = population / dimension).
    maxiter : int
        Maximum DE generations.
    seed : int | None
        RNG seed for reproducibility.
    tol : float
        DE convergence tolerance.
    workers : int
        Parallel workers (1 = serial; -1 = use all cores).
    initial_offsets : list[float] | None
        Optional warm-start — a seed member inserted into the initial
        population. Length must equal ``n_legs - 1``.
    """

    walker_factory: Callable[[], Walker]
    n_legs: int
    fitness: DynamicFitness
    world_config: WorldConfig | None = None
    popsize: int = 15
    maxiter: int = 30
    seed: int | None = None
    tol: float = 1e-3
    workers: int = 1
    initial_offsets: list[float] | None = None

    def __post_init__(self) -> None:
        if self.n_legs < 2:
            raise ValueError(
                f"n_legs must be >= 2 to have phase offsets to optimize "
                f"(got {self.n_legs})"
            )
        if self.initial_offsets is not None and len(self.initial_offsets) != self.n_legs - 1:
            raise ValueError(
                f"initial_offsets length {len(self.initial_offsets)} does "
                f"not match n_legs - 1 = {self.n_legs - 1}"
            )


@dataclass
class GaitOptimizationResult:
    """Outcome of :func:`optimize_gait`.

    Attributes
    ----------
    best_offsets : list[float]
        Phase offsets (radians, in ``[0, tau)``) for the highest-scoring
        candidate.
    best_score : float
        Fitness value of the best candidate.
    n_evaluations : int
        Total number of fitness evaluations the optimizer performed.
    converged : bool
        Whether scipy reported convergence (``OptimizeResult.success``).
    message : str
        Optimizer status message.
    history : list[float]
        Best score observed after each generation (populated only when
        the caller supplied a generation callback).
    """

    best_offsets: list[float]
    best_score: float
    n_evaluations: int
    converged: bool
    message: str
    history: list[float] = field(default_factory=list)


def optimize_gait(config: GaitOptimizationConfig) -> GaitOptimizationResult:
    """Evolve the phase-offset sequence of a multi-leg walker.

    Runs scipy's differential evolution over the ``n_legs - 1`` phase
    offsets (in radians, bounded to ``[0, tau)``). Each candidate rebuilds
    a fresh walker via ``config.walker_factory`` and evaluates
    ``config.fitness``; the score is negated internally because scipy
    minimizes.

    Parameters
    ----------
    config : GaitOptimizationConfig
        Search configuration.

    Returns
    -------
    GaitOptimizationResult
    """
    dim = config.n_legs - 1
    bounds = [(0.0, tau)] * dim

    def _negated_fitness(x: np.ndarray) -> float:
        offsets = [float(v) for v in x]
        walker = config.walker_factory()
        try:
            walker.add_legs(offsets)
        except Exception:
            return float("inf")
        try:
            result = config.fitness(
                deepcopy(walker.topology),
                deepcopy(walker.dimensions),
                config.world_config,
            )
        except Exception:
            return float("inf")
        if not result.valid:
            return float("inf")
        return -float(result.score)

    init: np.ndarray | str = "sobol"
    if config.initial_offsets is not None:
        # Seed the population with the caller's warm-start plus random
        # draws for the rest (DE needs at least 5 * dim members).
        total = max(5 * dim, config.popsize * dim)
        rng = np.random.default_rng(config.seed)
        pop = rng.uniform(0.0, tau, size=(total, dim))
        pop[0] = np.asarray(config.initial_offsets, dtype=float) % tau
        init = pop

    opt = differential_evolution(
        _negated_fitness,
        bounds=bounds,
        popsize=config.popsize,
        maxiter=config.maxiter,
        seed=config.seed,
        tol=config.tol,
        workers=config.workers,
        init=init,
        polish=False,  # phase offsets are noisy; local polishing is unhelpful
    )

    best_offsets = [float(v) % tau for v in opt.x]
    best_score = -float(opt.fun) if np.isfinite(opt.fun) else 0.0

    return GaitOptimizationResult(
        best_offsets=best_offsets,
        best_score=best_score,
        n_evaluations=int(opt.nfev),
        converged=bool(opt.success),
        message=str(opt.message),
    )
