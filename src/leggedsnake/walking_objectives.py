"""
Pre-built objective functions for multi-objective walking optimization.

These objectives wrap the standard pylinkage eval_func contract::

    objective(linkage, dimensions, init_positions) -> float

They can be passed directly to ``multi_objective_optimization`` or composed
with ``chain_optimizers``.

Example::

    from leggedsnake import (
        multi_objective_optimization,
        stride_length_objective,
        energy_efficiency_objective,
    )

    front = multi_objective_optimization(
        objectives=[
            stride_length_objective(duration=40, n_legs=4),
            energy_efficiency_objective(duration=40, n_legs=4),
        ],
        linkage=my_walker,
        bounds=my_bounds,
        objective_names=["stride length (maximize)", "energy per meter"],
    )
"""
from __future__ import annotations

from collections.abc import Sequence
from typing import Any, Callable

from pylinkage.optimization.collections import ParetoFront

from .physicsengine import World, WorldConfig
from .utility import step as step_check, stride
from .walker import Walker


def _run_physics(
    linkage: Walker,
    duration: float,
    config: WorldConfig | None = None,
) -> tuple[float, float, float]:
    """Simulate a walker and return (distance, total_efficiency, total_energy).

    Parameters
    ----------
    linkage : Walker
        The mechanism to simulate.
    duration : float
        Simulation duration in seconds.
    config : WorldConfig | None
        Simulation parameters. Uses ``DEFAULT_CONFIG`` when *None*.

    Returns (0, 0, 0) if the mechanism is unbuildable.
    """
    from . import UnbuildableError

    try:
        tuple(linkage.step())
    except UnbuildableError:
        return 0.0, 0.0, 0.0

    world = World(config=config)
    world.add_linkage(linkage)

    dt = world.config.physics_period
    steps = int(duration / dt)
    total_efficiency = 0.0
    total_energy = 0.0
    for _ in range(steps):
        result = world.update()
        if result is not None:
            total_efficiency += result[0]
            total_energy += result[1]

    distance = float(world.linkages[0].body.position.x)
    return distance, total_efficiency, total_energy


def _prepare_walker(
    linkage: Any,
    dimensions: Sequence[float],
    init_positions: Sequence[Any],
    n_legs: int,
    param_expander: Callable[..., Any] | None,
) -> Walker:
    """Configure a walker with dimensions and add legs."""
    if param_expander is not None:
        linkage.set_num_constraints(param_expander(dimensions), flat=False)
    else:
        linkage.set_num_constraints(dimensions)
    linkage.set_coords(init_positions)

    # Add legs if the walker doesn't have them yet
    if hasattr(linkage, 'get_feet'):
        current_legs = len(linkage.get_feet())
    elif hasattr(linkage, 'get_foots'):
        current_legs = len(linkage.get_foots())
    else:
        current_legs = 1
    if n_legs > 1 and current_legs <= 2:
        linkage.add_legs(n_legs - 1)

    return linkage


def stride_length_objective(
    *,
    lap_points: int = 12,
    step_height: float = 0.5,
    step_width: float = 0.2,
    stride_height: float = 0.2,
    foot_index: int = -2,
    param_expander: Callable[..., Any] | None = None,
) -> Callable[..., float]:
    """Create a kinematic stride-length objective (to maximize).

    This is a fast, physics-free evaluation that measures horizontal
    travel of the foot locus. Suitable for initial exploration stages.

    Parameters
    ----------
    lap_points : int
        Points per crank revolution for simulation.
    step_height, step_width : float
        Obstacle clearance requirements.
    stride_height : float
        Height threshold for stride extraction.
    foot_index : int
        Index of the foot joint in the locus output.
    param_expander : callable, optional
        Function to expand compact parameters to full dimensions
        (e.g., ``param2dimensions`` for symmetric linkages).

    Returns
    -------
    callable
        ``objective(linkage, dims, pos) -> float``
    """
    def _objective(
        linkage: Any,
        dims: Sequence[float],
        pos: Sequence[Any],
    ) -> float:
        if param_expander is not None:
            linkage.set_num_constraints(param_expander(dims), flat=False)
        else:
            linkage.set_num_constraints(dims)
        linkage.set_coords(pos)
        try:
            loci = tuple(
                tuple(i) for i in linkage.step(
                    iterations=lap_points,
                    dt=lap_points / lap_points,
                    skip_unbuildable=True,
                )
            )
        except Exception:
            return 0.0

        from pylinkage import extract_trajectory

        xs, ys = extract_trajectory(loci, foot_index)
        if xs.size == 0:
            return 0.0
        foot_locus = list(zip(xs.tolist(), ys.tolist()))
        if not step_check(foot_locus, step_height, step_width):
            return 0.0

        locus = stride(foot_locus, stride_height)
        return max(k[0] for k in locus) - min(k[0] for k in locus)

    _objective.__name__ = "stride_length"
    return _objective


def energy_efficiency_objective(
    *,
    duration: float = 40.0,
    n_legs: int = 4,
    param_expander: Callable[..., Any] | None = None,
    min_distance: float = 5.0,
    config: WorldConfig | None = None,
) -> Callable[..., float]:
    """Create a dynamic energy-efficiency objective (to maximize).

    Returns ``total_efficiency / total_energy`` from physics simulation.
    Returns 0 if the walker doesn't travel at least ``min_distance``.

    Parameters
    ----------
    duration : float
        Simulation duration in seconds.
    n_legs : int
        Number of leg pairs for the walker.
    param_expander : callable, optional
        Function to expand compact parameters to full dimensions.
    min_distance : float
        Minimum distance the walker must cover to get a non-zero score.
    config : WorldConfig, optional
        Simulation parameters (gravity, terrain, etc.).

    Returns
    -------
    callable
        ``objective(linkage, dims, pos) -> float``
    """

    def _objective(
        linkage: Any,
        dims: Sequence[float],
        pos: Sequence[Any],
    ) -> float:
        walker = _prepare_walker(linkage, dims, pos, n_legs, param_expander)
        distance, total_eff, total_energy = _run_physics(walker, duration, config)
        if distance < min_distance or total_energy == 0:
            return 0.0
        return total_eff / total_energy

    _objective.__name__ = "energy_efficiency"
    return _objective


def total_distance_objective(
    *,
    duration: float = 40.0,
    n_legs: int = 4,
    param_expander: Callable[..., Any] | None = None,
    config: WorldConfig | None = None,
) -> Callable[..., float]:
    """Create a dynamic total-distance objective (to maximize).

    Returns the horizontal position of the walker body after simulation.

    Parameters
    ----------
    duration : float
        Simulation duration in seconds.
    n_legs : int
        Number of leg pairs for the walker.
    param_expander : callable, optional
        Function to expand compact parameters to full dimensions.
    config : WorldConfig, optional
        Simulation parameters (gravity, terrain, etc.).

    Returns
    -------
    callable
        ``objective(linkage, dims, pos) -> float``
    """

    def _objective(
        linkage: Any,
        dims: Sequence[float],
        pos: Sequence[Any],
    ) -> float:
        walker = _prepare_walker(linkage, dims, pos, n_legs, param_expander)
        distance, _, _ = _run_physics(walker, duration, config)
        return distance

    _objective.__name__ = "total_distance"
    return _objective


def multi_objective_walking_optimization(
    linkage: Any,
    objectives: Sequence[Callable[..., float]],
    bounds: tuple[Sequence[float], Sequence[float]],
    objective_names: Sequence[str] | None = None,
    algorithm: str = "nsga2",
    n_generations: int = 100,
    pop_size: int = 100,
    seed: int | None = None,
    verbose: bool = True,
    **kwargs: Any,
) -> ParetoFront:
    """Multi-objective optimization for walking linkages.

    Convenience wrapper around ``pylinkage.optimization.multi_objective_optimization``
    that accepts walking-specific objective factories.

    Parameters
    ----------
    linkage : Walker or Linkage
        The linkage to optimize.
    objectives : sequence of callables
        Objective functions, each with signature
        ``(linkage, dims, pos) -> float``. Use the factory functions
        in this module (``stride_length_objective``, etc.) to create them.
    bounds : tuple of (lower, upper)
        Parameter bounds.
    objective_names : sequence of str, optional
        Names for plotting and identification.
    algorithm : {"nsga2", "nsga3"}
        Multi-objective algorithm. Default "nsga2".
    n_generations : int
        Number of generations. Default 100.
    pop_size : int
        Population size. Default 100.
    seed : int, optional
        Random seed for reproducibility.
    verbose : bool
        Show progress. Default True.

    Returns
    -------
    ParetoFront
        Collection of non-dominated solutions with ``.best_compromise()``,
        ``.plot()``, and ``.filter()`` methods.
    """
    from pylinkage.optimization import multi_objective_optimization

    return multi_objective_optimization(
        objectives=objectives,
        linkage=linkage,
        bounds=bounds,
        objective_names=objective_names,
        algorithm=algorithm,
        n_generations=n_generations,
        pop_size=pop_size,
        seed=seed,
        verbose=verbose,
        **kwargs,
    )
