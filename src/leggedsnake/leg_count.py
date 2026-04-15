"""Post-design leg-count sweeps.

Once a walker design is finalised (topology + link dimensions), the
question *"how many legs is best?"* is typically solved as a small
post-hoc sweep rather than as part of the main optimisation loop.
:func:`sweep_leg_counts` evaluates a fixed design across a range of
leg counts and returns the scores, so callers can pick the argmax or
plot the trade-off curve.
"""
from __future__ import annotations

from collections.abc import Iterable
from copy import deepcopy

from .fitness import DynamicFitness, FitnessResult
from .physicsengine import WorldConfig
from .walker import Walker


def sweep_leg_counts(
    walker: Walker,
    objective: DynamicFitness,
    n_legs_range: Iterable[int] = range(2, 9),
    opposite_leg: bool = False,
    world_config: WorldConfig | None = None,
) -> dict[int, FitnessResult]:
    """Evaluate a finished walker design across a range of leg counts.

    For each ``n`` in ``n_legs_range`` the walker is deep-copied,
    :meth:`Walker.add_legs(n - 1)` is called (plus
    :meth:`Walker.add_opposite_leg` when requested), and ``objective``
    is invoked on the resulting topology / dimensions. Results are
    returned as an ordered mapping so callers can pick the best leg
    count or plot the trade-off.

    The objective is expected to be configured with ``n_legs=1`` (its
    default add-legs expansion is skipped). If it exposes an ``n_legs``
    attribute, it is temporarily overridden to 1 for the duration of
    the call to avoid double-adding legs; the original value is
    restored before returning.

    Parameters
    ----------
    walker : Walker
        Finished single-leg design. Its topology and dimensions are
        deep-copied per sweep entry; the input walker is not mutated.
    objective : DynamicFitness
        Fitness evaluator. Built-in objectives like
        :class:`DistanceFitness` or :class:`EfficiencyFitness` all
        conform to the protocol.
    n_legs_range : iterable of int
        Leg counts to evaluate. Defaults to ``range(2, 9)``.
    opposite_leg : bool
        If True, :meth:`Walker.add_opposite_leg` is called before
        :meth:`Walker.add_legs`, creating a mirrored-pair baseline
        (so ``n_legs=2`` means two opposing legs, ``n_legs=3`` means
        two opposing plus a phase-offset copy, etc.).
    world_config : WorldConfig, optional
        Simulation config forwarded to the objective.

    Returns
    -------
    dict[int, FitnessResult]
        Ordered mapping from leg count to fitness result.
    """
    original_n_legs = getattr(objective, "n_legs", None)
    results: dict[int, FitnessResult] = {}
    try:
        if original_n_legs is not None:
            objective.n_legs = 1  # type: ignore[attr-defined]
        for n in n_legs_range:
            candidate = deepcopy(walker)
            if opposite_leg:
                candidate.add_opposite_leg()
            if n > 1:
                candidate.add_legs(n - 1)
            results[int(n)] = objective(
                candidate.topology, candidate.dimensions, world_config,
            )
    finally:
        if original_n_legs is not None:
            objective.n_legs = original_n_legs  # type: ignore[attr-defined]

    return results


__all__ = ["sweep_leg_counts"]
