#!/usr/bin/env python3
"""
Chebyshev Lambda Linkage walking mechanism.

The Chebyshev Lambda Linkage is a 4-bar mechanism invented by Russian
mathematician Pafnuty Chebyshev in 1878. It converts rotational motion
into approximate straight-line motion, making it ideal for walking.

This example uses :meth:`leggedsnake.Walker.from_chebyshev`, which builds
the canonical crank-rocker geometry with a coupler point P that traces
the straight-line walking locus.

References:
    - https://en.wikipedia.org/wiki/Chebyshev_lambda_linkage
    - https://en.etudes.ru/etudes/tchebyshev-plantigrade-machine/
    - Exposition Universelle (1878) "The Plantigrade Machine"

Run with: uv run python examples/chebyshev_linkage.py
"""
import leggedsnake as ls
from leggedsnake import Walker

# Scale factor for simulation.
SCALE = 1.5


def create_chebyshev_linkage() -> Walker:
    """Create a single Chebyshev Lambda leg unit with the classical ratios."""
    return Walker.from_chebyshev(
        crank=0.5 * SCALE,
        coupler=2.5 * SCALE,
        rocker=2.5 * SCALE,
        ground_length=2.0 * SCALE,
        # foot_ratio=1.5 reproduces the canonical walking locus where P
        # extends beyond B by half the coupler length.
        foot_ratio=1.5,
    )


def create_chebyshev_walker(n_legs: int = 3, opposite: bool = True) -> Walker:
    """
    Create a multi-legged Chebyshev walker.

    Like the Jansen and Klann mechanisms, the Chebyshev Lambda linkage
    is asymmetric. Multiple legs are created with phase offsets.

    Parameters
    ----------
    n_legs : int, optional
        Number of additional legs per side. Default is 3 (4 legs total without
        opposite leg, or 8 legs total with opposite leg).
    opposite : bool, optional
        If True, create an antisymmetric copy of the leg on the opposite side.
        This creates left/right leg pairs. Default is True.
    """
    walker = create_chebyshev_linkage()
    if opposite:
        walker.add_opposite_leg()
    if n_legs > 0:
        walker.add_legs(n_legs)
    return walker


def main():
    print("Chebyshev Lambda Linkage Walking Mechanism")
    print("=" * 50)
    print("4-bar mechanism from 1878 'Plantigrade Machine'")
    print()

    walker = create_chebyshev_walker(n_legs=3)
    walker.motor_rates = -4.0
    ls.video(walker, duration=15, dynamic_camera=True)


if __name__ == "__main__":
    main()
