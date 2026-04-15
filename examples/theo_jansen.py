#!/usr/bin/env python3
"""
Theo Jansen Linkage (Strandbeest) walking mechanism.

The Theo Jansen linkage is an 8-bar planar mechanism designed by Dutch kinetic
sculptor Theo Jansen in the 1990s for his "Strandbeest" sculptures.

This example uses :meth:`leggedsnake.Walker.from_jansen`, which builds the
canonical geometry from Jansen's famous "Holy Numbers" (the link dimensions
he discovered through decades of genetic-algorithm optimization).

Note: the Jansen mechanism is inherently asymmetric — in the real Strandbeest,
multiple identical legs are arranged in a row with phase offsets, not mirrored.

References:
    - https://en.wikipedia.org/wiki/Jansen%27s_linkage
    - https://observablehq.com/@sw1227/theo-jansens-linkage
    - https://javalab.org/en/theo_jansen_en/

Run with: uv run python examples/theo_jansen.py
"""
import leggedsnake as ls
from leggedsnake import Walker

# Scale factor — smaller values give more physics-friendly dimensions.
SCALE = 1 / 25.0


def create_theo_jansen_linkage() -> Walker:
    """Create a single Theo Jansen leg unit using the canonical Holy Numbers."""
    return Walker.from_jansen(scale=SCALE)


def create_theo_jansen_walker(n_legs: int = 3, opposite: bool = True) -> Walker:
    """
    Create a multi-legged Theo Jansen walker.

    In the real Strandbeest, legs are arranged in a row with phase offsets
    (not mirrored). This function uses add_legs to create phase-shifted copies.

    Parameters
    ----------
    n_legs : int, optional
        Number of additional legs per side. Default is 3 (4 legs total without
        opposite leg, or 8 legs total with opposite leg).
    opposite : bool, optional
        If True, create an antisymmetric copy of the leg on the opposite side.
        This creates left/right leg pairs. Default is True.
    """
    walker = create_theo_jansen_linkage()
    if opposite:
        walker.add_opposite_leg()
    if n_legs > 0:
        walker.add_legs(n_legs)
    return walker


def main():
    print("Theo Jansen Linkage (Strandbeest) Walking Mechanism")
    print("=" * 55)
    print("Using the authentic 'Holy Numbers' discovered by Jansen")
    print()

    walker = create_theo_jansen_walker(n_legs=3)
    walker.motor_rates = -4.0
    ls.video(walker, duration=15, dynamic_camera=True)


if __name__ == "__main__":
    main()
