#!/usr/bin/env python3
"""
Klann Linkage walking mechanism.

The Klann linkage is a 6-bar planar mechanism (Stephenson-III topology)
designed by Joe Klann in 1994 and patented in 2001 (US Patent 6,260,862).

This example uses :meth:`leggedsnake.Walker.from_klann`, which builds the
canonical geometry directly from the patent's dimensions (two rigid
triangles: the ternary coupler attached to the crank and the ternary leg
carrying the foot).

References:
    - https://en.wikipedia.org/wiki/Klann_linkage
    - US Patent 6,260,862
    - https://www.diywalkers.com/klann-linkage-optimizer.html

Run with: uv run python examples/klann_linkage.py
"""
import leggedsnake as ls
from leggedsnake import Walker

# Scale factor for better visualization.
SCALE = 3.0


def create_klann_linkage() -> Walker:
    """Create a single Klann leg unit using the canonical patent dimensions."""
    return Walker.from_klann(scale=SCALE)


def create_klann_walker(n_legs: int = 3, opposite: bool = True) -> Walker:
    """
    Create a multi-legged Klann walker.

    Like the Jansen mechanism, the Klann linkage is asymmetric. Multiple
    legs are created with phase offsets using add_legs().

    Parameters
    ----------
    n_legs : int, optional
        Number of additional legs per side. Default is 3 (4 legs total without
        opposite leg, or 8 legs total with opposite leg).
    opposite : bool, optional
        If True, create an antisymmetric copy of the leg on the opposite side.
        This creates left/right leg pairs. Default is True.
    """
    walker = create_klann_linkage()
    if opposite:
        walker.add_opposite_leg()
    if n_legs > 0:
        walker.add_legs(n_legs)
    return walker


def main():
    print("Klann Linkage Walking Mechanism")
    print("=" * 50)
    print("6-bar Stephenson III mechanism (US Patent 6,260,862)")
    print()

    walker = create_klann_walker(n_legs=2)
    walker.motor_rates = 4.0
    ls.video(walker, duration=15, dynamic_camera=True)


if __name__ == "__main__":
    main()
