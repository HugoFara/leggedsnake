#!/usr/bin/env python3
"""
Theo Jansen Linkage (Strandbeest) walking mechanism.

The Theo Jansen linkage is an 8-bar planar mechanism designed by Dutch kinetic
sculptor Theo Jansen in the 1990s for his "Strandbeest" sculptures.

This implementation uses the actual Jansen linkage topology with his famous
"Holy Numbers" - the link dimensions he discovered through genetic algorithm
optimization over many years.

Note: The Jansen mechanism is inherently asymmetric - in the real Strandbeest,
multiple identical legs are arranged in a row with phase offsets, not mirrored.

References:
    - https://en.wikipedia.org/wiki/Jansen%27s_linkage
    - https://observablehq.com/@sw1227/theo-jansens-linkage
    - https://javalab.org/en/theo_jansen_en/

Run with: uv run python examples/theo_jansen.py
"""
import numpy as np
from pylinkage import Static, Crank, Revolute
import leggedsnake as ls

# Simulation parameters
LAP_POINTS = 48  # More steps for numerical stability

# Theo Jansen's "Holy Numbers" - the sacred dimensions
HOLY_NUMBERS = {
    'a': 38.0,    # X-offset from O to B (frame)
    'b': 41.5,    # B to C
    'c': 39.3,    # B to D
    'd': 40.1,    # B to E
    'e': 55.8,    # C to E
    'f': 39.4,    # E to F
    'g': 36.7,    # D to F
    'h': 65.7,    # F to G (foot)
    'i': 49.0,    # D to G (foot)
    'j': 50.0,    # A to C
    'k': 61.9,    # A to D
    'l': 7.8,     # Y-offset from O to B (frame)
    'm': 15.0,    # Crank radius (O to A)
}

# Scale factor - smaller scale for better physics stability
SCALE = 1 / 25.0


def get_scaled_dimensions():
    """Get holy numbers scaled for simulation."""
    return {k: v * SCALE for k, v in HOLY_NUMBERS.items()}


def _solve_intersection(p1, r1, p2, r2):
    """Solve circle-circle intersection, returns (sol1, sol2) or (None, None)."""
    dx = p2[0] - p1[0]
    dy = p2[1] - p1[1]
    d = np.sqrt(dx**2 + dy**2)

    if d > r1 + r2 or d < abs(r1 - r2) or d == 0:
        return None, None

    a_val = (r1**2 - r2**2 + d**2) / (2 * d)
    h_val = np.sqrt(max(0, r1**2 - a_val**2))

    mx = p1[0] + a_val * dx / d
    my = p1[1] + a_val * dy / d

    sol1 = (mx + h_val * (-dy) / d, my + h_val * dx / d)
    sol2 = (mx - h_val * (-dy) / d, my - h_val * dx / d)

    return sol1, sol2


def compute_initial_coords(h, angle=0):
    """
    Compute initial joint coordinates for a given crank angle.

    Uses branch selections determined empirically for full-revolution stability.
    """
    # Fixed points
    O = (0, 0)
    B = (-h['a'], -h['l'])

    # Crank point
    A = (h['m'] * np.cos(angle), h['m'] * np.sin(angle))

    # Branch selections: C2, D1, E1, F1, G1
    C1, C2 = _solve_intersection(A, h['j'], B, h['b'])
    C = C2

    D1, D2 = _solve_intersection(A, h['k'], B, h['c'])
    D = D1

    E1, E2 = _solve_intersection(B, h['d'], C, h['e'])
    E = E1

    F1, F2 = _solve_intersection(D, h['g'], E, h['f'])
    F = F1

    G1, G2 = _solve_intersection(D, h['i'], F, h['h'])
    G = G1

    return {'O': O, 'B': B, 'A': A, 'C': C, 'D': D, 'E': E, 'F': F, 'G': G}


def create_theo_jansen_linkage():
    """
    Create a single Theo Jansen leg unit.

    Returns
    -------
    Walker
        The Theo Jansen walking linkage with one foot.
    """
    h = get_scaled_dimensions()
    coords = compute_initial_coords(h, angle=0)

    # Create joints
    O = Static(x=coords['O'][0], y=coords['O'][1], name="O (crank center)")
    B = Static(x=coords['B'][0], y=coords['B'][1], name="B (frame)")
    B.joint0 = O

    A = Crank(
        x=coords['A'][0], y=coords['A'][1],
        joint0=O,
        distance=h['m'],
        angle=-2 * np.pi / LAP_POINTS,
        name="A (crank)"
    )

    C = Revolute(
        x=coords['C'][0], y=coords['C'][1],
        joint0=A, joint1=B,
        distance0=h['j'], distance1=h['b'],
        name="C"
    )

    D = Revolute(
        x=coords['D'][0], y=coords['D'][1],
        joint0=A, joint1=B,
        distance0=h['k'], distance1=h['c'],
        name="D"
    )

    E = Revolute(
        x=coords['E'][0], y=coords['E'][1],
        joint0=B, joint1=C,
        distance0=h['d'], distance1=h['e'],
        name="E"
    )

    F = Revolute(
        x=coords['F'][0], y=coords['F'][1],
        joint0=D, joint1=E,
        distance0=h['g'], distance1=h['f'],
        name="F"
    )

    G = Revolute(
        x=coords['G'][0], y=coords['G'][1],
        joint0=D, joint1=F,
        distance0=h['i'], distance1=h['h'],
        name="G (foot)"
    )

    joints = [O, B, A, C, D, E, F, G]
    walker = ls.Walker(
        joints=joints,
        order=joints,
        name="TheoJansen"
    )

    return walker


def create_theo_jansen_walker(n_legs: int = 3):
    """
    Create a multi-legged Theo Jansen walker.

    In the real Strandbeest, legs are arranged in a row with phase offsets
    (not mirrored). This function uses add_legs to create phase-shifted copies.

    Parameters
    ----------
    n_legs : int, optional
        Number of additional legs. Default is 3 (4 legs total).

    Returns
    -------
    Walker
        Multi-legged Theo Jansen walking mechanism.
    """
    walker = create_theo_jansen_linkage()
    if n_legs > 0:
        walker.add_legs(n_legs)
    return walker


def main():
    print("Theo Jansen Linkage (Strandbeest) Walking Mechanism")
    print("=" * 55)
    print("Using the authentic 'Holy Numbers' discovered by Jansen")
    print()

    walker = create_theo_jansen_walker(n_legs=3)

    # Run the visualization
    ls.video(walker, duration=15, dynamic_camera=True)


if __name__ == "__main__":
    main()
