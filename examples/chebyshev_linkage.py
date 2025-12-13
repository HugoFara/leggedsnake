#!/usr/bin/env python3
"""
Chebyshev Lambda Linkage walking mechanism.

The Chebyshev Lambda Linkage is a 4-bar mechanism invented by Russian
mathematician Pafnuty Chebyshev in 1878. It converts rotational motion
to approximate straight-line motion, making it ideal for walking.

Topology (Lambda shape):
- Frame with 2 fixed pivots: O1 (crank center), O2 (rocker center)
- Crank: O1 -> A
- Coupler: A -> B
- Rocker: O2 -> B
- Foot point P: on the rigid coupler link, extended below

The mechanism looks like the Greek letter λ (lambda).

References:
    - https://en.wikipedia.org/wiki/Chebyshev_lambda_linkage
    - https://en.etudes.ru/etudes/tchebyshev-plantigrade-machine/
    - Exposition Universelle (1878) "The Plantigrade Machine"

Run with: uv run python examples/chebyshev_linkage.py
"""
import numpy as np
from pylinkage import Static, Crank, Fixed, Revolute
import leggedsnake as ls

# Simulation parameters
LAP_POINTS = 48  # Steps per revolution

# Chebyshev Lambda linkage dimensions
# Using the standard proportions where coupler = rocker = 1.0
# Ground link: 100*(5-sqrt(7))/3 / 100 ≈ 0.785
# Crank: 100*(3-sqrt(7)) / 100 ≈ 0.354
CHEBYSHEV_DIMENSIONS = {
    # Frame positions (O1 at origin)
    'O2_x': 0.785,      # Ground link length (distance between pivots)
    'O2_y': 0.0,        # O2 on same horizontal level
    # Link lengths
    'crank': 0.354,     # O1 to A
    'coupler': 1.0,     # A to B
    'rocker': 1.0,      # O2 to B
    # Foot point (on rigid coupler, extended below)
    'foot_dist': 1.5,   # Distance from A along coupler direction + extension
    'foot_angle': -np.pi/3,  # Angle below coupler line (toward foot)
}

# Scale factor for simulation
SCALE = 1.5


def get_scaled_dimensions():
    """Get dimensions scaled for simulation."""
    scaled = {}
    for k, v in CHEBYSHEV_DIMENSIONS.items():
        if k != 'foot_angle':
            scaled[k] = v * SCALE
        else:
            scaled[k] = v  # Angle doesn't scale
    return scaled


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


def compute_initial_coords(d, angle=0):
    """
    Compute initial joint coordinates for a given crank angle.
    """
    # Fixed frame pivots
    O1 = (0, 0)
    O2 = (d['O2_x'], d['O2_y'])

    # Crank point A
    A = (d['crank'] * np.cos(angle), d['crank'] * np.sin(angle))

    # B: intersection of circles from A (coupler) and O2 (rocker)
    B1, B2 = _solve_intersection(A, d['coupler'], O2, d['rocker'])
    # Choose B with lower Y for Lambda shape (rocker swings below)
    if B1 is None:
        # Fallback if no intersection
        B = (O2[0] - d['rocker'], O2[1])
    else:
        B = B1 if B1[1] < B2[1] else B2

    # Foot point P: on rigid coupler, below the A-B line
    # Calculate direction from A to B
    AB_dx = B[0] - A[0]
    AB_dy = B[1] - A[1]
    AB_len = np.sqrt(AB_dx**2 + AB_dy**2)

    if AB_len > 0:
        # Unit vector along coupler
        ux = AB_dx / AB_len
        uy = AB_dy / AB_len
        # Rotate by foot_angle to get foot direction
        cos_a = np.cos(d['foot_angle'])
        sin_a = np.sin(d['foot_angle'])
        foot_dx = ux * cos_a - uy * sin_a
        foot_dy = ux * sin_a + uy * cos_a
        # Foot position
        P = (A[0] + d['foot_dist'] * foot_dx, A[1] + d['foot_dist'] * foot_dy)
    else:
        P = (A[0], A[1] - d['foot_dist'])

    return {'O1': O1, 'O2': O2, 'A': A, 'B': B, 'P': P}


def create_chebyshev_linkage():
    """
    Create a single Chebyshev Lambda leg unit.

    The Chebyshev Lambda is a 4-bar mechanism with a coupler point
    that traces an approximate straight line for walking.

    Returns
    -------
    Walker
        The Chebyshev Lambda walking linkage with one foot.
    """
    d = get_scaled_dimensions()
    coords = compute_initial_coords(d, angle=0)

    # Create frame joints (Static)
    O1 = Static(x=coords['O1'][0], y=coords['O1'][1], name="O1 (crank center)")
    O2 = Static(x=coords['O2'][0], y=coords['O2'][1], name="O2 (rocker center)")
    O2.joint0 = O1

    # Crank joint
    A = Crank(
        x=coords['A'][0], y=coords['A'][1],
        joint0=O1,
        distance=d['crank'],
        angle=-2 * np.pi / LAP_POINTS,
        name="A (crank)"
    )

    # B: coupler meets rocker
    B = Revolute(
        x=coords['B'][0], y=coords['B'][1],
        joint0=A, joint1=O2,
        distance0=d['coupler'], distance1=d['rocker'],
        name="B (coupler-rocker)"
    )

    # P (foot): fixed point on the rigid coupler link
    # The foot is attached to A and oriented relative to B
    P = Fixed(
        x=coords['P'][0], y=coords['P'][1],
        joint0=A, joint1=B,
        distance=d['foot_dist'],
        angle=d['foot_angle'],
        name="P (foot)"
    )

    joints = [O1, O2, A, B, P]
    walker = ls.Walker(
        joints=joints,
        order=joints,
        name="Chebyshev"
    )

    return walker


def create_chebyshev_walker(n_legs: int = 3, mirror: bool = False):
    """
    Create a multi-legged Chebyshev walker.

    Like the Jansen and Klann mechanisms, the Chebyshev Lambda linkage
    is asymmetric. Multiple legs are created with phase offsets.

    Parameters
    ----------
    n_legs : int, optional
        Number of additional legs per side. Default is 3 (4 legs total without
        mirroring, or 8 legs total with mirroring).
    mirror : bool, optional
        If True, create a mirrored copy of the leg on the opposite side.
        This creates symmetric left/right leg pairs. Default is True.

    Returns
    -------
    Walker
        Multi-legged Chebyshev walking mechanism.
    """
    walker = create_chebyshev_linkage()
    if mirror:
        walker.mirror_leg()
    if n_legs > 0:
        walker.add_legs(n_legs)
    return walker


def main():
    print("Chebyshev Lambda Linkage Walking Mechanism")
    print("=" * 50)
    print("4-bar mechanism from 1878 'Plantigrade Machine'")
    print()

    walker = create_chebyshev_walker(n_legs=3)

    # Run the visualization
    ls.video(walker, duration=15, dynamic_camera=True)


if __name__ == "__main__":
    main()
