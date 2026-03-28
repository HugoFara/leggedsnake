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

# Chebyshev Lambda linkage dimensions for walking
# This is a crank-rocker mechanism that allows full 360° rotation
# Based on ground=1, crank short, coupler=rocker (creates the λ shape)
# The coupler point traces an approximate straight line during ground contact
CHEBYSHEV_DIMENSIONS = {
    # Frame positions (O1 at origin)
    'O2_x': 2.0,        # Ground link
    'O2_y': 0.0,        # O2 on same horizontal level
    # Link lengths for Lambda configuration (crank is the shortest link)
    'crank': 0.5,       # O1 to A (short crank for full rotation)
    'coupler': 2.5,     # A to B
    'rocker': 2.5,      # O2 to B
    # Foot point: extended along coupler for walking
    'foot_dist': 1.0,   # Ratio along coupler (1.0 = at point B)
    'foot_angle': 0,    # Along the coupler direction (A→B)
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
    # Choose B with LOWER Y - this puts the foot below the frame for walking
    if B1 is None:
        # Fallback if no intersection
        B = (O2[0] - d['rocker'], O2[1])
    else:
        B = B1 if B1[1] < B2[1] else B2

    # Foot point P: on the coupler line A-B
    # foot_dist is a ratio (0.5 = midpoint, which traces the straight line)
    ratio = d['foot_dist']
    P = (A[0] + ratio * (B[0] - A[0]), A[1] + ratio * (B[1] - A[1]))

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
    # foot_dist is a ratio (0.5 = midpoint), convert to absolute distance
    foot_distance = d['foot_dist'] * d['coupler']
    P = Fixed(
        x=coords['P'][0], y=coords['P'][1],
        joint0=A, joint1=B,
        distance=foot_distance,
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


def create_chebyshev_walker(n_legs: int = 3, opposite: bool = True):
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

    Returns
    -------
    Walker
        Multi-legged Chebyshev walking mechanism.
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
    # Motor rate: negative for clockwise rotation (walking forward)
    walker.motor_rate = -4.0

    # Run the visualization
    ls.video(walker, duration=15, dynamic_camera=True)


if __name__ == "__main__":
    main()
