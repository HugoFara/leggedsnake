#!/usr/bin/env python3
"""
Klann Linkage walking mechanism.

The Klann linkage is a 6-bar planar mechanism (Stephenson III topology)
designed by Joe Klann in 1994 and patented in 2001 (US Patent 6,260,862).

Topology:
- Frame with 3 grounded pivots: O1 (crank), O2 (upper rocker), O3 (lower rocker)
- Crank: O1 -> A
- Upper rocker: O2 -> B
- Coupler: A -> B
- Lower rocker: O3 -> C
- Leg (ternary link): B -> C -> F (foot)

References:
    - https://en.wikipedia.org/wiki/Klann_linkage
    - US Patent 6,260,862
    - https://www.diywalkers.com/klann-linkage-optimizer.html

Run with: uv run python examples/klann_linkage.py
"""
import numpy as np
from pylinkage import Static, Crank, Revolute
import leggedsnake as ls

# Simulation parameters
LAP_POINTS = 48  # More steps for numerical stability

# Klann linkage dimensions derived from patent US 6,260,862
# Scaled so crank radius = 1.0
KLANN_DIMENSIONS = {
    # Frame pivot positions (relative to O1 at origin)
    'O2_x': -0.87,      # Upper rocker pivot X
    'O2_y': 2.30,       # Upper rocker pivot Y
    'O3_x': -2.20,      # Lower rocker pivot X
    'O3_y': -0.66,      # Lower rocker pivot Y
    # Link lengths
    'crank': 1.0,       # O1 to A (crank radius)
    'upper_rocker': 1.93,  # O2 to B
    'coupler': 3.29,    # A to B
    'lower_rocker': 3.10,  # O3 to C
    'leg_BC': 3.35,     # B to C (leg segment)
    'leg_CF': 3.35,     # C to F (leg to foot)
}

# Scale factor for simulation
SCALE = 0.5


def get_scaled_dimensions():
    """Get dimensions scaled for simulation."""
    return {k: v * SCALE for k, v in KLANN_DIMENSIONS.items()}


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

    Uses branch selections determined empirically for full-revolution stability.
    """
    # Fixed frame pivots
    O1 = (0, 0)
    O2 = (d['O2_x'], d['O2_y'])
    O3 = (d['O3_x'], d['O3_y'])

    # Crank point A
    A = (d['crank'] * np.cos(angle), d['crank'] * np.sin(angle))

    # B: intersection of circles from A (coupler) and O2 (upper rocker)
    B1, B2 = _solve_intersection(A, d['coupler'], O2, d['upper_rocker'])
    # Choose B with lower Y (hip should be below upper rocker pivot in most configs)
    B = B1 if B1[1] < B2[1] else B2

    # C: intersection of circles from B (leg) and O3 (lower rocker)
    C1, C2 = _solve_intersection(B, d['leg_BC'], O3, d['lower_rocker'])
    # Choose C with lower Y (knee should be below hip)
    C = C1 if C1[1] < C2[1] else C2

    # F (foot): intersection of circles from B (leg_BF) and C (leg_CF)
    # For ternary link BCF, we need the distance B-F
    # In a rigid triangle, B-F is fixed. Calculate from geometry.
    # Using law of cosines: BF^2 = BC^2 + CF^2 - 2*BC*CF*cos(angle_BCF)
    # For Klann, the foot extends below the knee, so we use leg_CF from C
    F1, F2 = _solve_intersection(C, d['leg_CF'], B, d['leg_BC'] + d['leg_CF'] * 0.5)
    # Choose F with lower Y (foot should be lowest point)
    if F1 is None:
        # If intersection fails, use alternative approach
        # F is directly below C at distance leg_CF
        F = (C[0], C[1] - d['leg_CF'])
    else:
        F = F1 if F1[1] < F2[1] else F2

    return {'O1': O1, 'O2': O2, 'O3': O3, 'A': A, 'B': B, 'C': C, 'F': F}


def create_klann_linkage():
    """
    Create a single Klann leg unit.

    The Klann linkage is a 6-bar Stephenson III mechanism.

    Returns
    -------
    Walker
        The Klann walking linkage with one foot.
    """
    d = get_scaled_dimensions()
    coords = compute_initial_coords(d, angle=0)

    # Create frame joints (Static)
    O1 = Static(x=coords['O1'][0], y=coords['O1'][1], name="O1 (crank center)")
    O2 = Static(x=coords['O2'][0], y=coords['O2'][1], name="O2 (upper rocker)")
    O3 = Static(x=coords['O3'][0], y=coords['O3'][1], name="O3 (lower rocker)")
    O2.joint0 = O1
    O3.joint0 = O1

    # Crank joint
    A = Crank(
        x=coords['A'][0], y=coords['A'][1],
        joint0=O1,
        distance=d['crank'],
        angle=-2 * np.pi / LAP_POINTS,
        name="A (crank)"
    )

    # B (hip): on coupler from A, on upper rocker from O2
    B = Revolute(
        x=coords['B'][0], y=coords['B'][1],
        joint0=A, joint1=O2,
        distance0=d['coupler'], distance1=d['upper_rocker'],
        name="B (hip)"
    )

    # C (knee): on leg from B, on lower rocker from O3
    C = Revolute(
        x=coords['C'][0], y=coords['C'][1],
        joint0=B, joint1=O3,
        distance0=d['leg_BC'], distance1=d['lower_rocker'],
        name="C (knee)"
    )

    # F (foot): on ternary leg link, distance from both B and C
    # For a ternary link BCF where F extends from C
    F = Revolute(
        x=coords['F'][0], y=coords['F'][1],
        joint0=C, joint1=B,
        distance0=d['leg_CF'], distance1=d['leg_BC'] + d['leg_CF'] * 0.5,
        name="F (foot)"
    )

    joints = [O1, O2, O3, A, B, C, F]
    walker = ls.Walker(
        joints=joints,
        order=joints,
        name="Klann"
    )

    return walker


def create_klann_walker(n_legs: int = 3, opposite: bool = True):
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

    Returns
    -------
    Walker
        Multi-legged Klann walking mechanism.
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

    walker = create_klann_walker(n_legs=3)

    # Run the visualization
    ls.video(walker, duration=15, dynamic_camera=True)


if __name__ == "__main__":
    main()
