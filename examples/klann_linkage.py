#!/usr/bin/env python3
"""
Klann Linkage walking mechanism.

The Klann linkage is a 6-bar planar mechanism (Stephenson III topology)
designed by Joe Klann in 1994 and patented in 2001 (US Patent 6,260,862).

Topology (6-bar, 7 joints, DOF=1):
- Frame with 3 grounded pivots: O1 (crank), O2 (upper rocker), O3 (lower rocker)
- Crank: O1 -> A
- Coupler: A -> B (connects crank to hip)
- Upper rocker: O2 -> B (connects frame to hip)
- Lower rocker: O3 -> C (connects frame to knee)
- Ternary leg link: B-C-F (hip-knee-foot triangle)

The mechanism creates an approximately straight-line foot path during the
stance phase, with a curved lift during the swing phase.

References:
    - https://en.wikipedia.org/wiki/Klann_linkage
    - US Patent 6,260,862
    - https://www.diywalkers.com/klann-linkage-optimizer.html

Run with: uv run python examples/klann_linkage.py
"""
from math import pi

import numpy as np

import leggedsnake as ls
from leggedsnake import (
    Dimensions,
    DriverAngle,
    Edge,
    Hyperedge,
    HypergraphLinkage,
    Node,
    NodeRole,
    Walker,
)

# Simulation parameters
LAP_POINTS = 48  # Steps per crank revolution

# Klann linkage dimensions from US Patent 6,260,862
# Extracted from Wikipedia coordinates table
#
# CORRECT TOPOLOGY: The Klann is a 6-bar mechanism with TWO ternary links:
# - Links: Frame, Crank, Ternary-Coupler, Lower-Rocker, Upper-Rocker, Ternary-Leg
# - Ternary-Coupler: A → Elbow → Knee (rigid triangle attached to crank)
# - Ternary-Leg: Hip → Knee → Foot (rigid triangle for the leg)
#
# Connectivity:
# - Crank rotates at O_crank, endpoint A
# - Ternary-Coupler has A, Elbow, Knee as vertices (A fixed to crank)
# - Lower-Rocker: O_lower → Elbow (constrains Elbow position)
# - Upper-Rocker: O_upper → Hip (constrains Hip position)
# - Ternary-Leg connects Hip, Knee, Foot (Knee shared with coupler)
KLANN_PATENT = {
    # Frame pivot positions (relative to O_crank at origin)
    'O_upper_x': -0.233,        # First rocker arm axle (point 9)
    'O_upper_y': 0.616,
    'O_lower_x': -0.590,        # Second rocker arm axle (point 11)
    'O_lower_y': -0.176,
    # Link lengths from patent (verified constant between positions X and Y)
    'crank': 0.268,             # O_crank to A
    # Ternary coupler (A-Elbow-Knee triangle)
    'A_elbow': 0.590,           # A to Elbow
    'A_knee': 1.105,            # A to Knee (KEY: this is rigid!)
    'elbow_knee': 0.522,        # Elbow to Knee
    # Rockers
    'lower_rocker': 0.321,      # O_lower to Elbow
    'upper_rocker': 0.518,      # O_upper to Hip
    # Ternary leg (Hip-Knee-Foot triangle)
    'hip_knee': 0.897,          # Hip to Knee
    'knee_foot': 0.897,         # Knee to Foot
    'hip_foot': 1.732,          # Hip to Foot
}

# Scale factor for simulation
SCALE = 3.0  # Scale up for better visualization


def get_scaled_dimensions():
    """Get dimensions scaled for simulation."""
    return {k: v * SCALE for k, v in KLANN_PATENT.items()}


def _solve_intersection(p1, r1, p2, r2):
    """Solve circle-circle intersection, returns (sol1, sol2) or (None, None)."""
    dx = p2[0] - p1[0]
    dy = p2[1] - p1[1]
    d = np.sqrt(dx**2 + dy**2)

    if d > r1 + r2 or d < abs(r1 - r2) or d == 0:
        return None, None

    a_val = (r1**2 - r2**2 + d**2) / (2 * d)
    h_sq = r1**2 - a_val**2
    if h_sq < 0:
        return None, None
    h_val = np.sqrt(h_sq)

    mx = p1[0] + a_val * dx / d
    my = p1[1] + a_val * dy / d

    sol1 = (mx + h_val * (-dy) / d, my + h_val * dx / d)
    sol2 = (mx - h_val * (-dy) / d, my - h_val * dx / d)

    return sol1, sol2


def compute_frame_positions(d):
    """
    Get frame pivot positions from dimension dictionary.

    O_crank (crank pivot) is at origin.
    O_upper and O_lower positions are specified in the dimensions.
    """
    O_crank = (0.0, 0.0)
    O_upper = (d['O_upper_x'], d['O_upper_y'])
    O_lower = (d['O_lower_x'], d['O_lower_y'])
    return O_crank, O_upper, O_lower


def _compute_ternary_third_point(p1, p2, d12, d13, d23, lower=True):
    """
    Compute third point of a ternary link given two points and all three distances.

    Parameters
    ----------
    p1, p2 : tuple
        Known positions of first two points
    d12 : float
        Distance between p1 and p2 (for verification)
    d13 : float
        Distance from p1 to third point
    d23 : float
        Distance from p2 to third point
    lower : bool
        If True, choose solution with lower Y coordinate

    Returns
    -------
    tuple
        Position of third point
    """
    p3_1, p3_2 = _solve_intersection(p1, d13, p2, d23)
    if p3_1 is None:
        return None
    return p3_1 if (p3_1[1] < p3_2[1]) == lower else p3_2


def compute_initial_coords(d, angle=0):
    """
    Compute initial joint coordinates for a given crank angle.

    The Klann mechanism has two ternary links:
    1. Ternary-Coupler: A → Elbow → Knee (rigid triangle rotating with crank)
    2. Ternary-Leg: Hip → Knee → Foot (rigid triangle for leg)

    Uses fixed branch selection that produces full 360° rotation:
    - Elbow: higher Y (above crank axis)
    - Knee: lower Y
    - Hip: higher Y
    - Foot: lower Y
    """
    O_crank, O_upper, O_lower = compute_frame_positions(d)

    # Step 1: Crank endpoint A
    A = (d['crank'] * np.cos(angle), d['crank'] * np.sin(angle))

    # Step 2: Elbow - choose HIGHER Y for full rotation
    E1, E2 = _solve_intersection(A, d['A_elbow'], O_lower, d['lower_rocker'])
    if E1 is None:
        raise ValueError(f"Cannot compute Elbow at angle {angle}")
    Elbow = E1 if E1[1] > E2[1] else E2

    # Step 3: Knee - choose LOWER Y
    K1, K2 = _solve_intersection(A, d['A_knee'], Elbow, d['elbow_knee'])
    if K1 is None:
        raise ValueError(f"Cannot compute Knee at angle {angle}")
    Knee = K1 if K1[1] < K2[1] else K2

    # Step 4: Hip - choose HIGHER Y
    H1, H2 = _solve_intersection(O_upper, d['upper_rocker'], Knee, d['hip_knee'])
    if H1 is None:
        raise ValueError(f"Cannot compute Hip at angle {angle}")
    Hip = H1 if H1[1] > H2[1] else H2

    # Step 5: Foot - choose LOWER Y
    F1, F2 = _solve_intersection(Hip, d['hip_foot'], Knee, d['knee_foot'])
    if F1 is None:
        raise ValueError(f"Cannot compute Foot at angle {angle}")
    Foot = F1 if F1[1] < F2[1] else F2

    return {
        'O_crank': O_crank, 'O_upper': O_upper, 'O_lower': O_lower,
        'A': A, 'Elbow': Elbow, 'Knee': Knee, 'Hip': Hip, 'Foot': Foot
    }


def create_klann_linkage():
    """
    Create a single Klann leg unit.

    The Klann linkage is a 6-bar Stephenson III mechanism with:
    - 6 links: frame, crank, coupler, lower-rocker, upper-rocker, leg-ternary
    - 7 revolute joints (3 grounded)
    - 1 degree of freedom

    The two ternary links are:
    - Ternary coupler: A → Elbow → Knee (rigid triangle, Knee uses Hyperedge)
    - Ternary leg: Hip → Knee → Foot (rigid triangle, Foot uses Hyperedge)

    Returns
    -------
    Walker
        The Klann walking linkage with one foot.
    """
    d = get_scaled_dimensions()
    coords = compute_initial_coords(d, angle=0)

    # --- Topology ---
    hg = HypergraphLinkage(name="Klann")

    # Frame joints (ground)
    hg.add_node(Node("O_crank", role=NodeRole.GROUND, name="Frame"))
    hg.add_node(Node("O_upper", role=NodeRole.GROUND, name="Upper"))
    hg.add_node(Node("O_lower", role=NodeRole.GROUND, name="Frame2"))

    # Crank endpoint (driver, rotates around O_crank)
    hg.add_node(Node("A", role=NodeRole.DRIVER, name="Crank"))

    # Elbow: connects ternary coupler (from A) and lower rocker (from O_lower)
    hg.add_node(Node("elbow", role=NodeRole.DRIVEN, name="Elbow"))

    # Knee: part of ternary coupler (A-Elbow-Knee rigid triangle)
    hg.add_node(Node("knee", role=NodeRole.DRIVEN, name="Knee"))

    # Hip: connects upper rocker (from O_upper) and hip_knee (from Knee)
    hg.add_node(Node("hip", role=NodeRole.DRIVEN, name="Hip"))

    # Foot: part of leg-ternary (Hip-Knee-Foot rigid triangle)
    hg.add_node(Node("foot", role=NodeRole.DRIVEN, name="Foot"))

    # Edges
    hg.add_edge(Edge("O_crank_A", "O_crank", "A"))           # crank link
    hg.add_edge(Edge("A_elbow", "A", "elbow"))                # coupler link
    hg.add_edge(Edge("O_lower_elbow", "O_lower", "elbow"))    # lower rocker

    # Knee: rigid triangle with A and Elbow (ternary coupler)
    hg.add_edge(Edge("A_knee", "A", "knee"))
    hg.add_hyperedge(Hyperedge("ternary_coupler", nodes=("A", "elbow", "knee")))

    # Hip: upper rocker + connection to Knee
    hg.add_edge(Edge("O_upper_hip", "O_upper", "hip"))        # upper rocker
    hg.add_edge(Edge("knee_hip", "knee", "hip"))               # hip-knee link

    # Foot: rigid triangle with Hip and Knee (ternary leg)
    hg.add_edge(Edge("hip_foot", "hip", "foot"))
    hg.add_hyperedge(Hyperedge("ternary_leg", nodes=("hip", "knee", "foot")))

    # --- Dimensions ---
    dims = Dimensions(
        node_positions={
            "O_crank": coords['O_crank'],
            "O_upper": coords['O_upper'],
            "O_lower": coords['O_lower'],
            "A": coords['A'],
            "elbow": coords['Elbow'],
            "knee": coords['Knee'],
            "hip": coords['Hip'],
            "foot": coords['Foot'],
        },
        driver_angles={
            "A": DriverAngle(angular_velocity=2 * pi / LAP_POINTS),
        },
        edge_distances={
            "O_crank_A": d['crank'],
            "A_elbow": d['A_elbow'],
            "O_lower_elbow": d['lower_rocker'],
            "A_knee": d['A_knee'],
            "O_upper_hip": d['upper_rocker'],
            "knee_hip": d['hip_knee'],
            "hip_foot": d['hip_foot'],
        },
    )

    walker = Walker(hg, dims, name="Klann")

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

    walker = create_klann_walker(n_legs=2)
    # Motor rate: positive for counterclockwise rotation (walking forward)
    walker.motor_rates = 4.0

    # Run the visualization
    ls.video(walker, duration=15, dynamic_camera=True)


if __name__ == "__main__":
    main()
