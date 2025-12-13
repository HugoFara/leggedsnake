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
import numpy as np
from pylinkage import Static, Crank, Revolute, Fixed
import leggedsnake as ls

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


def _compute_fixed_angle(p0, p1, p_target):
    """
    Compute the angle for a Fixed joint.

    Parameters
    ----------
    p0 : tuple
        Position of joint0 (the anchor point)
    p1 : tuple
        Position of joint1 (the reference direction point)
    p_target : tuple
        Position of the Fixed joint itself

    Returns
    -------
    float
        Angle from (p0->p1) direction to (p0->p_target) direction
    """
    angle_to_ref = np.arctan2(p1[1] - p0[1], p1[0] - p0[0])
    angle_to_target = np.arctan2(p_target[1] - p0[1], p_target[0] - p0[0])
    return angle_to_target - angle_to_ref


def create_klann_linkage():
    """
    Create a single Klann leg unit.

    The Klann linkage is a 6-bar Stephenson III mechanism with:
    - 6 links: frame, crank, coupler, lower-rocker, upper-rocker, leg-ternary
    - 7 revolute joints (3 grounded)
    - 1 degree of freedom

    The two ternary links are:
    - Ternary coupler: A → Elbow → Knee (rigid triangle, Knee uses Fixed joint)
    - Ternary leg: Hip → Knee → Foot (rigid triangle, Foot uses Fixed joint)

    Returns
    -------
    Walker
        The Klann walking linkage with one foot.
    """
    d = get_scaled_dimensions()
    coords = compute_initial_coords(d, angle=0)

    # Create frame joints (Static)
    O_crank = Static(x=coords['O_crank'][0], y=coords['O_crank'][1], name="Frame")
    O_upper = Static(x=coords['O_upper'][0], y=coords['O_upper'][1], name="Upper")
    O_lower = Static(x=coords['O_lower'][0], y=coords['O_lower'][1], name="Frame2")
    O_upper.joint0 = O_crank
    O_lower.joint0 = O_crank

    # Crank endpoint (rotates around O_crank)
    A = Crank(
        x=coords['A'][0], y=coords['A'][1],
        joint0=O_crank,
        distance=d['crank'],
        angle=-2 * np.pi / LAP_POINTS,
        name="Crank"
    )

    # Elbow: connects ternary coupler (from A) and lower_rocker (from O_lower)
    Elbow = Revolute(
        x=coords['Elbow'][0], y=coords['Elbow'][1],
        joint0=A, joint1=O_lower,
        distance0=d['A_elbow'], distance1=d['lower_rocker'],
        name="Elbow"
    )

    # Knee: part of ternary coupler (A-Elbow-Knee rigid triangle)
    # Use Fixed joint to signal rigid structure for physics
    knee_angle = _compute_fixed_angle(coords['A'], coords['Elbow'], coords['Knee'])
    Knee = Fixed(
        x=coords['Knee'][0], y=coords['Knee'][1],
        joint0=A, joint1=Elbow,
        distance=d['A_knee'],
        angle=knee_angle,
        name="Knee"
    )

    # Hip: connects upper_rocker (from O_upper) and hip_knee (from Knee)
    Hip = Revolute(
        x=coords['Hip'][0], y=coords['Hip'][1],
        joint0=O_upper, joint1=Knee,
        distance0=d['upper_rocker'], distance1=d['hip_knee'],
        name="Hip"
    )

    # Foot: part of leg-ternary (Hip-Knee-Foot rigid triangle)
    # Use Fixed joint to signal rigid structure for physics
    foot_angle = _compute_fixed_angle(coords['Hip'], coords['Knee'], coords['Foot'])
    Foot = Fixed(
        x=coords['Foot'][0], y=coords['Foot'][1],
        joint0=Hip, joint1=Knee,
        distance=d['hip_foot'],
        angle=foot_angle,
        name="Foot"
    )

    joints = [O_crank, O_upper, O_lower, A, Elbow, Knee, Hip, Foot]
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

    walker = create_klann_walker(n_legs=2)

    # Run the visualization
    ls.video(walker, duration=15, dynamic_camera=True)


if __name__ == "__main__":
    main()
