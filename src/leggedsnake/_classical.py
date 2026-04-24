#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Canonical geometries for classical walking linkages.

Internal module: shared geometry solvers and topology/dimension builders
for Theo Jansen, Klann, Chebyshev Lambda, Strider, and TrotBot
mechanisms. Exposed as class methods on :class:`leggedsnake.walker.Walker`.
"""
from __future__ import annotations

from math import atan2, cos, sin, sqrt

from pylinkage.dimensions import Dimensions, DriverAngle
from pylinkage.hypergraph import HypergraphLinkage, NodeRole
from pylinkage.hypergraph.core import Edge, Hyperedge, Node


Point = tuple[float, float]


# Jansen's "Holy Numbers" — unscaled link lengths of the canonical 8-bar.
JANSEN_HOLY_NUMBERS: dict[str, float] = {
    "a": 38.0,   # X-offset from O to B (frame)
    "b": 41.5,   # B to C
    "c": 39.3,   # B to D
    "d": 40.1,   # B to E
    "e": 55.8,   # C to E
    "f": 39.4,   # E to F
    "g": 36.7,   # D to F
    "h": 65.7,   # F to G (foot)
    "i": 49.0,   # D to G (foot)
    "j": 50.0,   # A to C
    "k": 61.9,   # A to D
    "l": 7.8,    # Y-offset from O to B (frame)
    "m": 15.0,   # Crank radius (O to A)
}

# Klann dimensions from US Patent 6,260,862.
KLANN_PATENT_DIMENSIONS: dict[str, float] = {
    "O_upper_x": -0.233,
    "O_upper_y": 0.616,
    "O_lower_x": -0.590,
    "O_lower_y": -0.176,
    "crank": 0.268,
    "A_elbow": 0.590,
    "A_knee": 1.105,
    "elbow_knee": 0.522,
    "lower_rocker": 0.321,
    "upper_rocker": 0.518,
    "hip_knee": 0.897,
    "knee_foot": 0.897,
    "hip_foot": 1.732,
}


def _solve_intersection(
    p1: Point, r1: float, p2: Point, r2: float
) -> tuple[Point, Point] | tuple[None, None]:
    """Return the two circle-circle intersections, or ``(None, None)`` if none."""
    dx = p2[0] - p1[0]
    dy = p2[1] - p1[1]
    d = sqrt(dx * dx + dy * dy)
    if d > r1 + r2 or d < abs(r1 - r2) or d == 0:
        return None, None
    a_val = (r1 * r1 - r2 * r2 + d * d) / (2 * d)
    h_sq = r1 * r1 - a_val * a_val
    if h_sq < 0:
        return None, None
    h_val = sqrt(h_sq)
    mx = p1[0] + a_val * dx / d
    my = p1[1] + a_val * dy / d
    sol1 = (mx + h_val * (-dy) / d, my + h_val * dx / d)
    sol2 = (mx - h_val * (-dy) / d, my - h_val * dx / d)
    return sol1, sol2


def _jansen_positions(h: dict[str, float], angle: float) -> dict[str, Point]:
    """Compute initial joint coordinates for Jansen at a given crank angle.

    Uses the canonical branch selections (C2, D1, E1, F1, G1) that yield a
    full-revolution buildable mechanism with the Holy Numbers.
    """
    origin: Point = (0.0, 0.0)
    frame_b: Point = (-h["a"], -h["l"])
    crank_a: Point = (h["m"] * cos(angle), h["m"] * sin(angle))

    c1, c2 = _solve_intersection(crank_a, h["j"], frame_b, h["b"])
    if c1 is None or c2 is None:
        raise ValueError(f"Jansen C unsolvable at angle {angle}")
    node_c = c2

    d1, d2 = _solve_intersection(crank_a, h["k"], frame_b, h["c"])
    if d1 is None or d2 is None:
        raise ValueError(f"Jansen D unsolvable at angle {angle}")
    node_d = d1

    e1, e2 = _solve_intersection(frame_b, h["d"], node_c, h["e"])
    if e1 is None or e2 is None:
        raise ValueError(f"Jansen E unsolvable at angle {angle}")
    node_e = e1

    f1, f2 = _solve_intersection(node_d, h["g"], node_e, h["f"])
    if f1 is None or f2 is None:
        raise ValueError(f"Jansen F unsolvable at angle {angle}")
    node_f = f1

    g1, g2 = _solve_intersection(node_d, h["i"], node_f, h["h"])
    if g1 is None or g2 is None:
        raise ValueError(f"Jansen G unsolvable at angle {angle}")
    node_g = g1

    return {
        "O": origin, "B": frame_b, "A": crank_a,
        "C": node_c, "D": node_d, "E": node_e,
        "F": node_f, "G": node_g,
    }


def build_jansen(
    scale: float,
    initial_crank_angle: float,
    angular_velocity: float,
    name: str,
    lengths: dict[str, float] | None = None,
) -> tuple[HypergraphLinkage, Dimensions]:
    """Build the (topology, dimensions) of a single Theo Jansen leg.

    The Jansen linkage is an 8-bar planar mechanism. All mobile joints
    are Revolute (circle-circle intersection) — no ternary links.

    When ``lengths`` is given, its entries override the canonical
    :data:`JANSEN_HOLY_NUMBERS` *before* scaling — use this for
    optimization over the 13 length parameters. Partial dicts are
    allowed (missing keys fall back to the canonical values).
    """
    base = dict(JANSEN_HOLY_NUMBERS)
    if lengths is not None:
        base.update(lengths)
    h = {k: v * scale for k, v in base.items()}
    coords = _jansen_positions(h, initial_crank_angle)

    hg = HypergraphLinkage(name=name)
    hg.add_node(Node("O", role=NodeRole.GROUND, name="O (crank center)"))
    hg.add_node(Node("B", role=NodeRole.GROUND, name="B (frame)"))
    hg.add_node(Node("A", role=NodeRole.DRIVER, name="A (crank)"))
    for nid in ("C", "D", "E", "F"):
        hg.add_node(Node(nid, role=NodeRole.DRIVEN, name=nid))
    hg.add_node(Node("G", role=NodeRole.DRIVEN, name="G (foot)"))

    edges: list[tuple[str, str, str, float]] = [
        ("O_A", "O", "A", h["m"]),
        ("A_C", "A", "C", h["j"]),
        ("B_C", "B", "C", h["b"]),
        ("A_D", "A", "D", h["k"]),
        ("B_D", "B", "D", h["c"]),
        ("B_E", "B", "E", h["d"]),
        ("C_E", "C", "E", h["e"]),
        ("D_F", "D", "F", h["g"]),
        ("E_F", "E", "F", h["f"]),
        ("D_G", "D", "G", h["i"]),
        ("F_G", "F", "G", h["h"]),
    ]
    edge_distances: dict[str, float] = {}
    for eid, src, tgt, length in edges:
        hg.add_edge(Edge(eid, src, tgt))
        edge_distances[eid] = length

    dims = Dimensions(
        node_positions=coords,
        driver_angles={"A": DriverAngle(angular_velocity=angular_velocity)},
        edge_distances=edge_distances,
    )
    return hg, dims


def _klann_positions(d: dict[str, float], angle: float) -> dict[str, Point]:
    """Compute initial joint coordinates for Klann at a given crank angle."""
    o_crank: Point = (0.0, 0.0)
    o_upper: Point = (d["O_upper_x"], d["O_upper_y"])
    o_lower: Point = (d["O_lower_x"], d["O_lower_y"])
    crank_a: Point = (d["crank"] * cos(angle), d["crank"] * sin(angle))

    e1, e2 = _solve_intersection(
        crank_a, d["A_elbow"], o_lower, d["lower_rocker"],
    )
    if e1 is None or e2 is None:
        raise ValueError(f"Klann elbow unsolvable at angle {angle}")
    elbow = e1 if e1[1] > e2[1] else e2

    k1, k2 = _solve_intersection(crank_a, d["A_knee"], elbow, d["elbow_knee"])
    if k1 is None or k2 is None:
        raise ValueError(f"Klann knee unsolvable at angle {angle}")
    knee = k1 if k1[1] < k2[1] else k2

    h1, h2 = _solve_intersection(o_upper, d["upper_rocker"], knee, d["hip_knee"])
    if h1 is None or h2 is None:
        raise ValueError(f"Klann hip unsolvable at angle {angle}")
    hip = h1 if h1[1] > h2[1] else h2

    f1, f2 = _solve_intersection(hip, d["hip_foot"], knee, d["knee_foot"])
    if f1 is None or f2 is None:
        raise ValueError(f"Klann foot unsolvable at angle {angle}")
    foot = f1 if f1[1] < f2[1] else f2

    return {
        "O_crank": o_crank, "O_upper": o_upper, "O_lower": o_lower,
        "A": crank_a, "elbow": elbow, "knee": knee, "hip": hip, "foot": foot,
    }


def build_klann(
    scale: float,
    initial_crank_angle: float,
    angular_velocity: float,
    name: str,
) -> tuple[HypergraphLinkage, Dimensions]:
    """Build the (topology, dimensions) of a single Klann leg.

    6-bar Stephenson-III with two rigid triangles:
    ``ternary_coupler`` (A, elbow, knee) and ``ternary_leg`` (hip, knee, foot).
    """
    scaled = dict(KLANN_PATENT_DIMENSIONS)
    scaled["O_upper_x"] *= scale
    scaled["O_upper_y"] *= scale
    scaled["O_lower_x"] *= scale
    scaled["O_lower_y"] *= scale
    for key in (
        "crank", "A_elbow", "A_knee", "elbow_knee",
        "lower_rocker", "upper_rocker",
        "hip_knee", "knee_foot", "hip_foot",
    ):
        scaled[key] *= scale

    coords = _klann_positions(scaled, initial_crank_angle)

    hg = HypergraphLinkage(name=name)
    hg.add_node(Node("O_crank", role=NodeRole.GROUND, name="O_crank"))
    hg.add_node(Node("O_upper", role=NodeRole.GROUND, name="O_upper"))
    hg.add_node(Node("O_lower", role=NodeRole.GROUND, name="O_lower"))
    hg.add_node(Node("A", role=NodeRole.DRIVER, name="A (crank)"))
    hg.add_node(Node("elbow", role=NodeRole.DRIVEN, name="elbow"))
    hg.add_node(Node("knee", role=NodeRole.DRIVEN, name="knee"))
    hg.add_node(Node("hip", role=NodeRole.DRIVEN, name="hip"))
    hg.add_node(Node("foot", role=NodeRole.DRIVEN, name="foot"))

    edges: list[tuple[str, str, str, float]] = [
        ("O_crank_A", "O_crank", "A", scaled["crank"]),
        ("A_elbow", "A", "elbow", scaled["A_elbow"]),
        ("O_lower_elbow", "O_lower", "elbow", scaled["lower_rocker"]),
        ("A_knee", "A", "knee", scaled["A_knee"]),
        ("O_upper_hip", "O_upper", "hip", scaled["upper_rocker"]),
        ("knee_hip", "knee", "hip", scaled["hip_knee"]),
        ("hip_foot", "hip", "foot", scaled["hip_foot"]),
    ]
    edge_distances: dict[str, float] = {}
    for eid, src, tgt, length in edges:
        hg.add_edge(Edge(eid, src, tgt))
        edge_distances[eid] = length

    hg.add_hyperedge(Hyperedge(
        "ternary_coupler", nodes=("A", "elbow", "knee"),
    ))
    hg.add_hyperedge(Hyperedge(
        "ternary_leg", nodes=("hip", "knee", "foot"),
    ))

    dims = Dimensions(
        node_positions=coords,
        driver_angles={"A": DriverAngle(angular_velocity=angular_velocity)},
        edge_distances=edge_distances,
    )
    return hg, dims


def _chebyshev_positions(
    crank: float,
    coupler: float,
    rocker: float,
    ground_length: float,
    foot_ratio: float,
    angle: float,
) -> dict[str, Point]:
    """Compute initial joint coordinates for the Chebyshev Lambda."""
    o1: Point = (0.0, 0.0)
    o2: Point = (ground_length, 0.0)
    a_pt: Point = (crank * cos(angle), crank * sin(angle))

    b1, b2 = _solve_intersection(a_pt, coupler, o2, rocker)
    if b1 is None or b2 is None:
        raise ValueError(
            f"Chebyshev coupler/rocker do not meet at angle {angle}"
        )
    b_pt = b1 if b1[1] < b2[1] else b2
    p_pt: Point = (
        a_pt[0] + foot_ratio * (b_pt[0] - a_pt[0]),
        a_pt[1] + foot_ratio * (b_pt[1] - a_pt[1]),
    )
    return {"O1": o1, "O2": o2, "A": a_pt, "B": b_pt, "P": p_pt}


def build_chebyshev(
    crank: float,
    coupler: float,
    rocker: float,
    ground_length: float,
    foot_ratio: float,
    initial_crank_angle: float,
    angular_velocity: float,
    name: str,
) -> tuple[HypergraphLinkage, Dimensions]:
    """Build the (topology, dimensions) of a single Chebyshev Lambda leg.

    4-bar crank-rocker with a coupler point P that traces an approximate
    straight line. P is rigidly attached to the A-B coupler (modelled as
    a hyperedge ``triangle_P``).
    """
    coords = _chebyshev_positions(
        crank, coupler, rocker, ground_length, foot_ratio,
        initial_crank_angle,
    )
    foot_distance = foot_ratio * coupler

    hg = HypergraphLinkage(name=name)
    hg.add_node(Node("O1", role=NodeRole.GROUND, name="O1 (crank)"))
    hg.add_node(Node("O2", role=NodeRole.GROUND, name="O2 (rocker)"))
    hg.add_node(Node("A", role=NodeRole.DRIVER, name="A (crank)"))
    hg.add_node(Node("B", role=NodeRole.DRIVEN, name="B (coupler-rocker)"))
    hg.add_node(Node("P", role=NodeRole.DRIVEN, name="P (foot)"))

    # Rigid A-B-P triangle: two edges + hyperedge convention.
    edges: list[tuple[str, str, str, float]] = [
        ("O1_A", "O1", "A", crank),
        ("A_B", "A", "B", coupler),
        ("O2_B", "O2", "B", rocker),
        ("A_P", "A", "P", foot_distance),
    ]
    edge_distances: dict[str, float] = {}
    for eid, src, tgt, length in edges:
        hg.add_edge(Edge(eid, src, tgt))
        edge_distances[eid] = length

    hg.add_hyperedge(Hyperedge("triangle_P", nodes=("A", "B", "P")))

    dims = Dimensions(
        node_positions=coords,
        driver_angles={"A": DriverAngle(angular_velocity=angular_velocity)},
        edge_distances=edge_distances,
    )
    return hg, dims


# --- Ghassaei (boim.com/Walkin8r reference figure, Figure 5.4.4 of her 2011
# Pomona thesis) -----------------------------------------------------------
# A = crank axle ground, B = frame hinge ground, C = crank tip driver.
# The crank starts nearly vertical (0.085 rad off +y from A). Five RRR
# dyads: D = RRR(C, B), F = RRR(C, B) (other branch), H = RRR(D, B)
# (the "unnamed" intermediate joint), E = RRR(H, B) (real labelled E),
# G = RRR(E, F) (the foot). ``H_to_E`` is unspecified on the Boim figure
# and is assumed 75 (matching the other outer bars).
GHASSAEI_DIMENSIONS: dict[str, float] = {
    "crank": 26.0,           # A-C
    "ground": 53.0,          # A-B
    "C_to_outer": 56.0,      # C-D, C-F
    "B_to_outer": 77.0,      # B-D, B-F, B-H, B-E
    "D_to_H": 75.0,          # intermediate dyad arm
    "H_to_E": 130.0,         # not on figure; fit against Wikibooks locus
    "outer_to_foot": 75.0,   # E-G, F-G
}

_GHASSAEI_EDGES: tuple[tuple[str, str, str, float], ...] = (
    ("AC", "A", "C", 26.0),
    ("CD", "C", "D", 56.0),
    ("BD", "B", "D", 77.0),
    ("CF", "C", "F", 56.0),
    ("BF", "B", "F", 77.0),
    ("DH", "D", "H", 75.0),
    ("BH", "B", "H", 77.0),
    ("HE", "H", "E", 130.0),
    ("BE", "B", "E", 77.0),
    ("EG", "E", "G", 75.0),
    ("FG", "F", "G", 75.0),
)


def _circle_intersect(
    p1: Point, r1: float, p2: Point, r2: float,
) -> tuple[Point, Point] | None:
    """Return the two intersection points of two circles (or ``None``)."""
    dx, dy = p2[0] - p1[0], p2[1] - p1[1]
    d = sqrt(dx * dx + dy * dy)
    if d < abs(r1 - r2) or d > r1 + r2:
        return None
    a = (r1 ** 2 - r2 ** 2 + d ** 2) / (2 * d)
    h = sqrt(max(r1 ** 2 - a ** 2, 0.0))
    mx = p1[0] + a * dx / d
    my = p1[1] + a * dy / d
    return (
        (mx + h * dy / d, my - h * dx / d),
        (mx - h * dy / d, my + h * dx / d),
    )


def _pick_branch(
    pair: tuple[Point, Point] | None,
    pick: str,
    what: str = "RRR dyad",
) -> Point:
    """Pick one branch of a circle-circle intersection by side."""
    if pair is None:
        raise ValueError(f"{what}: circles do not intersect")
    p, q = pair
    if pick == "upper":
        return p if p[1] > q[1] else q
    if pick == "lower":
        return p if p[1] < q[1] else q
    if pick == "left":
        return p if p[0] < q[0] else q
    if pick == "right":
        return p if p[0] > q[0] else q
    raise ValueError(f"unknown branch {pick!r}")


def build_ghassaei(
    scale: float,
    initial_crank_angle: float,
    angular_velocity: float,
    name: str,
) -> tuple[HypergraphLinkage, Dimensions]:
    """Build Amanda Ghassaei's 5-dyad leg (Boim/Walkin8r / thesis Fig. 5.4.4).

    Nodes: A, B grounds; C crank tip; D, F outer dyads off (C, B);
    H intermediate (unnamed on the figure) off (D, B); E real joint off
    (H, B); G foot off (E, F). Classical Ghassaei dimensions are applied
    exactly (crank=26, ground=53, 56/77 inner+outer, 75 closing bars).
    Initial crank angle is 0.085 rad off vertical. H-to-E is not on the
    figure; 130 reproduces the Wikibooks reference foot-locus aspect.
    """
    s = scale
    A: Point = (0.0, 0.0)
    B: Point = (-53.0 * s, 0.0)
    # Crank starts 0.085 rad CCW from vertical (+y) above A.
    phi = initial_crank_angle + 0.085
    C: Point = (-26.0 * s * sin(phi), 26.0 * s * cos(phi))

    df_pair = _circle_intersect(C, 56.0 * s, B, 77.0 * s)
    D = _pick_branch(df_pair, "upper", "Ghassaei D")
    F = _pick_branch(df_pair, "lower", "Ghassaei F")

    # H is the "unnamed" intermediate joint, left of D. E is the real
    # lower-left joint. The figure does not specify H-E, and a value of
    # 130 reproduces the Wikibooks foot-locus aspect (~0.24).
    H = _pick_branch(_circle_intersect(D, 75.0 * s, B, 77.0 * s), "left", "Ghassaei H")
    E = _pick_branch(_circle_intersect(H, 130.0 * s, B, 77.0 * s), "lower", "Ghassaei E")
    G = _pick_branch(_circle_intersect(E, 75.0 * s, F, 75.0 * s), "lower", "Ghassaei G")

    positions: dict[str, Point] = {
        "A": A, "B": B, "C": C, "D": D, "F": F, "H": H, "E": E, "G": G,
    }

    hg = HypergraphLinkage(name=name)
    hg.add_node(Node("A", role=NodeRole.GROUND))
    hg.add_node(Node("B", role=NodeRole.GROUND))
    hg.add_node(Node("C", role=NodeRole.DRIVER))
    for nid in ("D", "F", "H", "E", "G"):
        hg.add_node(Node(nid, role=NodeRole.DRIVEN))

    edge_distances: dict[str, float] = {}
    for eid, src, tgt, length in _GHASSAEI_EDGES:
        hg.add_edge(Edge(eid, src, tgt))
        edge_distances[eid] = length * s

    dims = Dimensions(
        node_positions=positions,
        driver_angles={"C": DriverAngle(angular_velocity=angular_velocity)},
        edge_distances=edge_distances,
    )
    return hg, dims


# TrotBot bar lengths from DIY Walkers' combined Python simulator
# (https://www.diywalkers.com/python-linkage-simulator.html, 2024 release).
# Indices match the bar[0..17] naming in the upstream simulator so that
# cross-referencing against the source code stays obvious.
TROTBOT_BARS: dict[int, float] = {
    0: 4.0,     # crank
    1: 6.0,
    2: 8.0,
    3: 2.0,    # 3→2 extension
    4: 6.0,
    5: 2.0,    # 5→4 extension
    6: 11.0,
    7: 3.0,
    8: 9.0,
    9: 8.0,
    10: 1.0,   # 1→2 extension
    17: 7.55,
}

# TrotBot frame offset (crank-center at origin; fixed frame pivot at (-7, 6)).
TROTBOT_FRAME_OFFSET: Point = (-7.0, 6.0)


def build_strider(
    crank: float,
    triangle: float,
    femur: float,
    rocker_l: float,
    rocker_s: float,
    tibia: float,
    foot: float,
    angular_velocity: float,
    name: str,
) -> tuple[HypergraphLinkage, Dimensions]:
    """Build the (topology, dimensions) of one pair of Strider legs.

    The Strider is a symmetric 11-node walker with four rigid triangles:
    two attach the frame points ``B`` and ``B_p`` to the ground pair
    (``A``, ``Y``); two attach the ankles ``F`` and ``G`` rigidly to the
    ``C-E`` and ``C-D`` rocker links. Feet ``H`` and ``I`` hang off the
    knee-ankle pairs. Symmetry halves the parameter count: only 6
    lengths (plus the crank) describe both legs.

    Parameters
    ----------
    crank : float
        ``A-C`` length (crank arm).
    triangle : float
        Distance from ground pair to frame points (``A-B`` = ``Y-B``
        and mirror).
    femur : float
        ``B_p-D`` / ``B-E`` length.
    rocker_l : float
        ``C-D`` / ``C-E`` length.
    rocker_s : float
        Rocker-to-ankle length (``C-F``, ``E-F``, ``C-G``, ``D-G``).
    tibia : float
        ``D-H`` / ``E-I`` length.
    foot : float
        ``F-H`` / ``G-I`` length.
    angular_velocity, name
        Standard factory kwargs.
    """
    # Initial positions are *derived* from the 7 link lengths so the declared
    # edge distances match the geometry exactly. Otherwise optimizers that
    # mutate the dimensions get silently clamped to the hardcoded layout.
    #
    # Frame pair (A, Y) lies on the y-axis with |AY| = 1 (this is the
    # chassis height, kept as an internal constant — not exposed as a
    # param to preserve the 7-parameter signature). B and B_p form rigid
    # isoceles triangles with AY: |AB| = |YB| = triangle, so B sits on the
    # perpendicular bisector of AY at x = √(triangle² − 1/4). For the
    # geometry to close, triangle must exceed 1/2.
    if triangle <= 0.5:
        raise ValueError(
            f"triangle={triangle} ≤ 0.5: isoceles frame collapses onto A-Y"
        )
    A: Point = (0.0, 0.0)
    Y: Point = (0.0, 1.0)
    bx = sqrt(triangle * triangle - 0.25)
    B: Point = (bx, 0.5)
    B_p: Point = (-bx, 0.5)

    # Crank starts pointing straight down (angle = 3π/2 from +x).
    C: Point = (0.0, -crank)

    # Knees via RRR dyad off the corresponding frame point and the crank.
    D = _pick_branch(
        _circle_intersect(B_p, femur, C, rocker_l), "lower", "Strider D (left knee)",
    )
    E = _pick_branch(
        _circle_intersect(B, femur, C, rocker_l), "lower", "Strider E (right knee)",
    )

    # Ankles: F rigid on isoceles triangle C-D-F; G rigid on C-E-G.
    # Requires 2·rocker_s ≥ rocker_l for the isoceles triangle to close.
    F = _pick_branch(
        _circle_intersect(C, rocker_s, D, rocker_s), "lower",
        "Strider F (left ankle) — need 2·rocker_s ≥ rocker_l",
    )
    G = _pick_branch(
        _circle_intersect(C, rocker_s, E, rocker_s), "lower",
        "Strider G (right ankle) — need 2·rocker_s ≥ rocker_l",
    )

    # Feet via RRR dyad off the knee (tibia) and ankle (foot) of the same side.
    H = _pick_branch(
        _circle_intersect(D, tibia, F, foot), "lower", "Strider H (left foot)",
    )
    I = _pick_branch(  # noqa: E741 — joint label, mirrors A-H above
        _circle_intersect(E, tibia, G, foot), "lower", "Strider I (right foot)",
    )

    positions: dict[str, Point] = {
        "A": A, "Y": Y, "B": B, "B_p": B_p, "C": C,
        "D": D, "E": E, "F": F, "G": G, "H": H, "I": I,
    }

    hg = HypergraphLinkage(name=name)
    hg.add_node(Node("A", role=NodeRole.GROUND, name="A"))
    hg.add_node(Node("Y", role=NodeRole.GROUND, name="Y"))
    hg.add_node(Node("B", role=NodeRole.DRIVEN, name="frame right"))
    hg.add_node(Node("B_p", role=NodeRole.DRIVEN, name="frame left"))
    hg.add_node(Node("C", role=NodeRole.DRIVER, name="crank"))
    hg.add_node(Node("D", role=NodeRole.DRIVEN, name="left knee"))
    hg.add_node(Node("E", role=NodeRole.DRIVEN, name="right knee"))
    hg.add_node(Node("F", role=NodeRole.DRIVEN, name="left ankle"))
    hg.add_node(Node("G", role=NodeRole.DRIVEN, name="right ankle"))
    hg.add_node(Node("H", role=NodeRole.DRIVEN, name="left foot"))
    hg.add_node(Node("I", role=NodeRole.DRIVEN, name="right foot"))

    # Uncrossed topology: left ankle F rigid on C-D (left knee side), right
    # ankle G rigid on C-E (right knee side). The pre-refactor code had
    # F/G attached to the *opposite*-side rocker, which was inconsistent
    # with the INIT_COORD positions it shipped alongside.
    edges: list[tuple[str, str, str, float]] = [
        ("A_B", "A", "B", triangle),
        ("Y_B", "Y", "B", triangle),
        ("A_B_p", "A", "B_p", triangle),
        ("Y_B_p", "Y", "B_p", triangle),
        ("A_C", "A", "C", crank),
        ("B_p_D", "B_p", "D", femur),
        ("C_D", "C", "D", rocker_l),
        ("B_E", "B", "E", femur),
        ("C_E", "C", "E", rocker_l),
        ("C_F", "C", "F", rocker_s),
        ("D_F", "D", "F", rocker_s),
        ("C_G", "C", "G", rocker_s),
        ("E_G", "E", "G", rocker_s),
        ("D_H", "D", "H", tibia),
        ("F_H", "F", "H", foot),
        ("E_I", "E", "I", tibia),
        ("G_I", "G", "I", foot),
    ]
    edge_distances: dict[str, float] = {}
    for eid, src, tgt, length in edges:
        hg.add_edge(Edge(eid, src, tgt))
        edge_distances[eid] = length

    hg.add_hyperedge(Hyperedge("triangle_B", nodes=("A", "Y", "B")))
    hg.add_hyperedge(Hyperedge("triangle_B_p", nodes=("A", "Y", "B_p")))
    hg.add_hyperedge(Hyperedge("triangle_F", nodes=("C", "D", "F")))
    hg.add_hyperedge(Hyperedge("triangle_G", nodes=("C", "E", "G")))

    dims = Dimensions(
        node_positions=positions,
        driver_angles={"C": DriverAngle(angular_velocity=angular_velocity)},
        edge_distances=edge_distances,
    )
    return hg, dims


def _line_extend(p1: Point, p2: Point, length: float) -> Point:
    """Point on the line through p1 and p2, extended by ``length`` past p1.

    Matches DIY Walkers' ``lineextend(X1,Y1,X2,Y2,Length)``: returns the
    point at distance ``length`` from ``p1`` along the direction opposite
    to ``p2`` (i.e. past ``p1`` away from ``p2``).
    """
    slope = atan2(p2[1] - p1[1], p2[0] - p1[0])
    return (
        p1[0] - length * cos(slope),
        p1[1] - length * sin(slope),
    )


def _pick_intersection(
    sols: tuple[Point, Point] | tuple[None, None], branch: str,
) -> Point:
    """Pick one of the two circle intersections by DIY Walkers' branch rule."""
    sol1, sol2 = sols
    if sol1 is None or sol2 is None:
        raise ValueError(f"TrotBot joint unsolvable (branch={branch!r})")
    if branch == "high":
        return sol1 if sol1[1] > sol2[1] else sol2
    if branch == "low":
        return sol1 if sol1[1] < sol2[1] else sol2
    if branch == "left":
        return sol1 if sol1[0] < sol2[0] else sol2
    if branch == "right":
        return sol1 if sol1[0] > sol2[0] else sol2
    raise ValueError(f"Unknown branch {branch!r}")


def _trotbot_positions(
    bars: dict[int, float], angle: float, frame_offset: Point,
) -> dict[str, Point]:
    """Compute initial TrotBot joint positions for a given crank angle.

    Replicates the joint-assembly order used in DIY Walkers' combined
    Python simulator (``trotbot_strider_strandbeest_and_klann_ver_3.py``).
    """
    # Ground and driver.
    j0: Point = (0.0, 0.0)
    j3: Point = frame_offset
    j1: Point = (bars[0] * cos(angle), bars[0] * sin(angle))
    # j2: circle-circle of (j1, bar1) and (j3, bar2), high branch.
    j2 = _pick_intersection(
        _solve_intersection(j1, bars[1], j3, bars[2]), "high",
    )
    # j4: collinear extension of 3 → 2, past 2 by bar3.
    j4 = _line_extend(j2, j3, bars[3])
    # j5: circle-circle of (j4, bar4) and (j1, bar6), low branch.
    j5 = _pick_intersection(
        _solve_intersection(j4, bars[4], j1, bars[6]), "low",
    )
    # j6: collinear extension of 5 → 4, past 4 by bar5.
    j6 = _line_extend(j4, j5, bars[5])
    # j9: collinear extension of 1 → 2, past 2 by bar10.
    j9 = _line_extend(j2, j1, bars[10])
    # j8: circle-circle of (j1, bar7) and (j2, bar17), left branch.
    j8 = _pick_intersection(
        _solve_intersection(j1, bars[7], j2, bars[17]), "left",
    )
    # j7 (foot): circle-circle of (j8, bar8) and (j6, bar9), low branch.
    j7 = _pick_intersection(
        _solve_intersection(j8, bars[8], j6, bars[9]), "low",
    )
    return {
        "j0": j0, "j1": j1, "j2": j2, "j3": j3, "j4": j4,
        "j5": j5, "j6": j6, "j7": j7, "j8": j8, "j9": j9,
    }


def build_trotbot(
    scale: float,
    initial_crank_angle: float,
    angular_velocity: float,
    name: str,
) -> tuple[HypergraphLinkage, Dimensions]:
    """Build the (topology, dimensions) of one TrotBot leg.

    Uses the bar lengths from Wade & Ben Vagle's combined Python
    simulator at https://www.diywalkers.com (TrotBot ver 3, 2024). The
    mechanism has two ground nodes (``j0`` crank-axle, ``j3`` frame),
    one driver (``j1``), and seven driven joints; ``j7`` is the foot.

    Three collinear rigid ternaries encode the "line-extension" joints
    (``j4`` past ``j2``, ``j6`` past ``j4``, ``j9`` past ``j2``), each
    modelled as a :class:`Hyperedge` with two binary edges.
    """
    bars = {i: v * scale for i, v in TROTBOT_BARS.items()}
    frame = (TROTBOT_FRAME_OFFSET[0] * scale, TROTBOT_FRAME_OFFSET[1] * scale)
    coords = _trotbot_positions(bars, initial_crank_angle, frame)

    hg = HypergraphLinkage(name=name)
    hg.add_node(Node("j0", role=NodeRole.GROUND, name="crank axle"))
    hg.add_node(Node("j3", role=NodeRole.GROUND, name="frame"))
    hg.add_node(Node("j1", role=NodeRole.DRIVER, name="crank end"))
    for nid in ("j2", "j4", "j5", "j6", "j8", "j9"):
        hg.add_node(Node(nid, role=NodeRole.DRIVEN, name=nid))
    hg.add_node(Node("j7", role=NodeRole.DRIVEN, name="foot"))

    # Distances between the collinear ternaries' outer endpoints.
    d_3_4 = bars[2] + bars[3]   # 3-2-4 collinear
    d_5_6 = bars[4] + bars[5]   # 5-4-6 collinear
    d_1_9 = bars[1] + bars[10]  # 1-2-9 collinear

    edges: list[tuple[str, str, str, float]] = [
        ("j0_j1", "j0", "j1", bars[0]),
        ("j1_j2", "j1", "j2", bars[1]),
        ("j2_j3", "j2", "j3", bars[2]),
        ("j2_j4", "j2", "j4", bars[3]),
        ("j3_j4", "j3", "j4", d_3_4),
        ("j1_j5", "j1", "j5", bars[6]),
        ("j4_j5", "j4", "j5", bars[4]),
        ("j4_j6", "j4", "j6", bars[5]),
        ("j5_j6", "j5", "j6", d_5_6),
        ("j2_j9", "j2", "j9", bars[10]),
        ("j1_j9", "j1", "j9", d_1_9),
        ("j1_j8", "j1", "j8", bars[7]),
        ("j2_j8", "j2", "j8", bars[17]),
        ("j6_j7", "j6", "j7", bars[9]),
        ("j8_j7", "j8", "j7", bars[8]),
    ]
    edge_distances: dict[str, float] = {}
    for eid, src, tgt, length in edges:
        hg.add_edge(Edge(eid, src, tgt))
        edge_distances[eid] = length

    # Collinear triples expressed as rigid hyperedges.
    hg.add_hyperedge(Hyperedge("ternary_3_2_4", nodes=("j3", "j2", "j4")))
    hg.add_hyperedge(Hyperedge("ternary_5_4_6", nodes=("j5", "j4", "j6")))
    hg.add_hyperedge(Hyperedge("ternary_1_2_9", nodes=("j1", "j2", "j9")))

    dims = Dimensions(
        node_positions=coords,
        driver_angles={"j1": DriverAngle(angular_velocity=angular_velocity)},
        edge_distances=edge_distances,
    )
    return hg, dims


__all__ = [
    "GHASSAEI_DIMENSIONS",
    "JANSEN_HOLY_NUMBERS",
    "KLANN_PATENT_DIMENSIONS",
    "TROTBOT_BARS",
    "TROTBOT_FRAME_OFFSET",
    "build_chebyshev",
    "build_ghassaei",
    "build_jansen",
    "build_klann",
    "build_strider",
    "build_trotbot",
]
