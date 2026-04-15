#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Canonical geometries for classical walking linkages.

Internal module: shared geometry solvers and topology/dimension builders
for Theo Jansen, Klann, and Chebyshev Lambda mechanisms. Exposed as
class methods on :class:`leggedsnake.walker.Walker`.
"""
from __future__ import annotations

from math import cos, sin, sqrt

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
) -> tuple[HypergraphLinkage, Dimensions]:
    """Build the (topology, dimensions) of a single Theo Jansen leg.

    The Jansen linkage is an 8-bar planar mechanism. All mobile joints
    are Revolute (circle-circle intersection) — no ternary links.
    """
    h = {k: v * scale for k, v in JANSEN_HOLY_NUMBERS.items()}
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


__all__ = [
    "JANSEN_HOLY_NUMBERS",
    "KLANN_PATENT_DIMENSIONS",
    "build_chebyshev",
    "build_jansen",
    "build_klann",
]
