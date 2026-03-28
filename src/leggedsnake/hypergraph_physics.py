#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Hypergraph-based physics body generation.

This module provides functions to convert a pylinkage hypergraph representation
into pymunk physics bodies and constraints. Each edge in the hypergraph becomes
a rigid body (bar), and nodes where multiple bodies meet get PivotJoint constraints.
"""
from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import pymunk as pm
from pylinkage.dimensions import Dimensions
from pylinkage.hypergraph import HypergraphLinkage, NodeRole


@dataclass
class PhysicsMapping:
    """Maps hypergraph elements to pymunk physics objects.

    Attributes
    ----------
    edge_to_body : dict[str, pm.Body]
        Maps edge IDs to their corresponding pymunk bodies.
    node_to_bodies : dict[str, set[pm.Body]]
        Maps node IDs to all bodies that share that joint position.
    node_to_anchors : dict[str, dict[pm.Body, pm.Vec2d]]
        Maps node IDs to local anchor positions on each body.
    constraints : list[pm.PivotJoint]
        All pivot joint constraints created.
    motors : list[pm.SimpleMotor]
        All motor constraints for driver nodes.
    motor_pivots : list[pm.PivotJoint]
        Pivot joints associated with motors (currently unused, kept for compatibility).
    """

    edge_to_body: dict[str, pm.Body] = field(default_factory=dict)
    node_to_bodies: dict[str, set[pm.Body]] = field(default_factory=dict)
    node_to_anchors: dict[str, dict[pm.Body, pm.Vec2d]] = field(default_factory=dict)
    constraints: list[pm.PivotJoint] = field(default_factory=list)
    motors: list[pm.SimpleMotor] = field(default_factory=list)
    motor_pivots: list[pm.PivotJoint] = field(default_factory=list)


def _get_node_position(node_id: str, dimensions: Dimensions) -> pm.Vec2d:
    """Get node position as Vec2d from Dimensions, defaulting to (0, 0)."""
    pos = dimensions.get_node_position(node_id)
    if pos is None:
        return pm.Vec2d(0.0, 0.0)
    return pm.Vec2d(pos[0], pos[1])


def _create_segment(
    body: pm.Body,
    pos_a: pm.Vec2d,
    pos_b: pm.Vec2d,
    thickness: float,
    density: float,
    shape_filter: pm.ShapeFilter | None,
) -> pm.Segment:
    """Create a pymunk segment on a body between two world positions."""
    local_a = body.world_to_local(pos_a)
    local_b = body.world_to_local(pos_b)
    seg = pm.Segment(body, local_a, local_b, thickness)
    seg.density = density
    if shape_filter is not None:
        seg.filter = shape_filter
    return seg


def _is_ground_node(hg: HypergraphLinkage, node_id: str) -> bool:
    """Check if a node is a ground node."""
    return bool(hg.nodes[node_id].role == NodeRole.GROUND)


def _is_driver_node(hg: HypergraphLinkage, node_id: str) -> bool:
    """Check if a node is a driver (motor) node."""
    return bool(hg.nodes[node_id].role == NodeRole.DRIVER)


def _find_effective_ground_nodes(
    hg: HypergraphLinkage,
    joints: tuple[Any, ...] | None = None,
) -> set[str]:
    """Find all nodes that should be treated as ground (fixed to frame).

    A node is an effective ground node if:
    1. It is explicitly marked as GROUND, OR
    2. It is a Fixed joint where BOTH parent joints (joint0, joint1) are ground, OR
    3. It is not a DRIVER and ALL its neighbors (through edges) are effective ground nodes

    This handles cases like Fixed joints that connect two ground points -
    they should also be fixed to the frame body.

    Parameters
    ----------
    hg : HypergraphLinkage
        The hypergraph representation of the linkage.
    joints : tuple[Joint, ...] | None
        Optional original linkage joints for detecting Fixed joints with
        static parents. If provided, enables detection of ground-constrained
        Fixed joints.

    Returns
    -------
    set[str]
        Set of node IDs that should be treated as ground nodes.
    """
    # Import legacy classes with deprecation suppressed
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore", category=DeprecationWarning, message=r"pylinkage\.joints"
        )
        from pylinkage import Static, Fixed

    # Start with explicit ground nodes
    effective_ground: set[str] = {node.id for node in hg.ground_nodes()}

    # If we have the original joints, detect Fixed joints with both parents being ground
    if joints is not None:
        # Build a map from joint name to joint object
        joint_by_name: dict[str, Any] = {j.name: j for j in joints if j.name}

        # Iteratively find Fixed joints where both parents are effective ground
        changed = True
        while changed:
            changed = False
            for joint in joints:
                if not isinstance(joint, Fixed):
                    continue
                if joint.name in effective_ground:
                    continue

                # Check if both parent joints are effective ground
                j0_name = joint.joint0.name if hasattr(joint, 'joint0') and joint.joint0 else None
                j1_name = joint.joint1.name if hasattr(joint, 'joint1') and joint.joint1 else None

                j0_is_ground = (
                    j0_name in effective_ground or
                    (hasattr(joint, 'joint0') and isinstance(joint.joint0, Static))
                )
                j1_is_ground = (
                    j1_name in effective_ground or
                    (hasattr(joint, 'joint1') and isinstance(joint.joint1, Static))
                )

                if j0_is_ground and j1_is_ground:
                    effective_ground.add(joint.name)
                    changed = True

    # Build adjacency map: node_id -> set of neighbor node_ids
    neighbors: dict[str, set[str]] = {node_id: set() for node_id in hg.nodes}
    for edge in hg.edges.values():
        neighbors[edge.source].add(edge.target)
        neighbors[edge.target].add(edge.source)

    # Iteratively find nodes where all neighbors are ground
    # (excluding driver nodes which must remain movable)
    changed = True
    while changed:
        changed = False
        for node_id, node in hg.nodes.items():
            if node_id in effective_ground:
                continue
            # Driver nodes (cranks) should never be ground
            if node.role == NodeRole.DRIVER:
                continue
            # Check if all neighbors are effective ground
            node_neighbors = neighbors[node_id]
            if node_neighbors and all(n in effective_ground for n in node_neighbors):
                effective_ground.add(node_id)
                changed = True

    return effective_ground


def _find_rigid_triangles(
    hg: HypergraphLinkage,
    joints: tuple[Any, ...] | None,
) -> list[set[str]]:
    """Find groups of edges that form rigid triangles.

    Fixed joints with two parents create rigid triangles - all edges
    in such triangles should be on the same rigid body.

    Returns a list of edge ID sets, where each set contains edges
    that should be merged into a single body.
    """
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore", category=DeprecationWarning, message=r"pylinkage\.joints"
        )
        from pylinkage import Fixed

    if joints is None:
        return []

    triangles = []

    # Build edge lookup: (node1, node2) -> edge_id
    edge_lookup: dict[tuple[str, str], str] = {}
    for edge_id, edge in hg.edges.items():
        edge_lookup[(edge.source, edge.target)] = edge_id
        edge_lookup[(edge.target, edge.source)] = edge_id

    for joint in joints:
        if not isinstance(joint, Fixed):
            continue

        fixed_id = joint.name
        if fixed_id is None:
            continue

        j0_name = joint.joint0.name if hasattr(joint, 'joint0') and joint.joint0 else None
        j1_name = joint.joint1.name if hasattr(joint, 'joint1') and joint.joint1 else None

        if j0_name is None or j1_name is None:
            continue

        # Find edges forming the triangle: F-j0, F-j1, and j0-j1
        edge_f_j0 = edge_lookup.get((fixed_id, j0_name))
        edge_f_j1 = edge_lookup.get((fixed_id, j1_name))
        edge_j0_j1 = edge_lookup.get((j0_name, j1_name))

        # Collect all edges that form this triangle
        triangle_edges: set[str] = set()
        if edge_f_j0:
            triangle_edges.add(edge_f_j0)
        if edge_f_j1:
            triangle_edges.add(edge_f_j1)
        if edge_j0_j1:
            triangle_edges.add(edge_j0_j1)

        if len(triangle_edges) >= 2:  # At least 2 edges to form a rigid structure
            triangles.append(triangle_edges)

    # Merge overlapping triangles (edges shared between triangles)
    merged = _merge_overlapping_sets(triangles)
    return merged


def _merge_overlapping_sets(sets: list[set[str]]) -> list[set[str]]:
    """Merge sets that have overlapping elements."""
    if not sets:
        return []

    result: list[set[str]] = []
    for s in sets:
        merged = False
        for i, existing in enumerate(result):
            if s & existing:  # If there's overlap
                result[i] = existing | s
                merged = True
                break
        if not merged:
            result.append(s.copy())

    # Keep merging until no more changes
    changed = True
    while changed:
        changed = False
        new_result: list[set[str]] = []
        for s in result:
            merged = False
            for i, existing in enumerate(new_result):
                if s & existing:
                    new_result[i] = existing | s
                    merged = True
                    changed = True
                    break
            if not merged:
                new_result.append(s)
        result = new_result

    return result


def create_bodies_from_hypergraph(
    hg: HypergraphLinkage,
    dimensions: Dimensions,
    space: pm.Space,
    load_body: pm.Body,
    density: float,
    thickness: float,
    shape_filter: pm.ShapeFilter | None,
    joints: tuple[Any, ...] | None = None,
    motor_rate: float | None = None,
) -> PhysicsMapping:
    """Create pymunk bodies from a hypergraph representation.

    Each edge in the hypergraph becomes a rigid body with a segment shape,
    except for edges that form rigid triangles (from Fixed joints) which
    are merged into single bodies.

    Parameters
    ----------
    hg : HypergraphLinkage
        The hypergraph representation of the linkage.
    dimensions : Dimensions
        Geometric data (positions, distances, angles) for the hypergraph.
    space : pm.Space
        The pymunk space to add bodies to.
    load_body : pm.Body
        The frame/chassis body that ground nodes attach to.
    density : float
        Density for body mass calculation.
    thickness : float
        Radius of segment shapes.
    shape_filter : pm.ShapeFilter | None
        Collision filter for segments (typically to prevent self-collision).
    joints : tuple[Joint, ...] | None
        Optional original linkage joints for detecting Fixed joints with
        static parents. Enables proper detection of frame-fixed nodes.
    motor_rate : float | None
        Motor angular velocity in rad/s. If None, falls back to kinematic
        angle from the Crank joints (not recommended).

    Returns
    -------
    PhysicsMapping
        Mapping of hypergraph elements to physics objects.
    """
    mapping = PhysicsMapping()

    # Find all nodes that should be treated as ground (fixed to frame)
    effective_ground = _find_effective_ground_nodes(hg, joints)

    # Initialize effective ground nodes with load_body
    for node_id in effective_ground:
        mapping.node_to_bodies.setdefault(node_id, set()).add(load_body)
        pos = _get_node_position(node_id, dimensions)
        mapping.node_to_anchors.setdefault(node_id, {})[load_body] = (
            load_body.world_to_local(pos)
        )

    # Find rigid triangles that need to be merged
    triangles = _find_rigid_triangles(hg, joints)

    # Build map: edge_id -> triangle index (or None if not in triangle)
    edge_to_triangle: dict[str, int] = {}
    for i, triangle in enumerate(triangles):
        for edge_id in triangle:
            edge_to_triangle[edge_id] = i

    # Track which triangles have been created
    triangle_bodies: dict[int, pm.Body] = {}

    # Create bodies for edges
    for edge_id, edge in hg.edges.items():
        source_is_ground = edge.source in effective_ground
        target_is_ground = edge.target in effective_ground

        source_pos = _get_node_position(edge.source, dimensions)
        target_pos = _get_node_position(edge.target, dimensions)

        if source_is_ground and target_is_ground:
            # Both endpoints are ground - add segment to load_body
            seg = _create_segment(
                load_body, source_pos, target_pos, thickness, density, shape_filter
            )
            space.add(seg)
            mapping.edge_to_body[edge_id] = load_body
        elif edge_id in edge_to_triangle:
            # This edge is part of a rigid triangle
            tri_idx = edge_to_triangle[edge_id]

            if tri_idx not in triangle_bodies:
                # Create a new body for this triangle
                # Calculate center from all nodes in the triangle
                triangle_edges = triangles[tri_idx]
                triangle_nodes: set[str] = set()
                for te_id in triangle_edges:
                    te = hg.edges[te_id]
                    triangle_nodes.add(te.source)
                    triangle_nodes.add(te.target)

                # Calculate center of mass
                positions = [_get_node_position(n, dimensions) for n in triangle_nodes]
                center = sum(positions, pm.Vec2d(0, 0)) / len(positions)

                body = pm.Body()
                body.mass = 1
                body.position = center
                space.add(body)
                triangle_bodies[tri_idx] = body

            body = triangle_bodies[tri_idx]

            # Add segment to the triangle body
            seg = _create_segment(
                body, source_pos, target_pos, thickness, density, shape_filter
            )
            space.add(seg)
            mapping.edge_to_body[edge_id] = body

            # Track node-to-body mapping
            mapping.node_to_bodies.setdefault(edge.source, set()).add(body)
            mapping.node_to_bodies.setdefault(edge.target, set()).add(body)

            # Store local anchor positions
            mapping.node_to_anchors.setdefault(edge.source, {})[body] = (
                body.world_to_local(source_pos)
            )
            mapping.node_to_anchors.setdefault(edge.target, {})[body] = (
                body.world_to_local(target_pos)
            )
        else:
            # Create a new body for this bar
            body = pm.Body()
            body.mass = 1
            body.position = (source_pos + target_pos) / 2

            seg = _create_segment(
                body, source_pos, target_pos, thickness, density, shape_filter
            )
            space.add(body, seg)
            mapping.edge_to_body[edge_id] = body

            # Track node-to-body mapping
            mapping.node_to_bodies.setdefault(edge.source, set()).add(body)
            mapping.node_to_bodies.setdefault(edge.target, set()).add(body)

            # Store local anchor positions
            mapping.node_to_anchors.setdefault(edge.source, {})[body] = (
                body.world_to_local(source_pos)
            )
            mapping.node_to_anchors.setdefault(edge.target, {})[body] = (
                body.world_to_local(target_pos)
            )

    # Create pivot constraints at shared nodes
    _create_pivot_constraints(hg, dimensions, mapping, space, effective_ground, load_body)

    # Create motors for driver nodes
    _create_motor_constraints(hg, dimensions, mapping, space, load_body, motor_rate)

    return mapping


def _create_pivot_constraints(
    hg: HypergraphLinkage,
    dimensions: Dimensions,
    mapping: PhysicsMapping,
    space: pm.Space,
    effective_ground: set[str],
    load_body: pm.Body,
) -> None:
    """Create PivotJoint constraints at nodes shared by multiple bodies.

    Uses chain topology: for N bodies at a node, creates N-1 constraints
    by connecting each body to the first one (preferring load_body as anchor).
    This avoids over-constraining closed kinematic loops.
    """
    for node_id, bodies in mapping.node_to_bodies.items():
        if len(bodies) < 2:
            continue

        node_pos = _get_node_position(node_id, dimensions)

        # Use chain topology to avoid over-constraining
        body_list = list(bodies)

        # Prefer load_body as the anchor for stability
        if load_body in body_list:
            body_list.remove(load_body)
            body_list.insert(0, load_body)

        anchor_body = body_list[0]
        for other_body in body_list[1:]:
            # Skip if both bodies are the load_body
            if anchor_body is load_body and other_body is load_body:
                continue
            pivot = pm.PivotJoint(anchor_body, other_body, node_pos)
            pivot.collide_bodies = False
            space.add(pivot)
            mapping.constraints.append(pivot)


def _create_motor_constraints(
    hg: HypergraphLinkage,
    dimensions: Dimensions,
    mapping: PhysicsMapping,
    space: pm.Space,
    load_body: pm.Body,
    motor_rate: float | None = None,
) -> None:
    """Create SimpleMotor constraints for driver (crank) nodes.

    Note: The pivot constraint at the ground connection point is already
    created by _create_pivot_constraints. We only need to add the motor here.

    Parameters
    ----------
    hg : HypergraphLinkage
        The hypergraph representation.
    dimensions : Dimensions
        Geometric data for the linkage.
    mapping : PhysicsMapping
        The physics mapping to update.
    space : pm.Space
        The pymunk space.
    load_body : pm.Body
        The frame/chassis body.
    motor_rate : float | None
        Motor angular velocity in rad/s. If None, falls back to driver
        angular velocity from dimensions (not recommended).
    """
    for driver_node in hg.driver_nodes():
        driver_id = driver_node.id

        # Find the edge connecting driver to ground - this is the crank arm
        crank_edge_id = None

        for edge_id, edge in hg.edges.items():
            if edge.source == driver_id and _is_ground_node(hg, edge.target):
                crank_edge_id = edge_id
                break
            elif edge.target == driver_id and _is_ground_node(hg, edge.source):
                crank_edge_id = edge_id
                break

        if crank_edge_id is None:
            # No ground connection found for this driver
            continue

        # Get the crank body from the edge mapping
        driver_body = mapping.edge_to_body.get(crank_edge_id)

        if driver_body is None or driver_body is load_body:
            # Crank edge might be entirely on load_body (both ends ground)
            continue

        # Create motor - use provided motor_rate, or fall back to driver angle
        # Note: driver angular_velocity is the kinematic step size, not an ideal motor rate
        if motor_rate is not None:
            rate = motor_rate
        else:
            driver_angle = dimensions.get_driver_angle(driver_id)
            rate = driver_angle.angular_velocity if driver_angle is not None else 0.0
        motor = pm.SimpleMotor(driver_body, load_body, rate)
        space.add(motor)
        mapping.motors.append(motor)


def get_node_world_position(
    node_id: str,
    mapping: PhysicsMapping,
) -> pm.Vec2d:
    """Get the current world position of a node from physics bodies.

    Parameters
    ----------
    node_id : str
        The node ID to query.
    mapping : PhysicsMapping
        The physics mapping containing body references.

    Returns
    -------
    pm.Vec2d
        The world position of the node.
    """
    bodies = mapping.node_to_bodies.get(node_id)
    if not bodies:
        return pm.Vec2d(0, 0)

    # Use the first body's anchor to get world position
    body = next(iter(bodies))
    anchors = mapping.node_to_anchors.get(node_id, {})
    local_anchor = anchors.get(body, pm.Vec2d(0, 0))
    return body.local_to_world(local_anchor)
