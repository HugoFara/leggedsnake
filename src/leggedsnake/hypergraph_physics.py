#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Hypergraph-based physics body generation.

This module provides functions to convert a pylinkage hypergraph representation
into pymunk physics bodies and constraints. Each edge in the hypergraph becomes
a rigid body (bar), and nodes where multiple bodies meet get PivotJoint constraints.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from itertools import combinations
from typing import TYPE_CHECKING

import pymunk as pm
from pylinkage.hypergraph import HypergraphLinkage, NodeRole

if TYPE_CHECKING:
    from pylinkage.hypergraph import Node


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
        Pivot joints associated with motors.
    """

    edge_to_body: dict[str, pm.Body] = field(default_factory=dict)
    node_to_bodies: dict[str, set[pm.Body]] = field(default_factory=dict)
    node_to_anchors: dict[str, dict[pm.Body, pm.Vec2d]] = field(default_factory=dict)
    constraints: list[pm.PivotJoint] = field(default_factory=list)
    motors: list[pm.SimpleMotor] = field(default_factory=list)
    motor_pivots: list[pm.PivotJoint] = field(default_factory=list)


def _get_node_position(node: Node) -> pm.Vec2d:
    """Get node position as Vec2d, handling None values."""
    x = node.position[0] if node.position[0] is not None else 0.0
    y = node.position[1] if node.position[1] is not None else 0.0
    return pm.Vec2d(x, y)


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


def create_bodies_from_hypergraph(
    hg: HypergraphLinkage,
    space: pm.Space,
    load_body: pm.Body,
    density: float,
    thickness: float,
    shape_filter: pm.ShapeFilter | None,
) -> PhysicsMapping:
    """Create pymunk bodies from a hypergraph representation.

    Each edge in the hypergraph becomes a rigid body with a segment shape.
    Ground-only edges attach segments to the load_body instead.

    Parameters
    ----------
    hg : HypergraphLinkage
        The hypergraph representation of the linkage.
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

    Returns
    -------
    PhysicsMapping
        Mapping of hypergraph elements to physics objects.
    """
    mapping = PhysicsMapping()

    # Initialize ground nodes with load_body
    for node in hg.ground_nodes():
        mapping.node_to_bodies.setdefault(node.id, set()).add(load_body)
        pos = _get_node_position(node)
        mapping.node_to_anchors.setdefault(node.id, {})[load_body] = (
            load_body.world_to_local(pos)
        )

    # Create a body for each edge
    for edge_id, edge in hg.edges.items():
        source_is_ground = _is_ground_node(hg, edge.source)
        target_is_ground = _is_ground_node(hg, edge.target)

        source_pos = _get_node_position(hg.nodes[edge.source])
        target_pos = _get_node_position(hg.nodes[edge.target])

        if source_is_ground and target_is_ground:
            # Both endpoints are ground - add segment to load_body
            seg = _create_segment(
                load_body, source_pos, target_pos, thickness, density, shape_filter
            )
            space.add(seg)
            mapping.edge_to_body[edge_id] = load_body
        else:
            # Create a new body for this bar
            body = pm.Body()
            body.mass = 1  # Will be recalculated by segment density
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
    _create_pivot_constraints(hg, mapping, space)

    # Create motors for driver nodes
    _create_motor_constraints(hg, mapping, space, load_body)

    return mapping


def _create_pivot_constraints(
    hg: HypergraphLinkage,
    mapping: PhysicsMapping,
    space: pm.Space,
) -> None:
    """Create PivotJoint constraints at nodes shared by multiple bodies."""
    for node_id, bodies in mapping.node_to_bodies.items():
        if len(bodies) < 2:
            continue

        # Skip driver nodes - they get motor constraints instead
        if _is_driver_node(hg, node_id):
            continue

        node_pos = _get_node_position(hg.nodes[node_id])

        # Create pivot between each pair of bodies at this node
        body_list = list(bodies)
        for body_a, body_b in combinations(body_list, 2):
            pivot = pm.PivotJoint(body_a, body_b, node_pos)
            pivot.collide_bodies = False
            space.add(pivot)
            mapping.constraints.append(pivot)


def _create_motor_constraints(
    hg: HypergraphLinkage,
    mapping: PhysicsMapping,
    space: pm.Space,
    load_body: pm.Body,
) -> None:
    """Create SimpleMotor constraints for driver (crank) nodes."""
    for driver_node in hg.driver_nodes():
        driver_id = driver_node.id
        driver_pos = _get_node_position(driver_node)

        # Find the body for this driver (should be on an edge from ground)
        driver_bodies = mapping.node_to_bodies.get(driver_id, set())

        # Find the driver body (not the load_body)
        driver_body = None
        for body in driver_bodies:
            if body is not load_body:
                driver_body = body
                break

        if driver_body is None:
            # Driver might be directly on load_body in some edge cases
            continue

        # Find the reference body (ground connection point)
        # Look for ground neighbors of this driver
        ref_body = load_body
        ref_pos = None

        # Find edge connecting driver to ground
        for edge in hg.edges.values():
            if edge.source == driver_id and _is_ground_node(hg, edge.target):
                ref_pos = _get_node_position(hg.nodes[edge.target])
                break
            elif edge.target == driver_id and _is_ground_node(hg, edge.source):
                ref_pos = _get_node_position(hg.nodes[edge.source])
                break

        if ref_pos is None:
            # Fallback: use driver's neighbor that's on load_body
            ref_pos = driver_pos

        # Create pivot joint at the ground connection
        pivot = pm.PivotJoint(ref_body, driver_body, ref_pos)
        pivot.collide_bodies = False
        space.add(pivot)
        mapping.motor_pivots.append(pivot)

        # Create motor - rate comes from node angle attribute
        angle = driver_node.angle if driver_node.angle is not None else 0.0
        motor = pm.SimpleMotor(driver_body, ref_body, angle)
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
