#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Dynamic linkage: pymunk physics wrapper for hypergraph-based linkages.

Provides ``DynamicLinkage`` which converts a ``HypergraphLinkage`` + ``Dimensions``
into pymunk rigid bodies and constraints for physics simulation.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

import pymunk as pm

from pylinkage.dimensions import Dimensions
from pylinkage.hypergraph import HypergraphLinkage, NodeRole

from .hypergraph_physics import (
    MotorRates,
    PhysicsMapping,
    create_bodies_from_hypergraph,
    get_node_world_position,
)

if TYPE_CHECKING:
    from .walker import Walker

#: Collision filter for ground / road segments.
#:
#: Category bit 2 (``0x4``).  Mask ``0x1`` means the ground only
#: collides with foot edges (category ``0x1``), ignoring non-foot
#: linkage parts (category ``0x2``).  When no foot edges are specified
#: (legacy mode) every linkage shape stays in pymunk's default collision
#: regime and this filter is unused.
GROUND_FILTER = pm.ShapeFilter(categories=0x4, mask=0x1)


class NodeProxy:
    """Lightweight proxy for reading a node's physics position.

    Provides ``.x``, ``.y``, ``.coord()``, ``.name``, ``.role``, and
    ``.reload()`` for compatibility with physics engine and visualizer code.
    """

    __slots__ = ['_node_id', '_role', '_mapping', 'name', '_x', '_y']

    def __init__(
        self,
        node_id: str,
        role: NodeRole,
        mapping: PhysicsMapping,
        name: str | None = None,
    ) -> None:
        self._node_id = node_id
        self._role = role
        self._mapping = mapping
        self.name = name or node_id
        # Initialize from current physics state
        pos = get_node_world_position(node_id, mapping)
        self._x = float(pos.x)
        self._y = float(pos.y)

    @property
    def role(self) -> NodeRole:
        return self._role

    @property
    def x(self) -> float:
        return self._x

    @property
    def y(self) -> float:
        return self._y

    def coord(self) -> tuple[float, float]:
        return (self._x, self._y)

    def reload(self) -> None:
        """Update cached coordinates from physics body positions."""
        pos = get_node_world_position(self._node_id, self._mapping)
        self._x = float(pos.x)
        self._y = float(pos.y)


class DynamicLinkage:
    """Physics simulation wrapper for a hypergraph linkage.

    Converts a ``HypergraphLinkage`` + ``Dimensions`` into pymunk rigid
    bodies and constraints. Each edge becomes a rigid body, and nodes
    where multiple bodies meet get PivotJoint constraints.

    Attributes
    ----------
    body : pm.Body
        The frame/chassis body (load).
    rigidbodies : list[pm.Body]
        All physics bodies including the load body.
    joints : tuple[NodeProxy, ...]
        Proxy objects for reading node positions from physics.
    mass : float
        Total mass of all bodies.
    space : pm.Space
        The pymunk simulation space.
    physics_mapping : PhysicsMapping
        Mapping from hypergraph elements to pymunk objects.
    """

    __slots__ = [
        'body', 'rigidbodies', 'joints', 'mass', 'space',
        'density', '_thickness', 'filter', '_non_foot_filter',
        'physics_mapping',
        '_hypergraph', '_dimensions', 'height', 'mechanical_energy',
        'name',
    ]

    body: pm.Body
    rigidbodies: list[pm.Body]
    joints: tuple[NodeProxy, ...]
    mass: float
    space: pm.Space
    density: float
    _thickness: float
    filter: pm.ShapeFilter
    physics_mapping: PhysicsMapping
    _hypergraph: HypergraphLinkage
    _dimensions: Dimensions
    height: float
    mechanical_energy: float
    name: str

    def __init__(
        self,
        topology: HypergraphLinkage,
        dimensions: Dimensions,
        space: pm.Space,
        motor_rates: MotorRates | None = None,
        density: float = 1,
        load: float = 0,
        thickness: float = 0.1,
        name: str = "",
        foot_edge_ids: set[str] | None = None,
    ) -> None:
        """Create a DynamicLinkage from a hypergraph.

        Parameters
        ----------
        topology : HypergraphLinkage
            The mechanism topology.
        dimensions : Dimensions
            Geometric data (positions, distances, driver angles).
        space : pm.Space
            Pymunk simulation space.
        motor_rates : MotorRates | None
            Motor angular velocities. Single float for all drivers, or
            dict mapping driver node IDs to individual rates (multi-DOF).
        density : float
            Density for body mass calculation.
        load : float
            Mass of the load/chassis body.
        thickness : float
            Radius of segment shapes.
        name : str
            Human-readable name.
        foot_edge_ids : set[str] | None
            Edge IDs whose segments should collide with the ground.
            All other edges (and the load body) will pass through the
            ground.  *None* disables selective collision (all edges
            collide with the ground, legacy behaviour).
        """
        self._hypergraph = topology
        self._dimensions = dimensions
        self.density = density
        self._thickness = thickness
        self.space = space
        self.name = name

        # --- collision filters ---
        # When foot_edge_ids is given we use pymunk collision categories:
        #   bit 0 (0x1): foot edges
        #   bit 1 (0x2): non-foot linkage edges / load body
        #   bit 2 (0x4): ground / road  (assigned in World)
        # Foot filter:     category=foot, collides with ground + linkage
        # Non-foot filter: category=non-foot, collides with linkage only
        if foot_edge_ids is not None:
            self.filter = pm.ShapeFilter(
                group=1, categories=0x1, mask=pm.ShapeFilter.ALL_MASKS(),
            )
            self._non_foot_filter: pm.ShapeFilter | None = pm.ShapeFilter(
                group=1, categories=0x2, mask=0x2 | 0x1,
            )
        else:
            self.filter = pm.ShapeFilter(group=1)
            self._non_foot_filter = None

        # Build load body (the frame/chassis)
        first_ground = next(iter(topology.ground_nodes()), None)
        if first_ground is not None:
            pos = dimensions.get_node_position(first_ground.id)
            init_pos = pm.Vec2d(pos[0], pos[1]) if pos else pm.Vec2d(0, 0)
        else:
            init_pos = pm.Vec2d(0, 0)

        self.body = self._build_load(init_pos, load, foot_edge_ids is not None)

        # Generate physics bodies from hypergraph
        self.physics_mapping = create_bodies_from_hypergraph(
            topology,
            dimensions,
            space,
            self.body,
            density,
            thickness,
            self.filter,
            motor_rates=motor_rates,
            foot_edge_ids=foot_edge_ids,
            non_foot_filter=self._non_foot_filter,
        )

        # Create NodeProxy wrappers for position tracking
        proxies: list[NodeProxy] = []
        for node_id, node in topology.nodes.items():
            proxy = NodeProxy(
                node_id=node_id,
                role=node.role,
                mapping=self.physics_mapping,
                name=node.name,
            )
            proxies.append(proxy)
        self.joints = tuple(proxies)

        # Collect all rigid bodies
        self.rigidbodies = [self.body]
        for body in self.physics_mapping.edge_to_body.values():
            if body is not self.body and body not in self.rigidbodies:
                self.rigidbodies.append(body)

        self.mass = sum(b.mass for b in self.rigidbodies)
        self.height = 0.0
        self.mechanical_energy = 0.0

    def _build_load(
        self,
        position: pm.Vec2d,
        load_mass: float,
        use_non_foot: bool = False,
    ) -> pm.Body:
        """Create the load/chassis body."""
        load = pm.Body()
        load.position = position
        # The load body is never a foot — use non-foot filter when
        # selective collision is enabled so the chassis doesn't scrape
        # the ground.
        filt = self._non_foot_filter if use_non_foot else self.filter
        if filt is None:
            filt = self.filter
        vertices = (-.5, -.5), (-.5, .5), (.5, .5), (.5, -.5)
        segs: list[pm.Segment] = []
        for i, vertex in enumerate(vertices):
            segment = pm.Segment(
                load, vertex, vertices[(i + 1) % len(vertices)],
                self._thickness,
            )
            segment.density = self.density
            segment.filter = filt
            segs.append(segment)
        self.space.add(load, *segs)

        if load_mass > 0:
            load.mass = load_mass
            load.moment = load_mass * (1.0 + 1.0) / 12.0

        return load

    def get_all_positions(self) -> dict[str, tuple[float, float]]:
        """Get current world positions of all nodes."""
        return {
            proxy._node_id: proxy.coord()
            for proxy in self.joints
        }


def convert_to_dynamic_linkage(
    source: Walker | object,
    space: pm.Space,
    density: float = 1,
    load: float = 1,
    motor_rates: MotorRates | None = None,
) -> DynamicLinkage:
    """Convert a Walker (or legacy Linkage) to a DynamicLinkage.

    Parameters
    ----------
    source : Walker or Linkage
        The mechanism to convert.
    space : pm.Space
        Pymunk simulation space.
    density : float
        Density for body mass calculation.
    load : float
        Mass of the load/chassis.
    motor_rates : MotorRates | None
        Motor angular velocities. If None and source is a Walker,
        uses ``source.motor_rates``.

    Returns
    -------
    DynamicLinkage
    """
    # Import here to avoid circular imports
    from .walker import Walker

    if isinstance(source, Walker):
        if motor_rates is None:
            motor_rates = source.motor_rates
        foot_edges = source.get_foot_edges()
        return DynamicLinkage(
            topology=source.topology,
            dimensions=source.dimensions,
            space=space,
            motor_rates=motor_rates,
            density=density,
            load=load,
            name=source.name,
            foot_edge_ids=set(foot_edges) if foot_edges else None,
        )

    # Legacy Linkage path
    from pylinkage.hypergraph import from_linkage

    hg, dims = from_linkage(source)  # type: ignore[arg-type]
    if motor_rates is None and hasattr(source, 'motor_rate'):
        motor_rates = source.motor_rate
    return DynamicLinkage(
        topology=hg,
        dimensions=dims,
        space=space,
        motor_rates=motor_rates,
        density=density,
        load=load,
        name=getattr(source, 'name', '') or '',
    )
