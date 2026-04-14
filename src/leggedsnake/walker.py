#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Hypergraph-native Walker for simulating and optimizing walking linkages.

A Walker stores a mechanism as a HypergraphLinkage (topology) plus Dimensions
(geometry). It delegates kinematic simulation to pylinkage's Mechanism class
via ``to_mechanism()``, naturally supporting multi-DOF mechanisms through
multiple DRIVER nodes with independent angular velocities.
"""
from __future__ import annotations

from collections.abc import Generator
from math import tau
from typing import TYPE_CHECKING

from pylinkage.dimensions import Dimensions, DriverAngle
from pylinkage.hypergraph import HypergraphLinkage, NodeRole, to_mechanism
from pylinkage.hypergraph.core import Edge, Hyperedge, Node

if TYPE_CHECKING:
    from pylinkage.hypergraph import HierarchicalLinkage
    from pylinkage.mechanism import Mechanism
    from pylinkage.topology.catalog import CatalogEntry


class Walker:
    """A walking mechanism represented as a hypergraph.

    The Walker holds a ``HypergraphLinkage`` (topology) and ``Dimensions``
    (geometry). It supports multi-DOF mechanisms: each DRIVER node can have
    an independent angular velocity stored in ``Dimensions.driver_angles``.

    Kinematic simulation is delegated to pylinkage's ``Mechanism`` class
    via ``to_mechanism()``. Physics simulation is handled by converting
    to a ``DynamicLinkage`` (see ``physicsengine.World.add_linkage``).

    Attributes
    ----------
    topology : HypergraphLinkage
        The mechanism topology (nodes, edges, hyperedges).
    dimensions : Dimensions
        Geometric data (node positions, edge distances, driver angles).
    name : str
        Human-readable name.
    motor_rates : dict[str, float] | float
        Motor angular velocities for dynamic (physics) simulation.
        Single float applies to all drivers. Dict maps driver node IDs
        to individual rates. Default -4.0 rad/s (clockwise).
    """

    topology: HypergraphLinkage
    dimensions: Dimensions
    name: str
    motor_rates: dict[str, float] | float
    _mechanism: Mechanism | None
    _foot_edge_ids: list[str] | None

    def __init__(
        self,
        topology: HypergraphLinkage,
        dimensions: Dimensions,
        name: str = "",
        motor_rates: dict[str, float] | float = -4.0,
        foot_edge_ids: list[str] | None = None,
    ) -> None:
        self.topology = topology
        self.dimensions = dimensions
        self.name = name or topology.name
        self.motor_rates = motor_rates
        self._mechanism = None
        self._foot_edge_ids = foot_edge_ids

    # --- Factory class methods ---

    @classmethod
    def from_catalog(
        cls,
        entry: CatalogEntry,
        dimensions: Dimensions,
        motor_rates: dict[str, float] | float = -4.0,
    ) -> Walker:
        """Create a Walker from a topology catalog entry.

        The catalog provides topology only (nodes, edges, roles).
        Dimensions (positions, distances, driver angles) must be supplied.

        Parameters
        ----------
        entry : CatalogEntry
            A topology from ``pylinkage.topology.load_catalog()``.
        dimensions : Dimensions
            Geometric data for the topology's nodes and edges.
        motor_rates : float | dict[str, float]
            Motor angular velocities for dynamic simulation.
        """
        topology = entry.to_graph()
        return cls(topology, dimensions, name=entry.name, motor_rates=motor_rates)

    @classmethod
    def from_hierarchy(
        cls,
        hierarchy: HierarchicalLinkage,
        dimensions: Dimensions,
        motor_rates: dict[str, float] | float = -4.0,
    ) -> Walker:
        """Create a Walker from a hierarchical linkage composition.

        Flattens the hierarchy (merging shared ports) into a single
        ``HypergraphLinkage`` suitable for simulation.

        Parameters
        ----------
        hierarchy : HierarchicalLinkage
            Composed mechanism with component instances and connections.
        dimensions : Dimensions
            Geometric data for the flattened topology's nodes and edges.
        motor_rates : float | dict[str, float]
            Motor angular velocities for dynamic simulation.
        """
        flat = hierarchy.flatten()
        return cls(flat, dimensions, name=hierarchy.name, motor_rates=motor_rates)

    def _invalidate_cache(self) -> None:
        """Invalidate cached Mechanism after topology/dimension changes."""
        self._mechanism = None

    def to_mechanism(self) -> Mechanism:
        """Convert to a pylinkage Mechanism for kinematic simulation.

        The result is cached and invalidated when topology or dimensions change.
        """
        if self._mechanism is None:
            self._mechanism = to_mechanism(self.topology, self.dimensions)
        return self._mechanism

    def step(
        self,
        iterations: int | None = None,
        dt: float = 1.0,
        skip_unbuildable: bool = False,
    ) -> Generator[tuple[tuple[float, float] | tuple[float | None, float | None], ...], None, None]:
        """Simulate one full rotation of the mechanism.

        Delegates to ``Mechanism.step()``. Each driver advances independently.

        Parameters
        ----------
        iterations : int | None
            Number of simulation steps. If None, uses one full rotation period.
        dt : float
            Time step multiplier.
        skip_unbuildable : bool
            If True, yield ``(None, None)`` tuples for iterations where the
            mechanism cannot be assembled, instead of raising
            ``UnbuildableError``. Drivers keep advancing so the trajectory
            resumes on the buildable side of dead zones. Mirrors
            ``pylinkage.linkage.Linkage.step``'s parameter of the same name.

        Yields
        ------
        tuple of (x, y) coordinate tuples
            Joint positions at each step.
        """
        from pylinkage import UnbuildableError

        mechanism = self.to_mechanism()
        if iterations is None:
            iterations = mechanism.get_rotation_period()

        if skip_unbuildable:
            none_positions = tuple(
                (None, None) for _ in mechanism.joints
            )
            for _ in range(iterations):
                try:
                    mechanism._step_once(dt)
                except UnbuildableError:
                    yield none_positions
                else:
                    yield tuple(j.coord() for j in mechanism.joints)
        else:
            for _ in range(iterations):
                mechanism._step_once(dt)
                yield tuple(j.coord() for j in mechanism.joints)

    def get_rotation_period(self) -> int:
        """Number of steps for one full rotation cycle."""
        return self.to_mechanism().get_rotation_period()

    # --- Topological analysis (pylinkage 0.9 adoption) ---

    @property
    def dof(self) -> int:
        """Mobility of the underlying hypergraph via Grübler's formula.

        Returns 1 for a well-formed single-input walking mechanism,
        0 for a rigid structure, negative for overconstrained, >1 for
        under-constrained (needs more inputs).

        Delegates to :func:`pylinkage.topology.compute_dof` on
        ``self.topology``. Intended for fast pre-flight screening of
        candidates during optimization — reject DOF != 1 without
        paying the cost of building the mechanism and stepping it.
        """
        from pylinkage.topology import compute_dof
        return compute_dof(self.topology)

    @property
    def mobility(self) -> Any:
        """Full ``MobilityInfo`` (DOF, link count, joint counts).

        See :attr:`dof` for the scalar-only variant. Delegates to
        :func:`pylinkage.topology.compute_mobility`.
        """
        from pylinkage.topology import compute_mobility
        return compute_mobility(self.topology)

    # --- Leg management ---

    def add_legs(self, number: int = 1) -> None:
        """Add phase-offset copies of all non-ground mechanism parts.

        Each copy shares the same ground nodes but has its driver(s)
        phase-offset by ``tau / (number + 1)`` from the existing legs.
        Cloned drivers remain DRIVER nodes with offset ``initial_angle``,
        so each leg is independently driven at the same angular velocity.

        Parameters
        ----------
        number : int
            Number of additional legs to add (default 1).
        """
        if number < 1:
            return

        # Identify non-ground nodes (the "leg" template)
        ground_ids = {n.id for n in self.topology.ground_nodes()}
        driver_ids = {n.id for n in self.topology.driver_nodes()}
        template_node_ids = [nid for nid in self.topology.nodes if nid not in ground_ids]

        total_legs = number + 1  # existing + new
        edge_counter = len(self.topology.edges)
        node_counter = 0

        for leg_idx in range(1, number + 1):
            suffix = f" ({leg_idx})"
            phase_offset = tau * leg_idx / total_legs
            old_to_new: dict[str, str] = {}

            # Map ground nodes to themselves
            for gid in ground_ids:
                old_to_new[gid] = gid

            # Clone non-ground nodes
            for nid in template_node_ids:
                orig_node = self.topology.nodes[nid]
                new_id = f"{nid}{suffix}"
                while new_id in self.topology.nodes:
                    node_counter += 1
                    new_id = f"{nid}_{node_counter}"

                new_node = Node(
                    id=new_id,
                    role=orig_node.role,
                    joint_type=orig_node.joint_type,
                    name=f"{orig_node.name}{suffix}" if orig_node.name else new_id,
                )
                self.topology.add_node(new_node)
                old_to_new[nid] = new_id

                # Copy position from original
                orig_pos = self.dimensions.get_node_position(nid)
                if orig_pos is not None:
                    self.dimensions.node_positions[new_id] = orig_pos

                # For driver nodes, copy DriverAngle with phase offset
                if nid in driver_ids and nid in self.dimensions.driver_angles:
                    orig_da = self.dimensions.driver_angles[nid]
                    self.dimensions.driver_angles[new_id] = DriverAngle(
                        angular_velocity=orig_da.angular_velocity,
                        initial_angle=orig_da.initial_angle + phase_offset,
                    )

            # Clone edges
            for edge_id, edge in list(self.topology.edges.items()):
                src_is_template = edge.source in old_to_new and edge.source not in ground_ids
                tgt_is_template = edge.target in old_to_new and edge.target not in ground_ids
                if not (src_is_template or tgt_is_template):
                    continue

                new_src = old_to_new.get(edge.source, edge.source)
                new_tgt = old_to_new.get(edge.target, edge.target)
                new_edge_id = f"edge_{edge_counter}"
                edge_counter += 1

                new_edge = Edge(id=new_edge_id, source=new_src, target=new_tgt)
                self.topology.add_edge(new_edge)

                # Copy edge distance
                dist = self.dimensions.get_edge_distance(edge_id)
                if dist is not None:
                    self.dimensions.edge_distances[new_edge_id] = dist

            # Clone hyperedges
            for he_id, he in list(self.topology.hyperedges.items()):
                he_has_template = any(n in old_to_new and n not in ground_ids for n in he.nodes)
                if not he_has_template:
                    continue
                new_nodes = tuple(old_to_new.get(n, n) for n in he.nodes)
                new_he = Hyperedge(
                    id=f"{he_id}{suffix}",
                    nodes=new_nodes,
                    name=f"{he.name}{suffix}" if he.name else None,
                )
                self.topology.add_hyperedge(new_he)

        self._invalidate_cache()

    def add_opposite_leg(self, axis_x: float = 0.0) -> None:
        """Create an antisymmetric (mirrored) copy of the mechanism.

        The opposite leg has X-coordinates reflected across ``axis_x``
        and driver initial angles offset by pi for phase opposition.

        Parameters
        ----------
        axis_x : float
            X-coordinate of the vertical mirror axis. Default 0.0.
        """
        ground_ids = {n.id for n in self.topology.ground_nodes()}
        template_node_ids = [nid for nid in self.topology.nodes if nid not in ground_ids]
        axis_tolerance = 1e-9

        suffix = " (opposite)"
        old_to_new: dict[str, str] = {}

        edge_counter = len(self.topology.edges)

        # Map ground nodes: share if on axis, clone if off-axis
        for gid in ground_ids:
            pos = self.dimensions.get_node_position(gid)
            if pos is not None and abs(pos[0] - axis_x) < axis_tolerance:
                old_to_new[gid] = gid  # Share on-axis ground
            else:
                # Clone off-axis ground
                new_id = f"{gid}{suffix}"
                old_to_new[gid] = new_id
                new_node = Node(
                    id=new_id,
                    role=NodeRole.GROUND,
                    joint_type=self.topology.nodes[gid].joint_type,
                    name=f"{self.topology.nodes[gid].name}{suffix}",
                )
                self.topology.add_node(new_node)
                if pos is not None:
                    self.dimensions.node_positions[new_id] = (
                        2 * axis_x - pos[0],
                        pos[1],
                    )

        # Clone non-ground nodes
        driver_ids = {n.id for n in self.topology.driver_nodes()}
        for nid in template_node_ids:
            orig_node = self.topology.nodes[nid]
            new_id = f"{nid}{suffix}"
            new_node = Node(
                id=new_id,
                role=orig_node.role,
                joint_type=orig_node.joint_type,
                name=f"{orig_node.name}{suffix}" if orig_node.name else new_id,
            )
            self.topology.add_node(new_node)
            old_to_new[nid] = new_id

            # Mirror position
            pos = self.dimensions.get_node_position(nid)
            if pos is not None:
                self.dimensions.node_positions[new_id] = (
                    2 * axis_x - pos[0],
                    pos[1],
                )

            # For driver nodes, copy DriverAngle with pi phase offset
            if nid in driver_ids and nid in self.dimensions.driver_angles:
                orig_da = self.dimensions.driver_angles[nid]
                self.dimensions.driver_angles[new_id] = DriverAngle(
                    angular_velocity=orig_da.angular_velocity,
                    initial_angle=orig_da.initial_angle + tau / 2,
                )

        # Clone edges
        for edge_id, edge in list(self.topology.edges.items()):
            src_is_template = edge.source in old_to_new
            tgt_is_template = edge.target in old_to_new
            if not (src_is_template or tgt_is_template):
                continue
            # Skip if both map to themselves (shared ground)
            new_src = old_to_new.get(edge.source, edge.source)
            new_tgt = old_to_new.get(edge.target, edge.target)
            if new_src == edge.source and new_tgt == edge.target:
                continue

            new_edge_id = f"edge_{edge_counter}"
            edge_counter += 1
            new_edge = Edge(id=new_edge_id, source=new_src, target=new_tgt)
            self.topology.add_edge(new_edge)

            dist = self.dimensions.get_edge_distance(edge_id)
            if dist is not None:
                self.dimensions.edge_distances[new_edge_id] = dist

        # Clone hyperedges
        for he_id, he in list(self.topology.hyperedges.items()):
            he_has_template = any(n in old_to_new for n in he.nodes)
            if not he_has_template:
                continue
            new_nodes = tuple(old_to_new.get(n, n) for n in he.nodes)
            if new_nodes == he.nodes:
                continue
            new_he = Hyperedge(
                id=f"{he_id}{suffix}",
                nodes=new_nodes,
                name=f"{he.name}{suffix}" if he.name else None,
            )
            self.topology.add_hyperedge(new_he)

        self._invalidate_cache()

    # Backward compat alias
    mirror_leg = add_opposite_leg

    # --- Topology queries ---

    def get_feet(self) -> list[str]:
        """Return node IDs of foot joints.

        Detection heuristic (in priority order):

        1. Terminal nodes (degree 1, non-ground, non-driver) — classic
           walking linkages like Theo Jansen or Klann.
        2. *Outermost driven nodes* — nodes that are DRIVEN and whose
           neighbours are all either GROUND, DRIVER, or already
           identified as feet.  This catches coupler points (``P``) in
           synthesised four-bars where P connects to B and C but is the
           true foot.

        If auto-detection returns nothing, every DRIVEN node is
        considered a candidate (safe fallback that preserves the old
        "all edges collide" behaviour until the user specifies manually).
        """
        ground_ids = {n.id for n in self.topology.ground_nodes()}
        driver_ids = {n.id for n in self.topology.driver_nodes()}
        privileged = ground_ids | driver_ids

        degree: dict[str, int] = {nid: 0 for nid in self.topology.nodes}
        neighbors: dict[str, set[str]] = {nid: set() for nid in self.topology.nodes}
        for edge in self.topology.edges.values():
            degree[edge.source] += 1
            degree[edge.target] += 1
            neighbors[edge.source].add(edge.target)
            neighbors[edge.target].add(edge.source)

        # 1. Terminal non-ground/non-driver nodes
        feet = [
            nid for nid, d in degree.items()
            if d == 1 and nid not in privileged
        ]
        if feet:
            return feet

        # 2. Outermost driven nodes: DRIVEN nodes all of whose
        #    neighbours are ground, driver, or other driven nodes
        #    already connected to ground/driver.  In a four-bar with
        #    coupler point, P's neighbours are B (driver) and C
        #    (driven), so P qualifies.
        driven = {
            nid for nid, node in self.topology.nodes.items()
            if node.role == NodeRole.DRIVEN
        }
        for nid in driven:
            if all(nb in privileged or nb in driven for nb in neighbors[nid]):
                feet.append(nid)
        if feet:
            return feet

        return []

    def get_foot_edges(self) -> list[str]:
        """Return edge IDs of edges that connect to foot nodes.

        Foot edges are the only edges that should collide with the ground
        during physics simulation.  By default, every edge with at least
        one foot endpoint (as returned by :meth:`get_feet`) is considered
        a foot edge.  Override or set ``foot_edge_ids`` to customise.
        """
        if self._foot_edge_ids is not None:
            return list(self._foot_edge_ids)

        foot_ids = set(self.get_feet())
        return [
            eid
            for eid, edge in self.topology.edges.items()
            if edge.source in foot_ids or edge.target in foot_ids
        ]

    @property
    def foot_edge_ids(self) -> list[str] | None:
        """Explicit foot-edge override, or *None* for auto-detection."""
        return self._foot_edge_ids

    @foot_edge_ids.setter
    def foot_edge_ids(self, value: list[str] | None) -> None:
        self._foot_edge_ids = value

    # --- Optimization interface ---
    # These methods bridge the pylinkage optimizer contract.

    def get_num_constraints(self, flat: bool = True) -> list[float]:
        """Get optimizable constraints as a flat list of floats.

        Returns edge distances from the Mechanism (link lengths),
        compatible with pylinkage optimizers.
        """
        return self.to_mechanism().get_constraints()

    def set_num_constraints(
        self,
        constraints: list[float] | list[list[float]],
        flat: bool = True,
    ) -> None:
        """Set constraints from a flat list of floats.

        Updates the Mechanism, then syncs edge distances back to Dimensions.
        """
        mechanism = self.to_mechanism()
        if not flat and isinstance(constraints, list) and constraints and isinstance(constraints[0], list):
            # Flatten nested list
            flat_constraints = [v for sub in constraints for v in sub]
        else:
            flat_constraints = list(constraints)  # type: ignore[arg-type]

        mechanism.set_constraints(flat_constraints)

        # Sync back: update Dimensions.edge_distances from mechanism link lengths
        self._sync_dimensions_from_mechanism(mechanism)

    def get_coords(self) -> list[tuple[float, float]]:
        """Get current joint positions as a list of (x, y) tuples."""
        mechanism = self.to_mechanism()
        return mechanism.get_joint_positions()

    def set_coords(
        self,
        coords: list[tuple[float, float]] | list[tuple[float | None, float | None]],
    ) -> None:
        """Set joint positions."""
        mechanism = self.to_mechanism()
        mechanism.set_joint_positions(coords)  # type: ignore[arg-type]

        # Sync back to Dimensions
        for joint in mechanism.joints:
            pos = joint.position
            if pos[0] is not None and pos[1] is not None:
                self.dimensions.node_positions[joint.id] = (pos[0], pos[1])

    def _sync_dimensions_from_mechanism(self, mechanism: Mechanism) -> None:
        """Sync Dimensions edge distances from Mechanism link state."""
        from pylinkage.mechanism.link import DriverLink, ArcDriverLink, GroundLink

        # Rebuild edge distance mapping from mechanism's links
        # This matches the order used in get_constraints/set_constraints
        constraints = mechanism.get_constraints()
        idx = 0
        for link in mechanism.links:
            if isinstance(link, GroundLink):
                continue
            if isinstance(link, (DriverLink, ArcDriverLink)):
                if link.radius is not None and idx < len(constraints):
                    # Find the edge in our topology corresponding to this link
                    self._update_edge_distance_for_link(link, constraints[idx])
                    idx += 1
            else:
                if link.length is not None and idx < len(constraints):
                    self._update_edge_distance_for_link(link, constraints[idx])
                    idx += 1

    def _update_edge_distance_for_link(self, link: object, distance: float) -> None:
        """Update the edge distance in Dimensions for a given Mechanism link."""
        from pylinkage.mechanism.link import Link
        if not isinstance(link, Link) or len(link.joints) < 2:
            return
        j0_id = link.joints[0].id
        j1_id = link.joints[1].id
        # Find matching edge in topology
        edge = self.topology.get_edge_between(j0_id, j1_id)
        if edge is not None:
            self.dimensions.edge_distances[edge.id] = distance
