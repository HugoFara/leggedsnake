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
from typing import TYPE_CHECKING

from pylinkage.dimensions import Dimensions
from pylinkage.hypergraph import HypergraphLinkage, NodeRole, to_mechanism
from pylinkage.hypergraph.core import Edge, Hyperedge, Node

if TYPE_CHECKING:
    from pylinkage.mechanism import Mechanism


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

    def __init__(
        self,
        topology: HypergraphLinkage,
        dimensions: Dimensions,
        name: str = "",
        motor_rates: dict[str, float] | float = -4.0,
    ) -> None:
        self.topology = topology
        self.dimensions = dimensions
        self.name = name or topology.name
        self.motor_rates = motor_rates
        self._mechanism = None

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
    ) -> Generator[tuple[tuple[float, float] | tuple[float | None, float | None], ...], None, None]:
        """Simulate one full rotation of the mechanism.

        Delegates to ``Mechanism.step()``. Each driver advances independently.

        Parameters
        ----------
        iterations : int | None
            Number of simulation steps. If None, uses one full rotation period.
        dt : float
            Time step multiplier.

        Yields
        ------
        tuple of (x, y) coordinate tuples
            Joint positions at each step.
        """
        mechanism = self.to_mechanism()
        if iterations is None:
            yield from mechanism.step(dt)
        else:
            for _ in range(iterations):
                mechanism._step_once(dt)
                yield tuple(j.coord() for j in mechanism.joints)

    def get_rotation_period(self) -> int:
        """Number of steps for one full rotation cycle."""
        return self.to_mechanism().get_rotation_period()

    # --- Leg management ---

    def add_legs(self, number: int = 1) -> None:
        """Add phase-offset copies of all non-ground mechanism parts.

        Each copy shares the same ground nodes but has its driver(s)
        phase-offset by ``tau / (number + 1)`` from the existing legs.

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

        # Compute positions at each phase offset by stepping kinematically
        mechanism = self.to_mechanism()
        period = mechanism.get_rotation_period()
        total_legs = number + 1  # existing + new
        step_size = period // total_legs
        if step_size < 1:
            step_size = 1

        # Run mechanism to collect positions at each phase offset
        phase_positions: list[dict[str, tuple[float, float]]] = []
        steps_done = 0
        for positions in mechanism.step():
            steps_done += 1
            if steps_done % step_size == 0 and len(phase_positions) < number:
                pos_dict = {}
                for joint, pos in zip(mechanism.joints, positions):
                    if pos[0] is not None and pos[1] is not None:
                        pos_dict[joint.id] = (pos[0], pos[1])
                phase_positions.append(pos_dict)

        # For each phase offset, clone the non-ground nodes and edges
        edge_counter = max(
            (int(eid.split("_")[-1]) for eid in self.topology.edges if "_" in eid),
            default=0,
        ) + 1
        node_counter = 0

        for leg_idx, offset_positions in enumerate(phase_positions, start=1):
            suffix = f" ({leg_idx})"
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

                # For drivers in additional legs, make them DRIVEN (Fixed joint
                # sharing the original crank's motor, like the old Walker)
                new_role = NodeRole.DRIVEN if nid in driver_ids else orig_node.role
                new_node = Node(
                    id=new_id,
                    role=new_role,
                    joint_type=orig_node.joint_type,
                    name=f"{orig_node.name}{suffix}" if orig_node.name else new_id,
                )
                self.topology.add_node(new_node)
                old_to_new[nid] = new_id

                # Set position from kinematic simulation
                if nid in offset_positions:
                    self.dimensions.node_positions[new_id] = offset_positions[nid]
                else:
                    # Fallback: use original position
                    orig_pos = self.dimensions.get_node_position(nid)
                    if orig_pos is not None:
                        self.dimensions.node_positions[new_id] = orig_pos

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

                # If the original driver has a crank edge to ground, create
                # a new edge from original driver to the new "driven" copy
                # (mechanical linkage sharing the motor)
                if edge.source in driver_ids and edge.target in ground_ids:
                    link_edge_id = f"edge_{edge_counter}"
                    edge_counter += 1
                    link_edge = Edge(
                        id=link_edge_id,
                        source=old_to_new[edge.source],  # the new driven copy
                        target=edge.source,  # the original driver
                    )
                    self.topology.add_edge(link_edge)
                    if dist is not None:
                        self.dimensions.edge_distances[link_edge_id] = dist

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

        edge_counter = max(
            (int(eid.split("_")[-1]) for eid in self.topology.edges if "_" in eid),
            default=0,
        ) + 1

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
        for nid in template_node_ids:
            orig_node = self.topology.nodes[nid]
            new_id = f"{nid}{suffix}"
            # Drivers in opposite leg become DRIVEN (share original motor)
            new_role = NodeRole.DRIVEN if orig_node.role == NodeRole.DRIVER else orig_node.role
            new_node = Node(
                id=new_id,
                role=new_role,
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

        # Clone edges
        driver_ids = {n.id for n in self.topology.driver_nodes()}
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

            # Link opposite driver copy to original driver (shared motor)
            if edge.source in driver_ids and edge.target in ground_ids:
                link_edge_id = f"edge_{edge_counter}"
                edge_counter += 1
                link_edge = Edge(
                    id=link_edge_id,
                    source=old_to_new[edge.source],
                    target=edge.source,
                )
                self.topology.add_edge(link_edge)
                if dist is not None:
                    self.dimensions.edge_distances[link_edge_id] = dist

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
        """Return node IDs of foot joints (terminal, non-ground, non-driver).

        Feet are nodes that don't appear as parents of other nodes —
        i.e., nodes with degree 1 that are not ground or driver nodes.
        """
        ground_ids = {n.id for n in self.topology.ground_nodes()}
        driver_ids = {n.id for n in self.topology.driver_nodes()}

        # Count how many edges each node participates in
        degree: dict[str, int] = {nid: 0 for nid in self.topology.nodes}
        for edge in self.topology.edges.values():
            degree[edge.source] += 1
            degree[edge.target] += 1

        return [
            nid
            for nid, d in degree.items()
            if d == 1 and nid not in ground_ids and nid not in driver_ids
        ]

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


def walker_from_legacy(linkage: object) -> Walker:
    """Create a Walker from a legacy pylinkage Linkage.

    Uses ``from_linkage()`` to convert the joint-based representation
    to a hypergraph. Useful for migrating existing code.

    Parameters
    ----------
    linkage : pylinkage.Linkage
        A legacy joint-based linkage.

    Returns
    -------
    Walker
        A hypergraph-native Walker.
    """
    from pylinkage.hypergraph import from_linkage

    hg, dims = from_linkage(linkage)  # type: ignore[arg-type]

    motor_rate: dict[str, float] | float = -4.0
    if hasattr(linkage, 'motor_rate'):
        motor_rate = linkage.motor_rate

    return Walker(
        topology=hg,
        dimensions=dims,
        name=getattr(linkage, 'name', '') or '',
        motor_rates=motor_rate,
    )
