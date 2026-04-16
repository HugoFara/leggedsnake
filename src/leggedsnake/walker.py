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
from typing import TYPE_CHECKING, Any

from pylinkage.dimensions import Dimensions, DriverAngle
from pylinkage.hypergraph import HypergraphLinkage, NodeRole, to_mechanism
from pylinkage.hypergraph.core import Edge, Hyperedge, Node

if TYPE_CHECKING:
    from pylinkage.hypergraph import HierarchicalLinkage
    from pylinkage.mechanism import Mechanism
    from pylinkage.topology.catalog import CatalogEntry
    from pylinkage.topology.analysis import MobilityInfo


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

    @classmethod
    def from_watt(
        cls,
        crank: float,
        coupler1: float,
        rocker1: float,
        link4: float,
        link5: float,
        rocker2: float,
        ground_length: float,
        ground_pivot_a: tuple[float, float] = (0.0, 0.0),
        initial_crank_angle: float = 0.0,
        motor_rates: float | dict[str, float] = -4.0,
        name: str = "watt",
    ) -> Walker:
        """Build a Walker from a Watt six-bar specification.

        Wraps :func:`pylinkage.synthesis.watt_from_lengths` and feeds its
        result through the SimLinkage shim (``_walker_from_sim_linkage``).
        A Watt six-bar has two four-bar loops sharing the crank output,
        yielding two coupled rocker trajectories that can serve as foot
        paths.

        Parameters
        ----------
        crank, coupler1, rocker1, link4, link5, rocker2 : float
            Link lengths. See
            :func:`pylinkage.synthesis.watt_from_lengths` for the
            kinematic chain diagram.
        ground_length : float
            Distance between ground pivots A and D.
        ground_pivot_a : (float, float)
            World position of ground pivot A.
        initial_crank_angle : float
            Starting crank angle (radians).
        motor_rates : float | dict[str, float]
            Motor angular velocities applied to the resulting Walker.
        name : str
            Name for the linkage.

        Raises
        ------
        ValueError
            If pylinkage cannot assemble the mechanism at the given
            link lengths / initial angle.
        """
        from pylinkage.synthesis import watt_from_lengths

        sim = watt_from_lengths(
            crank=crank,
            coupler1=coupler1,
            rocker1=rocker1,
            link4=link4,
            link5=link5,
            rocker2=rocker2,
            ground_length=ground_length,
            ground_pivot_a=ground_pivot_a,
            initial_crank_angle=initial_crank_angle,
            name=name,
        )
        return _walker_from_sim_linkage(sim, motor_rates=motor_rates)

    @classmethod
    def from_stephenson(
        cls,
        crank: float,
        coupler: float,
        rocker: float,
        link4: float,
        link5: float,
        link6: float,
        ground_length: float,
        ground_pivot_a: tuple[float, float] = (0.0, 0.0),
        initial_crank_angle: float = 0.0,
        motor_rates: float | dict[str, float] = -4.0,
        name: str = "stephenson",
    ) -> Walker:
        """Build a Walker from a Stephenson six-bar specification.

        Wraps :func:`pylinkage.synthesis.stephenson_from_lengths` and
        feeds its result through the SimLinkage shim.

        Stephenson differs from Watt in where the second loop attaches:
        on a Stephenson the second chain branches from the coupler
        joint C and ground D (so the two ternary links are separated);
        on a Watt the second chain branches from B and C (ternary
        links adjacent). See
        :func:`pylinkage.synthesis.stephenson_from_lengths` for the
        kinematic chain.

        Raises
        ------
        ValueError
            If pylinkage cannot assemble the mechanism at the given
            link lengths / initial angle.
        """
        from pylinkage.synthesis import stephenson_from_lengths

        sim = stephenson_from_lengths(
            crank=crank,
            coupler=coupler,
            rocker=rocker,
            link4=link4,
            link5=link5,
            link6=link6,
            ground_length=ground_length,
            ground_pivot_a=ground_pivot_a,
            initial_crank_angle=initial_crank_angle,
            name=name,
        )
        return _walker_from_sim_linkage(sim, motor_rates=motor_rates)

    @classmethod
    def from_jansen(
        cls,
        scale: float = 1.0,
        initial_crank_angle: float = 0.0,
        angular_velocity: float = -tau / 48,
        motor_rates: float | dict[str, float] = -4.0,
        name: str = "jansen",
    ) -> Walker:
        """Build a Walker for Theo Jansen's 8-bar "Strandbeest" leg.

        Uses Jansen's canonical "Holy Numbers" (link lengths he discovered
        through decades of optimization), scaled by ``scale``. The single
        foot ``G`` traces a stable walking locus for a full crank rotation.

        Parameters
        ----------
        scale : float
            Multiplier applied to every link length. Default 1.0 yields
            the raw Holy Numbers (~65 length units at widest); drop to
            ~0.04 for physics-friendly metric dimensions.
        initial_crank_angle : float
            Starting crank angle (radians).
        angular_velocity : float
            Kinematic crank step (rad per ``step()`` iteration). Default
            traces 48 samples per revolution.
        motor_rates : float | dict[str, float]
            Motor angular velocity (rad/s) for physics simulation.
        name : str
            Name for the linkage.
        """
        from ._classical import build_jansen

        hg, dims = build_jansen(
            scale=scale,
            initial_crank_angle=initial_crank_angle,
            angular_velocity=angular_velocity,
            name=name,
        )
        return cls(hg, dims, name=name, motor_rates=motor_rates)

    @classmethod
    def from_klann(
        cls,
        scale: float = 3.0,
        initial_crank_angle: float = 0.0,
        angular_velocity: float = tau / 48,
        motor_rates: float | dict[str, float] = 4.0,
        name: str = "klann",
    ) -> Walker:
        """Build a Walker for Joe Klann's 6-bar walking linkage.

        Uses the dimensions of US Patent 6,260,862, scaled by ``scale``.
        The mechanism is a Stephenson-III topology with two rigid
        triangles: the ternary coupler (A-elbow-knee) rotates with the
        crank, while the ternary leg (hip-knee-foot) carries the foot.

        Parameters
        ----------
        scale : float
            Multiplier applied to every length and offset. Default 3.0
            matches the canonical example visualization; patent values
            themselves are dimensionless ratios around 1.0.
        initial_crank_angle : float
            Starting crank angle (radians).
        angular_velocity : float
            Kinematic crank step (rad per ``step()`` iteration). Positive
            by default since the Klann mechanism walks forward under
            counter-clockwise rotation.
        motor_rates : float | dict[str, float]
            Motor angular velocity (rad/s) for physics simulation.
        name : str
            Name for the linkage.
        """
        from ._classical import build_klann

        hg, dims = build_klann(
            scale=scale,
            initial_crank_angle=initial_crank_angle,
            angular_velocity=angular_velocity,
            name=name,
        )
        return cls(hg, dims, name=name, motor_rates=motor_rates)

    @classmethod
    def from_strider(
        cls,
        crank: float = 1.0,
        triangle: float = 2.0,
        femur: float = 1.8,
        rocker_l: float = 2.6,
        rocker_s: float = 1.4,
        tibia: float = 2.5,
        foot: float = 1.8,
        angular_velocity: float = -tau / 10,
        motor_rates: float | dict[str, float] = -4.0,
        name: str = "strider",
    ) -> Walker:
        """Build a Walker for the Strider mechanism (Vagle, DIY Walkers).

        The Strider is a symmetric 11-node mechanism producing two feet
        per crank. Its kinematic chain: ground pair (``A``, ``Y``) →
        rigidly attached frame points (``B``, ``B_p``) → knees
        (``D``, ``E``) driven off crank ``C`` → ankles (``F``, ``G``)
        rigid on the ``C-E`` / ``C-D`` rockers → feet (``H``, ``I``).

        Parameters
        ----------
        crank, triangle, femur, rocker_l, rocker_s, tibia, foot : float
            Link lengths. Defaults reproduce the Strider example shipped
            with ``leggedsnake`` (``examples/strider.py``).
        angular_velocity : float
            Kinematic crank step (rad per ``step()`` iteration). Default
            traces 10 samples per revolution (matches the example).
        motor_rates : float | dict[str, float]
            Motor angular velocity (rad/s) for physics simulation.
        name : str
            Name for the linkage.
        """
        from ._classical import build_strider

        hg, dims = build_strider(
            crank=crank,
            triangle=triangle,
            femur=femur,
            rocker_l=rocker_l,
            rocker_s=rocker_s,
            tibia=tibia,
            foot=foot,
            angular_velocity=angular_velocity,
            name=name,
        )
        return cls(hg, dims, name=name, motor_rates=motor_rates)

    @classmethod
    def from_ghassaei(
        cls,
        scale: float = 1.0,
        initial_crank_angle: float = 0.0,
        angular_velocity: float = -tau / 48,
        motor_rates: float | dict[str, float] = -4.0,
        name: str = "ghassaei",
    ) -> Walker:
        """Build Amanda Ghassaei's 5-dyad leg (Boim/Walkin8r, thesis Fig. 5.4.4).

        8 nodes (A, B grounds; C crank tip; D, F off (C, B); H unnamed
        intermediate off (D, B); E real lower-left joint off (H, B); G foot
        off (E, F)), 11 bars, 5 RRR dyads. Classical Ghassaei dimensions
        are applied exactly: crank=26, ground=53, 56/77 inner/outer bars,
        75 closing bars. H-E (not given on the figure) defaults to 130 to
        reproduce the Wikibooks reference foot-locus aspect (~0.243).
        Initial crank angle is 0.085 rad off vertical.
        """
        from ._classical import build_ghassaei

        hg, dims = build_ghassaei(
            scale=scale,
            initial_crank_angle=initial_crank_angle,
            angular_velocity=angular_velocity,
            name=name,
        )
        return cls(hg, dims, name=name, motor_rates=motor_rates)

    @classmethod
    def from_trotbot(
        cls,
        scale: float = 1.0,
        initial_crank_angle: float = 0.0,
        angular_velocity: float = -tau / 48,
        motor_rates: float | dict[str, float] = -4.0,
        name: str = "trotbot",
    ) -> Walker:
        """Build a Walker for the TrotBot mechanism (Vagle, DIYwalkers).

        Uses the bar lengths from Wade & Ben Vagle's combined Python
        simulator published at https://www.diywalkers.com, scaled by
        ``scale``. The mechanism has 10 joints, 15 binary edges, and
        three collinear rigid ternaries (``j3-j2-j4``, ``j5-j4-j6``,
        ``j1-j2-j9``) modelled as hyperedges. The foot is ``j7``.

        Parameters
        ----------
        scale : float
            Multiplier applied to every link length and the frame offset.
        initial_crank_angle : float
            Starting crank angle (radians).
        angular_velocity : float
            Kinematic crank step (rad per ``step()`` iteration).
        motor_rates : float | dict[str, float]
            Motor angular velocity (rad/s) for physics simulation.
        name : str
            Name for the linkage.

        Raises
        ------
        ValueError
            If the mechanism cannot be assembled at the given crank angle.
        """
        from ._classical import build_trotbot

        hg, dims = build_trotbot(
            scale=scale,
            initial_crank_angle=initial_crank_angle,
            angular_velocity=angular_velocity,
            name=name,
        )
        return cls(hg, dims, name=name, motor_rates=motor_rates)

    @classmethod
    def from_chebyshev(
        cls,
        crank: float = 0.75,
        coupler: float = 3.75,
        rocker: float = 3.75,
        ground_length: float = 3.0,
        foot_ratio: float = 1.0,
        initial_crank_angle: float = 0.0,
        angular_velocity: float = -tau / 48,
        motor_rates: float | dict[str, float] = -4.0,
        name: str = "chebyshev",
    ) -> Walker:
        """Build a Walker for Chebyshev's Lambda linkage (1878).

        A 4-bar crank-rocker whose coupler point traces an approximate
        straight line — the basis of Chebyshev's "plantigrade machine".
        The foot ``P`` rides rigidly on the coupler at distance
        ``foot_ratio * coupler`` from A (``foot_ratio=1.0`` puts P at B).

        Parameters
        ----------
        crank, coupler, rocker : float
            Link lengths. Defaults reproduce the working ratio used in
            the Chebyshev example.
        ground_length : float
            Distance between ground pivots O1 and O2 (both on y=0).
        foot_ratio : float
            Position of the foot along the A→B coupler, as a fraction
            of ``coupler`` (0.5 is the midpoint straight-line tracing
            point, 1.0 extends it to B).
        initial_crank_angle : float
            Starting crank angle (radians).
        angular_velocity : float
            Kinematic crank step (rad per ``step()`` iteration).
        motor_rates : float | dict[str, float]
            Motor angular velocity (rad/s) for physics simulation.
        name : str
            Name for the linkage.

        Raises
        ------
        ValueError
            If the coupler and rocker cannot meet at the chosen angle.
        """
        from ._classical import build_chebyshev

        hg, dims = build_chebyshev(
            crank=crank,
            coupler=coupler,
            rocker=rocker,
            ground_length=ground_length,
            foot_ratio=foot_ratio,
            initial_crank_angle=initial_crank_angle,
            angular_velocity=angular_velocity,
            name=name,
        )
        return cls(hg, dims, name=name, motor_rates=motor_rates)

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
        return int(self.to_mechanism().get_rotation_period())

    @property
    def joints(self) -> list[Any]:
        """Mechanism joints in solve order.

        Surface pylinkage's compatibility hook: ``pylinkage._compat.get_parts``
        sniffs for ``.joints`` or ``.components`` to iterate over a
        linkage's parts. Optimizers such as ``chain_optimizers`` /
        ``minimize_linkage`` / ``particle_swarm_optimization`` rely on it.
        """
        return list(self.to_mechanism().joints)

    @property
    def _solve_order(self) -> list[Any]:
        """Expose the underlying mechanism's solve order.

        pylinkage 0.9's ``linkage_to_solver_data`` (invoked from ``Ensemble``
        and every optimizer that wraps it) reads ``linkage._solve_order``.
        Delegating to the cached ``Mechanism`` keeps joint identities in sync
        with ``self.joints`` so ``id(joint)`` lookups resolve correctly.
        """
        return list(self.to_mechanism()._solve_order)

    # --- SimLinkage bridge (temporary compat shim) ---
    #
    # Until pylinkage 1.0 ships a stable SimLinkage → HypergraphLinkage path,
    # this module provides its own minimal conversion covering the component
    # types emitted by pylinkage's N-bar synthesis and catalog-based
    # co_optimize (Ground, Crank/ArcCrank, RRRDyad, FixedDyad). See
    # :func:`_walker_from_sim_linkage` below.

    # --- Kinematic derivatives (temporary compat shim) ---
    #
    # pylinkage's ``Mechanism.step_with_derivatives`` / ``set_input_velocity`` /
    # ``get_velocities`` / ``get_accelerations`` are committed upstream but not
    # yet released (expected in pylinkage 1.0). Until the first release carries
    # them, we provide a finite-difference implementation here. Consumers should
    # depend on ``Walker.step_with_derivatives`` (this module), not on the raw
    # pylinkage surface.
    #
    # DEPRECATION PLAN: once a pylinkage release pins the upstream API, drop
    # this finite-difference path and delegate to ``Mechanism.step_with_derivatives``
    # directly. Keep the Walker method signature stable across the switch.

    def step_with_derivatives(
        self,
        iterations: int | None = None,
        dt: float = 1.0,
        skip_unbuildable: bool = False,
    ) -> Generator[
        tuple[
            tuple[tuple[float, float] | tuple[float | None, float | None], ...],
            tuple[tuple[float, float] | tuple[None, None], ...],
            tuple[tuple[float, float] | tuple[None, None], ...],
        ],
        None,
        None,
    ]:
        """Simulate one rotation, yielding per-frame positions, velocities, accelerations.

        Velocities and accelerations are computed by three-point central
        finite differences against ``dt`` (forward / backward at the ends).
        Frames where a joint is unbuildable yield ``(None, None)`` for
        that joint in all three tuples — no derivative across a dead zone.

        This is a **temporary shim** awaiting pylinkage 1.0's
        ``Mechanism.step_with_derivatives`` becoming generally available.

        Parameters
        ----------
        iterations : int | None
            Number of simulation steps. If *None*, one full rotation period.
        dt : float
            Time step used for the finite-difference denominator.
        skip_unbuildable : bool
            Forwarded to :meth:`step`.

        Yields
        ------
        (positions, velocities, accelerations)
            Each is a tuple of per-joint ``(x, y)`` (or ``(None, None)``
            where undefined). The stream's length equals ``iterations``.
        """
        positions = list(self.step(
            iterations=iterations, dt=dt, skip_unbuildable=skip_unbuildable,
        ))
        n = len(positions)
        if n == 0:
            return

        n_joints = len(positions[0])
        none_pair = (None, None)

        def _central(i: int, j: int) -> tuple[float, float] | tuple[None, None]:
            # Three-point derivative of position j at frame i.
            if i == 0:
                prev_, next_ = positions[0], positions[1] if n > 1 else positions[0]
                denom = dt if n > 1 else 1.0
            elif i == n - 1:
                prev_, next_ = positions[n - 2], positions[n - 1]
                denom = dt
            else:
                prev_, next_ = positions[i - 1], positions[i + 1]
                denom = 2 * dt
            p0, p1 = prev_[j], next_[j]
            if p0[0] is None or p0[1] is None or p1[0] is None or p1[1] is None:
                return none_pair
            return ((p1[0] - p0[0]) / denom, (p1[1] - p0[1]) / denom)

        # Pre-compute velocities so we can difference them for acceleration.
        velocities: list[tuple[tuple[float, float] | tuple[None, None], ...]] = [
            tuple(_central(i, j) for j in range(n_joints))
            for i in range(n)
        ]

        def _accel_central(i: int, j: int) -> tuple[float, float] | tuple[None, None]:
            if n < 2:
                return none_pair
            if i == 0:
                v0, v1 = velocities[0][j], velocities[1][j]
                denom = dt
            elif i == n - 1:
                v0, v1 = velocities[n - 2][j], velocities[n - 1][j]
                denom = dt
            else:
                v0, v1 = velocities[i - 1][j], velocities[i + 1][j]
                denom = 2 * dt
            if v0[0] is None or v0[1] is None or v1[0] is None or v1[1] is None:
                return none_pair
            return ((v1[0] - v0[0]) / denom, (v1[1] - v0[1]) / denom)

        for i in range(n):
            accel = tuple(_accel_central(i, j) for j in range(n_joints))
            yield positions[i], velocities[i], accel

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
        return int(compute_dof(self.topology))

    @property
    def mobility(self) -> MobilityInfo:
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

        # Pre-compute each clone's kinematic pose by stepping the template
        # mechanism to the phase-offset point. We can't just rotate the
        # crank's position: ``to_mechanism`` derives the crank's initial
        # angle from ``atan2(crank_pos - motor_pos)``, so without this the
        # cloned drivers stay synchronized with the template and the legs
        # don't desynchronize. Rotating only the crank puts downstream
        # joints outside their circle-circle solution, which makes the
        # first solve fail. Simulating the template is the cleanest fix.
        total_legs = number + 1  # existing + new
        clone_positions: list[dict[str, tuple[float, float]]] = []
        if template_node_ids:
            import math as _math
            omega_vals = [
                da.angular_velocity
                for da in self.dimensions.driver_angles.values()
                if da.angular_velocity
            ]
            omega = omega_vals[0] if omega_vals else 1.0
            omega_sign = 1.0 if omega >= 0 else -1.0
            for leg_idx in range(1, number + 1):
                phase_offset = tau * leg_idx / total_legs
                iters = int(round(phase_offset / abs(omega))) if omega else 0
                self._invalidate_cache()
                mech = self.to_mechanism()
                for _ in range(iters):
                    mech._step_once(_math.copysign(1.0, omega) if omega else 0.0)
                pose = {}
                for j in mech.joints:
                    jid = getattr(j, 'id', None)
                    if jid in template_node_ids:
                        pose[jid] = j.coord()
                clone_positions.append(pose)
                _ = omega_sign  # quiet lint; retained for future asymmetric drivers
            self._invalidate_cache()

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

                # Use the phase-offset pose we pre-computed by stepping the
                # template mechanism — this places every cloned joint (not
                # just the crank) at a kinematically valid pose where the
                # crank's atan2 angle reflects the phase offset.
                pose = clone_positions[leg_idx - 1] if clone_positions else {}
                clone_pos = pose.get(nid)
                if clone_pos is None:
                    clone_pos = self.dimensions.get_node_position(nid)
                if clone_pos is not None:
                    self.dimensions.node_positions[new_id] = clone_pos

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

    def get_constraints(self) -> list[float]:
        """Get optimizable constraints as a flat list of floats.

        Returns edge distances from the Mechanism (link lengths),
        compatible with pylinkage optimizers that sniff
        ``linkage.get_constraints()``.
        """
        return list(self.to_mechanism().get_constraints())

    def set_constraints(self, values: list[float]) -> None:
        """Set constraints from a flat list of floats.

        Updates the Mechanism, then syncs edge distances back to Dimensions.
        """
        mechanism = self.to_mechanism()
        mechanism.set_constraints(list(values))
        self._sync_dimensions_from_mechanism(mechanism)

    # Back-compat aliases (will be deprecated with pylinkage 0.10 which
    # does the same rename). The ``num_`` prefix meant "numeric"
    # historically but reads as "number of", so upstream dropped it.

    def get_num_constraints(self, flat: bool = True) -> list[float]:
        """Alias for :meth:`get_constraints` (back-compat)."""
        return self.get_constraints()

    def set_num_constraints(
        self,
        constraints: list[float] | list[list[float]],
        flat: bool = True,
    ) -> None:
        """Alias for :meth:`set_constraints` (back-compat).

        Accepts a flat list of floats or (with ``flat=False``) a nested
        list that will be flattened in order.
        """
        flat_constraints: list[float]
        if (
            not flat
            and constraints
            and isinstance(constraints[0], list)
        ):
            nested: list[list[float]] = constraints  # type: ignore[assignment]
            flat_constraints = [v for sub in nested for v in sub]
        else:
            flat_constraints = [float(v) for v in constraints]  # type: ignore[arg-type]
        self.set_constraints(flat_constraints)

    def get_coords(self) -> list[tuple[float, float]]:
        """Get current joint positions as a list of (x, y) tuples."""
        mechanism = self.to_mechanism()
        return list(mechanism.get_joint_positions())

    def set_coords(
        self,
        coords: list[tuple[float, float]] | list[tuple[float | None, float | None]],
    ) -> None:
        """Set joint positions."""
        mechanism = self.to_mechanism()
        mechanism.set_joint_positions(coords)

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


# ---------------------------------------------------------------------------
# Temporary SimLinkage → Walker shim
# ---------------------------------------------------------------------------
#
# Pylinkage's ``simulation.Linkage`` (SimLinkage) is the new joint-free
# component/actuator/dyad API. It is what ``pylinkage.synthesis`` and
# ``pylinkage.optimization.co_optimize`` produce. Leggedsnake's Walker is
# hypergraph-native — until pylinkage 1.0 ships a supported bridge, we
# convert here. Only the component types the upstream co-optimization
# actually emits are handled: ``Ground``, ``Crank``, ``ArcCrank``,
# ``RRRDyad``, ``FixedDyad``. Anything else raises ``NotImplementedError``
# so we fail loudly when pylinkage adds a new component type.


def _walker_from_sim_linkage(
    sim_linkage: object,
    motor_rates: float | dict[str, float] = -4.0,
) -> Walker:
    """Convert a pylinkage SimLinkage to a Walker (TEMPORARY SHIM).

    Handles component types emitted by pylinkage's N-bar synthesis and
    catalog-backed co-optimization.

    Parameters
    ----------
    sim_linkage : pylinkage.simulation.Linkage
        A SimLinkage built from ``Ground`` / ``Crank`` / ``RRRDyad`` /
        ``FixedDyad`` components (what ``co_optimize`` produces).
    motor_rates : float | dict[str, float]
        Forwarded to the resulting Walker.

    Raises
    ------
    NotImplementedError
        If ``sim_linkage`` contains a component type this shim doesn't
        yet handle. Fix the upstream feature gap or extend the shim.
    """
    components = getattr(sim_linkage, "components", None)
    if components is None:
        raise TypeError(
            f"Expected SimLinkage, got {type(sim_linkage).__name__}"
        )

    hg = HypergraphLinkage(name=getattr(sim_linkage, "name", "") or "")
    node_positions: dict[str, tuple[float, float]] = {}
    edge_distances: dict[str, float] = {}
    driver_angles: dict[str, DriverAngle] = {}
    component_to_node: dict[int, str] = {}

    # Pass 1: create a Node per Component.
    for i, comp in enumerate(components):
        cls_name = type(comp).__name__
        if cls_name == "Ground":
            role = NodeRole.GROUND
        elif cls_name in ("Crank", "ArcCrank"):
            role = NodeRole.DRIVER
        else:
            role = NodeRole.DRIVEN
        node_id = getattr(comp, "name", None) or f"n{i}"
        hg.add_node(Node(node_id, role=role))
        component_to_node[id(comp)] = node_id
        node_positions[node_id] = (float(comp.x), float(comp.y))

    def _anchor_node_id(anchor: object) -> str:
        # Handle _AnchorProxy: delegate to its parent component.
        parent = getattr(anchor, "_parent", anchor)
        key = id(parent)
        if key not in component_to_node:
            raise NotImplementedError(
                f"Anchor {anchor!r} not found in component index"
            )
        return component_to_node[key]

    # Pass 2: edges / hyperedges + driver angles.
    edge_counter = 0
    for comp in components:
        cls_name = type(comp).__name__
        this_id = component_to_node[id(comp)]

        if cls_name == "Ground":
            continue  # ground nodes have no outgoing edge

        if cls_name in ("Crank", "ArcCrank"):
            anchor_id = _anchor_node_id(comp.anchor)
            eid = f"e{edge_counter}"
            edge_counter += 1
            hg.add_edge(Edge(eid, anchor_id, this_id))
            # Crank uses ``radius``; ArcCrank / older variants may use
            # ``distance``.
            raw_len: Any = getattr(comp, "radius", None)
            if raw_len is None:
                raw_len = getattr(comp, "distance", 1.0)
            edge_distances[eid] = float(raw_len)
            omega = float(getattr(comp, "angular_velocity", -tau / 12))
            driver_angles[this_id] = DriverAngle(angular_velocity=omega)
            continue

        if cls_name == "RRRDyad":
            a1 = _anchor_node_id(comp.anchor1)
            a2 = _anchor_node_id(comp.anchor2)
            e1, e2 = f"e{edge_counter}", f"e{edge_counter + 1}"
            edge_counter += 2
            hg.add_edge(Edge(e1, a1, this_id))
            hg.add_edge(Edge(e2, a2, this_id))
            edge_distances[e1] = float(comp.distance1)
            edge_distances[e2] = float(comp.distance2)
            continue

        if cls_name == "FixedDyad":
            # Rigid triangle — the coupler point rides a fixed triangle
            # with anchor1 and anchor2.
            a1 = _anchor_node_id(comp.anchor1)
            a2 = _anchor_node_id(comp.anchor2)
            he_id = f"he{edge_counter}"
            edge_counter += 1
            hg.add_hyperedge(Hyperedge(he_id, (a1, a2, this_id)))
            # Record the anchor-to-point distances as auxiliary edges for
            # simulators that want binary edges. The hyperedge itself
            # carries the rigidity; the edge distances give a usable
            # numeric fallback.
            e1, e2 = f"e{edge_counter}", f"e{edge_counter + 1}"
            edge_counter += 2
            hg.add_edge(Edge(e1, a1, this_id))
            hg.add_edge(Edge(e2, a2, this_id))
            edge_distances[e1] = float(getattr(comp, "distance", 1.0))
            # Distance from a2 to this: approximate from current positions.
            from math import hypot
            a2_pos = node_positions[a2]
            this_pos = node_positions[this_id]
            edge_distances[e2] = hypot(
                this_pos[0] - a2_pos[0], this_pos[1] - a2_pos[1]
            )
            continue

        raise NotImplementedError(
            f"SimLinkage component {cls_name!r} is not yet handled by the "
            "leggedsnake shim. Extend _walker_from_sim_linkage or wait "
            "for pylinkage 1.0's hypergraph-native bridge."
        )

    dims = Dimensions(
        node_positions=node_positions,
        driver_angles=driver_angles,
        edge_distances=edge_distances,
    )
    return Walker(
        topology=hg,
        dimensions=dims,
        name=getattr(sim_linkage, "name", "") or "",
        motor_rates=motor_rates,
    )
