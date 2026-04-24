"""
Topology co-optimization for walking mechanisms.

Jointly optimizes the discrete mechanism topology (four-bar, six-bar,
eight-bar variants) alongside continuous link dimensions — and, when
``evolve_offsets=True``, the per-leg phase offsets that determine
gait — using NSGA-II/III. Each candidate is a mixed chromosome:
``[topology_index, [n_legs,] dim_1, ..., dim_N, [off_1, ..., off_M]]``.

Leverages pylinkage's topology catalog and co-optimization infrastructure
while evaluating candidates through leggedsnake's physics-based fitness
functions.

Example::

    from leggedsnake import (
        DistanceFitness,
        StabilityFitness,
        NsgaWalkingConfig,
    )
    from leggedsnake.topology_optimization import (
        TopologyCoOptConfig,
        topology_walking_optimization,
    )

    result = topology_walking_optimization(
        objectives=[DistanceFitness(duration=10), StabilityFitness(duration=10)],
        objective_names=["distance", "stability"],
        config=TopologyCoOptConfig(
            max_links=6,
            n_generations=50,
            pop_size=40,
        ),
    )

    for sol in result.pareto_front.solutions:
        print(sol.scores, sol.metadata.get("topology_name"))
"""
from __future__ import annotations

from collections.abc import Sequence
from copy import deepcopy
from dataclasses import dataclass, field
from typing import Any, Literal

import numpy as np

from pylinkage.dimensions import Dimensions, DriverAngle
from pylinkage.optimization.collections import ParetoFront, ParetoSolution

from .fitness import CompositeFitness, DynamicFitness
from .gait_analysis import GaitAnalysisResult, analyze_gait
from .nsga_optimizer import NsgaWalkingConfig, _check_pymoo
from .stability import StabilityTimeSeries
from .walker import Walker


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass
class TopologySolutionInfo:
    """Topology metadata for a single Pareto solution.

    Attributes
    ----------
    topology_name : str
        Human-readable name from catalog (e.g., "Four-bar linkage").
    topology_id : str
        Catalog ID (e.g., "four-bar").
    topology_idx : int
        Index in the catalog used during optimization.
    num_links : int
        Number of links in the topology.
    n_legs : int
        Number of legs in the evaluated walker. Equal to
        ``config.n_legs`` when the leg-count range is fixed, or the
        value chosen by the evolutionary search when
        ``n_legs_min != n_legs_max``.
    phase_offsets : list[float] | None
        Evolved per-leg phase offsets (radians) when
        ``TopologyCoOptConfig.evolve_offsets`` is set. ``None`` when
        offsets are not part of the chromosome — the walker was built
        with the classic evenly-spaced rotating-stack gait. Length
        equals ``n_legs - 1``.
    """

    topology_name: str
    topology_id: str
    topology_idx: int
    num_links: int
    n_legs: int = 1
    phase_offsets: list[float] | None = None


@dataclass
class TopologyWalkingResult:
    """Result of topology co-optimization.

    Extends ``NsgaWalkingResult`` with per-solution topology metadata.

    Attributes
    ----------
    pareto_front : ParetoFront
        Non-dominated solutions.
    topology_info : dict[int, TopologySolutionInfo]
        Topology metadata for each Pareto solution (index -> info).
    gait_analyses : dict[int, GaitAnalysisResult] | None
        Gait analysis per solution.
    stability_series : dict[int, StabilityTimeSeries] | None
        Stability per solution.
    config : NsgaWalkingConfig
        The NSGA configuration used.
    co_opt_config : TopologyCoOptConfig
        The topology co-opt configuration used.
    """

    pareto_front: ParetoFront
    topology_info: dict[int, TopologySolutionInfo] = field(default_factory=dict)
    gait_analyses: dict[int, GaitAnalysisResult] | None = None
    stability_series: dict[int, StabilityTimeSeries] | None = None
    config: NsgaWalkingConfig = field(default_factory=NsgaWalkingConfig)
    co_opt_config: TopologyCoOptConfig = field(default_factory=lambda: TopologyCoOptConfig())

    def best_for_objective(self, objective_index: int) -> ParetoSolution:
        """Return best Pareto solution for a single objective."""
        return min(
            self.pareto_front.solutions,
            key=lambda s: s.scores[objective_index],
        )

    def best_compromise(
        self, weights: Sequence[float] | None = None,
    ) -> ParetoSolution:
        """Return the best compromise solution."""
        return self.pareto_front.best_compromise(weights)

    def solutions_by_topology(self) -> dict[str, list[int]]:
        """Group solution indices by topology ID."""
        groups: dict[str, list[int]] = {}
        for idx, info in self.topology_info.items():
            groups.setdefault(info.topology_id, []).append(idx)
        return groups


@dataclass
class TopologyCoOptConfig:
    """Configuration for topology co-optimization.

    Attributes
    ----------
    max_links : int
        Maximum number of links to consider from catalog.
    n_generations : int
        Number of evolutionary generations.
    pop_size : int
        Population size per generation.
    algorithm : {"nsga2", "nsga3"}
        Multi-objective algorithm.
    seed : int | None
        Random seed for reproducibility.
    verbose : bool
        Print progress.
    n_legs : int
        Fixed number of legs per walker candidate. Applied when
        ``n_legs_min`` and ``n_legs_max`` are both unset (the default).
    n_legs_min, n_legs_max : int | None
        If both are set and differ, leg count joins the chromosome as
        an integer gene sampled from ``[n_legs_min, n_legs_max]``.
        When equal (or either is None), the fixed ``n_legs`` is used.
    motor_rates : float | dict[str, float]
        Motor angular velocities.
    dimension_lower : float
        Lower bound for link lengths.
    dimension_upper : float
        Upper bound for link lengths.
    topology_mutation_rate : float
        Probability of mutating the topology gene.
    evolve_offsets : bool
        If True, append per-leg phase-offset genes (radians, bounded
        ``[0, tau)``) to the chromosome so the optimizer co-evolves
        gait pattern alongside topology + dimensions. The offset
        region is sized for ``max(leg_bounds) - 1`` so the chromosome
        length stays fixed under a variable ``n_legs``; surplus
        offset genes are ignored when a candidate uses fewer legs
        (mirroring the existing dimension-padding scheme). Requires
        ``max(leg_bounds) >= 2``. Default ``False`` keeps the
        classical evenly-spaced rotating-stack gait of
        ``Walker.add_legs(n)``.
    """

    max_links: int = 8
    n_generations: int = 100
    pop_size: int = 100
    algorithm: Literal["nsga2", "nsga3"] = "nsga2"
    seed: int | None = None
    verbose: bool = True
    n_legs: int = 2
    n_legs_min: int | None = None
    n_legs_max: int | None = None
    motor_rates: float | dict[str, float] = -4.0
    dimension_lower: float = 0.3
    dimension_upper: float = 5.0
    topology_mutation_rate: float = 0.1
    evolve_offsets: bool = False
    n_workers: int = 1
    """Number of parallel workers. 1 = sequential. >1 uses process pool."""

    def __post_init__(self) -> None:
        if self.evolve_offsets and max(self.leg_bounds) < 2:
            raise ValueError(
                "evolve_offsets=True requires max(leg_bounds) >= 2 — "
                "a single-leg walker has no phase offsets to evolve."
            )

    @property
    def leg_gene_active(self) -> bool:
        """True when leg count joins the chromosome as a variable gene."""
        return (
            self.n_legs_min is not None
            and self.n_legs_max is not None
            and self.n_legs_min < self.n_legs_max
        )

    @property
    def leg_bounds(self) -> tuple[int, int]:
        """Effective (low, high) leg-count bounds. Inclusive."""
        if self.n_legs_min is not None and self.n_legs_max is not None:
            return int(self.n_legs_min), int(self.n_legs_max)
        fixed = int(self.n_legs)
        return fixed, fixed

    @property
    def n_offset_genes(self) -> int:
        """Number of phase-offset genes appended to the chromosome.

        Zero when ``evolve_offsets`` is False. Otherwise sized for the
        upper bound on leg count (``max(leg_bounds) - 1``) so the
        chromosome remains fixed-length under a variable ``n_legs``;
        only the first ``current_n_legs - 1`` offsets are read per
        evaluation and the rest are ignored.
        """
        if not self.evolve_offsets:
            return 0
        return max(0, max(self.leg_bounds) - 1)


# ---------------------------------------------------------------------------
# Topology context: manages catalog → chromosome mapping
# ---------------------------------------------------------------------------


class _TopologyContext:
    """Maps between catalog topologies and chromosome encoding.

    Each topology has a variable number of edges (link lengths).
    Chromosomes are zero-padded to the maximum edge count so pymoo
    can operate on fixed-length arrays.
    """

    def __init__(
        self,
        max_links: int = 8,
        catalog: Any | None = None,
    ) -> None:
        from pylinkage.topology.catalog import TopologyCatalog

        if catalog is None:
            catalog = TopologyCatalog.load_builtin()

        entries = catalog.compatible_topologies(max_links=max_links)
        self.entries = sorted(entries, key=lambda e: (e.num_links, e.id))
        self.n_topologies = len(self.entries)

        if self.n_topologies == 0:
            raise ValueError(
                f"No topologies found with max_links={max_links}"
            )

        # Count edges per topology (continuous variables)
        self._edges_per_topo: list[int] = []
        for entry in self.entries:
            graph = entry.to_graph()
            n_edges = len(graph.edges)
            self._edges_per_topo.append(n_edges)

        self.max_edges = max(self._edges_per_topo)

    def n_edges(self, topo_idx: int) -> int:
        """Number of continuous variables for a topology."""
        return self._edges_per_topo[topo_idx]

    def build_walker(
        self,
        topo_idx: int,
        dimensions: list[float],
        motor_rates: float | dict[str, float] = -4.0,
    ) -> Walker | None:
        """Build a Walker from a topology index and dimensions.

        Returns None if construction fails.
        """
        topo_idx = max(0, min(topo_idx, self.n_topologies - 1))
        entry = self.entries[topo_idx]
        graph = entry.to_graph()

        n_edges = self._edges_per_topo[topo_idx]
        dims_to_use = dimensions[:n_edges]

        # Build Dimensions from graph structure
        try:
            dim_obj = self._build_dimensions(graph, dims_to_use, motor_rates)
        except Exception:
            return None

        try:
            walker = Walker(
                deepcopy(graph), dim_obj,
                name=entry.name,
                motor_rates=motor_rates,
            )
            # Verify buildable
            list(walker.step(iterations=1))
            return walker
        except Exception:
            return None

    def _build_dimensions(
        self,
        graph: Any,
        edge_lengths: list[float],
        motor_rates: float | dict[str, float],
    ) -> Dimensions:
        """Build Dimensions for a graph given edge lengths.

        Places ground nodes along x-axis, driver nodes at offset,
        and leaves driven nodes to be solved by kinematic simulation.
        """
        from math import tau

        node_positions: dict[str, tuple[float, float]] = {}
        edge_distances: dict[str, float] = {}
        driver_angles: dict[str, DriverAngle] = {}

        # Assign edge distances
        edges = list(graph.edges.values())
        for i, edge in enumerate(edges):
            if i < len(edge_lengths):
                edge_distances[edge.id] = abs(edge_lengths[i])
            else:
                edge_distances[edge.id] = 1.0

        # Place ground nodes along x-axis
        ground_ids = [n.id for n in graph.ground_nodes()]
        for i, gid in enumerate(ground_ids):
            node_positions[gid] = (i * 2.0, 0.0)

        # Place driver nodes
        driver_ids = [n.id for n in graph.driver_nodes()]
        if isinstance(motor_rates, dict):
            rate_map = motor_rates
        else:
            rate_map = {did: float(motor_rates) for did in driver_ids}

        for i, did in enumerate(driver_ids):
            # Find edge connecting driver to its anchor
            anchor_pos = (0.0, 0.0)
            radius = 1.0
            for eid, edge in graph.edges.items():
                if edge.source == did or edge.target == did:
                    other = edge.target if edge.source == did else edge.source
                    if other in node_positions:
                        anchor_pos = node_positions[other]
                        radius = edge_distances.get(eid, 1.0)
                        break

            node_positions[did] = (
                anchor_pos[0] + radius,
                anchor_pos[1],
            )
            rate = rate_map.get(did, -4.0)
            driver_angles[did] = DriverAngle(
                angular_velocity=tau / 12 if rate >= 0 else -tau / 12,
            )

        # Place remaining nodes at approximate positions
        placed = set(node_positions.keys())
        remaining = [nid for nid in graph.nodes if nid not in placed]

        for nid in remaining:
            # Find connected nodes that are placed
            connected_placed = []
            for edge in graph.edges.values():
                if edge.source == nid and edge.target in placed:
                    connected_placed.append(
                        (node_positions[edge.target], edge_distances.get(edge.id, 1.0))
                    )
                elif edge.target == nid and edge.source in placed:
                    connected_placed.append(
                        (node_positions[edge.source], edge_distances.get(edge.id, 1.0))
                    )

            if len(connected_placed) >= 2:
                # Circle-circle intersection approximation
                p1, r1 = connected_placed[0]
                p2, r2 = connected_placed[1]
                mid_x = (p1[0] + p2[0]) / 2
                mid_y = (p1[1] + p2[1]) / 2
                node_positions[nid] = (mid_x, mid_y + (r1 + r2) / 2)
            elif connected_placed:
                p, r = connected_placed[0]
                node_positions[nid] = (p[0] + r * 0.7, p[1] + r * 0.7)
            else:
                node_positions[nid] = (1.0, 1.0)

            placed.add(nid)

        return Dimensions(
            node_positions=node_positions,
            driver_angles=driver_angles,
            edge_distances=edge_distances,
        )


# ---------------------------------------------------------------------------
# Pymoo Problem for mixed topology + dimensions
# ---------------------------------------------------------------------------


class _TopologyWalkingProblem:
    """Pymoo Problem for topology co-optimization.

    Chromosome layout:
      * ``[topology_idx, dim_1, ..., dim_max]`` (default — length
        ``1 + max_edges``)
      * ``[topology_idx, n_legs, dim_1, ..., dim_max]`` when
        :attr:`TopologyCoOptConfig.leg_gene_active` (length
        ``2 + max_edges``)
      * ``[topology_idx, [n_legs,] dim_1, ..., dim_max,
        off_1, ..., off_M]`` when
        :attr:`TopologyCoOptConfig.evolve_offsets` is also set, with
        ``M = max(leg_bounds) - 1`` (length ``... + M``)

    Integer genes (topology index and, when present, leg count) are
    rounded and clamped. Dimension genes for topologies with fewer
    than ``max_edges`` links are ignored, and offset genes beyond
    ``current_n_legs - 1`` are likewise ignored — the same
    fixed-length-padded encoding scheme on both ends of the
    chromosome.
    """

    def __init__(
        self,
        ctx: _TopologyContext,
        objectives: Sequence[DynamicFitness],
        config: TopologyCoOptConfig,
        world_config: Any | None = None,
    ) -> None:
        _check_pymoo()
        from math import tau

        from pymoo.core.problem import Problem

        self.ctx = ctx
        self.objectives = list(objectives)
        self.world_config = world_config
        self._config = config
        self._n_obj = len(objectives)

        leg_active = config.leg_gene_active
        n_leg_genes = 1 if leg_active else 0
        n_offset_genes = config.n_offset_genes
        n_var = 1 + n_leg_genes + ctx.max_edges + n_offset_genes
        xl = np.zeros(n_var, dtype=float)
        xu = np.ones(n_var, dtype=float)
        xl[0] = 0.0
        xu[0] = float(ctx.n_topologies - 1)
        if leg_active:
            lo, hi = config.leg_bounds
            xl[1] = float(lo)
            xu[1] = float(hi)
        dim_start = 1 + n_leg_genes
        dim_end = dim_start + ctx.max_edges
        xl[dim_start:dim_end] = config.dimension_lower
        xu[dim_start:dim_end] = config.dimension_upper
        if n_offset_genes > 0:
            xl[dim_end:] = 0.0
            xu[dim_end:] = float(tau)

        outer = self

        class _Problem(Problem):  # type: ignore[misc]  # pymoo Problem is Any-typed
            def __init__(self_inner) -> None:
                super().__init__(
                    n_var=n_var,
                    n_obj=outer._n_obj,
                    n_ieq_constr=0,
                    xl=xl,
                    xu=xu,
                )

            def _evaluate(
                self_inner,
                X: np.ndarray,
                out: dict[str, Any],
                *args: Any,
                **kwargs: Any,
            ) -> None:
                out["F"] = outer._evaluate_batch(X)

        self._problem = _Problem()

    @property
    def problem(self) -> Any:
        return self._problem

    def _evaluate_batch(self, X: np.ndarray) -> np.ndarray:
        """Evaluate batch, optionally in parallel."""
        n_pop = X.shape[0]
        F = np.full((n_pop, self._n_obj), float("inf"))
        n_workers = max(1, self._config.n_workers)

        if n_workers <= 1:
            for i in range(n_pop):
                scores = self._evaluate_candidate(X[i])
                F[i] = [-s for s in scores]
        else:
            from concurrent.futures import ProcessPoolExecutor, as_completed

            with ProcessPoolExecutor(max_workers=n_workers) as pool:
                futures = {
                    pool.submit(
                        _topology_evaluate_worker,
                        self.ctx,
                        self.objectives,
                        self._config,
                        self.world_config,
                        X[i].copy(),
                    ): i
                    for i in range(n_pop)
                }
                for future in as_completed(futures):
                    idx = futures[future]
                    try:
                        scores = future.result()
                        F[idx] = [-s for s in scores]
                    except Exception:
                        F[idx] = [float("inf")] * self._n_obj

        return F

    def _evaluate_candidate(self, x: np.ndarray) -> list[float]:
        """Evaluate a single mixed chromosome."""
        topo_idx, n_legs, dims, offsets = _decode_chromosome(
            x, self._config, max_edges=self.ctx.max_edges,
        )

        walker = self.ctx.build_walker(
            topo_idx, dims, self._config.motor_rates,
        )
        if walker is None:
            return [0.0] * self._n_obj

        if not _apply_legs(walker, n_legs, offsets):
            return [0.0] * self._n_obj

        scores: list[float] = []
        for objective in self.objectives:
            try:
                result = objective(
                    deepcopy(walker.topology),
                    deepcopy(walker.dimensions),
                    self.world_config,
                )
                scores.append(result.score if result.valid else 0.0)
            except Exception:
                scores.append(0.0)

        return scores


# ---------------------------------------------------------------------------
# Top-level worker for parallel evaluation (must be picklable)
# ---------------------------------------------------------------------------


def _decode_chromosome(
    x: np.ndarray,
    config: TopologyCoOptConfig,
    max_edges: int | None = None,
) -> tuple[int, int, list[float], list[float] | None]:
    """Extract ``(topo_idx, n_legs, dimensions, offsets)`` from a chromosome.

    Mirrors the bounds and layout set up in
    :class:`_TopologyWalkingProblem`. Works for the fixed-leg default
    layout, the leg-gene layout, and either with the optional
    ``evolve_offsets`` tail.

    Parameters
    ----------
    x : np.ndarray
        Encoded chromosome from pymoo.
    config : TopologyCoOptConfig
        Provides leg-bound and offset-region sizing.
    max_edges : int, optional
        Number of dimension genes. Required when
        ``config.evolve_offsets`` is True so the decoder can split the
        dimension region from the offset region. When ``None`` and
        ``evolve_offsets`` is False, the entire post-leg suffix is
        treated as dimensions (preserves the historical signature).

    Returns
    -------
    tuple
        ``(topology_index, n_legs, dimensions, offsets)``. ``offsets``
        is ``None`` when ``evolve_offsets`` is False — callers fall
        back to ``Walker.add_legs(n_legs - 1)`` for the classical
        evenly-spaced rotating-stack gait. Otherwise it is the first
        ``n_legs - 1`` offsets read from the chromosome's offset
        region (any surplus genes are dropped).
    """
    topo_idx = int(round(float(x[0])))
    leg_offset = 1
    if config.leg_gene_active:
        lo, hi = config.leg_bounds
        n_legs = max(lo, min(hi, int(round(float(x[1])))))
        leg_offset = 2
    else:
        # Fixed leg count — prefer the collapsed range bounds when the
        # user set them explicitly (handles n_legs_min==n_legs_max), else
        # fall back to the scalar ``n_legs`` field.
        n_legs = config.leg_bounds[0]

    if config.evolve_offsets:
        if max_edges is None:
            raise ValueError(
                "_decode_chromosome requires max_edges when "
                "config.evolve_offsets is True — needed to split the "
                "dimension region from the offset region."
            )
        dim_end = leg_offset + max_edges
        dims = x[leg_offset:dim_end].tolist()
        # Read only as many offsets as the candidate's leg count
        # actually needs; surplus genes are padding.
        n_used = max(0, n_legs - 1)
        offsets: list[float] | None = x[dim_end:dim_end + n_used].tolist()
    else:
        dims = x[leg_offset:].tolist()
        offsets = None
    return topo_idx, n_legs, dims, offsets


def _apply_legs(
    walker: Walker,
    n_legs: int,
    offsets: list[float] | None,
) -> bool:
    """Add legs to a walker, returning False on failure.

    When ``offsets`` is None the classical evenly-spaced gait is used
    (``add_legs(n_legs - 1)``). When offsets are provided the explicit
    sequence drives the gait; the optimizer evolves these alongside
    topology + dimensions.
    """
    if n_legs <= 1:
        return True
    try:
        if offsets is None:
            walker.add_legs(n_legs - 1)
        else:
            walker.add_legs(offsets)
    except Exception:
        return False
    return True


def _topology_evaluate_worker(
    ctx: _TopologyContext,
    objectives: list[DynamicFitness],
    config: TopologyCoOptConfig,
    world_config: Any | None,
    x: np.ndarray,
) -> list[float]:
    """Evaluate a single topology chromosome in a worker process."""
    topo_idx, n_legs, dims, offsets = _decode_chromosome(
        x, config, max_edges=ctx.max_edges,
    )

    walker = ctx.build_walker(topo_idx, dims, config.motor_rates)
    if walker is None:
        return [0.0] * len(objectives)

    if not _apply_legs(walker, n_legs, offsets):
        return [0.0] * len(objectives)

    scores: list[float] = []
    for objective in objectives:
        try:
            result = objective(
                deepcopy(walker.topology),
                deepcopy(walker.dimensions),
                world_config,
            )
            scores.append(result.score if result.valid else 0.0)
        except Exception:
            scores.append(0.0)

    return scores


# ---------------------------------------------------------------------------
# High-level entry point
# ---------------------------------------------------------------------------


def topology_walking_optimization(
    objectives: Sequence[DynamicFitness],
    objective_names: Sequence[str] | None = None,
    config: TopologyCoOptConfig | None = None,
    world_config: Any | None = None,
    catalog: Any | None = None,
    include_gait: bool = False,
    include_stability: bool = False,
) -> TopologyWalkingResult:
    """Topology co-optimization for walking mechanisms.

    Jointly optimizes the mechanism topology (from pylinkage's catalog)
    and link dimensions using NSGA-II/III with physics-based fitness
    evaluation.

    .. note::
       This function reimplements the NSGA-II + mixed-chromosome plumbing
       that now ships in ``pylinkage.optimization.co_optimize``. The
       pylinkage-backed path — :func:`leggedsnake.optimize_walking_mechanism`
       — delegates the optimizer loop to pylinkage and wraps walking
       fitness through :func:`leggedsnake.fitness.co_optimize_objective`.
       Prefer it for new code. This standalone implementation is kept
       for backwards compatibility and will be deprecated once the
       pylinkage-backed path reaches feature parity (multi-process
       evaluation, gait / stability post-analysis).

    Parameters
    ----------
    objectives : sequence of DynamicFitness
        Fitness evaluators (e.g., ``DistanceFitness``, ``StabilityFitness``).
    objective_names : sequence of str, optional
        Human-readable names for each objective.
    config : TopologyCoOptConfig, optional
        Optimization configuration. Uses defaults if None.
    world_config : WorldConfig, optional
        Simulation config passed to fitness evaluators.
    catalog : TopologyCatalog, optional
        Topology catalog. Uses built-in catalog if None.
    include_gait : bool
        If True, run gait analysis on Pareto-front solutions.
    include_stability : bool
        If True, collect stability time series for Pareto-front solutions.

    Returns
    -------
    NsgaWalkingResult
        Pareto front with optional gait/stability analysis.
        Each ``ParetoSolution.metadata`` contains ``topology_name``
        and ``topology_id`` identifying the mechanism type.
    """
    _check_pymoo()
    from pymoo.algorithms.moo.nsga2 import NSGA2
    from pymoo.optimize import minimize as pymoo_minimize

    cfg = config or TopologyCoOptConfig()
    n_obj = len(objectives)

    if objective_names is None:
        objective_names = [f"objective_{i}" for i in range(n_obj)]

    # Build topology context
    ctx = _TopologyContext(max_links=cfg.max_links, catalog=catalog)

    if cfg.verbose:
        print(f"Topology co-optimization: {ctx.n_topologies} topologies, "
              f"{ctx.max_edges} max edges")
        for i, entry in enumerate(ctx.entries):
            print(f"  [{i}] {entry.name} ({entry.num_links} links, "
                  f"{ctx.n_edges(i)} edges)")

    # Build problem
    problem = _TopologyWalkingProblem(
        ctx=ctx,
        objectives=objectives,
        config=cfg,
        world_config=world_config,
    )

    # Build algorithm
    if cfg.algorithm == "nsga3" and n_obj > 2:
        from pymoo.algorithms.moo.nsga3 import NSGA3
        from pymoo.util.ref_dirs import get_reference_directions

        n_partitions = max(4, 12 - n_obj)
        ref_dirs = get_reference_directions(
            "das-dennis", n_obj, n_partitions=n_partitions,
        )
        algo = NSGA3(pop_size=cfg.pop_size, ref_dirs=ref_dirs)
    else:
        algo = NSGA2(pop_size=cfg.pop_size)

    # Run optimization
    res = pymoo_minimize(
        problem.problem,
        algo,
        ("n_gen", cfg.n_generations),
        seed=cfg.seed,
        verbose=cfg.verbose,
    )

    # Package results
    nsga_cfg = NsgaWalkingConfig(
        n_generations=cfg.n_generations,
        pop_size=cfg.pop_size,
        algorithm=cfg.algorithm,
        seed=cfg.seed,
        verbose=cfg.verbose,
    )

    if res.F is None or res.X is None:
        return TopologyWalkingResult(
            pareto_front=ParetoFront([], tuple(objective_names)),
            config=nsga_cfg,
            co_opt_config=cfg,
        )

    solutions: list[ParetoSolution] = []
    topo_info: dict[int, TopologySolutionInfo] = {}
    # Normalize to (n_solutions, n_objectives). pymoo may return 1-D for
    # a single solution (shape becomes (n_obj,)) or for a single-objective
    # multi-solution run (shape (n_sol,)). Reshape against the known
    # objective count to disambiguate.
    n_obj = len(objective_names)
    F = np.asarray(res.F).reshape(-1, n_obj)
    X = np.asarray(res.X).reshape(F.shape[0], -1)
    for i in range(F.shape[0]):
        scores = tuple(-float(v) for v in F[i])
        topo_idx, n_legs, _, offsets = _decode_chromosome(
            X[i], cfg, max_edges=ctx.max_edges,
        )
        topo_idx = max(0, min(topo_idx, ctx.n_topologies - 1))
        entry = ctx.entries[topo_idx]

        solutions.append(ParetoSolution(
            scores=scores,
            dimensions=X[i],
            init_positions=[(0.0, 0.0)],
        ))
        topo_info[i] = TopologySolutionInfo(
            topology_name=entry.name,
            topology_id=entry.id,
            topology_idx=topo_idx,
            num_links=entry.num_links,
            n_legs=n_legs,
            phase_offsets=offsets,
        )

    pareto = ParetoFront(solutions, tuple(objective_names))

    # Optional post-hoc analysis
    gait_analyses: dict[int, GaitAnalysisResult] | None = None
    stability_map: dict[int, StabilityTimeSeries] | None = None

    if include_gait or include_stability:
        gait_analyses = {} if include_gait else None
        stability_map = {} if include_stability else None

        for idx, sol in enumerate(solutions):
            topo_idx, n_legs, dims, offsets = _decode_chromosome(
                sol.dimensions, cfg, max_edges=ctx.max_edges,
            )
            walker = ctx.build_walker(topo_idx, dims, cfg.motor_rates)
            if walker is None:
                continue

            if not _apply_legs(walker, n_legs, offsets):
                continue

            composite = CompositeFitness(
                objectives=("distance", "efficiency", "stability"),
                n_legs=1,  # legs already added
            )
            result = composite(
                deepcopy(walker.topology),
                deepcopy(walker.dimensions),
                world_config,
            )

            if include_gait and result.valid and result.loci:
                foot_ids = walker.get_feet()
                gait = analyze_gait(
                    loci=result.loci,
                    foot_ids=foot_ids,
                    dt=0.02,
                )
                gait_analyses[idx] = gait  # type: ignore[index]

            if include_stability and result.valid:
                from .fitness import _run_simulation
                stab_result = _run_simulation(
                    deepcopy(walker.topology),
                    deepcopy(walker.dimensions),
                    world_config,
                    duration=composite.duration,
                    n_legs=1,
                    motor_rates=composite.motor_rates,
                    record_loci=False,
                    record_stability=True,
                )
                if stab_result.stability is not None:
                    stability_map[idx] = stab_result.stability  # type: ignore[index]

    return TopologyWalkingResult(
        pareto_front=pareto,
        topology_info=topo_info,
        gait_analyses=gait_analyses,
        stability_series=stability_map,
        config=nsga_cfg,
        co_opt_config=cfg,
    )
