"""
Topology co-optimization for walking mechanisms.

Jointly optimizes the discrete mechanism topology (four-bar, six-bar,
eight-bar variants) alongside continuous link dimensions using NSGA-II.
Each candidate is a mixed chromosome: ``[topology_index, dim_1, ..., dim_N]``.

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

from .fitness import CompositeFitness, DynamicFitness, FitnessResult
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
    """

    topology_name: str
    topology_id: str
    topology_idx: int
    num_links: int


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
        Number of legs per walker candidate.
    motor_rates : float | dict[str, float]
        Motor angular velocities.
    dimension_lower : float
        Lower bound for link lengths.
    dimension_upper : float
        Upper bound for link lengths.
    topology_mutation_rate : float
        Probability of mutating the topology gene.
    """

    max_links: int = 8
    n_generations: int = 100
    pop_size: int = 100
    algorithm: Literal["nsga2", "nsga3"] = "nsga2"
    seed: int | None = None
    verbose: bool = True
    n_legs: int = 2
    motor_rates: float | dict[str, float] = -4.0
    dimension_lower: float = 0.3
    dimension_upper: float = 5.0
    topology_mutation_rate: float = 0.1


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

    Chromosome: ``[topology_index, dim_1, ..., dim_max_edges]``
    where ``topology_index`` is rounded to the nearest integer.
    Unused dimension genes (for simpler topologies) are ignored.
    """

    def __init__(
        self,
        ctx: _TopologyContext,
        objectives: Sequence[DynamicFitness],
        config: TopologyCoOptConfig,
        world_config: Any | None = None,
    ) -> None:
        _check_pymoo()
        from pymoo.core.problem import Problem

        self.ctx = ctx
        self.objectives = list(objectives)
        self.world_config = world_config
        self._config = config
        self._n_obj = len(objectives)

        n_var = 1 + ctx.max_edges  # topology_idx + dimensions
        xl = np.zeros(n_var, dtype=float)
        xu = np.ones(n_var, dtype=float)
        xl[0] = 0.0
        xu[0] = float(ctx.n_topologies - 1)
        xl[1:] = config.dimension_lower
        xu[1:] = config.dimension_upper

        outer = self

        class _Problem(Problem):
            def __init__(self_inner) -> None:
                super().__init__(
                    n_var=n_var,
                    n_obj=outer._n_obj,
                    n_ieq_constr=0,
                    xl=xl,
                    xu=xu,
                )

            def _evaluate(
                self_inner, X: np.ndarray, out: dict, *args: Any, **kwargs: Any,
            ) -> None:
                F = np.full((X.shape[0], outer._n_obj), float("inf"))
                for i in range(X.shape[0]):
                    scores = outer._evaluate_candidate(X[i])
                    F[i] = [-s for s in scores]
                out["F"] = F

        self._problem = _Problem()

    @property
    def problem(self) -> Any:
        return self._problem

    def _evaluate_candidate(self, x: np.ndarray) -> list[float]:
        """Evaluate a single mixed chromosome."""
        topo_idx = int(round(x[0]))
        dims = x[1:].tolist()

        walker = self.ctx.build_walker(
            topo_idx, dims, self._config.motor_rates,
        )
        if walker is None:
            return [0.0] * self._n_obj

        # Add legs
        if self._config.n_legs > 1:
            try:
                walker.add_legs(self._config.n_legs - 1)
            except Exception:
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
    # Ensure F is 2D (single-objective pymoo may return 1D)
    F = np.atleast_2d(res.F)
    X = np.atleast_2d(res.X)
    for i in range(F.shape[0]):
        scores = tuple(-float(v) for v in F[i])
        topo_idx = int(round(X[i][0]))
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
        )

    pareto = ParetoFront(solutions, tuple(objective_names))

    # Optional post-hoc analysis
    gait_analyses: dict[int, GaitAnalysisResult] | None = None
    stability_map: dict[int, StabilityTimeSeries] | None = None

    if include_gait or include_stability:
        gait_analyses = {} if include_gait else None
        stability_map = {} if include_stability else None

        for idx, sol in enumerate(solutions):
            topo_idx = int(round(sol.dimensions[0]))
            dims = sol.dimensions[1:].tolist()
            walker = ctx.build_walker(topo_idx, dims, cfg.motor_rates)
            if walker is None:
                continue

            if cfg.n_legs > 1:
                try:
                    walker.add_legs(cfg.n_legs - 1)
                except Exception:
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
