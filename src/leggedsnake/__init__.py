#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
LeggedSnake: simulate and optimize planar walking linkages.

Please see https://hugofara.github.io/leggedsnake/ for complete documentation.
"""
from __future__ import annotations

from pylinkage import (
    UnbuildableError,
    HypostaticError,
    generate_bounds,
    kinematic_maximization,
    kinematic_minimization,
    particle_swarm_optimization,
    trials_and_errors_optimization,
)
from pylinkage.optimization import (
    chain_optimizers,
    differential_evolution_optimization,
    dual_annealing_optimization,
    minimize_linkage,
    multi_objective_optimization,
    OptimizationProgress,
    ParetoFront,
    ParetoSolution,
    # Async variants
    differential_evolution_optimization_async,
    minimize_linkage_async,
    particle_swarm_optimization_async,
    trials_and_errors_optimization_async,
)
from pylinkage.optimization.collections import Agent, MutableAgent

# Hypergraph API re-exports (primary construction API)
from pylinkage.hypergraph import (
    HypergraphLinkage,
    Node,
    Edge,
    Hyperedge,
    NodeRole,
)
from pylinkage.dimensions import Dimensions, DriverAngle

from .dynamiclinkage import DynamicLinkage, convert_to_dynamic_linkage
from .co_design import (
    WalkingDesignResult,
    WalkingDesignSpec,
    optimize_walking_mechanism,
)
from .fitness import (
    DynamicFitness,
    DistanceFitness,
    EfficiencyFitness,
    FitnessResult,
    StrideFitness,
    as_eval_func,
    as_ga_fitness,
    co_optimize_objective,
)
from .geneticoptimizer import GeneticOptimization, genetic_algorithm_optimization
from .physicsengine import DEFAULT_CONFIG, TerrainConfig, World, WorldConfig, params
from .walking_objectives import (
    energy_efficiency_objective,
    multi_objective_walking_optimization,
    stride_length_objective,
    total_distance_objective,
)
from .show_evolution import load_data, show_genetic_optimization
from .utility import step, stride
from .walker import Walker, walker_from_legacy

__version__ = "0.5.0"

# Lazy imports for modules that require a display (pyglet)
_VISUALIZER_NAMES = frozenset({
    "CAMERA", "VisualWorld", "all_linkages_video", "video", "video_debug",
})


def __getattr__(name: str):
    if name in _VISUALIZER_NAMES:
        from . import worldvisualizer as _wv
        # Populate module namespace so subsequent access is direct
        globals().update({n: getattr(_wv, n) for n in _VISUALIZER_NAMES})
        return globals()[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

__all__ = [
    # pylinkage re-exports
    "generate_bounds",
    "trials_and_errors_optimization",
    "particle_swarm_optimization",
    "kinematic_minimization",
    "kinematic_maximization",
    "UnbuildableError",
    "HypostaticError",
    # pylinkage optimization (new)
    "Agent",
    "MutableAgent",
    "chain_optimizers",
    "differential_evolution_optimization",
    "dual_annealing_optimization",
    "minimize_linkage",
    "multi_objective_optimization",
    "OptimizationProgress",
    "ParetoFront",
    "ParetoSolution",
    # Async optimization variants
    "differential_evolution_optimization_async",
    "minimize_linkage_async",
    "particle_swarm_optimization_async",
    "trials_and_errors_optimization_async",
    # Hypergraph API
    "HypergraphLinkage",
    "Node",
    "Edge",
    "Hyperedge",
    "NodeRole",
    "Dimensions",
    "DriverAngle",
    # utility
    "step",
    "stride",
    # walker
    "Walker",
    "walker_from_legacy",
    # dynamiclinkage
    "DynamicLinkage",
    "convert_to_dynamic_linkage",
    # fitness protocol
    "DynamicFitness",
    "DistanceFitness",
    "EfficiencyFitness",
    "FitnessResult",
    "StrideFitness",
    "as_eval_func",
    "as_ga_fitness",
    "co_optimize_objective",
    # co_design
    "WalkingDesignSpec",
    "WalkingDesignResult",
    "optimize_walking_mechanism",
    # geneticoptimizer
    "GeneticOptimization",
    "genetic_algorithm_optimization",
    # physicsengine
    "DEFAULT_CONFIG",
    "params",
    "TerrainConfig",
    "World",
    "WorldConfig",
    # worldvisualizer
    "all_linkages_video",
    "video",
    "video_debug",
    "VisualWorld",
    "CAMERA",
    # walking_objectives
    "stride_length_objective",
    "energy_efficiency_objective",
    "total_distance_objective",
    "multi_objective_walking_optimization",
    # show_evolution
    "load_data",
    "show_genetic_optimization",
]
