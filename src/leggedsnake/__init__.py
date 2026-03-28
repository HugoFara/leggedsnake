#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Main module of the leggedsnake library, to build walking machines easily.

Please see https://hugofara.github.io/leggedsnake/ for complete documentation.
A local copy should have been shipped with this package under the docs/ folder.

Created on Thu Jun 10 20:35:00 2021

@author: HugoFara
"""
from __future__ import annotations

import warnings

# Pylinkage is a sister project and some kind of backend for leggedsnake.
# pylinkage 0.8.0 deprecated the joints module in favor of components/actuators/dyads.
# leggedsnake still requires legacy joint classes for DynamicJoint inheritance;
# suppress the deprecation warning until the dynamic layer is fully migrated.
with warnings.catch_warnings():
    warnings.filterwarnings(
        "ignore", category=DeprecationWarning, message=r"pylinkage\.joints"
    )
    from pylinkage import (
        Crank,
        Fixed,
        Pivot,  # Deprecated, use Revolute instead
        Revolute,
        Static,
    )

from pylinkage import (
    HypostaticError,
    Linkage,
    UnbuildableError,
    bounding_box,
    generate_bounds,
    kinematic_maximization,
    kinematic_minimization,
    particle_swarm_optimization,
    show_linkage,
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

# New pylinkage 0.8.0 API re-exports for forward compatibility
from pylinkage.components import Ground
from pylinkage.dyads import FixedDyad, RRRDyad

from .dynamiclinkage import (
    DynamicLinkage,
    DynamicPivot,
    Motor,
    Nail,
    PinUp,
    convert_to_dynamic_linkage,
)
from .geneticoptimizer import GeneticOptimization, genetic_algorithm_optimization
from .physicsengine import World, params
from .walking_objectives import (
    energy_efficiency_objective,
    multi_objective_walking_optimization,
    stride_length_objective,
    total_distance_objective,
)
from .show_evolution import load_data, show_genetic_optimization
from .utility import step, stride
from .walker import Walker
from .worldvisualizer import (
    CAMERA,
    VisualWorld,
    all_linkages_video,
    video,
    video_debug,
)

__version__ = "0.4.0"

__all__ = [
    # pylinkage re-exports (legacy joint names)
    "bounding_box",
    "generate_bounds",
    "Static",
    "Fixed",
    "Pivot",  # Deprecated, use Revolute
    "Revolute",
    "Crank",
    "Linkage",
    "trials_and_errors_optimization",
    "particle_swarm_optimization",
    "kinematic_minimization",
    "kinematic_maximization",
    "UnbuildableError",
    "HypostaticError",
    "show_linkage",
    # pylinkage 0.8.0 new API re-exports
    "Ground",
    "FixedDyad",
    "RRRDyad",
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
    # utility
    "step",
    "stride",
    # walker
    "Walker",
    # geneticoptimizer
    "GeneticOptimization",
    "genetic_algorithm_optimization",
    # dynamiclinkage
    "Nail",
    "PinUp",
    "DynamicPivot",
    "Motor",
    "DynamicLinkage",
    "convert_to_dynamic_linkage",
    # physicsengine
    "params",
    "World",
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
