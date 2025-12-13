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

# Pylinkage is a sister project and some kind of backend for leggedsnake
from pylinkage import (
    Crank,
    Fixed,
    HypostaticError,
    Linkage,
    Pivot,  # Deprecated, use Revolute instead
    Revolute,
    Static,
    UnbuildableError,
    bounding_box,
    generate_bounds,
    kinematic_maximization,
    kinematic_minimization,
    particle_swarm_optimization,
    show_linkage,
    trials_and_errors_optimization,
)

from .dynamiclinkage import (
    DynamicLinkage,
    DynamicPivot,
    Motor,
    Nail,
    PinUp,
    convert_to_dynamic_linkage,
)
from .geneticoptimizer import GeneticOptimization
from .physicsengine import World, params
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
    # pylinkage re-exports
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
    # utility
    "step",
    "stride",
    # walker
    "Walker",
    # geneticoptimizer
    "GeneticOptimization",
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
    # show_evolution
    "load_data",
    "show_genetic_optimization",
]
