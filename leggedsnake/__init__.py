#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Main module of the leggedsnake library, to build walking machines easily.

Please see https://hugofara.github.io/leggedsnake/ for complete documentation.
A local copy should have been shipped with this package under the docs/ folder.

Created on Thu Jun 10 20:35:00 2021

@author: HugoFara
"""

# Pylinkage is a sister project and some kind of backend for leggedsnake
from pylinkage import (
    bounding_box, generate_bounds,
    Static, Fixed, Pivot, Crank,
    Linkage,
    trials_and_errors_optimization, particle_swarm_optimization,
    kinematic_minimization, kinematic_maximization,
    UnbuildableError, HypostaticError,
    show_linkage,
)

from .utility import step, stride
from .walker import Walker
from .geneticoptimizer import (
    evolutionary_optimization, GeneticOptimization
)
from .dynamiclinkage import (
    Nail, PinUp, DynamicPivot, Motor,
    DynamicLinkage,
    convert_to_dynamic_linkage,
)
from .physicsengine import (
    video,
    video_debug,
    all_linkages_video,
    params,
    World,
    VisualWorld
)
from .show_evolution import load_data, show_genetic_optimization

__version__ = "0.3.1"
