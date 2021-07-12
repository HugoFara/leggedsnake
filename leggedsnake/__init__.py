#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Main module of leggedsnake library, to build walking machines easily.

Please see https://hugofara.github.io/leggedsnake/ for a complete documentation.
A local copy should have been shipped with this package under the docs folder.

Created on Thu Jun 10 20:35:00 2021

@author: HugoFara
"""

from .utility import step, stride
from .walker import Walker
from .geneticoptimizer import (
    evolutionnary_optimization,
    evolutionnary_optimization_legacy
)
from .dynamiclinkage import (
    Nail, PinUp, DynamicPivot, Motor,
    DynamicLinkage,
    convert_to_dynamic_linkage,
)
from .physicsengine import (
    video, video_debug,
    params,
    World, VisualWorld
)

__version__ = "0.1.3"