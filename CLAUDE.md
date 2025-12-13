# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

LeggedSnake is a Python library for simulating and optimizing planar walking linkages (leg mechanisms). It combines kinematic analysis from pylinkage with dynamic simulation using pymunk physics engine, plus genetic algorithm optimization.

## Commands

```bash
# Install dependencies
uv sync

# Run tests
uv run pytest

# Run a single test
uv run pytest tests/test_utility.py::TestStride::test_minimal_stride

# Run tests with coverage
uv run pytest --cov=leggedsnake

# Lint with ruff
uv run ruff check src/

# Type check with mypy (strict mode)
uv run mypy

# Run examples (must use __main__ guard due to multiprocessing)
uv run python examples/strider.py
```

## Architecture

The library has three main layers:

### 1. Kinematic Layer (pylinkage integration)

- `Walker` (in `walker.py`): Extends pylinkage's `Linkage` class with leg-specific methods
  - `add_legs(n)`: Adds phase-offset copies of the leg along the same axis
  - `add_opposite_leg()`: Creates antisymmetric (mirrored) leg across vertical axis
  - `get_foots()`: Returns terminal joints (feet) - joints without children
  - `motor_rate`: Motor angular velocity in rad/s (default -4.0, clockwise)
- Joint types from pylinkage: `Static`, `Crank`, `Fixed`, `Pivot` (deprecated), `Revolute`

### 2. Dynamic Layer (pymunk integration)

- `dynamiclinkage.py`: Converts kinematic joints to physics-enabled equivalents
  - `DynamicJoint` (ABC) -> `Nail`, `PinUp`, `DynamicPivot`, `Motor`
  - `DynamicLinkage`: A `Linkage` with pymunk bodies and constraints
  - `convert_to_dynamic_linkage()`: Helper to convert kinematic linkage
- `hypergraph_physics.py`: Graph-based physics body generation
  - Converts pylinkage's hypergraph representation to pymunk bodies
  - Handles rigid triangles (Fixed joints) by merging edges into single bodies
  - `PhysicsMapping`: Dataclass tracking edge→body and node→bodies mappings
- `physicsengine.py`: `World` class manages simulation space, road generation, and physics stepping
  - Global `params` dict controls simulation parameters (gravity, torque, ground properties)
  - `set_space_constraints()`: Auto-tunes solver iterations based on constraint count

### 3. Optimization Layer

- `geneticoptimizer.py`: `GeneticOptimization` class implements GA with multiprocessing
  - DNA format: `[fitness_score, dimensions_list, coordinates_list]`
  - Supports checkpoint/resume via `startnstop` parameter (save to JSON)
  - `run(iters, processes)`: Main optimization loop with parallel evaluation
- pylinkage also provides `particle_swarm_optimization` and `trials_and_errors_optimization`

### Visualization

- `worldvisualizer.py`: `VisualWorld` extends `World` with matplotlib/pyglet animation
  - `video(linkage, duration, save)`: Single linkage simulation
  - `all_linkages_video(linkages)`: Multiple linkages racing
  - `video_debug()`: Frame-by-frame with force visualization

### Utility Functions

- `utility.py`: Fitness function helpers for kinematic optimization
  - `stride(locus, height)`: Returns the "step" portion of a foot locus (horizontal travel at low height)
  - `step(locus, height, width)`: Tests if locus can clear an obstacle of given dimensions

## Key Patterns

- Linkages are defined as collections of joints with parent-child relationships
- Kinematic linkages (pylinkage) are automatically converted to dynamic (pymunk) when added to a `World`
- Fitness functions receive DNA and return `(score, initial_positions)` tuples
- The `params` dict in `physicsengine.py` is the central configuration point:
  - `params["simul"]["physics_period"]`: Time step (default 0.02s)
  - `params["linkage"]["torque"]`: Motor max force (default 1000 N·m)
  - `params["linkage"]["load"]`: Frame mass (default 10 kg)
  - `params["ground"]["friction"]`, `slope`, `noise`: Terrain parameters

## Examples

The `examples/` directory contains complete working examples:
- `strider.py`: Full workflow demonstrating Strider linkage with PSO and GA optimization
- `theo_jansen.py`, `klann_linkage.py`, `chebyshev_linkage.py`: Classic mechanisms
- `simple_fourbar.py`, `simple_walker.py`: Minimal examples for getting started
