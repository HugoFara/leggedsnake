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

# Build documentation (from docs_src/)
sphinx-build -b html source docs

# Bump version (e.g., minor)
bump2version minor
```

## Architecture

The library has three main layers:

### 1. Kinematic Layer (pylinkage integration)

- `Walker` (in `walker.py`): Extends pylinkage's `Linkage` class with leg-specific methods like `add_legs()` and `get_foots()`
- Re-exports pylinkage joint types: `Static`, `Crank`, `Fixed`, `Pivot`

### 2. Dynamic Layer (pymunk integration)

- `dynamiclinkage.py`: Converts kinematic joints to physics-enabled equivalents
  - `DynamicJoint` (ABC) -> `Nail`, `PinUp`, `DynamicPivot`, `Motor`
  - `DynamicLinkage`: A `Linkage` with pymunk bodies and constraints
- `physicsengine.py`: `World` class manages simulation space, road generation, and physics stepping
  - Global `params` dict controls simulation parameters (gravity, torque, ground properties)

### 3. Optimization Layer

- `geneticoptimizer.py`: `GeneticOptimization` class implements GA with multiprocessing support
  - DNA format: `[fitness_score, dimensions_list, coordinates_list]`
  - Supports checkpoint/resume via `startnstop` parameter

### Visualization

- `worldvisualizer.py`: `VisualWorld` extends `World` with matplotlib animation
  - `video()`, `all_linkages_video()` for viewing simulations

### Utility Functions

- `utility.py`: `stride()` and `step()` functions for evaluating locus quality (used as fitness functions)

## Key Patterns

- Linkages are defined as collections of joints with parent-child relationships
- Kinematic linkages (pylinkage) are automatically converted to dynamic (pymunk) when added to a `World`
- Fitness functions receive DNA and return `(score, initial_positions)` tuples
- The `params` dict in `physicsengine.py` is the central configuration point for simulation behavior
