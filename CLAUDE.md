# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

LeggedSnake simulates and optimizes planar walking linkages. It layers a pymunk physics engine and multi-objective optimization on top of [pylinkage](../pylinkage/)'s kinematic/hypergraph model. The canonical mechanism is the Strider; classical Jansen/Klann/Chebyshev builders are also first-class.

This package depends on pylinkage as an editable path dep (`[tool.uv.sources]` in `pyproject.toml`) — changes to pylinkage are picked up without reinstalling.

## Commands

```bash
uv sync                                       # install deps (pulls editable pylinkage)
uv run pytest                                 # all tests
uv run pytest tests/test_utility.py::TestStride::test_minimal_stride   # single test
uv run pytest -k "strider"                    # tests matching pattern
uv run pytest --cov=leggedsnake               # coverage
uv run ruff check src/                        # lint
uv run mypy                                   # type-check (strict)
uv run jupyter lab examples/                  # numbered tutorial notebooks 01–06
uv run python examples/tools/verify_mechanisms.py  # smoke-test classical builders
```

Notebooks that spawn worker processes (GA, NSGA) execute their long-running cells under `if __name__ == "__main__":`-equivalent guards — `multiprocessing` workers fork-bomb on import otherwise.

## Architecture

The library composes into four layers. The public API is re-exported from `leggedsnake/__init__.py` — check there before chasing symbols into submodules.

### 1. Mechanism model — `walker.py`

`Walker` wraps a `HypergraphLinkage` (topology: nodes + edges + `NodeRole`) and a `Dimensions` (geometry: node positions, edge distances, `DriverAngle` per driver node). It is **hypergraph-native** — the older joint-first API (`Static`/`Crank`/`Fixed`/`Pivot`) is no longer the primary path.

Key surface:
- `Walker(topology, dimensions, name=..., motor_rates=...)` — `motor_rates` is `float` (all drivers) or `dict[str, float]` (per-driver, enables multi-DOF).
- `Walker.from_catalog`, `from_hierarchy`, `from_watt`, `from_jansen`, etc. — factory builders for classical mechanisms.
- `add_legs(n)` phase-offsets copies along the axis; `add_opposite_leg(axis_x)` mirrors across the frame.
- `to_mechanism()` hands off to pylinkage's `Mechanism` for kinematic stepping.
- `get_num_constraints()` / `set_num_constraints()` and `get_coords()` / `set_coords()` are the (de)serialization hooks used by optimizers.
- `get_foots()` returns terminal joints (feet).

### 2. Physics — `physicsengine.py`, `dynamiclinkage.py`, `hypergraph_physics.py`

- `World` owns a pymunk space, road generation, and stepping. `World.add_linkage(walker)` auto-converts kinematic → dynamic.
- `WorldConfig` / `TerrainConfig` / `SlopeProfile` / `TerrainPreset` are the structured config types; `SLOPE_PROFILES` and `DEFAULT_CONFIG` are prebuilt presets.
- Legacy global `params` dict still works for quick tuning (`params["simul"]["physics_period"]`, `params["linkage"]["torque"]`, `params["linkage"]["load"]`, `params["ground"]["friction"|"slope"|"noise"]`). New code should prefer the dataclass configs.
- `dynamiclinkage.py`: `DynamicJoint` ABC → `Nail`, `PinUp`, `DynamicPivot`, `Motor`; `DynamicLinkage` + `convert_to_dynamic_linkage()`.
- `hypergraph_physics.py`: builds pymunk bodies directly from the hypergraph, merging Fixed-triangle edges into rigid bodies. `PhysicsMapping` tracks edge→body and node→bodies.

### 3. Evaluation — `fitness.py`, `stability.py`, `gait_analysis.py`

`fitness.py` defines the `DynamicFitness` protocol and `FitnessResult` dataclass. Built-ins: `DistanceFitness`, `EfficiencyFitness`, `StrideFitness`, `StabilityFitness`, `CompositeFitness`. Adapters `as_eval_func()` / `as_ga_fitness()` bridge to pylinkage's optimizer contracts and the GA's `(score, initial_positions)` tuple.

`stability.py` computes CoM, ZMP, support polygon, tip-over margin — collected as `StabilityTimeSeries` from pymunk data.

`gait_analysis.py` extracts foot-contact events and gait cycles (`GaitCycle`, `FootEvent`, `analyze_gait`).

### 4. Optimization

Three layers, listed from general → walking-specific:

| Module | Entry point | Purpose |
|---|---|---|
| `geneticoptimizer.py` | `GeneticOptimization(dna, fitness, max_pop).run(iters, processes)` or `genetic_algorithm_optimization()` | Built-in GA with multiprocessing + JSON checkpoint/resume (`startnstop=`). DNA = `[score, dimensions, coords]`. |
| `walking_objectives.py` | `stride_length_objective`, `energy_efficiency_objective`, `total_distance_objective`, `multi_objective_walking_optimization` | Walking-specific objective functions reusing pylinkage's optimizer infrastructure. |
| `nsga_optimizer.py` | `nsga_walking_optimization(walker_factory, config: NsgaWalkingConfig)` | Multi-objective NSGA-II/III via pymoo. Returns `ParetoFront`. |
| `topology_optimization.py` | `topology_walking_optimization(config: TopologyCoOptConfig)` | Joint topology + dimensions co-optimization over pylinkage's topology catalog (mixed chromosome `[topology_index, dim...]`). |
| `co_design.py` | `optimize_walking_mechanism(spec: WalkingDesignSpec)` | End-to-end pipeline: topology discovery + kinematic prefilter + dynamic fitness. |
| `leg_count.py` | `sweep_leg_counts()` | Parameter sweep over number of legs. |

`agents_to_ensemble()` wraps optimization outputs into pylinkage's `Ensemble` for ranking/filtering.

### 5. I/O and visualization

- `serialization.py`: `save_walker`/`load_walker`, `save_result`/`load_result`, plus dict converters.
- `urdf_export.py`: `to_urdf`, `to_urdf_file`, `URDFConfig` — export to ROS URDF.
- `plotting.py`: matplotlib/plotly figures — `plot_pareto_front`, `plot_gait_diagram`, `plot_stability_timeseries`, `plot_com_trajectory`, `plot_foot_trajectories`, `plot_optimization_dashboard`, `plot_walker_plotly`, `save_walker_svg`.
- `show_evolution.py`: `show_genetic_optimization` replays a saved GA checkpoint.
- `worldvisualizer.py`: `VisualWorld` + `video(walker, duration, save=...)`, `all_linkages_video(walkers)`, `video_debug()` (frame-stepping with force vectors). **These symbols are lazy-imported** in `__init__.py` via `__getattr__` because they pull in pyglet and need a display — referencing them on a headless box triggers the import only at access time.
- `utility.py`: `stride(locus, height)` and `step(locus, height, width)` — kinematic fitness helpers used by PSO-style optimizers.

## Key patterns and gotchas

- **Fitness function contract**: GA/DNA path expects `fitness(dna) -> (score, initial_positions)`. The physics-aware protocol (`DynamicFitness`) takes `(topology, dimensions) -> FitnessResult`. Convert between them with `as_ga_fitness()` / `as_eval_func()`.
- **Multi-DOF mechanisms** are expressed by supplying multiple DRIVER-role nodes in the topology and per-driver rates in `Dimensions.driver_angles` / `Walker.motor_rates`. No separate API.
- **Kinematic vs dynamic optimization**: prefer kinematic (`stride`/`step` + PSO) for fast inner loops and fall back to dynamic GA/NSGA only for the final selection. The README advises exploiting mechanism symmetry (e.g. optimize a half-Strider kinematically).
- **Checkpointing**: `GeneticOptimization(..., startnstop="path.json")` auto-resumes. Long runs should always use this.
- **pymunk solver tuning**: `set_space_constraints()` in `physicsengine.py` auto-scales solver iterations to constraint count — don't hardcode iterations for large mechanisms.

## Tests

`tests/` mirrors `src/leggedsnake/` — `test_<module>.py` per module plus `test_integration.py` and `test_parallel_evaluation.py`. The integration test exercises the full optimize→simulate→visualize pipeline and is the best smoke test after architectural changes.

## Examples

Notebook-first. The tutorial surface is the numbered notebooks plus `discover_walker`; classical mechanisms (Jansen, Klann, Chebyshev, Strider) are reached through `Walker.from_*` factories inside the notebooks rather than per-mechanism scripts.

- `examples/01_walkers_gallery.ipynb` — `Walker.from_*` factories and the classical-walker gallery.
- `examples/02_physics_and_fitness.ipynb` — `World`, `WorldConfig` / `TerrainConfig` / `SLOPE_PROFILES`, and the `DynamicFitness` protocol.
- `examples/03_genetic_optimization.ipynb` — single-objective GA with checkpointing.
- `examples/04_multi_objective_and_gait.ipynb` — NSGA-II Pareto fronts, gait analysis, stability time series.
- `examples/05_topology_co_optimization.ipynb` — topology + dimensions co-optimization plus Phase 8.3 `evolve_offsets` (chromosome co-evolves topology, dimensions, and per-leg phase offsets in one NSGA sweep).
- `examples/06_export_and_share.ipynb` — JSON (``save_walker``/``load_walker``), URDF (``to_urdf``, ``URDFConfig``), SVG (``save_walker_svg``), interactive plotly (``plot_walker_plotly``).
- `examples/tools/` — maintenance scripts, not tutorials. `verify_mechanisms.py` smoke-tests the classical builders; `generate_readme_gifs.py` regenerates the README animations.
