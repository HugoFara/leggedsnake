# Changelog

All notable changes to the LeggedSnake will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.5.0] - Unreleased

### Added

- **Selective foot–ground collision**: only edges touching foot nodes
  collide with the ground surface, preventing non-foot linkage parts
  (frame, crank, coupler) from scraping the road and distorting the gait.
  - ``Walker.get_foot_edges()`` auto-detects which edges should touch
    the ground based on ``get_feet()`` topology analysis.
  - ``Walker.foot_edge_ids`` property allows explicit override.
  - Uses pymunk collision categories internally: foot edges (``0x1``),
    non-foot edges (``0x2``), ground segments (``0x4``).
  - Fully backward-compatible: when no feet are detected or
    ``foot_edge_ids`` is empty, all edges collide as before.
- **Improved foot detection** in ``Walker.get_feet()``: now detects
  "outermost driven nodes" in addition to terminal (degree-1) nodes,
  correctly identifying coupler points (P) in synthesised four-bars.
- **Multi-DOF mechanism support**: mechanisms can now have multiple
  independent drivers, each with its own angular velocity.
  - ``Walker.motor_rates`` accepts a ``dict[str, float]`` mapping driver
    node IDs to individual rates, or a single ``float`` for all drivers.
  - ``PhysicsMapping.motor_node_ids`` tracks which driver each motor
    corresponds to.
- **Hypergraph-native Walker**: ``Walker`` now stores a
  ``HypergraphLinkage`` (topology) and ``Dimensions`` (geometry) directly.
  Mechanisms are constructed with ``Node``, ``Edge``, ``Hyperedge``,
  and ``Dimensions`` from pylinkage's hypergraph API.
- ``walker_from_legacy(linkage)`` factory function to convert existing
  joint-based ``Linkage`` objects to the new ``Walker``.
- Re-exports of pylinkage hypergraph API: ``HypergraphLinkage``, ``Node``,
  ``Edge``, ``Hyperedge``, ``NodeRole``, ``Dimensions``, ``DriverAngle``.
- ``Walker.to_mechanism()`` converts to pylinkage's ``Mechanism`` for
  kinematic simulation with proper multi-driver support.
- ``Walker.get_feet()`` returns terminal node IDs (replaces ``get_foots()``).
- ``DynamicLinkage`` accepts ``HypergraphLinkage`` + ``Dimensions`` directly
  (no intermediate ``from_linkage()`` conversion).
- ``NodeProxy`` lightweight class replaces the ``DynamicJoint`` hierarchy
  for reading physics body positions.
- ``World.add_linkage()`` now accepts ``Walker`` directly.
- Re-exports of pylinkage optimization pipeline:
  ``Agent``, ``MutableAgent``, ``differential_evolution_optimization``,
  ``dual_annealing_optimization``, ``minimize_linkage``,
  ``chain_optimizers``, ``multi_objective_optimization``,
  ``ParetoFront``, ``ParetoSolution``, ``OptimizationProgress``,
  and async variants.
- ``genetic_algorithm_optimization``: standard-signature wrapper for the
  built-in GA, compatible with ``chain_optimizers``.
- ``walking_objectives`` module with factory functions:
  ``stride_length_objective``, ``energy_efficiency_objective``,
  ``total_distance_objective``, ``multi_objective_walking_optimization``.
- ``add_opposite_leg()`` mirrors a leg across a vertical axis.
- **``WorldConfig`` dataclass**: structured replacement for the global
  ``params`` dict. Pass ``config=WorldConfig(...)`` to ``World()`` to
  parameterize gravity, physics period, torque, terrain, and friction.
  - ``TerrainConfig`` dataclass for terrain generation parameters.
  - ``DEFAULT_CONFIG`` module-level instance with the previous defaults.
- **Physics-aware fitness protocol** (``fitness`` module):
  - ``FitnessResult`` dataclass with ``score``, ``metrics``, ``valid``,
    and ``loci`` fields for rich evaluation results.
  - ``DynamicFitness`` runtime-checkable Protocol for standardized
    fitness function signatures: ``(topology, dimensions, config) → FitnessResult``.
  - Built-in implementations: ``DistanceFitness`` (total walking distance),
    ``EfficiencyFitness`` (energy efficiency ratio), ``StrideFitness``
    (kinematic stride length, no physics).
  - ``as_eval_func()`` adapter: wraps ``DynamicFitness`` into pylinkage's
    ``(linkage, dims, pos) → float`` optimizer contract.
  - ``as_ga_fitness()`` adapter: wraps ``DynamicFitness`` into the GA
    optimizer's ``(dna) → (score, positions)`` contract.
  - Walking objective factories (``total_distance_objective``,
    ``energy_efficiency_objective``) accept an optional ``config`` parameter.
- **Dynamic Co-Design** (``co_design`` module): connect pylinkage's topology
  co-optimization with leggedsnake's physics simulation.
  - ``Walker.from_catalog(entry, dimensions)``: build Walker from a
    topology ``CatalogEntry``.
  - ``Walker.from_hierarchy(hierarchy, dimensions)``: build Walker from a
    ``HierarchicalLinkage`` (flattened automatically).
  - ``Walker.from_synthesis(solution)``: build Walker from
    ``TopologySolution`` or ``CoOptSolution``.
  - ``co_optimize_objective(fitness)``: adapt ``DynamicFitness`` to
    pylinkage's ``co_optimize()`` contract (``Linkage → float``, minimized).
    Supports two-stage kinematic pre-filter + dynamic evaluation.
  - ``optimize_walking_mechanism(spec)``: end-to-end pipeline from
    ``WalkingDesignSpec`` to ranked ``Walker`` solutions with metrics.
  - ``WalkingDesignSpec``, ``WalkingDesignResult`` dataclasses.
- **URDF export** (``urdf_export`` module):
  - ``to_urdf(walker)`` generates a URDF XML string for a Walker.
  - ``to_urdf_file(walker, path)`` writes to file.
  - ``URDFConfig`` dataclass for export options.
- **Stability metrics** (``stability`` module):
  - ``StabilitySnapshot`` dataclass: single-timestep CoM, ZMP, support
    polygon, tip-over margin, and body angle.
  - ``StabilityTimeSeries`` dataclass with properties:
    ``mean_tip_over_margin``, ``min_tip_over_margin``, ``zmp_excursion``,
    ``angular_stability``, ``com_trajectory``, ``summary_metrics()``.
  - ``compute_com(linkage)`` and ``compute_com_velocity(linkage)``:
    mass-weighted center of mass from pymunk rigid bodies.
  - ``approximate_zmp()``: zero-moment-point x-coordinate via linear
    inverted pendulum model.
  - ``get_support_polygon()``: convex hull of foot positions near ground.
  - ``compute_tip_over_margin()``: signed distance from CoM projection to
    support polygon boundary.
  - ``compute_stability_snapshot()``: one-call assembly of all metrics.
- **Gait analysis** (``gait_analysis`` module):
  - ``FootEvent`` dataclass for touchdown/liftoff events.
  - ``GaitCycle`` dataclass with duty factor, stride period, stance/swing
    duration.
  - ``GaitAnalysisResult`` dataclass with ``mean_duty_factor``,
    ``mean_stride_frequency``, ``mean_stride_length``, ``phase_offsets``,
    ``summary_metrics()``.
  - ``detect_foot_events()``: y-threshold crossing detection.
  - ``extract_gait_cycles()``: groups events into stride cycles.
  - ``compute_phase_offsets()``: normalized [0,1) phase between foot pairs.
  - ``compute_foot_trajectory_metrics()``: max height, horizontal range,
    path length, smoothness.
  - ``analyze_gait()``: one-call entry point from simulation loci.
- **NSGA-II/III multi-objective optimizer** (``nsga_optimizer`` module):
  - ``NsgaWalkingConfig`` dataclass: generations, population, algorithm,
    seed, crossover/mutation parameters, and ``n_workers`` for parallel
    evaluation.
  - ``NsgaWalkingResult`` dataclass: Pareto front with optional per-solution
    gait analysis and stability time series.
  - ``WalkingNsgaProblem``: pymoo Problem wrapper that evaluates
    ``DynamicFitness`` objectives on Walker candidates.
  - ``nsga_walking_optimization()``: high-level entry point returning
    ``ParetoFront`` of non-dominated solutions.
  - ``StabilityFitness``: scores by mean tip-over margin.
  - ``CompositeFitness``: evaluates distance + efficiency + stability in a
    single physics simulation (avoids redundant runs).
- **Visualization and reporting** (``plotting`` module):
  - ``plot_pareto_front()``: 2D and 3D Pareto front scatter plots with
    best-compromise highlighting.
  - ``plot_gait_diagram()``: gait timing diagram (stance/swing bars per
    foot).
  - ``plot_stability_timeseries()``: four-panel plot of tip-over margin,
    ZMP, body angle, and CoM height.
  - ``plot_com_trajectory()``: 2D CoM path colored by tip-over margin with
    support polygon snapshots.
  - ``plot_foot_trajectories()``: per-foot trajectory shape plots.
  - ``plot_optimization_dashboard()``: combined four-panel dashboard for a
    single Pareto solution.
- **Topology co-optimization** (``topology_optimization`` module):
  - ``TopologyCoOptConfig`` dataclass: max links, bounds, mutation rate,
    ``n_workers`` for parallel evaluation.
  - ``TopologyWalkingResult`` dataclass: extends NSGA result with
    per-solution ``TopologySolutionInfo`` (topology name, ID, link count).
  - ``topology_walking_optimization()``: jointly optimizes mechanism
    topology (from pylinkage's 19-entry catalog) and link dimensions
    using NSGA-II. Mixed chromosome: ``[topology_idx, dim_1, ..., dim_N]``.
  - ``solutions_by_topology()``: groups Pareto solutions by mechanism type.
- **Parallel fitness evaluation**:
  - ``n_workers`` parameter on ``NsgaWalkingConfig`` and
    ``TopologyCoOptConfig``. When > 1, candidate evaluation uses
    ``concurrent.futures.ProcessPoolExecutor``.
- **Walker serialization** (``serialization`` module):
  - ``walker_to_dict()`` / ``walker_from_dict()``: serialize Walker
    (topology + dimensions + motor rates) to/from plain dicts.
  - ``save_walker()`` / ``load_walker()``: JSON file I/O for Walkers.
  - ``result_to_dict()`` / ``result_from_dict()``: serialize
    ``NsgaWalkingResult`` (Pareto front scores, dimensions, config,
    topology metadata) to/from plain dicts.
  - ``save_result()`` / ``load_result()``: JSON file I/O for optimization
    results.
- ``examples/optimization_pipeline.py``: end-to-end example demonstrating
  Walker definition, NSGA-II optimization, gait/stability analysis, and
  all visualization plots.
- **Expanded terrain generation** in ``TerrainConfig`` / ``World``:
  - ``seed`` field for reproducible terrain via a seeded
    ``np.random.Generator`` (replaces bare ``np.random`` calls).
  - ``friction_range`` field: per-segment friction randomized uniformly
    within ``(lo, hi)``, overriding the global ``friction`` value.
  - ``gap_freq`` / ``gap_width``: configurable chasms (empty space) in the
    road that the walker must step over.
  - ``obstacle_freq`` / ``obstacle_height`` / ``obstacle_width``:
    rectangular bumps placed on the road surface.
  - ``slope_profile`` field: deterministic slope generators for repeatable
    benchmarking. Accepts a ``SlopeProfile`` enum (``RANDOM``, ``FLAT``,
    ``CONSTANT``, ``VALLEY``, ``SAWTOOTH``), a string key, or a custom
    callable with signature ``(terrain, rng, step) → angle``.
  - ``SLOPE_PROFILES`` registry mapping string keys to generator callables.
  - Re-enabled discrete step generation (was disabled with ``and False``).
  - ``TerrainPreset`` enum with ``TerrainConfig.from_preset()`` factory:
    ``FLAT``, ``HILLY``, ``ROUGH``, ``STAIRS``, ``MIXED`` ready-made
    terrain configurations.
  - ``SlopeProfile``, ``SLOPE_PROFILES``, and ``TerrainPreset`` are
    re-exported from the package.

### Changed

- **Breaking**: ``Walker`` no longer inherits from ``pylinkage.Linkage``.
  It is now a standalone class with ``topology`` and ``dimensions``
  attributes. Use ``walker_from_legacy()`` to migrate existing code.
- **Breaking**: ``DynamicLinkage`` no longer inherits from ``Linkage``.
  It takes ``(topology, dimensions, space)`` instead of ``(joints, space)``.
- **Breaking**: ``motor_rate`` parameter renamed to ``motor_rates`` in
  ``create_bodies_from_hypergraph()`` and ``DynamicLinkage``. Accepts
  ``dict[str, float]`` for per-driver rates or ``float`` for uniform rate.
- **Breaking**: Rigid triangle detection in ``hypergraph_physics`` now uses
  ``Hyperedge`` objects instead of ``isinstance(joint, Fixed)`` checks.
  Add ``Hyperedge`` to your topology for Fixed/ternary joints.
- ``_find_effective_ground_nodes()`` no longer requires a ``joints``
  parameter; ground detection is purely topology-based.
- ``World.update()`` power calculation uses ``sum(power)`` across all
  motors instead of ``power[0]`` only. Fixes incorrect energy accounting
  for multi-motor mechanisms.
- ``World.__update_linkage__()`` enables/checks all motors via
  ``physics_mapping.motors`` instead of ``isinstance`` checks.
- Joint color assignment in ``VisualWorld`` now uses ``NodeRole``
  (GROUND/DRIVER/DRIVEN) instead of ``isinstance`` checks on legacy
  joint classes.
- ``GeneticOptimization.run()`` returns ``list[Agent]`` (was raw lists).
- All examples rewritten to use hypergraph construction pattern.
- ``examples/`` is now in the main folder (was in ``docs/``).
- Minimum Python version is now 3.10 (was 3.7).
- Support for Python 3.12, 3.13, and 3.14.
- Requires ``pylinkage>=0.8.0``.
- Requires ``pymoo>=0.6.1.6`` (for NSGA-II/III and topology co-optimization).
- Requires ``scipy>=1.15.3``.
- Version bumped to 0.5.0.

### Fixed

- Multi-motor energy accounting: ``World.update()`` now sums power from
  all motors (was using only the first motor's power).
- ``add_legs()`` / ``add_opposite_leg()`` no longer crash with semantic
  edge IDs (e.g., ``"frame_crank"``). The edge counter no longer assumes
  numeric suffixes.
- ``add_legs()`` no longer requires ``to_mechanism()`` to succeed on the
  current topology. Cloned drivers use ``DriverAngle.initial_angle``
  phase offsets instead of kinematic simulation.
- ``add_opposite_leg()`` now creates independent DRIVER nodes with pi
  phase offset (was creating DRIVEN nodes linked to original driver).
- Project links fixed in pyproject.toml.
- Road step/gap/obstacle direction logic: new road features now extend
  in the correct direction (forward or backward) matching slope segments.

### Removed

- ``DynamicJoint`` abstract base class and subclasses: ``Nail``, ``Motor``,
  ``PinUp``, ``DynamicPivot``. Replaced by ``NodeProxy``.
- ``_joint_adapters.py`` module (adapter classes for legacy pylinkage API).
- ``convert_to_dynamic_joints()`` method on ``DynamicLinkage``.
- Re-exports of legacy pylinkage joint classes: ``Static``, ``Crank``,
  ``Fixed``, ``Pivot``, ``Revolute``, ``Linkage``, ``Ground``,
  ``FixedDyad``, ``RRRDyad``, ``bounding_box``, ``show_linkage``.
  Import these from ``pylinkage`` directly if needed.
- ``Walker.get_foots()`` (replaced by ``get_feet()``).
- ``setup.cfg``, ``setup.py``, ``requirements.txt``,
  ``requirements-dev.txt``, ``environment.yml``.

## [0.4.0] - 2023-06-21

### Added in 0.4.0

- View all walkers!
  - ``show_all_walkers`` in ``docs/examples/strider.py`` let you see all
    walkers in one simulation!
  - You can set the color of walkers during display.
- Genetic optimization:
  - ``GeneticOptimization`` class in ``geneticoptimizer.py`` that will
    replace the previous functional paradigm.
  - The average score is now displayed.
- ``VisualWorld`` has a new method called ``reload_visuals``.
- ``show_evolution.py`` is a new script plotting various data about the
  Walkers population's evolution during genetic optimization.
- In ``docs/examples/strider.py`` we recommend to use ``total_distance``
  as the fitness function.

### Changed in 0.4.0

- Genetic optimization:
  - During genetic optimization, population is now stable at max_pop
    (it used to fluctuate a lot).
  - Genetic optimization do no longer display all dimensions in the
    progress bar.
  - ``startnstop`` argument may now be the name of the file to use
    (a string).
  - ``max_genetic_distance`` was changed from 0.7 to 10.
    Results are much better now!
- Visuals:
  - ``update`` method of ``VisualWorld`` replaced by ``visual_update``.
    It clearly separates physics and display time.
  - Frame rate and physics speed are now independent parameters.
  - Visuals go to a new file ``worldvisualizer.py``.
  - Camera parameters should now be accessed from ``CAMERA`` instead of
    ``params["camera"]``.
  - The camera feels more cinematic.
- You can define a custom load when using ``World.add_linkage`` or
  ``VisualWorld.add_linkage``. The default is 0.
- ``pyproject.toml`` updated with the data of ``setup.cfg``.
  This is now the recommended metadata for the project.
- In ``docs/example/strider.py``, simulation time was increased from 30
  seconds to 40. It was just not enough.

### Fixed in 0.4.0

- Documentation of ``evolutionary_optimization_builtin`` was wrong: returned
  data were in order (fitness, dimensions, position), but
  (fitness, position, dimensions) was indicated.
- After a genetic optimization, the example script was assigning wrong data
  to the demo walker.
- ``kwargs_switcher`` from ``geneticoptimizer.py`` do no longer pop (destroy)
  argument from the input dictionary.

### Deprecated in 0.4.0

- ``setup.cfg`` should no longer be used, as it is replaced by ``pyproject.toml``.

### Removed in 0.4.0

- ``evolutionary_optimization`` function is removed.
  Use ``GeneticOptimization`` class instead.
  - You can no longer use the argument "init_pop" to change the size of
    the initial population. It now always set to max_pop.
- ``time_coef``, ``calc_rate`` and ``max_sub`` parameters of
  ``params["simul"]`` replaced by a unique ``physics_period`` set to
  0.02 (s).
- ``leggedsnake/Population evolution.json`` removed. It contained data
  about an evolution run and is not relevant for users.

## [0.3.1] - 2023-06-14

Starting from 0.3.1, we won't include "-alpha" or "-beta" in the naming scheme,
as it is considered irrelevant.

### Added in 0.3.1

- ``requirements-dev.txt`` that contain dev requirements. It makes contribution easier.
- PyCharm configuration files.

### Changed in 0.3.1

- Animations are now all stored in local variables, and no longer in an "ani"
global list of animations.

### Fixed in 0.3.1

- The main example file ``strider.py`` was launching animations for each subprocess.
This file is now considered an executable.
- ``evolutionary_optimization_builtin`` was during the last evaluation of linkages.
- ``data_descriptors`` were not save for the first line of data only in
  ``geneticoptimizer``.
- Multiple grammar corrections.
- The ``video`` function of ``physicsengine.py`` now effectively launches
  the video (no call to plt.show required).
- The ``video`` function of ``physicsengine.py`` using ``debug=True`` was crashing.

## [0.3.0-beta] - 2021-07-21

### Added in 0.3.0

- Multiprocessing is here! The genetic optimization can now be run in parallel!
  Performances got improved by 65 % using 4 processes only.

### Changed in 0.3.0

- We now save data using JSON! Slow computer users, you can relax and stop
  computing when you want.
- The sidebar in the documentation is a bit more useful.
- Not having tqdm will cause an exception.

### Fixed in 0.3.0

- Corrected the example, the genetic optimization is now properly fixed but
slower.

### Removed in 0.3.0

- Native support for PyGAD is no longer present.
- ``evolutionnary_optimization`` (replaced by ``evolutionary_optimization``).
- Data saved in the old txt format are no longer readable (were they readable?)

## [0.2.0-alpha] - 2021-07-14

### Added in 0.2.0

- Dependency to [tqdm](https://tqdm.github.io/) and matplotlib.
- The ``evolutionary_optimization`` replaces ``evolutionnary_optimization``.
  - The ``ite`` parameter renamed ``iters`` for consistency with pylinkage.
  - The new parameter ``verbose`` let you display a nice progress bar, more
    information on optimization state, or nothing.
- The best solution can be displayed with PyGAD as well.

### Changed in 0.2.0

- Typos and cleans-up in ``docs/examples/strider.py``.
- ``evolutionnary_optimization_legacy`` renamed to
  ``evolutionary_optimization_builtin``.

### Deprecated in 0.2.0

- ``evolutionnary_optimization`` is now deprecated. Please use
  ``evolutionary_optimization``.

### Removed in 0.2.0

- Explicit dependency to PyGAD. There is no longer an annoying message when
  PyGAD is not installed.

## [0.1.4-alpha] - 2021-07-12

### Added in 0.1.4

- It is now possible and advised to import class and functions using quick
  paths, for instance ``from leggedsnake import Walker`` instead of
  ``from leggedsnake.walker import Walker``.
- You do no longer have to manually import
  [pylinkage](https://hugofara.github.io/pylinkage/), we silently import the
  useful stuff for you.
- We now use [bump2version](https://pypi.org/project/bump2version/) for version
  maintenance.
- This is fixed by the ``road_y`` parameter in ``World`` let you define a
  custom height for the base ground.

### Changed in 0.1.4

- ``docs/examples/strider.py`` has been updated to the latest version of
  leggedsnake 0.1.4.

### Fixed in 0.1.4

- The full swarm representation in polar graph has been repaired in
  ``docs/examples/strider.py``.
- During a dynamic simulation, linkages with long legs could appear through the
  road.
- The documentation was not properly rendered because Napoleon (NumPy coding
  style) was not integrated.

## [0.1.3-alpha] - 2021-07-10

This package was lacking real documentation, it is fixed in this version.

### Added in 0.1.3

- Sphinx documentation!
- Website hosted on GitHub pages, check
  [hugofara.github.io/leggedsnake](https://hugofara.github.io/leggedsnake/)!
- Expanded README with the quick links section.

### Changed in 0.1.3

- Tests moved from ``leggedsnake/tests`` to ``tests/``.
- Examples moved from ``leggedsnake/examples/`` to ``docs/examples/``.
- I was testing my code on ``leggedsnake/examples/strider.py``  (the old path) and
  that's why it was a big mess. I cleaned up that all. Sorry for the
  inconvenience!

### Fixed in 0.1.3

- A lot of outdated code in the ``leggedsnake/examples/strider.py``
- Changelog URL was broken in ``setup.cfg``.

## [0.1.2-alpha] - 2021-07-07

### Added in 0.1.2

- Security: tests with ``tox.ini`` now include Python 3.9 and Flake 8.

### Changed in 0.1.2

- The ``step`` function execution speed has been increased by 25% when
  ``return_res`` is ``True``! Small performance improvement when ``return_res``
  is ``False``.
- The ``size`` argument of ``step`` function is now known as ``witdh``.
- We now require pylinkage>=0.4.0.

### Fixed in 0.1.2

- Files in ``leggedsnake/examples/`` were not included in the PyPi package.
- The example was incompatible with pylinkage 0.4.0.
- Test suite was unusable by tox.
- Tests fixed.
- Incompatible argument between PyGAD init_pop and built-in GA.

## [0.1.1-alpha] - 2021-06-26

### Added in 0.1.1

- The example file ``examples/strider.py`` is now shipped with the Python
  package.
- ``leggedsnake/geneticoptimizer.py`` can now automatically switch to the
  built-in GA algorithm if PyGAD is not installed.

### Changed in 0.1.1

- ``setup.cfg`` metadata

## [0.1.0-alpha] - 2021-06-25

### Added in 0.1.0

- Code vulnerabilities automatic checks
- Example videos in ``examples/images/``

### Changed in 0.1.0

- Many reforms in code style in order to make the dynamic part of naming conventions
  consistent with Pymunk.
- Images in the ``README.md``!

### Fixed in 0.1.0

- You can now define linkages with an enormous number of legs. Systems with
  many should no longer break physics but your CPU instead :)

## [0.0.3-alpha] - 2021-06-23

### Added in 0.0.3

- Started walkthrough demo in ``README.md``
- Automatic release to PyPi

### Fixed in 0.0.3

- Pymunk version should be at least 6.0.0 in requirement files.
- Some URLs typos in ``README.md``
- Versioning tests not executing (GitHub action)

## [0.0.2-alpha] - 2021-06-22

### Added in 0.0.2

- ``requirement.txt`` was absent due to ``.gitignore`` misconfiguration.

### Changed in 0.0.2

- ``.gitignore`` now ignores .txt files only in the leggedsnake folder.
- ``environment.yml`` more flexible (versions can be superior to the selected).
  pymunk>5.0.0 and pylinkage added.
- ``leggedsnake/utility.py`` not having zipfile or xml modules error
  encapsulation.

### Fixed in 0.0.2

- ``setup.cfg`` was not PyPi compatible. Removed mail (use GitHub!), we now
  explicitly say that ``README.md`` is markdown (PyPi is conservative)

## [0.0.1-alpha] - 2021-06-22

Basic version, supporting Genetic Algorithm optimization, but with various
problems.

### Added in 0.0.1

- ``CODE_OF_CONDUCT.md`` to help community.
- ``LICENSE`` MIT License.
- ``MANIFEST.in`` to include more files.
- ``README.md`` as a very minimal version.
- ``environment.yml`` with matplotlib, numpy, and pygad requirement.
- ``examples/strider.py`` a complete demo with Strider linkage.
- ``leggedsnake/__init__.py``.
- ``leggedsnake/dynamiclinkage.py``.
- ``leggedsnake/geneticoptimizer.py``.
- ``leggedsnake/physicsengine.py``.
- ``leggedsnake/show_evolution.py`` just a legacy package, no utility.
- ``leggedsnake/tests/test_utility.py`` untested test case
- ``leggedsnake/utility.py`` contain some useful evaluation function (``step``
  and ``stride``) and a broken GeoGebra interface.
- ``walker.py`` defines the ``Walker`` object.
- ``pyproject.toml``.
- ``setup.cfg``.
- ``setup.py`` empty, for compatibility purposes only.
- ``tox.ini`` tox with Python 3.7 and 3.8.
