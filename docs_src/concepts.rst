========
Concepts
========

LeggedSnake's API is small once you see the three pieces that compose
into everything else: how a mechanism is *described*, how a candidate
is *scored*, and how the optimizer landscape lays itself out.  This
page is the orientation; the per-module API pages have the exhaustive
detail.

The mechanism model: topology + dimensions = walker
===================================================

A walker is two orthogonal pieces of state, plus a thin object that
holds them together.

**Topology — what is connected to what.**
  :class:`~pylinkage.hypergraph.HypergraphLinkage` is a graph of
  :class:`~pylinkage.hypergraph.Node` (joints, with a
  :class:`~pylinkage.hypergraph.NodeRole` of ``GROUND``, ``DRIVER``,
  or ``DRIVEN``) connected by :class:`~pylinkage.hypergraph.Edge`
  (binary links) and :class:`~pylinkage.hypergraph.Hyperedge` (rigid
  triangles / N-ary clusters).  The topology describes the graph; it
  has no metric information.

**Dimensions — how big and where.**
  :class:`~pylinkage.dimensions.Dimensions` carries node positions,
  edge distances, and a :class:`~pylinkage.dimensions.DriverAngle`
  per driver node (initial angle + angular velocity).  Two walkers
  with the same topology but different dimensions are different
  designs; this is what an optimizer mutates.

**Walker — the bag that holds them.**
  :class:`~leggedsnake.Walker` wraps ``(topology, dimensions)`` plus
  a ``motor_rates`` override (``float`` for all drivers, or
  ``dict[str, float]`` for per-driver rates — the multi-DOF path).
  It exposes ``add_legs(n)`` for phase-offset copies,
  ``add_opposite_leg()`` for mirroring, ``to_mechanism()`` for
  kinematic stepping, and the classical-mechanism factories
  (``from_jansen``, ``from_klann``, ``from_chebyshev``,
  ``from_strider``, ``from_watt``, ``from_catalog``, …).

Why the split?  Optimization perturbs *dimensions* on a fixed
*topology* — separating the two means the optimizer never has to think
about graph mutation.  Topology co-design (see below) is the
exception, and it picks topologies from a finite catalog rather than
mutating edges.

The fitness protocol
====================

All physics-aware evaluators share one signature, captured by the
:class:`~leggedsnake.DynamicFitness` ``Protocol``:

.. code-block:: python

   def __call__(
       self,
       topology: HypergraphLinkage,
       dimensions: Dimensions,
       config: WorldConfig | None = None,
   ) -> FitnessResult: ...

A :class:`~leggedsnake.FitnessResult` carries:

* ``score`` — primary value, **higher is better** by convention.
* ``metrics`` — dict of secondary numbers (``"distance"``,
  ``"energy"``, ``"buildable_fraction"``, ``"froude_number"``,
  ``"cost_of_transport"``, …).  Multi-objective optimizers consume
  these; you also read them post-hoc.
* ``valid`` — did the run complete?  Failed designs return
  ``valid=False`` with a meaningful ``buildable_fraction`` so the
  optimizer still has a gradient.
* ``loci`` — per-joint trajectories when ``record_loci=True``.

Built-in implementations
------------------------

.. list-table::
   :header-rows: 1
   :widths: 30 35 35

   * - Class
     - Primary signal
     - Use when
   * - :class:`~leggedsnake.DistanceFitness`
     - Forward distance walked
     - You only care if it walks far.
   * - :class:`~leggedsnake.EfficiencyFitness`
     - Distance per unit motor energy
     - Energy-aware single-objective.
   * - :class:`~leggedsnake.StabilityFitness`
     - Mean tip-over margin
     - Robustness under disturbance.
   * - :class:`~leggedsnake.GaitFitness`
     - Mean stride length + gait metrics
     - You want gait shape, not just total distance.
   * - :class:`~leggedsnake.CompositeFitness`
     - Multiple objectives, *one* sim
     - NSGA front; avoids duplicate physics.
   * - :class:`~leggedsnake.StrideFitness`
     - Kinematic stride length, no physics
     - Inner loop of fast PSO; use for prefilter.

Adapters
--------

Two glue functions bridge :class:`~leggedsnake.DynamicFitness` to the
older optimizer contracts:

* :func:`~leggedsnake.as_eval_func` — wraps a fitness into pylinkage's
  ``(linkage, dimensions, positions) → float`` minimizer contract
  (use ``negate=True`` for any minimizer).
* :func:`~leggedsnake.as_ga_fitness` — wraps a fitness into the GA's
  ``(dna) → (score, initial_positions)`` tuple contract.

The same fitness object can therefore drive
:func:`~leggedsnake.particle_swarm_optimization`,
:func:`~leggedsnake.differential_evolution_optimization`,
:func:`~leggedsnake.minimize_linkage`,
:class:`~leggedsnake.GeneticOptimization`, and
:func:`~leggedsnake.nsga_walking_optimization` without rewriting it.

The optimizer landscape
=======================

Pick the cheapest tool that answers your question.  In rough cost
order from fastest-per-eval to slowest:

.. list-table::
   :header-rows: 1
   :widths: 35 25 40

   * - Function / class
     - What it varies
     - Use when
   * - :func:`~pylinkage.particle_swarm_optimization`,
       :func:`~pylinkage.kinematic_maximization`
     - Dimensions only (no physics)
     - Quick kinematic search; pair with
       :class:`~leggedsnake.StrideFitness` for the inner loop.
   * - :func:`~leggedsnake.chain_walking_optimizers`
     - Dimensions only
     - Stage global → local pipelines
       (DE → dual annealing → Nelder-Mead).
   * - :class:`~leggedsnake.GeneticOptimization`,
       :func:`~leggedsnake.genetic_algorithm_optimization`
     - Dimensions; JSON-checkpointed GA with multiprocessing
     - Single-objective dynamic GA; good for the final
       selection round.
   * - :func:`~leggedsnake.nsga_walking_optimization`
     - Dimensions; multi-objective
     - Pareto front of distance vs. energy vs. stability vs. gait.
   * - :func:`~leggedsnake.topology_walking_optimization`
     - Topology *and* dimensions (catalog index)
     - Pick a mechanism family + tune it jointly. Mixed
       chromosome.
   * - :func:`~leggedsnake.optimize_walking_mechanism`
     - End-to-end pipeline
     - Topology discovery → kinematic prefilter → dynamic
       fitness in one call.
   * - :func:`~leggedsnake.optimize_gait`
     - Phase offsets only
     - Tune trot / pace / canter on a finished walker.
   * - :func:`~leggedsnake.sweep_leg_counts`
     - Number of legs
     - Post-hoc "how many legs?" study.

Recommended pipeline
--------------------

The README phrases this as "exploit symmetry"; in concept terms it's
just *use a coarser model in the inner loop*:

1. **Kinematic stage.** PSO or DE on the half-mechanism with
   :class:`~leggedsnake.StrideFitness`.  Hundreds to thousands of
   evals per second; finds promising regions of the design space.
2. **Dynamic stage.** Hand the survivor to
   :class:`~leggedsnake.GeneticOptimization` or
   :func:`~leggedsnake.nsga_walking_optimization` with
   :class:`~leggedsnake.CompositeFitness` (distance + efficiency +
   stability + gait in one run).
3. **Gait tuning (optional).**
   :func:`~leggedsnake.optimize_gait` on the winner if you want a
   non-classical gait.

Everything visible — :func:`~leggedsnake.video`,
:func:`~leggedsnake.plot_pareto_front`,
:func:`~leggedsnake.plot_gait_diagram`,
:func:`~leggedsnake.plot_optimization_dashboard` — is for *inspecting*
results, not driving them.  The optimizer hands you a Walker; the
visualizer tells you whether it walks the way you wanted.

Where to go next
================

* :doc:`migration_world_config` if you have legacy code using the
  ``params`` dict.
* The grouped API toctree on :doc:`index` covers each module in detail.
* The numbered tutorial notebooks (``examples/01_walkers_gallery.ipynb``
  through ``examples/04_multi_objective_and_gait.ipynb``) walk through
  the full pipeline end-to-end.
