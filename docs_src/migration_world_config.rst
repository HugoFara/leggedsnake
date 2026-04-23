============================================
Migrating from ``params`` to ``WorldConfig``
============================================

LeggedSnake 0.5.0 retires the global ``params`` dict in favour of a
structured :class:`~leggedsnake.WorldConfig` dataclass.  The dict is
still importable for backward compatibility, but **no library code
reads it any more** — only what you pass into ``WorldConfig`` (or the
default ``DEFAULT_CONFIG``) actually drives a simulation.  Mutating
``params`` after the move is silent dead code; this guide shows the
one-to-one replacement.

Why the change
==============

* **Discoverability.** Dataclass fields show up in autocompletion,
  type-checkers, and ``help(WorldConfig)``; nested dict keys do not.
* **Immutability by default.** ``WorldConfig`` is a frozen-by-convention
  dataclass — pass a fresh one per simulation instead of mutating
  shared global state.
* **Validation.** Field types are explicit; mistyped keys fail at
  construction rather than silently being ignored.
* **Composition.** Terrain knobs live on
  :class:`~leggedsnake.TerrainConfig`, so you can reuse a tuned
  terrain across many runs by passing the same ``terrain=`` instance.

Quick-reference mapping
=======================

.. list-table::
   :header-rows: 1
   :widths: 45 45 10

   * - Legacy ``params`` key
     - ``WorldConfig`` field
     - Notes
   * - ``params["physics"]["gravity"]``
     - ``WorldConfig.gravity``
     - ``tuple[float, float]``
   * - ``params["simul"]["physics_period"]``
     - ``WorldConfig.physics_period``
     - seconds
   * - ``params["linkage"]["torque"]``
     - ``WorldConfig.torque``
     - **Default lowered** ``1e3`` → ``1e2`` N·m in 0.5.0
   * - ``params["linkage"]["load"]``
     - ``WorldConfig.load_mass``
     - kg; now applied automatically in
       :meth:`World.add_linkage`
   * - ``params["ground"]["friction"]``
     - ``WorldConfig.ground_friction``
     - global ground friction
   * - ``params["ground"]["slope"]``
     - ``WorldConfig.terrain.slope``
     - radians
   * - ``params["ground"]["max_step"]``
     - ``WorldConfig.terrain.max_step``
     - meters
   * - ``params["ground"]["step_freq"]``
     - ``WorldConfig.terrain.step_freq``
     - probability
   * - ``params["ground"]["noise"]``
     - ``WorldConfig.terrain.noise``
     - 0–1
   * - ``params["ground"]["section_len"]``
     - ``WorldConfig.terrain.section_len``
     - meters

Side-by-side: a typical setup
=============================

Before — mutating the global dict:

.. code-block:: python

   import leggedsnake as ls

   ls.params["physics"]["gravity"] = (0, -5.0)
   ls.params["simul"]["physics_period"] = 0.01
   ls.params["linkage"]["torque"] = 50
   ls.params["linkage"]["load"] = 8.0
   ls.params["ground"]["friction"] = 0.7
   ls.params["ground"]["slope"] = 0.0
   ls.params["ground"]["noise"] = 0.0

   world = ls.World()
   world.add_linkage(walker)

After — structured config:

.. code-block:: python

   import leggedsnake as ls

   cfg = ls.WorldConfig(
       gravity=(0, -5.0),
       physics_period=0.01,
       torque=50,
       load_mass=8.0,
       ground_friction=0.7,
       terrain=ls.TerrainConfig(slope=0.0, noise=0.0),
   )

   world = ls.World(config=cfg)
   world.add_linkage(walker)  # load_mass is applied automatically

Terrain presets
===============

Common terrain setups ship as :class:`~leggedsnake.TerrainPreset`
values, so you don't have to hand-tune the dozen ``TerrainConfig``
fields:

.. code-block:: python

   from leggedsnake import TerrainConfig, TerrainPreset, WorldConfig

   cfg = WorldConfig(terrain=TerrainConfig.from_preset(TerrainPreset.ROUGH))

Available presets: ``FLAT``, ``HILLY``, ``ROUGH``, ``STAIRS``,
``MIXED``, ``SLOPE_UP``, ``SLOPE_DOWN``, ``SINUSOIDAL``.

New ``WorldConfig`` fields with no legacy equivalent
====================================================

These were introduced alongside ``WorldConfig`` and are *only*
reachable through the new API:

* ``payload_offset: tuple[float, float]`` — offset the chassis centre
  of gravity (body-local meters) to simulate uneven payload.
* ``wind_force: tuple[float, float]`` — constant external force (N)
  applied to the chassis every physics step.  ``(+x, 0)`` models a
  steady headwind.
* ``drag_coefficient: float`` — linear drag (``F = -c·v``) applied to
  the chassis.  ``0`` disables drag.

On the terrain side: ``seed``, ``friction_range``, ``gap_freq`` /
``gap_width``, ``obstacle_freq`` / ``obstacle_height`` /
``obstacle_width``, ``slope_profile``, ``wave_period``,
``wave_sweep_rate``.  See :class:`~leggedsnake.TerrainConfig`.

Legacy-only knobs (no ``WorldConfig`` mapping)
==============================================

A handful of ``params`` keys had no live consumer at the time of the
0.5.0 refactor and were intentionally not migrated:

* ``params["linkage"]["crank_len"]`` — was already documented as
  "unused" in the legacy code.
* ``params["linkage"]["masses"]`` — bar masses are now derived from
  edge geometry by ``hypergraph_physics``, not a global linear
  density.
* ``params["physics"]["max_force"]`` — pymunk's per-constraint force
  limit; was never threaded into the physics step.

If you actually need any of these, please open an issue on the
`tracker <https://github.com/HugoFara/leggedsnake/issues>`_ describing
the use case.
