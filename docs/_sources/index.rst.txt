.. LeggedSnake documentation master file, created by
   sphinx-quickstart on Wed Jul  7 13:42:25 2021.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to LeggedSnake's documentation!
=======================================

.. toctree::
   :maxdepth: 2
   :caption: Introduction

   Readme <readmelink>
   changeloglink

.. toctree::
   :maxdepth: 2
   :caption: Guides

   concepts
   migration_world_config

.. toctree::
   :maxdepth: 1
   :caption: Mechanism

   api/walker
   api/utility

.. toctree::
   :maxdepth: 1
   :caption: Physics

   api/physics_engine
   api/dynamic_linkage
   api/hypergraph_physics
   api/world_visualizer

.. toctree::
   :maxdepth: 1
   :caption: Evaluation

   api/fitness
   api/stability
   api/gait_analysis

.. toctree::
   :maxdepth: 1
   :caption: Optimization

   api/walking_objectives
   api/genetic_optimizer
   api/nsga_optimizer
   api/topology_optimization
   api/co_design
   api/gait_optimization
   api/leg_count
   api/show_evolution

.. toctree::
   :maxdepth: 1
   :caption: I/O & Plotting

   api/serialization
   api/urdf_export
   api/plotting

.. toctree::
   :maxdepth: 2
   :caption: Quick Links

   Source Code <https://github.com/HugoFara/leggedsnake>
   Download on PyPi <https://pypi.org/project/leggedsnake/>
   Discuss <https://github.com/HugoFara/leggedsnake/discussions>

.. toctree::
   :maxdepth: 2
   :caption: See Also

   Pylinkage (Sister Project/Backend) <https://hugofara.github.io/pylinkage/>

.. mdinclude::
   ../README.md


Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
