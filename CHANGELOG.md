# Changelog

All notable changes to the LeggedSnake will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.4.0] - 2023-06-21

### Added in 0.4.0

- View all walkers!
  - ``show_all_walkers`` in ``docs/examples/strider.py`` let you see all walkers in one simulation!
  - You can set the color of walkers during display.
- Genetic optimization:
  - ``GeneticOptimization`` class in ``geneticoptimizer.py`` that will replace the previous functional paradigm.
  - The average score is now displayed.
- ``VisualWorld`` has a new method called ``reload_visuals``.
- ``show_evolution.py`` is a new script plotting various data about the Walkers population's evolution during 
genetic optimization.
- In ``docs/examples/strider.py`` we recommend to use ``total_distance`` as the fitness function.

### Changed in 0.4.0

- Genetic optimization:
  - During genetic optimization, population is now stable at max_pop (it used to fluctuate a lot).
  - Genetic optimization do no longer display all dimensions in the progress bar.
  - ``startnstop`` argument may now be the name of the file to use (a string).
  - ``max_genetic_distance`` was changed from 0.7 to 10. Results are much better now!
- Visuals:
  - ``update`` method of ``VisualWorld`` replaced by ``visual_update``. It clearly separates physics and display time.
  - Frame rate and physics speed are now independent parameters.
  - Visuals go to a new file ``worldvisualizer.py``.
  - Camera parameters should now be accessed from ``CAMERA`` instead of ``params["camera"]``.
  - The camera feels more cinematic.
- You can define a custom load when using ``World.add_linkage`` or ``VisualWorld.add_linkage``. The default is 0.
- ``pyproject.toml`` updated with the data of ``setup.cfg``. This is now the recommended metadata for the project. 
- In ``docs/example/strider.py``, simulation time was increased from 30 seconds to 40. It was just not enough.

### Fixed in 0.4.0

- Documentation of ``evolutionary_optimization_builtin`` was wrong: returned data were in order (fitness, dimensions, position),
but (fitness, position, dimensions) was indicated.
- After a genetic optimization, the example script was assigning wrong data to the demo walker.
- ``kwargs_switcher`` from ``geneticoptimizer.py`` do no longer pop (destroy) argument from the input dictionary.

### Deprecated in 0.4.0

- ``setup.cfg`` should no longer be used, as it is replaced by ``pyproject.toml``. 

### Removed in 0.4.0

- ``evolutionary_optimization`` function is removed. Use  ``GeneticOptimization`` class instead.
  - You can no longer use the argument "init_pop" to change the size of the initial population. 
  It now always set to max_pop.
- ``time_coef``, ``calc_rate`` and ``max_sub`` parameters of ``params["simul"]`` replaced by a unique 
``physics_period`` set to 0.02 (s).
- ``leggedsnake/Population evolution.json`` removed. 
It contained data about an evolution run and is not relevant for users.

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
- The ``video`` function of ``physicsengine.py`` now effectively launches the video (no call to plt.show required).
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
