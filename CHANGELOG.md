# Changelog
All notable changes to the LeggedSnake will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Fixed
 - `data_descriptors` were not save for the first line of data only in 
   ``geneticoptimizer``. 
   
## [0.3.0-beta] - 2021-07-21
### Added
 - Multiprocessing is here! The genetic optimization can now be run in parallel!
   Performances got improved by 65 % using 4 processes only.

### Changed
 - We now save data using JSON! Slow computer users, you can relax and stop 
   computing when you want.
 - The sidebar in the documentation a bit more useful.
 - Not having tqdm will cause an exception.

### Fixed
 - Corrected the example, the genetic optimization is now properly fixed but
slower.

### Removed
 - native support for PyGAD is no longer present.
 - ``evolutionnary_optimization`` (replaced by ``evolutionary_optimization``).
 - Data saved in the old txt format are no longer readable (were they?)

## [0.2.0-alpha] - 2021-07-14
### Added
 - Dependency to [tqdm](https://tqdm.github.io/) and matplotlib.
 - The ``evolutionary_optimization`` replaces ``evolutionnary_optimization``.
   - The ``ite`` parameter renamed ``iters`` for consistency with pylinkage.
   - The new parameter ``verbose`` let you display a nice progress bar, more 
     information on optimization state, or nothing.
 - The best solution can be displayed with PyGAD as well.

### Changed
 - Typos and cleans-up in ``docs/examples/strider.py``.
 - ``evolutionnary_optimization_legacy`` renamed to 
   ``evolutionary_optimization_builtin``.
   
### Deprecated
 - ``evolutionnary_optimization`` is now deprecated. Please use 
   ``evolutionary_optimization``.

### Removed
 - Explicit dependency to PyGAD. There is no longer an annoying message when 
   PyGAD is not installed.

## [0.1.4-alpha] - 2021-07-12
### Added
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

### Changed
 - ``docs/examples/strider.py`` has been updated to the latest version of 
   leggedsnake 0.1.4.
   
### Fixed
 - The full swarm representation in polar graph has been repaired in 
   ``docs/examples/strider.py``.
 - During a dynamic simulation, linkages with long legs could appear through the
   road.
 - The documentation was not properly rendered because Napoleon (NumPy coding 
   style) was not integrated.

## [0.1.3-alpha] - 2021-07-10
This package was lacking real documentation, it is fixed in this version.
### Added
 - Sphinx documentation!
 - Website hosted on GitHub pages, check 
   [hugofara.github.io/leggedsnake](https://hugofara.github.io/leggedsnake/)!
 - Expanded README with the quick links section.

### Changed
 - Tests moved from ``leggedsnake/tests`` to ``tests/``.
 - Examples moved from ``leggedsnake/examples/`` to ``docs/examples/``.
 - I was testing my code on ``leggedsnake/examples/strider.py``  (old path) and
   that's why it was a big mess. I cleaned up that all. Sorry for the 
   inconvenience!

### Fixed
 - A lot of outdated code in the ``leggedsnake/examples/strider.py``
 - Changelog URL was broken in ``setup.cfg``.

## [0.1.2-alpha] - 2021-07-07
### Changed
 - The ``step`` function execution speed has been increased by 25% when 
   ``return_res`` is ``True``! Small performance improvement when ``return_res``
   is ``False``.
 - The ``size`` argument of ``step`` function is now known as ``witdh``.
 - We now require pylinkage>=0.4.0.

### Fixed
 - Files in ``leggedsnake/examples/`` were not included in the PyPi package.
 - The example was incompatible with pylinkage 0.4.0.
 - Test suite was unusable by tox.
 - Tests fixed.
 - Incompatible argument between PyGAD init_pop and built-in GA.

### Security
 - Tests with ``tox.ini`` now include Python 3.9 and Flake 8.

## [0.1.1-alpha] - 2021-06-26
### Added
 - The example file ``examples/strider.py`` is now shipped with the Python 
   package.
 - ``leggedsnake/geneticoptimizer.py`` can now automatically switch to the 
   built-in GA algorithm if PyGAD is not installed.

### Changed
 - ``setup.cfg`` metadata

## [0.1.0-alpha] - 2021-06-25
### Added
 - Code vulnerabilities automatic checks
 - Example videos in ``examples/images/``

### Changed
 - Manny reforms in code style, to make the dynamic part naming conventions 
   consistent with Pymunk. 
 - Images in the ``README.md``!

### Fixed
 - You can now define linkages with an enormous number of legs. Systems with
   many should no longer break physics but your CPU instead :)

## [0.0.3-alpha] - 2021-06-23
### Added
 - Started walktrough demo in ``README.md``
 - Automatic release to PyPi

### Fixed
 - Pymunk version should be at least 6.0.0 in requirement files.
 - Some URLs typos in ``README.md``
 - Versioning tests not executing (GitHub action)

## [0.0.2-alpha] - 2021-06-22
### Added
- ``requirement.txt`` was absent due to ``.gitignore`` misconfiguration.

### Changed
 - ``.gitignore`` now ignores .txt files only in the leggedsnake folder.
 - ``environment.yml`` more flexible (versions can be superior to the selected). 
   pymunk>5.0.0 and pylinkage added.
 - ``leggedsnake/utility.py`` not having zipfile or xml modules error 
   encapsulation.

### Fixed
 - ``setup.cfg`` was not PyPi compatible. Removed mail (use GitHub!), we now 
   explicitly say that ``README.md`` is markdown (PyPi is conservative)
   
## [0.0.1-alpha] - 2021-06-22
Basic version, supporting Genetic Algorithm optimization, but with various 
problems.
### Added
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
 - ``tox.ini`` tox with Python 3.7 and 3.8