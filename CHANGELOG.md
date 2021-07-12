# Changelog
All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]
## Added
 - We now use [bump2version](https://pypi.org/project/bump2version/) for version maintenance.

## [0.1.3-alpha] - 2021-07-10
This package was lacking real documentation, it is fixed in this version.
### Added
 - Sphinx documentation!
 - Web site hosted on GitHub pages, check [hugofara.github.io/leggedsnake](https://hugofara.github.io/leggedsnake/)!
 - Expanded README with the quick links section.

### Changed
 - Tests moved from ``leggedsnake/tests`` to ``tests/``.
 - Examples moved from ``leggedsnake/examples/`` to ``docs/examples/``.
 - I was testing my code on ``leggedsnake/examples/strider.py``  (old path) and that's why it was a big mess. I cleaned up that all. Sorry for the inconvenience!

### Fixed
 - A lot of outdated code in the ``leggedsnake/examples/strider.py``
 - Changelog URL was broken in ``setup.cfg``.

## [0.1.2-alpha] - 2021-07-07
### Changed
 - The ``step`` function execution speed has been increased by 25% when ``return_res`` is ``True``! Small performance improvement when ``return_res`` is ``False``.
 - The ``size`` argument of ``step`` function is now known as ``witdh``.
 - We now require pylinkage>=0.4.0.

### Fixed
 - Files in ``leggedsnake/examples/`` were not included in the PyPi package.
 - The example was incompatible with [pylinkage](https://pypi.org/project/pylinkage/) 0.4.0.
 - Test suite was unusable by tox.
 - Tests fixed.
 - Incompatible argument between PyGAD init_pop and built-in GA.

### Security
 - Tests with `tox.ini`` now include Python 3.9 and Flake 8.

## [0.1.1-alpha] - 2021-06-26
### Added
 - The example file ``examples/strider.py`` is now shipped with the Python package.
 - ``leggedsnake/geneticoptimizer.py`` can now automatically switch to the built-in GA algorithm if PyGAD is not installed.

### Changed
 - ``setup.cfg`` metadata

## [0.1.0-alpha] - 2021-06-25
### Added
 - Code vulnerabilities automatic checks
 - Example videos in ``examples/images/``

### Changed
 - Manny reforms in code style, to make the dynamic part naming conventions consistent with Pymunk. 
 - Images in the ``README.md``!

### Fixed
 - You can now define linkages with an enormous number of legs. Systems with many should no longer break physics but your CPU instead :)

## [0.0.3-alpha] - 2021-06-23
### Added
 - Started walktrough demo in ``README.md``
 - Automatic release to PyPi

### Fixed
 - Pymunk version should be at least 6.0.0 in requirement files.
 - Some URLs typos in ``README.md``
 - Versionning tests not executing (GitHub action)

## [0.0.2-alpha] - 2021-06-22
### Added
- ``requirement.txt`` was absent due to ``.gitignore`` misconfiguration.

### Changed
 - ``.gitignore`` now ignores .txt files only in the leggedsnake folder.
 - ``environment.yml`` more flexible (versions can be superior to the selected). pymunk>5.0.0 and pylinkage added.
 - ``leggedsnake/utility.py`` not having zipfile or xml modules error encapsulation.

### Fixed
 - ``setup.cfg`` was not PyPi compatible.
     Removed mail (use GitHub!), we now explicitly say that ``README.md`` is markdown (PyPi is conservative)
   
## [0.0.1-alpha] - 2021-06-22
Basic version, supporting Genetic Algorithm optimization, but with various problems.
### Added
 - ``CODE_OF_CONDUCT.md`` to help community
 - ``LICENSE`` MIT License
 - ``MANIFEST.in`` to include more files
 - ``README.md`` as a very minimal version
 - ``environment.yml`` with matplotlib, numpy, and pygad requirement
 - ``examples/strider.py`` a complete demo with Strider linkage
 - ``leggedsnake/__init__.py``
 - ``leggedsnake/dynamiclinkage.py``
 - ``leggedsnake/geneticoptimizer.py``
 - ``leggedsnake/physicsengine.py``
 - ``leggedsnake/show_evolution.py`` just a legacy package, no utility
 - ``leggedsnake/tests/test_utility.py`` untested test case
 - ``leggedsnake/utility.py`` contain some useful evalution function (``step`` and ``stride``) and a broken GeoGebra interface.
 - ``walker.py`` defines the ``Walker`` object.
 - ``pyproject.toml`` 
 - ``setup.cfg``
 - ``setup.py`` empty, for compatibility purposes only
 - ``tox.ini`` tox with Python 3.7 and 3.8