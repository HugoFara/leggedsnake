# Changelog
All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]
### Added
 - The example file ``examples/strider.py`` is now shipped with the Python package.
 - ``leggedsnake/geneticoptimizer.py`` can now automatically switch to the built-in GA algorithm if PyGAD is not installed.

### Changed
 - ``setup.cfg`` metadata

## [0.1.0-alpha] - 2021-06-23
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
- ``requirement.txt`` was absent due to ``.gitignore`` missconfiguration.

### Changed
 - ``.gitignore`` ignore txt file only in leggedsnake folder
 - ``environment.yml`` more flexible (versions can be superior to the selected). Added pymunk>5.0.0, pylinkage
 - ``leggedsnake/utility.py`` not having zipfile or xml module error encapsulation

### Fixed
 - ``setup.cfg`` was not PyPi compatibl.
     Removed mail (use GitHub!), explicitly say that ``README.md`` is markdown (PyPi is conservative)


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


