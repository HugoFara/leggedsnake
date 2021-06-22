# leggedsnake

This package aims to provide reliable computation techniques in Python to build, simulate and optimize planar [legged mechanisms](https://en.wikipedia.org/wiki/Leg_mechanism). It is divided in three main parts:
* Linkage conception in simple Python and kinematic optimization relying on pylinkage.
* Leg mechanism definition, with ``Walker`` heriting from the ``Linkage`` class.
* Dynamic optimization thanks to genetic algorithms

## Requirements

Python 3, numpy for calculation, matplotlib for drawing, and standard libraries. 

For kinematic optimization you can either use the built-in algorithm, or [PySwarms](https://pyswarms.readthedocs.io/en/latest/), under MIT license. PySwarms is a much more complexe package which provides quick calculations, however with modern laptops the built-in swarm optimization should be quick enough to fit your needs.

Dynamic optimization relies on multiple packages. First of all it uses [Pymunk](http://www.pymunk.org/en/latest/index.html), made by Victor Blomqvist, as a physics engine. Then you can either use the built-in algorithm, or the GA module from [PyGAD](https://pygad.readthedocs.io/en/latest/). PyGAD is a complete library providing much more than genetic algorithms, so it might be heavy. Once again, the benchmarks showed than PyGAD is way quicker than the built-in, however the dynamic simulation takes all the execution time so feel frre to use the package that fits the most to your needs.

## Usage

The demo script is [strider.py](https://github.com/HugoFara/leggedsnake/blob/main/leggedsnake/examples/strider.py), which demonstrates all the techniques about the [Strider linkage](https://www.diywalkers.com/strider-linkage-plans.html).


## Requirements

Python 3, numpy for calculation, matplotlib for drawing, pylinkage and pygad. If you do not want to use Pygad, the built-in evolutive algorithms should be enough, even if less flexible.
