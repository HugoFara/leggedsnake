# leggedsnake

This package aims to provide reliable computation techniques in Python to build, simulate and optimize planar [leg mechanisms](https://en.wikipedia.org/wiki/Leg_mechanism). It is divided in three main parts:
* Linkage conception in simple Python and kinematic optimization relying on [pylinkage](https://github.com/HugoFara/pylinkage).
* Leg mechanism definition, with ``Walker`` heriting from the ``Linkage`` class.
* Dynamic optimization thanks to genetic algorithms.

## Installation
### Using pip
To download the package from PyPi, use:
``pip install leggedsnake``

### Setting up Virtual Environment
We provide an [environment.yml](https://github.com/HugoFara/leggedsnake/environment.yml) file for conda. Use ``conda env update --file environment.yml --name leggedsnake-env`` to install the requirements in a separate environment. 

## Requirements

Python 3, numpy for calculation, matplotlib for drawing, and standard libraries. 

For kinematic optimization you can either use the built-in algorithm, or [PySwarms](https://pyswarms.readthedocs.io/en/latest/), under MIT license. PySwarms is a much more complexe package which provides quick calculations, however with modern laptops the built-in swarm optimization should be quick enough to fit your needs.

Dynamic optimization relies on multiple packages. First of all it uses [Pymunk](http://www.pymunk.org/en/latest/index.html), made by Victor Blomqvist, as a physics engine. Then you can either use the built-in algorithm, or the GA module from [PyGAD](https://pygad.readthedocs.io/en/latest/). PyGAD is a complete library providing much more than genetic algorithms, so it might be heavy. PyGAD is more complete than the built-in however, so I haven't decided to continue on PyGAD or switch for another solution in the future.

## Usage

The demo script is [strider.py](https://github.com/HugoFara/leggedsnake/blob/main/leggedsnake/examples/strider.py), which demonstrates all the techniques about the [Strider linkage](https://www.diywalkers.com/strider-linkage-plans.html).

### Defining a ``Walker``
First, you need to define joints for your ``Walker`` as described in [pylinkage](https://github.com/HugoFara/pylinkage) documentation. Once your joints (let's say they are in a joint object), you should have something like that:
```python
from pylinkage.linkage import Static, Pivot, Fixed, Crank

from leggedsnake.walker import Walker

# Center of the Walker
A = Static(x=0, y=0, name="A")
B = Crank(1, 0, distance=1, angle=0.31, name="Crank")
# etc... let's say with have joints up to E
my_walker = Walker(
  joints=(A, B, C, D, E),
  name="My Walker"
)
```

``Walker`` is an just herited class of ``Linkage``, with some useful methods, and behaves quite the same.

### Kinematic optimization using Particle Swarm Optimization (PSO)
No change compared to a classic linkage optimization. You should use the ``step`` and ``stride`` method from the [utility module](https://github.com/HugoFara/leggedsnake/blob/main/leggedsnake/leggedsnake/utility.py) as fitness functions. 
This set of rules should work well for a stride **maximisation** problem:
#. Rebuild the Walker with the provided set of dimensions, and do a complete turn.
#. If the Walker raise an UnbuildableError, its score is 0 (or ``- float('inf')`` if you use other evaluation functions.
#. Verify if it can pass a certain obstacke using ``step`` function. If not, its score is 0.
#. Eventually mesure the length of its stide with the ``stride`` function. Return this length as its score.

This optimization is really quick, however it is only kinematically based, so you should be precise on what kind of movement you want.

### Dynamic Optimization using Genetic Algorithm (GA)
Kinematic optimization is fast, however it can return weird results, and it has no sense of gravity while walking heavily relies on gravity. This is why you may need to use dynamic optimization thanks to [Pymunk](http://www.pymunk.org/en/latest/index.html). However the calculation is much more slower, and you can no longer tests millions of linkages as in PSO (or you will need time). This is why we use [genetic algorithm](https://en.wikipedia.org/wiki/Genetic_algorithm), because it can provide good results with less parents.

We handle everything almost evything world definition to linkage conversion. Appart from the GA parameters, you just have to define a fitness function. Here are the main steps for a **maximisation problem**:
1. Create a function of two arguments, the first one should be the paramaters of the linkage, the second the initial positions for the joints.
2. Try to do a revolution in **kinematic simulation**. If the Walker raises an ``UnbuildableError`` set its score to ``-float('inf')``.
3. Otherwise use this procedure 
```python
from leggedsnake import physicsengine as pe

def dynamic_linkage_fitness(walker):
  """
  Make the dynamic evalutaion of a Walker.
  
  Return yield and initial position of joints.
  """
  world = pe.World()
  # We handle all the conversions
  world.add_linkage(walker)
  # Simulation duration (in seconds)
  duration = 40
  # Somme of yields
  tot = 0
  # Motor turned on duration
  dur = 0
  n = duration * pe.params["camera"]["fps"]
  n /= pe.params["simul"]["time_coef"]
  for j in range(int(n)):
      efficiency, energy = world.update(j)
      tot += efficiency
      dur += energy
  if dur == 0:
      return - float('inf'), list()
  print("Score:", tot / dur)
  # Return 100 times average yield, and initial positions as the final score
  return tot / dur, pos
```

And now, relax while your computer recreates a civilisation of walking machines!
