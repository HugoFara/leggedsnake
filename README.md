[![PyPI version fury.io](https://badge.fury.io/py/leggedsnake.svg)](https://pypi.python.org/pypi/leggedsnake/)
# leggedsnake

LeggedSnake is a Python library providing reliable computationnal techniques to build, simulate and optimize planar [leg mechanisms](https://en.wikipedia.org/wiki/Leg_mechanism). It is divided in three main parts:
* Linkage conception in simple Python and kinematic optimization relying on [pylinkage](https://github.com/HugoFara/pylinkage).
* Leg mechanism definition, with ``Walker`` heriting from the ``Linkage`` class.
* Dynamic simulation and its optimization thanks to genetic algorithms.

## Installation
### Using pip
The package is hosted on PyPi as [leggedsnake](https://pypi.org/project/leggedsnake/), use:
``pip install leggedsnake``

### Setting up Virtual Environment
We provide an [environment.yml](https://github.com/HugoFara/leggedsnake/blob/master/environment.yml) file for conda. Use ``conda env update --file environment.yml --name leggedsnake-env`` to install the requirements in a separate environment. 

If you are looking for a development version, check the GitHub repo under [HugoFara/leggedsnake](https://github.com/HugoFara/leggedsnake). 

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

``Walker`` is just a herited class of ``Linkage``, with some useful methods, and behaves quite the same way.

### Kinematic optimization using Particle Swarm Optimization (PSO)
No change compared to a classic linkage optimization. You should use the ``step`` and ``stride`` method from the [utility module](https://github.com/HugoFara/leggedsnake/blob/master/leggedsnake/utility.py) as fitness functions. 
This set of rules should work well for a stride **maximisation** problem:
1. Rebuild the Walker with the provided set of dimensions, and do a complete turn.
1. If the Walker raise an UnbuildableError, its score is 0 (or ``- float('inf')`` if you use other evaluation functions.
1. Verify if it can pass a certain obstacke using ``step`` function. If not, its score is 0.
1. Eventually mesure the length of its stide with the ``stride`` function. Return this length as its score.

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

### Visualization
For this part we will focus on the [Strider linkage](https://www.diywalkers.com/strider-linkage-plans.html), an exemple file is provided at [strider.py](https://github.com/HugoFara/leggedsnake/blob/master/examples/strider.py). The linkage looks like this:
![A Kinematic representation of Strider linkage](https://github.com/HugoFara/leggedsnake/raw/master/leggedsnake/examples/images/Kinematic%20unoptimized%20Strider.gif)

Looks cool? Let's simulate it dynamically!

![Dynamic one-leg-pair Strider being tested](https://github.com/HugoFara/leggedsnake/raw/master/leggedsnake/examples/images/Dynamic%20unoptimized%20one-legged%20Strider.gif)

Oops! Here is what you get when you forget to add more legs! There is **real danger here**, because your walker crawls well, you will be able to optimize efficiently the "crawler", *which may be not your goal*. 

Let's add three more leg pairs. Why three? Many legs means more mass and constraints, so less yield and more intensive computations. On the other hand, we always want the center of mass over the [support line](https://en.wikipedia.org/wiki/Support_polygon), which means that if the walker begins to lift a foot (let's say a front foot), and another doesn't come on the ground ahead of it, the linkage will to fall nose to the ground. With more foots we make the "snooping" time shorter, and a total of four leg pairs is a minimum for this *unoptimized* version. 

Let's have a look at the artist:

![Dynamic four-leg-pair unoptimized Strider](https://github.com/HugoFara/leggedsnake/raw/master/leggedsnake/examples/images/Dynamic%20unoptimized%20strider.gif)

## Advice
Use the vizualisation tools provided! The optimization tools should always give you a score with a better fitness, but it might not be what you expected. Tailor your optimization and *then* go for a long run will make you save a lot of time.

**Do not** use optimized linkages from the start! The risk is to fall to quickly into a suboptimal solution. They are several mechanisms to prevent that (starting from random position), however it can always have an impact on the rest of the optimization.

Try to minimize the number of elements in the optimizations! You can often use some linkage's properties to reduce the number of simulation parameters. For instance, the Strider linkage has an axial symmetry. While it is irrelevant to use this property in dynamic simulation, you can use "half" your Strider in a kinematic optimization, which is much faster:
![A Kinematic half Strider](https://github.com/HugoFara/leggedsnake/raw/master/leggedsnake/examples/images/Kinematic%20half-Strider.gif)
