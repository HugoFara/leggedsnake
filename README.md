# leggedsnake

[![PyPI version fury.io](https://badge.fury.io/py/leggedsnake.svg)](https://pypi.python.org/pypi/leggedsnake/)
[![Downloads](https://static.pepy.tech/personalized-badge/leggedsnake?period=total&units=international_system&left_color=grey&right_color=green&left_text=downloads)](https://pepy.tech/project/leggedsnake)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg )](https://raw.githubusercontent.com/HugoFara/leggedsnake/main/LICENSE.rst)

LeggedSnake makes the simulation of walking linkages fast and easy.
We believe that building walking linkages is fun and could be useful.
Our philosophy is to provide a quick way of building, optimizing and testing walking linkages.

*An example of a Strider linkage built with the project, failing to climb a slope.*

## Usage

First, you define the linkage you want to use.

```python3
import leggedsnake as ls

# Quick definition of a linkage as a dict of joints
linkage = {
    "A": ls.Static(x=0, y=0, name="A"),
    "B": ls.Crank(1, 0, distance=1, angle=0.31, name="Crank")
    # etc...
}
# Conversion to a dynamic linkage
my_walker = ls.Walker(
    joints=linkage.values(),
    name="My Walker"
)
# It is often faster to add pairs of legs this way
my_walker.add_legs(3)


# Then, run launch a GUI simulation with
ls.video(my_walker)
```
It should display something like the following.

![Dynamic four-leg-pair unoptimized Strider](https://github.com/HugoFara/leggedsnake/raw/main/docs/examples/images/Dynamic%20unoptimized%20strider.gif)

The next step is to optimize your linkage. We use a genetic algorithm here.

```python
# Definition of an individual as (fitness, dimensions, initial coordinates)
dna = [0, list(my_walker.get_num.constraints()), list(my_walker.get_coords())]
population = 10

# Prepare the optimization, with any fitness_function(dna) -> score 
optimizer = ls.GeneticOptimization(
        dna=dna, 
        fitness=fitness_function,
        max_pop=population,
)
# Run for 100 iterations, on 4 processes
optimized_walkers = optimizer.run(iters=100, processes=4)

# The following line will display the results
ls.all_linkages_video(optimized_walkers)
```
For 100 iterations, 10 linkages will be simulated and evaluated by fitness_function.
The fittest individuals are kept and will propagate their genes (with mutations).

Now you should see something like the following.

![10 optimized striders](https://github.com/HugoFara/leggedsnake/raw/main/docs/examples/images/Striders%20run.gif)

This is a simulation from the last generation of 10 linkages. 
Most of them cover a larger distance (this is the target of our ``fitness_function``).

Finally, only the best linkage may be kept. 
As , you can find it at ``optimized_linkages[0]``.

```python
# Results are sorted by best fitness first, 
# so we use the walker with the best score
best_dna = optimized_walkers[0]

# Change the dimensions
my_walker.set_num_constraints(best_dna[1])
my_walker.set_coords(best_dna[2])

# Once again launch the video
ls.video(my_walker)
```

![Dynamic optimized Strider](https://github.com/HugoFara/leggedsnake/raw/main/docs/examples/images/Dynamic%20optimized%20strider.gif)

So now it has a small ski pole, does not fall and goes much farther away!

## Installation

The package is hosted on PyPi as [leggedsnake](https://pypi.org/project/leggedsnake/), use:

```bash
pip install leggedsnake
```

## Build from source

Download this repository.

```shell
git clone https://github.com/hugofara/leggedsnake
```

### Conda Virtual Environment

We provide an [environment.yml](https://github.com/HugoFara/leggedsnake/blob/main/environment.yml) file for conda.

```shell
conda env update --file environment.yml --name leggedsnake-env
``` 
It will install the requirements in a separate environment.

### Other installation

If you are looking for a development version, check the GitHub repo under
[HugoFara/leggedsnake](https://github.com/HugoFara/leggedsnake).

## Main features

We handle planar [leg mechanisms](https://en.wikipedia.org/wiki/Leg_mechanism) in three main parts:

* Linkage conception in simple Python relies on [pylinkage](https://github.com/HugoFara/pylinkage).
* *Optional* kinematic optimization with ``Walker`` class, inherits from pylinkage's ``Linkage`` class.
* Dynamic simulation and its optimization use genetic algorithms.

## Usage

The demo script is [strider.py](https://github.com/HugoFara/leggedsnake/blob/main/docs/examples/strider.py), which
demonstrates all the techniques about the [Strider linkage](https://www.diywalkers.com/strider-linkage-plans.html).

In a nutshell, you have at least two parts:

1. Define a Linkage.
2. Run the optimization. 


### Defining a ``Walker``

First, you need to define joints for your ``Walker`` as described in [pylinkage](https://github.com/HugoFara/pylinkage)
documentation. Once your joints (let's say they are in a joint object), you should have something like that:

```python3
import leggedsnake as ls

# Center of the Walker
linkage = {
    "A": ls.Static(x=0, y=0, name="A"),
    "B": ls.Crank(1, 0, distance=1, angle=0.31, name="Crank")
    # etc...
}
my_walker = ls.Walker(
    joints=linkage.values(),
    name="My Walker"
)
```

``Walker`` is just an inherited class of ``Linkage``, with some useful methods, and behaves quite the same way.

### Dynamic Optimization using Genetic Algorithm (GA)

Kinematic optimization is fast, but it can return weird results, and it has no sense of gravity while walking heavily
relies on gravity. This is why you may need to use dynamic optimization thanks to
[Pymunk](http://www.pymunk.org/en/latest/index.html). However, the calculation is much slower, and you can no
longer test millions of linkages as in PSO (or you will need time). This is why we
use [genetic algorithm](https://en.wikipedia.org/wiki/Genetic_algorithm), because it can provide good results with
fewer parents.

We handle everything almost everything world definition to linkage conversion. Apart from the GA parameters, you just
have to define a fitness function. Here are the main steps for a **maximisation problem**:

1. Create a function of two arguments, the first one should be the parameters of the linkage, the second the initial positions for the joints.
2. Try to do a revolution in **kinematic simulation**. If the Walker raises an ``UnbuildableError`` set its score to ``-float('inf')``.
3. Otherwise, use the following procedure:

```python3
import leggedsnake as ls

def total_distance(walker):
    """
    Evaluates the final horizontal position of the input linkage.
    
    Return final distance and initial position of joints.
    """
    pos = tuple(walker.step())[-1]
    world = ls.World()
    # We handle all the conversions
    world.add_linkage(walker)
    # Simulation duration (in seconds)
    duration = 30
    steps = int(duration / ls.params["simul"]["physics_period"])
    for _ in range(steps):
        world.update()
    return world.linkages[0].body.position.x, pos
```

And now, relax while your computer creates a civilization of walking machines!

### Visualization

For this part we will focus on the [Strider linkage](https://www.diywalkers.com/strider-linkage-plans.html),
an example file is provided at ``docs/examples/strider.py``.

The linkage looks like this:
![A Kinematic representation of Strider linkage](https://github.com/HugoFara/leggedsnake/raw/main/docs/examples/images/Kinematic%20unoptimized%20Strider.gif)

Looks cool? Let's simulate it dynamically!

![Dynamic one-leg-pair Strider being tested](https://github.com/HugoFara/leggedsnake/raw/main/docs/examples/images/Dynamic%20unoptimized%20one-legged%20Strider.gif)

Oops! Here is what you get when you forget to add more legs! There is **real danger here**, because your walker crawls
well, you will be able to optimize efficiently the "crawler", *which may be not your goal*.

Let's add three more leg pairs. Why three? Many legs mean more mass and constraints, so less yield and more intensive
computations. On the other hand, we always want the center of mass over the
[support line](https://en.wikipedia.org/wiki/Support_polygon), which means that if the walker begins to lift a foot
(let's say a front foot), and another doesn't come on the ground ahead of it, the linkage will fall nose to the
ground. With more feet, we make the "snooping" time shorter, and a total of four leg pairs is a minimum for this
*unoptimized* version.

A simple way to do it is:

```python
my_linkage.add_legs(3) # Replace "my_linkage" with your Walker object
```

Let's have a look at the artist:

![Dynamic four-leg-pair unoptimized Strider](https://github.com/HugoFara/leggedsnake/raw/main/docs/examples/images/Dynamic%20unoptimized%20strider.gif)


### Kinematic optimization using Particle Swarm Optimization (PSO)

This is a kinematic linkage optimization, depending solely on pylinkage. 
You should use the ``step`` and ``stride`` method from the
[utility module](https://github.com/HugoFara/leggedsnake/blob/main/leggedsnake/utility.py) as fitness functions.
This set of rules should work well for a stride **maximisation** problem:

1. Rebuild the Walker with the provided set of dimensions, and do a complete turn.
2. If the Walker raises an UnbuildableError, its score is 0 (or ``-float('inf')`` if you use other evaluation functions).
3. Verify if it can pass a certain obstacle using ``step`` function. If not, its score is 0.
4. Eventually measure the length of its stride with the ``stride`` function. Return this length as its score.

## Advice

Use the visualisation tools provided! The optimization tools should always give you a score with a better fitness,
but it might not be what you expected. Tailor your optimization and *then* go for a long run will make you save a lot
of time.

**Do not** use optimized linkages from the start! The risk is to fall to quickly into a suboptimal solution. They are
several mechanisms to prevent that (starting from random position), but it can always have an impact on the rest of
the optimization.

Try to minimize the number of elements in the optimizations! You can often use some linkage properties to reduce the
number of simulation parameters. For instance, the Strider linkage has axial symmetry. While it is irrelevant to use
this property in dynamic simulation, you can use "half" your Strider in a kinematic optimization, which is much faster.

![A Kinematic half Strider](https://github.com/HugoFara/leggedsnake/raw/main/docs/examples/images/Kinematic%20half-Strider.gif)

## Contribute

This project is open to contribution and actively looking for contributors.
You can help making it better!

### For everyone

You can [drop a star](https://github.com/HugoFara/leggedsnake/stargazers),
[fork this project](https://github.com/HugoFara/leggedsnake/forks) or simply share the link to your best media.

The more people get engaged into this project, the better it will develop!

### For developers

You can follow the guide at [CONTRIBUTING.md](CONTRIBUTING.md). Feel free to me any pull request.

## Quick links

* For the documentation, check the docs at [hugofara.github.io/leggedsnake](https://hugofara.github.io/leggedsnake/)!
* Source code is hosted on GitHub as [HugoFara/leggedsnake](https://github.com/HugoFara/leggedsnake)
* We also provide a Python package on PyPi, test [leggedsnake](https://pypi.org/project/leggedsnake/).
* If you just want to chill out looking at walking linkages striving to survive, join the [discussions](https://github.com/HugoFara/leggedsnake/discussions).

Contributors are welcome!
