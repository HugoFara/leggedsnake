# leggedsnake

[![PyPI version fury.io](https://badge.fury.io/py/leggedsnake.svg)](https://pypi.python.org/pypi/leggedsnake/)
[![Downloads](https://static.pepy.tech/personalized-badge/leggedsnake?period=total&units=international_system&left_color=grey&right_color=green&left_text=downloads)](https://pepy.tech/project/leggedsnake)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg )](https://raw.githubusercontent.com/HugoFara/leggedsnake/main/LICENSE.rst)

LeggedSnake is a Python toolkit for designing, simulating, and optimizing
planar walking linkages. It layers a pymunk physics engine and multi-objective
optimizers on top of [pylinkage](https://github.com/HugoFara/pylinkage)'s
kinematic model, so you can go from a mechanism sketch to an evolved walker
in a few lines of code.

![Strider optimized over 10 generations](https://github.com/HugoFara/leggedsnake/raw/main/examples/images/Striders%20run.gif)

*One Strider mechanism optimized over 10 generations of a genetic
algorithm, rendered walking on flat ground with 4 phase-offset legs
(2 per side, mirrored).*

## Installation

From PyPI:

```bash
pip install leggedsnake
```

From source with [uv](https://docs.astral.sh/uv/):

```bash
git clone https://github.com/hugofara/leggedsnake
cd leggedsnake
uv sync
```

## Quick start

A walker in five lines — the canonical Theo Jansen "Strandbeest" built
from the Holy Numbers, mirrored left/right and cloned into four
phase-offset pairs for stance stability, rendered in a live pyglet
window:

```python
import leggedsnake as ls

walker = ls.Walker.from_jansen(scale=0.1)
walker.add_opposite_leg()            # mirror for left/right pair
walker.add_legs(3)                   # 4 legs per side → 8-leg Strandbeest
ls.video(walker, duration=10)        # live simulation
```

![Theo Jansen Strandbeest walking](https://github.com/HugoFara/leggedsnake/raw/main/examples/images/Jansen%20quick%20start.gif)

Fewer legs pitch 10–20° and can tip over; four legs per side keeps at
least two feet in stance at every crank angle, which is why real
Strandbeests have many legs.

Other classical mechanisms ship as one-line factories too:
`Walker.from_strider`, `Walker.from_klann`, `Walker.from_chebyshev`,
`Walker.from_watt`, and `Walker.from_catalog` (pylinkage's topology
catalog).

To build a custom mechanism, declare its topology (nodes + edges) and
dimensions separately:

```python
from math import tau
import leggedsnake as ls
from leggedsnake import (
    HypergraphLinkage, Node, Edge, NodeRole,
    Dimensions, DriverAngle, Walker,
)

hg = HypergraphLinkage(name="MyWalker")
for node_id, role in [
    ("frame", NodeRole.GROUND), ("frame2", NodeRole.GROUND),
    ("crank", NodeRole.DRIVER),
    ("upper", NodeRole.DRIVEN), ("foot", NodeRole.DRIVEN),
]:
    hg.add_node(Node(node_id, role=role))
for edge in [
    ("frame_crank", "frame", "crank"), ("frame2_upper", "frame2", "upper"),
    ("crank_upper", "crank", "upper"),
    ("crank_foot", "crank", "foot"), ("upper_foot", "upper", "foot"),
]:
    hg.add_edge(Edge(*edge))

dims = Dimensions(
    node_positions={
        "frame": (0, 0), "frame2": (2, 0),
        "crank": (1, 0), "upper": (1, 2), "foot": (1, 3),
    },
    edge_distances={
        "frame_crank": 1.0, "frame2_upper": 2.24,
        "crank_upper": 2.0, "crank_foot": 3.16, "upper_foot": 1.0,
    },
    driver_angles={"crank": DriverAngle(angular_velocity=-tau / 12)},
)

walker = Walker(hg, dims, name="My Walker")
walker.add_opposite_leg(axis_x=1.0)  # mirror for left/right pair
walker.add_legs(1)                   # add a phase-offset copy
ls.video(walker)
```

## What you can do with it

| Capability | Entry points |
| --- | --- |
| Build mechanisms | `Walker`, `HypergraphLinkage`, `Walker.from_strider/jansen/klann/chebyshev/watt/catalog` |
| Kinematic fitness | `leggedsnake.utility.stride`, `leggedsnake.utility.step` |
| Physics simulation | `World`, `video`, `all_linkages_video`, `video_debug` |
| Dynamic fitness | `DistanceFitness`, `EfficiencyFitness`, `StrideFitness`, `StabilityFitness`, `CompositeFitness` |
| Single-objective GA | `GeneticOptimization`, `genetic_algorithm_optimization` |
| Multi-objective (NSGA) | `nsga_walking_optimization`, `NsgaWalkingConfig` |
| Topology co-design | `topology_walking_optimization`, `optimize_walking_mechanism` |
| Gait & stability | `analyze_gait`, `StabilityTimeSeries`, `compute_tip_over_margin` |
| Export | `to_urdf` (ROS), `save_walker` (JSON), `save_walker_svg` |
| Plotting | `plot_pareto_front`, `plot_gait_diagram`, `plot_foot_trajectories`, `plot_optimization_dashboard` |

## Documentation

- **[Concepts guide](https://hugofara.github.io/leggedsnake/concepts.html)**
  — orientation to the three core ideas: topology + dimensions = walker,
  the `DynamicFitness` protocol, and the optimizer landscape from
  fast-kinematic to dynamic-multi-objective.
- **[`params` → `WorldConfig` migration guide](https://hugofara.github.io/leggedsnake/migration_world_config.html)**
  — for code written against the legacy global `params` dict.
- **[Full API reference](https://hugofara.github.io/leggedsnake/)** — modules
  grouped by capability (Mechanism, Physics, Evaluation, Optimization,
  I/O & Plotting).

## Tutorials and deeper examples

Start with the numbered notebooks — they walk through the full pipeline end
to end:

1. [`examples/01_walkers_gallery.ipynb`](examples/01_walkers_gallery.ipynb)
   — build and inspect the classical linkages.
2. [`examples/02_physics_and_fitness.ipynb`](examples/02_physics_and_fitness.ipynb)
   — physics simulation and fitness evaluation.
3. [`examples/03_genetic_optimization.ipynb`](examples/03_genetic_optimization.ipynb)
   — evolve a walker with the genetic algorithm.
4. [`examples/04_multi_objective_and_gait.ipynb`](examples/04_multi_objective_and_gait.ipynb)
   — NSGA Pareto fronts plus gait / stability analysis.

The scripted examples cover specific mechanisms and full pipelines:
[`strider.py`](examples/strider.py) (PSO + GA on the Strider),
[`theo_jansen.py`](examples/theo_jansen.py),
[`klann_linkage.py`](examples/klann_linkage.py),
[`chebyshev_linkage.py`](examples/chebyshev_linkage.py),
[`simple_fourbar.py`](examples/simple_fourbar.py),
[`simple_walker.py`](examples/simple_walker.py),
[`optimization_pipeline.py`](examples/optimization_pipeline.py),
[`compare_linkages.py`](examples/compare_linkages.py).

The two GIFs above are regenerated deterministically by
[`examples/generate_readme_gifs.py`](examples/generate_readme_gifs.py)
(matplotlib `PillowWriter`, headless).

## Tips for faster experiments

- **Visualize early and often.** Every optimizer will hand you a linkage with
  a better score; only the animation tells you whether it walks the way you
  wanted.
- **Don't start from a hand-tuned optimum.** A random starting population is
  more robust against collapsing into a nearby suboptimum.
- **Exploit symmetry.** A Strider half-leg has the same kinematic stride as
  the full mechanism and evaluates an order of magnitude faster. Use
  kinematic PSO on the reduced problem, then hand off the winner to a
  dynamic GA on the full mechanism.
  ![Kinematic half Strider](https://github.com/HugoFara/leggedsnake/raw/main/examples/images/Kinematic%20half-Strider.gif)
- **Checkpoint long runs.** `GeneticOptimization(..., startnstop="run.json")`
  resumes automatically on the next launch.
- **Wrap optimization scripts in `if __name__ == "__main__":`** — the GA and
  NSGA optimizers spawn worker processes.

## Contributing

Contributions, feature requests, and "look at this weird walker" submissions
are all welcome. See [CONTRIBUTING.md](CONTRIBUTING.md) for the developer
workflow, or drop by the
[GitHub discussions](https://github.com/HugoFara/leggedsnake/discussions).
A [star](https://github.com/HugoFara/leggedsnake/stargazers) or a link to
your favourite walker also helps.

## Quick links

- Documentation: [hugofara.github.io/leggedsnake](https://hugofara.github.io/leggedsnake/)
- Concepts: [concepts.html](https://hugofara.github.io/leggedsnake/concepts.html)
- Migration guide: [migration_world_config.html](https://hugofara.github.io/leggedsnake/migration_world_config.html)
- Source: [HugoFara/leggedsnake](https://github.com/HugoFara/leggedsnake)
- PyPI: [leggedsnake](https://pypi.org/project/leggedsnake/)
- Changelog: [CHANGELOG.md](CHANGELOG.md)
