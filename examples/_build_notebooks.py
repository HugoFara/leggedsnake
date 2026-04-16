"""Generate the numbered notebook series from compact Python specs.

Run once to (re)create examples/01_*.ipynb .. examples/04_*.ipynb. The
heavy lifting (execution, output capture) is delegated to
``jupyter nbconvert --to notebook --execute`` after this script writes
the empty-cell scaffolds.

Keep this helper in-tree as a regeneration tool — notebooks drift from
their markdown; rebuilding from source keeps the narrative the canonical
artifact.
"""

from __future__ import annotations

from pathlib import Path

import nbformat


def _md(src: str) -> nbformat.NotebookNode:
    return nbformat.v4.new_markdown_cell(src.strip() + "\n")


def _code(src: str) -> nbformat.NotebookNode:
    return nbformat.v4.new_code_cell(src.strip() + "\n")


def _build(cells: list[nbformat.NotebookNode], path: Path) -> None:
    nb = nbformat.v4.new_notebook()
    nb.cells = cells
    nb.metadata = {
        "kernelspec": {
            "display_name": ".venv",
            "language": "python",
            "name": "python3",
        },
        "language_info": {
            "codemirror_mode": {"name": "ipython", "version": 3},
            "file_extension": ".py",
            "mimetype": "text/x-python",
            "name": "python",
            "nbconvert_exporter": "python",
            "pygments_lexer": "ipython3",
        },
    }
    with path.open("w") as f:
        nbformat.write(nb, f)
    print(f"wrote {path}")


# ---------------------------------------------------------------------------
# 01 — Walkers gallery
# ---------------------------------------------------------------------------

NB01_INTRO = """
# A Gallery of Classical Walking Linkages

Leggedsnake ships **one-call factories** for the canonical planar leg
mechanisms, each preloaded with its published (or published-equivalent)
link lengths. This notebook walks through every factory, plots the foot
trajectory for one full crank revolution, and flags the design
trade-offs that distinguish each family.

**What you'll learn:**
- Building a `Walker` from a factory (`Walker.from_jansen()`, `from_klann()`, ...)
- Extracting the foot locus with `walker.step()` and `extract_trajectory()`
- Reading a foot curve as a walking-gait diagnostic
- Picking between four-bar, six-bar, and eight-bar topologies
"""

NB01_IMPORTS = """
import warnings

import matplotlib.pyplot as plt
import numpy as np

import leggedsnake as ls
from pylinkage import extract_trajectory
from pylinkage.visualizer import plot_static_linkage

warnings.filterwarnings('ignore', category=DeprecationWarning)
"""

NB01_HELPER_MD = """
## 1. A helper: draw a Walker with its foot locus

Every `Walker` wraps a cached `Mechanism`. pylinkage's
`plot_static_linkage(mechanism, ax, loci, ...)` renders the bars at
`t=0` and overlays each joint's locus over one full cycle. We then
bold-stroke the feet identified by `walker.get_feet()` so the stance
trajectory stands out against the auxiliary joint paths.
"""

NB01_HELPER = """
def show_walker(walker, title, iterations=None, figsize=(8, 5),
                skip_unbuildable=False, foot_labels=None):
    # Draw a Walker's bars at t=0 plus every joint locus; highlight the feet.
    # Pass ``skip_unbuildable=True`` for non-Grashof mechanisms so dead frames
    # are silently dropped. ``foot_labels`` overrides auto-detection when
    # get_feet() is ambiguous (e.g. all non-ground joints look like "feet").
    mech = walker.to_mechanism()
    raw = list(walker.step(iterations=iterations,
                           skip_unbuildable=skip_unbuildable))
    loci = [l for l in raw if l[0][0] is not None] if skip_unbuildable else raw

    fig, ax = plt.subplots(figsize=figsize)
    plot_static_linkage(
        mech, ax, loci,
        show_loci=True, show_labels=True, show_legend=False,
        title=title,
    )

    # Overlay the feet trajectories in bold. Some factories assign
    # cosmetic joint names ("left foot") that don't match the node IDs
    # returned by get_feet(); fall back to a substring match on "foot".
    if foot_labels is not None:
        feet_ids = set(foot_labels)
        def _is_foot(name):
            return name in feet_ids
    else:
        feet_ids = set(walker.get_feet())
        def _is_foot(name):
            return (
                name in feet_ids
                or any(name.startswith(fid + ' ') for fid in feet_ids)
                or 'foot' in name.lower()
            )
    for i, joint in enumerate(mech.joints):
        name = getattr(joint, 'name', '') or ''
        if _is_foot(name):
            xs, ys = extract_trajectory(loci, i)
            if xs.size:
                ax.plot(xs, ys, color='crimson', lw=2.2, alpha=0.9, zorder=10)

    ax.set_aspect('equal'); ax.grid(True, alpha=0.3)
    plt.show()
"""

NB01_JANSEN_MD = """
## 2. Theo Jansen's 8-bar (the holy number)

Jansen's linkage is an 8-bar mechanism tuned over 40 years of
experimentation. Its defining property is a **flat bottom** on the foot
curve — the foot glides horizontally during stance, keeping the body
level.
"""

NB01_JANSEN = """
jansen = ls.Walker.from_jansen()
print(f'DOF: {jansen.dof}, feet: {jansen.get_feet()}')

show_walker(jansen, "Theo Jansen — flat-bottom locus (holy number lengths)")
"""

NB01_KLANN_MD = """
## 3. Klann's 6-bar (US Patent 6,260,862)

Klann's linkage (6 links, one crank) approximates a scissor-gait pattern
with fewer parts than Jansen's. The foot lifts higher in swing — useful
for stepping over obstacles but worse for body stability.
"""

NB01_KLANN = """
klann = ls.Walker.from_klann()
show_walker(klann, "Klann — 6-bar, taller swing apex")
"""

NB01_CHEBYSHEV_MD = """
## 4. Chebyshev's lambda (4-bar, 1850s)

The simplest mechanism here: a **four-bar** with a coupler point. The
classical Chebyshev ratio `crank:coupler:rocker:ground = 1:2.5:2.5:2`,
with the foot at a coupler **extension** past the coupler-rocker joint
(`foot_ratio=2.0` places it at distance `2 × coupler` from the crank
pin A — i.e., as far past B as A-B). The foot hangs below the
mechanism in a Λ-shape, and its lower stroke is the approximate
straight line this linkage is named for. Four bars is the minimum for
a 1-DOF single-leg mechanism.
"""

NB01_CHEBYSHEV = """
cheb = ls.Walker.from_chebyshev(
    crank=1.0, coupler=2.5, rocker=2.5, ground_length=2.0, foot_ratio=2.0,
)
show_walker(cheb, "Chebyshev lambda — foot on coupler extension past B",
            figsize=(8, 6))
"""

NB01_STRIDER_MD = """
## 5. Strider (Vagle / DIY Walkers)

Strider is a symmetric 10-bar popularized by Wade Vagle — two mirrored
four-bar stacks driven from a shared crank. It's the mechanism that
`sandbox_legged.py` at the monorepo root optimizes. Its default
rotation period is small (10 frames), so we pass a higher iteration
count for a smooth curve.
"""

NB01_STRIDER = """
# Strider's factory angular velocity is coarse (10 steps per rotation);
# we pass a finer value so the foot curve is smooth in the plot.
from math import tau
strider = ls.Walker.from_strider(angular_velocity=-tau / 60)
show_walker(strider, "Strider — two mirrored four-bars, shared crank",
            figsize=(9, 6))
"""

NB01_GHASSAEI_MD = """
## 6. Ghassaei's 5-dyad (2011 Pomona thesis)

Amanda Ghassaei's leg is a five-RRR-dyad chain driven by a single
crank. `Walker.from_ghassaei()` uses the thesis's classical dimensions
(crank=26, ground=53, inner=56, outer=77, closing=75, H→E arm=130).
The H→E arm length is not given on Figure 5.4.4 and is tuned here to
reproduce the Wikibooks Walkin8r foot-locus aspect (x:y ≈ 1:0.24).
"""

NB01_GHASSAEI = """
ghassaei = ls.Walker.from_ghassaei()
show_walker(ghassaei, "Ghassaei (thesis / Boim Walkin8r) — 5-dyad",
            figsize=(9, 6))
"""

NB01_WATT_MD = """
## 7. Watt I — a six-bar walker from the literature

The two six-bar families (Watt and Stephenson) open richer foot-path
geometries than a pure four-bar: classical synthesis uses them for
coupler-curve targets a 4-bar cannot reach.

The dimensions below come from Mehdigholi's Watt-6-bar walking
mechanism (also featured in
[kenaycock/Six-Bar-Walking-Mechanism](https://github.com/kenaycock/Six-Bar-Walking-Mechanism)).
They produce a non-Grashof four-bar A-B-C-D (the crank cannot rotate
full 360°), so we pass ``skip_unbuildable=True`` and visualize only
the buildable stroke — in a real walker two cranks would drive the
mechanism through the dead zone.
"""

NB01_WATT = """
from math import pi

watt = ls.Walker.from_watt(
    crank=1.0,
    coupler1=4.6875,   # R3
    rocker1=2.3125,    # R4
    link4=4.6875,      # R6 (parallel leg link)
    link5=3.8125,      # R5 (top connector)
    rocker2=2.3125,    # R3'
    ground_length=1.875,  # R1
    initial_crank_angle=pi,
)
show_walker(watt, "Watt I — Mehdigholi walking dimensions",
            figsize=(8, 7), skip_unbuildable=True, foot_labels={'F'})
"""

NB01_STEPHENSON_MD = """
## 8. Stephenson I — synthesized from rigid-body poses

Where Watt's two rigid triangles are adjacent (joined at B), a
Stephenson has them **separated** — the second loop branches from
coupler joint C rather than crank-tip B.

Rather than handing it fixed dimensions, we run
`pylinkage.synthesis.motion_generation` with three target **poses**
(position + orientation) along a horizontal stroke, and Burmester
theory returns link lengths that thread those poses. The three poses
specify a body moving horizontally while rotating slightly — the
approximation of a walker's body during stance.

Pure collinear, same-orientation poses are singular under Burmester
synthesis (infinitely many solutions); the orientation sweep below
breaks that symmetry.
"""

NB01_STEPHENSON = """
import math
from pylinkage.synthesis import motion_generation, Pose, solution_to_linkage

# Three poses: horizontal motion with gentle orientation sweep
poses = [
    Pose(0.0, 0.0, 0.0),
    Pose(1.0, 0.5, math.pi / 10),
    Pose(2.0, 0.0, math.pi / 5),
]
syn = motion_generation(poses=poses, max_solutions=5, require_grashof=False)
print(f"motion_generation → {len(syn.solutions)} four-bar solution(s)")

# motion_generation returns four-bar solutions. To illustrate the
# Stephenson six-bar (and its separated ternary links), we use
# pylinkage's reference dimensions from its own notebook 14.
from pylinkage.synthesis import stephenson_from_lengths

stephenson = stephenson_from_lengths(
    crank=0.8, coupler=3.5, rocker=3.0,
    link4=2.0, link5=2.5, link6=3.0,
    ground_length=4.0, initial_crank_angle=math.pi / 4,
)
loci = list(stephenson.step(iterations=200))
fig, ax = plt.subplots(figsize=(9, 6))
plot_static_linkage(stephenson, ax, loci, show_loci=True, show_labels=True,
                    show_legend=False, n_ghosts=3,
                    title="Stephenson I — pylinkage reference dimensions")
ax.set_aspect('equal'); ax.grid(True, alpha=0.3)
plt.show()

# Show what motion_generation placed the coupler through
for i, p in enumerate(poses):
    print(f"  pose {i}: ({p.x:.1f}, {p.y:.1f}, {math.degrees(p.angle):.0f}°)")
"""

NB01_SUMMARY = """
## Summary

| Mechanism   | Links | Defining property                          |
|-------------|-------|--------------------------------------------|
| Chebyshev   | 4     | Simplest — straight-line approximation     |
| Klann       | 6     | Scissor-gait, high swing apex              |
| Watt I      | 6     | Inline coupler, double-dwell curves         |
| Stephenson  | 6     | Branched coupler, wider coupler-curve space |
| Jansen      | 8     | Flat-bottom stance (Jansen's "holy number") |
| Strider     | 10    | Mirrored four-bars, symmetric gait          |
| Ghassaei    | 12    | Five-dyad chain, broad horizontal stride    |

Pick a factory as a starting point, then tune link lengths with the
optimization pipeline (notebook 03) or swap topologies entirely with
`topology_walking_optimization` (see `discover_walker.ipynb`).
"""

NB01 = [
    _md(NB01_INTRO),
    _code(NB01_IMPORTS),
    _md(NB01_HELPER_MD), _code(NB01_HELPER),
    _md(NB01_JANSEN_MD), _code(NB01_JANSEN),
    _md(NB01_KLANN_MD), _code(NB01_KLANN),
    _md(NB01_CHEBYSHEV_MD), _code(NB01_CHEBYSHEV),
    _md(NB01_STRIDER_MD), _code(NB01_STRIDER),
    _md(NB01_GHASSAEI_MD), _code(NB01_GHASSAEI),
    _md(NB01_WATT_MD), _code(NB01_WATT),
    _md(NB01_STEPHENSON_MD), _code(NB01_STEPHENSON),
    _md(NB01_SUMMARY),
]


# ---------------------------------------------------------------------------
# 02 — Physics and fitness
# ---------------------------------------------------------------------------

NB02_INTRO = """
# Physics Simulation and Fitness Evaluation

Kinematic foot-loci (notebook 01) are free but they don't tell you
whether the walker **actually walks**. For that we drop the Walker
into a pymunk physics `World`, let it step for a few simulated
seconds, and score how far it got.

**What you'll learn:**
- Configuring gravity, friction, and terrain with `WorldConfig` / `TerrainConfig`
- Running a simulated walk with `World` + `Walker`
- Three built-in fitness evaluators (`DistanceFitness`,
  `EfficiencyFitness`, `StrideFitness`) and when each one applies
- Generating obstacle and slope terrains from `TerrainPreset`
"""

NB02_IMPORTS = """
import warnings

import matplotlib.pyplot as plt

import leggedsnake as ls

warnings.filterwarnings('ignore', category=DeprecationWarning)
"""

NB02_CONFIG_MD = """
## 1. The `WorldConfig` dataclass

Before `0.5.0`, physics parameters lived in a module-level `params`
dict. They now live on a `WorldConfig` dataclass you pass into `World()`
(the old `params` dict is still populated for legacy scripts). Defaults
match the historical dict: Earth gravity, 1000 N·m motor torque,
`physics_period=0.02`.
"""

NB02_CONFIG = """
cfg = ls.WorldConfig()
print(f"gravity = {cfg.gravity} m/s²")
print(f"physics period = {cfg.physics_period} s")
print(f"torque = {cfg.torque} N·m, load = {cfg.load_mass} kg")
print(f"terrain: slope={cfg.terrain.slope:.3f} rad, noise={cfg.terrain.noise}")
"""

NB02_SIM_MD = """
## 2. Simulate a Jansen walker for a few seconds

`World(config=...).add_linkage(walker)` converts the kinematic Walker
to its pymunk-backed twin, drops it on the road, and stands ready to
step. We bump the Jansen walker to four legs so it has enough ground
contact for a stable gait.
"""

NB02_SIM = """
def simulate(walker, duration_s=3.0, config=None):
    # Step the walker in a fresh World and return the final body x.
    cfg = config or ls.WorldConfig()
    world = ls.World(config=cfg)
    world.add_linkage(walker)
    n_steps = int(duration_s / cfg.physics_period)
    for _ in range(n_steps):
        world.update()
    return world.linkages[0].body.position.x

jansen = ls.Walker.from_jansen(scale=0.5)
jansen.add_legs(3)  # four legs total
x = simulate(jansen, duration_s=3.0)
print(f"Jansen travelled {x:.2f} m in 3 simulated seconds")
"""

NB02_FITNESS_MD = """
## 3. The `DynamicFitness` protocol

Optimizers need a *single number*. `DynamicFitness` is a Protocol:
any callable with signature `(topology, dimensions, config) → FitnessResult`
qualifies. Leggedsnake ships three implementations:

| Fitness            | Physics? | Maximizes                          |
|--------------------|----------|-------------------------------------|
| `StrideFitness`    | No       | Kinematic horizontal stride         |
| `DistanceFitness`  | Yes      | Total distance walked in T seconds  |
| `EfficiencyFitness`| Yes      | Distance / energy consumed          |

`CompositeFitness` evaluates distance + efficiency + stability in a
**single physics run**, so NSGA-II with three objectives costs one
simulation per candidate instead of three.
"""

NB02_FITNESS = """
fit_distance = ls.DistanceFitness(duration=3.0, n_legs=4)
fit_efficiency = ls.EfficiencyFitness(duration=3.0, n_legs=4, min_distance=0.5)
fit_stride = ls.StrideFitness()

walker = ls.Walker.from_jansen(scale=0.5)
for fit in (fit_distance, fit_efficiency, fit_stride):
    r = fit(walker.topology, walker.dimensions, ls.WorldConfig())
    print(f"{type(fit).__name__:20s} score={r.score:8.3f}  valid={r.valid}  metrics={dict(r.metrics)}")
"""

NB02_TERRAIN_MD = """
## 4. Terrain presets

`TerrainConfig` exposes a library of rough-ground generators. The
`TerrainPreset` enum gives five ready-made configurations:
"""

NB02_TERRAIN = """
for preset in ls.TerrainPreset:
    t = ls.TerrainConfig.from_preset(preset)
    print(f"{preset.name:10s} slope={t.slope:+.3f}rad  noise={t.noise:.2f}  "
          f"gap={t.gap_freq:.2f}  obstacle={t.obstacle_freq:.2f}  "
          f"profile={t.slope_profile}")
"""

NB02_ROUGH_MD = """
## 5. Walk on rough vs flat ground

Ground roughness costs distance. We simulate the same Jansen walker
on three presets and compare.
"""

NB02_ROUGH = """
results = {}
for preset in (ls.TerrainPreset.FLAT, ls.TerrainPreset.HILLY, ls.TerrainPreset.ROUGH):
    cfg = ls.WorldConfig(terrain=ls.TerrainConfig.from_preset(preset))
    walker = ls.Walker.from_jansen(scale=0.5)
    walker.add_legs(3)
    results[preset.name] = simulate(walker, duration_s=3.0, config=cfg)

fig, ax = plt.subplots(figsize=(6, 3.5))
ax.bar(results.keys(), results.values(), color=['#4c72b0', '#dd8452', '#c44e52'])
ax.set_ylabel("Distance walked (m, 3 s)")
ax.set_title("Terrain preset vs walking distance — Jansen, fixed dimensions")
ax.grid(True, axis='y')
plt.show()
"""

NB02_SUMMARY = """
## Summary

- **`WorldConfig` + `TerrainConfig`** are the structured entry points
  — pass one into `World(config=...)` instead of mutating the global
  `params` dict.
- **Fitness evaluators** implement `DynamicFitness`. Prefer
  `StrideFitness` (kinematic, cheap) inside inner loops and
  `DistanceFitness` / `CompositeFitness` (physics, expensive) for the
  decisive scoring pass.
- **Terrain presets** keep benchmarks reproducible (they carry a
  `seed`), which matters when comparing two candidates or re-scoring
  a checkpoint.

Next (notebook 03): run a GA over the Strider's link lengths to find
a better walker.
"""

NB02 = [
    _md(NB02_INTRO),
    _code(NB02_IMPORTS),
    _md(NB02_CONFIG_MD), _code(NB02_CONFIG),
    _md(NB02_SIM_MD), _code(NB02_SIM),
    _md(NB02_FITNESS_MD), _code(NB02_FITNESS),
    _md(NB02_TERRAIN_MD), _code(NB02_TERRAIN),
    _md(NB02_ROUGH_MD), _code(NB02_ROUGH),
    _md(NB02_SUMMARY),
]


# ---------------------------------------------------------------------------
# 03 — Genetic optimization
# ---------------------------------------------------------------------------

NB03_INTRO = """
# Optimizing a Walker with Genetic Algorithms

The Strider is parameterized by seven link lengths. We score candidates
on a cheap **kinematic** fitness (`StrideFitness`) with a small GA,
then look at before/after foot loci.

**What you'll learn:**
- Parameterizing a factory walker for optimization
- Running `genetic_algorithm_optimization` on a kinematic objective
- Reading an `Ensemble` result (`top`, `filter_by_score`)
- When to chain a GA with `chain_walking_optimizers` for local polish
"""

NB03_IMPORTS = """
import warnings

import matplotlib.pyplot as plt

import leggedsnake as ls
from pylinkage import extract_trajectory
from pylinkage.optimization import generate_bounds

warnings.filterwarnings('ignore', category=DeprecationWarning)
"""

NB03_START_MD = """
## 1. The starting Strider

`Walker.from_strider(...)` takes seven lengths. Most of pylinkage's
optimizers talk to a linkage via its `.joints` list (dimensions per
joint). Walker exposes `.joints` for exactly this reason.
"""

NB03_START = """
START = dict(crank=1.0, triangle=2.0, femur=1.8, rocker_l=2.6,
             rocker_s=1.4, tibia=2.5, foot=1.8)

def make_strider():
    return ls.Walker.from_strider(**START)

prototype = make_strider()
print(f"Strider start: DOF={prototype.dof}, feet={prototype.get_feet()}")
edge_names = list(prototype.dimensions.edge_distances.keys())
dim_values = list(prototype.dimensions.edge_distances.values())
print(f"{len(dim_values)} free edge distances to optimize")
"""

NB03_FIT_MD = """
## 2. A kinematic fitness on the Strider foot locus

`StrideFitness` returns the horizontal stride length (the portion of
the foot locus that stays below a given height — i.e., the stance
phase). We adapt it to the GA's `(eval_func(dna))` contract with
`as_ga_fitness`.
"""

NB03_FIT = """
fit = ls.StrideFitness(step_height=0.5, stride_height=0.2, foot_index=-1)
r = fit(prototype.topology, prototype.dimensions, ls.WorldConfig())
print(f"Starting stride score: {r.score:.3f}  (valid={r.valid})")

# Adapt to the optimizer contract: (linkage, dims, pos) -> float
eval_func = ls.as_eval_func(fit, walker_factory=make_strider)
"""

NB03_GA_MD = """
## 3. Search — `genetic_algorithm_optimization`

Small population and few iterations keep the demo fast. On a real
problem use `max_pop≈40, iters≈80+` and multiprocessing via
`processes=...`.
"""

NB03_GA = """
bounds = generate_bounds(dim_values, min_ratio=2.0, max_factor=2.0)

ensemble = ls.genetic_algorithm_optimization(
    eval_func=eval_func,
    linkage=prototype,
    center=dim_values,
    bounds=bounds,
    max_pop=10,
    iters=4,
    processes=1,
    verbose=False,
)

print(f"Ensemble size: {len(ensemble)}")
for i, agent in enumerate(ensemble.top(3)):
    print(f"  rank {i}: score={agent.score:.3f}")
"""

NB03_VIZ_MD = """
## 4. Visualize before vs after

Plot the starting and best-after-GA foot loci side-by-side.
"""

NB03_VIZ = """
def foot_xy(walker):
    mech = walker.to_mechanism()
    loci = list(walker.step(iterations=120))
    xs, ys = extract_trajectory(loci, len(mech.joints) - 1)
    return xs, ys

best_agent = ensemble[0]
# Build a fresh Strider, then swap in the optimized edge distances.
best_walker = make_strider()
best_walker.set_num_constraints(list(best_agent.dimensions))

fig, ax = plt.subplots(figsize=(7, 4))
xs, ys = foot_xy(prototype)
ax.plot(xs, ys, '--', color='#888', label=f"start (stride={r.score:.2f})")
xs, ys = foot_xy(best_walker)
ax.plot(xs, ys, '-', color='#c44e52', label=f"best GA (stride={best_agent.score:.2f})")
ax.set_aspect('equal'); ax.grid(True); ax.legend()
ax.set_title("Strider foot locus — GA before/after")
plt.show()
"""

NB03_ENSEMBLE_MD = """
## 5. Ensemble inspection

`Ensemble` wraps the Pareto-ordered population with helpers: `top(n)`,
`filter_by_score(min_score=...)`. Handy for checkpointing or reseeding
a follow-up run.
"""

NB03_ENSEMBLE = """
survivors = ensemble.filter_by_score(min_val=r.score)
print(f"{len(survivors)} / {len(ensemble)} candidates beat the starting score")
"""

NB03_SUMMARY = """
## Summary

- `genetic_algorithm_optimization` takes a *Walker* via `linkage=`
  and a `([lo...], [hi...])` bounds tuple, and returns a pylinkage
  `Ensemble`.
- `StrideFitness` + `as_ga_fitness` is the cheap kinematic pairing —
  budget this one in the inner loop, physics fitness in the outer.
- **Chaining**: `chain_walking_optimizers(fitness, walker, stages=[...])`
  feeds each stage the previous best — the go-to pattern for *global
  search → local polish* (differential evolution → Nelder-Mead).

Next (notebook 04): multi-objective NSGA-II over distance and
efficiency, plus gait analysis on the Pareto winners.
"""

NB03 = [
    _md(NB03_INTRO),
    _code(NB03_IMPORTS),
    _md(NB03_START_MD), _code(NB03_START),
    _md(NB03_FIT_MD), _code(NB03_FIT),
    _md(NB03_GA_MD), _code(NB03_GA),
    _md(NB03_VIZ_MD), _code(NB03_VIZ),
    _md(NB03_ENSEMBLE_MD), _code(NB03_ENSEMBLE),
    _md(NB03_SUMMARY),
]


# ---------------------------------------------------------------------------
# 04 — Multi-objective optimization and gait analysis
# ---------------------------------------------------------------------------

NB04_INTRO = """
# Multi-Objective Optimization and Gait Analysis

Real walking design is a trade-off: a walker that goes far usually
wastes energy doing it. Single-objective optimization collapses that
trade-off into one number; **NSGA-II** keeps both on the table and
returns a Pareto front of non-dominated designs.

**What you'll learn:**
- Running `nsga_walking_optimization` with two physics objectives
- Reading a Pareto front and picking the best compromise
- `analyze_gait()`: duty factor, stride period, phase offsets
- Visualization helpers (`plot_pareto_front`, `plot_gait_diagram`)
"""

NB04_IMPORTS = """
import warnings

import matplotlib.pyplot as plt
import numpy as np

import leggedsnake as ls

warnings.filterwarnings('ignore', category=DeprecationWarning)
np.random.seed(42)
"""

NB04_NSGA_MD = """
## 1. NSGA-II over distance and efficiency

Both objectives are physics-based (`DistanceFitness`,
`EfficiencyFitness`), so each candidate is a short pymunk rollout.
We keep `duration=2s`, `population_size=6`, `generations=3` to bound
the demo runtime — use order-of-magnitude larger settings for a
serious run.
"""

NB04_NSGA = """
def walker_factory():
    return ls.Walker.from_jansen(scale=0.5)

prototype = walker_factory()
edge_names = list(prototype.dimensions.edge_distances.keys())
dim_values = list(prototype.dimensions.edge_distances.values())
lo = [0.7 * v for v in dim_values]
hi = [1.3 * v for v in dim_values]

cfg = ls.NsgaWalkingConfig(n_generations=3, pop_size=6, seed=42, verbose=False)

result = ls.nsga_walking_optimization(
    walker_factory=walker_factory,
    objectives=[
        ls.DistanceFitness(duration=2.0, n_legs=4),
        ls.EfficiencyFitness(duration=2.0, n_legs=4, min_distance=0.1),
    ],
    bounds=(lo, hi),
    objective_names=['distance', 'efficiency'],
    nsga_config=cfg,
    include_gait=False,
    include_stability=False,
)
print(f"Pareto front: {len(result.pareto_front.solutions)} solutions")
for i, sol in enumerate(result.pareto_front.solutions[:5]):
    print(f"  sol {i}: scores={[round(float(s), 3) for s in sol.scores]}")
"""

NB04_PLOT_MD = """
## 2. Plot the Pareto front

`plot_pareto_front` is a thin matplotlib wrapper that highlights the
"best compromise" (closest to the utopia corner in objective space).
"""

NB04_PLOT = """
fig = ls.plot_pareto_front(result, figsize=(6, 4))
plt.show()
"""

NB04_BEST_MD = """
## 3. Inspect the best-compromise solution

`result.best_compromise` is the Pareto solution with smallest L2
distance to the utopia corner in objective space.
"""

NB04_BEST = """
best = result.best_compromise()
print(f"best compromise scores: {best.scores}")
print(f"  dimensions: {dict(zip(edge_names, best.dimensions))}")
"""

NB04_GAIT_MD = """
## 4. Gait analysis on the winner

`analyze_gait` takes the per-foot trajectories and returns **duty
factor** (fraction of the stride in stance), **stride period**, and
inter-foot **phase offsets**. Re-evaluating the winner with
`DistanceFitness(record_loci=True)` captures the trajectories we need.
"""

NB04_GAIT = """
best_walker = walker_factory()
best_walker.set_num_constraints(list(best.dimensions))

fit_record = ls.DistanceFitness(duration=3.0, n_legs=4, record_loci=True)
fr = fit_record(best_walker.topology, best_walker.dimensions, ls.WorldConfig())
print(f"recorded distance={fr.score:.3f}, feet tracked={list(fr.loci.keys())}")

if fr.loci:
    foot_ids = list(fr.loci.keys())
    gait = ls.analyze_gait(fr.loci, foot_ids, dt=0.02)
    print(f"mean duty factor : {gait.mean_duty_factor:.2f}")
    print(f"mean stride freq : {gait.mean_stride_frequency:.2f} Hz")
    print(f"mean stride len  : {gait.mean_stride_length:.3f} m")
"""

NB04_DIAG_MD = """
## 5. Gait diagram

`plot_gait_diagram` shows each foot as a stance/swing bar across
time. Symmetric gaits (trot, pace) produce simple alternating
patterns; chaotic gaits fragment.
"""

NB04_DIAG = """
if fr.loci:
    fig = ls.plot_gait_diagram(gait, figsize=(8, 2.8))
    plt.show()
else:
    print("No recorded loci — skipping gait diagram")
"""

NB04_SUMMARY = """
## Summary

| Step                | API                                               |
|---------------------|---------------------------------------------------|
| Multi-obj search    | `nsga_walking_optimization(factory, objectives, bounds)` |
| Pareto plot         | `plot_pareto_front(result)`                       |
| Best compromise     | `result.best_compromise`                          |
| Record loci         | `DistanceFitness(record_loci=True)`               |
| Gait analysis       | `analyze_gait(loci, foot_ids)`                    |
| Stability           | `compute_stability_snapshot` / `StabilityTimeSeries` |

See `discover_walker.ipynb` for the next step — **topology
co-optimization**, where the search treats the mechanism's graph
structure itself as a decision variable (four-bar vs six-bar vs
eight-bar, determined by fitness rather than assumed).
"""

NB04 = [
    _md(NB04_INTRO),
    _code(NB04_IMPORTS),
    _md(NB04_NSGA_MD), _code(NB04_NSGA),
    _md(NB04_PLOT_MD), _code(NB04_PLOT),
    _md(NB04_BEST_MD), _code(NB04_BEST),
    _md(NB04_GAIT_MD), _code(NB04_GAIT),
    _md(NB04_DIAG_MD), _code(NB04_DIAG),
    _md(NB04_SUMMARY),
]


def main() -> None:
    root = Path(__file__).parent
    _build(NB01, root / "01_walkers_gallery.ipynb")
    _build(NB02, root / "02_physics_and_fitness.ipynb")
    _build(NB03, root / "03_genetic_optimization.ipynb")
    _build(NB04, root / "04_multi_objective_and_gait.ipynb")


if __name__ == "__main__":
    main()
