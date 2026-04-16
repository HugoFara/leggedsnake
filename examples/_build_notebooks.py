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
## 7. Watt I — six-bar walker (full rotation, flat stance)

Watt and Stephenson six-bars (the only two 1-DOF six-link topologies,
per Grübler) open richer foot-path geometries than a pure four-bar:
classical synthesis uses them for coupler-curve targets a 4-bar
cannot reach.

For a practical walker we want two things from the foot trajectory:
1. **Full crank rotation** — the driving four-bar A-B-C-D must be
   Grashof crank-rocker, i.e., crank is the shortest link and
   `crank + ground ≤ coupler1 + rocker1`.
2. **A near-horizontal stretch of the foot locus** — the *stance*
   phase where the foot glides along the ground.

For Watt I the foot rides on the **ternary triangle B-C-E** — E is a
natural coupler-curve tracing point, rigidly attached to the
floating link BC. Sampling the parameter space for a Grashof crank-
rocker that also puts a near-flat segment in E's locus gives:
"""

NB01_WATT = """
watt = ls.Walker.from_watt(
    crank=1.49,
    coupler1=3.12,
    rocker1=4.33,
    link4=2.02,
    link5=3.79,
    rocker2=5.43,
    ground_length=4.24,
)
show_walker(watt, "Watt I — full rotation, flat stance on E",
            figsize=(9, 7), foot_labels={'E'})
"""

NB01_STEPHENSON_MD = """
## 8. Stephenson I — full rotation, separated ternary links

Where Watt's two rigid triangles are adjacent (joined at B), a
Stephenson has them **separated** — the second loop branches from
coupler joint C and ground D rather than from crank-tip B. Same
number of links, different graph, different coupler-curve family.

Same recipe as the Watt: we keep the driving four-bar A-B-C-D
Grashof crank-rocker and pick the second-loop lengths so F's locus
has a flat upper portion. `Walker.from_stephenson` routes to
pylinkage's `stephenson_from_lengths` under the hood and through the
SimLinkage → Walker shim.
"""

NB01_STEPHENSON = """
stephenson = ls.Walker.from_stephenson(
    crank=1.03,
    coupler=4.46,
    rocker=2.38,
    link4=3.01,
    link5=2.54,
    link6=3.67,
    ground_length=4.3,
)
show_walker(stephenson, "Stephenson I — full rotation, flat stance on F",
            figsize=(9, 7), foot_labels={'F'})
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
from matplotlib import animation
from IPython.display import HTML

import leggedsnake as ls
from pylinkage import extract_trajectory
from pylinkage.visualizer import plot_static_linkage

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

NB02_KINVIZ_MD = """
## 2. An eight-legged Jansen — kinematic preview

Before dropping anything on a road, let's see what we're about to
simulate. The canonical Strandbeest has **four legs per side**
mirrored across the chassis — we reproduce that arrangement with
`add_opposite_leg()` (left/right mirror) followed by `add_legs(3)`
(three phase-offset copies per side), matching the pattern in
`examples/theo_jansen.py`. Eight legs means at least two feet in
stance at any crank angle, which is what keeps the body level during
physics simulation.
"""

NB02_KINVIZ = """
jansen = ls.Walker.from_jansen(scale=0.1)
jansen.add_opposite_leg()  # mirrored copy of the leg on the opposite side
jansen.add_legs(3)         # 3 phase-offset copies per side → 8 legs total

# Jansen's foot is node 'G'. Cloned legs inherit the base name plus a
# suffix like '(opposite)' or '(2)', so 'G' + whitespace is the reliable
# filter. (walker.get_feet() also returns secondary degree-1 joints in
# hypergraph form; we want only the true feet.)
def is_foot(name):
    return name == 'G' or name.startswith('G ')

feet_count = sum(1 for nid in jansen.topology.nodes if is_foot(nid))
print(f"legs: {feet_count}, DOF: {jansen.dof}")

mech = jansen.to_mechanism()
loci = list(jansen.step(iterations=180))

fig, ax = plt.subplots(figsize=(9, 4.8))
plot_static_linkage(
    mech, ax, loci,
    show_loci=True, show_labels=False, show_legend=False,
    title="Jansen — 8 legs (4 per side, phase-offset), kinematic preview",
)
for i, joint in enumerate(mech.joints):
    if is_foot(getattr(joint, 'name', '') or ''):
        xs, ys = extract_trajectory(loci, i)
        if xs.size:
            ax.plot(xs, ys, color='crimson', lw=1.8, alpha=0.9, zorder=10)
ax.set_aspect('equal'); ax.grid(True, alpha=0.3)
plt.show()
"""

NB02_SIM_MD = """
## 3. Simulate the same walker in physics

`World(config=...).add_linkage(walker)` converts the kinematic Walker
to its pymunk-backed twin, drops it on the road, and stands ready to
step. Two gotchas worth addressing up front:

1. `World` defaults to `road_y=-5`. That's fine for small walkers but
   mis-matched for Jansen's large Holy-Number geometry. `linkage_bb(walker)`
   gives us the pre-physics bounding box and we slot the road just
   beneath `min_y` — exactly what `ls.video()` does.
2. The default `TerrainConfig` has a 10° slope, heavy noise, and an
   unseeded RNG, so every run is a different hilly uphill. For a
   reproducible demo we want flat ground — `TerrainPreset.FLAT` with a
   fixed `seed`.
"""

NB02_SIM = """
FLAT = ls.TerrainConfig.from_preset(ls.TerrainPreset.FLAT)
FLAT.seed = 0  # any int works; fixes road-building RNG

def simulate(walker, duration_s=3.0, config=None):
    # Step the walker in a fresh World and return the final body x.
    cfg = config or ls.WorldConfig(terrain=FLAT)
    min_y = ls.linkage_bb(walker)[0]
    world = ls.World(config=cfg, road_y=min_y - 1)
    world.add_linkage(walker)
    n_steps = int(duration_s / cfg.physics_period)
    for _ in range(n_steps):
        world.update()
    return world.linkages[0].body.position.x

x = simulate(jansen, duration_s=5.0)
print(f"Jansen travelled {x:.2f} units in 5 simulated seconds")
"""

NB02_ANIM_MD = """
## 4. Animate the dynamic walker

The distance number is terse — let's watch the walker move. We re-run
the simulation, sample world positions every few physics ticks, and
render an inline HTML animation. Each bar is drawn between the two
world-space node positions on its rigid body; the polyline along the
bottom is the procedurally built road.

The animation uses the full **8-leg** Strandbeest from §2 (four legs
per side at 0 / 90° / 180° / 270° phase offsets). Dense stance coverage
keeps the chassis level — in our tests the 8-leg walker stays within a
few degrees of horizontal over 5 s, while a 4-leg version pitches by
10° – 20°. That's the 2D analogue of why real Strandbeests have many
legs: the more legs overlapping in stance, the less the body reacts to
any single crank's swing.
"""

NB02_ANIM = """
anim_walker = ls.Walker.from_jansen(scale=0.1)
anim_walker.add_opposite_leg()
anim_walker.add_legs(3)  # 4 legs per side → canonical Strandbeest

bb_min_y, bb_max_x, bb_max_y, bb_min_x = ls.linkage_bb(anim_walker)
road_y = bb_min_y - 1
walker_h = bb_max_y - bb_min_y
walker_w = bb_max_x - bb_min_x

cfg = ls.WorldConfig(terrain=FLAT)
world = ls.World(config=cfg, road_y=road_y)
world.add_linkage(anim_walker)
dl = world.linkages[0]

edges = list(anim_walker.topology.edges.items())

duration_s = 10.0
stride = 10  # record every 10th physics tick (~200 ms per frame, 50 frames)
n_steps = int(duration_s / cfg.physics_period)

frames = []
for i in range(n_steps):
    world.update()
    if i % stride:
        continue
    pos = dl.get_all_positions()
    segs = [
        (pos[e.source], pos[e.target])
        for _, e in edges
        if e.source in pos and e.target in pos
    ]
    frames.append({
        'segments': segs,
        'road': list(world.road),
        'cx': dl.body.position.x,
    })

# Camera window sized to fit a full walker + a margin, centred on chassis x.
half_w = walker_w * 0.75
y_lo = road_y - 2
y_hi = road_y + walker_h + 8

fig, ax = plt.subplots(figsize=(7, 7 * (y_hi - y_lo) / (2 * half_w)))
plt.close(fig)  # suppress the static snapshot; we want the animation only

def draw(frame):
    ax.clear()
    for a, b in frame['segments']:
        ax.plot([a[0], b[0]], [a[1], b[1]], '-', color='#222', lw=1.0, alpha=0.75)
    xs, ys = zip(*frame['road'])
    ax.plot(xs, ys, color='#555', lw=1.2)
    ax.set_xlim(frame['cx'] - half_w, frame['cx'] + half_w)
    ax.set_ylim(y_lo, y_hi)
    ax.set_aspect('equal'); ax.grid(True, alpha=0.3)
    ax.set_title(f"Jansen — physics (x={frame['cx']:.2f})")

ani = animation.FuncAnimation(
    fig, draw, frames=frames, interval=200, blit=False,
)
HTML(ani.to_jshtml())
"""

NB02_FITNESS_MD = """
## 5. The `DynamicFitness` protocol

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

walker = ls.Walker.from_jansen(scale=0.1)
for fit in (fit_distance, fit_efficiency, fit_stride):
    r = fit(walker.topology, walker.dimensions, ls.WorldConfig())
    print(f"{type(fit).__name__:20s} score={r.score:8.3f}  valid={r.valid}  metrics={dict(r.metrics)}")
"""

NB02_TERRAIN_MD = """
## 6. Terrain presets

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
## 7. Walk on rough vs flat ground

Ground roughness costs distance. We simulate the same Jansen walker
on three presets and compare.
"""

NB02_ROUGH = """
results = {}
for preset in (ls.TerrainPreset.FLAT, ls.TerrainPreset.HILLY, ls.TerrainPreset.ROUGH):
    cfg = ls.WorldConfig(terrain=ls.TerrainConfig.from_preset(preset))
    walker = ls.Walker.from_jansen(scale=0.1)
    walker.add_opposite_leg()
    walker.add_legs(3)
    results[preset.name] = simulate(walker, duration_s=3.0, config=cfg)

fig, ax = plt.subplots(figsize=(6, 3.5))
ax.bar(results.keys(), results.values(), color=['#4c72b0', '#dd8452', '#c44e52'])
ax.set_ylabel("Distance walked (units, 3 s)")
ax.set_title("Terrain preset vs walking distance — Jansen, fixed dimensions")
ax.grid(True, axis='y')
plt.show()
"""

NB02_ROUGH_ANIM_MD = """
## 8. Animate the walker on rough terrain

Numbers in a bar chart hide the drama. Let's watch the same 8-leg
Jansen take on the ROUGH preset — randomized segment friction,
scattered obstacles, 5° slope — and see which legs carry it, where
it stumbles, and where the gait recovers. Rough ground tests not
just raw speed but the *robustness margin* of a design; a walker
that sails over flat ground can still stall in the first obstacle.
"""

NB02_ROUGH_ANIM = """
rough_walker = ls.Walker.from_jansen(scale=0.1)
rough_walker.add_opposite_leg()
rough_walker.add_legs(3)

rough_terrain = ls.TerrainConfig.from_preset(ls.TerrainPreset.ROUGH)
rough_terrain.seed = 1
rough_cfg = ls.WorldConfig(terrain=rough_terrain)

bb_min_y, bb_max_x, bb_max_y, bb_min_x = ls.linkage_bb(rough_walker)
r_road_y = bb_min_y - 1
r_walker_h = bb_max_y - bb_min_y
r_walker_w = bb_max_x - bb_min_x

r_world = ls.World(config=rough_cfg, road_y=r_road_y)
r_world.add_linkage(rough_walker)
r_dl = r_world.linkages[0]

r_edges = list(rough_walker.topology.edges.items())

r_duration = 10.0
r_stride = 10
r_n_steps = int(r_duration / rough_cfg.physics_period)

r_frames = []
for i in range(r_n_steps):
    r_world.update()
    if i % r_stride:
        continue
    pos = r_dl.get_all_positions()
    r_frames.append({
        'segments': [
            (pos[e.source], pos[e.target])
            for _, e in r_edges
            if e.source in pos and e.target in pos
        ],
        'road': list(r_world.road),
        'cx': r_dl.body.position.x,
    })

r_half_w = r_walker_w * 0.75
r_y_lo = r_road_y - 3
r_y_hi = r_road_y + r_walker_h + 8

r_fig, r_ax = plt.subplots(figsize=(7, 7 * (r_y_hi - r_y_lo) / (2 * r_half_w)))
plt.close(r_fig)

def draw_rough(frame):
    r_ax.clear()
    for a, b in frame['segments']:
        r_ax.plot([a[0], b[0]], [a[1], b[1]], '-', color='#222', lw=1.0, alpha=0.75)
    xs, ys = zip(*frame['road'])
    r_ax.plot(xs, ys, color='#8b5a2b', lw=1.4)
    r_ax.fill_between(xs, ys, r_y_lo, color='#d9c7a5', alpha=0.5)
    r_ax.set_xlim(frame['cx'] - r_half_w, frame['cx'] + r_half_w)
    r_ax.set_ylim(r_y_lo, r_y_hi)
    r_ax.set_aspect('equal'); r_ax.grid(True, alpha=0.3)
    r_ax.set_title(f"Jansen on ROUGH terrain (x={frame['cx']:.2f})")

r_ani = animation.FuncAnimation(
    r_fig, draw_rough, frames=r_frames, interval=100, blit=False,
)
HTML(r_ani.to_jshtml())
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
    _md(NB02_KINVIZ_MD), _code(NB02_KINVIZ),
    _md(NB02_SIM_MD), _code(NB02_SIM),
    _md(NB02_ANIM_MD), _code(NB02_ANIM),
    _md(NB02_FITNESS_MD), _code(NB02_FITNESS),
    _md(NB02_TERRAIN_MD), _code(NB02_TERRAIN),
    _md(NB02_ROUGH_MD), _code(NB02_ROUGH),
    _md(NB02_ROUGH_ANIM_MD), _code(NB02_ROUGH_ANIM),
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
from pylinkage.visualizer import plot_static_linkage

warnings.filterwarnings('ignore', category=DeprecationWarning)
"""

NB03_START_MD = """
## 1. The starting Strider

The Strider's 17 edge distances are coupled by symmetry — only seven
independent lengths really parameterize it (`crank, triangle, femur,
rocker_l, rocker_s, tibia, foot`). We optimize in that compact
7-dimensional space: the GA proposes seven numbers, we rebuild a fresh
Strider from them, and we score that walker. Mutations that break
symmetry are avoided by construction.

The factory's default crank rate traces just 10 samples per rotation —
enough for `StrideFitness`, too coarse for a smooth drawing. The
`show_strider` helper oversamples by shrinking `dt`: one revolution
rendered in 60 sub-steps without touching the stored angular velocity.
"""

NB03_START = """
PARAM_NAMES = ['crank', 'triangle', 'femur', 'rocker_l', 'rocker_s', 'tibia', 'foot']
START_PARAMS = [1.0, 2.0, 1.8, 2.6, 1.4, 2.5, 1.8]

def make_strider(params=START_PARAMS):
    return ls.Walker.from_strider(**dict(zip(PARAM_NAMES, params)))

def show_strider(walker, ax, title, n_frames=60):
    # Draw bars at t=0, every joint locus, and overlay the two feet in bold.
    mech = walker.to_mechanism()
    period = mech.get_rotation_period()  # integer steps per revolution at dt=1
    dt = period / n_frames
    loci = list(walker.step(iterations=n_frames, dt=dt, skip_unbuildable=True))
    plot_static_linkage(
        mech, ax, loci,
        show_loci=True, show_labels=False, show_legend=False,
        title=title,
    )
    for i, joint in enumerate(mech.joints):
        name = (getattr(joint, 'name', '') or '').lower()
        if 'foot' in name:
            xs, ys = extract_trajectory(loci, i)
            if xs.size:
                ax.plot(xs, ys, color='crimson', lw=2.2, alpha=0.9, zorder=10)
    ax.set_aspect('equal'); ax.grid(True, alpha=0.3)

prototype = make_strider()
print(f"Strider start: DOF={prototype.dof}, feet={prototype.get_feet()}")
print(f"{len(PARAM_NAMES)} independent params to optimize: {PARAM_NAMES}")

fig, ax = plt.subplots(figsize=(7, 5))
show_strider(prototype, ax, "Starting Strider — bars at t=0 + joint loci")
plt.show()
"""

NB03_FIT_MD = """
## 2. A kinematic fitness on the Strider foot locus

`StrideFitness` returns the horizontal stride length (the portion of
the foot locus that stays below a given height — i.e., the stance
phase). We wrap it in an `eval_func(linkage, dims, pos) -> float` that
**rebuilds a fresh Strider from the 7 compact params** each evaluation
— bypassing `set_num_constraints`, which can only partially propagate
changes through the hypergraph's rigid-triangle constraints.
"""

NB03_FIT = """
fit = ls.StrideFitness(step_height=0.5, stride_height=0.2, foot_index=-1)
r = fit(prototype.topology, prototype.dimensions, ls.WorldConfig())
print(f"Starting stride score: {r.score:.3f}  (valid={r.valid})")

def eval_func(linkage, dims, pos):
    # ``dims`` is the GA's current 7-tuple; ``linkage`` / ``pos`` are ignored.
    # Rebuild from scratch so the factory can recompute consistent node
    # positions for every candidate.
    try:
        w = make_strider(dims)
        result = fit(w.topology, w.dimensions, ls.WorldConfig())
        return result.score if result.valid else 0.0
    except Exception:
        # Infeasible geometries (e.g. non-intersecting RRR dyads) get score 0.
        return 0.0
"""

NB03_GA_MD = """
## 3. Search — `genetic_algorithm_optimization`

Small population and few iterations keep the demo fast. On a real
problem use `max_pop≈40, iters≈80+` and multiprocessing via
`processes=...`. Bounds are ±40% of each default — tight enough that
most mutations land in the RRR buildability window.
"""

NB03_GA = """
bounds = ([0.6 * v for v in START_PARAMS], [1.4 * v for v in START_PARAMS])

ensemble = ls.genetic_algorithm_optimization(
    eval_func=eval_func,
    linkage=prototype,
    center=START_PARAMS,
    bounds=bounds,
    max_pop=12,
    iters=15,
    processes=1,
    verbose=False,
)

print(f"Ensemble size: {len(ensemble)}")
for i in range(min(3, len(ensemble))):
    print(f"  rank {i}: score={ensemble[i].score:.3f}  "
          f"params={dict(zip(PARAM_NAMES, [round(float(d), 2) for d in ensemble[i].dimensions]))}")
"""

NB03_VIZ_MD = """
## 4. Visualize before vs after

Draw the full mechanism (bars at t=0, every joint locus, feet in bold)
for the starting Strider and the best GA candidate side-by-side. A
pure foot-locus line-plot hides which body deformation the optimizer
actually favored — seeing the whole mechanism tells you that story.
"""

NB03_VIZ = """
best_agent = ensemble[0]
best_walker = make_strider(list(best_agent.dimensions))

fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)
show_strider(prototype, axes[0],
             f"Start (stride={r.score:.2f})")
show_strider(best_walker, axes[1],
             f"Best GA (stride={best_agent.score:.2f})")
plt.tight_layout()
plt.show()
"""

NB03_ENSEMBLE_MD = """
## 5. Ensemble inspection

`Ensemble` wraps the Pareto-ordered population with helpers:
`filter_by_score(min_score=...)`, `top(n, ascending=False)` for the
best few. The 1×3 panel below shows the three top survivors — they
usually cluster around one attractor, revealing how much (or little)
the GA has diversified at this budget.
"""

NB03_ENSEMBLE = """
survivors = ensemble.filter_by_score(min_val=r.score)
print(f"{len(survivors)} / {len(ensemble)} candidates beat the starting score")

fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), sharey=True)
for ax, agent in zip(axes, [ensemble[i] for i in range(min(3, len(ensemble)))]):
    w = make_strider(list(agent.dimensions))
    show_strider(w, ax, f"stride={agent.score:.2f}")
plt.tight_layout()
plt.show()
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
from pylinkage import extract_trajectory
from pylinkage.visualizer import plot_static_linkage
from pylinkage.optimization import multi_objective_optimization
from pylinkage.optimization.collections import ParetoFront, ParetoSolution

warnings.filterwarnings('ignore', category=DeprecationWarning)
np.random.seed(42)

# Reproducible flat ground — the default terrain has a 10° slope + heavy
# noise + unseeded RNG, so repeated runs disagree.
FLAT = ls.TerrainConfig.from_preset(ls.TerrainPreset.FLAT)
FLAT.seed = 0
WORLD_CFG = ls.WorldConfig(terrain=FLAT)
"""

NB04_PROTO_MD = """
## 1. Starting walker — a de-tuned Jansen

Jansen's canonical Holy Numbers are already the product of decades
of hand-optimization; starting NSGA there and expecting it to
improve in 240 evaluations is unrealistic. Instead we start from a
**de-tuned** walker — the canonical lengths with four parameters
perturbed by ±15 %. This gives NSGA real headroom to recover.

The 11 optimizable length parameters (Jansen's naming convention —
see `leggedsnake._classical.JANSEN_HOLY_NUMBERS`):

| key | meaning                              |
|-----|--------------------------------------|
| m   | crank radius (O → A)                 |
| j,k | crank-to-knee distances (A → C, D)   |
| b,c,d | frame-to-knee distances (B → C, D, E) |
| e   | diagonal C → E                       |
| f,g | knee-to-ankle (E → F, D → F)         |
| h,i | ankle-to-foot (F → G, D → G)         |

Each fitness evaluator adds `add_opposite_leg()` plus three
phase-offset copies internally (`n_legs=4, mirror=True`), so the
*search candidate* stays 11-dimensional (one leg's length
parameters) even though the *simulated walker* is the canonical
8-leg Strandbeest (4 per side, mirrored). Below is the starting
foot locus (one crank revolution, kinematic only — no physics yet).
"""

NB04_PROTO = """
from leggedsnake._classical import JANSEN_HOLY_NUMBERS

# The 11 length parameters that control a Jansen leg, in a fixed
# order that the NSGA will search over.
LENGTH_KEYS = ['m', 'j', 'b', 'k', 'c', 'd', 'e', 'g', 'f', 'i', 'h']
CANONICAL = {k: JANSEN_HOLY_NUMBERS[k] for k in LENGTH_KEYS}

# De-tune four parameters so the starting walker has visible room to
# improve. These multipliers stay ±15 % so the de-tuned geometry is
# still buildable.
PERTURB = {'j': 1.15, 'k': 0.85, 'f': 1.15, 'i': 0.85}
START = [CANONICAL[k] * PERTURB.get(k, 1.0) for k in LENGTH_KEYS]

def walker_from_lengths(vec):
    # Build a fresh Jansen Walker with the given 11-length vector.
    # Using the factory's ``lengths=`` override sidesteps the hypergraph
    # ``set_num_constraints`` limitation (which can only propagate
    # mobile-dyad edges — the 9 rigid-triangle edges stay pinned).
    return ls.Walker.from_jansen(
        scale=0.1,
        lengths=dict(zip(LENGTH_KEYS, vec)),
    )

prototype = walker_from_lengths(START)
print(f"de-tuned start: DOF={prototype.dof}")
print(f"perturbed params: {PERTURB}")

mech = prototype.to_mechanism()
loci = list(prototype.step(iterations=96))
fig, ax = plt.subplots(figsize=(7, 4.2))
plot_static_linkage(
    mech, ax, loci,
    show_loci=True, show_labels=False, show_legend=False,
    title="De-tuned Jansen (scale=0.1) — kinematic preview",
)
for i, joint in enumerate(mech.joints):
    if (getattr(joint, 'name', '') or '').startswith('G'):
        xs, ys = extract_trajectory(loci, i)
        if xs.size:
            ax.plot(xs, ys, color='crimson', lw=2.0, alpha=0.9, zorder=10)
ax.set_aspect('equal'); ax.grid(True, alpha=0.3)
plt.show()
"""

NB04_NSGA_MD = """
## 2. NSGA-II over distance and efficiency

Both objectives are physics-based (`DistanceFitness`,
`EfficiencyFitness`), so each candidate is a short pymunk rollout.
We budget `pop=24 × generations=10 = 240` evaluations (×2 objectives
≈ 480 rollouts at `duration=5.0 s`) — enough to populate the Pareto
front without slowing the notebook. For a serious run push both
counts up an order of magnitude and add `n_workers=4` to parallelize.

Three things matter for this search to produce diverse, stable walkers:

1. **Rebuild per candidate.** `Walker.set_num_constraints` on a
   hypergraph walker can't propagate through rigid triangles, so
   most edge-length changes get silently dropped. We bypass that
   by building a fresh `Walker.from_jansen(scale=0.1, lengths=...)`
   every evaluation — the factory rewires positions from scratch,
   guaranteeing all 11 length parameters actually land.
2. **`mirror=True`.** Without it, every candidate walks on all legs
   stacked on one side of the chassis — unstable, high variance
   (stdev ~4 m on distance, run-to-run). Mirroring halves the
   variance to ~1 m and gives the NSGA a real signal to optimize.
3. **Flat seeded terrain (`WORLD_CFG`).** The default terrain has
   a 10° slope plus noise with an unseeded RNG; without overriding
   that, every candidate walks on different hills.

Bounds are ±20 % of the canonical values — tight enough that most
samples are buildable, wide enough to move the gait.
"""

NB04_NSGA = """
# Shared fitness evaluators — one simulation per objective per candidate.
# ``mirror=True, n_legs=4`` = 4 legs per side, 8 total (canonical Strandbeest).
fit_distance = ls.DistanceFitness(duration=5.0, n_legs=4, mirror=True)
fit_efficiency = ls.EfficiencyFitness(
    duration=5.0, n_legs=4, mirror=True, min_distance=1.0,
)

def _score(fit, vec):
    try:
        w = walker_from_lengths(vec)
        r = fit(w.topology, w.dimensions, WORLD_CFG)
        return r.score if r.valid else 0.0
    except Exception:
        return 0.0  # unbuildable geometry

# ``multi_objective_optimization`` minimizes — walking scores maximize,
# so we negate here. The ``linkage``/``pos`` args are unused: we always
# rebuild from the 11-length ``dims`` vector.
def eval_distance(linkage, dims, pos):
    return -_score(fit_distance, list(dims))

def eval_efficiency(linkage, dims, pos):
    return -_score(fit_efficiency, list(dims))

lo = [0.80 * v for v in START]
hi = [1.20 * v for v in START]

ensemble = multi_objective_optimization(
    objectives=[eval_distance, eval_efficiency],
    linkage=prototype,
    bounds=(lo, hi),
    objective_names=['distance', 'efficiency'],
    algorithm='nsga2',
    n_generations=10, pop_size=24, seed=42, verbose=False,
)

# Bridge pylinkage's Ensemble → leggedsnake's NsgaWalkingResult so the
# built-in plotting helpers still work. Scores are un-negated for display
# (higher distance / efficiency = better).
solutions = [
    ParetoSolution(
        scores=(-float(ensemble.scores['distance'][i]),
                -float(ensemble.scores['efficiency'][i])),
        dimensions=np.asarray(ensemble.dimensions[i], dtype=float),
        initial_positions=(),
    )
    for i in range(ensemble.n_members)
]
pareto = ParetoFront(solutions, ('distance', 'efficiency'))
result = ls.NsgaWalkingResult(
    pareto_front=pareto,
    config=ls.NsgaWalkingConfig(
        n_generations=10, pop_size=24, seed=42, verbose=False,
    ),
)

print(f"Pareto front: {len(result.pareto_front.solutions)} solutions")
for i, sol in enumerate(result.pareto_front.solutions):
    print(f"  sol {i}: distance={sol.scores[0]:+.2f} m, "
          f"efficiency={sol.scores[1]:+.3f}")
"""

NB04_PLOT_MD = """
## 3. Plot the Pareto front

`plot_pareto_front` is a thin matplotlib wrapper that highlights the
"best compromise" (closest to the utopia corner in objective space).
"""

NB04_PLOT = """
fig = ls.plot_pareto_front(result, figsize=(6, 4))
plt.show()
"""

NB04_BEST_MD = """
## 4. Inspect the best-compromise solution

`result.best_compromise` is the Pareto solution with smallest L2
distance to the utopia corner in objective space.
"""

NB04_BEST = """
best = result.best_compromise()
print(f"best compromise: distance={best.scores[0]:+.2f} m, "
      f"efficiency={best.scores[1]:+.3f}")
print()
print(f"{'param':>6}  {'canonical':>10}  {'start':>10}  {'optimized':>10}  {'Δ vs start':>11}")
for key, start, opt in zip(LENGTH_KEYS, START, best.dimensions):
    canonical = CANONICAL[key]
    change = (float(opt) - start) / start * 100
    print(f"  {key:>4}  {canonical:>10.2f}  {start:>10.2f}  "
          f"{float(opt):>10.2f}  {change:>+10.1f}%")
"""

NB04_GAIT_MD = """
## 5. Gait analysis on a winning walker

`analyze_gait` returns **duty factor** (fraction of stride in
stance), **stride frequency**, and inter-foot **phase offsets**. A
gait diagram only tells a useful story for a walker that *actually
walks*, so we want the Pareto solution with the best distance
score. `best_for_objective(0)` picks by search-time score — but
pymunk's LCP solver is slightly non-deterministic, so that single
rollout can be a lucky outlier. We instead re-evaluate every
Pareto candidate with median-of-3 rollouts and pick the one with
the highest robust distance.

We re-simulate with `DistanceFitness(record_loci=True)` to record
every joint, then filter to the foot nodes (names starting with
``'G'``). The default `contact_threshold=0.1` assumes ground near
``y=0``; our road sits at ``bb_min_y − 1 ≈ −10``, so we self-tune
the threshold to each foot's observed y-range.
"""

NB04_GAIT = """
import statistics as _stats

# Re-score each Pareto candidate with median-of-3 and pick the most
# robustly-fastest — avoids the single-rollout variance that can make
# ``best_for_objective(0)`` pick an outlier.
_dist_fit = ls.DistanceFitness(duration=5.0, n_legs=4, mirror=True)
def _median_d(vec, n=3):
    w = walker_from_lengths(vec)
    return _stats.median(
        _dist_fit(w.topology, w.dimensions, WORLD_CFG).score for _ in range(n)
    )
ranked = sorted(
    result.pareto_front.solutions,
    key=lambda s: _median_d(list(s.dimensions)),
    reverse=True,
)
fastest = ranked[0]
print(f"most robustly-fastest Pareto solution: "
      f"distance={fastest.scores[0]:+.2f} m (search), "
      f"efficiency={fastest.scores[1]:+.3f}")

best_walker = walker_from_lengths(list(fastest.dimensions))

fit_record = ls.DistanceFitness(
    duration=5.0, n_legs=4, mirror=True, record_loci=True,
)
# A single rollout can be an LCP-solver outlier (walker tips over even
# when the median behavior is fine). Retry up to 3× until we capture a
# rollout that moved at least 1 m — that's the run we analyse gait on.
for _ in range(3):
    fr = fit_record(best_walker.topology, best_walker.dimensions, WORLD_CFG)
    if fr.score >= 1.0:
        break
foot_ids = [k for k in fr.loci if k.startswith('G')]
print(f"re-simulated distance = {fr.score:+.3f} m")
print(f"feet tracked ({len(foot_ids)}): {foot_ids[:3]}...")

# ``analyze_gait`` decides \"in stance\" via ``y <= contact_threshold``
# and defaults to ``0.1`` — valid when the ground is near y=0. Here the
# road sits at ``bb_min_y - 1 ≈ -10`` and individual walkers have
# different swing amplitudes, so we pick each walker's own threshold at
# the median of all foot-y samples — roughly 50% stance / 50% swing —
# then verify the feet actually cross it.
import numpy as np
all_ys = np.array([y for f in foot_ids for _, y in fr.loci[f]])
y_min, y_max = float(all_ys.min()), float(all_ys.max())
contact_threshold = float(np.median(all_ys))

gait = ls.analyze_gait(
    {f: fr.loci[f] for f in foot_ids}, foot_ids,
    dt=WORLD_CFG.physics_period,
    contact_threshold=contact_threshold,
)
print(f"contact threshold  : {contact_threshold:.2f} (foot y in [{y_min:.1f}, {y_max:.1f}])")
print(f"mean duty factor   : {gait.mean_duty_factor:.2f}")
print(f"mean stride freq   : {gait.mean_stride_frequency:.3f} Hz")
print(f"mean stride length : {gait.mean_stride_length:.3f} m")
"""

NB04_DIAG_MD = """
## 6. Gait diagram

`plot_gait_diagram` shows each foot as a stance/swing bar across
time. An 8-leg Strandbeest run at the same crank rate produces
evenly spaced stance bars — the canonical 4/4 Jansen gait, each
foot in stance ~½ of the cycle, offset by ~¼ period from its
neighbors.
"""

NB04_DIAG = """
fig = ls.plot_gait_diagram(gait, figsize=(8, 3.0))
plt.show()
"""

NB04_COMPARE_MD = """
## 7. Initial vs optimized — side by side

The Pareto front reports numbers; the side-by-side shows the
geometry. Left: the starting Jansen (factory Holy Numbers).
Right: the fastest Pareto survivor. The bars at ``t=0`` plus all
joint loci plus the foot trajectory in red — same visualization as
§1 — let you see *what the optimizer actually changed* and whether
the foot locus gets bigger, flatter, or more symmetric along the way.

We also re-score both walkers with the same `DistanceFitness` used
in the search so the numbers on each title are directly comparable.
"""

NB04_COMPARE = """
def preview(walker, ax, title):
    mech = walker.to_mechanism()
    loci = list(walker.step(iterations=96))
    plot_static_linkage(
        mech, ax, loci,
        show_loci=True, show_labels=False, show_legend=False,
        title=title,
    )
    for i, joint in enumerate(mech.joints):
        if (getattr(joint, 'name', '') or '').startswith('G'):
            xs, ys = extract_trajectory(loci, i)
            if xs.size:
                ax.plot(xs, ys, color='crimson', lw=2.0, alpha=0.9, zorder=10)
    ax.set_aspect('equal'); ax.grid(True, alpha=0.3)

# Pymunk distance varies run-to-run (float-order in the LCP solver), so
# we take the median of 3 rollouts per walker for a stable comparison.
import statistics
score_fit = ls.DistanceFitness(duration=5.0, n_legs=4, mirror=True)

def median_distance(walker, n=3):
    return statistics.median(
        score_fit(walker.topology, walker.dimensions, WORLD_CFG).score
        for _ in range(n)
    )

canonical_walker = walker_from_lengths([CANONICAL[k] for k in LENGTH_KEYS])

start_distance = median_distance(prototype)
opt_distance = median_distance(best_walker)
canonical_distance = median_distance(canonical_walker)

fig, axes = plt.subplots(1, 3, figsize=(16, 5), sharey=True)
preview(prototype, axes[0],
        f"De-tuned start (median distance={start_distance:+.2f} m)")
preview(canonical_walker, axes[1],
        f"Canonical Jansen (median distance={canonical_distance:+.2f} m)")
preview(walker_from_lengths(list(fastest.dimensions)), axes[2],
        f"NSGA fastest (median distance={opt_distance:+.2f} m)")
plt.tight_layout()
plt.show()
delta = opt_distance - start_distance
print(f"Δ vs de-tuned start: {delta:+.2f} m", end='')
# Report % only when the baseline is not ~0 (otherwise the ratio blows up).
if abs(start_distance) >= 1.0:
    print(f"  ({delta / abs(start_distance) * 100:+.0f}%)")
else:
    print("  (start barely moves — de-tuning broke its gait)")
print(f"Δ vs canonical Jansen: {opt_distance - canonical_distance:+.2f} m "
      f"(Jansen's hand-tuned numbers are hard to beat in 240 evals)")
"""

NB04_RACE_MD = """
## 8. Watch them race

Static pictures hide the dynamics — let's drop both walkers into
pymunk worlds on the **same plot** and let them race for 10 s.
The de-tuned starting walker (gray) and the NSGA optimized walker
(blue) run on identical flat terrain with identical motor rates;
the only difference is the 11 length parameters. Whoever's chassis
is further right at the end wins.
"""

NB04_RACE = """
from matplotlib import animation
from IPython.display import HTML

def prepare_race(walker):
    # same 4-per-side mirrored Strandbeest the search used
    walker.add_opposite_leg()
    walker.add_legs(3)
    bb_min_y = ls.linkage_bb(walker)[0]
    w = ls.World(config=ls.WorldConfig(terrain=FLAT), road_y=bb_min_y - 1)
    w.add_linkage(walker)
    return w, list(walker.topology.edges.items())

start_race = walker_from_lengths(START)
opt_race = walker_from_lengths(list(fastest.dimensions))
world_s, edges_s = prepare_race(start_race)
world_o, edges_o = prepare_race(opt_race)

race_duration = 10.0
race_stride = 10  # record every 10th physics tick → 50 frames at 200 ms
n_race_steps = int(race_duration / world_s.config.physics_period)

race_frames = []
for i in range(n_race_steps):
    world_s.update()
    world_o.update()
    if i % race_stride:
        continue
    ps = world_s.linkages[0].get_all_positions()
    po = world_o.linkages[0].get_all_positions()
    race_frames.append({
        's_segs': [(ps[e.source], ps[e.target]) for _, e in edges_s
                    if e.source in ps and e.target in ps],
        'o_segs': [(po[e.source], po[e.target]) for _, e in edges_o
                    if e.source in po and e.target in po],
        # Each World only extends its road as *its* walker advances, so we
        # store both — the blue walker overruns the gray walker's road,
        # and vice-versa, whenever their x-positions diverge.
        's_road': list(world_s.road),
        'o_road': list(world_o.road),
        's_cx': world_s.linkages[0].body.position.x,
        'o_cx': world_o.linkages[0].body.position.x,
        't': i * world_s.config.physics_period,
    })

bb_s = ls.linkage_bb(start_race)  # (min_y, max_x, max_y, min_x)
bb_o = ls.linkage_bb(opt_race)
walker_half_w = max(bb_s[1] - bb_s[3], bb_o[1] - bb_o[3]) * 0.6
y_lo = min(bb_s[0], bb_o[0]) - 2
y_hi = max(bb_s[2], bb_o[2]) + 2

fig_r, ax_r = plt.subplots(figsize=(12, 5))
plt.close(fig_r)  # suppress static snapshot; animation follows

def draw_race(frame):
    ax_r.clear()
    # Union the two roads — each World extends only as its walker moves,
    # so the blue-winning side needs the blue road and the gray-winning
    # side needs the gray road. They're at the same road_y, so an
    # x-sorted merge gives a single continuous ground line.
    combined = sorted(
        {(round(x, 3), round(y, 3)) for x, y in frame['s_road']}
        | {(round(x, 3), round(y, 3)) for x, y in frame['o_road']}
    )
    xs, ys = zip(*combined)
    ax_r.plot(xs, ys, color='#555', lw=1.2)
    for (a, b) in frame['s_segs']:
        ax_r.plot([a[0], b[0]], [a[1], b[1]], '-',
                  color='#999', lw=1.2, alpha=0.8)
    for (a, b) in frame['o_segs']:
        ax_r.plot([a[0], b[0]], [a[1], b[1]], '-',
                  color='#1f77b4', lw=1.4, alpha=0.9)
    ax_r.plot([frame['s_cx']], [0], 'o', color='#666',
              ms=6, label=f"start x={frame['s_cx']:+.2f} m")
    ax_r.plot([frame['o_cx']], [0], 'o', color='#1f77b4',
              ms=6, label=f"optimized x={frame['o_cx']:+.2f} m")
    # Camera: center on midpoint, wide enough to contain both walkers
    mid = 0.5 * (frame['s_cx'] + frame['o_cx'])
    half_w = max(abs(frame['o_cx'] - frame['s_cx']) * 0.6 + walker_half_w,
                 walker_half_w * 2)
    ax_r.set_xlim(mid - half_w, mid + half_w)
    ax_r.set_ylim(y_lo, y_hi)
    ax_r.set_aspect('equal'); ax_r.grid(True, alpha=0.3)
    ax_r.set_title(f"Start (gray) vs NSGA optimized (blue) — t={frame['t']:.1f}s")
    ax_r.legend(loc='upper right', fontsize=9)

ani = animation.FuncAnimation(
    fig_r, draw_race, frames=race_frames, interval=200, blit=False,
)
HTML(ani.to_jshtml())
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
    _md(NB04_PROTO_MD), _code(NB04_PROTO),
    _md(NB04_NSGA_MD), _code(NB04_NSGA),
    _md(NB04_PLOT_MD), _code(NB04_PLOT),
    _md(NB04_BEST_MD), _code(NB04_BEST),
    _md(NB04_GAIT_MD), _code(NB04_GAIT),
    _md(NB04_DIAG_MD), _code(NB04_DIAG),
    _md(NB04_COMPARE_MD), _code(NB04_COMPARE),
    _md(NB04_RACE_MD), _code(NB04_RACE),
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
