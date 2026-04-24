"""
Regenerate the animated GIFs referenced from ``README.md``.

Produces two files in ``examples/images/``:

* ``Striders run.gif`` — one Strider mechanism, optimized over 10 GA
  generations on a kinematic stride objective, then rendered walking
  in physics with 8 phase-offset legs (4 per side, mirrored).
* ``Jansen quick start.gif`` — the quick-start Strandbeest from
  ``Walker.from_jansen`` with the same 8-leg arrangement (this is
  notebook 02's proven-stable config).

Rendering goes through ``matplotlib.animation.PillowWriter``, which is
headless and reproducible — unlike ``leggedsnake.video()`` (pyglet,
interactive only).

Run from the repo root:

    uv run python examples/tools/generate_readme_gifs.py
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import animation

import leggedsnake as ls


IMAGES_DIR = Path(__file__).parent.parent / "images"

STRIDER_PARAM_NAMES = [
    "crank", "triangle", "femur", "rocker_l", "rocker_s", "tibia", "foot",
]
STRIDER_START_PARAMS = [1.0, 2.0, 1.8, 2.6, 1.4, 2.5, 1.8]


def _flat_world_cfg() -> ls.WorldConfig:
    """Flat, seeded ground — reproducible across runs."""
    terrain = ls.TerrainConfig.from_preset(ls.TerrainPreset.FLAT)
    terrain.seed = 0
    return ls.WorldConfig(terrain=terrain)


def _record(walker, duration: float, record_stride: int) -> list[dict]:
    """Run physics on ``walker`` for ``duration`` seconds, store key frames."""
    cfg = _flat_world_cfg()
    bb_min_y = ls.linkage_bb(walker)[0]
    world = ls.World(config=cfg, road_y=bb_min_y - 1)
    world.add_linkage(walker)

    edges = list(walker.topology.edges.items())
    dt = cfg.physics_period
    n_steps = int(duration / dt)

    frames: list[dict] = []
    for i in range(n_steps):
        world.update()
        if i % record_stride:
            continue
        pos = world.linkages[0].get_all_positions()
        frames.append({
            "segs": [
                (pos[e.source], pos[e.target])
                for _, e in edges
                if e.source in pos and e.target in pos
            ],
            "road": list(world.road),
            "cx": float(world.linkages[0].body.position.x),
            "t": i * dt,
        })
    return frames


def _render_gif(
    walker,
    frames: list[dict],
    path: Path,
    title: str,
    fps: int = 20,
    figsize: tuple[float, float] = (8, 4.5),
    dpi: int = 80,
    color: str = "#1f77b4",
) -> None:
    """Animate ``frames`` and write an animated GIF to ``path``.

    Camera math follows notebook 02's pattern: the viewport is sized
    once from the walker's pre-physics bounding box and moves with the
    chassis centre, so the walker stays the same size throughout.
    """
    bb_min_y, bb_max_x, bb_max_y, bb_min_x = ls.linkage_bb(walker)
    walker_w = bb_max_x - bb_min_x
    walker_h = bb_max_y - bb_min_y
    road_y = bb_min_y - 1
    half_w = walker_w * 0.75
    y_lo = road_y - 1
    y_hi = road_y + walker_h + 3

    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    plt.close(fig)

    def draw(frame: dict) -> None:
        ax.clear()
        xs, ys = zip(*frame["road"])
        ax.plot(xs, ys, color="#8b5a2b", lw=1.2)
        ax.fill_between(xs, ys, y_lo, color="#d9c7a5", alpha=0.35)
        for a, b in frame["segs"]:
            ax.plot([a[0], b[0]], [a[1], b[1]],
                    color=color, lw=1.4, alpha=0.85)
        mid = frame["cx"]
        ax.set_xlim(mid - half_w, mid + half_w)
        ax.set_ylim(y_lo, y_hi)
        ax.set_aspect("equal")
        ax.grid(True, alpha=0.2)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(
            f"{title} — t={frame['t']:.1f}s, x={frame['cx']:+.2f} m",
            fontsize=10,
        )

    anim = animation.FuncAnimation(
        fig, draw, frames=frames, interval=int(1000 / fps)
    )
    writer = animation.PillowWriter(fps=fps)
    path.parent.mkdir(parents=True, exist_ok=True)
    anim.save(str(path), writer=writer, dpi=dpi)
    size_kb = path.stat().st_size / 1024
    print(f"  wrote {path} ({size_kb:.0f} KB)")


def _build_strider(params=STRIDER_START_PARAMS):
    """Build a Strider Walker from 7 compact length parameters."""
    return ls.Walker.from_strider(**dict(zip(STRIDER_PARAM_NAMES, params)))


_STRIDER_FITNESS = ls.DistanceFitness(duration=5.0, n_legs=2, mirror=True)


def _eval_strider_distance(linkage, dims, pos):
    """Physics distance score for the GA — ignores ``linkage``/``pos``.

    Rebuilds a fresh Strider from the 7 compact params on each
    evaluation (``set_num_constraints`` can't propagate through the
    hypergraph's rigid triangles). Uses physics simulation directly
    because kinematic stride alone can select for walkers that tip
    over under gravity.
    """
    try:
        w = _build_strider(list(dims))
        result = _STRIDER_FITNESS(w.topology, w.dimensions, _flat_world_cfg())
        return float(result.score) if result.valid else 0.0
    except Exception:
        return 0.0


def generate_strider_gif() -> None:
    """Run 10 GA generations on a Strider and render the winner walking."""
    print("Building Strider seed...")
    seed = _build_strider()
    seed_score = _eval_strider_distance(None, STRIDER_START_PARAMS, None)
    print(f"  seed distance: {seed_score:+.2f} m")

    print("Running 10-generation GA (pop=12, single process)...")
    np.random.seed(42)
    bounds = (
        [0.7 * v for v in STRIDER_START_PARAMS],
        [1.3 * v for v in STRIDER_START_PARAMS],
    )
    # processes=1: the helper wraps eval_func in a local closure that
    # can't be pickled across process boundaries (see notebook 03).
    # DistanceFitness is ~0.08 s per 5-second rollout, so 120 evals
    # finish in ~10 s on a single core.
    ensemble = ls.genetic_algorithm_optimization(
        eval_func=_eval_strider_distance,
        linkage=seed,
        center=STRIDER_START_PARAMS,
        bounds=bounds,
        max_pop=12,
        iters=10,
        processes=1,
        verbose=False,
    )
    # Physics is non-deterministic (LCP solver float order): the
    # ensemble's rank-0 candidate may have been a lucky single rollout.
    # Re-score the top 5 with median-of-3 and pick the most robust.
    cfg = _flat_world_cfg()
    import statistics as _st
    top5 = [ensemble[i] for i in range(min(5, len(ensemble)))]
    ranked = []
    for agent in top5:
        w = _build_strider(list(agent.dimensions))
        med = _st.median(
            _STRIDER_FITNESS(w.topology, w.dimensions, cfg).score
            for _ in range(3)
        )
        ranked.append((med, agent))
    ranked.sort(key=lambda r: r[0], reverse=True)
    best_score, best = ranked[0]
    print(f"  best median distance: {best_score:+.2f} m")
    print(f"  best params: {dict(zip(STRIDER_PARAM_NAMES, [round(float(d), 2) for d in best.dimensions]))}")

    # Rebuild the winner with the same 2-per-side mirrored arrangement
    # used at search time. A Strider leg pair already has two feet
    # (symmetric mechanism), so 2 pairs × 2 feet = 4 feet is enough
    # stance coverage, while the thinner leg tangle reads cleanly as a
    # walker in the rendered GIF.
    winner = _build_strider(list(best.dimensions))
    winner.add_opposite_leg()
    winner.add_legs(1)
    tuple(winner.step())

    print("Recording winner simulation...")
    frames = _record(winner, duration=8.0, record_stride=4)
    print(f"  {len(frames)} frames")

    print("Rendering GIF...")
    _render_gif(
        winner,
        frames,
        IMAGES_DIR / "Striders run.gif",
        title="Strider optimized over 10 generations",
        color="#1f77b4",
    )


def generate_jansen_gif() -> None:
    """Render the Walker.from_jansen quick-start example walking."""
    print("Building Jansen walker...")
    # Match notebook 02's proven 8-leg Strandbeest config: 4 legs per
    # side, mirrored. A 2- or 4-leg Jansen pitches by 10-20° and can
    # tip over; 8 legs keep the body level.
    walker = ls.Walker.from_jansen(scale=0.1)
    walker.add_opposite_leg()
    walker.add_legs(3)

    print("Recording simulation...")
    frames = _record(walker, duration=8.0, record_stride=4)
    print(f"  {len(frames)} frames")

    print("Rendering GIF...")
    _render_gif(
        walker,
        frames,
        IMAGES_DIR / "Jansen quick start.gif",
        title="Theo Jansen Strandbeest (Holy Numbers)",
        color="#d62728",
    )


def main() -> None:
    generate_strider_gif()
    generate_jansen_gif()


if __name__ == "__main__":
    main()
