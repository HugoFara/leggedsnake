#!/usr/bin/env python3
"""
End-to-end walking mechanism optimization pipeline.

Demonstrates the full workflow:
  1. Define a walker mechanism (four-bar with rigid triangle foot)
  2. Run multi-objective optimization (NSGA-II) for distance + stability
  3. Analyze gait cycles and stability of the best solutions
  4. Visualize results: Pareto front, gait diagram, stability, foot paths

Run with: uv run python examples/optimization_pipeline.py

This example uses very small populations and generations to run quickly.
For real optimization, increase pop_size to 50-100 and n_generations to
50-200.
"""
from __future__ import annotations

import matplotlib
matplotlib.use("Agg")  # Use non-interactive backend; change to "TkAgg" for display

from math import tau

from leggedsnake import (
    Dimensions,
    DistanceFitness,
    DriverAngle,
    Edge,
    Hyperedge,
    HypergraphLinkage,
    Node,
    NodeRole,
    NsgaWalkingConfig,
    StabilityFitness,
    Walker,
    nsga_walking_optimization,
    # Plotting
    plot_pareto_front,
    plot_gait_diagram,
    plot_stability_timeseries,
    plot_com_trajectory,
    plot_foot_trajectories,
    plot_optimization_dashboard,
)


# ── Step 1: Define the walker mechanism ────────────────────────────────


def make_walker() -> Walker:
    """Build a four-bar walker with a rigid-triangle foot.

    This is a minimal walking mechanism: a crank-rocker four-bar
    with a triangular extension that forms the foot. The foot traces
    an approximately D-shaped path that can produce walking motion.
    """
    hg = HypergraphLinkage(name="OptDemo")
    hg.add_node(Node("frame", role=NodeRole.GROUND))
    hg.add_node(Node("frame2", role=NodeRole.GROUND))
    hg.add_node(Node("crank", role=NodeRole.DRIVER))
    hg.add_node(Node("upper", role=NodeRole.DRIVEN))
    hg.add_node(Node("foot", role=NodeRole.DRIVEN))

    hg.add_edge(Edge("frame_crank", "frame", "crank"))
    hg.add_edge(Edge("crank_upper", "crank", "upper"))
    hg.add_edge(Edge("frame2_upper", "frame2", "upper"))
    hg.add_edge(Edge("upper_foot", "upper", "foot"))
    hg.add_hyperedge(Hyperedge("tri_foot", nodes=("upper", "crank", "foot")))

    dims = Dimensions(
        node_positions={
            "frame": (0, 0),
            "crank": (0.5, 0),
            "frame2": (-1, -0.3),
            "upper": (0, 1),
            "foot": (0, -1.5),
        },
        driver_angles={
            "crank": DriverAngle(angular_velocity=tau / 12),
        },
        edge_distances={
            "frame_crank": 0.5,
            "crank_upper": 1.5,
            "frame2_upper": 2.0,
            "upper_foot": 2.0,
        },
    )

    walker = Walker(hg, dims, name="OptDemo")
    list(walker.step(iterations=1))  # solve initial positions
    return walker


# ── Step 2: Configure and run NSGA-II optimization ─────────────────────


def run_optimization():
    """Run multi-objective optimization and return results."""
    # Define objectives: maximize distance and stability
    objectives = [
        DistanceFitness(duration=4, n_legs=2, record_loci=True),
        StabilityFitness(duration=4, n_legs=2, min_distance=0.5),
    ]

    # Parameter bounds: [frame_crank, crank_upper, frame2_upper, upper_foot]
    lower = [0.3, 1.0, 1.5, 1.5]
    upper = [1.0, 2.5, 3.0, 3.0]

    config = NsgaWalkingConfig(
        n_generations=5,   # Use 50+ for real optimization
        pop_size=10,       # Use 50-100 for real optimization
        seed=42,
        verbose=False,
    )

    print("Running NSGA-II optimization...")
    result = nsga_walking_optimization(
        walker_factory=make_walker,
        objectives=objectives,
        bounds=(lower, upper),
        objective_names=["distance", "stability"],
        nsga_config=config,
        include_gait=True,
        include_stability=True,
    )

    n_solutions = len(result.pareto_front.solutions)
    print(f"Found {n_solutions} Pareto-optimal solutions")

    if n_solutions > 0:
        best = result.best_compromise()
        print(f"Best compromise: distance={best.scores[0]:.3f}, "
              f"stability={best.scores[1]:.3f}")

    return result


# ── Step 3: Visualize results ──────────────────────────────────────────


def visualize(result):
    """Generate and save all visualization plots."""
    print("\nGenerating plots...")

    # 3a. Pareto front
    if len(result.pareto_front.solutions) >= 2:
        fig = plot_pareto_front(result)
        fig.savefig("examples/verification_plots/pareto_front.png", dpi=150)
        print("  Saved pareto_front.png")

    # 3b. Gait diagram (for solution 0)
    if result.gait_analyses and 0 in result.gait_analyses:
        fig = plot_gait_diagram(result.gait_analyses[0])
        fig.savefig("examples/verification_plots/gait_diagram.png", dpi=150)
        print("  Saved gait_diagram.png")

    # 3c. Stability time series
    if result.stability_series and 0 in result.stability_series:
        series = result.stability_series[0]
        fig = plot_stability_timeseries(series)
        fig.savefig("examples/verification_plots/stability.png", dpi=150)
        print("  Saved stability.png")

        fig = plot_com_trajectory(series)
        fig.savefig("examples/verification_plots/com_trajectory.png", dpi=150)
        print("  Saved com_trajectory.png")

    # 3d. Foot trajectories
    if result.gait_analyses and 0 in result.gait_analyses:
        fig = plot_foot_trajectories(result.gait_analyses[0])
        fig.savefig("examples/verification_plots/foot_trajectories.png", dpi=150)
        print("  Saved foot_trajectories.png")

    # 3e. Combined dashboard
    fig = plot_optimization_dashboard(result, solution_index=0)
    fig.savefig("examples/verification_plots/dashboard.png", dpi=150)
    print("  Saved dashboard.png")

    # Print summary metrics
    print("\n── Metrics Summary ──")
    if result.gait_analyses and 0 in result.gait_analyses:
        gait = result.gait_analyses[0]
        print(f"  Mean duty factor:      {gait.mean_duty_factor:.3f}")
        print(f"  Mean stride frequency: {gait.mean_stride_frequency:.3f} Hz")
        print(f"  Mean stride length:    {gait.mean_stride_length:.3f}")

    if result.stability_series and 0 in result.stability_series:
        stab = result.stability_series[0]
        metrics = stab.summary_metrics()
        print(f"  Mean tip-over margin:  {metrics['mean_tip_over_margin']:.3f}")
        print(f"  Min tip-over margin:   {metrics['min_tip_over_margin']:.3f}")
        print(f"  ZMP excursion:         {metrics['zmp_excursion']:.3f}")
        print(f"  Angular stability:     {metrics['angular_stability']:.4f}")


# ── Main ───────────────────────────────────────────────────────────────


def main():
    print("=" * 50)
    print("Walking Mechanism Optimization Pipeline")
    print("=" * 50)
    print()

    result = run_optimization()
    visualize(result)

    print("\nDone! Check examples/verification_plots/ for output.")


if __name__ == "__main__":
    main()
