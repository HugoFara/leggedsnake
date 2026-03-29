#!/usr/bin/env python3
"""
Compare multiple linkages side by side.

This example creates several linkages with different parameters
and visualizes them together with varying opacities.

Run with: uv run python examples/compare_linkages.py
"""
from math import pi

import leggedsnake as ls
from leggedsnake import (
    Dimensions,
    DriverAngle,
    Edge,
    Hyperedge,
    HypergraphLinkage,
    Node,
    NodeRole,
    Walker,
)


def create_linkage(crank_length: float, coupler_length: float, name: str):
    """Create a four-bar linkage with specified dimensions."""
    # --- Topology ---
    hg = HypergraphLinkage(name=name)

    hg.add_node(Node("ground", role=NodeRole.GROUND))
    hg.add_node(Node("rocker_ground", role=NodeRole.GROUND))
    hg.add_node(Node("crank", role=NodeRole.DRIVER))
    hg.add_node(Node("coupler", role=NodeRole.DRIVEN))
    hg.add_node(Node("output", role=NodeRole.DRIVEN))

    hg.add_edge(Edge("ground_crank", "ground", "crank"))
    hg.add_edge(Edge("crank_coupler", "crank", "coupler"))
    hg.add_edge(Edge("rocker_coupler", "rocker_ground", "coupler"))

    # Output forms a rigid triangle with crank and coupler
    hg.add_edge(Edge("crank_output", "crank", "output"))
    hg.add_hyperedge(Hyperedge("triangle_output", nodes=("crank", "coupler", "output")))

    # --- Dimensions ---
    dims = Dimensions(
        node_positions={
            "ground": (0, 0),
            "crank": (crank_length, 0),
            "rocker_ground": (3, 0),
            "coupler": (crank_length, coupler_length),
            "output": (crank_length * 0.5, coupler_length * 0.5),
        },
        driver_angles={
            "crank": DriverAngle(angular_velocity=pi / 12),  # 15 degrees per step
        },
        edge_distances={
            "ground_crank": crank_length,
            "crank_coupler": coupler_length,
            "rocker_coupler": 2.5,
            "crank_output": 1.2,
        },
    )

    walker = Walker(hg, dims, name=name)

    # Perform one kinematic step to solve initial joint positions
    list(walker.step(iterations=1))

    return walker


def main():
    print("Multi-Linkage Comparison Demo")
    print("=" * 40)
    print("Comparing linkages with different parameters")
    print("Opacity indicates which variant:")
    print("  - Brightest: variant 1 (small crank)")
    print("  - Medium: variant 2 (medium crank)")
    print("  - Faintest: variant 3 (large crank)")
    print()
    print("Controls:")
    print("  - Press Q or ESC to quit")
    print("  - Window is resizable")
    print("=" * 40)

    # Create multiple linkages with different parameters
    linkages = [
        create_linkage(crank_length=0.8, coupler_length=1.5, name="Small"),
        create_linkage(crank_length=1.0, coupler_length=2.0, name="Medium"),
        create_linkage(crank_length=1.2, coupler_length=2.5, name="Large"),
    ]

    # Opacities from brightest to faintest
    opacities = [1.0, 0.6, 0.3]

    # Run visualization with all linkages
    # dynamic_camera=False shows the whole scene
    ls.all_linkages_video(
        linkages,
        duration=15,
        colors=opacities,
        dynamic_camera=False
    )


if __name__ == "__main__":
    main()
