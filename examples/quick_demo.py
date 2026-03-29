#!/usr/bin/env python3
"""
Quick demo of the Pyglet visualization.

This is the simplest possible example - creates a basic mechanism
and runs the visualization immediately.

Run with: uv run python examples/quick_demo.py
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


def main():
    print("Quick Visualization Demo")
    print("Press Q or ESC to quit")
    print("-" * 30)

    # --- Topology: a simple crank-rocker mechanism ---
    hg = HypergraphLinkage(name="QuickDemo")

    hg.add_node(Node("base", role=NodeRole.GROUND))
    hg.add_node(Node("crank", role=NodeRole.DRIVER))
    hg.add_node(Node("follower", role=NodeRole.DRIVEN))
    hg.add_node(Node("output", role=NodeRole.DRIVEN))

    hg.add_edge(Edge("base_crank", "base", "crank"))
    hg.add_edge(Edge("base_follower", "base", "follower"))
    hg.add_edge(Edge("crank_follower", "crank", "follower"))

    # Output forms a rigid triangle with crank and follower
    hg.add_edge(Edge("crank_output", "crank", "output"))
    hg.add_hyperedge(Hyperedge("triangle_output", nodes=("crank", "follower", "output")))

    # --- Dimensions ---
    dims = Dimensions(
        node_positions={
            "base": (0, 0),
            "crank": (1, 0),
            "follower": (0, 2),
            "output": (0.5, -0.5),  # approximate, will be solved
        },
        driver_angles={
            "crank": DriverAngle(angular_velocity=0.5),
        },
        edge_distances={
            "base_crank": 1.0,
            "base_follower": 2.0,
            "crank_follower": 1.5,
            "crank_output": 1.5,
        },
    )

    walker = Walker(hg, dims, name="QuickDemo")

    # Perform one kinematic step to solve initial joint positions
    list(walker.step(iterations=1))

    # Run for 10 seconds
    ls.video(walker, duration=10, dynamic_camera=True)


if __name__ == "__main__":
    main()
