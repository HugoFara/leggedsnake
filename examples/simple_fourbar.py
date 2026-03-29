#!/usr/bin/env python3
"""
Simple four-bar linkage visualization demo.

This example creates a basic four-bar linkage and simulates it
using the Pyglet-based visualizer.

Run with: uv run python examples/simple_fourbar.py
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


def create_fourbar_linkage():
    """Create a simple four-bar linkage."""
    # --- Topology ---
    hg = HypergraphLinkage(name="FourBar")

    # Ground joints (fixed frame points)
    hg.add_node(Node("ground", role=NodeRole.GROUND))
    hg.add_node(Node("rocker_ground", role=NodeRole.GROUND))

    # Crank - the driving link that rotates
    hg.add_node(Node("crank", role=NodeRole.DRIVER))

    # Coupler - connects crank to rocker (circle-circle intersection)
    hg.add_node(Node("coupler", role=NodeRole.DRIVEN))

    # Output - rigid triangle vertex on the crank-coupler link
    hg.add_node(Node("output", role=NodeRole.DRIVEN))

    # Edges (links between joints)
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
            "crank": (1, 0),
            "rocker_ground": (3, 0),
            "coupler": (1.5, 1.5),
            "output": (0.5, 1.0),  # approximate, will be solved
        },
        driver_angles={
            "crank": DriverAngle(angular_velocity=pi / 12),  # 15 degrees per step
        },
        edge_distances={
            "ground_crank": 1.0,
            "crank_coupler": 2.0,        # Distance from crank tip
            "rocker_coupler": 2.5,       # Distance from rocker ground
            "crank_output": 1.5,
        },
    )

    walker = Walker(hg, dims, name="FourBar")

    # Perform one kinematic step to solve initial joint positions
    list(walker.step(iterations=1))

    return walker


def main():
    print("Four-Bar Linkage Visualization Demo")
    print("=" * 40)
    print("Controls:")
    print("  - Press Q or ESC to quit")
    print("  - Window is resizable")
    print("=" * 40)

    # Create the linkage
    walker = create_fourbar_linkage()

    # Run the visualization for 10 seconds with dynamic camera (follows the linkage)
    ls.video(walker, duration=10, dynamic_camera=True)


if __name__ == "__main__":
    main()
