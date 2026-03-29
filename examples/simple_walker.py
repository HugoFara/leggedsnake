#!/usr/bin/env python3
"""
Simple walker mechanism visualization demo.

This example creates a basic Klann-style walking linkage with two legs
and simulates it using the Pyglet-based visualizer.

Run with: uv run python examples/simple_walker.py
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


def create_klann_walker(opposite: bool = True):
    """
    Create a simplified Klann-style walking linkage.

    The Klann linkage is a planar mechanism designed to simulate
    the gait of legged animals.

    Parameters
    ----------
    opposite : bool, optional
        If True, create an antisymmetric copy of the leg on the opposite side.
        This creates left/right leg pairs. Default is True.
    """
    # --- Topology ---
    hg = HypergraphLinkage(name="SimpleWalker")

    # Frame/chassis anchor points
    hg.add_node(Node("frame", role=NodeRole.GROUND))
    hg.add_node(Node("frame2", role=NodeRole.GROUND))

    # Crank - the motor-driven input link
    hg.add_node(Node("crank", role=NodeRole.DRIVER))

    # Upper linkage - connects crank to frame2 (circle-circle intersection)
    hg.add_node(Node("upper", role=NodeRole.DRIVEN))

    # Foot - rigid triangle vertex on the upper-crank link
    hg.add_node(Node("foot", role=NodeRole.DRIVEN))

    # Edges
    hg.add_edge(Edge("frame_crank", "frame", "crank"))
    hg.add_edge(Edge("crank_upper", "crank", "upper"))
    hg.add_edge(Edge("frame2_upper", "frame2", "upper"))

    # Foot forms a rigid triangle with upper and crank
    hg.add_edge(Edge("upper_foot", "upper", "foot"))
    hg.add_hyperedge(Hyperedge("triangle_foot", nodes=("upper", "crank", "foot")))

    # --- Dimensions ---
    dims = Dimensions(
        node_positions={
            "frame": (0, 0),
            "crank": (0.5, 0),
            "frame2": (-1, -0.3),
            "upper": (0, 1),
            "foot": (0, -1.5),  # approximate, will be solved
        },
        driver_angles={
            "crank": DriverAngle(angular_velocity=pi / 6),  # 30 degrees per step
        },
        edge_distances={
            "frame_crank": 0.5,
            "crank_upper": 1.5,
            "frame2_upper": 2.0,
            "upper_foot": 2.0,
        },
    )

    # Create a Walker
    walker = Walker(hg, dims, name="SimpleWalker")

    # Perform one kinematic step to solve initial joint positions
    list(walker.step(iterations=1))

    # Optionally add opposite leg to create left/right pair
    if opposite:
        walker.add_opposite_leg()

    # Add a second leg offset by 180 degrees for alternating gait
    # add_legs(n) adds n additional legs with phase offsets
    walker.add_legs(1)

    return walker


def main():
    print("Simple Walker Visualization Demo")
    print("=" * 40)
    print("This demonstrates a Klann-style walking mechanism")
    print()
    print("Controls:")
    print("  - Press Q or ESC to quit")
    print("  - Window is resizable")
    print("=" * 40)

    # Create the walker
    walker = create_klann_walker()

    # Run the visualization for 15 seconds with dynamic camera
    ls.video(walker, duration=15, dynamic_camera=True)


if __name__ == "__main__":
    main()
