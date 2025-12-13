#!/usr/bin/env python3
"""
Quick demo of the Pyglet visualization.

This is the simplest possible example - creates a basic mechanism
and runs the visualization immediately.

Run with: uv run python examples/quick_demo.py
"""
import numpy as np
from pylinkage import Static, Crank, Fixed, Revolute, Linkage
import leggedsnake as ls


def main():
    print("Quick Visualization Demo")
    print("Press Q or ESC to quit")
    print("-" * 30)

    # Create a simple crank-rocker mechanism
    base = Static(0, 0, name="Base")
    crank = Crank(1, 0, name="Crank", angle=0.5, distance=1, joint0=base)
    follower = Revolute(
        0, 2,
        joint0=base,
        joint1=crank,
        distance0=2,
        distance1=1.5,
        name="Follower"
    )
    output = Fixed(
        joint0=crank,
        joint1=follower,
        distance=1.5,
        angle=-np.pi/2,
        name="Output"
    )

    linkage = Linkage(
        name='QuickDemo',
        joints=(base, crank, follower, output),
    )

    # Perform one kinematic step to solve initial joint positions
    list(linkage.step())

    # Run for 10 seconds
    ls.video(linkage, duration=10, dynamic_camera=True)


if __name__ == "__main__":
    main()
