#!/usr/bin/env python3
"""
Simple four-bar linkage visualization demo.

This example creates a basic four-bar linkage and simulates it
using the Pyglet-based visualizer.

Run with: uv run python examples/simple_fourbar.py
"""
import numpy as np
from pylinkage import Static, Crank, Fixed, Revolute, Linkage
import leggedsnake as ls


def create_fourbar_linkage():
    """Create a simple four-bar linkage."""
    # Ground joint (fixed point)
    ground = Static(x=0, y=0, name="Ground")

    # Crank - the driving link that rotates
    # angle is the rotation per kinematic step
    crank = Crank(
        x=1, y=0,
        joint0=ground,
        distance=1,
        angle=np.pi / 12,  # 15 degrees per step
        name="Crank"
    )

    # Rocker pivot point (second ground point)
    rocker_ground = Static(x=3, y=0, name="RockerGround")

    # Coupler - connects crank to rocker
    coupler = Revolute(
        x=1.5, y=1.5,
        joint0=crank,
        joint1=rocker_ground,
        distance0=2,      # Distance from crank tip
        distance1=2.5,    # Distance from rocker ground
        name="Coupler"
    )

    # Output link (rocker)
    output = Fixed(
        joint0=crank,
        joint1=coupler,
        distance=1.5,
        angle=-np.pi/3,
        name="Output"
    )

    linkage = Linkage(
        joints=(ground, crank, rocker_ground, coupler, output),
        name="FourBar"
    )

    # Perform one kinematic step to solve initial joint positions
    # This is required before visualization
    list(linkage.step())

    return linkage


def main():
    print("Four-Bar Linkage Visualization Demo")
    print("=" * 40)
    print("Controls:")
    print("  - Press Q or ESC to quit")
    print("  - Window is resizable")
    print("=" * 40)

    # Create the linkage
    linkage = create_fourbar_linkage()

    # Run the visualization for 10 seconds with dynamic camera (follows the linkage)
    ls.video(linkage, duration=10, dynamic_camera=True)


if __name__ == "__main__":
    main()
