#!/usr/bin/env python3
"""
Simple walker mechanism visualization demo.

This example creates a basic Klann-style walking linkage with two legs
and simulates it using the Pyglet-based visualizer.

Run with: uv run python examples/simple_walker.py
"""
import numpy as np
from pylinkage import Static, Crank, Fixed, Revolute
import leggedsnake as ls


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
    # Frame/chassis anchor point
    frame = Static(x=0, y=0, name="Frame")

    # Crank - the motor-driven input link
    # angle is the rotation per step (must be non-zero)
    crank = Crank(
        x=0.5, y=0,
        joint0=frame,
        distance=0.5,
        angle=np.pi / 6,  # 30 degrees per step
        name="Crank"
    )

    # Second frame point for the rocker
    frame2 = Static(x=-1, y=-0.3, name="Frame2")

    # Upper linkage - connects crank to rocker
    upper = Revolute(
        x=0, y=1,
        joint0=crank,
        joint1=frame2,
        distance0=1.5,
        distance1=2.0,
        name="Upper"
    )

    # Lower leg segment - the "foot"
    foot = Fixed(
        joint0=upper,
        joint1=crank,
        distance=2.0,
        angle=-2 * np.pi / 3,
        name="Foot"
    )

    # Create a Walker (extends Linkage with leg-specific methods)
    joints = (frame, crank, frame2, upper, foot)
    walker = ls.Walker(
        joints=joints,
        order=joints,
        name="SimpleWalker"
    )

    # Perform one kinematic step to solve initial joint positions
    list(walker.step())

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
