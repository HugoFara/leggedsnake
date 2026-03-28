#!/usr/bin/env python3
"""
Compare multiple linkages side by side.

This example creates several linkages with different parameters
and visualizes them together with varying opacities.

Run with: uv run python examples/compare_linkages.py
"""
import numpy as np
from pylinkage import Static, Crank, Fixed, Revolute, Linkage
import leggedsnake as ls


def create_linkage(crank_length: float, coupler_length: float, name: str):
    """Create a four-bar linkage with specified dimensions."""
    ground = Static(x=0, y=0, name=f"{name}_Ground")

    crank = Crank(
        x=crank_length, y=0,
        joint0=ground,
        distance=crank_length,
        angle=np.pi / 12,  # 15 degrees per step
        name=f"{name}_Crank"
    )

    rocker_ground = Static(x=3, y=0, name=f"{name}_RockerGround")

    coupler = Revolute(
        x=crank_length, y=coupler_length,
        joint0=crank,
        joint1=rocker_ground,
        distance0=coupler_length,
        distance1=2.5,
        name=f"{name}_Coupler"
    )

    output = Fixed(
        joint0=crank,
        joint1=coupler,
        distance=1.2,
        angle=-np.pi/4,
        name=f"{name}_Output"
    )

    linkage = Linkage(
        joints=(ground, crank, rocker_ground, coupler, output),
        name=name
    )

    # Perform one kinematic step to solve initial joint positions
    list(linkage.step())

    return linkage


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
