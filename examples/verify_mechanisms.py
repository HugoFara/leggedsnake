#!/usr/bin/env python3
"""
Verification script for leg mechanisms.

This script generates and saves foot trajectory plots for all implemented
leg mechanisms to verify they produce proper walking motion.

Run with: uv run python examples/verify_mechanisms.py
"""
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

from theo_jansen import create_theo_jansen_linkage
from klann_linkage import create_klann_linkage
from chebyshev_linkage import create_chebyshev_linkage


def plot_mechanism_trajectory(ax, walker, name, iterations=48):
    """
    Plot the mechanism and its foot trajectory.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Axes to plot on.
    walker : Walker
        The walking mechanism.
    name : str
        Name of the mechanism.
    iterations : int
        Number of simulation iterations.
    """
    # Get all positions during one full revolution
    loci = list(walker.step(iterations=iterations))

    # Get foot node IDs
    feet = walker.get_feet()
    if not feet:
        ax.set_title(f'{name}\n(ERROR: No feet found)')
        return

    foot_id = feet[0]

    # Get the mechanism to find foot index among joints
    mechanism = walker.to_mechanism()
    joint_ids = [j.id for j in mechanism.joints]
    foot_idx = joint_ids.index(foot_id)

    # Extract foot trajectory
    foot_positions = [pos[foot_idx] for pos in loci]
    xs = [p[0] for p in foot_positions if p[0] is not None]
    ys = [p[1] for p in foot_positions if p[1] is not None]

    if not xs or not ys:
        ax.set_title(f'{name}\n(ERROR: No valid positions)')
        return

    # Plot foot trajectory
    ax.plot(xs, ys, 'b-', linewidth=2, label='Foot trajectory')
    ax.scatter(xs[0], ys[0], color='green', s=100, zorder=5, label='Start')

    # Plot initial mechanism configuration
    initial_pos = loci[0]

    # Plot joints
    for i, pos in enumerate(initial_pos):
        if pos[0] is not None:
            ax.scatter(pos[0], pos[1], color='red', s=50, zorder=4)
            short_name = joint_ids[i].split()[0]
            ax.annotate(short_name, (pos[0], pos[1]),
                       fontsize=8, ha='center', va='bottom')

    # Plot links (edges from topology)
    for edge in walker.topology.edges.values():
        if edge.source in joint_ids and edge.target in joint_ids:
            src_idx = joint_ids.index(edge.source)
            tgt_idx = joint_ids.index(edge.target)
            if (initial_pos[src_idx][0] is not None and
                    initial_pos[tgt_idx][0] is not None):
                ax.plot(
                    [initial_pos[src_idx][0], initial_pos[tgt_idx][0]],
                    [initial_pos[src_idx][1], initial_pos[tgt_idx][1]],
                    'k-', linewidth=1.5, alpha=0.5,
                )

    # Calculate stride metrics
    stride_width = max(xs) - min(xs)
    stride_height = max(ys) - min(ys)

    ax.set_title(f'{name}\n(stride: {stride_width:.2f} x {stride_height:.2f})')
    ax.set_xlabel('X position')
    ax.set_ylabel('Y position')
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)
    ax.legend(loc='upper right', fontsize=8)


def main():
    """Generate verification plots for all mechanisms."""
    print("Generating mechanism verification plots...")

    # Create output directory
    output_dir = Path(__file__).parent / "verification_plots"
    output_dir.mkdir(exist_ok=True)

    mechanisms = [
        ('Theo Jansen (8-bar)', create_theo_jansen_linkage),
        ('Klann (6-bar Stephenson III)', create_klann_linkage),
        ('Chebyshev Lambda (4-bar)', create_chebyshev_linkage),
    ]

    # Create individual plots
    for name, create_func in mechanisms:
        print(f"  Plotting {name}...")
        try:
            fig, ax = plt.subplots(figsize=(10, 8))
            walker = create_func()
            plot_mechanism_trajectory(ax, walker, name)
            filename = output_dir / f"{name.split()[0].lower()}_linkage.png"
            fig.savefig(filename, dpi=150, bbox_inches='tight')
            plt.close(fig)
            print(f"    Saved: {filename}")
        except Exception as e:
            print(f"    Error: {e}")

    # Create comparison plot
    print("  Creating comparison plot...")
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    for ax, (name, create_func) in zip(axes, mechanisms):
        try:
            walker = create_func()
            plot_mechanism_trajectory(ax, walker, name)
        except Exception as e:
            ax.set_title(f'{name}\n(ERROR: {e})')

    fig.suptitle('Comparison of Leg Mechanisms', fontsize=14, fontweight='bold')
    plt.tight_layout()
    comparison_file = output_dir / "comparison_all_mechanisms.png"
    fig.savefig(comparison_file, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"    Saved: {comparison_file}")

    print("\nVerification complete!")
    print(f"Plots saved to: {output_dir}")


if __name__ == "__main__":
    main()
