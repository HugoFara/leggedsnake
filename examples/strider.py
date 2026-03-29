# -*- coding: utf-8 -*-
"""
Complete simulator for strider mechanism, a type of walking mechanism.

Directions:
    - The first section gives objects definition, and links between them
    - The second section explains how simulation works.
    - Third section is for display.

Created on Sun Dec 23 2018 21:03:11.

@author: HugoFara
"""

import matplotlib.pyplot as plt
import matplotlib.animation as anim
import numpy as np
import leggedsnake as ls
from math import tau
from pylinkage.hypergraph import HypergraphLinkage, Node, Edge, Hyperedge, NodeRole
from pylinkage.dimensions import Dimensions, DriverAngle

# Simulation parameters
# Number of points for crank complete turn
LAP_POINTS = 10
# Time (in seconds) for a crank revolution
LAP_PER_SECOND = 100
# Number of pairs of legs for the dynamic simulation
LEGS_NUMBER = 4

"""
Parameters that can change without changing joints between objects.

Can be distance between joints, or an angle.
Units are given relative to crank length, which is normalized to 1.
"""
DIM_NAMES = (
    "triangle", "ape", "femur", "rockerL", "rockerS", "f", "tibia", "phi"
)

DIMENSIONS = (
    # AB distance (=AB_p) "triangle":
    2,
    # "ape":
    np.pi / 4,
    # femur = 3 for higher steps, 2 for the standard size, but 1.8 is good enough
    1.8,
    # "rockerL":
    2.6,
    # "rockerS":
    1.4,
    # "phi":
    np.pi + .2,
    # "tibia":
    2.5,
    # "f":
    1.8,
)
# Limits for parameters, will be used in optimizers
BOUNDS = (
    (0, 0, 0, 0, 0, 0, 0, 0),
    (8, 2 * np.pi, 7.2, 10.4, 5.6, 2 * np.pi, 10, 7.6)
)

# Initial coordinates according to previous dimensions
INIT_COORD = [
    (0, 0), (0, 1), (1.41, 1.41), (-1.41, 1.41), (0, -1), (-2.25, 0),
    (2.25, 0), (-1.4, -1.2), (1.4, -1.2), (-2.7, -2.7), (2.7, -2.7)
]

# Node names in topology order
NODE_NAMES = ["A", "Y", "B", "B_p", "C", "D", "E", "F", "G", "H", "I"]


def param2dimensions(param=DIMENSIONS):
    """
    Expand compact symmetric parameters to edge distances.

    The strider is symmetric so only 8 parameters are needed
    to define all edge lengths.

    Returns a dict mapping edge IDs to distances.
    """
    return {
        # B is Fixed relative to A and Y: distance + angle
        "A_B": param[0],     # AB distance
        "Y_B": param[0],     # YB distance (same by symmetry)
        # B_p is mirror of B
        "A_B_p": param[0],
        "Y_B_p": param[0],
        # Crank C
        "A_C": 1.0,          # Crank arm length (normalized)
        # D and E: femur + rocker
        "B_p_D": param[2],   # femur
        "C_D": param[3],     # rockerL
        "B_E": param[2],     # femur (symmetric)
        "C_E": param[3],     # rockerL (symmetric)
        # F and G: ankles (Fixed joints)
        "C_F": param[4],     # rockerS
        "E_F": param[4],     # rockerS (symmetric edge in Fixed triangle)
        "C_G": param[4],     # rockerS
        "D_G": param[4],     # rockerS (symmetric)
        # H and I: feet
        "D_H": param[6],     # tibia
        "F_H": param[7],     # f
        "E_I": param[6],     # tibia (symmetric)
        "G_I": param[7],     # f (symmetric)
    }


def _build_strider_topology() -> HypergraphLinkage:
    """Build the strider topology (nodes, edges, hyperedges)."""
    hg = HypergraphLinkage(name="Strider")

    # Ground nodes
    hg.add_node(Node("A", role=NodeRole.GROUND, name="A"))
    hg.add_node(Node("Y", role=NodeRole.GROUND, name="Point (0, 1)"))

    # Frame points (Fixed joints relative to A and Y)
    hg.add_node(Node("B", role=NodeRole.DRIVEN, name="Frame right (B)"))
    hg.add_node(Node("B_p", role=NodeRole.DRIVEN, name="Frame left (B_p)"))

    # Crank (driver)
    hg.add_node(Node("C", role=NodeRole.DRIVER, name="Crank link (C)"))

    # Knee links
    hg.add_node(Node("D", role=NodeRole.DRIVEN, name="Left knee link (D)"))
    hg.add_node(Node("E", role=NodeRole.DRIVEN, name="Right knee link (E)"))

    # Ankle links (Fixed joints)
    hg.add_node(Node("F", role=NodeRole.DRIVEN, name="Left ankle link (F)"))
    hg.add_node(Node("G", role=NodeRole.DRIVEN, name="Right ankle link (G)"))

    # Feet
    hg.add_node(Node("H", role=NodeRole.DRIVEN, name="Left foot (H)"))
    hg.add_node(Node("I", role=NodeRole.DRIVEN, name="Right foot (I)"))

    # Edges
    # Frame: B is Fixed on A-Y, B_p is Fixed on A-Y
    hg.add_edge(Edge("A_B", "A", "B"))
    hg.add_edge(Edge("Y_B", "Y", "B"))
    hg.add_edge(Edge("A_B_p", "A", "B_p"))
    hg.add_edge(Edge("Y_B_p", "Y", "B_p"))

    # Crank
    hg.add_edge(Edge("A_C", "A", "C"))

    # Knees (Revolute: two parent edges)
    hg.add_edge(Edge("B_p_D", "B_p", "D"))
    hg.add_edge(Edge("C_D", "C", "D"))
    hg.add_edge(Edge("B_E", "B", "E"))
    hg.add_edge(Edge("C_E", "C", "E"))

    # Ankles: F is Fixed on C-E, G is Fixed on C-D
    hg.add_edge(Edge("C_F", "C", "F"))
    hg.add_edge(Edge("E_F", "E", "F"))
    hg.add_edge(Edge("C_G", "C", "G"))
    hg.add_edge(Edge("D_G", "D", "G"))

    # Feet (Revolute)
    hg.add_edge(Edge("D_H", "D", "H"))
    hg.add_edge(Edge("F_H", "F", "H"))
    hg.add_edge(Edge("E_I", "E", "I"))
    hg.add_edge(Edge("G_I", "G", "I"))

    # Hyperedges for rigid triangles (Fixed joints)
    # B is rigid on A-Y frame
    hg.add_hyperedge(Hyperedge("triangle_B", nodes=("A", "Y", "B")))
    # B_p is rigid on A-Y frame
    hg.add_hyperedge(Hyperedge("triangle_B_p", nodes=("A", "Y", "B_p")))
    # F is Fixed on C-E
    hg.add_hyperedge(Hyperedge("triangle_F", nodes=("C", "E", "F")))
    # G is Fixed on C-D
    hg.add_hyperedge(Hyperedge("triangle_G", nodes=("C", "D", "G")))

    return hg


def complete_strider(param=DIMENSIONS, prev=INIT_COORD):
    """
    Build a complete strider Walker from parameters and initial positions.

    Parameters
    ----------
    param : tuple[float]
        The 8 compact parameters (expanded via param2dimensions).
    prev : list[tuple[float, float]]
        Initial coordinates for each node.
    """
    hg = _build_strider_topology()
    edge_dists = param2dimensions(param)

    # Build node positions from prev coordinates
    node_positions = {}
    for name, coord in zip(NODE_NAMES, prev):
        node_positions[name] = coord

    dims = Dimensions(
        node_positions=node_positions,
        driver_angles={"C": DriverAngle(angular_velocity=-tau / LAP_POINTS)},
        edge_distances=edge_dists,
    )

    return ls.Walker(hg, dims, name="Strider")


def strider_builder(param=DIMENSIONS, prev=INIT_COORD, n_leg_pairs=1, minimal=False):
    """
    Quickly build a strider with various parameters.

    Parameters
    ----------
    param : tuple[float]
        The 8 compact parameters.
    prev : list[tuple[float, float]]
        Initial coordinates.
    n_leg_pairs : int, optional
        Number of leg pairs. The default is 1.
    minimal : bool, optional
        Minimal representation (one foot only). The default is False.
    """
    if minimal:
        # Build minimal topology: remove G and I nodes
        hg = _build_strider_topology()
        # Remove right ankle and right foot
        hg.remove_node("G")
        hg.remove_node("I")

        edge_dists = param2dimensions(param)
        # Remove edges related to G and I
        for eid in ["C_G", "D_G", "E_I", "G_I"]:
            if eid in edge_dists:
                del edge_dists[eid]

        minimal_names = [n for n in NODE_NAMES if n not in ("G", "I")]
        node_positions = {}
        prev_list = list(prev)
        prev_filtered = [prev_list[NODE_NAMES.index(n)] for n in minimal_names]
        for name, coord in zip(minimal_names, prev_filtered):
            node_positions[name] = coord

        dims = Dimensions(
            node_positions=node_positions,
            driver_angles={"C": DriverAngle(angular_velocity=-tau / LAP_POINTS)},
            edge_distances=edge_dists,
        )
        strider = ls.Walker(hg, dims, name="Strider")
    else:
        strider = complete_strider(param, prev)

    if n_leg_pairs > 1:
        strider.add_legs(n_leg_pairs - 1)
    return strider


def show_all_walkers(dnas, duration=40, save=False):
    """Show multiple walkers racing."""
    linkages = []
    for dna in dnas:
        dummy_strider = complete_strider(DIMENSIONS, INIT_COORD)
        dummy_strider.add_legs(LEGS_NUMBER - 1)
        dummy_strider.set_num_constraints(dna[1])
        dummy_strider.set_coords(dna[2])
        linkages.append(dummy_strider)
    ls.all_linkages_video(
        linkages, duration, save,
        np.random.rand(len(dnas), 3)
    )


def show_physics(linkage, prev=None, debug=False, duration=40, save=False):
    """Give mechanism a dynamic model and launch video."""
    if prev is not None:
        linkage.set_coords(prev)
    try:
        tuple(linkage.step())
    except ls.UnbuildableError:
        print("Warning: mechanism is unbuildable at given positions")
        return

    if debug:
        ls.video_debug(linkage)
    else:
        ls.video(linkage, duration, save)


# Ugly way to save (position + cost) history
history = []


def sym_stride_evaluator(linkage, dims, pos):
    """
    Score each dimension set for symmetric strider.

    Uses compact 8-parameter form, expands to full edge distances.
    """
    expanded = param2dimensions(dims)
    # Update edge distances in the walker's dimensions
    for eid, dist in expanded.items():
        if eid in linkage.dimensions.edge_distances:
            linkage.dimensions.edge_distances[eid] = dist
    linkage._invalidate_cache()
    linkage.set_coords(pos)

    points = 12
    try:
        loci = tuple(
            tuple(i) for i in linkage.step(
                iterations=points, dt=LAP_POINTS / points
            )
        )
    except ls.UnbuildableError:
        return 0

    history.append(list(dims) + [0])
    foot_locus = tuple(x[-2] for x in loci)
    if not ls.step(foot_locus, .5, .2):
        return 0
    locus = ls.stride(foot_locus, .2)
    score = max(k[0] for k in locus) - min(k[0] for k in locus)
    history[-1][-1] = score
    return score


def repr_polar_swarm(current_swarm, fig=None, lines=None, t=0):
    """Represent a swarm in a polar graph."""
    best_cost = max(x[-1] for x in current_swarm)
    fig.suptitle(f"Best cost: {best_cost}")
    for line, dimension_set in zip(lines, current_swarm):
        line.set_data(t, dimension_set)
    return lines


def swarm_optimizer(
        linkage, dims=DIMENSIONS, show=False, save_each=0, age=300,
        iters=400, *args
):
    """Optimize linkage geometrically using PSO."""
    print("Initial dimensions:", dims)

    if show == 1:
        out = ls.particle_swarm_optimization(
            sym_stride_evaluator, linkage,
            center=dims, n_particles=age, iters=iters,
            bounds=BOUNDS, dimensions=len(dims), *args,
        )

        fig = plt.figure("Swarm in polar graph")
        ax = fig.add_subplot(111, projection='polar')
        lines = [ax.plot([], [], lw=.5, animated=False)[0] for i in range(age)]
        t = np.linspace(0, 2 * np.pi, len(dims) + 2)[:-1]
        ax.set_xticks(t)
        ax.set_rmax(7)
        ax.set_xticklabels(DIM_NAMES + ("score",))
        formatted_history = [
            history[i:i + age] for i in range(0, len(history), age)
        ]
        animation = anim.FuncAnimation(
            fig,
            func=repr_polar_swarm,
            frames=formatted_history,
            fargs=(fig, lines, t), blit=True,
            interval=10, repeat=True,
            save_count=(iters - 1) * bool(save_each)
        )
        plt.show()
        if save_each:
            writer = anim.FFMpegWriter(
                fps=24, bitrate=1800,
                metadata={
                    'title': "Particle swarm looking for R^8 in R "
                    "application maximum",
                    'comment': "Made with Python and Matplotlib",
                    'description': "The swarm tries to find best dimension"
                    " set for Strider legged mechanism"
                }
            )
            animation.save(r"PSO.mp4", writer=writer)
        if animation:
            pass
        return out
    elif show == 2:
        out = ls.particle_swarm_optimization(
            sym_stride_evaluator, linkage,
            center=dims, n_particles=age, iters=iters,
            bounds=BOUNDS, dimensions=len(dims),
            *args
        )

        fig = plt.figure("Swarm in tiled mode")
        cells = int(np.ceil(np.sqrt(age)))
        axes = fig.subplots(cells, cells)
        lines = [ax.plot([], [], lw=.5, animated=False)[0]
                 for ax in axes.flatten()]
        formatted_history = [
            history[i:i + age][:-1] for i in range(0, len(history), age)
        ]
        animation = anim.FuncAnimation(
            fig, lambda *args: ls.swarm_tiled_repr(linkage, *args),
            formatted_history, fargs=(fig, axes, param2dimensions), blit=False,
            interval=40, repeat=False, save_count=(iters - 1) * bool(save_each)
        )
        plt.show(block=not save_each)
        if save_each:
            writer = anim.FFMpegWriter(
                fps=24, bitrate=1800,
                metadata={
                    'title': "Particle swarm looking for R^8 in R "
                    "application maximum",
                    'comment': "Made with Python and Matplotlib",
                    'description': "The swarm looks for best dimension "
                    "set for Strider legged mechanism"
                }
            )
            animation.save("Particle swarm optimization.mp4", writer=writer)
        if animation:
            pass
        return out
    else:
        out = tuple(
            ls.particle_swarm_optimization(
                sym_stride_evaluator,
                linkage,
                dims,
                n_particles=age,
                bounds=BOUNDS,
                dimensions=len(dims),
                iters=iters,
                *args
            )
        )
        return out


def dna_interpreter(dna):
    """Reconstruct a strider Walker from DNA."""
    linkage = complete_strider(DIMENSIONS, INIT_COORD)
    linkage.add_legs(LEGS_NUMBER - 1)
    linkage.set_num_constraints(dna[1])
    linkage.set_coords(dna[2])
    return linkage


def move_linkage(linkage):
    """Make the linkage do a movement; return positions or False if impossible."""
    try:
        pos = tuple(linkage.step(iterations=LAP_POINTS))[-1]
        return pos
    except ls.UnbuildableError:
        return False


def total_distance(dna):
    """
    Evaluate the final horizontal position of the input linkage.

    Parameters
    ----------
    dna : list of 3 elements
        [score, dimensions, initial_positions]

    Returns
    -------
    tuple
        (distance, positions)
    """
    linkage = dna_interpreter(dna)
    pos = move_linkage(linkage)
    if not pos:
        return -2, list()
    world = ls.World()
    world.add_linkage(linkage)
    duration = 40
    steps = int(duration / world.config.physics_period)
    for _ in range(steps):
        world.update()
    return world.linkages[0].body.position.x, pos


def efficiency(dna):
    """Individual yield: return average efficiency and initial coordinates."""
    linkage = dna_interpreter(dna)
    pos = move_linkage(linkage)
    if not pos:
        return -2, list()
    world = ls.World()
    world.add_linkage(linkage)
    duration = 40
    tot = 0
    dur = 0
    steps = int(duration / world.config.physics_period)
    for _ in range(steps):
        eff, energy = world.update()
        tot += eff
        dur += energy
    if dur == 0:
        return -1, list()
    if world.linkages[0].body.position.x < 5:
        return 0, pos
    return tot / dur, pos


def evolutive_optimizer(
        linkage, dims=DIMENSIONS, prev=None, pop=10, iters=10,
        startnstop=False, gui=None
):
    """Optimization of the linkage by genetic algorithm."""
    if prev is not None:
        linkage.set_coords(prev)
    tuple(linkage.step())

    fitness_function = total_distance
    dna = [0, list(dims), list(linkage.get_coords())]
    dna[0] = fitness_function(dna)
    optimizer = ls.GeneticOptimization(
        dna=dna,
        prob=.07,
        fitness=fitness_function,
        max_pop=pop,
        startnstop=startnstop,
        gui=gui
    )
    return optimizer.run(iters, processes=4)


def chained_optimizer(linkage, dims=DIMENSIONS, prev=None):
    """
    Two-stage optimization: PSO for kinematic exploration, then DE to refine.
    """
    if prev is not None:
        linkage.set_coords(prev)

    return ls.chain_optimizers(
        eval_func=sym_stride_evaluator,
        linkage=linkage,
        stages=[
            (ls.particle_swarm_optimization, {
                "center": dims,
                "n_particles": 40,
                "iters": 40,
                "bounds": BOUNDS,
                "dimensions": len(dims),
            }),
            (ls.differential_evolution_optimization, {
                "bounds": BOUNDS,
                "maxiter": 100,
                "popsize": 15,
            }),
        ],
    )


def show_optimized(linkage, data, n_show=10, duration=5, symmetric=True):
    """Show the optimized linkages."""
    for datum in data[:min(len(data), n_show)]:
        if datum[0] <= 0:
            continue
        if symmetric:
            expanded = param2dimensions(datum[1])
            for eid, dist in expanded.items():
                if eid in linkage.dimensions.edge_distances:
                    linkage.dimensions.edge_distances[eid] = dist
            linkage._invalidate_cache()
        else:
            linkage.set_num_constraints(datum[1])
        print(f"Score: {datum[0]}")


def main(trials_and_errors, particle_swarm, genetic, chained=False):
    strider = complete_strider(DIMENSIONS, INIT_COORD)
    print(
        "Initial striding score:",
        sym_stride_evaluator(strider, DIMENSIONS, INIT_COORD)
    )
    if trials_and_errors:
        optimized_striders = ls.trials_and_errors_optimization(
            sym_stride_evaluator, strider, DIMENSIONS, divisions=4, verbose=True
        )
        print(
            "Striding score after trials and errors optimization:",
            optimized_striders[0].score
        )

    if particle_swarm:
        optimized_striders = swarm_optimizer(
            strider, show=1, save_each=0, age=40, iters=40, bounds=BOUNDS,
        )
        print(
            "Striding score after particle swarm optimization:",
            optimized_striders[0].score
        )

    if chained:
        results = chained_optimizer(strider, DIMENSIONS, INIT_COORD)
        best = results[0]
        print(
            "Striding score after chained PSO+DE optimization:",
            best.score
        )

    if genetic:
        strider.add_legs(LEGS_NUMBER - 1)
        init_coords = strider.get_coords()
        show_physics(strider, save=False)
        print(
            "Distance ran score before genetic optimization",
            total_distance([0, strider.get_num_constraints(), strider.get_coords()])[0]
        )
        file = "Population evolution.json"
        file = False
        optimized_striders = evolutive_optimizer(
            strider,
            dims=strider.get_num_constraints(),
            prev=init_coords,
            pop=10,
            iters=30,
            startnstop=file,
        )
        print(
            "Distance ran score after genetic optimization:",
            optimized_striders[0].score
        )
        strider = dna_interpreter(optimized_striders[0])
        input("Press enter to show result ")
        show_physics(strider, save=False)
        show_all_walkers(optimized_striders, save=False)
        if file:
            data = ls.load_data(file)
            ls.show_genetic_optimization(data)


if __name__ == "__main__":
    main(trials_and_errors=False, particle_swarm=False, genetic=True)
