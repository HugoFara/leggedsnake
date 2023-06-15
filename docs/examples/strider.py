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

# Simulation parameters
# Number of points for crank complete turn
n = 10
# Time (in seconds) for a crank revolution
speed = 100

"""
Parameters that can change without changing joints between objects.

Can be distance between joints, or an angle.
Units are given relative to crank length, which is normalized to 1.
"""
param_names = (
    "triangle", "ape", "femur", "rockerL", "rockerS", "f", "tibia", "phi"
)

param = (
    # AB distance (=AB_p) "triangle":
    2,
    # "ape":
    np.pi / 4,
    # femur = 3 for higher steps, 2 for the standard size but 1.8 is good enough
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
# Optimized but useless strider with a step of size 5.05
# param = (2.62484195, 1.8450077, 2.41535873, 2.83669735, 2.75235715,
#         4.60386788, 3.49814371, 3.51517851)
# Limits for parameters, will be used in optimizers
bounds = (
    (0, 0, 0, 0, 0, 0, 0, 0),
    (8, 2 * np.pi, 7.2, 10.4, 5.6, 2 * np.pi, 10, 7.6)
)

# Initial coordinates according to previous dimensions
begin = (
    (0, 0), (0, 1), (1.41, 1.41), (-1.41, 1.41), (0, -1), (-2.25, 0),
    (2.25, 0), (-1.4, -1.2), (1.4, -1.2), (-2.7, -2.7), (2.7, -2.7)
)


def param2dimensions(param=param, flat=False):
    """
    Parameters are written in short form due to symmetry.

    This function expands them to fit in strider.set_num_constraints.
    """
    out = (
        # Static joints (A and Y)
        (), (),
        # B, B_p
        (param[0], -param[1]), (param[0], param[1]),
        # Crank (C)
        (1, ),
        # D and E
        (param[2], param[3]), (param[2], param[3]),
        # F and G
        (param[4], -param[5]), (param[4], param[5]),
        # H and I
        (param[6], param[7]), (param[6], param[7])
    )
    if not flat:
        return out
    flat_dims = []
    for constraint in out:
        if constraint == ():
            flat_dims.append(0)
        else:
            flat_dims.extend(constraint)
    return tuple(flat_dims)


def complete_strider(constraints, prev):
    """
    Take two sequences to define strider linkage.

    Parameters
    ----------
    constraints : Union[tuple[float], tuple[tuple[float]]]
        The sequence of geometrical constraints
    prev : tuple[tuple[float]]
        Coordinates to set by default.
    """
    linka = {
        # Fixed points (mechanism body)
        # A is the origin
        "A": ls.Static(x=0, y=0, name="A"),
        # Vertical axis for convenience
        "Y": ls.Static(0, 1, name="Point (0, 1)"),
    }
    # For drawing only
    linka["Y"].joint0 = linka["A"]
    linka.update({
        # Not fixed because we will optimize this position
        "B": ls.Fixed(joint0=linka["A"], joint1=linka["Y"], name="Frame right (B)"),
        "B_p": ls.Fixed(joint0=linka["A"], joint1=linka["Y"], name="Frame left (B_p)"),
        # Pivot joints, explicitly defined to be modified later
        # Joint linked to crank. Coordinates are chosen in each frame
        "C": ls.Crank(joint0=linka["A"], angle=2 * np.pi / n, name="Crank link (C)")
    })
    linka.update({
        "D": ls.Pivot(joint0=linka["B_p"], joint1=linka["C"], name="Left knee link (D)"),
        "E": ls.Pivot(joint0=linka["B"], joint1=linka["C"], name="Right knee link (E)")
    })
    linka.update({
        # F is fixed relative to C and E
        "F": ls.Fixed(joint0=linka["C"], joint1=linka["E"], name='Left ankle link (F)'),
        # G fixed to C and D
        "G": ls.Fixed(joint0=linka["C"], joint1=linka["D"], name='Right ankle link (G)')
    })
    linka.update({
        "H": ls.Pivot(joint0=linka["D"], joint1=linka["F"], name="Left foot (H)"),
        "I": ls.Pivot(joint0=linka["E"], joint1=linka["G"], name="Right foot (I)")
    })
    # Mechanism definition
    strider = ls.Walker(
        joints=linka.values(),
        order=linka.values(),
        name="Strider"
    )
    strider.set_coords(prev)
    strider.set_num_constraints(constraints, flat=False)
    return strider


def strider_builder(constraints, prev, n_leg_pairs=1, minimal=False):
    """
    Quickly build a strider with various parameters.

    Parameters
    ----------
    constraints : iterable of 2-tuple
        Iterable of all the constraints to set.
    prev : tuple of 2-tuples
        Initial coordinates.
    n_leg_pairs : int, optional
        The number of leg pairs that the strider should have. The default is 1.
    minimal : bool, optional
        Minimal representation is with one foot only. The default is False.

    Returns
    -------
    strider : leggedssnake.walker.Walker
        The requested strider linkage.
    """
    # Fixed points (mechanism body)
    # A is the origin
    A = ls.Static(x=0, y=0, name="A")
    # Vertical axis for convenience,
    Y = ls.Static(0, 1, name="Point (0, 1)")
    # For drawing only
    Y.joint0 = A
    # Not fixed because we will optimize this position
    B = ls.Fixed(joint0=A, joint1=Y, name="Frame right (B)")
    B_p = ls.Fixed(joint0=A, joint1=Y, name="Frame left (B_p)")
    # Pivot joints, explicitly defined to be modified later
    # Joint linked to crank. Coordinates are chosen in each frame
    C = ls.Crank(joint0=A, angle=2 * np.pi / n, name="Crank link (C)")
    D = ls.Pivot(joint0=B_p, joint1=C, name="Left knee link (D)")
    E = ls.Pivot(joint0=B, joint1=C, name="Right knee link (E)")
    # F is fixed relative to C and E
    F = ls.Fixed(joint0=C, joint1=E, name='Left ankle link (F)')
    H = ls.Pivot(joint0=D, joint1=F, name="Left foot (H)")
    joints = [A, Y, B, B_p, C, D, E, F, H]
    if not minimal:
        # G fixed to C and D
        G = ls.Fixed(joint0=C, joint1=D, name='Right ankle link (G)')
        joints.insert(-1, G)
        joints.append(ls.Pivot(joint0=E, joint1=G, name="Right foot (I)"))
    # Mechanism definition
    strider = ls.Walker(
        joints=joints,
        order=joints,
        name="Strider"
    )
    if minimal and len(prev) > len(joints):
        prev = list(prev)
        constraints = list(constraints)
        # Joint G
        prev.pop(-3)
        constraints.pop(-3)
        # Joint I
        prev.pop(-1)
        constraints.pop(-1)
    strider.set_coords(prev)
    strider.set_num_constraints(constraints, flat=False)
    if n_leg_pairs > 1:
        strider.add_legs(n_leg_pairs - 1)
    return strider


def show_physics(linkage, prev=None, debug=False, duration=40, save=False):
    """
    Give mechanism a dynamic model and launch video.

    Parameters
    ----------
    linkage : leggedsnake.walker.Walker
        Linkage to simulate.
    prev : tuple[tuple[float]], optional
        Previous coordinates to use. The default is None.
    debug : bool, optional
        Launch in debug mode (frame by frame, with forces visualization).
        The default is False.
    duration : float, optional
        Simulation duration (in seconds). The default is 40.
    save : bool, optional
        Save the video file instead of displaying it. The default is False.
    """
    # Define initial positions
    linkage.rebuild(prev)
    if debug:
        ls.video_debug(linkage)
    else:
        ls.video(linkage, duration, save)


# Ugly way to save (position + cost) history
history = []


def sym_stride_evaluator(linkage, dims, pos):
    """Give score to each dimension set for symmetric strider."""
    linkage.set_num_constraints(param2dimensions(dims), flat=False)
    linkage.set_coords(pos)
    try:
        points = 12
        # Complete revolution with 12 points
        tuple(
            tuple(i) for i in linkage.step(
                iterations=points + 1, dt=n / points
            )
        )
        # Again with n points, and at least 12 iterations
        # L = tuple(tuple(i) for i in linkage.step(iterations=n))
        factor = int(points / n) + 1
        loci = tuple(
            tuple(i) for i in linkage.step(
                iterations=n * factor, dt=n / n / factor
            )
        )
        history.append(list(dims) + [0])
    except ls.UnbuildableError:
        return 0
    else:
        foot_locus = tuple(x[-2] for x in loci)
        # Constraints check
        if not ls.step(foot_locus, .5, .2):
            return 0
        # Performances evaluation
        locus = ls.stride(foot_locus, .2)
        score = max(k[0] for k in locus) - min(k[0] for k in locus)
        history[-1][-1] = score
        return score


def repr_polar_swarm(current_swarm, fig=None, lines=None, t=0):
    """
    Represent a swarm in a polar graph.

    Parameters
    ----------
    current_swarm : list[list[float]]
        List of dimensions + cost (concatenated).
    fig : matplotlib.pyplot.Figure, optional
        Figure to draw on. The default is None.
    lines : list[matplotlib.pyplot.Artist], optional
        Lines to be modified. The default is None.
    t : int, optional
        Frame index. The default is 0.

    Returns
    -------
    lines : list[matplotlib.pyplot.Artist]
        Lines with coordinates modified.

    """
    best_cost = max(x[-1] for x in current_swarm)
    fig.suptitle("Best cost: {}".format(best_cost))
    for line, dimension_set in zip(lines, current_swarm):
        line.set_data(t, dimension_set)
    return lines


def swarm_optimizer(linkage, dims=param, show=False, save_each=0, age=300,
                    iters=400, *args, **kwargs):
    """
    Optimize linkage geometrically using PSO.

    Parameters
    ----------
    linkage : pylinkage.linkage.Linkage
        The linkage to optimize.
    dims : list[float], optional
        The dimensions that should vary. The default is param.
    show : int, optional
        Type of visualization.
        - 0 for None
        - 1 for polar graph
        - 2 for tiled 2D representation
        The default is False.
    save_each : int, optional
        If show is 0, save the image each {save_each} frame. The default is 0.
    age : int, optional
        Number of agents to simulate. The default is 300.
    iters : int, optional
        Number of iterations to run through. The default is 400.
    blind_ite : int, optional
        Number of iterations without evaluation. The default is 200.
    *args : list
        Arguments to pass to the particle swarm optimization.
    **kwargs : dict
        DESCRIPTION.

    Returns
    -------
    list
        List of best fit linkages.

    """
    print("Initial dimensions:", dims)

    if show == 1:
        out = ls.particle_swarm_optimization(
            sym_stride_evaluator, linkage,
            center=dims, n_particles=age, iters=iters,
            bounds=bounds, dimensions=len(dims), *args,
        )

        fig = plt.figure("Swarm in polar graph")
        ax = fig.add_subplot(111, projection='polar')
        lines = [ax.plot([], [], lw=.5, animated=False)[0] for i in range(age)]
        t = np.linspace(0, 2 * np.pi, len(dims) + 2)[:-1]
        ax.set_xticks(t)
        ax.set_rmax(7)
        ax.set_xticklabels(param_names + ("score",))
        formatted_history = [
            history[i:i + age] for i in range(0, len(history), age)
        ]
        animation = anim.FuncAnimation(
            fig,
            func=repr_polar_swarm,
            frames=formatted_history,
            fargs=(fig, lines, t), blit=True,
            interval=10, repeat=True,
            save_count=(iters - 1) * bool(save_each))
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
        # Tiled representation of swarm
        out = ls.particle_swarm_optimization(
            sym_stride_evaluator, linkage,
            center=dims, n_particles=age, iters=iters,
            bounds=bounds, dimensions=len(dims),
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
        # Don't let the animation be garbage-collected!
        if animation:
            pass
        return out

    elif save_each:
        for dim, i in ls.particle_swarm_optimization(
                sym_stride_evaluator,
                linkage,
                dims,
                age,
                iters=iters,
                bounds=bounds,
                dimensions=len(dims),
                # *args
        ):
            if not i % save_each:
                f = open('PSO optimizer.txt', 'w')
                # We only keep the best results
                dim.sort(key=lambda x: x[1], reverse=True)
                for j in range(min(10, len(dim))):
                    par = {}
                    for k in range(len(dim[j][0])):
                        par[param_names[k]] = dim[j][0][k]
                    f.write('{}\n{}\n{}\n'.format(par, dim[j][1], dim[j][2]))
                    f.write('----\n')
                f.close()
    else:
        out = tuple(
            ls.particle_swarm_optimization(
                sym_stride_evaluator,
                linkage,
                dims,
                n_particles=age,
                bounds=bounds,
                dimensions=len(dims),
                iters=iters,
                *args
            )
        )
        return out


def fitness(dna, linkage_hollow, gui=False):
    """
    Individual yield, return average efficiency and initial coordinates.

    Parameters
    ----------
    dna : list of 3 elements
        The first element is dimensions. The second element is score (unused).
        The third element is initial positions.
    linkage_hollow : Linkage
        A which will integrate this DNA (avoid redefining a new linkage).

    Returns
    -------
    list
        List of two elements: score (a float), and initial positions.
        Score is -float('inf') when mechanism building is impossible.
    """
    linkage_hollow.set_num_constraints(dna[1])
    linkage_hollow.rebuild(dna[2])
    # Check if the mechanism is buildable
    try:
        # Save initial coordinates
        pos = tuple(linkage_hollow.step())[-1]
    except ls.UnbuildableError:
        return -2, list()
    world = ls.World()
    world.add_linkage(linkage_hollow)
    # Simulation duration (in seconds)
    duration = 30
    # Somme of yields
    tot = 0
    # Motor turned on duration
    dur = 0
    steps = int(duration / ls.params["simul"]["physics_period"])
    for _ in range(steps):
        efficiency, energy = world.update()
        tot += efficiency
        dur += energy
    if dur == 0:
        return -1, list()
    if world.linkages[0].body.position.x > -5:
        return 0, pos
    if gui:
        ls.video(linkage_hollow, duration)
    # Return 100 times average yield, and initial positions
    return tot / dur, pos


def evolutive_optimizer(
        linkage, 
        dims=param, 
        prev=None, 
        pop=10, 
        iters=10,
        init_pop=None, 
        startnstop=False,
        gui=False
    ):
    """
    Optimization of the linkage by genetic algorithm.

    Parameters
    ----------
    linkage : Linkage
        Linkage to optimize.
    dims : sequence of floats
        Initial dimensions to use.
    prev : tuple of 2-tuples of float, optional
        Initial positions. The default is None.
    pop : int, optional
        Number of individuals. The default is 10.
    iters : int, optional
        Number of iterations to perform. The default is 10.
    init_pop : int, optional
        Initial population for the highest initial genetic diversity.
        The default is None.
    startnstop : bool, optional
        To save results to a file regularly, and fetch initial data from this file. The default is False.

    Returns
    -------
    list
        List of optimized linkages with dimensions, score and initial
        positions.

    """
    linkage.rebuild(prev)
    linkage.step()
    dna = 0, list(dims), list(linkage.get_coords())
    optimizer = ls.genetic_optimization(
        dna=dna, 
        prob=.07,
        fitness=fitness,
        iters=iters,
        max_pop=pop,
        init_pop=init_pop,
        startnstop=startnstop,
        fitness_args=(linkage, gui),
        processes=4
    )
    return optimizer.run(iters)


def show_optimized(linkage, data, n_show=10, duration=5, symmetric=True):
    """Show the optimized linkages."""
    for datum in data[:min(len(data), n_show)]:
        if datum[0] <= 0:
            continue
        if symmetric:
            linkage.set_num_constraints(param2dimensions(datum[1]), flat=False)
        else:
            linkage.set_num_constraints(datum[1], flat=False)
        ls.show_linkage(
            linkage, prev=begin, title=str(datum[0]), duration=duration
        )


def main(trials_and_errors, particle_swarm, genetic):
    """
    Optimize a strider with different settings.

    :param trials_and_errors: True to use trial and errors optimization.
    :type trials_and_errors: bool
    :param particle_swarm: True to use a particle swarm optimization
    :type particle_swarm: bool
    """
    strider = complete_strider(param2dimensions(param), begin)
    print(
        "Initial score:",
        sym_stride_evaluator(strider, param, begin)
    )
    if trials_and_errors:
        # Trials and errors optimization as comparison
        optimized_striders = ls.trials_and_errors_optimization(
            sym_stride_evaluator, strider, param, divisions=4, verbose=True
        )
        print(
            "Score after trials and errors optimization:",
            optimized_striders[0][0]
        )

    # Particle swarm optimization
    if particle_swarm:
        optimized_striders = swarm_optimizer(
            strider, show=1, save_each=0, age=40, iters=40, bounds=bounds,
        )
        print(
            "Score after particle swarm optimization:",
            optimized_striders[0][0]
        )

    if genetic:
        # ls.show_linkage(strider, save=False, duration=10, iteration_factor=n)
        # Add legs more legs to avoid falling
        strider.add_legs(3)
        init_coords = strider.get_coords()
        show_physics(strider, debug=False, duration=40, save=False)
        # Reload the position: the show_optimized
        optimized_striders = evolutive_optimizer(
            strider,
            dims=strider.get_num_constraints(),
            prev=init_coords,
            pop=30,
            iters=30,
            startnstop=False,
            gui=False
        )
        print(
            "Fitness after genetic optimization:", 
            optimized_striders[0][0]
        )
        strider.set_coords(optimized_striders[0][2])
        strider.set_num_constraints(optimized_striders[0][1], flat=False)
        show_physics(strider, debug=False, duration=40, save=False)


# The file will be imported as a module if using multiprocessing
if __name__ == "__main__":
    main(trials_and_errors=False, particle_swarm=False, genetic=True)
