# -*- coding: utf-8 -*-
"""
Complete simulator for strider mechanism, a type of walking mechanism.

Directions:
    - First section gives objects definition, and links between them
    - Second section explains how simulation works.
    - Third section is for display.

Created on Sun Dec 23 21:03:11 2018.

@author: HugoFara
"""

import matplotlib.pyplot as plt
import matplotlib.animation as anim
import numpy as np
import leggedsnake as ls

# Simulation parameters
# Nunber of points for crank complet turn
n = 10
# Time (in seconds) for a crank revolution
speed = 100

"""
Parameters that can change without changing joints between objects.

Can be distance between joints, or an angle.
Units are given relative to crank length, which is normalized to 1.
"""
param_names = ("triangle", "ape", "femur", "rockerL",
               "rockerS", "f", "tibia", "phi")
param = (
    # AB distance (=AB_p) "triangle":
    2,
    # "ape":
    np.pi/4,
    # femur = 3 for higher steps, 2 for standard, 1.8 is good enough
    1.8,
    # "rockerL":
    2.6,
    # "rockerS":
    1.4,
    # "phi":
    np.pi+.2,
    # "tibia":
    2.5,
    # "f":
    1.8,
)
# Optimized but useless stridder with step of size 5.05
# param = (2.62484195, 1.8450077, 2.41535873, 2.83669735, 2.75235715,
#         4.60386788, 3.49814371, 3.51517851)
# Limits for parameters, will be used in optimizers
bounds = ((0, 0, 0, 0, 0, 0, 0, 0),
          (8, 2 * np.pi, 7.2, 10.4, 5.6, 2 * np.pi, 10, 7.6))

# Initial coordinates according to previous dimensions
begin = ((0, 0), (0, 1), (1.41, 1.41), (-1.41, 1.41), (0, -1), (-2.25, 0),
         (2.25, 0), (-1.4, -1.2), (1.4, -1.2), (-2.7, -2.7), (2.7, -2.7))

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
    flattout = []
    for c in out:
        if c == ():
            flattout.append(0)
        else:
            flattout.extend(c)
    return tuple(flattout)


def complete_strider(constraints, prev):
    """
    Take two sequences to define strider linkage.

    Arguments
    ---------
    * constraints: the sequence of geometrical constraints
    * prev: coordinates to set by default.
    """
    # Fixed points (mechanism body)
    # A is the origin
    A = ls.Static(x=0, y=0, name="A")
    # Vertical axis for convience,
    Y = ls.Static(0, 1, name="Point (0, 1)")
    # For drawing only
    Y.joint0 = A
    # Not fixed because we will optimize this position
    B = ls.Fixed(joint0=A, joint1=Y, name="Frame right (B)")
    B_p = ls.Fixed(joint0=A, joint1=Y, name="Frame left (B_p)")
    # Pivot joints, explicitely defined to be modified later
    # Joint linked to crank. Coordinates are chosen in each frame
    C = ls.Crank(joint0=A, angle=2*np.pi/n, name="Crank link (C)")
    D = ls.Pivot(joint0=B_p, joint1=C, name="Left knee link (D)")
    E = ls.Pivot(joint0=B, joint1=C, name="Right knee link (E)")
    # F is fixed relative to C and E
    F = ls.Fixed(joint0=C, joint1=E, name='Left ankle link (F)')
    # G fixed to C and D
    G = ls.Fixed(joint0=C, joint1=D, name='Right ankle link (G)')
    H = ls.Pivot(joint0=D, joint1=F, name="Left foot (H)")
    Ii = ls.Pivot(joint0=E, joint1=G, name="Right foot (I)")
    # Mechanisme definition
    strider = ls.Walker(
        joints=(A, Y, B, B_p, C, D, E, F, G, H, Ii),
        order=(A, Y, B, B_p, C, D, E, F, G, H, Ii),
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
    strider : Linkage
        The requested strider linkage.
    """
    # Fixed points (mechanism body)
    # A is the origin
    A = ls.Static(x=0, y=0, name="A")
    # Vertical axis for convience,
    Y = ls.Static(0, 1, name="Point (0, 1)")
    # For drawing only
    Y.joint0 = A
    # Not fixed because we will optimize this position
    B = ls.Fixed(joint0=A, joint1=Y, name="Frame right (B)")
    B_p = ls.Fixed(joint0=A, joint1=Y, name="Frame left (B_p)")
    # Pivot joints, explicitely defined to be modified later
    # Joint linked to crank. Coordinates are chosen in each frame
    C = ls.Crank(joint0=A, angle=2*np.pi/n, name="Crank link (C)")
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
    # Mechanisme definition
    strider = Walker(
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
    linkage : pylinkage.linkage.Linkage
        Linkage to simulate
    prev : tuple[tuple[float]], optional
        Previous coordinates to use. The default is None.
    debug : bool, optional
        Launch in debug mode (frame by frame, with forces visualisation).
        The default is False.
    duration : float, optional
        Simulation duration (in seconds). The default is 40.
    save : bool, optional
        Save the video file instead of displaying it. The default is False.
    """
    # Define intial positions
    linkage.rebuild(prev)
    if debug:
        ls.video_debug(linkage)
    else:
        ls.video(linkage, duration, save)
    plt.show()

# Ugly way to save (position + cost) history
history = []

def sym_stride_evaluator(linkage, dims, pos):
    """Give score to each dimension set for symmetric strider."""
    linkage.set_num_constraints(param2dimensions(dims), flat=False)
    linkage.set_coords(pos)
    try:
        points = 12
        # Complete revolution with 12 points
        tuple(tuple(i) for i in linkage.step(iterations=points + 1,
                                             dt=n/points))
        # Again with n points, and at least 12 iterations
        # L = tuple(tuple(i) for i in linkage.step(iterations=n))
        factor = int(points / n) + 1
        loci = tuple(tuple(i) for i in linkage.step(
            iterations=n * factor, dt=n / n / factor))
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
        score =  max(k[0] for k in locus) - min(k[0] for k in locus)
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
        Figuer to draw on. The default is None.
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


ani = []


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
        Type of visualisation.
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
    print("Initial dimensions: ", dims)

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
            history[i:i+age] for i in range(0, len(history), age)
        ]
        animation = anim.FuncAnimation(
            fig,
            func=repr_polar_swarm,
            frames=formatted_history,
            fargs=(fig, lines, t), blit=True,
            interval=10, repeat=True,
            save_count=(iters - 1) * bool(save_each))
        ani.append(animation)
        plt.show()
        if save_each:
            writer = anim.FFMpegWriter(
                fps=24, bitrate=1800,
                metadata={
                        'title': "Particule swarm looking for R^8 in R "
                        "application maximum",
                        'comment': "Made with Python and Matplotlib",
                        'description': "The swarm tries to find best dimension"
                        " set for Strider legged mechanism"
                }
            )
            ani[-1].save(r"PSO.mp4", writer=writer)
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
            history[i:i+age][:-1] for i in range(0, len(history), age)
        ]
        animation = anim.FuncAnimation(
            fig, lambda *args: ls.swarm_tiled_repr(linkage, *args),
            formatted_history, fargs=(fig, axes, param2dimensions), blit=False,
            interval=40, repeat=False, save_count=(iters - 1) * bool(save_each)
        )
        ani.append(animation)
        plt.show(block=not save_each)
        if save_each:
            writer = anim.FFMpegWriter(
                fps=24, bitrate=1800,
                metadata={
                        'title': "Particule swarm looking for R^8 in R "
                        "application maximum",
                        'comment': "Made with Python and Matplotlib",
                        'description': "The swarm looks for best dimension "
                        "set for Strider legged mechanism"}
                )

            ani[-1].save("Particle swarm optimization.mp4", writer=writer)
        return out

    elif save_each:
        for dim, i in ls.particle_swarm_optimization(
                eval_func, linkage, dims, age, iters=iters,
                bounds=bounds, #iterable=True,
                dimensions=len(dims), # *args
        ):
            if not i % save_each:
                f = open('PSO optimizer.txt', 'w')
                # We only keep best results
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
                eval_func, linkage, dims, n_particles=age, bounds=bounds,
                dimensions=len(dims),
                iters=iters, # iterable=False,
                *args
            )
        )
        return out
        return out.sort(key=lambda x: x[1], reverse=True)


def fitness(dna, linkage_hollow):
    """
    Individual yield, return average efficiency and initial coordinates.

    Parameters
    ----------
    dna : list of 3 elements
        First element is dimensions. Second element is score (unused).
        Third element is initial positions.
    linkage_hollow : Linkage
        A which will integrate this DNA (avoid redifining a new linkage).

    Returns
    -------
    list
        List of two elements: score (a float), and initial positions.
        Score is -float('inf') when mechanism building is impossible.
    """
    linkage_hollow.set_num_constraints(param2dimensions(dna[0]), flat=False)
    linkage_hollow.rebuild(dna[2])
    # Check if mechanism is buildable
    try:
        # Save initial coordinates
        pos = tuple(linkage_hollow.step())[-1]
    except ls.UnbuildableError:
        return - float('inf'), list()
    else:
        world = ls.World()
        world.add_linkage(linkage_hollow)
        # Simulation duration (in seconds)
        duration = 40
        # Somme of yields
        tot = 0
        # Motor turned on duration
        dur = 0
        n = duration * ls.params["camera"]["fps"]
        n /= ls.params["simul"]["time_coef"]
        for j in range(int(n)):
            efficiency, energy = world.update(j)
            tot += efficiency
            dur += energy
        if dur == 0:
            return - float('inf'), list()
        print("Score:", tot / dur)
        # Return 100 times average yield, and initial positions
        return tot / dur, pos


def evolutive_optimizer(linkage, dims=param, prev=None, pop=10, iters=10,
                        init_pop=None, save=False, startnstop=False):
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
        Initial population for a highest initial genetic diversity.
        The default is None.
    save : bool, optional
        To save the optimized data. The default is False.
    startnstop : bool, optional
        To use a save save and save results regularly. The default is False.

    Returns
    -------
    list
        List of optimized linkages with dimensions, score and initial
        positions.

    """
    linkage.rebuild(prev)
    linkage.add_legs(3)
    linkage.step()
    dna = list(dims), 0, list(linkage.get_coords())
    o = ls.evolutionnary_optimization(
        dna=dna, prob=.07, fitness=fitness, ite=iters, max_pop=pop,
        startnstop=startnstop, fitness_args=[linkage]
    )
    if save:
        file = open('Evolutive optimizer.txt', 'w')
        # We only keep 10 best results
        for i in range(min(10, len(o))):
            file.write('{}\n{}\n{}\n----\n'.format(o[i][0], o[i][1], o[i][2]))
        file.close()
    return o


def show_optimized(linkage, data, n_show=10, duration=5, symmetric=True):
    """Show the optimized linkages."""
    for datum in zip(data.swarm.current_cost[:n_show], data.swarm.position):
        if datum[0] == 0:
            continue
        if symmetric:
            linkage.set_num_constraints(param2dimensions(datum[1]), flat=False)
        else:
            linkage.set_num_constraints(datum[1], flat=False)
        ls.show_linkage(
            linkage, prev=begin, title=str(datum[0]), duration=10
        )

strider = complete_strider(param2dimensions(param), begin)
print(
    "Initial score: {}"
        .format(sym_stride_evaluator(strider, param, begin))
)
# Trials and errors optimization as comparison
optimized_striders = ls.trials_and_errors_optimization(
    sym_stride_evaluator, strider, param, divisions=4
)
print(
    "Score after trials and errors optimization: {}"
        .format(optimized_striders[0][0])
)

# Particle swarm optimization
optimized_striders = swarm_optimizer(
    strider, show=1, save_each=0, age=40, iters=40, bounds=bounds,
)
print(
    "Score after particle swarm optimization: {}"
        .format(optimized_striders[0][0])
)
show_optimized(strider, optimized_striders)
ls.show_linkage(strider, save=False, duration=10, iteration_factor=n)
# We add some legs
strider.add_legs(3)
show_physics(strider, debug=False, duration=40, save=False)
optimized_striders = evolutive_optimizer(
    strider, dims=param, prev=begin, pop=10, iters=100,
    save=False, startnstop=False
)
print(
    "Fitness after evolutive optimization: {}"
        .format(optimized_striders[0][0])
)