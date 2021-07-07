# -*- coding: utf-8 -*-
"""
Created on Sun Dec 23 21:03:11 2018.

@author: HugoFara

Complete simulator for strider mecanism, a type of walking mecanism.

Directions:
    - First section gives objects definition, and links between them
    - Second section explains how simulation works.
    - Third section is for display.
"""

import matplotlib.pyplot as plt
import matplotlib.animation as anim
import numpy as np
import pylinkage.optimizer as wo
from pylinkage.exceptions import UnbuildableError
from pylinkage.linkage import Static, Pivot, Fixed, Crank
import pylinkage.visualizer as visu

from leggedsnake import utility as wu
from leggedsnake.walker import Walker
from leggedsnake import physicsengine as pe
from leggedsnake import geneticoptimizer as go


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
"""
param = {1.5707963267948966: 1.6087602528022467, 3.141592653589793: 3.36478019266876, 4.71238898038469: 4.982560396344311, 'rockerL': 2.584774044408933, 'one': 1.1394106709324356, 'phi': 3.3257392473742047, 'femur': 2.098116925770731, 'tibia': 2.4696281593446883, 'triangle': 1.9080888204659163, 'rockerS': 1.451669502360432, 'ape': 0.6108747637740084, 'f': 1.5197400988301764}
# Set of initial positions the goes well with previous dimensions
begin = [(0, 0), (0, 1), (1.0944496744512664, 1.563004432776993), (-1.0944496744512662, 1.563004432776993), (6.123233995736766e-17, -1.0), (-2.4116997155170004, -0.0700745498561018), (2.411699715517001, -0.070074549856102), (-1.2359355451543963, -1.7614510308005955), (1.235935545154396, -1.761451030800596), (-2.543371738668845, -2.5361900629145095), (2.543371738668845, -2.5361900629145095), (1.138589676216803, 0.043246112509899), (-1.385210184363346, -0.514867786605443), (3.1919664890008175, 1.6131825842727139), (0.1663113596606027, -1.0347238403357382), (2.5894451182451443, 0.091855066624656), (0.0178909396676118, -2.5471990787847707), (3.841518904259255, -0.7694934124293479), (-0.2521962696426615, 1.1111497282603409), (-2.656493683094839, 0.1622499399173982), (1.84845375652758, -0.3949466591601567), (-1.5669070372523302, 1.72668197080705), (1.17286598517539, 1.3878118217539663), (-0.297066585737247, 0.891747724480835), (-0.2621340406012742, 0.8874270757701106), (-1.0980787313568554, -0.3041048779227688), (-3.1681379353978905, 1.2437684219358172), (1.479298183094035, -0.4995149633987277), (-2.541216000676675, -0.46126456099549), (-0.1143206787728823, -1.3716087130725885), (-3.9155070920685544, -1.1100587996515308), (0.4942397142541866, -2.76418338469027)]
"""


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
    # Fixed points (mecanism body)
    # A is the origin
    A = Static(x=0, y=0, name="A")
    # Vertical axis for convience,
    Y = Static(0, 1, name="Point (0, 1)")
    # For drawing only
    Y.joint0 = A
    # Not fixed because we will optimize this position
    B = Fixed(joint0=A, joint1=Y, name="Frame right (B)")
    B_p = Fixed(joint0=A, joint1=Y, name="Frame left (B_p)")
    # Pivot joints, explicitely defined to be modified later
    # Joint linked to crank. Coordinates are chosen in each frame
    C = Crank(joint0=A, angle=2*np.pi/n, name="Crank link (C)")
    D = Pivot(joint0=B_p, joint1=C, name="Left knee link (D)")
    E = Pivot(joint0=B, joint1=C, name="Right knee link (E)")
    # F is fixed relative to C and E
    F = Fixed(joint0=C, joint1=E, name='Left ankle link (F)')
    # G fixed to C and D
    G = Fixed(joint0=C, joint1=D, name='Right ankle link (G)')
    H = Pivot(joint0=D, joint1=F, name="Left foot (H)")
    Ii = Pivot(joint0=E, joint1=G, name="Right foot (I)")
    # Mechanisme definition
    strider = Walker(
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
    # Fixed points (mecanism body)
    # A is the origin
    A = Static(x=0, y=0, name="A")
    # Vertical axis for convience,
    Y = Static(0, 1, name="Point (0, 1)")
    # For drawing only
    Y.joint0 = A
    # Not fixed because we will optimize this position
    B = Fixed(joint0=A, joint1=Y, name="Frame right (B)")
    B_p = Fixed(joint0=A, joint1=Y, name="Frame left (B_p)")
    # Pivot joints, explicitely defined to be modified later
    # Joint linked to crank. Coordinates are chosen in each frame
    C = Crank(joint0=A, angle=2*np.pi/n, name="Crank link (C)")
    D = Pivot(joint0=B_p, joint1=C, name="Left knee link (D)")
    E = Pivot(joint0=B, joint1=C, name="Right knee link (E)")
    # F is fixed relative to C and E
    F = Fixed(joint0=C, joint1=E, name='Left ankle link (F)')
    H = Pivot(joint0=D, joint1=F, name="Left foot (H)")
    joints = [A, Y, B, B_p, C, D, E, F, H]
    if not minimal:
        # G fixed to C and D
        G = Fixed(joint0=C, joint1=D, name='Right ankle link (G)')
        joints.insert(-1, G)
        joints.append(Pivot(joint0=E, joint1=G, name="Right foot (I)"))
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
    Give mecanism a dynamic model and launch video.

    Arguments
    ---------
    prev: previous coordinates to use
    debug: launch in debug mode (frame by frame, with forces visualisation)
    duration: simulation duration (in seconds)
    save: save the video file instead of displaying it
    """
    # Define intial positions
    linkage.rebuild(prev)
    if debug:
        pe.video_debug(linkage)
    else:
        pe.video(linkage, duration, save)


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
        L = tuple(tuple(i) for i in linkage.step(
            iterations=n * factor, dt=n / n / factor))
    except UnbuildableError:
        return 0
    else:
        foot_locus = tuple(x[-2] for x in L)
        # Constraints check
        if not wu.step(foot_locus, .5, .2):
            return 0
        # Performances evaluation
        locus = wu.stride(foot_locus, .2)
        return max(k[0] for k in locus) - min(k[0] for k in locus)


def repr_polar_swarm(swarm, fig=None, lines=None, t=0):
    """
    Represent a swarm in a polar graph.

    Parameters
    ----------
    swarm : tuple
        List of dimensions then best cost.
    fig : TYPE, optional
        Figuer to draw on. The default is None.
    lines : TYPE, optional
        Lines to be modified. The default is None.
    t : TYPE, optional
        DESCRIPTION. The default is 0.

    Returns
    -------
    lines : TYPE
        Lines modified.

    """
    dims, cost = swarm
    for line, dimension_set in zip(lines, dims):
        fig.suptitle("Best cost: {}".format(cost))
        line.set_data(t, tuple(dimension_set) + (-cost,))
    return lines


ani = []


def swarm_optimizer(linkage, dims=param, show=False, save_each=0, age=300,
                    ite=400, blind_ite=200, *args, **kwargs):
    """
    Optimize linkage geometrically using PSO.

    Parameters
    ----------
    linkage : TYPE
        DESCRIPTION.
    dims : TYPE, optional
        DESCRIPTION. The default is param.
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
    ite : int, optional
        Number of iterations to run through. The default is 400.
    blind_ite : int, optional
        Number of iterations without evaluation. The default is 200.
    *args : TYPE
        DESCRIPTION.
    **kwargs : TYPE
        DESCRIPTION.

    Returns
    -------
    TYPE
        DESCRIPTION.

    """
    print("Initial dimensions: ", dims)

    def eval_func(dims, pos):
        return sym_stride_evaluator(linkage, dims, pos)
    if show == 1:
        out = wo.particle_swarm_optimization(
            eval_func, linkage,
            dims, age, ite=ite, iterable=True,
            bounds=bounds, neigh=.5, *args,
        )

        fig = plt.figure("Swarm in polar graph")
        ax = fig.add_subplot(111, projection='polar')
        lines = [ax.plot([], [], lw=.5, animated=False)[0] for i in range(age)]
        t = np.linspace(0, 2 * np.pi, len(dims) + 2)[:-1]
        ax.set_xticks(t)
        ax.set_rmax(7)
        ax.set_xticklabels(param_names + ("score",))
        animation = anim.FuncAnimation(
            fig, func=repr_polar_swarm,
            frames=zip(out.pos_history, out.cost_history),
            fargs=(fig, lines, t), blit=True,
            interval=10, repeat=True,
            save_count=(ite-1) * bool(save_each))
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
                        " set for Strider legged mecanism"})

            ani[-1].save(r"PSO.mp4", writer=writer)
        return out
    elif show == 2:
        out = wo.particle_swarm_optimization(
                eval_func, linkage,
                dims, age, ite=ite, iterable=True,
                bounds=bounds, neigh=.5, *args)

        fig = plt.figure("Swarm in tiled mode")
        cells = int(np.ceil(np.sqrt(age)))
        axes = fig.subplots(cells, cells)
        lines = [ax.plot([], [], lw=.5, animated=False)[0]
                 for ax in axes.flatten()]
        animation = anim.FuncAnimation(
            fig, lambda *args: visu.swarm_tiled_repr(linkage, *args),
            out.pos_history, fargs=(fig, axes, param2dimensions), blit=False,
            interval=40, repeat=False, save_count=(ite-1) * bool(save_each)
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
                        "set for Strider legged mecanism"}
                )

            ani[-1].save("Particle swarm optimization.mp4", writer=writer)
        return out

    elif save_each:
        for dim, i in wo.particle_swarm_optimization(
                eval_func, linkage, dims, age, ite=ite,
                bounds=bounds, iterable=True, *args):
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
            wo.particle_swarm_optimization(
                eval_func, linkage, dims, n_indi=age, bounds=bounds,
                ite=ite, blind_iter=blind_ite, iterable=False, *args))
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
    # Check if mecanism is buildable
    try:
        # Save initial coordinates
        pos = tuple(linkage_hollow.step())[-1]
    except UnbuildableError:
        return - float('inf'), list()
    else:
        world = pe.World()
        world.add_linkage(linkage_hollow)
        # Simulation duration (in seconds)
        duration = 40
        # Somme of yields
        tot = 0
        # Motor turned on duration
        dur = 0
        n = duration * pe.params["camera"]["fps"]
        n /= pe.params["simul"]["time_coef"]
        for j in range(int(n)):
            efficiency, energy = world.update(j)
            tot += efficiency
            dur += energy
        if dur == 0:
            return - float('inf'), list()
        print("Score:", tot / dur)
        # Return 100 times average yield, and initial positions
        return tot / dur, pos


def evolutive_optimizer(linkage, dims=param, prev=None, pop=10, ite=10,
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
    ite : int, optional
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
    o : list
        List of optimized linkages with dimensions, score and initial
        positions.

    """
    linkage.rebuild(prev)
    linkage.add_legs(3)
    linkage.step()
    dna = list(dims), 0, list(linkage.get_pos())
    if not init_pop:
        init_pop = pop
    o = go.evolutionnary_optimization(
        dna=dna, prob=.07, fitness=fitness, ite=ite, max_pop=pop,
        init_pop=init_pop, startnstop=startnstop, fitness_args=[linkage])
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
        visu.show_linkage(linkage, prev=begin, title=str(datum[0]),
                          duration=10)

#wu.step([(0, 0), (-1, 0), (-1, 1), (0, 1), (0, .5)], 0, .5)
from cProfile import run
#strider = complete_strider(param2dimensions(param), begin)
strider = strider_builder(param2dimensions(param), begin,
                          n_leg_pairs=5, minimal=False)
#o = swarm_optimizer(show=1, save_each=1, age=10, ite=10, blind_ite=10)
run('sym_stride_evaluator(strider, param, begin)')
#optimized_striders = wo.exhaustive_optimization(
#    sym_stride_evaluator, strider, param, delta_dim=.5)
#optimized_striders = swarm_optimizer(strider, show=1, save_each=0, age=250,
#                                     ite=200, blind_ite=1, bounds=bounds)
#show_optimized(strider, optimized_striders)
#strider.add_legs(3)
#visu.show_linkage(strider, save=False, duration=10, iteration_factor=n)
#show_physics(strider, debug=False, duration=40, save=False)
#o = evolutive_optimizer(
#    strider, dims=param, prev=begin, pop=10, ite=100, init_pop=100,
#    save=False, startnstop=False)
