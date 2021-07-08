#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
The geneticoptimizer module provides optimizers and wrappers for GA.

As for now I didn't tried a convincing Genetic Algorithm library. This is why
you can either use PyGAD or the builtt-in version. The switch between versions
is made automatically on wether or not you have PyGAD installed.

Created on Thu Jun 10 21:20:47 2021.

@author: HugoFara
"""
import numpy as np
from numpy.random import rand, normal, randint
# Optimization by genetic algorithm
try:
    from pygad import GA
except ModuleNotFoundError:
    print("PyGad not installed. We will use legacy Genetic Optimizer.")
from pylinkage.geometry import dist


def birth(par1, par2, prob):
    """
    Return a new individual with par1 and par2 as parents (two sequences).

    Child are genrated by a uniform crossover followed by a random "resetting"
    mutation of each gene. The resetting is a normal law.

    Initial positions come from one of the two parents randomly.

    Parameters
    ----------
    par1 : list of three elements
        Dna of first parent.
    par2 : list of three elements
        Dna of second parent.
    prob : float
        Probability for each gene to mutate, width of a normal law.

    Returns
    -------
    child : list of three elements
        Dna of the child.

    """
    child = [[], 0, []]
    for gene1, gene2 in zip(par1[0], par2[0]):
        child[0].append(normal((gene1 if rand() < .5 else gene2), prob))
    for pos1, pos2 in zip(par1[2], par2[2]):
        child[2].append(pos1 if rand() < .5 else pos2)
    return child


def evolutionnary_optimization_legacy(
        dna, prob, fitness, ite, max_pop=11, init_pop=None,
        max_genetic_dist=.7, startnstop=False, fitness_args=None):
    """
    Optimization by genetic algorithm (GA).

    Parameters
    ----------
    dna : dict
        Dictionary of editable values.
    prob : list of floats
        List of probabilities of the good transmission of one
        characteristics.
    fitness : callable
        Evaluation function.
        Return float.
        The
    ite : int
        Number of iterations.
    max_pop : int, optional
        Maximum number of individuals. The default is 11.
    init_pop : sequence of object, optional
        Initial population, for wider INITIAL genetic diversity.
        The default is None.
    max_genetic_dist : float, optional
        Maximum genetic distance, before individuals
        cannot reproduce (separated species). The default is .7.
    startnstop : bool, optional
        Ability to close program without loosing population.
        If True, we verify at initialization the existence of a data file.
        Population is save every int(250 / max_pop) iterations.
        The default is False.
    fitness_args : sequence, optional
        Positional arguments to send to the fitness function.
        The default is None (no argument sent).

    Returns
    -------
    list
        List of 3-tuples: best dimensions, best score and initial positions.
        The list is sorted by score order.
    """
    try:
        f = open('Population data.txt', 'r')
        assert startnstop, "Population data file not found"
    except FileNotFoundError:
        pop = [[dna[0].copy(), dna[1], dna[2].copy()] for i in range(2)]
    else:
        pop = []
        print("Population data file found")
        for line in f.readlines():
            if line[0] == '{' and line[-2] == '}':
                pop.append([{}, 0, []])
                for i in line[1:-2].split(", "):
                    j = i.replace("'", '').split(": ")
                    pop[-1][0][j[0]] = float(j[1])
                    ############################
                    # Warning: does not consider user's choice
                    if j[0] != prob:
                        prob = .07
            elif line[0] == '[' and line[-2] == ']':
                for i in line[2:-3].split('), ('):
                    j = i.split(', ')
                    pop[-1][2].append((float(j[0]), float(j[1])))

    # "Garden of Eden" phase, add enough children to get as much individuals as
    # required
    if not init_pop:
        init_pop = max_pop
    for i in range(len(pop), init_pop):
        pop.append(birth(pop[randint(len(pop) - 1)],
                         pop[randint(len(pop) - 1)],
                         prob))
    for i in range(ite):
        # Individuals evaluation
        for dna in pop:
            if fitness_args is not None:
                fit = fitness(dna, *fitness_args)
            else:
                fit = fitness(dna)
            # Length of bars
            dna[1] = fit[0]
            if len(fit[1]):
                # Unbuildable individual, we don't change initial positions
                dna[2] = fit[1]
        j = np.linalg.norm(np.var([tuple(j[0].values()) for j in pop], axis=0))
        print("Turn : %s, %s individuals, scores : \n%s\nGenetic diversity: \
%s" % (i, len(pop), [j[1] for j in pop], j))
        # Population selection
        # Minimal score before death
        death_score = np.quantile([j[1] for j in pop], 1 - max_pop / len(pop))
        if np.isnan(death_score):
            death_score = - float('inf')
        # We only keep max_pop individuals
        pop = list(filter(lambda x: x[1] >= death_score, pop))
        median = np.median([j[1] for j in pop])
        # Index of best individual
        best = max(range(len(pop)), key=lambda x: pop[x][1])
        # Parents selection, 1/4 of population
        parents = []
        indexes = []
        for j, individual in enumerate(pop):
            # Parents whose score is above median.
            # Individuals with best fitness are more likely to be selected
            if (
                    .5
                    * (individual[1] - pop[best][1])
                    / (pop[best][1] - median) + 1
                    ) > max(rand(), .5):
                parents.append(individual)
                indexes.append(j)
        # Add best individual if needed
        if best not in indexes:
            parents.insert(0, pop[best])
            indexes.append(best)
        # Add a random parent if odd number
        if len(parents) % 2:
            for j, individual in enumerate(pop):
                if j not in indexes:
                    parents.append(individual)
                    indexes.append(j)
                    break
        print("Median score: %s, %s parents\n----" % (median, len(parents)))
        if startnstop and not i % int(250 / max_pop):
            file = open('Population data.txt', 'w')
            for j in pop:
                file.write('%s\n%s\n%s\n----\n' % (j[0], j[1], j[2]))
            file.close()
            print('Data saved.')
        # Children generation
        children = []
        j = 0
        while len(parents) > 1 and j < 100:
            par1 = parents.pop(randint(len(parents) - 1))
            if len(parents) > 1:
                par2 = parents.pop(randint(len(parents) - 1))
            else:
                par2 = parents.pop()
            if dist(par1[0].values(), par2[0].values()) < max_genetic_dist:
                children.append(birth(par1, par2, prob))
            elif parents:
                parents.append(par1)
                parents.append(par2)
            j += 1
        # Add to population
        pop.extend(children)

    out = []
    for i in pop:
        fit = fitness(i)
        if isinstance(fit, tuple):
            out.append((i[0], fit[0], fit[1]))
        else:
            out.append((i[0], fit, i[2]))
    out.sort(key=lambda x: x[1], reverse=True)
    return out


def evolutionnary_optimization(
        dna, prob, fitness, ite, max_pop=11, init_pop=None,
        max_genetic_dist=.7, startnstop=False, fitness_args=None):
    """
    Run the Genetic Optimizer.

    Genetic Optimization is a procedural algorithm based on Darwinian evolution
    models.

    This function can use either PySwarms.GA if found, and automatically
    falls backs to the legacy algorithm if not.

    Parameters
    ----------
    dna : TYPE
        DESCRIPTION.
    prob : TYPE
        DESCRIPTION.
    fitness : callable
        Evaluation function for an %AXIMISATION problem.
        Must return a float.
    ite : int
        Number of iterations.
    max_pop : int, optional
        Maximum number of individuals. The default is 11.
    init_pop : list of object, optional
        Initial population, for wider INITIAL genetic diversity.
        The default is None.
    max_genetic_dist : TYPE, optional
        DESCRIPTION. The default is .7.
    startnstop : bool, optional
        Ability to close program without loosing population.
        If True, we verify at initialization the existence of a data file.
        Population is save every int(250 / max_pop) iterations.
        The default is False.
    fitness_args : sequence, optional
        Positional arguments to send to the fitness function.
        The default is None (no argument sent).

    Returns
    -------
    TYPE
        An iterable of the best fit individuals.

    """
    def fitness_func(dims, index):
        fit = fitness([dims, 0, dna[2]], *fitness_args)
        return fit[0]
    # If pyswarms.GA is installed
    if GA:
        natural_history = GA(
           num_generations=ite,
           num_parents_mating=int(np.ceil(max_pop / 10)),
           fitness_func=fitness_func,
           initial_population=init_pop,
           sol_per_pop=max_pop,
           num_genes=len(dna[0]),
           init_range_low=0, init_range_high=5,
           crossover_type='uniform', crossover_probability=.5,
           mutation_type="random", mutation_probability=prob
           )
        natural_history.run()
        natural_history.plot_result()
        return natural_history

    # Legacy fallback
    return evolutionnary_optimization_legacy(
        dna, prob, fitness, ite,
        max_pop=max_pop,
        init_pop=len(init_pop),
        max_genetic_dist=max_genetic_dist,
        startnstop=startnstop,
        fitness_args=fitness_args
        )
