#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
The geneticoptimizer module provides optimizers and wrappers for GA.

As for now, I didn't try a convincing Genetic Algorithm library. This is why
it is built-in here. Feel free to propose a copyleft library on GitHub!

Created on Thu Jun 10 21:20:47 2021.

@author: HugoFara
"""
import os.path
import json
import multiprocessing as mp
import numpy as np
from numpy.random import rand, normal, randint
# Progress bar
import tqdm
from pylinkage.geometry import dist


def tqdm_verbosity(iterable, verbose=True, *args, **kwargs):
    """Wrapper for tqdm, that let you specify if you want verbosity."""
    if verbose:
        for i in tqdm.tqdm(iterable, *args, **kwargs):
            yield i
    else:
        for i in iterable:
            yield i


def kwargs_switcher(arg_name, kwargs, default=None):
    """Simple function to return the good element from a kwargs dict."""
    out = default
    if arg_name in kwargs:
        out = kwargs.pop(arg_name) or out
    return out


def load_population(file_path):
    """Return a population from a given file."""
    with open(file_path, 'r') as file:
        data = json.load(file)
        pop = data[-1]['population']
    return pop


def save_population(file_path, population, verbose=False, data_descriptors=None):
    """
    Save the population to a json file.

    Parameters
    ----------
    file_path : str
    population : list of dna
    verbose : bool
    data_descriptors : dict
        Any additional value you want to save for the current generation.
    """
    if not os.path.exists(file_path):
        data = []
        turn = 0
    else:
        with open(file_path, 'r') as file:
            data = json.load(file)
        turn = data[-1]['turn'] + 1
    data.append({
        'turn': turn,
        'population': population
    })
    if data_descriptors is not None:
        data[-1].update(data_descriptors)
    with open(file_path, 'w') as file:
        json.dump(data, file)
    if verbose:
        print('Data saved.')


def birth(par1, par2, prob):
    """
    Return a new individual with par1 and par2 as parents (two sequences).

    Child are generated by a uniform crossover followed by a random "resetting"
    mutation of each gene. The resetting is a normal law.

    Initial positions come from one of the two parents randomly.

    Parameters
    ----------
    par1 : list[float, tuple of float, tuple of tuple of float]
        Dna of first parent.
    par2 : list[float, tuple of float, tuple of tuple of float]
        Dna of second parent.
    prob : list[float]
        Probability for each gene to mutate, width of a normal law.

    Returns
    -------
    child : list[float, tuple of float, tuple of tuple of float]
        Dna of the child.

    """
    child = [0, [], []]
    for gene1, gene2 in zip(par1[1], par2[1]):
        child[1].append(normal((gene1 if rand() < .5 else gene2), prob))
    for pos1, pos2 in zip(par1[2], par2[2]):
        child[2].append(pos1 if rand() < .5 else pos2)
    return child


def evaluate_individual(dna, fitness, fitness_args):
    """Simple evaluation for a single individual.

    Parameters
    ---------
    dna : list[float, tuple of float, tuple of tuple of float]
        List of the individuals' DNAs
    fitness : callable
        Fitness function of signature fitness(dna, fitness_args) → float.
    fitness_args : tuple
        Additional arguments to pass to the fitness function. Usually the
        initial positions of the joints.

    Returns
    -------
    tuple
        Score then initial coordinates.

    See Also
    --------
    evaluate_population : counterpart for an entire population.
    """
    if fitness_args is None:
        fit = fitness(dna)
    else:
        fit = fitness(dna, *fitness_args)
    if len(fit[1]):
        return fit[0], fit[1]
    # Don't change initial positions for unbuildable individuals.
    return fit[0], dna[2]


def evaluate_population(pop, fitness, fitness_args, verbose=True, processes=1):
    """
    Evaluate the whole population, attribute scores.

    Parameters
    ---------
    pop : list of list[float, tuple of float, tuple of tuple of float]
        List of the individuals' DNAs
    fitness : callable
        Fitness function of signature fitness(dna, fitness_args) → float.
    fitness_args : tuple
        Additional arguments to pass to the fitness function. Usually the
        initial positions of the joints.
    verbose : bool, default=True
        To display informations about population evaluation.
    processes : int, default=1
        Number of processes involved for a multiprocessors evaluation.

    See Also
    --------
    evaluate_individual : same function but on a single DNA.
    """
    # For multiprocessing we load the processes
    if processes > 1:
        res = [None] * len(pop)
        with mp.Pool(processes=processes) as pool:
            # Load the processes
            for i, dna in enumerate(pop):
                res[i] = pool.apply_async(
                    evaluate_individual, (dna, fitness, fitness_args)
                )
            # Then get data
            for result, dna in zip(res, pop):
                dna[0], dna[2] = result.get()
    else:
        for dna in pop:
            dna[0], dna[2] = evaluate_individual(dna, fitness, fitness_args)

    if verbose:
        diversity = np.linalg.norm(
            np.var([dna[1] for dna in pop], axis=0)
        )
        print("Scores:", [dna[1] for dna in pop])
        print("Genetic diversity: ", diversity)


def select_parents(pop, verbose=True):
    """Selection 1/4 of the population as parents."""
    median = np.median([dna[0] for dna in pop])
    # Index of the best individual
    best_index, best_dna = max(enumerate(pop), key=lambda x: x[1][0])
    # Parents selection, 1/4 of population
    parents = []
    indexes = []
    for j, individual in enumerate(pop):
        # Parents whose score is above median.
        # Individuals with the best fitness are more likely to be selected
        if (
                .5 * (individual[0] - best_dna[0]) / (best_dna[0] - median)
        ) + 1 > max(rand(), .5):
            parents.append(individual)
            indexes.append(j)
    # Add best individual if needed
    if best_index not in indexes:
        parents.insert(0, best_dna)
        indexes.append(best_index)
    # Add a random parent if odd number
    if len(parents) % 2:
        for j, individual in enumerate(pop):
            if j not in indexes:
                parents.append(individual)
                indexes.append(j)
                break
    if verbose:
        print(f"Median score: {median}, {len(parents)} parents")
    return parents


def make_children(parents, prob, max_genetic_dist=float('inf')):
    children = []
    j = 0
    while len(parents) > 1 and j < 100:
        par1 = parents.pop(randint(len(parents) - 1))
        if len(parents) > 1:
            par2 = parents.pop(randint(len(parents) - 1))
        else:
            par2 = parents.pop()
        if dist(par1[1], par2[1]) < max_genetic_dist:
            children.append(birth(par1, par2, prob))
        elif parents:
            parents.append(par1)
            parents.append(par2)
        j += 1
    return children


def evolutionary_optimization_builtin(
        dna,
        prob,
        fitness,
        iters,
        **kwargs
):
    """
    Optimization by genetic algorithm (GA).

    Parameters
    ----------
    dna : list[float, tuple of float, tuple of tuple of float]
        DNA of individuals.
    prob : list of float
        List of probabilities of the good transmission of one
        characteristics.
    fitness : callable
        Evaluation function.
        Return float.
        The
    iters : int
        Number of iterations.
    **kwargs : dict
        Other useful parameters for the optimization.

        max_pop : int, default=11
            Maximum number of individuals. The default is 11.
        init_pop : sequence of object, default=None
            Initial population, for wider INITIAL genetic diversity.
            The default is None.
        max_genetic_dist : float, default=.7
            Maximum genetic distance, before individuals
            cannot reproduce (separated species). The default is .7.
        startnstop : bool, default=False
            Ability to close program without loosing population.
            If True, we verify at initialization the existence of a data file.
            Population is save every int(250 / max_pop) iterations.
            The default is False.
        fitness_args : tuple
            Keyword arguments to send to the fitness function.
            The default is None (no argument sent).
        verbose : int
            Level of verbosity.
            0 : no verbose, do not print anything.
            1 : show a progress bar.
            2 : complete report for each turn.
            The default is 1.
        processes : int, default=1
            Number of processes that will evaluate the linkages.

    Returns
    -------
    list[float, tuple of float, tuple of tuple of float]
        List of 3-tuples: best dimensions, best score and initial positions.
        The list is sorted by score order.
    """
    file_path = 'Population evolution.json'
    startnstop = kwargs_switcher('startnstop', kwargs, False)
    if startnstop and os.path.exists(file_path):
        pop = load_population(file_path)
    else:
        # At least two parents to begin with
        pop = [[dna[0], list(dna[1]), list(dna[2])] for _ in range(2)]

    max_pop = kwargs_switcher('max_pop', kwargs, 11)
    max_genetic_dist = kwargs_switcher('max_genetic_dist', kwargs, .7)
    verbose = kwargs_switcher('verbose', kwargs, 1)
    fitness_args = kwargs_switcher('fitness_args', kwargs, None)
    # Number of evaluations to run in parallel
    processes = kwargs_switcher('processes', kwargs, default=1)
    # "Garden of Eden" phase, add enough children to get as much individuals as
    # required
    init_pop = kwargs_switcher('init_pop', kwargs, default=max_pop)
    for i in range(len(pop), init_pop):
        pop.append(
            birth(
                pop[randint(len(pop) - 1)],
                pop[randint(len(pop) - 1)],
                prob
            )
        )
    postfix = [
        "best_score", max(x[0] for x in pop),
        "best_dimensions", max(pop, key=lambda x: x[0])[1]
    ]
    iterations = tqdm_verbosity(
        range(iters),
        verbose=verbose == 1,
        total=iters,
        desc='Evolutionary optimization',
        postfix=postfix
    )
    for i in iterations:
        if verbose > 1:
            print(f"Turn: {i}, {len(pop)} individuals.")
        # Individuals evaluation
        evaluate_population(
            pop, fitness, fitness_args,
            verbose=verbose > 1,
            processes=processes
        )
        # Population selection
        # Minimal score before death
        death_score = np.quantile([j[0] for j in pop], 1 - max_pop / len(pop))
        if np.isnan(death_score):
            death_score = - float('inf')
        # We only keep max_pop individuals
        pop = list(filter(lambda x: x[0] >= death_score, pop))
        parents = select_parents(pop, verbose=verbose > 1)
        # We select the best fit individual to show off, we now he is a parent
        best_id = max(enumerate(parents), key=lambda x: x[1][0])[0]
        postfix[1] = parents[best_id][0]
        postfix[3] = parents[best_id][1]
        if startnstop:
            save_population(
                file_path, pop, verbose > 1,
                {
                    'best_score': parents[best_id][0],
                    'best_individual_id': best_id
                }
            )
        # Children generation
        children = make_children(parents, prob, max_genetic_dist)
        # Add to population
        pop.extend(children)

    out = []
    for dna in pop:
        fit = fitness(dna)
        if isinstance(fit, tuple):
            out.append((fit[0], fit[1], dna[2]))
        else:
            out.append((fit, dna[1], dna[2]))
    out.sort(key=lambda x: x[0], reverse=True)
    return out


def evolutionary_optimization(
        dna,
        fitness,
        iters,
        prob=.07,
        **kwargs
):
    """
    Run the Genetic Optimizer.

    Genetic Optimization is a procedural algorithm based on Darwinian evolution
    models.

    As of today, this function will only use the built-in algorithm, but you
    should still use it because we may implement another GA library.

    Parameters
    ----------
    dna : list[float, tuple of float, tuple of tuple of float]
        DNA of a linkage in format (score, dimensions, initial coordinates).
    prob : float or list of float, default=.07
        Mutation probability for each gene.
    fitness : callable
        Evaluation function for an MAXIMISATION problem.
        Must return a float.
    iters : int
        Number of iterations.
    **kwargs : dict
        Other useful parameters for the optimization.

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
        fitness_args : tuple
            Positional arguments to send to the fitness function.
            The default is None (no argument sent).
        verbose : int, optional
            Level of verbosity.
            0 : no verbose, do not print anything.
            1 : show a progress bar.
            2 : complete report for each turn.
            The default is 1.
        processes : int, default=1
            Number of processes that will evaluate the linkages.

    Returns
    -------
    list[dna]
        An iterable of the best fit individuals, in format
        (score, dimensions, initial coordinates).

    See Also
    --------
    evolutionary_optimization_builtin : built-in genetic algorithm.

    """
    # Legacy fallback
    return evolutionary_optimization_builtin(
        dna, prob, fitness,
        iters=iters,
        **kwargs
    )
