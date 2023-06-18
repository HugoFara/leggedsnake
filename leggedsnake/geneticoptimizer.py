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
import numpy.random as nprand
# Progress bar
import tqdm
from pylinkage.geometry import dist


def kwargs_switcher(arg_name, kwargs, default=None):
    """Simple function to return the good element from a kwargs dict."""
    out = default
    if arg_name in kwargs:
        out = kwargs[arg_name] or out
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
        Path of the file to write to.
    population : list
        Sequence of dna
    verbose : bool
        Enable or not verbosity (outputs success).
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


class GeneticOptimization:
    def __init__(
            self, 
            dna,
            fitness,
            iters,
            prob=.07,
            **kwargs
    ) -> None:
        """

        Parameters
        ----------
        dna : list
        fitness : callable
        iters : int
        prob : float or tuple[float]
        kwargs : dict
            Other useful parameters for the optimization.

            max_pop : int, default=11
                Maximum number of individuals. The default is 11.
            max_genetic_dist : float, default=.7
                Maximum genetic distance, before individuals
                cannot reproduce (separated species). The default is .7.
            startnstop : bool, default=False
                Ability to close program without loosing population.
                If True, we verify at initialization the existence of a data file.
                Population is saved every int(250 / max_pop) iterations.
                The default is False.
            fitness_args : tuple
                Keyword arguments to send to the fitness function.
                The default is None (no argument sent).
            verbose : int
                Level of verbosity.
                0: no verbose, do not print anything.
                1: show a progress bar.
                2: complete report for each turn.
                The default is 1.
        """
        self.dna = dna
        self.fitness = fitness
        self.iters = iters
        self.prob = prob
        self.kwargs = kwargs
        self.pop = None
        self.max_pop = kwargs_switcher('max_pop', self.kwargs, 11)
        self.verbosity = kwargs_switcher('verbose', self.kwargs, 1)
        self.startnstop = kwargs_switcher('startnstop', self.kwargs, False)
        if self.startnstop and os.path.exists(self.startnstop):
            self.pop = load_population(self.startnstop)
        else:
            # At least two parents to begin with
            self.pop = [[self.dna[0], list(self.dna[1]), list(self.dna[2])]]

    def birth(self, par1, par2, prob):
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
        prob : list[float] or float
            Probability for each gene to mutate, width of a normal law.

        Returns
        -------
        child : list[float, tuple of float, tuple of tuple of float]
            Dna of the child.

        """
        child = [0, [], []]
        for gene1, gene2 in zip(par1[1], par2[1]):
            child[1].append(nprand.normal((gene1 if nprand.rand() < .5 else gene2), prob))
        for pos1, pos2 in zip(par1[2], par2[2]):
            child[2].append(pos1 if nprand.rand() < .5 else pos2)
        return child

    def evaluate_individual(self, dna, fitness_args):
        """Simple evaluation for a single individual.

        Parameters
        ---------
        dna : list[float, tuple of float, tuple of tuple of float]
            List of the individuals' DNAs
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
            fit = self.fitness(dna)
        else:
            fit = self.fitness(dna, *fitness_args)
        if len(fit[1]):
            return fit[0], fit[1]
        # Don't change initial positions for unbuildable individuals.
        return fit[0], dna[2]

    def evaluate_population(self, fitness_args, verbose=True, processes=1):
        """
        Evaluate the whole population, attribute scores.

        Parameters
        ---------
        fitness_args : tuple
            Additional arguments to pass to the fitness function. Usually the
            initial positions of the joints.
        verbose : bool, default=True
            To display information about population evaluation.
        processes : int, default=1
            Number of processes involved for a multiprocessor evaluation.

        See Also
        --------
        evaluate_individual : same function but on a single DNA.
        """
        # For multiprocessing, we load the processes
        if processes > 1:
            with mp.Pool(processes=processes) as pool:
                # Load the processes
                res = [
                    pool.apply_async(
                        self.evaluate_individual, (dna, fitness_args)
                    ) for dna in self.pop
                ]
                # Then get data
                for result, dna in zip(res, self.pop):
                    dna[0], dna[2] = result.get()
        else:
            for dna in self.pop:
                dna[0], dna[2] = self.evaluate_individual(dna, fitness_args)

        if verbose:
            diversity = np.linalg.norm(
                np.var([dna[1] for dna in self.pop], axis=0)
            )
            print("Scores:", [dna[1] for dna in self.pop])
            print("Genetic diversity: ", diversity)

    def select_parents(self, verbose=True):
        """Selection 1/4 of the population as parents."""
        median = np.median([dna[0] for dna in self.pop])
        # Index of the best individual
        best_index, best_dna = max(enumerate(self.pop), key=lambda x: x[1][0])
        # Parents selection, 1/4 of population
        parents = []
        indexes = []
        for j, individual in enumerate(self.pop):
            # Parents whose score is above median.
            # Individuals with the best fitness are more likely to be selected
            if best_dna[0] == median:
                score = 1
            else:
                score = .5 * (individual[0] - best_dna[0]) / (best_dna[0] - median)
            if score + 1 > max(nprand.rand(), .5):
                parents.append(individual)
                indexes.append(j)
        # Add the best individual if needed
        if best_index not in indexes:
            parents.insert(0, best_dna)
            indexes.append(best_index)
        # Add a random parent if odd number
        if len(parents) % 2:
            for j, individual in enumerate(self.pop):
                if j not in indexes:
                    parents.append(individual)
                    indexes.append(j)
                    break
        if verbose:
            print(f"Median score: {median}, {len(parents)} parents")
        return parents

    def make_children(self, parents, prob, max_genetic_dist=float('inf')):
        children = []
        j = 0
        while len(parents) > 1 and j < 100:
            par1 = parents.pop(nprand.randint(len(parents) - 1))
            if len(parents) > 1:
                par2 = parents.pop(nprand.randint(len(parents) - 1))
            else:
                par2 = parents.pop()
            if dist(par1[1], par2[1]) < max_genetic_dist:
                children.append(self.birth(par1, par2, prob))
            elif parents:
                parents.append(par1)
                parents.append(par2)
            j += 1
        return children

    def reduce_population(self):
        """
        Reduce the population down to max_pop.

        Returns
        -------
        new_population : list of dna
            At most self.max_pop individuals, sorted by score.
        """
        # We only keep max_pop individuals
        sorted_pop = sorted(self.pop, key=lambda x: x[0], reverse=True)
        return sorted_pop[:self.max_pop]

    def run(self, iters, processes=1):
        """
        Optimization by genetic algorithm (GA).

        Parameters
        ----------
        iters : int
            Number of iterations.
        processes : int, default=1
            Number of processes that will evaluate the linkages.

        Returns
        -------
        list[float, tuple[float], tuple[tuple[float]]]
            List of 3-tuples: best score, best dimensions and initial positions.
            The list is sorted by score order.
        """

        max_genetic_dist = kwargs_switcher('max_genetic_dist', self.kwargs, .7)
        fitness_args = kwargs_switcher('fitness_args', self.kwargs, None)
        # Random children to get as many individuals as required
        for _ in range(self.max_pop - len(self.pop)):
            self.pop.append(
                self.birth(
                    self.pop[int(nprand.rand() * len(self.pop))],
                    self.pop[int(nprand.rand() * len(self.pop))],
                    self.prob
                )
            )
        # Individuals evaluation
        self.evaluate_population(
            fitness_args,
            verbose=self.verbosity > 1,
            processes=processes
        )
        postfix = {"best score": max(x[0] for x in self.pop)}
        iterations = tqdm.trange(
            iters, desc='Evolutionary optimization',
            disable=self.verbosity != 1, postfix=postfix
        )
        for i in iterations:
            if self.verbosity > 1:
                print(f"Turn: {i}, {len(self.pop)} individuals.")
            # Population selection
            self.pop = self.reduce_population()
            # Display
            if kwargs_switcher('gui', self.kwargs, False):
                kwargs_switcher('gui', self.kwargs, False)(self.pop)
            parents = self.select_parents(verbose=self.verbosity > 1)
            # We select the best fit individual to show off, we know it is a parent
            best_id = max(enumerate(parents), key=lambda x: x[1][0])[0]
            # Update progress bar
            postfix["best score"] = parents[best_id][0]
            iterations.set_postfix(postfix)
            if self.startnstop:
                save_population(
                    self.startnstop, self.pop, self.verbosity > 1,
                    {
                        'best_score': parents[best_id][0],
                        'best_individual_id': best_id
                    }
                )
            # Children generation
            children = self.make_children(parents, self.prob, max_genetic_dist)
            # Add to population
            self.pop.extend(children)
            # Individuals evaluation
            self.evaluate_population(
                fitness_args,
                verbose=self.verbosity > 1,
                processes=processes
            )

        out = []
        for dna in self.pop:
            # Return (fitness, dimensions, initial positions)
            out.append(dna)
        out.sort(key=lambda x: x[0], reverse=True)
        return out
