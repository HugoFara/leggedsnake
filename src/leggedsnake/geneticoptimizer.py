#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
The geneticoptimizer module provides optimizers and wrappers for GA.

As for now, I didn't try a convincing Genetic Algorithm library. This is why
it is built-in here. Feel free to propose a copyleft library on GitHub!

Created on Thu Jun 10 21:20:47 2021.

@author: HugoFara
"""
from __future__ import annotations

import os.path
import json
import multiprocessing as mp
from collections.abc import Sequence
from typing import Any, Callable, TypeAlias

import numpy as np
import numpy.random as nprand
# Progress bar
import tqdm

from pylinkage.optimization.collections import Agent
from pylinkage.population import Ensemble, Member

# Type aliases for DNA structure
# DNA format: [fitness_score, dimensions_list, coordinates_list]
DNA: TypeAlias = list[Any]  # [float, list[float], list[tuple[float, float]]]
Population: TypeAlias = list[DNA]
FitnessFunc: TypeAlias = Callable[..., tuple[float, list[tuple[float, float]]]]


def agents_to_ensemble(agents: Sequence[Agent], linkage: Any) -> Ensemble:
    """Wrap a list of Agents in a pylinkage Ensemble.

    Provides ``.rank()``, ``.top()``, ``.filter()``, ``.filter_by_score()``
    and numpy-style indexing over optimization results. The template
    linkage is only used for topology metadata; batch simulation via
    ``Ensemble.simulate()`` is not supported for Walker-based mechanisms.

    Parameters
    ----------
    agents : sequence of Agent
        Optimization results (e.g. from ``genetic_algorithm_optimization``).
    linkage : Walker, Mechanism, or Linkage
        Template. A ``Walker`` is converted to its underlying Mechanism.

    Returns
    -------
    Ensemble
        One member per agent, with ``scores={"score": agent.score}``.
    """
    template = linkage.to_mechanism() if hasattr(linkage, "to_mechanism") else linkage

    if not agents:
        n_constraints = len(linkage.get_num_constraints()) if hasattr(linkage, "get_num_constraints") else 0
        n_joints = len(template.joints) if hasattr(template, "joints") else 0
        return Ensemble(
            template,
            np.zeros((0, n_constraints), dtype=np.float64),
            np.zeros((0, n_joints, 2), dtype=np.float64),
            {"score": np.zeros(0, dtype=np.float64)},
        )

    n_joints = len(agents[0].init_positions)
    members = [Member.from_agent(a, n_joints) for a in agents]
    dims = np.stack([m.dimensions for m in members])
    positions = np.stack([m.initial_positions for m in members])
    scores = {"score": np.array([a.score for a in agents], dtype=np.float64)}
    return Ensemble(template, dims, positions, scores)


def kwargs_switcher(arg_name: str, kwargs: dict[str, Any], default: Any = None) -> Any:
    """Simple function to return the good element from a kwargs dict."""
    out = default
    if arg_name in kwargs:
        out = kwargs[arg_name] or out
    return out


def load_population(file_path: str) -> Population:
    """Return a population from a given file."""
    with open(file_path, 'r') as file:
        data = json.load(file)
        pop: Population = data[-1]['population']
    return pop


def save_population(
    file_path: str,
    population: Population,
    verbose: bool = False,
    data_descriptors: dict[str, Any] | None = None,
) -> None:
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
    dna: DNA
    fitness: FitnessFunc
    iters: int
    prob: float
    kwargs: dict[str, Any]
    pop: Population
    max_pop: int
    verbosity: int
    startnstop: str | bool

    def __init__(
        self,
        dna: DNA,
        fitness: FitnessFunc,
        prob: float = 0.07,
        **kwargs: Any,
    ) -> None:
        """

        Parameters
        ----------
        dna : list
        fitness : callable
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
        self.iters = 0
        self.prob = prob
        self.kwargs = kwargs
        self.max_pop = kwargs_switcher('max_pop', self.kwargs, 11)
        self.verbosity = kwargs_switcher('verbose', self.kwargs, 1)
        self.startnstop = kwargs_switcher('startnstop', self.kwargs, False)
        if self.startnstop and os.path.exists(str(self.startnstop)):
            self.pop = load_population(str(self.startnstop))
        else:
            # At least two parents to begin with
            self.pop = [[self.dna[0], list(self.dna[1]), list(self.dna[2])]]

    def birth(self, par1: DNA, par2: DNA) -> DNA:
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

        Returns
        -------
        child : list[float, tuple of float, tuple of tuple of float]
            Dna of the child.

        """
        child: DNA = [0, [], []]
        for gene1, gene2 in zip(par1[1], par2[1]):
            child[1].append(nprand.normal((gene1 if nprand.rand() < .5 else gene2), self.prob))
        for pos1, pos2 in zip(par1[2], par2[2]):
            child[2].append(pos1 if nprand.rand() < .5 else pos2)
        return child

    def evaluate_individual(
        self, dna: DNA, fitness_args: tuple[Any, ...] | None
    ) -> tuple[float, list[tuple[float, float]]]:
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

    def evaluate_population(
        self, fitness_args: tuple[Any, ...] | None, verbose: bool = True, processes: int = 1
    ) -> None:
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

    def select_parents(self, verbose: bool = True) -> Population:
        """Selection 1/4 of the population as parents."""
        median = np.median([dna[0] for dna in self.pop])
        # Index of the best individual
        best_index, best_dna = max(enumerate(self.pop), key=lambda x: x[1][0])
        # Parents selection, 1/4 of population
        parents: Population = []
        indexes: list[int] = []
        for j, individual in enumerate(self.pop):
            # Parents whose score is above median.
            # Individuals with the best fitness are more likely to be selected
            if best_dna[0] == median:
                score = 1.0
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

    def make_children(
        self, parents: Population, max_genetic_dist: float = float('inf')
    ) -> Population:
        children: Population = []
        j = 0
        while len(parents) > 1 and j < 100:
            par1 = parents.pop(nprand.randint(len(parents) - 1))
            par2 = parents.pop(int(nprand.rand() * len(parents)))
            if np.linalg.norm(np.array(par1[1]) - np.array(par2[1])) < max_genetic_dist:
                children.append(self.birth(par1, par2))
            elif parents:
                parents.append(par1)
                parents.append(par2)
            j += 1
        return children

    def reduce_population(self) -> Population:
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

    def run(self, iters: int, processes: int = 1) -> list[Agent]:
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
        list[Agent]
            List of Agent(score, dimensions, init_positions) sorted by
            score in descending order. Compatible with pylinkage's
            chain_optimizers pipeline.
        """

        max_genetic_dist = kwargs_switcher('max_genetic_dist', self.kwargs, 10)
        fitness_args = kwargs_switcher('fitness_args', self.kwargs, None)
        # Random children to get as many individuals as required
        for _ in range(self.max_pop - len(self.pop)):
            self.pop.append(
                self.birth(
                    self.pop[int(nprand.rand() * len(self.pop))],
                    self.pop[int(nprand.rand() * len(self.pop))]
                )
            )
        # Individuals evaluation
        self.evaluate_population(
            fitness_args,
            verbose=self.verbosity > 1,
            processes=processes
        )
        postfix = {
            "best score": max(x[0] for x in self.pop),
            "average score": np.mean([x[0] for x in self.pop])
        }
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
            if kwargs_switcher('gui', self.kwargs, None) is not None:
                kwargs_switcher('gui', self.kwargs, None)(self.pop)
            parents = self.select_parents(verbose=self.verbosity > 1)
            # We select the best fit individual to show off, we know it is a parent
            best_id = max(enumerate(parents), key=lambda x: x[1][0])[0]
            # Update progress bar
            postfix.update({
                "best score": parents[best_id][0],
                "average score": np.mean([x[0] for x in self.pop])
            })
            iterations.set_postfix(postfix)
            if self.startnstop and isinstance(self.startnstop, str):
                save_population(
                    self.startnstop, self.pop, self.verbosity > 1,
                    {
                        'best_score': parents[best_id][0],
                        'best_individual_id': best_id
                    }
                )
            # Children generation
            children = self.make_children(parents, max_genetic_dist)
            # Add to population
            self.pop.extend(children)
            # Individuals evaluation
            self.evaluate_population(
                fitness_args,
                verbose=self.verbosity > 1,
                processes=processes
            )

        # Return as Agent namedtuples, sorted by score descending
        sorted_pop = sorted(self.pop, key=lambda x: x[0], reverse=True)
        return [
            Agent(score=dna[0], dimensions=dna[1], init_positions=dna[2])
            for dna in sorted_pop
        ]


def genetic_algorithm_optimization(
    eval_func: Callable[..., float],
    linkage: Any,
    center: Sequence[float] | None = None,
    bounds: tuple[Sequence[float], Sequence[float]] | None = None,
    order_relation: Callable[[float, float], float] = max,
    max_pop: int = 30,
    iters: int = 100,
    prob: float = 0.07,
    max_genetic_dist: float = 10.0,
    processes: int = 1,
    startnstop: str | bool = False,
    verbose: bool = True,
    **kwargs: Any,
) -> Ensemble:
    """Genetic algorithm optimization with the standard pylinkage interface.

    This wrapper bridges leggedsnake's ``GeneticOptimization`` to pylinkage's
    optimizer contract, making it usable with ``chain_optimizers`` and
    interchangeable with PSO, DE, etc.

    The evaluation function receives ``(linkage, dimensions, init_positions)``
    and returns a scalar score — exactly like pylinkage optimizers.

    Parameters
    ----------
    eval_func : callable
        Evaluation function with signature
        ``(linkage, dimensions, init_positions) -> float``.
    linkage : Walker or Linkage
        The mechanism to optimize. Must provide ``get_num_constraints()``,
        ``set_num_constraints()``, ``get_coords()``, ``set_coords()``.
    center : sequence of float, optional
        Initial dimensions. If *None*, read from ``linkage``.
        ``chain_optimizers`` injects the previous stage's best here.
    bounds : tuple of (lower, upper), optional
        Not directly used by the GA, but accepted for API compatibility.
        When provided, initial random children are clamped to these bounds.
    order_relation : callable, optional
        ``max`` (default) for maximization, ``min`` for minimization.
    max_pop : int
        Maximum population size. Default 30.
    iters : int
        Number of generations. Default 100.
    prob : float
        Mutation standard deviation. Default 0.07.
    max_genetic_dist : float
        Speciation threshold. Default 10.0.
    processes : int
        Number of parallel processes for evaluation. Default 1.
    startnstop : str or bool
        Path to checkpoint file, or False to disable. Default False.
    verbose : bool
        Show progress bar. Default True.

    Returns
    -------
    Ensemble
        Population wrapped in a pylinkage ``Ensemble`` (one member per
        candidate, columnar scores). Ranking is already applied: the
        member at index 0 is the best under ``order_relation``. Iterate,
        slice, or call ``.top()`` / ``.rank()`` / ``.filter_by_score()``
        to drill down. Use ``ensemble[i].score`` / ``.dimensions`` /
        ``.initial_positions`` to access fields — or call
        ``ensemble[i].to_agent()`` for the legacy tuple shape.
    """
    minimize = order_relation is min
    dims = list(center) if center is not None else linkage.get_num_constraints()
    init_pos = list(linkage.get_coords())

    # Bridge eval_func from pylinkage's (linkage, dims, pos) -> float
    # to GeneticOptimization's (dna, linkage) -> (score, positions) contract.
    def _ga_fitness(dna: DNA, lk: Any) -> tuple[float, list[Any]]:
        lk.set_num_constraints(dna[1])
        lk.set_coords(dna[2])
        raw_score = eval_func(lk, dna[1], dna[2])
        score = -raw_score if minimize else raw_score
        return score, list(lk.get_coords())

    # Seed DNA
    seed_score = eval_func(linkage, dims, init_pos)
    if minimize:
        seed_score = -seed_score
    dna: DNA = [seed_score, list(dims), list(init_pos)]

    optimizer = GeneticOptimization(
        dna=dna,
        fitness=_ga_fitness,
        prob=prob,
        max_pop=max_pop,
        max_genetic_dist=max_genetic_dist,
        startnstop=startnstop,
        fitness_args=(linkage,),
        verbose=1 if verbose else 0,
    )
    results = optimizer.run(iters, processes=processes)

    if minimize:
        # Flip scores back to original sign
        results = [
            Agent(score=-a.score, dimensions=a.dimensions, init_positions=a.init_positions)
            for a in results
        ]
        results.sort(key=lambda a: a.score)
    return agents_to_ensemble(results, linkage)
