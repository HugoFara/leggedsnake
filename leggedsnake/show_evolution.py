# -*- coding: utf-8 -*-
"""
The show_evolution module provides data visualization about GA.

Created on Mon Jun 10 2019 14:30:05.

@author: HugoFara
"""

import json
import matplotlib.pyplot as plt
import numpy as np

DATA = []


def draw_any_func(axis, scores, func):
    """Draw any kind of function."""
    vector = tuple(map(func, scores))
    axis.plot(vector)


def draw_median_score(axis, scores):
    """Compute a median vector from a score matrix and draw it."""
    median = tuple(map(np.nanmedian, scores))
    axis.plot(median, c='g', label="Median score")


def draw_best_score(axis, scores):
    """Compute the best score vector form a score matrix and draw it."""
    best = tuple(map(np.nanmax, scores))
    axis.plot(best, c='r', label="Best score")


def draw_standard_deviation(axis, scores):
    """Draw the standard deviation of scores from a score matrix."""
    std = tuple(map(np.nanstd, scores))
    axis.plot(std, c='b', label="Standard deviation")


def draw_population(axis, populations):
    """Just draw the number of individuals in the population."""
    pop_vector = tuple(map(len, populations))
    axis.plot(pop_vector, label="Population")
    axis.set_ylabel("Total population")
    axis.tick_params(direction="in")


def draw_diversity(axis, dimensions):
    """Draw the standard deviation in the dimensions."""
    diversity_vector = tuple(map(np.nanstd, dimensions))
    axis.plot(diversity_vector, label="Genetic diversity")


def show_genetic_optimization(data=DATA):
    """
    Show a graph representing an optimization with a genetic algorithm.

    Parameters
    ----------
    data : Any, default=None
        Use json.load({path to file}) to generate the data.
    """
    scores = [
        [dna[0] if not np.isinf(dna[0]) else 0 for dna in turn["population"]]
        for turn in data
    ]

    fig = plt.figure()

    # Data relative to score
    scores_axis = fig.add_subplot(211)
    scores_axis.grid()
    draw_median_score(scores_axis, scores)

    draw_standard_deviation(scores_axis, scores)

    draw_best_score(scores_axis, scores)

    draw_any_func(scores_axis, scores, np.nanmin)

    scores_axis.set_xlabel('Turn')

    # Other data
    dimensions = [[dna[1] for dna in turn["population"]] for turn in data]
    second_axis = fig.add_subplot(212)
    draw_diversity(second_axis, dimensions)
    # Another axis is necessary due to large values
    pop_axis = second_axis.twinx()
    draw_population(pop_axis, scores)
    fig.suptitle("Evolution of linkages using a genetic algorithm")
    fig.legend()
    plt.show()


if __name__ == "__main__":
    with open("Population evolution.json") as file:
        DATA.extend(json.load(file))

    show_genetic_optimization()
