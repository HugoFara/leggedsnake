# -*- coding: utf-8 -*-
"""
The show_evolution module provides data visualization about GA.

Created on Mon Jun 10 2019 14:30:05.

@author: HugoFara
"""
from __future__ import annotations

import argparse
import json
from typing import Any, Callable

import matplotlib.pyplot as plt
from matplotlib.axes import Axes
import numpy as np

DATA: list[dict[str, Any]] = []


def draw_any_func(
    axis: Axes,
    scores: list[list[float]],
    func: Callable[[list[float]], float],
) -> None:
    """Draw any kind of function."""
    vector = tuple(map(func, scores))
    axis.plot(vector)


def draw_median_score(axis: Axes, scores: list[list[float]]) -> None:
    """Compute a median vector from a score matrix and draw it."""
    median = tuple(map(np.nanmedian, scores))
    axis.plot(median, label="Median score")


def draw_best_score(axis: Axes, scores: list[list[float]]) -> None:
    """Compute the best score vector form a score matrix and draw it."""
    best = tuple(map(np.nanmax, scores))
    axis.plot(best, label="Best score")


def draw_standard_deviation(axis: Axes, scores: list[list[float]]) -> None:
    """Draw the standard deviation of scores from a score matrix."""
    std = tuple(map(np.nanstd, scores))
    axis.plot(std, label="Standard deviation")


def draw_population(axis: Axes, populations: list[list[Any]]) -> None:
    """Just draw the number of individuals in the population."""
    pop_vector = tuple(map(len, populations))
    axis.plot(pop_vector, label="Population")
    axis.set_ylabel("Total population")
    axis.tick_params(direction="in")


def draw_diversity(axis: Axes, dimensions: list[list[list[float]]]) -> None:
    """Draw the standard deviation in the dimensions."""
    diversity_vector = tuple(map(np.nanstd, dimensions))
    axis.plot(diversity_vector, label="Genetic diversity")


def load_data(json_file: str) -> list[dict[str, Any]]:
    """Load the population from a JSON file, and return it."""
    with open(json_file) as file:
        data: list[dict[str, Any]] = json.load(file)
        return data


def show_genetic_optimization(data: list[dict[str, Any]] = DATA) -> None:
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
    parser = argparse.ArgumentParser(
        description="Visualize genetic algorithm optimization data."
    )
    parser.add_argument(
        "file",
        nargs="?",
        default="Population evolution.json",
        help="Path to the JSON data file (default: Population evolution.json)"
    )
    args = parser.parse_args()

    data = load_data(args.file)
    show_genetic_optimization(data)
