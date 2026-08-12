"""
Walker and optimization result serialization.

Provides JSON save/load for Walker instances and NsgaWalkingResult.
Delegates to pylinkage's ``graph_to_dict``/``graph_from_dict`` for
topology and ``Dimensions.to_dict``/``Dimensions.from_dict`` for
geometry (positions, distances, driver angles).

Files written by leggedsnake < 0.6.0 remain readable: pylinkage's
``Dimensions.from_dict`` accepts both the current ``[node_a, node_b,
distance]`` triples and the older stringified-tuple hyperedge keys.

Example::

    from leggedsnake.serialization import save_walker, load_walker

    save_walker(walker, "my_walker.json")
    loaded = load_walker("my_walker.json")

    # Save full optimization results
    from leggedsnake.serialization import save_result, load_result
    save_result(nsga_result, "optimization.json")
    loaded_result = load_result("optimization.json")
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .topology_optimization import TopologyWalkingResult

import numpy as np

from pylinkage.dimensions import Dimensions
from pylinkage.hypergraph.serialization import graph_from_dict, graph_to_dict
from pylinkage.optimization.collections import ParetoFront, ParetoSolution

from .nsga_optimizer import NsgaWalkingConfig, NsgaWalkingResult



# ---------------------------------------------------------------------------
# Walker serialization
# ---------------------------------------------------------------------------


def walker_to_dict(walker: Any) -> dict[str, Any]:
    """Serialize a Walker to a plain dict.

    Parameters
    ----------
    walker : Walker
        The walker to serialize.

    Returns
    -------
    dict[str, Any]
        JSON-serializable dictionary.
    """
    return {
        "version": 1,
        "name": walker.name,
        "topology": graph_to_dict(walker.topology),
        "dimensions": walker.dimensions.to_dict(),
        "motor_rates": walker.motor_rates,
    }


def walker_from_dict(data: dict[str, Any]) -> Any:
    """Deserialize a Walker from a plain dict.

    Parameters
    ----------
    data : dict[str, Any]
        Dictionary from ``walker_to_dict``.

    Returns
    -------
    Walker
    """
    from .walker import Walker

    topology = graph_from_dict(data["topology"])
    dimensions = Dimensions.from_dict(data["dimensions"])
    motor_rates = data.get("motor_rates", -4.0)
    name = data.get("name", "")

    return Walker(
        topology=topology,
        dimensions=dimensions,
        name=name,
        motor_rates=motor_rates,
    )


def save_walker(walker: Any, path: str | Path) -> None:
    """Save a Walker to a JSON file.

    Parameters
    ----------
    walker : Walker
        The walker to save.
    path : str | Path
        Output file path.
    """
    data = walker_to_dict(walker)
    with open(path, "w") as f:
        json.dump(data, f, indent=2)


def load_walker(path: str | Path) -> Any:
    """Load a Walker from a JSON file.

    Parameters
    ----------
    path : str | Path
        Path to JSON file created by ``save_walker``.

    Returns
    -------
    Walker
    """
    with open(path) as f:
        data = json.load(f)
    return walker_from_dict(data)


# ---------------------------------------------------------------------------
# NsgaWalkingResult serialization
# ---------------------------------------------------------------------------


def result_to_dict(result: NsgaWalkingResult) -> dict[str, Any]:
    """Serialize an NsgaWalkingResult to a plain dict.

    Serializes the Pareto front (scores, dimensions), configuration,
    and topology metadata. Gait and stability time series are omitted
    (too large and can be recomputed).

    Parameters
    ----------
    result : NsgaWalkingResult
        Optimization result to serialize.

    Returns
    -------
    dict[str, Any]
        JSON-serializable dictionary.
    """
    solutions_data = []
    for sol in result.pareto_front.solutions:
        solutions_data.append({
            "scores": list(sol.scores),
            "dimensions": sol.dimensions.tolist(),
            "init_positions": [
                list(p) if isinstance(p, tuple) else p
                for p in sol.initial_positions
            ] if sol.initial_positions else [],
        })

    config_data = {
        "n_generations": result.config.n_generations,
        "pop_size": result.config.pop_size,
        "algorithm": result.config.algorithm,
        "seed": result.config.seed,
        "verbose": result.config.verbose,
        "crossover_prob": result.config.crossover_prob,
        "mutation_eta": result.config.mutation_eta,
        "n_workers": result.config.n_workers,
    }

    data: dict[str, Any] = {
        "version": 1,
        "objective_names": list(result.pareto_front.objective_names),
        "solutions": solutions_data,
        "config": config_data,
    }

    # Include topology info if present (TopologyWalkingResult)
    if hasattr(result, "topology_info") and result.topology_info:
        topo_data = {}
        for idx, info in result.topology_info.items():
            topo_data[str(idx)] = {
                "topology_name": info.topology_name,
                "topology_id": info.topology_id,
                "topology_idx": info.topology_idx,
                "num_links": info.num_links,
            }
        data["topology_info"] = topo_data

    return data


def result_from_dict(
    data: dict[str, Any],
) -> "NsgaWalkingResult | TopologyWalkingResult":
    """Deserialize an NsgaWalkingResult from a plain dict.

    Parameters
    ----------
    data : dict[str, Any]
        Dictionary from ``result_to_dict``.

    Returns
    -------
    NsgaWalkingResult
    """
    objective_names = tuple(data.get("objective_names", []))

    solutions = []
    for sol_data in data.get("solutions", []):
        solutions.append(ParetoSolution(
            scores=tuple(sol_data["scores"]),
            dimensions=np.array(sol_data["dimensions"], dtype=float),
            init_positions=[
                tuple(p) if isinstance(p, list) else p
                for p in sol_data.get("init_positions", [])
            ],
        ))

    pareto = ParetoFront(solutions, objective_names)

    cfg_data = data.get("config", {})
    config = NsgaWalkingConfig(
        n_generations=cfg_data.get("n_generations", 100),
        pop_size=cfg_data.get("pop_size", 100),
        algorithm=cfg_data.get("algorithm", "nsga2"),
        seed=cfg_data.get("seed"),
        verbose=cfg_data.get("verbose", True),
        crossover_prob=cfg_data.get("crossover_prob", 0.9),
        mutation_eta=cfg_data.get("mutation_eta", 20.0),
        n_workers=cfg_data.get("n_workers", 1),
    )

    result: NsgaWalkingResult | TopologyWalkingResult = NsgaWalkingResult(
        pareto_front=pareto,
        config=config,
    )

    # Reconstruct topology info if present
    if "topology_info" in data:
        from .topology_optimization import TopologySolutionInfo, TopologyWalkingResult

        topo_info: dict[int, TopologySolutionInfo] = {}
        for idx_str, info_data in data["topology_info"].items():
            topo_info[int(idx_str)] = TopologySolutionInfo(
                topology_name=info_data["topology_name"],
                topology_id=info_data["topology_id"],
                topology_idx=info_data["topology_idx"],
                num_links=info_data["num_links"],
            )

        # Return as TopologyWalkingResult
        result = TopologyWalkingResult(
            pareto_front=pareto,
            topology_info=topo_info,
            config=config,
        )

    return result


def save_result(result: NsgaWalkingResult, path: str | Path) -> None:
    """Save an NsgaWalkingResult to a JSON file.

    Parameters
    ----------
    result : NsgaWalkingResult
        Optimization result to save.
    path : str | Path
        Output file path.
    """
    data = result_to_dict(result)
    with open(path, "w") as f:
        json.dump(data, f, indent=2)


def load_result(
    path: str | Path,
) -> "NsgaWalkingResult | TopologyWalkingResult":
    """Load an NsgaWalkingResult from a JSON file.

    Parameters
    ----------
    path : str | Path
        Path to JSON file created by ``save_result``.

    Returns
    -------
    NsgaWalkingResult
    """
    with open(path) as f:
        data = json.load(f)
    return result_from_dict(data)
