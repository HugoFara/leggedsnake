#!/usr/bin/env python3
"""Tests for Walker and optimization result serialization."""

import json
import tempfile
import unittest
from math import tau
from pathlib import Path

import numpy as np

from pylinkage.dimensions import Dimensions, DriverAngle
from pylinkage.hypergraph import HypergraphLinkage, Node, Edge, Hyperedge, NodeRole
from pylinkage.optimization.collections import ParetoFront, ParetoSolution

from leggedsnake.nsga_optimizer import NsgaWalkingConfig, NsgaWalkingResult
from leggedsnake.serialization import (
    load_result,
    load_walker,
    result_from_dict,
    result_to_dict,
    save_result,
    save_walker,
    walker_from_dict,
    walker_to_dict,
)
from leggedsnake.topology_optimization import (
    TopologySolutionInfo,
    TopologyWalkingResult,
)
from leggedsnake.walker import Walker


def _make_fourbar_walker():
    hg = HypergraphLinkage(name="test_fourbar")
    hg.add_node(Node("frame", role=NodeRole.GROUND))
    hg.add_node(Node("crank", role=NodeRole.DRIVER))
    hg.add_node(Node("follower", role=NodeRole.DRIVEN))
    hg.add_edge(Edge("frame_crank", "frame", "crank"))
    hg.add_edge(Edge("frame_follower", "frame", "follower"))
    hg.add_edge(Edge("crank_follower", "crank", "follower"))
    dims = Dimensions(
        node_positions={"frame": (0, 0), "crank": (1, 0), "follower": (0, 2)},
        driver_angles={"crank": DriverAngle(angular_velocity=-tau / 12)},
        edge_distances={
            "frame_crank": 1.0, "frame_follower": 2.0, "crank_follower": 1.5,
        },
    )
    return Walker(hg, dims, name="test_fourbar", motor_rates=-4.0)


def _make_walker_with_hyperedge():
    """Walker with a hyperedge (rigid triangle)."""
    hg = HypergraphLinkage(name="triangle_walker")
    hg.add_node(Node("frame", role=NodeRole.GROUND))
    hg.add_node(Node("frame2", role=NodeRole.GROUND))
    hg.add_node(Node("crank", role=NodeRole.DRIVER))
    hg.add_node(Node("upper", role=NodeRole.DRIVEN))
    hg.add_node(Node("foot", role=NodeRole.DRIVEN))
    hg.add_edge(Edge("frame_crank", "frame", "crank"))
    hg.add_edge(Edge("crank_upper", "crank", "upper"))
    hg.add_edge(Edge("frame2_upper", "frame2", "upper"))
    hg.add_edge(Edge("upper_foot", "upper", "foot"))
    hg.add_hyperedge(Hyperedge("tri_foot", nodes=("upper", "crank", "foot")))

    dims = Dimensions(
        node_positions={
            "frame": (0, 0), "crank": (0.5, 0), "frame2": (-1, -0.3),
            "upper": (0, 1), "foot": (0, -1.5),
        },
        driver_angles={"crank": DriverAngle(angular_velocity=tau / 12)},
        edge_distances={
            "frame_crank": 0.5, "crank_upper": 1.5,
            "frame2_upper": 2.0, "upper_foot": 2.0,
        },
    )
    return Walker(hg, dims, name="triangle_walker")


def _make_nsga_result():
    """Create a synthetic NsgaWalkingResult."""
    rng = np.random.RandomState(42)
    solutions = []
    for _ in range(5):
        solutions.append(ParetoSolution(
            scores=tuple(rng.uniform(0, 10, 2).tolist()),
            dimensions=rng.uniform(0.5, 2.0, 3),
            init_positions=[(0.0, 0.0), (1.0, 0.0)],
        ))
    return NsgaWalkingResult(
        pareto_front=ParetoFront(solutions, ("distance", "stability")),
        config=NsgaWalkingConfig(
            n_generations=50, pop_size=40, seed=42, verbose=False,
        ),
    )


class TestWalkerToDict(unittest.TestCase):

    def test_roundtrip_fourbar(self):
        """Walker survives serialization roundtrip."""
        walker = _make_fourbar_walker()
        data = walker_to_dict(walker)
        loaded = walker_from_dict(data)

        self.assertEqual(loaded.name, walker.name)
        self.assertEqual(loaded.motor_rates, walker.motor_rates)
        self.assertEqual(
            set(loaded.topology.nodes.keys()),
            set(walker.topology.nodes.keys()),
        )
        self.assertEqual(
            set(loaded.topology.edges.keys()),
            set(walker.topology.edges.keys()),
        )
        # Check dimensions
        for nid in walker.dimensions.node_positions:
            orig = walker.dimensions.node_positions[nid]
            loaded_pos = loaded.dimensions.node_positions[nid]
            self.assertAlmostEqual(orig[0], loaded_pos[0])
            self.assertAlmostEqual(orig[1], loaded_pos[1])

        for eid in walker.dimensions.edge_distances:
            self.assertAlmostEqual(
                walker.dimensions.edge_distances[eid],
                loaded.dimensions.edge_distances[eid],
            )

    def test_roundtrip_with_hyperedge(self):
        """Walker with hyperedge survives roundtrip."""
        walker = _make_walker_with_hyperedge()
        data = walker_to_dict(walker)
        loaded = walker_from_dict(data)

        self.assertEqual(loaded.name, walker.name)
        self.assertEqual(
            set(loaded.topology.hyperedges.keys()),
            set(walker.topology.hyperedges.keys()),
        )

    def test_json_serializable(self):
        """walker_to_dict produces JSON-serializable output."""
        walker = _make_fourbar_walker()
        data = walker_to_dict(walker)
        json_str = json.dumps(data)
        self.assertIsInstance(json_str, str)

    def test_driver_angles_preserved(self):
        """Driver angles roundtrip correctly."""
        walker = _make_fourbar_walker()
        data = walker_to_dict(walker)
        loaded = walker_from_dict(data)

        for nid in walker.dimensions.driver_angles:
            orig = walker.dimensions.driver_angles[nid]
            loaded_da = loaded.dimensions.driver_angles[nid]
            self.assertAlmostEqual(orig.angular_velocity, loaded_da.angular_velocity)
            self.assertAlmostEqual(orig.initial_angle, loaded_da.initial_angle)

    def test_dict_motor_rates(self):
        """Walker with dict motor_rates roundtrips."""
        walker = _make_fourbar_walker()
        walker.motor_rates = {"crank": -3.5}
        data = walker_to_dict(walker)
        loaded = walker_from_dict(data)
        self.assertEqual(loaded.motor_rates, {"crank": -3.5})


class TestSaveLoadWalker(unittest.TestCase):

    def test_save_load_file(self):
        """Walker saves to and loads from JSON file."""
        walker = _make_fourbar_walker()
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            path = f.name

        save_walker(walker, path)
        loaded = load_walker(path)

        self.assertEqual(loaded.name, walker.name)
        self.assertEqual(
            set(loaded.topology.nodes.keys()),
            set(walker.topology.nodes.keys()),
        )
        Path(path).unlink()

    def test_file_is_valid_json(self):
        """Saved file is valid JSON."""
        walker = _make_fourbar_walker()
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            path = f.name

        save_walker(walker, path)
        with open(path) as f:
            data = json.load(f)

        self.assertIn("topology", data)
        self.assertIn("dimensions", data)
        self.assertEqual(data["version"], 1)
        Path(path).unlink()


class TestResultToDict(unittest.TestCase):

    def test_roundtrip(self):
        """NsgaWalkingResult survives roundtrip."""
        result = _make_nsga_result()
        data = result_to_dict(result)
        loaded = result_from_dict(data)

        self.assertEqual(
            len(loaded.pareto_front.solutions),
            len(result.pareto_front.solutions),
        )
        self.assertEqual(
            loaded.pareto_front.objective_names,
            result.pareto_front.objective_names,
        )
        self.assertEqual(loaded.config.n_generations, result.config.n_generations)
        self.assertEqual(loaded.config.seed, result.config.seed)

    def test_scores_preserved(self):
        """Solution scores are preserved."""
        result = _make_nsga_result()
        data = result_to_dict(result)
        loaded = result_from_dict(data)

        for orig, loaded_sol in zip(
            result.pareto_front.solutions, loaded.pareto_front.solutions,
        ):
            for a, b in zip(orig.scores, loaded_sol.scores):
                self.assertAlmostEqual(a, b)

    def test_dimensions_preserved(self):
        """Solution dimensions are preserved."""
        result = _make_nsga_result()
        data = result_to_dict(result)
        loaded = result_from_dict(data)

        for orig, loaded_sol in zip(
            result.pareto_front.solutions, loaded.pareto_front.solutions,
        ):
            np.testing.assert_array_almost_equal(
                orig.dimensions, loaded_sol.dimensions,
            )

    def test_json_serializable(self):
        """result_to_dict produces JSON-serializable output."""
        result = _make_nsga_result()
        data = result_to_dict(result)
        json_str = json.dumps(data)
        self.assertIsInstance(json_str, str)

    def test_empty_result(self):
        """Empty result roundtrips."""
        result = NsgaWalkingResult(
            pareto_front=ParetoFront([], ("a", "b")),
        )
        data = result_to_dict(result)
        loaded = result_from_dict(data)
        self.assertEqual(len(loaded.pareto_front.solutions), 0)


class TestSaveLoadResult(unittest.TestCase):

    def test_save_load_file(self):
        """Result saves to and loads from JSON."""
        result = _make_nsga_result()
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            path = f.name

        save_result(result, path)
        loaded = load_result(path)

        self.assertEqual(
            len(loaded.pareto_front.solutions),
            len(result.pareto_front.solutions),
        )
        Path(path).unlink()


class TestTopologyResultSerialization(unittest.TestCase):

    def test_topology_info_roundtrip(self):
        """TopologyWalkingResult with topology info roundtrips."""
        rng = np.random.RandomState(42)
        solutions = [
            ParetoSolution(
                scores=(5.0, 3.0),
                dimensions=rng.uniform(0.5, 2.0, 5),
                init_positions=[(0.0, 0.0)],
            ),
        ]
        result = TopologyWalkingResult(
            pareto_front=ParetoFront(solutions, ("dist", "stab")),
            topology_info={
                0: TopologySolutionInfo(
                    topology_name="Four-bar linkage",
                    topology_id="four-bar",
                    topology_idx=0,
                    num_links=4,
                ),
            },
            config=NsgaWalkingConfig(n_generations=10, pop_size=5),
        )

        data = result_to_dict(result)
        self.assertIn("topology_info", data)

        loaded = result_from_dict(data)
        self.assertIsInstance(loaded, TopologyWalkingResult)
        self.assertIn(0, loaded.topology_info)
        self.assertEqual(loaded.topology_info[0].topology_id, "four-bar")
        self.assertEqual(loaded.topology_info[0].num_links, 4)


if __name__ == "__main__":
    unittest.main()
