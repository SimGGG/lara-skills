"""Candidate coverage, split leakage and isolated-node behavior."""

import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from lara_skills.training_inputs import candidate_graph, group_patents, assign_splits
from lara_skills.graph_encoder import SelfAndNeighborLayer


class TrainingInputTests(unittest.TestCase):
    def test_candidate_universe_survives_missing_and_isolated_nodes(self):
        index = pd.DataFrame({"skill_code": list("ABCD"), "embedding_row": [0, 1, 2, 3]})
        graph = pd.DataFrame({"node_id": [0, 1, 2], "skill_code": ["B", "A", "C"], "posting_count": [5, 4, 1]})
        edges = pd.DataFrame({"source_node": [0], "target_node": [1], "weight": [.5]})
        nodes, remapped, adjacency = candidate_graph(index, graph, edges)
        self.assertEqual(nodes.skill_code.tolist(), list("ABCD"))
        self.assertEqual(nodes.graph_status.tolist(), ["connected", "connected", "isolated", "unobserved"])
        self.assertEqual((remapped.source_node.iloc[0], remapped.target_node.iloc[0]), (1, 0))
        np.testing.assert_array_equal(adjacency.toarray(), [[0, 1, 0, 0], [1, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]])

    def test_isolated_representation_depends_only_on_self_features(self):
        layer = SelfAndNeighborLayer(2, 2)
        with torch.no_grad():
            layer.self_projection.weight.copy_(torch.eye(2))
            layer.self_projection.bias.zero_()
            layer.neighbor_projection.weight.copy_(torch.eye(2))
        adjacency = torch.sparse_coo_tensor(torch.tensor([[0, 1], [1, 0]]), torch.ones(2), (3, 3)).coalesce()
        features = torch.tensor([[1., 2.], [3., 4.], [5., 6.]])
        result = layer(features, adjacency)
        torch.testing.assert_close(result[2], features[2])
        changed = features.clone()
        changed[:2] *= 100
        torch.testing.assert_close(layer(changed, adjacency)[2], result[2])
        self.assertFalse(torch.equal(layer(changed, adjacency)[0], result[0]))
        result.sum().backward()
        self.assertTrue(torch.isfinite(layer.self_projection.weight.grad).all())

    def test_transitive_application_family_and_exact_text_groups(self):
        frame = pd.DataFrame({"id": ["p1", "p2", "p3", "p4", "p5"],
                              "application": ["a1", "a1", "a3", "a4", "a5"],
                              "family": ["f1", "f2", "f2", None, None],
                              "year": [2019, 2019, 2020, 2021, 2021],
                              "text": ["first", "second", "THIRD text", "third  text", "unique"]})
        grouped = group_patents(frame, {c: c for c in frame.columns})
        self.assertEqual(grouped.group_id.nunique(), 2)
        self.assertEqual(grouped.iloc[:4].group_id.nunique(), 1)
        split, _ = assign_splits(grouped, {"train": .6, "validation": .2, "test": .2}, 42)
        self.assertEqual(split.iloc[:4].split.nunique(), 1)

    def test_balanced_splits_reproducible_without_labels(self):
        n = 200
        frame = pd.DataFrame({"id": [f"P{i:03d}" for i in range(n)],
                              "application": [f"A{i}" for i in range(n)],
                              "family": [f"F{i}" for i in range(n)],
                              "year": np.repeat(np.arange(2019, 2024), 40),
                              "text": [f"unique technology {i}" for i in range(n)]})
        grouped = group_patents(frame, {c: c for c in frame.columns})
        first, summary = assign_splits(grouped, {"train": .6, "validation": .2, "test": .2}, 42)
        second, _ = assign_splits(group_patents(frame.sample(frac=1, random_state=5), {c: c for c in frame.columns}),
                                  {"train": .6, "validation": .2, "test": .2}, 42)
        self.assertTrue(summary["year_balance_exact"])
        self.assertEqual(first.split.value_counts().to_dict(), {"train": 120, "validation": 40, "test": 40})
        pd.testing.assert_frame_equal(first, second)
        for field in ["application", "family", "text_hash", "group_id"]:
            self.assertEqual(first.groupby(field).split.nunique().max(), 1)


if __name__ == "__main__":
    unittest.main()
