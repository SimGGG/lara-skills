"""Occupation dictionary parsing, posting marginals and the shared small graph fixture."""

import sys
import tempfile
import unittest
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
from scipy import sparse
import torch
from torch.nn import functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from lara_skills.occupations import aggregate_postings, demand_relations, occupation_dictionary
from lara_skills.hierarchy import build_hierarchy, row_normalize
from lara_skills.ranking import torch_adjacency

CFG = {"hidden_dim": 8, "decoder_dim": 8, "layers": 2, "dropout": 0., "query_neighbors": 2,
       "query_temperature": .1, "gate_dim": 4, "gate_temperature": 1.}


def fixture(n=4, occupations=2):
    candidates = pd.DataFrame({"candidate_id": range(n), "skill_code": [str(i) for i in range(n)], "main": ["A"] * n})
    edges = pd.DataFrame({"source_node": [0], "target_node": [1], "weight": [1.]})
    hierarchy = build_hierarchy(candidates, edges, ["main"])
    forward = sparse.coo_matrix(([1., 2.], ([0, 0], [1, 2])), shape=(occupations, n)).tocsr()
    bundle = {"features": F.normalize(torch.randn(occupations, 6), dim=1).numpy(),
              "skill_to_occupation": row_normalize(forward), "occupation_to_skill": row_normalize(forward.T),
              "available": np.diff(forward.T.tocsr().indptr) > 0}
    graph = sparse.coo_matrix(([1., 1.], ([0, 1], [1, 0])), shape=(n, n))
    return hierarchy, bundle, torch_adjacency(graph, "cpu")


class OccupationTests(unittest.TestCase):
    def test_merged_dictionary_paths_and_leading_zero(self):
        frame = pd.DataFrame({"code": [111, 112], "name": ["first", "second"], "main": ["A", None]})
        config = {"path_columns": ["main"], "code_column": "code", "name_column": "name", "code_width": 4,
                  "embedding_template": "{name}: {path}"}
        nodes = occupation_dictionary(frame, config)
        self.assertEqual(nodes.occupation_code.tolist(), ["0111", "0112"])
        self.assertEqual(nodes.embedding_text.tolist(), ["first: A", "second: A"])
        frame.loc[1, "code"] = 111
        with self.assertRaisesRegex(ValueError, "duplicate"):
            occupation_dictionary(frame, config)

    def test_empty_skill_postings_in_denominator_and_no_double_count(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "postings.parquet"
            pd.DataFrame({"occupation_code": ["01", "01", "01", "02"],
                          "skills": [["a", "b"], ["a"], [], ["b"]]}).to_parquet(path)
            with duckdb.connect() as con:
                counts, skill_counts, pairs = aggregate_postings(con, [path])
        nodes = pd.DataFrame({"occupation_id": [0, 1, 2], "occupation_code": ["01", "02", "03"]})
        candidates = pd.DataFrame({"candidate_id": [0, 1, 2], "skill_code": ["a", "b", "c"]})
        n, e, forward, reverse = demand_relations(nodes, candidates, counts, skill_counts, pairs, 1)
        edge = e[(e.occupation_code == "01") & (e.skill_code == "a")].iloc[0]
        self.assertAlmostEqual(edge.weight, 2 / np.sqrt(3 * 2))
        self.assertEqual(n.posting_count.tolist(), [3, 1, 0])
        np.testing.assert_allclose(np.asarray(forward.sum(1)).ravel(), [1, 1, 0])
        self.assertEqual(reverse.shape, (3, 3))
        n, e, forward, reverse = demand_relations(nodes, candidates, counts, skill_counts, pairs, 2)
        self.assertEqual(len(e), 1)
        self.assertEqual(reverse.getrow(2).nnz, 0)
        bad = counts.copy(); bad.loc[0, "occupation_code"] = "unknown"
        with self.assertRaisesRegex(ValueError, "Unknown"):
            demand_relations(nodes, candidates, bad, skill_counts, pairs, 1)


if __name__ == "__main__":
    unittest.main()
