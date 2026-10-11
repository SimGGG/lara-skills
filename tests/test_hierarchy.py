"""Taxonomy identity, size-corrected market pooling and empty-market behavior."""

import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from lara_skills.hierarchy import build_hierarchy


def fixture():
    candidates = pd.DataFrame({"candidate_id": range(5), "skill_code": list("abcde"),
                               "main": ["A", "A", "B", "B", "C"],
                               "detail": ["shared", "shared", "shared", "other", "solo"]})
    edges = pd.DataFrame({"source_node": [0, 1, 0], "target_node": [2, 3, 4], "weight": [1., 3., 2.]})
    return candidates, edges


class HierarchyTests(unittest.TestCase):
    def test_path_identity_and_size_corrected_market(self):
        candidates, edges = fixture()
        h = build_hierarchy(candidates, edges, ["main", "detail"])
        self.assertEqual([len(p.toarray()) for p in h["pools"]], [3, 4])
        self.assertNotEqual(h["membership"][0, 1], h["membership"][2, 1])
        np.testing.assert_allclose(h["pools"][0].toarray(), [[.5, .5, 0, 0, 0], [0, 0, .5, .5, 0], [0, 0, 0, 0, 1]])
        np.testing.assert_allclose(h["coarse"][0].toarray(), [[0, .5, .5], [1, 0, 0], [1, 0, 0]])

    def test_missing_taxonomy_and_duplicate_edges_are_rejected(self):
        candidates, edges = fixture()
        candidates.loc[0, "detail"] = None
        with self.assertRaisesRegex(ValueError, "Missing"):
            build_hierarchy(candidates, edges, ["main", "detail"])
        candidates, _ = fixture()
        with self.assertRaisesRegex(ValueError, "each undirected edge once"):
            build_hierarchy(candidates, pd.concat([edges, edges.iloc[:1]]), ["main"])

    def test_empty_market_keeps_all_taxonomy_paths(self):
        candidates, edges = fixture()
        h = build_hierarchy(candidates, edges.iloc[:0], ["main", "detail"])
        self.assertFalse(h["local_market_available"].any())
        self.assertTrue(all(m.nnz == 0 for m in h["coarse"]))
        self.assertEqual(h["membership"].shape, (5, 2))


if __name__ == "__main__":
    unittest.main()
