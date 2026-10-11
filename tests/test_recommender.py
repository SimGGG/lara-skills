"""Ranking, missing-judgment semantics and inductive inference contracts."""

import math
import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from lara_skills.evaluation import full_ranking_coverage, judged_metrics, ranking_pairs
from lara_skills.ranking import SkillRanker, baseline_scores, pairwise_ranking_loss, torch_adjacency
from lara_skills.recommender_data import load_development_split


class RecommenderTests(unittest.TestCase):
    def test_training_loss_averages_tasks_not_pairs(self):
        labels = pd.DataFrame({"query_id": [0, 0, 1, 1, 1], "relevance_score": [2, 0, 1, 0, 0]})
        high, low, weights, count = ranking_pairs(labels)
        scores = torch.zeros(5, requires_grad=True)
        loss = pairwise_ranking_loss(scores, torch.tensor(high), torch.tensor(low), torch.tensor(weights), count)
        self.assertAlmostEqual(float(loss.detach()), 1.5 * math.log(2), places=6)
        loss.backward()
        self.assertLess(float(scores.grad[0]), 0)
        self.assertGreater(float(scores.grad[1]), 0)

    def test_metrics_distinguish_unknown_from_negative(self):
        labels = pd.DataFrame({"query_id": [0, 0], "sample_id": ["P", "P"], "perspective": ["A", "A"],
                               "candidate_id": [0, 1], "relevance_score": [2, 0]})
        report, _ = judged_metrics(labels, np.array([3., 1.]))
        self.assertAlmostEqual(report["perspective_macro"]["judged_ndcg_at_5"], 1.)
        coverage = full_ranking_coverage(labels, np.array([[2., 1., 3.]]), k=2)
        self.assertEqual(coverage["unjudged_slots"], 1)
        self.assertEqual(coverage["judgment_coverage"], .5)
        self.assertEqual(coverage["relevant_fraction_among_judged_slots"], 1.)
        self.assertFalse(coverage["full_universe_precision_or_recall_estimated"])

    def test_inference_is_query_independent_and_scores_isolates(self):
        cfg = {"hidden_dim": 4, "decoder_dim": 4, "layers": 2, "dropout": 0., "query_neighbors": 2, "query_temperature": .1}
        torch.manual_seed(42)
        x = torch.nn.functional.normalize(torch.rand(3, 4), dim=1)
        query = torch.nn.functional.normalize(torch.rand(2, 4), dim=1)
        a = torch_adjacency(sparse.csr_matrix([[0, 1, 0], [1, 0, 0], [0, 0, 0]], dtype=np.float32), "cpu")
        for graph in [False, True]:
            model = SkillRanker(4, 2, cfg, graph=graph).eval()
            with torch.no_grad():
                scores = model.score_all(query, torch.tensor([0, 1]), x, a, torch.zeros(3), chunk_size=2)
                alone = model.score_all(query[:1], torch.tensor([0]), x, a, torch.zeros(3), chunk_size=5)
            self.assertEqual(scores.shape, (2, 3))
            self.assertTrue(torch.isfinite(scores).all())
            torch.testing.assert_close(scores[:1], alone)

    def test_diffusion_preserves_isolated_semantic_score(self):
        x = np.eye(3, dtype=np.float32)
        q = np.array([[.1, .2, .9]], dtype=np.float32)
        a = sparse.csr_matrix([[0, 1, 0], [1, 0, 0], [0, 0, 0]], dtype=np.float32)
        scores = baseline_scores("diffusion", q, x, np.array([2, 2, 0]), a, alpha=.5)
        np.testing.assert_allclose(scores, [[.15, .15, .9]])

    def test_model_selection_loader_rejects_test(self):
        with self.assertRaisesRegex(ValueError, "test split"):
            load_development_split(Path("nonexistent"), {}, "test", None)


if __name__ == "__main__":
    unittest.main()
