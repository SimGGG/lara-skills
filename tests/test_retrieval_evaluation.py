"""Global retrieval differs from ordering inside the judged pool."""

import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from lara_skills.retrieval_evaluation import retrieval_metrics


class RetrievalTests(unittest.TestCase):
    def inputs(self):
        queries = pd.DataFrame({"query_id": [0, 1, 2], "sample_id": ["a", "b", "c"], "perspective": ["develop", "use", "use"]})
        labels = pd.DataFrame({"query_id": [0, 0, 0, 1], "candidate_id": [0, 1, 2, 0], "relevance_score": [2, 1, 0, 1]})
        return queries, labels

    def test_unjudged_candidates_can_outrank_all_known_positives(self):
        q, labels = self.inputs()
        scores = np.array([[3., 2., 1., 9., 8., 7.]] * 3)
        summary, detail, ranks, top = retrieval_metrics(scores, q, labels, (2, 4, 6))
        row = summary[(summary.positive_grade_min == 1) & (summary.k == 2)].iloc[0]
        self.assertEqual(row.recall_macro, 0.)
        self.assertEqual(row.hit_macro, 0.)
        self.assertEqual(row.unjudged_slots, 6)
        self.assertEqual(row.known_grade0_slots, 0)
        # Within judged candidates, 2 > 1 > 0 is perfect; globally positives are 4th/5th.
        self.assertEqual(ranks[(ranks.query_id == 0) & (ranks.candidate_id == 0)].full_universe_rank.iloc[0], 4)
        row = summary[(summary.positive_grade_min == 1) & (summary.k == 4)].iloc[0]
        self.assertEqual(row.recall_macro, .75)
        self.assertEqual(row.hit_macro, 1.)

    def test_no_known_positives_is_undefined_not_failure(self):
        q, labels = self.inputs()
        summary, detail, _, _ = retrieval_metrics(np.zeros((3, 6)), q, labels, (2,))
        self.assertTrue(detail.loc[detail.query_id == 2, 'known_positive_recall'].isna().all())
        self.assertEqual(summary[summary.positive_grade_min == 1].eligible_queries.iloc[0], 2)
        self.assertEqual(summary[summary.positive_grade_min == 2].eligible_queries.iloc[0], 1)

    def test_stable_ties_and_full_universe_recall(self):
        q, labels = self.inputs()
        summary, _, _, top = retrieval_metrics(np.zeros((3, 6)), q, labels, (1, 6))
        np.testing.assert_array_equal(top[0], np.arange(6))
        self.assertTrue(summary.loc[summary.k == 6, 'recall_macro'].eq(1.).all())

    def test_invalid_cutoffs_scores_and_labels(self):
        q, labels = self.inputs()
        for ks in [(0,), (7,), (2, 2)]:
            with self.assertRaises(ValueError):
                retrieval_metrics(np.zeros((3, 6)), q, labels, ks)
        labels.loc[0, 'relevance_score'] = np.nan
        with self.assertRaises(ValueError):
            retrieval_metrics(np.zeros((3, 6)), q, labels, (2,))


if __name__ == "__main__":
    unittest.main()
