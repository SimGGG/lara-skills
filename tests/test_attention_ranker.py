"""Typed attention normalization, graph sensitivity and inductive inference."""

import sys
import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
from scipy import sparse
import torch
from torch.nn import functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from lara_skills.attention_ranker import AttentionRanker, incoming_softmax
from lara_skills.ranking import torch_adjacency
from test_occupations import fixture, CFG

CONFIG = {**CFG, "attention_heads": 2, "edge_weight_prior": 1.}


class AttentionTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(104)
        torch.set_num_threads(1)

    def test_attention_joint_normalization_and_extreme_logits(self):
        logits = torch.tensor([[1000., -1000.], [1001., -999.], [-1000., 1000.]], requires_grad=True)
        targets = torch.tensor([0, 0, 2])
        weights = incoming_softmax(logits, targets, 4)
        torch.testing.assert_close(weights[:2].sum(0), torch.ones(2))
        torch.testing.assert_close(weights[2], torch.ones(2))
        self.assertTrue(torch.isfinite(weights).all())
        (weights * torch.arange(6).reshape(3, 2)).sum().backward()
        self.assertTrue(torch.isfinite(logits.grad).all())

    def test_shared_initial_backbone_and_factory(self):
        h, o, _ = fixture()
        torch.manual_seed(42)
        base = AttentionRanker(6, 2, CONFIG, h, o, "hgt")
        torch.manual_seed(42)
        proposal = AttentionRanker(6, 2, CONFIG, h, o, "hgt_query")
        for key, value in base.state_dict().items():
            torch.testing.assert_close(value, proposal.state_dict()[key])
        self.assertFalse(any("edge_" in k or k == "membership" for k in proposal.state_dict()))

    def test_gradients_batch_chunk_independence_and_isolates(self):
        h, o, a = fixture()
        model = AttentionRanker(6, 2, CONFIG, h, o, "hgt_query").eval()
        x, q = F.normalize(torch.randn(4, 6), dim=1), F.normalize(torch.randn(2, 6), dim=1)
        p, freq = torch.tensor([0, 1]), torch.zeros(4)
        scores = model.score_all(q, p, x, a, freq, chunk_size=3)
        self.assertTrue(torch.isfinite(scores).all())
        # Opposing pairs require candidate-specific learning, not only a query bias.
        (F.softplus(scores[0, 0] - scores[0, 2]) + F.softplus(scores[1, 3] - scores[1, 1])).backward()
        for parameter in [model.readout_query.weight, model.readout_key.weight,
                          model.attention_layers[0].attention_transform["skill_to_occupation"],
                          model.attention_layers[0].message_transform["occupation_to_skill"]]:
            self.assertIsNotNone(parameter.grad)
            self.assertTrue(torch.isfinite(parameter.grad).all())
            self.assertGreater(float(parameter.grad.abs().sum()), 0.)
        with torch.no_grad():
            alone = model.score_all(q[:1], p[:1], x, a, freq, chunk_size=7)
            torch.testing.assert_close(scores[:1], alone)
            views = model.encode_market(x, a)
            z = model.encode_queries(q, p, x, views)
            ids = torch.tensor([0, 2])
            weights = model.fusion_weights(z, views[ids], ids)
            torch.testing.assert_close(weights.sum(1), torch.ones(2))
            self.assertEqual(float(weights[0, -1]), 0.)
            self.assertGreater(float(weights[1, -1]), 0.)
            repeated = views[2:3].expand(2, -1, -1)
            other = model.fusion_weights(z, repeated, torch.tensor([2, 2]))
            self.assertFalse(torch.allclose(other[0], other[1]))

    def test_checkpoint_reuse_with_new_node_counts(self):
        h, o, _ = fixture()
        first = AttentionRanker(6, 2, CONFIG, h, o, "hgt_query")
        h, o, a = fixture(n=5, occupations=3)
        second = AttentionRanker(6, 2, CONFIG, h, o, "hgt_query").eval()
        second.load_state_dict(first.state_dict())
        with torch.no_grad():
            scores = second.score_all(F.normalize(torch.randn(1, 6), dim=1), torch.tensor([0]),
                                      F.normalize(torch.randn(5, 6), dim=1), a, torch.zeros(5))
        self.assertEqual(scores.shape, (1, 5))
        self.assertTrue(torch.isfinite(scores).all())

    def test_graph_changes_encoder_with_identical_parameters(self):
        h, o, a = fixture()
        model = AttentionRanker(6, 2, CONFIG, h, o, "hgt").eval()
        x = F.normalize(torch.randn(4, 6), dim=1)
        empty = torch_adjacency(sparse.csr_matrix((4, 4)), "cpu")
        with torch.no_grad():
            original, removed = model.encode_market(x, a), model.encode_market(x, empty)
        torch.testing.assert_close(original[:, 0], removed[:, 0])
        self.assertFalse(torch.allclose(original[:, 1], removed[:, 1]))

    def test_invalid_heads_and_edge_weights(self):
        h, o, a = fixture()
        with self.assertRaisesRegex(ValueError, "divisible"):
            AttentionRanker(6, 2, {**CONFIG, "attention_heads": 3}, h, o, "hgt")
        model = AttentionRanker(6, 2, CONFIG, h, o, "hgt")
        bad = torch.sparse_coo_tensor(a.indices(), -a.values(), a.shape).coalesce()
        with self.assertRaisesRegex(ValueError, "positive weights"):
            model.encode_market(torch.randn(4, 6), bad)


if __name__ == "__main__":
    unittest.main()
