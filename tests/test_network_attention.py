"""Actual node selection, directed routing, gradient and unseen-query inference."""

import sys
import unittest
from pathlib import Path

import torch
from torch.nn import functional as F
from scipy import sparse

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from lara_skills.attention_ranker import AttentionRanker
from lara_skills.models import make_ranker
from lara_skills.ranking import torch_adjacency
from test_attention_ranker import CONFIG
from test_occupations import fixture


class NetworkAttentionTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(37)
        torch.set_num_threads(1)
        self.h, self.o, self.a = fixture()
        self.x = F.normalize(torch.randn(4, 6), dim=1)
        self.q = F.normalize(torch.randn(2, 6), dim=1)
        self.p = torch.tensor([0, 0])
        self.f = torch.zeros(4)

    def model(self, config=None):
        return make_ranker("hgt_network", 6, 2, config or CONFIG, self.h, self.o).eval()

    def test_node_distributions_change_with_query_and_route_correctly(self):
        model = self.model()
        market = model.encode_market(self.x, self.a)
        z = model.encode_queries(self.q, self.p, self.x, market)
        for weights in z["attention"].values():
            torch.testing.assert_close(weights.sum(1), torch.ones(2))
        torch.testing.assert_close(z["skill_mass"].sum(1), torch.ones(2))
        self.assertFalse(torch.allclose(z["attention"]["skill"][0], z["attention"]["skill"][1]))
        self.assertEqual(float(z["attention"]["occupation"][:, 1].sum().detach()), 0.)
        # Source occupation 0 has links to skills 1 and 2 with weights 1:2.
        routed = model.route_attention({"occupation": torch.tensor([[1., 0.]])})
        torch.testing.assert_close(routed, torch.tensor([[0., 1 / 3, 2 / 3, 0.]]))
        # Selecting a different skill node must alter candidate-specific scores.
        ids = torch.arange(4)
        qids = torch.zeros(4, dtype=torch.long)
        left = {**z, "skill_mass": model.route_attention({"skill": torch.tensor([[1., 0., 0., 0.]])})}
        right = {**z, "skill_mass": model.route_attention({"skill": torch.tensor([[0., 0., 0., 1.]])})}
        delta = (model.score_pairs(left, market, self.q, self.x, self.f, qids, ids) -
                 model.score_pairs(right, market, self.q, self.x, self.f, qids, ids))
        self.assertGreater(float(delta[0].detach()), 0.)
        self.assertLess(float(delta[3].detach()), 0.)
        torch.testing.assert_close(delta[1:3], torch.zeros(2))

    def test_gradient_isolates_and_batch_chunk_independence(self):
        model = self.model()
        scores = model.score_all(self.q, self.p, self.x, self.a, self.f, chunk_size=3)
        self.assertTrue(torch.isfinite(scores).all())
        F.softplus(scores[0, 1] - scores[0, 3]).backward()
        for parameter in [model.node_query.weight, model.node_key["skill"].weight, model.network_scale_raw,
                          model.attention_layers[0].message_transform["occupation_to_skill"]]:
            self.assertIsNotNone(parameter.grad)
            self.assertTrue(torch.isfinite(parameter.grad).all())
            self.assertGreater(float(parameter.grad.abs().sum()), 0.)
        with torch.no_grad():
            alone = model.score_all(self.q[:1], self.p[:1], self.x, self.a, self.f, chunk_size=7)
        torch.testing.assert_close(scores[:1], alone)

    def test_same_backbone_and_checkpoint_without_node_ids(self):
        torch.manual_seed(42)
        base = AttentionRanker(6, 2, CONFIG, self.h, self.o, "hgt")
        torch.manual_seed(42)
        model = self.model()
        for key, value in base.state_dict().items():
            torch.testing.assert_close(value, model.state_dict()[key], rtol=0, atol=0)
        self.assertFalse(any("route_" in k or "routable_" in k for k in model.state_dict()))
        h, o, a = fixture(n=5, occupations=3)
        other = make_ranker("hgt_network", 6, 2, CONFIG, h, o).eval()
        other.load_state_dict(model.state_dict())
        with torch.no_grad():
            scores = other.score_all(self.q, self.p, F.normalize(torch.randn(5, 6), dim=1), a, torch.zeros(5))
        self.assertEqual(scores.shape, (2, 5))
        self.assertTrue(torch.isfinite(scores).all())

    def test_uniform_attention_and_all_occupation_edges_absent(self):
        uniform = self.model({**CONFIG, "node_attention_mode": "uniform"})
        z = uniform.encode_queries(self.q, self.p, self.x, uniform.encode_market(self.x, self.a))
        torch.testing.assert_close(z["attention"]["skill"], torch.full((2, 4), .25))
        torch.testing.assert_close(z["skill_mass"][0], z["skill_mass"][1])
        self.o["skill_to_occupation"] = sparse.csr_matrix((2, 4))
        self.o["occupation_to_skill"] = sparse.csr_matrix((4, 2))
        self.o["available"][:] = False
        model = self.model()
        z = model.encode_queries(self.q, self.p, self.x, model.encode_market(self.x, self.a))
        self.assertNotIn("occupation", z["attention"])
        torch.testing.assert_close(z["skill_mass"].sum(1), torch.ones(2))

    def test_node_reordering_preserves_scores(self):
        model = self.model()
        h, o = dict(self.h), dict(self.o)
        order = [3, 1, 0, 2]
        h["membership"] = h["membership"][order]
        h["pools"] = [pool[:, order] for pool in h["pools"]]
        o["skill_to_occupation"] = o["skill_to_occupation"][:, order]
        o["occupation_to_skill"] = o["occupation_to_skill"][order]
        o["available"] = o["available"][order]
        other = make_ranker("hgt_network", 6, 2, CONFIG, h, o).eval()
        other.load_state_dict(model.state_dict())
        permuted = torch_adjacency(sparse.csr_matrix(self.a.to_dense().numpy()[order][:, order]), "cpu")
        with torch.no_grad():
            before = model.score_all(self.q, self.p, self.x, self.a, self.f)
            after = other.score_all(self.q, self.p, self.x[order], permuted, self.f[order])
        torch.testing.assert_close(before[:, order], after)

    def test_invalid_temperature_and_routing(self):
        with self.assertRaisesRegex(ValueError, "temperature"):
            self.model({**CONFIG, "node_attention_temperature": float("nan")})
        self.o["skill_to_occupation"] = self.o["skill_to_occupation"] * 2
        with self.assertRaisesRegex(ValueError, "sum to one"):
            self.model()

    def test_large_group_normalization_uses_precise_validation(self):
        h, o, _ = fixture(n=3000)
        model = make_ranker("hgt_network", 6, 2, CONFIG, h, o)
        self.assertEqual(model.route_taxonomy_0.shape, (3000, 1))


if __name__ == "__main__":
    unittest.main()
