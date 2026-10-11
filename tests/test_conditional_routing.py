"""Conditional edge normalization, content bounds and inductive graph reuse."""

import sys
import unittest
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F
from scipy import sparse

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from lara_skills.models import make_ranker
from lara_skills.ranking import torch_adjacency
from test_retrieval_training import CFG
from test_occupations import fixture


class ConditionalRoutingTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(42); torch.set_num_threads(1)
        self.h, self.o, self.a = fixture()
        self.x = F.normalize(torch.randn(4, 6), dim=1)
        self.q = F.normalize(torch.randn(2, 6), dim=1)
        self.p = torch.tensor([0, 0]); self.f = torch.zeros(4)

    def model(self, **extra):
        return make_ranker('hgt_conditional', 6, 2, {**CFG, **extra}, self.h, self.o).eval()

    def test_baseline_content_bound_and_family_mass(self):
        m = self.model()
        h = m.encode_market(self.x, self.a)
        z = m.encode_queries(self.q, self.p, self.x, h)
        torch.testing.assert_close(z['content'], self.q)
        torch.testing.assert_close(m.score_all(self.q, self.p, self.x, self.a, self.f), self.q @ self.x.T)
        torch.testing.assert_close(z['skill_mass'].sum(1), torch.ones(2))
        torch.testing.assert_close(z['reference_mass'].sum(1), torch.ones(2))
        torch.testing.assert_close(z['family_mass'].sum(2), torch.ones(2, 3))
        torch.testing.assert_close(z['family_weights'], torch.full((2, 3), 1/3))
        with torch.no_grad():
            m.content_adapter[-1].weight.fill_(100.)
            m.decoder[-1].weight.fill_(100.)
        anchor, _, _ = m.content_queries(self.q, self.p)
        self.assertTrue(((anchor * self.q).sum(1) >= (1 - m.query_bound**2)**.5 - 1e-6).all())
        values = m.score_all(self.q, self.p, self.x, self.a, self.f)
        self.assertLessEqual(float((values - self.q @ self.x.T).abs().max().detach()), .100001)

    def test_same_occupation_routes_to_different_skills_for_different_patents(self):
        m = self.model(learned_attention_bound=0.)
        x = torch.eye(6)[:4]; q = x[[1, 2]]
        h = m.encode_market(x, self.a)
        anchor, _, _ = m.content_queries(q, self.p)
        weights = m.occupation_route_probabilities(anchor @ x.T, F.normalize(m.route_query(anchor), dim=1), h['edge_keys'])
        torch.testing.assert_close(weights.sum(1), torch.ones(2))
        mass = m.conditional_occupation_route(torch.tensor([[1., 0.], [1., 0.]]), anchor @ x.T, anchor, h['edge_keys'])
        self.assertGreater(float(mass[0, 1].detach()), float(mass[0, 2].detach()))
        self.assertGreater(float(mass[1, 2].detach()), float(mass[1, 1].detach()))
        torch.testing.assert_close(mass[:, [0, 3]], torch.zeros(2, 2))
        torch.testing.assert_close(mass.sum(1), torch.ones(2))

    def test_chunk_batch_and_gradient_consistency(self):
        m = self.model(routing_query_chunk=1, dropout=.2).train()
        plain = self.model(routing_query_chunk=1, dropout=.2, checkpoint_scoring=False).train()
        with torch.no_grad():
            m.network_scale_raw.fill_(.2); m.decoder[-1].weight.normal_(0, .05)
        plain.load_state_dict(m.state_dict())
        outs=[]
        for model in [m, plain]:
            torch.manual_seed(16)
            y=model.score_all(self.q,self.p,self.x,self.a,self.f,3)
            (y[0,1]-y[1,2]).backward();outs.append(y.detach())
        torch.testing.assert_close(*outs)
        for (_,a),(_,b) in zip(m.named_parameters(),plain.named_parameters()):
            if a.grad is not None: torch.testing.assert_close(a.grad,b.grad,atol=1e-6,rtol=1e-4)
        for parameter in [m.route_query.weight,m.family_gate[-1].weight,m.content_adapter[-1].weight,
                          m.attention_layers[0].message_transform['occupation_to_skill']]:
            self.assertIsNotNone(parameter.grad)
            self.assertTrue(torch.isfinite(parameter.grad).all())
            self.assertGreater(float(parameter.grad.abs().sum()),0.)
        m.eval()
        with torch.no_grad():
            together=m.score_all(self.q,self.p,self.x,self.a,self.f,3)
            alone=m.score_all(self.q[:1],self.p[:1],self.x,self.a,self.f,7)
        torch.testing.assert_close(together[:1],alone)

    def test_missing_edges_and_new_market_size(self):
        m=self.model()
        self.o['skill_to_occupation']=sparse.csr_matrix((2,4))
        self.o['occupation_to_skill']=sparse.csr_matrix((4,2));self.o['available'][:]=False
        empty=self.model();empty.load_state_dict(m.state_dict())
        z=empty.encode_queries(self.q,self.p,self.x,empty.encode_market(self.x,self.a))
        torch.testing.assert_close(z['family_weights'][:,1],torch.zeros(2))
        torch.testing.assert_close(z['skill_mass'].sum(1),torch.ones(2))
        h,o,a=fixture(n=5,occupations=3)
        other=make_ranker('hgt_conditional',6,2,CFG,h,o).eval();other.load_state_dict(m.state_dict())
        y=other.score_all(self.q,self.p,F.normalize(torch.randn(5,6),dim=1),a,torch.zeros(5))
        self.assertEqual(y.shape,(2,5));self.assertTrue(torch.isfinite(y).all())
        self.assertFalse(any('route_skill_ids' in k or 'reference' in k for k in m.state_dict()))

    def test_reordering_and_invalid_settings(self):
        m=self.model();order=[3,1,0,2]
        h,o=dict(self.h),dict(self.o)
        h['membership']=h['membership'][order];h['pools']=[p[:,order] for p in h['pools']]
        o['skill_to_occupation']=o['skill_to_occupation'][:,order];o['occupation_to_skill']=o['occupation_to_skill'][order]
        o['available']=o['available'][order]
        other=make_ranker('hgt_conditional',6,2,CFG,h,o).eval();other.load_state_dict(m.state_dict())
        a=torch_adjacency(sparse.csr_matrix(self.a.to_dense().numpy()[order][:,order]),'cpu')
        with torch.no_grad():
            m.network_scale_raw.fill_(.2);other.load_state_dict(m.state_dict())
            first=m.score_all(self.q,self.p,self.x,self.a,self.f)
            second=other.score_all(self.q,self.p,self.x[order],a,self.f[order])
        torch.testing.assert_close(first[:,order],second)
        with self.assertRaises(ValueError):self.model(content_query_bound=1.)
        with self.assertRaises(ValueError):self.model(occupation_route_temperature=0.)


if __name__=='__main__':unittest.main()
