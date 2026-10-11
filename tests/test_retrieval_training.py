"""Partial supervision, baseline preservation, gradients and checkpoint selection."""

import contextlib
import io
import json
import sys
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
from lara_skills.models import make_ranker
from lara_skills.retrieval_objective import partial_retrieval_targets, retrieval_selection_key
from lara_skills.retrieval_objective import retrieval_training_loss
from test_attention_ranker import CONFIG
from test_occupations import fixture

CFG = {**CONFIG, "training_objective": "partial_label_retrieval", "score_mode": "bounded_residual",
       "residual_bound": .1, "retrieval_temperature": .03, "positive_target_mass": .5,
       "rank_temperature": .03, "auxiliary_rank_weight": .05, "residual_regularization": .01,
       "checkpoint_scoring": True}


class RetrievalTrainingTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(14); torch.set_num_threads(1)

    def test_unknown_soft_mass_and_only_explicit_zeros_are_removed(self):
        cosine = torch.zeros(3, 4)
        target = partial_retrieval_targets(cosine, torch.tensor([0, 0, 1]), torch.tensor([0, 1, 2]),
                                           torch.tensor([2, 0, 0]), .03, .5)
        torch.testing.assert_close(target.sum(1), torch.ones(3))
        self.assertEqual(float(target[0, 1]), 0.)
        self.assertGreater(float(target[0, 2]), 0.)
        self.assertGreater(float(target[0, 0]), float(target[0, 2]))
        torch.testing.assert_close(target[2], torch.full((4,), .25))
        torch.testing.assert_close(target[1, [0, 1, 3]], torch.full((3,), 1 / 3))
        with self.assertRaisesRegex(ValueError, "Duplicate"):
            partial_retrieval_targets(cosine, torch.tensor([0, 0]), torch.tensor([1, 1]), torch.tensor([0, 2]), .03, .5)

    def test_full_dictionary_loss_detects_unjudged_intruder(self):
        cosine = torch.zeros(1, 4)
        qi, si = torch.tensor([0, 0]), torch.tensor([0, 1])
        target = partial_retrieval_targets(cosine, qi, si, torch.tensor([2, 0]), .03, .5)
        a = torch.tensor([[.08, -.01, 0., 0.]], requires_grad=True)
        b = a.detach().clone(); b[0, 2] = .3; b.requires_grad_()
        args = (cosine, target, torch.tensor([0]), qi, si, torch.tensor([0]), torch.tensor([1]), torch.ones(1), 1, CFG)
        la, _ = retrieval_training_loss(a, *args)
        lb, _ = retrieval_training_loss(b, *args)
        self.assertGreater(float(lb.detach()), float(la.detach()))
        lb.backward()
        self.assertGreater(float(b.grad[0, 2]), 0.)
        self.assertLess(float(b.grad[0, 0]), 0.)

    def test_baseline_at_initialization_bounds_and_uniform_reference(self):
        h, o, a = fixture()
        x, q = F.normalize(torch.randn(4, 6), dim=1), F.normalize(torch.randn(2, 6), dim=1)
        p, f = torch.tensor([0, 1]), torch.zeros(4)
        for name in ['hgt_network', 'hgt_conditional']:
            model = make_ranker(name, 6, 2, CFG, h, o).eval()
            torch.testing.assert_close(model.score_all(q, p, x, a, f, 3), q @ x.T)
            with torch.no_grad():
                model.decoder[-1].weight.fill_(100.)
                model.network_scale_raw.fill_(2.)
            values = model.score_all(q, p, x, a, f, 3)
            self.assertLessEqual(float((values - q @ x.T).abs().max().detach()), .100001)
            model.residual_bound = 0.
            torch.testing.assert_close(model.score_all(q, p, x, a, f), q @ x.T)
        model = make_ranker('hgt_network', 6, 2, {**CFG, 'node_attention_mode': 'uniform'}, h, o).eval()
        with torch.no_grad():
            model.network_scale_raw.fill_(5.)
            z = model.encode_queries(q, p, x, model.encode_market(x, a))
            torch.testing.assert_close(z['skill_mass'], model.uniform_skill_mass[None, :].expand(2, -1))
            torch.testing.assert_close(model.score_all(q, p, x, a, f), q @ x.T)

    def test_checkpointed_scoring_has_same_gradients_and_graph_path_learns(self):
        h, o, a = fixture()
        x, q = F.normalize(torch.randn(4, 6), dim=1), F.normalize(torch.randn(2, 6), dim=1)
        models = [make_ranker('hgt_network', 6, 2, {**CFG, 'dropout': .2, 'checkpoint_scoring': v}, h, o).train() for v in [False, True]]
        with torch.no_grad():
            models[0].decoder[-1].weight.normal_(0, .05)
            models[0].network_scale_raw.fill_(.1)
        models[1].load_state_dict(models[0].state_dict())
        values = []
        for model in models:
            torch.manual_seed(77)
            scores = model.score_all(q, torch.tensor([0, 1]), x, a, torch.zeros(4), 3)
            (scores[0, 0] - scores[1, 2]).backward()
            values.append(scores.detach())
        torch.testing.assert_close(*values)
        for (name, one), (_, two) in zip(models[0].named_parameters(), models[1].named_parameters()):
            if one.grad is not None:
                torch.testing.assert_close(one.grad, two.grad, atol=1e-6, rtol=1e-4)
        for parameter in [models[1].node_query.weight, models[1].attention_layers[0].message_transform['occupation_to_skill']]:
            self.assertGreater(float(parameter.grad.abs().sum()), 0.)

    def test_selection_is_recall_then_hit(self):
        def key(r, h):
            return retrieval_selection_key(pd.DataFrame([{'positive_grade_min': 1, 'k': 5, 'recall_macro': r, 'hit_macro': h}]))
        self.assertGreater(key(.2, .1), key(.1, .9))
        self.assertGreater(key(.2, .3), key(.2, .1))
        self.assertFalse(key(.2, .3) > key(.2, .3))

    def test_training_roundtrip_and_no_test_access(self):
        self.training_roundtrip(single=False)

    def test_single_task_training_roundtrip_and_no_test_access(self):
        self.training_roundtrip(single=True)

    def training_roundtrip(self, single):
        from lara_skills.train_retrieval import fit_retrieval_recommenders
        from lara_skills.recommender import RecommendationEngine
        h, o, a = fixture(n=6)
        candidates = pd.DataFrame({'candidate_id': range(6), 'skill_code': list('abcdef'),
                                   'skill_name': list('ABCDEF'), 'posting_count': [5, 4, 3, 2, 1, 0]})
        x, q = F.normalize(torch.randn(6, 6), dim=1).numpy(), F.normalize(torch.randn(2, 6), dim=1).numpy()
        market = candidates, x, sparse.csr_matrix(a.to_dense().numpy()), np.zeros(6, dtype=np.float32)
        called = []
        def loader(root, config, split, candidates):
            self.assertIn(split, ['train', 'validation']); called.append(split)
            queries = pd.DataFrame({'sample_id': [split] * 2, 'perspective': ['dev', 'use'],
                                    'perspective_id': [0, 1], 'query_id': [0, 1]})
            labels = pd.DataFrame({'sample_id': [split] * 4, 'perspective': ['dev', 'dev', 'use', 'use'],
                                   'query_id': [0, 0, 1, 1], 'candidate_id': [0, 1, 2, 3],
                                   'skill_code': list('abcd'), 'relevance_score': [2, 0, 1, 1]})
            if single:
                queries['sample_id'] = [split + '0', split + '1']
                queries['perspective'] = 'skills'
                queries['perspective_id'] = 0
                labels['sample_id'] = [split + '0'] * 2 + [split + '1'] * 2
                labels['perspective'] = 'skills'
            return {'queries': queries, 'features': q.copy() if split == 'validation' else -q.copy(), 'labels': labels}
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            config = {**CFG, 'candidate_dir': 'candidates', 'hierarchy_dir': 'hierarchy', 'occupation_dir': 'occupations',
                      'patent_embedding_dir': 'patents', 'split_dir': 'splits', 'output_dir': 'models', 'device': 'cpu',
                      'seed': 42, 'cpu_threads': 1, 'top_k': 5, 'epochs': 2, 'learning_rate': .001,
                      'weight_decay': .0001, 'gradient_clip': 5., 'diffusion_alphas': [.5, 0.], 'score_chunk_size': 3,
                      'retrieval_ks': [5], 'training_models': ['hgt_network', 'hgt_conditional'], 'default_engine': 'hgt_conditional'}
            encoding = {'model_id': 'fixture', 'revision': 'fixed', 'perspectives': ['dev', 'use']}
            if single:
                config['task_mode'] = 'single'
                encoding.update(task_mode='single', perspectives=['skills'], query_template='{description}')
            for name in ['candidates', 'hierarchy', 'occupations', 'patents', 'splits', 'features', 'scripts']:
                (root / name).mkdir(); (root / name / 'manifest.json').write_text(json.dumps({'config': encoding}))
            for name in ['occupations', 'patents']:
                (root / name / 'encoder_config.json').write_text(json.dumps(encoding))
            (root / 'candidates/feature_reference.json').write_text(json.dumps({'path_relative_to_project_root': 'features/x.npy'}))
            (root / 'scripts/train_recommender.py').write_text('# fixture\n')
            with contextlib.ExitStack() as stack:
                for module in ['train_retrieval', 'recommender']:
                    prefix = f'lara_skills.{module}.'
                    stack.enter_context(patch(prefix + 'load_market', return_value=market))
                    stack.enter_context(patch(prefix + 'load_hierarchy', return_value=h))
                    stack.enter_context(patch(prefix + 'load_occupations', return_value=o))
                stack.enter_context(patch('lara_skills.train_retrieval.load_development_split', side_effect=loader))
                with contextlib.redirect_stdout(io.StringIO()):
                    fit_retrieval_recommenders(root, config)
                for name in config['training_models']:
                    engine = RecommendationEngine(root, 'models', model=name)
                    scores = engine.score_embeddings(q, np.array([0, 0] if single else [0, 1]))
                    np.testing.assert_allclose(scores, np.load(root / f'models/{name}_validation_all_scores.npy'), atol=1e-6)
                    history = pd.read_csv(root / f'models/{name}_history.csv')
                    best = history.sort_values(['validation_known_recall_at_5', 'validation_known_hit_at_5', 'epoch'], ascending=[False, False, True]).iloc[0]
                    checkpoint = torch.load(root / f'models/{name}.pt', weights_only=True)
                    self.assertEqual(checkpoint['best_epoch'], best.epoch)
                    self.assertGreaterEqual(best.validation_known_recall_at_5, history.iloc[0].validation_known_recall_at_5)
            self.assertEqual(called, ['train', 'validation'])
            self.assertFalse(list(root.glob('*.partial_*')))


if __name__ == '__main__':
    unittest.main()
