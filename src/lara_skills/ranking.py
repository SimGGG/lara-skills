"""Shared feature rankers with optional inductive GraphSAGE context."""

import math
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from .graph_encoder import SelfAndNeighborLayer


def torch_adjacency(matrix, device):
    coo = matrix.tocoo()
    indices = torch.from_numpy(np.vstack([coo.row, coo.col]).astype(np.int64))
    return torch.sparse_coo_tensor(indices, torch.from_numpy(coo.data.astype(np.float32)), matrix.shape).coalesce().to(device)


class NoWorkTypeEmbedding(nn.Module):
    """Parameter-free zero contribution for a single recommendation task."""

    def __init__(self, hidden):
        super().__init__()
        self.register_buffer("zero", torch.zeros(hidden), persistent=False)

    def forward(self, ids):
        if ids.ndim != 1 or bool((ids != 0).any()):
            raise ValueError("Single-task queries require the internal zero task ID")
        return self.zero.expand(len(ids), -1)


class SkillRanker(nn.Module):
    """No trainable skill/patent IDs; scores every skill including isolates."""

    def __init__(self, input_dim, perspective_count, config, graph=False):
        super().__init__()
        self.graph = graph
        self.config = dict(config)
        hidden = config["hidden_dim"]
        dims = [input_dim] + [hidden] * config["layers"]
        self.skill_layers = nn.ModuleList([
            SelfAndNeighborLayer(a, b) if graph else nn.Linear(a, b)
            for a, b in zip(dims[:-1], dims[1:])])
        self.query_projection = nn.Linear(input_dim, hidden)
        task_mode = config.get("task_mode", "work_types")
        if task_mode not in {"single", "work_types"} or (task_mode == "single" and perspective_count != 1):
            raise ValueError("Invalid task mode or work type count")
        self.perspective = NoWorkTypeEmbedding(hidden) if task_mode == "single" else nn.Embedding(perspective_count, hidden)
        if graph:
            self.query_neighbor_projection = nn.Linear(hidden, hidden, bias=False)
        self.dropout = nn.Dropout(config["dropout"])
        self.decoder = nn.Sequential(nn.Linear(4 * hidden + 2, config["decoder_dim"]), nn.ReLU(),
                                     nn.Dropout(config["dropout"]), nn.Linear(config["decoder_dim"], 1))
        self.score_mode = config.get("score_mode", "legacy")
        if self.score_mode not in {"legacy", "bounded_residual"}:
            raise ValueError("Unknown score_mode")
        self.residual_bound = float(config.get("residual_bound", .1))
        if not math.isfinite(self.residual_bound) or self.residual_bound < 0:
            raise ValueError("residual_bound must be finite and nonnegative")
        if self.score_mode == "bounded_residual":
            # Start exactly at the frozen semantic baseline, including epoch 0.
            nn.init.zeros_(self.decoder[-1].weight)
            nn.init.zeros_(self.decoder[-1].bias)

    def combine_scores(self, cosine, residual):
        if self.score_mode == "bounded_residual":
            return cosine + self.residual_bound * torch.tanh(residual)
        return cosine + residual

    def encode_market(self, features, adjacency):
        hidden = features
        for layer in self.skill_layers:
            hidden = layer(hidden, adjacency) if self.graph else F.relu(layer(hidden))
            hidden = self.dropout(hidden)
        return F.normalize(hidden, dim=1)

    def encode_queries(self, query_features, perspective_ids, skill_features, skill_hidden):
        hidden = self.query_projection(query_features) + self.perspective(perspective_ids)
        if self.graph:
            semantic = query_features @ skill_features.T
            count = min(self.config["query_neighbors"], skill_features.shape[0])
            if count <= 0:
                raise ValueError("query_neighbors must be positive")
            # Semantic neighbors use text alone, never patent relevance labels.
            values, indices = torch.topk(semantic, k=count, dim=1)
            weights = torch.softmax(values / self.config["query_temperature"], dim=1)
            neighbors = (skill_hidden[indices] * weights.unsqueeze(-1)).sum(dim=1)
            hidden = hidden + self.query_neighbor_projection(neighbors)
        return F.normalize(F.relu(hidden), dim=1)

    def score_pairs(self, query_hidden, skill_hidden, query_features, skill_features,
                    frequency_feature, query_ids, skill_ids):
        q, s = query_hidden[query_ids], skill_hidden[skill_ids]
        cosine = (query_features[query_ids] * skill_features[skill_ids]).sum(dim=1)
        features = torch.cat([q, s, q * s, torch.abs(q - s), cosine[:, None],
                              frequency_feature[skill_ids, None]], dim=1)
        return self.combine_scores(cosine, self.decoder(features).squeeze(-1))

    def score_all(self, query_features, perspective_ids, skill_features, adjacency, frequency_feature,
                  chunk_size=16384):
        """All-candidate ranking; semantic query neighbors do not restrict outputs."""
        if chunk_size < 1:
            raise ValueError("chunk_size must be positive")
        skill_hidden = self.encode_market(skill_features, adjacency)
        query_hidden = self.encode_queries(query_features, perspective_ids, skill_features, skill_hidden)
        qn, sn = len(query_features), len(skill_features)
        result = torch.empty((qn, sn), device=skill_features.device)
        flat = result.view(-1)
        for offset in range(0, qn * sn, chunk_size):
            ids = torch.arange(offset, min(offset + chunk_size, qn * sn), device=skill_features.device)
            qids, sids = ids // sn, ids % sn
            args = (query_hidden, skill_hidden, query_features, skill_features, frequency_feature, qids, sids)
            if self.training and self.config.get("checkpoint_scoring", False) and torch.is_grad_enabled():
                from torch.utils.checkpoint import checkpoint
                values = checkpoint(self.score_pairs, *args, use_reentrant=False)
            else:
                values = self.score_pairs(*args)
            flat[offset:offset + len(ids)] = values
        return result


def pairwise_ranking_loss(scores, high, low, weights, task_count, temperature=1.0):
    if task_count < 1 or temperature <= 0:
        raise ValueError("Ranking loss needs informative tasks and positive temperature")
    return (weights * F.softplus(-(scores[high] - scores[low]) / temperature)).sum() / task_count


def baseline_scores(name, query_features, skill_features, posting_count, adjacency, alpha=.5):
    if name == "popularity":
        return np.broadcast_to(np.log1p(posting_count), (len(query_features), len(posting_count))).copy()
    semantic = query_features @ skill_features.T
    if name == "semantic":
        return semantic
    if name != "diffusion" or not 0 <= alpha <= 1:
        raise ValueError("Unknown baseline or invalid diffusion alpha")
    # Row-vector a @ P equals the column-vector P.T @ a used in the plan.
    result = (1 - alpha) * semantic + alpha * (adjacency.T @ semantic.T).T
    isolated = np.diff(adjacency.indptr) == 0
    result[:, isolated] = semantic[:, isolated]
    return result
