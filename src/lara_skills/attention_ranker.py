"""Static typed graph attention and patent-conditioned skill readout."""

import math

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from .ranking import SkillRanker, torch_adjacency

NETWORK_ATTENTION_MODELS = {"hgt_network", "hgt_conditional"}
ATTENTION_MODELS = {"hgt", "hgt_query"} | NETWORK_ATTENTION_MODELS


def incoming_softmax(logits, targets, count):
    """Normalize jointly over ALL relation types entering each target, per head."""
    index = targets[:, None].expand_as(logits)
    maxima = logits.new_full((count, logits.shape[1]), -torch.inf)
    maxima.scatter_reduce_(0, index, logits.detach(), reduce="amax", include_self=True)
    values = torch.exp(logits - maxima[targets])
    totals = logits.new_zeros(maxima.shape).scatter_add_(0, index, values)
    return values / totals[targets].clamp_min(torch.finfo(logits.dtype).tiny)


class TypedAttentionLayer(nn.Module):
    """HGT core: typed Q/K/V, relation transforms, joint incoming attention.

    A fixed log edge-weight prior extends the original static HGT operator.
    No temporal encoding or trainable node identifiers are used.
    """

    def __init__(self, node_types, relations, hidden, heads, dropout, weight_prior):
        super().__init__()
        if heads < 1 or hidden < 1 or hidden % heads:
            raise ValueError("Hidden dimension must be divisible by positive attention heads")
        if not math.isfinite(weight_prior) or weight_prior < 0:
            raise ValueError("Attention edge prior must be finite and nonnegative")
        self.heads, self.width, self.hidden = heads, hidden // heads, hidden
        self.weight_prior, self.relations = weight_prior, dict(relations)
        self.query = nn.ModuleDict({t: nn.Linear(hidden, hidden) for t in node_types})
        self.key = nn.ModuleDict({t: nn.Linear(hidden, hidden) for t in node_types})
        self.value = nn.ModuleDict({t: nn.Linear(hidden, hidden) for t in node_types})
        self.output = nn.ModuleDict({t: nn.Linear(hidden, hidden, bias=False) for t in node_types})
        self.norm = nn.ModuleDict({t: nn.LayerNorm(hidden) for t in node_types})
        self.attention_transform = nn.ParameterDict()
        self.message_transform = nn.ParameterDict()
        self.relation_scale = nn.ParameterDict()
        for name in relations:
            self.attention_transform[name] = nn.Parameter(torch.empty(heads, self.width, self.width))
            self.message_transform[name] = nn.Parameter(torch.empty(heads, self.width, self.width))
            self.relation_scale[name] = nn.Parameter(torch.ones(heads))
            for head in range(heads):
                nn.init.xavier_uniform_(self.attention_transform[name][head])
                nn.init.xavier_uniform_(self.message_transform[name][head])
        self.dropout = nn.Dropout(dropout)

    def forward(self, states, edges):
        if set(edges) != set(self.relations):
            raise ValueError("Attention relation schema differs from the model")
        q = {t: self.query[t](x).reshape(-1, self.heads, self.width) for t, x in states.items()}
        k = {t: self.key[t](x).reshape(-1, self.heads, self.width) for t, x in states.items()}
        v = {t: self.value[t](x).reshape(-1, self.heads, self.width) for t, x in states.items()}
        incoming = {t: [] for t in states}
        for name, (source, target) in self.relations.items():
            src, dst, weight = edges[name]
            if src.numel() == 0:
                continue
            # Transform once per source node, then gather its edge occurrences.
            keys = torch.einsum("nhd,hde->nhe", k[source], self.attention_transform[name])
            values = torch.einsum("nhd,hde->nhe", v[source], self.message_transform[name])
            logits = (keys[src] * q[target][dst]).sum(-1) / math.sqrt(self.width)
            logits = logits * self.relation_scale[name]
            if self.weight_prior:
                logits = logits + self.weight_prior * weight.log()[:, None]
            incoming[target].append((dst, logits, values[src]))
        result = {}
        for target, state in states.items():
            parts = incoming[target]
            if not parts:
                result[target] = self.norm[target](state)
                continue
            dst = torch.cat([p[0] for p in parts])
            logits = torch.cat([p[1] for p in parts])
            messages = torch.cat([p[2] for p in parts])
            weights = incoming_softmax(logits, dst, len(state))
            update = state.new_zeros((len(state), self.heads, self.width))
            update.index_add_(0, dst, messages * weights[:, :, None])
            update = self.output[target](F.gelu(update.reshape(-1, self.hidden)))
            result[target] = self.norm[target](state + self.dropout(update))
        return result


class AttentionRanker(SkillRanker):
    """Same graph and decoder for HGT baseline and query-attention proposal."""

    def __init__(self, input_dim, perspective_count, config, hierarchy, occupations, mode):
        if mode not in {"hgt", "hgt_query"} or hierarchy is None or occupations is None:
            raise ValueError("Attention ranker requires hierarchy and occupation bundles")
        if config["layers"] < 1 or config["query_neighbors"] < 1 or config["query_temperature"] <= 0:
            raise ValueError("Invalid graph depth or query attachment settings")
        super().__init__(input_dim, perspective_count, config, graph=False)
        self.mode = mode
        hidden = config["hidden_dim"]
        membership = np.asarray(hierarchy["membership"])
        self.level_count = membership.shape[1]
        self.node_types = ["skill", "occupation"] + [f"taxonomy_{i}" for i in range(self.level_count)]
        if (occupations["features"].ndim != 2 or occupations["features"].shape[1] != input_dim
                or occupations["skill_to_occupation"].shape != (len(occupations["features"]), len(membership))
                or occupations["occupation_to_skill"].shape != (len(membership), len(occupations["features"]))):
            raise ValueError("Occupation and candidate dimensions differ")
        self.register_buffer("membership", torch.as_tensor(membership, dtype=torch.long), persistent=False)
        self.register_buffer("occupation_features", torch.as_tensor(occupations["features"], dtype=torch.float32), persistent=False)
        self.register_buffer("occupation_available", torch.as_tensor(occupations["available"], dtype=torch.bool), persistent=False)
        self.register_buffer("occupation_to_skill", torch_adjacency(occupations["occupation_to_skill"], "cpu"), persistent=False)
        self.input_projection = nn.ModuleDict({t: nn.Linear(input_dim, hidden) for t in self.node_types})
        relations = {"skill_cooccurrence": ("skill", "skill")}
        self.buffer_relations = []

        def relation(name, source, target, matrix):
            if not np.isfinite(matrix.data).all() or (matrix.data <= 0).any():
                raise ValueError("Attention edges require finite positive weights")
            relations[name] = (source, target)
            self.register_buffer(f"edge_{name}", torch_adjacency(matrix, "cpu"), persistent=False)
            self.buffer_relations.append(name)

        relation("skill_to_occupation", "skill", "occupation", occupations["skill_to_occupation"])
        relation("occupation_to_skill", "occupation", "skill", occupations["occupation_to_skill"])
        for i, (pool, coarse) in enumerate(zip(hierarchy["pools"], hierarchy["coarse"])):
            group = f"taxonomy_{i}"
            if pool.shape[1] != len(membership) or coarse.shape != (pool.shape[0], pool.shape[0]):
                raise ValueError("Invalid taxonomy dimensions")
            self.register_buffer(f"pool_{i}", torch_adjacency(pool, "cpu"), persistent=False)
            relation(f"skill_to_{group}", "skill", group, pool)
            # One ancestor per level; reverse membership weights equal one.
            reverse = pool.T.tocsr().copy()
            reverse.data[:] = 1.
            relation(f"{group}_to_skill", group, "skill", reverse)
            relation(f"{group}_cooccurrence", group, group, coarse)
        self.relations = relations
        self.attention_layers = nn.ModuleList([
            TypedAttentionLayer(self.node_types, relations, hidden, config.get("attention_heads", 4),
                                config["dropout"], config.get("edge_weight_prior", 1.))
            for _ in range(config["layers"])])
        self.query_neighbor_projection = nn.Linear(hidden, hidden, bias=False)
        self.view_names = ["self_text", "hgt_skill"] + [f"taxonomy_{i}" for i in range(self.level_count)] + ["occupation_context"]
        if mode == "hgt_query":
            self.readout_query = nn.Linear(hidden, hidden, bias=False)
            self.readout_key = nn.Linear(hidden, hidden, bias=False)

    def self_encode(self, features):
        hidden = features
        for layer in self.skill_layers:
            hidden = self.dropout(F.relu(layer(hidden)))
        return F.normalize(hidden, dim=1)

    def encode_graph(self, features, adjacency):
        if len(features) != len(self.membership) or adjacency.shape != (len(features), len(features)):
            raise ValueError("Candidate dimensions differ from attention graph")
        adjacency = adjacency.coalesce()
        if not torch.isfinite(adjacency.values()).all() or (adjacency.values() <= 0).any():
            raise ValueError("Attention edges require finite positive weights")
        inputs = {"skill": features, "occupation": self.occupation_features}
        for i in range(self.level_count):
            inputs[f"taxonomy_{i}"] = F.normalize(torch.sparse.mm(getattr(self, f"pool_{i}"), features), dim=1)
        states = {t: F.gelu(self.input_projection[t](x)) for t, x in inputs.items()}

        def edge_tuple(matrix):
            return matrix.indices()[1], matrix.indices()[0], matrix.values()

        edges = {name: edge_tuple(getattr(self, f"edge_{name}")) for name in self.buffer_relations}
        edges["skill_cooccurrence"] = edge_tuple(adjacency)
        for layer in self.attention_layers:
            states = layer(states, edges)
        return states

    def encode_market(self, features, adjacency):
        states = self.encode_graph(features, adjacency)
        views = [self.self_encode(features), F.normalize(states["skill"], dim=1)]
        for i in range(self.level_count):
            views.append(F.normalize(states[f"taxonomy_{i}"], dim=1)[self.membership[:, i]])
        views.append(F.normalize(torch.sparse.mm(self.occupation_to_skill, states["occupation"]), dim=1))
        return torch.stack(views, dim=1)

    def encode_queries(self, query_features, perspective_ids, skill_features, skill_hidden):
        # Keep the exact same text-only query attachment for both variants.
        hidden = self.query_projection(query_features) + self.perspective(perspective_ids)
        values, indices = torch.topk(query_features @ skill_features.T,
                                    k=min(self.config["query_neighbors"], len(skill_features)), dim=1)
        weights = torch.softmax(values / self.config["query_temperature"], dim=1)
        neighbors = (skill_hidden[indices, 0] * weights[:, :, None]).sum(1)
        return F.normalize(F.relu(hidden + self.query_neighbor_projection(neighbors)), dim=1)

    def fusion_weights(self, query_hidden, candidate_views, candidate_ids):
        if self.mode == "hgt":
            weights = candidate_views.new_zeros(candidate_views.shape[:2])
            weights[:, 1] = 1.
            return weights
        logits = (self.readout_query(query_hidden)[:, None, :] * self.readout_key(candidate_views)).sum(-1)
        logits = logits / math.sqrt(self.config["hidden_dim"])
        valid = torch.ones_like(logits, dtype=torch.bool)
        valid[:, -1] = self.occupation_available[candidate_ids]
        return torch.softmax(logits.masked_fill(~valid, -torch.inf), dim=1)

    def score_pairs(self, query_hidden, skill_hidden, query_features, skill_features,
                    frequency_feature, query_ids, skill_ids):
        q, views = query_hidden[query_ids], skill_hidden[skill_ids]
        weights = self.fusion_weights(q, views, skill_ids)
        s = F.normalize((weights[:, :, None] * views).sum(1), dim=1)
        cosine = (query_features[query_ids] * skill_features[skill_ids]).sum(1)
        decoder_input = torch.cat([q, s, q * s, torch.abs(q - s), cosine[:, None], frequency_feature[skill_ids, None]], dim=1)
        return self.combine_scores(cosine, self.decoder(decoder_input).squeeze(-1))
