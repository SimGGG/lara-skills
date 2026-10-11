"""Patent-to-node attention routed to skills through fixed market relations."""

import math

import torch
from torch import nn
from torch.nn import functional as F

from .attention_ranker import AttentionRanker
from .ranking import torch_adjacency


class NetworkAttentionRanker(AttentionRanker):
    """HGT baseline plus a query-dependent, candidate-specific network prior.

    Each node type receives equal total mass; within a type, the patent query
    attends to actual nodes. Column-stochastic routes distribute that mass to
    skills. No patent IDs, labels, or query-to-gold edges enter inference.
    """

    def __init__(self, input_dim, perspective_count, config, hierarchy, occupations):
        super().__init__(input_dim, perspective_count, config, hierarchy, occupations, "hgt")
        self.mode = "hgt_network"
        self.node_temperature = float(config.get("node_attention_temperature", .2))
        self.node_attention_mode = config.get("node_attention_mode", "learned")
        if not math.isfinite(self.node_temperature) or self.node_temperature <= 0:
            raise ValueError("Node attention temperature must be finite and positive")
        if self.node_attention_mode not in {"learned", "uniform"}:
            raise ValueError("Node attention mode must be learned or uniform")
        hidden = config["hidden_dim"]
        self.node_query = nn.Linear(hidden, hidden, bias=False)
        self.node_key = nn.ModuleDict({t: nn.Linear(hidden, hidden, bias=False) for t in self.node_types})
        self.network_scale_raw = nn.Parameter(torch.tensor(math.log(math.expm1(1.))))
        if self.score_mode == "bounded_residual":
            nn.init.zeros_(self.network_scale_raw)
        # These matrices already normalize over the outgoing skills per node.
        # Transpose yields [target skill, source node], without changing weights.
        routes = {"occupation": occupations["skill_to_occupation"].T.tocsr()}
        routes.update({f"taxonomy_{i}": pool.T.tocsr() for i, pool in enumerate(hierarchy["pools"])})
        for node_type, matrix in routes.items():
            route = torch_adjacency(matrix, "cpu")
            # Large taxonomy groups accumulate noticeable float32 roundoff.
            # Validate the stored weights in float64; inference stays float32.
            totals = torch.sparse.sum(route.to(dtype=torch.float64), dim=0).to_dense()
            valid = totals > 0
            # Validate routing rather than silently normalizing a wrong direction.
            if not torch.allclose(totals[valid], torch.ones_like(totals[valid]), atol=1e-5):
                raise ValueError(f"Node-to-skill routes for {node_type} must sum to one per connected source node")
            self.register_buffer(f"route_{node_type}", route, persistent=False)
            self.register_buffer(f"routable_{node_type}", valid, persistent=False)
        if self.score_mode == "bounded_residual":
            # A reference with the SAME routing removes small-group mass bias.
            uniform = {"skill": torch.full((1, len(hierarchy["membership"])),
                                          1. / len(hierarchy["membership"]))}
            for node_type in self.node_types:
                if node_type == "skill":
                    continue
                valid = getattr(self, f"routable_{node_type}")
                if valid.any():
                    uniform[node_type] = (valid.float() / valid.sum())[None, :]
            self.register_buffer("uniform_skill_mass", self.route_attention(uniform)[0], persistent=False)

    def encode_market(self, features, adjacency):
        return {"states": self.encode_graph(features, adjacency), "self_text": self.self_encode(features)}

    def encode_queries(self, query_features, perspective_ids, skill_features, skill_hidden):
        # Reuse the baseline's query processing and semantic attachment exactly.
        z = super().encode_queries(query_features, perspective_ids, skill_features,
                                   skill_hidden["self_text"][:, None, :])
        attention = self.node_attention(z, skill_hidden["states"])
        return {"text": z, "attention": attention, "skill_mass": self.route_attention(attention)}

    def node_attention(self, query_hidden, states):
        q = F.normalize(self.node_query(query_hidden), dim=1)
        result = {}
        for node_type in self.node_types:
            valid = (torch.ones(len(states[node_type]), dtype=torch.bool, device=q.device)
                     if node_type == "skill" else getattr(self, f"routable_{node_type}"))
            if not bool(valid.any()):
                continue
            keys = F.normalize(self.node_key[node_type](states[node_type]), dim=1)
            logits = q @ keys.T / self.node_temperature
            if self.node_attention_mode == "uniform":
                logits = torch.zeros_like(logits)
            result[node_type] = torch.softmax(logits.masked_fill(~valid[None, :], -torch.inf), dim=1)
        return result

    def route_attention(self, attention):
        # Skill mass stays at the attended skill; its HGT state already encodes
        # the cooccurrence neighborhood. Other types route through membership.
        routed = []
        for node_type, weights in attention.items():
            routed.append(weights if node_type == "skill" else
                          torch.sparse.mm(getattr(self, f"route_{node_type}"), weights.T).T)
        return torch.stack(routed).mean(0)

    def score_pairs(self, query_hidden, skill_hidden, query_features, skill_features,
                    frequency_feature, query_ids, skill_ids):
        q = query_hidden["text"][query_ids]
        s = F.normalize(skill_hidden["states"]["skill"][skill_ids], dim=1)
        cosine = (query_features[query_ids] * skill_features[skill_ids]).sum(1)
        decoder_input = torch.cat([q, s, q * s, torch.abs(q - s), cosine[:, None],
                                   frequency_feature[skill_ids, None]], dim=1)
        mass = query_hidden["skill_mass"][query_ids, skill_ids]
        if self.score_mode == "bounded_residual":
            # Uniform attention has zero excess evidence for every skill.
            reference = self.uniform_skill_mass[skill_ids]
            prior = torch.log(mass.clamp_min(1e-12) / reference.clamp_min(1e-12))
            residual = self.decoder(decoder_input).squeeze(-1) + torch.tanh(self.network_scale_raw) * prior
            return self.combine_scores(cosine, residual)
        prior = torch.log1p(len(skill_features) * mass)
        return cosine + self.decoder(decoder_input).squeeze(-1) + F.softplus(self.network_scale_raw) * prior

    def attention_diagnostics(self, query_hidden):
        """Per-query diagnostics; node indices refer to the bound graph bundle."""
        rows = []
        for node_type, weights in query_hidden["attention"].items():
            probability, index = weights.max(dim=1)
            entropy = -(weights * weights.clamp_min(torch.finfo(weights.dtype).tiny).log()).sum(1)
            for qid in range(len(weights)):
                rows.append({"query_id": qid, "node_type": node_type,
                             "top_node_index": int(index[qid]), "top_node_probability": float(probability[qid]),
                             "entropy": float(entropy[qid]), "type_mass": 1. / len(query_hidden["attention"])})
        return rows
