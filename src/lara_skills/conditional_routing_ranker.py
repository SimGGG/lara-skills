"""Content-preserving patent queries and query-conditioned occupation routes."""

import math

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint

from .network_attention_ranker import NetworkAttentionRanker


class ConditionalRoutingRanker(NetworkAttentionRanker):
    """Three relation families; all candidates and original graph edges retained."""

    def __init__(self, input_dim, perspective_count, config, hierarchy, occupations):
        super().__init__(input_dim, perspective_count, config, hierarchy, occupations)
        if self.score_mode != "bounded_residual":
            raise ValueError("Conditional routing requires bounded_residual scoring")
        self.mode = "hgt_conditional"
        self.occupation_route_mode = config.get("occupation_route_mode", "conditional")
        if self.occupation_route_mode not in {"conditional", "fixed"}:
            raise ValueError("Unknown occupation route mode")
        self.query_bound = float(config.get("content_query_bound", .1))
        self.semantic_temperature = float(config.get("semantic_attention_temperature", .05))
        self.route_temperature = float(config.get("occupation_route_temperature", .1))
        self.learned_attention_bound = float(config.get("learned_attention_bound", .1))
        self.route_chunk = config.get("routing_query_chunk", 8)
        rank = config.get("routing_rank", 16)
        for value in [self.semantic_temperature, self.route_temperature]:
            if not math.isfinite(value) or value <= 0:
                raise ValueError("Routing temperatures must be finite and positive")
        if not math.isfinite(self.query_bound) or not 0 <= self.query_bound < 1:
            raise ValueError("content_query_bound must be in [0, 1)")
        if not math.isfinite(self.learned_attention_bound) or self.learned_attention_bound < 0:
            raise ValueError("learned_attention_bound must be finite and nonnegative")
        if not isinstance(self.route_chunk, int) or self.route_chunk < 1 or not isinstance(rank, int) or rank < 1:
            raise ValueError("Routing rank and chunk must be positive integers")
        hidden = config["hidden_dim"]
        # These parent modules are not used by the new content query encoder.
        del self.skill_layers
        del self.query_neighbor_projection
        self.content_adapter = nn.Sequential(nn.Linear(input_dim + hidden, 32), nn.GELU(), nn.Linear(32, input_dim))
        nn.init.zeros_(self.content_adapter[-1].weight)
        nn.init.zeros_(self.content_adapter[-1].bias)
        self.family_gate = nn.Sequential(nn.Linear(input_dim + hidden, 32), nn.GELU(), nn.Linear(32, 3))
        nn.init.zeros_(self.family_gate[-1].weight)
        nn.init.zeros_(self.family_gate[-1].bias)
        self.route_query = nn.Linear(input_dim, rank, bias=False)
        self.route_occupation_key = nn.Linear(hidden, rank, bias=False)
        self.route_skill_key = nn.Linear(hidden, rank, bias=False)
        # Parent route is [skill, occupation], normalized over each occupation.
        edge = self.route_occupation.coalesce()
        self.register_buffer("route_skill_ids", edge.indices()[0], persistent=False)
        self.register_buffer("route_occupation_ids", edge.indices()[1], persistent=False)
        self.register_buffer("route_log_weight", edge.values().log(), persistent=False)
        reference = torch.sparse.mm(self.route_occupation,
                                   (self.routable_occupation.float() / self.routable_occupation.sum().clamp_min(1))[:, None])[:, 0]
        self.register_buffer("occupation_reference", reference, persistent=False)
        taxonomy_reference = []
        for i in range(self.level_count):
            route = getattr(self, f"route_taxonomy_{i}")
            taxonomy_reference.append(torch.sparse.mm(route, torch.full((route.shape[1], 1), 1. / route.shape[1]))[:, 0])
        self.register_buffer("taxonomy_reference", torch.stack(taxonomy_reference).mean(0), persistent=False)

    def encode_market(self, features, adjacency):
        states = self.encode_graph(features, adjacency)
        raw = {"skill": features, "occupation": F.normalize(self.occupation_features, dim=1)}
        for i in range(self.level_count):
            raw[f"taxonomy_{i}"] = F.normalize(torch.sparse.mm(getattr(self, f"pool_{i}"), features), dim=1)
        edge_keys = F.normalize(self.route_occupation_key(states["occupation"])[self.route_occupation_ids] *
                                self.route_skill_key(states["skill"])[self.route_skill_ids], dim=1)
        route = adjacency.transpose(0, 1).coalesce()
        isolated = torch.sparse.sum(adjacency, dim=1).to_dense() == 0
        uniform = features.new_full((1, len(features)), 1. / len(features))
        reference = self.cooccurrence_route(uniform, route, isolated)[0]
        return {"states": states, "raw": raw, "edge_keys": edge_keys,
                "cooccurrence_route": route, "cooccurrence_isolated": isolated, "cooccurrence_reference": reference}

    @staticmethod
    def cooccurrence_route(attention, route, isolated):
        propagated = torch.sparse.mm(route, attention.T).T + attention * isolated[None, :]
        # Keep a direct skill path, including isolated/unobserved candidates.
        return .5 * (attention + propagated)

    def content_queries(self, features, perspective_ids):
        work = .1 * F.normalize(self.perspective(perspective_ids), dim=1)
        inputs = torch.cat([features, work], dim=1)
        delta = self.content_adapter(inputs)
        delta = delta / delta.norm(dim=1, keepdim=True).clamp_min(1.)
        anchor = F.normalize(features + self.query_bound * delta, dim=1)
        return anchor, F.normalize(self.query_projection(anchor), dim=1), inputs

    def content_attention(self, anchor, projected, market):
        learned_query = F.normalize(self.node_query(projected), dim=1)
        result = {}
        for node_type in self.node_types:
            valid = (torch.ones(len(market["raw"][node_type]), dtype=torch.bool, device=anchor.device)
                     if node_type == "skill" else getattr(self, f"routable_{node_type}"))
            if not bool(valid.any()):
                continue
            learned_key = F.normalize(self.node_key[node_type](market["states"][node_type]), dim=1)
            logits = (anchor @ market["raw"][node_type].T + self.learned_attention_bound *
                      (learned_query @ learned_key.T)) / self.semantic_temperature
            if self.node_attention_mode == "uniform":
                logits = torch.zeros_like(logits)
            result[node_type] = torch.softmax(logits.masked_fill(~valid[None, :], -torch.inf), dim=1)
        return result

    def occupation_route_probabilities(self, semantic, route_query, edge_keys):
        """[query, edge] probabilities normalized over outgoing edges per occupation."""
        index = self.route_occupation_ids[None, :].expand(len(semantic), -1)
        logits = self.route_log_weight[None, :] + (semantic[:, self.route_skill_ids] +
                  self.learned_attention_bound * (route_query @ edge_keys.T)) / self.route_temperature
        maxima = logits.new_full((len(logits), len(self.occupation_features)), -torch.inf)
        maxima.scatter_reduce_(1, index, logits.detach(), reduce="amax", include_self=True)
        values = (logits - maxima.gather(1, index)).exp()
        totals = values.new_zeros(maxima.shape).scatter_add(1, index, values)
        return values / totals.gather(1, index).clamp_min(torch.finfo(values.dtype).tiny)

    def _route_chunk(self, occupation_attention, semantic, query, edge_keys):
        weights = self.occupation_route_probabilities(semantic, query, edge_keys)
        mass = weights * occupation_attention[:, self.route_occupation_ids]
        return semantic.new_zeros(semantic.shape).scatter_add(1, self.route_skill_ids[None, :].expand_as(mass), mass)

    def conditional_occupation_route(self, attention, semantic, anchor, edge_keys):
        if self.occupation_route_mode == "fixed":
            return torch.sparse.mm(self.route_occupation, attention.T).T
        query = F.normalize(self.route_query(anchor), dim=1)
        parts = []
        for start in range(0, len(semantic), self.route_chunk):
            stop = start + self.route_chunk
            args = (attention[start:stop], semantic[start:stop], query[start:stop], edge_keys)
            if self.training and self.config.get("checkpoint_scoring", False) and torch.is_grad_enabled():
                parts.append(checkpoint(self._route_chunk, *args, use_reentrant=False))
            else:
                parts.append(self._route_chunk(*args))
        return torch.cat(parts)

    def encode_queries(self, query_features, perspective_ids, skill_features, skill_hidden):
        anchor, projected, inputs = self.content_queries(query_features, perspective_ids)
        attention = self.content_attention(anchor, projected, skill_hidden)
        cooccurrence = self.cooccurrence_route(attention["skill"], skill_hidden["cooccurrence_route"],
                                               skill_hidden["cooccurrence_isolated"])
        occupation = (self.conditional_occupation_route(attention["occupation"], anchor @ skill_features.T,
                                                        anchor, skill_hidden["edge_keys"])
                      if "occupation" in attention else torch.zeros_like(cooccurrence))
        taxonomy = torch.stack([torch.sparse.mm(getattr(self, f"route_taxonomy_{i}"),
                                                attention[f"taxonomy_{i}"].T).T for i in range(self.level_count)]).mean(0)
        gate_logits = self.family_gate(inputs)
        valid = torch.tensor([True, "occupation" in attention, True], device=anchor.device)
        gates = torch.softmax(gate_logits.masked_fill(~valid[None, :], -torch.inf), dim=1)
        families = torch.stack([cooccurrence, occupation, taxonomy], dim=1)
        references = torch.stack([skill_hidden["cooccurrence_reference"], self.occupation_reference, self.taxonomy_reference])
        return {"text": projected, "content": anchor, "attention": attention, "family_weights": gates,
                "family_mass": families, "skill_mass": (families * gates[:, :, None]).sum(1),
                "reference_mass": gates @ references}

    def score_pairs(self, query_hidden, skill_hidden, query_features, skill_features,
                    frequency_feature, query_ids, skill_ids):
        q = query_hidden["text"][query_ids]
        s = F.normalize(skill_hidden["states"]["skill"][skill_ids], dim=1)
        cosine = (query_features[query_ids] * skill_features[skill_ids]).sum(1)
        features = torch.cat([q, s, q * s, torch.abs(q - s), cosine[:, None], frequency_feature[skill_ids, None]], dim=1)
        mass = query_hidden["skill_mass"][query_ids, skill_ids].clamp_min(1e-12)
        reference = query_hidden["reference_mass"][query_ids, skill_ids].clamp_min(1e-12)
        residual = self.decoder(features).squeeze(-1) + torch.tanh(self.network_scale_raw) * (mass / reference).log()
        return self.combine_scores(cosine, residual)

    def attention_diagnostics(self, query_hidden):
        rows = super().attention_diagnostics(query_hidden)
        for row in rows:
            qid, node_type = row["query_id"], row["node_type"]
            family = 0 if node_type == "skill" else 1 if node_type == "occupation" else 2
            row["type_mass"] = float(query_hidden["family_weights"][qid, family]) / (self.level_count if family == 2 else 1)
        return rows
