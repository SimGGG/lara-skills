"""Model variants exposed by the application and the ranker factory.

``lara`` is the full model of the paper (patent-conditioned attention, three
relation routes, query-dependent gate, bounded correction). ``lara-na`` keeps the
patent-conditioned node attention but routes evidence with fixed market weights.
The internal identifiers are kept because checkpoints and configs record them.
"""

from .conditional_routing_ranker import ConditionalRoutingRanker
from .network_attention_ranker import NetworkAttentionRanker

VARIANTS = {"lara": "hgt_conditional", "lara-na": "hgt_network"}
MODEL_NAMES = set(VARIANTS.values())


def variant_model(variant):
    if variant in VARIANTS:
        return VARIANTS[variant]
    if variant in MODEL_NAMES:
        return variant
    raise ValueError(f"Unknown variant {variant!r}; choose one of {sorted(VARIANTS)}")


def make_ranker(name, input_dim, perspective_count, config, hierarchy=None, occupations=None):
    if name == "hgt_conditional":
        return ConditionalRoutingRanker(input_dim, perspective_count, config, hierarchy, occupations)
    if name == "hgt_network":
        return NetworkAttentionRanker(input_dim, perspective_count, config, hierarchy, occupations)
    raise ValueError(f"Unsupported model {name!r}; expected one of {sorted(MODEL_NAMES)}")
