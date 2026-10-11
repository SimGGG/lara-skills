"""LARA: labor-market-aware relational attention for technology-to-skill recommendation."""

from ._version import __version__
from .models import MODEL_NAMES, VARIANTS, make_ranker, variant_model
from .recommender import RecommendationEngine

__all__ = ["__version__", "MODEL_NAMES", "VARIANTS", "RecommendationEngine", "make_ranker", "variant_model"]
