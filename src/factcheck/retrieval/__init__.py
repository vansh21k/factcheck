"""Retrieval stage: query expansion, hybrid search, rank fusion, and reranking.

See ``expand.py``, ``fusion.py``, ``hybrid.py``, and ``rerank.py`` for the pieces;
``hybrid.RetrievalPipeline`` is the composed entry point the ``FactChecker`` calls.
"""

from __future__ import annotations

from .expand import IdentityExpander, NegationAwareExpander
from .fusion import ReciprocalRankFusion, WeightedScoreFusion, get_fusion
from .hybrid import HybridRetriever, RetrievalPipeline
from .rerank import LLMListwiseReranker, PassThroughReranker, get_reranker

__all__ = [
    "HybridRetriever",
    "IdentityExpander",
    "LLMListwiseReranker",
    "NegationAwareExpander",
    "PassThroughReranker",
    "ReciprocalRankFusion",
    "RetrievalPipeline",
    "WeightedScoreFusion",
    "get_fusion",
    "get_reranker",
]
