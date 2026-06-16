"""Contextual Prompt Expansion (CPE).

Augments each bare class name with context from three sources: lexical
canonicalization, semantic-neighbour tokens, and WordNet-hypernym
tokens. The neighbour and hypernym tokens are appended to
the decoder's text sequence; the lexical axis is an input-side string
substitution. M = 2 per axis.
"""
from __future__ import annotations

from .lexical import install_lexical_canonicalizer  # noqa: F401 (re-exported)
from .memory_bank import CompositeContextBank, ContextMemoryBank
from .neighbor_tokens import SemanticNeighborBank
from .hypernym_tokens import WordNetHypernymBank


def build_context_memory_bank(
    neighbor_tokens: int = 2,
    hypernym_tokens: int = 2,
) -> ContextMemoryBank:
    """Build the composite semantic-neighbour + WordNet-hypernym context bank."""
    return CompositeContextBank([
        SemanticNeighborBank(num_tokens=neighbor_tokens),
        WordNetHypernymBank(num_tokens=hypernym_tokens),
    ])
