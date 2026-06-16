"""WordNet-hypernym context tokens.
For each class (e.g. "cat"), look up its first noun synset, take that synset's
top-M hypernyms
"""
from __future__ import annotations

from typing import List, Optional

import torch

from .memory_bank import (
    ContextMemoryBank,
    MemoryBatch,
    pooled_text_embeddings,
    project_text_to_prompt_dim,
)


def wordnet_hypernyms(name: str, max_count: int) -> List[str]:
    """Return up to ``max_count`` hypernym names for ``name``.

    Takes the first noun synset (the most common WordNet sense) and returns up
    to ``max_count`` of its immediate hypernyms. Empty list if WordNet has no
    entry (common for compound class names, which lexical canonicalization
    addresses upstream).
    """
    try:
        from nltk.corpus import wordnet
    except (ImportError, LookupError):
        return []
    candidate = name.strip().lower().replace(" ", "_").replace("-", "_")
    synsets = wordnet.synsets(candidate, pos=wordnet.NOUN)
    if not synsets:
        return []
    hypernyms = synsets[0].hypernyms()
    if not hypernyms:
        return []
    out: List[str] = []
    for h in hypernyms[:max_count]:
        lemma = h.lemmas()[0].name().replace("_", " ")
        if lemma and lemma not in out:
            out.append(lemma)
    return out


class WordNetHypernymBank(ContextMemoryBank):
    """Memory tokens from WordNet noun-hypernym lookups."""

    def __init__(self, num_tokens: int = 2):
        super().__init__()
        self.num_tokens = int(num_tokens)
        self.register_buffer("_tokens", torch.empty(0), persistent=False)
        # Which (class, slot) positions carry a hypernym (vs padding).
        self.register_buffer("_has_hyp", torch.empty(0), persistent=False)

    @torch.inference_mode()
    def precompute_global(self, query_words, pooled_text, resizer, device, model=None, **kwargs):
        """Look up hypernyms, encode them via SAM 3's text encoder, project to 256-d."""
        if model is None:
            raise RuntimeError(
                "WordNetHypernymBank.precompute_global requires `model=` to "
                "encode hypernym text."
            )
        N = len(query_words)
        M = int(self.num_tokens)
        d_text = int(pooled_text.shape[-1])

        # 1. Look up hypernyms for each class, on the canonicalized class name.
        from .lexical import _canonicalize
        hypernyms_per_class: List[List[str]] = [
            wordnet_hypernyms(_canonicalize(name, universal=True), max_count=M)
            for name in query_words
        ]

        # 2. Collect the unique set of hypernym strings.
        unique: List[str] = []
        seen = set()
        for hyps in hypernyms_per_class:
            for h in hyps:
                if h not in seen:
                    seen.add(h)
                    unique.append(h)

        if not unique:
            # No hypernyms found at all; the bank produces no tokens.
            self._tokens = torch.empty(0, device=device)
            self._has_hyp = torch.empty(0, device=device)
            return

        # 3. Encode all unique hypernyms once.
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            unique_pooled = pooled_text_embeddings(model, unique, device)
        unique_pooled = unique_pooled.to(dtype=torch.float32)        # [n_unique, 1024]
        name_to_idx = {h: i for i, h in enumerate(unique)}

        # 4. Per-class hypernym table [N, M, 1024], zero-padded for missing entries.
        tokens_1024 = torch.zeros(N, M, d_text, device=device, dtype=torch.float32)
        has_hyp = torch.zeros(N, M, device=device, dtype=torch.bool)
        for c, hyps in enumerate(hypernyms_per_class):
            for m, h in enumerate(hyps[:M]):
                tokens_1024[c, m] = unique_pooled[name_to_idx[h]]
                has_hyp[c, m] = True

        # 5. Project 1024 -> 256 via SAM 3's resizer (cast to its dtype first).
        rw_dtype = next(resizer.parameters()).dtype
        tokens_256 = project_text_to_prompt_dim(
            tokens_1024.to(dtype=rw_dtype), resizer,
        ).float()                                                    # [N, M, 256]

        self._tokens = tokens_256.contiguous().to(device=device, dtype=torch.float32)
        self._has_hyp = has_hyp.to(device=device).contiguous()

    def extra_memory_tokens(self, bucket_class_slots: List[int],
                            device: torch.device) -> Optional[MemoryBatch]:
        if self._tokens.numel() == 0:
            return None
        K = len(bucket_class_slots)
        M = int(self.num_tokens)
        d = int(self._tokens.size(-1))

        tokens = torch.zeros(M, K, d, device=device, dtype=self._tokens.dtype)
        # Mask convention: True = ignore. Unset only where the class has a hypernym.
        mask = torch.ones(K, M, device=device, dtype=torch.bool)
        for k, cls_idx in enumerate(bucket_class_slots):
            for m in range(M):
                if bool(self._has_hyp[cls_idx, m].item()):
                    tokens[m, k, :] = self._tokens[cls_idx, m].to(device=device)
                    mask[k, m] = False
        return MemoryBatch(tokens=tokens, mask=mask)
