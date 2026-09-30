import re
import torch
from nltk.corpus import wordnet

COMPOUND_WORDS = {
    "trafficlight": "traffic light",
    "trafficsign": "traffic sign",
    "pottedplant": "potted plant",
    "tvmonitor": "tv monitor",
    "windowpane": "window pane",
    "coffeetable": "coffee table",
    "streetlight": "street light",
    "signboard": "sign board",
    "arcademachine": "arcade machine",
    "kitchenisland": "kitchen island",
    "dirttrack": "dirt track",
    "chestofdrawers": "chest of drawers",
    "televisionreceiver": "television receiver",
    "conveyerbelt": "conveyor belt",
    "bedclothes": "bed clothes",
    "sportsball": "sports ball",
    "baseballbat": "baseball bat",
    "baseballglove": "baseball glove",
    "tennisracket": "tennis racket",
    "wineglass": "wine glass",
    "cellphone": "cell phone",
    "hairdrier": "hair drier",
    "firehydrant": "fire hydrant",
    "stopsign": "stop sign",
    "parkingmeter": "parking meter",
    "aeroplane": "airplane",
    "diningtable": "dining table",
    "motorbike": "motorcycle",
}

WORD_ORDER = {
    "wallbrick": "brick wall",
    "wallconcrete": "concrete wall",
    "wallstone": "stone wall",
    "walltile": "tile wall",
    "wallwood": "wooden wall",
    "wallpanel": "wall panel",
    "floormarble": "marble floor",
    "floorstone": "stone floor",
    "floortile": "tile floor",
    "floorwood": "wooden floor",
    "ceilingtile": "tile ceiling",
    "windowblind": "window blind",
}

AMBIGUOUS_NAMES = {
    "tie": "necktie",
    "remote": "remote control",
    "mouse": "computer mouse",
    "monitor": "computer monitor",
    "ashcan": "trash can",
    "plaything": "toy",
    "buffet": "sideboard",
}

CANONICAL_NAMES = {**COMPOUND_WORDS, **WORD_ORDER, **AMBIGUOUS_NAMES}

TEXT_KEYS = {"language_features": 1, "language_mask": 0, "language_embeds": 1}  # key -> prompt (batch) dimension


def load_vocabulary(path):
    """One class per line; comma-separated names on a line are aliases of that class.

    Returns the prompt names and, for each prompt, the index of its class.
    """
    names, class_ids = [], []
    with open(path) as f:
        for class_id, line in enumerate(f):
            aliases = [name.strip() for name in line.split(",")]
            names += aliases
            class_ids += [class_id] * len(aliases)
    return names, class_ids


def canonicalize(name):
    """Split compound words, repair word order and disambiguate homonyms, e.g. "pottedplant" -> "potted plant"."""
    return CANONICAL_NAMES.get(re.sub(r"[^a-z0-9]+", "", name.lower()), name)


def wordnet_hypernyms(name, max_count):
    """The first hypernyms of the most common noun sense of `name`."""
    synsets = wordnet.synsets(name.strip().lower().replace(" ", "_").replace("-", "_"), pos=wordnet.NOUN)
    if not synsets:
        return []
    lemmas = [h.lemmas()[0].name().replace("_", " ") for h in synsets[0].hypernyms()[:max_count]]
    return list(dict.fromkeys(lemma for lemma in lemmas if lemma))


def mean_token_embedding(text):
    """Average of the token embeddings of each prompt, [K, 1024]."""
    embeds = text["language_embeds"]
    valid = (~text["language_mask"]).to(embeds.dtype).transpose(0, 1).unsqueeze(-1)
    return (embeds * valid).sum(dim=0) / valid.sum(dim=0).clamp_min(1.0)


class TextEncoder:
    """SAM 3's text encoder with a per-prompt cache.

    Names are canonicalized before encoding when `canonical` is set. Every name is encoded once, in the batch of the
    first call that contains it.
    """

    def __init__(self, backbone, device, canonical=True):
        self.backbone = backbone
        self.device = device
        self.canonical = canonical
        self.cache = {}

    def __call__(self, names):
        if self.canonical:
            names = [canonicalize(name) for name in names]
        missing = [name for name in names if name not in self.cache]
        if missing:
            text = self.backbone.forward_text(missing, device=self.device)
            for i, name in enumerate(missing):
                self.cache[name] = {key: text[key].narrow(dim, i, 1).clone() for key, dim in TEXT_KEYS.items()}
        return {key: torch.cat([self.cache[name][key] for name in names], dim=dim) for key, dim in TEXT_KEYS.items()}


class ContextTokens:
    """Contextual prompt expansion (CPE): text tokens appended to every prompt.

    Each prompt gets the similarity-weighted embeddings of its nearest prompts in the vocabulary and the embeddings of
    its WordNet hypernyms, projected to SAM 3's prompt width. A hypernym slot a prompt does not fill is masked out.
    """

    def __init__(self, text_encoder, names, projection, num_neighbors, num_hypernyms):
        with torch.inference_mode():
            with torch.autocast("cuda", dtype=torch.bfloat16):
                pooled = mean_token_embedding(text_encoder(names)).float()
            neighbors = self._neighbor_tokens(pooled, projection, num_neighbors)
            hypernyms, missing = self._hypernym_tokens(names, text_encoder, projection, num_hypernyms)
        self.tokens = torch.cat([neighbors, hypernyms], dim=1)
        self.mask = torch.cat([torch.zeros_like(neighbors[..., 0], dtype=torch.bool), missing], dim=1)

    def __call__(self, prompt_ids):
        """Tokens [M, K, 256] and padding mask [K, M] for a batch of prompts."""
        return self.tokens[prompt_ids].transpose(0, 1), self.mask[prompt_ids]

    @staticmethod
    def _neighbor_tokens(pooled, projection, num_tokens):
        unit = pooled / pooled.norm(dim=-1, keepdim=True).clamp_min(1e-8)
        similarity = unit @ unit.T
        similarity.fill_diagonal_(-float("inf"))
        values, neighbors = similarity.topk(num_tokens, dim=1)
        return projection(pooled[neighbors] * values.softmax(dim=1).unsqueeze(-1)).float()

    @staticmethod
    def _hypernym_tokens(names, text_encoder, projection, num_tokens):
        hypernyms = [wordnet_hypernyms(canonicalize(name), num_tokens) for name in names]
        unique = list(dict.fromkeys(h for per_name in hypernyms for h in per_name))
        if not unique:
            empty = torch.zeros(len(names), 0, projection.out_features, device=projection.weight.device)
            return empty, empty[..., 0].bool()

        with torch.autocast("cuda", dtype=torch.bfloat16):
            pooled = mean_token_embedding(text_encoder(unique)).float()
        table = pooled.new_zeros(len(names), num_tokens, pooled.shape[-1])
        found = torch.zeros(len(names), num_tokens, dtype=torch.bool, device=pooled.device)
        for i, per_name in enumerate(hypernyms):
            for slot, hypernym in enumerate(per_name):
                table[i, slot] = pooled[unique.index(hypernym)]
                found[i, slot] = True
        tokens = projection(table.to(projection.weight.dtype)).float()
        return tokens.masked_fill(~found[..., None], 0), ~found
