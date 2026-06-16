"""Lexical canonicalization of class names, before the text encoder.
"""
from __future__ import annotations

import re
from typing import Dict


# Compound-word splits 
COMPOUND_SPLIT_MAP: Dict[str, str] = {
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

WORDORDER_REPAIR_MAP: Dict[str, str] = {
    "wallbrick":     "brick wall",
    "wallconcrete":  "concrete wall",
    "wallstone":     "stone wall",
    "walltile":      "tile wall",
    "wallwood":      "wooden wall",
    "wallpanel":     "wall panel",
    "floormarble":   "marble floor",
    "floorstone":    "stone floor",
    "floortile":     "tile floor",
    "floorwood":     "wooden floor",
    "ceilingtile":   "tile ceiling",
    "windowblind":   "window blind",
}

# Homonym / archaic class names mapped to a more specific noun phrase.
AMBIGUOUS_LABEL_MAP: Dict[str, str] = {
    "tie":         "necktie",          # clothing item, not "to tie"
    "remote":      "remote control",   # device, not "remote location"
    "mouse":       "computer mouse",   # device in indoor scenes
    "monitor":     "computer monitor", # computer display
    "ashcan":      "trash can",        # archaic -> modern
    "plaything":   "toy",              # archaic -> modern
    "buffet":      "sideboard",        # ambiguous (food/furniture) -> furniture
}


def _normalize_key(name: str) -> str:
    """Lowercase + strip non-alphanumerics so matching is robust to
    punctuation / casing (e.g. ``"Traffic-Light"`` matches ``"trafficlight"``)."""
    return re.sub(r"[^a-z0-9]+", "", name.lower())


def _canonicalize(name: str, universal: bool = False) -> str:
    key = _normalize_key(name)
    if key in COMPOUND_SPLIT_MAP:
        return COMPOUND_SPLIT_MAP[key]
    if universal:
        if key in WORDORDER_REPAIR_MAP:
            return WORDORDER_REPAIR_MAP[key]
        if key in AMBIGUOUS_LABEL_MAP:
            return AMBIGUOUS_LABEL_MAP[key]
    return name


def install_lexical_canonicalizer(model, segmentor, universal: bool = False) -> None:
    """Wrap ``model.backbone.forward_text`` to canonicalize class names.
    """
    backbone = model.backbone
    if not hasattr(segmentor, "_lex_original_forward_text"):
        segmentor._lex_original_forward_text = backbone.forward_text
    original = segmentor._lex_original_forward_text
    segmentor._lex_universal = bool(universal)

    def canonicalizing_forward_text(captions, input_boxes=None, additional_text=None, device="cuda"):
        if input_boxes is not None or additional_text is not None:
            return original(captions, input_boxes=input_boxes,
                            additional_text=additional_text, device=device)
        u = getattr(segmentor, "_lex_universal", False)
        substituted = [_canonicalize(c, universal=u) for c in captions]
        return original(substituted, input_boxes=None, additional_text=None, device=device)

    backbone.forward_text = canonicalizing_forward_text
