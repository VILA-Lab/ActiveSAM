import math
import os.path as osp
from contextlib import contextmanager

import sam3
import torch
from sam3 import build_sam3_image_model
from sam3.model import vitdet
from sam3.model.data_misc import FindStage
from torch import nn

BPE_PATH = osp.join(osp.dirname(sam3.__file__), "assets", "bpe_simple_vocab_16e6.txt.gz")


def build_sam3(checkpoint):
    enable_variable_resolution_rope()
    return build_sam3_image_model(bpe_path=BPE_PATH, checkpoint_path=checkpoint, device="cuda")


def enable_variable_resolution_rope():
    """SAM 3's global attention layers store RoPE tables for a 1008 px input; rebuild them for other resolutions."""

    def apply_rope(self, q, k):
        num_tokens = q.shape[-2]
        if self.freqs_cis.shape[0] != num_tokens:
            side = math.isqrt(num_tokens)
            scale = self.rope_pt_size[0] / side if self.rope_interp else 1.0
            self.freqs_cis = self.compute_cis(end_x=side, end_y=side, scale_pos=scale).to(q.device)
        return vitdet.apply_rotary_enc(q, k, freqs_cis=self.freqs_cis)

    vitdet.Attention._apply_rope = apply_rope


@contextmanager
def shortened_decoder(decoder, num_layers, num_queries=None):
    """Run SAM 3's DETR decoder with its first `num_layers` layers and, optionally, its first `num_queries` queries."""
    full = decoder.layers, decoder.num_layers, decoder.query_embed, decoder.reference_points, decoder.num_queries
    decoder.layers = decoder.layers[:num_layers]
    decoder.num_layers = num_layers
    if num_queries is not None:
        decoder.query_embed = nn.Embedding.from_pretrained(decoder.query_embed.weight[:num_queries].detach())
        decoder.reference_points = nn.Embedding.from_pretrained(decoder.reference_points.weight[:num_queries].detach())
        decoder.num_queries = num_queries
    try:
        yield
    finally:
        decoder.layers, decoder.num_layers, decoder.query_embed, decoder.reference_points, decoder.num_queries = full


def find_stage(num_prompts, device):
    """K text prompts, all on the single cached image."""
    return FindStage(img_ids=torch.zeros(num_prompts, dtype=torch.long, device=device),
                     text_ids=torch.arange(num_prompts, device=device), input_boxes=None, input_boxes_mask=None,
                     input_boxes_label=None, input_points=None, input_points_mask=None)


def select_prompts(encoder_out, keep):
    """The fusion-encoder output restricted to the prompts `keep` (its batch dimension)."""
    return {**encoder_out,
            "encoder_hidden_states": encoder_out["encoder_hidden_states"][:, keep],
            "pos_embed": encoder_out["pos_embed"][:, keep],
            "valid_ratios": encoder_out["valid_ratios"][keep]}
