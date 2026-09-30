import torch
import torch.nn.functional as F
from mmengine.structures import PixelData
from mmseg.models.segmentors import BaseSegmentor
from mmseg.registry import MODELS
from PIL import Image
from sam3.model.sam3_image_processor import Sam3Processor

from .ecd import ExclusiveConceptDecoder
from .prompts import ContextTokens, TextEncoder, load_vocabulary
from .sam3_utils import build_sam3, find_stage, select_prompts, shortened_decoder
from .signatures import Signatures


@MODELS.register_module()
class ActiveSAM(BaseSegmentor):
    """
    Args:
        vocabulary: class-name file.
        signatures: signature file of the vocabulary, estimated by calibrate.py.
        background_label: index of the background class, or None.
        resolution, preview_resolution: input sizes of the full pass and of the preview.
        preview_layers, preview_queries: decoder layers and object queries used by the preview.
        tau: presence threshold of EPG; 0 decodes every prompt at full resolution.
        bucket_size: prompts decoded together in one pass.
        confidence_threshold: instances below this score do not enter the class maps.
        neighbor_tokens, hypernym_tokens: context tokens per prompt (contextual prompt expansion).
        canonical_names: canonicalize class names before encoding them.
        sigma: ECD score noise.
        background_offset: added to the log prior of the background null concept.
        background_threshold: argmax decoding only, see above.
    """

    def __init__(self, vocabulary, signatures=None, background_label=None, sam3_checkpoint="weights/sam3/sam3.pt",
                 resolution=1008, preview_resolution=336, preview_layers=2, preview_queries=50, tau=0.1,
                 bucket_size=32, confidence_threshold=0.3, neighbor_tokens=2, hypernym_tokens=2,
                 canonical_names=True, sigma=0.15, background_offset=1.0, background_threshold=0.0):
        super().__init__()
        self.device = torch.device("cuda")
        self.sam3 = build_sam3(sam3_checkpoint)
        self.processor = Sam3Processor(self.sam3, resolution=resolution, device=self.device)
        self.preview_processor = Sam3Processor(self.sam3, resolution=preview_resolution, device=self.device)

        self.prompts, class_of_prompt = load_vocabulary(vocabulary)
        self.class_of_prompt = torch.tensor(class_of_prompt, device=self.device)
        self.num_classes = max(class_of_prompt) + 1
        self.text_encoder = TextEncoder(self.sam3.backbone, self.device, canonical_names)
        self.context = ContextTokens(self.text_encoder, self.prompts, self.sam3.backbone.language_backbone.resizer,
                                     neighbor_tokens, hypernym_tokens)

        self.tau = tau
        self.preview_layers, self.preview_queries = preview_layers, preview_queries
        self.bucket_size = bucket_size
        self.confidence_threshold = confidence_threshold
        self.background_label = background_label
        self.always_admitted = torch.zeros_like(self.class_of_prompt, dtype=torch.bool)
        if background_label is not None:
            self.always_admitted = self.class_of_prompt == background_label

        self.background_threshold = background_threshold
        self.ecd = None
        if signatures is not None:
            self.ecd = ExclusiveConceptDecoder(Signatures.load(signatures), sigma, background_label,
                                               background_offset, self.device)

    def predict(self, inputs, data_samples):
        for sample in data_samples:
            image = self.load_image(sample.img_path)
            class_maps, presence, active = self.class_maps(image, tuple(sample.ori_shape))
            labels = self.decode(class_maps, presence, active)
            sample.set_data({"seg_logits": PixelData(data=class_maps), "pred_sem_seg": PixelData(data=labels[None])})
        return data_samples

    def load_image(self, path):
        return Image.open(path).convert("RGB")

    def decode(self, class_maps, presence, active):
        if self.ecd is not None:
            return self.ecd(class_maps, presence, active)
        labels = class_maps.argmax(0)
        if self.background_label is not None:
            labels[class_maps.amax(0) < self.background_threshold] = self.background_label
        return labels

    def class_maps(self, image, shape):
        """Score map [C, *shape] and presence [C] of every class, and the classes admitted for the image."""
        maps, presence, admitted = self.ground(image)
        if maps.shape[-2:] != shape:
            maps = F.interpolate(maps[None], size=shape, mode="bilinear", align_corners=False)[0]
        return self.max_over_aliases(maps), self.max_over_aliases(presence), self.class_of_prompt[admitted].unique()

    def max_over_aliases(self, values):
        """Per-class maximum over the prompts of each class, along dim 0."""
        index = self.class_of_prompt.view(-1, *[1] * (values.dim() - 1)).expand_as(values)
        return values.new_zeros(self.num_classes, *values.shape[1:]).scatter_reduce_(0, index, values, "amax")

    def ground(self, image):
        """EPG. Per-prompt score maps at the image size, the prompts' presence and the admitted prompts."""
        width, height = image.size
        num_prompts = len(self.prompts)
        maps = torch.zeros(num_prompts, height, width, device=self.device)
        presence = torch.zeros(num_prompts, device=self.device)
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            if self.tau > 0:
                admitted = self.admit(self.preview(image))
            else:
                admitted = torch.arange(num_prompts, device=self.device)
            state = self.processor.set_image(image)
            for prompt_ids in admitted.split(self.bucket_size):
                self.decode_prompts(state, prompt_ids, maps, presence)
        return maps * presence[:, None, None], presence, admitted

    def preview(self, image):
        """Presence of every prompt at the preview resolution, from a shortened decoder without segmentation head."""
        state = self.preview_processor.set_image(image)
        presence = torch.zeros(len(self.prompts), device=self.device)
        with shortened_decoder(self.sam3.transformer.decoder, self.preview_layers, self.preview_queries):
            for prompt_ids in torch.arange(len(self.prompts), device=self.device).split(self.bucket_size):
                _, encoder_out, prompt, prompt_mask = self.encode(state, prompt_ids, with_context=False)
                out, _ = self.run_decoder(encoder_out, prompt, prompt_mask)
                presence[prompt_ids] = out["presence_logit_dec"].sigmoid().flatten().float()
        return presence

    def admit(self, presence):
        """Prompts with presence >= tau (the best one if none passes) and the background prompts."""
        admitted = presence >= self.tau
        if not admitted.any():
            admitted[presence.argmax()] = True
        return (admitted | self.always_admitted).nonzero().flatten()

    def decode_prompts(self, state, prompt_ids, maps, presence):
        """Full-resolution decoding of one bucket of prompts; writes their score maps and presence."""
        backbone_out, encoder_out, prompt, prompt_mask = self.encode(state, prompt_ids, with_context=True)
        survivors = prompt_ids
        if self.tau > 0:
            with shortened_decoder(self.sam3.transformer.decoder, num_layers=1):
                out, _ = self.run_decoder(encoder_out, prompt, prompt_mask)
            screened = out["presence_logit_dec"].sigmoid().flatten().float()
            presence[prompt_ids] = screened
            keep = (screened >= self.tau).nonzero().flatten()
            if keep.numel() == 0:
                keep = screened.argmax()[None]
            survivors = prompt_ids[keep]
            encoder_out, prompt, prompt_mask = select_prompts(encoder_out, keep), prompt[:, keep], prompt_mask[keep]

        out, hs = self.run_decoder(encoder_out, prompt, prompt_mask)
        self.sam3._run_segmentation_heads(out=out, backbone_out=backbone_out, img_ids=torch.zeros_like(survivors),
                                          vis_feat_sizes=encoder_out["vis_feat_sizes"],
                                          encoder_hidden_states=encoder_out["encoder_hidden_states"],
                                          prompt=prompt, prompt_mask=prompt_mask, hs=hs)

        survivor_presence = out["presence_logit_dec"].sigmoid().flatten().float()
        presence[survivors] = survivor_presence
        scores = out["pred_logits"].sigmoid().squeeze(-1).float() * survivor_presence[:, None]
        for i, prompt_id in enumerate(survivors.tolist()):
            maps[prompt_id] = self.fuse_heads(out["semantic_seg"][i, 0], out["pred_masks"][i], scores[i],
                                              maps.shape[-2:])

    def fuse_heads(self, semantic_logits, mask_logits, scores, size):
        """max(semantic map, max over confident instances of mask * score), upsampled to `size`."""
        score_map = F.interpolate(semantic_logits[None, None].float(), size=size, mode="bilinear",
                                  align_corners=False)[0, 0].sigmoid()
        confident = scores > self.confidence_threshold
        if confident.any():
            masks = F.interpolate(mask_logits[confident][:, None].float(), size=size, mode="bilinear",
                                  align_corners=False)[:, 0].sigmoid()
            score_map = torch.maximum(score_map, (masks * scores[confident][:, None, None]).amax(0))
        return score_map

    def encode(self, state, prompt_ids, with_context):
        """Prompt encoding and fusion encoder for a bucket of prompts on the cached image features."""
        text = self.text_encoder([self.prompts[i] for i in prompt_ids.tolist()])
        if with_context:
            tokens, mask = self.context(prompt_ids)
            features = text["language_features"]
            text["language_features"] = torch.cat([features, tokens.to(features.dtype)])
            text["language_mask"] = torch.cat([text["language_mask"], mask], dim=1)
        backbone_out = {**state["backbone_out"], **text}
        find = find_stage(len(prompt_ids), self.device)
        geometric_prompt = self.sam3._get_dummy_prompt(num_prompts=len(prompt_ids))
        prompt, prompt_mask, backbone_out = self.sam3._encode_prompt(backbone_out, find, geometric_prompt)
        backbone_out, encoder_out, _ = self.sam3._run_encoder(backbone_out, find, prompt, prompt_mask)
        return backbone_out, encoder_out, prompt, prompt_mask

    def run_decoder(self, encoder_out, prompt, prompt_mask):
        return self.sam3._run_decoder(memory=encoder_out["encoder_hidden_states"], pos_embed=encoder_out["pos_embed"],
                                      src_mask=encoder_out["padding_mask"], out={}, prompt=prompt,
                                      prompt_mask=prompt_mask, encoder_out=encoder_out)

    def _forward(self, inputs, data_samples=None):
        raise NotImplementedError

    def encode_decode(self, inputs, batch_img_metas):
        raise NotImplementedError

    def extract_feat(self, inputs):
        raise NotImplementedError

    def loss(self, inputs, data_samples):
        raise NotImplementedError("ActiveSAM is training-free")
