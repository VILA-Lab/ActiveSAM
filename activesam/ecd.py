"""Exclusive concept decoding (ECD): each pixel takes the class whose signature best explains its score vector."""
import numpy as np
import torch

PRIOR_EPS = 1e-3


def nearest_signature(scores, log_prior, signatures, sigma):
    """argmax_k  log pi_k - ||S(p) - R_k||^2 / (2 sigma^2)  over the classes k with S_k(p) > 0.

    scores [n, P], log_prior [n], signatures [n, n] with row k = R_k. Returns row indices [P]. A pixel where no class
    scores keeps the plain argmax.
    """
    scores = scores.clamp(0, 1)
    scored = scores > 0
    fit = (signatures @ scores - 0.5 * (signatures * signatures).sum(1, keepdim=True)) / sigma ** 2
    logits = (log_prior[:, None] + fit).masked_fill(~scored, torch.finfo(fit.dtype).min)
    return torch.where(scored.any(0), logits.argmax(0), scores.argmax(0))


class ExclusiveConceptDecoder:
    """ECD over the active classes of an image.

    Both signature estimates R_a and R_e decode every pixel; where they disagree, or where an uncalibrated class is
    involved, the pixel keeps the argmax label. With a background class, background is a null concept (signature 0,
    prior m_bg e^b) that competes with the decoded label.
    """

    def __init__(self, signatures, sigma, background_label=None, background_offset=1.0, device="cuda"):
        def tensor(x):
            return torch.as_tensor(x, dtype=torch.float32, device=device)

        self.R_a, self.R_e, self.log_mass = tensor(signatures.R_a), tensor(signatures.R_e), tensor(signatures.log_mass)
        self.uncalibrated = torch.as_tensor(signatures.uncalibrated, device=device)
        self.sigma = sigma
        self.background_label = background_label
        if background_label is not None:
            self.log_prior_background = float(np.log(max(signatures.m_bg, 1e-6)) + background_offset)

    def __call__(self, class_maps, presence, active):
        """class_maps [C, H, W], presence [C], active [n] class ids. Returns labels [H, W]."""
        height, width = class_maps.shape[-2:]
        scores = class_maps[active].flatten(1)
        mass = self.log_mass[active].exp()
        R_a = self.R_a[active][:, active]
        R_e = self.R_e[active][:, active]

        argmax = scores.argmax(0)
        log_prior = torch.log(presence[active] * mass + PRIOR_EPS)
        decoded_a = nearest_signature(scores, log_prior, R_a, self.sigma)
        decoded_e = nearest_signature(scores, log_prior, R_e, self.sigma)
        label = torch.where(decoded_a == decoded_e, decoded_a, argmax)

        uncalibrated = self.uncalibrated[active]
        keep_argmax = uncalibrated[argmax] | uncalibrated[label]
        label = torch.where(keep_argmax, argmax, label)
        labels = active[label]

        if self.background_label is not None:
            log_prior = torch.log(presence[active].clamp_min(PRIOR_EPS) * mass)
            residual = scores - R_a[label].T
            label_fit = log_prior[label] - (residual * residual).sum(0) / (2 * self.sigma ** 2)
            null_fit = self.log_prior_background - (scores * scores).sum(0) / (2 * self.sigma ** 2)
            labels = torch.where((null_fit > label_fit) & ~keep_argmax, self.background_label, labels)
        return labels.reshape(height, width)
