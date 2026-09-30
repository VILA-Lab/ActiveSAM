"""Class signatures and their label-free estimation from a pool of unlabeled images."""
import json
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F

SIGNATURE_EPS = 1e-6


@dataclass
class Signatures:
    """Signatures of one vocabulary with C classes.
    R_a[k] is the mean class-score vector of the pixels class k wins, R_e[k] the mean over its most exclusive pixel
    group with at least N pixels, log_mass[k] the log share of the pool's pixels won by k, and m_bg the share won by
    the background prompt.
    """

    R_a: np.ndarray
    R_e: np.ndarray
    log_mass: np.ndarray
    mass_argmax: np.ndarray
    mass_excl: np.ndarray
    m_bg: float
    meta: dict

    @property
    def uncalibrated(self):
        """Classes that won fewer than N pooled pixels; ECD leaves their pixels to the argmax."""
        return self.mass_argmax < self.meta["constants"]["CORE_MIN_PX"]

    def save(self, path):
        np.savez(path, R_a=self.R_a, R_e=self.R_e, log_mass=self.log_mass, mass_argmax=self.mass_argmax,
                 mass_excl=self.mass_excl, m_bg=np.array(self.m_bg, dtype=np.float64), meta=json.dumps(self.meta))

    @classmethod
    def load(cls, path):
        f = np.load(path)
        return cls(f["R_a"], f["R_e"], f["log_mass"], f["mass_argmax"], f["mass_excl"], float(f["m_bg"]),
                   json.loads(str(f["meta"])))


class SignatureEstimator:
    """Accumulates three pixel groups per class over the pool and averages the class scores inside them.
    For a class k the groups are its argmax pixels (k has the highest score), its confident pixels (S_k > core_high)
    and its exclusive pixels (k wins with S_k > core_high while every other score is below core_low).
    """

    GROUPS = ("argmax", "exclusive", "confident")

    def __init__(self, num_classes, core_high=0.5, core_low=0.3, min_pixels=2000, device="cuda"):
        self.num_classes = num_classes
        self.core_high, self.core_low, self.min_pixels = core_high, core_low, min_pixels
        self.sums = {g: torch.zeros(num_classes, num_classes, dtype=torch.float64, device=device) for g in self.GROUPS}
        self.sizes = {g: torch.zeros(num_classes, dtype=torch.float64, device=device) for g in self.GROUPS}
        self.num_images = 0

    def add(self, class_maps, active):
        """class_maps [C, H, W] of one pool image, active [n] the classes admitted for it."""
        height, width = class_maps.shape[-2:]
        pooled = F.adaptive_avg_pool2d(class_maps[None], (height // 4, width // 4))[0][active]
        scores = (pooled.clamp(0, 1) * 255).round().to(torch.uint8).flatten(1).float() / 255.0

        pixels = torch.arange(scores.shape[1], device=scores.device)
        winner = torch.zeros_like(scores, dtype=torch.bool)
        winner[scores.argmax(0), pixels] = True
        competitor = scores.masked_fill(winner, 0).amax(0)
        confident = scores > self.core_high
        groups = {"argmax": winner, "exclusive": winner & confident & (competitor < self.core_low),
                  "confident": confident}
        for name, members in groups.items():
            members = members.float()
            self.sums[name][active[:, None], active[None, :]] += (members @ scores.T).double()
            self.sizes[name][active] += members.sum(1).double()
        self.num_images += 1

    def signatures(self, background_label=None, meta=None):
        mean = {g: self._mean(g) for g in self.GROUPS}
        sizes = self.sizes
        R_e = torch.where((sizes["exclusive"] >= self.min_pixels)[:, None], mean["exclusive"],
                          torch.where((sizes["confident"] >= self.min_pixels)[:, None], mean["confident"],
                                      mean["argmax"]))
        share = sizes["argmax"] / sizes["argmax"].sum()
        m_bg = float("nan") if background_label is None else float(share[background_label])
        constants = dict(CORE_HI=self.core_high, CORE_LO=self.core_low, CORE_MIN_PX=self.min_pixels)
        meta = dict(meta or {}, n_images=self.num_images, C=self.num_classes, background_label=background_label,
                    constants=constants)
        return Signatures(mean["argmax"].cpu().numpy(), R_e.cpu().numpy(),
                          torch.log(share.float().clamp_min(1e-6)).cpu().numpy(),
                          sizes["argmax"].cpu().numpy(), sizes["exclusive"].cpu().numpy(), m_bg, meta)

    def _mean(self, group):
        size = self.sizes[group][:, None]
        mean = torch.where(size > 0, self.sums[group] / size.clamp_min(1e-9), 0.5)
        return mean.float().clamp(SIGNATURE_EPS, 1 - SIGNATURE_EPS)
