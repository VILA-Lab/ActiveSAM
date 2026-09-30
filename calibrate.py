"""Estimate the signatures of a vocabulary from the unlabeled calibration pool (no labels are read).

    python calibrate.py configs/cfg_voc21.py --out signatures/voc21.npz
"""
import argparse
import os
import os.path as osp

import torch
from mmengine.config import Config, DictAction
from mmengine.registry import init_default_scope
from mmengine.utils import ProgressBar
from mmseg.registry import DATASETS, MODELS
from torch.utils.data import DataLoader, Subset

import activesam
from activesam.signatures import SignatureEstimator

ROOT = osp.dirname(osp.abspath(__file__))


def parse_args():
    parser = argparse.ArgumentParser(description="Estimate the signatures of a vocabulary.")
    parser.add_argument("config")
    parser.add_argument("--out", required=True, help="output .npz file")
    parser.add_argument("--cfg-options", nargs="+", action=DictAction, help="override config entries")
    return parser.parse_args()


def main():
    args = parse_args()
    os.chdir(ROOT)
    cfg = Config.fromfile(args.config)
    if args.cfg_options:
        cfg.merge_from_dict(args.cfg_options)
    init_default_scope("mmseg")
    calibration = cfg.calibration

    images = DATASETS.build(calibration.pool)
    pool = Subset(images, range(0, len(images), calibration.stride))
    loader = DataLoader(pool, batch_size=None, num_workers=4)

    model = MODELS.build(dict(cfg.model, signatures=None)).eval()
    estimator = SignatureEstimator(model.num_classes, calibration.core_high, calibration.core_low,
                                   calibration.min_pixels)
    progress = ProgressBar(len(pool))
    with torch.no_grad():
        for sample in loader:
            meta = sample["data_samples"].metainfo
            image = model.load_image(meta["img_path"])
            class_maps, _, active = model.class_maps(image, tuple(meta["ori_shape"]))
            estimator.add(class_maps, active)
            progress.update()

    source = f"{calibration.pool.data_root}/{calibration.pool.data_prefix.img_path}, every {calibration.stride}th image"
    signatures = estimator.signatures(cfg.model.get("background_label"),
                                      meta=dict(pool=source, vocabulary=cfg.model.vocabulary))
    signatures.save(args.out)
    print(f"\n{args.out}: {len(pool)} images, {int((~signatures.uncalibrated).sum())} of "
          f"{model.num_classes} classes calibrated")


if __name__ == "__main__":
    main()
