from __future__ import annotations

import argparse
import json
import os
import os.path as osp
import sys
import time

_ROOT = osp.dirname(osp.abspath(__file__))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import activesam  

from mmengine.config import Config
from mmengine.runner import Runner


def parse_args():
    p = argparse.ArgumentParser(
        description="ActiveSAM open-vocabulary segmentation eval")
    p.add_argument("config",
                   help="dataset config, e.g. configs/cfg_voc21.py")
    p.add_argument("--tag", default="",
                   help="suffix appended to the work_dir name")
    p.add_argument("--n-images", type=int, default=None,
                   help="evaluate only the first N images (quick smoke test)")
    p.add_argument("--sam3-ckpt-path", default=None,
                   help="override the SAM 3 checkpoint path")
    p.add_argument("--corruption-type", default=None,
                   help="ImageNet-C corruption applied at the input "
                        "(gaussian_noise, motion_blur, jpeg_compression, fog); "
                        "switches the model to ActiveSAMCorrupted")
    p.add_argument("--corruption-severity", type=int, default=5,
                   help="severity 1-5 for --corruption-type (default 5)")
    p.add_argument("--seed", type=int, default=None,
                   help="for reproducibility")
    args = p.parse_args()
    os.environ.setdefault("LOCAL_RANK", "0")
    return args


def main():
    args = parse_args()
    os.chdir(_ROOT)

    cfg = Config.fromfile(args.config)
    cfg.launcher = "none"
    if args.sam3_ckpt_path:
        cfg.model.sam3_ckpt_path = args.sam3_ckpt_path
    if args.n_images:
        cfg.test_dataloader.dataset.indices = list(range(args.n_images))

    # ImageNet-C corruption: use the corruption-aware model.
    if args.corruption_type:
        cfg.model.type = "ActiveSAMCorrupted"
        cfg.model.corruption_type = args.corruption_type
        cfg.model.corruption_severity = args.corruption_severity

    # Fix the RNG seed 
    if args.seed is not None:
        cfg.randomness = dict(seed=args.seed)

    cfg_name = osp.splitext(osp.basename(args.config))[0]
    tag = args.tag
    if args.corruption_type:
        ctag = f"{args.corruption_type}_s{args.corruption_severity}"
        tag = f"{tag}__{ctag}" if tag else ctag
    cfg.work_dir = osp.join(
        _ROOT, "work_dirs", cfg_name + (f"__{tag}" if tag else ""))
    os.makedirs(cfg.work_dir, exist_ok=True)

    runner = Runner.from_cfg(cfg)
    num_images = len(runner.test_dataloader.dataset)

    t0 = time.time()
    metrics = runner.test()
    elapsed = time.time() - t0

    if runner.rank == 0:
        miou = metrics.get("mIoU", float("nan"))
        aacc = metrics.get("aAcc", float("nan"))
        fps = num_images / elapsed if elapsed else 0.0
        print(f"\n{'=' * 56}")
        print(f"  Dataset : {cfg.dataset_type}")
        print(f"  mIoU    : {miou:.2f}    aAcc: {aacc:.2f}")
        print(f"  Images  : {num_images}    FPS: {fps:.2f}")
        print(f"{'=' * 56}\n")
        summary = {
            "dataset": cfg.dataset_type,
            "num_images": num_images,
            "total_time_s": round(elapsed, 2),
            "fps": round(fps, 2),
            **{k: round(float(v), 4) for k, v in metrics.items()},
        }
        with open(osp.join(cfg.work_dir, "results.json"), "w") as f:
            json.dump(summary, f, indent=2)


if __name__ == "__main__":
    main()
