import argparse
import json
import os
import os.path as osp
import time

import torch
from mmengine.config import Config, DictAction
from mmengine.runner import Runner

import activesam

ROOT = osp.dirname(osp.abspath(__file__))


def parse_args():
    p = argparse.ArgumentParser(description="ActiveSAM open-vocabulary segmentation eval")
    p.add_argument("config", help="dataset config, e.g. configs/cfg_voc21.py")
    p.add_argument("--tag", default="", help="suffix appended to the work_dir name")
    p.add_argument("--n-images", type=int, default=None, help="evaluate only the first N images (quick smoke test)")
    p.add_argument("--corruption-type", default=None,
                   help="ImageNet-C corruption applied at the input (gaussian_noise, motion_blur, "
                        "jpeg_compression, fog); switches the model to ActiveSAMCorrupted")
    p.add_argument("--corruption-severity", type=int, default=5, help="severity 1-5 for --corruption-type (default 5)")
    p.add_argument("--seed", type=int, default=None, help="random seed of the stochastic corruptions")
    p.add_argument("--cfg-options", nargs="+", action=DictAction,
                   help="override config settings, e.g. model.tau=0")
    return p.parse_args()


def main():
    args = parse_args()
    os.chdir(ROOT)

    cfg = Config.fromfile(args.config)
    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)
    if args.n_images:
        cfg.test_dataloader.dataset.indices = args.n_images
    if args.corruption_type:
        cfg.model.type = "ActiveSAMCorrupted"
        cfg.model.corruption_type = args.corruption_type
        cfg.model.corruption_severity = args.corruption_severity
    if args.seed is not None:
        cfg.randomness = dict(seed=args.seed)

    cfg_name = osp.splitext(osp.basename(args.config))[0]
    tag = args.tag
    if args.corruption_type:
        ctag = f"{args.corruption_type}_s{args.corruption_severity}"
        tag = f"{tag}__{ctag}" if tag else ctag
    cfg.work_dir = osp.join(ROOT, "work_dirs", cfg_name + (f"__{tag}" if tag else ""))
    os.makedirs(cfg.work_dir, exist_ok=True)

    runner = Runner.from_cfg(cfg)
    num_images = len(runner.test_dataloader.dataset)
    torch.cuda.synchronize()
    t0 = time.time()
    metrics = runner.test()
    torch.cuda.synchronize()
    elapsed = time.time() - t0

    fps = num_images / elapsed
    print(f"\n{'=' * 56}")
    print(f"  Dataset : {cfg.test_dataloader.dataset.type}")
    print(f"  mIoU    : {metrics['mIoU']:.2f}    aAcc: {metrics['aAcc']:.2f}")
    print(f"  Images  : {num_images}    FPS: {fps:.2f}")
    print(f"{'=' * 56}\n")
    summary = {
        "dataset": cfg.test_dataloader.dataset.type,
        "num_images": num_images,
        "total_time_s": round(elapsed, 2),
        "fps": round(fps, 2),
        **{k: round(float(v), 4) for k, v in metrics.items()},
        "per_class": runner.test_evaluator.metrics[0].per_class,
    }
    with open(osp.join(cfg.work_dir, "results.json"), "w") as f:
        json.dump(summary, f, indent=2)


if __name__ == "__main__":
    main()
