#!/usr/bin/env bash
# bash run_corruption.sh                 # all six benchmarks, four corruptions at severity 5
# bash run_corruption.sh voc21           # a subset
cd "$(dirname "$0")"
for ds in ${@:-voc20 voc21 city_scapes ade20k context60 coco_stuff164k}; do
    for c in gaussian_noise motion_blur jpeg_compression fog; do
        python eval.py configs/cfg_${ds}.py --corruption-type $c --corruption-severity 5 --seed 0
    done
done
