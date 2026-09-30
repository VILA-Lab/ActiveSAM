#!/usr/bin/env bash
# bash run_eval.sh                       # all eight benchmarks
# bash run_eval.sh voc21 city_scapes     # a subset
cd "$(dirname "$0")"
for ds in ${@:-voc20 voc21 context59 context60 coco_object coco_stuff164k city_scapes ade20k}; do
    python eval.py configs/cfg_${ds}.py
done
