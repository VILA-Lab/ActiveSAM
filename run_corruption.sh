#!/usr/bin/env bash
# Evaluate ActiveSAM under ImageNet-C corruption (Table 2): four corruptions at
# severity 5 over the six robustness benchmarks.
#
# Usage:
#   bash run_corruption.sh                                # all six benchmarks
#   bash run_corruption.sh --datasets voc21 cityscapes    # only the listed ones

set -uo pipefail
cd "$(dirname "$0")"

ALL=(voc20 voc21 city_scapes ade20k context60 coco_stuff164k)
CORRUPTIONS=(gaussian_noise motion_blur jpeg_compression fog)
SEVERITY=5

# Default to all benchmarks; otherwise take the list after --datasets
# (space- or comma-separated).
DATASETS=("${ALL[@]}")
if [ "$#" -gt 0 ]; then
    { [ "$1" = "--datasets" ] || [ "$1" = "-d" ]; } && shift
    read -r -a DATASETS <<< "${*//,/ }"
fi
if [ "${#DATASETS[@]}" -eq 0 ]; then
    echo "no datasets given, e.g.: bash run_corruption.sh --datasets voc21 cityscapes" >&2
    exit 1
fi


alias_of() {
    case "$1" in
        cityscapes|city)      echo city_scapes ;;
        coco_stuff|cocostuff) echo coco_stuff164k ;;
        cocoobject)           echo coco_object ;;
        *)                    echo "$1" ;;
    esac
}

for ds in "${DATASETS[@]}"; do
    key="$(alias_of "$ds")"
    cfg="configs/cfg_${key}.py"
    if [ ! -f "$cfg" ]; then
        echo "!! skipping '${ds}': no config at ${cfg}" >&2
        continue
    fi
    for c in "${CORRUPTIONS[@]}"; do
        echo "==================== ${key} | ${c} s${SEVERITY} ===================="
        python eval.py "$cfg" \
            --corruption-type "${c}" --corruption-severity "${SEVERITY}"
    done
done
