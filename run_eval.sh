#!/usr/bin/env bash
# Evaluate ActiveSAM on the open-vocabulary segmentation benchmarks.
#
# Usage:
#   bash run_eval.sh                                      # all eight benchmarks
#   bash run_eval.sh --datasets voc21 cityscapes ade20k   # only the listed ones

set -uo pipefail
cd "$(dirname "$0")"

ALL=(voc20 voc21 context59 context60 coco_object coco_stuff164k city_scapes ade20k)

# Default to all benchmarks; otherwise take the list after --datasets
# (space- or comma-separated).
DATASETS=("${ALL[@]}")
if [ "$#" -gt 0 ]; then
    { [ "$1" = "--datasets" ] || [ "$1" = "-d" ]; } && shift
    read -r -a DATASETS <<< "${*//,/ }"
fi
if [ "${#DATASETS[@]}" -eq 0 ]; then
    echo "no datasets given, e.g.: bash run_eval.sh --datasets voc21 cityscapes" >&2
    exit 1
fi

# Map a few friendly names to their config suffix.
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
    echo "==================== ${key} ===================="
    python eval.py "$cfg"
done
