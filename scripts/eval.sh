#!/usr/bin/env bash
# Evaluate a checkpoint (a directory with model.pt and model.yaml) at every exit under the three
# protocols of the paper: self-evaluation, gallery-fixed, and one threshold held fixed across exits.
#
#   CVLFACE=/path/to/CVLface bash scripts/eval.sh /path/to/sdvit_vit_s12_wf4m [num_gpu]
set -euo pipefail

CKPT=$(realpath "${1:?usage: eval.sh <checkpoint_dir> [num_gpu]}")
NUM_GPU=${2:-1}

cd "${CVLFACE:?set CVLFACE to the root of the CVLface repository}/cvlface/research/recognition/code/run_v1"
for PROTOCOL in self gallery_fixed threshold; do
  python sdvit_eval.py --ckpt_dir "$CKPT" --protocol $PROTOCOL --exits all --num_gpu $NUM_GPU
done
