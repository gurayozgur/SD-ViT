#!/usr/bin/env bash
# Train SD-ViT (or the Baseline) with the settings of the paper:
# 40 epochs, total batch 1024, AdamW (lr 1e-4, weight decay 0.05), cosine schedule with 3 warm-up epochs,
# AdaFace (m=0.4, h=0.333), PartialFC (sampling rate 0.6), omega=0.7, seed 2048.
#
#   CVLFACE=/path/to/CVLface bash scripts/train.sh vit_s12_wf4m 4          # SD-ViT on 4 GPUs
#   CVLFACE=/path/to/CVLface bash scripts/train.sh vit_s12_wf4m 4 baseline # same backbone, final exit only
set -euo pipefail

CONFIG=${1:?usage: train.sh {vit_s12_wf4m|vit_s15_wf4m|vit_b24_wf4m|vit_b24_wf12m} [num_gpu] [sdvit|baseline]}
NUM_GPU=${2:-4}
METHOD=${3:-sdvit}
TOTAL_BATCH=1024

case $CONFIG in
  vit_s12_wf4m)  ARCH=v1_small;    DATA=webface4m_ldmktheta_RA10 ;;
  vit_s15_wf4m)  ARCH=v1_small_15; DATA=webface4m_ldmktheta_RA10 ;;
  vit_b24_wf4m)  ARCH=v1_base;     DATA=webface4m_ldmktheta_RA10 ;;
  vit_b24_wf12m) ARCH=v1_base;     DATA=webface12m_ldmktheta_RA10 ;;
  *) echo "unknown config: $CONFIG"; exit 1 ;;
esac
case $METHOD in
  sdvit)    MODEL=vit_SD/configs/$ARCH.yaml; PIPELINE=configs/train_model_sd_cls.yaml ;;
  baseline) MODEL=vit/configs/$ARCH.yaml;    PIPELINE=configs/train_model_cls.yaml ;;
  *) echo "unknown method: $METHOD"; exit 1 ;;
esac
if (( TOTAL_BATCH % NUM_GPU )); then echo "num_gpu must divide $TOTAL_BATCH"; exit 1; fi

cd "${CVLFACE:?set CVLFACE to the root of the CVLface repository}/cvlface/research/recognition/code/run_v1"
if (( NUM_GPU == 1 )); then
  LAUNCH=(python)  # CVLface's train.py launches a single process itself
else
  LAUNCH=(lightning run model --strategy=ddp --devices=$NUM_GPU --precision="32-true")
fi
LIGHTING_TESTING=1 "${LAUNCH[@]}" \
      train.py trainers.prefix=${METHOD}_${CONFIG} \
      trainers.num_gpu=$NUM_GPU \
      trainers.batch_size=$((TOTAL_BATCH / NUM_GPU)) \
      trainers.gradient_acc=1 \
      trainers.num_workers=8 \
      trainers.precision='32-true' \
      trainers.float32_matmul_precision='high' \
      trainers.using_wandb=False \
      trainers.skip_final_eval=True \
      dataset=configs/$DATA.yaml \
      data_augs=configs/gridsample_v2.yaml \
      models=$MODEL \
      pipelines=$PIPELINE \
      classifiers=configs/partial_fc_sample60.yaml \
      optims=configs/cosine_wd5_long.yaml \
      losses=configs/adaface.yaml \
      evaluations=configs/hq5.yaml
