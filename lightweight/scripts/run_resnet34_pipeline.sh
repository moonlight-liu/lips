#!/usr/bin/env bash
set -euo pipefail

cd /root/lx/LipFD
source ~/anaconda3/etc/profile.d/conda.sh
conda activate lips

STAGE="${1:-train}"
RUN_NAME="${RUN_NAME:-region_resnet34_official_preprocess_ra1}"
GPU_VISIBLE="${CUDA_VISIBLE_DEVICES:-3}"
GPU_ARG="${GPU_ARG:-0}"
BATCH_SIZE="${BATCH_SIZE:-16}"
NUM_WORKERS="${NUM_WORKERS:-8}"
EPOCHS="${EPOCHS:-5}"
LR="${LR:-1e-4}"

RUN_DIR="./lightweight/results/checkpoints/${RUN_NAME}"
BEST_CKPT="${RUN_DIR}/best.pth"

train() {
  CUDA_VISIBLE_DEVICES="${GPU_VISIBLE}" python lightweight/scripts/train_region_light.py \
    --backbone resnet34 \
    --teacher_ckpt ./checkpoints/ckpt.pth \
    --real_list_path ./datasets/AVLips/0_real \
    --fake_list_path ./datasets/AVLips/1_fake \
    --val_real_list_path ./datasets/val/0_real \
    --val_fake_list_path ./datasets/val/1_fake \
    --batch_size "${BATCH_SIZE}" \
    --num_workers "${NUM_WORKERS}" \
    --epochs "${EPOCHS}" \
    --lr "${LR}" \
    --max_train_batches -1 \
    --max_val_batches -1 \
    --loss_freq 100 \
    --log_loss_every 20 \
    --name "${RUN_NAME}" \
    --save_val_scores
}

validate_best() {
  CUDA_VISIBLE_DEVICES="${GPU_VISIBLE}" python lightweight/scripts/validate_region_light.py \
    --backbone resnet34 \
    --ckpt "${BEST_CKPT}" \
    --real_list_path ./datasets/val/0_real \
    --fake_list_path ./datasets/val/1_fake \
    --batch_size "${BATCH_SIZE}" \
    --num_workers "${NUM_WORKERS}" \
    --gpu "${GPU_ARG}" \
    --save_scores "${RUN_DIR}/val_scores_best.csv"
}

measure_params() {
  python lightweight/scripts/measure_region_light_params.py \
    --backbone resnet34 \
    --output "${RUN_DIR}/params_resnet34.json"
}

benchmark_speed() {
  CUDA_VISIBLE_DEVICES="${GPU_VISIBLE}" python lightweight/scripts/benchmark_region_light.py \
    --backbone resnet34 \
    --ckpt "${BEST_CKPT}" \
    --device "cuda:${GPU_ARG}" \
    --batch_sizes 1 4 8 16 \
    --warmup 20 \
    --iters 100 \
    --output "${RUN_DIR}/speed_resnet34_best.json"
}

scan_thresholds() {
  python lightweight/scripts/scan_threshold_from_scores.py \
    --scores "${RUN_DIR}/val_scores_best.csv" \
    --output "${RUN_DIR}/threshold_scan_best.json"
}

case "${STAGE}" in
  train)
    train
    ;;
  validate)
    validate_best
    ;;
  params)
    measure_params
    ;;
  speed)
    benchmark_speed
    ;;
  scan)
    scan_thresholds
    ;;
  all)
    train
    validate_best
    measure_params
    benchmark_speed
    scan_thresholds
    ;;
  *)
    echo "Unknown stage: ${STAGE}" >&2
    echo "Usage: bash lightweight/scripts/run_resnet34_pipeline.sh [train|validate|params|speed|scan|all]" >&2
    exit 2
    ;;
esac
