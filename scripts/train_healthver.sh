#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export MULTIVERS_DATA_ROOT="${MULTIVERS_DATA_ROOT:-$PROJECT_ROOT/data_train}"
cd "$PROJECT_ROOT/multivers"
export PYTHONPATH="$PROJECT_ROOT/multivers/multivers${PYTHONPATH:+:$PYTHONPATH}"
PYTHON_BIN="${PYTHON_BIN:-$PROJECT_ROOT/.venv/bin/python}"

GPU_COUNT="${GPU_COUNT:-1}"
RESULT_DIR="${RESULT_DIR:-checkpoints_user}"
EXPERIMENT_NAME="${EXPERIMENT_NAME:-healthver_from_fever_sci}"
# Original MultiVerS fine-tunes HealthVer from the FEVER+PubMedQA+EvidenceInference
# pretrained model (script/train_target.py). Download: python script/get_checkpoint.py fever_sci
# The released pickle needs an old Lightning, so it was re-saved weights-only as
# fever_sci_weights.ckpt (identical state_dict).
# Unset -> fever_sci. Set to empty (STARTING_CHECKPOINT=) for the baseline from the plain encoder.
STARTING_CHECKPOINT="${STARTING_CHECKPOINT-checkpoints/fever_sci_weights.ckpt}"

"$PYTHON_BIN" multivers/train.py \
  --result_dir "$RESULT_DIR" \
  --datasets "${DATASETS:-healthver}" \
  ${VALID_DATASETS:+--valid_datasets "$VALID_DATASETS"} \
  ${STARTING_CHECKPOINT:+--starting_checkpoint "$STARTING_CHECKPOINT"} \
  --experiment_name "$EXPERIMENT_NAME" \
  --num_workers "${NUM_WORKERS:-4}" \
  --gpus "$GPU_COUNT" \
  --accumulate_grad_batches 8 \
  --lr 1e-5 \
  --precision 16 \
  --max_epochs "${MAX_EPOCHS:-20}" \
  --scheduler_total_epochs "${MAX_EPOCHS:-20}" \
  --train_batch_size 1 \
  --eval_batch_size 2 \
  --encoder_name longformer-large-science \
  --no_reweight_labels \
  --gradient_checkpointing
