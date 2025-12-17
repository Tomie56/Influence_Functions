#!/usr/bin/env bash
set -euo pipefail

# ============================================================
# run_multimodal.sh  (Multimodal full pipeline)
#
# 1) (optional) Train base multimodal transformer on jsonl
# 2) (optional) Run IF:
#      - single test_id  -> save if_scores_test_X.npy + topk_ids_test_X.txt
#      - all tests       -> save per-test npy under ./outputs/multimodal/if_all/
# 3) (optional) Run LOO on Top-K train ids (resume jsonl)
# 4) (optional) Visualize after LOO
# ============================================================

# -----------------------------
# [A] Switches (0/1)
# -----------------------------
SKIP_TRAIN=0
SKIP_IF=0
SKIP_LOO=0
SKIP_VIS=0

RUN_ALL_TEST_IF=0      # 1 = compute IF for all test ids (save per-test npy)
TEST_ID=0              # used when RUN_ALL_TEST_IF=0

# -----------------------------
# [B] Paths (hard-coded)
# -----------------------------
MODELS_DIR="./models"
OUT_DIR="./outputs/multimodal"
mkdir -p "${MODELS_DIR}" "${OUT_DIR}"

BASE_CKPT="${MODELS_DIR}/multimodal_base.pth"

# single-test outputs
IF_SCORES_NPY="${OUT_DIR}/if_scores_test_${TEST_ID}.npy"
TOPK_IDS_TXT="${OUT_DIR}/topk_ids_test_${TEST_ID}.txt"
LOO_JSONL="${OUT_DIR}/loo_losses_test_${TEST_ID}.jsonl"

# all-tests IF outputs (each test -> one npy)
IF_ALL_DIR="${OUT_DIR}/if_all"   # contains if_scores_test_{tid}.npy + index.jsonl

# -----------------------------
# [C] Dataset config (jsonl)
# -----------------------------
TRAIN_JSONL="./data/train_data_new.jsonl"
TEST_JSONL="./data/test_data_new.jsonl"
IMAGE_ROOT="./"

# If you want to limit #tests for all-tests IF (for debugging)
ALL_TEST_LIMIT=0   # 0 = all tests in dataset

# -----------------------------
# [D] Multimodal model shape
# (must match train_base.py and run_if.py init)
# -----------------------------
IMAGE_SIZE=224
EMBED_DIM=384
NUM_HEADS=6
NUM_LAYERS=4
NUM_IMAGE_TOKENS_PER_PATCH=12
MAX_SEQ_LEN=768

# -----------------------------
# [E] Base training params (multimodal)
# -----------------------------
# These are "fine-tune-like" defaults.
BASE_EPOCHS=3
BASE_LR=1e-4
WEIGHT_DECAY=1e-4
BATCH_SIZE=8
NUM_WORKERS=2

# -----------------------------
# [F] IF params (stochastic approx, Figure 2 right style)
# -----------------------------
RECURSION_DEPTH=1000
DAMP=0.01
SCALE=25.0

# IF/LOO compute is heavy; IF per-test uses batch_size=1 internally in run_if.py anyway.
IF_BATCH_SIZE=1

# -----------------------------
# [G] LOO params
# -----------------------------
TOPK_LOO=500
LOO_EPOCHS=1
LOO_LR=1e-4
# ============================================================

echo "============================================================"
echo "[run_multimodal.sh] OUT_DIR=${OUT_DIR}"
echo "[run_multimodal.sh] BASE_CKPT=${BASE_CKPT}"
echo "[run_multimodal.sh] SKIP_TRAIN=${SKIP_TRAIN} SKIP_IF=${SKIP_IF} SKIP_LOO=${SKIP_LOO} SKIP_VIS=${SKIP_VIS}"
echo "[run_multimodal.sh] RUN_ALL_TEST_IF=${RUN_ALL_TEST_IF} TEST_ID=${TEST_ID}"
echo "[run_multimodal.sh] TRAIN_JSONL=${TRAIN_JSONL}"
echo "[run_multimodal.sh] TEST_JSONL=${TEST_JSONL}"
echo "============================================================"
echo

# ============================================================
# 1) Train base model (or load)
# ============================================================
if [[ "${SKIP_TRAIN}" -eq 1 ]]; then
  [[ -f "${BASE_CKPT}" ]] || { echo "[ERROR] Missing ckpt: ${BASE_CKPT}"; exit 1; }
  echo "[1/4] Skip training. Using: ${BASE_CKPT}"
else
  echo "[1/4] Train base multimodal transformer -> ${BASE_CKPT}"

  python3 -u scripts/train_base.py \
    --experiment multimodal \
    --train_jsonl "${TRAIN_JSONL}" \
    --test_jsonl "${TEST_JSONL}" \
    --image_root "${IMAGE_ROOT}" \
    --image_size "${IMAGE_SIZE}" \
    --embed_dim "${EMBED_DIM}" \
    --num_heads "${NUM_HEADS}" \
    --num_layers "${NUM_LAYERS}" \
    --num_image_tokens_per_patch "${NUM_IMAGE_TOKENS_PER_PATCH}" \
    --max_seq_len "${MAX_SEQ_LEN}" \
    --epochs "${BASE_EPOCHS}" \
    --lr "${BASE_LR}" \
    --weight_decay "${WEIGHT_DECAY}" \
    --batch_size "${BATCH_SIZE}" \
    --num_workers "${NUM_WORKERS}" \
    --save_path "${BASE_CKPT}"
fi
echo

# ============================================================
# 2) Run IF
# ============================================================
if [[ "${SKIP_IF}" -eq 1 ]]; then
  echo "[2/4] Skip IF."
else
  echo "[2/4] Run IF (stochastic approx)."

  if [[ "${RUN_ALL_TEST_IF}" -eq 1 ]]; then
    echo "      Mode: ALL test ids -> per-test npy under ${IF_ALL_DIR}"
    python3 -u scripts/run_if.py \
      --experiment multimodal \
      --train_jsonl "${TRAIN_JSONL}" \
      --test_jsonl "${TEST_JSONL}" \
      --image_root "${IMAGE_ROOT}" \
      --ckpt_path "${BASE_CKPT}" \
      --all_test_ids 1 \
      --save_all_dir "${IF_ALL_DIR}" \
      --all_test_limit "${ALL_TEST_LIMIT}" \
      --resume_all_test 1 \
      --recursion_depth "${RECURSION_DEPTH}" \
      --damp "${DAMP}" \
      --scale "${SCALE}" \
      --batch_size "${IF_BATCH_SIZE}" \
      --num_workers "${NUM_WORKERS}" \
      --image_size "${IMAGE_SIZE}" \
      --embed_dim "${EMBED_DIM}" \
      --num_heads "${NUM_HEADS}" \
      --num_layers "${NUM_LAYERS}" \
      --num_image_tokens_per_patch "${NUM_IMAGE_TOKENS_PER_PATCH}" \
      --max_seq_len "${MAX_SEQ_LEN}"
  else
    echo "      Mode: single test_id=${TEST_ID}"
    python3 -u scripts/run_if.py \
      --experiment multimodal \
      --train_jsonl "${TRAIN_JSONL}" \
      --test_jsonl "${TEST_JSONL}" \
      --image_root "${IMAGE_ROOT}" \
      --ckpt_path "${BASE_CKPT}" \
      --test_id "${TEST_ID}" \
      --recursion_depth "${RECURSION_DEPTH}" \
      --damp "${DAMP}" \
      --scale "${SCALE}" \
      --batch_size "${IF_BATCH_SIZE}" \
      --num_workers "${NUM_WORKERS}" \
      --save_scores_npy "${IF_SCORES_NPY}" \
      --save_topk_ids_txt "${TOPK_IDS_TXT}" \
      --topk "${TOPK_LOO}" \
      --image_size "${IMAGE_SIZE}" \
      --embed_dim "${EMBED_DIM}" \
      --num_heads "${NUM_HEADS}" \
      --num_layers "${NUM_LAYERS}" \
      --num_image_tokens_per_patch "${NUM_IMAGE_TOKENS_PER_PATCH}" \
      --max_seq_len "${MAX_SEQ_LEN}"
  fi
fi
echo

# ============================================================
# 3) Run LOO (single-test only)
# ============================================================
if [[ "${SKIP_LOO}" -eq 1 ]]; then
  echo "[3/4] Skip LOO."
else
  if [[ "${RUN_ALL_TEST_IF}" -eq 1 ]]; then
    echo "[3/4] RUN_ALL_TEST_IF=1 -> LOO skipped (too expensive)."
  else
    [[ -f "${TOPK_IDS_TXT}" ]] || { echo "[ERROR] Missing topk ids: ${TOPK_IDS_TXT}"; exit 1; }
    echo "[3/4] Run LOO on Top-${TOPK_LOO} ids -> append ${LOO_JSONL} (resume)"

    python3 -u scripts/run_loo.py \
      --experiment multimodal \
      --train_jsonl "${TRAIN_JSONL}" \
      --test_jsonl "${TEST_JSONL}" \
      --image_root "${IMAGE_ROOT}" \
      --topk_ids_txt "${TOPK_IDS_TXT}" \
      --test_id "${TEST_ID}" \
      --epochs "${LOO_EPOCHS}" \
      --lr "${LOO_LR}" \
      --weight_decay "${WEIGHT_DECAY}" \
      --batch_size "${BATCH_SIZE}" \
      --num_workers "${NUM_WORKERS}" \
      --save_jsonl "${LOO_JSONL}" \
      --image_size "${IMAGE_SIZE}" \
      --embed_dim "${EMBED_DIM}" \
      --num_heads "${NUM_HEADS}" \
      --num_layers "${NUM_LAYERS}" \
      --num_image_tokens_per_patch "${NUM_IMAGE_TOKENS_PER_PATCH}" \
      --max_seq_len "${MAX_SEQ_LEN}"
  fi
fi
echo

# ============================================================
# 4) Visualization (after LOO)
# ============================================================
if [[ "${SKIP_VIS}" -eq 1 ]]; then
  echo "[4/4] Skip visualization."
else
  if [[ "${RUN_ALL_TEST_IF}" -eq 1 ]]; then
    echo "[4/4] RUN_ALL_TEST_IF=1 -> visualization skipped (no LOO)."
  else
    echo "[4/4] Visualization -> ${OUT_DIR}"
    python3 -u scripts/run_plot.py \
      --if_scores_npy "${IF_SCORES_NPY}" \
      --topk_ids_txt "${TOPK_IDS_TXT}" \
      --loo_jsonl "${LOO_JSONL}" \
      --out_dir "${OUT_DIR}" \
      --test_id "${TEST_ID}" \
      --top_k_plot "${TOPK_LOO}" \
      --model_type "Multimodal Transformer (approx)"
  fi
fi

echo
echo "==================== DONE (MULTIMODAL) ===================="
echo "Base ckpt: ${BASE_CKPT}"
if [[ "${RUN_ALL_TEST_IF}" -eq 1 ]]; then
  echo "IF all-tests dir: ${IF_ALL_DIR}"
else
  echo "IF scores:   ${IF_SCORES_NPY}"
  echo "TopK ids:    ${TOPK_IDS_TXT}"
  echo "LOO jsonl:   ${LOO_JSONL}"
fi
echo "==========================================================="
