#!/usr/bin/env bash
set -euo pipefail

# ============================================================
# run.sh
#
# 1) Choose experiment: mnist | multimodal(jsonl)
# 2) Train base model once (optional skip). Save to ./models/
# 3) Run IF:
#    - single test_id: save IF scores + topK train ids for LOO
#    - all test_ids  : save one npy per test under OUT_DIR/if_all/
# 4) Run LOO on Top-K train ids (optional skip), append resume jsonl
# ============================================================

# -----------------------------
# [A] Experiment selector
# -----------------------------
EXPERIMENT="mnist"          # mnist | multimodal

# -----------------------------
# [B] Switches
# -----------------------------
SKIP_TRAIN=0                # 1: load ./models/...ckpt, do not train
SKIP_IF=0                   # 1: do not compute IF
SKIP_LOO=0                  # 1: do not compute LOO

RUN_ALL_TEST_IF=0           # 1: compute IF for ALL test ids and save per-test npy
TEST_ID=0                   # used when RUN_ALL_TEST_IF=0

# -----------------------------
# [C] Paths (hard-coded)
# -----------------------------
MODELS_DIR="./models"
OUT_DIR="./outputs/${EXPERIMENT}"
mkdir -p "$MODELS_DIR" "$OUT_DIR"

BASE_CKPT="${MODELS_DIR}/${EXPERIMENT}_base.pth"

# single-test outputs
IF_SCORES_NPY="${OUT_DIR}/if_scores_test_${TEST_ID}.npy"
TOPK_IDS_TXT="${OUT_DIR}/topk_ids_test_${TEST_ID}.txt"
LOO_JSONL="${OUT_DIR}/loo_losses_test_${TEST_ID}.jsonl"

# all-test outputs (per-test npy)
IF_ALL_DIR="${OUT_DIR}/if_all"   # will contain if_scores_test_{tid}.npy + index.jsonl

# -----------------------------
# [D] Dataset config
# -----------------------------
# MNIST
MNIST_ROOT="./mnist"
MNIST_TRAIN_LIMIT=55000
MNIST_TEST_LIMIT=10000
MNIST_OPT="lbfgs"
MNIST_L2=0.01

# Multimodal(jsonl)
TRAIN_JSONL="./data/train_data_new.jsonl"
TEST_JSONL="./data/test_data_new.jsonl"
IMAGE_ROOT="./"

# -----------------------------
# [E] IF (approx) params
# -----------------------------
RECURSION_DEPTH=1000
DAMP=0.01
SCALE=25.0

BATCH_SIZE=32
NUM_WORKERS=2

# -----------------------------
# [F] Base training params
# -----------------------------
BASE_EPOCHS=10
BASE_LR=1e-3
WEIGHT_DECAY=0.0

# -----------------------------
# [G] LOO params
# -----------------------------
TOPK_LOO=500
LOO_EPOCHS=3
LOO_LR=1e-3

# -----------------------------
# [H] All-tests IF options
# -----------------------------
ALL_TEST_LIMIT=0            # 0 = run all tests; otherwise run first N tests
RESUME_ALL_TEST=1           # 1 = skip if_scores_test_{tid}.npy if already exists
# ============================================================

echo "============================================================"
echo "[run.sh] EXPERIMENT=$EXPERIMENT"
echo "[run.sh] OUT_DIR=$OUT_DIR"
echo "[run.sh] BASE_CKPT=$BASE_CKPT"
echo "[run.sh] SKIP_TRAIN=$SKIP_TRAIN SKIP_IF=$SKIP_IF SKIP_LOO=$SKIP_LOO"
echo "[run.sh] RUN_ALL_TEST_IF=$RUN_ALL_TEST_IF TEST_ID=$TEST_ID"
echo "============================================================"
echo

# ============================================================
# 1) Train base model (or load)
# ============================================================
if [[ "$SKIP_TRAIN" -eq 1 ]]; then
  [[ -f "$BASE_CKPT" ]] || { echo "[ERROR] Missing ckpt: $BASE_CKPT"; exit 1; }
  echo "[1/3] Skip training. Using: $BASE_CKPT"
else
  echo "[1/3] Train base model -> $BASE_CKPT"

  if [[ "$EXPERIMENT" == "mnist" ]]; then
    python3 -u scripts/train_base.py \
      --experiment mnist \
      --mnist_root "$MNIST_ROOT" \
      --mnist_train_limit "$MNIST_TRAIN_LIMIT" \
      --optimizer "$MNIST_OPT" \
      --l2_reg "$MNIST_L2" \
      --epochs "$BASE_EPOCHS" \
      --lr "$BASE_LR" \
      --weight_decay "$WEIGHT_DECAY" \
      --batch_size "$BATCH_SIZE" \
      --num_workers "$NUM_WORKERS" \
      --save_path "$BASE_CKPT"
  else
    python3 -u scripts/train_base.py \
      --experiment multimodal \
      --train_jsonl "$TRAIN_JSONL" \
      --test_jsonl "$TEST_JSONL" \
      --image_root "$IMAGE_ROOT" \
      --epochs "$BASE_EPOCHS" \
      --lr "$BASE_LR" \
      --weight_decay "$WEIGHT_DECAY" \
      --batch_size "$BATCH_SIZE" \
      --num_workers "$NUM_WORKERS" \
      --save_path "$BASE_CKPT"
  fi
fi
echo

# ============================================================
# 2) Run IF
#   - single test: save if_scores_test_X.npy + topk_ids_test_X.txt
#   - all tests : save per-test npy under OUT_DIR/if_all/
# ============================================================
if [[ "$SKIP_IF" -eq 1 ]]; then
  echo "[2/3] Skip IF."
else
  echo "[2/3] Run IF."

  if [[ "$RUN_ALL_TEST_IF" -eq 1 ]]; then
    mkdir -p "$IF_ALL_DIR"
    echo "      Mode: ALL test ids -> $IF_ALL_DIR/if_scores_test_{tid}.npy (+ index.jsonl)"

    if [[ "$EXPERIMENT" == "mnist" ]]; then
      python3 -u scripts/run_if.py \
        --experiment mnist \
        --mnist_root "$MNIST_ROOT" \
        --mnist_train_limit "$MNIST_TRAIN_LIMIT" \
        --mnist_test_limit "$MNIST_TEST_LIMIT" \
        --ckpt_path "$BASE_CKPT" \
        --all_test_ids 1 \
        --save_all_dir "$IF_ALL_DIR" \
        --all_test_limit "$ALL_TEST_LIMIT" \
        --resume_all_test "$RESUME_ALL_TEST" \
        --recursion_depth "$RECURSION_DEPTH" \
        --damp "$DAMP" \
        --scale "$SCALE" \
        --batch_size "$BATCH_SIZE" \
        --num_workers "$NUM_WORKERS"
    else
      python3 -u scripts/run_if.py \
        --experiment multimodal \
        --train_jsonl "$TRAIN_JSONL" \
        --test_jsonl "$TEST_JSONL" \
        --image_root "$IMAGE_ROOT" \
        --ckpt_path "$BASE_CKPT" \
        --all_test_ids 1 \
        --save_all_dir "$IF_ALL_DIR" \
        --all_test_limit "$ALL_TEST_LIMIT" \
        --resume_all_test "$RESUME_ALL_TEST" \
        --recursion_depth "$RECURSION_DEPTH" \
        --damp "$DAMP" \
        --scale "$SCALE" \
        --batch_size "$BATCH_SIZE" \
        --num_workers "$NUM_WORKERS"
    fi

  else
    echo "      Mode: single test_id=$TEST_ID -> $IF_SCORES_NPY + $TOPK_IDS_TXT"

    if [[ "$EXPERIMENT" == "mnist" ]]; then
      python3 -u scripts/run_if.py \
        --experiment mnist \
        --mnist_root "$MNIST_ROOT" \
        --mnist_train_limit "$MNIST_TRAIN_LIMIT" \
        --mnist_test_limit "$MNIST_TEST_LIMIT" \
        --test_id "$TEST_ID" \
        --recursion_depth "$RECURSION_DEPTH" \
        --damp "$DAMP" \
        --scale "$SCALE" \
        --batch_size "$BATCH_SIZE" \
        --num_workers "$NUM_WORKERS" \
        --save_scores_npy "$IF_SCORES_NPY" \
        --save_topk_ids_txt "$TOPK_IDS_TXT" \
        --topk "$TOPK_LOO"
    else
      python3 -u scripts/run_if.py \
        --experiment multimodal \
        --train_jsonl "$TRAIN_JSONL" \
        --test_jsonl "$TEST_JSONL" \
        --image_root "$IMAGE_ROOT" \
        --test_id "$TEST_ID" \
        --recursion_depth "$RECURSION_DEPTH" \
        --damp "$DAMP" \
        --scale "$SCALE" \
        --batch_size "$BATCH_SIZE" \
        --num_workers "$NUM_WORKERS" \
        --save_scores_npy "$IF_SCORES_NPY" \
        --save_topk_ids_txt "$TOPK_IDS_TXT" \
        --topk "$TOPK_LOO"
    fi
  fi
fi
echo

# ============================================================
# 3) Run LOO (only for single-test mode)
# ============================================================
if [[ "$SKIP_LOO" -eq 1 ]]; then
  echo "[3/3] Skip LOO."
else
  if [[ "$RUN_ALL_TEST_IF" -eq 1 ]]; then
    echo "[3/3] RUN_ALL_TEST_IF=1 -> LOO skipped by design (too expensive)."
  else
    echo "[3/3] Run LOO for Top-$TOPK_LOO ids -> append $LOO_JSONL (resume)"
    [[ -f "$TOPK_IDS_TXT" ]] || { echo "[ERROR] Missing topk ids: $TOPK_IDS_TXT"; exit 1; }

    if [[ "$EXPERIMENT" == "mnist" ]]; then
      python3 -u scripts/run_loo.py \
        --experiment mnist \
        --mnist_root "$MNIST_ROOT" \
        --mnist_train_limit "$MNIST_TRAIN_LIMIT" \
        --mnist_test_limit "$MNIST_TEST_LIMIT" \
        --optimizer "$MNIST_OPT" \
        --l2_reg "$MNIST_L2" \
        --topk_ids_txt "$TOPK_IDS_TXT" \
        --test_id "$TEST_ID" \
        --epochs "$LOO_EPOCHS" \
        --lr "$LOO_LR" \
        --weight_decay "$WEIGHT_DECAY" \
        --batch_size "$BATCH_SIZE" \
        --num_workers "$NUM_WORKERS" \
        --save_jsonl "$LOO_JSONL"
    else
      python3 -u scripts/run_loo.py \
        --experiment multimodal \
        --train_jsonl "$TRAIN_JSONL" \
        --test_jsonl "$TEST_JSONL" \
        --image_root "$IMAGE_ROOT" \
        --topk_ids_txt "$TOPK_IDS_TXT" \
        --test_id "$TEST_ID" \
        --epochs "$LOO_EPOCHS" \
        --lr "$LOO_LR" \
        --weight_decay "$WEIGHT_DECAY" \
        --batch_size "$BATCH_SIZE" \
        --num_workers "$NUM_WORKERS" \
        --save_jsonl "$LOO_JSONL"
    fi
  fi
fi


# ============================================================
# 4) Plot after LOO
# ============================================================
if [[ "$SKIP_LOO" -eq 1 ]]; then
  echo "[4/4] Skip plots (LOO skipped)."
else
  if [[ "$RUN_ALL_TEST_IF" -eq 1 ]]; then
    echo "[4/4] ALL-TEST IF mode -> plots skipped by design."
  else
    echo "[4/4] Plot IF distribution + IF vs LOO"

    python3 -u scripts/run_plot.py \
      --test_id "$TEST_ID" \
      --if_scores_npy "$IF_SCORES_NPY" \
      --topk_ids_txt "$TOPK_IDS_TXT" \
      --loo_jsonl "$LOO_JSONL" \
      --out_dir "$OUT_DIR" \
      --top_k_scatter "$TOPK_LOO" \
      --model_type "$EXPERIMENT (approx)"
  fi
fi


echo
echo "==================== DONE ===================="
echo "Base ckpt: $BASE_CKPT"
if [[ "$RUN_ALL_TEST_IF" -eq 1 ]]; then
  echo "IF all-tests dir: $IF_ALL_DIR"
  echo "  - per-test npy: if_scores_test_{tid}.npy"
  echo "  - index file : index.jsonl"
else
  echo "IF scores:   $IF_SCORES_NPY"
  echo "TopK ids:    $TOPK_IDS_TXT"
  echo "LOO jsonl:   $LOO_JSONL"
fi
echo "============================================="
