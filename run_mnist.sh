#!/usr/bin/env bash
set -euo pipefail

# ============================================================
# run_mnist.sh  (MNIST full pipeline)
#
# 1) (optional) Train base logistic regression on MNIST(55k)
# 2) (optional) Run IF:
#      - single test_id  -> save if_scores_test_X.npy + topk_ids_test_X.txt
#      - all tests       -> save per-test npy under ./outputs/mnist/if_all/
# 3) (optional) Run LOO on Top-K train ids (resume jsonl)
# 4) (optional) Visualize after LOO (hist + IF-vs-LOO scatter)
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
OUT_DIR="./outputs/mnist"
mkdir -p "${MODELS_DIR}" "${OUT_DIR}"

BASE_CKPT="${MODELS_DIR}/mnist_base.pth"

# single-test outputs
IF_SCORES_NPY="${OUT_DIR}/if_scores_test_${TEST_ID}.npy"
TOPK_IDS_TXT="${OUT_DIR}/topk_ids_test_${TEST_ID}.txt"
LOO_JSONL="${OUT_DIR}/loo_losses_test_${TEST_ID}.jsonl"

# all-tests IF outputs (each test -> one npy)
IF_ALL_DIR="${OUT_DIR}/if_all"   # will contain if_scores_test_{tid}.npy + index.jsonl

# -----------------------------
# [C] MNIST dataset config
# -----------------------------
MNIST_ROOT="./mnist"
MNIST_TRAIN_LIMIT=55000      # paper uses 55,000
MNIST_TEST_LIMIT=10000        

# -----------------------------
# [D] Base training (LogReg) params (paper-aligned)
# -----------------------------
MNIST_OPT="lbfgs"            # paper: L-BFGS
MNIST_L2=0.01                # paper: L2 regularization 0.01
BASE_EPOCHS=1                # for LBFGS, think "fit to convergence" in one run
BASE_LR=1.0                  # LBFGS step size (often 1.0); your train_base.py can ignore if not used
WEIGHT_DECAY=0.0             # keep 0 if you already add L2 via loss (l2_reg). avoid double-counting.

# Batch size:
# - LBFGS is typically full-batch; for MNIST 55k it's reasonable on CPU.
# - If your implementation uses full-batch closure, set BATCH_SIZE=55000.
BATCH_SIZE=55000
NUM_WORKERS=2

# -----------------------------
# [E] IF params (stochastic approx, Figure 2 middle)
# -----------------------------
RECURSION_DEPTH=1000
DAMP=0.01
SCALE=25.0

# -----------------------------
# [F] LOO params
# -----------------------------
TOPK_LOO=500                 # paper middle: 500 most influential by |IF|
LOO_EPOCHS=1                 # logistic regression retrain; keep small for speed (increase if needed)
LOO_LR=1.0                   # if LOO uses LBFGS, LR is step size
# ============================================================

echo "============================================================"
echo "[run_mnist.sh] OUT_DIR=${OUT_DIR}"
echo "[run_mnist.sh] BASE_CKPT=${BASE_CKPT}"
echo "[run_mnist.sh] SKIP_TRAIN=${SKIP_TRAIN} SKIP_IF=${SKIP_IF} SKIP_LOO=${SKIP_LOO} SKIP_VIS=${SKIP_VIS}"
echo "[run_mnist.sh] RUN_ALL_TEST_IF=${RUN_ALL_TEST_IF} TEST_ID=${TEST_ID}"
echo "[run_mnist.sh] MNIST_TRAIN_LIMIT=${MNIST_TRAIN_LIMIT} MNIST_TEST_LIMIT=${MNIST_TEST_LIMIT}"
echo "============================================================"
echo

# ============================================================
# 1) Train base model (or load)
# ============================================================
if [[ "${SKIP_TRAIN}" -eq 1 ]]; then
  [[ -f "${BASE_CKPT}" ]] || { echo "[ERROR] Missing ckpt: ${BASE_CKPT}"; exit 1; }
  echo "[1/4] Skip training. Using: ${BASE_CKPT}"
else
  echo "[1/4] Train base logistic regression -> ${BASE_CKPT}"
  python3 -u scripts/train_base.py \
    --experiment mnist \
    --mnist_root "${MNIST_ROOT}" \
    --mnist_train_limit "${MNIST_TRAIN_LIMIT}" \
    --mnist_test_limit "${MNIST_TEST_LIMIT}" \
    --optimizer "${MNIST_OPT}" \
    --l2_reg "${MNIST_L2}" \
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
      --experiment mnist \
      --mnist_root "${MNIST_ROOT}" \
      --mnist_train_limit "${MNIST_TRAIN_LIMIT}" \
      --mnist_test_limit "${MNIST_TEST_LIMIT}" \
      --ckpt_path "${BASE_CKPT}" \
      --all_test_ids 1 \
      --save_all_dir "${IF_ALL_DIR}" \
      --all_test_limit "${MNIST_TEST_LIMIT}" \
      --resume_all_test 1 \
      --recursion_depth "${RECURSION_DEPTH}" \
      --damp "${DAMP}" \
      --scale "${SCALE}" \
      --batch_size 1 \
      --num_workers "${NUM_WORKERS}"
  else
    echo "      Mode: single test_id=${TEST_ID}"
    python3 -u scripts/run_if.py \
      --experiment mnist \
      --mnist_root "${MNIST_ROOT}" \
      --mnist_train_limit "${MNIST_TRAIN_LIMIT}" \
      --mnist_test_limit "${MNIST_TEST_LIMIT}" \
      --ckpt_path "${BASE_CKPT}" \
      --test_id "${TEST_ID}" \
      --recursion_depth "${RECURSION_DEPTH}" \
      --damp "${DAMP}" \
      --scale "${SCALE}" \
      --batch_size 1 \
      --num_workers "${NUM_WORKERS}" \
      --save_scores_npy "${IF_SCORES_NPY}" \
      --save_topk_ids_txt "${TOPK_IDS_TXT}" \
      --topk "${TOPK_LOO}"
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
      --experiment mnist \
      --mnist_root "${MNIST_ROOT}" \
      --mnist_train_limit "${MNIST_TRAIN_LIMIT}" \
      --mnist_test_limit "${MNIST_TEST_LIMIT}" \
      --optimizer "${MNIST_OPT}" \
      --l2_reg "${MNIST_L2}" \
      --topk_ids_txt "${TOPK_IDS_TXT}" \
      --test_id "${TEST_ID}" \
      --epochs "${LOO_EPOCHS}" \
      --lr "${LOO_LR}" \
      --weight_decay "${WEIGHT_DECAY}" \
      --batch_size "${BATCH_SIZE}" \
      --num_workers "${NUM_WORKERS}" \
      --save_jsonl "${LOO_JSONL}"
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
      --model_type "MNIST Logistic Regression (approx)"
  fi
fi

echo
echo "==================== DONE (MNIST) ===================="
echo "Base ckpt: ${BASE_CKPT}"
if [[ "${RUN_ALL_TEST_IF}" -eq 1 ]]; then
  echo "IF all-tests dir: ${IF_ALL_DIR}"
else
  echo "IF scores:   ${IF_SCORES_NPY}"
  echo "TopK ids:    ${TOPK_IDS_TXT}"
  echo "LOO jsonl:   ${LOO_JSONL}"
fi
echo "======================================================"
