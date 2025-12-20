#!/bin/bash
set -e  # Exit on any error

# -------------------------- Configurable Parameters (Modify as Needed) --------------------------
# Basic settings
EXPERIMENT_DIR="./mnist_experiment"  # Root directory for all outputs
GPU_ID=0                              # GPU id (set to -1 for CPU)
TEST_ID=5                             # Target test sample ID for IF/LOO
SEED=42                               # Random seed

# Data settings
MNIST_ROOT="./mnist_data"             # MNIST data storage path
TRAIN_LIMIT=55000                     # Limit training data size (set to -1 for full dataset)
TEST_LIMIT=10000                      # Limit test data size (set to -1 for full dataset)

# Base Model Training settings
BASE_OPTIMIZER="lbfgs"                # Optimizer for base model training (lbfgs/sgd/adamw)
BASE_EPOCHS=1                         # Epochs for base model training
BASE_BATCH_SIZE=55000                 # Batch size for base model training, LBFGS uses full batch
BASE_LR=0.01                          # Learning rate for base model training

# IF settings
RECURSION_DEPTH=5500                  # Recursion depth for s_test calculation
R_AVERAGING=10                        # Number of averaging iterations for IF
DAMP=0.01                           # Dampening factor for IF
SCALE=60.0                            # Scaling factor for IF
TOPK=200                               # Top-K influential training samples to select

# LOO settings
LOO_OPTIMIZER="lbfgs"                 # Optimizer for LOO retraining (lbfgs/sgd/adamw)
LOO_LR=1.0                            # Learning rate for LOO retraining
LOO_EPOCHS=1                          # Epochs for LOO retraining
LOO_BATCH_SIZE=55000                  # Batch size for LOO retraining, LBFGS uses full batch

# Visualization settings
TOPN_DIST=200                          # Top N helpful/harmful samples for distribution plot
TOPK_CORR=200                          # Top K samples for IF-LOO correlation plot
MODEL_TYPE="MNIST_IF"                 # Model type label for visualization

# -------------------------- Directory Setup --------------------------
IF_SAVE_DIR="${EXPERIMENT_DIR}/if_results"
LOO_JSONL_PATH="${EXPERIMENT_DIR}/loo_results.jsonl"
VIS_SAVE_DIR="${EXPERIMENT_DIR}/visualizations"
BASE_MODEL_DIR="${EXPERIMENT_DIR}/base_model"

# Create directories
mkdir -p "${EXPERIMENT_DIR}"
mkdir -p "${IF_SAVE_DIR}"
mkdir -p "${VIS_SAVE_DIR}"
mkdir -p "${BASE_MODEL_DIR}"

echo "======================================"
echo "MNIST Experiment Setup Completed"
echo "Experiment Dir: ${EXPERIMENT_DIR}"
echo "GPU ID: ${GPU_ID}"
echo "Test Sample ID: ${TEST_ID}"
echo "======================================"

# -------------------------- Step 1: Train Base Model (Mandatory, for run_if.py to load weights) --------------------------
echo -e "\n[Step 1/4] Training MNIST Base Model"
python ./scripts/train_base.py \
    --mnist_root "${MNIST_ROOT}" \
    --train_limit "${TRAIN_LIMIT}" \
    --epochs "${BASE_EPOCHS}" \
    --batch_size "${BASE_BATCH_SIZE}" \
    --lr "${BASE_LR}" \
    --weight_decay 0.01 \
    --lbfgs_tolerance_grad 1e-8 \
    --lbfgs_tolerance_change 1e-10 \
    --gpu "${GPU_ID}" \
    --save_dir "${BASE_MODEL_DIR}"

# -------------------------- Step 2: Calculate Influence Functions (IF) --------------------------
echo -e "\n[Step 2/4] Calculating Influence Functions for Test ID ${TEST_ID}"
python ./scripts/run_if.py \
    --experiment mnist \
    --ckpt_path "${BASE_MODEL_DIR}/base_model.pth" \
    --mnist_root "${MNIST_ROOT}" \
    --mnist_train_limit "${TRAIN_LIMIT}" \
    --mnist_test_limit "${TEST_LIMIT}" \
    --test_id "${TEST_ID}" \
    --recursion_depth "${RECURSION_DEPTH}" \
    --r_averaging "${R_AVERAGING}" \
    --damp "${DAMP}" \
    --scale "${SCALE}" \
    --topk "${TOPK}" \
    --save_dir "${IF_SAVE_DIR}" \
    --batch_size 32 \
    --num_workers 2 \
    --gpu "${GPU_ID}"

# -------------------------- Step 3: Leave-One-Out (LOO) Retraining --------------------------
echo -e "\n[Step 3/4] Starting LOO Retraining for Top-${TOPK} Samples"
python ./scripts/run_loo.py \
    --experiment mnist \
    --topk_ids_txt "${IF_SAVE_DIR}/topk_ids_test_${TEST_ID}.txt" \
    --test_id "${TEST_ID}" \
    --mnist_root "${MNIST_ROOT}" \
    --mnist_train_limit "${TRAIN_LIMIT}" \
    --mnist_test_limit "${TEST_LIMIT}" \
    --optimizer "${LOO_OPTIMIZER}" \
    --lr "${LOO_LR}" \
    --epochs "${LOO_EPOCHS}" \
    --lbfgs_tolerance_grad 1e-8 \
    --lbfgs_tolerance_change 1e-10 \
    --batch_size "${LOO_BATCH_SIZE}" \
    --num_workers 2 \
    --save_jsonl "${LOO_JSONL_PATH}" \
    --gpu "${GPU_ID}"

# -------------------------- Step 4: Visualize Results --------------------------
echo -e "\n[Step 4/4] Generating Visualizations"
python ./scripts/vis.py \
    --if_save_dir "${IF_SAVE_DIR}" \
    --loo_jsonl "${LOO_JSONL_PATH}" \
    --vis_save_dir "${VIS_SAVE_DIR}" \
    --test_id "${TEST_ID}" \
    --top_n_dist "${TOPN_DIST}" \
    --top_k_corr "${TOPK_CORR}" \
    --model_type "${MODEL_TYPE}"

# -------------------------- Completion --------------------------
echo -e "\n======================================"
echo "MNIST Experiment Completed Successfully"
echo "Results Summary:"
echo "  - Base Model: ${BASE_MODEL_DIR}/base_model.pth"
echo "  - IF Results: ${IF_SAVE_DIR}"
echo "  - LOO Results: ${LOO_JSONL_PATH}"
echo "  - Visualizations: ${VIS_SAVE_DIR}"
echo "======================================"