#!/bin/bash
set -e  # Exit on any error
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# -------------------------- Configurable Parameters (Modify as Needed) --------------------------
# Basic settings
EXPERIMENT_DIR="./multimodal_experiment"  # Root directory for all outputs
GPU_ID=0                                  # GPU id (set to -1 for CPU)
TEST_ID=1                                 # Target test sample ID for IF/LOO
SEED=42                                   # Random seed

# Multimodal Data settings 
DATA_ROOT="./"                        # Root directory for multimodal data
TRAIN_JSONL="./data/train_data_new.jsonl"    # Multimodal train data jsonl path
TEST_JSONL="./data/test_data_new.jsonl"      # Multimodal test data jsonl path
TRAIN_LIMIT=500                           # Limit training data size (set to -1 for full dataset)
TEST_LIMIT=25                             # Limit test data size (set to -1 for full dataset)

# Multimodal Model Config 
EMBED_DIM=384                             # Embedding dimension
NUM_HEADS=4                               # Number of attention heads
NUM_LAYERS=2                              # Number of transformer layers
MAX_SEQ_LEN=256                           # Max text sequence length
VOCAB_SIZE=10000                          # Tokenizer vocab size

# Base Model Training settings
BASE_OPTIMIZER="sgd"                    # Optimizer for base model training (lbfgs/sgd/adamw)
BASE_EPOCHS=10                            # Epochs for base model training
BASE_BATCH_SIZE=4                        # Batch size for base model training, LBFGS uses full batch
BASE_LR=0.0001                            # Learning rate for base model training
BASE_WEIGHT_DECAY=0.01                    # Weight decay for base model

# IF settings
RECURSION_DEPTH=100                       # Recursion depth for s_test calculation
R_AVERAGING=5                             # Number of averaging iterations for IF
DAMP=0.01                                # Dampening factor for IF
SCALE=60.0                               # Scaling factor for IF
TOPK=200                                  # Top-K influential training samples to select
LOSS_FUNC="cross_entropy"                 # Loss function for IF calculation

# LOO settings
LOO_OPTIMIZER="sgd"                     # Optimizer for LOO retraining (lbfgs/sgd/adamw)
LOO_LR=0.0001                             # Learning rate for LOO retraining
LOO_EPOCHS=8                              # Epochs for LOO retraining
LOO_BATCH_SIZE=4                         # Batch size for LOO retraining, LBFGS uses full batch
LOO_NUM_WORKERS=2                         # Number of dataloader workers for LOO

# Visualization settings
TOPN_DIST=200                             # Top N helpful/harmful samples for distribution plot
TOPK_CORR=200                             # Top K samples for IF-LOO correlation plot
MODEL_TYPE="Multimodal-Transformer"       # Model type label for visualization

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
echo "Multimodal Experiment Setup Completed"
echo "Experiment Dir: ${EXPERIMENT_DIR}"
echo "GPU ID: ${GPU_ID}"
echo "Test Sample ID: ${TEST_ID}"
echo "Model Type: ${MODEL_TYPE}"
echo "======================================"

# -------------------------- Step 1: Train Multimodal Base Model --------------------------
echo -e "\n[Step 1/4] Training Multimodal-Transformer Base Model"
# python ./scripts/train_base.py \
#     --experiment multimodal \
#     --train_jsonl "${TRAIN_JSONL}" \
#     --test_jsonl "${TEST_JSONL}" \
#     --image_root "${DATA_ROOT}" \
#     --data_root "${DATA_ROOT}" \
#     --train_limit "${TRAIN_LIMIT}" \
#     --embed_dim "${EMBED_DIM}" \
#     --num_heads "${NUM_HEADS}" \
#     --num_layers "${NUM_LAYERS}" \
#     --max_seq_len "${MAX_SEQ_LEN}" \
#     --vocab_size "${VOCAB_SIZE}" \
#     --optimizer "${BASE_OPTIMIZER}" \
#     --epochs "${BASE_EPOCHS}" \
#     --batch_size "${BASE_BATCH_SIZE}" \
#     --lr "${BASE_LR}" \
#     --weight_decay "${BASE_WEIGHT_DECAY}" \
#     --lbfgs_tolerance_grad 1e-8 \
#     --lbfgs_tolerance_change 1e-10 \
#     --gpu "${GPU_ID}" \
#     --save_dir "${BASE_MODEL_DIR}"

# -------------------------- Step 2: Calculate Influence Functions (IF) --------------------------
echo -e "\n[Step 2/4] Calculating Influence Functions for Test ID ${TEST_ID}"
python ./scripts/run_if.py \
    --experiment multimodal \
    --ckpt_path "${BASE_MODEL_DIR}/base_model.pth" \
    --data_root "${DATA_ROOT}" \
    --train_jsonl "${TRAIN_JSONL}" \
    --test_jsonl "${TEST_JSONL}" \
    --image_root "${DATA_ROOT}" \
    --train_limit "${TRAIN_LIMIT}" \
    --test_limit "${TEST_LIMIT}" \
    --embed_dim "${EMBED_DIM}" \
    --num_heads "${NUM_HEADS}" \
    --num_layers "${NUM_LAYERS}" \
    --max_seq_len "${MAX_SEQ_LEN}" \
    --test_id "${TEST_ID}" \
    --recursion_depth "${RECURSION_DEPTH}" \
    --r_averaging "${R_AVERAGING}" \
    --damp "${DAMP}" \
    --scale "${SCALE}" \
    --loss_func "${LOSS_FUNC}" \
    --topk "${TOPK}" \
    --save_dir "${IF_SAVE_DIR}" \
    --batch_size 32 \
    --num_workers 2 \
    --gpu "${GPU_ID}"

# -------------------------- Step 3: Leave-One-Out (LOO) Retraining --------------------------
echo -e "\n[Step 3/4] Starting LOO Retraining for Top-${TOPK} Samples"
python ./scripts/run_loo.py \
    --experiment multimodal \
    --topk_ids_txt "${IF_SAVE_DIR}/topk_ids_test_${TEST_ID}.txt" \
    --test_id "${TEST_ID}" \
    --data_root "${DATA_ROOT}" \
    --train_jsonl "${TRAIN_JSONL}" \
    --test_jsonl "${TEST_JSONL}" \
    --image_root "${DATA_ROOT}" \
    --train_limit "${TRAIN_LIMIT}" \
    --test_limit "${TEST_LIMIT}" \
    --embed_dim "${EMBED_DIM}" \
    --num_heads "${NUM_HEADS}" \
    --num_layers "${NUM_LAYERS}" \
    --max_seq_len "${MAX_SEQ_LEN}" \
    --optimizer "${LOO_OPTIMIZER}" \
    --lr "${LOO_LR}" \
    --epochs "${LOO_EPOCHS}" \
    --batch_size "${LOO_BATCH_SIZE}" \
    --num_workers "${LOO_NUM_WORKERS}" \
    --lbfgs_tolerance_grad 1e-8 \
    --lbfgs_tolerance_change 1e-10 \
    --save_jsonl "${LOO_JSONL_PATH}" \
    --gpu "${GPU_ID}"

# -------------------------- Step 4: Visualize Multimodal Results --------------------------
echo -e "\n[Step 4/4] Generating Multimodal Experiment Visualizations"
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
echo "Multimodal Experiment Completed Successfully"
echo "Results Summary:"
echo "  - Base Model: ${BASE_MODEL_DIR}/base_model.pth"
echo "  - IF Results: ${IF_SAVE_DIR}"
echo "  - LOO Results: ${LOO_JSONL_PATH}"
echo "  - Visualizations: ${VIS_SAVE_DIR}"
echo "======================================"