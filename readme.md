# Influence Functions + Leave-One-Out (MNIST / Multimodal)

This repository provides a lightweight, self-contained implementation of Influence Functions (IF) and Leave-One-Out (LOO) retraining, supporting two application scenarios:

- **MNIST**: Multi-class logistic regression (paper-style sanity check).

- **Multimodal (jsonl)**: A lightweight multimodal language model with image tokens.

Development is based on:

- [https://github.com/kohpangwei/influence-release.git](https://github.com/kohpangwei/influence-release.git)

Influence Function computation is adapted from the PyTorch implementation ideas in:

- [https://github.com/nimarb/pytorch_influence_functions.git](https://github.com/nimarb/pytorch_influence_functions.git)

## What This Project Does

Given a trained model and a chosen **test sample**, we perform the following steps:

1. **Influence Functions (IF)**: Approximate the effect of each training point on the test loss.

2. **Top-K Selection**: Select the most influential training indices by `abs(IF)` for verification.

3. **Leave-One-Out Retraining (LOO)**: Retrain the model with one training point removed and record the test loss.

4. **Visualization**: Plot IF distributions and IF-vs-LOO correlation (Pearson correlation coefficient).

## Results

<center class="half">
    <img src="https://github.com/Tomie56/Influence_Functions/blob/master/figures/influence_vs_loo_test_5.png" width="400"/><img src="https://github.com/Tomie56/Influence_Functions/blob/master/figures/influence_vs_loo_test_0.png" width="400"/><img src="https://github.com/Tomie56/Influence_Functions/blob/master/figures/influence_vs_loo_test_1.png" width="400"/>
</center>

## Repository Layout

```plaintext

.
├── models/                          # Store base model weight checkpoints
├── eval/  
│   └── vis.ipynb                    # ipynb for  
├── outputs/
│   ├── mnist/                       # MNIST experiment outputs
│   │   ├── tmp_if/                  # Temporary IF computation files (internal use)
│   │   ├── influences_test_0.npz    # Single test sample IF results (scores + train_ids)
│   │   ├── topk_ids_test_0.txt      # Top-K training sample IDs for LOO validation
│   │   ├── loo_losses_test_0.jsonl  # LOO retraining loss results (JSON stored line by line)
│   │   └── figs/                    # Visualization images (distribution + correlation)
│   └── multimodal/                  # Multimodal experiment outputs (same structure as MNIST)
│       ├── tmp_if/
│       ├── influences_test_0.npz
│       ├── topk_ids_test_0.txt
│       ├── loo_losses_test_0.jsonl
│       └── figs/
├── scripts/
│   ├── train_base.py                # Train base model and save checkpoint
│   ├── run_if.py                    # Compute IF (supports single/all test samples)
│   ├── run_loo.py                    # LOO retraining (supports resuming from breakpoints)
│   ├── vis.py                       # Visualization tool (IF distribution + IF-LOO correlation)
│   ├── run_mnist.sh                 # One-click execution script for MNIST full pipeline
│   └── run_multimodal.sh            # One-click execution script for multimodal full pipeline
├── model.py                         # MNIST logistic regression + multimodal model + Tokenizer
├── data_utils_mnist.py              # MNIST data loading utilities
├── data_utils.py                    # Multimodal (jsonl) data loading utilities
└── influence_function.py            # IF core logic (grad_z / s_test / influence calculation)

```

## Data Format

### MNIST

MNIST data is loaded via `data_utils_mnist.py`. The data loader returns batches in either of the following formats:

- Tuple-style: `(x, y)` or `(x, y, idx)`

- Dict-style: `{"x": ..., "labels": ..., "idx": ...}`

`idx` is used as the stable sample ID for alignment across IF and LOO processes.

### Multimodal (jsonl)

The multimodal pipeline expects `train_jsonl` and `test_jsonl` containing your dataset fields, plus an `image_root` (directory for image files).

The data loader yields dict batches including:

- `input_ids`, `labels`, and optionally `attention_mask`, `pixel_values`, `image_flags`

- `sample_index` for stable sample identification (recommended)

## Outputs

### Influence Functions (Single Test)

- `outputs/<exp>/influences_test_<test_id>.npz`

    - `scores`: float32 array of influence scores aligned with `train_ids`

    - `train_ids`: int64 array of the corresponding training sample IDs

- `outputs/<exp>/topk_ids_test_<test_id>.txt`

    - One integer training ID per line, selected by `abs(score)` top-K

### Influence Functions (All Tests)

- `outputs/<exp>/all_tests/if_scores_test_<test_id>.npy` (or `.npz` depending on configuration)

- `outputs/<exp>/all_tests/index.jsonl`

    - One line per test sample: `{"test_id": ..., "path": ...}`

### Leave-One-Out

- `outputs/<exp>/loo_losses_test_<test_id>.jsonl`

    - One JSON object per line:
                

        - Base run (no training sample removed): `{"id": -1, "loss": <float>}`

        - LOO run (one training sample removed): `{"id": <train_id>, "loss": <float>}`

    - Supports resuming: Already-present `id` values are skipped to avoid redundant computation.

## Quickstart

### 1) MNIST Full Pipeline

Run the full MNIST pipeline (train base model → compute IF → select Top-K → LOO retraining → generate plots):

```bash

bash scripts/run_mnist.sh

```

Typical outputs:

- models/mnist_base.pth

- outputs/mnist/influences_test_0.npz

- outputs/mnist/topk_ids_test_0.txt

- outputs/mnist/loo_losses_test_0.jsonl

- outputs/mnist/figs/influence_distribution_test_0.png

- outputs/mnist/figs/influence_vs_loo_test_0.png

### 2) Multimodal Full Pipeline

Run the multimodal full pipeline:

```bash

bash scripts/run_multimodal.sh
```

Ensure the script correctly points to:

- TRAIN_JSONL: Path to the training jsonl file

- TEST_JSONL: Path to the test jsonl file

- IMAGE_ROOT: Root directory for image files

Also ensure the model hyperparameters in the script match those used during training.

### Running Individual Steps

#### Train Base Model

MNIST:

```bash

python3 -u scripts/train_base.py \
  --experiment mnist \
  --mnist_root ./mnist \
  --mnist_train_limit 55000 \
  --optimizer lbfgs \
  --l2_reg 0.01 \
  --epochs 1 \
  --batch_size 55000 \
  --save_path ./models/mnist_base.pth

```

Multimodal:

```bash

python3 -u scripts/train_base.py \
  --experiment multimodal \
  --train_jsonl ./data/train_data_new.jsonl \
  --image_root ./ \
  --epochs 10 \
  --batch_size 32 \
  --save_path ./models/multimodal_base.pth

```

#### Run IF (Single Test)

```bash

python3 -u scripts/run_if.py \
  --experiment mnist \
  --ckpt_path ./models/mnist_base.pth \
  --test_id 0 \
  --recursion_depth 1000 \
  --damp 0.01 \
  --scale 25.0 \
  --save_scores_npy ./outputs/mnist/if_scores_test_0.npy \
  --save_topk_ids_txt ./outputs/mnist/topk_ids_test_0.txt \
  --topk 500

```

#### Run IF (All Tests)

```bash

python3 -u scripts/run_if.py \
  --experiment mnist \
  --ckpt_path ./models/mnist_base.pth \
  --all_test_ids 1 \
  --save_all_dir ./outputs/mnist/all_tests \
  --resume_all_test 1

```

#### Run LOO

```bash

python3 -u scripts/run_loo.py \
  --experiment mnist \
  --topk_ids_txt ./outputs/mnist/topk_ids_test_0.txt \
  --test_id 0 \
  --save_jsonl ./outputs/mnist/loo_losses_test_0.jsonl \
  --optimizer lbfgs \
  --l2_reg 0.01 \
  --epochs 1

```
