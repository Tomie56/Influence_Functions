# Influence Functions + Leave-One-Out (MNIST / Multimodal)

This repository is a small, self-contained implementation of Influence Functions (IF) and Leave-One-Out retraining (LOO) for two settings:

- **MNIST**: multi-class logistic regression (paper-style sanity check).

- **Multimodal (jsonl)**: a lightweight multimodal language model with image tokens.

Development was based on:

- https://github.com/kohpangwei/influence-release.git

Influence Function computation is adapted from the PyTorch implementation ideas in:

- https://github.com/nimarb/pytorch_influence_functions.git

## What this project does

Given a trained model and a chosen **test sample**, we compute:

1. **Influence Functions (IF)**: approximate the effect of each training point on the test loss.

2. **Top-K selection**: select the most influential training indices by `abs(IF)` for verification.

3. **Leave-One-Out retraining (LOO)**: retrain the model with one training point removed and record the test loss.

4. **Visualization**: plot IF distributions and IF-vs-LOO correlation (Pearson).

For "all-tests IF", we compute IF for multiple test samples and save **one file per test**.

## Repository layout

```plaintext

.
├── models/ # saved checkpoints (base model)
├── outputs/
│ ├── mnist/
│ │ ├── tmp_if/ # temporary IF outputs (internal)
│ │ ├── influences_test.npz # per-test IF results (scores + train_ids)
│ │ ├── topk_ids_test_.txt # train ids selected for LOO
│ │ ├── loo_losses_test_*.jsonl# LOO results (one JSON per line)
│ │ └── figs/ # plots produced by vis.py (optional)
│ └── multimodal/ # same structure for multimodal runs
├── scripts/
│ ├── train_base.py # train base model and save checkpoint
│ ├── run_if.py # run IF (single test or all tests)
│ ├── run_loo.py # run LOO retraining and append jsonl (resume)
│ ├── vis.py # plotting helpers
│ ├── run_mnist.sh # MNIST full pipeline entrypoint
│ └── run_multimodal.sh # Multimodal full pipeline entrypoint
├── model.py # MNIST logistic regression + multimodal model + tokenizer
├── data_utils_mnist.py # MNIST dataloader helpers
├── data_utils.py # multimodal dataloader helpers (jsonl)
└── influence_function.py # IF core (grad_z, s_test, calculate_influences)
```

## Data format

### MNIST

MNIST is read via `data_utils_mnist.py`. The loader returns batches that can be either:

- tuple-style: `(x, y)` or `(x, y, idx)`, or

- dict-style: `{"x": ..., "labels": ..., "idx": ...}`

`idx` is used as the stable sample id for alignment across IF/LOO.

### Multimodal (jsonl)

The multimodal pipeline expects `train_jsonl` and `test_jsonl` with your dataset fields, plus an `image_root`.

The dataloader yields dict batches including:

- `input_ids`, `labels`, and optionally `attention_mask`, `pixel_values`, `image_flags`

- `sample_index` for stable sample id (recommended)

## Outputs

### Influence Functions (single test)

- `outputs/<exp>/influences_test_<test_id>.npz`

    - `scores`: float32 array of influence scores aligned with `train_ids`

    - `train_ids`: int64 array of the corresponding training ids

- `outputs/<exp>/topk_ids_test_<test_id>.txt`

    - one integer training id per line, selected by`abs(score)` top-k

### Influence Functions (all tests)

- `outputs/<exp>/all_tests/if_scores_test_<test_id>.npy` (or `.npz` depending on config)

- `outputs/<exp>/all_tests/index.jsonl`

    - one line per test: `{"test_id": ..., "path": ...}`

### Leave-One-Out

- `outputs/<exp>/loo_losses_test_<test_id>.jsonl`

    - one JSON object per line:
                

        - base run: `{"id": -1, "loss": <float>}`

        - LOO run:  `{"id": <train_id>, "loss": <float>}`

    - Supports resuming: already-present `id` values are skipped.

## Quickstart

### 1) MNIST full pipeline

Run the full MNIST pipeline (train base -> IF -> Top-K -> LOO -> plots):

```bash

bash scripts/run_mnist.sh
```

Typical outputs:

models/mnist_base.pth

outputs/mnist/influences_test_0.npz

outputs/mnist/topk_ids_test_0.txt

outputs/mnist/loo_losses_test_0.jsonl

outputs/mnist/figs/influence_distribution_test_0.png

outputs/mnist/figs/influence_vs_loo_test_0.png

### 2) Multimodal full pipeline

Run the multimodal full pipeline:

```bash

bash scripts/run_multimodal.sh
```

Make sure the script points to:

TRAIN_JSONL, TEST_JSONL, IMAGE_ROOT

and uses the same model hyperparameters as training.

### Running individual steps

#### Train base model

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

#### Run IF (single test)

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

#### Run IF (all tests)

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

Note:

run_loo.py does not accept --ckpt_path. It retrains from scratch for each removal.

The base run is recorded as {"id": -1, "loss": ...}.

## Notes on sign convention (Helpful vs Harmful)

This code follows the common convention:

More negative influence values correspond to training points that reduce the test loss when upweighted (often called "helpful").

More positive influence values correspond to training points that increase the test loss when upweighted (often called "harmful").

Always validate the sign convention on your setup by comparing IF predictions against LOO loss changes.

## Environment

Typical requirements:

Python 3.8+

PyTorch

NumPy

tqdm

matplotlib

scipy (for Pearson correlation in visualization)

Install (example):

```bash

pip install torch numpy tqdm matplotlib scipy
```

## Acknowledgements

Core IF idea and experimental framing: https://github.com/kohpangwei/influence-release.git

PyTorch IF implementation reference: https://github.com/nimarb/pytorch_influence_functions.git
> （注：文档部分内容可能由 AI 生成）