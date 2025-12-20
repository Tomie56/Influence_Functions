# Influence Functions + Leave-One-Out (MNIST / Multimodal)

This repository provides a lightweight, self-contained implementation of Influence Functions (IF) and Leave-One-Out (LOO) retraining, supporting two application scenarios:

- **MNIST**: Multi-class logistic regression (paper-style sanity check).

- **Multimodal (jsonl)**: A lightweight multimodal language model with image tokens. (incoming)

Development is based on:

- [https://github.com/kohpangwei/influence-release.git](https://github.com/kohpangwei/influence-release.git)

Influence Function computation is adapted from the PyTorch implementation ideas in:

- [https://github.com/nimarb/pytorch_influence_functions.git](https://github.com/nimarb/pytorch_influence_functions.git)

## What This Project Does

Given a trained model and a chosen **test sample**, this repo performs the following steps:

1. **Influence Functions (IF)**: Approximate the effect of each training point on the test loss.

2. **Top-K Selection**: Select the most influential training indices by `abs(IF)` for verification.

3. **Leave-One-Out Retraining (LOO)**: Retrain the model with one training point removed and record the test loss.

4. **One-Step-Train Loss (OST)**: Train the model with one single step and record the loss changes.

5. **Visualization**: Plot IF distributions and IF-vs-LOO correlation (Pearson correlation coefficient) and IF-vs-OST plot.

## Results

## LOO vs. Influence Functions (IF) Experiments

Experiments replicate the core content from the paper *Understanding Black-box Predictions via Influence Functions* (corresponding to Figure 2 in the original work). Visualization results are presented below:
<center class="half">
    <img src="https://github.com/Tomie56/Influence_Functions/blob/master/figures/influence_vs_loo_test_5.png" width="600"/><img src="https://github.com/Tomie56/Influence_Functions/blob/master/figures/influence_vs_loo_test_0.png" width="600"/><img src="https://github.com/Tomie56/Influence_Functions/blob/master/figures/influence_vs_loo_test_1.png" width="600"/>
</center>

### Key Experimental Observations

#### 1. Approximation Performance of IF

Influence Functions (IF) achieve effective approximation of Leave-One-Out (LOO) cross-validation results overall. However, approximation accuracy deteriorates significantly for data points with influence scores close to 0. Such data points contribute minimally to model parameter updates, leading to larger discrepancies between IF approximations and true LOO outcomes.

#### 2. Characteristics of Helpful and Harmful Data on MNIST Dataset

Analysis of the most helpful and most harmful training samples identified by IF for specific test cases on the MNIST dataset reveals results consistent with intuitive expectations:

- **Most Helpful Data**: Samples share the same label as the target test sample. Consistent label information provides robust support for the model’s correct classification of the test case, which aligns with both theoretical intuition and experimental evidence.

- **Most Harmful Data**: Samples exhibit high input similarity to the target test sample (i.e., visually similar images) but have different labels. Conflicting label information confuses the model’s decision boundary for the test case, thereby adversely affecting prediction performance.

In summary, influence scores computed via the IF implementation yield highly interpretable results, effectively capturing both qualitative and quantitative impacts of individual training samples on model predictions.

---

### Test Case Visualizations

#### Test id 0
![image](https://github.com/Tomie56/Influence_Functions/blob/master/figures/test_0.png)
- **Helpful Training Samples**:
![image](https://github.com/Tomie56/Influence_Functions/blob/master/figures/helpful_test_0.png)
- **Harmful Training Samples**:
![image](https://github.com/Tomie56/Influence_Functions/blob/master/figures/harmful_test_0.png)

#### Test id 5
![image](https://github.com/Tomie56/Influence_Functions/blob/master/figures/test_5.png)
- **Helpful Training Samples**:
![image](https://github.com/Tomie56/Influence_Functions/blob/master/figures/helpful_test_5.png)
- **Harmful Training Samples**:
![image](https://github.com/Tomie56/Influence_Functions/blob/master/figures/harmful_test_5.png)

#### Test id 10
![image](https://github.com/Tomie56/Influence_Functions/blob/master/figures/test_10.png)
- **Helpful Training Samples**:
![image](https://github.com/Tomie56/Influence_Functions/blob/master/figures/helpful_test_10.png)
- **Harmful Training Samples**:
![image](https://github.com/Tomie56/Influence_Functions/blob/master/figures/harmful_test_10.png)

## One-step-train LOSS (OST) vs. Influence Functions (IF) Experiments
Traditional Influence Functions (IF) are based on Leave-One-Out (LOO) cross-validation, which approximates model parameter changes after removing each training sample to evaluate its impact on target test sample predictions. The LOO-based approach is theoretically most accurate for capturing true sample influence, as it fully considers each sample’s holistic role in training.

However, traditional IF involves complex calculations (e.g., Hessian matrix inversion), leading to high computational costs and limited applicability to large-scale datasets or real-time tasks. To address this, we proposed a simplified incremental approximation: the one_step_train_loss method. Specifically, we first calculate the base loss of the target test sample on the pre-trained model, then fine-tune the model with each training sample as a single batch (batch=1) for one step, compute the new loss of the test sample on the updated model, and define the influence score as data_influence = base_loss - new_loss.

This simplified method sacrifices some precise interpretability—it cannot capture the cumulative impact of training samples during full training, only reflecting the marginal impact of one-step updates. Nevertheless, in practical tasks like data selection (e.g., selecting influential samples for retraining/pruning), we only care about the relative order of influence scores rather than absolute values. Thus, we conducted the one_step_train_loss experiment to verify if this method can maintain the relative order of sample influence, providing a low-cost alternative.

### Key Experimental Observations

#### 1. Performance Characteristics of the one_step_train loss (ost) Method in Data Selection

The ost method exhibits significant advantages in selecting the most influential samples, while its advantages diminish when selecting samples with less obvious influence. Specifically, for top 100 and top 500 influential samples, the ost method achieves a much higher selection accuracy compared to the random selection baseline (baseline accuracy < 1%). However, when selecting top 25000 samples, the advantage of the ost method becomes insignificant, as the baseline accuracy for this scenario is approximately 50%. The underlying reason is likely that samples with less obvious influence have ambiguous decision boundaries. For such samples, the loss change-based ost method is less accurate in capturing their true influence, leading to reduced differentiation from random selection.

- **Top 100**:
![image](https://github.com/Tomie56/Influence_Functions/blob/master/figures/ost_vs_if_top100.png)
- **Top 500**:
![image](https://github.com/Tomie56/Influence_Functions/blob/master/figures/ost_vs_if_top500.png)
- **Top 25000**:
![image](https://github.com/Tomie56/Influence_Functions/blob/master/figures/ost_vs_if_top25000.png)

#### 2. Characteristics of Helpful and Harmful Data on MNIST Dataset

Consistent with the observations from traditional IF experiments, the helpful and harmful training samples identified by the ost method on the MNIST dataset also align with intuitive expectations: 

- **Most Helpful Data**: Samples share the same label as the target test sample. Consistent label information provides robust support for the model’s correct classification of the test case.

- **Most Harmful Data**: Samples exhibit high input similarity to the target test sample (i.e., visually similar images) but have different labels. Conflicting label information confuses the model’s decision boundary for the test case, thereby adversely affecting prediction performance.


### Test Case Visualizations

#### Test id 0
![image](https://github.com/Tomie56/Influence_Functions/blob/master/figures/ost_test_0.png)
- **Helpful Training Samples**:
![image](https://github.com/Tomie56/Influence_Functions/blob/master/figures/ost_helpful_test_0.png)
- **Harmful Training Samples**:
![image](https://github.com/Tomie56/Influence_Functions/blob/master/figures/ost_harmful_test_0.png)


## Repository Layout

```plaintext

.
├── models/                          # Store base model weight checkpoints
├── eval/  
│   ├── vis.ipynb                    # ipynb for eval IF
│   └── vis_ost.ipynb                # ipynb for eval OST
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
│   ├── data_utils.py                
│   ├── data_utils_mnist.py          
│   ├── one_step_train.py            # OST loss
│   ├── train_base.py                # Train base model and save checkpoint
│   ├── run_if.py                    # Compute IF (supports single/all test samples)
│   ├── run_loo.py                   # LOO retraining (supports resuming from breakpoints)
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

### One-Step-Train Loss

- `outputs/<exp>/one_step_train_influences_test_<test_id>.npy`

    - `scores`: float32 array of influence scores aligned with `train_ids`

    - `train_ids`: int64 array of the corresponding training sample IDs
    
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

#### Run OST

```bash

python3 -u scripts/one_step_train.py 
