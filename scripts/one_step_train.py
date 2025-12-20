import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, Subset
from torchvision import datasets, transforms
import numpy as np
from pathlib import Path
import os

# -------------------------- MNIST Dataloader (Provided) --------------------------
_DEFAULT_MNIST_TRANSFORM = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

def mnist_collate_fn(batch):
    """Collate function for MNIST dataset (packages batch into tensors with IDs)"""
    xs = torch.stack([item[0] for item in batch])
    ys = torch.tensor([item[1] for item in batch])
    ids = torch.tensor([item[2] if len(item) == 3 else idx for idx, item in enumerate(batch)])
    return xs, ys, ids

def build_dataloader_mnist(
    data_root: str,
    train: bool = True,
    batch_size: int = 32,
    shuffle: bool = False,
    num_workers: int = 2,
    limit_size: int = -1,
    transform=None,
    return_dataset: bool = False
):
    """
    Build MNIST dataloader/dataset with optional size limit and sample IDs.
    
    Args:
        data_root: Root directory for MNIST data storage.
        train: If True, load training dataset; else load test dataset.
        batch_size: Number of samples per batch.
        shuffle: Whether to shuffle the dataset.
        num_workers: Number of subprocesses for data loading.
        limit_size: Limit dataset size to this value (-1 for full dataset).
        transform: Data transformation pipeline (default: MNIST standard transform).
        return_dataset: If True, return both dataloader and dataset.
    
    Returns:
        Tuple of (dataloader, dataset) if return_dataset=True, else (dataloader, None).
    """
    if transform is None:
        transform = _DEFAULT_MNIST_TRANSFORM
    
    # Load raw MNIST dataset
    mnist_dataset = datasets.MNIST(
        root=data_root,
        train=train,
        download=True,
        transform=transform
    )

    # Wrapper dataset to add sample IDs
    class MNISTWithID(Dataset):
        def __init__(self, base_dataset):
            self.base_dataset = base_dataset
        
        def __len__(self):
            return len(self.base_dataset)
        
        def __getitem__(self, idx):
            x, y = self.base_dataset[idx]
            return x, y, idx  # Return (data, label, sample_id)
    
    dataset_with_id = MNISTWithID(mnist_dataset)

    # Limit dataset size if specified
    if limit_size > 0 and limit_size < len(dataset_with_id):
        dataset_with_id = Subset(dataset_with_id, list(range(limit_size)))
        print(f"[INFO] MNIST dataset size limited to {limit_size} samples")

    # Construct dataloader
    dataloader = DataLoader(
        dataset_with_id,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=mnist_collate_fn
    )
    
    if return_dataset:
        return dataloader, dataset_with_id
    
    return dataloader, None

# -------------------------- Model & Helper Functions --------------------------
def log_clip(x, eps=1e-10):
    """
    Clip tensor values to avoid log(0) error.
    
    Args:
        x: Input tensor.
        eps: Minimum/maximum clipping value.
    
    Returns:
        Clipped tensor.
    """
    return torch.log(torch.clamp(x, min=eps, max=1-eps))

class MNISTLogisticRegression(nn.Module):
    """
    Logistic Regression model for MNIST classification (binary/multi-class).
    
    Args:
        weight_decay: L2 regularization coefficient.
        is_multi: If True, use multi-class classification (10 classes); else binary.
    """
    def __init__(self, weight_decay, is_multi=False):
        super(MNISTLogisticRegression, self).__init__()
        self.is_multi = is_multi
        self.weight_decay = weight_decay
        self.flatten = nn.Flatten()  # Flatten 28x28 image to 784-dim vector
        
        # Define weight parameter manually
        if self.is_multi:
            # [num_classes, input_dim] for multi-class classification
            self.w = torch.nn.Parameter(torch.zeros([10, 784], requires_grad=True))
        else:
            # [input_dim] for binary classification
            self.w = torch.nn.Parameter(torch.zeros([784], requires_grad=True))

    def forward(self, x: torch.Tensor, labels=None):
        """
        Forward pass of the model.
        
        Args:
            x: Input tensor of shape [batch, 1, 28, 28].
            labels: Label tensor (unused in forward pass, for compatibility).
        
        Returns:
            Logits tensor of shape [batch, 10] (multi-class) or [batch, 1] (binary).
        """
        # Flatten input: [batch, 1, 28, 28] -> [batch, 784]
        x_flat = self.flatten(x)
        
        # Compute logits
        if self.is_multi:
            # Matrix multiplication: [batch, 784] @ [784, 10] -> [batch, 10]
            logits = torch.matmul(x_flat, self.w.T)
        else:
            # Reshape weight and multiply: [batch, 784] @ [784, 1] -> [batch, 1]
            logits = torch.matmul(x_flat, torch.reshape(self.w, [-1, 1]))
        
        return logits

    def loss(self, logits, y, train=True):
        """
        Compute total loss (task loss + L2 regularization).
        
        Args:
            logits: Model output logits.
            y: Ground truth labels.
            train: If True, include L2 regularization; else exclude.
        
        Returns:
            Scalar total loss.
        """
        if self.is_multi:
            # Multi-class cross entropy loss
            criterion = torch.nn.CrossEntropyLoss()
            y = y.to(logits.device)
            ce_loss = criterion(logits, y.long())
        else:
            # Binary cross entropy loss (sigmoid + log loss)
            preds = torch.sigmoid(logits)
            bce_loss = -torch.mean(y * log_clip(preds) + (1 - y) * log_clip(1 - preds))
        
        # L2 regularization loss (only for training)
        l2_loss = 0.5 * self.weight_decay * torch.norm(self.w, p=2) ** 2 if train else 0.0
        
        # Total loss
        total_loss = (ce_loss if self.is_multi else bce_loss) + l2_loss
        return total_loss

# -------------------------- Configuration Parameters --------------------------
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
CKPT_PATH = "/mnt/afs/jingjinhao/project/influence_functions/mnist_experiment/base_model/base_model.pth"
MNIST_DATA_ROOT = "./mnist_data"  # Modify to your MNIST data path
TEST_ID = 0  # Specify the test sample ID to calculate influence for
LEARNING_RATE = 0.01  # Learning rate for one-step training
OPTIMIZER_TYPE = "sgd"  # Optimizer type: "sgd" or "adamw"
TRAIN_LIMIT_SIZE = 55000  # Limit training data size (-1 for full dataset)
TEST_LIMIT_SIZE = -1   # Limit test data size (-1 for full dataset)
# 结果保存路径配置
RESULT_SAVE_DIR = "/mnt/afs/jingjinhao/project/influence_functions/mnist_experiment/if_results"
FULL_INFLUENCE_SAVE_NAME = f"one_step_train_influences_test_{TEST_ID}.npy"
TOPK_INFLUENCE_PATH = f"{RESULT_SAVE_DIR}/influences_test_{TEST_ID}.npz"

# -------------------------- Core Functional Functions --------------------------
def load_pretrained_model_and_original(ckpt_path, device):
    """
    Load pre-trained MNISTLogisticRegression model and create an original copy (untrained).
    
    Args:
        ckpt_path: Path to the pre-trained model checkpoint.
        device: Computing device (cuda/cpu).
    
    Returns:
        Tuple of (working_model, original_model, checkpoint dictionary).
        - working_model: Model for one-step training (will be restored after each step)
        - original_model: Untouched original model (permanent initial state)
    """
    # Load checkpoint with weight only (safe loading)
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=True)
    print(f"[INFO] Successfully loaded checkpoint from {ckpt_path}")
    
    # Initialize working model (for training)
    working_model = MNISTLogisticRegression(
        weight_decay=ckpt["weight_decay"],
        is_multi=ckpt.get("experiment") == "mnist"
    ).to(device)
    
    # Initialize original model (untouched, for restoration and base loss calculation)
    original_model = MNISTLogisticRegression(
        weight_decay=ckpt["weight_decay"],
        is_multi=ckpt.get("experiment") == "mnist"
    ).to(device)
    
    # Load weights to both models
    working_model.load_state_dict(ckpt["model_state_dict"])
    original_model.load_state_dict(ckpt["model_state_dict"])
    
    # Set original model to eval mode permanently (no training)
    original_model.eval()
    print(f"[INFO] Working model & original model loaded with weight_decay: {working_model.weight_decay}, is_multi: {working_model.is_multi}")
    
    return working_model, original_model, ckpt

def calculate_single_sample_loss(model, x, y, device):
    """
    Calculate loss for a single sample (no gradient computation).
    
    Args:
        model: Trained model (or original model).
        x: Single sample data (shape [1, 28, 28]).
        y: Single sample label (int -> will be converted to tensor).
        device: Computing device (cuda/cpu).
    
    Returns:
        Scalar loss value.
    """
    model.eval()  # Set model to evaluation mode
    with torch.no_grad():  # Disable gradient computation
        x = x.unsqueeze(0).to(device)
        y = torch.tensor(y, dtype=torch.long).unsqueeze(0).to(device)
        
        # Forward pass and loss calculation
        logits = model(x)
        loss = model.loss(logits, y, train=False)
    
    return loss.item()

def one_step_train_and_calc_influence(working_model, original_model, train_x, train_y, test_x, test_y, fixed_base_loss, device, lr, optimizer_type):
    """
    Perform one-step training with a single training sample and calculate data influence.
    Restore working model from original model after training (guarantee initial state).
    
    Args:
        working_model: Model for one-step training (will be restored).
        original_model: Untouched original model (for restoration).
        train_x: Single training sample data.
        train_y: Single training sample label (int).
        test_x: Single test sample data.
        test_y: Single test sample label (int).
        fixed_base_loss: Pre-calculated fixed base loss (from original model).
        device: Computing device (cuda/cpu).
        lr: Learning rate for one-step training.
        optimizer_type: Optimizer type ("sgd" or "adamw").
    
    Returns:
        Tuple of (data_influence, fixed_base_loss, new_loss):
            - data_influence: fixed_base_loss - new_loss (influence of training sample on test sample)
            - fixed_base_loss: Pre-calculated test sample loss (unchanged)
            - new_loss: Test sample loss after one-step training
    """
    # 1. One-step training with single sample
    working_model.train()  # Set working model to training mode
    
    # Initialize optimizer
    if optimizer_type == "sgd":
        optimizer = optim.SGD(working_model.parameters(), lr=lr)
    elif optimizer_type == "adamw":
        optimizer = optim.AdamW(working_model.parameters(), lr=lr)
    else:
        raise ValueError(f"[ERROR] Unsupported optimizer type: {optimizer_type}. Use 'sgd' or 'adamw'")
    
    # Single-step training loop
    train_x = train_x.unsqueeze(0).to(device)
    train_y = torch.tensor(train_y, dtype=torch.long).unsqueeze(0).to(device) 
    
    optimizer.zero_grad()  # Zero gradients
    logits = working_model(train_x)  # Forward pass
    train_loss = working_model.loss(logits, train_y, train=True)  # Calculate training loss
    train_loss.backward()  # Backward pass (compute gradients)
    optimizer.step()  # Update parameters (one step)
    
    # 2. Calculate new loss (after training)
    new_loss = calculate_single_sample_loss(working_model, test_x, test_y, device)
    
    # 3. Restore working model to original state (from original_model, not state_dict)
    working_model.load_state_dict(original_model.state_dict())
    working_model.eval()  # Reset to eval mode after restoration
    
    # 4. Calculate data influence
    data_influence = fixed_base_loss - new_loss
    
    return data_influence, fixed_base_loss, new_loss

def save_full_influence_results(influence_results, save_dir, save_name):
    """
    Save full influence results to .npy file (contains all training samples' influence).
    
    Args:
        influence_results: List of dictionaries containing influence info for each training sample.
        save_dir: Directory to save the result file.
        save_name: Name of the .npy file.
    
    Returns:
        Full path of the saved file.
    """
    # Create save directory if not exists
    os.makedirs(save_dir, exist_ok=True)
    
    # Extract core data (train_id, train_label, influence_value) for saving
    save_data = {
        "train_sample_ids": np.array([item["train_sample_id"] for item in influence_results]),
        "train_labels": np.array([item["train_label"] for item in influence_results]),
        "influence_values": np.array([item["data_influence"] for item in influence_results]),
        "fixed_base_loss": influence_results[0]["base_loss"],
        "test_id": TEST_ID
    }
    
    # Save to .npy file
    save_path = os.path.join(save_dir, save_name)
    np.save(save_path, save_data)
    print(f"[INFO] Full influence results saved to: {save_path}")
    
    return save_path

def compare_full_and_topk_influence(full_influence_path, topk_influence_path):
    """
    Compare full influence results (.npy) with TopK influence results (.npz) by matching train sample IDs.
    
    Args:
        full_influence_path: Path to full influence .npy file.
        topk_influence_path: Path to TopK influence .npz file.
    """
    # 1. Load full influence data
    if not os.path.exists(full_influence_path):
        raise FileNotFoundError(f"[ERROR] Full influence file not found: {full_influence_path}")
    full_data = np.load(full_influence_path, allow_pickle=True).item()
    full_train_ids = full_data["train_sample_ids"]
    full_influences = full_data["influence_values"]
    full_label_map = dict(zip(full_train_ids, full_data["train_labels"]))
    full_influence_map = dict(zip(full_train_ids, full_influences))
    
    # 2. Load TopK influence data
    if not os.path.exists(topk_influence_path):
        raise FileNotFoundError(f"[ERROR] TopK influence file not found: {topk_influence_path}")
    topk_data = np.load(topk_influence_path)
    # Common keys for TopK influence (adjust if your npz has different keys)
    topk_train_ids = topk_data.get("train_ids") or topk_data.get("train_sample_ids")
    topk_influences = topk_data.get("influences") or topk_data.get("influence_values")
    topk_labels = topk_data.get("train_labels") or None
    
    if topk_train_ids is None or topk_influences is None:
        raise ValueError("[ERROR] TopK influence file missing required keys (train_ids/influences)")
    
    print(f"\n[INFO] Starting comparison between full and TopK influence results (Test ID: {TEST_ID})")
    print(f"[INFO] Full influence data: {len(full_train_ids)} training samples")
    print(f"[INFO] TopK influence data: {len(topk_train_ids)} training samples")
    
    # 3. Match IDs between full and TopK data
    common_train_ids = np.intersect1d(full_train_ids, topk_train_ids)
    print(f"[INFO] Common train sample IDs between two files: {len(common_train_ids)}")
    
    if len(common_train_ids) == 0:
        print("[WARNING] No common train sample IDs found between full and TopK data")
        return
    
    # 4. Extract matched data
    topk_id_to_influence = dict(zip(topk_train_ids, topk_influences))
    topk_id_to_label = dict(zip(topk_train_ids, topk_labels)) if topk_labels is not None else {}
    
    matched_data = []
    for train_id in common_train_ids:
        matched_data.append({
            "train_id": train_id,
            "train_label": full_label_map[train_id],
            "full_influence": full_influence_map[train_id],
            "topk_influence": topk_id_to_influence[train_id],
            "topk_label": topk_id_to_label.get(train_id, "N/A")
        })
    
    # 5. Print comparison results
    print(f"\n==================== Comparison Results (Test ID: {TEST_ID}) ====================")
    print(f"{'Train ID':<10} {'Label':<8} {'Full Influence':<15} {'TopK Influence':<15} {'Difference':<15}")
    print("-" * 70)
    for item in matched_data[:20]:  # Print first 20 matches for readability
        diff = item["full_influence"] - item["topk_influence"]
        print(f"{item['train_id']:<10} {item['train_label']:<8} {item['full_influence']:<15.6f} {item['topk_influence']:<15.6f} {diff:<15.6f}")
    
    # 6. Statistical analysis
    full_matched_influences = np.array([item["full_influence"] for item in matched_data])
    topk_matched_influences = np.array([item["topk_influence"] for item in matched_data])
    abs_diff = np.abs(full_matched_influences - topk_matched_influences)
    
    print(f"\n[STATISTICS] Comparison of Matched Influences:")
    print(f"  Mean Absolute Difference: {np.mean(abs_diff):.6f}")
    print(f"  Max Absolute Difference: {np.max(abs_diff):.6f}")
    print(f"  Min Absolute Difference: {np.min(abs_diff):.6f}")
    print(f"  Pearson Correlation: {np.corrcoef(full_matched_influences, topk_matched_influences)[0,1]:.6f}")
    print(f"==================== Comparison Completed ====================")

# -------------------------- Main Function --------------------------
def main():
    """Main function to run one-step train influence calculation, save results and compare with TopK."""
    # 1. Load working model and original (untouched) model
    working_model, original_model, ckpt = load_pretrained_model_and_original(CKPT_PATH, DEVICE)
    print(f"[INFO] Computing device: {DEVICE}")
    
    # 2. Load MNIST dataset with IDs using build_dataloader_mnist
    # Load test dataset (for specified TEST_ID)
    _, test_dataset = build_dataloader_mnist(
        data_root=MNIST_DATA_ROOT,
        train=False,
        return_dataset=True,
        limit_size=TEST_LIMIT_SIZE
    )
    
    # Load training dataset (for iterating over all training samples)
    _, train_dataset = build_dataloader_mnist(
        data_root=MNIST_DATA_ROOT,
        train=True,
        return_dataset=True,
        limit_size=TRAIN_LIMIT_SIZE
    )
    
    # Validate TEST_ID
    if TEST_ID < 0 or TEST_ID >= len(test_dataset):
        raise ValueError(f"[ERROR] TEST_ID {TEST_ID} is out of range (test dataset size: {len(test_dataset)})")
    
    # 3. Get specified test sample (with ID)
    test_x, test_y, test_sample_id = test_dataset[TEST_ID]
    print(f"\n[INFO] Target test sample - ID: {test_sample_id}, Label: {test_y}, Shape: {test_x.shape}")
    
    # 4. Calculate FIXED base loss (only once, using original model)
    fixed_base_loss = calculate_single_sample_loss(original_model, test_x, test_y, DEVICE)
    print(f"[INFO] Fixed base loss (from original model): {fixed_base_loss:.6f} (will not change during training)")
    
    # 5. Calculate influence for each training sample
    influence_results = []
    train_dataset_size = len(train_dataset)
    print(f"\n[INFO] Starting influence calculation for {train_dataset_size} training samples (test ID: {test_sample_id})")
    
    for train_idx in range(train_dataset_size):
        # Get training sample with ID
        train_x, train_y, train_sample_id = train_dataset[train_idx]
        
        # Calculate influence (use fixed base loss, restore from original model)
        data_influence, base_loss, new_loss = one_step_train_and_calc_influence(
            working_model=working_model,
            original_model=original_model,
            train_x=train_x,
            train_y=train_y,
            test_x=test_x,
            test_y=test_y,
            fixed_base_loss=fixed_base_loss,
            device=DEVICE,
            lr=LEARNING_RATE,
            optimizer_type=OPTIMIZER_TYPE
        )
        
        # Save results
        influence_results.append({
            "train_idx": train_idx,
            "train_sample_id": train_sample_id,
            "train_label": train_y,
            "base_loss": base_loss,  # Fixed base loss (same for all samples)
            "new_loss": new_loss,
            "data_influence": data_influence
        })
        
        # Print progress every 1000 samples
        if (train_idx + 1) % 1000 == 0:
            print(f"[PROGRESS] Processed {train_idx + 1}/{train_dataset_size} training samples")
    
    # 6. Save full influence results
    full_save_path = save_full_influence_results(influence_results, RESULT_SAVE_DIR, FULL_INFLUENCE_SAVE_NAME)
    
    # 7. Compare with TopK influence results
    try:
        compare_full_and_topk_influence(full_save_path, TOPK_INFLUENCE_PATH)
    except Exception as e:
        print(f"[WARNING] Comparison failed: {str(e)}")
    
    # 8. Print summary results
    print(f"\n==================== Test Sample {test_sample_id} Influence Summary ====================")
    print(f"[SUMMARY] Fixed base loss (original model): {fixed_base_loss:.6f}")
    
    # Sort results by influence (descending order)
    influence_results.sort(key=lambda x: x["data_influence"], reverse=True)
    top_influence = influence_results[0]
    bottom_influence = influence_results[-1]
    
    print(f"\n[TOP INFLUENCE] Most influential training sample:")
    print(f"  Train Index: {top_influence['train_idx']}, Sample ID: {top_influence['train_sample_id']}")
    print(f"  Train Label: {top_influence['train_label']}, Influence Value: {top_influence['data_influence']:.6f}")
    print(f"  Pre-Train Test Loss: {top_influence['base_loss']:.6f}, Post-Train Test Loss: {top_influence['new_loss']:.6f}")
    
    print(f"\n[BOTTOM INFLUENCE] Least influential training sample:")
    print(f"  Train Index: {bottom_influence['train_idx']}, Sample ID: {bottom_influence['train_sample_id']}")
    print(f"  Train Label: {bottom_influence['train_label']}, Influence Value: {bottom_influence['data_influence']:.6f}")
    print(f"  Pre-Train Test Loss: {bottom_influence['base_loss']:.6f}, Post-Train Test Loss: {bottom_influence['new_loss']:.6f}")
    print(f"\n==================== Influence Calculation Completed ====================")

if __name__ == "__main__":
    main()