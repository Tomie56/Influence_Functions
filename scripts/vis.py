import os
import argparse
import json
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
from pathlib import Path

# Set plot style
plt.rcParams["font.family"] = ["DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["font.size"] = 10
plt.rcParams["axes.linewidth"] = 0.8
plt.rcParams["grid.alpha"] = 0.3

# -------------------------- Data Loading Helpers (Compatible with Previous Code) --------------------------
def load_influence_results(if_save_dir: str, test_id: int):
    """Load influence scores and train IDs from run_if.py output"""
    if_path = Path(if_save_dir) / f"influences_test_{test_id}.npz"
    if not if_path.exists():
        raise FileNotFoundError(f"Influence file not found: {if_path}")
    
    data = np.load(if_path)
    influences = data["scores"].flatten()
    train_ids = data["train_ids"].flatten()
    return influences, train_ids

def load_loo_results(loo_jsonl_path: str):
    """Load LOO results and calculate loss changes (compatible with run_loo.py output)"""
    loo_data = {}
    base_loss = None
    with open(loo_jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            item = json.loads(line.strip())
            train_id = item["id"]
            loss = item["loss"]
            if train_id == -1:
                base_loss = loss
            else:
                loo_data[train_id] = loss
    
    if base_loss is None:
        raise ValueError("Base model loss (id=-1) not found in LOO results")
    
    # Calculate loss change: loo_loss - base_loss
    loss_changes = {}
    for train_id, loo_loss in loo_data.items():
        loss_changes[train_id] = loo_loss - base_loss
    return loss_changes, base_loss

def align_if_loo_data(influences: np.ndarray, train_ids_if: np.ndarray, loss_changes: dict):
    """Align IF scores and LOO loss changes by train ID"""
    aligned_if = []
    aligned_loo = []
    for if_score, train_id in zip(influences, train_ids_if):
        if train_id in loss_changes:
            aligned_if.append(if_score)
            aligned_loo.append(loss_changes[train_id])
    return np.array(aligned_if), np.array(aligned_loo)

# -------------------------- Visualization Functions --------------------------
def plot_influence_vs_leave_one_out(
    actual_loss_changes: np.ndarray,
    predicted_influences: np.ndarray,
    save_dir: str,
    test_idx: int,
    top_k: int = 500,
    model_type: str = "Approx",
    ) -> None:
    os.makedirs(save_dir, exist_ok=True)

    # Align and filter top-k
    if actual_loss_changes.shape[0] != predicted_influences.shape[0]:
        n = min(actual_loss_changes.shape[0], predicted_influences.shape[0])
        actual_loss_changes = actual_loss_changes[:n]
        predicted_influences = predicted_influences[:n]

    if top_k is not None and top_k < len(actual_loss_changes):
        top_idx = np.argsort(np.abs(predicted_influences))[-top_k:]
        actual_loss_changes = actual_loss_changes[top_idx]
        predicted_influences = predicted_influences[top_idx]

    # Calculate correlation
    corr, p_value = pearsonr(actual_loss_changes, predicted_influences)

    # Plot
    fig, ax = plt.subplots(figsize=(8, 6), dpi=150)
    ax.scatter(actual_loss_changes, predicted_influences, alpha=0.6, s=20, color="#1f77b4", edgecolors="none")

    # Ideal y=x line
    min_val = float(min(np.min(actual_loss_changes), np.min(predicted_influences)))
    max_val = float(max(np.max(actual_loss_changes), np.max(predicted_influences)))
    ax.plot(
        [min_val, max_val],
        [min_val, max_val],
        "r--",
        linewidth=2,
        label=f"Ideal (y=x), r={corr:.3f} (p={p_value:.2e})",
    )

    ax.set_xlabel("Actual Change in Loss (LOO)")
    ax.set_ylabel("Predicted Influence (IF)")
    ax.set_title(f"{model_type}: Influence vs. Leave-One-Out (Test {test_idx})", pad=15)
    ax.legend(fontsize=10)
    ax.grid(True, linestyle="--")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    save_path = os.path.join(save_dir, f"influence_vs_loo_test_{test_idx}.png")
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches="tight", dpi=300)
    plt.close()
    print(f"Saved IF vs LOO plot: {save_path}")
    print(f"Pearson r={corr:.3f}, p={p_value:.2e}")

# -------------------------- Main Logic --------------------------
def parse_args():
    parser = argparse.ArgumentParser("Visualization for Influence Function (IF) and LOO Results")
    # Data paths
    parser.add_argument("--if_save_dir", type=str, required=True, help="Directory of run_if.py output")
    parser.add_argument("--loo_jsonl", type=str, required=True, help="Path to run_loo.py output jsonl")
    parser.add_argument("--vis_save_dir", type=str, required=True, help="Directory to save visualizations")
    # Config
    parser.add_argument("--test_id", type=int, required=True, help="Target test sample ID")
    parser.add_argument("--top_n_dist", type=int, default=50, help="Top N helpful/harmful for distribution plot")
    parser.add_argument("--top_k_corr", type=int, default=500, help="Top K samples for correlation plot")
    parser.add_argument("--model_type", type=str, default="IF", help="Model type label for plot title")
    return parser.parse_args()

def main():
    args = parse_args()

    # Load and align data
    influences, train_ids_if = load_influence_results(args.if_save_dir, args.test_id)
    loss_changes, _ = load_loo_results(args.loo_jsonl)
    aligned_if, aligned_loo = align_if_loo_data(influences, train_ids_if, loss_changes)

    # Plot IF vs LOO correlation
    plot_influence_vs_leave_one_out(aligned_loo, aligned_if, args.vis_save_dir, args.test_id, args.top_k_corr, args.model_type)

    print("All visualizations completed!")

if __name__ == "__main__":
    main()